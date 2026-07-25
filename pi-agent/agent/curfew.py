"""Curfew sleep/wake: halt the Pi outside playing hours, wake by RTC.

Power math that motivates this: the rig draws ~6-8W recording but
essentially nothing halted, so sleeping the ~10 dark hours stretches a
100Ah battery from ~5-6 days to ~10-12 — and battery swaps happen at
night, when the camera is asleep anyway, costing zero footage.

Mechanism (Pi 5): write the next wake time (epoch seconds) into
/sys/class/rtc/rtc0/wakealarm, then power off. The RTC (with its coin
cell) fires the alarm and the PMIC boots the Pi at dawn. For the
lowest halted draw set POWER_OFF_ON_HALT=1 in the EEPROM config
(rpi-eeprom-config); wake-on-alarm works either way.

Safety rails:
  * Clock sanity — never halt on a clock that predates this code
    (unsynced boot clock could put "21:00" anywhere in history).
  * Boot grace — if the Pi boots INSIDE the curfew window (battery
    reconnected early, manual power-up for maintenance), stay awake
    for boot_grace_seconds so the operator can SSH in and disable the
    curfew before it re-halts.
  * Arm-before-halt — the wakealarm write must SUCCEED or the halt is
    skipped; a Pi that can't promise to wake up stays awake.
"""

from __future__ import annotations

import datetime as dt
import logging
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger("golfreelz_agent.curfew")

_WAKEALARM = Path("/sys/class/rtc/rtc0/wakealarm")
_MIN_SANE_YEAR = 2025


def _parse_hhmm(s: str) -> tuple[int, int]:
    h, m = str(s).strip().split(":")
    return int(h), int(m)


class CurfewThread(threading.Thread):
    """Checks the local clock every 30s; inside the sleep window it
    arms the RTC for the wake time and powers the Pi off."""

    def __init__(self, cfg: dict | None):
        super().__init__(daemon=True, name="curfew")
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.sleep_hm = _parse_hhmm(cfg.get("sleep", "21:00"))
        self.wake_hm = _parse_hhmm(cfg.get("wake", "06:00"))
        self.boot_grace = int(cfg.get("boot_grace_seconds", 600))
        self.dry_run = bool(cfg.get("dry_run", False))
        self._started_at = time.monotonic()
        self.stopping = threading.Event()

    # ── window math ──────────────────────────────────────────────────
    def _in_sleep_window(self, now: dt.datetime) -> bool:
        sh, sm = self.sleep_hm
        wh, wm = self.wake_hm
        sleep_t = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
        wake_t = now.replace(hour=wh, minute=wm, second=0, microsecond=0)
        if sleep_t <= wake_t:
            # window inside one day (e.g. sleep 01:00 wake 06:00)
            return sleep_t <= now < wake_t
        # window spans midnight (the normal case: sleep 21:00 wake 06:00)
        return now >= sleep_t or now < wake_t

    def _next_wake(self, now: dt.datetime) -> dt.datetime:
        wh, wm = self.wake_hm
        wake_t = now.replace(hour=wh, minute=wm, second=0, microsecond=0)
        if wake_t <= now:
            wake_t += dt.timedelta(days=1)
        return wake_t

    # ── halt machinery ───────────────────────────────────────────────
    def _sleep_until(self, wake_at: dt.datetime) -> bool:
        """Arm the RTC and power off. Primary path is `rtcwake -m off`
        (one atomic arm+halt; the agent user gets a NOPASSWD sudoers
        entry for it from install.sh). Fallback: direct sysfs arm +
        shutdown for root-run agents. Returns False only when NOTHING
        could both guarantee a wake-up and halt."""
        epoch = int(wake_at.timestamp())
        for cmd in (
            ["sudo", "-n", "rtcwake", "-m", "off", "-t", str(epoch)],
            ["rtcwake", "-m", "off", "-t", str(epoch)],
        ):
            try:
                subprocess.run(
                    cmd, timeout=30, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                )
                return True  # (normally unreached — the Pi is off)
            except Exception:  # noqa: BLE001
                continue
        if self._arm_rtc(wake_at):
            self._halt()
            return True
        return False

    def _arm_rtc(self, wake_at: dt.datetime) -> bool:
        epoch = int(wake_at.timestamp())
        try:
            _WAKEALARM.write_text("0\n")  # clear any stale alarm
        except OSError:
            pass  # empty/unset alarm errors on some kernels — fine
        try:
            _WAKEALARM.write_text(f"{epoch}\n")
        except OSError as exc:
            log.error("curfew: could not arm RTC wakealarm: %s", exc)
            return False
        # Read back — the kernel silently drops past-times.
        try:
            armed = _WAKEALARM.read_text().strip()
            if not armed or int(armed) != epoch:
                log.error(
                    "curfew: wakealarm readback mismatch (%r != %d)",
                    armed, epoch,
                )
                return False
        except (OSError, ValueError) as exc:
            log.error("curfew: wakealarm readback failed: %s", exc)
            return False
        return True

    def _halt(self) -> None:
        for cmd in (
            ["sudo", "-n", "shutdown", "-h", "now"],
            ["shutdown", "-h", "now"],
            ["systemctl", "poweroff"],
        ):
            try:
                subprocess.run(cmd, timeout=15, check=True)
                return
            except Exception:  # noqa: BLE001
                continue
        log.error(
            "curfew: every shutdown command failed — staying awake "
            "(agent user may need a sudoers entry for shutdown)",
        )

    # ── main loop ────────────────────────────────────────────────────
    def run(self) -> None:
        if not self.enabled:
            return
        log.info(
            "curfew: enabled — sleep %02d:%02d wake %02d:%02d%s",
            *self.sleep_hm, *self.wake_hm,
            " (DRY RUN)" if self.dry_run else "",
        )
        while not self.stopping.is_set():
            now = dt.datetime.now()
            if now.year < _MIN_SANE_YEAR:
                # Clock not yet synced (no NTP over LTE yet) — a bogus
                # clock must never decide to power off the rig.
                self.stopping.wait(30)
                continue
            if self._in_sleep_window(now):
                grace_left = self.boot_grace - (
                    time.monotonic() - self._started_at
                )
                if grace_left > 0:
                    log.info(
                        "curfew: in sleep window but %.0fs of boot grace "
                        "remain (operator window) — halting after",
                        grace_left,
                    )
                    self.stopping.wait(min(grace_left + 1, 60))
                    continue
                wake_at = self._next_wake(now)
                log.info(
                    "curfew: sleep window reached — arming RTC for %s "
                    "and powering off", wake_at.isoformat(),
                )
                if self.dry_run:
                    log.info("curfew: DRY RUN — skipping halt")
                    self.stopping.wait(300)
                    continue
                if not self._sleep_until(wake_at):
                    log.error(
                        "curfew: could not guarantee an RTC wake-up — "
                        "staying awake (will retry)",
                    )
                # Either the halt failed or (dry paths) returned —
                # don't spin-loop the shutdown attempt.
                self.stopping.wait(300)
                continue
            self.stopping.wait(30)

    def stop(self) -> None:
        self.stopping.set()
