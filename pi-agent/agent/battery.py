"""INA226 battery telemetry.

Reads bus voltage + shunt current from an INA226 breakout wired inline
on the 12V battery feed (battery -> fuse -> INA226 -> buck -> Pi), over
I2C. The readings ride along on every heartbeat so the admin dashboard
can show per-camera battery state and warn before a battery dies.

Best-effort by design: no sensor, no smbus2 package, or a wiring fault
just means read() returns None and heartbeats go out without battery
fields — never crashes the capture loop.

Wiring (see docs/battery-power-wiring.md for the full build):
    INA226 VCC  -> Pi 3.3V (pin 1)
    INA226 GND  -> Pi GND  (pin 6)
    INA226 SDA  -> Pi SDA  (pin 3, GPIO2)
    INA226 SCL  -> Pi SCL  (pin 5, GPIO3)
    IN+ / IN-   -> in series on the battery's positive lead

Enable I2C once per Pi:  sudo raspi-config nonint do_i2c 0
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger("golfreelz_agent.battery")

# INA226 registers
_REG_SHUNT_V = 0x01  # 2.5 uV / LSB, signed
_REG_BUS_V = 0x02    # 1.25 mV / LSB

# The '0-36V 20A' screw-terminal breakout ships with a 20A shunt —
# nominally 0.002 ohm. The plain CJMCU-style blue board uses 0.1 ohm.
# Override in config.yaml (battery.shunt_ohms) to match your board;
# the Dallas bench test against a known load is the calibration step.
_DEFAULT_SHUNT_OHMS = 0.002


class BatteryMonitor:
    """Lazy INA226 reader. Construction never raises; the first read
    attempts to open the bus and disables itself after repeated
    failures so a missing sensor costs nothing at runtime."""

    def __init__(self, cfg: dict | None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.bus_num = int(cfg.get("i2c_bus", 1))
        self.addr = int(cfg.get("i2c_address", 0x40))
        self.shunt_ohms = float(cfg.get("shunt_ohms", _DEFAULT_SHUNT_OHMS))
        self._bus = None
        self._fails = 0
        self._dead = False

    def _open(self):
        if self._bus is not None:
            return self._bus
        import smbus2  # type: ignore

        self._bus = smbus2.SMBus(self.bus_num)
        return self._bus

    def _read16(self, reg: int) -> int:
        raw = self._open().read_word_data(self.addr, reg)
        # SMBus returns little-endian; INA226 is big-endian.
        return ((raw & 0xFF) << 8) | (raw >> 8)

    def read(self) -> dict | None:
        """One sample: {"voltage": V, "current_a": A} or None."""
        if not self.enabled or self._dead:
            return None
        try:
            bus_raw = self._read16(_REG_BUS_V)
            shunt_raw = self._read16(_REG_SHUNT_V)
            if shunt_raw > 0x7FFF:  # sign
                shunt_raw -= 0x10000
            voltage = bus_raw * 1.25e-3
            current = (shunt_raw * 2.5e-6) / self.shunt_ohms
            if not (0.0 < voltage < 30.0):
                raise ValueError(f"implausible bus voltage {voltage:.2f}V")
            self._fails = 0
            return {
                "voltage": round(voltage, 3),
                "current_a": round(current, 3),
            }
        except Exception as exc:  # noqa: BLE001
            self._fails += 1
            if self._fails == 1:
                log.warning("battery: INA226 read failed: %s", exc)
            if self._fails >= 5:
                if not self._dead:
                    log.warning(
                        "battery: %d consecutive failures — disabling "
                        "telemetry until restart", self._fails,
                    )
                self._dead = True
            return None

    def read_averaged(self, samples: int = 3, gap_sec: float = 0.05) -> dict | None:
        """A few samples averaged — smooths LTE-modem TX current spikes
        so the dashboard number is representative."""
        vals = []
        for _ in range(max(1, samples)):
            r = self.read()
            if r:
                vals.append(r)
            if gap_sec:
                time.sleep(gap_sec)
        if not vals:
            return None
        return {
            "voltage": round(sum(v["voltage"] for v in vals) / len(vals), 3),
            "current_a": round(sum(v["current_a"] for v in vals) / len(vals), 3),
        }
