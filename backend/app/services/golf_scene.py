"""Non-golf clip scanner (dev course-testing tool).

The camera fires on any motion, so the Production queue collects garbage
(kitchen walk-bys, empty rooms, pets). This module decides whether a clip
contains a real golf SWING, using the pose swing detector (MediaPipe
wrist-speed + back-bend — the same detector that drives production cutting).
A clip with **no identified swing** is flagged non-golf, so the operator can
bulk-select and delete the garbage.

Fail-safe: pose unavailable (mediapipe not installed) or any per-clip error
→ treated as GOLF (unflagged), so we never mass-flag a whole queue. Scanning
runs on a background worker with progress.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

log = logging.getLogger("golfreelz.golf_scene")


# Scan the clip in chunks and STOP at the first real swing — a golf clip
# rarely needs the whole video read, and we only need "is there a swing".
# A non-golf clip (no swing) still gets fully scanned (that's unavoidable —
# you can't prove absence early). Lower sample rate than production since we
# don't need precise timing here.
_SCAN_CHUNK_SEC = 20.0
_SCAN_SAMPLE_HZ = 10.0


def classify_clip(path: Path) -> dict:
    """Return {"is_golf": bool, "reason": str} for one clip.

    A clip is non-golf when the pose detector finds no swing in it. Scans in
    chunks and returns GOLF as soon as one swing is found. Fail-safe: when
    pose is unavailable or anything errors, return is_golf=True so a real shot
    is never dropped and the queue is never mass-flagged."""
    try:
        from . import pose_swing

        if not pose_swing.available():
            return {"is_golf": True, "reason": "pose unavailable (mediapipe not installed)"}
        t = 0.0
        for _ in range(200):  # safety bound
            dbg: dict = {}
            swings = pose_swing.detect_swings_from_pose(
                path, sample_hz=_SCAN_SAMPLE_HZ,
                start_sec=t, max_scan_sec=_SCAN_CHUNK_SEC, debug=dbg,
            )
            if swings:
                return {"is_golf": True, "reason": f"swing ~{swings[0]['peak_time_sec']:.0f}s"}
            # RESCUE-AWARE (same rule as produce): a burst dropped ONLY
            # by the wrist-speed ratio gate but with a swing-posture
            # bend is very possibly a real swing seen from far away —
            # NEVER classify (and delete) on that evidence. Produce's
            # ratio-rescue + ball-departure check will arbitrate.
            for b in (dbg.get("bursts_detail") or []):
                if (
                    b.get("status") == "ratio_low"
                    and b.get("bend") is not None
                    and float(b["bend"]) >= 15.0
                ):
                    return {
                        "is_golf": True,
                        "reason": (
                            f"rescueable burst ~{b['t']:.0f}s (ratio "
                            f"{b.get('ratio')}, bend {b.get('bend')}) — "
                            "benefit of the doubt"
                        ),
                    }
            if dbg.get("reached_eof", True):
                break  # scanned to the end with no swing
            t += _SCAN_CHUNK_SEC
        return {"is_golf": False, "reason": "no golf swing detected"}
    except Exception as exc:  # noqa: BLE001
        log.warning("golf_scene: classify failed for %s: %s", path, exc)
        return {"is_golf": True, "reason": "error"}


# ── background scan job ────────────────────────────────────────────────
_state = {
    "running": False,
    "total": 0,
    "done": 0,
    "flagged": [],       # upload row ids classified NON-golf
    "finished_at": None,
}
_lock = threading.Lock()


def status() -> dict:
    with _lock:
        return dict(_state, flagged=list(_state["flagged"]))


def start_scan(items: list[tuple[int, Path]]) -> bool:
    """Kick a background scan. `items` is [(upload_id, clip_path), ...].
    Returns False if a scan is already running."""
    with _lock:
        if _state["running"]:
            return False
        _state.update({
            "running": True, "total": len(items), "done": 0,
            "flagged": [], "finished_at": None,
        })

    def _run():
        flagged = []
        try:
            for upload_id, path in items:
                try:
                    verdict = classify_clip(path)
                    if not verdict["is_golf"]:
                        flagged.append(upload_id)
                        log.info(
                            "golf_scene: flagged upload %s (%s)",
                            upload_id, verdict["reason"],
                        )
                except Exception as exc:  # noqa: BLE001
                    log.warning("golf_scene: scan item %s failed: %s", upload_id, exc)
                with _lock:
                    _state["done"] += 1
                    _state["flagged"] = list(flagged)
        finally:
            with _lock:
                _state["running"] = False
                _state["flagged"] = list(flagged)
                import time as _t
                _state["finished_at"] = _t.time()
            log.info("golf_scene: scan done — %d flagged of %d", len(flagged), len(items))

    threading.Thread(target=_run, name="golf-scene-scan", daemon=True).start()
    return True
