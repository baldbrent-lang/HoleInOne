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


def classify_clip(path: Path) -> dict:
    """Return {"is_golf": bool, "reason": str} for one clip.

    A clip is non-golf when the pose detector finds no swing in it. Fail-safe:
    when pose is unavailable or anything errors, return is_golf=True so a real
    shot is never dropped and the queue is never mass-flagged."""
    try:
        from . import pose_swing

        if not pose_swing.available():
            return {"is_golf": True, "reason": "pose unavailable (mediapipe not installed)"}
        dbg: dict = {}
        swings = pose_swing.detect_swings_from_pose(path, debug=dbg)
        if swings:
            return {"is_golf": True, "reason": f"{len(swings)} swing(s)"}
        return {"is_golf": False, "reason": dbg.get("reason") or "no golf swing detected"}
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
