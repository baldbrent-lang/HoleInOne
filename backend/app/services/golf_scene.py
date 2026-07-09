"""Non-golf clip scanner (dev course-testing tool).

The camera fires on any motion, so the Production queue collects garbage:
someone walking through a kitchen, an empty indoor room, a pet — anything
that isn't a golfer taking a shot. This module scans a clip and decides
whether it looks like a real golf shot, so the operator can bulk-select
the garbage and delete it in one pass instead of eyeballing each card.

Verdict per clip (we scan the tee angle; fall back to green):
  * NON-GOLF if no person appears in any sampled frame ("no people"), OR
  * NON-GOLF if the scene has almost no grass/turf ("indoor / no golf
    scene") — catches the kitchen walk-by, which HAS a person but no grass.
  * GOLF only when a person is present AND the scene is outdoors-green.

Detection is deliberately conservative (biased toward keeping) because a
false NON-GOLF only *pre-checks* a box — the operator still reviews and
clicks delete — but we'd rather under-flag than tempt a real shot's
deletion. Person detection reuses the same YOLOv8n ONNX the tee Pi runs;
if the model can't be found it degrades to the grass check alone.

Everything is fail-safe: any per-clip error → treated as GOLF (unflagged),
never a crash. Scanning runs on a background worker with progress, mirrored
on services/mirror.py.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("golfreelz.golf_scene")

# COCO class 0 is "person" in YOLOv8. The in-repo ONNX was exported at
# 320×320 (the tee Pi's input_size), and its detection-head reshape is
# baked to that size — cv2.dnn throws on any other input. Must match.
_PERSON_CLASS = 0
_INPUT = 320
_PERSON_CONF = 0.35

# Grass/turf: OpenCV HSV hue is 0–179; greens sit roughly 30–90 with real
# saturation/value (excludes grey walls, skin, wood). A golf frame is
# grass-dominated; an indoor frame is near zero. Low bar so we only flag
# CLEARLY grass-less scenes on this criterion.
_GRASS_H_LO, _GRASS_H_HI = 30, 95
_GRASS_S_MIN, _GRASS_V_MIN = 40, 40
_GRASS_MIN_FRAC = 0.08

_SAMPLE_FRAMES = 10

_net = None
_net_tried = False


def _load_net():
    """Lazy-load the YOLOv8n ONNX from the in-repo Pi model. Returns None
    (person detection disabled) if it can't be found/loaded — the scan then
    relies on the grass check alone."""
    global _net, _net_tried
    if _net is not None or _net_tried:
        return _net
    _net_tried = True
    # golf_scene.py → services → app → backend → <repo root>
    root = Path(__file__).resolve().parents[3]
    candidates = [
        root / "backend" / "app" / "models" / "yolov8n.onnx",
        root / "pi-agent" / "models" / "yolov8n.onnx",
    ]
    for p in candidates:
        if p.exists():
            try:
                _net = cv2.dnn.readNetFromONNX(str(p))
                log.info("golf_scene: person detector loaded from %s", p)
                return _net
            except Exception as exc:  # noqa: BLE001
                log.warning("golf_scene: failed to load %s: %s", p, exc)
    log.info("golf_scene: no person model found — grass check only")
    return None


def _person_present(frame) -> bool:
    net = _load_net()
    if net is None:
        return True  # unknown → don't let the person gate flag anything
    try:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (_INPUT, _INPUT), swapRB=True, crop=False,
        )
        net.setInput(blob)
        out = net.forward()[0].T  # (anchors, 84)
        scores = out[:, 4 + _PERSON_CLASS]
        return bool(np.any(scores >= _PERSON_CONF))
    except Exception as exc:  # noqa: BLE001
        log.debug("golf_scene: person detect failed: %s", exc)
        return True


def _grass_fraction(frame) -> float:
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            (_GRASS_H_LO, _GRASS_S_MIN, _GRASS_V_MIN),
            (_GRASS_H_HI, 255, 255),
        )
        return float(cv2.countNonZero(mask)) / float(mask.size or 1)
    except Exception:  # noqa: BLE001
        return 1.0  # unknown → treat as grass so we don't flag on error


def classify_clip(path: Path) -> dict:
    """Return {"is_golf": bool, "reason": str} for one clip. Fail-safe:
    any error yields is_golf=True (unflagged)."""
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return {"is_golf": True, "reason": "unreadable"}
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            total = _SAMPLE_FRAMES
        idxs = np.linspace(0, max(0, total - 1), _SAMPLE_FRAMES).astype(int)
        person = False
        grass_fracs = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            grass_fracs.append(_grass_fraction(frame))
            if not person and _person_present(frame):
                person = True
        cap.release()
        if not grass_fracs:
            return {"is_golf": True, "reason": "no frames"}
        grass = float(np.median(grass_fracs))
        outdoor = grass >= _GRASS_MIN_FRAC
        if not person:
            return {"is_golf": False, "reason": "no people"}
        if not outdoor:
            return {"is_golf": False, "reason": f"indoor / no grass ({grass:.0%})"}
        return {"is_golf": True, "reason": f"golfer + grass ({grass:.0%})"}
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
