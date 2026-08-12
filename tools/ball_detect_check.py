#!/usr/bin/env python3
"""Does the ball detector still find a BALL rather than a SHOE?

Run:  python3 tools/ball_detect_check.py

This exists because the detector spent two rounds confidently ringing a
white shoe while a plainly visible ball sat a few feet away, and nothing
in the pipeline could say why. The cause was not a threshold: the scan
resized every frame to 360p and then demanded a blob of at least
0.00002 of the frame area. On 1080p that is a 3x downscale and a 4.6px^2
floor, and a golf ball is ~8px across natively -- so the ball was being
erased (by the downscale, and then by a 3x3 MORPH_OPEN that takes a pixel
off every edge) BEFORE any threshold was consulted, while a 30px shoe
sailed through with 14x the margin.

That failure is invisible from the outside: "found no ball" and "threw
the ball away in the first ten lines" look identical in the output. So
this builds a synthetic clip whose truth is known -- one ball-sized blob
that rests and then vanishes, one shoe-sized blob that never moves -- and
runs the REAL detector over it (not a copy of the logic, which would
drift away from the thing it is supposed to be checking).

Exits non-zero if the ball is missed or the shoe is reported.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    print("needs opencv + numpy:  pip install opencv-python-headless numpy")
    raise SystemExit(2)

ROOT = Path(__file__).resolve().parent.parent
AI_TRACER = ROOT / "backend" / "app" / "services" / "ai_tracer.py"

# The synthetic scene, in native pixels. These are the numbers that
# matter: a golf ball at tee-camera distance really is a handful of
# pixels across, and a shoe really is several times that.
W, H = 1920, 1080
FPS = 30.0
BALL_XY, BALL_R = (1200, 600), 4        # 8px across
SHOE_XY, SHOE_WH = (1000, 640), (15, 6)  # 30x12
REST_SEC, GONE_SEC = 3.0, 3.0            # ball sits, then is struck
ROI = {"x": 0.45, "y": 0.45, "w": 0.30, "h": 0.20}


def build_clip(path: Path) -> float:
    """A ball that rests then vanishes, and a shoe that never leaves."""
    vw = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)
    )
    if not vw.isOpened():
        print("could not open a VideoWriter (no mp4v encoder?)")
        raise SystemExit(2)
    n_rest = int(REST_SEC * FPS)
    for i in range(int((REST_SEC + GONE_SEC) * FPS)):
        f = np.zeros((H, W, 3), np.uint8)
        f[:] = (40, 90, 55)  # grass
        # A little noise so the clip is not unrealistically clean.
        cv2.randn(noise := np.zeros((H, W, 3), np.int16), 0, 6)
        f = np.clip(f.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.ellipse(f, SHOE_XY, SHOE_WH, 0, 0, 360, (235, 240, 240), -1)
        if i < n_rest:
            cv2.circle(f, BALL_XY, BALL_R, (240, 245, 245), -1, cv2.LINE_AA)
        vw.write(f)
    vw.release()
    return n_rest / FPS


def main() -> int:
    spec = importlib.util.spec_from_file_location("ai_tracer", AI_TRACER)
    ai = importlib.util.module_from_spec(spec)
    sys.modules["ai_tracer"] = ai
    spec.loader.exec_module(ai)

    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "synthetic.mp4"
        depart_at = build_clip(clip)
        dbg: dict = {}
        segs = ai.detect_swings_from_ball(clip, fps=FPS, roi=ROI, debug=dbg)

        print(f"clip: {W}x{H} @{FPS:g}fps, ball vanishes at {depart_at:.1f}s")
        print(f"accepted radius window : {dbg.get('accept_radius_px')} px")
        print(f"scanned at native res  : {dbg.get('native_scan')}")
        print(f"candidates / in box    : {dbg.get('n_cand_total')} / "
              f"{dbg.get('n_cand_in_roi')}")
        print(f"rejected wrong size    : {dbg.get('n_drop_size')}")
        print(f"rejected wrong shape   : {dbg.get('n_drop_shape')}")
        print(f"departures             : {dbg.get('departures')}")
        print(f"reason                 : {dbg.get('reason')}")

        fails = []
        if len(segs) != 1:
            fails.append(f"expected exactly 1 departure, got {len(segs)}")
        for s in segs:
            dx = abs(s["ball_x"] - BALL_XY[0])
            dy = abs(s["ball_y"] - BALL_XY[1])
            if abs(s["ball_x"] - SHOE_XY[0]) < 40 and abs(s["ball_y"] - SHOE_XY[1]) < 40:
                fails.append(
                    f"reported the SHOE at ({s['ball_x']:.0f},{s['ball_y']:.0f})"
                )
            elif dx > 25 or dy > 25:
                fails.append(
                    f"departure at ({s['ball_x']:.0f},{s['ball_y']:.0f}) is not "
                    f"the ball at {BALL_XY}"
                )
            elif abs(s["peak_time_sec"] - depart_at) > 0.5:
                fails.append(
                    f"departure timed at {s['peak_time_sec']:.2f}s, expected "
                    f"~{depart_at:.2f}s"
                )

        print()
        if fails:
            for f in fails:
                print(f"FAIL: {f}")
            return 1
        print("PASS: found the ball, timed the departure, ignored the shoe.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
