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
that rests and then vanishes, plus the three things that were being
ringed instead of it on real footage -- a white shoe, a bare leg, and the
ball-sized bright TOE of a shoe -- and runs the REAL detector over it
(not a copy of the logic, which would drift away from the thing it is
supposed to be checking).

Exits non-zero if the ball is missed, mistimed, or if any decoy is
reported as a ball.
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
# Two resolutions, because they fail differently. At 1080p the ball is
# ~4px of radius; at 720p -- which is what the cameras actually shoot --
# it is ~2px, and at that size cv2.contourArea undercounts the blob by a
# third and every area-ratio test rejects a correctly-found ball.
RESOLUTIONS = ((1920, 1080, 4), (1280, 720, 2))
FPS = 30.0
BALL_XY, BALL_R = (1200, 600), 4        # 8px across
SHOE_XY, SHOE_WH = (1000, 640), (15, 6)  # 30x12 white shoe
# The three things that were being ringed instead of the ball on real
# footage: a bare leg (sunlit skin is bright and only moderately
# saturated), the small bright TOE of a shoe (ball-sized on its own), and
# the leg it is attached to. Each is placed inside the search box.
SKIN_XY, SKIN_WH = (1120, 560), (7, 26)   # a calf
TOE_XY, TOE_R = (1105, 600), 5            # shoe toe, ball-sized...
TOE_BODY = ((1085, 604), (18, 7))         # ...but attached to a shoe
# The hard cases: ball-SIZED, ball-SHAPED and ball-WHITE highlights that
# only their surroundings give away -- a sunlit spot on a knee, and a lit
# toe on a shoe whose body is in shade. Size and roundness cannot touch
# these; "is it surrounded by grass" is the only thing that can.
LEG_XY, LEG_WH = (1250, 640), (14, 34)    # a bare leg
KNEE_HL, KNEE_R = (1250, 632), 4          # sunlit spot ON that leg
DARKSHOE_XY, DARKSHOE_WH = (1150, 690), (22, 9)   # shoe in shade
LITTOE_XY, LITTOE_R = (1150, 688), 4      # its one sunlit patch
REST_SEC = 3.0        # the ball sits this long, then is struck
# ...and the PEOPLE walk off later. This matters: a decoy that never
# leaves can never produce a false departure, so a clip where the golfer
# stands still forever would pass no matter how bad the filters are. They
# leave far enough after the ball (> min_separation_sec) that a false
# departure cannot be merged into the real one and hidden.
PEOPLE_SEC = 8.0
TOTAL_SEC = 10.0
ROI = {"x": 0.45, "y": 0.45, "w": 0.30, "h": 0.20}


def scaled(xy, k):
    return (int(xy[0] * k), int(xy[1] * k))


def build_clip(path: Path, W: int, H: int, ball_r: int, k: float) -> float:
    """A ball that rests then is struck, and people who later walk off."""
    vw = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)
    )
    if not vw.isOpened():
        print("could not open a VideoWriter (no mp4v encoder?)")
        raise SystemExit(2)
    n_rest = int(REST_SEC * FPS)
    n_people = int(PEOPLE_SEC * FPS)
    S = lambda xy: scaled(xy, k)
    SW = lambda wh: (max(1, int(wh[0] * k)), max(1, int(wh[1] * k)))
    for i in range(int(TOTAL_SEC * FPS)):
        f = np.zeros((H, W, 3), np.uint8)
        f[:] = (40, 90, 55)  # grass
        # A little noise so the clip is not unrealistically clean.
        cv2.randn(noise := np.zeros((H, W, 3), np.int16), 0, 6)
        f = np.clip(f.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if i < n_people:
            cv2.ellipse(f, S(SHOE_XY), SW(SHOE_WH), 0, 0, 360, (235, 240, 240), -1)
            # sunlit skin: bright, but it has a hue -- S~68 vs the ball's ~5
            cv2.ellipse(f, S(SKIN_XY), SW(SKIN_WH), 0, 0, 360, (150, 175, 205), -1)
            # a ball-sized toe that is part of a bigger white shoe
            cv2.ellipse(f, S(TOE_BODY[0]), SW(TOE_BODY[1]), 0, 0, 360, (235, 240, 240), -1)
            cv2.circle(f, S(TOE_XY), max(1, int(TOE_R * k)), (238, 243, 243), -1, cv2.LINE_AA)
            # a bare leg with a ball-white, ball-sized sunlit spot on it
            cv2.ellipse(f, S(LEG_XY), SW(LEG_WH), 0, 0, 360, (150, 175, 205), -1)
            cv2.circle(f, S(KNEE_HL), max(1, int(KNEE_R * k)), (238, 242, 244), -1, cv2.LINE_AA)
            # a shaded shoe (too dark to be "white") with one lit toe patch
            cv2.ellipse(f, S(DARKSHOE_XY), SW(DARKSHOE_WH), 0, 0, 360, (120, 125, 125), -1)
            cv2.circle(f, S(LITTOE_XY), max(1, int(LITTOE_R * k)), (240, 244, 244), -1, cv2.LINE_AA)
        if i < n_rest:
            cv2.circle(f, S(BALL_XY), ball_r, (240, 245, 245), -1, cv2.LINE_AA)
        vw.write(f)
    vw.release()
    return n_rest / FPS


def run_one(ai, td: Path, W: int, H: int, ball_r: int) -> list[str]:
    """One resolution, end to end. Returns a list of failures."""
    k = W / 1920.0
    clip = td / f"synthetic_{W}x{H}.mp4"
    depart_at = build_clip(clip, W, H, ball_r, k)
    dbg: dict = {}
    segs = ai.detect_swings_from_ball(clip, fps=FPS, roi=ROI, debug=dbg)

    print(f"\n=== {W}x{H}, ball radius {ball_r}px, struck at {depart_at:.1f}s")
    print(f"  accepted radius window : {dbg.get('accept_radius_px')} px")
    print(f"  candidates in box      : {dbg.get('n_cand_in_roi')}")
    print(f"  rejected wrong size    : {dbg.get('n_drop_size')}")
    print(f"  rejected wrong shape   : {dbg.get('n_drop_shape')}")
    print(f"  rejected not on grass  : {dbg.get('n_drop_touch')}")
    print(f"  departures             : {dbg.get('departures')}")

    decoys = [("SHOE", SHOE_XY), ("SKIN", SKIN_XY), ("SHOE TOE", TOE_XY),
              ("KNEE HIGHLIGHT", KNEE_HL), ("LIT SHOE TOE", LITTOE_XY)]
    ball = scaled(BALL_XY, k)
    tol = max(25 * k, 12)
    fails = []
    if len(segs) != 1:
        fails.append(f"{W}x{H}: expected exactly 1 departure, got {len(segs)}")
    for s in segs:
        at = (s["ball_x"], s["ball_y"])
        decoy = next(
            (n for n, xy in decoys
             if abs(at[0] - scaled(xy, k)[0]) < tol
             and abs(at[1] - scaled(xy, k)[1]) < tol),
            None,
        )
        if decoy:
            fails.append(
                f"{W}x{H}: reported the {decoy} at ({at[0]:.0f},{at[1]:.0f})"
            )
        elif abs(at[0] - ball[0]) > tol or abs(at[1] - ball[1]) > tol:
            fails.append(
                f"{W}x{H}: departure at ({at[0]:.0f},{at[1]:.0f}) is not the "
                f"ball at {ball}"
            )
        elif abs(s["peak_time_sec"] - depart_at) > 0.5:
            fails.append(
                f"{W}x{H}: departure timed at {s['peak_time_sec']:.2f}s, "
                f"expected ~{depart_at:.2f}s"
            )
    return fails


def main() -> int:
    spec = importlib.util.spec_from_file_location("ai_tracer", AI_TRACER)
    ai = importlib.util.module_from_spec(spec)
    sys.modules["ai_tracer"] = ai
    spec.loader.exec_module(ai)

    fails = []
    with tempfile.TemporaryDirectory() as td:
        for W, H, ball_r in RESOLUTIONS:
            fails += run_one(ai, Path(td), W, H, ball_r)

    print()
    if fails:
        for f in fails:
            print(f"FAIL: {f}")
        return 1
    print("PASS: at every resolution, found the ball, timed the departure, "
          "and ignored all 5 decoys.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
