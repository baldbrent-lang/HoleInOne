"""The ball's last few frames on the GREEN camera, and a comet on them.

The tee tracer answers "where did it go". This answers "watch it come
down": on the green camera the ball is close, large and moving fast
against a still background, so the frames just before it touches down
usually contain an obvious chain of blobs -- the one visible in the
landing scan as a run of numbered dots dropping into the green.

TWO THINGS MAKE THIS EASIER THAN THE TEE SIDE.

The end is known. The operator marks the landing frame and the landing
spot, so the search is not "find a flight somewhere in this clip" but
"walk backwards from THIS pixel on THIS frame and see whether a chain
leads into it". A backwards walk from a known point is a far smaller
question than a forward search, and it fails cleanly: no chain, no
tracer, which is exactly the "only when it finds an obvious path"
condition.

And the background is still. The green camera is bolted down looking at
grass; a plain frame difference against the previous frame is enough,
and the same one the landing scan already uses.

What comes back is drawn as a COMET rather than the tee's dashed line:
a bright head at the ball with a short tail fading behind it. The tee
tracer's job is to show a whole flight at once, so it persists; here the
ball is only in view for a moment and the interesting thing is the
motion itself.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

log = logging.getLogger("golfreelz.green_flight")

# How far back from the landing to look, in seconds. The ball is only
# recognisable on this camera for the last part of its descent -- before
# that it is a speck against trees -- so a long window buys noise, not
# track.
LOOK_BACK_SEC = 1.2

# A blob has to be ball-sized. Generous at the top end because a fast
# ball smears across several pixels in one exposure.
MIN_AREA, MAX_AREA = 2, 900

# Frame-diff threshold. Lower than the tee scan's: the green camera is
# closer and the ball is brighter against grass, but it is also often in
# shadow under trees.
DIFF_THRESH = 10

# The chain has to be at least this long to be worth drawing. Three
# points is a coincidence; six is a path.
MIN_CHAIN = 6


def _blobs(cur_g, prev_g, cv2, np):
    d = cv2.absdiff(cur_g, prev_g)
    _, th = cv2.threshold(d, DIFF_THRESH, 255, cv2.THRESH_BINARY)
    th = cv2.dilate(th, None, iterations=1)
    n, _lbl, stats, cents = cv2.connectedComponentsWithStats(th)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if MIN_AREA <= area <= MAX_AREA:
            out.append((float(cents[i][0]), float(cents[i][1]), area))
    return out


def find_path(video: Path, landing_frame: int, landing_xy, fps: float,
              look_back_sec: float = LOOK_BACK_SEC):
    """Walk back from the marked landing and return the ball's chain.

    Returns (points, reason). `points` is [{frame, x, y}, ...] in green
    pixels, oldest first and ending on the landing; or None with a
    sentence saying why, which the caller shows rather than silently
    drawing nothing.
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    try:
        lf = int(landing_frame)
        lx, ly = float(landing_xy[0]), float(landing_xy[1])
    except (TypeError, ValueError, IndexError):
        return None, "the landing frame or spot is not usable"

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None, "could not open the green video"
    try:
        _fps = float(fps or cap.get(cv2.CAP_PROP_FPS) or 30.0)
        first = max(0, lf - int(round(look_back_sec * _fps)))
        if lf - first < MIN_CHAIN:
            return None, "the landing is too near the start of the clip"

        # One sequential pass. Seeking per frame on a long clip is
        # slower than decoding straight through a second of video.
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, first - 1)))
        prev = None
        per_frame: dict[int, list] = {}
        for f in range(max(0, first - 1), lf + 1):
            ok, frame = cap.read()
            if not ok:
                break
            g = cv2.GaussianBlur(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
            if prev is not None and f >= first:
                per_frame[f] = _blobs(g, prev, cv2, np)
            prev = g
    finally:
        cap.release()

    if not per_frame:
        return None, "no frames were readable before the landing"

    # BACKWARDS FROM THE LANDING. The operator's mark is the one point
    # known to be the ball, so it anchors the walk; each earlier frame
    # contributes the blob that best continues the motion already
    # established. Going forwards instead would have to guess a start.
    chain = [{"frame": lf, "x": lx, "y": ly}]
    step = None                      # (dx, dy) of the last accepted hop
    for f in range(lf - 1, first - 1, -1):
        cands = per_frame.get(f) or []
        if not cands:
            # One missing frame is a blink -- a dark background, a blob
            # merged with a shadow. Two in a row is the chain ending.
            if len(chain) >= 2 and chain[-1]["frame"] - f > 2:
                break
            continue
        cx, cy = chain[-1]["x"], chain[-1]["y"]
        # How far the ball could have come in one frame. Before there is
        # a step to go on, allow a generous radius; after, expect
        # something close to the same hop again.
        reach = 90.0 if step is None else max(18.0, 2.2 * math.hypot(*step))
        best, best_score = None, None
        for bx, by, area in cands:
            dx, dy = bx - cx, by - cy
            dist = math.hypot(dx, dy)
            if dist > reach or dist < 1.0:
                continue
            score = dist
            if step is not None:
                # Direction has to agree with the hop before it: the
                # ball keeps coming from the same way. Reversals are
                # grass, shadows and the golfer's shadow moving.
                _sl = math.hypot(*step) or 1.0
                cosang = (dx * step[0] + dy * step[1]) / (dist * _sl)
                if cosang < 0.55:
                    continue
                # Prefer a hop the same LENGTH as the last one as well
                # as the same direction -- a still background throws up
                # near-duplicates otherwise.
                score = abs(dist - _sl) * 2.0 + dist * 0.2
            if best_score is None or score < best_score:
                best_score, best = score, (bx, by, dx, dy)
        if best is None:
            if len(chain) >= 2 and chain[-1]["frame"] - f > 2:
                break
            continue
        chain.append({"frame": f, "x": best[0], "y": best[1]})
        step = (best[2], best[3])

    chain.reverse()
    if len(chain) < MIN_CHAIN:
        return None, (
            f"only {len(chain)} frames chain into the landing — no obvious "
            f"ball path on the green camera for this swing"
        )
    # A chain that never goes anywhere is the background flickering in
    # place, not a ball.
    travel = math.hypot(chain[-1]["x"] - chain[0]["x"],
                        chain[-1]["y"] - chain[0]["y"])
    if travel < 40.0:
        return None, (
            f"the chain only travels {travel:.0f}px — that is background "
            f"flicker, not a ball"
        )
    return [{"frame": int(p["frame"]), "x": round(float(p["x"]), 1),
             "y": round(float(p["y"]), 1)} for p in chain], None


def render_comet(src: Path, out: Path, points, first_frame: int,
                 tail_frames: int = 8) -> bool:
    """Draw a comet along `points` onto `src`, writing `out`.

    `points` are in the SOURCE's frame numbering; `first_frame` is the
    source frame the cut segment starts on, so the two can be lined up
    without the caller renumbering anything.

    A comet, not a persistent line: the ball is in view for well under a
    second here and what the eye wants is the motion. The head sits on
    the ball and a short tail fades behind it, so the shot reads as
    "coming down THERE" rather than as a diagram.
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    by_frame = {int(p["frame"]): (float(p["x"]), float(p["y"]))
                for p in (points or [])}
    if not by_frame:
        return False

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return False
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if W <= 0 or H <= 0:
        cap.release()
        return False
    tmp = out.with_suffix(".comet.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (W, H))
    if not vw.isOpened():
        cap.release()
        return False

    _r = max(3, int(round(W / 220)))
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            f = first_frame + idx
            idx += 1
            # The tail is the ball's own recent positions, so it curves
            # with the flight instead of being a straight streak.
            trail = [(g, by_frame[g]) for g in range(f - tail_frames, f + 1)
                     if g in by_frame]
            if trail:
                overlay = frame.copy()
                for k, (g, (x, y)) in enumerate(trail):
                    age = (len(trail) - 1 - k)
                    t = 1.0 - (age / float(max(1, tail_frames)))
                    if t <= 0.02:
                        continue
                    cv2.circle(overlay, (int(round(x)), int(round(y))),
                               max(1, int(round(_r * (0.35 + 0.65 * t)))),
                               (255, 245, 225), -1, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                # The head again, opaque and ringed, so it stays the
                # brightest thing on the frame over pale bunker sand.
                hx, hy = trail[-1][1]
                cv2.circle(frame, (int(round(hx)), int(round(hy))),
                           _r + 1, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (int(round(hx)), int(round(hy))),
                           _r + 3, (60, 160, 255), 1, cv2.LINE_AA)
            vw.write(frame)
    finally:
        cap.release()
        vw.release()

    if not tmp.exists() or tmp.stat().st_size < 1000:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(out)
    return True
