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


# How long the resting ball takes to reach full brightness, in frames.
# Short enough to be there when the viewer looks, long enough that it
# settles rather than pops.
_REST_FADE_FRAMES = 6


def render_comet(src: Path, out: Path, points, first_frame: int,
                 tail_frames: int = 15) -> bool:
    """Draw a comet along `points` onto `src`, writing `out`.

    `points` are in the SOURCE's frame numbering; `first_frame` is the
    source frame the cut segment starts on, so the two can be lined up
    without the caller renumbering anything.

    A comet, not a persistent line: the ball is in view for well under a
    second here and what the eye wants is the motion. The head sits on
    the ball and a short tail fades behind it, so the shot reads as
    "coming down THERE" rather than as a diagram.

    ONE CONTINUOUS STREAK, NOT A ROW OF DOTS. The obvious way to draw a
    tail is a blob per recent position, and it looks it: the ball moves
    tens of pixels per frame here, so those blobs land far apart and
    read as beads on a string rather than as something moving. It is
    drawn instead as a single tapered stroke through the same positions
    -- full width at the head, down to nothing at the far end -- built
    up in a soft mask so the edges fall off rather than ending in a hard
    line. The head is the wide end of that stroke, so there is no
    separate circle to see.

    HOW LONG THE TAIL RUNS is a look rather than a measurement: it is
    however many of the ball's recent positions the stroke is drawn
    through. Fifteen frames is about a third of a second on this
    camera — long enough to read as speed, short enough that it stays a
    comet instead of a line drawn down the frame.
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

    # Width of the head, in pixels. Everything else is a fraction of it.
    _r = max(3, int(round(W / 220)))
    # The stroke is walked at this many steps per frame-gap. The ball
    # covers tens of pixels between frames, and stepping along that gap
    # is what makes the width and the brightness change SMOOTHLY down
    # the tail instead of in visible bands.
    _SUB = 8
    # THE SAME BLUE AS THE TEE TRACER, taken from the tracer itself
    # rather than copied: one clip, one shot, one colour. The two
    # halves used to disagree -- a broadcast-blue line on the tee and a
    # warm-white streak on the green -- which reads as two graphics
    # rather than one flight followed to the ground.
    #
    # Two of the tracer's three tones are used the way it uses them: the
    # core blue through the tail, brightening to the pale-azure inner
    # highlight where the stroke is densest, which is the head. So the
    # comet has the tracer's own core-and-glow build rather than a flat
    # fill of its colour.
    from .ai_tracer import TRACER_CORE_BGR, TRACER_INNER_BGR

    _CORE = np.array(TRACER_CORE_BGR, dtype=np.float32)
    _HOT = np.array(TRACER_INNER_BGR, dtype=np.float32)
    # Where the ball came to rest, and when it got there: the last
    # plotted point. Everything after that frame gets the resting glow.
    _last_f = max(by_frame)
    _rest_xy = by_frame[_last_f]
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
            trail = [by_frame[g] for g in range(f - tail_frames, f + 1)
                     if g in by_frame]
            if trail:
                # ONE STROKE, TAIL TO HEAD. Drawn into a single-channel
                # mask so the taper is in the coverage rather than in
                # the colour -- which is what lets it be blurred into a
                # soft edge and composited once, instead of stacking
                # semi-transparent shapes on top of each other.
                mask = np.zeros((H, W), np.uint8)
                n = len(trail)
                px, py = trail[0]
                for k in range(1, n):
                    x1, y1 = trail[k - 1]
                    x2, y2 = trail[k]
                    for s in range(1, _SUB + 1):
                        u = s / float(_SUB)
                        cx = x1 + (x2 - x1) * u
                        cy = y1 + (y2 - y1) * u
                        # How far along the whole tail this step is:
                        # 0 at the oldest position, 1 at the ball.
                        t = ((k - 1) + u) / float(max(1, n - 1))
                        # Squared so the tail thins quickly behind the
                        # head and then lingers, the way a real one does.
                        w = max(1, int(round(_r * (0.12 + 0.88 * t * t))))
                        # Later (brighter, wider) steps are drawn last,
                        # so overlapping ones simply win.
                        cv2.line(mask, (int(round(px)), int(round(py))),
                                 (int(round(cx)), int(round(cy))),
                                 int(round(40 + 215 * (0.25 + 0.75 * t))),
                                 w * 2, cv2.LINE_AA)
                        px, py = cx, cy
                # A blur the width of the head: the stroke stops being a
                # shape with an outline and becomes a glow.
                _k = max(3, (_r | 1))
                mask = cv2.GaussianBlur(mask, (_k, _k), 0)
                # Composited over the stroke's own corner of the frame.
                # The rest is the untouched camera, and doing the float
                # work on a few thousand pixels instead of a million
                # keeps this off the critical path of a produce.
                _pad = 3 * _r + _k
                x0 = max(0, int(min(p[0] for p in trail)) - _pad)
                x1b = min(W, int(max(p[0] for p in trail)) + _pad)
                y0 = max(0, int(min(p[1] for p in trail)) - _pad)
                y1b = min(H, int(max(p[1] for p in trail)) + _pad)
                if x1b > x0 and y1b > y0:
                    roi = frame[y0:y1b, x0:x1b]
                    a = (mask[y0:y1b, x0:x1b].astype(np.float32)
                         / 255.0)[:, :, None]
                    # Core blue where the stroke is thin, the hot pale
                    # centre where it is dense: the tracer's own build.
                    col = _CORE + (_HOT - _CORE) * (a ** 2)
                    roi[:] = np.clip(
                        roi.astype(np.float32) * (1.0 - a) + col * a,
                        0, 255,
                    ).astype(np.uint8)
            elif _rest_xy is not None and f > _last_f:
                # THE BALL, SITTING WHERE IT FINISHED.
                #
                # The comet ends and the green half runs on for another
                # second or two, and until now that tail was of an empty
                # green -- the shot arrives and then vanishes, which is
                # the moment the viewer is actually looking for. A small
                # glow left on the last plotted point holds the answer
                # on screen: the ball is THERE.
                #
                # Drawn the same way the comet is, through a blurred
                # mask, so it reads as the same graphic settling rather
                # than a marker appearing. It fades UP over a few frames
                # for the same reason -- a hard cut-in looks like a
                # different object.
                _age = f - _last_f
                _in = min(1.0, _age / max(1.0, _REST_FADE_FRAMES))
                _rx, _ry = _rest_xy
                _pad2 = 4 * _r
                x0 = max(0, int(_rx) - _pad2)
                x1b = min(W, int(_rx) + _pad2)
                y0 = max(0, int(_ry) - _pad2)
                y1b = min(H, int(_ry) + _pad2)
                if x1b > x0 and y1b > y0:
                    m2 = np.zeros((H, W), np.uint8)
                    cv2.circle(m2, (int(round(_rx)), int(round(_ry))),
                               max(2, int(round(_r * 0.55))), 255, -1,
                               cv2.LINE_AA)
                    _k2 = max(3, ((_r * 2) | 1))
                    m2 = cv2.GaussianBlur(m2, (_k2, _k2), 0)
                    roi = frame[y0:y1b, x0:x1b]
                    a2 = ((m2[y0:y1b, x0:x1b].astype(np.float32) / 255.0)
                          * _in)[:, :, None]
                    col2 = _CORE + (_HOT - _CORE) * (a2 ** 2)
                    roi[:] = np.clip(
                        roi.astype(np.float32) * (1.0 - a2) + col2 * a2,
                        0, 255,
                    ).astype(np.uint8)
            vw.write(frame)
    finally:
        cap.release()
        vw.release()

    if not tmp.exists() or tmp.stat().st_size < 1000:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(out)
    return True
