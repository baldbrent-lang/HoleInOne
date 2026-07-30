"""Debug3 — isolate the ball by BLOB SHAPE and TRACKING, not heat geometry.

Debug2 reasons about the shape the swing draws in a motion composite: the
club fan, the band above it, chains that rise and point back at the ball.
That works, but everything it looks at is an accumulation, so a single
frame's evidence is never examined on its own terms.

This path does the opposite. It asks, per frame, "which connected blobs are
ball-sized and not part of the golfer?", then links those detections over
time and fits a flight to them. Nothing here looks at a composite:

  A. MOG2                 per-frame foreground, model warmed up before the
                          window so the golfer's body is already background
                          where it has not moved.
  B. COMPONENT FILTER      connected components. The big ones ARE the golfer
                          (and their shadow) — they become a mask, not a
                          detection. What survives is small and off-body.
  C. AREA FILTER           keep only ball-sized blobs. A golf ball at these
                          distances is a handful of pixels; anything much
                          larger is a limb, a shadow edge or a leaf.
  D. TRACK                 link detections across frames with a
                          constant-velocity predictor and a nearest-
                          neighbour gate — a Kalman filter without the
                          tuning, which for a ballistic target over 2
                          seconds is the same answer.
  E. RANSAC PARABOLA       a ball's image path is x linear in time, y
                          quadratic. Fit that with RANSAC so a few bad
                          links cannot bend the curve, then keep the fit
                          with the most support that also rises and points
                          back at the ball at the impact frame.

Every stage reports its counts and its drop reasons, because the useful
question is almost never "did it work" but "which filter ate the ball".
"""
from __future__ import annotations

import itertools
import logging
import math
from pathlib import Path

log = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    HAS_CV = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    np = None  # type: ignore
    HAS_CV = False

# Shared with Debug2 so the two are directly comparable on the same window.
from .debug2 import WIN_POST, WIN_PRE, _label  # noqa: E402

# A golf ball is TINY, but "20-30 pixels of area" is a 1080p number and it
# does not travel. Measured on a synthetic flight at 720p: a ball 7px across
# is ~29px of area geometrically, but MOG2's foreground for a MOVING object
# plus the morphological open bleeds it to 35-50px, so a 30px cap detected
# the ball in 11 of 101 frames. At 60 it was 100 of 101.
#
# So the cap is derived from the biggest plausible ball DIAMETER at this
# frame height, with headroom for that bleed. BALL_AREA_STRICT is kept only
# so the report can say how many blobs a 30px cap would have kept — useful
# for judging whether the derived cap is letting junk in.
BALL_AREA_MIN = 2
BALL_AREA_STRICT = 30
BALL_DIAM_FRAC = 0.014        # of frame height
BALL_AREA_BLEED = 1.6         # MOG2 + open, vs the geometric disc
MIN_KEPT_FOR_TRACKING = 6

# Detections kept per frame. This used to be 10, keeping the SMALLEST, on
# the reasoning that a ball is small. On a real wide shot that threw the
# ball away: the golfer is tiny in frame so the ball is only a few pixels,
# and there are dozens of even smaller movers -- grass shimmer, leaf
# flicker, compression noise at the frame border -- that outrank it. One
# swing dropped 1913 detections to the cap and kept 10 blobs sitting on the
# frame edges. The sequential tracker is cheap enough not to need a tight
# cap, so this is now a runaway guard rather than a filter, and what it
# keeps is the SMALLEST-AREA blobs only as a last resort.
MAX_DETS_PER_FRAME = 40

# A pixel that is foreground in more than this fraction of the window is
# not a ball. The ball crosses any given pixel ONCE -- about 1% of a 106
# frame window -- while the golfer's body occupies its own region for the
# whole swing and wind-blown foliage re-lights the same leaves over and
# over. This one test removes both, and unlike an area threshold it does
# not care that MOG2 renders a person as scattered edge fragments rather
# than a silhouette.
BUSY_PIXEL_FRAC = 0.20

# Foreground is dilated by this much before the occupancy count, so the
# measure is "is this REGION always active" rather than "is this exact
# pixel". Without it the mask comes out empty on real motion.
BUSY_DILATE_PX = 9

# Kernel used ONLY to decide what is the golfer. Closing the foreground
# joins a person's separate edge fragments -- collar, hems, limb
# boundaries -- into one blob big enough to recognise. The ball detection
# itself still runs on the un-closed mask, or closing would merge the ball
# into whatever it flies past.
GOLFER_CLOSE_PX = 13

# Ignore this many pixels around the frame border. Compression and sensor
# noise light up the outermost columns every frame, and those blobs are
# both plentiful and perfectly ball-sized. A ball that is genuinely in the
# outermost few pixels has already left the shot.
EDGE_MARGIN_PX = 6

MAX_TRACKS_TESTED = 120

# Gate for a track's FIRST link, in multiples of r. A track with one point
# has no velocity yet, so its prediction is "stay put" -- and a golf ball
# covers far more ground than the steady-state gate allows. Measured on real
# footage: consecutive flight detections were ~60px apart at 1080p while the
# gate was 31px, so the ball could never acquire and only slow junk ever
# formed tracks. Wide here, tight once a velocity exists.
ACQUIRE_GATE_R = 12.0

# Extra gate slack proportional to speed. A prediction's error grows with
# how fast the thing is moving, so a fixed radius is either too tight for
# the ball or too loose for everything else.
GATE_SPEED_FRAC = 0.25

# A track this good does not need the ball's permission. The aim test exists
# to reject noise, but a long run of detections lying on a parabola, rising
# most of the frame, IS a ball flight -- it is stronger evidence than a rest
# detection that may have locked onto a white shoe. Observed: a 33-point
# track with 26 inliers rising 374px was thrown out for aiming 262px from a
# ball marker sitting on the golfer's trainer. When a track clears these,
# aim becomes a reported disagreement rather than a veto, and the panel says
# the ball position is the thing to doubt.
SELF_EVIDENT_INLIERS = 8
# Residual budget for "this is obviously a flight". A flat 4px came from
# synthetic footage, where the fit lands at 0.9px because the ball is a
# clean disc on a clean background. A real ball smears at speed and its
# MOG2 centroid jitters, so the honest test is whether the residual is
# small COMPARED TO THE TRACK -- scale-free, and it does not punish a long
# flight for being long.
SELF_EVIDENT_RMS_R = 0.6        # of the ball scale r
SELF_EVIDENT_RMS_SPAN = 0.02    # of the track's own span
SELF_EVIDENT_RISE_FRAC = 0.15      # of frame height
SELF_EVIDENT_SPAN_FRAC = 0.15      # of frame width

# Seed points for the parabola RANSAC. 14 gives C(14,3) = 364 candidate
# fits, which is fast and plenty for a curve with three parameters.
RANSAC_SEED_PTS = 14

# MOG2 needs to have seen the scene before the window or the golfer's whole
# body reads as foreground on frame one.
WARMUP_FRAMES = 40

# Work at or below this height. MOG2 plus a connectedComponentsWithStats
# label array per frame at 1080p is ~25MB of allocation per frame, for every
# frame of every candidate's window -- enough to get a container killed
# mid-request, which surfaces as a 502 with no log line of its own. A ball
# is 5-7px across at 720p and was detected in 100 of 101 frames there, so
# there is nothing to gain from the extra resolution.
WORK_HEIGHT = 720


# ── stages A-C: MOG2, drop the golfer, keep ball-sized blobs ────────────

def ball_area_cap(frame_h: int) -> tuple[float, float]:
    """(max area, max bbox side) for a ball at this frame height."""
    d = max(6.0, BALL_DIAM_FRAC * float(frame_h))
    return BALL_AREA_BLEED * (math.pi / 4.0) * d * d, 3.0 * d



def body_box_from_pose(head_xy, feet_xy, frame_w: int, frame_h: int,
                       pad_frac: float = 0.40):
    """(x0, y0, x1, y1) covering the golfer, in FULL-RES coordinates.

    Spans head to feet vertically with a little headroom for the club at the
    top of the backswing, and pad_frac of a body height either side for the
    arms. Returns None without pose.
    """
    if not (head_xy and feet_xy and len(head_xy) == 2 and len(feet_xy) == 2):
        return None
    hx, hy = float(head_xy[0]), float(head_xy[1])
    fx, fy = float(feet_xy[0]), float(feet_xy[1])
    body = max(40.0, fy - hy)
    cx = 0.5 * (hx + fx)
    return (
        max(0.0, cx - pad_frac * body),
        max(0.0, hy - 0.15 * body),
        min(float(frame_w), cx + pad_frac * body),
        min(float(frame_h), fy + 0.08 * body),
    )


def detect_ball_blobs(
    input_path: Path,
    f0: int,
    f1: int,
    max_area: int | None = None,
    min_area: int = BALL_AREA_MIN,
    max_per_frame: int = MAX_DETS_PER_FRAME,
    body_box=None,
    debug_dir: Path | None = None,
    debug_prefix: str = "d3",
) -> dict:
    """Per-frame ball-sized detections over frames f0..f1.

    Returns {ok, dets, stats, areas, images, reason}. Never raises.

    dets is [{frame, x, y, area, w, h}]. stats counts what each filter
    threw away, which is the only way to tell "no ball in shot" apart from
    "the area cap was wrong".
    """
    out = {
        "ok": False, "dets": [], "reason": None,
        "stats": {"frames": 0, "components": 0, "golfer": 0,
                  "on_golfer": 0, "too_small": 0, "too_big": 0, "kept": 0},
        "areas": [], "images": {}, "max_area": max_area,
    }
    if not HAS_CV:
        out["reason"] = "opencv not installed"
        return out
    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            out["reason"] = "could not open video"
            return out
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Downscale for the pass, then report detections in FULL-RES
        # coordinates so the caller never has to know this happened.
        scale = min(1.0, float(WORK_HEIGHT) / float(max(1, src_h)))
        w = max(1, int(round(src_w * scale)))
        h = max(1, int(round(src_h * scale)))
        out["scale"] = round(scale, 4)
        out["work_frame"] = [w, h]
        inv = 1.0 / scale if scale else 1.0
        # "Big" = the golfer. Scaled to the frame so it survives a change of
        # camera or resolution.
        golfer_area = max(400.0, 0.0008 * float(w) * float(h))
        golfer_h = 0.12 * float(h)
        # The ball-size window, derived from the frame unless overridden.
        # POSE BODY BOX, in working coordinates. This is the only body
        # exclusion that does not depend on how MOG2 happens to render a
        # person. Area and height thresholds assume the body arrives as one
        # big component; on real footage it does not -- a person in flat
        # clothing has no interior motion, so MOG2 returns scattered edge
        # fragments and one swing produced under one "golfer" component per
        # frame. Pose already told us where the golfer is, so use that.
        #
        # It costs the first couple of detections after impact, while the
        # ball is still inside the box. That is affordable: the ball clears
        # it in two or three frames, and the parabola run back to the impact
        # frame recovers the launch point to about a pixel.
        bbox = None
        if body_box and len(body_box) == 4:
            bbox = tuple(int(round(v * scale)) for v in body_box)
            out["body_box"] = list(bbox)
        _cap, _side = ball_area_cap(h)
        if max_area is None:
            max_area = int(round(_cap))
        out["max_area"] = int(max_area)
        out["max_side"] = int(round(_side))
        n_strict = 0            # what a flat 30px cap would have kept

        mog = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=25, detectShadows=False,
        )
        start = max(0, int(f0) - WARMUP_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        kernel = np.ones((3, 3), np.uint8)
        close_k = np.ones((GOLFER_CLOSE_PX, GOLFER_CLOSE_PX), np.uint8)
        busy_k = np.ones((BUSY_DILATE_PX, BUSY_DILATE_PX), np.uint8)
        # How often each pixel is foreground across the window.
        occ = np.zeros((h, w), np.uint16)

        base = None                     # first frame of the window, for draws
        best_frame_img = None           # frame with the most kept blobs
        best_frame_n = -1
        best_frame_draw = None
        dets: list[dict] = []
        areas: list[int] = []

        for f in range(start, int(f1) + 1):
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            if scale < 1.0:
                fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
            fg = mog.apply(fr)
            if f < int(f0):
                continue                # warm-up only
            if base is None:
                base = fr.copy()
            out["stats"]["frames"] += 1
            # Open to kill single-pixel speckle without eating a 3px ball.
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
            # Accumulate a DILATED foreground. Per-pixel occupancy measured
            # nothing: a body fragment wobbling a few pixels never lands on
            # the same pixel often enough, so the mask came out empty. What
            # is actually constant is the REGION -- the golfer's outline
            # keeps re-lighting slightly different pixels in the same place.
            # A ball crossing that region still contributes its own footprint
            # once, so this separates "always busy here" from "something
            # passed through".
            occ += (cv2.dilate(fg, busy_k) > 0).astype(np.uint16)
            n, lab, stats, cent = cv2.connectedComponentsWithStats(fg, 8)
            out["stats"]["components"] += max(0, n - 1)
            # Golfer detection runs on a CLOSED copy so a fragmented body
            # reads as one blob; the ball pass below keeps the raw mask.
            fg_body = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, close_k)
            nb, labb, statsb, _cb = cv2.connectedComponentsWithStats(
                fg_body, 8,
            )

            # Pass 1: find the golfer. Big blobs are not candidates, they are
            # an exclusion zone — a ball crossing the body cannot be told
            # from a moving sleeve, so we do not try.
            golfer = np.zeros((h, w), np.uint8)
            big = False
            for i in range(1, nb):
                a = int(statsb[i, cv2.CC_STAT_AREA])
                bh = int(statsb[i, cv2.CC_STAT_HEIGHT])
                if a >= golfer_area or bh >= golfer_h:
                    golfer[labb == i] = 255
                    out["stats"]["golfer"] += 1
                    big = True
            if big:
                golfer = cv2.dilate(golfer, np.ones((21, 21), np.uint8))

            # Pass 2: the small stuff.
            kept_this = []
            for i in range(1, n):
                a = int(stats[i, cv2.CC_STAT_AREA])
                bw = int(stats[i, cv2.CC_STAT_WIDTH])
                bh = int(stats[i, cv2.CC_STAT_HEIGHT])
                if a >= golfer_area or bh >= golfer_h:
                    continue
                cx, cy = float(cent[i][0]), float(cent[i][1])
                areas.append(a)
                if (cx < EDGE_MARGIN_PX or cy < EDGE_MARGIN_PX
                        or cx > w - EDGE_MARGIN_PX
                        or cy > h - EDGE_MARGIN_PX):
                    out["stats"]["on_edge"] = (
                        out["stats"].get("on_edge", 0) + 1
                    )
                    continue
                if bbox and (bbox[0] <= cx <= bbox[2]
                             and bbox[1] <= cy <= bbox[3]):
                    out["stats"]["in_body_box"] = (
                        out["stats"].get("in_body_box", 0) + 1
                    )
                    continue
                if golfer[min(h - 1, max(0, int(cy))),
                          min(w - 1, max(0, int(cx)))]:
                    out["stats"]["on_golfer"] += 1
                    continue
                if a < min_area:
                    out["stats"]["too_small"] += 1
                    continue
                if a > max_area:
                    out["stats"]["too_big"] += 1
                    continue
                # A limb or a shadow edge can be small in area and still be
                # long and thin. A ball is neither.
                if bw > _side or bh > _side:
                    out["stats"]["too_long"] = (
                        out["stats"].get("too_long", 0) + 1
                    )
                    continue
                if a <= BALL_AREA_STRICT:
                    n_strict += 1
                kept_this.append({
                    "frame": f, "x": cx * inv, "y": cy * inv, "area": a,
                    "w": bw, "h": bh,
                    # working-frame position, for the debug draws
                    "wx": cx, "wy": cy,
                })

            if len(kept_this) > max_per_frame:
                kept_this.sort(key=lambda d: d["area"])
                out["stats"]["over_cap"] = (
                    out["stats"].get("over_cap", 0)
                    + len(kept_this) - max_per_frame
                )
                kept_this = kept_this[:max_per_frame]
                out["stats"]["kept"] -= 0     # counted below instead
            dets.extend(kept_this)
            out["stats"]["kept"] += len(kept_this)

            if len(kept_this) > best_frame_n:
                best_frame_n = len(kept_this)
                best_frame_img = fr.copy()
                best_frame_draw = (kept_this, golfer.copy(), f)

        cap.release()

        # Drop anything sitting in a pixel that is foreground most of the
        # time. Done after the loop so the threshold is a fraction of the
        # frames actually read rather than the frames requested.
        n_frames = max(1, out["stats"]["frames"])
        busy = (occ >= max(2, int(BUSY_PIXEL_FRAC * n_frames)))
        out["stats"]["busy_px"] = int(busy.sum())
        kept_dets = []
        for d in dets:
            iy = min(h - 1, max(0, int(d["wy"])))
            ix = min(w - 1, max(0, int(d["wx"])))
            if busy[iy, ix]:
                out["stats"]["in_busy_region"] = (
                    out["stats"].get("in_busy_region", 0) + 1
                )
                continue
            kept_dets.append(d)
        dets = kept_dets
        out["stats"]["kept"] = len(dets)
        out["busy_mask"] = busy

        out["dets"] = dets
        out["areas"] = areas
        out["n_at_strict_cap"] = n_strict
        if base is None:
            out["reason"] = "could not read the window"
            return out
        out["ok"] = True
        s = out["stats"]
        out["reason"] = (
            f"{s['kept']} ball-sized blob(s) kept from {s['components']} "
            f"component(s) over {s['frames']} frames "
            f"(area {min_area}-{max_area}px, max side {int(_side)}px; "
            f"dropped {s['golfer']} golfer, {s.get('in_body_box', 0)} in "
            f"the pose body box, {s['on_golfer']} on the golfer, "
            f"{s['too_big']} too big, {s.get('too_long', 0)} too long, "
            f"{s['too_small']} too small, {s.get('on_edge', 0)} on the "
            f"frame border, {s.get('over_cap', 0)} over the "
            f"{max_per_frame}/frame cap, {s.get('in_busy_region', 0)} in a "
            f"busy region -- pixels lit in >={int(BUSY_PIXEL_FRAC * 100)}% "
            f"of frames, which is the golfer and the moving trees). "
            f"A flat {BALL_AREA_STRICT}px cap would have kept {n_strict}."
        )

        if debug_dir is not None:
            # Image 1: one frame, fully classified. This is the frame to look
            # at when the counts look wrong.
            if best_frame_draw is not None:
                img = best_frame_img
                kept_this, gmask, fno = best_frame_draw
                img[gmask > 0] = (
                    0.6 * img[gmask > 0]
                    + 0.4 * np.array([40, 40, 220])
                ).astype(np.uint8)
                for d in kept_this:
                    cv2.circle(img, (int(d["wx"]), int(d["wy"])),
                               max(8, int(0.012 * h)), (0, 255, 0), 2,
                               cv2.LINE_AA)
                _label(
                    img,
                    f"frame {fno}: red = golfer mask (excluded), green = "
                    f"ball-sized blobs kept ({len(kept_this)}). "
                    f"area window {min_area}-{max_area}px",
                )
                nm = f"{debug_prefix}-frame.jpg"
                cv2.imwrite(str(Path(debug_dir) / nm), img,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 86])
                out["images"]["frame"] = nm

            # Image 2: every kept detection over the window, coloured by
            # time. A flight shows as a coherent colour ramp; noise does not.
            img = base.copy()
            # Show the excluded region, since "why was my flight dropped"
            # is answered by looking at what this covers.
            if bbox:
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              (0, 140, 255), 2)
            _b = out.get("busy_mask")
            if _b is not None:
                img[_b] = (0.55 * img[_b]
                           + 0.45 * np.array([40, 40, 200])).astype(np.uint8)
            span = max(1, int(f1) - int(f0))
            for d in dets:
                t = (d["frame"] - int(f0)) / span
                col = (int(255 * (1 - t)), int(80 + 100 * t), int(255 * t))
                cv2.circle(img, (int(d["wx"]), int(d["wy"])), 4, col, -1,
                           cv2.LINE_AA)
            _label(
                img,
                f"all {len(dets)} kept detections, f{f0}-{f1}. "
                f"blue = early, orange = late. red tint = busy region "
                f"(lit in >={int(BUSY_PIXEL_FRAC * 100)}% of frames -- the "
                f"golfer and the moving trees), excluded",
            )
            nm = f"{debug_prefix}-dets.jpg"
            cv2.imwrite(str(Path(debug_dir) / nm), img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 86])
            out["images"]["dets"] = nm
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("debug3 detect_ball_blobs failed: %s", exc)
        out["reason"] = f"failed: {exc}"
        return out


# ── stage D: link detections into tracks ───────────────────────────────

def build_tracks(
    dets: list,
    r: float,
    max_gap: int = 4,
    min_len: int = 3,
) -> list:
    """Link detections into constant-velocity tracks, one track per object.

    A SEQUENTIAL multi-target tracker: walk the frames in order, predict
    where each open track should be, associate the nearest detection inside
    a gate that widens when a frame is missed, and start a new track from
    anything left over. Tracks that go unmatched for longer than max_gap are
    closed.

    This replaced a version that seeded a track from every pair of
    detections and extended each one. That was combinatorial: on real
    footage a single object produced hundreds of overlapping near-duplicate
    tracks -- 2132 of them on one swing -- and since the caller can only
    afford to fit the first N, the real flight was pushed off the end of the
    list by copies of noise. One object should yield one track.

    Established tracks associate FIRST (longest first), so a two-frame piece
    of noise cannot steal a detection from a flight that has been running
    for thirty frames.

    Velocity is smoothed as the track grows, so one noisy detection bends
    the prediction a little rather than throwing it off. That plus the
    widening gate is what a Kalman filter would give a ballistic target,
    without the covariance tuning, and every accepted link is a distance you
    can print.

    Returns tracks sorted longest-first, each {points, span_px, rise_px}.
    """
    pool = sorted(
        [d for d in (dets or [])],
        key=lambda d: (d["frame"], d["x"]),
    )
    if len(pool) < min_len:
        return []
    by_frame: dict[int, list] = {}
    for d in pool:
        by_frame.setdefault(int(d["frame"]), []).append(d)

    open_tracks: list[dict] = []
    closed: list[dict] = []
    for f in sorted(by_frame):
        cands = by_frame[f]
        taken: set[int] = set()
        # Longest first: seniority decides who gets an ambiguous detection.
        open_tracks.sort(key=lambda t: -len(t["pts"]))
        for tr in open_tracks:
            last = tr["pts"][-1]
            df = f - last["frame"]
            if df <= 0:
                continue
            px = last["x"] + tr["vx"] * df
            py = last["y"] + tr["vy"] * df
            if len(tr["pts"]) < 2:
                # No velocity yet: acquisition, not tracking.
                gate = ACQUIRE_GATE_R * r * df
            else:
                speed = math.hypot(tr["vx"], tr["vy"])
                gate = (1.5 + 0.9 * df) * r + GATE_SPEED_FRAC * speed * df
            best_i = None
            best_d = None
            for i, c in enumerate(cands):
                if i in taken:
                    continue
                dist = math.hypot(c["x"] - px, c["y"] - py)
                if dist <= gate and (best_d is None or dist < best_d):
                    best_i, best_d = i, dist
            if best_i is None:
                continue
            taken.add(best_i)
            c = cands[best_i]
            nvx = (c["x"] - last["x"]) / df
            nvy = (c["y"] - last["y"]) / df
            if len(tr["pts"]) == 1:
                tr["vx"], tr["vy"] = nvx, nvy      # first link sets it
            else:
                tr["vx"] = 0.5 * tr["vx"] + 0.5 * nvx
                tr["vy"] = 0.5 * tr["vy"] + 0.5 * nvy
            tr["pts"].append(c)
        # Retire anything that has gone quiet.
        still = []
        for tr in open_tracks:
            if f - tr["pts"][-1]["frame"] > max_gap:
                closed.append(tr)
            else:
                still.append(tr)
        open_tracks = still
        # Everything unclaimed opens a track of its own.
        for i, c in enumerate(cands):
            if i not in taken:
                open_tracks.append({"pts": [c], "vx": 0.0, "vy": 0.0})
    closed.extend(open_tracks)

    tracks: list[dict] = []
    for tr in closed:
        pts = tr["pts"]
        if len(pts) < min_len:
            continue
        span = math.hypot(pts[-1]["x"] - pts[0]["x"],
                          pts[-1]["y"] - pts[0]["y"])
        # A track that never went anywhere is a piece of background flicker
        # rediscovered every frame, not an object in motion.
        if span < 2.0 * r:
            continue
        tracks.append({
            "points": [
                {"frame": int(p["frame"]), "x": int(p["x"]),
                 "y": int(p["y"]), "area": int(p["area"])}
                for p in pts
            ],
            "span_px": round(span, 1),
            "rise_px": round(pts[0]["y"] - pts[-1]["y"], 1),
        })
    tracks.sort(key=lambda t: -len(t["points"]))
    return tracks


# ── stage E: RANSAC parabola ───────────────────────────────────────────

def ransac_parabola(
    points: list,
    impact_frame: int,
    ball_xy=None,
    tol: float | None = None,
    r: float = 12.0,
) -> dict:
    """Fit x = a·t + b, y = c·t² + d·t + e by RANSAC over `points`.

    A ball's image path really is this: horizontal speed is near constant
    over a couple of seconds, vertical is gravity plus foreshortening. So
    the fit is the physics, not a curve-fitting convenience — and RANSAC
    means a couple of bad links cannot bend it.

    Seeds are 3-point subsets of an EVENLY SPACED SUBSAMPLE of the track
    rather than random draws. Deterministic, so re-running a swing shows the
    operator what they saw last time; and bounded, because a 100-point track
    has C(100,3) = 161,700 subsets and a polyfit each is minutes of work per
    track. Spacing the seeds also picks better ones: three points spread
    across the flight constrain a parabola far better than three adjacent
    ones. Inliers are always counted against every point, not the subsample.

    Returns {ok, n_inliers, rms_px, inliers, outliers, at_impact, aim_px,
    coef, reason}.
    """
    out = {
        "ok": False, "n_inliers": 0, "rms_px": None, "inliers": [],
        "outliers": [], "at_impact": None, "aim_px": None, "reason": None,
    }
    if not HAS_CV or len(points) < 3:
        out["reason"] = f"need 3+ points, have {len(points)}"
        return out
    if tol is None:
        tol = max(6.0, 1.2 * r)
    # X IS NOT LINEAR IN TIME. The model was x linear, y quadratic -- true
    # for a projectile in the world, but this is an IMAGE. A ball flying
    # away from the camera covers less and less image width per frame, so
    # its x decelerates through perspective alone. Forcing x straight makes
    # the fit split the difference, and the frames where the ball moves
    # fastest across the frame -- the first ones after the strike -- get
    # thrown out as outliers. Those are the points closest to the ball and
    # the ones the launch extrapolation most needs.
    #
    # Fit x quadratic once there are enough points to support it. Same rule
    # the production tracer already uses in _robust_quadratic_fit.
    x_deg = 2 if len(points) >= 6 else 1
    t = np.array([p["frame"] for p in points], float)
    x = np.array([p["x"] for p in points], float)
    y = np.array([p["y"] for p in points], float)
    best = None
    m = len(points)
    if m <= RANSAC_SEED_PTS:
        seed_idx = list(range(m))
    else:
        seed_idx = sorted({
            int(round(v)) for v in
            np.linspace(0, m - 1, RANSAC_SEED_PTS)
        })
    for idx in itertools.combinations(seed_idx, 3):
        ti, xi, yi = t[list(idx)], x[list(idx)], y[list(idx)]
        if len(set(ti.tolist())) < 3:
            continue
        try:
            cx = np.polyfit(ti, xi, x_deg)
            cy = np.polyfit(ti, yi, 2)
        except Exception:  # noqa: BLE001
            continue
        dx = x - np.polyval(cx, t)
        dy = y - np.polyval(cy, t)
        dist = np.hypot(dx, dy)
        inl = dist <= tol
        n_inl = int(inl.sum())
        if n_inl < 3:
            continue
        rms = float(np.sqrt(float(np.mean(dist[inl] ** 2))))
        key = (n_inl, -rms)
        if best is None or key > best[0]:
            best = (key, inl, cx, cy, rms)
    if best is None:
        out["reason"] = f"no 3-point fit held {3} points within {tol:.0f}px"
        return out
    _key, inl, cx, cy, _rms = best
    # Refit on the inliers — the 3 seed points chose the model, they should
    # not define it.
    try:
        _d = 2 if int(inl.sum()) >= 6 else 1
        cx = np.polyfit(t[inl], x[inl], _d)
        cy = np.polyfit(t[inl], y[inl], 2) if int(inl.sum()) >= 3 else cy
    except Exception:  # noqa: BLE001
        pass
    dist = np.hypot(x - np.polyval(cx, t), y - np.polyval(cy, t))
    inl = dist <= tol
    out["n_inliers"] = int(inl.sum())
    out["rms_px"] = round(float(np.sqrt(float(np.mean(dist[inl] ** 2)))), 2)
    out["inliers"] = [points[i] for i in range(len(points)) if inl[i]]
    out["outliers"] = [points[i] for i in range(len(points)) if not inl[i]]
    ix = float(np.polyval(cx, float(impact_frame)))
    iy = float(np.polyval(cy, float(impact_frame)))
    out["at_impact"] = [int(ix), int(iy)]
    out["coef"] = {"x": [float(v) for v in cx], "y": [float(v) for v in cy]}
    if ball_xy and len(ball_xy) == 2:
        out["aim_px"] = round(
            math.hypot(ix - float(ball_xy[0]), iy - float(ball_xy[1])), 1,
        )
    out["ok"] = True
    out["x_degree"] = int(len(cx) - 1)
    out["reason"] = (
        f"{out['n_inliers']}/{len(points)} points on a parabola "
        f"(x deg {out['x_degree']}), rms {out['rms_px']}px"
        + (f"; run back to impact it lands {out['aim_px']}px from the ball"
           if out["aim_px"] is not None else "")
    )
    return out


def rest_check_image(
    input_path: Path,
    frame_no: int,
    ball_xy,
    r: float,
    debug_dir: Path,
    debug_prefix: str = "d3rest",
    zoom: int = 6,
) -> str | None:
    """A frame from BEFORE impact with our ball estimate ringed on it.

    The point is to look at the ball sitting there. Every other panel shows
    frames from the swing itself, by which time the ball has gone -- and a
    ball position can look plausible on an empty patch of turf. Ringing it
    on a frame where the ball is still present is the only check that
    settles it by eye.

    Returns the written filename, or None.
    """
    if not HAS_CV or not ball_xy or len(ball_xy) != 2:
        return None
    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_no)))
        ok, fr = cap.read()
        cap.release()
        if not ok or fr is None:
            return None
        h, w = fr.shape[:2]
        bx, by = int(ball_xy[0]), int(ball_xy[1])
        cv2.circle(fr, (bx, by), max(10, int(2.0 * r)), (0, 255, 0), 2,
                   cv2.LINE_AA)
        # A zoomed inset, because a golf ball at this range is a few pixels
        # and "is it there" is not answerable at 1:1.
        pad = int(max(30.0, 6.0 * r))
        x0, y0 = max(0, bx - pad), max(0, by - pad)
        x1, y1 = min(w, bx + pad), min(h, by + pad)
        if x1 - x0 > 8 and y1 - y0 > 8:
            crop = cv2.resize(fr[y0:y1, x0:x1], None, fx=zoom, fy=zoom,
                              interpolation=cv2.INTER_NEAREST)
            ch, cw = crop.shape[:2]
            ch, cw = min(ch, h // 2), min(cw, w // 2)
            crop = crop[:ch, :cw]
            fr[0:ch, w - cw:w] = crop
            cv2.rectangle(fr, (w - cw, 0), (w - 1, ch), (0, 255, 0), 2)
        _label(fr, f"frame {frame_no} -- BEFORE impact, ball should still be "
                   f"here. green = our estimate ({bx},{by}). inset {zoom}x")
        nm = f"{debug_prefix}.jpg"
        cv2.imwrite(str(Path(debug_dir) / nm), fr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return nm
    except Exception as exc:  # noqa: BLE001
        log.warning("debug3 rest_check_image failed: %s", exc)
        return None


def launch_from_ground(fit: dict, ground_y: float | None) -> dict:
    """Where the fitted flight leaves the ground. THE ball position.

    No image search. Once a flight is fitted, the ball's launch point is a
    property of the curve: solve y(t) = ground_y and take the earlier root
    (y opens upward in image coordinates, so the later root is the landing).
    x follows from the linear fit at that t.

    This replaced a search for a stationary bright blob near the same point.
    That search measurably added nothing -- on both synthetics the
    extrapolation alone landed 1.4-2.2px from truth and the "refined"
    answer was also 1.4px -- while adding a failure mode that put the ball
    400px up a tree, because a branch is stationary and bright too. When the
    line is this clear, the line IS the answer.

    Returns {ok, xy, frame, reason}.
    """
    out = {"ok": False, "xy": None, "frame": None, "reason": None}
    co = (fit or {}).get("coef") or {}
    if not co.get("y") or not co.get("x"):
        out["reason"] = "no fitted curve"
        return out
    if ground_y is None:
        out["reason"] = "no ground line (pose gave no feet)"
        return out
    a, b, c0 = (float(v) for v in co["y"])
    if abs(a) < 1e-9:
        out["reason"] = "the fit is a straight line, it never meets the ground"
        return out
    disc = b * b - 4.0 * a * (c0 - float(ground_y))
    if disc < 0.0:
        out["reason"] = "the flight never reaches the ground line"
        return out
    t_lo = (-b - math.sqrt(disc)) / (2.0 * a)
    t_hi = (-b + math.sqrt(disc)) / (2.0 * a)
    t = min(t_lo, t_hi)
    x = float(np.polyval(co["x"], t)) if HAS_CV else None
    if x is None:
        out["reason"] = "numpy unavailable"
        return out
    out["ok"] = True
    out["xy"] = [int(round(x)), int(round(float(ground_y)))]
    out["frame"] = int(round(t))
    out["reason"] = (
        f"the fitted flight leaves the ground line at f{out['frame']}, "
        f"({out['xy'][0]}, {out['xy'][1]})"
    )
    return out


def refine_ball_from_flight(
    input_path: Path,
    fit: dict,
    impact_frame: int,
    r: float,
    ground_y: float | None = None,
    search_r: float = 5.0,
    look_back: int = 14,
    debug_dir: Path | None = None,
    debug_prefix: str = "d3ball",
) -> dict:
    """Locate the ball at rest by looking where the FLIGHT says it was.

    Every other ball finder here searches blind and can be beaten by
    anything small, white and round -- a trainer, a sprinkler head, a daisy.
    But once a flight has been fitted, running its parabola back to the
    impact frame gives a launch point good to a couple of pixels, and that
    turns the problem into a tiny constrained search: find the stationary
    bright blob inside a box a few ball-widths across, in the frames BEFORE
    impact while the ball is still sitting there.

    A shoe cannot win this because it is not in the box.

    Stationarity is the test that separates the ball from the club sole
    arriving at address: across the frames examined, the ball does not move.

    Returns {ok, xy, moved_px, seen_in, spread_px, reason, image}.
    """
    out = {"ok": False, "xy": None, "moved_px": None, "seen_in": 0,
           "spread_px": None, "reason": None, "image": None,
           "launch_frame": None, "from": None}
    if not HAS_CV or not fit or not fit.get("at_impact"):
        out["reason"] = "no fitted flight to search from"
        return out
    try:
        bx, by = float(fit["at_impact"][0]), float(fit["at_impact"][1])
        f_ball = int(impact_frame)
        out["from"] = "the impact frame"
        # WHERE THE FLIGHT MEETS THE GROUND, not where it is at a guessed
        # impact frame. The pose peak is maximum wrist speed, which is only
        # approximately impact -- and if it is off by a few frames the
        # parabola evaluated there lands in mid-air, where any static branch
        # or gap of sky passes a "stationary bright blob" test. A ball at
        # rest is on the ground, so solve for the ground crossing.
        _co = (fit or {}).get("coef") or {}
        if ground_y is not None and _co.get("y") and _co.get("x"):
            a, b, c0 = (float(v) for v in _co["y"])
            if abs(a) > 1e-9:
                disc = b * b - 4.0 * a * (c0 - float(ground_y))
                if disc >= 0.0:
                    # y opens upward in image coordinates (the ball rises,
                    # y falls, then it comes back down), so the EARLIER root
                    # is the launch and the later one is the landing.
                    t_lo = (-b - math.sqrt(disc)) / (2.0 * a)
                    t_hi = (-b + math.sqrt(disc)) / (2.0 * a)
                    t_g = min(t_lo, t_hi)
                    gx = float(np.polyval(_co["x"], t_g))
                    # Only trust it if it lands somewhere sane: within the
                    # frame and not miles from the assumed impact frame.
                    if abs(t_g - impact_frame) <= 60:
                        bx, by = gx, float(ground_y)
                        f_ball = int(round(t_g))
                        out["from"] = "where the flight meets the ground"
        out["launch_frame"] = f_ball
        box = max(18.0, search_r * r)
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            out["reason"] = "could not open video"
            return out
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        x0 = int(max(0, bx - box))
        y0 = int(max(0, by - box))
        x1 = int(min(W, bx + box))
        y1 = int(min(H, by + box))
        if x1 - x0 < 6 or y1 - y0 < 6:
            out["reason"] = "search box fell outside the frame"
            return out
        # BEFORE impact only: after it, the ball is gone and the club is
        # sweeping through.
        f0 = max(0, f_ball - look_back)
        f1 = max(f0, f_ball - 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        area_max, _side = ball_area_cap(H)
        hits: list[tuple[float, float]] = []
        first_crop = None
        for _f in range(f0, f1 + 1):
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            crop = fr[y0:y1, x0:x1]
            if first_crop is None:
                first_crop = crop.copy()
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # WHITE TOP-HAT, not a brightness threshold. The crop is mostly
            # grass, so any percentile of the raw image sits at grass level
            # and the mask fills with texture -- measured: the "ball" then
            # wandered 27px frame to frame because it was picking a
            # different blade of grass each time. A top-hat with a kernel a
            # little larger than the ball keeps small bright objects and
            # discards everything smoothly varying, which is exactly the
            # distinction between a ball and the turf it sits on.
            kk = int(max(5, 2.5 * r)) | 1
            th = cv2.morphologyEx(
                g, cv2.MORPH_TOPHAT,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)),
            )
            thr = max(12.0, float(np.percentile(th, 99.0)))
            m = (th >= thr).astype(np.uint8) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            n, _lab, st, cent = cv2.connectedComponentsWithStats(m, 8)
            best = None
            for i in range(1, n):
                a = int(st[i, cv2.CC_STAT_AREA])
                bw = int(st[i, cv2.CC_STAT_WIDTH])
                bh = int(st[i, cv2.CC_STAT_HEIGHT])
                if a < 2 or a > area_max:
                    continue
                if max(bw, bh) > 4.0 * r or min(bw, bh) < 1:
                    continue
                # Roundish: a ball's bounding box is near-square.
                if max(bw, bh) > 2.2 * max(1, min(bw, bh)):
                    continue
                d = math.hypot(float(cent[i][0]) + x0 - bx,
                               float(cent[i][1]) + y0 - by)
                if best is None or d < best[0]:
                    best = (d, float(cent[i][0]) + x0,
                            float(cent[i][1]) + y0)
            if best:
                hits.append((best[1], best[2]))
        cap.release()
        if len(hits) < 3:
            out["reason"] = (
                f"only {len(hits)} bright blob(s) in the "
                f"{int(2 * box)}px box over f{f0}-{f1}"
            )
            return out
        xs = np.array([h[0] for h in hits])
        ys = np.array([h[1] for h in hits])
        mx, my = float(np.median(xs)), float(np.median(ys))
        spread = float(np.median(np.hypot(xs - mx, ys - my)))
        out["seen_in"] = len(hits)
        out["spread_px"] = round(spread, 1)
        # A ball at rest does not move. Anything wandering more than about a
        # ball width across these frames is the club, a shadow edge, or
        # glare -- not the thing we are looking for.
        if spread > 1.5 * r:
            out["reason"] = (
                f"the brightest blob wandered {spread:.0f}px over "
                f"{len(hits)} frames -- not a ball at rest"
            )
            return out
        # A ball at rest is ON THE GROUND. Whatever was found in the sky is
        # a branch or a gap between leaves -- both perfectly stationary, both
        # perfectly bright, neither a golf ball. This is the guard that would
        # have caught a marker sitting 400px above the turf.
        if ground_y is not None and abs(my - float(ground_y)) > 4.0 * r:
            out["reason"] = (
                f"the blob found is {abs(my - float(ground_y)):.0f}px off "
                f"the ground line at the feet -- not a ball at rest"
            )
            return out
        out["ok"] = True
        out["xy"] = [int(round(mx)), int(round(my))]
        out["moved_px"] = round(math.hypot(mx - bx, my - by), 1)
        out["reason"] = (
            f"stationary bright blob in {len(hits)} of {f1 - f0 + 1} frames "
            f"before f{f_ball}, spread {spread:.1f}px, {out['moved_px']}px "
            f"from where the flight said to look ({out['from']})"
        )
        if debug_dir is not None and first_crop is not None:
            z = 6
            img = cv2.resize(first_crop, None, fx=z, fy=z,
                             interpolation=cv2.INTER_NEAREST)
            cv2.circle(img, (int((bx - x0) * z), int((by - y0) * z)),
                       int(1.2 * r * z / 2), (255, 200, 0), 2, cv2.LINE_AA)
            cv2.circle(img, (int((mx - x0) * z), int((my - y0) * z)),
                       int(1.2 * r * z / 2), (0, 255, 0), 2, cv2.LINE_AA)
            _label(img, f"ball search box, {z}x. cyan = where the flight "
                        f"said to look, green = the stationary blob found. "
                        f"{out['reason']}")
            nm = f"{debug_prefix}-refine.jpg"
            cv2.imwrite(str(Path(debug_dir) / nm), img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            out["image"] = nm
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("debug3 refine_ball_from_flight failed: %s", exc)
        out["reason"] = f"failed: {exc}"
        return out


def pick_flight(
    tracks: list,
    impact_frame: int,
    ball_xy=None,
    frame_w: int = 1920,
    frame_h: int = 1080,
    r: float = 12.0,
) -> dict:
    """Score every track's RANSAC fit and return the best real flight.

    The gates are Debug2's, deliberately: it has to RISE, and its curve run
    back to the impact frame has to land near the ball. Keeping the tests
    identical is what makes the two methods comparable — if Debug3 finds a
    flight Debug2 rejected, the difference is in the detection, not in a
    looser standard.
    """
    out = {"ok": False, "flight": None, "fit": None, "tried": [],
           "reason": None}
    # The old combinatorial tracker produced thousands of near-duplicate
    # tracks and only the first 40 could be afforded, which is how a real
    # flight got pushed off the end by copies of noise. One track per object
    # plus a fit that costs ~10ms means the limit can be generous; it is a
    # runaway guard now, not a budget.
    if len(tracks) > MAX_TRACKS_TESTED:
        out["note"] = (
            f"{len(tracks)} tracks built, testing the {MAX_TRACKS_TESTED} "
            f"longest"
        )
    for tr in tracks[:MAX_TRACKS_TESTED]:
        fit = ransac_parabola(tr["points"], impact_frame, ball_xy, r=r)
        rec = {
            "n_points": len(tr["points"]),
            "span_px": tr["span_px"], "rise_px": tr["rise_px"],
            "n_inliers": fit.get("n_inliers"), "rms_px": fit.get("rms_px"),
            "aim_px": fit.get("aim_px"), "verdict": None,
        }
        if not fit.get("ok"):
            rec["verdict"] = f"no fit: {fit.get('reason')}"
            out["tried"].append(rec)
            continue
        if fit["n_inliers"] < 4:
            rec["verdict"] = f"only {fit['n_inliers']} inliers"
            out["tried"].append(rec)
            continue
        if tr["rise_px"] < 30:
            rec["verdict"] = f"does not rise ({tr['rise_px']}px)"
            out["tried"].append(rec)
            continue
        # Same relative aim gate as Debug2: a miss only means something
        # measured against how far the track is from the ball.
        if fit.get("aim_px") is not None and ball_xy:
            d0 = math.hypot(
                tr["points"][0]["x"] - float(ball_xy[0]),
                tr["points"][0]["y"] - float(ball_xy[1]),
            )
            # The floor was a flat 60px, which is a 1080p number and too
            # tight besides: the BALL POSITION is itself only good to a few
            # tens of pixels, so a 60px floor demands the back-projection
            # agree more precisely than the thing it is compared against.
            # 6r scales with the frame and is about the ball's own
            # uncertainty. The 0.30*d0 term still dominates for any track
            # that starts far away, which is where the test does real work.
            limit = min(max(120.0, 0.20 * float(frame_w)),
                        max(6.0 * r, 0.30 * d0))
            _rms_budget = max(SELF_EVIDENT_RMS_R * r,
                              SELF_EVIDENT_RMS_SPAN * float(tr["span_px"]))
            rec["rms_budget"] = round(_rms_budget, 1)
            _self_evident = (
                fit["n_inliers"] >= SELF_EVIDENT_INLIERS
                and (fit["rms_px"] or 99.0) <= _rms_budget
                and tr["rise_px"] >= SELF_EVIDENT_RISE_FRAC * float(frame_h)
                and tr["span_px"] >= SELF_EVIDENT_SPAN_FRAC * float(frame_w)
            )
            if fit["aim_px"] > limit and not _self_evident:
                rec["verdict"] = (
                    f"aims {fit['aim_px']:.0f}px from the ball "
                    f"(limit {limit:.0f})"
                )
                out["tried"].append(rec)
                continue
            if fit["aim_px"] > limit:
                rec["aim_disagrees"] = True
                rec["note"] = (
                    f"accepted on its own fit despite aiming "
                    f"{fit['aim_px']:.0f}px from the ball (limit "
                    f"{limit:.0f}) -- {fit['n_inliers']} inliers on a "
                    f"parabola at {fit['rms_px']}px rms, rising "
                    f"{tr['rise_px']:.0f}px, is a flight. SUSPECT THE BALL "
                    f"POSITION."
                )
            # And the baseline has to support that aim — see Debug2: a
            # short run's back-projection is luck, not evidence.
            need = 2.0 * r * max(1.0, d0) / max(1.0, limit)
            if tr["span_px"] < max(4.0 * r, need):
                rec["verdict"] = (
                    f"baseline {tr['span_px']:.0f}px too short to aim from "
                    f"(need {max(4.0 * r, need):.0f})"
                )
                out["tried"].append(rec)
                continue
        score = (
            2.0 * fit["n_inliers"]
            + 2.0 * min(1.0, tr["span_px"] / max(1.0, 0.5 * float(frame_w)))
            - (fit["rms_px"] or 0.0) / 10.0
        )
        rec["verdict"] = (
            f"accepted, score {score:.2f}"
            + (" (aim disagrees)" if rec.get("aim_disagrees") else "")
        )
        rec["score"] = round(score, 2)
        out["tried"].append(rec)
        if out["flight"] is None or score > out["flight"]["score"]:
            out["flight"] = {"score": score, "track": tr, "fit": fit,
                             "note": rec.get("note")}
    if out["flight"] is None:
        # Say WHICH test rejected them, and how close the best one came.
        # "none survived" sends you back to the panel to expand a table; the
        # caption should carry the answer.
        buckets: dict[str, int] = {}
        for rec in out["tried"]:
            v = str(rec.get("verdict") or "?")
            key = (
                "too few inliers" if "inliers" in v
                else "does not rise" if "rise" in v
                else "aims wide of the ball" if "aims" in v
                else "baseline too short" if "baseline" in v
                else "no parabola fit" if "no fit" in v
                else v
            )
            buckets[key] = buckets.get(key, 0) + 1
        why = ", ".join(f"{n} {k}" for k, n in
                        sorted(buckets.items(), key=lambda kv: -kv[1]))
        # The near miss is the useful one -- it is the track to argue about.
        near = None
        for rec in out["tried"]:
            if rec.get("aim_px") is not None and (rec.get("n_inliers") or 0) >= 3:
                if near is None or rec["aim_px"] < near["aim_px"]:
                    near = rec
        detail = ""
        if near:
            detail = (
                f". Closest: {near['n_points']} points, "
                f"{near['n_inliers']} inliers, rms {near.get('rms_px')}px "
                f"(budget {near.get('rms_budget')}), spans "
                f"{near.get('span_px')}px, rises {near['rise_px']}px, "
                f"aims {near['aim_px']}px from the ball -- "
                f"{near.get('verdict')}"
            )
        out["reason"] = (
            f"{len(tracks)} track(s) built, none survived the flight tests"
            + (f" ({why})" if why else "") + detail
        )
        return out
    out["ok"] = True
    out["fit"] = out["flight"]["fit"]
    out["reason"] = out["flight"]["fit"]["reason"]
    _w = out["flight"].get("note")
    if _w:
        out["aim_disagrees"] = True
        out["reason"] += ". " + _w
    return out


def draw_flight(
    canvas_path: Path,
    out_path: Path,
    ball_xy,
    tracks: list,
    winner: dict | None,
    fit: dict | None,
    caption: str,
    scale: float = 1.0,
) -> bool:
    """Draw every track dim, the winner bright, and the fitted parabola.

    Detections come back in FULL-RES coordinates while the canvas is the
    downscaled working frame, so everything drawn here is multiplied by the
    same scale the detection pass used. Getting this wrong does not throw --
    it just draws the whole flight 1.5x off the ball, which looks like a
    tracking failure.
    """
    if not HAS_CV:
        return False
    try:
        img = cv2.imread(str(canvas_path))
        if img is None:
            return False
        h, w = img.shape[:2]
        sc = float(scale) or 1.0

        def _p(px, py):
            return int(round(float(px) * sc)), int(round(float(py) * sc))

        win_pts = {(p["frame"], p["x"], p["y"])
                   for p in (winner or {}).get("points", [])}
        for tr in tracks[:40]:
            pts = tr["points"]
            if {(p["frame"], p["x"], p["y"]) for p in pts} == win_pts:
                continue
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, _p(a["x"], a["y"]), _p(b["x"], b["y"]),
                         (120, 120, 120), 1, cv2.LINE_AA)
        if fit and fit.get("coef"):
            cx, cy = fit["coef"]["x"], fit["coef"]["y"]
            fs = [p["frame"] for p in (winner or {}).get("points", [])]
            if fs:
                prev = None
                for f in range(min(fs) - 2, max(fs) + 3):
                    cur = _p(float(np.polyval(cx, float(f))),
                             float(np.polyval(cy, float(f))))
                    if prev is not None:
                        cv2.line(img, prev, cur, (255, 200, 0), 2,
                                 cv2.LINE_AA)
                    prev = cur
            for p in fit.get("outliers") or []:
                cv2.drawMarker(img, _p(p["x"], p["y"]), (0, 0, 255),
                               cv2.MARKER_TILTED_CROSS, 12, 2)
            for p in fit.get("inliers") or []:
                cv2.circle(img, _p(p["x"], p["y"]),
                           max(7, int(0.010 * h)), (0, 255, 0), 2,
                           cv2.LINE_AA)
            if fit.get("at_impact"):
                cv2.drawMarker(img, _p(*fit["at_impact"]), (255, 0, 255),
                               cv2.MARKER_CROSS, 22, 2)
        if ball_xy and len(ball_xy) == 2:
            cv2.circle(img, _p(ball_xy[0], ball_xy[1]),
                       max(10, int(0.014 * h)), (0, 255, 0), 3, cv2.LINE_AA)
        _label(img, caption)
        cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("debug3 draw_flight failed: %s", exc)
        return False


__all__ = [
    "BALL_AREA_MIN", "BALL_AREA_STRICT", "ball_area_cap", "body_box_from_pose",
    "MIN_KEPT_FOR_TRACKING", "WIN_POST", "WIN_PRE",
    "build_tracks", "detect_ball_blobs", "draw_flight", "pick_flight",
    "ransac_parabola", "launch_from_ground", "rest_check_image", "refine_ball_from_flight",
]
