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
import time
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

# ...but the `* df` in that gate is capped here. A track with one point
# has no velocity, so a missed frame widens its gate with nothing to aim
# it — at df=4 that reached 622px at 1080p, which is a third of the frame,
# and the tracker duly built 3-point "tracks" spanning 900px out of
# unrelated noise. On the rainy swing at Snee Farm those junk tracks had
# the biggest rises in the whole run and crowded the real flight out of
# the shortlist. Acquisition over one or two frames is a ball that was
# briefly missed; over four it is a guess.
ACQUIRE_MAX_DF = 2

# A real flight is SEEN on most of the frames it crosses. Junk built from
# long-range links is not: every one of the frame-spanning tracks above
# had a point on barely a third of its frames. This is reported, not
# enforced — in bad light a real ball can be missed for a few frames, and
# a threshold that quietly deleted the shot would be the same class of
# bug as the gate that invented it.
def track_density(points: list) -> float:
    """Points per frame spanned, 1.0 = detected on every frame."""
    if not points:
        return 0.0
    span = int(points[-1]["frame"]) - int(points[0]["frame"]) + 1
    return round(len(points) / max(1, span), 2)

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
    # A GOLFER IS A TEE-CAMERA IDEA. Everything below that suppresses
    # "the big moving thing" and "the pixels that are always busy" was
    # written for a view with a person swinging in it, where a ball
    # crossing a moving sleeve genuinely cannot be told from the sleeve.
    #
    # A green camera has no golfer. What it does have on a bright windy
    # day is a CLOUD SHADOW sweeping across the green -- a single
    # connected component far bigger than the golfer threshold, dilated
    # by 21px, and therefore an exclusion zone laid over the exact part
    # of the frame the ball lands in. The busy-pixel filter compounds
    # it: shadow crossing the same turf repeatedly makes that turf
    # "always busy", and every detection there is dropped. The result is
    # detections in the tree line and none on the green, which is not a
    # tuning problem, it is the wrong question being asked of the view.
    suppress_bodies: bool = True,
    # WHICH BLOBS SURVIVE THE PER-FRAME CAP. "small" keeps the smallest,
    # which is right on a tee view where the ball is the tiniest thing
    # that matters and the big stuff is the golfer. On a green view it
    # is exactly backwards: everything left after the area filter is
    # already ball-sized, the frame is full of one- and two-pixel
    # speckle off the tree line, and keeping the smallest forty of
    # several hundred keeps the speckle and throws the ball away.
    cap_prefer: str = "small",
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
            if suppress_bodies:
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
                kept_this.sort(key=lambda d: d["area"],
                               reverse=(cap_prefer == "large"))
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
        # Off with body suppression, and for the same reason: on a green
        # view "this turf keeps being foreground" is a cloud crossing it,
        # not a golfer standing on it, and blanking it removes the
        # landing zone.
        busy = ((occ >= max(2, int(BUSY_PIXEL_FRAC * n_frames)))
                if suppress_bodies
                else np.zeros_like(occ, dtype=bool))
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
                # No velocity yet: acquisition, not tracking. Capped in df
                # — see ACQUIRE_MAX_DF; an uncapped one reaches a third of
                # the frame and links noise to noise.
                gate = ACQUIRE_GATE_R * r * min(df, ACQUIRE_MAX_DF)
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


def follow_to_rest(
    dets: list,
    seed: list,
    fps: float,
    gate_min_px: float = 22.0,
    gate_k: float = 2.5,
    max_gap_frames: int = 10,
    rest_frames: int = 15,
    still_px: float = 8.0,
    bounce_keep: float = 0.6,
    bounce_sec: float = 1.0,
    bounce_decay: float = 0.93,
) -> dict:
    """Follow the ball from the end of its descent through the ground.

    The descent track stops the moment the ball hits, because a parabola
    and a bounce are different objects to a tracker. What happens next is
    the part a viewer actually cares about -- where it pitched, how it
    kicked, where it finished -- and it is all still there in the same
    per-frame detections.

    THE BALL COMING TO REST IS THE DETECTIONS STOPPING. These come from
    MOG2, which sees motion, so a ball that has settled produces nothing
    at all. That is not a gap to be tolerated, it is the answer: after
    rest_frames with nothing inside the gate, the ball is at rest where
    it was last seen.

    The gate scales with how fast the ball is actually going -- a first
    bounce moves 40px a frame and a ball trickling to a stop moves one --
    with a floor so a slow roll is not lost to detector jitter. Prediction
    is deliberately NOT ballistic: the whole point is that the ground has
    stopped obeying the parabola.

    Returns {path, landing_frame, landing_xy, rest_frame, rest_xy,
    n_gaps, reason}. Never raises.
    """
    out = {"path": [], "landing_frame": None, "landing_xy": None,
           "rest_frame": None, "rest_xy": None, "n_gaps": 0, "reason": None}
    if not seed or len(seed) < 2:
        out["reason"] = "no descent to follow from"
        return out

    by_frame: dict[int, list] = {}
    for d in dets or []:
        by_frame.setdefault(int(d["frame"]), []).append(d)

    # How fast it was falling when it arrived, which is the yardstick for
    # deciding it has landed.
    _p0, _p1 = seed[-2], seed[-1]
    _df = max(1, int(_p1["frame"]) - int(_p0["frame"]))
    vy_descent = (float(_p1["y"]) - float(_p0["y"])) / _df
    vx = (float(_p1["x"]) - float(_p0["x"])) / _df
    vy = vy_descent

    x, y = float(_p1["x"]), float(_p1["y"])
    f = int(_p1["frame"])
    last_seen = f
    path = [{"frame": f, "x": int(round(x)), "y": int(round(y)),
             "phase": "descent"}]
    last_frames = sorted(by_frame) or [f]
    f_end = last_frames[-1]

    speed_arrival = max(1.0, (vx * vx + vy * vy) ** 0.5)
    vx_arrival = vx

    while f < f_end:
        f += 1
        gap = f - last_seen
        speed = max(1.0, (vx * vx + vy * vy) ** 0.5)
        # THE GATE MUST NOT COLLAPSE AT THE MOMENT OF THE BOUNCE.
        #
        # It scales with how fast the ball is currently going, and the
        # landing test is a COLLAPSE in the rate of fall -- so the two
        # fight each other exactly once, at the landing frame, and the
        # gate wins. The ball arrives at 50px a frame, the vertical
        # component goes to nothing as it pitches, the measured speed
        # drops with it, and the gate shuts to its 22px floor on the one
        # frame the ball is about to kick forward hardest. The bounce
        # lands outside it, nothing matches for rest_frames, and the
        # ball is declared at rest a frame after it touched -- which is
        # what happened: landed f316, "rest" f317, twelve pixels away,
        # with the bounce plainly visible carrying on across the frame.
        #
        # A bounce keeps most of the horizontal speed it arrived with,
        # so that arrival speed is the floor, decaying over about a
        # second as the ball skids, kicks and starts to roll.
        if out["landing_frame"] is not None:
            _since = f - int(out["landing_frame"])
            if _since <= int(bounce_sec * max(1.0, fps or 30.0)):
                speed = max(speed, bounce_keep * speed_arrival
                            * (bounce_decay ** max(0, _since)))
        # The gate grows with the gap -- a ball missed for three frames
        # has had three frames to travel -- but only so far. Ungapped, it
        # reaches whatever it likes: measured on a real clip, a ten-frame
        # gap opened the gate to 220px and the track jumped 190px UPWARD
        # onto unrelated noise, inventing a roll that never happened out
        # of a ball that had already stopped.
        gate = max(gate_min_px, gate_k * speed) * min(max(1, gap), 3)
        best, bestd = None, gate
        _bouncing = (out["landing_frame"] is not None
                     and f - int(out["landing_frame"])
                     <= int(bounce_sec * max(1.0, fps or 30.0)))
        for d in by_frame.get(f, []):
            _ddx = float(d["x"]) - x
            # Opening the gate for the bounce also opens it to whatever
            # else is moving nearby, so the direction has to earn it: a
            # ball that pitched going one way does not set off back the
            # other. Only applied while the gate is widened, and only
            # when the arrival had a real horizontal component to speak
            # of -- a ball dropping straight down has no "back".
            if (_bouncing and abs(vx_arrival) >= 2.0
                    and _ddx * vx_arrival < -gate_min_px):
                continue
            dd = (_ddx ** 2 + (float(d["y"]) - y) ** 2) ** 0.5
            if dd < bestd:
                bestd, best = dd, d
        if best is None:
            if gap >= rest_frames:
                break
            continue
        # ...and a ball can also come to rest while still producing
        # detections -- sand settling around it, a shadow shifting. If
        # the followed point has not gone anywhere for rest_frames, it is
        # at rest whatever the detector keeps reporting.
        _recent = [q for q in path if f - q["frame"] < rest_frames]
        if len(_recent) >= rest_frames // 2:
            _sx = max(q["x"] for q in _recent) - min(q["x"] for q in _recent)
            _sy = max(q["y"] for q in _recent) - min(q["y"] for q in _recent)
            if max(_sx, _sy) <= still_px:
                last_seen = _recent[0]["frame"]
                x, y = float(_recent[0]["x"]), float(_recent[0]["y"])
                break
        nx, ny = float(best["x"]), float(best["y"])
        _dt = max(1, f - last_seen)
        vx, vy = (nx - x) / _dt, (ny - y) / _dt
        # LANDED when it stops falling like it was falling. The ground
        # does not have to bounce it back up -- into sand it simply stops
        # -- so the test is a collapse in the rate of fall, not a reversal.
        if out["landing_frame"] is None and vy < 0.5 * vy_descent:
            out["landing_frame"] = int(f)
            out["landing_xy"] = [int(round(nx)), int(round(ny))]
        if _dt > 1:
            out["n_gaps"] += 1
        x, y = nx, ny
        last_seen = f
        path.append({
            "frame": int(f), "x": int(round(x)), "y": int(round(y)),
            "phase": ("descent" if out["landing_frame"] is None
                      else "ground"),
        })

    out["path"] = path
    out["rest_frame"] = int(last_seen)
    out["rest_xy"] = [int(round(x)), int(round(y))]
    _n_ground = sum(1 for p in path if p["phase"] == "ground")
    if out["landing_frame"] is None:
        out["reason"] = (
            "never stopped falling inside the window — the ball left the "
            "frame, or the window ends before it lands"
        )
    else:
        out["reason"] = (
            f"landed f{out['landing_frame']} at "
            f"{out['landing_xy'][0]},{out['landing_xy'][1]}; followed "
            f"{_n_ground} frame(s) of bounce and roll to rest at f"
            f"{out['rest_frame']} ({out['rest_xy'][0]},{out['rest_xy'][1]})"
        )
    return out


def draw_ball_path(canvas_path: Path, out_path: Path, descent: list,
                   follow: dict, caption: str = "") -> bool:
    """Descent, landing, bounce/roll and rest, on one picture.

    The numbers are checkable in a table; where the ball pitched and where
    it finished are only checkable by looking.
    """
    if not HAS_CV:
        return False
    try:
        img = cv2.imread(str(canvas_path))
        if img is None:
            return False
        pts = [(int(p["x"]), int(p["y"])) for p in (descent or [])]
        gp = [(int(p["x"]), int(p["y"])) for p in (follow.get("path") or [])
              if p.get("phase") == "ground"]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, (0, 160, 255), 2, cv2.LINE_AA)  # BGR orange
        # JOIN THE TWO. The descent track ends on the last frame the ball
        # was seen falling and the ground path starts on the first frame
        # it was seen after -- between them is the frame it was actually
        # landing on, and leaving that segment undrawn puts a gap exactly
        # where the eye is looking for the pitch mark.
        if pts and gp:
            cv2.line(img, pts[-1], gp[0], (0, 160, 255), 2, cv2.LINE_AA)
        for a, b in zip(gp, gp[1:]):
            cv2.line(img, a, b, (0, 215, 255), 2, cv2.LINE_AA)
        for q in gp:
            cv2.circle(img, q, 3, (0, 215, 255), -1, cv2.LINE_AA)

        # LANDING and REST, each with its frame. They can be a few dozen
        # pixels apart -- a ball that pitches and stops, which is most of
        # what lands in sand -- so the second label is pushed clear rather
        # than written over the first.
        _lxy = follow.get("landing_xy")
        _rxy = follow.get("rest_xy")
        if _lxy:
            lx, ly = int(_lxy[0]), int(_lxy[1])
            cv2.drawMarker(img, (lx, ly), (0, 0, 255), cv2.MARKER_CROSS, 26, 3)
            cv2.circle(img, (lx, ly), 15, (0, 0, 255), 2, cv2.LINE_AA)
            _lt = f"LANDING f{follow.get('landing_frame')}"
            cv2.putText(img, _lt, (lx + 20, ly - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, _lt, (lx + 20, ly - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        if _rxy:
            rx, ry = int(_rxy[0]), int(_rxy[1])
            cv2.circle(img, (rx, ry), 13, (0, 255, 0), 2, cv2.LINE_AA)
            _dy = 30
            if _lxy and abs(rx - int(_lxy[0])) < 190 and abs(ry - int(_lxy[1])) < 30:
                _dy = 52  # the landing label is right there; drop below it
            _rt = f"REST f{follow.get('rest_frame')}"
            cv2.putText(img, _rt, (rx + 18, ry + _dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, _rt, (rx + 18, ry + _dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA)
        if caption:
            _label(img, caption)
        cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("debug3 draw_ball_path failed: %s", exc)
        return False


def bezier_continuation(points: list, landing_xy, n_dir: int = 6,
                        ctrl_frac: float = 0.60, ctrl_min: float = 160.0,
                        ctrl_max: float = 250.0, n_samples: int = 60) -> dict:
    """Carry the measured flight on to a known landing, as one parabola.

    MOG2 gives the first part of the flight and then loses the ball --
    against trees, against sky, or simply off the top of the frame. The
    landing is known from the OTHER camera. What is missing is the middle,
    and a quadratic Bezier is exactly the right tool because a quadratic
    Bezier IS a parabola: it leaves P0 along the direction the ball was
    already travelling, rises to an apex, and comes down to P2.

    THE DIRECTION COMES FROM A REGRESSION, NOT THE LAST TWO POINTS. At
    50fps a MOG2 centroid jitters a couple of pixels frame to frame, and
    a tangent taken from two points multiplies that jitter by the hundreds
    of pixels being projected -- a 2px wobble over a 20px step is a 6
    degree error, which is tens of pixels wrong by the landing. A
    least-squares fit over the last several points averages the jitter out.

    P1 sits along that direction at 60% of the distance to the landing
    (clamped, floor 160px), which is what keeps the curve going the way
    the ball was going before it bends. The control distance IS the apex
    height: push P1 further along the launch ray and the ball climbs
    further before turning over. 40% with a 40px floor left the apex
    sitting almost on top of P0 whenever the landing mapped close to the
    last measured point, which on a shot that exits the top of frame is
    most of the time -- the curve turned over immediately instead of
    arcing. Placing P1 anywhere off that ray would make the
    tracer turn immediately, which reads as wrong even when the endpoints
    are right.

    Returns {ok, p0, p1, p2, apex, curve, direction, ctrl_px, reason}.
    Never raises. Screen space only -- no attempt at 3D, which needs
    depth, launch conditions and lens geometry that are not available and
    buys nothing for a tracer that only has to look right.
    """
    out = {"ok": False, "p0": None, "p1": None, "p2": None, "apex": None,
           "curve": [], "direction": None, "ctrl_px": None, "reason": None,
           "n_dir_used": 0, "n_stalled_dropped": 0}
    pts = [p for p in (points or []) if p is not None]
    if len(pts) < 2:
        out["reason"] = "fewer than two measured points to extend from"
        return out
    if not landing_xy:
        out["reason"] = "no landing point to aim at"
        return out

    # THE LAST RELIABLE POINT IS NOT THE LAST POINT. Measured on a real
    # flight: the ball rises cleanly for twelve frames and then sits at
    # (1064,15) -- the top edge -- for eight more, because the blob is
    # clipped by the border and its centroid stops moving. A regression
    # over that tail reads a stationary ball and refuses to extend at all.
    # Trailing points that do not move are dropped before anything else.
    _stall = 0
    while len(pts) - _stall > 2:
        a, b = pts[-1 - _stall - 1], pts[-1 - _stall]
        if abs(float(b["x"]) - float(a["x"])) <= 2.0 and \
           abs(float(b["y"]) - float(a["y"])) <= 2.0:
            _stall += 1
            continue
        break
    if _stall:
        pts = pts[:len(pts) - _stall]
    out["n_stalled_dropped"] = _stall
    if len(pts) < 2:
        out["reason"] = "nothing left to extend from once stalled points are dropped"
        return out

    tail = pts[-max(2, int(n_dir)):]
    out["n_dir_used"] = len(tail)
    n = len(tail)
    # Least squares of FRAME -> x and FRAME -> y. Against the index it
    # would treat a three-frame gap as one step and bend the tangent
    # toward whatever came after it.
    _i = [float(q.get("frame", j)) for j, q in enumerate(tail)]
    _mi = sum(_i) / n
    _den = sum((v - _mi) ** 2 for v in _i) or 1.0
    dx = sum((_i[j] - _mi) * float(tail[j]["x"]) for j in range(n)) / _den
    dy = sum((_i[j] - _mi) * float(tail[j]["y"]) for j in range(n)) / _den
    mag = (dx * dx + dy * dy) ** 0.5
    if mag < 1e-6:
        out["reason"] = "the measured points do not move — no direction to carry on"
        return out
    ux, uy = dx / mag, dy / mag

    p0 = (float(pts[-1]["x"]), float(pts[-1]["y"]))
    p2 = (float(landing_xy[0]), float(landing_xy[1]))
    dist = ((p2[0] - p0[0]) ** 2 + (p2[1] - p0[1]) ** 2) ** 0.5
    if dist < 1.0:
        out["reason"] = "the last measured point is already at the landing"
        return out
    k = max(ctrl_min, min(ctrl_max, dist * float(ctrl_frac)))
    p1 = (p0[0] + ux * k, p0[1] + uy * k)

    curve = []
    for j in range(int(n_samples) + 1):
        t = j / float(n_samples)
        w0, w1, w2 = (1 - t) ** 2, 2 * (1 - t) * t, t * t
        curve.append((
            int(round(w0 * p0[0] + w1 * p1[0] + w2 * p2[0])),
            int(round(w0 * p0[1] + w1 * p1[1] + w2 * p2[1])),
        ))
    apex = min(curve, key=lambda q: q[1])   # smallest y = highest point
    out.update({
        "ok": True,
        "p0": [int(round(p0[0])), int(round(p0[1]))],
        "p1": [int(round(p1[0])), int(round(p1[1]))],
        "p2": [int(round(p2[0])), int(round(p2[1]))],
        "apex": [int(apex[0]), int(apex[1])],
        "curve": curve, "ctrl_px": round(k, 1),
        "direction": [round(ux, 4), round(uy, 4)],
        "reason": (
            f"carried on from the last {len(tail)} measured point(s)"
            + (f" ({_stall} stalled at the frame edge dropped)" if _stall else "")
            + f", heading ({ux:.2f},{uy:.2f}), control {k:.0f}px along that "
            f"line, apex ({apex[0]},{apex[1]})"
        ),
    })
    return out


def draw_bezier_continuation(canvas_path: Path, out_path: Path,
                             points: list, bez: dict,
                             caption: str = "") -> bool:
    """Measured flight solid, projected continuation dashed.

    The distinction is the point: one half is where the ball was seen,
    the other is where it must have gone. Drawing them alike would make a
    projection look like a measurement.
    """
    if not HAS_CV:
        return False
    try:
        img = cv2.imread(str(canvas_path))
        if img is None:
            return False
        mp = [(int(p["x"]), int(p["y"])) for p in (points or [])]
        for a, b in zip(mp, mp[1:]):
            cv2.line(img, a, b, (255, 220, 0), 3, cv2.LINE_AA)   # measured
        for q in mp:
            cv2.circle(img, q, 3, (255, 220, 0), -1, cv2.LINE_AA)
        curve = bez.get("curve") or []
        for j in range(0, max(0, len(curve) - 1), 2):            # dashed
            cv2.line(img, curve[j], curve[j + 1], (0, 90, 255), 3, cv2.LINE_AA)
        for nm, key, col in (("P0", "p0", (255, 220, 0)),
                             ("apex", "apex", (0, 200, 255)),
                             ("P2 landing", "p2", (0, 0, 255))):
            q = bez.get(key)
            if not q:
                continue
            qq = (int(q[0]), int(q[1]))
            cv2.circle(img, qq, 8, col, 2, cv2.LINE_AA)
            for _c, _t in ((( 0, 0, 0), 4), (col, 2)):
                cv2.putText(img, nm, (qq[0] + 12, qq[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _c, _t, cv2.LINE_AA)
        if caption:
            _label(img, caption)
        cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("debug3 draw_bezier_continuation failed: %s", exc)
        return False


def extrapolate_flight(points: list, frame_h: int, fps: float,
                       max_sec: float = 4.0, n_fit: int = 8) -> dict:
    """Carry the flight on with no landing to aim at.

    When the green camera never saw the ball -- missed the green, flew
    long, out of that camera's view entirely -- there is no P2 and the
    Bezier has nothing to bend toward. The ball still went somewhere, and
    the measurement itself says where: x moves at a constant rate and y
    is a parabola, which is the same model the RANSAC fit already uses.
    Continue those and the line follows the shot instead of stopping dead
    where MOG2 lost it.

    The landing is ASSUMED, not measured, and is reported as such: the
    curve runs until the ball returns to the height it was struck from
    (the ground line at that distance), or until max_sec is up, whichever
    comes first.

    Returns {ok, points, landing_xy, landing_frame, reason}. Never raises.
    """
    out = {"ok": False, "points": [], "landing_xy": None,
           "landing_frame": None, "reason": None}
    pts = [p for p in (points or []) if p is not None]
    if len(pts) < 3:
        out["reason"] = "fewer than three measured points to fit"
        return out
    # Drop a stalled tail first, for the same reason the Bezier does: a
    # ball clipped by the frame border stops moving and a fit over that
    # reads as a ball hanging in the air.
    while len(pts) > 3:
        a, b = pts[-2], pts[-1]
        if abs(float(b["x"]) - float(a["x"])) <= 2.0 and \
           abs(float(b["y"]) - float(a["y"])) <= 2.0:
            pts = pts[:-1]
            continue
        break
    tail = pts[-max(3, int(n_fit)):]
    f0 = float(tail[0]["frame"])
    tt = [float(q["frame"]) - f0 for q in tail]
    xx = [float(q["x"]) for q in tail]
    yy = [float(q["y"]) for q in tail]
    try:
        if not HAS_CV:      # this module gates cv2 and numpy together
            out["reason"] = "numpy not installed"
            return out
        mx = np.polyfit(tt, xx, 1)          # x linear in t
        my = np.polyfit(tt, yy, 2)          # y quadratic
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"could not fit the flight: {exc}"
        return out
    if my[0] <= 0:
        # Curving upward forever: no descent to follow, so nothing to say
        # about where it came down.
        out["reason"] = "the fitted flight never turns over — nothing to extend"
        return out

    y_ground = float(pts[0]["y"])           # the height it was struck from
    f_last = int(tail[-1]["frame"])
    n_max = int(max(1, round(float(max_sec) * float(fps or 30.0))))
    res = []
    for i in range(1, n_max + 1):
        f = f_last + i
        t = float(f) - f0
        x = float(np.polyval(mx, t))
        y = float(np.polyval(my, t))
        res.append({"frame": int(f), "x": int(round(x)), "y": int(round(y))})
        if y >= y_ground:                   # back down to where it started
            break
    if not res:
        out["reason"] = "nothing to extend"
        return out
    out.update({
        "ok": True, "points": res,
        "landing_xy": [res[-1]["x"], res[-1]["y"]],
        "landing_frame": res[-1]["frame"],
        "reason": (
            f"no landing seen on the green camera — carried the fitted "
            f"flight on for {len(res)} frame(s) to an ASSUMED landing at "
            f"({res[-1]['x']},{res[-1]['y']})"
            + ("" if res[-1]["y"] >= y_ground else ", still airborne when the extension ran out")
        ),
    })
    return out


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
        "aim_path_px": None, "aim_at_impact_px": None, "aim_basis": None,
        "aim_frame": None,
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
    # Keep the seed model. The refit below can come out WORSE than the
    # model it replaces, and we need something to fall back to.
    _seed_inl, _seed_cx, _seed_cy = inl, cx, cy
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
    # A refit that holds nothing is not an improvement. Left unguarded,
    # dist[inl] is empty, np.mean warns "Mean of empty slice" and returns
    # nan, and nan is not JSON-serialisable — which 500s the debug3
    # status endpoint AFTER produce has already succeeded. Fall back to
    # the seed model, which by construction held at least 3 points.
    if int(inl.sum()) < 3:
        inl, cx, cy = _seed_inl, _seed_cx, _seed_cy
        dist = np.hypot(x - np.polyval(cx, t), y - np.polyval(cy, t))
    out["n_inliers"] = int(inl.sum())
    _in_dist = dist[inl]
    out["rms_px"] = (
        round(float(np.sqrt(float(np.mean(_in_dist ** 2)))), 2)
        if _in_dist.size else None
    )
    out["inliers"] = [points[i] for i in range(len(points)) if inl[i]]
    out["outliers"] = [points[i] for i in range(len(points)) if not inl[i]]
    ix = float(np.polyval(cx, float(impact_frame)))
    iy = float(np.polyval(cy, float(impact_frame)))
    out["at_impact"] = [int(ix), int(iy)]
    out["coef"] = {"x": [float(v) for v in cx], "y": [float(v) for v in cy]}
    if ball_xy and len(ball_xy) == 2:
        _bx, _by = float(ball_xy[0]), float(ball_xy[1])
        # TWO WAYS TO ASK "does this track come from the ball", and only
        # one of them is any good.
        #
        # The old one evaluates the fit AT THE POSE IMPACT FRAME. That
        # bakes in pose's timing, and pose's timing is the weakest number
        # in the run — it fires on peak wrist speed, which is not the
        # moment of impact. On the swing at Snee Farm it was EIGHT frames
        # late, which put the impact frame three frames INSIDE the ball's
        # own track. Evaluating there lands the "launch point" 200px up in
        # the air, and the real flight was rejected for aiming 234px wide.
        #
        # The right question has no clock in it: run the path DOWN to the
        # ball's own height and see how far to the side it passes. Same
        # arithmetic as launch_from_ground, asked earlier. On that swing
        # it answers 3px.
        out["aim_at_impact_px"] = round(math.hypot(ix - _bx, iy - _by), 1)
        _path = None
        _a, _b, _c = (float(v) for v in cy)
        _t = None
        if abs(_a) > 1e-9:
            _disc = _b * _b - 4.0 * _a * (_c - _by)
            if _disc >= 0.0:
                _rt = math.sqrt(_disc)
                # The earlier root is the launch; the later one is where
                # the ball comes back down.
                _t = min((-_b - _rt) / (2.0 * _a), (-_b + _rt) / (2.0 * _a))
        elif abs(_b) > 1e-9:
            _t = (_by - _c) / _b            # degenerate: straight line
        if _t is not None:
            # Don't extrapolate a parabola halfway to next week. Past a
            # couple of track-lengths back, the curve is inventing.
            _f0 = float(points[0]["frame"])
            _span = max(1.0, float(points[-1]["frame"]) - _f0)
            if _f0 - _t <= max(30.0, 2.0 * _span):
                _path = round(
                    abs(float(np.polyval(cx, _t)) - _bx), 1,
                )
                out["aim_frame"] = int(round(_t))
        out["aim_path_px"] = _path
        out["aim_px"] = _path if _path is not None else out["aim_at_impact_px"]
        out["aim_basis"] = (
            "path run down to the ball's height" if _path is not None
            else "the fit at the pose impact frame (the path never gets "
                 "down to the ball)"
        )
    out["ok"] = True
    out["x_degree"] = int(len(cx) - 1)
    out["reason"] = (
        f"{out['n_inliers']}/{len(points)} points on a parabola "
        f"(x deg {out['x_degree']}), rms {out['rms_px']}px"
        + (f"; run down to the ball's height it passes "
           f"{out['aim_px']}px from it (at f{out['aim_frame']})"
           if out.get("aim_path_px") is not None else
           f"; at the pose impact frame it lands {out['aim_px']}px from "
           f"the ball" if out["aim_px"] is not None else "")
    )
    return out


# HOW FAST A FALLING GOLF BALL LOOKS, as a fraction of frame height per
# second, so the numbers survive a change of lens or resolution.
#
# The lower bound is what separates a ball from everything else that
# moves on a green. Measured across two clips: real descents fell at
# 0.53 and 1.29 frame-heights/sec, and the fastest of the 47 other
# tracks -- players walking, a flag, wind in the tree line, MOG2
# speckle -- managed 0.16. There is nearly a factor of four of daylight
# between them.
#
# The upper bound is physics. A par-3 tee shot comes down at 20-35 m/s;
# the one track that beat 3.0 was doing the equivalent of 96 m/s, and
# was three 9-pixel specks in black tree shadow that the tracker had
# chained into a straight line.
DESCENT_RATE_LO = 0.30
DESCENT_RATE_HI = 3.0
# How much of its peak fall speed a track must still have to count as
# still descending. Below this it has landed and is rolling.
DESCENT_FLATTEN_FRAC = 0.4


def _path_bend_px(pts: list) -> float:
    """How far a track strays from a straight line, in pixels rms.

    Fitted x against y rather than against time, because a descent is
    mostly vertical: y is the long axis and is monotonic through the
    fall, so the fit is well conditioned where fitting against time is
    not when frames are dropped.
    """
    try:
        ys = np.array([float(p["y"]) for p in pts], dtype=float)
        xs = np.array([float(p["x"]) for p in pts], dtype=float)
        if len(pts) < 3 or float(ys.max() - ys.min()) < 1.0:
            return 0.0
        coef = np.polyfit(ys, xs, 1)
        return float(np.sqrt(np.mean((np.polyval(coef, ys) - xs) ** 2)))
    except Exception:  # noqa: BLE001
        return 0.0


# HOW MANY FRAMES THE BACKGROUND PLATE IS MEDIANED OVER. A ceiling, not
# a target: the step widens to hit it on a long range and is left alone
# on a short one. See the note inside for why the ceiling exists at all.
MAX_PLATE_SAMPLES = 48


def detect_movers_by_plate(
    input_path: Path,
    f0: int,
    f1: int,
    plate_step: int = 8,
    thr: int = 24,
    edge_px: int = 30,
    min_area: int = 3,
    max_area: int = 900,
    max_side: int = 70,
) -> list:
    """Ball-sized things that differ from a still picture of the scene.

    WHY THIS EXISTS ALONGSIDE MOG2. `detect_ball_blobs` learns its
    background as it goes, and that adaptation is exactly wrong for a
    ball that crosses a busy band of the frame in a few frames: on a real
    clip a shot that a person can see land produced ONE detection in the
    whole descent, which no tracker can do anything with. The same ball
    against the same footage gives four or five here, because a median
    over the whole clip is a background that cannot be talked into
    absorbing something that was only ever there for a tenth of a second.

    It is not better than MOG2 in general -- it has no notion of a
    moving camera and it lights up on anything that shifted -- so the
    caller unions the two rather than choosing. Measured on two clips,
    the union finds every descent a person can see and neither detector
    finds all of them alone.

    Solid blobs bigger than a ball are dropped: a person is filled in,
    a ball smeared by its own speed is a thin streak.

    Returns the same [{frame, x, y, area}] shape `detect_ball_blobs`
    returns, so the two lists concatenate. Never raises.
    """
    out: list = []
    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            return out
        # A BOUNDED NUMBER OF SAMPLES, however long the range is.
        #
        # Every eighth frame of a two-minute clip is 450 frames, and 450
        # greyscale 720p frames is a 415MB stack -- which `np.median`
        # then partitions on a copy of, so the peak is over 800MB in one
        # allocation, on top of everything else the process is holding.
        # On a small container that is not slow, it is an OOM kill: the
        # worker dies, the upload sits at "processing" forever, and
        # nothing anywhere says what happened.
        #
        # The cap costs nothing that matters. The plate is a median of a
        # background that is not moving; forty-odd samples describe it as
        # well as four hundred do, and the ball -- there for a tenth of a
        # second -- is outvoted either way.
        _n_rng = max(1, int(f1) - int(f0) + 1)
        _step = max(int(plate_step), _n_rng // MAX_PLATE_SAMPLES + 1)
        plate_frames = []
        for f in range(int(f0), int(f1) + 1, _step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, im = cap.read()
            if ok and im is not None:
                plate_frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        if not plate_frames:
            cap.release()
            return out
        plate = np.median(np.stack(plate_frames), axis=0).astype(np.int16)
        del plate_frames
        h, w = plate.shape
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f0))
        f = int(f0)
        while f <= int(f1):
            ok, im = cap.read()
            if not ok or im is None:
                break
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.int16)
            d = np.abs(g - plate).astype(np.uint8)
            _, mask = cv2.threshold(d, int(thr), 255, cv2.THRESH_BINARY)
            n, _lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
            for i in range(1, n):
                a = int(st[i, cv2.CC_STAT_AREA])
                bw = int(st[i, cv2.CC_STAT_WIDTH])
                bh = int(st[i, cv2.CC_STAT_HEIGHT])
                if not (min_area <= a <= max_area):
                    continue
                if bw > max_side or bh > max_side:
                    continue
                if max(bw, bh) > 14 and a > 0.55 * bw * bh:
                    continue
                x, y = float(cen[i][0]), float(cen[i][1])
                if (x < edge_px or y < edge_px
                        or x > w - edge_px or y > h - edge_px):
                    continue
                out.append({"frame": f, "x": x, "y": y, "area": a})
            f += 1
        cap.release()
    except Exception:  # noqa: BLE001
        return out
    return out


# The sensitivity ladder the operator's map exposes as 🔍 Scan, and now
# the descent search's third detector. (threshold, per-frame keep,
# max blob area)
# How many diff blobs per frame the DESCENT search keeps. Deeper than
# the map's own scan uses, because that one runs on a region an operator
# has already pointed at and this one runs on a whole green view.
DESCENT_DIFF_PER_FRAME = 30

DIFF_SENS = {
    1: (12, 6, 600),
    2: (8, 10, 900),
    3: (5, 16, 1400),
}


def detect_movers_by_diff(
    input_path: Path,
    f0: int,
    f1: int,
    sens: int = 2,
    roi=None,
    cap: int = 20000,
    # HOW MANY BLOBS SURVIVE EACH FRAME. Largest-first beats speckle,
    # which is why it is the order -- but it is extremal either way, and
    # a green view has big things that are not the ball: a tree edge
    # lit by a pixel of camera shake is a far larger blob than a ball
    # forty yards off. Keeping ten of those per frame can crowd the ball
    # out exactly as keeping the smallest forty did, in the other
    # direction. So the descent search asks for a deeper keep and lets
    # the fall-rate, drop and straightness gates do the discriminating,
    # which is what they are for and what a size ranking is only ever a
    # proxy for.
    per_frame: int | None = None,
) -> list:
    """Frame-to-frame difference. The detector that actually sees the ball.

    WHY THIS IS HERE. MOG2 learns a background and the plate medians one;
    both describe what the scene usually looks like, and both then ask
    whether a pixel departs from it. On a green that is mostly smooth
    turf under moving cloud, what departs most is the tree line and the
    rough -- high-frequency texture that a pixel of camera shake lights
    up -- and a small ball crossing plain grass is not what either model
    is most surprised by. Measured on the clip this was written for:
    19,260 detections between the two of them, densely over the trees,
    and the putting surface all but bare.

    A plain difference against the PREVIOUS FRAME has no model to be
    unsurprised by. Something that was not there a thirtieth of a second
    ago and is there now is exactly a ball in flight, and this is the
    detector the operator's own map has been finding these descents with
    all along.

    Returns [{frame, x, y, area}], the shape the others return.
    """
    thr, _pf_default, area_max = DIFF_SENS[int(sens)]
    per_frame = int(per_frame) if per_frame else _pf_default
    out: list = []
    if not HAS_CV:
        return out
    try:
        cap_v = cv2.VideoCapture(str(input_path))
        if not cap_v.isOpened():
            return out
        try:
            x0, y0 = (int(roi[0]), int(roi[1])) if roi else (0, 0)
            cap_v.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, int(f0))))
            prev = None
            for f in range(max(0, int(f0)), int(f1) + 1):
                ok, frame = cap_v.read()
                if not ok or frame is None:
                    break
                if roi:
                    frame = frame[y0:y0 + int(roi[3]), x0:x0 + int(roi[2])]
                gray = cv2.GaussianBlur(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
                if prev is not None:
                    diff = cv2.absdiff(gray, prev)
                    _, th = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
                    th = cv2.dilate(th, None, iterations=1)
                    n, _l, st, cen = cv2.connectedComponentsWithStats(th)
                    hits = []
                    for i in range(1, n):
                        a = int(st[i, cv2.CC_STAT_AREA])
                        if 1 <= a <= area_max:
                            hits.append((a, float(cen[i][0]),
                                         float(cen[i][1])))
                    # LARGEST FIRST, which is the whole difference between
                    # this and the blob detector's cap. Inside a
                    # ball-sized window the ball is the BIG one and the
                    # speckle is the small one, so keeping the smallest
                    # few of several hundred keeps the speckle and throws
                    # the ball away.
                    hits.sort(reverse=True)
                    for a, cx, cy in hits[:per_frame]:
                        out.append({"frame": int(f),
                                    "x": int(round(x0 + cx)),
                                    "y": int(round(y0 + cy)),
                                    "area": int(a)})
                prev = gray
                if len(out) >= cap:
                    break
        finally:
            cap_v.release()
    except Exception as exc:  # noqa: BLE001
        log.debug("diff detector failed on %s: %s", input_path, exc)
    return out


def find_descents(
    input_path: Path,
    fps: float | None = None,
    window: tuple[int, int] | None = None,
    r: float = 14.0,
    # THREE POINTS IS A DESCENT. Four was the first guess and it threw a
    # real shot away: on a clip whose two landings a person can point at,
    # the earlier one gave three detections and then nothing for the
    # eleven frames before it hit the ground. That is not a tracking
    # failure to tune around -- the ball genuinely is not in the picture
    # in any way a detector can see. Three is enough because the
    # fall-rate band does the discriminating, not the point count.
    min_points: int = 3,
    min_drop_frac: float = 0.05,
    rate_lo: float = DESCENT_RATE_LO,
    rate_hi: float = DESCENT_RATE_HI,
    # A FALLING BALL IS STRAIGHT. Over the half-second a descent lasts,
    # gravity bends the path far less than the image is wide, so x against
    # y is very nearly a line. A chain of unrelated speckles the tracker
    # happened to link is not. Measured rms across three real descents on
    # two clips: 0.1, 2.2 and 5.3 px. The two false positives: 17.6 and
    # 21.7. This is the cleanest single discriminator of the lot.
    max_bend_px: float = 10.0,
    merge_sec: float = 1.0,
    max_events: int = 20,
) -> dict:
    """Every ball descent in a green-camera clip, found without being told
    where to look.

    WORKING BACKWARDS. Finding the swing on the tee camera is hard --
    people stand in front of the ball, there is more than one ball on the
    tee, and the ball is four pixels wide. Finding the ball ARRIVING on
    the green is easy, and this is why: on a green camera nothing else in
    the picture falls. Players walk, the flag moves, the tree line shifts
    in the wind, and all of it is slow and mostly sideways. A ball coming
    down off a par 3 crosses a quarter of the frame in half a second, and
    it is the only thing that does.

    So the count of descents is the count of shots that reached the
    green, and each one dates itself. A tee shot is airborne about 5 to 7
    seconds, so a descent that ends at green-clock T puts its swing in a
    two-second window at tee-clock T minus 5 to T minus 7 -- which is a
    small enough haystack that the tee-side detector only has to be right
    about WHERE, not WHEN.

    Each event carries `last_descent_frame`: the frame the fall
    flattened out, which is the landing, not the last frame the tracker
    still had something to follow. Those differ by the length of the
    bounce.

    FOR A GREEN CAMERA ONLY. Everything above rests on the view being
    quiet, and a tee camera is not: pointed at the same footage this
    returns 27 tracks on a green clip and 1611 on a tee clip, because a
    tee is full of people. Run on a tee view it will happily report
    twenty "descents". `n_tracks` in the result is the tell.

    Returns {ok, events, n_tracks, reason}. Never raises.
    """
    out: dict = {"ok": False, "events": [], "n_tracks": 0, "reason": None}
    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            out["reason"] = "could not open video"
            return out
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        _fps = float(fps or cap.get(cv2.CAP_PROP_FPS) or 30.0)
        cap.release()
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"could not read video: {exc}"
        return out
    if _fps <= 1.0:
        _fps = 30.0
    if frame_h <= 0 or n_frames <= 1:
        out["reason"] = "video reports no frames"
        return out

    f_lo, f_hi = window if window else (0, n_frames - 1)
    f_lo = max(0, int(f_lo))
    f_hi = min(max(f_lo + 1, int(f_hi)), n_frames - 1)

    # BOTH DETECTORS, unioned. Neither finds every descent alone --
    # measured on two clips, MOG2 saw one of the two landings with a
    # single detection and the plate saw the other with ten. Together
    # they find both, and the fall-rate band below is what keeps the
    # extra noise from the plate out of the answer.
    try:
        # NO BODY SUPPRESSION ON A GREEN VIEW. This function is for a
        # green camera and says so at the top; the golfer mask and the
        # busy-pixel filter are tee-camera machinery, and on a cloudy day
        # they mask out the green itself. A person walking the green is
        # not a problem here -- the fall-rate, drop and straightness
        # gates below throw them out, because a person does not fall at
        # a third of a frame-height per second in a straight line.
        det = detect_ball_blobs(input_path, f_lo, f_hi,
                                suppress_bodies=False,
                                cap_prefer="large")
        dets = list(det.get("dets") or [])
        n_mog = len(dets)
        dets.extend(detect_movers_by_plate(input_path, f_lo, f_hi))
        n_plate = len(dets) - n_mog
        # THE THIRD DETECTOR, and on a green view the one that works.
        # The other two model the scene and report departures from it,
        # which on smooth turf under moving cloud means the tree line;
        # this one asks only "was that there a frame ago", which is what
        # a falling ball is.
        dets.extend(detect_movers_by_diff(input_path, f_lo, f_hi, sens=2,
                                          per_frame=DESCENT_DIFF_PER_FRAME))
        n_diff = len(dets) - n_mog - n_plate
        dets.sort(key=lambda d: d["frame"])
        out["n_dets"] = len(dets)
        out["n_dets_mog2"] = n_mog
        out["n_dets_plate"] = n_plate
        out["n_dets_diff"] = n_diff
        # WHAT THE DETECTORS THREW AWAY, carried out. "No descent found"
        # has two completely different causes -- the ball was never
        # detected, or it was detected and never linked -- and they call
        # for opposite fixes. Without these counts the two look identical
        # from the outside, which is how a masked-out green reads as a
        # tracking problem.
        out["det_stats"] = dict(det.get("stats") or {})
        out["det_stats"]["plate"] = n_plate
        out["det_stats"]["diff"] = n_diff
        # EVERY DETECTION, to be drawn. A picture of the chains answers
        # "which tracks were found and why were they rejected"; it cannot
        # answer "was the ball detected at all", which is the prior
        # question and the one that decides whether the problem is in the
        # detector or in the linker. Capped so a busy clip cannot make
        # the payload enormous, keeping an even spread rather than the
        # first N, which would be one corner of the clip.
        _cap = 4000
        _step = max(1, len(dets) // _cap)
        out["dets_preview"] = [
            {"frame": int(d["frame"]), "x": int(d["x"]), "y": int(d["y"])}
            for d in dets[::_step]
        ]
        tracks = build_tracks(dets, r, min_len=int(min_points))
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"detection failed: {exc}"
        return out
    out["n_tracks"] = len(tracks)

    min_drop = float(min_drop_frac) * frame_h
    events = []
    for tk in tracks:
        pts = tk.get("points") or []
        if len(pts) < int(min_points):
            continue
        drop = float(pts[-1]["y"]) - float(pts[0]["y"])
        if drop < min_drop:
            continue
        span_f = max(1, int(pts[-1]["frame"]) - int(pts[0]["frame"]))
        rate = (drop / span_f) * _fps / frame_h
        if not (float(rate_lo) <= rate <= float(rate_hi)):
            continue
        bend = _path_bend_px(pts)
        if bend > float(max_bend_px):
            continue

        # WHERE IT LANDED, not where the tracker gave up. The seed keeps
        # producing points through the bounce and the roll, so the last
        # point of the track can be a second after the ball touched
        # down. Walk back to the last frame it was still falling at a
        # decent share of its peak speed.
        peak = 0.0
        vys = []
        for a, b in zip(pts, pts[1:]):
            df = max(1, int(b["frame"]) - int(a["frame"]))
            v = (float(b["y"]) - float(a["y"])) / df
            vys.append(v)
            peak = max(peak, v)
        last_i = len(pts) - 1
        if peak > 0:
            for i in range(len(vys) - 1, -1, -1):
                if vys[i] >= DESCENT_FLATTEN_FRAC * peak:
                    last_i = i + 1
                    break
        land = pts[last_i]
        events.append({
            "first_frame": int(pts[0]["frame"]),
            "last_frame": int(pts[-1]["frame"]),
            "last_descent_frame": int(land["frame"]),
            "last_descent_sec": round(int(land["frame"]) / _fps, 2),
            "landing_xy": [int(round(land["x"])), int(round(land["y"]))],
            "n_points": len(pts),
            "drop_px": int(round(drop)),
            "fall_rate": round(rate, 3),
            "bend_px": round(bend, 2),
            "peak_px_per_frame": round(peak, 1),
            # THE CHAIN ITSELF, up to the frame it stopped falling.
            # Callers that only want to know a ball came down are served
            # by the summary above; a caller that wants to DRAW the
            # descent needs the points, and re-finding them from the
            # landing is both slower and capable of a different answer.
            "points": [
                {"frame": int(q["frame"]),
                 "x": int(round(q["x"])), "y": int(round(q["y"]))}
                for q in pts[:last_i + 1]
            ],
        })

    # One descent can arrive as two tracks when the ball blinks out
    # against the tree line on the way down. Anything landing within a
    # second of an earlier event is the same shot; keep the one with
    # more evidence behind it.
    events.sort(key=lambda e: e["last_descent_frame"])
    merged: list[dict] = []
    for ev in events:
        if merged and (ev["last_descent_frame"] - merged[-1]["last_descent_frame"]) \
                <= merge_sec * _fps:
            if ev["n_points"] > merged[-1]["n_points"]:
                merged[-1] = ev
            continue
        merged.append(ev)

    out["events"] = merged[:int(max_events)]
    out["ok"] = bool(out["events"])
    out["fps"] = round(_fps, 3)
    out["reason"] = (
        f"{len(out['events'])} descent(s) in {len(tracks)} track(s): "
        + ", ".join(f"f{e['last_descent_frame']} ({e['last_descent_sec']}s)"
                    for e in out["events"])
        if out["events"] else
        f"no descent among {len(tracks)} track(s) — "
        f"nothing fell at {rate_lo}-{rate_hi} frame-heights/sec"
    )
    return out


def find_flight(
    input_path: Path,
    fps: float,
    impact_frame: int | None = None,
    head_xy=None,
    feet_xy=None,
    frame_w: int | None = None,
    frame_h: int | None = None,
    ball_side: str | None = None,
    rest_ball: dict | None = None,
    ball_locked: bool = False,
    debug_dir: Path | None = None,
    debug_prefix: str = "d3",
) -> dict:
    """THE pipeline. One implementation, used by produce and by Debug3.

    Debug3 and produce must not be two pieces of code that happen to agree
    today -- they had already drifted, with Debug3 taking its ball from the
    club arc at the launch frame while produce took the extrapolation, a
    difference measured at 3px versus 67px on the same swing. So this is the
    whole thing, and `debug_dir` is the only difference between the two
    callers: pass it and every stage also writes its image and its numbers
    into the returned `debug` block; leave it None and the identical
    arithmetic runs silently.

    Stages: blobs -> tracks -> RANSAC flight -> launch frame from the ground
    crossing -> ball from the club arc measured AT that frame.

    `ball_locked` makes `rest_ball` the ANSWER rather than a hint. By
    default this measures the ball itself from the club arc at the launch
    frame and that measurement wins -- correct when nobody knows better,
    and wrong the moment somebody does. An operator who placed the ball by
    eye in the edit wizard knows better, and having their placement
    silently replaced by a detector is worse than not offering the control
    at all. When locked, the club-arc pass is skipped entirely (it can no
    longer change anything, and it is not free).

    Returns {ok, ball, ball_source, launch_frame, points, reason, debug}.
    """
    out = {"ok": False, "ball": None, "ball_source": None,
           "launch_frame": None, "points": [], "reason": None, "debug": {}}
    dbg = out["debug"]
    if not HAS_CV:
        out["reason"] = "opencv not installed"
        return out
    try:
        if frame_w is None or frame_h is None:
            cap = cv2.VideoCapture(str(input_path))
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        else:
            n_frames = 0
        r = max(6.0, 0.012 * float(frame_h))
        if impact_frame is None:
            f_lo, f_hi = 0, (n_frames - 1 if n_frames else 400)
        else:
            f_lo = max(0, int(impact_frame) - WIN_PRE)
            # ONE WINDOW, NOT A PARAMETER. This used to take a win_post
            # override so the swing test could ask for 3 seconds where
            # produce uses WIN_POST. That made the panel meant to show
            # what produce does see a different number of frames than
            # produce, which is a way for the two to disagree about the
            # flight on the same swing -- the exact drift that sharing
            # this function was supposed to end.
            f_hi = int(impact_frame) + WIN_POST
        dbg["window"] = [f_lo, f_hi]
        dbg["r_px"] = round(r, 1)

        # WALL CLOCK PER PHASE. Which stage costs the run is not guessable
        # from the source: the MOG2 pass decodes and segments every frame of
        # the window while the RANSAC is a few hundred polyfits on at most a
        # few dozen points, and the debug image writes are pure overhead
        # that only the panel pays. Timing is a perf_counter read per phase,
        # so it costs nothing to leave on in production too.
        _laps: dict[str, float] = {}
        dbg["timing"] = _laps
        _mark = time.perf_counter()

        def _lap(name: str) -> None:
            nonlocal _mark
            now = time.perf_counter()
            _laps[name] = round(now - _mark, 3)
            _mark = now
            # Kept current so the two early returns below (no detections /
            # no flight) still report a total rather than a bare phase list.
            _laps["total"] = round(
                sum(v for k, v in _laps.items() if k != "total"), 3,
            )

        # A-C: detections.
        bbox = body_box_from_pose(head_xy, feet_xy, frame_w, frame_h)
        dbg["body_box"] = list(bbox) if bbox else None
        det = detect_ball_blobs(
            input_path, f_lo, f_hi, body_box=bbox,
            debug_dir=debug_dir, debug_prefix=f"{debug_prefix}blob",
        )
        _lap("detect")
        dbg["detect"] = {
            "reason": det.get("reason"), "stats": det.get("stats"),
            "max_area": det.get("max_area"), "max_side": det.get("max_side"),
            "n_at_strict_cap": det.get("n_at_strict_cap"),
            "images": det.get("images"),
        }
        _a = sorted(det.get("areas") or [])
        if _a:
            dbg["detect"]["area_summary"] = {
                "n": len(_a), "median": _a[len(_a) // 2],
                "p90": _a[int(0.90 * (len(_a) - 1))], "max": _a[-1],
            }
        if not det.get("ok"):
            out["reason"] = det.get("reason")
            return out

        # D: tracks.
        tracks = build_tracks(det.get("dets") or [], r)
        _lap("tracks")
        dbg["n_tracks"] = len(tracks)
        # The preview covers exactly the tracks that get drawn, and each
        # row carries the colour of its line — a table whose rows cannot
        # be matched to the picture is just numbers.
        _sel = select_preview_tracks(tracks)
        _shown = [tracks[_i] for _i, _ in _sel]
        # `idx` is the track's position in the FULL list — the same number
        # the tested-tracks table below uses. Numbering the shortlist 1..12
        # instead would give the same track two different names in two
        # tables on the same page, which is worse than no number.
        dbg["tracks_preview"] = [
            {"idx": _i + 1,
             "color": TRACK_COLORS[_k % len(TRACK_COLORS)],
             "why": _why,
             "n": len(tracks[_i]["points"]),
             "span_px": tracks[_i]["span_px"],
             "rise_px": tracks[_i]["rise_px"],
             "density": track_density(tracks[_i]["points"]),
             "from": [int(tracks[_i]["points"][0]["x"]),
                      int(tracks[_i]["points"][0]["y"])],
             "to": [int(tracks[_i]["points"][-1]["x"]),
                    int(tracks[_i]["points"][-1]["y"])],
             "frames": [tracks[_i]["points"][0]["frame"],
                        tracks[_i]["points"][-1]["frame"]]}
            for _k, (_i, _why) in enumerate(_sel)
        ]
        if debug_dir is not None:
            _canvas = (det.get("images") or {}).get("dets")
            if _canvas and (Path(debug_dir) / _canvas).exists():
                _tn = f"{debug_prefix}tracks.jpg"
                _nr = sum(1 for _, _w in _sel if "rises" in _w)
                if draw_tracks(
                    Path(debug_dir) / _canvas, Path(debug_dir) / _tn,
                    _shown,
                    f"TRACK CANDIDATES: {len(_sel)} of {len(tracks)} built "
                    f"-- the longest, plus the {_nr} that RISE most (a "
                    f"branch in the wind outlasts a struck ball, so length "
                    f"alone hides the shot). Numbers and colours match the "
                    f"table. Hollow ring = first frame, filled dot = last.",
                    scale=det.get("scale") or 1.0,
                    labels=[_i + 1 for _i, _ in _sel],
                ):
                    dbg["tracks_image"] = _tn
            _lap("draw_tracks")

        # E: the flight.
        gy = float(feet_xy[1]) if feet_xy and len(feet_xy) == 2 else None
        # The ball, BEFORE picking the flight, so the aim gate is armed.
        # This used to pass None, which made the aim / self-evident /
        # baseline gates dead code — and that is how a fit whose
        # back-extrapolated impact landed at (4828, -238) on a 1280x720
        # frame was accepted as a ball flight. The club arc knows where
        # the ball is; the flight picker just never got told.
        # `rest_ball` lets a caller that ALREADY ran this pass hand the
        # answer in rather than paying for it twice. Debug3 does exactly
        # that: it needs the ball before the club-fan judge runs, so it
        # measures it there and passes it down. Without it, this runs the
        # pass itself and nothing about the result differs.
        from .debug2 import club_bottom_ball

        _pre = rest_ball if rest_ball is not None else (
            club_bottom_ball(
                input_path, int(impact_frame or f_lo), fps,
                feet_xy=feet_xy, head_xy=head_xy, ball_side=ball_side,
                debug_dir=debug_dir, debug_prefix=f"{debug_prefix}hint",
            ) if feet_xy else {}
        )
        _ball_hint = _pre.get("xy") if _pre.get("ok") else None
        dbg["ball_hint"] = _ball_hint
        dbg["ball_hint_reason"] = _pre.get("reason")
        # The picture goes up whether or not the hint landed: when this
        # comes back empty the aim gate is disarmed, and the only way to
        # see WHY is to look at where it searched.
        dbg["ball_hint_image"] = _pre.get("image")
        _lap("club_arc")
        res = pick_flight(
            tracks, int(impact_frame or f_lo), _ball_hint,
            frame_w=frame_w, frame_h=frame_h, r=r,
        )
        _lap("flight")
        # Which coloured line the fit actually chose. Identified by its
        # points, not by index, because pick_flight ranks its own way and
        # an index would silently point at the wrong row the day that
        # changes.
        _win = {(p["frame"], p["x"], p["y"])
                for p in ((res.get("flight") or {}).get("track") or {})
                .get("points", [])}
        # Each drawn track's VERDICT, joined back onto its row. The answer
        # to "why wasn't the one I can see picked?" was sitting in a
        # 71-row collapsed table keyed by a number you had to go and find.
        # It belongs next to the line you are looking at.
        _by_idx = {t.get("idx"): t for t in (res.get("tried") or [])}
        for _row in dbg["tracks_preview"]:
            # NOT `_t` — that name is the timing dict this function's
            # `_lap` closure writes into. Rebinding it here pointed _lap
            # at a tested-track record instead, whose "frames" value is a
            # list, so the next _lap summed a list into a float and threw
            # `unsupported operand type(s) for +: 'int' and 'list'` --
            # which surfaced as the whole flight stage failing and a
            # produced clip never being built.
            _rec = _by_idx.get(_row["idx"]) or {}
            _row["verdict"] = _rec.get("verdict")
            _row["n_inliers"] = _rec.get("n_inliers")
            _row["rms_px"] = _rec.get("rms_px")
            _row["aim_px"] = _rec.get("aim_px")
            _row["score"] = _rec.get("score")
        if _win:
            for _row, (_i, _) in zip(dbg["tracks_preview"], _sel):
                _row["winner"] = ({(p["frame"], p["x"], p["y"])
                                   for p in tracks[_i]["points"]} == _win)
            # If the fit chose a track that did not make the shortlist,
            # say so rather than showing a table with no winner in it.
            dbg["winner_not_shown"] = not any(
                _r.get("winner") for _r in dbg["tracks_preview"])
        fit = res.get("fit") or {}
        dbg["flight"] = {
            "reason": res.get("reason"), "tried": res.get("tried"),
            "n_inliers": fit.get("n_inliers"), "rms_px": fit.get("rms_px"),
            "at_impact": fit.get("at_impact"),
            "x_degree": fit.get("x_degree"),
            "aim_px": fit.get("aim_px"), "aim_basis": fit.get("aim_basis"),
        }
        # THE CLICKABLE POOL, SET BEFORE THE FAILURE RETURN. These are
        # every ball-sized detection, not just the ones a fit kept, and
        # they are what click-to-plot offers the operator to click. They
        # used to be filled in below -- past the early return -- so a
        # flight that found nothing threw away the very evidence the
        # operator needs to plot the ball by hand. A failed flight is
        # exactly when the manual path matters.
        out["candidates"] = [
            {"frame": int(d["frame"]), "x": int(d["x"]), "y": int(d["y"])}
            for d in (det.get("dets") or [])
        ]
        if not res.get("ok"):
            out["reason"] = res.get("reason")
            return out
        # ACROSS THE GAP. The fit only ever saw one track, and the
        # tracker closes a track after four blind frames -- so the rest
        # of the ascent, sitting in a straight line with the part that
        # was found, was being left out. Anything on the same parabola
        # joins it.
        _grown = grow_flight(fit, det.get("dets") or [], r=r)
        dbg["grown"] = {
            "n_added": _grown.get("n_added"),
            "rms_px": _grown.get("rms_px"),
            "reason": _grown.get("reason"),
        }
        _fpts = (_grown.get("points")
                 if _grown.get("ok") and _grown.get("points")
                 else (fit.get("inliers") or []))
        out["points"] = [
            {"frame": int(q["frame"]), "x": int(q["x"]), "y": int(q["y"])}
            for q in _fpts
        ]
        # The launch FRAME, from where the flight meets the ground.
        lg = launch_from_ground(fit, gy)
        _lap("launch")
        dbg["launch"] = lg
        if lg.get("ok"):
            out["launch_frame"] = lg["frame"]
            out["ball"] = lg["xy"]
            out["ball_source"] = "flight extrapolated to the ground"
        else:
            out["launch_frame"] = int(impact_frame or f_lo)
            out["ball"] = fit.get("at_impact")
            out["ball_source"] = "the fit at the impact frame"

        # The ball POSITION, from the club arc measured at that frame. The
        # flight gives the better frame; the arc gives the better position,
        # measured at full resolution over the real downswing. On the swing
        # this was settled against, the arc landed 3px from a ball visible
        # in the check frame while the extrapolation was 67px right of it.
        _locked_xy = None
        if ball_locked and isinstance(rest_ball, dict):
            _rb = rest_ball.get("xy")
            if _rb and len(_rb) >= 2:
                _locked_xy = [float(_rb[0]), float(_rb[1])]

        if _locked_xy is not None:
            # The operator placed it. Record what the flight extrapolated
            # to as the alternative -- useful for judging the fit -- but
            # the placement is the answer.
            dbg["ball_alt"] = out["ball"]
            dbg["ball_alt_source"] = out["ball_source"]
            dbg["club_arc"] = {
                "reason": "skipped — the ball was placed by the operator",
            }
            out["ball"] = _locked_xy
            out["ball_source"] = "placed by the operator"
        elif out["launch_frame"] is not None:
            from .debug2 import club_bottom_ball

            club = club_bottom_ball(
                input_path, int(out["launch_frame"]), fps,
                feet_xy=feet_xy, head_xy=head_xy, ball_side=ball_side,
                debug_dir=debug_dir, debug_prefix=f"{debug_prefix}club",
            )
            _lap("club_arc")
            dbg["club_arc"] = {
                "frame": int(out["launch_frame"]), "xy": club.get("xy"),
                "reason": club.get("reason"), "image": club.get("image"),
                "vs_extrapolated_px": (
                    round(math.hypot(club["xy"][0] - out["ball"][0],
                                     club["xy"][1] - out["ball"][1]), 1)
                    if club.get("xy") and out["ball"] else None
                ),
            }
            if club.get("ok") and club.get("xy"):
                dbg["ball_alt"] = out["ball"]
                dbg["ball_alt_source"] = out["ball_source"]
                out["ball"] = club["xy"]
                out["ball_source"] = "club arc at the flight's launch frame"

        # THE ANSWER IS COMPLETE HERE. Everything below this line draws
        # pictures for the panel, and NONE of it may change the verdict.
        #
        # It used to sit inside the one big try, above the line that sets
        # `ok` — so a JPEG that failed to write, or an int() on a NaN in
        # a caption, turned an accepted flight into "no flight" and the
        # clip was never produced. The panel then showed a track marked
        # "accepted, score 15.89" next to a stage reading "0 flights
        # accepted", with nothing to say why.
        out["ok"] = bool(out["points"]) and out["ball"] is not None
        out["reason"] = (
            f"{len(out['points'])} tracer point(s), ball {out['ball']} "
            f"({out['ball_source']}), launch f{out['launch_frame']} "
            f"({fit.get('n_inliers')} inliers at {fit.get('rms_px')}px rms)"
        )
        if not out["ok"]:
            out["reason"] = (
                f"the fit was accepted but there is nothing to render: "
                f"{len(out['points'])} tracer point(s), ball {out['ball']}"
            )
        try:
            # A frame from before the strike with the answer ringed on it.
            if debug_dir is not None and out["ball"]:
                _rf = max(0, int(out["launch_frame"]) - 5)
                dbg["rest_check_frame"] = _rf
                dbg["rest_check_image"] = rest_check_image(
                    input_path, _rf, out["ball"], r, debug_dir,
                    debug_prefix=f"{debug_prefix}rest",
                )
                _lap("rest_check_image")

            # The flight drawing, on the detections canvas.
            if debug_dir is not None:
                _canvas = (det.get("images") or {}).get("dets")
                if _canvas and (Path(debug_dir) / _canvas).exists():
                    _nm = f"{debug_prefix}flight.jpg"
                    if draw_flight(
                        Path(debug_dir) / _canvas, Path(debug_dir) / _nm,
                        out["ball"], tracks,
                        (res.get("flight") or {}).get("track"), fit,
                        f"BALL: {out['ball_source']} {out['ball']} at f"
                        f"{out['launch_frame']}. {fit.get('n_inliers')} "
                        f"inliers, rms {fit.get('rms_px')}px. green=inliers "
                        f"red x=outliers cyan=fit magenta=impact "
                        f"grey=rejected",
                        scale=det.get("scale") or 1.0,
                    ):
                        dbg["flight_image"] = _nm
                _lap("draw_flight")
        except Exception as exc:  # noqa: BLE001
            # Say so loudly, but keep the flight.
            log.warning(
                "debug3 panel images failed (the flight is unaffected): %s",
                exc, exc_info=True,
            )
            dbg["images_error"] = f"{type(exc).__name__}: {exc}"
            try:
                import traceback as _tb

                dbg["images_traceback"] = "".join(
                    _tb.format_exception(type(exc), exc, exc.__traceback__),
                )[-2000:]
            except Exception:  # noqa: BLE001
                pass
        return out
    except Exception as exc:  # noqa: BLE001
        # exc_info, because "failed: cannot convert float NaN to integer"
        # names the symptom and not one line of the code that raised it.
        log.warning("debug3 find_flight failed: %s", exc, exc_info=True)
        out["reason"] = f"failed: {type(exc).__name__}: {exc}"
        out["failed"] = True
        # THE TRACEBACK, ON THE REPORT. The message alone sent us hunting
        # through the source for an operator that turned out not to be
        # there: "unsupported operand type(s) for +: 'int' and 'list'"
        # was chased through three wrong candidates because nothing said
        # WHICH line. The operator is looking at the panel, not at the
        # server's journal, so the answer belongs on the panel.
        try:
            import traceback as _tb

            out["traceback"] = "".join(
                _tb.format_exception(type(exc), exc, exc.__traceback__),
            )[-2000:]
            dbg["traceback"] = out["traceback"]
        except Exception:  # noqa: BLE001
            pass
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
    # A non-finite coefficient makes every comparison below False — nan is
    # not < 0 — so disc, the roots and the frame all come out nan, and
    # int(round(nan)) raises ValueError. That exception surfaced as the
    # whole flight stage failing, with the panel reporting "no flight" for
    # a track it had just marked accepted. Refuse up front instead.
    if not all(math.isfinite(v) for v in (a, b, c0, float(ground_y))):
        out["reason"] = "the fit has non-finite coefficients"
        return out
    if abs(a) < 1e-9:
        out["reason"] = "the fit is a straight line, it never meets the ground"
        return out
    disc = b * b - 4.0 * a * (c0 - float(ground_y))
    if not math.isfinite(disc) or disc < 0.0:
        out["reason"] = "the flight never reaches the ground line"
        return out
    t_lo = (-b - math.sqrt(disc)) / (2.0 * a)
    t_hi = (-b + math.sqrt(disc)) / (2.0 * a)
    t = min(t_lo, t_hi)
    if not math.isfinite(t):
        out["reason"] = "the ground crossing is not a finite frame number"
        return out
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


# A struck ball's vertical profile has ONE shape: it rises, it peaks, it
# falls. It cannot resume rising. Everything below measures how far a
# track departs from that.
#
# Deadband, as a fraction of the ball scale r, per frame. Detection
# centroids jitter by a pixel or two; without this, a dead-flat noise
# track and a real flight both look like they reverse constantly.
SHAPE_DEADBAND_R = 0.15
# Apex count. One rise→fall is the apex. More than one means the track
# went up, came down, and went up again — not a ball.
MAX_RISE_TO_FALL = 1
# Falling and then rising again is the physically impossible one. Zero is
# the honest answer; 1 is tolerated because a single mis-linked detection
# near the apex can manufacture one.
MAX_FALL_TO_RISE = 1


def flight_shape(points: list, r: float) -> dict:
    """How closely a track follows the one profile a struck ball can draw.

    Image y is DOWN-positive, so rising is dy < 0.

    Two guards keep this from firing on a real flight. Steps are
    normalised by the frame gap (`dy/df`), so a missed frame reads as one
    long step rather than a lurch; and steps below the deadband are
    treated as flat and dropped, so centroid jitter is not a reversal.

    Returns {n_rise_to_fall, n_fall_to_rise, monotonicity, directness,
    n_steps}. `monotonicity` is 1.0 for a clean rise-peak-fall and falls
    towards 0 as the track zig-zags; `directness` is net displacement
    over path length — 1.0 for a straight line, ~0 for something that
    wanders in place.
    """
    out = {"n_rise_to_fall": 0, "n_fall_to_rise": 0,
           "monotonicity": 0.0, "directness": 0.0, "n_steps": 0}
    pts = [p for p in (points or []) if p.get("y") is not None]
    if len(pts) < 3:
        return out

    dead = SHAPE_DEADBAND_R * float(r)
    signs = []
    path = 0.0
    for a, b in zip(pts, pts[1:]):
        df = max(1, int(b.get("frame", 0)) - int(a.get("frame", 0)))
        dy = (float(b["y"]) - float(a["y"])) / df
        path += math.hypot(float(b["x"]) - float(a["x"]),
                           float(b["y"]) - float(a["y"]))
        if abs(dy) >= dead:                 # flat steps carry no direction
            signs.append(1 if dy > 0 else -1)   # +1 falling, -1 rising

    out["n_steps"] = len(signs)
    for s0, s1 in zip(signs, signs[1:]):
        if s0 == -1 and s1 == 1:
            out["n_rise_to_fall"] += 1
        elif s0 == 1 and s1 == -1:
            out["n_fall_to_rise"] += 1

    reversals = out["n_rise_to_fall"] + out["n_fall_to_rise"]
    # One reversal is free — that's the apex.
    out["monotonicity"] = (
        1.0 if len(signs) < 2
        else max(0.0, 1.0 - max(0, reversals - 1) / float(len(signs) - 1))
    )
    if path > 0:
        out["directness"] = round(
            math.hypot(float(pts[-1]["x"]) - float(pts[0]["x"]),
                       float(pts[-1]["y"]) - float(pts[0]["y"])) / path, 3,
        )
    out["monotonicity"] = round(out["monotonicity"], 3)
    return out


def grow_flight(fit: dict, dets: list, r: float = 12.0,
                tol: float | None = None, max_reach: int = 90,
                n_iter: int = 8, min_run: int = 3,
                run_gap: int = 5) -> dict:
    """Pull in every OTHER detection that lies on the winning fit.

    The flight is fitted over ONE track, and build_tracks closes a track
    after four frames with nothing to link to. So a ball that MOG2 loses
    for five frames -- against a bright sky, behind a branch, through
    the top of a tree line -- comes back as a SECOND track, and the fit
    never sees it. On a real swing that left the tracer covering the
    strike and the first part of the ascent while the rest of the climb
    sat there in plain sight, in a straight line with the part that was
    drawn, in its own track, unused.

    The fitted parabola already says where those points would have to
    be. So this sweeps every detection, keeps the ones within `tol` of
    the model, refits over the union, and repeats -- the standard
    grow-and-refit, with the model doing the deciding rather than the
    tracker's gap rule.

    What stops it swallowing noise is that the model is a parabola, not
    a corridor: `tol` is the same distance RANSAC used to call a point
    an inlier, and a detection has to sit on the curve at ITS OWN FRAME.
    Drifting foliage a hundred frames away has to be in exactly the
    right place at exactly the right time to qualify. `max_reach` bounds
    how far past the measured track that is even considered, because an
    extrapolated parabola is a guess and a distant one is a bad guess.

    It runs until a pass finds nothing new, not for a fixed couple of
    rounds. A fit made from the first eight points of a flight is a
    parabola constrained over a quarter of a second, and its
    extrapolation is only good for a little way past that -- so the far
    end of the flight sits outside `tol` until the near end has been
    joined and the curve refitted through it. Reaching the whole flight
    takes several passes, and stopping at two recovered 13 of 22 points
    where running to convergence recovers all of them.

    Returns {ok, points, n_added, rms_px, reason}. Never raises.
    """
    out = {"ok": False, "points": [], "n_added": 0, "rms_px": None,
           "reason": None}
    inl = list((fit or {}).get("inliers") or [])
    coef = (fit or {}).get("coef") or {}
    if not HAS_CV or len(inl) < 3 or not coef.get("x") or not coef.get("y"):
        out["reason"] = "no fit to grow from"
        return out
    if tol is None:
        tol = max(6.0, 1.2 * r)
    try:
        cx = np.array(coef["x"], dtype=float)
        cy = np.array(coef["y"], dtype=float)
        keep = {(int(p["frame"]), int(p["x"]), int(p["y"])) for p in inl}
        f_lo = min(int(p["frame"]) for p in inl)
        f_hi = max(int(p["frame"]) for p in inl)
        pool = [d for d in (dets or [])
                if (int(d["frame"]), int(d["x"]), int(d["y"])) not in keep
                and f_lo - max_reach <= int(d["frame"]) <= f_hi + max_reach]
        pts = list(inl)
        added = 0
        # Two rounds of the same loop. The first only trusts points
        # within RANSAC's own tolerance, which is the right suspicion of
        # a curve fitted through three. Once that has converged the
        # curve is constrained by the whole flight, and holding it to a
        # tolerance set for three points leaves the far tail out -- 15
        # of 22 on the reference flight, the last seven all real. So the
        # second round relaxes it by half, and the rms check at the end
        # is what says whether that was earned.
        for _tol in (tol, tol * 1.5):
            for _ in range(max(1, int(n_iter))):
                have = {int(p["frame"]) for p in pts}
                best_at: dict[int, tuple] = {}
                for d in pool:
                    _f = int(d["frame"])
                    if _f in have:
                        continue
                    _e = math.hypot(float(d["x"]) - float(np.polyval(cx, _f)),
                                    float(d["y"]) - float(np.polyval(cy, _f)))
                    if _e > _tol:
                        continue
                    # One ball per frame: the nearest thing to where the
                    # model says it is.
                    if _f not in best_at or _e < best_at[_f][0]:
                        best_at[_f] = (_e, d)
                _new = [d for _e, d in best_at.values()]
                if not _new:
                    break
                pts = sorted(pts + [{"frame": int(d["frame"]), "x": int(d["x"]),
                                     "y": int(d["y"])} for d in _new],
                             key=lambda p: p["frame"])
                added += len(_new)
                _t = np.array([p["frame"] for p in pts], float)
                cx = np.polyfit(_t, np.array([p["x"] for p in pts], float),
                                2 if len(pts) >= 6 else 1)
                cy = np.polyfit(_t, np.array([p["y"] for p in pts], float), 2)
        if not added:
            out.update({"ok": True, "points": pts, "n_added": 0,
                        "reason": "no other detection lies on the flight"})
            return out
        # A CONTINUATION ARRIVES AS A RUN; NOISE ARRIVES ALONE.
        #
        # Filling a hole INSIDE the measured track is safe -- both ends
        # are pinned by real points, so the curve through the hole is
        # interpolation. Reaching out PAST the last measurement is
        # extrapolation, and an extrapolated parabola will eventually
        # pass near something. On the reference flight it picked up a
        # single stray at f1365 and a pair at f1372.
        #
        # A ball that MOG2 re-acquires gives several frames in a row on
        # the curve, because it is a ball and it is still there. So
        # points beyond the original span have to come with company:
        # part of a run of at least `min_run`, no more than `run_gap`
        # frames apart. Inside the span nothing is dropped.
        _outside = sorted((p for p in pts
                           if not (f_lo <= int(p["frame"]) <= f_hi)),
                          key=lambda p: p["frame"])
        _drop = set()
        _run: list = []
        for _p in _outside + [None]:
            if (_run and _p is not None
                    and int(_p["frame"]) - int(_run[-1]["frame"]) <= run_gap):
                _run.append(_p)
                continue
            if len(_run) < min_run:
                _drop |= {(int(q["frame"]), int(q["x"]), int(q["y"]))
                          for q in _run}
            _run = [] if _p is None else [_p]
        if _drop:
            pts = [p for p in pts
                   if (int(p["frame"]), int(p["x"]), int(p["y"]))
                   not in _drop]
            added -= len(_drop)
            if added <= 0:
                out.update({
                    "ok": True, "points": list(inl), "n_added": 0,
                    "rms_px": fit.get("rms_px"),
                    "reason": (f"{len(_drop)} detection(s) lay on the "
                               f"flight past its last measured point but "
                               f"none had {min_run} in a row to back it "
                               f"up — kept the original fit")})
                return out
            _t = np.array([p["frame"] for p in pts], float)
            cx = np.polyfit(_t, np.array([p["x"] for p in pts], float),
                            2 if len(pts) >= 6 else 1)
            cy = np.polyfit(_t, np.array([p["y"] for p in pts], float), 2)
        _t = np.array([p["frame"] for p in pts], float)
        _res = np.hypot(
            np.array([p["x"] for p in pts], float) - np.polyval(cx, _t),
            np.array([p["y"] for p in pts], float) - np.polyval(cy, _t))
        _rms = float(np.sqrt(float(np.mean(_res ** 2))))
        # A GROWN FIT THAT FITS WORSE IS NOT AN IMPROVEMENT. If pulling
        # the extra points in has bent the curve off the ones that were
        # measured cleanly, keep what RANSAC had.
        if _rms > max(tol, 1.6 * float(fit.get("rms_px") or tol)):
            out.update({"ok": True, "points": list(inl), "n_added": 0,
                        "rms_px": fit.get("rms_px"),
                        "reason": (f"{added} point(s) lay on the flight but "
                                   f"refitting through them raised the rms "
                                   f"to {_rms:.1f}px — kept the original fit")})
            return out
        out.update({
            "ok": True, "points": pts, "n_added": added,
            "rms_px": round(_rms, 2),
            "reason": (
                f"joined {added} detection(s) from outside the winning "
                f"track that sit on the same flight (within {tol:.0f}px), "
                f"spanning f{pts[0]['frame']}–f{pts[-1]['frame']} at "
                f"{_rms:.2f}px rms"),
        })
        return out
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"could not grow the flight: {exc}"
        out["points"] = list(inl)
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
    for _ti, tr in enumerate(tracks[:MAX_TRACKS_TESTED]):
        fit = ransac_parabola(tr["points"], impact_frame, ball_xy, r=r)
        _p0, _p1 = tr["points"][0], tr["points"][-1]
        rec = {
            # WHICH track this row is. Without an id and endpoints, a
            # 79-row table cannot be matched to anything on the picture --
            # the operator circles a trail on screen and has no way to
            # find its row, which is exactly what happened.
            "idx": _ti + 1,
            "frames": [int(_p0["frame"]), int(_p1["frame"])],
            "from": [int(_p0["x"]), int(_p0["y"])],
            "to": [int(_p1["x"]), int(_p1["y"])],
            "density": track_density(tr["points"]),
            "n_points": len(tr["points"]),
            "span_px": tr["span_px"], "rise_px": tr["rise_px"],
            "n_inliers": fit.get("n_inliers"), "rms_px": fit.get("rms_px"),
            "aim_px": fit.get("aim_px"),
            "aim_path_px": fit.get("aim_path_px"),
            "aim_at_impact_px": fit.get("aim_at_impact_px"),
            "aim_basis": fit.get("aim_basis"),
            "aim_frame": fit.get("aim_frame"),
            "verdict": None,
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
                    f"(limit {limit:.0f}, measured by "
                    f"{fit.get('aim_basis') or 'the fit at impact'})"
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
        # ── shape: does this track obey ball physics? ──────────────
        shape = flight_shape(tr["points"], r)
        rec.update({
            "monotonicity": shape["monotonicity"],
            "directness": shape["directness"],
            "n_rise_to_fall": shape["n_rise_to_fall"],
            "n_fall_to_rise": shape["n_fall_to_rise"],
        })
        # A ball rises, peaks once, and falls. It never resumes rising.
        # Observed: a 38-point blob inside a 50px box in the tree canopy
        # beat a 636px real flight by 0.21 because it had one more inlier.
        # Its dy sign flips constantly; the flight's does not.
        if shape["n_fall_to_rise"] > MAX_FALL_TO_RISE:
            rec["verdict"] = (
                f"falls then rises again x{shape['n_fall_to_rise']} — "
                f"not a ball flight"
            )
            out["tried"].append(rec)
            continue
        if shape["n_rise_to_fall"] > MAX_RISE_TO_FALL:
            rec["verdict"] = (
                f"peaks {shape['n_rise_to_fall']}x — a ball peaks once"
            )
            out["tried"].append(rec)
            continue

        # ── score, every term relative to the track's own scale ────
        # The old score was ~97% raw inlier count: span was capped at 2.0
        # (one inlier), and rms was compared in absolute pixels, which
        # REWARDED blobs — a tight cluster fits any parabola closely by
        # construction. Measured against its own span instead, the same
        # 3.78px that looked excellent on a 50px track (7.6% of span) is
        # correctly worse than 4.19px on a 637px flight (0.66%).
        _span = float(tr["span_px"] or 0.0)
        _npts = max(1, len(tr["points"]))
        span_frac = min(1.0, _span / max(1.0, 0.5 * float(frame_w)))
        inlier_frac = fit["n_inliers"] / float(_npts)
        rms_frac = min(
            1.0,
            (fit["rms_px"] or 0.0) / max(1.0, SELF_EVIDENT_RMS_SPAN * _span),
        )
        score = (
            10.0 * span_frac              # went somewhere
            + 10.0 * shape["monotonicity"]  # went there like a ball
            + 5.0 * inlier_frac           # and the parabola explains it
            - 5.0 * rms_frac              # scale-free residual
            + 0.05 * fit["n_inliers"]     # count breaks ties, nothing more
        )
        # What the old formula would have picked, carried alongside so a
        # disagreement is visible in the panel instead of silent.
        legacy = (
            2.0 * fit["n_inliers"]
            + 2.0 * span_frac
            - (fit["rms_px"] or 0.0) / 10.0
        )
        rec["verdict"] = (
            f"accepted, score {score:.2f}"
            + (" (aim disagrees)" if rec.get("aim_disagrees") else "")
        )
        rec["score"] = round(score, 2)
        rec["score_legacy"] = round(legacy, 2)
        out["tried"].append(rec)
        if out["flight"] is None or score > out["flight"]["score"]:
            out["flight"] = {"score": score, "track": tr, "fit": fit,
                             "note": rec.get("note"), "shape": shape}
    if out["flight"] is None:
        # Say WHICH test rejected them, and how close the best one came.
        # "none survived" sends you back to the panel to expand a table; the
        # caption should carry the answer.
        buckets: dict[str, int] = {}
        for rec in out["tried"]:
            v = str(rec.get("verdict") or "?")
            key = (
                "falls then rises again" if "falls then rises" in v
                else "peaks more than once" if "peaks" in v
                else "too few inliers" if "inliers" in v
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
    # Carry the winner's shape into the caption and the log — the number
    # that decides between a flight and a blob should be readable without
    # expanding the table.
    _sh = out["flight"].get("shape") or {}
    if _sh:
        out["shape"] = _sh
        out["reason"] += (
            f"; rises {_sh.get('monotonicity')} monotonic, "
            f"{_sh.get('directness')} direct"
        )
    _w = out["flight"].get("note")
    if _w:
        out["aim_disagrees"] = True
        out["reason"] += ". " + _w
    return out


# ONE palette, drawn by the backend and printed by the panel. The table
# below the picture is useless unless its swatch is the same colour as
# the line, so both read this list — hex here, converted to BGR at the
# point of drawing, rather than two lists that agree until one is edited.
# Chosen to stay apart from each other and from the frame: nothing in
# the fairway-green band, and nothing close enough to the orange body
# box or the blue/amber detection dots to be mistaken for them.
TRACK_COLORS = [
    "#e6194b", "#f58231", "#ffe119", "#bfef45", "#42d4f4", "#4363d8",
    "#911eb4", "#f032e6", "#fabed4", "#9a6324", "#800000", "#000075",
]
# How many tracks get a line and a table row. All 79 drawn at once is a
# ball of wool, and the caption says how many were left out.
TRACKS_DRAWN = 12


def select_preview_tracks(tracks: list, limit: int = TRACKS_DRAWN) -> list:
    """Which tracks get a line and a table row.

    Longest-first alone answers the wrong question. Over a 105-frame
    window a branch swaying in the wind is in view the whole time and
    banks 19 points, while a struck ball crosses the frame in a dozen —
    so ranking by duration ranks the trees above the shot, and on a rainy
    day at Snee Farm it pushed the actual flight off a 12-row list.

    So half the slots go to the longest tracks and half to the tallest
    RISING ones. Rise is the one shape only a struck ball makes: shimmer
    oscillates, walkers travel sideways, nothing else in frame goes UP
    and keeps going. A track cannot be the ball and be invisible here.

    Returns [(index_into_tracks, why)], ordered longest-first for display.
    """
    n = len(tracks or [])
    if n == 0:
        return []
    by_len = sorted(range(n), key=lambda i: -len(tracks[i]["points"]))
    # Rising, and SEEN on most of its frames. Rise alone is not enough:
    # the biggest risers in the Snee Farm run were 3-point tracks spanning
    # 900px, built by linking noise to noise across a third of the frame,
    # and they took every rise slot. A struck ball is detected on most of
    # the frames it crosses (0.72 there); those had 0.33. Sparse risers
    # are still eligible, just last — a real ball in bad light can drop
    # frames, and a hard cut-off would be the same class of mistake as
    # the gate that invented the junk.
    _rise = [i for i in range(n) if tracks[i]["rise_px"] > 0]
    by_rise = sorted(
        _rise,
        key=lambda i: (track_density(tracks[i]["points"]) < 0.5,
                       -tracks[i]["rise_px"]),
    )
    n_rise = min(limit // 2, len(by_rise))
    # Rising tracks are claimed FIRST — they are the ones that go missing.
    picked: dict[int, set] = {}
    for i in by_rise[:n_rise]:
        picked.setdefault(i, set()).add("rises")
    for i in by_len:
        if len(picked) >= limit:
            break
        picked.setdefault(i, set()).add("longest")
    # ...and anything that qualified both ways says so.
    for i in by_len[:limit]:
        if i in picked:
            picked[i].add("longest")
    for i in by_rise[:n_rise]:
        picked[i].add("rises")
    out = []
    for i in sorted(picked, key=lambda j: -len(tracks[j]["points"])):
        why = picked[i]
        out.append((i, "longest + rises" if len(why) > 1 else next(iter(why))))
    return out


def _track_bgr(i: int) -> tuple:
    """Palette entry i as an OpenCV BGR tuple."""
    hx = TRACK_COLORS[i % len(TRACK_COLORS)].lstrip("#")
    r, g, b = (int(hx[j:j + 2], 16) for j in (0, 2, 4))
    return (b, g, r)


def draw_tracks(
    canvas_path: Path,
    out_path: Path,
    tracks: list,
    caption: str,
    scale: float = 1.0,
    labels: list | None = None,
) -> bool:
    """Draw the track candidates as thin coloured polylines, numbered.

    The detections image answers "what did we see"; this answers "what did
    we think went with what", which is the question the tracks table is
    really about. Without it the table is a list of numbers with no way to
    tell which row is the arc through the sky and which is a branch moving
    in the wind.

    `tracks` is the ALREADY-SELECTED subset (see select_preview_tracks) —
    this draws what it is given, in order, so the table cannot disagree
    with the picture about which track is number 4. `labels` carries each
    one's number in the FULL list, so the number on the line is the same
    number the tested-tracks table uses.

    Same scale contract as draw_flight: detections are full-res, the
    canvas is the downscaled working frame.
    """
    if not HAS_CV:
        return False
    try:
        img = cv2.imread(str(canvas_path))
        if img is None:
            return False
        sc = float(scale) or 1.0

        def _p(px, py):
            return int(round(float(px) * sc)), int(round(float(py) * sc))

        for i, tr in enumerate(tracks):
            col = _track_bgr(i)
            pts = tr.get("points") or []
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, _p(a["x"], a["y"]), _p(b["x"], b["y"]),
                         col, 2, cv2.LINE_AA)
            if not pts:
                continue
            # Endpoints marked differently so the direction of travel is
            # readable: hollow ring where the track starts, filled dot
            # where it ends.
            cv2.circle(img, _p(pts[0]["x"], pts[0]["y"]), 5, col, 2,
                       cv2.LINE_AA)
            cv2.circle(img, _p(pts[-1]["x"], pts[-1]["y"]), 4, col, -1,
                       cv2.LINE_AA)
            _lbl = str(labels[i]) if labels and i < len(labels) else str(i + 1)
            _tx, _ty = _p(pts[0]["x"], pts[0]["y"])
            # Black underlay so the number survives a light background.
            for _c, _t in ((( 0, 0, 0), 3), (col, 1)):
                cv2.putText(img, _lbl, (_tx + 7, _ty - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _c, _t,
                            cv2.LINE_AA)
        _label(img, caption)
        cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("debug3 draw_tracks failed: %s", exc)
        return False


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
    "build_tracks", "find_flight", "detect_ball_blobs", "draw_flight",
    "follow_to_rest", "draw_ball_path", "bezier_continuation", "extrapolate_flight", "draw_bezier_continuation",
    "draw_tracks", "select_preview_tracks", "TRACK_COLORS", "TRACKS_DRAWN",
    "pick_flight",
    "ransac_parabola", "launch_from_ground", "rest_check_image", "refine_ball_from_flight",
]
