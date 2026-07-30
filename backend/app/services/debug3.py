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

# Seed points for the parabola RANSAC. 14 gives C(14,3) = 364 candidate
# fits, which is fast and plenty for a curve with three parameters.
RANSAC_SEED_PTS = 14

# MOG2 needs to have seen the scene before the window or the golfer's whole
# body reads as foreground on frame one.
WARMUP_FRAMES = 40


# ── stages A-C: MOG2, drop the golfer, keep ball-sized blobs ────────────

def ball_area_cap(frame_h: int) -> tuple[float, float]:
    """(max area, max bbox side) for a ball at this frame height."""
    d = max(6.0, BALL_DIAM_FRAC * float(frame_h))
    return BALL_AREA_BLEED * (math.pi / 4.0) * d * d, 3.0 * d


def detect_ball_blobs(
    input_path: Path,
    f0: int,
    f1: int,
    max_area: int | None = None,
    min_area: int = BALL_AREA_MIN,
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
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # "Big" = the golfer. Scaled to the frame so it survives a change of
        # camera or resolution.
        golfer_area = max(400.0, 0.0008 * float(w) * float(h))
        golfer_h = 0.12 * float(h)
        # The ball-size window, derived from the frame unless overridden.
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
            fg = mog.apply(fr)
            if f < int(f0):
                continue                # warm-up only
            if base is None:
                base = fr.copy()
            out["stats"]["frames"] += 1
            # Open to kill single-pixel speckle without eating a 3px ball.
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
            n, lab, stats, cent = cv2.connectedComponentsWithStats(fg, 8)
            out["stats"]["components"] += max(0, n - 1)

            # Pass 1: find the golfer. Big blobs are not candidates, they are
            # an exclusion zone — a ball crossing the body cannot be told
            # from a moving sleeve, so we do not try.
            golfer = np.zeros((h, w), np.uint8)
            big = False
            for i in range(1, n):
                a = int(stats[i, cv2.CC_STAT_AREA])
                bh = int(stats[i, cv2.CC_STAT_HEIGHT])
                if a >= golfer_area or bh >= golfer_h:
                    golfer[lab == i] = 255
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
                d = {"frame": f, "x": cx, "y": cy, "area": a,
                     "w": bw, "h": bh}
                dets.append(d)
                kept_this.append(d)
                out["stats"]["kept"] += 1

            if len(kept_this) > best_frame_n:
                best_frame_n = len(kept_this)
                best_frame_img = fr.copy()
                best_frame_draw = (kept_this, golfer.copy(), f)

        cap.release()
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
            f"dropped {s['golfer']} golfer, {s['on_golfer']} on the golfer, "
            f"{s['too_big']} too big, {s.get('too_long', 0)} too long, "
            f"{s['too_small']} too small). "
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
                    cv2.circle(img, (int(d["x"]), int(d["y"])),
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
            span = max(1, int(f1) - int(f0))
            for d in dets:
                t = (d["frame"] - int(f0)) / span
                col = (int(255 * (1 - t)), int(80 + 100 * t), int(255 * t))
                cv2.circle(img, (int(d["x"]), int(d["y"])), 4, col, -1,
                           cv2.LINE_AA)
            _label(
                img,
                f"all {len(dets)} kept detections, f{f0}-{f1}. "
                f"blue = early, red = late",
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
    """Link detections into constant-velocity tracks.

    A nearest-neighbour tracker with a predicted position and a gate that
    widens with the frame gap. This is what a Kalman filter buys you for a
    ballistic target — a prediction plus an uncertainty that grows when you
    miss a frame — without the covariance tuning, and it is inspectable:
    every accepted link is a distance you can print.

    Velocity is smoothed as the track grows, so one noisy detection bends
    the prediction a little rather than throwing it off.

    Returns tracks sorted longest-first, each {points, span_px, rise_px}.
    """
    pool = sorted(
        [d for d in (dets or [])],
        key=lambda d: (d["frame"], d["x"]),
    )
    n = len(pool)
    tracks: list[dict] = []
    if n < min_len:
        return tracks
    # Index by frame. Seeding over every PAIR and then scanning the whole
    # pool to extend is O(n^3) in Python, and a two-second window with a few
    # detections per frame is 500+ points -- tens of millions of operations
    # for work that only ever looks a few frames ahead. Both the seeding and
    # the extension are frame-local, so index once and the cost becomes
    # linear in detections times the square of the per-frame count.
    by_frame: dict[int, list] = {}
    for d in pool:
        by_frame.setdefault(int(d["frame"]), []).append(d)
    frames = sorted(by_frame)

    def _ahead(f: int):
        """Detections in frames f+1 .. f+max_gap."""
        for g in range(f + 1, f + max_gap + 1):
            for d in by_frame.get(g, ()):
                yield g, d

    for a in pool:
        for df0, b in _ahead(int(a["frame"])):
            df0 = df0 - int(a["frame"])
            vx = (b["x"] - a["x"]) / df0
            vy = (b["y"] - a["y"]) / df0
            # A ball in flight moves. A pair that has not moved is two
            # frames of the same piece of background noise.
            if math.hypot(vx, vy) < 0.35 * r:
                continue
            run = [a, b]
            while True:
                last = run[-1]
                best_c = None
                best_d = None
                for gf, c in _ahead(int(last["frame"])):
                    df = gf - int(last["frame"])
                    px = last["x"] + vx * df
                    py = last["y"] + vy * df
                    gate = (1.5 + 0.9 * df) * r
                    dist = math.hypot(c["x"] - px, c["y"] - py)
                    if dist > gate:
                        continue
                    # NEAREST neighbour, not first-found: with several
                    # detections in a frame the closest to the prediction is
                    # the one the tracker should take.
                    if best_d is None or dist < best_d:
                        best_c, best_d, best_df = c, dist, df
                if best_c is None:
                    break
                nvx = (best_c["x"] - last["x"]) / best_df
                nvy = (best_c["y"] - last["y"]) / best_df
                # Smoothed velocity update — the Kalman-lite step.
                vx = 0.5 * vx + 0.5 * nvx
                vy = 0.5 * vy + 0.5 * nvy
                run.append(best_c)
            if len(run) < min_len:
                continue
            span = math.hypot(run[-1]["x"] - run[0]["x"],
                              run[-1]["y"] - run[0]["y"])
            tracks.append({
                "points": [
                    {"frame": int(p["frame"]), "x": int(p["x"]),
                     "y": int(p["y"]), "area": int(p["area"])}
                    for p in run
                ],
                "span_px": round(span, 1),
                "rise_px": round(run[0]["y"] - run[-1]["y"], 1),
            })
    # Collapse near-duplicates. The seed loop finds the same flight from
    # every starting pair along it, so one ball produces dozens of tracks
    # differing by an endpoint. A strict subset test barely helps -- they
    # are not subsets, they overlap -- so drop anything sharing most of its
    # points with a longer track already kept. Without this a real flight
    # can be pushed past the caller's try-limit by 40 copies of a bird.
    tracks.sort(key=lambda t: -len(t["points"]))
    kept: list[dict] = []
    keysets: list[set] = []
    for t in tracks:
        keyset = {(p["frame"], p["x"], p["y"]) for p in t["points"]}
        if any(
            len(keyset & ks) >= 0.7 * len(keyset) for ks in keysets
        ):
            continue
        kept.append(t)
        keysets.append(keyset)
    return kept


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
            cx = np.polyfit(ti, xi, 1)
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
        cx = np.polyfit(t[inl], x[inl], 1)
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
    out["reason"] = (
        f"{out['n_inliers']}/{len(points)} points on a parabola, "
        f"rms {out['rms_px']}px"
        + (f"; run back to impact it lands {out['aim_px']}px from the ball"
           if out["aim_px"] is not None else "")
    )
    return out


def pick_flight(
    tracks: list,
    impact_frame: int,
    ball_xy=None,
    frame_w: int = 1920,
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
    for tr in tracks[:40]:
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
            limit = min(max(120.0, 0.20 * float(frame_w)),
                        max(60.0, 0.30 * d0))
            if fit["aim_px"] > limit:
                rec["verdict"] = (
                    f"aims {fit['aim_px']:.0f}px from the ball "
                    f"(limit {limit:.0f})"
                )
                out["tried"].append(rec)
                continue
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
        rec["verdict"] = f"accepted, score {score:.2f}"
        rec["score"] = round(score, 2)
        out["tried"].append(rec)
        if out["flight"] is None or score > out["flight"]["score"]:
            out["flight"] = {"score": score, "track": tr, "fit": fit}
    if out["flight"] is None:
        out["reason"] = (
            f"{len(tracks)} track(s) built, none survived the flight tests"
        )
        return out
    out["ok"] = True
    out["fit"] = out["flight"]["fit"]
    out["reason"] = out["flight"]["fit"]["reason"]
    return out


def draw_flight(
    canvas_path: Path,
    out_path: Path,
    ball_xy,
    tracks: list,
    winner: dict | None,
    fit: dict | None,
    caption: str,
) -> bool:
    """Draw every track dim, the winner bright, and the fitted parabola."""
    if not HAS_CV:
        return False
    try:
        img = cv2.imread(str(canvas_path))
        if img is None:
            return False
        h, w = img.shape[:2]
        win_pts = {(p["frame"], p["x"], p["y"])
                   for p in (winner or {}).get("points", [])}
        for tr in tracks[:40]:
            pts = tr["points"]
            if {(p["frame"], p["x"], p["y"]) for p in pts} == win_pts:
                continue
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, (a["x"], a["y"]), (b["x"], b["y"]),
                         (120, 120, 120), 1, cv2.LINE_AA)
        if fit and fit.get("coef"):
            cx, cy = fit["coef"]["x"], fit["coef"]["y"]
            fs = [p["frame"] for p in (winner or {}).get("points", [])]
            if fs:
                prev = None
                for f in range(min(fs) - 2, max(fs) + 3):
                    px = int(np.polyval(cx, float(f)))
                    py = int(np.polyval(cy, float(f)))
                    if prev is not None:
                        cv2.line(img, prev, (px, py), (255, 200, 0), 2,
                                 cv2.LINE_AA)
                    prev = (px, py)
            for p in fit.get("outliers") or []:
                cv2.drawMarker(img, (int(p["x"]), int(p["y"])), (0, 0, 255),
                               cv2.MARKER_TILTED_CROSS, 12, 2)
            for p in fit.get("inliers") or []:
                cv2.circle(img, (int(p["x"]), int(p["y"])),
                           max(7, int(0.010 * h)), (0, 255, 0), 2,
                           cv2.LINE_AA)
            if fit.get("at_impact"):
                cv2.drawMarker(img, tuple(fit["at_impact"]), (255, 0, 255),
                               cv2.MARKER_CROSS, 22, 2)
        if ball_xy and len(ball_xy) == 2:
            cv2.circle(img, (int(ball_xy[0]), int(ball_xy[1])),
                       max(10, int(0.014 * h)), (0, 255, 0), 3, cv2.LINE_AA)
        _label(img, caption)
        cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("debug3 draw_flight failed: %s", exc)
        return False


__all__ = [
    "BALL_AREA_MIN", "BALL_AREA_STRICT", "ball_area_cap",
    "MIN_KEPT_FOR_TRACKING", "WIN_POST", "WIN_PRE",
    "build_tracks", "detect_ball_blobs", "draw_flight", "pick_flight",
    "ransac_parabola",
]
