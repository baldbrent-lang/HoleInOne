"""Sky-phase probe: is the ball still findable past the blob detector?

The measurement behind the predictive-tracker design. It does not track
-- it probes. For every frame past the hand-off it predicts where the
ball should be, opens a small window there, and asks three detectors
what they see:

  diff  frame differencing against the previous frame -- what the
        current pipeline uses, included as the baseline to beat
  log   Laplacian-of-Gaussian blob response. Sky has no texture, so a
        small isolated extremum on a smooth background is exactly what
        this is good at and what differencing is bad at
  ncc   normalised cross-correlation against a template cut from the
        last confident detection, refreshed as it goes because the ball
        shrinks and its blur changes through the flight

The flight only advances when two detectors AGREE and each clears a
credibility floor. Every one of them returns its best-of-a-bad-lot when
nothing is there, and two such picks can land on the same pixel and
agree about a ball that is not in the frame -- which would be the worst
possible outcome, since it would send us building a tracker for a signal
the video does not contain.

Used by the /sky-probe admin endpoint and by backend/tools/sky_probe.py.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


DEFAULT_FLOORS = {"diff": 6.0, "log": 8.0, "ncc": 0.6}


# --------------------------------------------------------------------
# the model: where should the ball be next
# --------------------------------------------------------------------
def fit_state(points: list, window: int = 12, target=None,
              land_frame=None) -> dict | None:
    """x linear, y quadratic over the last `window` points.

    Deliberately re-fit over a sliding window rather than propagating one
    model the whole way: image-space vertical acceleration is not really
    constant under perspective -- the ball is flying away from the camera,
    so its apparent motion decelerates on top of gravity. A local fit
    absorbs that; a global one drifts.
    """
    pts = points[-window:]
    if len(pts) < 3:
        return None
    t = np.array([p["frame"] for p in pts], dtype=float)
    t0 = t[-1]
    t = t - t0
    x = np.array([p["x"] for p in pts], dtype=float)
    y = np.array([p["y"] for p in pts], dtype=float)
    try:
        cx = np.polyfit(t, x, 1)
        cy = np.polyfit(t, y, 2)
    except Exception:
        return None
    state = {"t0": float(t0), "cx": cx.tolist(), "cy": cy.tolist(),
             "t_land": None, "target": None}

    # PIN THE FAR END IF WE KNOW IT.
    #
    # Free extrapolation is the weak part of this: the model is fitted
    # over a dozen noisy centroids and then run forward for a second or
    # more, so a small error in the seed compounds into a prediction
    # that leaves the flight entirely -- which is how the search window
    # ends up over a treeline instead of a ball.
    #
    # But the far end is not unknown. The operator has already marked
    # the target (the green, 200 yards out on hole 1), and the ball
    # finishes there. So this stops being extrapolation and becomes
    # interpolation between two known points: horizontal speed says
    # WHEN the ball gets there, and the vertical fit is constrained to
    # pass through it. The curve can then only be wrong in the middle.
    #
    # `land_frame` pins WHEN as well as where. The solve below is done
    # once on the seed, where the whole remaining flight is still ahead;
    # every re-fit during the probe then reuses that answer rather than
    # re-deriving it from a window that is closer and closer to the
    # target -- which would shrink t_land toward zero and, in the last
    # frames, fail its own sanity guard and silently unpin the curve.
    if target is not None and len(target) >= 2:
        tx, ty = float(target[0]), float(target[1])
        vx = float(cx[0])
        # x IS THE CLOCK, so it has to be a working one. On a flight that
        # is near-vertical in the image -- the ball hit straight down the
        # camera axis -- x barely changes over the seed, and dividing by
        # that tiny, noise-dominated vx puts the landing anywhere at all.
        # A pin from a bad clock is worse than no pin, so require x to
        # carry a real share of the motion.
        speed = float(np.median(np.hypot(np.diff(x), np.diff(y)))) \
            if len(pts) > 1 else 0.0
        if land_frame is not None:
            t_land = float(land_frame) - float(t0)
            ok_land = t_land > 1.0
        elif abs(vx) > 1e-3 and abs(vx) >= 0.25 * speed:
            t_land = (tx - float(cx[1])) / vx
            # Only ahead of us, and not so far ahead that a near-zero
            # horizontal speed puts the landing in another minute.
            ok_land = 1.0 < t_land < 20.0 * max(1.0, len(pts))
        else:
            t_land, ok_land = 0.0, False
        if ok_land:
            # y = a·t² + b·t + c with y(t_land) = ty, so
            # c = ty - a·t_land² - b·t_land. Substituting leaves a
            # two-parameter least squares over the seed.
            A = np.column_stack([
                t * t - t_land * t_land,
                t - t_land,
            ])
            try:
                ab, *_ = np.linalg.lstsq(A, y - ty, rcond=None)
                a_, b_ = float(ab[0]), float(ab[1])
                c_ = ty - a_ * t_land * t_land - b_ * t_land
                state["cy"] = [a_, b_, c_]
                state["t_land"] = float(t_land)
                state["target"] = [tx, ty]
            except Exception:
                pass
    return state


def predict(state: dict, frame: int) -> tuple[float, float]:
    dt = frame - state["t0"]
    x = np.polyval(state["cx"], dt)
    y = np.polyval(state["cy"], dt)
    return float(x), float(y)


def predicted_step(state: dict, frame: int) -> float:
    """Pixels the ball is expected to move this frame. The search window
    is sized from THIS, not from a constant: at hand-off the ball is
    moving fastest -- exactly where a fixed 40x40 box would lose it --
    and by apex it barely moves, where the same box would be all noise."""
    x0, y0 = predict(state, frame - 1)
    x1, y1 = predict(state, frame)
    return math.hypot(x1 - x0, y1 - y0)


# --------------------------------------------------------------------
# the three detectors, each answering "what is in this window"
# --------------------------------------------------------------------
def det_diff(cur_g, prev_g, box, thresh=10):
    x0, y0, x1, y1 = box
    d = cv2.absdiff(cur_g[y0:y1, x0:x1], prev_g[y0:y1, x0:x1])
    _, th = cv2.threshold(d, thresh, 255, cv2.THRESH_BINARY)
    th = cv2.dilate(th, None, iterations=1)
    n, _lbl, stats, cents = cv2.connectedComponentsWithStats(th)
    best = None
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (1 <= area <= 400):
            continue
        score = float(d[int(cents[i][1]), int(cents[i][0])])
        if best is None or score > best[0]:
            best = (score, x0 + float(cents[i][0]), y0 + float(cents[i][1]),
                    area)
    if best is None:
        return None
    return {"x": best[1], "y": best[2], "score": best[0], "area": best[3]}


def det_log(cur_g, box, sigma=1.6):
    """Laplacian-of-Gaussian: the ball as a bright OR dark speck on a
    smooth background. Both polarities matter -- against bright sky the
    ball is often DARKER than its surroundings, which is the case the
    brightness gates in the existing tracer kept getting wrong."""
    x0, y0, x1, y1 = box
    crop = cur_g[y0:y1, x0:x1].astype(np.float32)
    if crop.size == 0:
        return None
    blur = cv2.GaussianBlur(crop, (0, 0), sigma)
    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    mag = np.abs(lap)
    # Ignore the border: a window edge is a discontinuity and always the
    # strongest response there.
    m = 3
    if mag.shape[0] <= 2 * m or mag.shape[1] <= 2 * m:
        return None
    inner = mag[m:-m, m:-m]
    idx = int(np.argmax(inner))
    cy, cx = np.unravel_index(idx, inner.shape)
    peak = float(inner[cy, cx])
    # Contrast against the window's own noise floor, so a flat sky and a
    # busy treeline are scored on the same scale.
    med = float(np.median(mag))
    mad = float(np.median(np.abs(mag - med))) or 1e-3
    return {
        "x": x0 + m + float(cx), "y": y0 + m + float(cy),
        "score": (peak - med) / (1.4826 * mad),   # robust z
        "polarity": "dark" if lap[m + cy, m + cx] > 0 else "bright",
    }


def det_ncc(cur_g, box, template):
    if template is None:
        return None
    x0, y0, x1, y1 = box
    crop = cur_g[y0:y1, x0:x1]
    th, tw = template.shape[:2]
    if crop.shape[0] < th or crop.shape[1] < tw:
        return None
    res = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
    _minv, maxv, _minl, maxl = cv2.minMaxLoc(res)
    return {
        "x": x0 + maxl[0] + tw / 2.0,
        "y": y0 + maxl[1] + th / 2.0,
        "score": float(maxv),
    }


def cut_template(gray, x, y, half=5):
    h, w = gray.shape[:2]
    x0, y0 = int(round(x - half)), int(round(y - half))
    x1, y1 = x0 + 2 * half + 1, y0 + 2 * half + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return gray[y0:y1, x0:x1].copy()


# --------------------------------------------------------------------
def seed_from_pipeline(video: Path, impact: int, ball, fps: float) -> list:
    """Run the real detector so the probe starts exactly where production
    starts -- no point measuring a hand-off the pipeline never reaches."""
    from app.services import debug3 as d3

    cap = cv2.VideoCapture(str(video))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    res = d3.find_flight(
        video, fps, impact_frame=impact, frame_w=fw, frame_h=fh,
        rest_ball={"ok": True, "xy": [float(ball[0]), float(ball[1])],
                   "reason": "sky_probe"},
        ball_locked=True,
    )
    return {
        "ok": bool(res.get("ok")),
        "reason": res.get("reason"),
        "points": res.get("points") or [],
        "candidates": res.get("candidates") or [],
    }


def probe(video: Path, seed_points: list, frames_dir: Path | None,
          max_frames: int, win_min: int, win_pad: float,
          floors: dict, target=None) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    NB = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    pts = sorted(seed_points, key=lambda p: p["frame"])
    handoff = int(pts[-1]["frame"])

    # THE SEED HAS TO BE ABLE TO CARRY AN EXTRAPOLATION.
    #
    # Three points is enough to SOLVE a quadratic and nowhere near enough
    # to trust one a hundred frames later. Marks clustered into a few
    # pixels over a few frames -- which is what you get plotting the ball
    # near apex, where it barely moves -- leave the curvature term
    # entirely to noise. The fit still succeeds; it just describes
    # nothing. Extrapolated, it shoots up, turns over, and comes back
    # down through the golfer, and every detector then follows it there.
    #
    # So refuse, and say which part is short. The operator can plot a few
    # more marks earlier in the flight, where the ball is moving.
    _span = int(pts[-1]["frame"]) - int(pts[0]["frame"])
    _travel = math.hypot(pts[-1]["x"] - pts[0]["x"], pts[-1]["y"] - pts[0]["y"])
    if len(pts) < 4:
        raise ValueError("need at least 4 seed points to extrapolate from")
    if _span < 4:
        raise ValueError(
            f"the seed spans only {_span} frames — plot marks further "
            "apart in time, early in the flight where the ball moves")
    if _travel < 12.0:
        raise ValueError(
            f"the seed travels only {_travel:.0f}px — too little motion "
            "to fit a flight to; plot marks earlier, nearer impact")

    # WHICH WAY THE BALL IS GOING, from the seed as a whole. Everything
    # below is checked against this: a ball in the tee view travels one
    # way until it lands, so a prediction or a pick that heads back the
    # other way is not the ball.
    ux, uy = (pts[-1]["x"] - pts[0]["x"]) / _travel, \
             (pts[-1]["y"] - pts[0]["y"]) / _travel

    state = fit_state(pts, target=target)
    if state is None:
        raise ValueError("could not fit a flight to the seed points")

    # Template from the hand-off frame itself.
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(handoff))
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"could not read the hand-off frame {handoff}")
    prev_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template = cut_template(prev_g, pts[-1]["x"], pts[-1]["y"])

    rows = []
    last_seen = {"frame": handoff, "x": float(pts[-1]["x"]),
                 "y": float(pts[-1]["y"])}
    end = handoff + max_frames if NB <= 0 else min(handoff + max_frames, NB - 1)
    # WHEN the ball gets to the target, solved once off the seed while
    # the whole flight is still ahead. Every re-fit below is handed this
    # rather than re-solving it, and the probe stops there: anything
    # found past the target is something else.
    land_f = None
    if state.get("t_land") is not None:
        land_f = int(round(state["t0"] + state["t_land"]))
        end = min(end, land_f)
    for f in range(handoff + 1, end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        cur_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        px, py = predict(state, f)
        step = predicted_step(state, f)

        # THE MODEL MAY NOT DOUBLE BACK. A parabola fitted in image space
        # has an apex, and past it the curve descends -- back down the
        # way it came, toward the tee. The ball does not do that: it is
        # flying away from the camera, so it climbs toward the horizon
        # and then settles at the green, never returning to where it was
        # struck. Once the predicted step points back along the launch
        # direction the fit has left the flight, and probing on just
        # feeds the detectors a window over the golfer.
        _ax, _ay = predict(state, f - 1)
        if (px - _ax) * ux + (py - _ay) * uy < 0:
            rows.append({"frame": f, "reason": "prediction turned back"})
            break
        # Window: the predicted step plus padding, floored so it never
        # collapses at apex.
        half = int(max(win_min, round(step * win_pad)))
        x0 = max(0, int(px) - half); x1 = min(W, int(px) + half)
        y0 = max(0, int(py) - half); y1 = min(H, int(py) + half)
        if x1 - x0 < 8 or y1 - y0 < 8:
            rows.append({"frame": f, "reason": "window left the frame"})
            break
        box = (x0, y0, x1, y1)

        got = {
            "diff": det_diff(cur_g, prev_g, box),
            "log": det_log(cur_g, box),
            "ncc": det_ncc(cur_g, box, template),
        }
        row = {
            "frame": f,
            "pred": [round(px, 1), round(py, 1)],
            "step_px": round(step, 2),
            "half_px": half,
            # How smooth the window is. Sky is near-flat; a treeline is
            # not. The whole premise is that the detectors behave
            # differently across that line, so it has to be measured.
            "win_std": round(float(cur_g[y0:y1, x0:x1].std()), 2),
        }
        # MOTION IS THE OTHER HALF OF THE FILTER. A score floor says the
        # window contained something with the right LOOK; it says nothing
        # about whether that thing is where a ball could have got to. The
        # golfer's shoulder and a moving branch both score well, and
        # following them is what draws those zig-zags back down the
        # frame. So each pick is also asked to move like the ball: onward
        # along the launch direction, and no further than the model says
        # is possible.
        _lead = math.hypot(last_seen["x"] - px, last_seen["y"] - py)
        _reach = max(win_min, step * (f - last_seen["frame"]) * 2.0) + _lead
        for k, v in got.items():
            if v is None:
                row[k] = None
                continue
            _dx = v["x"] - last_seen["x"]
            _dy = v["y"] - last_seen["y"]
            _onward = _dx * ux + _dy * uy
            _jump = math.hypot(_dx, _dy)
            row[k] = {
                "xy": [round(v["x"], 1), round(v["y"], 1)],
                "err_px": round(math.hypot(v["x"] - px, v["y"] - py), 1),
                "score": round(v["score"], 3),
                # Kept in the report even when it fails: the rejected
                # scores are what the floors get calibrated against.
                "credible": bool(v["score"] >= floors[k]),
                # ...and separately, whether a ball could be there at all.
                "plausible": bool(_onward >= -1.0 and _jump <= _reach),
                "onward_px": round(float(_onward), 1),
            }
            if "polarity" in v:
                row[k]["polarity"] = v["polarity"]
        # AGREEMENT is the number that matters -- but only between
        # detections that are credible ON THEIR OWN. Every one of these
        # returns its best-of-a-bad-lot when there is nothing there: NCC
        # reports its highest correlation however low, LoG the strongest
        # extremum in the noise. Two such picks can land on the same
        # pixel and "agree" about a ball that is not in the frame. That
        # failure is worse than finding nothing, because it would send us
        # building a tracker for a signal the video does not contain.
        #
        # The floors below are STARTING POINTS measured on a synthetic
        # clip, not truth. Calibrating them against real sky is part of
        # what this harness is for -- run with --min-log-z 0 to see the
        # raw scores and pick the knee.
        picks = [
            (k, got[k]) for k in ("diff", "log", "ncc")
            if got[k] and row[k]["credible"] and row[k]["plausible"]
        ]
        agree = []
        for i in range(len(picks)):
            for j in range(i + 1, len(picks)):
                a, b = picks[i][1], picks[j][1]
                if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) <= 4.0:
                    agree.append(f"{picks[i][0]}+{picks[j][0]}")
        row["agree"] = agree
        rows.append(row)

        if frames_dir is not None:
            vis = frame.copy()
            cv2.rectangle(vis, (x0, y0), (x1, y1), (60, 200, 255), 1)
            cv2.drawMarker(vis, (int(px), int(py)), (60, 200, 255),
                           cv2.MARKER_CROSS, 12, 1)
            for k, col in (("diff", (0, 0, 255)), ("log", (0, 255, 0)),
                           ("ncc", (255, 0, 255))):
                if got[k]:
                    cv2.circle(vis, (int(got[k]["x"]), int(got[k]["y"])),
                               5, col, 1)
            cv2.putText(vis, f"f{f} step={step:.1f} win={half} "
                             f"agree={','.join(agree) or '-'}",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(frames_dir / f"probe_{f:06d}.png"), vis)

        # Advance the model only on agreement -- following a single
        # detector is how a probe turns into a tracker that wanders off
        # into the treeline and reports success.
        if agree:
            best = max(picks, key=lambda kv: 1 if kv[0] == "log" else 0)[1]
            last_seen = {"frame": f, "x": best["x"], "y": best["y"]}
            pts.append({"frame": f, "x": best["x"], "y": best["y"]})
            _st = fit_state(pts, target=target, land_frame=land_f)
            if _st:
                state = _st
            _t = cut_template(cur_g, best["x"], best["y"])
            if _t is not None:
                template = _t
        prev_g = cur_g
    cap.release()

    probed = [r for r in rows if "pred" in r]
    def _hits(k):
        return [r for r in probed if r.get(k)]
    summary = {
        "handoff_frame": handoff,
        "target": list(target) if target else None,
        "landing_frame": land_f,
        "frames_probed": len(probed),
        "stopped_because": next(
            (r["reason"] for r in reversed(rows) if r.get("reason")), None),
        "last_agreed_frame": last_seen["frame"],
        "frames_extended": last_seen["frame"] - handoff,
        "agreement_frames": sum(1 for r in probed if r.get("agree")),
        "per_detector": {
            k: {
                "found": len(_hits(k)),
                "credible": sum(1 for r in _hits(k) if r[k]["credible"]),
                # Cleared the floor AND could physically be the ball.
                # The gap between these two is the foliage.
                "usable": sum(1 for r in _hits(k)
                              if r[k]["credible"] and r[k]["plausible"]),
                "floor": floors[k],
                "median_err_px": round(float(np.median(
                    [r[k]["err_px"] for r in _hits(k)])), 2) if _hits(k) else None,
                "median_score": round(float(np.median(
                    [r[k]["score"] for r in _hits(k)])), 3) if _hits(k) else None,
            }
            for k in ("diff", "log", "ncc")
        },
        "window_std": {
            "median": round(float(np.median(
                [r["win_std"] for r in probed])), 2) if probed else None,
        },
    }
    return {"summary": summary, "rows": rows}




# --------------------------------------------------------------------
# what the operator looks at
# --------------------------------------------------------------------
def render_montage(video: Path, rep: dict, out_path: Path,
                   cell: int = 96, cols: int = 10, max_cells: int = 60):
    """A contact sheet of the probed windows, upscaled.

    The summary numbers say a detector "found" something; only the
    picture says whether that something was the ball. Each cell is the
    search window blown up with the prediction (cross) and each
    detector's pick marked, so a run that quietly followed a leaf is
    obvious at a glance rather than hidden behind a good-looking
    agreement rate.
    """
    rows = [r for r in rep.get("rows") or [] if "pred" in r]
    if not rows:
        return None
    rows = rows[:max_cells]
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    n = len(rows)
    ncols = min(cols, n)
    nrows = (n + ncols - 1) // ncols
    pad = 2
    sheet = np.full(
        ((cell + 16) * nrows + pad, cell * ncols + pad, 3), 24, np.uint8,
    )
    try:
        for i, r in enumerate(rows):
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(r["frame"]))
            ok, frame = cap.read()
            if not ok:
                continue
            px, py = r["pred"]
            half = max(8, int(r.get("half_px") or 16))
            H, W = frame.shape[:2]
            x0 = max(0, int(px) - half); x1 = min(W, int(px) + half)
            y0 = max(0, int(py) - half); y1 = min(H, int(py) + half)
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            sc = cell / max(crop.shape[0], crop.shape[1])
            vis = cv2.resize(crop, None, fx=sc, fy=sc,
                             interpolation=cv2.INTER_NEAREST)
            vis = cv2.copyMakeBorder(
                vis, 0, max(0, cell - vis.shape[0]),
                0, max(0, cell - vis.shape[1]),
                cv2.BORDER_CONSTANT, value=(24, 24, 24),
            )[:cell, :cell]
            # Prediction, then each detector that cleared its floor.
            cv2.drawMarker(vis, (int((px - x0) * sc), int((py - y0) * sc)),
                           (60, 200, 255), cv2.MARKER_CROSS, 9, 1)
            for k, col in (("diff", (0, 0, 255)), ("log", (0, 255, 0)),
                           ("ncc", (255, 0, 255))):
                d = r.get(k)
                if not d or not d.get("credible"):
                    continue
                cv2.circle(vis, (int((d["xy"][0] - x0) * sc),
                                 int((d["xy"][1] - y0) * sc)), 6, col, 1)
            gy = (i // ncols) * (cell + 16) + pad
            gx = (i % ncols) * cell + pad
            sheet[gy:gy + cell, gx:gx + cell] = vis
            # Agreement in the caption: the cells to trust at a glance.
            lab = f"f{r['frame']}"
            if r.get("agree"):
                lab += " *"
            cv2.putText(sheet, lab, (gx + 2, gy + cell + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (120, 255, 120) if r.get("agree") else (170, 170, 170),
                        1, cv2.LINE_AA)
    finally:
        cap.release()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return out_path
