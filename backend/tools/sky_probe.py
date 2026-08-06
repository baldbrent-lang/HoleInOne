#!/usr/bin/env python3
"""Measure where MOG2 dies and whether anything can pick the ball up after.

This answers ONE question, on real clips, before any tracker gets built:
past the point where the blob detector loses the ball, is there still a
detectable ball in the frame -- and which detector finds it?

It does not track. It probes. For every frame after the hand-off it
predicts where the ball should be, opens a small window there, and asks
three detectors what they see:

  diff  frame differencing against the previous frame -- what the
        current pipeline uses, included as the baseline to beat
  log   Laplacian-of-Gaussian blob response. Sky has no texture, so a
        small isolated extremum on a smooth background is exactly what
        this is good at and what differencing is bad at
  ncc   normalised cross-correlation against a template cut from the
        last confident detection, refreshed as it goes because the ball
        shrinks and its blur changes through the flight

Then it reports, per swing: the hand-off frame, how many frames each
detector plausibly extended the flight, how often they agreed, and how
far each pick sat from the prediction. Agreement between two independent
detectors is the signal worth trusting -- one detector alone will always
find SOMETHING in a 60x60 window.

Usage
-----
Seed from the existing pipeline (needs an impact frame and a ball):

    python3 backend/tools/sky_probe.py --video tee.mp4 \\
        --impact 1604 --ball 1180,655 --out report.json

Or seed from points the operator plotted in click-to-plot, which is the
honest way to probe a swing the pipeline failed on:

    python3 backend/tools/sky_probe.py --video tee.mp4 \\
        --seed seed.json --out report.json

    seed.json: [{"frame": 1604, "x": 1180, "y": 655}, ...]

Add --frames DIR to write an annotated PNG per probed frame: the window,
the prediction, and what each detector picked. Looking at twenty of
those tells you more than any summary number.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# --------------------------------------------------------------------
# the model: where should the ball be next
# --------------------------------------------------------------------
def fit_state(points: list, window: int = 12) -> dict | None:
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
    return {"t0": float(t0), "cx": cx.tolist(), "cy": cy.tolist()}


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
          floors: dict) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    NB = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    pts = sorted(seed_points, key=lambda p: p["frame"])
    handoff = int(pts[-1]["frame"])
    state = fit_state(pts)
    if state is None:
        raise SystemExit("need at least 3 seed points to fit a model")

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
    for f in range(handoff + 1, end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        cur_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        px, py = predict(state, f)
        step = predicted_step(state, f)
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
        for k, v in got.items():
            if v is None:
                row[k] = None
                continue
            row[k] = {
                "xy": [round(v["x"], 1), round(v["y"], 1)],
                "err_px": round(math.hypot(v["x"] - px, v["y"] - py), 1),
                "score": round(v["score"], 3),
                # Kept in the report even when it fails: the rejected
                # scores are what the floors get calibrated against.
                "credible": bool(v["score"] >= floors[k]),
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
            if got[k] and got[k]["score"] >= floors[k]
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
            _st = fit_state(pts)
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
        "frames_probed": len(probed),
        "last_agreed_frame": last_seen["frame"],
        "frames_extended": last_seen["frame"] - handoff,
        "agreement_frames": sum(1 for r in probed if r.get("agree")),
        "per_detector": {
            k: {
                "found": len(_hits(k)),
                "credible": sum(1 for r in _hits(k) if r[k]["credible"]),
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--impact", type=int)
    ap.add_argument("--ball", help="x,y of the ball at rest")
    ap.add_argument("--seed", type=Path,
                    help="JSON list of {frame,x,y} to seed from instead")
    ap.add_argument("--fps", type=float, default=0.0)
    ap.add_argument("--max-frames", type=int, default=120,
                    help="how far past the hand-off to probe")
    ap.add_argument("--win-min", type=int, default=12,
                    help="smallest search half-width, px")
    ap.add_argument("--win-pad", type=float, default=2.5,
                    help="search half-width as a multiple of predicted step")
    ap.add_argument("--min-diff", type=float, default=6.0,
                    help="floor on the frame-diff peak")
    ap.add_argument("--min-log-z", type=float, default=8.0,
                    help="floor on the LoG robust z-score")
    ap.add_argument("--min-ncc", type=float, default=0.6,
                    help="floor on the NCC correlation")
    ap.add_argument("--frames", type=Path, help="write annotated PNGs here")
    ap.add_argument("--out", type=Path, help="write the report JSON here")
    a = ap.parse_args(argv)

    if not a.video.exists():
        raise SystemExit(f"no such video: {a.video}")
    fps = a.fps
    if fps <= 0:
        cap = cv2.VideoCapture(str(a.video))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        cap.release()

    pipeline = None
    if a.seed:
        seed = json.loads(a.seed.read_text())
    else:
        if a.impact is None or not a.ball:
            raise SystemExit("give --seed, or --impact and --ball")
        bx, by = (float(v) for v in a.ball.split(","))
        pipeline = seed_from_pipeline(a.video, a.impact, (bx, by), fps)
        seed = pipeline["points"]
        if len(seed) < 3:
            raise SystemExit(
                "the pipeline found fewer than 3 flight points "
                f"({pipeline.get('reason')}) — plot a few in click-to-plot "
                "and pass them with --seed",
            )
    if a.frames:
        a.frames.mkdir(parents=True, exist_ok=True)

    floors = {"diff": a.min_diff, "log": a.min_log_z, "ncc": a.min_ncc}
    rep = probe(a.video, seed, a.frames, a.max_frames, a.win_min, a.win_pad,
                floors)
    rep["floors"] = floors
    rep["video"] = str(a.video)
    rep["fps"] = fps
    rep["seed_points"] = len(seed)
    if pipeline is not None:
        rep["pipeline"] = {"ok": pipeline["ok"], "reason": pipeline["reason"],
                           "n_points": len(pipeline["points"]),
                           "n_candidates": len(pipeline["candidates"])}

    s = rep["summary"]
    print(f"video            {a.video.name}  @{fps:.2f}fps")
    if pipeline is not None:
        print(f"pipeline         ok={pipeline['ok']} "
              f"points={len(pipeline['points'])} "
              f"reason={pipeline['reason']!r}")
    print(f"hand-off frame   {s['handoff_frame']}  "
          f"(seeded from {len(seed)} points)")
    print(f"probed           {s['frames_probed']} frames")
    print(f"extended         {s['frames_extended']} frames "
          f"(to f{s['last_agreed_frame']}) on 2-of-3 agreement")
    print(f"agreement        {s['agreement_frames']}/{s['frames_probed']} frames")
    print(f"window texture   median std {s['window_std']['median']} "
          "(low = sky, high = trees)")
    for k, v in s["per_detector"].items():
        print(f"  {k:<5} found {v['found']:>4}  "
              f"credible {v['credible']:>4} (>={v['floor']})  "
              f"median err {v['median_err_px']}px  "
              f"median score {v['median_score']}")
    if a.out:
        a.out.write_text(json.dumps(rep, indent=2))
        print(f"\nwrote {a.out}")
    if a.frames:
        print(f"annotated frames in {a.frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
