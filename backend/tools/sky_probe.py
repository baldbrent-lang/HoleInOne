#!/usr/bin/env python3
"""Command-line front end for the sky probe.

The measurement itself lives in app/services/sky_probe.py so the admin
endpoint and this share one implementation -- two copies of a detector
would drift, and then the number on the screen and the number in the
terminal would disagree about the same swing.

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
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.sky_probe import probe, seed_from_pipeline  # noqa: E402


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
    ap.add_argument("--target",
                    help="x,y of the target (the green) in tee pixels — "
                         "pins the far end of the flight, which turns the "
                         "prediction from extrapolation into interpolation")
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
    _target = None
    if a.target:
        _target = [float(v) for v in a.target.split(",")]
    try:
        rep = probe(a.video, seed, a.frames, a.max_frames, a.win_min,
                    a.win_pad, floors, _target)
    except ValueError as exc:
        # A seed too short or too still to extrapolate from -- a message,
        # not a traceback.
        print(f"cannot probe: {exc}")
        return 2
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
    if s.get("target"):
        print(f"target           {s['target']} "
              f"-> landing f{s.get('landing_frame')}  (flight pinned)")
    print(f"hand-off frame   {s['handoff_frame']}  "
          f"(seeded from {len(seed)} points)")
    print(f"probed           {s['frames_probed']} frames")
    print(f"extended         {s['frames_extended']} frames "
          f"(to f{s['last_agreed_frame']}) on 2-of-3 agreement")
    print(f"agreement        {s['agreement_frames']}/{s['frames_probed']} frames")
    if s.get("stopped_because"):
        print(f"stopped          {s['stopped_because']}")
    print(f"window texture   median std {s['window_std']['median']} "
          "(low = sky, high = trees)")
    for k, v in s["per_detector"].items():
        print(f"  {k:<5} found {v['found']:>4}  "
              f"credible {v['credible']:>4} (>={v['floor']})  "
              f"usable {v['usable']:>4}  "
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
