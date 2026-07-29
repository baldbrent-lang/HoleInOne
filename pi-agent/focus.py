#!/usr/bin/env python3
"""Focus meter for the HQ camera — turn the lens ring until the number peaks.

The IMX477 has NO autofocus: focus and aperture are physical rings on the
C/CS lens. So focusing is a measurement you tune against, and eyeballing a
preview on a laptop in sunlight is how a camera ends up half a turn off.

This prints a live sharpness score (variance of Laplacian — high = crisp
edges, low = mush) on a region of your choosing, at the camera's FULL
resolution rather than the 720p the agent captures at. Turn the ring
slowly; the number rises to a peak and falls off again. Stop at the peak.

It also separates the three things that all look like "out of focus":

  focus       a STATIC object at the golfer's distance scores low here
  motion blur this scores fine on static things but moving things smear
              in the video — that's shutter speed, not the lens
  compression this scores fine and the saved crop is sharp, but the
              uploaded clip is soft — that's bitrate (green uploads at
              720p/1500kbps), not the lens

If the saved 100% crop is sharp, focus is not your problem.

Usage on the Pi::

    sudo systemctl stop golfreelz-agent      # the service holds the camera
    python3 focus.py                         # centre region, live score
    python3 focus.py --roi tee               # use tee_box_roi from config
    python3 focus.py --roi 600,400,700,500   # explicit x,y,w,h
    python3 focus.py --save /tmp/focus.jpg   # write a still + 100% crop
    sudo systemctl start golfreelz-agent     # put it back when done

Reading the output::

    score 1284  (best 1301)  [############        ]  bright 118  zones L 940 C 1284 R 1102

  score    sharpness now. Bigger is sharper. The absolute value is
           meaningless across scenes — only its PEAK as you turn matters.
  best     the highest seen this run, so you can tell you have gone past.
  zones    left / centre / right thirds. A big spread means the sensor is
           not square to the scene (tilt) or the lens is soft in the
           corners — stopping the aperture down usually fixes the latter.
  bright   mean luma. Under ~40 or over ~220 and the score is unreliable;
           fix exposure first.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    print("opencv/numpy missing — run this on the Pi", file=sys.stderr)
    raise SystemExit(2)


def sharpness(gray) -> float:
    """Variance of the Laplacian: the standard focus metric. Edges are
    second-derivative energy, and a defocused image has almost none."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def open_full_res(device, want_w: int, want_h: int):
    """Open the camera and ask for the biggest frame it will give.

    The agent captures 720p (see CAPTURE_MODES) because that is what the
    encoder sustains — but a downscaled frame is soft BY CONSTRUCTION and
    would mask exactly what we are trying to measure. Focus is judged at
    the sensor's resolution."""
    cap = None
    if device == "auto":
        for idx in range(4):
            cand = cv2.VideoCapture(idx)
            if cand.isOpened():
                cap = cand
                break
            cand.release()
    else:
        try:
            cap = cv2.VideoCapture(int(device))
        except (TypeError, ValueError):
            cap = cv2.VideoCapture(str(device))
    if cap is None or not cap.isOpened():
        print(
            "could not open the camera.\n"
            "  Is the agent still running?  sudo systemctl stop golfreelz-agent",
            file=sys.stderr,
        )
        raise SystemExit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, got_w, got_h


def resolve_roi(spec: str | None, cfg_path: Path, w: int, h: int):
    """(x, y, rw, rh) to measure. Defaults to the middle third — big
    enough to catch real detail, small enough that sky and foreground
    do not dilute the score."""
    if spec in (None, "", "centre", "center"):
        return int(w * 0.33), int(h * 0.33), int(w * 0.34), int(h * 0.34)
    if spec == "tee":
        try:
            import yaml  # type: ignore

            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            roi = cfg.get("tee_box_roi") or {}
            if all(k in roi for k in ("x", "y", "w", "h")):
                return int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
            print(f"no tee_box_roi in {cfg_path} — using the centre", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"could not read {cfg_path} ({exc}) — using the centre", file=sys.stderr)
        return resolve_roi(None, cfg_path, w, h)
    try:
        x, y, rw, rh = (int(v) for v in spec.split(","))
        return x, y, rw, rh
    except Exception:  # noqa: BLE001
        print(f"bad --roi {spec!r}; expected x,y,w,h or 'tee'", file=sys.stderr)
        raise SystemExit(2)


def clamp_roi(x, y, rw, rh, w, h):
    x = max(0, min(w - 16, x))
    y = max(0, min(h - 16, y))
    rw = max(16, min(w - x, rw))
    rh = max(16, min(h - y, rh))
    return x, y, rw, rh


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Live focus meter for the HQ camera")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--roi", default=None, help="'tee', 'centre', or x,y,w,h")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--width", type=int, default=4056, help="requested capture width")
    ap.add_argument("--height", type=int, default=3040, help="requested capture height")
    ap.add_argument("--save", default=None, help="write a still + 100%% crop here")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds")
    args = ap.parse_args(argv[1:])

    cap, w, h = open_full_res(args.device, args.width, args.height)
    x, y, rw, rh = clamp_roi(
        *resolve_roi(args.roi, Path(args.config), w, h), w, h,
    )
    print(f"camera {w}x{h}   measuring region ({x},{y}) {rw}x{rh}")
    print("turn the focus ring slowly; stop where 'score' peaks.  Ctrl-C to quit.\n")

    best = 0.0
    t0 = time.time()
    last_frame = None
    dead = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                # The deadline is checked HERE too. Skipping it on a failed
                # read means a camera that stops delivering spins forever
                # instead of exiting - which is exactly when you want it to
                # say something is wrong.
                dead += 1
                if dead >= 100:
                    print("\ncamera stopped delivering frames - giving up")
                    break
                if args.seconds and (time.time() - t0) >= args.seconds:
                    break
                time.sleep(0.05)
                continue
            dead = 0
            last_frame = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = gray[y:y + rh, x:x + rw]
            score = sharpness(roi)
            best = max(best, score)
            bright = float(roi.mean())
            # Thirds of the measured region — a big L/C/R spread means the
            # camera is not square to the scene, or the lens is soft off-axis.
            t = rw // 3
            zl = sharpness(roi[:, :t]) if t >= 8 else 0.0
            zc = sharpness(roi[:, t:2 * t]) if t >= 8 else 0.0
            zr = sharpness(roi[:, 2 * t:]) if t >= 8 else 0.0
            bar = "#" * int(20 * (score / best if best else 0))
            warn = ""
            if bright < 40:
                warn = "  (too dark to judge — fix exposure first)"
            elif bright > 220:
                warn = "  (blown out — fix exposure first)"
            sys.stdout.write(
                f"\rscore {score:7.0f}  (best {best:7.0f})  [{bar:<20}]  "
                f"bright {bright:3.0f}  zones L {zl:5.0f} C {zc:5.0f} "
                f"R {zr:5.0f}{warn}   "
            )
            sys.stdout.flush()
            if args.seconds and (time.time() - t0) >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print()
        if args.save and last_frame is not None:
            out = Path(args.save)
            cv2.imwrite(str(out), last_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            # A 100% crop of the measured region: this is the picture that
            # settles "focus vs compression". If THIS is sharp and the
            # uploaded clip is not, the lens was never the problem.
            crop = out.with_name(out.stem + "_crop100" + out.suffix)
            cv2.imwrite(
                str(crop), last_frame[y:y + rh, x:x + rw],
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
            print(f"wrote {out}  and  {crop}  (view the crop at 100%)")
        cap.release()
    print(f"best score this run: {best:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
