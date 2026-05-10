# Par-3 Tracer Spike

A standalone script for answering one question:

> Can classical CV recover a usable golf-ball trajectory from real iPhone
> footage of a par-3 shot — without ML, without a paid SDK?

If the answer is yes, GolfReelz can ship its own in-house shot tracer for
par-3-only footage. If the answer is no, we know early and can budget for an
ML model or a third-party SDK before building the rest of the pipeline.

## What it does

For each input video:

1. Walks every frame and looks for ball candidates using:
   - HSV white-mask (golf balls are bright white)
   - Frame-difference motion mask (the ball moves frame-to-frame)
   - Contour area + radius + circularity gates (filters lens flares, line
     paint, cup edges, etc.)
2. Greedily links detections into trajectories, rejecting tracks where the
   per-frame pixel jump is too large or where the gap is too long.
3. Fits a parabola `y = ax² + bx + c` to each track and discards anything
   whose RMS residual is too high — real ball flight is parabolic in image
   space.
4. Renders the surviving trajectory as a dashed-white "tracer" line with a
   green glow, drawn progressively as the ball flies.

## Setup

```bash
cd tools
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

OpenCV ships its own ffmpeg wrapper, so you do not need a system install of
ffmpeg for this script (though you'll want it for compression elsewhere in
GolfReelz).

## Capturing test clips on iPhone

Two clips of the **same swing**, recorded simultaneously:

- **Tee-cam**: phone on a small tripod at the tee, framed so the ball-flight
  arc fits in the upper third of the frame for the first ~1 second of
  flight. Lock exposure on a brightly-lit patch of grass; if the camera
  auto-exposes to the sky the ball will blend into the clouds.
- **Green-cam**: phone on a tripod just left or right of the green, framing
  the cup in the lower third and the incoming ball-flight in the upper two
  thirds. Lock exposure on the green.

Settings on both phones:

- 1080p, **60 fps** if available (more samples = cleaner parabola fit)
- Tap-and-hold to lock AE/AF
- HEVC off if you have the option (`Settings ▸ Camera ▸ Formats ▸ Most
  Compatible`) — H.264 reads cleaner in OpenCV

Save them as `tee.mp4` and `green.mp4` somewhere on your machine.

## Running it

```bash
python tracer_spike.py /path/to/tee.mp4 /path/to/green.mp4
```

This writes four files next to the inputs:

- `tee_candidates.jpg` — first frame with every detected candidate circled
- `tee_traced.mp4` — original tee-cam clip with the picked trajectory
  overlaid as a dashed tracer
- `green_candidates.jpg` — same idea, for the green cam
- `green_traced.mp4` — green-cam clip with overlay

Console output prints, per camera:

```
tee.mp4
  frames=312 fps=60.0
  candidates=84
  trajectory length=27 frames, residual=4.1
green.mp4
  frames=308 fps=60.0
  candidates=61
  trajectory length=19 frames, residual=6.8
```

## Interpreting results

| residual | meaning |
| --- | --- |
| < 5 px | clean parabola, ball-flight clearly captured. Ship it. |
| 5–10 px | usable but jittery — try increasing `MIN_TRACK_LENGTH` or tightening `MOTION_DIFF_THRESHOLD`. |
| 10–18 px | borderline. Check `*_candidates.jpg` — likely a false-positive track latched onto. |
| > 18 px | rejected. The script will print "no usable trajectory". |

If `candidates` is tiny (< 20), the HSV/motion gates are too strict — open
`tracer_spike.py` and lower `MOTION_DIFF_THRESHOLD` or expand `HSV_LOWER` /
`HSV_UPPER`. If it's huge (> 500), the inverse: tighten them or increase
`MIN_BALL_AREA`. The debug JPG tells you which way to go.

## What this does *not* prove

Even a perfect trajectory in 2D image space is not a 3D ball position. The
follow-up spike — once we know detection works — is to fuse the tee-cam and
green-cam tracks via a known baseline (camera positions on the course) into
real-world XYZ. That's a separate script and a separate question.

This spike is just: **does the ball ever reliably show up in the pixels?**
