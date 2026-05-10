"""Par-3 stereo ball-tracer SPIKE.

Goal of this script: take a tee-cam clip and a green-cam clip from the same
swing and answer one question — *can we recover a usable ball trajectory
from real iPhone footage using only classical CV?*

This is intentionally simple. No ML model, no training data, no GPU. Just
OpenCV + NumPy + ffmpeg. We're trying to learn the failure modes on real
footage so we can decide whether the par-3-specialized CV pipeline is worth
investing in.

What it does
------------
For each input video, independently:
  1. Walk every frame, find small bright circular blobs that could be balls
  2. Reject obvious non-balls (too big, too non-circular, not moving)
  3. Group surviving detections into the longest trajectory that fits a
     parabola in image coordinates (real ball flight is parabolic in
     image space when seen from a roughly horizontal camera angle)
  4. Render a dashed-white tracer line on top of the original video
  5. Save the annotated MP4 + a debug image showing every candidate

What it doesn't do (yet)
------------------------
- ML-based detection. Pure HSV + motion + circularity heuristics.
- True 3D stereo reconstruction. Each camera tracks independently —
  we eyeball whether the two views' trajectories agree.
- Camera calibration. Skips the homography step for V0.
- Render anything fancy (glow, anti-aliasing, ball-shadow). Just a line.

Usage
-----
    python tools/tracer_spike.py path/to/tee.mp4 path/to/green.mp4 --out out/

Outputs:
    out/tee_traced.mp4          — original tee video with tracer overlay
    out/green_traced.mp4        — original green video with tracer overlay
    out/tee_debug.jpg           — sampled frame with every candidate
                                   detection drawn as a red circle (so
                                   you can see what the detector is finding)
    out/green_debug.jpg

Tunables
--------
The constants near the top of this file are worth tweaking on real footage.
The defaults are reasonable starting points, but real iPhone clips in
real lighting will need adjustment.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# --- Tunables (change these on real footage) --------------------------------

# White-ish HSV mask for the ball
HSV_LOWER = np.array([0, 0, 200])    # low saturation, high value = white
HSV_UPPER = np.array([180, 70, 255])

# Acceptable ball size (in pixels). Tune to your camera distance.
MIN_BALL_AREA = 4
MAX_BALL_AREA = 200
MIN_BALL_RADIUS = 1.0
MAX_BALL_RADIUS = 14.0

# Circularity threshold (1.0 = perfect circle)
MIN_CIRCULARITY = 0.55

# Frame-to-frame motion sensitivity
MOTION_DIFF_THRESHOLD = 22

# Trajectory linking
MAX_FRAME_GAP = 6        # how many frames the ball can disappear before we cut
MAX_PIXEL_JUMP_PER_FRAME = 60  # ball can't teleport
MIN_TRACK_LENGTH = 6     # reject any "trajectory" with fewer than N detections
MAX_PARABOLA_RESIDUAL = 18.0   # px; reject tracks that don't fit a parabola

# Render
TRACER_COLOR = (240, 240, 240)  # near-white; opencv is BGR
TRACER_THICKNESS = 3
DASH_LENGTH = 14
GAP_LENGTH = 10
BALL_HIGHLIGHT_COLOR = (0, 230, 255)  # yellow

# --- End tunables -----------------------------------------------------------


@dataclass
class Detection:
    frame: int
    x: float
    y: float
    radius: float


def find_ball_candidates(frame_bgr: np.ndarray, prev_gray: np.ndarray | None) -> list[Detection]:
    """Return all blobs in this frame that *could* be a ball."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    if prev_gray is not None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        _, motion_mask = cv2.threshold(diff, MOTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        # Dilate motion mask slightly so we don't miss ball that barely moved
        motion_mask = cv2.dilate(motion_mask, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.bitwise_and(white_mask, motion_mask)
    else:
        mask = white_mask

    # Tiny morphological close to fill speckles in the ball
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[Detection] = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (MIN_BALL_AREA <= area <= MAX_BALL_AREA):
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if not (MIN_BALL_RADIUS <= radius <= MAX_BALL_RADIUS):
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter ** 2)
        if circularity < MIN_CIRCULARITY:
            continue
        out.append(Detection(frame=-1, x=float(cx), y=float(cy), radius=float(radius)))
    return out


def collect_detections(video_path: Path, debug_frame_path: Path | None = None) -> tuple[list[Detection], float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detections: list[Detection] = []

    prev_gray: np.ndarray | None = None
    debug_frame_idx = -1
    debug_frame: np.ndarray | None = None
    debug_candidates: list[Detection] = []

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cands = find_ball_candidates(frame, prev_gray)
        for c in cands:
            detections.append(Detection(frame=idx, x=c.x, y=c.y, radius=c.radius))
        if debug_frame is None and len(cands) > 0:
            # Capture the FIRST frame that has any candidates as our debug image,
            # so we can see what the detector is finding right out of the gate.
            debug_frame = frame.copy()
            debug_frame_idx = idx
            debug_candidates = cands
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        idx += 1

    cap.release()

    if debug_frame_path and debug_frame is not None:
        for c in debug_candidates:
            cv2.circle(debug_frame, (int(c.x), int(c.y)), int(c.radius) + 2, (0, 0, 255), 2)
        cv2.putText(
            debug_frame, f"frame {debug_frame_idx}: {len(debug_candidates)} candidates",
            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )
        cv2.imwrite(str(debug_frame_path), debug_frame)

    return detections, fps, idx


def link_trajectories(detections: list[Detection]) -> list[list[Detection]]:
    """Greedy chain: a trajectory is a sequence of detections where each
    successive detection is within MAX_PIXEL_JUMP_PER_FRAME * gap of the last."""
    by_frame: dict[int, list[Detection]] = {}
    for d in detections:
        by_frame.setdefault(d.frame, []).append(d)

    frames = sorted(by_frame)
    if not frames:
        return []

    trajectories: list[list[Detection]] = []

    # Visit every detection as a potential start; pick the longest chain that follows
    used: set[tuple[int, int]] = set()  # (frame_idx, detection_index_in_frame)

    for start_frame in frames:
        for si, start in enumerate(by_frame[start_frame]):
            if (start_frame, si) in used:
                continue
            track = [start]
            track_used = {(start_frame, si)}
            cur = start

            for next_frame in frames:
                if next_frame <= cur.frame:
                    continue
                gap = next_frame - cur.frame
                if gap > MAX_FRAME_GAP:
                    break
                # Pick the closest candidate at next_frame that's plausible
                best = None
                best_dist = float("inf")
                best_idx = -1
                for ci, cand in enumerate(by_frame[next_frame]):
                    if (next_frame, ci) in used or (next_frame, ci) in track_used:
                        continue
                    dist = float(np.hypot(cand.x - cur.x, cand.y - cur.y))
                    if dist > MAX_PIXEL_JUMP_PER_FRAME * gap:
                        continue
                    if dist < best_dist:
                        best = cand
                        best_dist = dist
                        best_idx = ci
                if best is None:
                    continue
                track.append(best)
                track_used.add((next_frame, best_idx))
                cur = best

            if len(track) >= MIN_TRACK_LENGTH:
                trajectories.append(track)
                used.update(track_used)

    return trajectories


def parabola_residual(track: list[Detection]) -> float:
    """Fit y = a*x^2 + b*x + c through the trajectory and return RMS residual.
    Real ball flight is parabolic in image space (when camera sees the flight
    roughly side-on or behind). Bad detections rarely fit a parabola."""
    if len(track) < 3:
        return float("inf")
    xs = np.array([p.x for p in track])
    ys = np.array([p.y for p in track])
    if np.ptp(xs) < 5:
        # No horizontal motion — almost certainly stationary noise
        return float("inf")
    coeffs = np.polyfit(xs, ys, 2)
    pred = np.polyval(coeffs, xs)
    return float(np.sqrt(np.mean((ys - pred) ** 2)))


def pick_best_trajectory(trajectories: list[list[Detection]]) -> list[Detection]:
    """Among all candidate tracks, pick the one with the lowest parabolic
    residual that's also reasonably long. Real ball trajectories beat
    backgrounds because backgrounds don't follow physics."""
    scored: list[tuple[float, list[Detection]]] = []
    for t in trajectories:
        r = parabola_residual(t)
        if r > MAX_PARABOLA_RESIDUAL:
            continue
        # Score: lower residual + longer track wins. Length matters more.
        score = r - 0.5 * len(t)
        scored.append((score, t))
    if not scored:
        return []
    scored.sort(key=lambda s: s[0])
    return scored[0][1]


def draw_dashed_polyline(img: np.ndarray, points: list[tuple[int, int]]) -> None:
    if len(points) < 2:
        return
    # Draw the full polyline first as a faint glow
    cv2.polylines(img, [np.array(points, dtype=np.int32)], False, (255, 255, 255), TRACER_THICKNESS + 4, cv2.LINE_AA)
    # Then draw the dashed bright line on top
    accumulated = 0.0
    drawing = True
    for i in range(1, len(points)):
        p0 = np.array(points[i - 1], dtype=np.float32)
        p1 = np.array(points[i], dtype=np.float32)
        seg_len = float(np.linalg.norm(p1 - p0))
        if seg_len == 0:
            continue
        direction = (p1 - p0) / seg_len
        traveled = 0.0
        cur = p0.copy()
        while traveled < seg_len:
            target_seg = DASH_LENGTH if drawing else GAP_LENGTH
            remaining = target_seg - accumulated
            step = min(remaining, seg_len - traveled)
            nxt = cur + direction * step
            if drawing:
                cv2.line(img, (int(cur[0]), int(cur[1])), (int(nxt[0]), int(nxt[1])),
                         TRACER_COLOR, TRACER_THICKNESS, cv2.LINE_AA)
            cur = nxt
            traveled += step
            accumulated += step
            if accumulated >= target_seg:
                drawing = not drawing
                accumulated = 0.0


def render_with_tracer(video_path: Path, track: list[Detection], output_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    track_by_frame = {d.frame: (int(d.x), int(d.y)) for d in track}
    track_frames = sorted(track_by_frame)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Build the points seen up to and including the current frame
        seen_pts = [track_by_frame[f] for f in track_frames if f <= idx]
        draw_dashed_polyline(frame, seen_pts)
        if idx in track_by_frame:
            cv2.circle(frame, track_by_frame[idx], 7, BALL_HIGHLIGHT_COLOR, 2, cv2.LINE_AA)
        out.write(frame)
        idx += 1
    cap.release()
    out.release()


def process_one(video_path: Path, out_dir: Path, label: str) -> dict:
    print(f"[{label}] reading {video_path.name} …")
    debug_image = out_dir / f"{label}_debug.jpg"
    detections, fps, n_frames = collect_detections(video_path, debug_image)
    print(f"[{label}] {n_frames} frames @ {fps:.1f}fps, {len(detections)} candidate detections")
    print(f"[{label}] debug frame: {debug_image}")

    trajectories = link_trajectories(detections)
    print(f"[{label}] formed {len(trajectories)} candidate trajectory chain(s)")
    track = pick_best_trajectory(trajectories)
    if not track:
        print(f"[{label}] NO TRAJECTORY MET QUALITY GATE. "
              f"Inspect the debug frame and tune HSV / circularity / parabola residual.")
        return {"label": label, "found": False, "n_detections": len(detections)}

    print(f"[{label}] picked trajectory: {len(track)} points, "
          f"frames {track[0].frame}-{track[-1].frame}, "
          f"residual={parabola_residual(track):.2f}px")

    out_path = out_dir / f"{label}_traced.mp4"
    print(f"[{label}] rendering tracer to {out_path}")
    render_with_tracer(video_path, track, out_path)
    return {
        "label": label,
        "found": True,
        "n_points": len(track),
        "frame_range": (track[0].frame, track[-1].frame),
        "residual_px": parabola_residual(track),
        "out_path": str(out_path),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Par-3 ball-tracer spike (classical CV).")
    p.add_argument("tee", type=Path, help="path to tee-cam clip")
    p.add_argument("green", type=Path, help="path to green-cam clip")
    p.add_argument("--out", type=Path, default=Path("out"), help="output directory")
    args = p.parse_args()

    if not args.tee.is_file():
        print(f"tee video not found: {args.tee}", file=sys.stderr)
        return 1
    if not args.green.is_file():
        print(f"green video not found: {args.green}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    tee_result = process_one(args.tee, args.out, "tee")
    green_result = process_one(args.green, args.out, "green")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in (tee_result, green_result):
        if r["found"]:
            f0, f1 = r["frame_range"]
            print(f"  {r['label']:6}  ✓  {r['n_points']} pts  "
                  f"frames {f0}-{f1}  residual {r['residual_px']:.1f}px → {r['out_path']}")
        else:
            print(f"  {r['label']:6}  ✗  no usable trajectory "
                  f"(saw {r['n_detections']} raw candidates)")

    if tee_result["found"] and green_result["found"]:
        # Cheap stereo sanity: do the two views see the ball flying for
        # overlapping wall-clock windows? The clips should be roughly synced
        # so we expect the green track to start ~later than the tee track
        # (ball has to fly there).
        tee_dur = tee_result["frame_range"][1] - tee_result["frame_range"][0]
        green_dur = green_result["frame_range"][1] - green_result["frame_range"][0]
        print()
        print(f"  tee track lasts {tee_dur} frames; green track lasts {green_dur} frames")
        print(f"  if both tracks visually pass eye-test, par-3-specialized CV is viable")

    return 0


if __name__ == "__main__":
    sys.exit(main())
