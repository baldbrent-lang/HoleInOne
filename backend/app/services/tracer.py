"""Par-3 ball tracer (classical CV).

Mirrors the algorithm prototyped in tools/tracer_spike.py, repackaged for
the upload pipeline. Given a video clip, detects the ball, fits a
parabolic trajectory, and renders an annotated MP4 with a dashed tracer
overlay drawn progressively as the ball flies.

If OpenCV / NumPy aren't installed at runtime, `render_tracer` is a
no-op that returns ok=False. The upload flow treats that as a soft
failure — the clip is still saved with its original source_url, just
without a tracer_url, and the broadcast channel falls back to playing
the raw clip.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("golfreelz.tracer")

try:  # Optional dependency — keep the backend importable without it.
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    HAS_CV = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    np = None  # type: ignore
    HAS_CV = False


# Kept in sync with tools/tracer_spike.py — tune both files together if
# you change a threshold during validation.
HSV_LOWER = (0, 0, 210)
# Middle ground after iteration: 70 was too permissive (caught sky/clouds),
# 38 was too strict (caught nothing in real footage). 55 admits a real
# golf ball (which can pick up subtle saturation from grass reflections,
# motion blur, or shadow) while still rejecting most overcast sky.
HSV_UPPER = (180, 55, 255)

# Real-world tuning notes for GoPro Hero 13 at 1080p60, mounted 4-6ft
# behind a right-handed golfer:
#   - At impact the ball is ~6ft from camera → ~30px diameter (~700 area)
#     so MAX_BALL_AREA / MAX_BALL_RADIUS need headroom or the close-up
#     impact frames get rejected.
#   - At 50yd downrange the ball is ~3-4px diameter, near MIN floor.
#   - Motion blur at 1/240 or slower shutter elongates the ball, so the
#     circularity floor needs to be forgiving.
MIN_BALL_AREA = 3
MAX_BALL_AREA = 900
MIN_BALL_RADIUS = 0.8
MAX_BALL_RADIUS = 22.0

MIN_CIRCULARITY = 0.45
# Tuned middle ground after iteration: 18 caught too much noise, 28
# rejected real ball motion in lower-light footage. 22 is a reasonable
# compromise for most GoPro daylight clips.
MOTION_DIFF_THRESHOLD = 22

MAX_FRAME_GAP = 6
MAX_PIXEL_JUMP_PER_FRAME = 90
MIN_TRACK_LENGTH = 5
MAX_PARABOLA_RESIDUAL = 22.0
# A "trajectory" stuck in one corner of the sky (random noise) often
# has tiny total displacement. A real ball flight covers a big chunk
# of the frame. Reject anything whose total motion is smaller than
# this (in pixels, measured as max of x-range or y-range).
MIN_TRAJECTORY_SPAN_PX = 60.0

TRACER_COLOR = (240, 240, 240)
TRACER_THICKNESS = 3
DASH_LENGTH = 14
GAP_LENGTH = 10
BALL_HIGHLIGHT_COLOR = (0, 230, 255)


@dataclass
class _Det:
    frame: int
    x: float
    y: float
    radius: float


def have_tracer() -> bool:
    """True if OpenCV is installed and the tracer can actually run."""
    return HAS_CV


def render_tracer(input_path: Path, output_path: Path, debug_path: Path | None = None) -> dict:
    """Detect the ball + render a traced MP4 to output_path.

    If `debug_path` is provided, also save a JPG of the first frame that
    has any ball candidates (with every candidate circled in red),
    regardless of whether the trajectory passed the parabolic-fit gate.
    On total detection failure, saves the first frame of the video with
    a "0 candidates" overlay so the operator can see what the detector
    is staring at.

    Returns a dict shaped like::
        {ok: bool, residual_px: float|None, n_points: int,
         n_candidates: int, frame_range: [int,int]|None, error: str|None}

    Caller is responsible for handing the output to compress_for_email()
    if browser-playable H.264 is needed (cv2 writes mp4v by default).
    """
    if not HAS_CV:
        return {"ok": False, "error": "opencv not installed", "n_points": 0, "n_candidates": 0}
    try:
        return _render(input_path, output_path, debug_path)
    except Exception as exc:  # pragma: no cover
        log.warning("tracer crashed on %s: %s", input_path, exc)
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
        return {"ok": False, "error": str(exc), "n_points": 0, "n_candidates": 0}


# --- internals --------------------------------------------------------------

def _render(input_path: Path, output_path: Path, debug_path: Path | None = None) -> dict:
    hsv_lower = np.array(HSV_LOWER)
    hsv_upper = np.array(HSV_UPPER)

    detections: list[_Det] = []
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return {"ok": False, "error": "could not open video", "n_points": 0, "n_candidates": 0}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Snapshot frames for the debug JPG. We pick the *busiest* frame
    # (the one with the most candidate detections) rather than the first
    # frame with any candidates — that's almost always pre-swing sky
    # noise. The busiest frame is usually near impact, where a real ball
    # plus surrounding motion blur generates lots of candidates.
    first_frame_snapshot = None
    busiest_frame = None
    busiest_frame_idx = -1
    busiest_candidates: list[tuple[float, float, float]] = []
    busiest_count = 0

    prev_gray = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if first_frame_snapshot is None:
            first_frame_snapshot = frame.copy()
        cands = _candidates(frame, prev_gray, hsv_lower, hsv_upper)
        if len(cands) > busiest_count:
            busiest_count = len(cands)
            busiest_frame = frame.copy()
            busiest_frame_idx = idx
            busiest_candidates = list(cands)
        for cx, cy, r in cands:
            detections.append(_Det(idx, cx, cy, r))
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        idx += 1
    cap.release()

    if debug_path is not None:
        filmstrip = _sample_filmstrip(input_path, idx)
        _write_debug(
            debug_path, busiest_frame, busiest_frame_idx, busiest_candidates,
            first_frame_snapshot, detections, filmstrip, idx, fps,
        )

    track = _pick_best(_link(detections))
    if not track:
        return {
            "ok": False,
            "error": "no usable trajectory",
            "n_candidates": len(detections),
            "n_points": 0,
        }

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    cap2 = cv2.VideoCapture(str(input_path))
    track_by_frame = {d.frame: (int(d.x), int(d.y)) for d in track}
    track_frames = sorted(track_by_frame)
    i = 0
    while True:
        ok, frame = cap2.read()
        if not ok:
            break
        seen = [track_by_frame[f] for f in track_frames if f <= i]
        _draw_dashed(frame, seen)
        if i in track_by_frame:
            cv2.circle(frame, track_by_frame[i], 7, BALL_HIGHLIGHT_COLOR, 2, cv2.LINE_AA)
        writer.write(frame)
        i += 1
    cap2.release()
    writer.release()

    return {
        "ok": True,
        "residual_px": _residual(track),
        "n_points": len(track),
        "n_candidates": len(detections),
        "frame_range": [track[0].frame, track[-1].frame],
        "fps": float(fps),
        "error": None,
    }


def _sample_filmstrip(input_path: Path, n_frames: int) -> list:
    """Pull 5 evenly-spaced frames from the clip, downscale them, and
    return as a list of small BGR images. Used to embed a visual
    timeline of the clip in the debug JPG so the operator can see what
    actually happens in the video regardless of detection success."""
    if n_frames <= 0:
        return []
    out = []
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return []
    try:
        pcts = (0.05, 0.27, 0.5, 0.73, 0.95)
        for pct in pcts:
            target_idx = max(0, min(n_frames - 1, int(n_frames * pct)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            tile_h = 200
            tile_w = max(1, int(w * tile_h / max(h, 1)))
            small = cv2.resize(frame, (tile_w, tile_h))
            label = f"{int(pct * 100)}%  f{target_idx}"
            cv2.rectangle(small, (0, 0), (tile_w, 22), (0, 0, 0), -1)
            cv2.putText(small, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            out.append(small)
    finally:
        cap.release()
    return out


def _write_debug(path, busiest_frame, busiest_idx, busiest_cands, first_frame,
                 all_detections, filmstrip, total_frames, fps):
    """Diagnostic JPG composed of two parts stacked vertically.

    Top half = the busiest frame (most candidates seen) with:
      - tiny yellow dots for every detection across the entire clip
        (so a real ball flight shows as a curved sweep; sky noise shows
        as a cluster up top)
      - red circles around the busiest frame's specific candidates

    Bottom half = 5-frame filmstrip sampled evenly across the clip, so
    the operator can see what the video actually contains independent
    of whether anything was detected. Crucial when zero candidates
    came through the gates — we need to know if it's a filter problem
    or a "the clip has no swing in it" problem."""
    base = busiest_frame.copy() if busiest_frame is not None else (
        first_frame.copy() if first_frame is not None else None
    )
    if base is None and not filmstrip:
        return

    if base is not None:
        for d in all_detections:
            cv2.circle(base, (int(d.x), int(d.y)), 2, (0, 255, 255), -1, cv2.LINE_AA)
        if busiest_cands:
            for cx, cy, r in busiest_cands:
                cv2.circle(base, (int(cx), int(cy)), max(int(r) + 2, 4), (0, 0, 255), 2)
            secs = total_frames / fps if fps > 0 else 0.0
            msg = (
                f"clip: {total_frames}f / {secs:.1f}s  |  "
                f"busiest frame {busiest_idx}: {len(busiest_cands)} cands  |  "
                f"total cands across clip: {len(all_detections)}"
            )
        else:
            msg = "0 candidates across entire clip - filters too strict, or no white moving objects in view"
        cv2.rectangle(base, (0, 0), (base.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(base, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 255), 2, cv2.LINE_AA)

    strip = None
    if filmstrip:
        strip = np.hstack(filmstrip)
        # Pad / scale strip to match base width if base exists
        if base is not None:
            base_w = base.shape[1]
            sh, sw = strip.shape[:2]
            if sw < base_w:
                pad = np.zeros((sh, base_w - sw, 3), dtype=np.uint8)
                strip = np.hstack([strip, pad])
            elif sw > base_w:
                new_h = int(sh * base_w / sw)
                strip = cv2.resize(strip, (base_w, new_h))
        # Add a header bar over the filmstrip
        cv2.rectangle(strip, (0, 0), (strip.shape[1], 24), (40, 40, 40), -1)
        cv2.putText(strip, "Clip timeline (5 frames)", (8, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if base is not None and strip is not None:
        out = np.vstack([base, strip])
    elif base is not None:
        out = base
    else:
        out = strip
    cv2.imwrite(str(path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 80])


def _candidates(frame_bgr, prev_gray, hsv_lower, hsv_upper):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    if prev_gray is not None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        _, motion_mask = cv2.threshold(diff, MOTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.dilate(motion_mask, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.bitwise_and(white_mask, motion_mask)
    else:
        mask = white_mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
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
        circ = 4.0 * np.pi * area / (perimeter ** 2)
        if circ < MIN_CIRCULARITY:
            continue
        out.append((float(cx), float(cy), float(radius)))
    return out


def _link(detections):
    by_frame: dict[int, list[_Det]] = {}
    for d in detections:
        by_frame.setdefault(d.frame, []).append(d)
    frames = sorted(by_frame)
    if not frames:
        return []
    trajectories: list[list[_Det]] = []
    used: set[tuple[int, int]] = set()
    for sf in frames:
        for si, start in enumerate(by_frame[sf]):
            if (sf, si) in used:
                continue
            track = [start]
            track_used = {(sf, si)}
            cur = start
            for nf in frames:
                if nf <= cur.frame:
                    continue
                gap = nf - cur.frame
                if gap > MAX_FRAME_GAP:
                    break
                best = None
                best_dist = float("inf")
                best_idx = -1
                for ci, cand in enumerate(by_frame[nf]):
                    if (nf, ci) in used or (nf, ci) in track_used:
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
                track_used.add((nf, best_idx))
                cur = best
            if len(track) >= MIN_TRACK_LENGTH:
                trajectories.append(track)
                used.update(track_used)
    return trajectories


def _residual(track):
    if len(track) < 3:
        return float("inf")
    xs = np.array([p.x for p in track])
    ys = np.array([p.y for p in track])
    if np.ptp(xs) < 5:
        return float("inf")
    coeffs = np.polyfit(xs, ys, 2)
    pred = np.polyval(coeffs, xs)
    return float(np.sqrt(np.mean((ys - pred) ** 2)))


def _pick_best(trajectories):
    scored = []
    for t in trajectories:
        r = _residual(t)
        if r > MAX_PARABOLA_RESIDUAL:
            continue
        xs = [p.x for p in t]
        ys = [p.y for p in t]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        if span < MIN_TRAJECTORY_SPAN_PX:
            # Stuck-in-corner noise. Real ball flight sweeps across the frame.
            continue
        scored.append((r - 0.5 * len(t), t))
    if not scored:
        return []
    scored.sort(key=lambda s: s[0])
    return scored[0][1]


def _draw_dashed(img, points):
    if len(points) < 2:
        return
    cv2.polylines(img, [np.array(points, dtype=np.int32)], False, (255, 255, 255), TRACER_THICKNESS + 4, cv2.LINE_AA)
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
