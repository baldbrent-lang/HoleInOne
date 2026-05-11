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
# Kept in sync with tools/tracer_spike.py — tune both files together if
# you change a threshold during validation.
#
# v2: dropped the HSV "is it white?" color filter entirely. The old
# approach worked when the ball was on grass (white-on-green) but failed
# completely when the ball climbed into bright overcast sky, where it
# appears as a DARK silhouette against bright clouds, not as a white
# pixel. We now use MOG2 background subtraction to detect any pixel
# that differs from the static scene — lighter OR darker — then filter
# by size, radius, and circularity.

# MOG2 sensitivity. Higher = stricter, fewer foreground pixels.
BG_VAR_THRESHOLD = 16

# Skip the first N frames of candidate extraction so the background
# model has time to converge. The frames are still fed to the
# subtractor (to build the model), just not searched for candidates.
WARMUP_FRAMES = 12

# Real-world tuning notes for GoPro Hero 13 at 1080p60, mounted 4-6ft
# behind a right-handed golfer:
#   - Ball just past impact (frame 2 in test footage): ~4-6px wide,
#     ~12-30px area
#   - Ball mid-flight against sky: ~3-4px wide, ~7-15px area
#   - Body / club shaft: thousands of pixels — easily filtered by
#     MAX_BALL_AREA
MIN_BALL_AREA = 3
MAX_BALL_AREA = 400
MIN_BALL_RADIUS = 0.5
MAX_BALL_RADIUS = 18.0

# Motion blur elongates the ball; allow a forgiving floor.
# Bumped 0.40 → 0.55 after first MOG2 run showed body-edge fragments
# (slightly oblong) sneaking through.
MIN_CIRCULARITY = 0.55

# Any single foreground blob larger than this is treated as "the body
# or club arc" — the bounding box (plus a buffer) is excluded from
# candidate extraction so we don't pick up hundreds of silhouette
# fragments. The body is reliably the biggest motion blob in any
# par-3 shot from a tee-cam mount.
BODY_BLOB_MIN_AREA = 1500
BODY_BBOX_BUFFER_PX = 30

# Performance ceilings. Without these the tracer runs for minutes on
# long clips and the browser request times out before the server
# finishes. These caps trade some coverage for keeping the round-trip
# under ~2 minutes on Replit's shared CPU.
MAX_FRAMES_PROCESS = 900     # ~15s @ 60fps, ~30s @ 30fps
MAX_TOTAL_CANDIDATES = 60000  # Early abort if drowning in noise
# Downsample to at most this wide for detection. Track coordinates
# are scaled back to native resolution before rendering so the
# overlay still lines up.
DETECT_MAX_WIDTH = 1280

MAX_FRAME_GAP = 6
MAX_PIXEL_JUMP_PER_FRAME = 90
MIN_TRACK_LENGTH = 5
MAX_PARABOLA_RESIDUAL = 22.0
# A "trajectory" stuck in one corner of the sky (random noise) often
# has tiny total displacement. A real ball flight covers a big chunk
# of the frame. Reject anything whose total motion is smaller than
# this (in pixels, measured as max of x-range or y-range).
MIN_TRAJECTORY_SPAN_PX = 60.0

# --- v3 noise reducers ------------------------------------------------------
# Two stages added to address the "noise blanketing the whole frame" failure
# mode we saw on iPhone tee-cam footage (grass micro-motion + sky compression
# shimmer + handheld shake produced 600+ candidates per frame).

# Stage A — global motion compensation. Estimate a per-frame affine transform
# from the current frame to a reference frame using ORB + RANSAC, warp the
# current frame into reference coords, then feed the WARPED frame to MOG2.
# This collapses camera-shake noise so MOG2 only flags genuine scene motion.
# Detections are inverse-warped back to current-frame coords for rendering.
USE_MOTION_COMPENSATION = True
MC_ORB_FEATURES = 500
MC_MIN_MATCHES = 30
MC_RANSAC_REPROJ_PX = 3.0

# Stage B — upward-streak prefilter. The ball, immediately after impact,
# produces a chain of detections each clearly above the previous (smaller
# y in image coords). Random noise blobs occasionally form a 2-frame
# upward pair by chance but rarely 3+ in a row at a plausible pixel speed.
# We keep only candidates that participate in such a chain — the ball's
# initial-flight phase survives, the 60K-candidate noise floor doesn't.
# Trade-off: this discards the descent portion of the trajectory; we
# trace impact → apex only. That's fine for a visual tracer demo and we
# can soften this later to "anchor linker on upward seeds, extend both
# directions" if the half-arc looks short.
USE_UPWARD_STREAK_FILTER = True
MIN_UPWARD_CHAIN_LEN = 3
MIN_UPWARD_DY_PER_FRAME = 1.5
UPWARD_SEARCH_FRAMES = 2

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
    detections: list[_Det] = []
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return {"ok": False, "error": "could not open video", "n_points": 0, "n_candidates": 0}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Downsample for detection so MOG2 + contour finding don't drag.
    # Candidates are scaled back to native coords before being stored,
    # so the rendered overlay stays aligned with the source video.
    det_scale = 1.0
    if width > DETECT_MAX_WIDTH:
        det_scale = DETECT_MAX_WIDTH / float(width)
    det_w = max(1, int(round(width * det_scale)))
    det_h = max(1, int(round(height * det_scale)))
    log.info("tracer: %s %dx%d @ %.1ffps  detect@%dx%d", input_path.name, width, height, fps, det_w, det_h)

    bg = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=BG_VAR_THRESHOLD, detectShadows=False,
    )

    # Snapshot frames for the debug JPG. Always store the NATIVE-res
    # frame so the debug image is readable.
    first_frame_snapshot = None
    busiest_frame = None
    busiest_frame_idx = -1
    busiest_candidates: list[tuple[float, float, float]] = []
    busiest_count = 0

    # Motion compensation state. Reference frame is the first frame we
    # see (at detection resolution, grayscale). Every subsequent frame is
    # warped INTO the reference's coordinate space before MOG2 sees it.
    ref_gray = None
    mc_failed_frames = 0

    aborted_early = False
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= MAX_FRAMES_PROCESS:
            aborted_early = True
            log.info("tracer: hit MAX_FRAMES_PROCESS=%d cap; stopping detection", MAX_FRAMES_PROCESS)
            break
        if first_frame_snapshot is None:
            first_frame_snapshot = frame.copy()
        det_frame = cv2.resize(frame, (det_w, det_h)) if det_scale != 1.0 else frame

        # Estimate camera-motion transform mapping THIS frame's coords →
        # reference frame coords. Used both to stabilize the input to MOG2
        # (so panning/shake doesn't fire foreground) and to unwarp detected
        # candidates back to this frame's coords for overlay rendering.
        M_to_ref = None
        if USE_MOTION_COMPENSATION:
            cur_gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)
            if ref_gray is None:
                ref_gray = cur_gray
                M_to_ref = np.eye(2, 3, dtype=np.float32)
                bg_input = det_frame
            else:
                M_to_ref = _estimate_cur_to_ref(ref_gray, cur_gray)
                if M_to_ref is None:
                    mc_failed_frames += 1
                    bg_input = det_frame  # fall back; MOG2 may flicker briefly
                else:
                    bg_input = cv2.warpAffine(
                        det_frame, M_to_ref, (det_w, det_h),
                        borderMode=cv2.BORDER_REPLICATE,
                    )
        else:
            bg_input = det_frame

        fg_mask = bg.apply(bg_input)
        if idx < WARMUP_FRAMES:
            idx += 1
            continue
        cands_det = _candidates_from_mask(fg_mask)
        # Inverse-warp candidates back to current-frame detection coords
        # (they came out of MOG2 in reference-frame coords).
        if cands_det and USE_MOTION_COMPENSATION and M_to_ref is not None:
            M_inv = cv2.invertAffineTransform(M_to_ref)
            cands_det = [
                (
                    float(M_inv[0, 0] * cx + M_inv[0, 1] * cy + M_inv[0, 2]),
                    float(M_inv[1, 0] * cx + M_inv[1, 1] * cy + M_inv[1, 2]),
                    r,
                )
                for (cx, cy, r) in cands_det
            ]
        if cands_det:
            # Scale candidates from detection-res back to native coords.
            inv = 1.0 / det_scale if det_scale != 0 else 1.0
            cands = [(cx * inv, cy * inv, r * inv) for (cx, cy, r) in cands_det]
        else:
            cands = []
        if len(cands) > busiest_count:
            busiest_count = len(cands)
            busiest_frame = frame.copy()
            busiest_frame_idx = idx
            busiest_candidates = list(cands)
        for cx, cy, r in cands:
            detections.append(_Det(idx, cx, cy, r))
        if len(detections) > MAX_TOTAL_CANDIDATES:
            aborted_early = True
            log.info("tracer: hit MAX_TOTAL_CANDIDATES=%d; aborting (noise overwhelming)", MAX_TOTAL_CANDIDATES)
            break
        if idx and idx % 120 == 0:
            log.info("tracer: %d frames scanned, %d candidates so far", idx, len(detections))
        idx += 1
    cap.release()
    if USE_MOTION_COMPENSATION:
        log.info("tracer: motion-comp registration failed on %d/%d frames",
                 mc_failed_frames, max(idx, 1))

    # Stage B: prune candidates that aren't part of any short upward streak.
    # The ball's initial flight phase generates a chain of detections moving
    # consistently upward; random noise rarely does. Performed BEFORE the
    # noise-ceiling abort check so an over-the-ceiling clip can still
    # produce a usable result if the upward filter rescues it.
    if USE_UPWARD_STREAK_FILTER and detections:
        seeds = _find_upward_seeds(detections)
        log.info("tracer: upward-streak filter kept %d/%d candidates",
                 len(seeds), len(detections))
    else:
        seeds = detections

    if debug_path is not None:
        filmstrip = _sample_filmstrip(input_path, idx)
        _write_debug(
            debug_path, busiest_frame, busiest_frame_idx, busiest_candidates,
            first_frame_snapshot, detections, seeds, filmstrip, idx, fps,
        )

    if aborted_early and len(detections) >= MAX_TOTAL_CANDIDATES and len(seeds) < MIN_TRACK_LENGTH:
        # Detection blew past our noise ceiling AND the upward filter
        # didn't rescue enough seeds to form a trajectory. Don't waste
        # time on linking — the result won't be useful and the request
        # would time out. The debug JPG still shows what we saw.
        return {
            "ok": False,
            "error": f"noise overwhelming ({len(detections)}+ candidates) — check the debug image for the source (rain on lens, camera shake, etc.)",
            "n_candidates": len(detections),
            "n_points": 0,
        }

    log.info("tracer: linking %d detections (from %d raw)...", len(seeds), len(detections))
    track = _pick_best(_link(seeds))
    if not track:
        return {
            "ok": False,
            "error": "no usable trajectory",
            "n_candidates": len(detections),
            "n_points": 0,
        }
    log.info("tracer: picked track of %d points, residual %.2fpx", len(track), _residual(track))

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
                 all_detections, seed_detections, filmstrip, total_frames, fps):
    """Diagnostic JPG composed of two parts stacked vertically.

    Top half = the busiest frame (most candidates seen) with:
      - tiny yellow dots for every raw detection across the clip (so a
        real ball flight shows as a curved sweep and sky noise shows as
        a cluster up top)
      - red circles around the busiest frame's specific candidates
      - bright green dots for the candidates that survived the
        upward-streak prefilter — these should trace the actual ball arc

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
        # Surviving upward-streak seeds: bigger, brighter, opaque green.
        for d in seed_detections:
            cv2.circle(base, (int(d.x), int(d.y)), 4, (0, 255, 0), -1, cv2.LINE_AA)
        if busiest_cands:
            for cx, cy, r in busiest_cands:
                cv2.circle(base, (int(cx), int(cy)), max(int(r) + 2, 4), (0, 0, 255), 2)
            secs = total_frames / fps if fps > 0 else 0.0
            msg = (
                f"clip: {total_frames}f / {secs:.1f}s  |  "
                f"busiest frame {busiest_idx}: {len(busiest_cands)} cands  |  "
                f"raw: {len(all_detections)}  |  upward-streak: {len(seed_detections)}"
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


def _candidates_from_mask(fg_mask):
    """Extract small, roundish candidate ball positions from a binary
    foreground mask. Color-agnostic — works for the ball whether it's
    white-on-grass or dark-on-sky.

    Critical step: find the largest motion blob (the golfer's body /
    club arc) and exclude its bounding box from candidate extraction.
    Without this the body silhouette generates hundreds of small
    ball-shaped fragments per frame that completely drown out the real
    ball detection.
    """
    # Bigger close kernel (5x5) to merge body silhouette fragments
    # into one connected blob so we can find it as the largest contour.
    mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Largest contour = body / club arc. Bounding-box-exclude it.
    largest = max(contours, key=cv2.contourArea)
    body_bbox = None
    if cv2.contourArea(largest) >= BODY_BLOB_MIN_AREA:
        x, y, w, h = cv2.boundingRect(largest)
        b = BODY_BBOX_BUFFER_PX
        body_bbox = (x - b, y - b, x + w + b, y + h + b)

    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (MIN_BALL_AREA <= area <= MAX_BALL_AREA):
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if not (MIN_BALL_RADIUS <= radius <= MAX_BALL_RADIUS):
            continue
        if body_bbox is not None and (
            body_bbox[0] <= cx <= body_bbox[2] and body_bbox[1] <= cy <= body_bbox[3]
        ):
            # Inside the body / club arc — skip.
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


def _estimate_cur_to_ref(ref_gray, cur_gray):
    """Return a 2x3 affine mapping cur_gray pixel coords → ref_gray coords,
    or None if registration fails.

    ORB features + brute-force Hamming matching + RANSAC affine fit. Robust
    to the moderate frame-to-reference displacements we see across a single
    handheld iPhone clip (a few seconds, modest pan / shake). For very
    aggressive camera motion we'd need to re-anchor the reference or fall
    back to full homography, but neither shows up in the par-3 tee-cam use
    case yet.
    """
    try:
        orb = cv2.ORB_create(nfeatures=MC_ORB_FEATURES)
        kp1, des1 = orb.detectAndCompute(ref_gray, None)
        kp2, des2 = orb.detectAndCompute(cur_gray, None)
        if des1 is None or des2 is None:
            return None
        if len(kp1) < MC_MIN_MATCHES or len(kp2) < MC_MIN_MATCHES:
            return None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) < MC_MIN_MATCHES:
            return None
        matches.sort(key=lambda m: m.distance)
        matches = matches[: max(MC_MIN_MATCHES, int(0.8 * len(matches)))]
        src = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, _inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=MC_RANSAC_REPROJ_PX,
        )
        return M
    except Exception:
        return None


def _find_upward_seeds(detections):
    """Subset of `detections` that participate in a temporally-consistent
    upward chain of length >= MIN_UPWARD_CHAIN_LEN.

    Algorithm: DP in both directions. fwd[d] = longest upward chain
    starting at d going forward; bwd[d] = longest ending at d coming
    from earlier frames. A detection is kept iff fwd[d] + bwd[d] - 1 >=
    MIN_UPWARD_CHAIN_LEN — i.e., d sits somewhere on a chain that long.

    "Upward" means y_next < y_cur - MIN_UPWARD_DY_PER_FRAME * frame_gap
    (image coords: smaller y = higher on screen). Spatial gate reuses
    MAX_PIXEL_JUMP_PER_FRAME so we don't link blobs across the frame.

    Cost: O(N * UPWARD_SEARCH_FRAMES * avg_cands_per_frame). At 60K
    candidates and ~300 cands/frame this is well under a second on a
    laptop CPU.
    """
    by_frame: dict[int, list[_Det]] = {}
    for d in detections:
        by_frame.setdefault(d.frame, []).append(d)
    frames = sorted(by_frame.keys())
    if not frames:
        return []

    fwd: dict[int, int] = {}
    for f in reversed(frames):
        for d in by_frame[f]:
            best = 1
            for df in range(1, UPWARD_SEARCH_FRAMES + 1):
                for nd in by_frame.get(f + df, ()):
                    if d.y - nd.y < MIN_UPWARD_DY_PER_FRAME * df:
                        continue
                    if abs(nd.x - d.x) > MAX_PIXEL_JUMP_PER_FRAME * df:
                        continue
                    if abs(nd.y - d.y) > MAX_PIXEL_JUMP_PER_FRAME * df:
                        continue
                    candidate = 1 + fwd.get(id(nd), 1)
                    if candidate > best:
                        best = candidate
            fwd[id(d)] = best

    bwd: dict[int, int] = {}
    for f in frames:
        for d in by_frame[f]:
            best = 1
            for df in range(1, UPWARD_SEARCH_FRAMES + 1):
                for pd in by_frame.get(f - df, ()):
                    if pd.y - d.y < MIN_UPWARD_DY_PER_FRAME * df:
                        continue
                    if abs(pd.x - d.x) > MAX_PIXEL_JUMP_PER_FRAME * df:
                        continue
                    if abs(pd.y - d.y) > MAX_PIXEL_JUMP_PER_FRAME * df:
                        continue
                    candidate = 1 + bwd.get(id(pd), 1)
                    if candidate > best:
                        best = candidate
            bwd[id(d)] = best

    return [d for d in detections
            if fwd[id(d)] + bwd[id(d)] - 1 >= MIN_UPWARD_CHAIN_LEN]
