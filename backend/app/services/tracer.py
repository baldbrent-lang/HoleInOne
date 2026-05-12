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
# v3: bumped 16 → 50 after outdoor footage (wind-moved foliage / grass,
# HEVC shimmer) produced 600+ candidates/frame at threshold 16. 50
# corresponds to ~7σ above the per-pixel background gaussian, which
# suppresses random texture wiggle but easily lets the actual ball
# through (the ball traverses pixels whose σ is tiny because the
# background there is mostly uniform sky / grass).
BG_VAR_THRESHOLD = 50

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

# Motion-density "hot mask". After scanning the clip we have a per-pixel
# count of how often that pixel showed up in the MOG2 foreground. The
# golfer's body stays in roughly the same place across the entire swing,
# so its pixels accumulate huge counts. Wind-blown leaves and tree edges
# accumulate moderate counts. A real ball-flight pixel is touched in 1-2
# frames at most. Thresholding the heatmap therefore gives us a static
# "ignore" region that kills body silhouette, club corridor, and tree
# noise in one pass — leaving the sparse, transient ball detections.
# Threshold is taken as max(floor, pct * counted_frames) so it scales
# with clip length but never drops below the floor on very short clips.
HOT_MASK_PCT = 0.05
HOT_MASK_MIN_HITS = 8
HOT_MASK_DILATE_PX = 5

# Performance ceilings. Without these the tracer runs for minutes on
# long clips and the browser request times out before the server
# finishes. These caps trade some coverage for keeping the round-trip
# under ~2 minutes on Replit's shared CPU.
MAX_FRAMES_PROCESS = 1800    # ~15s @ 120fps, ~30s @ 60fps, ~60s @ 30fps
MAX_TOTAL_CANDIDATES = 60000  # Early abort if drowning in noise
# Downsample to at most this wide for detection. Track coordinates
# are scaled back to native resolution before rendering so the
# overlay still lines up.
DETECT_MAX_WIDTH = 1280

MAX_FRAME_GAP = 6
MAX_PIXEL_JUMP_PER_FRAME = 90
# Tracks shorter than this aren't considered real ball flight. Lowered
# from 8 — was hard-rejecting some real but partially-detected ball
# flights. Score formula's length bonus + alignment bonus still
# strongly prefer longer chains.
MIN_TRACK_LENGTH = 6
# Frame-parameterized RMS. Loosened to 50 — a real ball-flight chain
# typically runs 5-20 px, but with the alignment bonus heavily
# rewarding parabola-aligned yellow extensions, we can afford to let
# slightly noisier chains through. The bonus will pick out the right
# one anyway.
MAX_PARABOLA_RESIDUAL = 50.0
# A "trajectory" stuck in one corner of the sky (random noise) often
# has tiny total displacement. A real ball flight covers a big chunk
# of the frame. Reject anything whose total motion is smaller than
# this (in pixels, measured as max of x-range or y-range).
MIN_TRAJECTORY_SPAN_PX = 20.0

# --- v3 noise reducers ------------------------------------------------------
# Two stages added to address the "noise blanketing the whole frame" failure
# mode we saw on iPhone tee-cam footage (grass micro-motion + sky compression
# shimmer + handheld shake produced 600+ candidates per frame).

# Stage A — global motion compensation. Estimate a per-frame affine transform
# from the current frame to a reference frame using ORB + RANSAC, warp the
# current frame into reference coords, then feed the WARPED frame to MOG2.
# This collapses camera-shake noise so MOG2 only flags genuine scene motion.
# Detections are inverse-warped back to current-frame coords for rendering.
#
# Default OFF: on tripod-mounted tee-cam footage (the production case) MC
# isn't earning its keep — and when `_estimate_cur_to_ref` silently fails
# (ORB+RANSAC is flaky on HEVC iPhone footage with uniform grass/sky), the
# heatmap stops accumulating and the hot-mask filter degenerates to a
# no-op. Flip back to True only for handheld footage where camera shake
# is visible in the raw clip.
USE_MOTION_COMPENSATION = False
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
# v3 tightening: prior thresholds (chain_len=3, dy=1.5px) were so
# permissive that with 200+ cands/frame, random pairs satisfied the
# upward criterion by chance and the filter became a near no-op
# (kept 96%+ of raw). A real ball moves 10-30px/frame at launch — 6px
# is still permissive enough for the apex / blurred frames where it
# slows, but enough to discard wind-wiggle noise.
MIN_UPWARD_CHAIN_LEN = 5
MIN_UPWARD_DY_PER_FRAME = 6.0
UPWARD_SEARCH_FRAMES = 2

# Hard cap on the number of candidates we keep per frame. Even with
# tighter MOG2 + body-blob exclusion, outdoor footage with wind +
# foliage can still produce dozens of ball-shaped fragments per frame.
# At that density any temporal-coherence filter (upward streak,
# parabolic fit) finds chains by chance. Capping per-frame guarantees
# the temporal filters operate on a sparse field. Ranking is by
# circularity scaled by smallness, so the most ball-like survive.
MAX_CANDIDATES_PER_FRAME = 40

# Parabola extension. The upward-streak filter intentionally truncates
# the track at the apex (the ball moves <6px upward there, so no chain
# forms across it). Once we have a confident seed track of the rising
# arc, we fit y(frame) as a quadratic + x(frame) as linear, sweep ALL
# post-hot-mask detections, and pull in anything within `eps` of the
# predicted position. Eps grows with distance from the seed range
# because extrapolation is less certain the further out we go — and
# we refit after every accepted point so the curvature improves as the
# apex comes into view.
EXTEND_EPS_MIN_PX = 15.0
EXTEND_EPS_MULT = 3.0
EXTEND_EPS_GROWTH_PER_FRAME = 0.05
# Eps caps at base * (1 + EXTEND_EPS_MAX_GROWTH). Without this cap a
# 400-frame forward extension lets eps balloon to base * 40 and the
# extension grabs almost every cand in the clip — dragging the fit
# into a wild parabola spanning the whole frame.
EXTEND_EPS_MAX_GROWTH = 2.0
EXTEND_BACKWARD_FRAMES = 60
EXTEND_MAX_CONSECUTIVE_MISS = 5
EXTEND_OUT_OF_FRAME_MARGIN_PX = 80

# Ball-address detection. During the warmup frames the ball is stationary
# at address. Find it by looking for a small bright blob in the lower
# half of the frame that appears in multiple consecutive frames at the
# same position. Once we know which side of the body's x-center the
# ball sits on, we can reject post-mask candidates on the opposite
# side — those are body silhouette fragments or background noise that
# the ball physically can't be near.
#
# v2 detection uses top-hat morphology rather than a fixed brightness
# threshold. A golf ball on grass is a small *locally bright* feature
# — significantly brighter than its surrounding green — but its
# absolute brightness varies wildly with cloud cover, shadow, and
# camera exposure. Top-hat (image minus its morphological opening with
# a kernel bigger than the ball) isolates exactly that kind of
# local-bright peak and ignores evenly-lit grass texture.
BALL_ADDR_TOPHAT_KERNEL_PX = 15  # > expected ball diameter
# Local-brightness contrast (0-255) a blob must exceed in the top-hat
# output. Was 25 — that caught grass texture, cloud highlights, and
# compression noise, producing ~1000 blobs/frame on busy outdoor
# clips and making _find_hit_ball_from_snapshots' downstream matching
# loops run on absurdly large blob sets. Real golf balls produce
# top-hat values of 60-120, so 50 is a comfortable floor that keeps
# the real ball and cuts background noise 5-10×.
BALL_ADDR_TOPHAT_THRESH = 50
# Reject non-circular blobs in the top-hat output. Grass texture and
# compression artifacts produce elongated / irregular shapes; a real
# ball is roughly circular even partially occluded.
BALL_ADDR_MIN_CIRCULARITY = 0.5
BALL_ADDR_MIN_OCCURRENCES = 3
BALL_ADDR_POSITION_TOLERANCE_PX = 12
BALL_ADDR_Y_TOP_FRACTION = 0.45  # only search below this fraction of the frame
BALL_ADDR_SEARCH_SECONDS = 1.0   # convert to frames at runtime: int(fps * seconds)

# Club-shaft handedness detection. At address the club shaft is a long
# diagonal line extending from the golfer's hands (upper-middle of body)
# down to the clubhead at his feet. Its slope reveals handedness more
# reliably than the ball's position — a tee-cam clip may have other
# practice balls scattered in the grass that confuse ball-address
# detection, but only the actual club shaft will form a long consistent
# diagonal in the address phase.
#
# Slope sign convention (image y grows downward):
#   slope > 0  → clubhead to RIGHT of hands → LEFTY
#   slope < 0  → clubhead to LEFT of hands → RIGHTY
#
# For tee-cam from behind, target lies on the side of the body the
# clubhead points to, so "behind the golfer" = the OPPOSITE side, which
# is where we want to reject post-mask candidates.
CLUB_DETECT_SEARCH_SECONDS = 1.0  # convert to frames at runtime: int(fps * seconds)
CLUB_MIN_LINE_LEN_PX = 40
CLUB_CANNY_LOW = 50
CLUB_CANNY_HIGH = 150
CLUB_HOUGH_THRESHOLD = 40
CLUB_HOUGH_MAX_GAP = 8
CLUB_X_MID_LO_FRACTION = 0.25  # lower endpoint must land in middle half
CLUB_X_MID_HI_FRACTION = 0.75
CLUB_Y_REACH_FRACTION = 0.40   # line must reach into lower 60% of frame
CLUB_UPPER_Y_MAX_FRACTION = 0.55  # upper endpoint must be in upper 55%
# Reject anything close to horizontal — the grass edge / horizon / bench
# tops will otherwise dominate longest-line voting at slope ~0. At
# address from behind, a real shaft is at least 45° off horizontal.
CLUB_MIN_SLOPE_MAGNITUDE = 1.0
CLUB_SLOPE_SIGN_THRESHOLD = 0.3
CLUB_MIN_VOTES = 3
# A real club at address ends at the clubhead, which sits right at the
# golfer's feet — meaning its lower endpoint should be near the body
# bbox bottom (where the feet are). Lines whose lower endpoint is
# inside the body silhouette (arms / body edges / shirt seams) won't
# pass this proximity check after the loop.
CLUB_LOWER_NEAR_BODY_PX_ABOVE = 60   # tolerated above body bottom
CLUB_LOWER_NEAR_BODY_PX_BELOW = 100  # tolerated below body bottom

# Require ball-address candidates to be near the detected clubhead.
# Beyond this distance the blob is almost certainly a stray practice
# ball / shoe / belt-buckle reflection, not the ball being addressed.
# Tightened 120 → 80 after a clip where the golfer's white shoe at
# ~100 px from the clubhead won the disappearance vote, putting the
# ball anchor on the wrong side of the body and corrupting the picked
# trajectory. The actual ball sits within ~50 px of the clubhead even
# when the Hough line's lower endpoint lands slightly up the shaft,
# so 80 px gives the real ball headroom without admitting the
# off-axis false positives.
MAX_BALL_TO_CLUBHEAD_PX = 80

TRACER_COLOR = (0, 140, 255)        # bright orange (BGR)
TRACER_HALO_COLOR = (40, 90, 200)   # dimmer orange behind the dashes
TRACER_THICKNESS = 5
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

    # Per-pixel foreground-hit count at detection resolution. After the
    # scan finishes we threshold this to a binary "hot mask" of regions
    # the ball never visits but the body/club/foliage always does.
    heatmap = np.zeros((det_h, det_w), dtype=np.uint16)
    counted_frames = 0  # frames after warmup that contributed to heatmap

    # Raw candidates are stored at DETECTION resolution so we can index
    # them into the heatmap before scaling survivors back to native.
    raw_cands_det: list[tuple[int, float, float, float]] = []
    first_frame_snapshot = None

    # Motion compensation state. Reference frame is the first frame we
    # see (at detection resolution, grayscale). Every subsequent frame is
    # warped INTO the reference's coordinate space before MOG2 sees it.
    ref_gray = None
    mc_failed_frames = 0
    # Per-frame inverse affine transform (ref → current). Stored so we
    # can apply the hot-mask filter in ref-coords (where the heatmap
    # lives) and only then unwarp survivors back to current-frame coords.
    frame_M_inv: dict[int, "np.ndarray | None"] = {}
    # Stationary-ball-blob votes accumulated during warmup, used to fix
    # the ball's address position before the swing starts.
    ball_addr_votes: list[tuple[float, float]] = []
    # Frame-index snapshots saved during the main loop for the
    # disappearance-based ball detector. early_* sits in the first ~1.5s
    # (ball at address); late_* sits in the last ~1.5s of the clip
    # (ball gone). We cache here rather than seeking back to these
    # frames later — random-access seeks in HEVC .mov files are slow
    # enough (1 sec of decoding per seek) that a separate pass can
    # hang the whole request. If the late window is past
    # MAX_FRAMES_PROCESS, late_sample_indices stays empty and the
    # disappearance detector returns None, falling back to the
    # vote-based scan.
    total_frames_in_clip = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    early_sample_indices: set[int] = {
        int(t * fps) for t in (0.2, 0.4, 0.6, 0.8, 1.1, 1.4)
    }
    late_sample_indices: set[int] = set()
    if 0 < total_frames_in_clip <= MAX_FRAMES_PROCESS:
        late_sample_indices = {
            total_frames_in_clip - int(t * fps)
            for t in (1.5, 1.2, 0.9, 0.7, 0.5, 0.3)
            if 0 <= total_frames_in_clip - int(t * fps) < MAX_FRAMES_PROCESS
        }
    early_snapshots: dict[int, "np.ndarray"] = {}
    late_snapshots: dict[int, "np.ndarray"] = {}
    # Club-shaft slope votes for handedness inference. Each vote is a
    # signed slope (dy/dx with image-y down); sign indicates which side
    # of the body the clubhead is on. Plus a single representative
    # line (longest seen) used to draw on the debug image. Stored as
    # raw (line, slope) candidates; the body-bbox proximity filter is
    # applied after the main loop, once we have body geometry from
    # the hot mask.
    club_line_candidates: list[tuple[tuple[int, int, int, int], float]] = []

    # Per-clip windows scaled by fps. At 120fps a 1s address phase is
    # 120 frames; at 30fps it's 30. Constant frame counts would either
    # cut high-fps clips short or over-extend low-fps ones.
    ball_addr_search_n = max(WARMUP_FRAMES, int(round(fps * BALL_ADDR_SEARCH_SECONDS)))
    club_detect_search_n = max(WARMUP_FRAMES, int(round(fps * CLUB_DETECT_SEARCH_SECONDS)))
    # Per-frame upward-motion threshold also scales — the ball travels
    # 4× less per frame at 120fps than at 30fps, so a fixed 6px floor
    # filters out everything past the steepest rising portion.
    min_upward_dy = MIN_UPWARD_DY_PER_FRAME * (30.0 / max(fps, 1.0))

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
        # Snapshot caching for the disappearance-based ball detector.
        # We cache the raw det_frame at the pre-computed sample indices
        # so the post-loop detector can scan early-vs-late without
        # paying the HEVC random-access seek cost.
        if idx in early_sample_indices:
            early_snapshots[idx] = det_frame.copy()
        if idx in late_sample_indices:
            late_snapshots[idx] = det_frame.copy()
        # Address-position vote while the ball is plausibly still at
        # address. Independent of MOG2 warmup — the ball typically
        # stays motionless for ~1s while the golfer settles.
        if idx < ball_addr_search_n:
            ball_addr_votes.extend(_scan_for_ball_addr_blobs(det_frame))
        # Club-shaft candidates during the address window. We collect
        # all qualifying lines per frame and filter them against the
        # body bbox after the main loop.
        if idx < club_detect_search_n:
            for cand in _detect_club_candidates_in_frame(det_frame):
                club_line_candidates.append(cand)
        if idx < WARMUP_FRAMES:
            idx += 1
            continue
        # Heatmap only accumulates when fg_mask lives in a consistent
        # coord space across frames — i.e., motion-comp succeeded (or
        # is disabled). A motion-comp-failed frame's mask is in raw
        # current-frame coords and would smear the heatmap.
        if M_to_ref is not None or not USE_MOTION_COMPENSATION:
            heatmap += (fg_mask > 0).astype(np.uint16)
            counted_frames += 1
        if USE_MOTION_COMPENSATION:
            frame_M_inv[idx] = (
                cv2.invertAffineTransform(M_to_ref) if M_to_ref is not None else None
            )
        cands_det = _candidates_from_mask(fg_mask)
        # Store at REFERENCE-frame detection coords (no unwarp yet) so we
        # can apply the hot-mask filter (which lives in ref coords) before
        # paying the unwarp + scale cost. When motion-comp failed for
        # this frame, frame_M_inv[idx] is None and the cand stays in
        # current-frame coords; we'll skip the hot-mask filter for it
        # downstream.
        for cx, cy, r in cands_det:
            raw_cands_det.append((idx, cx, cy, r))
        if len(raw_cands_det) > MAX_TOTAL_CANDIDATES:
            aborted_early = True
            log.info("tracer: hit MAX_TOTAL_CANDIDATES=%d; aborting (noise overwhelming)", MAX_TOTAL_CANDIDATES)
            break
        if idx and idx % 120 == 0:
            log.info("tracer: %d frames scanned, %d raw cands so far", idx, len(raw_cands_det))
        idx += 1
    cap.release()
    if USE_MOTION_COMPENSATION:
        log.info("tracer: motion-comp registration failed on %d/%d frames",
                 mc_failed_frames, max(idx, 1))
    total_frames_scanned = idx

    # Hot-mask: threshold the per-pixel heatmap into a binary "always
    # moving" mask in ref-frame coords (body silhouette, club corridor,
    # wind-blown foliage). A real ball touches each pixel for 1-2 frames
    # so it sails through; static-corridor noise gets killed.
    hot_mask = _build_hot_mask(heatmap, counted_frames)
    hot_pixels = int(hot_mask.sum()) if hot_mask is not None else 0
    log.info(
        "tracer: heatmap counted_frames=%d  hot_mask non-zero px=%d (%.1f%% of detect area)",
        counted_frames, hot_pixels,
        100.0 * hot_pixels / max(1, det_w * det_h),
    )

    # Ball address (detection-coord) + body bbox → kept for display
    # and as fallback signals. The primary handedness signal now comes
    # from the club shaft's slope, which is robust to stray practice
    # balls in the grass.
    log.info("tracer: post-loop: computing body bbox + silhouette mask...")
    body_bbox_det, body_mask_det = _hot_mask_body_components(hot_mask)
    body_x_det = (
        float((body_bbox_det[0] + body_bbox_det[2]) / 2.0)
        if body_bbox_det is not None else None
    )
    body_bottom_det = float(body_bbox_det[3]) if body_bbox_det is not None else None
    log.info(
        "tracer: post-loop: body_bbox=%s, filtering %d club-line candidates...",
        body_bbox_det, len(club_line_candidates),
    )

    # Filter club-line candidates by body-bbox proximity: a real club
    # ends at the clubhead near the body's feet, not inside the body
    # silhouette. Apply this only if we have a body bbox; otherwise
    # fall back to the unfiltered candidate pool.
    club_slope_votes: list[float] = []
    club_best_line: tuple[int, int, int, int] | None = None
    club_best_len_sq = 0
    for (line, slope) in club_line_candidates:
        _x1, _y1, _x2, y2 = line
        if body_bottom_det is not None:
            if (y2 < body_bottom_det - CLUB_LOWER_NEAR_BODY_PX_ABOVE
                    or y2 > body_bottom_det + CLUB_LOWER_NEAR_BODY_PX_BELOW):
                continue
        club_slope_votes.append(slope)
        x1, y1, x2, y2c = line
        len_sq = (x2 - x1) ** 2 + (y2c - y1) ** 2
        if len_sq > club_best_len_sq:
            club_best_len_sq = len_sq
            club_best_line = line
    handedness = _handedness_from_slopes(club_slope_votes)

    # Decide handedness / behind_side from the club first.
    behind_side: str | None = None
    if handedness == "lefty":
        behind_side = "left"
    elif handedness == "righty":
        behind_side = "right"

    # Clubhead position (lower endpoint of detected club line) — used as
    # a prior for the vote-based ball-address picker if we fall back to it.
    clubhead_det = (
        (float(club_best_line[2]), float(club_best_line[3]))
        if club_best_line is not None else None
    )
    log.info(
        "tracer: post-loop: handedness=%s, behind_side=%s, clubhead=%s; "
        "running disappearance detector on %d early + %d late snapshots",
        handedness or "unknown", behind_side or "none", clubhead_det,
        len(early_snapshots), len(late_snapshots),
    )
    # Primary ball detector: identify the white blob that's present in
    # early frames but ABSENT in late frames. The hit ball is the only
    # thing that disappears between address and follow-through — practice
    # balls scattered in the foreground stay put. Uses snapshots cached
    # in the main loop above, so no extra video IO. Additionally
    # filtered by clubhead proximity + behind_side so a stray practice
    # ball that got kicked/hidden mid-clip can't pass as the hit ball.
    ball_addr_det = _find_hit_ball_from_snapshots(
        early_snapshots, late_snapshots,
        body_x=body_x_det, behind_side=behind_side, clubhead=clubhead_det,
    )
    if ball_addr_det is not None:
        log.info(
            "tracer: ball found via disappearance check at (%.0f, %.0f)  "
            "(%d early snapshots, %d late snapshots)",
            ball_addr_det[0], ball_addr_det[1],
            len(early_snapshots), len(late_snapshots),
        )
    else:
        # Fallback: in-loop vote-based scan (top-hat across the address
        # window). Less reliable than disappearance — confused by stray
        # practice balls — but catches cases where the clip doesn't have
        # a clear late-window "empty" state.
        ball_addr_det = _pick_ball_addr(
            ball_addr_votes,
            body_x=body_x_det,
            behind_side=behind_side,
            clubhead=clubhead_det,
        )
        if ball_addr_det is not None:
            log.info(
                "tracer: ball found via address-vote fallback at (%.0f, %.0f)",
                ball_addr_det[0], ball_addr_det[1],
            )
        else:
            log.info("tracer: ball not found (both detectors failed)")

    # If the club gave no decisive handedness, fall back to inferring it
    # from the (now side-unfiltered) ball-address position.
    if behind_side is None and ball_addr_det is not None and body_x_det is not None:
        if ball_addr_det[0] < body_x_det - 1:
            behind_side = "right"
        elif ball_addr_det[0] > body_x_det + 1:
            behind_side = "left"
    n_pos = sum(1 for s in club_slope_votes if s > CLUB_SLOPE_SIGN_THRESHOLD)
    n_neg = sum(1 for s in club_slope_votes if s < -CLUB_SLOPE_SIGN_THRESHOLD)
    log.info(
        "tracer: club slope votes %d (pos=%d neg=%d) → handedness=%s  "
        "ball_addr=%s  body_x=%s  behind_side=%s",
        len(club_slope_votes), n_pos, n_neg, handedness or "unknown",
        f"({ball_addr_det[0]:.0f},{ball_addr_det[1]:.0f})" if ball_addr_det else "None",
        f"{body_x_det:.0f}" if body_x_det is not None else "None",
        behind_side or "none",
    )

    # One-shot pass: filter by hot mask, foot line (everything below the
    # body bbox is divot debris / practice balls / shadow), the
    # behind-the-golfer half, then unwarp the survivor back to
    # current-frame coords and scale to native. Frames where motion-comp
    # failed keep their cand (no reliable hot-mask filtering possible)
    # and skip the unwarp.
    inv = 1.0 / det_scale if det_scale != 0 else 1.0
    foot_y_cutoff = (
        body_bottom_det + 15 if body_bottom_det is not None else float("inf")
    )
    # The body bbox bottom is often the bottom of the *feet*, but the
    # ball at rest can sit a few px BELOW that on the grass (camera-
    # angle dependent). Extend the cutoff to be at least 15 px below
    # the ball-at-rest, so impact-area detections at ground level
    # don't get foot-rejected.
    if ball_addr_det is not None:
        foot_y_cutoff = max(foot_y_cutoff, ball_addr_det[1] + 15)
    detections: list[_Det] = []
    hot_rejected = 0
    side_rejected = 0
    foot_rejected = 0
    for f, cx, cy, r in raw_cands_det:
        if cy > foot_y_cutoff:
            foot_rejected += 1
            continue
        M_inv = frame_M_inv.get(f) if USE_MOTION_COMPENSATION else None
        if M_inv is not None or not USE_MOTION_COMPENSATION:
            iy, ix = int(cy), int(cx)
            if 0 <= iy < det_h and 0 <= ix < det_w and hot_mask[iy, ix]:
                hot_rejected += 1
                continue
        if behind_side == "left" and body_x_det is not None and cx < body_x_det:
            side_rejected += 1
            continue
        if behind_side == "right" and body_x_det is not None and cx > body_x_det:
            side_rejected += 1
            continue
        if M_inv is not None:
            cx_cur = float(M_inv[0, 0] * cx + M_inv[0, 1] * cy + M_inv[0, 2])
            cy_cur = float(M_inv[1, 0] * cx + M_inv[1, 1] * cy + M_inv[1, 2])
        else:
            cx_cur, cy_cur = cx, cy
        detections.append(_Det(f, cx_cur * inv, cy_cur * inv, r * inv))
    log.info(
        "tracer: hot-mask rejected %d, foot rejected %d, side-filter rejected %d, kept %d / %d raw",
        hot_rejected, foot_rejected, side_rejected, len(detections), len(raw_cands_det),
    )

    # Stage B: prune candidates that aren't part of any short upward
    # streak. Done AFTER hot-mask so the upward filter operates on a
    # sparse field (no static-corridor noise contaminating the streaks).
    if USE_UPWARD_STREAK_FILTER and detections:
        seeds = _find_upward_seeds(detections, min_dy_per_frame=min_upward_dy)
        log.info(
            "tracer: upward-streak filter (min_dy=%.2fpx @ %.1ffps) kept %d/%d candidates",
            min_upward_dy, fps, len(seeds), len(detections),
        )
    else:
        seeds = detections

    # Recompute "busiest frame" from POST-hot-mask counts — what the
    # upward-streak filter and linker actually see.
    busiest_frame_idx = -1
    busiest_count = 0
    busiest_candidates: list[tuple[float, float, float]] = []
    busiest_frame = None
    if detections:
        per_frame: dict[int, list[_Det]] = {}
        for d in detections:
            per_frame.setdefault(d.frame, []).append(d)
        busiest_frame_idx = max(per_frame, key=lambda f: len(per_frame[f]))
        bf = per_frame[busiest_frame_idx]
        busiest_count = len(bf)
        busiest_candidates = [(d.x, d.y, d.radius) for d in bf]
        busiest_frame = _grab_frame(input_path, busiest_frame_idx)

    # Skip linking only when we hit the noise ceiling AND can't form a
    # minimum-length seed chain anyway.
    aborting_noise = (
        aborted_early
        and len(raw_cands_det) >= MAX_TOTAL_CANDIDATES
        and len(seeds) < MIN_TRACK_LENGTH
    )

    seed_track: list[_Det] = []
    track: list[_Det] = []
    if not aborting_noise:
        log.info(
            "tracer: linking %d seeds (post-hot-mask: %d, raw: %d)...",
            len(seeds), len(detections), len(raw_cands_det),
        )
        ball_addr_native = None
        if ball_addr_det is not None and det_scale > 0:
            ball_addr_native = (ball_addr_det[0] * inv, ball_addr_det[1] * inv)
        log.info("tracer: _link starting on %d seeds...", len(seeds))
        linked = _link(seeds)
        log.info("tracer: _link produced %d trajectories; scoring...", len(linked))
        seed_track = _pick_best(
            linked,
            ball_addr_native=ball_addr_native,
            body_mask_det=body_mask_det,
            det_scale=det_scale,
            all_detections=detections,
        )
        if seed_track:
            log.info(
                "tracer: seed track %d points, residual %.2fpx",
                len(seed_track), _residual(seed_track),
            )
            track = _extend_track(
                seed_track, detections, total_frames_scanned,
                frame_w=width, frame_h=height,
            )

    # Prefer the at-rest ball position from address-time detection.
    # That's the *actual* ball before it was struck — what the user
    # wants to see. track[0] is only the first frame where the linker
    # caught the ball in motion, which can be several frames into the
    # flight (already a few feet up). Fall back to it only when the
    # address-time top-hat scan finds nothing.
    picked_for_debug = track or seed_track
    ball_pos_native = None
    ball_pos_source = None
    if ball_addr_det is not None and det_scale > 0:
        ball_pos_native = (ball_addr_det[0] * inv, ball_addr_det[1] * inv)
        ball_pos_source = "address-vote"
    elif picked_for_debug:
        first = min(picked_for_debug, key=lambda p: p.frame)
        ball_pos_native = (float(first.x), float(first.y))
        ball_pos_source = "track"
    log.info(
        "tracer: ball_pos=%s (source=%s)",
        f"({ball_pos_native[0]:.0f},{ball_pos_native[1]:.0f})" if ball_pos_native else "None",
        ball_pos_source or "none",
    )

    if debug_path is not None:
        filmstrip = _sample_filmstrip(input_path, total_frames_scanned)
        _write_debug(
            debug_path, busiest_frame, busiest_frame_idx, busiest_candidates,
            first_frame_snapshot, detections, seeds, filmstrip,
            total_frames_scanned, fps,
            n_raw=len(raw_cands_det), hot_mask=hot_mask, det_scale=det_scale,
            native_size=(width, height),
            ball_addr_det=ball_addr_det, body_x_det=body_x_det,
            behind_side=behind_side,
            club_line_det=club_best_line, handedness=handedness,
            body_bottom_det=body_bottom_det, picked_track=picked_for_debug,
            ball_pos_native=ball_pos_native, ball_pos_source=ball_pos_source,
        )

    if aborting_noise:
        return {
            "ok": False,
            "error": f"noise overwhelming ({len(raw_cands_det)}+ candidates pre-mask) — check the debug image for the source (rain on lens, camera shake, etc.)",
            "n_candidates": len(detections),
            "n_points": 0,
        }
    if not seed_track:
        return {
            "ok": False,
            "error": "no usable trajectory",
            "n_candidates": len(detections),
            "n_points": 0,
        }
    if len(track) > len(seed_track):
        log.info(
            "tracer: extended track %d → %d points (residual %.2fpx)",
            len(seed_track), len(track), _residual(track),
        )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    cap2 = cv2.VideoCapture(str(input_path))
    # Pass the at-rest ball position into smoothing only when it came
    # from the disappearance / address-vote detector (the actual ball).
    # For "track" source, ball_pos == track[0], so adding it as an
    # anchor would be a no-op (or worse, slightly distort the fit).
    smooth_anchor = ball_pos_native if ball_pos_source == "address-vote" else None
    smoothed = _smooth_track_for_render(track, ball_pos_native=smooth_anchor)
    smooth_by_frame = {f: (x, y) for (f, x, y) in smoothed}
    smooth_frames = sorted(smooth_by_frame)
    detection_frames = {int(d.frame) for d in track}
    # If we know the ball-at-rest position, prepend it to the rendered
    # polyline so the dashed line starts at the actual ball, draws a
    # straight segment up to track[0], and then curves through the
    # smoothed parabola. Only do this for the verified at-rest position
    # (source=address-vote) — when source=track, ball_pos == track[0]
    # already so prepending is a no-op.
    # The smoothing function (_smooth_track_for_render) now anchors the
    # parabola at the ball's at-rest position when source == "address-vote",
    # so the smoothed curve already starts at the ball. No prepend
    # needed here — prepending an exact-ball-position point on top of
    # the fitted-at-anchor point would create a small kink between
    # the two slightly-different positions.
    i = 0
    while True:
        ok, frame = cap2.read()
        if not ok:
            break
        seen = [smooth_by_frame[f] for f in smooth_frames if f <= i]
        _draw_dashed(frame, seen)
        if i in detection_frames and i in smooth_by_frame:
            cv2.circle(frame, smooth_by_frame[i], 7, BALL_HIGHLIGHT_COLOR, 2, cv2.LINE_AA)
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


def _detect_club_candidates_in_frame(det_frame):
    """Find all plausible club-shaft line segments in a single frame.
    Returns a list of ((x_top, y_top, x_bot, y_bot), slope).

    These are *candidates*. Final filtering by the body bbox (the actual
    clubhead at address sits near the body's bottom edge, not inside the
    body silhouette) happens after the main loop, once the hot mask
    gives us body geometry. We can't apply that filter here because the
    body bbox doesn't exist yet during the warmup frames.
    """
    h, w = det_frame.shape[:2]
    gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CLUB_CANNY_LOW, CLUB_CANNY_HIGH)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=CLUB_HOUGH_THRESHOLD,
        minLineLength=CLUB_MIN_LINE_LEN_PX,
        maxLineGap=CLUB_HOUGH_MAX_GAP,
    )
    if lines is None:
        return []
    x_lo = w * CLUB_X_MID_LO_FRACTION
    x_hi = w * CLUB_X_MID_HI_FRACTION
    y_reach = h * CLUB_Y_REACH_FRACTION
    y_upper_max = h * CLUB_UPPER_Y_MAX_FRACTION
    out = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Put upper endpoint first (smaller y).
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        if not (x_lo <= x2 <= x_hi):
            continue  # lower end out of the middle band — bench/tree/etc.
        if y2 < y_reach:
            continue  # line doesn't reach into the lower portion
        if y1 > y_upper_max:
            continue  # upper end still in lower half — flat ground edges
        dx = x2 - x1
        dy = y2 - y1
        # Allow near-vertical lines (the actual club from tee-cam-behind
        # is mostly vertical). |dx|>=2 is enough to give a reliable
        # slope sign — anything below that is noise-level lateral.
        if abs(dx) < 2:
            continue
        slope = float(dy) / float(dx)
        if abs(slope) < CLUB_MIN_SLOPE_MAGNITUDE:
            continue  # too horizontal — grass / horizon / bench top
        out.append(((int(x1), int(y1), int(x2), int(y2)), slope))
    return out


def _handedness_from_slopes(slopes):
    """Vote on handedness from per-frame club slopes. Returns 'lefty',
    'righty', or None if the vote isn't decisive."""
    if not slopes:
        return None
    pos = sum(1 for s in slopes if s > CLUB_SLOPE_SIGN_THRESHOLD)
    neg = sum(1 for s in slopes if s < -CLUB_SLOPE_SIGN_THRESHOLD)
    if pos >= CLUB_MIN_VOTES and pos >= 2 * max(neg, 1):
        return "lefty"
    if neg >= CLUB_MIN_VOTES and neg >= 2 * max(pos, 1):
        return "righty"
    return None


def _cluster_positions(positions, tol):
    """Group (x, y) positions whose pairwise Chebyshev distance is
    within `tol` into clusters. Returns list of (mean_x, mean_y, count)
    per cluster.

    Uses a spatial hash (grid of size `tol`) so worst-case runtime is
    O(N · avg_bucket_size) instead of O(N²). On busy outdoor clips
    that can yield thousands of input positions, the difference is
    seconds vs. minutes.
    """
    if not positions:
        return []
    cell = max(1, int(tol))
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, (cx, cy) in enumerate(positions):
        buckets.setdefault((int(cx // cell), int(cy // cell)), []).append(i)
    used = [False] * len(positions)
    clusters = []
    for i, (cx, cy) in enumerate(positions):
        if used[i]:
            continue
        cluster = [(cx, cy)]
        used[i] = True
        bx, by = int(cx // cell), int(cy // cell)
        # Members within `tol` (Chebyshev) of (cx, cy) can only live in
        # the 3x3 neighbourhood of buckets around (bx, by).
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((bx + dx, by + dy), ()):
                    if j == i or used[j]:
                        continue
                    cx2, cy2 = positions[j]
                    if abs(cx2 - cx) <= tol and abs(cy2 - cy) <= tol:
                        cluster.append((cx2, cy2))
                        used[j] = True
        mean_x = sum(c[0] for c in cluster) / len(cluster)
        mean_y = sum(c[1] for c in cluster) / len(cluster)
        clusters.append((mean_x, mean_y, len(cluster)))
    return clusters


def _build_position_hash(positions, cell):
    """Return a {(bx, by): [(cx, cy), ...]} grid for fast neighborhood
    queries. `cell` is the grid size — should be the query tolerance,
    so the 3x3 cell neighborhood of any query point covers all points
    within that tolerance."""
    buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for (cx, cy) in positions:
        buckets.setdefault((int(cx // cell), int(cy // cell)), []).append((cx, cy))
    return buckets


def _has_position_within(hash_grid, cx, cy, tol):
    """True iff hash_grid contains any position within Chebyshev `tol`
    of (cx, cy). hash_grid must have been built with cell == tol."""
    cell = tol
    bx, by = int(cx // cell), int(cy // cell)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for (cx2, cy2) in hash_grid.get((bx + dx, by + dy), ()):
                if abs(cx2 - cx) <= tol and abs(cy2 - cy) <= tol:
                    return True
    return False


def _find_hit_ball_from_snapshots(
    early_snapshots, late_snapshots,
    body_x=None, behind_side=None, clubhead=None,
):
    """Find the ball that was hit by comparing white-blob positions in
    cached early vs late snapshot frames.

    Logic: the ball at address sits stationary in the early window; by
    the late window it's gone. Practice balls in the foreground stay
    put (same position in both windows). The hit ball is the ONLY blob
    present early and absent late.

    Disappearance alone isn't enough on a busy driving range — a
    practice ball that got kicked or hidden by the golfer walking
    through can also satisfy the "early-yes, late-no" condition. So
    candidates that survive disappearance are additionally filtered
    by:
      - `clubhead` proximity (within MAX_BALL_TO_CLUBHEAD_PX). The
        actual ball sits right next to the clubhead at address.
      - `behind_side` + `body_x` (reject the side of the body that's
        away from the ball-flight direction). A righty's ball is on
        the left of the body; if disappearance returns a right-side
        blob, it's a wrong-side false positive.

    Snapshots are passed in as {frame_idx: det_frame} dicts populated by
    the main detection loop — no extra video IO or HEVC random-access
    seeks (which are slow enough on .mov files to hang the request).

    Returns (cx, cy) in detection coords, or None.
    """
    if not late_snapshots or not early_snapshots:
        log.info("tracer:   disappearance: skipping (early=%d late=%d)",
                 len(early_snapshots) if early_snapshots else 0,
                 len(late_snapshots) if late_snapshots else 0)
        return None
    early_blobs: list[tuple[float, float]] = []
    for snap in early_snapshots.values():
        early_blobs.extend(_scan_for_ball_addr_blobs(snap))
    log.info("tracer:   disappearance: %d early blobs across %d snapshots",
             len(early_blobs), len(early_snapshots))
    late_blobs: list[tuple[float, float]] = []
    for snap in late_snapshots.values():
        late_blobs.extend(_scan_for_ball_addr_blobs(snap))
    log.info("tracer:   disappearance: %d late blobs across %d snapshots",
             len(late_blobs), len(late_snapshots))

    early_clusters = _cluster_positions(early_blobs, BALL_ADDR_POSITION_TOLERANCE_PX)
    log.info("tracer:   disappearance: %d early clusters", len(early_clusters))
    if not early_clusters:
        return None

    # Spatial hash of late blobs for O(1) "is there a late blob near
    # this cluster?" checks. Without this, on clips with thousands of
    # blobs, the matching loop was O(clusters · late_blobs) and could
    # take minutes.
    tol_late = BALL_ADDR_POSITION_TOLERANCE_PX * 2
    late_hash = _build_position_hash(late_blobs, tol_late)
    raw_candidates: list[tuple[float, float, int]] = []
    for (cx, cy, count) in early_clusters:
        if count < 2:
            continue
        if not _has_position_within(late_hash, cx, cy, tol_late):
            raw_candidates.append((cx, cy, count))
    log.info(
        "tracer:   disappearance: %d candidates have no late match",
        len(raw_candidates),
    )

    if not raw_candidates:
        return None

    # Apply the same plausibility filters the vote-based fallback uses.
    # Disappearance is a strong "this blob is gone" signal but doesn't
    # know anything about where the actual ball-at-address would be.
    candidates: list[tuple[float, float, int]] = []
    rejected_side = rejected_clubhead = 0
    for (cx, cy, count) in raw_candidates:
        if body_x is not None and behind_side in ("left", "right"):
            if behind_side == "left" and cx < body_x:
                rejected_side += 1
                continue
            if behind_side == "right" and cx > body_x:
                rejected_side += 1
                continue
        if clubhead is not None:
            d = float(np.hypot(cx - clubhead[0], cy - clubhead[1]))
            if d > MAX_BALL_TO_CLUBHEAD_PX:
                rejected_clubhead += 1
                continue
        candidates.append((cx, cy, count))
    log.info(
        "tracer:   disappearance: rejected %d on side, %d on clubhead-distance; %d kept",
        rejected_side, rejected_clubhead, len(candidates),
    )
    if not candidates:
        return None

    # Prefer the candidate closest to the clubhead when we have one;
    # otherwise pick the most stationary (highest occurrence count).
    if clubhead is not None:
        candidates.sort(
            key=lambda c: float(np.hypot(c[0] - clubhead[0], c[1] - clubhead[1]))
        )
    else:
        candidates.sort(key=lambda c: -c[2])
    return (candidates[0][0], candidates[0][1])


def _scan_for_ball_addr_blobs(det_frame):
    """Find small bright blobs in the lower portion of a single det-res
    frame, likely to be a stationary golf ball at address.

    Uses top-hat morphology: gray − opened(gray, kernel) leaves only
    features smaller than the kernel that are brighter than their
    surroundings. Robust to varied lighting (cloud cover, shadow) in a
    way a fixed brightness threshold isn't — a ball at 150/255 gray on
    grass at 100/255 has the same 50-unit local contrast as a ball at
    220 on grass at 170, both of which top-hat catches.
    """
    gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (BALL_ADDR_TOPHAT_KERNEL_PX, BALL_ADDR_TOPHAT_KERNEL_PX),
    )
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _ret, bright = cv2.threshold(tophat, BALL_ADDR_TOPHAT_THRESH, 255, cv2.THRESH_BINARY)
    contours, _h = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h = det_frame.shape[0]
    y_floor = h * BALL_ADDR_Y_TOP_FRACTION
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (MIN_BALL_AREA <= area <= MAX_BALL_AREA):
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if not (MIN_BALL_RADIUS <= radius <= MAX_BALL_RADIUS):
            continue
        if cy < y_floor:
            continue
        # Circularity gate: 4π·area / perimeter². 1.0 = perfect circle,
        # < 0.5 is elongated / irregular — grass texture / compression
        # noise that bright-thresholded into ball-sized blobs.
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        circ = 4.0 * np.pi * area / (perimeter ** 2)
        if circ < BALL_ADDR_MIN_CIRCULARITY:
            continue
        blobs.append((float(cx), float(cy)))
    return blobs


def _pick_ball_addr(all_blobs, body_x=None, behind_side=None, clubhead=None):
    """Pick the (cx, cy) ball-address position from voted warmup blobs.

    Priors (applied in order):
      - `behind_side` + `body_x`: drop blobs on the side already filtered
        out by the handedness call. Practice balls scattered on the
        rejected half (the body's side away from target) get excluded.
      - `clubhead`: if multiple position clusters satisfy the minimum
        occurrence count, prefer the one closest to the clubhead's
        detected position. The real ball at address is right beside
        the clubhead.
      - Fallback: among clusters meeting the occurrence threshold,
        pick the one with the most votes.
    """
    if not all_blobs:
        return None
    if body_x is not None and behind_side in ("left", "right"):
        if behind_side == "left":
            blobs = [(cx, cy) for (cx, cy) in all_blobs if cx >= body_x]
        else:
            blobs = [(cx, cy) for (cx, cy) in all_blobs if cx <= body_x]
    else:
        blobs = list(all_blobs)
    if not blobs:
        return None
    tol = BALL_ADDR_POSITION_TOLERANCE_PX
    # Spatial hash so we only check the 3x3 cell neighborhood for each
    # blob instead of every other blob. O(N) instead of O(N²) — busy
    # outdoor clips can produce 100k+ votes across the warmup window
    # and the original double loop was a multi-minute hang on those.
    hash_grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for (cx, cy) in blobs:
        hash_grid.setdefault((int(cx // tol), int(cy // tol)), []).append((cx, cy))
    candidates: list[tuple[int, float, float]] = []
    for (cx, cy) in blobs:
        bx, by = int(cx // tol), int(cy // tol)
        c = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for (cx2, cy2) in hash_grid.get((bx + dx, by + dy), ()):
                    if abs(cx2 - cx) <= tol and abs(cy2 - cy) <= tol:
                        c += 1
        if c >= BALL_ADDR_MIN_OCCURRENCES:
            candidates.append((c, cx, cy))
    if not candidates:
        return None
    if clubhead is not None:
        # Reject anything beyond MAX_BALL_TO_CLUBHEAD_PX of the clubhead
        # entirely. Practice balls scattered in the foreground were
        # winning the vote on busy ranges; they're meters away from the
        # actual address position and would mislead the side filter.
        near = [
            (c, cx, cy)
            for (c, cx, cy) in candidates
            if float(np.hypot(cx - clubhead[0], cy - clubhead[1])) <= MAX_BALL_TO_CLUBHEAD_PX
        ]
        if not near:
            return None
        near.sort(
            key=lambda x: float(np.hypot(x[1] - clubhead[0], x[2] - clubhead[1]))
        )
        return (near[0][1], near[0][2])
    candidates.sort(key=lambda x: -x[0])
    return (candidates[0][1], candidates[0][2])


def _hot_mask_body_components(hot_mask):
    """Return (bbox, filled_mask) for the golfer's body envelope.

    Bbox is the rectangular bounding box of the body's largest
    connected component — used for the foot-line cutoff.
    Filled mask is a uint8 image of just that connected component
    drawn as a filled shape, used for *pixel-accurate* "is this point
    inside the body" tests. A vertical ball flight passing just to the
    left of the body's legs lives inside the rectangular bbox (the
    bbox includes empty space around the body shape) but OUTSIDE the
    filled mask. The filled-mask check correctly accepts that ball
    flight; the bbox check falsely rejected it.

    Returns (None, None) if no large enough component.
    """
    closed = cv2.morphologyEx(
        hot_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8),
    )
    contours, _h = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 200:
        return None, None
    x, y, w, h = cv2.boundingRect(largest)
    body_mask = np.zeros(hot_mask.shape, dtype=np.uint8)
    cv2.drawContours(body_mask, [largest], -1, 255, -1)
    return (int(x), int(y), int(x + w), int(y + h)), body_mask


def _hot_mask_body_bbox(hot_mask):
    """Bbox-only convenience wrapper around _hot_mask_body_components.
    Kept for callers that don't need the filled mask."""
    bbox, _mask = _hot_mask_body_components(hot_mask)
    return bbox


def _hot_mask_x_center(hot_mask):
    """X-center of the body bbox (back-compat wrapper)."""
    bbox = _hot_mask_body_bbox(hot_mask)
    if bbox is None:
        return None
    return float((bbox[0] + bbox[2]) / 2.0)


def _build_hot_mask(heatmap, counted_frames: int):
    """Threshold per-pixel hit counts into a binary 'always-moving' mask.

    A real ball touches any given pixel for 1-2 frames, so the threshold
    is set well above that floor. The body and tree corridors accumulate
    counts in the dozens, so they cross the threshold easily.

    Returned mask is uint8 {0,1}-valued at detection resolution, with a
    small dilation applied to grow a buffer around hot regions.
    """
    if counted_frames <= 0:
        return np.zeros(heatmap.shape, dtype=np.uint8)
    threshold = max(HOT_MASK_MIN_HITS, int(counted_frames * HOT_MASK_PCT))
    raw = (heatmap >= threshold).astype(np.uint8)
    if HOT_MASK_DILATE_PX > 0:
        k = HOT_MASK_DILATE_PX * 2 + 1
        raw = cv2.dilate(raw, np.ones((k, k), np.uint8))
    return raw


def _grab_frame(input_path: Path, frame_idx: int):
    """Re-open the video and read a specific frame at native resolution.

    Used to fetch the post-filter busiest frame for the debug image
    without keeping every frame in memory during the detection pass.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


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
                 all_detections, seed_detections, filmstrip, total_frames, fps,
                 n_raw=None, hot_mask=None, det_scale=1.0, native_size=None,
                 ball_addr_det=None, body_x_det=None, behind_side=None,
                 club_line_det=None, handedness=None,
                 body_bottom_det=None, picked_track=None,
                 ball_pos_native=None, ball_pos_source=None):
    """Diagnostic JPG composed of two parts stacked vertically.

    Top half = the busiest frame (most candidates seen) with:
      - a red translucent overlay of the hot mask (the regions excluded
        by the motion-density filter — body, club corridor, leaves)
      - a cyan vertical line at the body's x-center, with a tint over
        the half that got rejected by the side filter
      - a cyan ring at the ball's detected address position
      - tiny yellow dots for every POST-hot-mask detection across the
        clip
      - bright green dots for the candidates that survived the
        upward-streak prefilter — these should trace the actual ball arc
      - red circles around the busiest frame's surviving candidates

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
        if hot_mask is not None:
            target_size = native_size if native_size else (base.shape[1], base.shape[0])
            scaled = cv2.resize(
                hot_mask, target_size, interpolation=cv2.INTER_NEAREST,
            )
            tint = np.zeros_like(base)
            tint[..., 2] = 200  # red in BGR
            mask3 = scaled.astype(bool)
            base[mask3] = (0.55 * base[mask3] + 0.45 * tint[mask3]).astype(np.uint8)
        # Body x-center axis + side-filter shading. body_x_det is in
        # detection-coord px, so scale up to native frame width for the
        # overlay.
        if body_x_det is not None and det_scale > 0:
            body_x_native = int(body_x_det / det_scale)
            cv2.line(base, (body_x_native, 0), (body_x_native, base.shape[0]),
                     (255, 220, 0), 2, cv2.LINE_AA)
            if behind_side in ("left", "right"):
                shade = np.zeros_like(base)
                shade[..., 0] = 80  # blue tint on the rejected half
                if behind_side == "left":
                    shade[:, :body_x_native] = (60, 60, 60)
                    base[:, :body_x_native] = (
                        0.65 * base[:, :body_x_native] + 0.35 * shade[:, :body_x_native]
                    ).astype(np.uint8)
                else:
                    shade[:, body_x_native:] = (60, 60, 60)
                    base[:, body_x_native:] = (
                        0.65 * base[:, body_x_native:] + 0.35 * shade[:, body_x_native:]
                    ).astype(np.uint8)
        if ball_pos_native is not None:
            bx_native = int(ball_pos_native[0])
            by_native = int(ball_pos_native[1])
            if ball_pos_source == "address-vote":
                # Ball-at-rest directly detected — the most accurate
                # signal. Solid cyan + "ball" label.
                cv2.circle(base, (bx_native, by_native), 14, (255, 220, 0), 3, cv2.LINE_AA)
                cv2.circle(base, (bx_native, by_native), 4, (255, 220, 0), -1, cv2.LINE_AA)
                cv2.putText(
                    base, "ball", (bx_native + 18, by_native + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 2, cv2.LINE_AA,
                )
            else:
                # Inferred from first frame of the picked track —
                # approximate (might be already mid-flight). Smaller
                # ring + "ball?" label so the operator knows it's
                # a fallback estimate.
                cv2.circle(base, (bx_native, by_native), 10, (180, 180, 255), 2, cv2.LINE_AA)
                cv2.putText(
                    base, "ball?", (bx_native + 14, by_native + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 255), 1, cv2.LINE_AA,
                )
        if club_line_det is not None and det_scale > 0:
            cx1, cy1, cx2, cy2 = club_line_det
            cv2.line(
                base,
                (int(cx1 / det_scale), int(cy1 / det_scale)),
                (int(cx2 / det_scale), int(cy2 / det_scale)),
                (0, 200, 255), 3, cv2.LINE_AA,
            )
            tag = f"club: {handedness}" if handedness else "club"
            cv2.putText(
                base, tag,
                (int(cx2 / det_scale) + 10, int(cy2 / det_scale) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA,
            )
        # Foot-line cutoff: anything below this y was rejected as
        # divot debris / practice balls / shadow.
        if body_bottom_det is not None and det_scale > 0:
            foot_y_native = int((body_bottom_det + 15) / det_scale)
            cv2.line(
                base, (0, foot_y_native), (base.shape[1], foot_y_native),
                (255, 100, 200), 2, cv2.LINE_AA,
            )
        for d in all_detections:
            cv2.circle(base, (int(d.x), int(d.y)), 2, (0, 255, 255), -1, cv2.LINE_AA)
        # Surviving upward-streak seeds: bigger, brighter, opaque green.
        for d in seed_detections:
            cv2.circle(base, (int(d.x), int(d.y)), 4, (0, 255, 0), -1, cv2.LINE_AA)
        # Picked track (what the rendered tracer will follow). Drawn in
        # magenta so the operator can spot when the linker chose the
        # club arc or some other chain over the obvious ball flight.
        if picked_track:
            pts = [(int(d.x), int(d.y)) for d in sorted(picked_track, key=lambda p: p.frame)]
            for i in range(1, len(pts)):
                cv2.line(base, pts[i - 1], pts[i], (255, 0, 255), 3, cv2.LINE_AA)
            for px, py in pts:
                cv2.circle(base, (px, py), 5, (255, 0, 255), 2, cv2.LINE_AA)
        if busiest_cands:
            for cx, cy, r in busiest_cands:
                cv2.circle(base, (int(cx), int(cy)), max(int(r) + 2, 4), (0, 0, 255), 2)
        secs = total_frames / fps if fps > 0 else 0.0
        raw_str = f"raw: {n_raw}  |  " if n_raw is not None else ""
        if busiest_cands:
            msg = (
                f"clip: {total_frames}f / {secs:.1f}s  |  "
                f"busiest frame {busiest_idx}: {len(busiest_cands)} cands  |  "
                f"{raw_str}post-mask: {len(all_detections)}  |  "
                f"upward-streak: {len(seed_detections)}"
            )
        else:
            msg = (
                f"clip: {total_frames}f / {secs:.1f}s  |  "
                f"0 cands survived hot-mask  |  {raw_str}post-mask: 0  |  upward-streak: 0"
            )
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

    scored = []
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
        # Ball-likeness score: high circularity * smallness. A real ball
        # is small AND round; disc-shaped body fragments are round but
        # bigger. Used to rank when we cap the per-frame candidate set.
        smallness = 1.0 - (area / MAX_BALL_AREA)
        score = circ * smallness
        scored.append((score, float(cx), float(cy), float(radius)))

    if len(scored) > MAX_CANDIDATES_PER_FRAME:
        scored.sort(key=lambda s: -s[0])
        scored = scored[:MAX_CANDIDATES_PER_FRAME]
    return [(cx, cy, r) for (_score, cx, cy, r) in scored]


def _link(detections):
    """Velocity-aware chain builder. Each detection starts its own
    chain; at each step we rank next-frame candidates by distance from
    the *predicted* position (extrapolating the chain's current
    velocity), not from the current point.

    This matters around the impact splash: noise dots are spatially
    closer to the current point than the actual next ball position,
    but they're far from where the velocity says the ball should be.
    Velocity-aware ranking keeps the chain on the real trajectory.

    First step has no velocity yet → falls back to current position
    (same as plain nearest-neighbor). Velocity is exponentially
    smoothed across steps to absorb single-frame jitter.

    Chains are allowed to overlap; _pick_best scores them by
    residual − 0.5·len, so the longest, cleanest chain wins.
    """
    by_frame: dict[int, list[_Det]] = {}
    for d in detections:
        by_frame.setdefault(d.frame, []).append(d)
    frames = sorted(by_frame)
    if not frames:
        return []
    trajectories: list[list[_Det]] = []
    for sf in frames:
        for si, start in enumerate(by_frame[sf]):
            track = [start]
            track_used = {(sf, si)}
            cur = start
            vel: tuple[float, float] | None = None  # (dx, dy) per frame
            for nf in frames:
                if nf <= cur.frame:
                    continue
                gap = nf - cur.frame
                if gap > MAX_FRAME_GAP:
                    break
                if vel is not None:
                    pred_x = cur.x + vel[0] * gap
                    pred_y = cur.y + vel[1] * gap
                else:
                    pred_x, pred_y = cur.x, cur.y
                best = None
                best_dist = float("inf")
                best_idx = -1
                for ci, cand in enumerate(by_frame[nf]):
                    if (nf, ci) in track_used:
                        continue
                    # Hard spatial constraint relative to current point
                    # (the ball physically can't teleport across the
                    # frame regardless of what velocity predicts).
                    spatial = float(np.hypot(cand.x - cur.x, cand.y - cur.y))
                    if spatial > MAX_PIXEL_JUMP_PER_FRAME * gap:
                        continue
                    # Rank by distance from predicted position.
                    pdist = float(np.hypot(cand.x - pred_x, cand.y - pred_y))
                    if pdist < best_dist:
                        best = cand
                        best_dist = pdist
                        best_idx = ci
                if best is None:
                    continue
                actual_dx = (best.x - cur.x) / gap
                actual_dy = (best.y - cur.y) / gap
                if vel is None:
                    vel = (actual_dx, actual_dy)
                else:
                    # Exponential smoothing: 60% new, 40% prior.
                    vel = (
                        0.6 * actual_dx + 0.4 * vel[0],
                        0.6 * actual_dy + 0.4 * vel[1],
                    )
                track.append(best)
                track_used.add((nf, best_idx))
                cur = best
            if len(track) >= MIN_TRACK_LENGTH:
                trajectories.append(track)
    return trajectories


def _residual(track):
    """RMS deviation of the track from a frame-parameterized motion
    model: y is quadratic in frame (gravity), x is linear in frame.

    Previously this fit y = f(x) as a quadratic, which returns infinity
    for near-vertical trajectories — exactly the shape a tee-cam
    captures for a straight shot. _pick_best would then reject the real
    ball flight column and grab a noisier chain with more lateral
    spread instead. Parameterizing by frame avoids the issue: a
    vertical chain has nearly-constant x and a clean y(t) parabola
    both fit perfectly.
    """
    if len(track) < 3:
        return float("inf")
    frames = np.array([p.frame for p in track], dtype=np.float64)
    xs = np.array([p.x for p in track], dtype=np.float64)
    ys = np.array([p.y for p in track], dtype=np.float64)
    if float(frames.max() - frames.min()) < 2:
        return float("inf")
    x_coef = np.polyfit(frames, xs, 1)
    y_coef = np.polyfit(frames, ys, 2)
    x_pred = np.polyval(x_coef, frames)
    y_pred = np.polyval(y_coef, frames)
    return float(np.sqrt(np.mean((xs - x_pred) ** 2 + (ys - y_pred) ** 2)))


def _fit_motion(track):
    """Fit a frame-parameterized motion model to a track. Returns
    (y_coef, x_coef, rms) or (None, None, None) if the frame range is
    too narrow to fit. y is quadratic in frame (gravity), x is linear
    (the ball mostly travels in a straight horizontal line in image
    coords — sliced/drawn balls violate this, but for v0 it's fine).
    """
    if len(track) < 3:
        return None, None, None
    frames = np.array([d.frame for d in track], dtype=np.float64)
    xs = np.array([d.x for d in track], dtype=np.float64)
    ys = np.array([d.y for d in track], dtype=np.float64)
    if float(frames.max() - frames.min()) < 4:
        return None, None, None
    y_coef = np.polyfit(frames, ys, 2)
    x_coef = np.polyfit(frames, xs, 1)
    xs_pred = np.polyval(x_coef, frames)
    ys_pred = np.polyval(y_coef, frames)
    rms = float(np.sqrt(np.mean((xs - xs_pred) ** 2 + (ys - ys_pred) ** 2)))
    return y_coef, x_coef, rms


def _extend_track(seed_track, all_detections, total_frames, frame_w=None, frame_h=None):
    """Extend a confident seed track (rising arc) with apex + descent
    detections from `all_detections`, using *incremental refitting*.

    For each frame past the seed range (and a few frames before), predict
    (px, py) from the current motion fit, find the closest in-frame
    detection, accept it if within an eps that grows with extrapolation
    distance, and refit. The refit step is the key — a fit on rising-only
    seeds has uncertain curvature, but every apex-region point we add
    sharpens the parabola and improves subsequent predictions.

    Stops extending in either direction once the predicted position
    leaves the frame (with a margin), since there can't be a real
    ball detection at impossible coordinates.
    """
    if len(seed_track) < 5:
        return list(seed_track)
    y_coef, x_coef, rms = _fit_motion(seed_track)
    if y_coef is None:
        return list(seed_track)

    seed_first = min(d.frame for d in seed_track)
    seed_last = max(d.frame for d in seed_track)

    seed_frames = {int(d.frame) for d in seed_track}
    by_frame: dict[int, list[_Det]] = {}
    for d in all_detections:
        f = int(d.frame)
        if f in seed_frames:
            continue
        by_frame.setdefault(f, []).append(d)

    track = list(seed_track)
    m = EXTEND_OUT_OF_FRAME_MARGIN_PX

    def _try_extend(frame_iter, anchor_frame):
        """Walk frame_iter, predicting from the current fit, adding any
        detection within an eps that grows with |frame - anchor_frame|
        (capped). Stops after EXTEND_MAX_CONSECUTIVE_MISS misses or when
        the predicted point leaves the frame."""
        nonlocal y_coef, x_coef, rms
        consecutive_miss = 0
        for f in frame_iter:
            distance = abs(f - anchor_frame)
            growth = min(EXTEND_EPS_GROWTH_PER_FRAME * distance, EXTEND_EPS_MAX_GROWTH)
            local_eps = max(EXTEND_EPS_MIN_PX, EXTEND_EPS_MULT * rms) * (1.0 + growth)
            py = float(np.polyval(y_coef, f))
            px = float(np.polyval(x_coef, f))
            if frame_w is not None and frame_h is not None and (
                px < -m or px > frame_w + m or py < -m or py > frame_h + m
            ):
                break
            cands = by_frame.get(f, [])
            if not cands:
                consecutive_miss += 1
                if consecutive_miss > EXTEND_MAX_CONSECUTIVE_MISS:
                    break
                continue
            best = None
            best_err = float("inf")
            for d in cands:
                err = float(np.hypot(d.x - px, d.y - py))
                if err < best_err:
                    best_err = err
                    best = d
            if best is not None and best_err <= local_eps:
                track.append(best)
                consecutive_miss = 0
                new_y, new_x, new_rms = _fit_motion(track)
                if new_y is not None:
                    y_coef, x_coef, rms = new_y, new_x, new_rms
            else:
                consecutive_miss += 1
                if consecutive_miss > EXTEND_MAX_CONSECUTIVE_MISS:
                    break

    _try_extend(range(seed_last + 1, total_frames + 1), seed_last)
    _try_extend(
        range(seed_first - 1, max(-1, seed_first - 1 - EXTEND_BACKWARD_FRAMES), -1),
        seed_first,
    )

    track.sort(key=lambda d: d.frame)
    return track


def _smooth_track_for_render(track, ball_pos_native=None):
    """Return a list of integer-frame (frame, x, y) points sampled from
    the motion fit of `track`, used in place of raw detection points
    when rendering the dashed tracer line.

    Fit is on the picked track only. When `ball_pos_native` is provided,
    we extrapolate the fitted parabola backward until y reaches the
    ball's at-rest y, and render from that frame instead of from
    track[0]. This way the natural parabolic motion handles the smooth
    transition from ball-at-rest to picked-track — no separate bridge
    segment, no kink.

    The very first rendered point is pinned to the exact ball coords
    so the line literally starts at the cyan ring (the parabola's
    extrapolated x at the same y is usually 5-15px off from ball_x,
    not enough to cause a visible kink at the pin).

    Truncates at apex (vertex of y(frame) parabola) so descent isn't
    rendered.
    """
    if len(track) < 3:
        return [(int(d.frame), int(d.x), int(d.y)) for d in track]
    y_coef, x_coef, _rms = _fit_motion(track)
    if y_coef is None:
        return [(int(d.frame), int(d.x), int(d.y)) for d in track]
    f_min = min(int(d.frame) for d in track)
    f_max = max(int(d.frame) for d in track)
    a, b, c = float(y_coef[0]), float(y_coef[1]), float(y_coef[2])

    # Extrapolate the fitted parabola backward to find the frame at
    # which y equals the ball-at-rest y. That's where the line should
    # start: the natural continuation of the parabola hits the ball's
    # actual position, no need for an artificial bridge.
    if ball_pos_native is not None and a > 1e-9:
        ball_y = float(ball_pos_native[1])
        # Solve a*f² + b*f + (c - ball_y) = 0 for f. The earlier (smaller)
        # root is where the ball's flight started; the later root is the
        # symmetric point on the descent which we don't care about.
        disc = b * b - 4 * a * (c - ball_y)
        if disc >= 0:
            sqrtd = disc ** 0.5
            f_root1 = (-b - sqrtd) / (2.0 * a)
            f_root2 = (-b + sqrtd) / (2.0 * a)
            f_at_ball_y = min(f_root1, f_root2)
            # Cap how far backward we'll extrapolate. If the parabola
            # would say the ball was at rest 50 frames ago, we don't
            # want a 50-frame straight line — just clamp to a sane
            # bridge length.
            extrapolation_cap_frames = 8
            f_min = max(int(round(f_at_ball_y)), f_min - extrapolation_cap_frames)

    # Apex truncation (no descent rendered).
    if a > 1e-9:
        vertex_frame = -b / (2.0 * a)
        if f_min < vertex_frame < f_max:
            f_max = int(round(vertex_frame))

    out = []
    for f in range(f_min, f_max + 1):
        if f == f_min and ball_pos_native is not None:
            # Pin the very first point to the exact ball coords. The
            # parabola's predicted x here is usually within a few px
            # of ball_x; pinning eliminates that small offset so the
            # tracer literally starts at the cyan ball ring.
            out.append((f, int(ball_pos_native[0]), int(ball_pos_native[1])))
        else:
            out.append((f, int(np.polyval(x_coef, f)), int(np.polyval(y_coef, f))))
    return out


def _count_aligned_detections(track, all_detections, eps=20.0):
    """Count post-mask detections (excluding those already in `track`)
    that fall within `eps` of the parabolic motion fit of `track` at
    their own frame. This is the "green-before-yellow" signature: a
    real ball-flight chain's parabola also predicts the yellow
    (post-mask but non-upward-streak) detections that follow it past
    the apex / wherever the upward-streak filter cut out. The club's
    follow-through arc doesn't, because its motion isn't sustained
    parabolic flight.
    """
    if len(track) < 3 or not all_detections:
        return 0
    y_coef, x_coef, _rms = _fit_motion(track)
    if y_coef is None:
        return 0
    track_frame_set = {int(d.frame) for d in track}
    aligned = 0
    for d in all_detections:
        if int(d.frame) in track_frame_set:
            continue
        py = float(np.polyval(y_coef, d.frame))
        px = float(np.polyval(x_coef, d.frame))
        if abs(d.y - py) <= eps and abs(d.x - px) <= eps:
            aligned += 1
    return aligned


def _pick_best(trajectories, ball_addr_native=None, body_mask_det=None,
               det_scale=1.0, all_detections=None):
    """Pick the most likely ball-flight trajectory. Lower score = better.

    Score = residual − 1.0·len + body_pct·30 + anchor − 0.3·aligned_count.

    The `aligned_count` bonus captures the "green-before-yellow" pattern:
    a real ball-flight chain (upward-streak seeds) extends along the
    same parabola the post-apex / decelerating yellow detections sit
    on. Club motion chains don't — they're brief arcs whose parabola
    fit doesn't predict additional sustained-motion detections.

    Hard reject tracks whose points sit mostly inside the golfer's body
    silhouette mask. Uses pixel-accurate filled body mask (not bbox)
    so vertical ball flights alongside the body but inside its bbox
    aren't falsely rejected.
    """
    scored = []
    rejected_body = 0
    rejected_residual = 0
    rejected_span = 0
    if body_mask_det is not None:
        bm_h, bm_w = body_mask_det.shape[:2]
    for t in trajectories:
        r = _residual(t)
        if r > MAX_PARABOLA_RESIDUAL:
            rejected_residual += 1
            continue
        xs = [p.x for p in t]
        ys = [p.y for p in t]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        if span < MIN_TRAJECTORY_SPAN_PX:
            rejected_span += 1
            continue
        # Reject body-resident tracks before they can win the score.
        # Pixel-accurate against the filled body silhouette (not bbox).
        inside_pct = 0.0
        if body_mask_det is not None:
            inside_count = 0
            for p in t:
                ix = int(p.x * det_scale)
                iy = int(p.y * det_scale)
                if 0 <= iy < bm_h and 0 <= ix < bm_w and body_mask_det[iy, ix] > 0:
                    inside_count += 1
            inside_pct = inside_count / len(t)
            if inside_pct > 0.9:
                rejected_body += 1
                continue
        score = r - 1.0 * len(t) + inside_pct * 30.0
        # Parabola-alignment bonus: this chain's fit predicts the
        # positions of many post-mask detections. Strong ball-flight
        # signal because the apex/descent dots that the upward-streak
        # filter dropped (the yellows) still sit on the same parabola.
        if all_detections is not None:
            aligned = _count_aligned_detections(t, all_detections, eps=20.0)
            score -= 0.7 * aligned
        if ball_addr_native is not None:
            first = t[0]
            d = float(np.hypot(first.x - ball_addr_native[0], first.y - ball_addr_native[1]))
            if d < 50:
                # Strong anchor when the chain literally starts at the
                # ball. Ball flight starts at the ball; nothing else
                # plausibly does.
                score -= 25.0
            elif d < 100:
                score -= 10.0
            elif d > 200:
                score += 5.0
        scored.append((score, t))
    if not scored:
        log.info(
            "tracer: _pick_best: NO trajectory survived. of %d input chains: "
            "rejected %d on residual>%.0f, %d on span<%d, %d on body-overlap>0.8",
            len(trajectories), rejected_residual, MAX_PARABOLA_RESIDUAL,
            rejected_span, int(MIN_TRAJECTORY_SPAN_PX), rejected_body,
        )
        return []
    scored.sort(key=lambda s: s[0])
    log.info(
        "tracer: _pick_best top picks (n=%d, body-rejected=%d): %s",
        len(scored), rejected_body,
        ", ".join(
            f"len={len(t)} r={_residual(t):.1f} score={s:.1f}"
            for s, t in scored[:3]
        ),
    )
    return scored[0][1]


def _draw_dashed(img, points):
    if len(points) < 2:
        return
    cv2.polylines(img, [np.array(points, dtype=np.int32)], False, TRACER_HALO_COLOR, TRACER_THICKNESS + 4, cv2.LINE_AA)
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


def _find_upward_seeds(detections, min_dy_per_frame=None):
    """Subset of `detections` that participate in a temporally-consistent
    upward chain of length >= MIN_UPWARD_CHAIN_LEN.

    `min_dy_per_frame` is the upward-motion threshold per frame; the
    caller should scale it with fps (a ball at 30fps moves 4× as far per
    frame as the same ball at 120fps), otherwise high-fps clips lose
    almost the entire ball trajectory near the apex where per-frame dy
    drops below the threshold. Defaults to the module constant when
    not provided.

    Algorithm: DP in both directions. fwd[d] = longest upward chain
    starting at d going forward; bwd[d] = longest ending at d coming
    from earlier frames. A detection is kept iff fwd[d] + bwd[d] - 1 >=
    MIN_UPWARD_CHAIN_LEN — i.e., d sits somewhere on a chain that long.

    "Upward" means y_next < y_cur - min_dy_per_frame * frame_gap
    (image coords: smaller y = higher on screen). Spatial gate reuses
    MAX_PIXEL_JUMP_PER_FRAME so we don't link blobs across the frame.
    """
    if min_dy_per_frame is None:
        min_dy_per_frame = MIN_UPWARD_DY_PER_FRAME
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
                    if d.y - nd.y < min_dy_per_frame * df:
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
                    if pd.y - d.y < min_dy_per_frame * df:
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
