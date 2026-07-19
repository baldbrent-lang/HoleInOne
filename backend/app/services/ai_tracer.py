"""AI-powered golf analysis. Step 2: find the address frame.

Camera is always behind the golfer; golfer hits toward a target away
from the camera. This module's only job (for now) is to identify the
single frame closest to ADDRESS — the golfer standing over the ball,
club at rest behind/next to the ball, just before takeaway begins.

Future steps will layer in ball-at-rest detection, impact frame, ball
tracking, and the dashed-overlay render — each verified before moving
on. The classical CV tracer in services/tracer.py is independent of
this module.
"""
from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger("golfreelz.ai_tracer")

try:
    import cv2  # type: ignore
    HAS_CV = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    HAS_CV = False

try:
    import numpy as np  # type: ignore
    HAS_NP = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    HAS_NP = False

try:
    from anthropic import Anthropic  # type: ignore
    HAS_ANTHROPIC = True
except Exception:  # pragma: no cover
    Anthropic = None  # type: ignore
    HAS_ANTHROPIC = False


# Per the claude-api skill default. Override via env (TRACER_AI_MODEL)
# if a cheaper or different vision model is preferred — e.g. set to
# claude-haiku-4-5 for a ~5x cost reduction on this kind of routine
# vision task.
MODEL = os.environ.get("TRACER_AI_MODEL", "claude-opus-4-7")

# Number of frames sampled across the clip and sent to Claude in one
# multi-image API call. 12 gives Claude a clear timeline view of the
# whole swing while keeping per-request token usage bounded; with
# typical clips of 3-10 seconds that's ~one sample every 250-800 ms,
# dense enough to catch address even if it's brief.
N_FRAMES = 12

# Per-frame width sent to Claude for general use (impact/address pass).
# Anthropic auto-resizes images above ~1568 px on the long edge anyway;
# 640 px gives Claude clear visibility of the golfer's stance and club
# position without bloating payload.
FRAME_W = 640

# Per-frame width sent for the dedicated handedness call. Higher than
# FRAME_W because handedness reads off a very specific feature — the
# club shaft, which is just a few pixels wide. Extra resolution makes
# the shaft direction much easier for Claude to call.
HANDEDNESS_FRAME_W = 1024

# Impact-finder parameters. After address we sample N_IMPACT_CANDIDATES
# frames evenly across IMPACT_WINDOW_SECONDS of clip time. 2.0 s covers
# takeaway → top → downswing → impact for a normal-tempo swing across
# the typical frame rates (30 / 60 / 120 fps). The address frame is
# sent alongside as a REFERENCE, and each candidate has the ball's
# starting position drawn on it as a blue circle so Claude can pick
# the frame whose club shaft is closest to pointing at that circle.
N_IMPACT_CANDIDATES = 12
IMPACT_WINDOW_SECONDS = 2.0

HANDEDNESS_FROM_ADDRESS_PROMPT = (
    "You are looking at a single still frame of a golfer at ADDRESS — "
    "set up over the ball, just before takeaway. The camera is "
    "positioned BEHIND the target line, and the golfer is hitting "
    "AWAY from the camera (target is down the image, beyond the "
    "golfer).\n\n"
    "Your job: locate the golfer's HANDS and the GOLF BALL on the "
    "ground, then determine whether the golfer is RIGHT-handed or "
    "LEFT-handed from the shaft direction between them.\n\n"
    "Reason in this exact order — do NOT skip steps:\n"
    "Step 1. Locate the golfer's HANDS gripping the club (the upper "
    "end of the shaft). Note their approximate (x, y) pixel "
    "coordinates in the image. (Image dimensions are given in the "
    "user message; x=0 is left, y=0 is top.)\n"
    "Step 2. Locate the GOLF BALL on the ground. The ball is whatever "
    "the club shaft is pointing toward — at address the clubhead "
    "rests right against the ball, so the ball is at (or within a "
    "few pixels of) the lower end of the shaft. Report the ball's "
    "approximate (x, y) pixel coordinates. (If the clubhead obscures "
    "the ball, just use the clubhead position; that's the ball "
    "location for our purposes.)\n"
    "Step 3. Compare: ball_x vs hands_x.\n"
    "  - If ball_x < hands_x (ball is to the LEFT of the hands in "
    "the image), the shaft points down-and-LEFT, and the golfer is "
    "LEFT-handed.\n"
    "  - If ball_x > hands_x (ball is to the RIGHT of the hands in "
    "the image), the shaft points down-and-RIGHT, and the golfer is "
    "RIGHT-handed.\n"
    "This rule holds because the camera sits behind the target line. "
    "A right-handed golfer's ball rests in front of his lead (left) "
    "foot on his target side; from a behind-the-line camera that "
    "target side appears on the image-RIGHT, so the ball is RIGHT of "
    "the hands and the shaft points down-and-right. Left-handed is "
    "the mirror.\n\n"
    "Apply the comparison literally. Do NOT second-guess it with "
    "ball-position reasoning or assumptions about typical setups. "
    "The ball_x-vs-hands_x comparison IS the answer.\n\n"
    "Reply with ONE JSON object and nothing else:\n"
    "{\n"
    '  "hands_x": <int approximate pixel x of the hands>,\n'
    '  "hands_y": <int approximate pixel y of the hands>,\n'
    '  "ball_x": <int approximate pixel x of the ball on the ground>,\n'
    '  "ball_y": <int approximate pixel y of the ball on the ground>,\n'
    '  "shaft_direction": "down_left" | "down_right" | "vertical",\n'
    '  "handedness": "right" | "left" | "unknown",\n'
    '  "confidence": "high" | "medium" | "low",\n'
    '  "notes": "<≤25 word description of the two landmarks and the resulting shaft direction>"\n'
    "}\n"
    "Use 'unknown' only if the hands or ball are genuinely not "
    "visible enough to estimate their pixel coordinates."
)


ADDRESS_SYSTEM_PROMPT = (
    "You are analyzing still frames from a golf swing video. The camera "
    "is positioned BEHIND the golfer; the golfer hits toward a target "
    "that is away from the camera (down the page). Frames are presented "
    "in chronological order with their frame numbers shown in a text "
    "block immediately before each image.\n\n"
    "Your job: pick the SINGLE frame that best shows the golfer at "
    "ADDRESS — the moment he is set up over the ball, club resting "
    "behind or next to the ball, just before he starts his takeaway "
    "(backswing). Cues for address:\n"
    "- Golfer is stationary, feet planted, bent at the hips over the ball.\n"
    "- Club is at the ball or just behind it (NOT raised, NOT mid-swing).\n"
    "- Hands are at waist or thigh level, not lifted.\n"
    "- The shoulders / hips have not yet rotated for the swing.\n\n"
    "Do NOT pick:\n"
    "- A frame where the club is already moving (any part of backswing, "
    "downswing, impact, or follow-through).\n"
    "- A frame where the golfer is still walking up or hasn't set his "
    "stance yet.\n"
    "- A frame after the ball has been struck.\n\n"
    "If multiple frames look like address (the golfer may waggle the "
    "club or hold the address position for a moment), pick the LAST "
    "one immediately before the club starts moving back.\n\n"
    "Reply with ONE JSON object and nothing else:\n"
    "{\n"
    '  "address_frame": <int — must equal one of the labeled frame numbers>,\n'
    '  "confidence": "high" | "medium" | "low",\n'
    '  "notes": "<≤25 word reasoning citing why this frame over the others>"\n'
    "}\n"
    "Never invent a frame number; pick from the labels you were shown."
)


REFINE_IMPACT_FRAME_W = 768

# Ball-tracking pass parameters. After impact we step through every
# frame and ask Claude to locate the ball, until it leaves the view.
# Each frame is sent at BALL_TRACK_FRAME_W wide; 1568 is Anthropic's
# vision tile threshold — above that, images get internally resized
# without benefit, so we sit right at the sweet spot. Parallel calls
# keep wall time manageable across the typical 30-60 frame flight.
# Width we send the ROI crop at on the per-frame ball-track call.
# 1568 is Anthropic's vision-tile threshold — above that, images get
# internally resampled down to ~1568 anyway, so we were paying full
# vision-token tariff for upload but Claude saw the same pixels. Sit
# at the threshold to keep ball footprint maxed out without paying
# for resampled-away resolution.
BALL_TRACK_FRAME_W = 1568

# Adaptive contrast enhancement (CLAHE on the L channel of LAB) is
# applied to the cropped frame BEFORE it's JPEG-encoded for the API
# whenever the crop's grayscale standard deviation is below
# CLAHE_CONTRAST_THRESHOLD. This rescues ball detection on overcast
# / flat-lit clips where the white ball sits at near-equal luminance
# to the grey sky. Skipped on high-contrast scenes (bright day with
# blue sky) where boosting contrast would just amplify false-positive
# whites like clouds and flags.
CLAHE_CONTRAST_THRESHOLD = 45.0
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8

# Default frame budget by source fps:
#   <50 fps : 20 consecutive frames (~0.4-0.7 s of flight at 30/60 fps)
#   50-100  : 40 consecutive frames (still every-frame, twice the
#             budget to cover the same wall-clock duration)
#   >100    : 40 frames sampled with STRIDE = 5 (every fifth frame,
#             200-frame span). Cost stays at 40 Claude calls; the
#             ball barely moves between consecutive frames at 120 fps
#             so sampling every other frame is wasteful — wider stride
#             covers ~1.7s of flight (vs ~0.7s in the lower buckets)
#             with the same API budget.
BALL_TRACK_MAX_FRAMES = 12
BALL_TRACK_MAX_FRAMES_HIGH_FPS = 40
BALL_TRACK_HIGH_FPS_THRESHOLD = 50.0
BALL_TRACK_VHIGH_FPS_THRESHOLD = 100.0
BALL_TRACK_VHIGH_FPS_STRIDE = 5
# <50 fps: 12 points every THIRD frame → 36-frame span (~1.2s at 30fps).
# Operator call: same 12 AI calls, half again more flight covered — the
# per-frame gaps get filled by the CV chain / fitted curve anyway.
BALL_TRACK_LOW_FPS_STRIDE = 3
BALL_TRACK_CONCURRENCY = 8

# Phase-2 retry sends a crop. Crop size is in NATIVE pixels (how much
# of the source frame around the prediction we hand to Claude); send
# size is the upscaled width we encode to JPEG before sending. The
# upscale turns a ~600 px region into a ~1024 px image — same content,
# more vision tiles, ball appears with ~1.7× more pixels per side.
BALL_TRACK_CROP_NATIVE_SIZE = 600
BALL_TRACK_CROP_SEND_W = 1568

# Phase-3 refinement: for every Phase 1/2 hit, do one more pass with
# a tighter crop centered on the reported position. Catches the
# "in the vicinity of the ball but not on it" failure mode where
# Claude found the right region but the click landed a few pixels
# off the actual speck. Crop is ~2.5× tighter than Phase 2, so the
# ball occupies a much larger share of the upscaled image — easier
# to pinpoint the exact pixel.
BALL_TRACK_REFINE_CROP_NATIVE_SIZE = 250
BALL_TRACK_REFINE_SEND_W = 1568
# Cap how far the refined position can move from the original. A
# big jump usually means Claude latched onto a distractor (cloud
# fragment, range ball on the ground); we'd rather keep the
# slightly-off Phase 1 hit than swap it for a confidently-wrong
# refinement.
BALL_TRACK_REFINE_MAX_DELTA_PX = 80
# Toggle: disable to skip Phase 3 entirely if API spend matters
# more than per-frame pixel precision.
BALL_TRACK_REFINEMENT_ENABLED = True

BALL_TRACK_REFINE_PROMPT = (
    "You are looking at a tightly zoomed crop of a single frame from "
    "a golf swing video. An earlier pass located what appears to be "
    "the airborne golf ball within this region. Your job: find the "
    "EXACT pixel position of the ball. The earlier identification "
    "may have been a few pixels off — your job is to refine it.\n\n"
    "Cues:\n"
    "- A small bright white spot OR (against bright sky) a small "
    "darker silhouette.\n"
    "- 5-15 pixels across after the crop's upscale.\n"
    "- May have slight motion blur in the direction of flight.\n"
    "- The ball is usually near the center of this crop (the earlier "
    "pass identified it from this region) but may not be exactly "
    "centered.\n\n"
    "If you can clearly see the airborne ball, return its exact "
    "(x, y) pixel coordinates in THIS image's coordinate system "
    "(top-left = 0,0). If you cannot find a ball in this crop, set "
    "found=false.\n\n"
    "Reply with ONE JSON object only:\n"
    "{\n"
    '  "found": true | false,\n'
    '  "x": <int>,\n'
    '  "y": <int>,\n'
    '  "confidence": "high" | "medium" | "low",\n'
    '  "notes": "<≤15 word description>"\n'
    "}"
)


BALL_TRACK_PROMPT = (
    "You are looking at a single still frame from a golf swing video. "
    "The ball was struck a fraction of a second ago and is now in "
    "flight, travelling AWAY from the camera and arcing upward toward "
    "the target.\n\n"
    "Your only job: locate the airborne golf BALL in this frame.\n\n"
    "Cues:\n"
    "- A small bright white spot (3-15 pixels across) or a short "
    "motion-blur streak in the direction of motion.\n"
    "- Against grass: white-on-green. Against sky: bright speck OR "
    "(overcast) a small darker silhouette.\n"
    "- The ball is rising as it flies, so it usually sits in the "
    "upper portion of the frame above the tee.\n"
    "- DO NOT confuse with: shoes, belt, hands, club head, range "
    "balls on the ground at the tee, leaves, flags, course markers, "
    "or clouds.\n\n"
    "If you cannot clearly see the airborne ball in this frame (it "
    "has left the frame, gone behind something, or is too small to "
    "pick out reliably), set found=false.\n\n"
    "Reply with ONE JSON object only:\n"
    "{\n"
    '  "found": true | false,\n'
    '  "x": <int pixel x, or null>,\n'
    '  "y": <int pixel y, or null>,\n'
    '  "confidence": "high" | "medium" | "low",\n'
    '  "notes": "<≤15 word description>"\n'
    "}"
)


REFINE_IMPACT_SYSTEM_PROMPT = (
    "You are looking at a tight cluster of candidate frames near the "
    "moment of IMPACT in a golf swing. The camera is behind the target "
    "line; the golfer is hitting AWAY from the camera. Frames are in "
    "chronological order, each preceded by a 'Frame N:' text block.\n\n"
    "Every candidate has a BLUE CIRCLE drawn on it at the ball's "
    "STARTING POSITION — the same pixel on every frame, so you can "
    "directly compare each candidate's clubhead to the circle.\n\n"
    "Two things to do in your reply:\n"
    "  1. Pick the SINGLE candidate whose CLUB SHAFT is closest to "
    "pointing at the blue circle — i.e., the clubhead is back at the "
    "ball (the bottom end of the shaft is on or right at the circle).\n"
    "  2. For that picked frame ONLY, report the (x, y) pixel "
    "coordinates of the golfer's HANDS (upper end of the shaft) and "
    "the CLUBHEAD (lower end of the shaft).\n\n"
    "Earlier candidates typically still have the clubhead above the "
    "ball (descending). Later candidates have the clubhead past the "
    "ball (follow-through). The winner is the transition point — "
    "clubhead on or nearest the circle. If two look equally close, "
    "pick the EARLIER one (true impact is the last pre-ball-leaving "
    "frame).\n\n"
    "Reply with ONE JSON object and nothing else:\n"
    "{\n"
    '  "impact_frame": <int — must equal one of the labeled candidate frame numbers>,\n'
    '  "hands_x": <int pixel x of the hands on the picked frame>,\n'
    '  "hands_y": <int pixel y of the hands on the picked frame>,\n'
    '  "clubhead_x": <int pixel x of the clubhead on the picked frame>,\n'
    '  "clubhead_y": <int pixel y of the clubhead on the picked frame>,\n'
    '  "confidence": "high" | "medium" | "low",\n'
    '  "notes": "<≤25 word reasoning about clubhead position relative to the blue circle>"\n'
    "}\n"
    "All (x, y) coordinates are in the IMAGE coordinate system of the "
    "frame you were shown (image dimensions are stated in the user "
    "message). Never invent a frame number; pick from the candidates "
    "shown."
)


IMPACT_SYSTEM_PROMPT = (
    "You are analyzing frames from a golf swing video. The camera is "
    "behind the target line and the golfer is hitting AWAY from the "
    "camera.\n\n"
    "The first image is a REFERENCE showing the golfer at ADDRESS, "
    "set up over the ball, just before takeaway. After the reference "
    "you will see a series of candidate frames from AFTER address, in "
    "chronological order, each preceded by a 'Frame N:' text block.\n\n"
    "Every candidate frame (and the reference) also has a BLUE CIRCLE "
    "drawn on it marking the ball's STARTING POSITION — the exact "
    "pixel where the ball was sitting at address. The blue circle is "
    "in the same image location on every frame, so you can directly "
    "compare each candidate's clubhead to the circle.\n\n"
    "Your job: pick the SINGLE candidate frame closest to IMPACT — "
    "the moment the clubhead has returned to the ball after the "
    "backswing and is just about to or just making contact.\n\n"
    "The swing progression is:\n"
    "  address (reference) → takeaway (club moves back) → top of "
    "backswing (club raised behind/above golfer) → downswing (club "
    "swinging back toward ball) → IMPACT → follow-through.\n\n"
    "IMPACT cues:\n"
    "- The clubhead is back AT the blue circle (or as close to it as "
    "any candidate shows).\n"
    "- The club shaft runs from the hands DOWN to the blue circle, "
    "roughly vertical or just past vertical — NOT raised behind the "
    "golfer, NOT high in the downswing.\n"
    "- The golfer's arms are extended down toward the ball; hips have "
    "rotated through.\n"
    "- The ball is either still visible at the blue circle or JUST "
    "struck (small motion blur near the circle).\n\n"
    "Do NOT pick:\n"
    "- A backswing frame (club raised behind/above the golfer).\n"
    "- An early downswing frame (club still high, clubhead well above "
    "the blue circle).\n"
    "- A post-impact frame where the ball is clearly gone and the "
    "clubhead has swung well past the blue circle.\n"
    "- A follow-through frame (club continuing across the golfer's "
    "body, clubhead nowhere near the circle).\n\n"
    "If the actual impact moment falls BETWEEN two candidates (the "
    "earlier one is mid-downswing and the next is already follow-"
    "through), pick the EARLIER of the two — true impact is the last "
    "pre-ball-leaving frame.\n\n"
    "Reply with ONE JSON object and nothing else:\n"
    "{\n"
    '  "impact_frame": <int — must equal one of the labeled candidate frame numbers>,\n'
    '  "confidence": "high" | "medium" | "low",\n'
    '  "notes": "<≤25 word reasoning citing where the clubhead is relative to the blue circle>"\n'
    "}\n"
    "Never invent a frame number; pick from the candidates shown."
)


def have_ai_tracer() -> bool:
    """True when the AI module can actually run end-to-end."""
    return HAS_CV and HAS_ANTHROPIC and bool(os.environ.get("ANTHROPIC_API_KEY"))


# The Anthropic SDK retries on 5xx by default but only twice, which
# isn't enough when the API is genuinely overloaded (HTTP 529). Six
# retries with the SDK's exponential backoff covers ~30 s of wait
# time, which rides out most transient overload spikes without making
# the pipeline feel hung.
ANTHROPIC_MAX_RETRIES = 6

# Models the per-run override is allowed to pick from. Lets the
# operator A/B Opus vs Haiku on the same clip without restarting the
# backend; anything outside this set falls back to the env-driven
# MODEL default.
SUPPORTED_MODELS = {"claude-opus-4-7", "claude-haiku-4-5"}


def _anthropic_client():
    """Construct an Anthropic client with our retry policy. Returns None
    if the SDK isn't installed (caller is expected to check have_ai_tracer
    first; this is a defensive fallback)."""
    if not HAS_ANTHROPIC:
        return None
    return Anthropic(max_retries=ANTHROPIC_MAX_RETRIES)


def _resolve_model(model_override: str | None = None) -> str:
    """Return the model id to use for an AI call. A non-empty
    `model_override` wins; anything not in SUPPORTED_MODELS is rejected
    so a typo doesn't silently route to the wrong endpoint. Falls back
    to the module-level MODEL default."""
    if model_override:
        if model_override in SUPPORTED_MODELS:
            return model_override
        log.warning(
            "ai_tracer: ignoring unsupported model_override %r; using %s",
            model_override, MODEL,
        )
    return MODEL


def _resolve_frame_picker_model(model_override: str | None = None) -> str:
    """Routes back to _resolve_model so the four frame-picker tasks run
    on whatever the operator picked (Opus by default). We tried Haiku
    here for cost — accuracy regressed too much in practice, so reverted.
    Kept the helper in place in case we want per-task model routing
    later (e.g. send handedness to Haiku but keep address on Opus)."""
    return _resolve_model(model_override)


def _maybe_apply_clahe(frame):
    """Run CLAHE on the frame's luma channel when the image looks low-
    contrast (overcast / flat lighting). Returns a (possibly enhanced)
    BGR ndarray and a boolean indicating whether the enhancement
    actually fired. Cheap when skipped (~1 ms for the std check) and
    cheap when applied (~5-10 ms for typical 1080p crops)."""
    if not HAS_CV or frame is None or frame.size == 0:
        return frame, False
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception:
        return frame, False
    if HAS_NP:
        try:
            std_dev = float(np.std(gray))
        except Exception:
            std_dev = float(gray.std()) if hasattr(gray, "std") else 100.0
    else:
        std_dev = 100.0
    if std_dev >= CLAHE_CONTRAST_THRESHOLD:
        return frame, False
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE),
        )
        l_eq = clahe.apply(l_chan)
        merged = cv2.merge((l_eq, a_chan, b_chan))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return enhanced, True
    except Exception:
        return frame, False


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _sample_frame_indices(total_frames: int, n_samples: int) -> list[int]:
    """Pick n_samples evenly-spaced indices, avoiding the very first
    and last few frames where the golfer may not be in shot."""
    if total_frames <= 1:
        return [0]
    pad = max(1, total_frames // 20)
    lo = pad
    hi = max(pad + 1, total_frames - pad - 1)
    if hi <= lo:
        lo, hi = 0, total_frames - 1
    if n_samples <= 1:
        return [(lo + hi) // 2]
    step = (hi - lo) / float(n_samples - 1)
    out: list[int] = []
    for i in range(n_samples):
        idx = int(round(lo + i * step))
        idx = max(0, min(total_frames - 1, idx))
        if idx not in out:
            out.append(idx)
    return out


def _grab_frame(input_path: Path, frame_idx: int):
    """Return the raw BGR ndarray for one frame, or None on failure."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _grab_frames_jpegs(
    input_path: Path, indices: list[int], frame_w: int = FRAME_W,
) -> list[tuple[int, bytes]]:
    """Return (frame_idx, jpeg_bytes) for each requested index, resized
    to frame_w wide. Failed reads are skipped silently."""
    out: list[tuple[int, bytes]] = []
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return out
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            if w > frame_w:
                scale = frame_w / float(w)
                frame = cv2.resize(
                    frame, (frame_w, int(round(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                out.append((idx, bytes(buf)))
    finally:
        cap.release()
    return out


def annotate_address_with_shaft(
    input_path: Path, address_frame_idx: int,
    handedness_info: dict, output_image_path: Path,
) -> bool:
    """Draw the shaft line + landmark dots Claude identified onto the
    native-resolution address frame and write it to output_image_path.
    Coordinates from `handedness_info` are in the SENT image space
    (typically HANDEDNESS_FRAME_W wide) and are scaled to native here.

    Returns True if a file was written, False otherwise (missing
    coordinates, frame extraction failure, encode failure).
    """
    if not HAS_CV:
        return False
    hx, hy = handedness_info.get("hands_x"), handedness_info.get("hands_y")
    bx, by = handedness_info.get("ball_x"), handedness_info.get("ball_y")
    sent_w = handedness_info.get("image_width") or 0
    sent_h = handedness_info.get("image_height") or 0
    if None in (hx, hy, bx, by) or sent_w <= 0 or sent_h <= 0:
        return False

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return False
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(address_frame_idx))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return False

    nh, nw = frame.shape[:2]
    sx = nw / float(sent_w)
    sy = nh / float(sent_h)
    p_hands = (int(round(hx * sx)), int(round(hy * sy)))
    p_ball = (int(round(bx * sx)), int(round(by * sy)))

    # Halo behind the shaft for legibility on busy backgrounds.
    cv2.line(frame, p_hands, p_ball, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.line(frame, p_hands, p_ball, (0, 230, 255), 4, cv2.LINE_AA)
    # Landmark dots.
    cv2.circle(frame, p_hands, 12, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(frame, p_hands, 9, (0, 230, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, p_ball, 12, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(frame, p_ball, 9, (255, 80, 80), -1, cv2.LINE_AA)
    # Short text labels.
    cv2.putText(
        frame, "hands", (p_hands[0] + 14, p_hands[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "hands", (p_hands[0] + 14, p_hands[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "ball start", (p_ball[0] + 14, p_ball[1] + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "ball start", (p_ball[0] + 14, p_ball[1] + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 80), 1, cv2.LINE_AA,
    )
    ok = cv2.imwrite(
        str(output_image_path), frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
    )
    return bool(ok)


def find_ball_at_address_cv(
    input_path: Path,
    address_frame_idx: int,
    target_w: int = 1024,
) -> dict:
    """Classical-CV ball-at-address detector. Cheap replacement for
    the handedness Claude call when all we need is the ball-at-rest
    pixel coords as a ball-tracking seed.

    Samples a handful of frames near address_frame_idx, runs tophat
    morphology + circularity filtering on each to find small bright
    blobs, and votes across frames so only a stationary white object
    (the ball) survives — scattered range balls in the foreground or
    a white tee marker that happens to look round don't get reinforced
    across frames the way the actual ball does.

    Returns a dict with the same fields detect_handedness_at_address
    fills for downstream consumers (ball_x/y, image_width/height) so
    the rest of the pipeline can treat the two interchangeably.
    handedness is left as 'unknown' since downstream no longer uses it.
    """
    info: dict = {
        "ok": False,
        "error": None,
        "handedness": "unknown",
        "confidence": None,
        "notes": None,
        "hands_x": None,
        "hands_y": None,
        "ball_x": None,
        "ball_y": None,
        "shaft_direction": None,
        "image_width": None,
        "image_height": None,
        "n_votes": 0,
        "method": "classical_cv",
        "model": None,
    }
    if not HAS_CV or not HAS_NP:
        info["error"] = "opencv or numpy not installed"
        return info

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        info["error"] = "could not open video"
        return info
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Sample five frames centered on the address frame. Address is
        # held for ~1 s before takeaway, so anything within +-15 frames
        # at 30 fps should still be the golfer over the ball.
        offsets = (-15, -8, 0, 8, 15)
        candidate_indices: list[int] = []
        for off in offsets:
            idx = max(0, min(total - 1, int(address_frame_idx) + off))
            if idx not in candidate_indices:
                candidate_indices.append(idx)

        per_frame_dets: list[tuple[int, int]] = []
        sent_w = 0
        sent_h = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        for idx in candidate_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            if w > target_w:
                scale = target_w / float(w)
                frame = cv2.resize(
                    frame, (target_w, int(round(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            sent_h, sent_w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            _, mask = cv2.threshold(tophat, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in contours:
                area = float(cv2.contourArea(c))
                if area < 3.0 or area > 200.0:
                    continue
                peri = float(cv2.arcLength(c, True))
                if peri <= 0:
                    continue
                circ = 4.0 * math.pi * area / (peri * peri)
                if circ < 0.5:
                    continue
                (cx, cy), _r = cv2.minEnclosingCircle(c)
                # Ball sits on the ground — anything above ~40 % of the
                # frame is probably sky / golfer's torso, drop it.
                if cy < sent_h * 0.40:
                    continue
                per_frame_dets.append((int(round(cx)), int(round(cy))))
    finally:
        cap.release()

    info["image_width"] = sent_w
    info["image_height"] = sent_h
    if sent_w == 0:
        info["error"] = "could not extract any frames"
        return info
    if not per_frame_dets:
        info["error"] = "no ball-shaped blobs detected near address frame"
        return info

    # Cluster nearby detections — same ball appearing across multiple
    # frames must land within a few pixels of itself.
    clusters: list[list[int | float]] = []  # [count, sum_x, sum_y]
    CLUSTER_RADIUS_PX = 10
    for x, y in per_frame_dets:
        merged = False
        for c in clusters:
            ax = c[1] / c[0]
            ay = c[2] / c[0]
            if abs(x - ax) <= CLUSTER_RADIUS_PX and abs(y - ay) <= CLUSTER_RADIUS_PX:
                c[0] += 1
                c[1] += x
                c[2] += y
                merged = True
                break
        if not merged:
            clusters.append([1, float(x), float(y)])

    clusters.sort(key=lambda c: -c[0])
    best = clusters[0]
    votes = int(best[0])
    if votes < 2:
        info["error"] = (
            f"no ball candidate confirmed across nearby frames "
            f"(best cluster had {votes} vote, need >= 2)"
        )
        return info

    ball_x = int(round(best[1] / votes))
    ball_y = int(round(best[2] / votes))
    info["ok"] = True
    info["ball_x"] = ball_x
    info["ball_y"] = ball_y
    info["n_votes"] = votes
    info["confidence"] = "high" if votes >= 4 else ("medium" if votes >= 3 else "low")
    info["notes"] = (
        f"classical-CV tophat vote: ball at ({ball_x}, {ball_y}) in "
        f"{sent_w}x{sent_h} image, {votes}/{len(candidate_indices)} "
        f"sampled frames agreed"
    )
    log.info(
        "ai_tracer: find_ball_at_address_cv — addr=%d ball=(%d,%d) votes=%d/%d "
        "in %dx%d (no Claude call)",
        address_frame_idx, ball_x, ball_y, votes, len(candidate_indices),
        sent_w, sent_h,
    )
    return info


def _ball_present_near(
    input_path: Path,
    frame_idx: int,
    x: int,
    y: int,
    target_w: int = 1024,
    radius_px: int = 16,
) -> bool | None:
    """Is there a small bright ball-like blob within `radius_px` of
    (x, y) in the frame at `frame_idx`? Coords are in a target_w-wide
    normalised image — the same space find_ball_at_address_cv returns.
    Uses the identical tophat + circularity test so 'is the ball still
    here' is judged exactly like 'where is the ball'. Returns True /
    False, or None if the frame can't be read."""
    if not HAS_CV or not HAS_NP:
        return None
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = max(0, min(total - 1, int(frame_idx)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        if w > target_w:
            scale = target_w / float(w)
            frame = cv2.resize(
                frame, (target_w, int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        _, mask = cv2.threshold(tophat, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        r2 = float(radius_px * radius_px)
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < 3.0 or area > 200.0:
                continue
            peri = float(cv2.arcLength(c, True))
            if peri <= 0:
                continue
            if (4.0 * math.pi * area / (peri * peri)) < 0.5:
                continue
            (cx, cy), _r = cv2.minEnclosingCircle(c)
            if (cx - x) ** 2 + (cy - y) ** 2 <= r2:
                return True
        return False
    finally:
        cap.release()


def ball_departed_for_swing(
    input_path: Path,
    peak_time_sec: float,
    fps: float,
    before_offset_sec: float = 1.5,
    after_offsets_sec: tuple[float, ...] = (1.8, 3.0),
    search_radius_px: int = 16,
    debug: dict | None = None,
) -> str:
    """Classify a candidate swing by what the ball did — this separates a
    real golf shot from a practice swing or non-golf motion, with no
    audio.

    Locates the ball at address (~before_offset_sec before the swing),
    then checks whether a ball-like blob is still at that exact spot a
    couple of seconds later. Returns one of:
      "departed"  — a ball was on the tee and is gone after → REAL shot
      "present"   — a ball is still on the tee after        → practice swing
      "no_ball"   — no ball found at address                → not a golf shot
                    (random motion, someone walking through frame, indoors)
      "uncertain" — a ball WAS found but the after-frame couldn't be read
                    → treat as a likely shot (don't drop on a glitch)
    """
    if not HAS_CV or not HAS_NP or not fps or fps <= 0:
        return "uncertain"
    addr_idx = max(0, int(round((peak_time_sec - before_offset_sec) * fps)))
    ball = find_ball_at_address_cv(input_path, addr_idx)
    if not ball.get("ok"):
        if debug is not None:
            debug.setdefault("ball_filter", []).append(
                {"peak_sec": round(peak_time_sec, 2), "result": "no_ball"}
            )
        return "no_ball"

    bx, by = int(ball["ball_x"]), int(ball["ball_y"])
    present_after = False
    checked = 0
    for off in after_offsets_sec:
        aidx = int(round((peak_time_sec + off) * fps))
        res = _ball_present_near(
            input_path, aidx, bx, by, radius_px=search_radius_px
        )
        if res is None:
            continue
        checked += 1
        if res:
            present_after = True
            break

    if checked == 0:
        if debug is not None:
            debug.setdefault("ball_filter", []).append(
                {"peak_sec": round(peak_time_sec, 2),
                 "ball_xy": [bx, by], "result": "uncertain"}
            )
        return "uncertain"

    result = "present" if present_after else "departed"
    if debug is not None:
        debug.setdefault("ball_filter", []).append({
            "peak_sec": round(peak_time_sec, 2),
            "ball_xy": [bx, by],
            "present_after": present_after,
            "result": result,
        })
    return result


# Keep a swing as a real shot only when the ball was confirmed to leave
# the tee — or a ball was present but the after-check glitched. "present"
# (practice swing, ball stayed) and "no_ball" (no golf ball at all) are
# dropped, so only actual golf shots reach Production.
_BALL_KEEP_RESULTS = {"departed", "uncertain"}


def filter_swings_by_ball_departure(
    input_path: Path,
    swings: list[dict],
    fps: float,
    debug: dict | None = None,
    keep_all: bool = False,
    drop_garbage: bool = False,
) -> list[dict]:
    """Classify each candidate swing by what the ball did and tag it with a
    `ball_verdict` ("departed" / "present" / "no_ball" / "uncertain").

    Every swing is tagged regardless, so the verdict is always available
    downstream (stored on the produced clip's diagnostics). What differs is
    whether non-shots are DROPPED:

    - `keep_all=False` (strict, the eventual production default): keep only
      real golf shots — a ball was on the tee and left it. Drops practice
      swings (ball stayed) and any motion with no ball at all. The risk: if
      the tee camera can't clearly see the ball (poor placement, occlusion),
      a legitimate shot is silently dropped.

    - `keep_all=True` (permissive, for course testing): keep every detected
      swing but still tag the verdict. Nothing is silently lost — we produce
      everything, see what the ball check thought of each, then tune the
      thresholds and flip back to strict once it's trustworthy.

    - `drop_garbage=True` (only meaningful with keep_all): a middle ground
      that weeds out CLEAR non-golf — a motion burst with no ball ever on the
      tee ("no_ball": someone walking through frame, indoor/kitchen motion) —
      while still keeping practice swings ("present") and shots the camera
      couldn't quite see the ball for ("uncertain"). This removes the
      "garbage" clips without reintroducing the strict-mode risk of dropping a
      real shot on a ball-not-visible glitch."""
    if not swings:
        return swings
    kept: list[dict] = []
    dropped = 0
    garbage = 0
    for sw in swings:
        peak = float(sw.get("peak_time_sec") or sw.get("start_sec") or 0.0)
        result = ball_departed_for_swing(input_path, peak, fps, debug=debug)
        # Tag every swing with the ball check's verdict so it rides along
        # to the produced clip's diagnostics, whether or not we drop it.
        sw["ball_verdict"] = result
        is_shot = result in _BALL_KEEP_RESULTS
        is_garbage = result == "no_ball"
        if keep_all:
            # Permissive: keep everything, EXCEPT clear garbage when the
            # garbage weeder is on (no ball ever = not golf).
            if drop_garbage and is_garbage:
                garbage += 1
                log.info(
                    "ai_tracer: swing@%.1fs dropped as garbage — no ball ever "
                    "on the tee (not golf)", peak,
                )
                continue
            kept.append(sw)
            if not is_shot:
                log.info(
                    "ai_tracer: swing@%.1fs kept (keep_all) — ball verdict=%s",
                    peak, result,
                )
        elif is_shot:
            kept.append(sw)
        else:
            dropped += 1
            log.info(
                "ai_tracer: swing@%.1fs dropped — %s", peak,
                "ball still on tee (practice swing)"
                if result == "present" else "no ball detected (not a shot)",
            )
    log.info(
        "ai_tracer: ball filter — kept %d of %d swings (dropped %d, garbage %d; "
        "keep_all=%s drop_garbage=%s)",
        len(kept), len(swings), dropped, garbage, keep_all, drop_garbage,
    )
    return kept


def detect_handedness_at_address(
    input_path: Path, address_frame_idx: int,
    frame_w: int = HANDEDNESS_FRAME_W,
    model: str | None = None,
    examples: list | None = None,
) -> dict:
    """Single-frame Claude call: given the picked address frame index,
    ask whether the golfer is right- or left-handed. Camera is assumed
    to be behind the golfer (which it always is per the upstream
    assumption).

    Returns::

        {
          "ok": bool,
          "error": str | None,
          "handedness": "right" | "left" | "unknown" | None,
          "confidence": "high" | "medium" | "low" | None,
          "notes": str | None,
          "model": str,
        }

    Never raises.
    """
    model = _resolve_frame_picker_model(model)
    info: dict = {
        "ok": False,
        "error": None,
        "handedness": None,
        "confidence": None,
        "notes": None,
        "hands_x": None,
        "hands_y": None,
        "ball_x": None,
        "ball_y": None,
        "shaft_direction": None,
        "image_width": None,
        "image_height": None,
        "model": model,
        "examples_used": [],
    }
    if not HAS_CV:
        info["error"] = "opencv not installed"
        return info
    if not HAS_ANTHROPIC:
        info["error"] = "anthropic SDK not installed"
        return info
    if not os.environ.get("ANTHROPIC_API_KEY"):
        info["error"] = "ANTHROPIC_API_KEY not set in environment"
        return info

    # Re-read the frame so we can record the actual sent width Claude
    # will see — used to make the hands_x / ball_x reasoning explicit.
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        info["error"] = "could not open video for handedness call"
        return info
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(address_frame_idx))
        ok, raw = cap.read()
    finally:
        cap.release()
    if not ok or raw is None:
        info["error"] = f"could not read address frame {address_frame_idx}"
        return info
    h, w = raw.shape[:2]
    if w > frame_w:
        scale = frame_w / float(w)
        raw = cv2.resize(
            raw, (frame_w, int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    sent_h, sent_w = raw.shape[:2]
    info["image_width"] = sent_w
    info["image_height"] = sent_h
    ok, buf = cv2.imencode(".jpg", raw, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        info["error"] = "could not jpeg-encode address frame"
        return info
    jpeg = bytes(buf)

    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"Address frame ({address_frame_idx}) from the clip. "
                f"Image is {sent_w}x{sent_h} pixels. "
                "Locate the hands and the golf ball, then determine handedness "
                "from ball_x vs hands_x as instructed. JSON only."
            ),
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(jpeg).decode("ascii"),
            },
        },
    ]

    # Few-shot prior: prepend operator-verified handedness examples
    # from this course before the task block above.
    if examples:
        from . import tracer_examples as _ex
        prior_blocks = _ex.example_blocks(examples, _ex.KIND_HANDEDNESS)
        if prior_blocks:
            content = prior_blocks + content
            info["examples_used"] = [
                {"lvu_id": e.lvu_id, "hole": e.hole_number} for e in examples
            ]

    client = _anthropic_client()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=200,
            system=[{
                "type": "text",
                "text": HANDEDNESS_FROM_ADDRESS_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": content}],
        )
    except Exception as exc:
        log.warning("ai_tracer: handedness API call failed: %s", exc)
        info["error"] = f"api_failed: {exc}"
        return info

    text_chunks = [
        b.text for b in resp.content
        if getattr(b, "type", None) == "text" and getattr(b, "text", None)
    ]
    parsed = _extract_json("\n".join(text_chunks))
    if not parsed:
        info["error"] = "no_json_in_response"
        return info

    handedness = str(parsed.get("handedness") or "").lower()
    if handedness not in {"right", "left", "unknown"}:
        info["error"] = f"unexpected handedness value: {handedness!r}"
        return info
    info["ok"] = True
    info["handedness"] = handedness
    info["confidence"] = str(parsed.get("confidence") or "").lower() or None
    info["notes"] = str(parsed.get("notes") or "")[:300] or None
    for key in ("hands_x", "hands_y", "ball_x", "ball_y"):
        try:
            info[key] = int(parsed[key])
        except (KeyError, TypeError, ValueError):
            info[key] = None
    sd = str(parsed.get("shaft_direction") or "").lower()
    info["shaft_direction"] = sd if sd in {"down_left", "down_right", "vertical"} else None
    log.info(
        "ai_tracer: handedness at address frame %d — %s (%s) "
        "hands=(%s,%s) ball=(%s,%s) shaft=%s — %s",
        address_frame_idx, info["handedness"], info["confidence"],
        info["hands_x"], info["hands_y"], info["ball_x"], info["ball_y"],
        info["shaft_direction"], info["notes"],
    )
    return info


def find_address_frame(
    input_path: Path, output_image_path: Path | None = None,
    model: str | None = None,
    examples: list | None = None,
) -> dict:
    """Ask Claude which frame in the clip is the golfer at address.

    Returns a dict::

        {
          "ok": bool,
          "error": str | None,
          "address_frame": int | None,
          "confidence": "high" | "medium" | "low" | None,
          "notes": str | None,
          "model": str,
          "frames_sent": [int, ...],   # frame indices shown to Claude
          "saved_image": bool,         # True if output_image_path was written
        }

    If `output_image_path` is provided AND Claude returned a frame
    index, the full-resolution version of that frame is written to
    disk so the caller can serve it for display. Never raises — every
    failure path ends with ok=False + a descriptive error.
    """
    model = _resolve_frame_picker_model(model)
    info: dict = {
        "ok": False,
        "error": None,
        "address_frame": None,
        "confidence": None,
        "notes": None,
        "model": model,
        "frames_sent": [],
        "saved_image": False,
        "examples_used": [],
    }

    if not HAS_CV:
        info["error"] = "opencv not installed"
        return info
    if not HAS_ANTHROPIC:
        info["error"] = "anthropic SDK not installed"
        return info
    if not os.environ.get("ANTHROPIC_API_KEY"):
        info["error"] = "ANTHROPIC_API_KEY not set in environment"
        return info

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        info["error"] = "could not open video"
        return info
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    finally:
        cap.release()
    if total_frames <= 1:
        info["error"] = "video has no frames"
        return info

    indices = _sample_frame_indices(total_frames, N_FRAMES)
    frames = _grab_frames_jpegs(input_path, indices)
    if not frames:
        info["error"] = "could not extract any frames"
        return info
    candidate_frames = [idx for idx, _ in frames]
    info["frames_sent"] = candidate_frames
    log.info(
        "ai_tracer: address — sending %d frames %s of %d total @ %.1f fps to %s",
        len(frames), candidate_frames, total_frames, fps, model,
    )

    content: list[dict] = []
    if examples:
        from . import tracer_examples as _ex
        prior_blocks = _ex.example_blocks(examples, _ex.KIND_ADDRESS)
        if prior_blocks:
            content.extend(prior_blocks)
            info["examples_used"] = [
                {"lvu_id": e.lvu_id, "hole": e.hole_number} for e in examples
            ]
    content.append({
        "type": "text",
        "text": (
            f"Below are {len(frames)} frames from a {total_frames}-frame "
            f"clip at {fps:.1f} fps, in chronological order. Each image "
            "is preceded by its frame number."
        ),
    })
    for idx, jpeg in frames:
        content.append({"type": "text", "text": f"Frame {idx}:"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(jpeg).decode("ascii"),
            },
        })
    content.append({
        "type": "text",
        "text": (
            "Identify which of the labeled frame numbers shows the "
            f"golfer at ADDRESS, just before takeoff. Valid frame "
            f"numbers: {candidate_frames}. Respond with JSON only."
        ),
    })

    client = _anthropic_client()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=256,
            system=[{
                "type": "text",
                "text": ADDRESS_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": content}],
        )
    except Exception as exc:
        log.warning("ai_tracer: address API call failed: %s", exc)
        info["error"] = f"api_failed: {exc}"
        return info

    text_chunks = [
        b.text for b in resp.content
        if getattr(b, "type", None) == "text" and getattr(b, "text", None)
    ]
    raw = "\n".join(text_chunks)
    parsed = _extract_json(raw)
    if not parsed:
        log.warning("ai_tracer: address — no JSON in response: %s", raw[:200])
        info["error"] = "no_json_in_response"
        return info

    try:
        addr = int(parsed.get("address_frame"))
    except (TypeError, ValueError):
        info["error"] = "could_not_parse_address_frame"
        return info
    # Snap to the nearest candidate if Claude returned something off-list.
    if addr not in candidate_frames:
        nearest = min(candidate_frames, key=lambda f: abs(f - addr))
        log.warning(
            "ai_tracer: address — Claude returned frame %d not in candidates; "
            "snapping to nearest candidate %d",
            addr, nearest,
        )
        addr = nearest
    if addr < 0 or addr >= total_frames:
        info["error"] = f"address_frame_out_of_range ({addr})"
        return info

    info["ok"] = True
    info["address_frame"] = addr
    info["confidence"] = str(parsed.get("confidence") or "").lower() or None
    info["notes"] = str(parsed.get("notes") or "")[:300] or None
    log.info(
        "ai_tracer: address — frame=%d confidence=%s notes=%s",
        addr, info["confidence"], info["notes"],
    )

    # Save the picked frame at native resolution so the UI can display it.
    if output_image_path is not None:
        full_frame = _grab_frame(input_path, addr)
        if full_frame is not None:
            ok = cv2.imwrite(
                str(output_image_path), full_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
            )
            info["saved_image"] = bool(ok)
        else:
            log.warning("ai_tracer: address — could not re-extract picked frame %d", addr)
    return info


def find_impact_frame_after_address(
    input_path: Path, address_frame_idx: int,
    ball_xy_sent: tuple[float, float] | None = None,
    ball_sent_dims: tuple[int, int] | None = None,
    output_image_path: Path | None = None,
    model: str | None = None,
    examples: list | None = None,
) -> dict:
    """Find the impact frame — when the clubhead has returned to the
    ball after the backswing.

    Pipeline:
      1. Sample N_IMPACT_CANDIDATES (12) frames evenly across
         [address+1, address+IMPACT_WINDOW_SECONDS] of clip time.
         2 s covers takeaway → top → downswing → impact at all
         typical frame rates.
      2. If `ball_xy_sent` + `ball_sent_dims` are provided (from the
         handedness pass), draw a BLUE CIRCLE at the ball's starting
         position on the address frame AND every candidate. Coords
         are scaled to native pixels internally.
      3. Send the address frame as a REFERENCE plus all candidates
         in a single multi-image API call. Claude picks the candidate
         whose clubhead is closest to the blue circle / address ball
         position.

    Returns the same shape as find_address_frame(). The written
    output_image_path file is the native-resolution impact frame
    with the same blue ball-circle drawn on it.
    """
    model = _resolve_frame_picker_model(model)
    info: dict = {
        "ok": False,
        "error": None,
        "impact_frame": None,
        "confidence": None,
        "notes": None,
        "model": model,
        "frames_sent": [],
        "saved_image": False,
        "examples_used": [],
    }

    if not HAS_CV:
        info["error"] = "opencv not installed"
        return info
    if not HAS_ANTHROPIC:
        info["error"] = "anthropic SDK not installed"
        return info
    if not os.environ.get("ANTHROPIC_API_KEY"):
        info["error"] = "ANTHROPIC_API_KEY not set in environment"
        return info

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        info["error"] = "could not open video"
        return info
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if total_frames <= 1:
        info["error"] = "video has no frames"
        return info

    address_frame_idx = int(address_frame_idx)
    window_frames = max(1, int(round(fps * IMPACT_WINDOW_SECONDS)))
    lo = min(address_frame_idx + 1, total_frames - 1)
    hi = min(address_frame_idx + window_frames, total_frames - 1)
    if hi <= lo:
        info["error"] = (
            f"address frame {address_frame_idx} too close to end of clip "
            f"({total_frames} frames @ {fps:.1f} fps)"
        )
        return info
    n = N_IMPACT_CANDIDATES
    if hi - lo + 1 <= n:
        candidate_indices = list(range(lo, hi + 1))
    else:
        step = (hi - lo) / float(n - 1)
        candidate_indices = sorted({
            min(hi, max(lo, int(round(lo + i * step)))) for i in range(n)
        })
    info["frames_sent"] = candidate_indices

    # Resolve the ball position to native pixel coords. Callers pass
    # ball_xy_sent in whatever resolution their pass used (e.g. 1024px
    # from the handedness call); we scale it up to native here so the
    # circle lands on the right pixel of each candidate frame.
    ball_xy_native: tuple[float, float] | None = None
    if (
        ball_xy_sent is not None and ball_sent_dims is not None
        and ball_sent_dims[0] > 0 and ball_sent_dims[1] > 0
        and native_w > 0 and native_h > 0
    ):
        sw, sh = ball_sent_dims
        ball_xy_native = (
            float(ball_xy_sent[0]) * native_w / float(sw),
            float(ball_xy_sent[1]) * native_h / float(sh),
        )

    # Helper: extract a frame at native res, draw the blue ball-rest
    # circle on it (when ball coords are known), resize to FRAME_W,
    # and JPEG-encode for the API payload.
    def _annotated_jpeg(idx: int) -> bytes | None:
        local_cap = cv2.VideoCapture(str(input_path))
        if not local_cap.isOpened():
            return None
        try:
            local_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = local_cap.read()
        finally:
            local_cap.release()
        if not ok or frame is None:
            return None
        if ball_xy_native is not None:
            bx = int(round(ball_xy_native[0]))
            by = int(round(ball_xy_native[1]))
            cv2.circle(frame, (bx, by), 18, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.circle(frame, (bx, by), 16, (255, 60, 0), 4, cv2.LINE_AA)
        h, w = frame.shape[:2]
        if w > FRAME_W:
            scale = FRAME_W / float(w)
            frame = cv2.resize(
                frame, (FRAME_W, int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        return bytes(buf) if ok else None

    # Reference: the address frame itself (also with the blue circle).
    reference_jpeg = _annotated_jpeg(address_frame_idx)
    if reference_jpeg is None:
        info["error"] = f"could not extract address reference frame {address_frame_idx}"
        return info

    annotated_jpegs: list[tuple[int, bytes]] = []
    for idx in candidate_indices:
        jpeg = _annotated_jpeg(idx)
        if jpeg is not None:
            annotated_jpegs.append((idx, jpeg))

    if not annotated_jpegs:
        info["error"] = "could not extract any candidate frames"
        return info
    candidate_indices_actual = [idx for idx, _ in annotated_jpegs]

    log.info(
        "ai_tracer: impact — address=%d candidates=%s ball_native=%s",
        address_frame_idx, candidate_indices_actual, ball_xy_native,
    )

    content: list[dict] = []
    if examples:
        from . import tracer_examples as _ex
        prior_blocks = _ex.example_blocks(examples, _ex.KIND_IMPACT)
        if prior_blocks:
            content.extend(prior_blocks)
            info["examples_used"] = [
                {"lvu_id": e.lvu_id, "hole": e.hole_number} for e in examples
            ]
    content.extend([
        {
            "type": "text",
            "text": (
                f"REFERENCE — address frame (frame {address_frame_idx}). "
                "Golfer set up over the ball, just before takeaway. The "
                "blue circle marks the ball's starting position; the same "
                "circle is drawn on every candidate frame below."
            ),
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(reference_jpeg).decode("ascii"),
            },
        },
        {
            "type": "text",
            "text": (
                f"Now {len(annotated_jpegs)} candidate frames from AFTER "
                "address, in chronological order. Each is preceded by its "
                "frame number. Each candidate has the same blue circle at "
                "the ball's starting position so you can compare the "
                "clubhead's distance to it across frames."
            ),
        },
    ])
    for idx, jpeg in annotated_jpegs:
        content.append({"type": "text", "text": f"Frame {idx}:"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(jpeg).decode("ascii"),
            },
        })
    content.append({
        "type": "text",
        "text": (
            "Identify the IMPACT frame — the candidate whose clubhead "
            "is back at (or closest to) the blue circle. Valid frame "
            f"numbers: {candidate_indices_actual}. JSON only."
        ),
    })

    client = _anthropic_client()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=300,
            system=[{
                "type": "text",
                "text": IMPACT_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": content}],
        )
    except Exception as exc:
        log.warning("ai_tracer: impact API call failed: %s", exc)
        info["error"] = f"api_failed: {exc}"
        return info

    text_chunks = [
        b.text for b in resp.content
        if getattr(b, "type", None) == "text" and getattr(b, "text", None)
    ]
    raw = "\n".join(text_chunks)
    parsed = _extract_json(raw)
    if not parsed:
        log.warning("ai_tracer: impact — no JSON in response: %s", raw[:200])
        info["error"] = "no_json_in_response"
        return info

    try:
        impact = int(parsed.get("impact_frame"))
    except (TypeError, ValueError):
        info["error"] = "could_not_parse_impact_frame"
        return info
    if impact not in candidate_indices_actual:
        nearest = min(candidate_indices_actual, key=lambda f: abs(f - impact))
        log.warning(
            "ai_tracer: impact — Claude returned frame %d not in candidates; "
            "snapping to nearest candidate %d",
            impact, nearest,
        )
        impact = nearest

    info["ok"] = True
    info["impact_frame"] = impact
    info["confidence"] = str(parsed.get("confidence") or "").lower() or None
    info["notes"] = str(parsed.get("notes") or "")[:300] or None
    log.info(
        "ai_tracer: impact — frame=%d confidence=%s notes=%s",
        impact, info["confidence"], info["notes"],
    )

    # Save the picked frame at native resolution with the same blue
    # ball-circle annotation drawn on it.
    if output_image_path is not None:
        full_frame = _grab_frame(input_path, impact)
        if full_frame is not None:
            if ball_xy_native is not None:
                bx = int(round(ball_xy_native[0]))
                by = int(round(ball_xy_native[1]))
                cv2.circle(full_frame, (bx, by), 22, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.circle(full_frame, (bx, by), 20, (255, 60, 0), 4, cv2.LINE_AA)
            ok = cv2.imwrite(
                str(output_image_path), full_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
            )
            info["saved_image"] = bool(ok)
        else:
            log.warning("ai_tracer: impact — could not re-extract picked frame %d", impact)
    return info


def refine_impact_frame(
    input_path: Path, approximate_impact_idx: int,
    ball_xy_sent: tuple[float, float] | None,
    ball_sent_dims: tuple[int, int] | None,
    output_image_path: Path | None = None,
    model: str | None = None,
) -> dict:
    """Refine the impact frame to within ±5 of the initial estimate AND
    locate the shaft (hands + clubhead) on the picked frame in a single
    multi-image Claude call.

    Pipeline:
      1. Sample 11 frames [impact-5, impact-4, ..., impact+5], clipped
         to clip length.
      2. Draw the BLUE CIRCLE at the ball's starting position on every
         candidate (at native res, then resize for the API).
      3. Single Claude call: pick the candidate whose shaft points
         closest at the blue circle, AND report hands/clubhead pixel
         positions on that picked frame.
      4. If output_image_path is provided, write the refined impact
         frame at native res with both the blue ball circle AND the
         yellow shaft line overlaid on it.

    Returns dict with refined frame index, landmarks, confidence, notes,
    frames_sent, saved_image, image dimensions, and any error.
    """
    model = _resolve_frame_picker_model(model)
    info: dict = {
        "ok": False,
        "error": None,
        "impact_frame": None,
        "hands_x": None,
        "hands_y": None,
        "clubhead_x": None,
        "clubhead_y": None,
        "image_width": None,
        "image_height": None,
        "confidence": None,
        "notes": None,
        "model": model,
        "frames_sent": [],
        "saved_image": False,
    }

    if not HAS_CV:
        info["error"] = "opencv not installed"
        return info
    if not HAS_ANTHROPIC:
        info["error"] = "anthropic SDK not installed"
        return info
    if not os.environ.get("ANTHROPIC_API_KEY"):
        info["error"] = "ANTHROPIC_API_KEY not set in environment"
        return info

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        info["error"] = "could not open video"
        return info
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if total_frames <= 1:
        info["error"] = "video has no frames"
        return info

    approximate_impact_idx = int(approximate_impact_idx)
    candidate_indices = sorted({
        max(0, min(total_frames - 1, approximate_impact_idx + off))
        for off in range(-5, 6)
    })
    info["frames_sent"] = candidate_indices

    # Resolve ball position to native pixel coords (so the circle lands
    # on the right pixel of each candidate at native res before resize).
    ball_xy_native: tuple[float, float] | None = None
    if (
        ball_xy_sent is not None and ball_sent_dims is not None
        and ball_sent_dims[0] > 0 and ball_sent_dims[1] > 0
        and native_w > 0 and native_h > 0
    ):
        sw, sh = ball_sent_dims
        ball_xy_native = (
            float(ball_xy_sent[0]) * native_w / float(sw),
            float(ball_xy_sent[1]) * native_h / float(sh),
        )

    # Extract + annotate each candidate; track the actual sent
    # dimensions so we can scale landmark coords back to native later.
    annotated_jpegs: list[tuple[int, bytes]] = []
    sent_dims: tuple[int, int] = (0, 0)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        info["error"] = "could not re-open video"
        return info
    try:
        for idx in candidate_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if ball_xy_native is not None:
                bx = int(round(ball_xy_native[0]))
                by = int(round(ball_xy_native[1]))
                cv2.circle(frame, (bx, by), 18, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.circle(frame, (bx, by), 16, (255, 60, 0), 4, cv2.LINE_AA)
            h, w = frame.shape[:2]
            if w > REFINE_IMPACT_FRAME_W:
                scale = REFINE_IMPACT_FRAME_W / float(w)
                frame = cv2.resize(
                    frame, (REFINE_IMPACT_FRAME_W, int(round(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            sent_dims = (frame.shape[1], frame.shape[0])
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
            if ok:
                annotated_jpegs.append((idx, bytes(buf)))
    finally:
        cap.release()

    if not annotated_jpegs:
        info["error"] = "could not extract any candidate frames"
        return info
    candidate_indices_actual = [idx for idx, _ in annotated_jpegs]
    info["image_width"] = sent_dims[0]
    info["image_height"] = sent_dims[1]

    log.info(
        "ai_tracer: refine_impact — approx=%d candidates=%s ball_native=%s sent=%sx%s",
        approximate_impact_idx, candidate_indices_actual, ball_xy_native,
        sent_dims[0], sent_dims[1],
    )

    content: list[dict] = [{
        "type": "text",
        "text": (
            f"Below are {len(annotated_jpegs)} candidate frames near "
            f"impact (image size {sent_dims[0]}x{sent_dims[1]} px). "
            "Each has the same blue circle at the ball's starting "
            "position."
        ),
    }]
    for idx, jpeg in annotated_jpegs:
        content.append({"type": "text", "text": f"Frame {idx}:"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(jpeg).decode("ascii"),
            },
        })
    content.append({
        "type": "text",
        "text": (
            "Pick the impact frame (clubhead closest to the blue "
            "circle) AND report hands + clubhead pixel positions on "
            f"that frame. Valid frame numbers: {candidate_indices_actual}. "
            "JSON only."
        ),
    })

    client = _anthropic_client()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=400,
            system=[{
                "type": "text",
                "text": REFINE_IMPACT_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": content}],
        )
    except Exception as exc:
        log.warning("ai_tracer: refine_impact API call failed: %s", exc)
        info["error"] = f"api_failed: {exc}"
        return info

    text_chunks = [
        b.text for b in resp.content
        if getattr(b, "type", None) == "text" and getattr(b, "text", None)
    ]
    parsed = _extract_json("\n".join(text_chunks))
    if not parsed:
        info["error"] = "no_json_in_response"
        return info

    try:
        impact = int(parsed.get("impact_frame"))
    except (TypeError, ValueError):
        info["error"] = "could_not_parse_impact_frame"
        return info
    if impact not in candidate_indices_actual:
        nearest = min(candidate_indices_actual, key=lambda f: abs(f - impact))
        log.warning(
            "ai_tracer: refine_impact returned %d not in candidates; "
            "snapping to nearest %d", impact, nearest,
        )
        impact = nearest

    info["ok"] = True
    info["impact_frame"] = impact
    info["confidence"] = str(parsed.get("confidence") or "").lower() or None
    info["notes"] = str(parsed.get("notes") or "")[:300] or None
    for key in ("hands_x", "hands_y", "clubhead_x", "clubhead_y"):
        try:
            info[key] = int(parsed[key])
        except (KeyError, TypeError, ValueError):
            info[key] = None
    log.info(
        "ai_tracer: refine_impact — frame=%d conf=%s hands=(%s,%s) clubhead=(%s,%s) "
        "notes=%s",
        impact, info["confidence"],
        info["hands_x"], info["hands_y"], info["clubhead_x"], info["clubhead_y"],
        info["notes"],
    )

    # Save the refined impact frame at native res with both annotations:
    # the blue ball-rest circle AND the yellow shaft line.
    if output_image_path is not None:
        full_frame = _grab_frame(input_path, impact)
        if full_frame is not None:
            if ball_xy_native is not None:
                bx = int(round(ball_xy_native[0]))
                by = int(round(ball_xy_native[1]))
                cv2.circle(full_frame, (bx, by), 22, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.circle(full_frame, (bx, by), 20, (255, 60, 0), 4, cv2.LINE_AA)
            sw, sh = sent_dims
            hx, hy = info["hands_x"], info["hands_y"]
            cx, cy = info["clubhead_x"], info["clubhead_y"]
            if (
                None not in (hx, hy, cx, cy) and sw > 0 and sh > 0
                and native_w > 0 and native_h > 0
            ):
                sx = native_w / float(sw)
                sy = native_h / float(sh)
                p_hands = (int(round(hx * sx)), int(round(hy * sy)))
                p_club = (int(round(cx * sx)), int(round(cy * sy)))
                cv2.line(full_frame, p_hands, p_club, (0, 0, 0), 8, cv2.LINE_AA)
                cv2.line(full_frame, p_hands, p_club, (0, 230, 255), 4, cv2.LINE_AA)
                cv2.circle(full_frame, p_hands, 12, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(full_frame, p_hands, 9, (0, 230, 255), -1, cv2.LINE_AA)
                cv2.circle(full_frame, p_club, 12, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(full_frame, p_club, 9, (255, 80, 80), -1, cv2.LINE_AA)
                cv2.putText(
                    full_frame, "hands", (p_hands[0] + 14, p_hands[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA,
                )
                cv2.putText(
                    full_frame, "hands", (p_hands[0] + 14, p_hands[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA,
                )
                cv2.putText(
                    full_frame, "clubhead", (p_club[0] + 14, p_club[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA,
                )
                cv2.putText(
                    full_frame, "clubhead", (p_club[0] + 14, p_club[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 80), 1, cv2.LINE_AA,
                )
            ok = cv2.imwrite(
                str(output_image_path), full_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
            )
            info["saved_image"] = bool(ok)
        else:
            log.warning(
                "ai_tracer: refine_impact — could not re-extract picked frame %d", impact,
            )
    return info


def _refine_crop_call(
    idx: int,
    current_pos_native: tuple[int, int],
    native_frame,
    crop_size: int,
    client,
    model: str,
) -> tuple[int, dict]:
    """Phase-3 refinement: send a tight crop centered on the current
    ball position back to Claude for pixel-accurate refinement.

    `current_pos_native` is the (x, y) Claude returned from Phase 1
    or Phase 2, already in native-frame coords. The crop is
    `crop_size` native pixels wide/tall, centered on that point and
    clamped to the frame. Result is upscaled to BALL_TRACK_REFINE_SEND_W
    so the ball — typically ~5 px at native — covers ~30 px in the
    image Claude sees.

    Returns (idx, parsed_dict). On success the returned dict carries
    the refined coords already translated back to native-frame space
    (top-left origin); on failure it has an `_error` key.
    """
    nh, nw = native_frame.shape[:2]
    cx = int(current_pos_native[0])
    cy = int(current_pos_native[1])
    half = crop_size // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(nw, cx + half)
    y1 = min(nh, cy + half)
    if x1 <= x0 or y1 <= y0:
        return idx, {"_error": "empty crop"}
    crop = native_frame[y0:y1, x0:x1]
    crop, _ = _maybe_apply_clahe(crop)
    crop_h, crop_w = crop.shape[:2]
    send_target_w = BALL_TRACK_REFINE_SEND_W
    if crop_w > 0 and crop_w < send_target_w:
        scale_up = send_target_w / float(crop_w)
        send_w = send_target_w
        send_h = int(round(crop_h * scale_up))
        send_crop = cv2.resize(
            crop, (send_w, send_h), interpolation=cv2.INTER_CUBIC,
        )
    else:
        send_crop = crop
        send_w = crop_w
        send_h = crop_h
    ok, buf = cv2.imencode(".jpg", send_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return idx, {"_error": "encode failed"}
    b64 = base64.standard_b64encode(bytes(buf)).decode("ascii")
    user_text = (
        f"Frame {idx} — REFINEMENT CROP. {send_w}x{send_h} px image "
        f"(a {crop_w}x{crop_h} px native region centered on the ball's "
        f"previously-identified position, upscaled for clarity). The "
        f"ball should be near the center but may not be exactly centered. "
        f"Return (x, y) in THIS image's coordinate system "
        f"(top-left = 0,0, width {send_w}, height {send_h}). JSON only."
    )
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=200,
            system=[{
                "type": "text",
                "text": BALL_TRACK_REFINE_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": user_text},
                ],
            }],
        )
    except Exception as exc:
        return idx, {"_error": str(exc)}
    text_chunks = [
        b.text for b in resp.content
        if getattr(b, "type", None) == "text" and getattr(b, "text", None)
    ]
    parsed = _extract_json("\n".join(text_chunks)) or {
        "_error": "no_json_in_response",
    }
    if (
        "_error" not in parsed and parsed.get("found")
        and parsed.get("x") is not None and parsed.get("y") is not None
    ):
        try:
            send_x = float(parsed["x"])
            send_y = float(parsed["y"])
        except (TypeError, ValueError):
            pass
        else:
            crop_x = send_x * (crop_w / float(send_w)) if send_w > 0 else send_x
            crop_y = send_y * (crop_h / float(send_h)) if send_h > 0 else send_y
            parsed["x"] = int(round(crop_x + x0))
            parsed["y"] = int(round(crop_y + y0))
            parsed["_native_coords"] = True
    return idx, parsed


def track_ball_after_impact(
    input_path: Path,
    impact_frame_idx: int,
    output_dir: Path,
    output_prefix: str,
    ball_xy_sent: tuple[float, float] | None = None,
    ball_sent_dims: tuple[int, int] | None = None,
    max_frames: int | None = None,
    send_width: int = BALL_TRACK_FRAME_W,
    concurrency: int = BALL_TRACK_CONCURRENCY,
    model: str | None = None,
) -> dict:
    """Track the ball forward frame-by-frame from `impact_frame_idx`.

    Two-phase strategy:
      Phase 1 — fan out up to `max_frames` parallel Claude calls, each
        asking for the ball's pixel coordinates on one frame. If the
        ball's at-rest position from address is supplied via
        ball_xy_sent + ball_sent_dims, it's passed as an initial hint
        (helpful on the first few frames where the ball is still near
        rest).
      Phase 2 — for every frame Claude couldn't find, look at the
        nearest neighbor where it WAS found. Re-query that frame in
        parallel with the neighbor position as a "look near here"
        hint. The ball doesn't move much frame-to-frame, so a rough
        anchor usually rescues a missed call.

    Every frame in the window gets a JPEG written to
    `output_dir / {output_prefix}_f{N}.jpg` — annotated with a yellow
    highlight ring when Claude found the ball, plain otherwise — so
    the operator can see exactly what was queried.

    Returns::

        {
          "ok": bool,
          "error": str | None,
          "frames": [
            {
              "frame": int,
              "found": bool,
              "x": int | None,    # native pixel x (None if not found)
              "y": int | None,    # native pixel y
              "confidence": str | None,
              "notes": str | None,
              "image_filename": str | None,    # always set if frame readable
              "retry": bool,                   # True iff found via the hint pass
            },
            ...
          ],
          "n_frames_processed": int,
          "n_frames_found": int,
          "n_frames_found_via_retry": int,
          "first_lost_run_start": int | None,
          "model": str,
        }
    """
    model = _resolve_model(model)
    info: dict = {
        "ok": False,
        "error": None,
        "frames": [],
        "n_frames_processed": 0,
        "n_frames_found": 0,
        "n_frames_found_via_retry": 0,
        "first_lost_run_start": None,
        "model": model,
    }

    if not HAS_CV:
        info["error"] = "opencv not installed"
        return info
    if not HAS_ANTHROPIC:
        info["error"] = "anthropic SDK not installed"
        return info
    if not os.environ.get("ANTHROPIC_API_KEY"):
        info["error"] = "ANTHROPIC_API_KEY not set in environment"
        return info

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        info["error"] = "could not open video"
        return info
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        clip_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    finally:
        cap.release()
    if total_frames <= 1:
        info["error"] = "video has no frames"
        return info

    # Resolve frame budget + stride from clip fps. Goal: roughly the
    # same wall-clock window of flight covered by ~20-40 Claude calls
    # regardless of source fps. The STRIDE is fps-derived even when the
    # caller overrides max_frames — a supplied max_frames used to force
    # stride 1, which bunched all the calls into consecutive frames right
    # at the launch (0.4s) instead of spreading them over the flight.
    if clip_fps > BALL_TRACK_VHIGH_FPS_THRESHOLD:
        stride = BALL_TRACK_VHIGH_FPS_STRIDE
        _default_max = BALL_TRACK_MAX_FRAMES_HIGH_FPS
    elif clip_fps >= BALL_TRACK_HIGH_FPS_THRESHOLD:
        stride = 1
        _default_max = BALL_TRACK_MAX_FRAMES_HIGH_FPS
    else:
        # <50 fps: 12 data points sampled every THIRD frame (stride 3),
        # so the same 12 Claude calls span 36 frames (~1.2s at 30fps).
        stride = BALL_TRACK_LOW_FPS_STRIDE
        _default_max = BALL_TRACK_MAX_FRAMES
    if max_frames is None:
        max_frames = _default_max
    log.info(
        "ai_tracer: ball_track — fps=%.1f → max_frames=%d, stride=%d "
        "(spanning %d frames)",
        clip_fps, max_frames, stride, max_frames * stride,
    )

    impact_frame_idx = int(impact_frame_idx)
    span = max_frames * stride
    last_frame = min(total_frames - 1, impact_frame_idx + span - 1)
    frame_indices = list(range(impact_frame_idx, last_frame + 1, stride))
    if not frame_indices:
        info["error"] = "no frames to process"
        return info
    info["n_frames_processed"] = len(frame_indices)
    log.info(
        "ai_tracer: ball_track — tracking %d frames [%d..%d] from impact",
        len(frame_indices), frame_indices[0], frame_indices[-1],
    )

    # Compute a "ball flight zone" ROI from the at-rest ball position
    # passed in by the caller. The ball arcs UP and across the frame
    # after impact, so anything below the ball's resting y AND outside
    # a generous horizontal band around the rest x is irrelevant —
    # cropping it out before sending each frame to Claude removes a
    # ton of distractor whites (range balls on the ground, shoes,
    # baskets, other golfers off to the side) and increases the ball's
    # pixel footprint as a fraction of the image. We don't crop side-
    # of-flight too aggressively because hooks/slices can drift.
    ball_xy_native: tuple[float, float] | None = None
    if (
        ball_xy_sent is not None and ball_sent_dims is not None
        and ball_sent_dims[0] > 0 and ball_sent_dims[1] > 0
    ):
        # Need the native frame dims; cheapest probe is one extra cap.
        _cap = cv2.VideoCapture(str(input_path))
        try:
            if _cap.isOpened():
                _nw = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                _nh = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if _nw > 0 and _nh > 0:
                    ball_xy_native = (
                        float(ball_xy_sent[0]) * _nw / float(ball_sent_dims[0]),
                        float(ball_xy_sent[1]) * _nh / float(ball_sent_dims[1]),
                    )
        finally:
            _cap.release()

    roi: tuple[int, int, int, int] | None = None  # (x0, y0, x1, y1) in native px

    # Extract all frames once up front: keep both a resized JPEG (for
    # the API) and the native ndarray (for drawing the highlight when
    # the ball is found, without a second cv2.VideoCapture pass).
    # Also stash the per-frame native dims so retry / coord-translation
    # downstream can map sent-coords back to native through the ROI.
    frames_data: dict[int, tuple[bytes, int, int, "np.ndarray"]] = {}
    clahe_count = 0  # how many Phase-1 frames triggered the CLAHE boost
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        info["error"] = "could not re-open video"
        return info
    try:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            native_h, native_w = frame.shape[:2]
            # Lazily compute the ROI on the first successful frame
            # (we needed at least one read to know native dims for sure).
            if roi is None and ball_xy_native is not None:
                bx = int(round(float(ball_xy_native[0])))
                by = int(round(float(ball_xy_native[1])))
                # Horizontal: ±30% of frame width around ball x — gives
                # 60% of the frame as a centered band on the ball flight
                # zone. Tight enough to remove the side baskets / off-
                # frame golfers / parked carts but wide enough to handle
                # typical hooks and slices. Vertical: from image top
                # down to ball_y + 60 (buffer for the very first
                # impact frame where the ball might still be at rest,
                # plus margin for the highlight ring).
                half_w = int(native_w * 0.30)
                buffer_below = 60
                roi = (
                    max(0, bx - half_w),
                    0,
                    min(native_w, bx + half_w),
                    min(native_h, by + buffer_below),
                )
                log.info(
                    "ai_tracer: ball_track ROI = (%d,%d)→(%d,%d) "
                    "(%.0f%%×%.0f%% of %dx%d)",
                    roi[0], roi[1], roi[2], roi[3],
                    100 * (roi[2] - roi[0]) / native_w,
                    100 * (roi[3] - roi[1]) / native_h,
                    native_w, native_h,
                )
            # Apply ROI crop. The cropped region is what Claude sees;
            # we stash the FULL native frame for annotation later, and
            # the ROI offset (via x0/y0) is recovered from `roi` when
            # translating coords back.
            if roi is not None:
                x0, y0, x1, y1 = roi
                api_frame = frame[y0:y1, x0:x1]
            else:
                api_frame = frame
            # Adaptive CLAHE: rescues low-contrast (overcast) frames
            # where the white ball blends into a flat grey sky. No-op
            # on normal-contrast images so it's safe to leave on.
            api_frame, clahe_fired = _maybe_apply_clahe(api_frame)
            if clahe_fired:
                clahe_count += 1
            api_h, api_w = api_frame.shape[:2]
            if api_w > send_width:
                scale = send_width / float(api_w)
                resized = cv2.resize(
                    api_frame, (send_width, int(round(api_h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            elif api_w < send_width:
                # Up-sample to send_width — the ball is small in pixel
                # area; giving Claude a higher-res image lets the model
                # spend more vision tiles on the ball region.
                scale = send_width / float(api_w)
                resized = cv2.resize(
                    api_frame, (send_width, int(round(api_h * scale))),
                    interpolation=cv2.INTER_CUBIC,
                )
            else:
                resized = api_frame
            ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
            if ok:
                # Store sent dims (for coord translation) + native frame
                # (for native-res annotation).
                frames_data[idx] = (
                    bytes(buf),
                    resized.shape[1], resized.shape[0],
                    frame.copy(),
                )
    finally:
        cap.release()
    if not frames_data:
        info["error"] = "could not extract any frames"
        return info

    client = _anthropic_client()

    # Pre-compute the ball-at-rest hint in each frame's sent-coord space
    # so we can include it as initial context on Phase 1 calls. The
    # handedness pass recorded ball position at HANDEDNESS_FRAME_W; the
    # track pass sends at BALL_TRACK_FRAME_W. We scale via fractions to
    # avoid having to know the exact sent dimensions for each frame.
    rest_hint_fraction: tuple[float, float] | None = None
    if (
        ball_xy_sent is not None and ball_sent_dims is not None
        and ball_sent_dims[0] > 0 and ball_sent_dims[1] > 0
    ):
        rest_hint_fraction = (
            float(ball_xy_sent[0]) / float(ball_sent_dims[0]),
            float(ball_xy_sent[1]) / float(ball_sent_dims[1]),
        )

    def _hint_for_rest(sent_w: int, sent_h: int) -> str:
        if rest_hint_fraction is None:
            return ""
        hx = int(round(rest_hint_fraction[0] * sent_w))
        hy = int(round(rest_hint_fraction[1] * sent_h))
        return (
            f" The ball started at rest at approximately ({hx}, {hy}); "
            "soon after impact it is still near there, then arcs up and "
            "to one side as it flies."
        )

    def _query(idx: int, hint_text: str = "") -> tuple[int, dict]:
        jpeg_bytes, sent_w, sent_h, _native = frames_data[idx]
        b64 = base64.standard_b64encode(jpeg_bytes).decode("ascii")
        user_text = (
            f"Frame {idx}. Image size {sent_w}x{sent_h} px."
            f"{hint_text} Locate the airborne golf ball. JSON only."
        )
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=200,
                system=[{
                    "type": "text",
                    "text": BALL_TRACK_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }],
            )
        except Exception as exc:
            log.warning("ai_tracer: ball_track frame %d API failed: %s", idx, exc)
            return idx, {"_error": str(exc)}
        text_chunks = [
            b.text for b in resp.content
            if getattr(b, "type", None) == "text" and getattr(b, "text", None)
        ]
        parsed = _extract_json("\n".join(text_chunks))
        return idx, (parsed or {"_error": "no_json_in_response"})

    # --- Phase 1: parallel pass with the rest-position hint ---
    phase1_results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = []
        for idx in frames_data:
            _jb, sw, sh, _nf = frames_data[idx]
            futures.append(ex.submit(_query, idx, _hint_for_rest(sw, sh)))
        for fut in as_completed(futures):
            try:
                idx, parsed = fut.result()
                phase1_results[idx] = parsed
            except Exception as exc:
                log.warning("ai_tracer: ball_track phase-1 exception: %s", exc)

    # Extract sent-coord positions of frames found in Phase 1 — used as
    # hint anchors for Phase 2 retries on missed frames.
    found_sent: dict[int, tuple[int, int, int, int]] = {}
    for idx, parsed in phase1_results.items():
        if "_error" in parsed or not parsed.get("found"):
            continue
        sx, sy = parsed.get("x"), parsed.get("y")
        if sx is None or sy is None:
            continue
        try:
            sent_x, sent_y = int(sx), int(sy)
        except (TypeError, ValueError):
            continue
        _jb, sw, sh, _nf = frames_data[idx]
        found_sent[idx] = (sent_x, sent_y, sw, sh)
    log.info(
        "ai_tracer: ball_track phase-1 — found %d / %d frames",
        len(found_sent), len(frames_data),
    )

    # --- Phase 2: crop-and-zoom retry on missed frames ---
    # For each frame Phase 1 missed, predict the ball's position via
    # linear interpolation through the nearest found neighbors, crop a
    # ~600 px square out of the native frame centered on that
    # prediction, and ask Claude to find the ball in just that crop.
    # Going from a 1568×N image with a 5-px ball to a 600×600 crop
    # with the ball near center turns "needle in a haystack" into a
    # routine vision task.
    BALL_TRACK_CROP_SIZE = BALL_TRACK_CROP_NATIVE_SIZE
    retry_results: dict[int, dict] = {}
    retry_targets = [idx for idx in frames_data if idx not in found_sent]
    if found_sent and retry_targets:
        # Cost guard: only retry frames inside the [first_found, last_found]
        # window — beyond that the ball has clearly left the frame and the
        # retry burns tokens for no signal. Also cap total retries at
        # max(3, half of phase-1 hits) so phase 2's spend stays a small
        # fraction of phase 1's. Pick the interior gaps closest to a found
        # neighbor first since they're most likely to succeed.
        _found_keys = sorted(found_sent.keys())
        _first_found, _last_found = _found_keys[0], _found_keys[-1]
        retry_targets = [
            idx for idx in retry_targets
            if _first_found <= idx <= _last_found
        ]
        _retry_cap = max(3, len(found_sent) // 2)
        if len(retry_targets) > _retry_cap:
            def _gap_score(idx: int) -> int:
                return min(abs(idx - f) for f in _found_keys)
            retry_targets.sort(key=_gap_score)
            retry_targets = retry_targets[:_retry_cap]
            retry_targets.sort()
        log.info(
            "ai_tracer: ball_track phase-2 — capped %d candidate retries to %d "
            "(found=%d, range=[%d, %d])",
            sum(1 for idx in frames_data if idx not in found_sent),
            len(retry_targets), len(found_sent), _first_found, _last_found,
        )
    if found_sent and retry_targets:
        # Resolve every Phase-1 found position to NATIVE coords once,
        # so prediction math is in a single coordinate system.
        found_native: list[tuple[int, tuple[int, int]]] = []
        for fidx, (sx, sy, sw, sh) in found_sent.items():
            _jb, _isw, _ish, nf = frames_data[fidx]
            nh, nw = nf.shape[:2]
            found_native.append((
                fidx,
                (
                    int(round(sx * nw / float(sw))),
                    int(round(sy * nh / float(sh))),
                ),
            ))
        found_native.sort(key=lambda t: t[0])

        def _predict_native(target_idx: int) -> tuple[int, int] | None:
            """Predict the ball's native pixel position at `target_idx`.

            With ≥3 found points, fit a quadratic separately in x and y
            against frame index — the ball flies on a parabolic arc so
            this hugs the curve much more tightly than linear interp,
            especially near the apex. With 2 points, linear interpolate
            (clamped to the segment between them). With 1 point, return
            it. With 0, return None.
            """
            if not found_native:
                return None
            if len(found_native) >= 3 and HAS_NP:
                try:
                    frames = np.array([t[0] for t in found_native], dtype=float)
                    xs = np.array([t[1][0] for t in found_native], dtype=float)
                    ys = np.array([t[1][1] for t in found_native], dtype=float)
                    px = np.polyfit(frames, xs, 2)
                    py = np.polyfit(frames, ys, 2)
                    return (
                        int(round(float(np.polyval(px, target_idx)))),
                        int(round(float(np.polyval(py, target_idx)))),
                    )
                except Exception as exc:
                    log.warning(
                        "ai_tracer: parabolic prediction failed (%s), "
                        "falling back to linear", exc,
                    )
            # Fall through: 1-2 points, no numpy, or polyfit failed.
            prev_n = None
            next_n = None
            for fidx, fpos in found_native:
                if fidx < target_idx:
                    prev_n = (fidx, fpos)
                elif fidx > target_idx:
                    next_n = (fidx, fpos)
                    break
            if prev_n and next_n:
                pidx, (px_, py_) = prev_n
                nidx, (nx_, ny_) = next_n
                if nidx == pidx:
                    return (px_, py_)
                t = (target_idx - pidx) / float(nidx - pidx)
                return (
                    int(round(px_ + t * (nx_ - px_))),
                    int(round(py_ + t * (ny_ - py_))),
                )
            if prev_n:
                return prev_n[1]
            if next_n:
                return next_n[1]
            return None

        def _retry_crop(idx: int) -> tuple[int, dict]:
            pred = _predict_native(idx)
            if pred is None:
                return idx, {"_error": "no neighbor for prediction"}
            _jb, _sw, _sh, native_frame = frames_data[idx]
            nh, nw = native_frame.shape[:2]
            half = BALL_TRACK_CROP_SIZE // 2
            x0 = max(0, pred[0] - half)
            y0 = max(0, pred[1] - half)
            x1 = min(nw, pred[0] + half)
            y1 = min(nh, pred[1] + half)
            if x1 <= x0 or y1 <= y0:
                return idx, {"_error": "empty crop"}
            crop = native_frame[y0:y1, x0:x1]
            # Same adaptive contrast boost as the Phase-1 ROI: helps
            # the ball pop on overcast / flat-lit clips.
            crop, _ = _maybe_apply_clahe(crop)
            crop_h, crop_w = crop.shape[:2]
            # Upscale the crop before sending so the ball occupies a
            # larger pixel footprint AND Anthropic processes it with
            # more vision tiles (~3 instead of ~1). Same content, more
            # attention budget spent on it. Coords returned will be in
            # the upscaled image space — convert back below.
            send_target_w = BALL_TRACK_CROP_SEND_W
            if crop_w > 0 and crop_w < send_target_w:
                scale_up = send_target_w / float(crop_w)
                send_w = send_target_w
                send_h = int(round(crop_h * scale_up))
                send_crop = cv2.resize(
                    crop, (send_w, send_h), interpolation=cv2.INTER_CUBIC,
                )
            else:
                send_crop = crop
                send_w = crop_w
                send_h = crop_h
            ok, buf = cv2.imencode(".jpg", send_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                return idx, {"_error": "crop encode failed"}
            b64 = base64.standard_b64encode(bytes(buf)).decode("ascii")
            user_text = (
                f"Frame {idx} — ZOOMED CROP. You are seeing a "
                f"{send_w}x{send_h} px image (a {crop_w}x{crop_h} region "
                f"of a {nw}x{nh} native frame, upscaled for clarity), "
                f"centered on the ball's predicted position. The ball "
                "should be NEAR THE CENTER of this image. The "
                "system-prompt rule about \"upper portion of the image\" "
                "does not apply here — find the ball wherever it is in "
                "this image. Return (x, y) in THIS image's coordinate "
                f"system (top-left = 0,0, width {send_w}, height {send_h}). "
                "JSON only."
            )
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=200,
                    system=[{
                        "type": "text",
                        "text": BALL_TRACK_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }],
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": user_text},
                        ],
                    }],
                )
            except Exception as exc:
                log.warning(
                    "ai_tracer: ball_track phase-2 frame %d API failed: %s", idx, exc,
                )
                return idx, {"_error": str(exc)}
            text_chunks = [
                b.text for b in resp.content
                if getattr(b, "type", None) == "text" and getattr(b, "text", None)
            ]
            parsed = _extract_json("\n".join(text_chunks)) or {
                "_error": "no_json_in_response",
            }
            # Translate returned coords: upscaled-image space → native
            # crop space → native frame. Flag the record so the
            # downstream save loop skips the usual sent→native scaling
            # (we already did the math here).
            if (
                "_error" not in parsed and parsed.get("found")
                and parsed.get("x") is not None and parsed.get("y") is not None
            ):
                try:
                    send_x = float(parsed["x"])
                    send_y = float(parsed["y"])
                except (TypeError, ValueError):
                    pass
                else:
                    crop_x = send_x * (crop_w / float(send_w)) if send_w > 0 else send_x
                    crop_y = send_y * (crop_h / float(send_h)) if send_h > 0 else send_y
                    parsed["x"] = int(round(crop_x + x0))
                    parsed["y"] = int(round(crop_y + y0))
                    parsed["_native_coords"] = True
                    parsed["_crop"] = {
                        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                        "pred_x": pred[0], "pred_y": pred[1],
                        "send_w": send_w, "send_h": send_h,
                    }
            return idx, parsed

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(_retry_crop, idx) for idx in retry_targets]
            for fut in as_completed(futures):
                try:
                    idx, parsed = fut.result()
                    retry_results[idx] = parsed
                except Exception as exc:
                    log.warning("ai_tracer: ball_track phase-2 exception: %s", exc)
        n_retry_hits = sum(
            1 for p in retry_results.values()
            if "_error" not in p and p.get("found")
            and p.get("x") is not None and p.get("y") is not None
        )
        log.info(
            "ai_tracer: ball_track phase-2 retried %d frames; %d additional hits",
            len(retry_results), n_retry_hits,
        )

    # --- Save annotated images for every frame and build records ---
    output_dir.mkdir(parents=True, exist_ok=True)
    n_found = 0
    n_retry_found = 0
    consec_lost = 0
    first_lost_run_start: int | None = None
    for idx in sorted(frames_data):
        parsed1 = phase1_results.get(idx, {"_error": "missing"})
        parsed2 = retry_results.get(idx)
        # Prefer Phase 1 if it found the ball; else Phase 2 if it did.
        chosen = None
        via_retry = False
        if "_error" not in parsed1 and parsed1.get("found"):
            chosen = parsed1
        elif parsed2 is not None and "_error" not in parsed2 and parsed2.get("found"):
            chosen = parsed2
            via_retry = True
        elif "_error" not in parsed1:
            chosen = parsed1  # carries Claude's "not found" notes
        elif parsed2 is not None and "_error" not in parsed2:
            chosen = parsed2

        record: dict = {
            "frame": idx,
            "found": False,
            "x": None,
            "y": None,
            "confidence": None,
            "notes": None,
            "image_filename": None,
            "retry": False,
        }
        if chosen is not None:
            found_flag = bool(chosen.get("found"))
            record["found"] = found_flag
            record["confidence"] = str(chosen.get("confidence") or "").lower() or None
            record["notes"] = str(chosen.get("notes") or "")[:200] or None
            sx = chosen.get("x")
            sy = chosen.get("y")
            if found_flag and sx is not None and sy is not None:
                try:
                    sent_x = int(sx)
                    sent_y = int(sy)
                except (TypeError, ValueError):
                    sent_x = sent_y = None
                if sent_x is not None and sent_y is not None:
                    if chosen.get("_native_coords"):
                        # Phase 2 crop retries already translated to
                        # native-frame coords inside _retry_crop.
                        native_x = sent_x
                        native_y = sent_y
                    else:
                        # Phase 1: Claude saw a sent-w × sent-h image
                        # that may be a cropped ROI of the native
                        # frame. Translate sent → ROI → native:
                        #   roi-space x = sent_x * roi_w / sent_w
                        #   native x   = roi-space x + roi_x0
                        # When no ROI was applied, roi_x0/y0 = 0 and
                        # roi_w/h = native_w/h, which collapses to
                        # the previous direct sent → native scaling.
                        _jb, sw, sh, native_frame = frames_data[idx]
                        nh, nw = native_frame.shape[:2]
                        if roi is not None:
                            roi_x0, roi_y0, roi_x1, roi_y1 = roi
                            roi_w = roi_x1 - roi_x0
                            roi_h = roi_y1 - roi_y0
                        else:
                            roi_x0 = roi_y0 = 0
                            roi_w, roi_h = nw, nh
                        native_x = int(round(sent_x * roi_w / float(sw) + roi_x0))
                        native_y = int(round(sent_y * roi_h / float(sh) + roi_y0))
                    record["x"] = native_x
                    record["y"] = native_y
                    record["retry"] = via_retry

        # Always write a JPEG for this frame so the operator can see
        # what Claude was actually looking at, even when the ball was
        # not found. Annotated when found, plain otherwise. Also draw
        # the ROI rectangle so it's obvious which region of the full
        # frame was actually fed to Claude.
        _jb, sw, sh, native_frame = frames_data[idx]
        annotated = native_frame.copy()
        if roi is not None:
            rx0, ry0, rx1, ry1 = roi
            # Thin lime-green rectangle marking the cropped region.
            cv2.rectangle(annotated, (rx0, ry0), (rx1, ry1), (0, 0, 0), 4, cv2.LINE_AA)
            cv2.rectangle(annotated, (rx0, ry0), (rx1, ry1), (40, 220, 80), 2, cv2.LINE_AA)
        if record["found"] and record["x"] is not None and record["y"] is not None:
            ring_color = (0, 230, 255) if not record["retry"] else (255, 200, 0)
            cv2.circle(annotated, (record["x"], record["y"]), 22, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.circle(annotated, (record["x"], record["y"]), 20, ring_color, 4, cv2.LINE_AA)
            cv2.circle(annotated, (record["x"], record["y"]), 4, ring_color, -1, cv2.LINE_AA)
        filename = f"{output_prefix}_f{idx:05d}.jpg"
        out_path = output_dir / filename
        ok = cv2.imwrite(
            str(out_path), annotated,
            [int(cv2.IMWRITE_JPEG_QUALITY), 88],
        )
        if ok:
            record["image_filename"] = filename
            if record["found"]:
                n_found += 1
                if via_retry:
                    n_retry_found += 1

        info["frames"].append(record)
        if record["found"]:
            consec_lost = 0
        else:
            consec_lost += 1
            if consec_lost == 3 and first_lost_run_start is None:
                first_lost_run_start = idx - 2

    info["n_frames_found"] = n_found
    info["n_frames_found_via_retry"] = n_retry_found
    info["first_lost_run_start"] = first_lost_run_start
    info["n_frames_clahe_enhanced"] = clahe_count
    info["ok"] = True
    log.info(
        "ai_tracer: ball_track — found ball in %d/%d frames; first_lost_run_start=%s",
        n_found, len(frame_indices), first_lost_run_start,
    )

    # --- Phase 3: precision refinement on found frames ---
    # Phase 1/2 already located the ball "near" its true position, but
    # at 1568×N a 5-pixel ball is easy to land "in the vicinity" rather
    # than ON. Re-query each found frame with a 250-px native crop
    # centered on the reported position so the ball occupies a much
    # larger share of the image — Claude can then pinpoint exact
    # pixels. Refined position is only accepted when it lands within
    # BALL_TRACK_REFINE_MAX_DELTA_PX of the original; bigger jumps
    # usually mean Claude latched onto a distractor in the crop.
    if BALL_TRACK_REFINEMENT_ENABLED and client is not None:
        refine_targets: list[tuple[int, tuple[int, int]]] = []
        for rec in info["frames"]:
            if not rec.get("found"):
                continue
            idx = rec["frame"]
            if idx not in frames_data:
                continue
            x = rec.get("x")
            y = rec.get("y")
            if x is None or y is None:
                continue
            refine_targets.append((int(idx), (int(x), int(y))))
        if refine_targets:
            refine_results: dict[int, dict] = {}
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futures = []
                for idx, pos in refine_targets:
                    _jb, _sw, _sh, native_frame = frames_data[idx]
                    futures.append(ex.submit(
                        _refine_crop_call, idx, pos, native_frame,
                        BALL_TRACK_REFINE_CROP_NATIVE_SIZE, client, model,
                    ))
                for fut in as_completed(futures):
                    try:
                        idx, parsed = fut.result()
                        refine_results[idx] = parsed
                    except Exception as exc:
                        log.warning("ai_tracer: ball_track phase-3 exception: %s", exc)
            n_refined = 0
            n_refused = 0
            for rec in info["frames"]:
                idx = rec["frame"]
                refined = refine_results.get(idx)
                if not refined or "_error" in refined or not refined.get("found"):
                    continue
                new_x = refined.get("x")
                new_y = refined.get("y")
                if new_x is None or new_y is None:
                    continue
                try:
                    new_x = int(new_x); new_y = int(new_y)
                except (TypeError, ValueError):
                    continue
                dx = abs(new_x - rec["x"])
                dy = abs(new_y - rec["y"])
                if dx > BALL_TRACK_REFINE_MAX_DELTA_PX or dy > BALL_TRACK_REFINE_MAX_DELTA_PX:
                    n_refused += 1
                    continue
                rec["x"] = new_x
                rec["y"] = new_y
                rec["refined"] = True
                n_refined += 1
            info["n_frames_refined"] = n_refined
            info["n_frames_refine_refused"] = n_refused
            log.info(
                "ai_tracer: ball_track phase-3 — refined %d / refused %d "
                "of %d found frames",
                n_refined, n_refused, len(refine_targets),
            )

    return info


# Visual style for the final tracer-overlay render. Emulates the blue
# TV broadcast shot-tracer (Toptracer / ProTracer): a bright azure core
# stroke wrapped in a soft, blurred outer glow, tapering slightly toward
# the apex. Colors are BGR (OpenCV order).
TRACER_CORE_BGR = (255, 110, 40)         # bright broadcast blue (RGB 40,110,255)
TRACER_GLOW_BGR = (255, 170, 95)         # lighter blue, feeds the soft glow
TRACER_INNER_BGR = (255, 205, 150)       # pale-azure hot inner highlight
# Legacy aliases (kept so any external reference still resolves). The
# broadcast renderer no longer uses these directly.
TRACER_LINE_COLOR = TRACER_CORE_BGR
TRACER_LINE_HALO = (120, 40, 10)
# Tapered thickness: thicker at the resting-ball end of the line,
# narrower toward the apex / end-of-flight, so the line visually
# narrows as the ball flies away from the camera (broadcast tracer
# convention). Now scaled to frame height inside the renderer; these
# remain as the default taper ratio anchors.
TRACER_LINE_THICKNESS_START = 10
TRACER_LINE_THICKNESS_END = 1
TRACER_LINE_THICKNESS = TRACER_LINE_THICKNESS_START  # legacy alias
# Dash/gap constants kept for the classical tracer path; the AI
# editing-wizard renderer is now a continuous solid line so these
# are unused there.
TRACER_DASH_LEN = 14
TRACER_GAP_LEN = 10
TRACER_BALL_RING = (0, 230, 255)         # yellow — ball at current frame
TRACER_REST_RING = (255, 60, 0)          # blue — ball-at-rest marker

# Trajectory smoothing: a golf ball flies on a parabola, so we fit a
# quadratic in y vs frame and a line in x vs frame to the anchor
# points (rest position + every per-frame detection), then sample
# that fit at every integer frame to get the rendered tracer line.
# Anchors with residual > TRAJ_OUTLIER_PX after fit are dropped as
# misidentifications and the curve refits without them.
TRAJ_OUTLIER_PX = 80
TRAJ_OUTLIER_MAX_ITERS = 6
# How many frames before the impact frame the rest-position anchor
# sits. Operator wanted the tracer line to literally start at the
# ball-at-rest position a beat before the swing strikes, so the
# rendered curve begins 2 frames before impact instead of at impact.
REST_ANCHOR_FRAMES_BEFORE_IMPACT = 2


def _robust_quadratic_fit(
    anchors: list[tuple[int, int, int]],
    threshold_px: float = TRAJ_OUTLIER_PX,
    max_iters: int = TRAJ_OUTLIER_MAX_ITERS,
    pinned_indices: set[int] | None = None,
    weights: list[float] | None = None,
):
    """Fit y = a·f² + b·f + c and x = m·f + k to `anchors`
    (list of (frame, x, y)), iteratively dropping the anchor with the
    largest residual until every kept residual is ≤ threshold_px or
    fewer than 3 anchors remain.

    `pinned_indices` (e.g. {0} for the rest position) are excluded
    from outlier rejection — they're ground-truth anchors that the fit
    should always respect, even if their residual is large. They still
    contribute to the fit; they just can't be thrown out.

    `weights`, if provided, must be the same length as `anchors`. Used
    as the `w=` parameter to numpy's polyfit so high-confidence anchors
    can pull the curve toward themselves. Pinned anchors are typically
    weighted heavily so the fit passes through (or very near) them.

    Returns (x_coef, y_coef, rejected_indices_set) on success, or
    None when fewer than 3 anchors are usable or numpy is missing.
    """
    if not HAS_NP or len(anchors) < 3:
        return None
    pinned: set[int] = set(pinned_indices) if pinned_indices else set()
    rejected: set[int] = set()
    last_coefs = None
    for _ in range(max_iters):
        kept_idxs = [i for i in range(len(anchors)) if i not in rejected]
        if len(kept_idxs) < 3:
            break
        frames = np.array([anchors[i][0] for i in kept_idxs], dtype=float)
        xs = np.array([anchors[i][1] for i in kept_idxs], dtype=float)
        ys = np.array([anchors[i][2] for i in kept_idxs], dtype=float)
        if weights is not None:
            ws = np.array(
                [weights[i] for i in kept_idxs], dtype=float,
            )
            # polyfit chokes on all-zero weights; clamp to tiny positive
            ws = np.where(ws > 0, ws, 1e-3)
        else:
            ws = None
        try:
            y_coef = np.polyfit(frames, ys, 2, w=ws)
            # x quadratic on longer tracks: a full flight (ascent + apex
            # + descent) often reverses x direction in image coords —
            # forcing a straight-line x made the fit reject the descent
            # points as outliers. Short tracks keep the stabler linear x.
            x_deg = 2 if len(kept_idxs) >= 8 else 1
            x_coef = np.polyfit(frames, xs, x_deg, w=ws)
        except Exception:
            return None
        last_coefs = (x_coef, y_coef)
        x_pred = np.polyval(x_coef, frames)
        y_pred = np.polyval(y_coef, frames)
        residuals = np.sqrt((xs - x_pred) ** 2 + (ys - y_pred) ** 2)
        # Find the worst residual among NON-pinned anchors only — a
        # pinned anchor (the rest position) is allowed to have a large
        # residual without being thrown out.
        worst_local = -1
        worst_residual = -1.0
        for j, idx in enumerate(kept_idxs):
            if idx in pinned:
                continue
            r = float(residuals[j])
            if r > worst_residual:
                worst_residual = r
                worst_local = j
        if worst_local < 0:
            # All remaining anchors are pinned (or below threshold).
            return last_coefs[0], last_coefs[1], rejected
        if worst_residual > threshold_px and len(kept_idxs) > 3:
            rejected.add(kept_idxs[worst_local])
            continue
        return last_coefs[0], last_coefs[1], rejected
    # Loop fell out without converging — return the last successful fit
    # if it exists and we still have ≥3 anchors.
    kept_after = [i for i in range(len(anchors)) if i not in rejected]
    if last_coefs is None or len(kept_after) < 3:
        return None
    return last_coefs[0], last_coefs[1], rejected


def _draw_dashed_tracer(
    img,
    points: list[tuple[int, int]],
    *,
    total_points: int | None = None,
    start_thickness: int = TRACER_LINE_THICKNESS_START,
    end_thickness: int = TRACER_LINE_THICKNESS_END,
) -> None:
    """Draw a solid tapered polyline through `points` with a halo
    behind it. No-op when fewer than 2 points are provided.

    Despite the legacy name, this renderer is no longer dashed — the
    operator preferred a continuous solid line so the tracer reads
    as one stroke instead of a chain. Thickness linearly tapers from
    `start_thickness` at the first point (the resting ball) to
    `end_thickness` at the last (apex / end of flight). `total_points`
    is the FULL final length of the polyline — passing it keeps the
    per-frame growth tapered against the same reference span so
    early frames don't briefly render as a thin stub before the full
    taper develops.
    """
    if len(points) < 2:
        return
    h, w = img.shape[:2]
    full_n = total_points if (total_points and total_points > 1) else len(points)

    # Core stroke weight scales with frame height so the tracer reads the
    # same bold width at any resolution (broadcast lines are heavy). It
    # tapers from ~1.35x near the ball to ~0.5x at the apex so the line
    # thins as the ball flies away — the classic TV-tracer cue.
    base = max(3.5, h * 0.008)

    def _core_t(i: int) -> int:
        frac = max(0.0, min(1.0, i / float(full_n - 1)))
        return max(2, int(round(base * (1.50 + (0.50 - 1.50) * frac))))

    pts_i = [(int(round(px)), int(round(py))) for px, py in points]
    xs = [p[0] for p in pts_i]
    ys = [p[1] for p in pts_i]

    # --- Soft outer glow ---------------------------------------------------
    # Draw the stroke thick on a padded ROI layer, Gaussian-blur it, then
    # screen-blend it back onto the frame. Screen (rather than a straight
    # add) keeps a bright sky from blowing out to white while still laying
    # a luminous blue halo over darker trees / grandstands. Working on a
    # cropped ROI keeps the per-frame blur cheap.
    if HAS_NP:
        glow_pad = int(base * 8) + 12
        x0 = max(0, min(xs) - glow_pad)
        y0 = max(0, min(ys) - glow_pad)
        x1 = min(w, max(xs) + glow_pad)
        y1 = min(h, max(ys) + glow_pad)
        if x1 > x0 and y1 > y0:
            roi = img[y0:y1, x0:x1]
            glow = np.zeros_like(roi)
            shifted = [(p[0] - x0, p[1] - y0) for p in pts_i]
            for i in range(1, len(shifted)):
                cv2.line(
                    glow, shifted[i - 1], shifted[i],
                    TRACER_GLOW_BGR, max(4, int(round(_core_t(i) * 2.8))),
                    cv2.LINE_AA,
                )
            k = max(5, (int(base * 4) | 1))  # odd kernel for GaussianBlur
            glow = cv2.GaussianBlur(glow, (k, k), 0)
            # Screen blend: out = 255 - (255-roi)(255-glow)/255. Done with
            # cv2 ops (bitwise_not = 255-x for uint8) so it's dtype-safe
            # regardless of numpy's scalar-casting rules.
            inv = cv2.multiply(
                cv2.bitwise_not(roi), cv2.bitwise_not(glow),
                scale=1.0 / 255.0,
            )
            cv2.bitwise_not(inv, dst=roi)
    else:
        # numpy-less fallback: a couple of translucent-looking passes so
        # the line still gets a halo, just without the blurred glow.
        for i in range(1, len(pts_i)):
            cv2.line(
                img, pts_i[i - 1], pts_i[i],
                TRACER_GLOW_BGR, _core_t(i) + 6, cv2.LINE_AA,
            )

    # --- Crisp core on top -------------------------------------------------
    # Bright blue body, then a thin pale-azure highlight down the centre so
    # the stroke reads as a rounded glossy tube rather than a flat band.
    for i in range(1, len(pts_i)):
        cv2.line(
            img, pts_i[i - 1], pts_i[i],
            TRACER_CORE_BGR, _core_t(i), cv2.LINE_AA,
        )
    for i in range(1, len(pts_i)):
        it = max(1, int(round(_core_t(i) * 0.4)))
        cv2.line(
            img, pts_i[i - 1], pts_i[i],
            TRACER_INNER_BGR, it, cv2.LINE_AA,
        )


def render_tracer_video(
    input_path: Path,
    output_path: Path,
    ball_rest_xy_native: tuple[float, float] | None,
    impact_frame_idx: int,
    track_frames: list[dict],
    target_xy: tuple[float, float] | None = None,
    write_start: int | None = None,
    write_end: int | None = None,
) -> dict:
    """Render an MP4 of the source video with a progressive dashed
    tracer line overlaid.

    The line stops at the parabola's apex (its mathematical vertex)
    unless the last kept anchor sits visually below the apex by at
    least DESCENT_THRESHOLD_PX — the signal that the operator (or
    AI) actually confirmed the ball coming back down. Without that
    signal we'd rather end at the peak than draw a "phantom descent"
    produced by the parabola fit's vertex landing between marks the
    operator placed near the apex.

    The tracer:
      - Starts at the ball's at-rest position (anchored at the impact
        frame so the line begins right where the ball was struck).
      - Extends through each subsequent frame's found ball position.
      - Holds the last-known geometry on frames where the ball wasn't
        found, so the line doesn't flicker.
      - A small blue ring continuously marks the at-rest position so
        the operator can see where the ball started.
      - A yellow ring on the current frame marks the live ball location
        when the tracker has one.

    `track_frames` is the per-frame list from track_ball_after_impact:
    each entry has `frame`, `found`, and (when found) native pixel
    `x`/`y`.

    Returns::

        {
          "ok": bool,
          "error": str | None,
          "n_points": int,      # length of the rendered tracer line
          "frame_range": [int, int] | None,
          "fps": float | None,
          "saved_path": str | None,
        }

    The written file is raw mp4v from OpenCV; caller is responsible
    for transcoding to H.264 (e.g. via services.video.compress_for_email)
    before serving to browsers.
    """
    info: dict = {
        "ok": False,
        "error": None,
        "n_points": 0,
        "frame_range": None,
        "fps": None,
        "saved_path": None,
    }

    if not HAS_CV:
        info["error"] = "opencv not installed"
        return info

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        info["error"] = "could not open source video"
        return info
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    info["fps"] = float(fps)

    # Build {frame_idx: (x, y, is_manual)} for every successfully-located
    # ball. Manual entries are operator-confirmed: the fit treats them
    # as pinned ground truth (never rejected, heavily weighted).
    points_by_frame: dict[int, tuple[int, int, bool]] = {}
    for rec in track_frames or []:
        if not rec.get("found"):
            continue
        x = rec.get("x")
        y = rec.get("y")
        f = rec.get("frame")
        if f is None or x is None or y is None:
            continue
        try:
            points_by_frame[int(f)] = (int(x), int(y), bool(rec.get("manual", False)))
        except (TypeError, ValueError):
            continue

    # The full ordered list of tracer anchors: rest position first,
    # then every found-ball position in chronological order. We also
    # track which anchor indices came from manual edits so they can
    # be pinned + weighted in the fit.
    anchors: list[tuple[int, int, int]] = []  # (frame, x, y)
    manual_anchor_idxs: set[int] = set()
    rest_anchor_frame: int | None = None
    rest_added = False
    if ball_rest_xy_native is not None:
        rest_anchor_frame = max(
            0, int(impact_frame_idx) - REST_ANCHOR_FRAMES_BEFORE_IMPACT,
        )
        rx = int(round(float(ball_rest_xy_native[0])))
        ry = int(round(float(ball_rest_xy_native[1])))
        # Guard: the resting-ball anchor is pinned, weighted 10x, and the
        # rendered curve is forced to START exactly on it — so a mis-
        # detected rest (the AI latching onto a bright object off to the
        # side, e.g. a white sign / backboard / reflection) drags the whole
        # tracer origin there, with no per-frame card to reveal it. Only
        # trust the rest when it's spatially consistent with the real ball
        # track: the ball starts AT rest and the first captured detection is
        # right after impact, so rest should sit near the track. If it's far
        # from EVERY detected position, it's almost certainly wrong — drop it
        # and let the line begin at the first real detection instead.
        manual_pts = [(x, y) for (x, y, m) in points_by_frame.values() if m]
        ref_pts = manual_pts or [
            (x, y) for (x, y, _m) in points_by_frame.values()
        ]
        drop_rest = False
        if len(ref_pts) >= 2:
            diag = math.hypot(width, height) or 1.0
            # Generous threshold: a real ball can rise a long way before the
            # tracker first locks on, so only reject a rest that's WILDLY off
            # (a bright sign / backboard / reflection halfway across frame).
            max_off = 0.33 * diag
            nearest = min(math.hypot(rx - dx, ry - dy) for dx, dy in ref_pts)
            if nearest > max_off:
                drop_rest = True
                info["rest_anchor_dropped"] = {
                    "xy": [rx, ry],
                    "nearest_detection_px": round(float(nearest), 1),
                    "threshold_px": round(max_off, 1),
                }
                log.info(
                    "ai_tracer: dropped resting-ball anchor at (%d,%d) — "
                    "%.0fpx from nearest ball detection (> %.0f) — starting "
                    "line at the ball's launch point instead",
                    rx, ry, nearest, max_off,
                )
        if drop_rest:
            rest_anchor_frame = None  # no longer a valid anchor-0 rest
        else:
            anchors.append((rest_anchor_frame, rx, ry))
            rest_added = True
    # Manual points MERGE with the auto-detected ones: the operator's
    # clicks are pinned ground truth (never rejected, 10x weight in the
    # fit) but the auto points stay in — the common workflow is mapping
    # a few frames the detector missed (e.g. the blurred launch) while
    # keeping the detector's good arc. The old behaviour dropped EVERY
    # auto point the moment one manual existed, so two early clicks
    # rendered as a tiny spline through just those clicks and the whole
    # detected flight vanished from the tracer.
    has_manual = any(m for (_x, _y, m) in points_by_frame.values())
    n_auto_pts = sum(1 for (_x, _y, m) in points_by_frame.values() if not m)
    for f in sorted(points_by_frame):
        x, y, is_manual = points_by_frame[f]
        if is_manual:
            manual_anchor_idxs.add(len(anchors))
        anchors.append((f, x, y))

    # Fit a smooth parabola through the anchors with iterative outlier
    # rejection. The rendered line is sampled from the fit at every
    # integer frame — a single quadratic — so the result is always a
    # smooth parabola. A bad click (operator or AI) that breaks the
    # parabolic shape gets thrown out before it can yank the curve;
    # legitimate marks with small click jitter just blend into the
    # fit.
    smoothed_points: list[tuple[int, int, int]] = []  # (frame, x, y)
    last_kept_frame_global: int | None = None
    rejected_frames: set[int] = set()

    # When the operator has manually plotted the trajectory, draw the line
    # straight THROUGH their points (per-frame linear interpolation between
    # consecutive plotted anchors) instead of the parametric x=linear /
    # y=quadratic parabola. That model assumes a down-the-line shot whose x
    # moves monotonically; a ball hit up-and-over (x goes one way then back)
    # can't be represented as a straight-line x, so the descent rendered on
    # the wrong side of the golfer. Interpolating through the marks honours
    # exactly what was plotted — up AND down, on the correct side.
    # Manual-ONLY render (Catmull spline through the clicks) applies only
    # when the operator plotted the entire path themselves and there are
    # no auto points to blend with — e.g. a shot the detector missed
    # completely, or an up-and-over flight the parametric fit can't
    # represent. With auto points present, manual marks act as pinned
    # anchors inside the robust fit instead.
    manual_render = False
    if has_manual and n_auto_pts == 0 and len(anchors) >= 2:
        # Draw a SMOOTH Catmull-Rom spline through the plotted anchors.
        # It still passes through every plotted point, but curves smoothly
        # between them instead of connecting them with straight segments
        # (which read as a squiggly, kinked line).
        manual_render = True
        pts = sorted(anchors, key=lambda a: a[0])
        n = len(pts)

        def _catmull(p0, p1, p2, p3, t):
            # Standard Catmull-Rom basis; p1→p2 is the drawn segment,
            # p0/p3 are the neighbouring points that set its tangents.
            t2 = t * t
            t3 = t2 * t
            return 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )

        for i in range(n - 1):
            f1, x1, y1 = pts[i]
            f2, x2, y2 = pts[i + 1]
            _, x0, y0 = pts[i - 1] if i >= 1 else pts[i]
            _, x3, y3 = pts[i + 2] if i + 2 < n else pts[i + 1]
            span = f2 - f1
            if span <= 0:
                continue
            for ff in range(f1, f2):
                t = (ff - f1) / float(span)
                xi = int(round(_catmull(x0, x1, x2, x3, t)))
                yi = int(round(_catmull(y0, y1, y2, y3, t)))
                if 0 <= xi < width and 0 <= yi < height:
                    smoothed_points.append((ff, xi, yi))
        f_last, x_last, y_last = pts[-1]
        if 0 <= x_last < width and 0 <= y_last < height:
            smoothed_points.append((f_last, x_last, y_last))

        # Continue past the last plotted point toward the target/landing
        # spot on a smooth ballistic arc — but ONLY if a target was marked.
        # A golf shot flies away and lands downrange near the horizon, not
        # back in the foreground, so we aim the descent at the plotted
        # landing point instead of fabricating a flat parabola that dives
        # in front of the golfer. The curve leaves the last plotted point
        # along the ball's current direction, then bends to the target
        # (quadratic Bézier), so there's no kink/squiggle at the join. With
        # no target we simply stop at the last plotted point.
        if target_xy is not None and len(pts) >= 2:
            tx_t, ty_t = float(target_xy[0]), float(target_xy[1])
            _, x_a, y_a = pts[-1]
            _, x_b, y_b = pts[-2]
            dx, dy = float(x_a - x_b), float(y_a - y_b)
            dlen = math.hypot(dx, dy) or 1.0
            ux, uy = dx / dlen, dy / dlen
            span = math.hypot(tx_t - x_a, ty_t - y_a)
            cx = x_a + ux * span * 0.4
            cy = y_a + uy * span * 0.4
            steps = int(round(fps * 1.5)) if fps else 45
            for i in range(1, steps + 1):
                t = i / float(steps)
                mt = 1.0 - t
                bx = int(round(mt * mt * x_a + 2 * mt * t * cx + t * t * tx_t))
                by = int(round(mt * mt * y_a + 2 * mt * t * cy + t * t * ty_t))
                if not (0 <= bx < width and 0 <= by < height):
                    break
                smoothed_points.append((f_last + i, bx, by))

        last_kept_frame_global = (
            smoothed_points[-1][0] if smoothed_points else None
        )

    # The rest position AND every manual click are immune to outlier
    # rejection — the operator's plotted points are ground truth, so the
    # fitted curve is forced to keep (and hug) them instead of tossing
    # them as "outliers" relative to a fit the AI points had skewed.
    rest_is_anchor_zero = rest_added
    pinned_set: set[int] = set(manual_anchor_idxs)
    if rest_is_anchor_zero:
        pinned_set.add(0)
    if not pinned_set:
        pinned_set = None  # type: ignore[assignment]
    if len(anchors) > 1:
        # Rest = 10×, manual = 10×, AI-detected = 1×. The bumped
        # weight on the rest + manual marks keeps the fitted parabola
        # within a few pixels of every confirmed point instead of
        # averaging them out with the AI detections.
        weight_list = []
        for i in range(len(anchors)):
            if (rest_is_anchor_zero and i == 0) or i in manual_anchor_idxs:
                weight_list.append(10.0)
            else:
                weight_list.append(1.0)
    else:
        weight_list = None
    fit = (
        None if manual_render
        else _robust_quadratic_fit(
            anchors, pinned_indices=pinned_set, weights=weight_list,
        )
    )
    if fit is not None:
        x_coef, y_coef, rejected_indices = fit
        rejected_frames = {anchors[i][0] for i in rejected_indices}
        kept = [a for i, a in enumerate(anchors) if i not in rejected_indices]
        kept_indices = [
            i for i in range(len(anchors)) if i not in rejected_indices
        ]

        # Re-fit constrained to pass EXACTLY through the rest anchor.
        # The unconstrained weighted fit lands the parabola "near" the
        # rest (weight=10x) but not on it; pinning the first rendered
        # point to the exact rest pixel then created a visible kink
        # between that pinned point and polyval's prediction at the
        # next frame. Constraining the curve through the rest moves
        # the math, not the geometry: every rendered point comes from
        # polyval and the line starts exactly on the ball with no
        # discontinuity.
        if (
            rest_is_anchor_zero and HAS_NP and len(kept) >= 3
            and rest_anchor_frame is not None and 0 in kept_indices
            # Only when the main fit used LINEAR x — this refit rebuilds a
            # linear x and would clobber the quadratic x that long
            # (descent-including) tracks need.
            and len(x_coef) == 2
        ):
            try:
                f0 = float(rest_anchor_frame)
                x0_rest = float(anchors[0][1])
                y0_rest = float(anchors[0][2])
                kept_frames = np.array([k[0] for k in kept], dtype=float)
                kept_xs = np.array([k[1] for k in kept], dtype=float)
                kept_ys = np.array([k[2] for k in kept], dtype=float)
                if weight_list is not None:
                    kept_ws = np.array(
                        [weight_list[i] for i in kept_indices],
                        dtype=float,
                    )
                else:
                    kept_ws = np.ones(len(kept), dtype=float)
                kept_ws = np.where(kept_ws > 0, kept_ws, 1e-3)
                sqrt_w = np.sqrt(kept_ws)
                # Substitute c = y0 - a·f0² - b·f0 into y = a·f² + b·f + c
                # → (y - y0) = a·(f² - f0²) + b·(f - f0). Standard
                # weighted lstsq on a/b, then back out c.
                u_q = (kept_frames**2 - f0**2) * sqrt_w
                v_q = (kept_frames - f0) * sqrt_w
                A_q = np.column_stack([u_q, v_q])
                b_q = (kept_ys - y0_rest) * sqrt_w
                sol_q, *_ = np.linalg.lstsq(A_q, b_q, rcond=None)
                a_q = float(sol_q[0])
                b_q_coef = float(sol_q[1])
                c_q = y0_rest - a_q * f0 * f0 - b_q_coef * f0
                y_coef = np.array([a_q, b_q_coef, c_q])
                # Same substitution for the linear x fit:
                # x = m·f + k constrained by x0 = m·f0 + k.
                v_x = (kept_frames - f0) * sqrt_w
                b_x = (kept_xs - x0_rest) * sqrt_w
                sol_x, *_ = np.linalg.lstsq(
                    v_x[:, None], b_x, rcond=None,
                )
                m_x = float(sol_x[0])
                k_x = x0_rest - m_x * f0
                x_coef = np.array([m_x, k_x])
            except Exception as exc:
                log.warning(
                    "ai_tracer: rest-constrained refit failed (%s) — "
                    "falling back to unconstrained fit",
                    exc,
                )
        if kept:
            first_frame = kept[0][0]
            last_kept_frame = kept[-1][0]
            # Always originate the drawn line at the struck ball. When the
            # rest anchor was missing or dropped (or the AI's first ball
            # detection is already well into the flight), kept[0] sits up in
            # the air — extend the sampled range back to just before impact
            # so the fitted parabola is drawn from the ball's launch point on
            # the ground, not from wherever tracking first locked on. With a
            # valid rest anchor this is a no-op (kept[0] already sits there).
            launch_frame = max(
                0, int(impact_frame_idx) - REST_ANCHOR_FRAMES_BEFORE_IMPACT,
            )
            if not manual_render and launch_frame < first_frame:
                first_frame = launch_frame
            # Apex truncation: stop the line at the parabola's vertex
            # unless the operator marked CLEAR descent past it. In
            # image coords y grows downward, so the vertex of a > 0
            # is the visual peak.
            #
            # Even in wizard mode we apply this: when manual marks
            # plateau near the peak (several marks at similar y) the
            # math places the vertex BETWEEN them and the polyval
            # samples between those marks dip slightly past it on the
            # way back up — a "phantom descent" the operator didn't
            # mark. Detecting genuine descent: the last anchor sits
            # visually below the predicted apex by at least
            # DESCENT_THRESHOLD_PX. If so the operator confirmed the
            # ball coming down and we render through; if not we stop
            # at the apex frame.
            a_y = float(y_coef[0])
            b_y = float(y_coef[1])
            render_end = last_kept_frame
            DESCENT_THRESHOLD_PX = 30.0
            # Apex truncation is a safeguard for AUTO detections that
            # plateau near the peak. When the operator has manually
            # plotted the arc, they've explicitly told us where the ball
            # goes (up AND down) — render the whole plotted trajectory
            # and never truncate.
            if a_y > 1e-6 and not has_manual:
                apex_frame_f = -b_y / (2.0 * a_y)
                apex_y_predicted = float(np.polyval(y_coef, apex_frame_f))
                last_anchor_y = float(kept[-1][2])
                user_marked_descent = (
                    last_anchor_y > apex_y_predicted + DESCENT_THRESHOLD_PX
                )
                if (
                    not user_marked_descent
                    and first_frame < apex_frame_f < last_kept_frame
                ):
                    render_end = int(round(apex_frame_f))
            last_kept_frame_global = render_end
            # Render every frame at the parabola's prediction — one
            # smooth quadratic from rest to render_end. When the
            # rest-constrained refit above succeeded, polyval at the
            # first frame returns the exact rest pixel by construction,
            # so the line starts on the ball with no kink.
            #
            # SKIP (don't break on) points the parabola predicts outside
            # the frame: a shot that peaks at the very top edge dips
            # briefly off-screen near the apex, and breaking there killed
            # the entire line "on the way up". Skipping lets it draw up to
            # the edge and resume on the descent.
            for f in range(first_frame, render_end + 1):
                x = int(round(float(np.polyval(x_coef, f))))
                y = int(round(float(np.polyval(y_coef, f))))
                if x < 0 or x >= width or y < 0 or y >= height:
                    continue
                smoothed_points.append((f, x, y))
        log.info(
            "ai_tracer: tracer fit — %d anchors, %d rejected as outliers, "
            "%d smoothed render points (parabola)",
            len(anchors), len(rejected_indices), len(smoothed_points),
        )
    elif not manual_render:
        # Not enough anchors for a stable fit (or numpy missing).
        # Fall back to the raw point-to-point line. (Skipped in
        # manual_render mode — smoothed_points is already built by
        # interpolating through the operator's plotted points above.)
        smoothed_points = list(anchors)
        log.info(
            "ai_tracer: tracer — falling back to raw %d anchors (no fit)",
            len(anchors),
        )

    info["n_points"] = len(smoothed_points)
    if smoothed_points:
        info["frame_range"] = [int(smoothed_points[0][0]), int(smoothed_points[-1][0])]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        info["error"] = "VideoWriter failed to open output path"
        return info

    rest_xy: tuple[int, int] | None = None
    if ball_rest_xy_native is not None:
        rest_xy = (
            int(round(float(ball_rest_xy_native[0]))),
            int(round(float(ball_rest_xy_native[1]))),
        )

    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Output window: only WRITE frames in [write_start, write_end]
            # so a multi-swing / long source renders just the selected
            # swing's segment instead of the whole clip (fast + short clip).
            if write_end is not None and frame_idx > write_end:
                break
            if write_start is not None and frame_idx < write_start:
                frame_idx += 1
                continue
            # Draw the tracer once we've reached the impact frame.
            if smoothed_points and frame_idx >= smoothed_points[0][0]:
                visible = [
                    (f, x, y) for f, x, y in smoothed_points if f <= frame_idx
                ]
                # Split into contiguous-frame runs. Off-screen (above the
                # top edge) apex points are skipped during sampling, which
                # leaves a frame gap; drawing one polyline across that gap
                # bridged the clipped ascent-top to the descent-top with a
                # flat horizontal bar across the top of the screen. Break
                # the line at the gap instead so a shot that peaks at/above
                # the top edge shows the ascent and descent reaching the
                # edge rather than a flat plateau.
                runs: list[list[tuple[int, int]]] = []
                cur: list[tuple[int, int]] = []
                prev_f = None
                for f, x, y in visible:
                    if prev_f is not None and (f - prev_f) > 2:
                        if len(cur) >= 2:
                            runs.append(cur)
                        cur = []
                    cur.append((x, y))
                    prev_f = f
                if len(cur) >= 2:
                    runs.append(cur)
                for run in runs:
                    # Pass the full final point count so the taper rate
                    # stays consistent as the line builds frame-by-frame
                    # (otherwise early frames briefly look like a thin
                    # stub before the full taper develops).
                    _draw_dashed_tracer(
                        frame, run,
                        total_points=len(smoothed_points),
                    )
                # Resting-ball indicator at the origin removed too — operator
                # wants nothing but the smoothed tracer line in the final
                # production output. rest_xy is still threaded through the
                # pipeline for the debug image / future use.
            # Per-frame ball ring removed for production output. The
            # smoothed tracer polyline is enough on its own; the ring
            # used to "pop" frame-to-frame which read as visual noise.
            # The resting-ball indicator (drawn above on the first
            # render frame) and the debug-image overlays are untouched.
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    info["ok"] = True
    info["saved_path"] = str(output_path)
    info["n_outliers_rejected"] = len(rejected_frames)
    info["rejected_frames"] = sorted(rejected_frames)
    return info


# Sample rate used when extracting audio for impact detection. 22050 Hz
# gives us ~45 µs precision — plenty to resolve a club-on-ball
# transient (a few ms wide) to the right video frame even at 120 fps.
AUDIO_SAMPLE_RATE = 22050

# Smoothing window for the RMS envelope, ~10 ms — wide enough to
# average out one-sample crackle but narrow enough to preserve the
# sharp leading edge of the impact transient.
AUDIO_ENVELOPE_WINDOW_MS = 10

# Minimum peak-to-median ratio for us to trust the audio peak as a
# real impact. Below this we treat the audio as too noisy to
# identify a clear "thwack" and fall back to the AI vision path.
AUDIO_MIN_PEAK_OVER_MEDIAN = 25.0

# Minimum audio peak-to-median ratio used by the *swing detector*
# (detect_swings_from_audio / detect_swings_combined) when scanning a
# long recording for club-on-ball impacts. Lower = more sensitive;
# catches quiet USB-mic recordings but admits more false positives.
# Higher = fewer spurious detections but may miss soft impacts.
# Tune this as you calibrate with real course footage.
# Changed from 5.0 → 3.0 to handle quiet USB mic captures where the
# impact transient is clear but the peak/median ratio is modest.
SWING_AUDIO_MIN_PEAK_RATIO = 3.0

# ── 4-signal heuristic tunables ──────────────────────────────────────────────
# These constants drive detect_swings_combined's per-candidate gate logic.
# All are also exposed as keyword arguments so you can override per-call
# without editing this file. Change here to shift the global default.

# Maximum rise-time (attack) for a valid impact transient.  Already
# enforced by detect_swings_from_audio; replicated here as a named
# constant so the log output references the same value.
SWING_AUDIO_MAX_RISE_MS: float = 30.0

# Maximum ring-out: how long (ms) the envelope stays above 20 % of the
# peak amplitude after the peak.  A real "thwack" decays in < 150 ms;
# a ball drop on a hard floor, a hand clap, or a voice sustains longer.
SWING_AUDIO_MAX_DURATION_MS: float = 200.0

# Minimum spectral centroid (Hz) of the 100 ms window around the peak,
# computed after the 1.5 kHz high-pass.  Club-on-ball is broadband and
# produces a centroid >> 3 kHz; footsteps and low-frequency thuds sit
# below.  Set to 0.0 to disable this sub-check entirely.
SWING_AUDIO_MIN_SPECTRAL_CENTROID_HZ: float = 1500.0

# How far back from the impact timestamp to require sustained motion
# (backswing window).  A typical golf swing takes 0.5–1.0 s from
# address-to-takeaway to impact.
SWING_BACKSWING_WINDOW_S: float = 0.7

# How far forward from the impact timestamp to require sustained motion
# (follow-through window).  Real swings continue 0.3–0.8 s post-impact.
SWING_FOLLOWTHROUGH_WINDOW_S: float = 0.4

# Minimum motion signal at the EXACT impact moment, expressed as a
# multiple of the clip's median motion.  A ball drop has ~1× median
# at the moment of the sound; a real swing spikes much higher.
SWING_MIN_MOTION_RATIO: float = 2.0

# Minimum mean motion ratio (vs. clip median) required within the
# backswing and follow-through windows.  Lower than SWING_MIN_MOTION_RATIO
# because the arc is sustained but not necessarily peaked.
SWING_MOTION_FLOOR_RATIO: float = 1.2

# Visual impact lands a few frames before the audio envelope peak: sound
# from the strike has to travel ~5–15 m of air to the mic (≈30 ms /
# ~1 frame at 30 fps), and the smoothed envelope's max sits a few ms
# after the leading edge of the transient. Two frames back is a good
# default for typical phone-mic distances; tune up if the mic was very
# far from the tee, down if it was on the player.
AUDIO_IMPACT_PRE_PEAK_FRAMES = 2


def find_impact_via_audio(
    input_path: Path,
    fps: float,
    min_ratio: float | None = None,
    highpass_hz: float = 1500.0,
) -> dict:
    """Locate impact from the audio track.

    A struck golf ball produces a sharp, isolated transient ("thwack")
    that's typically the loudest moment in the clip by a wide margin.
    We extract mono PCM via ffmpeg, compute a short-window RMS
    envelope, find the peak, sanity-check it against the median
    envelope value (so background noise / music / wind doesn't fool
    us), and convert the peak's timestamp to a frame index using the
    supplied `fps`.

    Returns::

        {
          "ok": bool,
          "error": str | None,
          "impact_frame": int | None,
          "confidence": "high" | "medium" | "low" | None,
          "method": "audio",
          "peak_time_sec": float | None,
          "peak_value": float | None,
          "median_envelope": float | None,
          "ratio": float | None,           # peak_value / median_envelope
          "duration_sec": float | None,
        }

    Never raises. Returns ok=False with a descriptive `error` whenever
    we can't run (ffmpeg missing, no audio stream, empty audio, etc.)
    or can't find a clear peak.
    """
    info: dict = {
        "ok": False,
        "error": None,
        "impact_frame": None,
        "confidence": None,
        "method": "audio",
        "peak_time_sec": None,
        "peak_value": None,
        "median_envelope": None,
        "ratio": None,
        "duration_sec": None,
        "highpass_hz": float(highpass_hz) if highpass_hz and highpass_hz > 0 else 0.0,
    }

    if not HAS_NP:
        info["error"] = "numpy not installed"
        return info
    if shutil.which("ffmpeg") is None:
        info["error"] = "ffmpeg not on PATH"
        return info
    if fps is None or fps <= 0:
        info["error"] = "invalid fps"
        return info

    # Extract mono PCM int16 audio at AUDIO_SAMPLE_RATE Hz to stdout.
    # -vn skips video, -ac 1 forces mono, -f s16le emits raw PCM.
    # Apply the same 1.5 kHz high-pass filter the test-cut detector
    # uses so the peak/median ratio reported here is comparable to
    # what the operator sees on the long-upload audio detector. Without
    # it, voices / rumble pump the median up and a real thwack reads
    # as ×8-10 instead of ×80-150.
    audio_filter_args: list[str] = []
    if highpass_hz and highpass_hz > 0:
        audio_filter_args = ["-af", f"highpass=f={float(highpass_hz)}"]
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-v", "error",
                "-i", str(input_path),
                "-vn", "-ac", "1",
                "-ar", str(AUDIO_SAMPLE_RATE),
                *audio_filter_args,
                "-f", "s16le", "-",
            ],
            capture_output=True, timeout=60, check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        info["error"] = f"ffmpeg failed: {exc}"
        return info
    if proc.returncode != 0:
        info["error"] = f"ffmpeg returned {proc.returncode}: {proc.stderr.decode('utf-8', 'ignore')[:200]}"
        return info
    raw = proc.stdout
    if not raw:
        info["error"] = "no audio data extracted (clip may have no audio track)"
        return info

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0:
        info["error"] = "empty audio buffer"
        return info
    info["duration_sec"] = float(samples.size) / AUDIO_SAMPLE_RATE

    # RMS-ish envelope: |x| then moving average with a short window.
    # |x| (rather than x²) keeps the units intuitive (peak-amplitude
    # ratios) and is good enough for finding the loudest transient.
    abs_samples = np.abs(samples)
    win = max(1, int(AUDIO_SAMPLE_RATE * AUDIO_ENVELOPE_WINDOW_MS / 1000.0))
    if abs_samples.size <= win:
        info["error"] = "audio too short for envelope"
        return info
    kernel = np.ones(win, dtype=np.float32) / float(win)
    envelope = np.convolve(abs_samples, kernel, mode="same")

    peak_idx = int(np.argmax(envelope))
    peak_value = float(envelope[peak_idx])
    median = float(np.median(envelope))
    info["peak_value"] = peak_value
    info["median_envelope"] = median
    if median <= 0:
        # Mostly-silent audio; no useful baseline. Treat the loudest
        # sample as impact only if it's well above 0.
        if peak_value < 0.05:
            info["error"] = "audio is silent / has no impact transient"
            return info
        ratio = float("inf")
    else:
        ratio = peak_value / median
    info["ratio"] = ratio if ratio != float("inf") else None
    effective_min_ratio = (
        float(min_ratio) if min_ratio is not None else AUDIO_MIN_PEAK_OVER_MEDIAN
    )
    info["min_ratio_used"] = effective_min_ratio
    if ratio < effective_min_ratio:
        info["error"] = (
            f"audio peak/median ratio too low ({ratio:.1f} < "
            f"{effective_min_ratio}); audio likely noisy or "
            "lacks a clear impact"
        )
        return info

    peak_time = peak_idx / float(AUDIO_SAMPLE_RATE)
    info["peak_time_sec"] = float(peak_time)
    audio_peak_frame = int(round(peak_time * fps))
    # Visual impact lands a few frames before the audio envelope peak
    # (see AUDIO_IMPACT_PRE_PEAK_FRAMES comment). Clamp to 0 so we don't
    # go negative on short clips with impact near the very start.
    adjusted_frame = max(0, audio_peak_frame - AUDIO_IMPACT_PRE_PEAK_FRAMES)
    info["audio_peak_frame"] = audio_peak_frame
    info["pre_peak_offset_frames"] = int(AUDIO_IMPACT_PRE_PEAK_FRAMES)
    info["impact_frame"] = adjusted_frame
    # Confidence buckets — gated at >=30 by AUDIO_MIN_PEAK_OVER_MEDIAN
    # above, so "low" is unreachable here. Kept for downstream code
    # that branches on the label.
    if ratio >= 60:
        info["confidence"] = "high"
    else:
        info["confidence"] = "medium"

    info["ok"] = True
    log.info(
        "ai_tracer: audio impact — frame=%d (audio_peak=%d - %d) t=%.3fs "
        "peak=%.3f median=%.3f ratio=%.1f confidence=%s",
        info["impact_frame"], audio_peak_frame, AUDIO_IMPACT_PRE_PEAK_FRAMES,
        peak_time, peak_value, median, ratio,
        info["confidence"],
    )
    return info


def detect_swings_from_motion(
    input_path: Path,
    fps: float | None = None,
    sample_height: int = 180,
    min_swing_sec: float = 0.5,
    max_swing_sec: float = 3.5,
    min_separation_sec: float = 5.0,
    before_motion_sec: float = 3.5,
    after_motion_sec: float = 5.0,
    motion_ratio: float = 4.0,
    debug: dict | None = None,
) -> list[dict]:
    """Find every swing in a long tee-side video by scanning for bursts
    of high pixel motion.

    The tee camera is essentially static between swings — the player
    stands almost still during setup, then the swing produces a ~0.8 s
    burst of huge frame-to-frame motion. We decode the video at low
    resolution (`sample_height` px tall, grayscale), compute the mean
    absolute difference between consecutive frames, smooth that signal,
    threshold against the median, and group contiguous high-motion runs.

    Returns the same shape as detect_swings_from_audio so the rest of
    the pipeline can swallow it interchangeably:

        [
          {
            "peak_time_sec": float,   # time of max motion in the burst
            "start_sec": float,       # max(0, burst_start - before_motion_sec)
            "end_sec": float,         # min(duration, burst_end + after_motion_sec)
            "ratio": float,           # burst-peak motion / median motion
            "confidence": "high" | "medium" | "low",
            "burst_duration_sec": float,
          },
          ...
        ]

    Empty list when opencv / numpy is missing, the video has < 2 frames,
    or no motion bursts pass the duration + amplitude gates. Never
    raises.
    """
    if debug is not None:
        debug.update({
            "reason": None,
            "motion_ratio_used": float(motion_ratio),
            "sample_height": int(sample_height),
            "min_swing_sec": float(min_swing_sec),
            "max_swing_sec": float(max_swing_sec),
        })

    if not HAS_CV or not HAS_NP:
        log.info("ai_tracer: detect_swings_from_motion — missing opencv / numpy")
        if debug is not None:
            debug["reason"] = "opencv or numpy not installed"
        return []

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        log.warning("ai_tracer: detect_swings_from_motion — cannot open %s", input_path)
        if debug is not None:
            debug["reason"] = "could not open video"
        return []

    try:
        src_fps = float(fps) if fps else float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if src_fps <= 0:
            src_fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 2:
            if debug is not None:
                debug["reason"] = f"video has only {total_frames} frame(s)"
            return []

        # We don't need full-rate motion — 10 Hz is plenty to find ~1 s
        # swing bursts and keeps the scan fast on long videos.
        target_hz = 10.0
        step = max(1, int(round(src_fps / target_hz)))
        effective_hz = src_fps / step

        diffs: list[float] = []
        prev_gray = None
        idx = -1
        kept = 0
        while True:
            idx += 1
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx % step != 0:
                continue
            h, w = frame.shape[:2]
            if h > sample_height:
                scale = sample_height / float(h)
                new_w = max(1, int(round(w * scale)))
                frame = cv2.resize(frame, (new_w, sample_height), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                d = cv2.absdiff(gray, prev_gray)
                diffs.append(float(d.mean()))
            prev_gray = gray
            kept += 1
    finally:
        cap.release()

    if len(diffs) < 4:
        if debug is not None:
            debug["reason"] = f"only {len(diffs)} usable frame diffs"
        return []

    motion = np.asarray(diffs, dtype=np.float32)
    # Smooth with a ~300 ms window so we get one peak per swing instead
    # of jitter from individual frames.
    smooth_win = max(1, int(round(0.3 * effective_hz)))
    if smooth_win > 1 and smooth_win < motion.size:
        kernel = np.ones(smooth_win, dtype=np.float32) / float(smooth_win)
        motion = np.convolve(motion, kernel, mode="same")

    median = float(np.median(motion))
    if median <= 1e-6:
        median = 1e-6
    threshold = median * motion_ratio
    above = motion > threshold

    # Walk contiguous True runs. Each run = one motion burst.
    duration_sec = (len(diffs) - 1) / effective_hz if effective_hz > 0 else 0.0
    bursts: list[tuple[int, int, int, float]] = []  # (start_i, end_i, peak_i, peak_v)
    i = 0
    n = above.size
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        # [i, j) is one above-threshold run.
        seg = motion[i:j]
        peak_off = int(np.argmax(seg))
        bursts.append((i, j - 1, i + peak_off, float(seg[peak_off])))
        i = j

    # Duration filter — real swings last ~0.6–2 s. Drop sub-blip
    # (twitches, single-frame flashes) and over-long runs (someone
    # walking through frame, camera bump, panning).
    accepted: list[tuple[int, int, int, float]] = []
    for s_i, e_i, p_i, p_v in bursts:
        burst_sec = (e_i - s_i) / effective_hz if effective_hz > 0 else 0.0
        if burst_sec < min_swing_sec or burst_sec > max_swing_sec:
            continue
        accepted.append((s_i, e_i, p_i, p_v))

    def _populate_debug(n_final: int) -> None:
        if debug is None:
            return
        ranked = sorted(bursts, key=lambda t: -t[3])[:10]
        # Downsample the motion waveform for plotting. Peak-preserving
        # (max per bin) so a ~0.8s swing burst survives decimation to a
        # few hundred points. Map index i -> t = i/(len-1) * duration_sec.
        _series = motion
        _max_pts = 600
        if _series.size > _max_pts:
            _bin = int(np.ceil(_series.size / _max_pts))
            _pad = (-_series.size) % _bin
            if _pad:
                _series = np.concatenate(
                    [_series, np.full(_pad, _series[-1], dtype=_series.dtype)]
                )
            _series = _series.reshape(-1, _bin).max(axis=1)
        debug["motion_series"] = [round(float(v), 4) for v in _series]
        debug.update({
            "effective_hz": float(effective_hz),
            "duration_sec": float(duration_sec),
            "n_motion_samples": int(motion.size),
            "median_motion": float(median),
            "threshold": float(threshold),
            "min_motion": float(motion.min()),
            "max_motion": float(motion.max()),
            "n_raw_bursts": len(bursts),
            "n_duration_ok": len(accepted),
            "n_final": int(n_final),
            "top_raw_bursts": [
                {
                    "start_sec": round(s_i / effective_hz, 2) if effective_hz else 0.0,
                    "end_sec": round(e_i / effective_hz, 2) if effective_hz else 0.0,
                    "duration_sec": round((e_i - s_i) / effective_hz, 2) if effective_hz else 0.0,
                    "peak_sec": round(p_i / effective_hz, 2) if effective_hz else 0.0,
                    "peak_ratio": round(p_v / median, 2) if median > 0 else None,
                    "passes_duration": (
                        min_swing_sec <= ((e_i - s_i) / effective_hz if effective_hz else 0) <= max_swing_sec
                    ),
                }
                for s_i, e_i, p_i, p_v in ranked
            ],
        })

    if not accepted:
        log.info(
            "ai_tracer: detect_swings_from_motion — %d raw bursts, 0 passed "
            "duration filter [%.1f, %.1f] s (median=%.4f threshold=%.4f)",
            len(bursts), min_swing_sec, max_swing_sec, median, threshold,
        )
        _populate_debug(0)
        return []

    # Non-max suppression — collapse bursts whose peaks land within
    # min_separation_sec of each other (e.g. practice swing + real
    # swing in quick succession) keeping the louder one.
    accepted.sort(key=lambda t: -t[3])
    chosen_peaks: list[int] = []
    keep: list[tuple[int, int, int, float]] = []
    min_sep = int(min_separation_sec * effective_hz)
    for s_i, e_i, p_i, p_v in accepted:
        if any(abs(p_i - cp) < min_sep for cp in chosen_peaks):
            continue
        chosen_peaks.append(p_i)
        keep.append((s_i, e_i, p_i, p_v))
    keep.sort(key=lambda t: t[2])

    segments: list[dict] = []
    for s_i, e_i, p_i, p_v in keep:
        start_t = s_i / effective_hz
        end_t = e_i / effective_hz
        peak_t = p_i / effective_hz
        ratio = p_v / median if median > 0 else float("inf")
        if ratio >= 12:
            conf = "high"
        elif ratio >= 6:
            conf = "medium"
        else:
            conf = "low"
        segments.append({
            "peak_time_sec": float(peak_t),
            "start_sec": float(max(0.0, start_t - before_motion_sec)),
            "end_sec": float(min(duration_sec, end_t + after_motion_sec)),
            "ratio": float(ratio) if ratio != float("inf") else None,
            "confidence": conf,
            "burst_duration_sec": float(end_t - start_t),
        })

    log.info(
        "ai_tracer: detect_swings_from_motion — %d swings (raw=%d duration_ok=%d) "
        "duration=%.1fs median_motion=%.4f threshold=%.4f hz=%.1f",
        len(segments), len(bursts), len(accepted),
        duration_sec, median, threshold, effective_hz,
    )
    _populate_debug(len(segments))
    return segments


_FIND_RESTING_BALL_PROMPT = (
    "You are looking at ONE frame from a golf tee camera positioned behind "
    "the golfer. Decide whether a golf ball is sitting AT REST on the "
    "ground / mat / tee — a small, roughly round, usually white ball that is "
    "stationary and ready to be hit. It may be partially hidden by the club "
    "head resting behind it. Do NOT count a ball in flight, balls in a "
    "bucket, distant range balls, or the hole. CRITICALLY: do NOT count "
    "white shoes, socks, shoe trim, tee markers, sprinkler heads, signs or "
    "any other white object — a golf ball is TINY (typically under 2% of "
    "the image width) and perfectly round; if the white thing is attached "
    "to the golfer or bigger than a few pixels, it is not the ball. When "
    "unsure, answer present=false. Reply with JSON only:\n"
    '{"present": true|false, "x": <int pixel x or null>, '
    '"y": <int pixel y or null>, "confidence": "high"|"medium"|"low"}\n'
    "Coordinates are pixels in THIS image (top-left = 0,0)."
)


def verify_rest_and_impact(
    input_path: Path,
    rest_xy: tuple[float, float],
    approx_impact_frame: int,
    fps: float,
    debug_dir: Path | None = None,
    debug_prefix: str = "anchorchk",
    window_sec: float = 1.0,
) -> dict:
    """Pixel-verify and TIGHTEN the two anchors the tracers lean on —
    no API calls, one sequential decode of ~2s of video.

    1) SNAP: vision ball coordinates are ±a-few-px after downscale →
       native scaling. On the pre-impact frames, find the small bright
       cluster nearest the claimed rest position (within ~15px) and
       snap to its centroid.
    2) DEPARTURE: watch the rest patch across [impact-1s, impact+1s].
       The ball is a bright blob that is THERE before the strike and
       GONE after — the first frame it stays absent (3+ consecutive,
       riding out the clubhead passing over) is impact to ~±1 frame.
       If the patch never had a ball, or the ball never leaves, the
       claimed rest position (or the swing) is wrong — verified=False
       and callers keep their original anchors.

    Writes a debug film-strip (each patch crop bordered green=ball
    present / red=absent, departure frame flagged) when debug_dir is
    set. Returns {available, verified, rest_xy, snapped, snap_px,
    impact_frame, impact_delta, present_ratio_pre, reason, image}.
    Never raises."""
    out = {
        "available": False, "verified": None,
        "rest_xy": [float(rest_xy[0]), float(rest_xy[1])],
        "snapped": False, "snap_px": None,
        "impact_frame": None, "impact_delta": None,
        "present_ratio_pre": None, "reason": None, "image": None,
    }
    if not HAS_CV or not HAS_NP:
        out["reason"] = "opencv/numpy not installed"
        return out
    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            out["reason"] = "could not open video"
            return out
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(fps or 30.0)
        imp = int(approx_impact_frame)
        r = max(10, int(round(0.015 * h)))  # ball scale
        # WIDE crop: the claimed rest position can be off by 50-100px
        # (vision downscale error, operator click) — search an 8r
        # half-size neighborhood (~260px square at 1080p) so the real
        # ball is in view even when the claim is well off. The
        # ball-sized area cap + nearest-to-claim selection keep bigger
        # bright objects from hijacking the snap. Clamp at frame edges
        # instead of bailing (the ball usually sits low in the frame).
        m = 8 * r
        cx0, cy0 = float(rest_xy[0]), float(rest_xy[1])
        x0, x1 = max(0, int(cx0 - m)), min(w, int(cx0 + m + 1))
        y0, y1 = max(0, int(cy0 - m)), min(h, int(cy0 + m + 1))
        if x1 - x0 < 3 * r or y1 - y0 < 3 * r:
            out["reason"] = "rest patch clipped by frame edge"
            cap.release()
            return out
        ccx, ccy = cx0 - x0, cy0 - y0  # claimed centre, crop coords
        f_lo = max(0, imp - int(round(float(window_sec) * fps)))
        f_hi = imp + int(round(float(window_sec) * fps))
        crops: dict[int, "np.ndarray"] = {}
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_lo)
        for f in range(f_lo, f_hi + 1):
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            crops[f] = cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        cap.release()
        if len(crops) < 10:
            out["reason"] = "window too short / read failed"
            return out
        out["available"] = True

        # Baseline = clearly pre-impact frames (up to impact - 0.35s).
        _pre_end = imp - int(round(0.35 * fps))
        base_fs = [f for f in crops if f <= _pre_end]
        if len(base_fs) < 4:
            base_fs = sorted(crops)[: max(4, len(crops) // 4)]
        base = np.median(
            np.stack([crops[f].astype(np.float32) for f in base_fs]), axis=0,
        ).astype(np.uint8)

        def _bright_centroid(g, tx, ty):
            """Centroid of the small bright cluster nearest (tx, ty),
            or None. Ball-sized only: the absolute-pixel area cap keeps
            shoes / tee markers / bright patio from qualifying in the
            wide crop."""
            mean = float(g.mean())
            thr = max(mean + 25.0, float(np.percentile(g, 92)))
            mask = (g >= thr).astype(np.uint8)
            n, lbl, stats_, cent = cv2.connectedComponentsWithStats(mask)
            best = None
            for i in range(1, n):
                area = int(stats_[i, cv2.CC_STAT_AREA])
                if area < 4 or area > (1.8 * r) ** 2:
                    continue
                d = ((cent[i][0] - tx) ** 2
                     + (cent[i][1] - ty) ** 2) ** 0.5
                if best is None or d < best[0]:
                    best = (d, float(cent[i][0]), float(cent[i][1]), area)
            return best  # (dist_to_target, cx, cy, area) | None

        # SNAP on the baseline composite: nearest ball-sized bright
        # cluster to the CLAIMED position, anywhere in the wide crop.
        snap = _bright_centroid(base, ccx, ccy)
        scx, scy = float(ccx), float(ccy)
        if snap is not None and snap[0] <= 6.0 * r:
            scx, scy = snap[1], snap[2]
            out["snapped"] = True
            out["snap_px"] = float(round(float(snap[0]), 1))
            out["rest_xy"] = [
                float(round(x0 + scx, 1)),
                float(round(y0 + scy, 1)),
            ]

        # Per-frame ball presence at the SNAPPED spot.
        def _present(g):
            b = _bright_centroid(g, scx, scy)
            if b is None:
                return False
            return b[0] <= r * 0.8

        present = {f: _present(g) for f, g in crops.items()}
        _fs_all = sorted(crops)
        _head = _fs_all[: max(6, int(round(0.4 * fps)))]
        pre_ratio = max(
            sum(1 for f in base_fs if present[f]) / max(1, len(base_fs)),
            sum(1 for f in _head if present[f]) / max(1, len(_head)),
        )
        out["present_ratio_pre"] = round(pre_ratio, 2)
        if pre_ratio < 0.6:
            out["verified"] = False
            out["reason"] = (
                f"no steady ball at the claimed rest spot before impact "
                f"(present {int(pre_ratio * 100)}% of baseline frames) — "
                f"rest position likely wrong"
            )
        else:
            # DEPARTURE: first frame absent and STAYS absent 3 frames
            # (clubhead passing over occludes 1-2 frames at most).
            fs_sorted = sorted(crops)
            dep = None
            for i, f in enumerate(fs_sorted[:-2]):
                # Scan the WHOLE window (not just past the baseline) —
                # the impact estimate can run late (pose peak), putting
                # the true departure well before it.
                if f < fs_sorted[0] + 3:
                    continue
                if (
                    not present[f]
                    and not present.get(f + 1, True)
                    and not present.get(f + 2, True)
                    and any(present.get(f - k, False) for k in (1, 2, 3, 4))
                ):
                    dep = f
                    break
            if dep is None:
                out["verified"] = False
                out["reason"] = (
                    "ball never left the rest spot within ±1s of the "
                    "estimated impact (practice swing, or impact estimate "
                    "far off)"
                )
            else:
                out["verified"] = True
                out["impact_frame"] = int(dep)
                out["impact_delta"] = int(dep - imp)
                out["reason"] = (
                    f"ball departed the rest spot at f{dep} "
                    f"({dep - imp:+d} vs estimate)"
                )

        # Debug film-strip — SHOW THE WORK: each patch crop bordered by
        # its presence verdict, the departure frame flagged.
        if debug_dir is not None:
            try:
                fs_sorted = sorted(crops)
                step = max(1, len(fs_sorted) // 30)
                sel = fs_sorted[::step]
                if out.get("impact_frame") is not None:
                    for extra in range(
                        out["impact_frame"] - 2, out["impact_frame"] + 3,
                    ):
                        if extra in crops and extra not in sel:
                            sel.append(extra)
                    sel = sorted(set(sel))
                _cw = crops[fs_sorted[0]].shape[1]
                Z = 1 if _cw >= 100 else (2 if _cw >= 60 else 3)
                tiles = []
                for f in sel:
                    g = crops[f]
                    t = cv2.cvtColor(
                        cv2.resize(
                            g, (g.shape[1] * Z, g.shape[0] * Z),
                            interpolation=cv2.INTER_NEAREST,
                        ),
                        cv2.COLOR_GRAY2BGR,
                    )
                    col = (0, 200, 0) if present.get(f) else (0, 0, 230)
                    cv2.rectangle(
                        t, (0, 0), (t.shape[1] - 1, t.shape[0] - 1), col, 3,
                    )
                    if out.get("impact_frame") == f:
                        cv2.rectangle(
                            t, (4, 4), (t.shape[1] - 5, t.shape[0] - 5),
                            (0, 255, 255), 3,
                        )
                    cv2.circle(
                        t, (int(scx * Z), int(scy * Z)), int(r * 0.8 * Z),
                        (255, 200, 0), 1, cv2.LINE_AA,
                    )
                    cv2.putText(
                        t, str(f), (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255, 255, 255), 1, cv2.LINE_AA,
                    )
                    tiles.append(t)
                per_row = 10
                rows = []
                for i in range(0, len(tiles), per_row):
                    row = tiles[i:i + per_row]
                    while len(row) < per_row:
                        row.append(np.zeros_like(tiles[0]))
                    rows.append(cv2.hconcat(row))
                strip = cv2.vconcat(rows)
                bar = np.zeros((56, strip.shape[1], 3), np.uint8)
                _lbl = (
                    f"anchor check @ rest ({out['rest_xy'][0]:.0f},"
                    f"{out['rest_xy'][1]:.0f}) "
                    f"[{x1 - x0}x{y1 - y0}px window]"
                    + (
                        f" snapped {out['snap_px']}px" if out["snapped"]
                        else " (no snap)"
                    )
                    + f" - {out['reason']}"
                )
                cv2.putText(
                    bar, _lbl, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255, 255, 255), 1, cv2.LINE_AA,
                )
                cv2.putText(
                    bar,
                    "green=ball present, red=absent, yellow box=departure "
                    "(impact), ring=watched spot",
                    (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (180, 180, 180), 1, cv2.LINE_AA,
                )
                img = cv2.vconcat([bar, strip])
                name = f"{debug_prefix}.jpg"
                cv2.imwrite(
                    str(Path(debug_dir) / name), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 88],
                )
                out["image"] = name
            except Exception as exc:  # noqa: BLE001
                log.warning("anchor check: debug strip failed: %s", exc)
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("verify_rest_and_impact failed: %s", exc)
        out["reason"] = str(exc)
        return out


def track_launch_from_rest(
    input_path: Path,
    rest_xy: tuple[float, float],
    impact_frame: int,
    fps: float,
    debug_dir: Path | None = None,
    debug_prefix: str = "launchtrk",
    max_seconds: float = 4.0,
) -> dict:
    """Adaptive launch tracker (operator-designed; pure pixels, no AI).

    Seeded at the PINNED rest ball + departure frame. The viewing
    square starts with the ball in its BOTTOM THIRD (centred left-
    right) — the ball is about to go UP. Each frame: find the moving
    ball-sized blob in the square (frame differencing);
      found  -> mark it, advance a frame, move the square along the
                ball's velocity, SHRINK back toward base size;
      missed -> advance a frame, keep extrapolating, WIDEN the square
                until it's found again.
    Stops after ~18 consecutive misses, the 4s cap, or the square
    leaving the frame. While ascending the ball rides the bottom third
    of the square; once descending, the top third (the square always
    looks ahead of the motion).

    Returns {available, points: [{frame,x,y}], n_found, boxes, image,
    reason}. When debug_dir is set, writes a film-strip of the moving
    square per frame (green border=found with the ball dotted,
    red=missed, box size labelled) — the work, visible. Never raises."""
    out = {
        "available": False, "points": [], "n_found": 0,
        "boxes": [], "image": None, "reason": None,
    }
    if not HAS_CV or not HAS_NP:
        out["reason"] = "opencv/numpy not installed"
        return out
    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            out["reason"] = "could not open video"
            return out
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(fps or 30.0)
        r = max(10, int(round(0.015 * h)))   # ball scale
        m0 = 8 * r                            # base box half-size
        f0 = int(impact_frame)
        f_end = f0 + int(round(float(max_seconds) * fps))
        rx, ry = float(rest_xy[0]), float(rest_xy[1])

        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f0 - 1))
        ok, prev = cap.read()
        if not ok or prev is None:
            out["reason"] = "could not read pre-impact frame"
            cap.release()
            return out
        prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        last_x, last_y, last_f = rx, ry, f0 - 1
        vx = vy = None
        recent_marks: list = []  # last few FOUND marks — the operator's
        # 'marks nearly on top of each other' apex signal is their NET
        # drift over the window (single-hop spacing is jitter-inflated
        # by diff-crescent centroids exactly when true motion is tiny)
        scale = 1.0
        consec_miss = 0
        ref_found_g = prev_g  # gray at the last FOUND frame (apex diffing)
        tiles = []  # (frame, crop_bgr, found, fx, fy, bw, bh)
        f = f0
        while f <= f_end:
            ok, cur = cap.read()
            if not ok or cur is None:
                out["reason"] = "end of video"
                break
            cur_g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
            gap = f - last_f
            # APEX MODE (operator insight): when the found marks land
            # nearly on top of each other, the ball is topping out —
            # its per-frame motion approaches zero and plain frame
            # differencing goes blind. Hold the box tight on the last
            # position, diff against the LAST FOUND frame (slow drift
            # accumulates into a visible blob), drop the motion-per-
            # frame demands, and be extra patient before giving up.
            _vm_now = (
                (vx * vx + vy * vy) ** 0.5 if vx is not None else None
            )
            _net_pf = None
            if len(recent_marks) >= 3:
                _f0m, _x0m, _y0m = recent_marks[0]
                _f1m, _x1m, _y1m = recent_marks[-1]
                if _f1m > _f0m:
                    _net_pf = (
                        ((_x1m - _x0m) ** 2 + (_y1m - _y0m) ** 2) ** 0.5
                        / (_f1m - _f0m)
                    )
            # Threshold 6px/f: mark jitter (diff-crescent centroids)
            # inflates measured drift by ~2-3px on a genuinely slow
            # ball, so 4.0 left a dead zone where neither apex mode nor
            # normal-mode gates would accept the real ball.
            apex_mode = (
                (_net_pf is not None and _net_pf < 6.0)
                or (_vm_now is not None and _vm_now < 6.0)
            )
            if apex_mode:
                scale = 1.0  # never widen at apex — drift hunts sparkle
            pred_x = last_x + (vx * gap if vx is not None else 0.0)
            pred_y = last_y + (vy * gap if vy is not None else 0.0)
            bw = bh = int(2 * m0 * scale)
            # Directional bias: the square looks AHEAD of the motion —
            # ball in the bottom third while ascending (or unknown),
            # top third once descending.
            if apex_mode:
                # Topping out: the ball is about to reverse — look both
                # ways (centred), not up.
                y0 = int(pred_y - bh / 2.0)
            elif vy is not None and vy > 2.0:
                y0 = int(pred_y - bh / 3.0)
            else:
                y0 = int(pred_y - 2.0 * bh / 3.0)
            x0 = int(pred_x - bw / 2.0)
            x0 = max(0, min(w - bw, x0)) if bw <= w else 0
            y0 = max(0, min(h - bh, y0)) if bh <= h else 0
            x1, y1 = min(w, x0 + bw), min(h, y0 + bh)
            if x1 - x0 < 3 * r or y1 - y0 < 3 * r:
                out["reason"] = "square left the frame"
                break
            crop_g = cur_g[y0:y1, x0:x1]
            _base_g = ref_found_g if apex_mode else prev_g
            diff = cv2.absdiff(crop_g, _base_g[y0:y1, x0:x1])
            _thr = 12 if apex_mode else 20  # faint apex/descent blob
            mask = (diff >= _thr).astype(np.uint8)
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
            n, lbl, stats_, cent = cv2.connectedComponentsWithStats(mask)
            _crop_mean = float(crop_g.mean())
            best = None
            for i in range(1, n):
                area = int(stats_[i, cv2.CC_STAT_AREA])
                if area < 3 or area > (3.5 * r) ** 2:
                    continue  # not ball-sized motion
                # SHAPE gate (operator's f484 catch): the club shaft
                # sweeps up through the ball's corridor right after
                # impact — its motion component is a long thin STREAK
                # (small area, huge extent) with the head at its tip.
                # A ball's diff blob can only be as long as its own
                # blur (speed x gap + diameter); anything longer is
                # club, not ball.
                _cw_ = int(stats_[i, cv2.CC_STAT_WIDTH])
                _ch_ = int(stats_[i, cv2.CC_STAT_HEIGHT])
                _dim_cap = 2 * r + 1.6 * max(
                    20.0, (_vm_now or 30.0) * gap,
                )
                if max(_cw_, _ch_) > _dim_cap:
                    continue
                cx_, cy_ = float(cent[i][0]) + x0, float(cent[i][1]) + y0
                # The vacated rest spot keeps firing as motion for ~10
                # frames (background ghost) — never the ball.
                if ((cx_ - rx) ** 2 + (cy_ - ry) ** 2) ** 0.5 < 1.5 * r:
                    continue
                # BALL-likeness during LOCK-ON only (first finds): the
                # ball must be a small BRIGHT blob, which rejects shirt
                # and arm motion near the golfer. Once the trajectory is
                # established the direction/step gates carry precision —
                # and against bright sky the ball is NOT brighter than
                # the crop average (real miss at f493: dark-trees crop
                # turned sky-white and the gate killed the visible ball,
                # then the widened box chained canopy sparkle instead).
                if len(out["points"]) < 3 or apex_mode:
                    # Also required in APEX mode: diffing against the
                    # last-found frame lights up the VACATED old spot
                    # (dark background now) as well as the ball — the
                    # ball is the bright one.
                    _px = int(round(cx_ - x0)); _py = int(round(cy_ - y0))
                    _p5 = crop_g[
                        max(0, _py - 2):_py + 3, max(0, _px - 2):_px + 3,
                    ]
                    _margin = 8.0 if apex_mode else 15.0
                    if (
                        _p5.size == 0
                        or float(_p5.mean()) < _crop_mean + _margin
                    ):
                        continue
                if vx is None:
                    # FIRST find: the ball just launched — it must be
                    # ABOVE the rest spot and clearly displaced from it
                    # (an up-cone, like the rest-lock's seed).
                    if cy_ > ry - 2.0:
                        continue
                    if abs(cx_ - rx) > (ry - cy_) + 60.0:
                        continue
                elif apex_mode:
                    # Topping out: accept only a small drift near the
                    # last mark — no direction demand (it reverses
                    # here), no minimum step (there barely is one).
                    _sm = ((cx_ - last_x) ** 2
                           + (cy_ - last_y) ** 2) ** 0.5
                    if _sm > 45.0:
                        continue
                else:
                    # Aim by trajectory: the hop from the last plot
                    # point must roughly agree with the velocity (flight
                    # never u-turns), stay within a velocity-scaled
                    # accept radius of the prediction, AND make real
                    # progress (near-static canopy sparkle produces
                    # ball-sized flickers that sit still).
                    _sx, _sy = cx_ - last_x, cy_ - last_y
                    _sm = (_sx ** 2 + _sy ** 2) ** 0.5
                    _vm = (vx ** 2 + vy ** 2) ** 0.5
                    if _vm > 6.0 and _sm > 1.0:
                        if (_sx * vx + _sy * vy) < 0.2 * _sm * _vm:
                            continue
                    if _vm > 10.0 and _sm < 0.3 * _vm * gap:
                        continue  # anti-sparkle: fast flight only
                d = ((cx_ - pred_x) ** 2 + (cy_ - pred_y) ** 2) ** 0.5
                if vx is not None:
                    _vm = (vx ** 2 + vy ** 2) ** 0.5
                    if d > max(40.0, 2.2 * _vm * gap):
                        continue
                if best is None or d < best[0]:
                    best = (d, cx_, cy_, area)
            found = best is not None
            fx = fy = None
            if found:
                _, fx, fy, _a = best
                nvx = (fx - last_x) / max(1, gap)
                nvy = (fy - last_y) / max(1, gap)
                recent_marks.append((f, float(fx), float(fy)))
                if len(recent_marks) > 6:
                    recent_marks.pop(0)
                # First hop from rest can be huge (launch) — take it;
                # afterwards smooth so one noisy hit can't yank aim.
                if vx is None:
                    vx, vy = nvx, nvy
                else:
                    vx, vy = 0.5 * vx + 0.5 * nvx, 0.5 * vy + 0.5 * nvy
                last_x, last_y, last_f = fx, fy, f
                ref_found_g = cur_g
                out["points"].append({
                    "frame": int(f), "x": round(float(fx), 1),
                    "y": round(float(fy), 1),
                })
                scale = max(1.0, scale * 0.75)
                consec_miss = 0
            else:
                consec_miss += 1
                scale = min(3.0, scale * 1.35)
            out["boxes"].append({
                "frame": int(f), "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "found": bool(found),
            })
            _crop_bgr = cur[y0:y1, x0:x1].copy()
            _heat_bgr = _crop_bgr.copy()
            _mm = mask > 0
            _heat_bgr[_mm] = (
                0.35 * _heat_bgr[_mm] + np.array([0, 0, 165])
            ).clip(0, 255).astype(np.uint8)
            tiles.append((f, _crop_bgr, _heat_bgr, found, fx, fy, x0, y0))
            _miss_limit = 48 if apex_mode else 18
            if consec_miss >= _miss_limit:
                out["reason"] = (
                    f"lost the ball ({consec_miss} straight misses)"
                )
                break
            prev_g = cur_g
            f += 1
        cap.release()
        out["available"] = True
        out["n_found"] = len(out["points"])
        if out["reason"] is None:
            out["reason"] = f"tracked to the {max_seconds:.0f}s cap"
        if out["n_found"] == 0:
            out["reason"] = "never re-found the ball after departure"

        if debug_dir is not None and tiles:
            try:
                sel = tiles[:150]  # EVERY frame (capped for sanity)
                TW = 130

                def _render_strip(use_heat: bool) -> "np.ndarray":
                    rendered = []
                    for (tf, crop_p, crop_h, found, fx, fy, bx0, by0) in sel:
                        crop = crop_h if use_heat else crop_p
                        ch, cw = crop.shape[:2]
                        z = TW / float(cw)
                        t = cv2.resize(crop, (TW, max(1, int(ch * z))))
                        col = (0, 200, 0) if found else (0, 0, 230)
                        cv2.rectangle(
                            t, (0, 0),
                            (t.shape[1] - 1, t.shape[0] - 1), col, 2,
                        )
                        if found and fx is not None:
                            cv2.circle(
                                t,
                                (int((fx - bx0) * z), int((fy - by0) * z)),
                                max(3, int(r * z)), (0, 255, 255), 1,
                                cv2.LINE_AA,
                            )
                        cv2.putText(
                            t, f"{tf}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, (255, 255, 255), 1, cv2.LINE_AA,
                        )
                        cv2.putText(
                            t, f"{cw}px", (3, t.shape[0] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                            (200, 200, 200), 1, cv2.LINE_AA,
                        )
                        rendered.append(t)
                    mh = max(t.shape[0] for t in rendered)
                    rendered = [
                        cv2.copyMakeBorder(
                            t, 0, mh - t.shape[0], 0, 0,
                            cv2.BORDER_CONSTANT, value=(0, 0, 0),
                        )
                        for t in rendered
                    ]
                    per_row = 12
                    rows = []
                    for i in range(0, len(rendered), per_row):
                        row = rendered[i:i + per_row]
                        while len(row) < per_row:
                            row.append(np.zeros_like(rendered[0]))
                        rows.append(cv2.hconcat(row))
                    strip = cv2.vconcat(rows)
                    bar = np.zeros((34, strip.shape[1], 3), np.uint8)
                    cv2.putText(
                        bar,
                        (
                            f"launch tracker from rest "
                            f"({rx:.0f},{ry:.0f}) @ f{f0} - "
                            f"{out['n_found']} found, {out['reason']} | "
                            f"square moves with the ball; widens on miss, "
                            f"shrinks on find"
                            + (" | RED TINT = motion mask" if use_heat else "")
                        ),
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA,
                    )
                    return cv2.vconcat([bar, strip])

                name = f"{debug_prefix}.jpg"
                cv2.imwrite(
                    str(Path(debug_dir) / name), _render_strip(False),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 86],
                )
                out["image"] = name
                # Heat variant — the SAME tiles with the frame-diff
                # motion mask tinted red (what the tracker actually
                # looked at), toggled from the debug UI.
                name_h = f"{debug_prefix}-heat.jpg"
                cv2.imwrite(
                    str(Path(debug_dir) / name_h), _render_strip(True),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 86],
                )
                out["image_heat"] = name_h
            except Exception as exc:  # noqa: BLE001
                log.warning("launch tracker: debug strip failed: %s", exc)
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("track_launch_from_rest failed: %s", exc)
        out["reason"] = str(exc)
        return out


def find_resting_ball(
    input_path: Path,
    frame_idx: int,
    frame_w: int = 1024,
    model: str | None = None,
    crop_center: tuple[float, float] | None = None,
    crop_frac: float = 0.45,
) -> dict:
    """Single-frame Claude vision call: is there a golf ball AT REST, and
    where? Returns {present, x, y, confidence, error} with x/y in NATIVE
    pixels. Never raises. This is the same 'Claude can see the ball'
    capability the tracer uses in flight, pointed at the resting ball.

    crop_center (native px, e.g. the golfer's hands from the pose detector)
    zooms the call in: only a crop_frac-sized box around that point is sent.
    On a wide tee shot the ball is ~4px in the downscaled full frame —
    routinely invisible to the model even when it's plainly in the picture.
    Cropping to the golfer makes the ball several times larger. Returned
    coords are mapped back to full-frame native pixels."""
    out = {
        "present": False, "x": None, "y": None, "confidence": None,
        "error": None, "crop_box": None,
    }
    if not HAS_CV:
        out["error"] = "opencv not installed"
        return out
    if not HAS_ANTHROPIC or not os.environ.get("ANTHROPIC_API_KEY"):
        out["error"] = "ANTHROPIC_API_KEY not set"
        return out
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        out["error"] = "could not open video"
        return out
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, raw = cap.read()
    finally:
        cap.release()
    if not ok or raw is None:
        out["error"] = f"could not read frame {frame_idx}"
        return out
    h, w = raw.shape[:2]
    crop_x0 = crop_y0 = 0
    if crop_center is not None and w > 0 and h > 0:
        cw = max(64, int(round(w * crop_frac)))
        ch = max(64, int(round(h * crop_frac)))
        # Bias the box slightly downward: the hands sit at waist height and
        # the ball is on the ground below them.
        cx = int(round(float(crop_center[0])))
        cy = int(round(float(crop_center[1]) + 0.08 * h))
        crop_x0 = max(0, min(w - cw, cx - cw // 2))
        crop_y0 = max(0, min(h - ch, cy - ch // 2))
        raw = raw[crop_y0:crop_y0 + ch, crop_x0:crop_x0 + cw]
        h, w = raw.shape[:2]
        out["crop_box"] = [crop_x0, crop_y0, w, h]
    scale = frame_w / float(w) if w > frame_w else 1.0
    if scale != 1.0:
        raw = cv2.resize(raw, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", raw, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        out["error"] = "jpeg encode failed"
        return out
    client = _anthropic_client()
    try:
        resp = client.messages.create(
            model=_resolve_frame_picker_model(model),
            max_tokens=150,
            system=[{
                "type": "text",
                "text": _FIND_RESTING_BALL_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": [
                {"type": "text", "text": f"Frame {frame_idx}. JSON only."},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(bytes(buf)).decode("ascii"),
                }},
            ]}],
        )
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"api_failed: {exc}"
        return out
    text = "".join(
        c.text for c in resp.content if getattr(c, "type", None) == "text"
    )
    data = _extract_json(text) or {}
    out["present"] = bool(data.get("present"))
    out["confidence"] = data.get("confidence")
    if out["present"] and data.get("x") is not None and data.get("y") is not None:
        try:
            out["x"] = int(round(float(data["x"]) / scale)) + crop_x0
            out["y"] = int(round(float(data["y"]) / scale)) + crop_y0
        except (TypeError, ValueError):
            pass
    return out


def _white_blob_at(input_path: Path, frame_idx: int, x: int, y: int) -> bool | None:
    """Cheap pixel check: does the marked point actually look like a golf
    ball — a small cluster of notably-bright pixels near (x, y)? Guards the
    practice-swing verdict against the vision call latching onto a tee
    marker / leaf / shadow: a misdetected static object would otherwise sit
    "unmoved" across the swing and wrongly classify a real shot as practice.
    Returns True/False, or None when the frame can't be read (no opinion)."""
    try:
        cap = cv2.VideoCapture(str(input_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, frame_idx)))
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        r = max(12, int(0.015 * ((w * w + h * h) ** 0.5)))
        x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
        y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
        if x1 - x0 < 6 or y1 - y0 < 6:
            return None
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        if not HAS_NP:
            return None
        mean = float(gray.mean())
        # "Bright" relative to the local patch — works in shade and sun.
        thr = max(mean + 25.0, float(np.percentile(gray, 92)))
        mask = gray >= thr
        n_bright = int(mask.sum())
        area = gray.shape[0] * gray.shape[1]
        # A ball is a SMALL bright cluster: some bright pixels, but only a
        # small fraction of the patch. A white shoe / sock / marker fills
        # far more of it (false positive: part of a white shoe was marked
        # as the ball while the golfer walked).
        if n_bright < 4 or n_bright > 0.22 * area:
            return False
        ys, xs = np.nonzero(mask)
        cy, cx = float(ys.mean()), float(xs.mean())
        # Cluster centred reasonably near the marked point.
        return (
            abs(cx - (gray.shape[1] / 2.0)) <= r * 0.8
            and abs(cy - (gray.shape[0] / 2.0)) <= r * 0.8
        )
    except Exception:  # noqa: BLE001
        return None


_JUDGE_SWING_HEAT_PROMPT = (
    "You are looking at a MOTION-HEAT visualization from a fixed golf tee "
    "camera: one video frame with accumulated motion overlaid in color "
    "(blue = pixels that moved only briefly, green/yellow/orange/red = "
    "progressively more constant motion).\n"
    "Question: does this show a GOLFER SWINGING A GOLF CLUB during this "
    "window?\n"
    "A real swing shows a standing human silhouette in heat AND a fan or "
    "arc of thin streaks sweeping around/above the figure — the club "
    "shaft painting successive positions — sometimes with a dotted trail "
    "of brief motion leaving the scene (the ball).\n"
    "Answer false for: a person walking (a diffuse smeared blob with no "
    "fan), someone bending over or placing a tee, foliage/tree shimmer, "
    "or an empty scene.\n"
    'Reply with JSON only:\n{"is_swing": true|false, '
    '"confidence": "high"|"medium"|"low", "reason": "<one short sentence>"}'
)


def judge_swing_heat_image(image_path, model: str | None = None) -> dict:
    """One Claude vision call: does this motion-heat composite look like a
    golfer swinging a club? The operator A/B-tested this against the
    ray-counting heuristic and the model read the gestalt correctly where
    the heuristic inverted (walking ghost kept, real fans dropped).
    Returns {available, is_swing (bool|None), confidence, reason}.
    Never raises."""
    out = {"available": False, "is_swing": None, "confidence": None, "reason": None}
    if not HAS_ANTHROPIC or not os.environ.get("ANTHROPIC_API_KEY"):
        out["reason"] = "ANTHROPIC_API_KEY not set"
        return out
    try:
        data = Path(image_path).read_bytes()
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"could not read image: {exc}"
        return out
    try:
        client = _anthropic_client()
        resp = client.messages.create(
            model=_resolve_frame_picker_model(model),
            max_tokens=150,
            system=[{
                "type": "text",
                "text": _JUDGE_SWING_HEAT_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "JSON only."},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(data).decode("ascii"),
                }},
            ]}],
        )
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"api_failed: {exc}"
        return out
    text = "".join(
        c.text for c in resp.content if getattr(c, "type", None) == "text"
    )
    parsed = _extract_json(text) or {}
    out["available"] = True
    if isinstance(parsed.get("is_swing"), bool):
        out["is_swing"] = parsed["is_swing"]
    out["confidence"] = parsed.get("confidence")
    out["reason"] = parsed.get("reason")
    return out


def classify_swing_shot(
    input_path: Path,
    peak_time_sec: float,
    fps: float,
    leads: tuple = (1.5, 1.0, 0.5),
    after_sec: float = 1.5,
    move_tol_frac: float = 0.06,
    hint_xy: tuple[float, float] | None = None,
) -> dict:
    """Real shot vs practice swing, by ball departure.

    A real swing makes a resting ball leave; a practice/air swing / whiff
    does not. Find the resting ball BEFORE the swing (trying each lead in
    turn — the club can hide it at 1.5s but not nearer the top) and check
    whether it's gone AFTER (club has followed through, so the spot is
    clear). Verdict:
      * no ball before        -> practice (air swing)
      * ball before, gone after -> real (ball departed)
      * ball before, still there after (same spot) -> practice (not struck)

    Returns {verdict: 'real'|'practice'|'unknown', reason, before, after}.
    'unknown' when the AI ball isn't available (no key) — callers should
    treat unknown as keep, never drop a real swing on a missing key."""
    if not (HAS_CV and HAS_ANTHROPIC and os.environ.get("ANTHROPIC_API_KEY")):
        return {"verdict": "unknown", "reason": "AI ball unavailable",
                "before": None, "after": None}

    # hint_xy (the golfer's hands from the pose detector) zooms every ball
    # look-up to the golfer instead of scanning the full wide frame — on a
    # course tee shot the ball is otherwise a ~4px dot the model misses.
    _hint = tuple(hint_xy) if hint_xy else None
    before = None
    probe = None  # last absent look-up, kept so the debug UI can show it
    _errs: list[str] = []  # vision-call FAILURES (rate limit, network) —
    # must never masquerade as "looked and saw no ball"
    for lead in leads:
        t_b = max(0.0, float(peak_time_sec) - lead)
        r = find_resting_ball(input_path, int(t_b * fps), crop_center=_hint)
        if r.get("error"):
            _errs.append(str(r["error"]))
        if r.get("present") and r.get("x") is not None:
            before = {
                "present": True, "x": r["x"], "y": r["y"],
                "t": round(t_b, 2), "lead": lead, "confidence": r.get("confidence"),
                "crop_box": r.get("crop_box"),
            }
            break
        probe = {
            "present": False, "x": None, "y": None,
            "t": round(t_b, 2), "lead": lead, "crop_box": r.get("crop_box"),
        }
    # Every zoomed look missed. The zoom crop is aimed at the pose wrist
    # point — if that point was garbled (pose dropout near the blurred
    # peak), the crop may not even contain the ball. One full-frame retry
    # at the nearest lead so a bad hint can't guarantee a miss.
    if before is None and _hint is not None:
        t_b = max(0.0, float(peak_time_sec) - leads[-1])
        r = find_resting_ball(input_path, int(t_b * fps))
        if r.get("error"):
            _errs.append(str(r["error"]))
        if r.get("present") and r.get("x") is not None:
            before = {
                "present": True, "x": r["x"], "y": r["y"],
                "t": round(t_b, 2), "lead": leads[-1],
                "confidence": r.get("confidence"), "crop_box": None,
            }
    before_out = before or probe

    t_a = float(peak_time_sec) + after_sec
    ra = find_resting_ball(input_path, int(t_a * fps), crop_center=_hint)
    if ra.get("error"):
        _errs.append(str(ra["error"]))
    after = {
        "present": bool(ra.get("present") and ra.get("x") is not None),
        "x": ra.get("x"), "y": ra.get("y"), "t": round(t_a, 2),
        "confidence": ra.get("confidence"), "crop_box": ra.get("crop_box"),
    }

    # Decision matrix, biased so a DETECTION FAILURE can never kill a real
    # shot — "practice" (the verdict that drops the swing from production)
    # requires positive evidence: a credible resting ball still sitting
    # there after the swing.
    #
    #   before found + after gone            -> real (ball departed)
    #   before found + after same spot       -> practice (not struck)
    #   before found + after different spot  -> unknown (re-teed ball vs a
    #        spare ball lying nearby — ambiguous either way, keep). NOTE:
    #        this used to say REAL ("ball moved"), which mislabelled
    #        placing/teeing the ball as a shot.
    #   before missed + after found          -> practice (a ball sat there
    #        through the swing; the address look-up just missed it)
    #   before missed + after gone           -> unknown (we never saw a
    #        ball at all — can't judge; used to say practice, which
    #        dropped real swings whenever the address probe failed)
    #
    # Any "practice" additionally pixel-verifies the marked point actually
    # looks like a white ball; a non-ball latch (tee marker / leaf) would
    # sit "unmoved" across every swing and silently kill real shots.
    if before is None and not after["present"]:
        if _errs:
            # The look-ups didn't fail to SEE a ball — the calls
            # themselves failed. Say so, loudly: an API rate-limit
            # looks exactly like "no ball" otherwise and sends the
            # operator hunting a detection bug that isn't there.
            log.warning(
                "classify_swing_shot: %d ball look-up call(s) FAILED: %s",
                len(_errs), _errs[0],
            )
            return {
                "verdict": "unknown",
                "reason": (
                    f"ball look-up calls FAILED ({_errs[0][:120]}) — "
                    f"not a detection miss; can't judge, keeping"
                ),
                "errors": _errs[:3],
                "before": before_out, "after": after,
            }
        return {
            "verdict": "unknown",
            "reason": "couldn't find a ball before or after — can't judge, keeping",
            "before": before_out, "after": after,
        }
    if before is None and after["present"]:
        a_ok = _white_blob_at(input_path, int(after["t"] * fps), after["x"], after["y"])
        if a_ok is False:
            return {
                "verdict": "unknown",
                "reason": "marked point doesn't look like a ball — not trusting it",
                "before": before_out, "after": after,
            }
        return {
            "verdict": "practice",
            "reason": "ball still resting after the swing (not struck)",
            "before": before_out, "after": after,
        }
    if not after["present"]:
        # Verify the BEFORE mark actually looks like a ball before
        # asserting a real shot — a white shoe marked as the "ball" while
        # the golfer walks away also reads as found-then-gone.
        b_ok = _white_blob_at(
            input_path, int(before["t"] * fps), before["x"], before["y"],
        )
        if b_ok is False:
            return {
                "verdict": "unknown",
                "reason": "before mark doesn't look like a ball — not "
                          "trusting 'departed' verdict",
                "before": before_out, "after": after,
            }
        return {"verdict": "real", "reason": "ball departed",
                "before": before_out, "after": after}

    # Ball found both before and after — same spot means not struck.
    try:
        cap = cv2.VideoCapture(str(input_path))
        w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920.0)
        h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080.0)
        cap.release()
    except Exception:  # noqa: BLE001
        w, h = 1920.0, 1080.0
    diag = (w * w + h * h) ** 0.5
    dist = ((before["x"] - after["x"]) ** 2 + (before["y"] - after["y"]) ** 2) ** 0.5
    if dist <= move_tol_frac * diag:
        b_ok = _white_blob_at(input_path, int(before["t"] * fps), before["x"], before["y"])
        a_ok = _white_blob_at(input_path, int(after["t"] * fps), after["x"], after["y"])
        if b_ok is False or a_ok is False:
            return {
                "verdict": "unknown",
                "reason": "marked point doesn't look like a ball — not trusting 'unmoved' verdict",
                "before": before_out, "after": after,
            }
        return {"verdict": "practice", "reason": "ball still at rest (not struck)",
                "before": before_out, "after": after}
    return {
        "verdict": "unknown",
        "reason": "resting ball visible after the swing at a different spot "
                  "(re-teed vs spare ball — ambiguous), keeping",
        "before": before_out, "after": after,
    }


def detect_swings_from_ai_ball(
    input_path: Path,
    fps: float | None = None,
    roi: dict | None = None,
    sample_every_sec: float = 1.3,
    max_frames: int = 20,
    min_rest_sec: float = 1.0,
    min_separation_sec: float = 4.0,
    before_sec: float = 3.5,
    after_sec: float = 5.0,
    model: str | None = None,
    debug: dict | None = None,
) -> list[dict]:
    """Find swings by using Claude to recognize the RESTING ball, then
    detecting when it departs. Samples the clip every ~sample_every_sec (cap
    max_frames Claude calls), asks Claude for the resting ball per frame,
    builds a presence timeline, and calls a swing where the ball was present
    (in the ROI) then vanished. Coarse in time but robust where classical
    white-blob detection fails (club occlusion, lighting). Costs one Claude
    call per sampled frame. Never raises."""
    if debug is not None:
        debug.update({"reason": None, "method": "ai_resting_ball", "available": False})
    if not HAS_CV:
        if debug is not None:
            debug["reason"] = "opencv not installed"
        return []
    if not HAS_ANTHROPIC or not os.environ.get("ANTHROPIC_API_KEY"):
        if debug is not None:
            debug["reason"] = "ANTHROPIC_API_KEY not set on this deployment"
        return []
    if debug is not None:
        debug["available"] = True

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        if debug is not None:
            debug["reason"] = "could not open video"
        return []
    try:
        src_fps = float(fps) if fps else float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if total <= 1 or src_fps <= 0:
        if debug is not None:
            debug["reason"] = "video too short"
        return []
    duration = total / src_fps

    # Evenly spaced sample frame indices, capped at max_frames.
    n = min(max_frames, max(2, int(duration / max(0.3, sample_every_sec)) + 1))
    idxs = [int(round(k * (total - 1) / (n - 1))) for k in range(n)]

    def _in_roi(x, y):
        if not roi or x is None or y is None or not fw or not fh:
            return True
        rx0 = float(roi.get("x", 0.0)) * fw
        ry0 = float(roi.get("y", 0.0)) * fh
        rx1 = rx0 + float(roi.get("w", 1.0)) * fw
        ry1 = ry0 + float(roi.get("h", 1.0)) * fh
        return rx0 <= x <= rx1 and ry0 <= y <= ry1

    samples = []  # (t, present_in_roi, x, y)
    for fi in idxs:
        r = find_resting_ball(input_path, fi, model=model)
        present = bool(r.get("present")) and _in_roi(r.get("x"), r.get("y"))
        samples.append((fi / src_fps, present, r.get("x"), r.get("y")))

    # Departures: a present run (>= min_rest) that ends with the ball gone.
    departures = []  # (t_departure, x, y)
    run_start = None
    last_pos = (None, None)
    for k, (t, present, x, y) in enumerate(samples):
        if present:
            if run_start is None:
                run_start = t
            last_pos = (x, y)
        else:
            if run_start is not None and (t - run_start) >= min_rest_sec:
                # ball was resting, now gone → departure at previous sample
                dep_t = samples[k - 1][0] if k > 0 else t
                departures.append((dep_t, last_pos[0], last_pos[1]))
            run_start = None

    departures.sort(key=lambda d: d[0])
    kept = []
    for d in departures:
        if kept and d[0] - kept[-1][0] < min_separation_sec:
            continue
        kept.append(d)

    segments = []
    for t_dep, x, y in kept:
        segments.append({
            "peak_time_sec": float(t_dep),
            "start_sec": float(max(0.0, t_dep - before_sec)),
            "end_sec": float(min(duration, t_dep + after_sec)),
            "confidence": "medium",
            "ball_x": x,
            "ball_y": y,
        })

    n_present = sum(1 for _t, p, _x, _y in samples if p)
    log.info(
        "ai_tracer: detect_swings_from_ai_ball — %d swings, ball seen in %d/%d "
        "sampled frames",
        len(kept), n_present, len(samples),
    )
    if debug is not None:
        debug.update({
            "duration_sec": float(duration),
            "n_samples": len(samples),
            "n_ball_seen": n_present,
            "n_departures": len(kept),
            "samples": [
                {"t": round(t, 2), "present": bool(p),
                 "x": (int(x) if x is not None else None),
                 "y": (int(y) if y is not None else None)}
                for t, p, x, y in samples
            ],
            "peaks": [round(float(t), 2) for t, _x, _y in kept],
        })
    return segments


def detect_swings_from_ball(
    input_path: Path,
    fps: float | None = None,
    sample_hz: float = 15.0,
    min_rest_sec: float = 0.8,
    occlusion_tol_sec: float = 1.5,
    min_separation_sec: float = 4.0,
    before_sec: float = 3.5,
    after_sec: float = 5.0,
    white_v_min: int = 170,
    white_s_max: int = 90,
    min_ball_frac: float = 0.00002,
    max_ball_frac: float = 0.006,
    circularity_min: float = 0.45,
    stationary_tol_frac: float = 0.018,
    roi: dict | None = None,
    debug: dict | None = None,
) -> list[dict]:
    """Find swings by tracking a resting white ball that suddenly departs.

    A real golf shot is the one event that makes a stationary ball leave:
    the ball sits still on the mat, then is struck and vanishes from that
    spot. Practice swings, walk-bys and ball pickups don't produce that
    signature, so counting ball departures gives the correct swing count —
    and hands the ball position + impact frame to the tracer for free.

    v2 — occlusion-tolerant. From behind the golfer the club sole settles
    behind the ball at address and partially (or briefly fully) hides it, so
    the ball is rarely visible cleanly for a full second at a stretch. Two
    changes handle that:
      * circularity_min is relaxed so a partially-occluded *crescent* of the
        ball still counts (the club covering part of it doesn't drop the
        detection), and
      * a track survives being unmatched for up to occlusion_tol_sec (the ball
        flickering behind the moving club) and only DEPARTS after a sustained
        absence longer than that — i.e. the real impact, when it's gone for
        good. Rest is measured as the span (first→last sighting), tolerating
        internal occlusion gaps.

    The caller (produce-debug) then motion-gates departures: only a departure
    that coincides with a swing motion burst counts, which separates "club
    arrived at address" from "ball actually struck".

    Returns the same segment shape as detect_swings_from_motion (plus
    ball_x / ball_y / rest_sec). Empty + a debug reason on any failure;
    never raises.
    """
    if debug is not None:
        debug.update({"reason": None, "method": "ball_departure"})
    if not HAS_CV or not HAS_NP:
        if debug is not None:
            debug["reason"] = "opencv or numpy not installed"
        return []

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        if debug is not None:
            debug["reason"] = "could not open video"
        return []

    samples: list[tuple[float, list[tuple[float, float, float]]]] = []
    frame_shape: tuple[int, int] | None = None
    n_cand_total = 0   # white/round/size-passing blobs before the ROI gate
    n_cand_in_roi = 0  # ... that also fell inside the ROI
    sample_cands: list[tuple[int, int]] = []  # native positions, for diagnostics
    try:
        src_fps = float(fps) if fps else float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if src_fps <= 0:
            src_fps = 30.0
        step = max(1, int(round(src_fps / sample_hz)))
        eff_hz = src_fps / step
        idx = -1
        while True:
            idx += 1
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx % step != 0:
                continue
            t = idx / src_fps
            h, w = frame.shape[:2]
            frame_shape = (h, w)
            # Downscale to ~360p for speed; keep the scale to map back.
            scale = 360.0 / h if h > 360 else 1.0
            fr = (
                cv2.resize(frame, (max(1, int(w * scale)), 360), interpolation=cv2.INTER_AREA)
                if scale != 1.0
                else frame
            )
            hs, ws = fr.shape[:2]
            hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(
                hsv, (0, 0, int(white_v_min)), (179, int(white_s_max), 255)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            frame_area = float(hs * ws)
            cands: list[tuple[float, float, float]] = []
            for c in cnts:
                a = cv2.contourArea(c)
                if a < min_ball_frac * frame_area or a > max_ball_frac * frame_area:
                    continue
                (cx, cy), rad = cv2.minEnclosingCircle(c)
                circ_area = np.pi * rad * rad
                if circ_area <= 0 or (a / circ_area) < circularity_min:
                    continue
                nx, ny = cx / scale, cy / scale  # native-pixel centroid
                n_cand_total += 1
                if len(sample_cands) < 400:
                    sample_cands.append((int(nx), int(ny)))
                # Tee-box ROI gate: drop candidates outside the drawn box
                # (fractions of the frame), killing shoes/glints elsewhere.
                if roi:
                    rx0 = float(roi.get("x", 0.0)) * w
                    ry0 = float(roi.get("y", 0.0)) * h
                    rx1 = rx0 + float(roi.get("w", 1.0)) * w
                    ry1 = ry0 + float(roi.get("h", 1.0)) * h
                    if not (rx0 <= nx <= rx1 and ry0 <= ny <= ry1):
                        continue
                n_cand_in_roi += 1
                cands.append((nx, ny, rad / scale))
            samples.append((t, cands))
    finally:
        cap.release()

    if not samples or frame_shape is None:
        if debug is not None and not debug.get("reason"):
            debug["reason"] = "no usable frames"
        return []

    h, w = frame_shape
    tol = stationary_tol_frac * float((h * h + w * w) ** 0.5)
    tracks: list[dict] = []
    departures: list[tuple[float, float, float, float]] = []  # (t, x, y, rest_dur)

    for t, cands in samples:
        used: set[int] = set()
        for tr in tracks:
            if not tr["alive"]:
                continue
            best, bestd = None, tol
            for ci, (cx, cy, _r) in enumerate(cands):
                if ci in used:
                    continue
                d = ((cx - tr["x"]) ** 2 + (cy - tr["y"]) ** 2) ** 0.5
                if d < bestd:
                    bestd, best = d, ci
            if best is not None:
                cx, cy, _r = cands[best]
                used.add(best)
                tr["x"] = 0.7 * tr["x"] + 0.3 * cx
                tr["y"] = 0.7 * tr["y"] + 0.3 * cy
                tr["last_t"] = t
                tr["hits"] += 1
            elif t - tr["last_t"] >= occlusion_tol_sec:
                rest_dur = tr["last_t"] - tr["first_t"]
                if rest_dur >= min_rest_sec:
                    departures.append((tr["last_t"], tr["x"], tr["y"], rest_dur))
                tr["alive"] = False
        for ci, (cx, cy, _r) in enumerate(cands):
            if ci in used:
                continue
            tracks.append(
                {"x": cx, "y": cy, "first_t": t, "last_t": t, "hits": 1, "alive": True}
            )

    departures.sort(key=lambda d: d[0])
    kept: list[tuple[float, float, float, float]] = []
    for d in departures:
        if kept and d[0] - kept[-1][0] < min_separation_sec:
            if d[3] > kept[-1][3]:
                kept[-1] = d
            continue
        kept.append(d)

    duration = samples[-1][0]
    segments: list[dict] = []
    for t_dep, x, y, rest_dur in kept:
        conf = "high" if rest_dur >= 1.5 else ("medium" if rest_dur >= 0.8 else "low")
        segments.append(
            {
                "peak_time_sec": float(t_dep),
                "start_sec": float(max(0.0, t_dep - before_sec)),
                "end_sec": float(min(duration, t_dep + after_sec)),
                "confidence": conf,
                "ball_x": float(x),
                "ball_y": float(y),
                "rest_sec": round(float(rest_dur), 2),
            }
        )

    log.info(
        "ai_tracer: detect_swings_from_ball — %d departures (raw=%d) "
        "duration=%.1fs hz=%.1f",
        len(kept), len(departures), duration, eff_hz,
    )
    n_rested = sum(
        1 for tr in tracks if (tr["last_t"] - tr["first_t"]) >= min_rest_sec
    )
    if debug is not None:
        debug.update(
            {
                "eff_hz": float(eff_hz),
                "duration_sec": float(duration),
                "n_departures": len(kept),
                "n_raw_departures": len(departures),
                # Diagnostics — where detection fell down when nothing is found:
                #   n_cand_total 0  -> ball not passing white/size filter
                #   in_roi 0        -> ball outside the drawn box
                #   n_rested 0      -> nothing held still >= min_rest_sec
                "n_cand_total": n_cand_total,
                "n_cand_in_roi": n_cand_in_roi,
                "n_tracks": len(tracks),
                "n_rested": n_rested,
                "min_rest_sec": float(min_rest_sec),
                "roi": roi,
                "sample_cands": sample_cands,
                "departures": [
                    {
                        "t": round(float(t), 2),
                        "x": round(float(x)),
                        "y": round(float(y)),
                        "rest_sec": round(float(rd), 2),
                    }
                    for t, x, y, rd in kept
                ],
            }
        )
    return segments


def compute_motion_trace(
    input_path: Path,
    fps: float | None = None,
    target_hz: float = 30.0,
    sample_height: int = 180,
    smooth_sec: float = 0.12,
    motion_ratio: float = 4.0,
    max_pts: int = 900,
) -> dict | None:
    """Full-rate motion trace for the DEBUG chart only.

    Same mean-pixel-difference signal as detect_swings_from_motion, but
    sampled at target_hz (default the full 30 Hz) instead of 10 Hz, so a
    ~33 ms downswing peak can't fall between samples and get aliased away —
    which made physically-identical swings plot at wildly different heights.
    Returns {series, hz, duration_sec, median, threshold} for plotting, or
    None if it can't run. Does not affect swing detection.
    """
    if not HAS_CV or not HAS_NP:
        return None
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return None
    try:
        src_fps = float(fps) if fps else float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if src_fps <= 0:
            src_fps = 30.0
        step = max(1, int(round(src_fps / target_hz)))
        eff_hz = src_fps / step
        diffs: list[float] = []
        prev = None
        idx = -1
        while True:
            idx += 1
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx % step != 0:
                continue
            h, w = frame.shape[:2]
            if h > sample_height:
                sc = sample_height / float(h)
                frame = cv2.resize(
                    frame, (max(1, int(w * sc)), sample_height),
                    interpolation=cv2.INTER_AREA,
                )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                diffs.append(float(cv2.absdiff(gray, prev).mean()))
            prev = gray
    finally:
        cap.release()

    if len(diffs) < 4:
        return None
    motion = np.asarray(diffs, dtype=np.float32)
    win = max(1, int(round(smooth_sec * eff_hz)))
    if 1 < win < motion.size:
        motion = np.convolve(motion, np.ones(win, np.float32) / win, mode="same")
    median = float(np.median(motion)) or 1e-6
    threshold = median * motion_ratio
    duration = (len(diffs) - 1) / eff_hz if eff_hz > 0 else 0.0
    series = motion
    if series.size > max_pts:
        b = int(np.ceil(series.size / max_pts))
        pad = (-series.size) % b
        if pad:
            series = np.concatenate([series, np.full(pad, series[-1], series.dtype)])
        series = series.reshape(-1, b).max(axis=1)
    return {
        "series": [round(float(v), 4) for v in series],
        "hz": float(eff_hz),
        "duration_sec": float(duration),
        "median": float(median),
        "threshold": float(threshold),
    }


def detect_swings_from_audio(
    input_path: Path,
    fps: float | None = None,
    min_separation_sec: float = 6.0,
    before_impact_sec: float = 4.5,
    after_impact_sec: float = 9.0,
    min_peak_ratio: float = 6.0,
    highpass_hz: float = 1500.0,
    max_attack_sec: float = 0.030,
    absolute_threshold_floor: float = 0.001,
    debug: dict | None = None,
    _cache_out: dict | None = None,
) -> list[dict]:
    """Find every club-on-ball impact in a long video by scanning its
    audio for sharp transients, then return one swing window per
    impact.

    Returns a list of segment dicts shaped like the manual-segments
    JSON the long-upload endpoint already accepts, but with no
    hole_number filled in (caller assigns those):

        [
          {
            "peak_time_sec": float,
            "start_sec": float,        # max(0, peak - before_impact_sec)
            "end_sec": float,          # min(duration, peak + after_impact_sec)
            "ratio": float,            # peak / median envelope
            "confidence": "high" | "medium" | "low",
          },
          ...
        ]

    Empty list when no audio, no peaks above threshold, or ffmpeg /
    numpy unavailable. Never raises.
    """
    if debug is not None:
        debug.update({
            "reason": None,
            "min_peak_ratio_used": float(min_peak_ratio),
            "min_separation_sec": float(min_separation_sec),
            "highpass_hz_used": float(highpass_hz),
            "max_attack_sec_used": float(max_attack_sec),
        })

    if not HAS_NP or shutil.which("ffmpeg") is None:
        log.info("ai_tracer: detect_swings_from_audio — missing numpy / ffmpeg")
        if debug is not None:
            debug["reason"] = "numpy or ffmpeg not installed"
        return []
    # High-pass before envelope extraction: a club-on-ball strike has
    # most of its energy > 1.5 kHz (the sharp 'crack'). Voices peak
    # 200-2 kHz, train rumble / wind / engine noise sit below 200 Hz.
    # Filtering at 1.5 kHz collapses those false-positive sources
    # while preserving the impact transient.
    audio_filter_args: list[str] = []
    if highpass_hz and highpass_hz > 0:
        audio_filter_args = ["-af", f"highpass=f={float(highpass_hz)}"]
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-v", "error",
                "-i", str(input_path),
                "-vn", "-ac", "1",
                "-ar", str(AUDIO_SAMPLE_RATE),
                *audio_filter_args,
                "-f", "s16le", "-",
            ],
            capture_output=True, timeout=120, check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        log.warning("ai_tracer: detect_swings ffmpeg failed: %s", exc)
        if debug is not None:
            debug["reason"] = f"ffmpeg failed: {exc}"
        return []
    if proc.returncode != 0 or not proc.stdout:
        log.info(
            "ai_tracer: detect_swings — no audio (rc=%d, stderr=%s)",
            proc.returncode,
            proc.stderr.decode("utf-8", "ignore")[:120],
        )
        if debug is not None:
            debug["reason"] = "no audio stream"
        return []
    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0:
        if debug is not None:
            debug["reason"] = "empty audio buffer"
        return []
    duration_sec = float(samples.size) / AUDIO_SAMPLE_RATE
    win = max(1, int(AUDIO_SAMPLE_RATE * AUDIO_ENVELOPE_WINDOW_MS / 1000.0))
    if samples.size <= win:
        if debug is not None:
            debug["reason"] = f"audio shorter than envelope window ({win} samples)"
        return []
    kernel = np.ones(win, dtype=np.float32) / float(win)
    envelope = np.convolve(np.abs(samples), kernel, mode="same")
    median = float(np.median(envelope))
    # Expose raw audio data to callers that need it for further signal
    # analysis (e.g. detect_swings_combined's ring-out / spectral checks).
    if _cache_out is not None:
        _cache_out["envelope"] = envelope
        _cache_out["samples"] = samples
        _cache_out["sr"] = int(AUDIO_SAMPLE_RATE)
        _cache_out["median"] = float(median)
        _cache_out["duration_sec"] = duration_sec
    raw_threshold = median * min_peak_ratio
    # Safety net for nearly-silent clips: without this, a clip with
    # almost no audio would have a tiny median and any digital noise
    # spike would become a huge ratio. The default (0.001) almost
    # never binds on a real recording but prevents nonsense on empty
    # ones. 0.02 (the old value) was tuned for un-filtered audio
    # where median ~ 0.013; with the 1.5 kHz high-pass it drops to
    # ~0.003 and the floor would dominate the ratio knob.
    threshold = max(raw_threshold, float(absolute_threshold_floor))
    threshold_floor_hit = threshold > raw_threshold
    min_sep = int(min_separation_sec * AUDIO_SAMPLE_RATE)
    if debug is not None:
        debug.update({
            "duration_sec": round(duration_sec, 2),
            "median_envelope": median,
            "threshold": threshold,
            "threshold_floor": float(absolute_threshold_floor),
            "threshold_floor_hit": threshold_floor_hit,
        })

    # Two-pass peak finder. First pass: every local maximum above
    # threshold. Second pass: greedy non-max suppression within
    # min_separation — keep the loudest peak in each cluster.
    above = envelope > threshold
    n = envelope.size
    # Raw peaks tagged with attack time: (pos, peak_value, attack_sec,
    # passes_attack). Attack time = (peak_pos - last sample below 10 %
    # of peak amplitude within a look-back window) / sample_rate. A
    # real club-on-ball impact rises from noise to peak in < 20 ms;
    # voices, rumble, and most ambient sounds are sustained and ramp
    # up over 50+ ms, so the attack filter drops them while keeping
    # impulsive transients.
    max_attack_samples = max(1, int(max_attack_sec * AUDIO_SAMPLE_RATE))
    attack_lookback = max(win, max_attack_samples * 3)
    raw_peaks: list[tuple[int, float, float, bool]] = []
    for i in range(5, n - 5):
        if not above[i]:
            continue
        v = float(envelope[i])
        if not (
            v >= envelope[i - 1] and v >= envelope[i + 1]
            and v >= envelope[i - 3] and v >= envelope[i + 3]
            and v >= envelope[i - 5] and v >= envelope[i + 5]
        ):
            continue
        thresh10 = 0.10 * v
        start_look = max(0, i - attack_lookback)
        # np.argwhere is slow per-peak, but the lookback window is small
        # (a few hundred samples). Walk back to the most recent sub-10 %
        # crossing.
        attack_samples: int | None = None
        for k in range(i - 1, start_look - 1, -1):
            if float(envelope[k]) < thresh10:
                attack_samples = i - k
                break
        if attack_samples is None:
            attack_samples = attack_lookback + 1  # never fell below — fail.
        attack_sec = attack_samples / float(AUDIO_SAMPLE_RATE)
        passes_attack = attack_sec <= max_attack_sec
        raw_peaks.append((i, v, attack_sec, passes_attack))

    n_raw = len(raw_peaks)
    attack_filtered = [(p, v) for p, v, _a, ok in raw_peaks if ok]
    if debug is not None:
        debug["n_raw_peaks"] = n_raw
        debug["n_after_attack"] = len(attack_filtered)
    if not attack_filtered:
        log.info(
            "ai_tracer: detect_swings — %d raw peaks, 0 passed attack filter "
            "(<= %dms); median=%.4f threshold=%.4f",
            n_raw, int(max_attack_sec * 1000), median, threshold,
        )
        if debug is not None:
            debug["reason"] = (
                f"{n_raw} raw peak(s) but none had attack <= {int(max_attack_sec*1000)}ms"
                if n_raw else
                f"no local maxima above threshold (median={median:.4f}, "
                f"threshold={threshold:.4f}, min_peak_ratio={min_peak_ratio})"
            )
            debug["n_after_nms"] = 0
            # Surface raw peaks anyway so the operator can see why they
            # all failed (which were too slow).
            top = sorted(raw_peaks, key=lambda t: -t[1])[:15]
            top_sorted_by_time = sorted(top, key=lambda t: t[0])
            debug["top_peaks"] = [
                {
                    "peak_sec": round(pos / float(AUDIO_SAMPLE_RATE), 2),
                    "ratio": round(float(v) / median, 2) if median > 0 else None,
                    "attack_ms": round(a * 1000, 1),
                    "passes_attack": ok,
                    "kept": False,
                }
                for pos, v, a, ok in top_sorted_by_time
            ]
        return []

    # Sort by amplitude desc, then walk; for each candidate peak, if
    # we haven't already accepted a peak within ±min_sep samples, take
    # this one.
    attack_filtered.sort(key=lambda t: -t[1])
    accepted_positions: list[int] = []
    for pos, _v in attack_filtered:
        if any(abs(pos - acc) < min_sep for acc in accepted_positions):
            continue
        accepted_positions.append(pos)
    accepted_positions.sort()

    if debug is not None:
        kept_set = set(accepted_positions)
        # Top 15 from ALL raw peaks (including attack-rejected) so the
        # operator can see which were dropped for being too slow vs.
        # which survived through NMS.
        top = sorted(raw_peaks, key=lambda t: -t[1])[:15]
        top_sorted_by_time = sorted(top, key=lambda t: t[0])
        debug["n_after_nms"] = len(accepted_positions)
        debug["top_peaks"] = [
            {
                "peak_sec": round(pos / float(AUDIO_SAMPLE_RATE), 2),
                "ratio": round(float(v) / median, 2) if median > 0 else None,
                "attack_ms": round(a * 1000, 1),
                "passes_attack": ok,
                "kept": pos in kept_set,
            }
            for pos, v, a, ok in top_sorted_by_time
        ]

    segments: list[dict] = []
    for pos in accepted_positions:
        peak_t = pos / float(AUDIO_SAMPLE_RATE)
        peak_value = float(envelope[pos])
        ratio = (peak_value / median) if median > 0 else float("inf")
        if ratio >= 15:
            conf = "high"
        elif ratio >= 8:
            conf = "medium"
        else:
            conf = "low"
        segments.append({
            "peak_time_sec": float(peak_t),
            "start_sec": max(0.0, peak_t - before_impact_sec),
            "end_sec": min(duration_sec, peak_t + after_impact_sec),
            "ratio": float(ratio) if ratio != float("inf") else None,
            "confidence": conf,
        })
    log.info(
        "ai_tracer: detect_swings — %d swing windows from audio "
        "(duration=%.1fs median=%.4f threshold=%.4f)",
        len(segments), duration_sec, median, threshold,
    )
    return segments


def _build_motion_array(
    input_path: Path,
    fps: float | None = None,
    sample_height: int = 180,
    target_hz: float = 10.0,
) -> "tuple[np.ndarray | None, float, float, float]":
    """Decode a video at low resolution and return a smoothed frame-diff
    motion timeseries.

    Returns ``(motion_array, effective_hz, median_motion, duration_sec)``.
    Returns ``(None, 0.0, 0.0, 0.0)`` when opencv / numpy is unavailable
    or the video cannot be opened.  Never raises.

    The returned array is the same signal used internally by
    detect_swings_from_motion, factored out so that
    detect_swings_combined can index into it directly for per-candidate
    signal checks without running the full motion detector again.
    """
    if not HAS_CV or not HAS_NP:
        return None, 0.0, 0.0, 0.0
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return None, 0.0, 0.0, 0.0
    try:
        src_fps = float(fps) if fps else float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if src_fps <= 0:
            src_fps = 30.0
        step = max(1, int(round(src_fps / target_hz)))
        effective_hz = src_fps / step
        diffs: list[float] = []
        prev_gray = None
        idx = -1
        while True:
            idx += 1
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx % step != 0:
                continue
            h, w = frame.shape[:2]
            if h > sample_height:
                scale = sample_height / float(h)
                new_w = max(1, int(round(w * scale)))
                frame = cv2.resize(
                    frame, (new_w, sample_height), interpolation=cv2.INTER_AREA
                )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                d = cv2.absdiff(gray, prev_gray)
                diffs.append(float(d.mean()))
            prev_gray = gray
    finally:
        cap.release()

    if len(diffs) < 4:
        return None, 0.0, 0.0, 0.0
    motion = np.asarray(diffs, dtype=np.float32)
    smooth_win = max(1, int(round(0.3 * effective_hz)))
    if smooth_win > 1 and smooth_win < motion.size:
        kernel = np.ones(smooth_win, dtype=np.float32) / float(smooth_win)
        motion = np.convolve(motion, kernel, mode="same")
    median = float(np.median(motion))
    if median <= 1e-6:
        median = 1e-6
    duration_sec = (len(diffs) - 1) / effective_hz if effective_hz > 0 else 0.0
    return motion, float(effective_hz), median, duration_sec


def detect_swings_combined(
    input_path: Path,
    fps: float | None = None,
    audio_min_peak_ratio: float = SWING_AUDIO_MIN_PEAK_RATIO,
    motion_ratio: float = 2.0,
    pair_window_sec: float = 3.0,
    before_impact_sec: float = 4.5,
    after_impact_sec: float = 9.0,
    audio_max_rise_ms: float = SWING_AUDIO_MAX_RISE_MS,
    audio_max_duration_ms: float = SWING_AUDIO_MAX_DURATION_MS,
    audio_min_spectral_centroid_hz: float = SWING_AUDIO_MIN_SPECTRAL_CENTROID_HZ,
    backswing_window_s: float = SWING_BACKSWING_WINDOW_S,
    followthrough_window_s: float = SWING_FOLLOWTHROUGH_WINDOW_S,
    min_motion_ratio_at_impact: float = SWING_MIN_MOTION_RATIO,
    motion_floor_ratio: float = SWING_MOTION_FLOOR_RATIO,
    debug: dict | None = None,
) -> list[dict]:
    """4-signal heuristic swing detector.

    Each audio candidate is evaluated against four independent gates.
    All hard gates must pass for an event to be accepted. Every
    candidate's per-signal outcome is logged so failures are easy to
    diagnose from server logs or the ``debug`` dict.

    Gate 1 — Sharp audio transient
        1a. Attack (rise time) ≤ audio_max_rise_ms.  This is already
            enforced upstream by detect_swings_from_audio's built-in
            attack filter, so only candidates that cleared it arrive
            here.  The gate is recorded as always-OK in the report.
        1b. Ring-out duration ≤ audio_max_duration_ms.  After the
            envelope peak, we measure how long the envelope stays above
            20 % of the peak value.  A real "thwack" decays in < 150 ms;
            a ball dropped on a hard floor, a hand clap, or voice
            sustains longer.
        1c. Spectral centroid ≥ audio_min_spectral_centroid_hz (Hz).
            Computed via rfft on the 100 ms window around the peak
            (after the 1.5 kHz high-pass already applied).  Club-on-ball
            is broadband; footsteps / clothing rustle / low thuds sit
            below 2.5 kHz.  Set audio_min_spectral_centroid_hz=0 to
            disable this sub-gate.

    Gate 2 — Motion at impact
        The raw frame-diff signal at the audio peak timestamp must be
        ≥ min_motion_ratio_at_impact × clip_median_motion.  A ball drop
        produces almost zero motion at the moment of the sound; a real
        swing produces a large spike.

    Gate 3 — Backswing + follow-through
        Mean motion in the window [peak − backswing_window_s, peak]
        must be ≥ motion_floor_ratio × clip_median AND mean motion in
        [peak, peak + followthrough_window_s] must also clear that
        floor.  Ensures the event is a true arc, not a sudden audio
        transient from something off-screen or a single-frame camera bump.

    Gate 4 — Person in swing zone (soft / log-only)
        Checks whether the audio peak falls inside a motion burst from
        detect_swings_from_motion (within ±pair_window_sec).  Logged but
        NOT a hard rejection gate — no person-tracking is available yet,
        so this is a best-effort coarse spatial sanity check.

    Motion-only fallback
        Activated only when audio detection returns zero candidates
        (quiet mic, no audio track, lossy codec stripped audio, etc.).
        Motion bursts from detect_swings_from_motion are passed through
        a lightweight Gate 3 check (burst duration covers backswing and
        follow-through windows) and returned.  This path is kept strictly
        separate from the 4-signal path so the two strategies don't
        interfere.
    """
    log.info(
        "ai_tracer: detect_swings_combined — tunables: "
        "audio_min_peak_ratio=%.1f audio_max_rise_ms=%.0f "
        "audio_max_duration_ms=%.0f audio_min_spectral_centroid_hz=%.0f "
        "backswing_window_s=%.2f followthrough_window_s=%.2f "
        "min_motion_ratio_at_impact=%.1f motion_floor_ratio=%.1f "
        "pair_window_sec=%.1f before_impact_sec=%.1f after_impact_sec=%.1f",
        audio_min_peak_ratio, audio_max_rise_ms,
        audio_max_duration_ms, audio_min_spectral_centroid_hz,
        backswing_window_s, followthrough_window_s,
        min_motion_ratio_at_impact, motion_floor_ratio,
        pair_window_sec, before_impact_sec, after_impact_sec,
    )

    audio_debug: dict = {}
    motion_debug: dict = {}
    audio_cache: dict = {}

    # ── Source 1: audio candidates ────────────────────────────────────
    audio_windows = detect_swings_from_audio(
        input_path,
        fps=fps,
        min_peak_ratio=audio_min_peak_ratio,
        max_attack_sec=audio_max_rise_ms / 1000.0,
        before_impact_sec=before_impact_sec,
        after_impact_sec=after_impact_sec,
        debug=audio_debug,
        _cache_out=audio_cache,
    )

    # ── Source 2: motion burst list (for Gate 4 / fallback) ───────────
    motion_windows = detect_swings_from_motion(
        input_path,
        fps=fps,
        motion_ratio=motion_ratio,
        debug=motion_debug,
    )

    # ── Source 3: raw motion timeseries (for Gates 2 & 3) ────────────
    motion_array, motion_hz, motion_median, _motion_dur = _build_motion_array(
        input_path, fps=fps, sample_height=180,
    )

    # Unpack audio cache (populated by detect_swings_from_audio via _cache_out)
    envelope: "np.ndarray | None" = audio_cache.get("envelope")
    raw_samples: "np.ndarray | None" = audio_cache.get("samples")
    audio_sr: int = int(audio_cache.get("sr", AUDIO_SAMPLE_RATE))

    # ── 4-signal evaluation of each audio candidate ───────────────────
    accepted_windows: list[dict] = []
    candidate_reports: list[dict] = []
    # Tally how many candidates each hard gate killed (a single candidate
    # can fail multiple gates; all are counted independently).
    rejection_counts: dict[str, int] = {
        "audio_transient": 0,
        "motion_intensity": 0,
        "backswing": 0,
        "followthrough": 0,
    }

    for aw in audio_windows:
        peak_t = aw.get("peak_time_sec")
        if peak_t is None:
            continue
        peak_t = float(peak_t)

        report: dict = {
            "peak_sec": round(peak_t, 2),
            "audio_ratio": round(float(aw.get("ratio") or 0.0), 1),
        }

        # Gate 1a — attack (pre-filtered; always True here) ──────────
        report["s1a_attack_ok"] = True

        # Gate 1b — ring-out duration ─────────────────────────────────
        s1b_duration_ok: bool = True
        ring_out_ms: float | None = None
        if envelope is not None and audio_sr > 0:
            pk_s = max(0, min(int(round(peak_t * audio_sr)), len(envelope) - 1))
            pk_val = float(envelope[pk_s])
            ring_threshold = pk_val * 0.20
            look_ahead = min(len(envelope), pk_s + int(0.350 * audio_sr))
            ring_end = pk_s
            for k in range(pk_s, look_ahead):
                if float(envelope[k]) >= ring_threshold:
                    ring_end = k
            ring_out_ms = (ring_end - pk_s) / float(audio_sr) * 1000.0
            if audio_max_duration_ms > 0:
                s1b_duration_ok = ring_out_ms <= audio_max_duration_ms
        report["s1b_ring_out_ms"] = (
            round(ring_out_ms, 1) if ring_out_ms is not None else None
        )
        report["s1b_duration_ok"] = s1b_duration_ok

        # Gate 1c — spectral centroid (broadband check) ───────────────
        s1c_broadband_ok: bool = True
        spectral_centroid_hz: float | None = None
        if (
            raw_samples is not None
            and HAS_NP
            and audio_sr > 0
            and audio_min_spectral_centroid_hz > 0
        ):
            hw = int(0.050 * audio_sr)
            s0 = max(0, int(round(peak_t * audio_sr)) - hw)
            s1_ = min(len(raw_samples), int(round(peak_t * audio_sr)) + hw)
            window = raw_samples[s0:s1_]
            if window.size >= 8:
                magnitudes = np.abs(np.fft.rfft(window))
                freqs = np.fft.rfftfreq(window.size, d=1.0 / audio_sr)
                denom = float(np.sum(magnitudes)) + 1e-10
                spectral_centroid_hz = float(np.sum(freqs * magnitudes) / denom)
                s1c_broadband_ok = spectral_centroid_hz >= audio_min_spectral_centroid_hz
        report["s1c_spectral_centroid_hz"] = (
            round(spectral_centroid_hz, 0)
            if spectral_centroid_hz is not None else None
        )
        report["s1c_broadband_ok"] = s1c_broadband_ok

        # Gate 2 — motion magnitude at impact ─────────────────────────
        s2_motion_ok: bool = True
        motion_at_impact_ratio: float | None = None
        if motion_array is not None and motion_hz > 0 and motion_median > 0:
            m_idx = max(0, min(
                int(round(peak_t * motion_hz)), len(motion_array) - 1
            ))
            motion_at_impact_ratio = float(motion_array[m_idx]) / motion_median
            s2_motion_ok = motion_at_impact_ratio >= min_motion_ratio_at_impact
        report["s2_motion_at_impact_ratio"] = (
            round(motion_at_impact_ratio, 2)
            if motion_at_impact_ratio is not None else None
        )
        report["s2_motion_ok"] = s2_motion_ok

        # Gate 3a — backswing motion ───────────────────────────────────
        s3a_backswing_ok: bool = True
        backswing_mean_ratio: float | None = None
        if motion_array is not None and motion_hz > 0 and motion_median > 0:
            bs_end = max(0, int(round(peak_t * motion_hz)))
            bs_start = max(0, int(round((peak_t - backswing_window_s) * motion_hz)))
            if bs_end > bs_start:
                bs_mean = float(np.mean(motion_array[bs_start:bs_end]))
                backswing_mean_ratio = bs_mean / motion_median
                s3a_backswing_ok = backswing_mean_ratio >= motion_floor_ratio
        report["s3a_backswing_mean_ratio"] = (
            round(backswing_mean_ratio, 2)
            if backswing_mean_ratio is not None else None
        )
        report["s3a_backswing_ok"] = s3a_backswing_ok

        # Gate 3b — follow-through motion ─────────────────────────────
        s3b_ft_ok: bool = True
        ft_mean_ratio: float | None = None
        if motion_array is not None and motion_hz > 0 and motion_median > 0:
            ft_start = int(round(peak_t * motion_hz))
            ft_end = min(
                len(motion_array),
                int(round((peak_t + followthrough_window_s) * motion_hz)),
            )
            if ft_end > ft_start:
                ft_mean = float(np.mean(motion_array[ft_start:ft_end]))
                ft_mean_ratio = ft_mean / motion_median
                s3b_ft_ok = ft_mean_ratio >= motion_floor_ratio
        report["s3b_ft_mean_ratio"] = (
            round(ft_mean_ratio, 2) if ft_mean_ratio is not None else None
        )
        report["s3b_ft_ok"] = s3b_ft_ok

        # Gate 4 — person in swing zone (soft / log-only) ─────────────
        # Proximity check: does a motion burst from detect_swings_from_motion
        # land within ±pair_window_sec of this audio peak?  Logged only;
        # does not affect the hard_pass decision.
        s4_person_ok: bool | None = None
        s4_dt: float | None = None
        for mw in motion_windows:
            m_t = mw.get("peak_time_sec")
            if m_t is None:
                continue
            dt = abs(peak_t - float(m_t))
            if s4_dt is None or dt < s4_dt:
                s4_dt = dt
        if s4_dt is not None:
            s4_person_ok = s4_dt <= pair_window_sec
        report["s4_person_in_zone_ok"] = s4_person_ok
        report["s4_nearest_burst_dt_sec"] = (
            round(s4_dt, 2) if s4_dt is not None else None
        )

        # ── Decision ──────────────────────────────────────────────────
        # Hard gates: S1b, S1c, S2, S3a, S3b.
        # S1b + S1c together form the "audio_transient" gate.
        # Soft (logged only): S4.
        audio_transient_pass = s1b_duration_ok and s1c_broadband_ok
        hard_pass = (
            audio_transient_pass
            and s2_motion_ok
            and s3a_backswing_ok
            and s3b_ft_ok
        )
        report["hard_pass"] = hard_pass
        candidate_reports.append(report)

        # Rise time: scan backward from the peak sample until the envelope
        # drops below 10 % of the peak value (max 200 ms lookback).
        rise_ms_int: int = 0
        if envelope is not None and audio_sr > 0:
            _pk_s = max(0, min(int(round(peak_t * audio_sr)), len(envelope) - 1))
            _pk_val = float(envelope[_pk_s])
            _lookback = max(0, _pk_s - int(0.200 * audio_sr))
            _rise_start = _pk_s
            for _j in range(_pk_s, _lookback, -1):
                if float(envelope[_j]) < _pk_val * 0.10:
                    _rise_start = _j
                    break
            rise_ms_int = max(0, int(round(
                (_pk_s - _rise_start) / float(audio_sr) * 1000.0
            )))

        log.info(
            "swing candidate t=%.2fs: "
            "audio_transient=%s(rise=%dms,dur=%dms,centroid=%.0fHz) "
            "motion_at_impact=%s(ratio=%.2f) "
            "backswing=%s followthrough=%s in_roi=%s -> %s",
            peak_t,
            "OK" if audio_transient_pass else "FAIL",
            rise_ms_int,
            int(round(ring_out_ms)) if ring_out_ms is not None else -1,
            spectral_centroid_hz if spectral_centroid_hz is not None else -1.0,
            "OK" if s2_motion_ok else "FAIL",
            motion_at_impact_ratio if motion_at_impact_ratio is not None else -1.0,
            "OK" if s3a_backswing_ok else "FAIL",
            "OK" if s3b_ft_ok else "FAIL",
            "OK" if s4_person_ok else ("FAIL" if s4_person_ok is False else "?"),
            "ACCEPTED" if hard_pass else "REJECTED",
        )

        # Track per-gate rejection tallies for the end-of-loop summary.
        if not hard_pass:
            if not audio_transient_pass:
                rejection_counts["audio_transient"] += 1
            if not s2_motion_ok:
                rejection_counts["motion_intensity"] += 1
            if not s3a_backswing_ok:
                rejection_counts["backswing"] += 1
            if not s3b_ft_ok:
                rejection_counts["followthrough"] += 1

        if hard_pass:
            accepted_windows.append(aw)

    # ── Rejection summary (logged when all audio candidates fail) ─────
    if audio_windows and not accepted_windows:
        breakdown = ", ".join(
            f"{gate}={count}"
            for gate, count in rejection_counts.items()
            if count > 0
        ) or "unknown"
        log.info(
            "no swings detected: %d candidate(s) considered, "
            "rejected by: %s",
            len(audio_windows),
            breakdown,
        )

    # ── Motion-only fallback (separate code path) ─────────────────────
    # Triggered ONLY when audio returns zero candidates.  Audio with
    # candidates that all failed the 4-signal gates does NOT fall back —
    # if audio found events and they all failed the gates, we trust the
    # gates and return nothing (operator can lower thresholds via constants
    # or per-call kwargs and reprocess).
    used_fallback = False
    if not audio_windows and motion_windows:
        log.info(
            "ai_tracer: detect_swings_combined — audio returned 0 candidates; "
            "entering motion-only fallback (%d motion bursts)",
            len(motion_windows),
        )
        fallback_accepted: list[dict] = []
        for mw in motion_windows:
            m_peak_t = float(mw.get("peak_time_sec", 0.0))
            burst_dur = float(mw.get("burst_duration_sec") or 0.0)
            # Approximate burst start/end from burst_duration_sec.
            burst_start_t = m_peak_t - burst_dur / 2.0
            burst_end_t = m_peak_t + burst_dur / 2.0
            # Gate 3 analogue: require meaningful pre- and post-impact span.
            fb_backswing_ok = (m_peak_t - burst_start_t) >= (backswing_window_s * 0.5)
            fb_ft_ok = (burst_end_t - m_peak_t) >= (followthrough_window_s * 0.5)
            fb_pass = fb_backswing_ok and fb_ft_ok
            log.info(
                "ai_tracer: motion-only candidate t=%.2fs burst=%.2fs "
                "backswing_check=%s followthrough_check=%s → %s",
                m_peak_t, burst_dur,
                "OK" if fb_backswing_ok else "FAIL",
                "OK" if fb_ft_ok else "FAIL",
                "ACCEPT" if fb_pass else "REJECT",
            )
            if fb_pass:
                fallback_accepted.append(mw)

        if fallback_accepted:
            accepted_windows = fallback_accepted
            used_fallback = True
        else:
            # All motion bursts failed Gate 3 — return them unfiltered so
            # the operator can diagnose from logs rather than get nothing.
            log.info(
                "ai_tracer: motion-only fallback Gate 3 rejected all %d bursts; "
                "returning unfiltered so operator can diagnose",
                len(motion_windows),
            )
            accepted_windows = list(motion_windows)
            used_fallback = True

    if debug is not None:
        debug["audio"] = audio_debug
        debug["motion"] = motion_debug
        debug["combined"] = {
            "before_impact_sec": float(before_impact_sec),
            "after_impact_sec": float(after_impact_sec),
            "fallback": "motion_only" if used_fallback else None,
            "n_audio_candidates": len(audio_windows),
            "n_motion_windows": len(motion_windows),
            "n_accepted": len(accepted_windows),
            "rejection_counts": dict(rejection_counts),
            "tunables": {
                "audio_min_peak_ratio": float(audio_min_peak_ratio),
                "audio_max_rise_ms": float(audio_max_rise_ms),
                "audio_max_duration_ms": float(audio_max_duration_ms),
                "audio_min_spectral_centroid_hz": float(audio_min_spectral_centroid_hz),
                "backswing_window_s": float(backswing_window_s),
                "followthrough_window_s": float(followthrough_window_s),
                "min_motion_ratio_at_impact": float(min_motion_ratio_at_impact),
                "motion_floor_ratio": float(motion_floor_ratio),
                "pair_window_sec": float(pair_window_sec),
            },
            "candidates": candidate_reports,
        }

    log.info(
        "ai_tracer: detect_swings_combined — accepted %d / %d audio candidates "
        "(motion_bursts=%d fallback=%s)",
        len(accepted_windows), len(audio_windows), len(motion_windows),
        "motion_only" if used_fallback else "none",
    )
    return accepted_windows


def run_full_ai_tracer_pipeline(
    input_path: Path,
    output_dir: Path,
    output_prefix: str,
    model: str | None = None,
    impact_frame_override: int | None = None,
    ball_track_max_frames_override: int | None = None,
    ball_at_rest_override: tuple[float, float] | None = None,
    manual_ball_positions: list[dict] | None = None,
    handedness_override: str | None = None,
    examples_by_kind: dict | None = None,
    rest_anchor_fallback: tuple[float, float] | None = None,
    render_video: bool = True,
    ball_track_enabled: bool = True,
) -> dict:
    """Run the complete AI tracer pipeline (address → handedness →
    impact → refine → ball-track → tracer render) on a single clip.

    Writes intermediate JPEGs and the final tracer MP4 into
    `output_dir`, prefixed with `output_prefix`. Returns a dict with
    every stage's raw output plus paths to the files it wrote and a
    computed `cutover_time_sec` (when the rendered tracer ends in
    clip-time, plus a 1s buffer) for downstream dual-camera composite
    use.

    Optional manual overrides — used by the /admin/clips/ai page so an
    operator can correct the AI's output without re-uploading:

    - `impact_frame_override`: int frame index. Bypasses audio impact
      detection and the AI vision fallback; address frame is derived
      from this (impact − 1.5 s) too.
    - `ball_track_max_frames_override`: int. Overrides the per-fps
      default in track_ball_after_impact (e.g. 12 → 20 to keep
      tracking a long flight).
    - `ball_at_rest_override`: (x, y) in NATIVE pixel coords. Bypasses
      the handedness Claude call; ball_xy_sent + ball_sent_dims get
      set from this directly.
    - `manual_ball_positions`: [{"frame": int, "x": int, "y": int},…]
      in NATIVE pixel coords. Merged into ball_track frames after AI
      tracking — overrides existing entries for matching frame
      numbers and inserts new entries for frames AI missed entirely.
      Each manual entry is flagged `manual: true` in the result.

    Never raises. The router that calls this is responsible for
    transcoding the tracer MP4 to H.264 for browser playback if it
    plans to display it directly, and for constructing public URLs.
    """
    result: dict = {
        "ok": False,
        "error": None,
        "address": None,
        "address_image_path": None,
        "handedness": None,
        "impact": None,
        "impact_refined": None,
        "impact_image_path": None,
        "ball_track": None,
        "ball_track_frames": [],
        "tracer_video_info": None,
        "tracer_video_path": None,
        "cutover_time_sec": None,
        "fps": None,
        "ball_rest_xy_native": None,
        "ball_xy_sent": None,
        "ball_sent_dims": None,
    }

    if not HAS_CV:
        result["error"] = "opencv not installed"
        return result
    # AI is only REQUIRED when a Claude call can actually happen: the
    # ball track (if enabled), impact detection (no override), or the
    # resting-ball fallback (no override). A fully-pinned swing with the
    # track disabled runs the whole pipeline with zero API calls.
    _may_need_api = (
        ball_track_enabled
        or impact_frame_override is None
        or ball_at_rest_override is None
    )
    if _may_need_api and not HAS_ANTHROPIC:
        result["error"] = "anthropic SDK not installed"
        return result
    if _may_need_api and not os.environ.get("ANTHROPIC_API_KEY"):
        result["error"] = "ANTHROPIC_API_KEY not set in environment"
        return result

    output_dir.mkdir(parents=True, exist_ok=True)

    # Probe fps up front — both the audio-first address shortcut below
    # and downstream impact / track passes need it.
    from .video import probe_fps as _probe_fps
    fps_val = _probe_fps(input_path) or 30.0
    result["fps"] = fps_val

    # --- Step 0: audio impact (used both to derive the address frame
    # and to short-circuit AI impact detection downstream) ---
    # Manual impact_frame_override bypasses this entirely.
    if impact_frame_override is not None:
        audio_impact_info = {
            "ok": False,
            "error": "manual override active",
            "method": "manual_override",
            "impact_frame": int(impact_frame_override),
        }
    else:
        audio_impact_info = find_impact_via_audio(input_path, fps_val)

    # --- Step 1: address frame ---
    # When audio impact is confident (ratio >= AUDIO_MIN_PEAK_OVER_MEDIAN
    # = 25 after high-pass), skip the find_address_frame Claude call.
    # A golf swing from address to impact is < ~1.5 s; the golfer
    # holds address for at least a beat. So address ≈ impact − 1.5 s
    # is a safe heuristic that avoids the $0.05/clip API call.
    # Manual override path derives the same way from the supplied
    # impact frame.
    address_image_path = output_dir / f"{output_prefix}_address.jpg"
    address_info: dict
    if impact_frame_override is not None:
        addr_idx = max(0, int(impact_frame_override) - int(round(1.5 * fps_val)))
        if HAS_CV:
            try:
                cap = cv2.VideoCapture(str(input_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, addr_idx)
                ok_read, frame = cap.read()
                cap.release()
                if ok_read and frame is not None:
                    cv2.imwrite(str(address_image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            except Exception as exc:  # pragma: no cover
                log.warning("ai_tracer: manual-override address frame grab failed: %s", exc)
        address_info = {
            "ok": True,
            "error": None,
            "address_frame": addr_idx,
            "confidence": "manual",
            "notes": (
                f"derived from manual impact override frame "
                f"{int(impact_frame_override)} − {int(round(1.5 * fps_val))}f"
            ),
            "model": None,
            "frames_sent": [],
            "saved_image": address_image_path.exists(),
            "method": "manual_derived",
        }
        log.info(
            "ai_tracer: address frame derived from manual impact override — addr=%d (impact=%d)",
            addr_idx, int(impact_frame_override),
        )
    elif audio_impact_info.get("ok") and audio_impact_info.get("impact_frame") is not None:
        addr_idx = max(0, int(audio_impact_info["impact_frame"]) - int(round(1.5 * fps_val)))
        if HAS_CV:
            try:
                cap = cv2.VideoCapture(str(input_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, addr_idx)
                ok_read, frame = cap.read()
                cap.release()
                if ok_read and frame is not None:
                    cv2.imwrite(str(address_image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            except Exception as exc:  # pragma: no cover
                log.warning("ai_tracer: audio-derived address frame grab failed: %s", exc)
        address_info = {
            "ok": True,
            "error": None,
            "address_frame": addr_idx,
            "confidence": "high",
            "notes": (
                f"derived from audio impact frame "
                f"{audio_impact_info['impact_frame']} − {int(round(1.5 * fps_val))}f "
                f"(1.5s @ {fps_val:.1f}fps)"
            ),
            "model": None,
            "frames_sent": [],
            "saved_image": address_image_path.exists(),
            "method": "audio_derived",
        }
        log.info(
            "ai_tracer: address frame derived from audio impact — addr=%d (impact=%d - %.1fs)",
            addr_idx, int(audio_impact_info["impact_frame"]), 1.5,
        )
    else:
        # No override and no confident audio: derive a conservative
        # address anyway (0.5s into the clip — produce cuts put impact
        # ~2s in, so the vision-impact window [addr+1, addr+2s] still
        # covers it). The find_address_frame vision call is retired
        # per operator: address is always impact-1.5s when impact is
        # known, and never worth an Opus call when it isn't.
        addr_idx = max(0, int(round(0.5 * fps_val)))
        if HAS_CV:
            try:
                cap = cv2.VideoCapture(str(input_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, addr_idx)
                ok_read, frame = cap.read()
                cap.release()
                if ok_read and frame is not None:
                    cv2.imwrite(
                        str(address_image_path), frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92],
                    )
            except Exception as exc:  # pragma: no cover
                log.warning(
                    "ai_tracer: heuristic address frame grab failed: %s",
                    exc,
                )
        address_info = {
            "ok": True, "error": None, "address_frame": addr_idx,
            "confidence": "low",
            "notes": "heuristic 0.5s fallback (vision address retired)",
            "model": None, "frames_sent": [],
            "saved_image": address_image_path.exists(),
            "method": "heuristic",
        }
    result["address"] = address_info
    if address_image_path.exists():
        result["address_image_path"] = address_image_path
    if not address_info.get("ok") or address_info.get("address_frame") is None:
        result["error"] = f"address: {address_info.get('error', 'no address frame')}"
        return result
    addr_idx = int(address_info["address_frame"])

    # --- Step 2: handedness + landmarks ---
    # Manual ball_at_rest_override bypasses the Claude handedness call;
    # we synthesize a handedness_info dict with the operator-supplied
    # ball position in native pixel coords and let downstream code
    # treat the native frame size as "what we sent to Claude" — no
    # coord-space conversion needed.
    if ball_at_rest_override is not None:
        nw, nh = 0, 0
        if HAS_CV:
            cap = cv2.VideoCapture(str(input_path))
            try:
                if cap.isOpened():
                    nw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    nh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            finally:
                cap.release()
        handedness_info = {
            "ok": True,
            "error": None,
            "handedness": (
                handedness_override
                if handedness_override in ("right", "left", "unknown")
                else "unknown"
            ),
            "ball_x": int(ball_at_rest_override[0]),
            "ball_y": int(ball_at_rest_override[1]),
            "image_width": nw,
            "image_height": nh,
            "confidence": "manual",
            "method": "manual_override",
            "model": None,
            "notes": "ball-at-rest position supplied by operator",
        }
        log.info(
            "ai_tracer: ball-at-rest overridden — (%d, %d) in %dx%d native",
            int(ball_at_rest_override[0]), int(ball_at_rest_override[1]), nw, nh,
        )
    else:
        # Vision handedness call retired per operator (handedness
        # doesn't matter downstream; the ball position comes from the
        # found-ball override or the resting-ball fallback at step 6).
        handedness_info = {
            "ok": True, "error": None,
            "handedness": (
                handedness_override
                if handedness_override in ("right", "left", "unknown")
                else "unknown"
            ),
            "ball_x": None, "ball_y": None,
            "image_width": None, "image_height": None,
            "confidence": None, "method": "retired",
            "model": None,
            "notes": "vision handedness call retired",
        }
    result["handedness"] = handedness_info
    if handedness_info.get("ok") and handedness_info.get("method") != "manual_override":
        annotate_address_with_shaft(
            input_path, addr_idx, handedness_info, address_image_path,
        )

    # Extract ball-rest position in handedness sent-image coords for
    # downstream impact / track passes.
    ball_xy_sent = None
    ball_sent_dims = None
    if handedness_info.get("ok"):
        bx = handedness_info.get("ball_x")
        by = handedness_info.get("ball_y")
        sw = handedness_info.get("image_width")
        sh = handedness_info.get("image_height")
        if bx is not None and by is not None and sw and sh:
            ball_xy_sent = (float(bx), float(by))
            ball_sent_dims = (int(sw), int(sh))
    result["ball_xy_sent"] = ball_xy_sent
    result["ball_sent_dims"] = ball_sent_dims

    # --- Step 3: impact ---
    impact_info: dict | None = None

    # Step 3-manual: operator supplied an explicit impact frame —
    # bypass audio AND AI vision detection.
    if impact_frame_override is not None:
        impact_info = {
            "ok": True,
            "error": None,
            "impact_frame": int(impact_frame_override),
            "confidence": "manual",
            "notes": f"impact frame {int(impact_frame_override)} set manually by operator",
            "method": "manual",
            "model": None,
            "frames_sent": [],
            "audio": audio_impact_info,
        }

    # Step 3a: audio impact (already computed above; reuse)
    if impact_info is None and audio_impact_info.get("ok"):
        audio_frame = audio_impact_info.get("impact_frame")
        if audio_frame is not None and audio_frame >= addr_idx:
            ratio_str = (
                f" (×{audio_impact_info['ratio']:.1f})"
                if audio_impact_info.get("ratio") is not None else ""
            )
            peak_str = (
                f" at {audio_impact_info['peak_time_sec']:.3f}s"
                if audio_impact_info.get("peak_time_sec") is not None else ""
            )
            offset = audio_impact_info.get("pre_peak_offset_frames") or 0
            offset_str = f" − {offset}f" if offset else ""
            impact_info = {
                "ok": True,
                "error": None,
                "impact_frame": int(audio_frame),
                "confidence": audio_impact_info.get("confidence"),
                "notes": f"audio peak{peak_str}{ratio_str}{offset_str}",
                "method": "audio",
                "model": None,
                "frames_sent": [],
                "audio": audio_impact_info,
            }
        else:
            audio_impact_info["error"] = (
                f"audio peak at frame {audio_frame} precedes address "
                f"frame {addr_idx}"
            )
            audio_impact_info["ok"] = False

    # --- Step 3b: AI vision impact (fallback) ---
    if impact_info is None:
        impact_info = find_impact_frame_after_address(
            input_path, addr_idx,
            ball_xy_sent=ball_xy_sent,
            ball_sent_dims=ball_sent_dims,
            output_image_path=None,
            model=model,
            examples=(examples_by_kind or {}).get("impact"),
        )
        impact_info["method"] = "ai_vision"
        impact_info["audio"] = audio_impact_info
    result["impact"] = impact_info
    if not impact_info.get("ok") or impact_info.get("impact_frame") is None:
        result["error"] = f"impact: {impact_info.get('error', 'no impact')}"
        return result

    # --- Step 4: refined impact + shaft on impact frame ---
    # When step 3 succeeded via audio (ratio >= AUDIO_MIN_PEAK_OVER_MEDIAN
    # = 25 after high-pass), trust the audio frame directly and skip
    # both the refine_impact_frame Claude call AND the AI-vision
    # fallback. Audio gives sub-frame impact timing once the high-pass
    # filter is in; the refine call's hands / clubhead landmarks
    # aren't used downstream by ball tracking, so the call was pure
    # cost. We still produce an impact-frame JPG via cv2 so the
    # frontend can show what we picked.
    impact_image_path = output_dir / f"{output_prefix}_impact.jpg"
    if impact_info.get("method") in ("audio", "manual"):
        if HAS_CV:
            try:
                cap = cv2.VideoCapture(str(input_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(impact_info["impact_frame"]))
                ok_read, frame = cap.read()
                cap.release()
                if ok_read and frame is not None:
                    cv2.imwrite(str(impact_image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            except Exception as exc:  # pragma: no cover
                log.warning("ai_tracer: audio-impact frame grab failed: %s", exc)
        refined_impact_info = {
            "ok": True,
            "error": None,
            "impact_frame": int(impact_info["impact_frame"]),
            "hands_x": None, "hands_y": None,
            "clubhead_x": None, "clubhead_y": None,
            "confidence": impact_info.get("confidence"),
            "notes": "skipped refine — audio impact trusted",
            "method": "audio_trusted",
            "model": None,
            "frames_sent": [],
        }
    else:
        refined_impact_info = refine_impact_frame(
            input_path, int(impact_info["impact_frame"]),
            ball_xy_sent=ball_xy_sent,
            ball_sent_dims=ball_sent_dims,
            output_image_path=impact_image_path,
            model=model,
        )
    result["impact_refined"] = refined_impact_info
    if impact_image_path.exists():
        result["impact_image_path"] = impact_image_path
    if not refined_impact_info.get("ok") or refined_impact_info.get("impact_frame") is None:
        result["error"] = (
            f"refined_impact: {refined_impact_info.get('error', 'no refined impact')}"
        )
        return result

    # --- Step 5: per-frame ball tracking ---
    track_prefix = f"{output_prefix}_track"
    if not ball_track_enabled:
        # AI ball track disabled (operator: the launch tracker +
        # rest-lock supply the flight now). NOT deleted — flip
        # settings.ai_ball_track_enabled / AI_BALL_TRACK_ENABLED to
        # bring it back.
        ball_track_info = {
            "ok": True, "error": None, "frames": [],
            "n_frames_found": 0, "n_frames_processed": 0,
            "skipped": "ai ball track disabled",
        }
        log.info("ai_tracer: ball track SKIPPED (disabled by config)")
    else:
        ball_track_info = track_ball_after_impact(
            input_path,
            int(refined_impact_info["impact_frame"]),
            output_dir=output_dir,
            output_prefix=track_prefix,
            ball_xy_sent=ball_xy_sent,
            ball_sent_dims=ball_sent_dims,
            max_frames=int(ball_track_max_frames_override)
                if ball_track_max_frames_override is not None else None,
            model=model,
        )

    # Step 5b: merge operator-supplied manual ball positions into the
    # AI-tracked frames. Existing entries with the same frame number
    # get overridden (ball coords + found=True + confidence=manual);
    # missing frames get a brand-new entry inserted in frame order.
    if manual_ball_positions:
        frames_list = ball_track_info.get("frames") or []
        by_frame = {
            int(rec["frame"]): rec
            for rec in frames_list if rec.get("frame") is not None
        }
        n_override = 0
        n_inserted = 0
        for manual in manual_ball_positions:
            try:
                fnum = int(manual.get("frame"))
                mx = int(manual.get("x"))
                my = int(manual.get("y"))
            except (TypeError, ValueError, KeyError):
                continue
            if fnum in by_frame:
                rec = by_frame[fnum]
                prior_note = rec.get("notes") or ""
                rec["x"] = mx
                rec["y"] = my
                rec["found"] = True
                rec["confidence"] = "manual"
                rec["manual"] = True
                rec["notes"] = (
                    f"manual override (was: {prior_note})"
                    if prior_note else "manual override"
                )
                n_override += 1
            else:
                frames_list.append({
                    "frame": fnum,
                    "found": True,
                    "x": mx,
                    "y": my,
                    "confidence": "manual",
                    "notes": "manually added by operator",
                    "manual": True,
                    "retry": False,
                    "image_filename": None,
                })
                n_inserted += 1
        frames_list.sort(key=lambda r: int(r.get("frame") or 0))
        ball_track_info["frames"] = frames_list
        ball_track_info["n_manual_override"] = n_override
        ball_track_info["n_manual_inserted"] = n_inserted
        log.info(
            "ai_tracer: merged %d manual ball position(s) — %d override, %d inserted",
            n_override + n_inserted, n_override, n_inserted,
        )

    result["ball_track"] = ball_track_info
    for rec in (ball_track_info.get("frames") or []):
        result["ball_track_frames"].append({
            "frame": rec.get("frame"),
            "found": rec.get("found"),
            "x": rec.get("x"),
            "y": rec.get("y"),
            "confidence": rec.get("confidence"),
            "notes": rec.get("notes"),
            "retry": rec.get("retry", False),
            "manual": rec.get("manual", False),
            "image_filename": rec.get("image_filename"),
        })

    # --- Step 6: render tracer overlay video ---
    ball_rest_xy_native: tuple[float, float] | None = None
    if ball_xy_sent and ball_sent_dims:
        cap = cv2.VideoCapture(str(input_path))
        try:
            if cap.isOpened():
                nw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                nh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if nw > 0 and nh > 0:
                    ball_rest_xy_native = (
                        ball_xy_sent[0] * nw / ball_sent_dims[0],
                        ball_xy_sent[1] * nh / ball_sent_dims[1],
                    )
        finally:
            cap.release()
    # The tracer must ALWAYS start from where the ball is struck. When the
    # handedness call didn't hand back a ball position, fall back to a
    # dedicated resting-ball vision call on the address frame so we still
    # get a ground anchor instead of the line starting mid-flight at the
    # first tracked point. (Uses the operator override when one was given.)
    if ball_rest_xy_native is None and ball_at_rest_override is None:
        rb = find_resting_ball(input_path, int(addr_idx), model=model)
        if rb.get("present") and rb.get("x") is not None and rb.get("y") is not None:
            ball_rest_xy_native = (float(rb["x"]), float(rb["y"]))
            result["ball_rest_source"] = "find_resting_ball_fallback"
            log.info(
                "ai_tracer: rest anchor via fallback find_resting_ball at "
                "address frame %s → (%s,%s)",
                addr_idx, rb["x"], rb["y"],
            )
        elif rest_anchor_fallback is not None:
            # Vision couldn't see the ball (backlit / dark ground). Anchor
            # the line at the caller-supplied point — the golfer's hands at
            # impact from the pose detector — so it starts at the strike
            # instead of picking up mid-flight.
            ball_rest_xy_native = (
                float(rest_anchor_fallback[0]), float(rest_anchor_fallback[1]),
            )
            result["ball_rest_source"] = "pose_wrist_fallback"
            log.info(
                "ai_tracer: rest anchor via pose-hands fallback → (%.0f,%.0f)",
                rest_anchor_fallback[0], rest_anchor_fallback[1],
            )
    result["ball_rest_xy_native"] = ball_rest_xy_native

    if not render_video:
        # Caller renders its own video from the returned track (e.g. the
        # wizard's ai_mog2 engine, which re-renders a windowed clip with
        # MOG2-extended points). Skipping the full-length render here
        # saves a whole read+write+transcode pass on long sources.
        result["ok"] = True
        log.info(
            "ai_tracer: pipeline complete (render skipped) for %s — "
            "addr=%s impact=%s tracked=%d/%d",
            input_path.name, addr_idx,
            refined_impact_info.get("impact_frame"),
            (ball_track_info or {}).get("n_frames_found", 0),
            (ball_track_info or {}).get("n_frames_processed", 0),
        )
        return result

    tracer_path = output_dir / f"{output_prefix}_ai_tracer.mp4"
    tracer_info = render_tracer_video(
        input_path, tracer_path,
        ball_rest_xy_native=ball_rest_xy_native,
        impact_frame_idx=int(refined_impact_info["impact_frame"]),
        track_frames=ball_track_info.get("frames") or [],
    )
    result["tracer_video_info"] = tracer_info
    if tracer_info.get("ok"):
        # cv2.VideoWriter strips audio. Mux the original input clip's
        # audio track back in so the deliverable plays with sound.
        from .video import mux_audio_into_video as _mux_audio
        try:
            audio_ok = _mux_audio(tracer_path, input_path)
            if not audio_ok:
                log.info(
                    "ai_tracer: pipeline — audio mux skipped/failed for %s "
                    "(source may have no audio)", tracer_path.name,
                )
        except Exception as exc:
            log.warning("ai_tracer: audio mux raised: %s", exc)
        result["tracer_video_path"] = tracer_path
        # Cutover time for dual-camera composite: when the rendered
        # smoothed line ends (last sampled frame in source-clip
        # frame-index space), plus a 1-second pause so the viewer
        # registers the tracer apex before the cut to the green
        # camera. frame_range is in clip-relative frame indices so
        # we just divide by fps.
        rng = tracer_info.get("frame_range")
        if rng and fps_val > 0:
            result["cutover_time_sec"] = float(rng[1]) / fps_val + 1.0

    result["ok"] = True
    log.info(
        "ai_tracer: pipeline complete for %s — addr=%s impact=%s "
        "tracked=%d/%d cutover=%.2fs",
        input_path.name, addr_idx,
        refined_impact_info.get("impact_frame"),
        (ball_track_info or {}).get("n_frames_found", 0),
        (ball_track_info or {}).get("n_frames_processed", 0),
        result["cutover_time_sec"] or 0.0,
    )
    return result
