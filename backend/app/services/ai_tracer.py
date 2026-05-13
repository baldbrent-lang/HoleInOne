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
# 2000 is above Anthropic's "auto-resize" threshold so we pay slightly
# more in vision tokens, but the ball footprint in pixel area grows
# proportionally — matters for the 3-15 px ball that's the whole
# point of this pipeline.
BALL_TRACK_FRAME_W = 2000

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
#   >100    : 40 frames sampled with STRIDE = 2 (every other frame,
#             80-frame span). Cost stays at 40 Claude calls but
#             the time window doubles to match what the lower-fps
#             buckets cover in absolute time.
BALL_TRACK_MAX_FRAMES = 20
BALL_TRACK_MAX_FRAMES_HIGH_FPS = 40
BALL_TRACK_HIGH_FPS_THRESHOLD = 50.0
BALL_TRACK_VHIGH_FPS_THRESHOLD = 100.0
BALL_TRACK_VHIGH_FPS_STRIDE = 2
BALL_TRACK_CONCURRENCY = 8

# Phase-2 retry sends a crop. Crop size is in NATIVE pixels (how much
# of the source frame around the prediction we hand to Claude); send
# size is the upscaled width we encode to JPEG before sending. The
# upscale turns a ~600 px region into a ~1024 px image — same content,
# more vision tiles, ball appears with ~1.7× more pixels per side.
BALL_TRACK_CROP_NATIVE_SIZE = 600
BALL_TRACK_CROP_SEND_W = 1568


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


def detect_handedness_at_address(
    input_path: Path, address_frame_idx: int,
    frame_w: int = HANDEDNESS_FRAME_W,
    model: str | None = None,
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
    model = _resolve_model(model)
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
    model = _resolve_model(model)
    info: dict = {
        "ok": False,
        "error": None,
        "address_frame": None,
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

    content: list[dict] = [{
        "type": "text",
        "text": (
            f"Below are {len(frames)} frames from a {total_frames}-frame "
            f"clip at {fps:.1f} fps, in chronological order. Each image "
            "is preceded by its frame number."
        ),
    }]
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
    model = _resolve_model(model)
    info: dict = {
        "ok": False,
        "error": None,
        "impact_frame": None,
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

    content: list[dict] = [
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
    ]
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
    model = _resolve_model(model)
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
    # regardless of source fps.
    stride = 1
    if max_frames is None:
        if clip_fps > BALL_TRACK_VHIGH_FPS_THRESHOLD:
            # Slow-mo (>100 fps): same 40-call budget, but every other
            # frame so we span 2× the time window.
            max_frames = BALL_TRACK_MAX_FRAMES_HIGH_FPS
            stride = BALL_TRACK_VHIGH_FPS_STRIDE
        elif clip_fps >= BALL_TRACK_HIGH_FPS_THRESHOLD:
            max_frames = BALL_TRACK_MAX_FRAMES_HIGH_FPS
        else:
            max_frames = BALL_TRACK_MAX_FRAMES
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
    return info


# Visual style for the final tracer-overlay render. Mirrors the
# classical tracer in services/tracer.py so the two outputs look at
# home next to each other.
TRACER_LINE_COLOR = (0, 140, 255)        # bright orange (BGR)
TRACER_LINE_HALO = (40, 90, 200)         # darker orange halo behind
TRACER_LINE_THICKNESS = 5
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
# How many sub-frame samples the fade-away tail is split into. After
# the last accepted anchor the tracer doesn't continue extrapolating;
# instead it shows a brief comet-style tail that spans roughly one
# frame's worth of ball motion (sampled along the same parabola the
# main line was drawn through) and fades to zero opacity along its
# length. Length 1.0 = exactly 1 frame of motion at the local
# velocity; smaller values make the fade shorter.
TRACER_FADE_FRAME_LENGTH = 1.0
TRACER_FADE_SEGMENTS = 10


def _robust_quadratic_fit(
    anchors: list[tuple[int, int, int]],
    threshold_px: float = TRAJ_OUTLIER_PX,
    max_iters: int = TRAJ_OUTLIER_MAX_ITERS,
):
    """Fit y = a·f² + b·f + c and x = m·f + k to `anchors`
    (list of (frame, x, y)), iteratively dropping the anchor with the
    largest residual until every kept residual is ≤ threshold_px or
    fewer than 3 anchors remain.

    Returns (x_coef, y_coef, rejected_indices_set) on success, or
    None when fewer than 3 anchors are usable or numpy is missing.
    """
    if not HAS_NP or len(anchors) < 3:
        return None
    rejected: set[int] = set()
    last_coefs = None
    for _ in range(max_iters):
        kept_idxs = [i for i in range(len(anchors)) if i not in rejected]
        if len(kept_idxs) < 3:
            break
        frames = np.array([anchors[i][0] for i in kept_idxs], dtype=float)
        xs = np.array([anchors[i][1] for i in kept_idxs], dtype=float)
        ys = np.array([anchors[i][2] for i in kept_idxs], dtype=float)
        try:
            y_coef = np.polyfit(frames, ys, 2)
            x_coef = np.polyfit(frames, xs, 1)
        except Exception:
            return None
        last_coefs = (x_coef, y_coef)
        x_pred = np.polyval(x_coef, frames)
        y_pred = np.polyval(y_coef, frames)
        residuals = np.sqrt((xs - x_pred) ** 2 + (ys - y_pred) ** 2)
        worst_local = int(np.argmax(residuals))
        worst_residual = float(residuals[worst_local])
        if worst_residual > threshold_px and len(kept_idxs) > 3:
            rejected.add(kept_idxs[worst_local])
            continue
        return x_coef, y_coef, rejected
    # Loop fell out without converging — return the last successful fit
    # if it exists and we still have ≥3 anchors.
    kept_after = [i for i in range(len(anchors)) if i not in rejected]
    if last_coefs is None or len(kept_after) < 3:
        return None
    return last_coefs[0], last_coefs[1], rejected


def _draw_dashed_tracer(img, points: list[tuple[int, int]]) -> None:
    """Draw a dashed polyline through `points` with a halo behind it.
    No-op when fewer than 2 points are provided. Style matches the
    classical tracer so the visual language is consistent."""
    if len(points) < 2:
        return
    if HAS_NP:
        cv2.polylines(
            img, [np.array(points, dtype=np.int32)],
            False, TRACER_LINE_HALO,
            TRACER_LINE_THICKNESS + 4, cv2.LINE_AA,
        )
    accumulated = 0.0
    drawing = True
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        dx = float(x1 - x0)
        dy = float(y1 - y0)
        seg_len = (dx * dx + dy * dy) ** 0.5
        if seg_len == 0:
            continue
        ux = dx / seg_len
        uy = dy / seg_len
        traveled = 0.0
        cur_x = float(x0)
        cur_y = float(y0)
        while traveled < seg_len:
            target_seg = TRACER_DASH_LEN if drawing else TRACER_GAP_LEN
            remaining = target_seg - accumulated
            step = min(remaining, seg_len - traveled)
            nx = cur_x + ux * step
            ny = cur_y + uy * step
            if drawing:
                cv2.line(
                    img,
                    (int(cur_x), int(cur_y)), (int(nx), int(ny)),
                    TRACER_LINE_COLOR, TRACER_LINE_THICKNESS, cv2.LINE_AA,
                )
            cur_x = nx
            cur_y = ny
            traveled += step
            accumulated += step
            if accumulated >= target_seg:
                drawing = not drawing
                accumulated = 0.0


def render_tracer_video(
    input_path: Path,
    output_path: Path,
    ball_rest_xy_native: tuple[float, float] | None,
    impact_frame_idx: int,
    track_frames: list[dict],
) -> dict:
    """Render an MP4 of the source video with a progressive dashed
    tracer line overlaid.

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

    # Build {frame_idx: (x, y)} for every successfully-located ball.
    points_by_frame: dict[int, tuple[int, int]] = {}
    for rec in track_frames or []:
        if not rec.get("found"):
            continue
        x = rec.get("x")
        y = rec.get("y")
        f = rec.get("frame")
        if f is None or x is None or y is None:
            continue
        try:
            points_by_frame[int(f)] = (int(x), int(y))
        except (TypeError, ValueError):
            continue

    # The full ordered list of tracer anchors: rest position first,
    # then every found-ball position in chronological order.
    anchors: list[tuple[int, int, int]] = []  # (frame, x, y)
    if ball_rest_xy_native is not None:
        anchors.append((
            int(impact_frame_idx),
            int(round(float(ball_rest_xy_native[0]))),
            int(round(float(ball_rest_xy_native[1]))),
        ))
    for f in sorted(points_by_frame):
        x, y = points_by_frame[f]
        anchors.append((f, x, y))

    # Fit a smooth parabola through the anchors, with iterative
    # outlier rejection to throw out frames where Claude latched onto
    # something that isn't the ball. The rendered tracer is sampled
    # from this fit at every frame — not from the raw anchors — so
    # the line is a smooth arc instead of a kinked polyline, AND a
    # single bad detection won't yank it at right angles.
    smoothed_points: list[tuple[int, int, int]] = []  # (frame, x, y)
    fade_tail_points: list[tuple[int, int]] = []  # native pixel coords
    last_kept_frame_global: int | None = None
    rejected_frames: set[int] = set()
    fit = _robust_quadratic_fit(anchors)
    if fit is not None:
        x_coef, y_coef, rejected_indices = fit
        rejected_frames = {anchors[i][0] for i in rejected_indices}
        kept = [a for i, a in enumerate(anchors) if i not in rejected_indices]
        if kept:
            first_frame = kept[0][0]
            last_kept_frame = kept[-1][0]
            # Truncate the rendered line at the parabola's apex when
            # apex falls within the kept-anchor range. In image coords
            # y grows downward, so the ball's HIGHEST visual point is
            # the parabola's MINIMUM (positive a, vertex at -b/2a).
            # Past that minimum the smoothed line would curve back
            # down — bad visual when the ball was clearly still
            # rising on the last detected frames. Stopping at apex
            # produces a clean upward arc that ends at its peak.
            a_y = float(y_coef[0])
            b_y = float(y_coef[1])
            render_end = last_kept_frame
            if a_y > 1e-6:
                apex_frame_f = -b_y / (2.0 * a_y)
                if first_frame < apex_frame_f < last_kept_frame:
                    render_end = int(round(apex_frame_f))
            last_kept_frame_global = render_end
            for f in range(first_frame, render_end + 1):
                x = int(round(float(np.polyval(x_coef, f))))
                y = int(round(float(np.polyval(y_coef, f))))
                if x < 0 or x >= width or y < 0 or y >= height:
                    break
                smoothed_points.append((f, x, y))
            # Fade-tail: extend along the TANGENT of the parabola at
            # `render_end` (which equals apex when truncated, otherwise
            # last_kept_frame) for one frame's worth of motion. The
            # tangent at apex has vy = 0 so the fade goes horizontally
            # — never loops down. On non-truncated lines it picks up
            # the local velocity and continues straight from there.
            base_x = float(np.polyval(x_coef, render_end))
            base_y = float(np.polyval(y_coef, render_end))
            vx = float(np.polyval(np.polyder(x_coef), render_end))
            vy = float(np.polyval(np.polyder(y_coef), render_end))
            for i in range(1, TRACER_FADE_SEGMENTS + 1):
                t = i * TRACER_FADE_FRAME_LENGTH / TRACER_FADE_SEGMENTS
                fx = int(round(base_x + vx * t))
                fy = int(round(base_y + vy * t))
                if not (0 <= fx < width and 0 <= fy < height):
                    break
                fade_tail_points.append((fx, fy))
        log.info(
            "ai_tracer: tracer fit — %d anchors, %d rejected as outliers, "
            "%d smoothed render points, %d fade-tail segments",
            len(anchors), len(rejected_indices), len(smoothed_points),
            len(fade_tail_points),
        )
    else:
        # Not enough anchors for a stable fit (or numpy missing).
        # Fall back to the raw point-to-point line.
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
            # Draw the tracer once we've reached the impact frame.
            if smoothed_points and frame_idx >= smoothed_points[0][0]:
                visible = [
                    (x, y) for f, x, y in smoothed_points if f <= frame_idx
                ]
                if len(visible) >= 2:
                    _draw_dashed_tracer(frame, visible)
                # Fade-away tail. Renders only once the main line has
                # reached its end (i.e. we're at or past the last
                # accepted anchor frame). Each sub-segment is
                # alpha-blended with linearly decreasing weight so the
                # line visibly trails off along the parabola tangent.
                if (
                    fade_tail_points
                    and last_kept_frame_global is not None
                    and frame_idx >= last_kept_frame_global
                    and visible
                ):
                    tail_anchor = visible[-1]
                    walk = [tail_anchor, *fade_tail_points]
                    for j in range(len(walk) - 1):
                        alpha = 1.0 - (j / float(len(walk) - 1))
                        if alpha <= 0.03:
                            break
                        overlay = frame.copy()
                        cv2.line(
                            overlay, walk[j], walk[j + 1],
                            TRACER_LINE_COLOR, TRACER_LINE_THICKNESS,
                            cv2.LINE_AA,
                        )
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                if rest_xy is not None:
                    cv2.circle(frame, rest_xy, 12, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.circle(frame, rest_xy, 10, TRACER_REST_RING, 3, cv2.LINE_AA)
            # Highlight the ball on frames where the tracker has a
            # fresh position — but skip outliers (where Claude
            # mis-identified the ball), since drawing a ring at a
            # rejected position would visually contradict the
            # smoothed line.
            if frame_idx in points_by_frame and frame_idx not in rejected_frames:
                x, y = points_by_frame[frame_idx]
                cv2.circle(frame, (x, y), 18, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 16, TRACER_BALL_RING, 3, cv2.LINE_AA)
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
AUDIO_MIN_PEAK_OVER_MEDIAN = 4.0


def find_impact_via_audio(input_path: Path, fps: float) -> dict:
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
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-v", "error",
                "-i", str(input_path),
                "-vn", "-ac", "1",
                "-ar", str(AUDIO_SAMPLE_RATE),
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
    if ratio < AUDIO_MIN_PEAK_OVER_MEDIAN:
        info["error"] = (
            f"audio peak/median ratio too low ({ratio:.1f} < "
            f"{AUDIO_MIN_PEAK_OVER_MEDIAN}); audio likely noisy or "
            "lacks a clear impact"
        )
        return info

    peak_time = peak_idx / float(AUDIO_SAMPLE_RATE)
    info["peak_time_sec"] = float(peak_time)
    info["impact_frame"] = int(round(peak_time * fps))
    # Confidence buckets — these are heuristic but visible in the UI
    # so the operator can sanity-check the audio call.
    if ratio >= 15:
        info["confidence"] = "high"
    elif ratio >= 8:
        info["confidence"] = "medium"
    else:
        info["confidence"] = "low"

    info["ok"] = True
    log.info(
        "ai_tracer: audio impact — frame=%d t=%.3fs peak=%.3f median=%.3f "
        "ratio=%.1f confidence=%s",
        info["impact_frame"], peak_time, peak_value, median, ratio,
        info["confidence"],
    )
    return info
