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
BALL_TRACK_FRAME_W = 1568
BALL_TRACK_MAX_FRAMES = 20
BALL_TRACK_CONCURRENCY = 8


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
        "model": MODEL,
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

    client = Anthropic()
    try:
        resp = client.messages.create(
            model=MODEL,
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
    info: dict = {
        "ok": False,
        "error": None,
        "address_frame": None,
        "confidence": None,
        "notes": None,
        "model": MODEL,
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
        len(frames), candidate_frames, total_frames, fps, MODEL,
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

    client = Anthropic()
    try:
        resp = client.messages.create(
            model=MODEL,
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
    info: dict = {
        "ok": False,
        "error": None,
        "impact_frame": None,
        "confidence": None,
        "notes": None,
        "model": MODEL,
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

    client = Anthropic()
    try:
        resp = client.messages.create(
            model=MODEL,
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
        "model": MODEL,
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

    client = Anthropic()
    try:
        resp = client.messages.create(
            model=MODEL,
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
    max_frames: int = BALL_TRACK_MAX_FRAMES,
    send_width: int = BALL_TRACK_FRAME_W,
    concurrency: int = BALL_TRACK_CONCURRENCY,
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
    info: dict = {
        "ok": False,
        "error": None,
        "frames": [],
        "n_frames_processed": 0,
        "n_frames_found": 0,
        "n_frames_found_via_retry": 0,
        "first_lost_run_start": None,
        "model": MODEL,
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
    finally:
        cap.release()
    if total_frames <= 1:
        info["error"] = "video has no frames"
        return info

    impact_frame_idx = int(impact_frame_idx)
    last_frame = min(total_frames - 1, impact_frame_idx + max_frames - 1)
    frame_indices = list(range(impact_frame_idx, last_frame + 1))
    if not frame_indices:
        info["error"] = "no frames to process"
        return info
    info["n_frames_processed"] = len(frame_indices)
    log.info(
        "ai_tracer: ball_track — tracking %d frames [%d..%d] from impact",
        len(frame_indices), frame_indices[0], frame_indices[-1],
    )

    # Extract all frames once up front: keep both a resized JPEG (for
    # the API) and the native ndarray (for drawing the highlight when
    # the ball is found, without a second cv2.VideoCapture pass).
    frames_data: dict[int, tuple[bytes, int, int, "np.ndarray"]] = {}
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
            if native_w > send_width:
                scale = send_width / float(native_w)
                resized = cv2.resize(
                    frame, (send_width, int(round(native_h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                resized = frame
            ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
            if ok:
                # Store sent dims (for landmark scaling) + native frame
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

    client = Anthropic()

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
                model=MODEL,
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

    # --- Phase 2: retry missed frames using the nearest Phase-1 hit ---
    retry_results: dict[int, dict] = {}
    retry_targets = [
        idx for idx in frames_data
        if idx not in found_sent
    ]
    if found_sent and retry_targets:
        def _retry(idx: int) -> tuple[int, dict]:
            # Choose the closest found frame as the anchor. The ball
            # moves a small amount per frame so even a 3-4 frame gap is
            # usually a useful prior.
            nearest = min(found_sent, key=lambda f: abs(f - idx))
            anchor_sent_x, anchor_sent_y, anchor_sw, anchor_sh = found_sent[nearest]
            _jb, sw, sh, _nf = frames_data[idx]
            # Scale the anchor pixel coords to this frame's sent dims.
            hint_x = int(round(anchor_sent_x * sw / float(anchor_sw)))
            hint_y = int(round(anchor_sent_y * sh / float(anchor_sh)))
            hint = (
                f" HINT: in nearby frame {nearest} the ball was at "
                f"approximately ({hint_x}, {hint_y}). The ball typically "
                "moves less than ~50 px between adjacent frames; look in "
                "that area first, then check the surrounding region. If "
                "you genuinely cannot see the ball, return found=false."
            )
            return _query(idx, hint)

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(_retry, idx) for idx in retry_targets]
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
                    _jb, sw, sh, native_frame = frames_data[idx]
                    nh, nw = native_frame.shape[:2]
                    native_x = int(round(sent_x * nw / float(sw)))
                    native_y = int(round(sent_y * nh / float(sh)))
                    record["x"] = native_x
                    record["y"] = native_y
                    record["retry"] = via_retry

        # Always write a JPEG for this frame so the operator can see
        # what Claude was actually looking at, even when the ball was
        # not found. Annotated when found, plain otherwise.
        _jb, sw, sh, native_frame = frames_data[idx]
        annotated = native_frame.copy()
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
    info["ok"] = True
    log.info(
        "ai_tracer: ball_track — found ball in %d/%d frames; first_lost_run_start=%s",
        n_found, len(frame_indices), first_lost_run_start,
    )
    return info
