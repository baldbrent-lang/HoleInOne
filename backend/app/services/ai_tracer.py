"""AI-powered golf ball tracker using Claude Vision.

Classical CV (motion + brightness + parabola scoring) struggles on busy
backgrounds, bright sky, motion blur, and the wide variety of ball-vs-
non-ball confusables a tee-cam sees (shoes, divots, foliage, range
balls). Claude's general vision understanding handles "is this the
golf ball?" far more reliably than any hand-tuned heuristic.

Pipeline:
    1. Sample ~8 frames evenly across the clip.
    2. Send each to Claude with a strict prompt and a JSON output spec.
    3. Parse the (x, y) for frames where Claude reports `found: true`
       with sufficient confidence.
    4. Fit a parabolic trajectory through those anchor points (y is
       quadratic in frame index, x linear).
    5. Emit a dense per-frame _Det track from impact through apex,
       matching the existing patch-tracker output shape.

Falls back silently (returns empty track + reason) if anthropic SDK
is missing, ANTHROPIC_API_KEY isn't set, or any API call fails — the
caller chains into the legacy classical CV pipeline.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("golfreelz.ai_tracer")

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    HAS_CV = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    np = None  # type: ignore
    HAS_CV = False

try:
    from anthropic import Anthropic  # type: ignore
    HAS_ANTHROPIC = True
except Exception:  # pragma: no cover
    Anthropic = None  # type: ignore
    HAS_ANTHROPIC = False


# Mirrors backend/app/services/tracer.py:_Det so the AI tracker can be
# dropped in alongside the classical trackers without an adapter layer.
@dataclass
class _Det:
    frame: int
    x: float
    y: float
    radius: float


# Per the claude-api skill default. Override via env if a cheaper or
# different vision model is preferred for cost (e.g. claude-haiku-4-5).
MODEL = os.environ.get("TRACER_AI_MODEL", "claude-opus-4-7")

# Number of anchor frames sent to Claude. 8 covers most par-3 flight
# arcs at typical 60fps clips (~3-5s flight = 180-300 frames) while
# keeping per-clip cost reasonable.
N_ANCHOR_FRAMES = 8

# Frames are downscaled to this max width before being sent so token
# count stays bounded regardless of source resolution (4K GoPros etc.).
MAX_IMAGE_WIDTH = 1280

# Confidence levels Claude can return; only "high" / "medium" anchors
# feed the parabola fit, "low" are discarded as too risky.
ACCEPTED_CONFIDENCE = {"high", "medium"}

# Need at least 3 anchor points for a quadratic fit. Below this we
# bail and let the caller fall back to classical CV.
MIN_ANCHORS_FOR_FIT = 3

# Concurrent API calls — small parallelism keeps per-clip latency
# under ~10s while staying well under rate limits.
MAX_CONCURRENT_REQUESTS = 4


SYSTEM_PROMPT = (
    "You are a sports-vision assistant identifying the precise pixel "
    "location of a golf ball in flight in still frames from a par-3 tee "
    "shot. The camera sits behind a right- or left-handed golfer; the "
    "ball travels away from the camera, rising and arcing toward the green.\n\n"
    "For each frame, locate the ball:\n"
    "- It appears as a small bright white dot or short motion-blur streak. "
    "Typical size 3-15 pixels wide.\n"
    "- Against grass: white-on-green. Against sky: a small bright speck "
    "OR (overcast) a darker silhouette — still small and round/oblong.\n"
    "- DO NOT confuse the ball with: the golfer's shoes, belt buckle, "
    "hands, club head, range balls on the tee mat, divots, leaves, "
    "clouds, course markers, or background props.\n"
    "- The ball at address (sitting on the tee) is NOT in flight — set "
    "found=false for those frames.\n"
    "- If you cannot clearly see a single airborne ball, set found=false.\n\n"
    "Reply with ONE JSON object and nothing else, matching this schema:\n"
    "{\n"
    '  "found": true|false,\n'
    '  "x": <int pixel x, 0 = left, or null if not found>,\n'
    '  "y": <int pixel y, 0 = top,  or null if not found>,\n'
    '  "confidence": "high"|"medium"|"low",\n'
    '  "notes": "<short reasoning, max 12 words>"\n'
    "}\n"
    "Coordinates are in the IMAGE coordinate system provided (the image "
    "may be resized before you see it). Aim for the center of the ball."
)


def have_ai_tracer() -> bool:
    """True when the AI tracker can actually run end-to-end.

    Requires OpenCV (for frame extraction), the anthropic SDK, and
    ANTHROPIC_API_KEY in the environment. Any miss = silent fallback.
    """
    return HAS_CV and HAS_ANTHROPIC and bool(os.environ.get("ANTHROPIC_API_KEY"))


def _sample_frame_indices(total_frames: int, n_samples: int) -> list[int]:
    """Pick `n_samples` evenly-spaced indices across the clip,
    avoiding the very first / last few frames where the ball is
    typically at rest or off-screen."""
    if total_frames <= 1:
        return [0]
    pad = max(1, total_frames // 20)
    lo = pad
    hi = max(pad + 1, total_frames - pad - 1)
    if hi <= lo:
        lo, hi = 0, total_frames - 1
    if n_samples <= 1:
        return [lo]
    step = (hi - lo) / float(n_samples - 1)
    out: list[int] = []
    for i in range(n_samples):
        idx = int(round(lo + i * step))
        idx = max(0, min(total_frames - 1, idx))
        if idx not in out:
            out.append(idx)
    return out


def _grab_frame_jpeg(
    input_path: Path, frame_idx: int, max_width: int = MAX_IMAGE_WIDTH,
) -> tuple[bytes, int, int, int, int] | None:
    """Seek to frame_idx and return (jpeg_bytes, native_w, native_h,
    sent_w, sent_h). Opens its own VideoCapture so callers can run
    several of these concurrently across threads."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        native_h, native_w = frame.shape[:2]
        if native_w > max_width:
            scale = max_width / float(native_w)
            new_w = int(round(native_w * scale))
            new_h = int(round(native_h * scale))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        sent_h, sent_w = frame.shape[:2]
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return None
        return bytes(buf), native_w, native_h, sent_w, sent_h
    finally:
        cap.release()


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Pull the first {...} block out of Claude's reply and parse it.
    Tolerates prose around the JSON (we ask for JSON-only but the model
    occasionally adds a sentence)."""
    if not text:
        return None
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _ask_claude_for_ball(
    client, jpeg_bytes: bytes, sent_w: int, sent_h: int, frame_idx: int,
) -> dict | None:
    """One Claude call for one frame. Returns the parsed dict or None
    on any failure (caught and logged; we never raise into the caller)."""
    b64 = base64.standard_b64encode(jpeg_bytes).decode("ascii")
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=256,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
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
                    {
                        "type": "text",
                        "text": (
                            f"Frame {frame_idx}. Image size: {sent_w}x{sent_h} px. "
                            "Locate the golf ball in flight (JSON only)."
                        ),
                    },
                ],
            }],
        )
    except Exception as exc:
        log.warning("ai_tracer: Claude API failed on frame %d: %s", frame_idx, exc)
        return None
    text_chunks = [
        block.text for block in resp.content
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    if not text_chunks:
        log.warning("ai_tracer: empty response on frame %d", frame_idx)
        return None
    return _extract_json("\n".join(text_chunks))


def _query_anchor(
    input_path: Path, client, frame_idx: int,
) -> tuple[int, dict | None, int, int, int, int] | None:
    """Worker run by the thread pool: extract one frame, ask Claude.
    Returns (frame_idx, parsed_dict, native_w, native_h, sent_w, sent_h)
    so the caller can scale coordinates back to native resolution."""
    grab = _grab_frame_jpeg(input_path, frame_idx)
    if grab is None:
        log.warning("ai_tracer: could not grab frame %d", frame_idx)
        return None
    jpeg_bytes, native_w, native_h, sent_w, sent_h = grab
    parsed = _ask_claude_for_ball(client, jpeg_bytes, sent_w, sent_h, frame_idx)
    return frame_idx, parsed, native_w, native_h, sent_w, sent_h


def track_ball_with_ai(
    input_path: Path, fps: float, frame_w: int, frame_h: int,
    n_anchors: int = N_ANCHOR_FRAMES,
) -> tuple[list[_Det], dict]:
    """Primary AI-based ball tracker.

    Returns (track, info_dict). `track` is a list of _Det in NATIVE
    pixel coordinates, dense over the rendered flight range (impact
    through apex). `info_dict` carries diagnostics for logging and for
    feeding into the existing debug-image writer.

    On any failure mode (no API key, no anchors, fit failure) returns
    ([], {"stop_reason": "..."}). The caller treats that as fallthrough
    to the classical CV pipeline.
    """
    info: dict = {
        "stop_reason": None,
        "model": MODEL,
        "n_anchors_requested": 0,
        "n_anchors_returned": 0,
        "anchors": [],  # list of {frame, x, y, confidence, notes}
        "impact_frame": None,
    }

    if not have_ai_tracer():
        info["stop_reason"] = "ai_tracer_unavailable"
        return [], info

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        info["stop_reason"] = "cap_failed"
        return [], info
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if total_frames <= 1:
        info["stop_reason"] = "no_frames"
        return [], info

    sample_indices = _sample_frame_indices(total_frames, n_anchors)
    info["n_anchors_requested"] = len(sample_indices)
    log.info(
        "ai_tracer: querying Claude (%s) for %d anchors out of %d frames",
        MODEL, len(sample_indices), total_frames,
    )

    client = Anthropic()  # ANTHROPIC_API_KEY picked up from env

    raw_results: list[tuple[int, dict | None, int, int, int, int]] = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as ex:
        futures = [
            ex.submit(_query_anchor, input_path, client, idx)
            for idx in sample_indices
        ]
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                raw_results.append(res)

    raw_results.sort(key=lambda r: r[0])

    # Filter to accepted anchors and scale coordinates back to native.
    anchors: list[tuple[int, float, float]] = []
    for frame_idx, parsed, native_w, native_h, sent_w, sent_h in raw_results:
        if not parsed:
            continue
        found = bool(parsed.get("found"))
        confidence = str(parsed.get("confidence") or "low").lower()
        x = parsed.get("x")
        y = parsed.get("y")
        notes = str(parsed.get("notes") or "")[:80]
        anchor_log = {
            "frame": frame_idx, "found": found, "confidence": confidence,
            "x": x, "y": y, "notes": notes,
        }
        info["anchors"].append(anchor_log)
        if not found or confidence not in ACCEPTED_CONFIDENCE:
            continue
        if x is None or y is None:
            continue
        try:
            xv = float(x); yv = float(y)
        except (TypeError, ValueError):
            continue
        # Scale from sent (resized) image coords back to native frame.
        if sent_w > 0 and sent_h > 0:
            xv = xv * (native_w / float(sent_w))
            yv = yv * (native_h / float(sent_h))
        # Sanity-clip to frame bounds; Claude occasionally rounds slightly past.
        xv = max(0.0, min(float(frame_w - 1), xv))
        yv = max(0.0, min(float(frame_h - 1), yv))
        anchors.append((frame_idx, xv, yv))

    info["n_anchors_returned"] = len(anchors)
    log.info(
        "ai_tracer: %d/%d anchors accepted (notes=%s)",
        len(anchors), len(sample_indices),
        [(a["frame"], a["confidence"], a["notes"]) for a in info["anchors"]],
    )

    if len(anchors) < MIN_ANCHORS_FOR_FIT:
        info["stop_reason"] = f"too_few_anchors ({len(anchors)} < {MIN_ANCHORS_FOR_FIT})"
        return [], info

    track = _fit_and_densify(anchors, total_frames, frame_w, frame_h)
    if not track:
        info["stop_reason"] = "fit_failed"
        return [], info

    info["impact_frame"] = track[0].frame
    info["stop_reason"] = "ok"
    return track, info


def _fit_and_densify(
    anchors: list[tuple[int, float, float]],
    total_frames: int,
    frame_w: int,
    frame_h: int,
) -> list[_Det]:
    """Fit y quadratic / x linear in frame index across the anchors,
    then emit a dense per-frame _Det list from the first anchor's frame
    forward to the parabola apex (or the last anchor's frame if no
    apex is reached). Discards descent — the tracer overlay should
    fade out at the top of the arc, not chase the ball back down."""
    frames = np.array([a[0] for a in anchors], dtype=np.float64)
    xs = np.array([a[1] for a in anchors], dtype=np.float64)
    ys = np.array([a[2] for a in anchors], dtype=np.float64)

    try:
        y_coef = np.polyfit(frames, ys, 2)  # [a, b, c] for y = a*f^2 + b*f + c
        x_coef = np.polyfit(frames, xs, 1)  # [m, k] for x = m*f + k
    except Exception as exc:
        log.warning("ai_tracer: polyfit failed: %s", exc)
        return []

    a_y = float(y_coef[0])
    b_y = float(y_coef[1])
    # If the parabola opens the wrong way (a_y <= 0 in image coords
    # means the ball gets HIGHER as frames go on, which is wrong for
    # an image where y grows downward — but our ball goes UP, so y
    # decreases then increases, i.e. a_y SHOULD be positive). When the
    # anchors don't show this expected curvature it usually means too
    # few mid-flight samples; just emit a linear-y interpolation.
    use_parabola = a_y > 1e-9

    f_min = int(min(frames))
    f_max = int(max(frames))

    if use_parabola:
        vertex_frame = -b_y / (2.0 * a_y)
        # Apex truncation: only render impact -> apex, no descent.
        if f_min < vertex_frame < f_max:
            f_max = int(round(vertex_frame))

    f_min = max(0, f_min)
    f_max = min(total_frames - 1, f_max)
    if f_max < f_min:
        return []

    out: list[_Det] = []
    for f in range(f_min, f_max + 1):
        if use_parabola:
            y = float(np.polyval(y_coef, f))
        else:
            # Straight linear interpolation in y as well.
            y_coef_lin = np.polyfit(frames, ys, 1)
            y = float(np.polyval(y_coef_lin, f))
        x = float(np.polyval(x_coef, f))
        x = max(0.0, min(float(frame_w - 1), x))
        y = max(0.0, min(float(frame_h - 1), y))
        out.append(_Det(frame=f, x=x, y=y, radius=5.0))
    return out
