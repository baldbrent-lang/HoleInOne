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

# Number of anchor frames sent to Claude. Two-pass strategy:
#   Pass 1: N_ANCHORS_SCOUT frames spread across the whole clip to
#           find which window contains the ball in flight.
#   Pass 2: N_ANCHORS_DENSE frames packed into that window for the
#           parabola fit.
# 8 evenly-spaced anchors on a 10s slow-mo clip easily stepped over a
# 1-2s ball-flight window. Scout-then-densify gives us reliable
# coverage regardless of clip length / flight duration.
N_ANCHOR_FRAMES = 8  # legacy single-pass count (still used as a fallback)
N_ANCHORS_SCOUT = 10
N_ANCHORS_DENSE = 10
# How far on either side of a "ball found" scout frame we densify.
# 0.5s in clip-time is wide enough to cover the rest of the flight
# even if the scout caught only one end of it.
DENSE_WINDOW_SEC = 1.5

# Frames are downscaled to this max width before being sent so token
# count stays bounded regardless of source resolution (4K GoPros etc.).
MAX_IMAGE_WIDTH = 1280

# Confidence levels Claude can return; only "high" / "medium" anchors
# feed the parabola fit, "low" are discarded as too risky.
ACCEPTED_CONFIDENCE = {"high", "medium"}

# Need at least 3 anchor points for a quadratic fit. Below this we
# bail and let the caller fall back to classical CV.
MIN_ANCHORS_FOR_FIT = 3

# Concurrent API calls — keep per-clip latency under ~15s while
# staying well under rate limits.
MAX_CONCURRENT_REQUESTS = 8


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


def _run_pass(
    input_path: Path, client, indices: list[int],
    frame_w: int, frame_h: int, info: dict,
) -> list[tuple[int, float, float]]:
    """Query Claude for each frame in `indices`, return accepted anchors
    in NATIVE coords. Appends per-frame Claude responses to info["anchors"]
    for diagnostics."""
    raw_results: list[tuple[int, dict | None, int, int, int, int]] = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as ex:
        futures = [
            ex.submit(_query_anchor, input_path, client, idx)
            for idx in indices
        ]
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                raw_results.append(res)
    raw_results.sort(key=lambda r: r[0])

    accepted: list[tuple[int, float, float]] = []
    for frame_idx, parsed, native_w, native_h, sent_w, sent_h in raw_results:
        if not parsed:
            continue
        found = bool(parsed.get("found"))
        confidence = str(parsed.get("confidence") or "low").lower()
        x = parsed.get("x")
        y = parsed.get("y")
        notes = str(parsed.get("notes") or "")[:80]
        info["anchors"].append({
            "frame": frame_idx, "found": found, "confidence": confidence,
            "x": x, "y": y, "notes": notes,
        })
        if not found or confidence not in ACCEPTED_CONFIDENCE:
            continue
        if x is None or y is None:
            continue
        try:
            xv = float(x); yv = float(y)
        except (TypeError, ValueError):
            continue
        if sent_w > 0 and sent_h > 0:
            xv = xv * (native_w / float(sent_w))
            yv = yv * (native_h / float(sent_h))
        xv = max(0.0, min(float(frame_w - 1), xv))
        yv = max(0.0, min(float(frame_h - 1), yv))
        accepted.append((frame_idx, xv, yv))
    return accepted


def _dense_indices_around(
    center_lo: int, center_hi: int, total_frames: int,
    fps: float, n_samples: int, window_sec: float = DENSE_WINDOW_SEC,
) -> list[int]:
    """Spread `n_samples` integer frame indices across a window centered
    on [center_lo, center_hi], extended by ±window_sec at the ends so
    we catch nearby flight frames the scout missed."""
    pad_frames = int(round(window_sec * max(fps, 1.0)))
    lo = max(0, center_lo - pad_frames)
    hi = min(total_frames - 1, center_hi + pad_frames)
    if hi <= lo:
        return [lo]
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


def track_ball_with_ai(
    input_path: Path, fps: float, frame_w: int, frame_h: int,
    n_anchors: int = N_ANCHOR_FRAMES,
) -> tuple[list[_Det], dict]:
    """Primary AI-based ball tracker.

    Two-pass strategy:
      Scout: sample N_ANCHORS_SCOUT frames evenly across the clip.
             Find which scout frames Claude says contain a ball.
      Dense: sample N_ANCHORS_DENSE more frames packed into a window
             around the scout hits. This nails down the parabola.

    Returns (track, info_dict). `track` is a list of _Det in NATIVE
    pixel coordinates, dense over the rendered flight range (impact
    through apex). On any failure mode (no API key, no anchors found,
    fit failure) returns ([], {"stop_reason": "..."}) and the caller
    falls through to the classical CV pipeline.
    """
    info: dict = {
        "stop_reason": None,
        "model": MODEL,
        "n_anchors_requested": 0,
        "n_anchors_returned": 0,
        "anchors": [],  # list of {frame, x, y, confidence, notes}
        "impact_frame": None,
        "scout_hits": [],
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

    client = Anthropic()  # ANTHROPIC_API_KEY picked up from env

    # --- Pass 1: scout -----------------------------------------------
    scout_indices = _sample_frame_indices(total_frames, N_ANCHORS_SCOUT)
    log.info(
        "ai_tracer: SCOUT — querying Claude (%s) for %d frames out of %d (fps=%.1f)",
        MODEL, len(scout_indices), total_frames, fps,
    )
    scout_anchors = _run_pass(input_path, client, scout_indices, frame_w, frame_h, info)
    info["scout_hits"] = [a[0] for a in scout_anchors]
    log.info(
        "ai_tracer: SCOUT — %d/%d hits at frames %s",
        len(scout_anchors), len(scout_indices), info["scout_hits"],
    )

    anchors: list[tuple[int, float, float]] = list(scout_anchors)

    # --- Pass 2: densify around the hit window ----------------------
    if scout_anchors:
        hit_lo = min(a[0] for a in scout_anchors)
        hit_hi = max(a[0] for a in scout_anchors)
        dense_indices = _dense_indices_around(
            hit_lo, hit_hi, total_frames, fps, N_ANCHORS_DENSE,
        )
        # Don't re-query frames already scouted (the dict in info["anchors"]
        # tracks every frame_idx we've seen).
        seen_frames = {a["frame"] for a in info["anchors"]}
        dense_indices = [i for i in dense_indices if i not in seen_frames]
        if dense_indices:
            log.info(
                "ai_tracer: DENSE — querying %d frames in window [%d, %d]",
                len(dense_indices), hit_lo, hit_hi,
            )
            dense_anchors = _run_pass(
                input_path, client, dense_indices, frame_w, frame_h, info,
            )
            anchors.extend(dense_anchors)
            log.info(
                "ai_tracer: DENSE — %d/%d additional hits",
                len(dense_anchors), len(dense_indices),
            )

    anchors.sort(key=lambda a: a[0])
    info["n_anchors_requested"] = len({a["frame"] for a in info["anchors"]})
    info["n_anchors_returned"] = len(anchors)
    log.info(
        "ai_tracer: total %d/%d anchors accepted (notes=%s)",
        len(anchors), info["n_anchors_requested"],
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
