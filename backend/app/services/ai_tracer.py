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
# How far on either side of the inferred flight bracket we densify.
# Small because we don't want to spend the dense pass budget querying
# frames clearly before/after the flight — e.g. when scout finds the
# at_rest→gone transition, the flight is BETWEEN those frames, not
# outside. A small pad just accounts for transition-boundary slop.
DENSE_WINDOW_SEC = 0.4

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
    "You are a sports-vision assistant analyzing still frames from a "
    "par-3 tee shot. The camera sits behind a right- or left-handed "
    "golfer; when the ball is struck it travels away from the camera, "
    "rising and arcing toward the green.\n\n"
    "Classify the frame's state and, if a ball is visible, locate it.\n\n"
    "State values:\n"
    "- 'at_rest': the ball is still sitting on the tee (or ground), not "
    "yet struck. Backswing / address / takeaway are all 'at_rest'.\n"
    "- 'in_flight': the ball is airborne after being struck. Appears as "
    "a small bright dot or short motion-blur streak, 3-15 px wide. "
    "Against grass: white-on-green. Against sky: bright speck OR (overcast) "
    "darker silhouette — still small and round/oblong.\n"
    "- 'gone': the ball has been struck and has already left the frame "
    "(or landed and is no longer visibly airborne). Use this for any "
    "post-swing frame where you can tell the shot happened but no "
    "airborne ball is visible.\n"
    "- 'unknown': you cannot tell what state the frame is in.\n\n"
    "Do NOT confuse the ball with: the golfer's shoes, belt buckle, "
    "hands, club head, range balls on the tee mat, divots, leaves, "
    "clouds, course markers, or background props.\n\n"
    "Reply with ONE JSON object and nothing else, matching this schema:\n"
    "{\n"
    '  "state": "at_rest"|"in_flight"|"gone"|"unknown",\n'
    '  "x": <int pixel x of the ball center, or null if no ball visible>,\n'
    '  "y": <int pixel y of the ball center, or null if no ball visible>,\n'
    '  "confidence": "high"|"medium"|"low",\n'
    '  "notes": "<short reasoning, max 12 words>"\n'
    "}\n"
    "Provide x/y for BOTH 'at_rest' and 'in_flight' when you can see the "
    "ball; use null for 'gone' or 'unknown'. Coordinates are in the IMAGE "
    "coordinate system provided (the image may have been resized before "
    "you see it). Aim for the center of the ball."
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
    """Query Claude for each frame in `indices`, return accepted
    in_flight anchors in NATIVE coords. Appends per-frame Claude
    responses (including at_rest / gone classifications) to
    info["anchors"] for diagnostics and downstream windowing."""
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
        # Accept both the new "state" schema and the legacy "found" bool
        # in case any cached / mid-deploy responses come back in the old
        # shape — never throw, just record what we got.
        state = str(parsed.get("state") or "").lower()
        if not state:
            state = "in_flight" if parsed.get("found") else "gone"
        confidence = str(parsed.get("confidence") or "low").lower()
        x = parsed.get("x")
        y = parsed.get("y")
        notes = str(parsed.get("notes") or "")[:80]
        info["anchors"].append({
            "frame": frame_idx, "state": state, "confidence": confidence,
            "x": x, "y": y, "notes": notes,
        })
        if state != "in_flight" or confidence not in ACCEPTED_CONFIDENCE:
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

    # --- Pass 2: densify around the inferred flight window ----------
    # Three ways to determine the dense window, in order of preference:
    #   1. Scout had in_flight hits — bracket [min, max] of those hits.
    #   2. Scout has the transition at_rest → gone — flight is between
    #      the latest at_rest frame and the earliest gone frame after it.
    #      Common case for slow-mo clips where flight is < 1 of total.
    #   3. No bracket — give up; classical CV will pick this up.
    dense_lo: int | None = None
    dense_hi: int | None = None
    if scout_anchors:
        dense_lo = min(a[0] for a in scout_anchors)
        dense_hi = max(a[0] for a in scout_anchors)
        log.info("ai_tracer: dense window from in_flight hits [%d, %d]", dense_lo, dense_hi)
    else:
        at_rest_frames = sorted(
            a["frame"] for a in info["anchors"]
            if a["state"] == "at_rest" and a["confidence"] in ACCEPTED_CONFIDENCE
        )
        gone_frames = sorted(
            a["frame"] for a in info["anchors"]
            if a["state"] == "gone" and a["confidence"] in ACCEPTED_CONFIDENCE
        )
        if at_rest_frames and gone_frames:
            latest_rest = at_rest_frames[-1]
            earliest_gone = next((f for f in gone_frames if f > latest_rest), None)
            if earliest_gone is not None:
                dense_lo = latest_rest
                dense_hi = earliest_gone
                log.info(
                    "ai_tracer: dense window from at_rest→gone transition [%d, %d]",
                    dense_lo, dense_hi,
                )

    if dense_lo is not None and dense_hi is not None:
        dense_indices = _dense_indices_around(
            dense_lo, dense_hi, total_frames, fps, N_ANCHORS_DENSE,
        )
        seen_frames = {a["frame"] for a in info["anchors"]}
        dense_indices = [i for i in dense_indices if i not in seen_frames]
        if dense_indices:
            log.info(
                "ai_tracer: DENSE — querying %d frames in window [%d, %d]",
                len(dense_indices), dense_lo, dense_hi,
            )
            dense_anchors = _run_pass(
                input_path, client, dense_indices, frame_w, frame_h, info,
            )
            anchors.extend(dense_anchors)
            log.info(
                "ai_tracer: DENSE — %d/%d additional in_flight hits",
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


# --- Rendering --------------------------------------------------------------
# Visual style mirrors backend/app/services/tracer.py so the AI-rendered
# output sits next to the classical one without looking out of place.

TRACER_COLOR = (0, 140, 255)        # bright orange (BGR)
TRACER_HALO_COLOR = (40, 90, 200)
TRACER_THICKNESS = 5
DASH_LENGTH = 14
GAP_LENGTH = 10
ANCHOR_RING_COLOR = (0, 230, 255)   # cyan rings on Claude's verified anchor frames


def _draw_dashed(img, points: list[tuple[int, int]]) -> None:
    """Dashed-line overlay with halo. Lifted from tracer.py so the AI
    pipeline doesn't have to import the classical module."""
    if len(points) < 2:
        return
    pts = np.array(points, dtype=np.int32)
    cv2.polylines(img, [pts], False, TRACER_HALO_COLOR, TRACER_THICKNESS + 4, cv2.LINE_AA)
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
                cv2.line(
                    img, (int(cur[0]), int(cur[1])), (int(nxt[0]), int(nxt[1])),
                    TRACER_COLOR, TRACER_THICKNESS, cv2.LINE_AA,
                )
            cur = nxt
            traveled += step
            accumulated += step
            if accumulated >= target_seg:
                drawing = not drawing
                accumulated = 0.0


def render_ai_tracer(
    input_path: Path, output_path: Path, debug_path: Path | None = None,
) -> dict:
    """AI-only ball tracer render — NO classical CV fallback.

    Flow: ask Claude to classify a scout pass of frames + dense pass
    around the inferred flight window, fit a parabola, render the dashed
    overlay over the source MP4. On any failure (missing API key, no
    in_flight anchors found, fit failure) returns ok=False with a clear
    error string; the caller treats that as "AI couldn't find the ball
    in this clip" rather than falling back to anything else.

    Returns dict shape compatible with services.tracer.render_tracer:
        {ok, error, n_points, frame_range, fps, ai_info}
    Plus `ai_info` with scout/dense diagnostics for debug display.
    """
    if not HAS_CV:
        return {"ok": False, "error": "opencv not installed", "n_points": 0, "ai_info": {}}
    if not HAS_ANTHROPIC:
        return {
            "ok": False,
            "error": "anthropic SDK not installed (add `anthropic` to requirements.txt)",
            "n_points": 0,
            "ai_info": {},
        }
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {
            "ok": False,
            "error": "ANTHROPIC_API_KEY not set in environment",
            "n_points": 0,
            "ai_info": {},
        }

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return {"ok": False, "error": "could not open video", "n_points": 0, "ai_info": {}}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    try:
        track, ai_info = track_ball_with_ai(input_path, fps, width, height)
    except Exception as exc:  # pragma: no cover
        log.warning("ai_tracer: render crashed during track_ball_with_ai: %s", exc)
        return {
            "ok": False,
            "error": f"AI tracker exception: {exc}",
            "n_points": 0,
            "ai_info": {},
        }

    if debug_path is not None:
        try:
            _write_ai_debug_image(input_path, debug_path, ai_info, width, height)
        except Exception as exc:  # pragma: no cover
            log.warning("ai_tracer: debug-image write failed: %s", exc)

    if not track:
        return {
            "ok": False,
            "error": f"AI tracker found no ball flight: {ai_info.get('stop_reason')}",
            "n_points": 0,
            "ai_info": ai_info,
        }

    # Render the dashed overlay over the source video.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        return {
            "ok": False,
            "error": "VideoWriter failed to open output path",
            "n_points": len(track),
            "ai_info": ai_info,
        }
    cap2 = cv2.VideoCapture(str(input_path))
    if not cap2.isOpened():
        writer.release()
        return {
            "ok": False,
            "error": "could not re-open source video for rendering",
            "n_points": len(track),
            "ai_info": ai_info,
        }

    track_by_frame: dict[int, tuple[int, int]] = {
        int(d.frame): (int(d.x), int(d.y)) for d in track
    }
    track_frames = sorted(track_by_frame)
    anchor_frames = {
        a["frame"] for a in ai_info.get("anchors", [])
        if a.get("state") == "in_flight"
        and a.get("confidence") in ACCEPTED_CONFIDENCE
    }

    i = 0
    while True:
        ok, frame = cap2.read()
        if not ok:
            break
        seen = [track_by_frame[f] for f in track_frames if f <= i]
        _draw_dashed(frame, seen)
        # Highlight Claude's verified anchor frames so it's obvious which
        # points were AI-identified vs interpolated by the parabola fit.
        if i in anchor_frames and i in track_by_frame:
            cv2.circle(frame, track_by_frame[i], 9, ANCHOR_RING_COLOR, 2, cv2.LINE_AA)
        writer.write(frame)
        i += 1
    cap2.release()
    writer.release()

    return {
        "ok": True,
        "n_points": len(track),
        "frame_range": [int(track[0].frame), int(track[-1].frame)],
        "fps": float(fps),
        "ai_info": ai_info,
        "error": None,
    }


def _write_ai_debug_image(
    input_path: Path, debug_path: Path, ai_info: dict, width: int, height: int,
) -> None:
    """Save a montage of every anchor frame with Claude's classification
    overlaid. Lets us see at a glance which frames were at_rest /
    in_flight / gone and where Claude placed the ball."""
    anchors = ai_info.get("anchors") or []
    if not anchors:
        return
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return
    try:
        tiles = []
        anchors_sorted = sorted(anchors, key=lambda a: a["frame"])
        for a in anchors_sorted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(a["frame"]))
            ok, frame = cap.read()
            if not ok:
                continue
            # Downscale to a tile so the montage fits.
            tile_w = 480
            scale = tile_w / float(frame.shape[1])
            tile = cv2.resize(
                frame, (tile_w, int(round(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
            x, y = a.get("x"), a.get("y")
            # Anchor coords are in the (resized for API) image space;
            # the API saw images at MAX_IMAGE_WIDTH wide. Scale into the
            # debug tile coord space for the marker.
            if x is not None and y is not None:
                api_w = min(width, MAX_IMAGE_WIDTH)
                api_scale = api_w / float(width)
                api_h = int(round(height * api_scale))
                tile_x = int(round(x / float(api_w) * tile_w))
                tile_y = int(round(y / float(api_h) * tile.shape[0]))
                color = {
                    "in_flight": (0, 230, 255),
                    "at_rest":   (0, 200, 0),
                    "gone":      (60, 60, 200),
                    "unknown":   (180, 180, 180),
                }.get(a.get("state", ""), (255, 255, 255))
                cv2.circle(tile, (tile_x, tile_y), 10, color, 2, cv2.LINE_AA)
            label = f"f{a['frame']} {a.get('state', '?')} ({a.get('confidence', '?')})"
            cv2.rectangle(tile, (0, 0), (tile.shape[1], 24), (0, 0, 0), -1)
            cv2.putText(
                tile, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA,
            )
            tiles.append(tile)
        if not tiles:
            return
        cols = 4
        rows = (len(tiles) + cols - 1) // cols
        th = tiles[0].shape[0]
        tw = tiles[0].shape[1]
        montage = np.zeros((th * rows, tw * cols, 3), dtype=np.uint8)
        for idx, tile in enumerate(tiles):
            r, c = divmod(idx, cols)
            montage[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile
        cv2.imwrite(str(debug_path), montage, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    finally:
        cap.release()
