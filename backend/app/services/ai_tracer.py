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

# Pipeline shape:
#   Pass 1 (impact finder): build a labeled montage of N_MONTAGE_FRAMES
#       frames spanning the whole clip, ask Claude which frame is
#       closest to ball impact (clubhead meeting ball). One API call.
#   Pass 2 (tracker): sample N_TRACK_FRAMES frames in a window starting
#       slightly before the estimated impact and extending TRACK_SECONDS
#       afterwards. Query each in parallel for ball coordinates.
#   Fit a parabola through the in_flight anchors and render.
N_MONTAGE_FRAMES = 16          # 4x4 grid for the impact-finder pass
N_TRACK_FRAMES = 10            # tracking anchors after impact
TRACK_SECONDS = 2.5            # how far past impact to sample
TRACK_LEAD_SECONDS = 0.25      # small back-pad in case impact estimate is late

# Per-tile size in the impact montage. 4 cols * 384 = 1536 px wide —
# stays under Anthropic's 1568-px sweet spot for token-efficient
# vision processing while keeping each tile big enough to read.
MONTAGE_TILE_W = 384
MONTAGE_TILE_H = 216           # 16:9
MONTAGE_LABEL_BAND_H = 28      # black strip at top of each tile with frame label

# Frames are downscaled to this max width before being sent so token
# count stays bounded regardless of source resolution (4K GoPros etc.).
MAX_IMAGE_WIDTH = 1280

# Confidence levels Claude can return; only "high" / "medium" anchors
# feed the parabola fit, "low" are discarded as too risky.
ACCEPTED_CONFIDENCE = {"high", "medium"}

# Need at least 3 anchor points for a quadratic fit.
MIN_ANCHORS_FOR_FIT = 3

# Concurrent API calls — keep per-clip latency under ~15s while
# staying well under rate limits.
MAX_CONCURRENT_REQUESTS = 8


IMPACT_PROMPT = (
    "You are analyzing a montage of still frames from a par-3 golf tee "
    "shot, arranged in a grid. Each tile has a label at the top showing "
    "its frame number from the source video.\n\n"
    "Identify the SINGLE frame closest to the moment of IMPACT — when "
    "the clubhead meets the ball (just before the ball launches). The "
    "ball is still on the tee but the clubhead is right on it.\n\n"
    "Reply with ONE JSON object and nothing else:\n"
    "{\n"
    '  "impact_frame": <int frame number from the tile labels>,\n'
    '  "confidence": "high"|"medium"|"low",\n'
    '  "notes": "<short reasoning, max 15 words>"\n'
    "}\n"
    "If no tile shows impact (e.g. all are pre-swing or all post-flight) "
    "pick the closest tile — the next pass will refine the window. "
    "Never invent a frame number that isn't shown on a tile."
)

BALL_LOCATE_PROMPT = (
    "You are locating a golf ball in flight in a single frame from a "
    "par-3 tee shot. The camera is behind the golfer; the ball was just "
    "struck and is travelling away from the camera, rising and arcing "
    "toward the green.\n\n"
    "Find the ball:\n"
    "- Appears as a small bright white dot or short motion-blur streak, "
    "  3-15 pixels wide. Against grass: white-on-green. Against sky: "
    "  bright speck OR (overcast) a darker silhouette — still small "
    "  and round/oblong.\n"
    "- DO NOT confuse the ball with: shoes, belt buckle, hands, club "
    "  head, range balls on the tee mat, divots, leaves, clouds, "
    "  course markers, or background props.\n"
    "- If the ball is not visible in this frame (already left the frame, "
    "  occluded, hasn't been struck yet, etc.) set found=false.\n\n"
    "Reply with ONE JSON object and nothing else:\n"
    "{\n"
    '  "found": true|false,\n'
    '  "x": <int pixel x of ball center, or null>,\n'
    '  "y": <int pixel y of ball center, or null>,\n'
    '  "confidence": "high"|"medium"|"low",\n'
    '  "notes": "<short reasoning, max 12 words>"\n'
    "}\n"
    "Coordinates are in the image coordinate system provided (image may "
    "have been resized). Aim for the center of the ball."
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


def _ask_claude_with_image(
    client, jpeg_bytes: bytes, system_prompt: str, user_text: str,
    max_tokens: int = 256, log_tag: str = "",
) -> dict | None:
    """Send one image + prompt to Claude, parse the JSON reply. Returns
    None on any failure (logged, never raised). Used for both the
    impact-finder montage call and the per-frame ball-locate calls."""
    b64 = base64.standard_b64encode(jpeg_bytes).decode("ascii")
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": system_prompt,
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
        log.warning("ai_tracer: Claude API failed (%s): %s", log_tag, exc)
        return None
    text_chunks = [
        block.text for block in resp.content
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    if not text_chunks:
        log.warning("ai_tracer: empty response (%s)", log_tag)
        return None
    return _extract_json("\n".join(text_chunks))


def _build_impact_montage(
    input_path: Path, total_frames: int,
) -> tuple[bytes, list[int]] | None:
    """Compose a labeled grid of N_MONTAGE_FRAMES frames into a single
    JPEG for the impact-finder pass. Each tile has a black bar at the
    top showing its frame number so Claude can reference it.

    Returns (jpeg_bytes, frame_indices_in_grid_order) or None on failure.
    """
    indices = _sample_frame_indices(total_frames, N_MONTAGE_FRAMES)
    if not indices:
        return None
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return None
    try:
        tiles: list[tuple[int, "np.ndarray"]] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            tile = cv2.resize(
                frame, (MONTAGE_TILE_W, MONTAGE_TILE_H),
                interpolation=cv2.INTER_AREA,
            )
            # Black band at top with the frame number.
            cv2.rectangle(
                tile, (0, 0), (MONTAGE_TILE_W, MONTAGE_LABEL_BAND_H),
                (0, 0, 0), -1,
            )
            cv2.putText(
                tile, f"frame {idx}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )
            tiles.append((idx, tile))
        if not tiles:
            return None
    finally:
        cap.release()

    # Pack into a square-ish grid (cols = ceil(sqrt(N))).
    n = len(tiles)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    canvas = np.zeros(
        (rows * MONTAGE_TILE_H, cols * MONTAGE_TILE_W, 3),
        dtype=np.uint8,
    )
    grid_order: list[int] = []
    for i, (frame_idx, tile) in enumerate(tiles):
        r, c = divmod(i, cols)
        y0, y1 = r * MONTAGE_TILE_H, (r + 1) * MONTAGE_TILE_H
        x0, x1 = c * MONTAGE_TILE_W, (c + 1) * MONTAGE_TILE_W
        canvas[y0:y1, x0:x1] = tile
        grid_order.append(frame_idx)
    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return None
    return bytes(buf), grid_order


def _find_impact_frame(
    client, input_path: Path, total_frames: int, fps: float,
) -> tuple[int | None, dict]:
    """Build a labeled montage and ask Claude which frame is impact.

    Returns (impact_frame_index_or_None, diagnostics_dict). Diagnostics
    includes the candidate frames shown to Claude, the raw response,
    and the confidence — all useful for the debug UI.
    """
    diag: dict = {
        "candidate_frames": [],
        "impact_frame": None,
        "confidence": None,
        "notes": None,
        "raw_response": None,
    }
    built = _build_impact_montage(input_path, total_frames)
    if built is None:
        diag["error"] = "montage_build_failed"
        return None, diag
    jpeg_bytes, grid_order = built
    diag["candidate_frames"] = grid_order

    user_text = (
        f"Montage of {len(grid_order)} frames from a golf swing video. "
        f"Tile labels show frame numbers (range {grid_order[0]} - {grid_order[-1]}; "
        "total clip length {tot} frames at {fps:.1f} fps). "
        "Identify the frame closest to impact."
    ).format(tot=total_frames, fps=fps)

    parsed = _ask_claude_with_image(
        client, jpeg_bytes, IMPACT_PROMPT, user_text,
        max_tokens=256, log_tag="impact-montage",
    )
    diag["raw_response"] = parsed
    if not parsed:
        diag["error"] = "no_response"
        return None, diag
    try:
        impact = int(parsed.get("impact_frame"))
    except (TypeError, ValueError):
        diag["error"] = "could_not_parse_impact_frame"
        return None, diag
    if impact < 0 or impact >= total_frames:
        diag["error"] = f"impact_frame_out_of_range ({impact} not in [0,{total_frames}))"
        return None, diag
    diag["impact_frame"] = impact
    diag["confidence"] = str(parsed.get("confidence") or "").lower()
    diag["notes"] = str(parsed.get("notes") or "")[:120]
    return impact, diag


def _post_impact_indices(
    impact_frame: int, total_frames: int, fps: float,
    n_samples: int = N_TRACK_FRAMES,
    lead_seconds: float = TRACK_LEAD_SECONDS,
    track_seconds: float = TRACK_SECONDS,
) -> list[int]:
    """Choose `n_samples` evenly-spaced frame indices from just before
    the estimated impact through `track_seconds` afterwards. A small
    lead window (`lead_seconds`) ahead of impact catches edge cases
    where the impact estimate runs slightly late."""
    fps = max(fps, 1.0)
    lo = max(0, impact_frame - int(round(lead_seconds * fps)))
    hi = min(total_frames - 1, impact_frame + int(round(track_seconds * fps)))
    if hi <= lo:
        hi = min(total_frames - 1, lo + 1)
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


def _query_ball_locate(
    input_path: Path, client, frame_idx: int,
) -> dict | None:
    """Thread-pool worker: grab one frame and ask Claude for ball coords.
    Returns the full diagnostic record (frame, found, x, y, confidence,
    notes, native_w/h, sent_w/h) or None if the frame couldn't be read."""
    grab = _grab_frame_jpeg(input_path, frame_idx)
    if grab is None:
        return None
    jpeg_bytes, native_w, native_h, sent_w, sent_h = grab
    user_text = (
        f"Frame {frame_idx}. Image size: {sent_w}x{sent_h} px. "
        "Locate the airborne golf ball (JSON only)."
    )
    parsed = _ask_claude_with_image(
        client, jpeg_bytes, BALL_LOCATE_PROMPT, user_text,
        max_tokens=256, log_tag=f"frame {frame_idx}",
    ) or {}
    return {
        "frame": frame_idx,
        "found": bool(parsed.get("found")),
        "x": parsed.get("x"),
        "y": parsed.get("y"),
        "confidence": str(parsed.get("confidence") or "low").lower(),
        "notes": str(parsed.get("notes") or "")[:120],
        "native_w": native_w, "native_h": native_h,
        "sent_w": sent_w, "sent_h": sent_h,
    }


def track_ball_with_ai(
    input_path: Path, fps: float, frame_w: int, frame_h: int,
) -> tuple[list[_Det], dict]:
    """Primary AI-based ball tracker.

    Two-stage pipeline:
      1. Impact finder — one Claude call with a labeled montage of
         ~16 frames; Claude returns the single frame number closest
         to ball impact.
      2. Tracking pass — N_TRACK_FRAMES single-frame Claude calls in
         parallel, sampled from just before impact through ~2.5s after.
         Each call returns (found, x, y, confidence).

    Anchors that come back found=true with high/medium confidence feed
    a parabola fit; the densified per-frame _Det list is what gets
    rendered.

    Returns ([], info) with a populated stop_reason on any failure.
    """
    info: dict = {
        "stop_reason": None,
        "model": MODEL,
        "impact_frame": None,
        "impact_diag": {},
        "n_anchors_requested": 0,
        "n_anchors_returned": 0,
        "anchors": [],
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

    client = Anthropic()

    # --- Stage 1: find impact ---------------------------------------
    log.info(
        "ai_tracer: IMPACT — sending %d-frame montage to %s (clip=%d frames @ %.1ffps)",
        N_MONTAGE_FRAMES, MODEL, total_frames, fps,
    )
    impact_frame, impact_diag = _find_impact_frame(client, input_path, total_frames, fps)
    info["impact_diag"] = impact_diag
    if impact_frame is None:
        info["stop_reason"] = f"impact_finder_failed: {impact_diag.get('error', 'unknown')}"
        log.info("ai_tracer: IMPACT — failed (%s)", info["stop_reason"])
        return [], info
    info["impact_frame"] = impact_frame
    log.info(
        "ai_tracer: IMPACT — frame=%d confidence=%s notes=%s",
        impact_frame, impact_diag.get("confidence"), impact_diag.get("notes"),
    )

    # --- Stage 2: track ball through post-impact frames -------------
    track_indices = _post_impact_indices(impact_frame, total_frames, fps)
    info["n_anchors_requested"] = len(track_indices)
    log.info(
        "ai_tracer: TRACK — querying %d frames [%d..%d] for ball coords",
        len(track_indices), track_indices[0], track_indices[-1],
    )
    raw_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as ex:
        futures = [
            ex.submit(_query_ball_locate, input_path, client, idx)
            for idx in track_indices
        ]
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                raw_results.append(r)
    raw_results.sort(key=lambda r: r["frame"])

    # Filter to accepted in-flight anchors and scale into native coords.
    anchors: list[tuple[int, float, float]] = []
    for r in raw_results:
        info["anchors"].append({
            "frame": r["frame"], "found": r["found"], "confidence": r["confidence"],
            "x": r["x"], "y": r["y"], "notes": r["notes"],
        })
        if not r["found"] or r["confidence"] not in ACCEPTED_CONFIDENCE:
            continue
        if r["x"] is None or r["y"] is None:
            continue
        try:
            xv = float(r["x"]); yv = float(r["y"])
        except (TypeError, ValueError):
            continue
        if r["sent_w"] > 0 and r["sent_h"] > 0:
            xv = xv * (r["native_w"] / float(r["sent_w"]))
            yv = yv * (r["native_h"] / float(r["sent_h"]))
        xv = max(0.0, min(float(frame_w - 1), xv))
        yv = max(0.0, min(float(frame_h - 1), yv))
        anchors.append((r["frame"], xv, yv))

    info["n_anchors_returned"] = len(anchors)
    log.info(
        "ai_tracer: TRACK — %d/%d anchors accepted (notes=%s)",
        len(anchors), len(track_indices),
        [(a["frame"], a["confidence"], a["notes"]) for a in info["anchors"]],
    )

    if len(anchors) < MIN_ANCHORS_FOR_FIT:
        info["stop_reason"] = f"too_few_anchors ({len(anchors)} < {MIN_ANCHORS_FOR_FIT})"
        return [], info

    track = _fit_and_densify(anchors, total_frames, frame_w, frame_h)
    if not track:
        info["stop_reason"] = "fit_failed"
        return [], info

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
        if a.get("found") and a.get("confidence") in ACCEPTED_CONFIDENCE
    }
    impact_frame = ai_info.get("impact_frame")

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
    """Save a montage of every tracking frame with Claude's ball pick
    overlaid plus the impact frame in its own row at the top. Lets the
    operator see at a glance whether the AI's impact identification was
    right and which tracking frames had the ball visible."""
    anchors = ai_info.get("anchors") or []
    impact_frame = ai_info.get("impact_frame")
    if not anchors and impact_frame is None:
        return
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return
    try:
        tiles = []
        # First tile (when available): the impact frame Claude picked.
        if impact_frame is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(impact_frame))
            ok, frame = cap.read()
            if ok and frame is not None:
                tile = _make_debug_tile(
                    frame,
                    label=f"IMPACT f{impact_frame} ({ai_info.get('impact_diag', {}).get('confidence', '?')})",
                    x=None, y=None, color=(0, 200, 200),
                )
                tiles.append(tile)
        anchors_sorted = sorted(anchors, key=lambda a: a["frame"])
        for a in anchors_sorted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(a["frame"]))
            ok, frame = cap.read()
            if not ok:
                continue
            x, y = a.get("x"), a.get("y")
            color = (0, 230, 255) if a.get("found") else (60, 60, 200)
            found_str = "ball" if a.get("found") else "no ball"
            label = f"f{a['frame']} {found_str} ({a.get('confidence', '?')})"
            # API coords are in the resized-for-Claude image space; remap
            # them onto the debug tile by passing native dims through.
            tile = _make_debug_tile(
                frame, label=label, x=x, y=y, color=color,
                native_w=width, native_h=height,
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


def _make_debug_tile(
    frame, label: str, x, y, color, native_w: int = 0, native_h: int = 0,
):
    """Build one labeled debug tile. If x/y are supplied (in API/sent
    coord space, with native_w/native_h provided for the source frame),
    draw a marker at the corresponding location on the tile."""
    tile_w = 480
    scale = tile_w / float(frame.shape[1])
    tile = cv2.resize(
        frame, (tile_w, int(round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    if x is not None and y is not None and native_w > 0 and native_h > 0:
        api_w = min(native_w, MAX_IMAGE_WIDTH)
        api_scale = api_w / float(native_w)
        api_h = int(round(native_h * api_scale))
        tile_x = int(round(float(x) / float(api_w) * tile_w))
        tile_y = int(round(float(y) / float(api_h) * tile.shape[0]))
        cv2.circle(tile, (tile_x, tile_y), 10, color, 2, cv2.LINE_AA)
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(
        tile, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return tile
