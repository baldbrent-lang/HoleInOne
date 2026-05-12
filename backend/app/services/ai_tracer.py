"""AI-powered golf analysis. Step 1: detect handedness.

This module is intentionally narrow — the broader plan is to build the
AI tracer up one capability at a time, verifying each step before
moving on. This file currently does ONE thing: given an uploaded clip,
ask Claude whether the golfer in the video is right- or left-handed.

The classical CV tracer in services/tracer.py is independent of this
module — `/admin/clips` runs only the classical pipeline; this module
is exercised exclusively from `/admin/clips/ai`.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
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

# Number of frames sampled across the clip and shown to Claude in a
# single multi-image API call. 6 frames covers address through
# follow-through with high probability of catching the backswing apex
# — the single most unambiguous cue for handedness. 3 was too few:
# evenly-spaced samples sometimes only captured address/follow-through
# poses where Claude has to guess from subtle grip/stance details that
# flip depending on camera orientation.
N_FRAMES = 6

# Per-frame resolution we send to Claude. 640 px wide gives the model
# enough detail to see the golfer's stance and grip clearly while
# staying under Anthropic's 1568-px efficient-token threshold and
# keeping per-request payload small.
FRAME_W = 640

HANDEDNESS_SYSTEM_PROMPT = (
    "You are analyzing still frames from a golf swing video. Your only "
    "job is to determine whether the golfer is right-handed or "
    "left-handed.\n\n"
    "CRITICAL — left/right reference frame:\n"
    "All cues below are in the GOLFER'S body frame, NOT the camera's "
    "image frame. The golfer's left hand is on the same side of HIS "
    "body regardless of where the camera sits. When the camera is in "
    "front of the golfer (you see his chest/face), the golfer's LEFT "
    "side appears on the RIGHT side of the image, and vice versa. When "
    "the camera is behind the golfer (you see his back), the sides "
    "match the image. ALWAYS reason about the GOLFER'S body sides — "
    "translate to image-side only at the very end, and only as a "
    "sanity check.\n\n"
    "Step 1: Identify the camera position relative to the golfer "
    "(behind / in-front / down-the-line / face-on / mixed).\n\n"
    "Step 2: For each frame, identify which swing phase it shows "
    "(address / takeaway / backswing-apex / downswing / impact / "
    "follow-through). Backswing-apex is the most informative for "
    "handedness; address and follow-through are second.\n\n"
    "Step 3: Apply the cues (in golfer's-body frame):\n"
    "  STRONGEST CUE — backswing direction:\n"
    "    - RIGHT-HANDED golfer: club travels back over his RIGHT "
    "shoulder; at the top, the clubhead is on his RIGHT side, "
    "high and behind.\n"
    "    - LEFT-HANDED golfer: mirror image — club travels back "
    "over his LEFT shoulder; clubhead at the top is on his LEFT side.\n"
    "  GRIP — at address the TOP hand on the grip (the one closer to "
    "the body, furthest from the clubhead) is:\n"
    "    - RIGHT-HANDED: LEFT hand on top.\n"
    "    - LEFT-HANDED: RIGHT hand on top.\n"
    "  STANCE — at address the ball sits roughly in front of the "
    "golfer's lead foot:\n"
    "    - RIGHT-HANDED: ball near his LEFT foot (lead is left).\n"
    "    - LEFT-HANDED: ball near his RIGHT foot (lead is right).\n"
    "  FOLLOW-THROUGH:\n"
    "    - RIGHT-HANDED: finishes facing target, club ends behind "
    "and above his LEFT shoulder.\n"
    "    - LEFT-HANDED: mirror — finishes with club behind his RIGHT "
    "shoulder.\n\n"
    "Step 4: Weight the evidence across frames. If backswing-apex is "
    "visible, that single frame should dominate. Disagree only if you "
    "have strong contrary evidence in another frame.\n\n"
    "Reply with ONE JSON object and nothing else:\n"
    "{\n"
    '  "camera_position": "behind"|"in_front"|"down_the_line"|"face_on"|"mixed",\n'
    '  "per_frame": [\n'
    '    {"frame": <int>, "phase": "address|takeaway|backswing_apex|downswing|impact|follow_through|other",\n'
    '     "evidence": "<≤15 word note about what side of the golfer\'s body the club is on>"}\n'
    "  ],\n"
    '  "handedness": "right"|"left"|"unknown",\n'
    '  "confidence": "high"|"medium"|"low",\n'
    '  "notes": "<≤30 word summary citing the single strongest cue>"\n'
    "}\n"
    "Use 'unknown' only if you genuinely cannot see the golfer's body "
    "in any frame. The per_frame array must include one entry per "
    "frame you were shown, in the same order."
)


def have_ai_tracer() -> bool:
    """True when the AI module can actually run end-to-end."""
    return HAS_CV and HAS_ANTHROPIC and bool(os.environ.get("ANTHROPIC_API_KEY"))


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Pull the first {...} block out of Claude's reply and parse it."""
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
    """Pick `n_samples` evenly-spaced indices, avoiding the very first
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


def _grab_frames(
    input_path: Path, indices: list[int], frame_w: int = FRAME_W,
) -> list[tuple[int, bytes]]:
    """Extract (frame_idx, jpeg_bytes) for each requested index, resized
    to frame_w wide. Frames that fail to read are skipped silently."""
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


def detect_handedness(input_path: Path) -> dict:
    """Ask Claude whether the golfer in this clip is right- or left-handed.

    Returns a dict::

        {
          "ok": bool,
          "error": str | None,
          "handedness": "right" | "left" | "unknown" | None,
          "confidence": "high" | "medium" | "low" | None,
          "notes": str | None,
          "model": str,
          "frames_sent": [int, ...],
        }

    Never raises — any failure (no API key, no SDK, clip unreadable,
    API error, parse failure) is reflected in `ok=False` + `error`.
    """
    info: dict = {
        "ok": False,
        "error": None,
        "handedness": None,
        "confidence": None,
        "notes": None,
        "camera_position": None,
        "per_frame": [],
        "model": MODEL,
        "frames_sent": [],
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
    frames = _grab_frames(input_path, indices)
    if not frames:
        info["error"] = "could not extract any frames"
        return info
    info["frames_sent"] = [idx for idx, _ in frames]
    log.info(
        "ai_tracer: handedness — sending %d frames %s of %d total @ %.1f fps to %s",
        len(frames), info["frames_sent"], total_frames, fps, MODEL,
    )

    # Build a single multi-image user message, labelling each frame so
    # Claude can refer to them in its reasoning.
    content: list[dict] = [{
        "type": "text",
        "text": (
            f"Below are {len(frames)} frames from a golf swing video, in "
            f"chronological order, sampled across a {total_frames}-frame "
            f"clip at {fps:.1f} fps. Each image is preceded by its frame "
            "number."
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
        "text": "Determine the golfer's handedness. Respond with JSON only.",
    })

    client = Anthropic()
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=256,
            system=[{
                "type": "text",
                "text": HANDEDNESS_SYSTEM_PROMPT,
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
    raw = "\n".join(text_chunks)
    parsed = _extract_json(raw)
    if not parsed:
        log.warning("ai_tracer: handedness — no JSON in response: %s", raw[:200])
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
    info["camera_position"] = str(parsed.get("camera_position") or "").lower() or None
    per_frame = parsed.get("per_frame")
    if isinstance(per_frame, list):
        info["per_frame"] = [
            {
                "frame": pf.get("frame"),
                "phase": str(pf.get("phase") or "").lower(),
                "evidence": str(pf.get("evidence") or "")[:200],
            }
            for pf in per_frame if isinstance(pf, dict)
        ]
    log.info(
        "ai_tracer: handedness — %s (%s) camera=%s — %s",
        info["handedness"], info["confidence"],
        info["camera_position"], info["notes"],
    )
    for pf in info["per_frame"]:
        log.info(
            "ai_tracer:   frame %s phase=%s evidence=%s",
            pf.get("frame"), pf.get("phase"), pf.get("evidence"),
        )
    return info
