"""Operator-verified examples used as few-shot reference in the
AI tracer's Claude vision calls.

Every time the Edit wizard saves an `edit_metrics` block, we
accumulate ground-truth answers for the questions the tracer asks:
which frame is address, which frame is impact, where's the ball,
which hand is the golfer. This module looks those answers up,
renders the reference JPEG (cached on disk), and turns them into
Claude API message blocks.

Quality gate (default): only clips whose produced VideoClip was
promoted to the Broadcast channel (`is_highlight=True`) are eligible.
That filters out operator first-passes that turned out to be wrong
or got superseded.

Per call we attach at most `limit` examples — default 3. Token cost
grows linearly; 3 keeps the Claude bill under control while still
giving the model meaningful prior.

The disk cache lives under uploads/clips/_examples/. Each example
gets one rendered JPEG keyed by (lvu_id, swing_idx, kind), so the
first call for a given example pays the encode cost and every
subsequent call reuses the file.

Best-effort throughout: a missing source video, unreadable frame, or
encode failure logs and returns no example. Callers fall back to
zero-shot.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from ..config import settings
from ..models import Course, LongVideoUpload, VideoClip

log = logging.getLogger("golfreelz.tracer_examples")

CLIPS_DIR = Path(__file__).resolve().parents[2] / settings.upload_dir / "clips"
EXAMPLES_CACHE_DIR = CLIPS_DIR / "_examples"

# Kinds the tracer asks about. Each is wired into a specific
# ai_tracer.py Claude call.
KIND_ADDRESS = "address"
KIND_IMPACT = "impact"
KIND_HANDEDNESS = "handedness"
ALL_KINDS = (KIND_ADDRESS, KIND_IMPACT, KIND_HANDEDNESS)

# Defaults match the scope. Raise the limit only after measuring the
# token-cost impact — every additional example adds ~1.5k input
# tokens to the prompt.
DEFAULT_LIMIT = 3
DEFAULT_FRAME_W = 768   # downscale target for example JPEGs


def is_enabled() -> bool:
    """Master switch via env var. Set AI_TRACER_USE_EXAMPLES=0 to
    fall back to pure zero-shot during incidents or A/B."""
    val = os.environ.get("AI_TRACER_USE_EXAMPLES", "1").strip().lower()
    return val not in ("", "0", "false", "off", "no")


@dataclass
class Example:
    """One operator-verified ground-truth point. Holds the source
    metadata; the rendered JPEG is materialized on demand via
    render_jpeg()."""

    lvu_id: int
    swing_idx: int                     # 0 for single-swing uploads
    kind: str                          # one of ALL_KINDS
    source_path: Path                  # raw video on disk
    frame_idx: int                     # frame the answer applies to
    course_id: int
    hole_number: int
    answer: dict[str, Any] = field(default_factory=dict)
    # Examples for "handedness" / "impact" kinds carry ball coords so
    # we can draw a circle on the rendered example JPEG.
    ball_xy_native: Optional[tuple[int, int]] = None


def _swing_entries(em: Optional[dict]) -> list[dict]:
    """Both single-swing and multi-swing uploads serialize the same
    fields, just at different nesting depths. Normalize to a list of
    swing-dicts so callers can iterate uniformly."""
    if not em:
        return []
    if isinstance(em.get("swings"), list) and em["swings"]:
        return [s for s in em["swings"] if isinstance(s, dict)]
    # Single-swing — pretend it's a one-element list.
    return [em]


def _ball_xy(swing: dict) -> Optional[tuple[int, int]]:
    ball = swing.get("ball") or {}
    if not isinstance(ball, dict):
        return None
    try:
        return (int(ball["x"]), int(ball["y"]))
    except (KeyError, TypeError, ValueError):
        return None


def _example_from_swing(
    lvu: LongVideoUpload, swing_idx: int, swing: dict, kind: str,
) -> Optional[Example]:
    """Try to build an Example for `kind` from one swing's
    edit_metrics. Returns None if the required ground truth isn't
    there (e.g. operator hasn't edited that aspect yet)."""
    if not lvu.tee_filename:
        return None
    source_path = CLIPS_DIR / lvu.tee_filename
    if not source_path.exists():
        return None

    if kind == KIND_ADDRESS:
        frame_idx = swing.get("address_frame")
        if frame_idx is None:
            return None
        return Example(
            lvu_id=lvu.id, swing_idx=swing_idx, kind=kind,
            source_path=source_path, frame_idx=int(frame_idx),
            course_id=lvu.course_id, hole_number=_swing_hole(swing, lvu),
            answer={"address_frame": int(frame_idx)},
            ball_xy_native=_ball_xy(swing),
        )
    if kind == KIND_IMPACT:
        impact = swing.get("impact_frame")
        address = swing.get("address_frame")
        if impact is None or address is None:
            return None
        return Example(
            lvu_id=lvu.id, swing_idx=swing_idx, kind=kind,
            source_path=source_path, frame_idx=int(impact),
            course_id=lvu.course_id, hole_number=_swing_hole(swing, lvu),
            answer={
                "impact_frame": int(impact),
                "address_frame": int(address),
            },
            ball_xy_native=_ball_xy(swing),
        )
    if kind == KIND_HANDEDNESS:
        address = swing.get("address_frame")
        handedness = swing.get("handedness")
        ball = _ball_xy(swing)
        if address is None or not handedness or ball is None:
            return None
        return Example(
            lvu_id=lvu.id, swing_idx=swing_idx, kind=kind,
            source_path=source_path, frame_idx=int(address),
            course_id=lvu.course_id, hole_number=_swing_hole(swing, lvu),
            answer={
                "handedness": str(handedness),
                "ball_x": ball[0], "ball_y": ball[1],
            },
            ball_xy_native=ball,
        )
    return None


def _swing_hole(swing: dict, lvu: LongVideoUpload) -> int:
    """Best-guess hole number for one swing within a long-upload row.
    Per-swing hole metadata isn't always populated; falls back to 0
    when missing so callers can still group by course."""
    try:
        return int(swing.get("hole_number"))
    except (TypeError, ValueError):
        return 0


def fetch_examples(
    db: Session,
    *,
    course_id: int,
    hole_number: Optional[int],
    kind: str,
    limit: int = DEFAULT_LIMIT,
    must_be_broadcast: bool = True,
    exclude_lvu_ids: Optional[set[int]] = None,
) -> list[Example]:
    """Pull the most-recent N operator-verified examples for one
    Claude question.

    Filters:
      - same course_id (always)
      - same hole_number first; if fewer than `limit` matches, pads
        with other holes at the same course
      - must_be_broadcast (default True): the produced clip must have
        is_highlight=True. This is the primary quality gate against
        polluting the example bank with bad edits.
      - exclude_lvu_ids: skip these long-upload ids (e.g. don't
        few-shot a clip with itself when re-processing).

    Newest first. Best-effort: anything that doesn't yield a usable
    ground-truth point is silently skipped, and we return whatever we
    found (possibly 0).
    """
    if kind not in ALL_KINDS:
        log.warning("tracer_examples: unknown kind=%r", kind)
        return []

    # Two-pass query: same hole, then any hole at same course.
    out: list[Example] = []
    seen_lvu: set[int] = set(exclude_lvu_ids or ())

    def _query(hole: Optional[int]) -> list[LongVideoUpload]:
        q = (
            db.query(LongVideoUpload)
            .filter(LongVideoUpload.course_id == course_id)
            .filter(LongVideoUpload.processing_status == "completed")
            .filter(LongVideoUpload.edit_metrics.isnot(None))
        )
        if must_be_broadcast:
            # Highlighted clips only. Join via VideoClip.long_upload_id.
            q = q.join(
                VideoClip, VideoClip.long_upload_id == LongVideoUpload.id,
            ).filter(VideoClip.is_highlight.is_(True))
            if hole is not None:
                q = q.filter(VideoClip.hole_number == int(hole))
        q = q.order_by(LongVideoUpload.created_at.desc())
        # Pull more than we need — multiple swings per row, filters
        # below may discard some.
        return q.limit(limit * 4).all()

    candidates: list[LongVideoUpload] = []
    if hole_number is not None:
        candidates.extend(_query(int(hole_number)))
    candidates.extend([r for r in _query(None) if r.id not in seen_lvu
                       and r not in candidates])

    for lvu in candidates:
        if lvu.id in seen_lvu:
            continue
        for s_idx, swing in enumerate(_swing_entries(lvu.edit_metrics)):
            ex = _example_from_swing(lvu, s_idx, swing, kind)
            if ex is None:
                continue
            out.append(ex)
            seen_lvu.add(lvu.id)
            # Only one swing per upload — keep the example set diverse.
            break
        if len(out) >= limit:
            break

    log.info(
        "tracer_examples: kind=%s course=%d hole=%s found=%d",
        kind, course_id, hole_number, len(out),
    )
    return out


def fetch_all_kinds(
    db: Session,
    *,
    course_id: int,
    hole_number: Optional[int],
    limit: int = DEFAULT_LIMIT,
    must_be_broadcast: bool = True,
    exclude_lvu_ids: Optional[set[int]] = None,
) -> dict[str, list[Example]]:
    """Convenience: fetch examples for every kind in one call."""
    if not is_enabled():
        return {k: [] for k in ALL_KINDS}
    return {
        k: fetch_examples(
            db,
            course_id=course_id, hole_number=hole_number, kind=k,
            limit=limit, must_be_broadcast=must_be_broadcast,
            exclude_lvu_ids=exclude_lvu_ids,
        )
        for k in ALL_KINDS
    }


# ---------------------------------------------------------------------
# Rendering: extract example frame, optionally annotate, JPEG-encode.
# ---------------------------------------------------------------------


def _cache_path(ex: Example) -> Path:
    EXAMPLES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return EXAMPLES_CACHE_DIR / (
        f"lvu{ex.lvu_id}_s{ex.swing_idx}_{ex.kind}.jpg"
    )


def render_jpeg(ex: Example, frame_w: int = DEFAULT_FRAME_W) -> Optional[bytes]:
    """Materialize the example's reference image. Caches on disk so
    repeated calls (and repeated tracer runs) only pay the encode
    cost once. Returns the raw JPEG bytes or None on failure."""
    cache = _cache_path(ex)
    if cache.exists():
        try:
            return cache.read_bytes()
        except OSError:
            pass  # fall through and re-render

    try:
        import cv2  # type: ignore
    except ImportError:
        log.warning("tracer_examples: opencv missing — can't render examples")
        return None

    cap = cv2.VideoCapture(str(ex.source_path))
    if not cap.isOpened():
        log.warning(
            "tracer_examples: could not open source for example "
            "lvu%d frame=%d", ex.lvu_id, ex.frame_idx,
        )
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ex.frame_idx))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        log.warning(
            "tracer_examples: could not read frame %d from %s",
            ex.frame_idx, ex.source_path.name,
        )
        return None

    # Annotate ball position when relevant — matches the blue circle
    # the impact/handedness calls draw on their working frames so
    # Claude sees consistent visual cues.
    if ex.ball_xy_native is not None and ex.kind in (KIND_IMPACT, KIND_HANDEDNESS):
        bx, by = ex.ball_xy_native
        try:
            cv2.circle(frame, (int(bx), int(by)), 18, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.circle(frame, (int(bx), int(by)), 16, (255, 60, 0), 4, cv2.LINE_AA)
        except Exception as exc:  # pragma: no cover
            log.warning("tracer_examples: ball circle failed: %s", exc)

    h, w = frame.shape[:2]
    if w > frame_w:
        scale = frame_w / float(w)
        try:
            frame = cv2.resize(
                frame, (frame_w, int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        except Exception as exc:  # pragma: no cover
            log.warning("tracer_examples: resize failed: %s", exc)
            return None

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
    if not ok:
        return None
    data = bytes(buf)
    try:
        cache.write_bytes(data)
    except OSError as exc:
        log.warning("tracer_examples: cache write failed (%s): %s", cache, exc)
    return data


# ---------------------------------------------------------------------
# Prompt-block builders. Each returns a list of {"type": ..., ...}
# blocks that the caller prepends to its content list.
# ---------------------------------------------------------------------


def _b64(b: bytes) -> str:
    return base64.standard_b64encode(b).decode("ascii")


def _intro_block(kind: str, n: int) -> dict:
    return {
        "type": "text",
        "text": (
            f"REFERENCE EXAMPLES ({n}) — operator-verified {kind} picks "
            f"from previous clips on this course. Use them as priors for "
            f"the visual cues / framing / lighting; the new clip below "
            f"is the actual question."
        ),
    }


def _outro_block() -> dict:
    return {
        "type": "text",
        "text": "End of reference examples. Now the actual clip:",
    }


def example_blocks(examples: list[Example], kind: str) -> list[dict]:
    """Turn a list of Examples into prompt content blocks for `kind`.
    Empty list / disabled → empty list (caller's prompt unchanged)."""
    if not examples or not is_enabled():
        return []
    blocks: list[dict] = []
    rendered: list[tuple[Example, bytes]] = []
    for ex in examples:
        jpeg = render_jpeg(ex)
        if jpeg is None:
            continue
        rendered.append((ex, jpeg))
    if not rendered:
        return []
    blocks.append(_intro_block(kind, len(rendered)))
    for ex, jpeg in rendered:
        caption = _caption_for(ex)
        blocks.append({"type": "text", "text": caption})
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": _b64(jpeg),
            },
        })
    blocks.append(_outro_block())
    return blocks


def _caption_for(ex: Example) -> str:
    hole = f"hole {ex.hole_number}" if ex.hole_number else "this course"
    if ex.kind == KIND_ADDRESS:
        return (
            f"Example from {hole}: this image IS the address frame "
            f"(operator-verified). Look at the golfer's stance, club "
            f"position, and framing — the new clip's address frame "
            f"will look similar."
        )
    if ex.kind == KIND_IMPACT:
        return (
            f"Example from {hole}: this image IS the impact frame "
            f"(operator-verified). The blue circle marks the ball's "
            f"starting position; clubhead has returned to it. Look "
            f"for the same clubhead-at-ball geometry in the new clip."
        )
    if ex.kind == KIND_HANDEDNESS:
        h = ex.answer.get("handedness", "?")
        bx, by = ex.answer.get("ball_x"), ex.answer.get("ball_y")
        return (
            f"Example from {hole}: golfer is {h}-handed. Ball at "
            f"({bx}, {by}) is circled. Hands sit on the opposite "
            f"side of the ball from the lead foot."
        )
    return f"Example from {hole}."
