"""Match incoming video clips to participants by visual appearance.

Matching pipeline:
- Eligibility window is per-golfer, anchored to *registration time*:
  a clip is eligible to match a participant if the clip's captured_at
  is between participant.created_at and participant.created_at +
  match_window_minutes (default 9 hours). Clips captured before a
  golfer registers are never matched to them.
- Gather all participants at this course inside that window.
- In real mode: embed the clip, compare cosine similarity against each
  participant's selfie embedding, assign top match if margin > threshold.
- In stub mode (no real embedder configured): round-robin assignment
  across candidate participants on the same hole, so the demo flow
  still produces sensible per-golfer galleries.

Ambiguous matches and out-of-window clips are flagged for human review.
"""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import (
    ClipProcessingStatus,
    Course,
    HoleInOneEvent,
    Participant,
    TeeTime,
    VideoClip,
)
from . import appearance


def _candidates_in_window(db: Session, course: Course, clip: VideoClip) -> list[Participant]:
    """Participants registered at this course within the match window."""
    window = timedelta(minutes=settings.match_window_minutes)
    participants = (
        db.execute(
            select(Participant)
            .join(TeeTime, Participant.tee_time_id == TeeTime.id)
            .where(
                and_(
                    TeeTime.course_id == course.id,
                    Participant.created_at <= clip.captured_at,
                    Participant.created_at >= clip.captured_at - window,
                )
            )
        )
        .scalars()
        .all()
    )
    return list(participants)


def match_clip(db: Session, clip: VideoClip) -> Participant | None:
    course = db.get(Course, clip.course_id)
    if not course:
        clip.processing_status = ClipProcessingStatus.unassigned.value
        return None

    candidates = _candidates_in_window(db, course, clip)
    if not candidates:
        clip.processing_status = ClipProcessingStatus.unassigned.value
        clip.issue_note = "no registered participants in clip window"
        return None

    if appearance.is_stub():
        participant = _round_robin_pick(db, course, clip, candidates)
    else:
        participant = _appearance_pick(clip, candidates)
        if participant is None:
            clip.processing_status = ClipProcessingStatus.flagged.value
            clip.issue_note = "ambiguous: top appearance matches too close"
            return None

    clip.participant_id = participant.id
    clip.processing_status = ClipProcessingStatus.assigned.value

    if clip.ball_in_cup and clip.camera_type in ("hole", "tee"):
        _ensure_hio_event(db, participant, clip)

    return participant


def _round_robin_pick(
    db: Session, course: Course, clip: VideoClip, candidates: list[Participant]
) -> Participant:
    """Stub mode: assign to candidate with fewest existing clips on this hole."""
    candidate_ids = [p.id for p in candidates]
    counts = {pid: 0 for pid in candidate_ids}
    rows = (
        db.execute(
            select(VideoClip.participant_id, func.count(VideoClip.id))
            .where(
                VideoClip.course_id == course.id,
                VideoClip.hole_number == clip.hole_number,
                VideoClip.camera_type == clip.camera_type,
                VideoClip.participant_id.in_(candidate_ids),
            )
            .group_by(VideoClip.participant_id)
        )
    ).all()
    for pid, n in rows:
        counts[pid] = n
    return min(candidates, key=lambda p: (counts[p.id], p.id))


def _appearance_pick(clip: VideoClip, candidates: list[Participant]) -> Participant | None:
    """Real mode: cosine match clip embedding against participant embeddings."""
    clip_emb = appearance.embed_clip_url(clip.source_url)
    scored = []
    for p in candidates:
        emb = p.appearance_embedding or []
        scored.append((p, appearance.cosine(clip_emb, emb)))
    scored.sort(key=lambda x: -x[1])
    if not scored:
        return None
    if len(scored) > 1 and (scored[0][1] - scored[1][1]) < settings.embedding_min_margin:
        return None  # ambiguous
    return scored[0][0]


def _ensure_hio_event(db: Session, participant: Participant, clip: VideoClip) -> HoleInOneEvent:
    existing = (
        db.execute(
            select(HoleInOneEvent).where(
                and_(
                    HoleInOneEvent.participant_id == participant.id,
                    HoleInOneEvent.hole_number == clip.hole_number,
                )
            )
        )
        .scalars()
        .first()
    )
    if existing:
        return existing
    evt = HoleInOneEvent(participant_id=participant.id, hole_number=clip.hole_number)
    if clip.camera_type == "hole":
        evt.hole_clip_id = clip.id
    elif clip.camera_type == "tee":
        evt.tee_clip_id = clip.id
    elif clip.camera_type == "wide_green":
        evt.wide_clip_id = clip.id
    db.add(evt)
    return evt
