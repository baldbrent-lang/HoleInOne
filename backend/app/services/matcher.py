"""Match incoming video clips to participants by tee time + playing order.

Heuristic:
- For a given hole, find all tee times at the course whose window plausibly
  contains the clip timestamp.
- The window for hole H starts at `tee_time.starts_at + (H-1) * minutes_per_hole`
  and ends `minutes_per_hole * 2` later (generous tolerance for pace of play).
- Within matching tee times, assign the clip to the participant whose
  playing_order matches the shot's sequence in the window.

If multiple candidates remain after ranking, the clip is left unassigned and
flagged for human review.
"""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from ..models import (
    ClipProcessingStatus,
    Course,
    HoleInOneEvent,
    Participant,
    TeeTime,
    VideoClip,
)


def _hole_window(tt: TeeTime, hole_number: int, minutes_per_hole: int) -> tuple:
    start_offset = timedelta(minutes=minutes_per_hole * (hole_number - 1))
    window_start = tt.starts_at + start_offset
    window_end = window_start + timedelta(minutes=minutes_per_hole * 2)
    return window_start, window_end


def match_clip(db: Session, clip: VideoClip) -> Participant | None:
    course = db.get(Course, clip.course_id)
    if not course:
        clip.processing_status = ClipProcessingStatus.unassigned.value
        return None

    mph = course.minutes_per_hole or 14

    candidates = (
        db.execute(
            select(TeeTime).where(
                and_(
                    TeeTime.course_id == course.id,
                    TeeTime.starts_at <= clip.captured_at,
                    TeeTime.starts_at >= clip.captured_at - timedelta(minutes=mph * 19),
                )
            )
        )
        .scalars()
        .all()
    )

    tee_time = None
    for tt in candidates:
        window_start, window_end = _hole_window(tt, clip.hole_number, mph)
        if window_start <= clip.captured_at <= window_end:
            if tee_time is not None:
                clip.processing_status = ClipProcessingStatus.flagged.value
                clip.issue_note = "ambiguous: multiple tee times overlap this window"
                return None
            tee_time = tt

    if tee_time is None:
        clip.processing_status = ClipProcessingStatus.unassigned.value
        return None

    existing_clips_on_hole = (
        db.execute(
            select(VideoClip).where(
                and_(
                    VideoClip.course_id == course.id,
                    VideoClip.hole_number == clip.hole_number,
                    VideoClip.captured_at >= tee_time.starts_at,
                    VideoClip.captured_at < clip.captured_at,
                    VideoClip.camera_type == clip.camera_type,
                )
            )
        )
        .scalars()
        .all()
    )
    shot_sequence = len(existing_clips_on_hole) + 1

    participant = (
        db.execute(
            select(Participant).where(
                and_(
                    Participant.tee_time_id == tee_time.id,
                    Participant.playing_order == shot_sequence,
                )
            )
        )
        .scalars()
        .first()
    )

    if participant is None:
        clip.processing_status = ClipProcessingStatus.flagged.value
        clip.issue_note = f"no participant with playing_order={shot_sequence}"
        return None

    clip.participant_id = participant.id
    clip.processing_status = ClipProcessingStatus.assigned.value

    if clip.ball_in_cup and clip.camera_type in ("hole", "tee"):
        _ensure_hio_event(db, participant, clip)

    return participant


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
