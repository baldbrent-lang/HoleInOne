from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import ClipProcessingStatus, Course, Participant, VideoClip
from ..schemas import ClipOut, GalleryOut, IssueFlag, ParticipantOut

router = APIRouter(prefix="/api/gallery", tags=["gallery"])


def _get_participant(db: Session, token: str) -> Participant:
    p = db.query(Participant).filter(Participant.gallery_token == token).first()
    if not p:
        raise HTTPException(404, "gallery not found")
    return p


@router.get("/{gallery_token}", response_model=GalleryOut)
def get_gallery(gallery_token: str, db: Session = Depends(get_db)):
    participant = _get_participant(db, gallery_token)
    course = db.get(Course, participant.tee_time.course_id)
    clips = (
        db.query(VideoClip)
        .filter(
            VideoClip.participant_id == participant.id,
            VideoClip.processing_status == ClipProcessingStatus.assigned.value,
        )
        .order_by(VideoClip.hole_number.asc(), VideoClip.captured_at.asc())
        .all()
    )
    return GalleryOut(
        participant=ParticipantOut.model_validate(participant),
        course_name=course.name if course else "",
        livestream_url=course.livestream_url if course else None,
        hole_yardages=(course.hole_yardages or {}) if course else {},
        clips=[ClipOut.model_validate(c) for c in clips],
    )


@router.post("/{gallery_token}/clips/{clip_id}/flag", response_model=ClipOut)
def flag_clip(gallery_token: str, clip_id: int, payload: IssueFlag, db: Session = Depends(get_db)):
    participant = _get_participant(db, gallery_token)
    clip = db.get(VideoClip, clip_id)
    if not clip or clip.participant_id != participant.id:
        raise HTTPException(404, "clip not found for this gallery")
    clip.processing_status = ClipProcessingStatus.flagged.value
    clip.issue_note = f"[golfer flag] {payload.note.strip()}"
    db.commit()
    db.refresh(clip)
    return ClipOut.model_validate(clip)
