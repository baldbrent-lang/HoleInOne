from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..deps import require_admin
from ..models import (
    AuditLog,
    ClipProcessingStatus,
    Course,
    HIOStatus,
    HoleInOneEvent,
    Participant,
    TeeTime,
    VideoClip,
)
from ..schemas import (
    CourseCreate,
    CourseOut,
    HIOEventOut,
    HIOReviewAction,
)
from ..services import notifications
from ..services.qr import generate_qr_png

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/courses", response_model=list[CourseOut])
def list_courses(db: Session = Depends(get_db)):
    return db.query(Course).order_by(Course.created_at.desc()).all()


@router.post("/courses", response_model=CourseOut)
def create_course(payload: CourseCreate, db: Session = Depends(get_db)):
    course = Course(
        name=payload.name,
        location=payload.location,
        par3_holes=payload.par3_holes,
        minutes_per_hole=payload.minutes_per_hole,
        tee_sheet_provider=payload.tee_sheet_provider,
        tee_sheet_config=payload.tee_sheet_config,
    )
    db.add(course)
    db.commit()
    db.refresh(course)
    return course


@router.get("/courses/{course_id}/qr.png")
def course_qr_png(course_id: int, db: Session = Depends(get_db)):
    course = db.get(Course, course_id)
    if not course:
        raise HTTPException(404, "course not found")
    url = f"{settings.app_base_url}/r/{course.qr_token}"
    png = generate_qr_png(url)
    return Response(content=png, media_type="image/png")


@router.get("/stats")
def stats(db: Session = Depends(get_db)):
    now = datetime.utcnow()
    day_ago = now - timedelta(days=1)
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    def _count_since(since):
        return db.query(Participant).filter(Participant.created_at >= since).count()

    total_participants = db.query(Participant).count()
    total_paid = db.query(Participant).filter(Participant.paid).count()
    revenue_cents = total_paid * settings.registration_price_cents

    by_course = (
        db.query(Course.name, func.count(Participant.id))
        .join(TeeTime, TeeTime.course_id == Course.id)
        .join(Participant, Participant.tee_time_id == TeeTime.id)
        .group_by(Course.name)
        .all()
    )

    return {
        "participants": {
            "total": total_participants,
            "day": _count_since(day_ago),
            "week": _count_since(week_ago),
            "month": _count_since(month_ago),
        },
        "revenue_cents": revenue_cents,
        "clips_by_status": dict(
            db.query(VideoClip.processing_status, func.count()).group_by(VideoClip.processing_status).all()
        ),
        "by_course": [{"course": c, "participants": n} for c, n in by_course],
    }


@router.get("/participants/{participant_id}")
def participant_detail(participant_id: int, db: Session = Depends(get_db)):
    p = db.get(Participant, participant_id)
    if not p:
        raise HTTPException(404, "participant not found")
    return {
        "id": p.id,
        "name": p.name,
        "mobile": p.mobile,
        "email": p.email,
        "paid": p.paid,
        "gallery_url": f"{settings.app_base_url}/g/{p.gallery_token}",
        "tee_time": {
            "id": p.tee_time.id,
            "starts_at": p.tee_time.starts_at,
            "course_id": p.tee_time.course_id,
        },
        "clips": len(p.clips),
    }


@router.post("/participants/{participant_id}/resend-gallery")
def resend_gallery(participant_id: int, db: Session = Depends(get_db)):
    p = db.get(Participant, participant_id)
    if not p:
        raise HTTPException(404, "participant not found")
    gallery_url = f"{settings.app_base_url}/g/{p.gallery_token}"
    notifications.notify_gallery_ready(p.name, p.mobile, p.email, gallery_url)
    db.add(AuditLog(actor="admin", action="resend_gallery", target=f"participant:{p.id}"))
    db.commit()
    return {"ok": True, "gallery_url": gallery_url}


@router.get("/flagged-clips")
def flagged_clips(db: Session = Depends(get_db)):
    clips = (
        db.query(VideoClip)
        .filter(VideoClip.processing_status.in_([
            ClipProcessingStatus.flagged.value,
            ClipProcessingStatus.unassigned.value,
        ]))
        .order_by(VideoClip.captured_at.desc())
        .all()
    )
    return [
        {
            "id": c.id,
            "course_id": c.course_id,
            "hole_number": c.hole_number,
            "camera_type": c.camera_type,
            "captured_at": c.captured_at,
            "status": c.processing_status,
            "participant_id": c.participant_id,
            "note": c.issue_note,
            "source_url": c.source_url,
        }
        for c in clips
    ]


@router.post("/clips/{clip_id}/assign")
def manually_assign_clip(clip_id: int, participant_id: int, db: Session = Depends(get_db)):
    clip = db.get(VideoClip, clip_id)
    participant = db.get(Participant, participant_id)
    if not clip or not participant:
        raise HTTPException(404, "clip or participant missing")
    clip.participant_id = participant.id
    clip.processing_status = ClipProcessingStatus.assigned.value
    clip.issue_note = None
    db.add(AuditLog(actor="admin", action="assign_clip", target=f"clip:{clip.id}->p:{participant.id}"))
    db.commit()
    return {"ok": True}


# --- Hole-in-one review queue ------------------------------------------------

@router.get("/hio", response_model=list[HIOEventOut])
def list_hio_events(db: Session = Depends(get_db), status: str | None = None):
    q = db.query(HoleInOneEvent)
    if status:
        q = q.filter(HoleInOneEvent.status == status)
    return q.order_by(HoleInOneEvent.created_at.desc()).all()


@router.get("/hio/{event_id}")
def hio_event_detail(event_id: int, db: Session = Depends(get_db)):
    evt = db.get(HoleInOneEvent, event_id)
    if not evt:
        raise HTTPException(404, "event not found")
    participant = db.get(Participant, evt.participant_id)
    clip_ids = [cid for cid in (evt.tee_clip_id, evt.wide_clip_id, evt.hole_clip_id) if cid]
    hole_clips = db.query(VideoClip).filter(VideoClip.id.in_(clip_ids)).all() if clip_ids else []
    context_clips = (
        db.query(VideoClip)
        .filter(
            VideoClip.participant_id == evt.participant_id,
            VideoClip.hole_number == evt.hole_number,
        )
        .all()
    )
    angles = {c.camera_type: c for c in context_clips}
    return {
        "event": {
            "id": evt.id,
            "status": evt.status,
            "hole_number": evt.hole_number,
            "reviewer": evt.reviewer,
            "decision_note": evt.decision_note,
            "decided_at": evt.decided_at,
            "created_at": evt.created_at,
        },
        "participant": {
            "id": participant.id if participant else None,
            "name": participant.name if participant else None,
            "mobile": participant.mobile if participant else None,
            "email": participant.email if participant else None,
        },
        "clips": [
            {
                "id": c.id,
                "camera_type": c.camera_type,
                "source_url": c.source_url,
                "thumbnail_url": c.thumbnail_url,
                "captured_at": c.captured_at,
                "ball_in_cup": c.ball_in_cup,
            }
            for c in (list(angles.values()) or hole_clips)
        ],
    }


@router.post("/hio/{event_id}/decision")
def hio_decide(event_id: int, payload: HIOReviewAction, db: Session = Depends(get_db)):
    evt = db.get(HoleInOneEvent, event_id)
    if not evt:
        raise HTTPException(404, "event not found")
    mapping = {
        "approve": HIOStatus.approved.value,
        "reject": HIOStatus.rejected.value,
        "needs_more": HIOStatus.needs_more.value,
    }
    new_status = mapping.get(payload.action)
    if not new_status:
        raise HTTPException(400, "invalid action")
    evt.status = new_status
    evt.reviewer = payload.reviewer
    evt.decision_note = payload.note
    evt.decided_at = datetime.utcnow()
    db.add(
        AuditLog(
            actor=payload.reviewer,
            action=f"hio_{payload.action}",
            target=f"hio_event:{evt.id}",
            detail=payload.note,
        )
    )

    if new_status == HIOStatus.approved.value:
        p = db.get(Participant, evt.participant_id)
        if p:
            gallery_url = f"{settings.app_base_url}/g/{p.gallery_token}"
            notifications.notify_hio_confirmed(p.name, p.mobile, p.email, gallery_url)
    db.commit()
    return {"ok": True, "status": new_status}
