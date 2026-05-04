"""Course-scoped operator portal: per-course login, dashboard, participant
list. Each course has its own operator password set by the system admin
on the Course record. Tokens carry course_id, NOT user_id, and only
unlock data for that one course."""
from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..deps import current_operator
from ..models import (
    ClipProcessingStatus,
    Course,
    HIOStatus,
    HoleInOneEvent,
    Participant,
    TeeTime,
    VideoClip,
)
from ..services.auth import issue_operator_token, verify_password

router = APIRouter(prefix="/api/operator", tags=["operator"])


class OperatorLogin(BaseModel):
    course_token: str  # Course.qr_token (already public)
    password: str


class OperatorTokenResponse(BaseModel):
    token: str
    course: dict


@router.post("/login", response_model=OperatorTokenResponse)
def operator_login(payload: OperatorLogin, db: Session = Depends(get_db)):
    course = db.query(Course).filter(Course.qr_token == payload.course_token).first()
    if not course or not course.operator_password_hash:
        raise HTTPException(401, "wrong course or password")
    if not verify_password(payload.password, course.operator_password_hash):
        raise HTTPException(401, "wrong course or password")
    return OperatorTokenResponse(
        token=issue_operator_token(course.id),
        course={"id": course.id, "name": course.name, "location": course.location},
    )


@router.get("/me")
def operator_me(course: Course = Depends(current_operator)):
    return {
        "course": {
            "id": course.id,
            "name": course.name,
            "location": course.location,
            "par3_holes": course.par3_holes or [],
            "hole_yardages": course.hole_yardages or {},
            "livestream_url": course.livestream_url,
        }
    }


@router.get("/dashboard")
def operator_dashboard(
    course: Course = Depends(current_operator),
    db: Session = Depends(get_db),
):
    """Course-scoped stats + recent activity."""
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    assigned = ClipProcessingStatus.assigned.value

    # Participants in scope
    p_today = (
        db.query(func.count(Participant.id))
        .join(TeeTime, TeeTime.id == Participant.tee_time_id)
        .filter(TeeTime.course_id == course.id, Participant.created_at >= today_start)
        .scalar()
    )
    p_week = (
        db.query(func.count(Participant.id))
        .join(TeeTime, TeeTime.id == Participant.tee_time_id)
        .filter(TeeTime.course_id == course.id, Participant.created_at >= week_ago)
        .scalar()
    )
    p_month = (
        db.query(func.count(Participant.id))
        .join(TeeTime, TeeTime.id == Participant.tee_time_id)
        .filter(TeeTime.course_id == course.id, Participant.created_at >= month_start)
        .scalar()
    )
    p_paid_total = (
        db.query(func.count(Participant.id))
        .join(TeeTime, TeeTime.id == Participant.tee_time_id)
        .filter(TeeTime.course_id == course.id, Participant.paid.is_(True))
        .scalar()
    )

    revenue_cents = p_paid_total * settings.registration_price_cents

    # Clip counts (scoped to this course)
    clips_total = (
        db.query(func.count(VideoClip.id))
        .filter(VideoClip.course_id == course.id, VideoClip.processing_status == assigned)
        .scalar()
    )
    aces_total = (
        db.query(func.count(VideoClip.id))
        .filter(
            VideoClip.course_id == course.id,
            VideoClip.processing_status == assigned,
            VideoClip.ball_in_cup.is_(True),
        )
        .scalar()
    )
    aces_pending = (
        db.query(func.count(HoleInOneEvent.id))
        .join(Participant, Participant.id == HoleInOneEvent.participant_id)
        .join(TeeTime, TeeTime.id == Participant.tee_time_id)
        .filter(TeeTime.course_id == course.id, HoleInOneEvent.status == HIOStatus.pending.value)
        .scalar()
    )

    # Recent participants for at-a-glance
    recent = (
        db.query(Participant, TeeTime)
        .join(TeeTime, TeeTime.id == Participant.tee_time_id)
        .filter(TeeTime.course_id == course.id)
        .order_by(Participant.created_at.desc())
        .limit(10)
        .all()
    )

    return {
        "course": {"id": course.id, "name": course.name, "location": course.location},
        "stats": {
            "participants_today": p_today or 0,
            "participants_week": p_week or 0,
            "participants_month": p_month or 0,
            "participants_paid_total": p_paid_total or 0,
            "revenue_cents": revenue_cents,
            "clips_total": clips_total or 0,
            "aces_total": aces_total or 0,
            "aces_pending": aces_pending or 0,
        },
        "recent_participants": [
            {
                "id": p.id,
                "name": p.name,
                "paid": p.paid,
                "tee_time": tt.starts_at,
                "selfie_url": f"/uploads/{p.selfie_path}" if p.selfie_path else None,
                "gallery_url": f"{settings.app_base_url}/g/{p.gallery_token}",
            }
            for (p, tt) in recent
        ],
    }


@router.get("/participants")
def operator_participants(
    course: Course = Depends(current_operator),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(Participant, TeeTime)
        .join(TeeTime, TeeTime.id == Participant.tee_time_id)
        .filter(TeeTime.course_id == course.id)
        .order_by(TeeTime.starts_at.desc())
        .limit(200)
        .all()
    )
    return [
        {
            "id": p.id,
            "name": p.name,
            "mobile": p.mobile,
            "email": p.email,
            "paid": p.paid,
            "refunded_at": p.refunded_at,
            "tee_time": tt.starts_at,
            "gallery_url": f"{settings.app_base_url}/g/{p.gallery_token}",
            "selfie_url": f"/uploads/{p.selfie_path}" if p.selfie_path else None,
        }
        for (p, tt) in rows
    ]
