"""Routes that the mobile-web registration flow hits (no auth, token-gated)."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Course, Participant, TeeTime
from ..schemas import (
    PublicCourseOut,
    RegistrationCreate,
    RegistrationResult,
    TeeTimeOut,
)
from ..services import notifications
from ..services.stripe_service import create_registration_payment_intent
from ..services.tee_sheet import list_available_tee_times
from ..config import settings

router = APIRouter(prefix="/api/public", tags=["public"])


def _get_course_by_token(db: Session, token: str) -> Course:
    course = db.query(Course).filter(Course.qr_token == token).first()
    if not course:
        raise HTTPException(404, "course not found")
    return course


@router.get("/courses/{course_token}", response_model=PublicCourseOut)
def course_by_token(course_token: str, db: Session = Depends(get_db)):
    return _get_course_by_token(db, course_token)


@router.get("/courses/{course_token}/tee-times", response_model=list[TeeTimeOut])
def available_tee_times(
    course_token: str,
    date: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    db: Session = Depends(get_db),
):
    course = _get_course_by_token(db, course_token)
    on_date = datetime.fromisoformat(date) if date else datetime.utcnow()
    tts = list_available_tee_times(db, course, on_date)
    db.commit()
    out: list[TeeTimeOut] = []
    for tt in tts:
        spots = db.query(Participant).filter(Participant.tee_time_id == tt.id).count()
        out.append(
            TeeTimeOut(
                id=tt.id,
                starts_at=tt.starts_at,
                max_players=tt.max_players,
                spots_taken=spots,
            )
        )
    return out


@router.post("/register", response_model=RegistrationResult)
def register(payload: RegistrationCreate, db: Session = Depends(get_db)):
    course = _get_course_by_token(db, payload.course_token)
    tt = db.get(TeeTime, payload.tee_time_id)
    if not tt or tt.course_id != course.id:
        raise HTTPException(400, "tee time does not belong to course")

    existing = (
        db.query(Participant)
        .filter(
            Participant.tee_time_id == tt.id,
            Participant.playing_order == payload.playing_order,
        )
        .first()
    )
    if existing:
        raise HTTPException(409, "that playing order is already taken for this tee time")

    participant = Participant(
        tee_time_id=tt.id,
        name=payload.name,
        mobile=payload.mobile,
        email=payload.email,
        playing_order=payload.playing_order,
        group_size=payload.group_size,
    )
    db.add(participant)
    db.flush()

    intent = create_registration_payment_intent(participant.id)
    participant.stripe_payment_intent_id = intent.id
    participant.paid = intent.status == "succeeded"

    gallery_url = f"{settings.app_base_url}/g/{participant.gallery_token}"

    if participant.paid:
        notifications.notify_registration_confirmed(
            participant.name, participant.mobile, participant.email, gallery_url
        )

    db.commit()

    return RegistrationResult(
        participant_id=participant.id,
        gallery_url=gallery_url,
        client_secret=intent.client_secret,
        paid=participant.paid,
    )
