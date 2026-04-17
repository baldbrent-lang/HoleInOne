"""Routes that the mobile-web registration flow hits (no auth, token-gated)."""
from __future__ import annotations

import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import Course, Participant, TeeTime
from ..schemas import (
    PublicCourseOut,
    RegistrationResult,
    TeeTimeOut,
)
from ..services import appearance, notifications
from ..services.qr import generate_qr_png
from ..services.stripe_service import create_registration_payment_intent
from ..services.tee_sheet import list_available_tee_times

router = APIRouter(prefix="/api/public", tags=["public"])

UPLOAD_ROOT = Path(__file__).resolve().parents[2] / settings.upload_dir
SELFIE_DIR = UPLOAD_ROOT / "selfies"
SELFIE_DIR.mkdir(parents=True, exist_ok=True)


def _get_course_by_token(db: Session, token: str) -> Course:
    course = db.query(Course).filter(Course.qr_token == token).first()
    if not course:
        raise HTTPException(404, "course not found")
    return course


@router.get("/courses/{course_token}", response_model=PublicCourseOut)
def course_by_token(course_token: str, db: Session = Depends(get_db)):
    return _get_course_by_token(db, course_token)


@router.get("/courses/{course_token}/qr.png")
def course_qr_png(course_token: str, db: Session = Depends(get_db)):
    course = _get_course_by_token(db, course_token)
    url = f"{settings.app_base_url}/r/{course.qr_token}"
    png = generate_qr_png(url)
    filename = f"parone-{course.name.replace(' ', '_')}.png"
    return Response(
        content=png,
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


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
async def register(
    course_token: str = Form(...),
    tee_time_id: int = Form(...),
    name: str = Form(...),
    mobile: str = Form(""),
    email: str = Form(""),
    group_size: int = Form(4),
    selfie: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not (mobile.strip() or email.strip()):
        raise HTTPException(400, "mobile or email required")

    course = _get_course_by_token(db, course_token)
    tt = db.get(TeeTime, tee_time_id)
    if not tt or tt.course_id != course.id:
        raise HTTPException(400, "tee time does not belong to course")

    if not (selfie.content_type or "").startswith("image/"):
        raise HTTPException(400, "selfie must be an image")

    selfie_bytes = await selfie.read()
    if len(selfie_bytes) > 8 * 1024 * 1024:
        raise HTTPException(413, "selfie too large (max 8MB)")
    if not selfie_bytes:
        raise HTTPException(400, "empty selfie upload")

    spots_taken = db.query(Participant).filter(Participant.tee_time_id == tt.id).count()
    if spots_taken >= tt.max_players:
        raise HTTPException(409, "this tee time is full")

    embedding = appearance.embed_image_bytes(selfie_bytes)

    participant = Participant(
        tee_time_id=tt.id,
        name=name.strip(),
        mobile=mobile.strip() or None,
        email=email.strip() or None,
        group_size=group_size,
        appearance_embedding=embedding,
    )
    db.add(participant)
    db.flush()

    # Save selfie to disk now that we have the participant id
    ext = (selfie.filename or "").rsplit(".", 1)[-1].lower() if "." in (selfie.filename or "") else "jpg"
    if ext not in ("jpg", "jpeg", "png", "webp", "heic"):
        ext = "jpg"
    fname = f"{participant.id}-{secrets.token_hex(4)}.{ext}"
    fpath = SELFIE_DIR / fname
    fpath.write_bytes(selfie_bytes)
    participant.selfie_path = f"selfies/{fname}"

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
