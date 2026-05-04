"""Routes that the mobile-web registration flow hits (no auth, token-gated)."""
from __future__ import annotations

import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..deps import optional_user
from ..models import Course, Participant, Showcase, TeeTime, User
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


@router.get("/courses", response_model=list[PublicCourseOut])
def list_public_courses(db: Session = Depends(get_db)):
    return db.query(Course).order_by(Course.name).all()


@router.get("/stripe-config")
def stripe_config():
    """Expose the publishable key + flag the frontend uses to decide whether
    to mount the real Stripe Elements (Apple Pay / Google Pay / card / Link)
    or fall through to the mock-paid happy path."""
    return {
        "publishable_key": settings.stripe_publishable_key or None,
        "configured": bool(settings.stripe_publishable_key and settings.stripe_secret_key),
        "price_cents": settings.registration_price_cents,
    }


@router.get("/showcase")
def list_showcase(db: Session = Depends(get_db)):
    rows = db.query(Showcase).order_by(Showcase.position.asc()).all()
    return [
        {
            "position": s.position,
            "source_url": s.source_url,
            "thumbnail_url": s.thumbnail_url,
            "title": s.title,
            "caption": s.caption,
        }
        for s in rows
    ]


@router.get("/courses/{course_token}", response_model=PublicCourseOut)
def course_by_token(course_token: str, db: Session = Depends(get_db)):
    return _get_course_by_token(db, course_token)


@router.get("/courses/{course_token}/qr.png")
def course_qr_png(course_token: str, db: Session = Depends(get_db)):
    course = _get_course_by_token(db, course_token)
    url = f"{settings.app_base_url}/r/{course.qr_token}"
    png = generate_qr_png(url)
    filename = f"golfreelz-{course.name.replace(' ', '_')}.png"
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
    group_members: str = Form("[]"),  # JSON string: [{name, email, mobile}, ...]
    selfie: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User | None = Depends(optional_user),
):
    # If logged in, default contact info to the user's account.
    if user:
        if not email.strip():
            email = user.email
        if not name.strip():
            name = user.name or user.email

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

    # Auto-claim by email even without a token (case-insensitive)
    auto_user_id = user.id if user else None
    if not auto_user_id and email.strip():
        match = db.query(User).filter(User.email == email.strip().lower()).first()
        if match:
            auto_user_id = match.id

    participant = Participant(
        tee_time_id=tt.id,
        user_id=auto_user_id,
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

    # Lead pays for the whole group: $20 per registered player.
    # Cap members count up-front so the charge matches the participants we'll
    # actually create below.
    try:
        _members_preview = json.loads(group_members or "[]")
    except json.JSONDecodeError:
        _members_preview = []
    spots_remaining = tt.max_players - 1  # lead already takes one
    member_count = min(
        len([m for m in _members_preview[:3]
             if (m.get("name") or "").strip()
             and ((m.get("email") or "").strip() or (m.get("mobile") or "").strip())]),
        spots_remaining,
    )
    total_players = 1 + member_count
    intent = create_registration_payment_intent(
        participant.id,
        amount_cents=settings.registration_price_cents * total_players,
    )
    participant.stripe_payment_intent_id = intent.id
    participant.paid = intent.status == "succeeded"

    gallery_url = f"{settings.app_base_url}/g/{participant.gallery_token}"

    if participant.paid:
        notifications.notify_registration_confirmed(
            participant.name, participant.mobile, participant.email, gallery_url
        )

    # Group members: lead pays for everyone; each invitee gets a link to add
    # their own selfie. They share the same tee time + are auto-paid.
    invited = []
    try:
        members = json.loads(group_members or "[]")
    except json.JSONDecodeError:
        members = []
    for m in members[:3]:  # cap at 3 additional players (4 total per group)
        m_name = (m.get("name") or "").strip()
        m_email = (m.get("email") or "").strip()
        m_mobile = (m.get("mobile") or "").strip()
        if not m_name or (not m_email and not m_mobile):
            continue
        spots = db.query(Participant).filter(Participant.tee_time_id == tt.id).count()
        if spots >= tt.max_players:
            break
        member = Participant(
            tee_time_id=tt.id,
            user_id=None,
            name=m_name,
            mobile=m_mobile or None,
            email=m_email or None,
            group_size=group_size,
            paid=True,  # lead's payment covers them
            stripe_payment_intent_id=f"group_via_{participant.id}",
        )
        db.add(member)
        db.flush()
        invite_url = f"{settings.app_base_url}/invite/{member.gallery_token}"
        # Reuse the existing 'registered' notification — copy adapted on the
        # invite page itself.
        notifications.send_email(
            m_email,
            f"You're registered for GolfReelz at {course.name}",
            (
                f"{name.strip()} signed you up for a round at {course.name}.\n\n"
                f"To get your shots matched, take a quick outfit photo:\n{invite_url}\n\n"
                f"You'll get an email with all your par-3 clips after the round."
            ),
        )
        notifications.send_sms(
            m_mobile,
            f"GolfReelz: {name.strip()} added you to a round at {course.name}. Snap an outfit photo: {invite_url}",
        )
        invited.append({"id": member.id, "name": m_name, "invite_url": invite_url})

    db.commit()

    return RegistrationResult(
        participant_id=participant.id,
        gallery_url=gallery_url,
        client_secret=intent.client_secret,
        paid=participant.paid,
    )


@router.get("/invite/{gallery_token}")
def invite_info(gallery_token: str, db: Session = Depends(get_db)):
    """Public details for an invited group member: course + tee time, plus
    whether they've already submitted their selfie."""
    p = db.query(Participant).filter(Participant.gallery_token == gallery_token).first()
    if not p:
        raise HTTPException(404, "invite not found")
    course = db.get(Course, p.tee_time.course_id) if p.tee_time else None
    return {
        "name": p.name,
        "mobile": p.mobile,
        "email": p.email,
        "tee_time": p.tee_time.starts_at if p.tee_time else None,
        "selfie_uploaded": bool(p.selfie_path),
        "course": {
            "name": course.name if course else "",
            "location": course.location if course else "",
            "qr_token": course.qr_token if course else None,
        },
    }


@router.post("/invite/{gallery_token}/selfie")
async def invite_selfie(
    gallery_token: str,
    selfie: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    p = db.query(Participant).filter(Participant.gallery_token == gallery_token).first()
    if not p:
        raise HTTPException(404, "invite not found")
    if not (selfie.content_type or "").startswith("image/"):
        raise HTTPException(400, "selfie must be an image")
    data = await selfie.read()
    if not data:
        raise HTTPException(400, "empty selfie")
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(413, "selfie too large (max 8MB)")

    p.appearance_embedding = appearance.embed_image_bytes(data)

    ext = (selfie.filename or "").rsplit(".", 1)[-1].lower() if "." in (selfie.filename or "") else "jpg"
    if ext not in ("jpg", "jpeg", "png", "webp", "heic"):
        ext = "jpg"
    fname = f"{p.id}-{secrets.token_hex(4)}.{ext}"
    fpath = SELFIE_DIR / fname
    fpath.write_bytes(data)
    p.selfie_path = f"selfies/{fname}"
    db.commit()

    return {"ok": True, "gallery_url": f"{settings.app_base_url}/g/{p.gallery_token}"}
