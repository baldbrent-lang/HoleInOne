"""Routes that the mobile-web registration flow hits (no auth, token-gated)."""
from __future__ import annotations

import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..deps import optional_user
from ..models import ClipProcessingStatus, Course, HIOStatus, HoleInOneEvent, Participant, Showcase, TeeTime, User, VideoClip
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


@router.get("/stats")
def public_stats(db: Session = Depends(get_db)):
    """Live numbers shown on the Home page. Hide-on-zero is left to the
    frontend so an empty install still gets a meaningful payload."""
    from datetime import datetime, timedelta

    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    assigned = ClipProcessingStatus.assigned.value

    return {
        "clips_this_week": db.query(VideoClip).filter(
            VideoClip.created_at >= week_ago,
            VideoClip.processing_status == assigned,
        ).count(),
        "golfers_today": db.query(Participant).filter(
            Participant.created_at >= today_start,
        ).count(),
        "aces_pending": db.query(HoleInOneEvent).filter(
            HoleInOneEvent.status == HIOStatus.pending.value,
        ).count(),
        "courses_live": db.query(Course).count(),
        "total_clips_delivered": db.query(VideoClip).filter(
            VideoClip.processing_status == assigned,
        ).count(),
    }


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


def _short_name(full_name: str | None) -> str:
    parts = (full_name or "").strip().split()
    if not parts:
        return "—"
    first = parts[0]
    if len(parts) == 1:
        return first
    return f"{first} {parts[-1][0].upper()}."


@router.get("/leaderboards")
def public_leaderboards(limit: int = 10, db: Session = Depends(get_db)):
    """Top entries across the meaningful axes. limit=3 for the home preview,
    higher for the full page."""
    limit = max(1, min(50, limit))
    assigned = ClipProcessingStatus.assigned.value

    # Longest single-shot carry
    longest_carry_rows = (
        db.query(Participant.name, Course.name.label("course"), VideoClip.hole_number, VideoClip.carry_yards)
        .join(VideoClip, VideoClip.participant_id == Participant.id)
        .join(TeeTime, TeeTime.id == Participant.tee_time_id)
        .join(Course, Course.id == TeeTime.course_id)
        .filter(VideoClip.carry_yards.isnot(None), VideoClip.processing_status == assigned)
        .order_by(desc(VideoClip.carry_yards))
        .limit(limit)
        .all()
    )

    # Fastest single-shot ball speed
    fastest_ball_rows = (
        db.query(Participant.name, Course.name.label("course"), VideoClip.hole_number, VideoClip.ball_speed_mph)
        .join(VideoClip, VideoClip.participant_id == Participant.id)
        .join(TeeTime, TeeTime.id == Participant.tee_time_id)
        .join(Course, Course.id == TeeTime.course_id)
        .filter(VideoClip.ball_speed_mph.isnot(None), VideoClip.processing_status == assigned)
        .order_by(desc(VideoClip.ball_speed_mph))
        .limit(limit)
        .all()
    )

    # Most hole-in-ones (aces) — counted at the participant level
    most_aces_rows = (
        db.query(Participant.name, func.count(VideoClip.id).label("aces"))
        .join(VideoClip, VideoClip.participant_id == Participant.id)
        .filter(VideoClip.ball_in_cup.is_(True), VideoClip.processing_status == assigned)
        .group_by(Participant.id)
        .order_by(desc(func.count(VideoClip.id)))
        .limit(limit)
        .all()
    )

    # Most rounds played — counted across registered Users
    most_rounds_rows = (
        db.query(User.name, User.email, func.count(Participant.id).label("rounds"))
        .join(Participant, Participant.user_id == User.id)
        .group_by(User.id)
        .order_by(desc(func.count(Participant.id)))
        .limit(limit)
        .all()
    )

    return {
        "longest_carry": [
            {"golfer": _short_name(name), "course": course, "hole": hole, "value": yards, "unit": "yds"}
            for (name, course, hole, yards) in longest_carry_rows
        ],
        "fastest_ball": [
            {"golfer": _short_name(name), "course": course, "hole": hole, "value": mph, "unit": "mph"}
            for (name, course, hole, mph) in fastest_ball_rows
        ],
        "most_aces": [
            {"golfer": _short_name(name), "value": aces, "unit": "aces"}
            for (name, aces) in most_aces_rows
        ],
        "most_rounds": [
            {"golfer": _short_name(name or email.split("@")[0]), "value": rounds, "unit": "rounds"}
            for (name, email, rounds) in most_rounds_rows
        ],
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
    member_2_selfie: UploadFile | None = File(None),
    member_3_selfie: UploadFile | None = File(None),
    member_4_selfie: UploadFile | None = File(None),
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

    # Group members: lead pays for everyone AND captures every player's
    # outfit photo at registration so each gets a personal gallery from
    # the moment the round starts. The /invite/<token> path is kept around
    # as a fallback for back-fills, but the canonical flow requires a
    # selfie up-front for each member.
    member_selfie_files = [member_2_selfie, member_3_selfie, member_4_selfie]
    try:
        members = json.loads(group_members or "[]")
    except json.JSONDecodeError:
        members = []
    for idx, m in enumerate(members[:3]):
        m_name = (m.get("name") or "").strip()
        m_email = (m.get("email") or "").strip()
        m_mobile = (m.get("mobile") or "").strip()
        if not m_name or (not m_email and not m_mobile):
            continue
        spots = db.query(Participant).filter(Participant.tee_time_id == tt.id).count()
        if spots >= tt.max_players:
            break

        m_selfie = member_selfie_files[idx]
        m_selfie_bytes = None
        if m_selfie is not None:
            if not (m_selfie.content_type or "").startswith("image/"):
                raise HTTPException(400, f"player {idx + 2} selfie must be an image")
            m_selfie_bytes = await m_selfie.read()
            if len(m_selfie_bytes) > 8 * 1024 * 1024:
                raise HTTPException(413, f"player {idx + 2} selfie too large (max 8MB)")

        m_embedding = appearance.embed_image_bytes(m_selfie_bytes) if m_selfie_bytes else None

        member = Participant(
            tee_time_id=tt.id,
            user_id=None,
            name=m_name,
            mobile=m_mobile or None,
            email=m_email or None,
            group_size=group_size,
            appearance_embedding=m_embedding,
            paid=True,  # lead's payment covers them
            stripe_payment_intent_id=f"group_via_{participant.id}",
        )
        db.add(member)
        db.flush()

        if m_selfie_bytes:
            ext = (m_selfie.filename or "").rsplit(".", 1)[-1].lower() if "." in (m_selfie.filename or "") else "jpg"
            if ext not in ("jpg", "jpeg", "png", "webp", "heic"):
                ext = "jpg"
            m_fname = f"{member.id}-{secrets.token_hex(4)}.{ext}"
            m_fpath = SELFIE_DIR / m_fname
            m_fpath.write_bytes(m_selfie_bytes)
            member.selfie_path = f"selfies/{m_fname}"

            # Selfie captured up-front — send a confirmation email/SMS with
            # their gallery link instead of an "add your selfie" invite.
            gallery = f"{settings.app_base_url}/g/{member.gallery_token}"
            notifications.send_email(
                m_email,
                f"You're registered for GolfReelz at {course.name}",
                (
                    f"{name.strip()} signed you up for a round at {course.name}.\n\n"
                    f"You're all set — your outfit photo was captured at registration.\n"
                    f"We'll email your par-3 clips after the round.\n\n"
                    f"Bookmark your gallery: {gallery}"
                ),
            )
            notifications.send_sms(
                m_mobile,
                f"GolfReelz: {name.strip()} added you to a round at {course.name}. Your gallery: {gallery}",
            )
        else:
            # Fallback: no selfie at registration time. Send the legacy invite
            # link so they can add one later.
            invite_url = f"{settings.app_base_url}/invite/{member.gallery_token}"
            notifications.send_email(
                m_email,
                f"You're registered for GolfReelz at {course.name}",
                (
                    f"{name.strip()} signed you up for a round at {course.name}.\n\n"
                    f"One more step — take a quick outfit photo so we can match\n"
                    f"your shots: {invite_url}"
                ),
            )
            notifications.send_sms(
                m_mobile,
                f"GolfReelz: {name.strip()} added you to a round at {course.name}. Snap an outfit photo: {invite_url}",
            )

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
