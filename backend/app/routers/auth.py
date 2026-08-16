"""User accounts: signup, login, /me dashboard with round history."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..deps import current_user
from ..models import (
    ClipProcessingStatus,
    Course,
    Participant,
    TeeTime,
    User,
    VideoClip,
)
from ..services.auth import hash_password, issue_token, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


class SignupBody(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)
    name: str | None = None


class LoginBody(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: EmailStr
    name: str | None = None

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    token: str
    user: UserOut


def _claim_existing_participants(db: Session, user: User) -> int:
    """Link any historic Participants (registered with this email) to user."""
    rows = (
        db.query(Participant)
        .filter(Participant.email == user.email, Participant.user_id.is_(None))
        .all()
    )
    for p in rows:
        p.user_id = user.id
    if rows:
        db.commit()
    return len(rows)


@router.post("/signup", response_model=TokenResponse)
def signup(payload: SignupBody, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(409, "email already registered — try logging in")
    user = User(
        email=email,
        password_hash=hash_password(payload.password),
        name=(payload.name or "").strip() or None,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    _claim_existing_participants(db, user)
    return TokenResponse(token=issue_token(user.id), user=UserOut.model_validate(user))


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginBody, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(401, "wrong email or password")
    _claim_existing_participants(db, user)
    return TokenResponse(token=issue_token(user.id), user=UserOut.model_validate(user))


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return user


@router.get("/me/rounds")
def my_rounds(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Each Participant row is one round. Returns oldest-first by tee time
    grouped under each course."""
    rows = (
        db.query(Participant, TeeTime, Course)
        .join(TeeTime, Participant.tee_time_id == TeeTime.id)
        .join(Course, TeeTime.course_id == Course.id)
        .filter(Participant.user_id == user.id)
        .order_by(TeeTime.starts_at.desc())
        .all()
    )

    # Pre-fetch clip counts per participant in one query
    p_ids = [p.id for (p, _tt, _c) in rows]
    counts: dict[int, dict] = {pid: {"total": 0, "assigned": 0} for pid in p_ids}
    if p_ids:
        for pid, status_, n in (
            db.query(VideoClip.participant_id, VideoClip.processing_status, func.count(VideoClip.id))
            .filter(VideoClip.participant_id.in_(p_ids))
            .group_by(VideoClip.participant_id, VideoClip.processing_status)
            .all()
        ):
            counts[pid]["total"] += n
            if status_ == ClipProcessingStatus.assigned.value:
                counts[pid]["assigned"] += n

    # Group by course
    by_course: dict[int, dict] = {}
    for p, tt, course in rows:
        bucket = by_course.setdefault(
            course.id,
            {
                "course": {"id": course.id, "name": course.name, "location": course.location},
                "rounds": [],
            },
        )
        bucket["rounds"].append({
            "participant_id": p.id,
            "gallery_token": p.gallery_token,
            "gallery_url": f"{settings.app_base_url}/g/{p.gallery_token}",
            "tee_time": tt.starts_at,
            "paid": p.paid,
            "summary_sent_at": p.summary_sent_at,
            "clips": counts.get(p.id, {"total": 0, "assigned": 0}),
            "selfie_url": f"/uploads/{p.selfie_path}" if p.selfie_path else None,
        })
    return list(by_course.values())


@router.get("/me/rounds/{participant_id}/clips")
def my_round_clips(
    participant_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    p = db.get(Participant, participant_id)
    if not p or p.user_id != user.id:
        raise HTTPException(404, "round not found")
    course = db.get(Course, p.tee_time.course_id) if p.tee_time else None
    clips = (
        db.query(VideoClip)
        .filter(
            VideoClip.participant_id == p.id,
            VideoClip.processing_status == ClipProcessingStatus.assigned.value,
        )
        .order_by(VideoClip.hole_number.asc(), VideoClip.captured_at.asc())
        .all()
    )
    return {
        "course": {
            "name": course.name if course else "",
            "hole_yardages": (course.hole_yardages or {}) if course else {},
        },
        "participant": {"id": p.id, "name": p.name},
        "clips": [
            {
                "id": c.id,
                "hole_number": c.hole_number,
                "camera_type": c.camera_type,
                "captured_at": c.captured_at,
                "source_url": c.source_url,
                "thumbnail_url": c.thumbnail_url,
                "ball_in_cup": c.ball_in_cup,
            }
            for c in clips
        ],
    }
