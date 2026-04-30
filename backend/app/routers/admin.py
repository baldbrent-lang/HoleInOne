from __future__ import annotations

import secrets
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
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
    Showcase,
    TeeTime,
    VideoClip,
)
from ..schemas import (
    CourseCreate,
    CourseOut,
    CourseUpdate,
    HIOEventOut,
    HIOReviewAction,
)
from ..services import notifications
from ..services.matcher import match_clip
from ..services.qr import generate_qr_png
from ..services.video import compress_for_email, extract_thumbnail

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
        hole_yardages=payload.hole_yardages or {},
        minutes_per_hole=payload.minutes_per_hole,
        tee_sheet_provider=payload.tee_sheet_provider,
        tee_sheet_config=payload.tee_sheet_config,
        livestream_url=payload.livestream_url,
    )
    db.add(course)
    db.commit()
    db.refresh(course)
    return course


@router.patch("/courses/{course_id}", response_model=CourseOut)
def update_course(course_id: int, payload: CourseUpdate, db: Session = Depends(get_db)):
    course = db.get(Course, course_id)
    if not course:
        raise HTTPException(404, "course not found")
    data = payload.model_dump(exclude_unset=True)
    for field, value in data.items():
        setattr(course, field, value)
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


@router.get("/participants")
def list_participants(
    course_id: int | None = None,
    date: str | None = None,  # YYYY-MM-DD (filters on tee_time.starts_at date)
    q: str | None = None,  # substring on name/mobile/email
    paid: bool | None = None,
    limit: int = 200,
    db: Session = Depends(get_db),
):
    from datetime import date as date_cls, datetime as datetime_cls, timedelta

    query = (
        db.query(Participant, TeeTime, Course)
        .join(TeeTime, Participant.tee_time_id == TeeTime.id)
        .join(Course, TeeTime.course_id == Course.id)
    )
    if course_id is not None:
        query = query.filter(Course.id == course_id)
    if paid is not None:
        query = query.filter(Participant.paid == paid)
    if date:
        try:
            d = date_cls.fromisoformat(date)
        except ValueError:
            raise HTTPException(400, "date must be YYYY-MM-DD")
        lo = datetime_cls.combine(d, datetime_cls.min.time())
        hi = lo + timedelta(days=1)
        query = query.filter(TeeTime.starts_at >= lo, TeeTime.starts_at < hi)
    if q:
        needle = f"%{q.strip()}%"
        query = query.filter(
            (Participant.name.ilike(needle))
            | (Participant.mobile.ilike(needle))
            | (Participant.email.ilike(needle))
        )

    rows = query.order_by(TeeTime.starts_at.desc(), Participant.id.desc()).limit(limit).all()

    # Pre-fetch clip counts
    ids = [p.id for (p, _tt, _c) in rows]
    clip_counts: dict[int, dict] = {i: {"total": 0, "assigned": 0} for i in ids}
    if ids:
        assigned_status = ClipProcessingStatus.assigned.value
        counts = (
            db.query(VideoClip.participant_id, VideoClip.processing_status, func.count(VideoClip.id))
            .filter(VideoClip.participant_id.in_(ids))
            .group_by(VideoClip.participant_id, VideoClip.processing_status)
            .all()
        )
        for pid, status_, n in counts:
            bucket = clip_counts[pid]
            bucket["total"] += n
            if status_ == assigned_status:
                bucket["assigned"] += n

    return [
        {
            "id": p.id,
            "name": p.name,
            "mobile": p.mobile,
            "email": p.email,
            "paid": p.paid,
            "selfie_url": f"/uploads/{p.selfie_path}" if p.selfie_path else None,
            "gallery_token": p.gallery_token,
            "gallery_url": f"{settings.app_base_url}/g/{p.gallery_token}",
            "course": {"id": course.id, "name": course.name, "location": course.location},
            "tee_time": {"id": tt.id, "starts_at": tt.starts_at},
            "clips": clip_counts.get(p.id, {"total": 0, "assigned": 0}),
            "created_at": p.created_at,
        }
        for (p, tt, course) in rows
    ]


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
        "selfie_url": f"/uploads/{p.selfie_path}" if p.selfie_path else None,
        "gallery_url": f"{settings.app_base_url}/g/{p.gallery_token}",
        "tee_time": {
            "id": p.tee_time.id,
            "starts_at": p.tee_time.starts_at,
            "course_id": p.tee_time.course_id,
        },
        "clips": len(p.clips),
    }


@router.get("/participants/{participant_id}/clips")
def participant_clips(participant_id: int, db: Session = Depends(get_db)):
    p = db.get(Participant, participant_id)
    if not p:
        raise HTTPException(404, "participant not found")
    course = db.get(Course, p.tee_time.course_id) if p.tee_time else None
    clips = (
        db.query(VideoClip)
        .filter(VideoClip.participant_id == p.id)
        .order_by(VideoClip.hole_number.asc(), VideoClip.captured_at.asc())
        .all()
    )
    return {
        "participant": {"id": p.id, "name": p.name},
        "course": {
            "id": course.id if course else None,
            "name": course.name if course else "",
            "hole_yardages": (course.hole_yardages or {}) if course else {},
        },
        "clips": [
            {
                "id": c.id,
                "hole_number": c.hole_number,
                "camera_type": c.camera_type,
                "captured_at": c.captured_at,
                "source_url": c.source_url,
                "thumbnail_url": c.thumbnail_url,
                "carry_yards": c.carry_yards,
                "apex_feet": c.apex_feet,
                "ball_speed_mph": c.ball_speed_mph,
                "processing_status": c.processing_status,
                "ball_in_cup": c.ball_in_cup,
            }
            for c in clips
        ],
    }


@router.post("/test-email")
def send_test_email(payload: dict, db: Session = Depends(get_db)):
    """Fire a single test email to confirm SMTP / SendGrid wiring works.

    Body: {"to": "you@example.com"} or {"participant_id": 5}
    """
    to = (payload or {}).get("to")
    if not to and payload.get("participant_id"):
        p = db.get(Participant, int(payload["participant_id"]))
        to = p.email if p else None
    if not to:
        raise HTTPException(400, "provide 'to' or 'participant_id' (with email)")

    provider = (
        "smtp" if (settings.smtp_host and settings.smtp_user and settings.smtp_password)
        else "sendgrid" if settings.sendgrid_api_key
        else "mock"
    )

    try:
        notifications.send_email(
            to,
            "GolfReelz test email",
            "If you can read this, your email wiring is working. — GolfReelz",
        )
    except Exception as exc:  # surface SMTP errors back to the admin
        raise HTTPException(502, f"send failed via {provider}: {exc}")

    return {"ok": True, "provider": provider, "to": to}


@router.post("/participants/{participant_id}/send-summary")
def send_round_summary(participant_id: int, force: bool = False, db: Session = Depends(get_db)):
    """Manually fire the round-summary email for a participant.

    Use force=true to resend even if summary_sent_at is already set.
    """
    p = db.get(Participant, participant_id)
    if not p:
        raise HTTPException(404, "participant not found")
    course = db.get(Course, p.tee_time.course_id) if p.tee_time else None
    if not course:
        raise HTTPException(404, "course not found")
    if force:
        p.summary_sent_at = None
    sent = notifications.maybe_send_round_summary(db, p, course)
    db.commit()
    return {
        "sent": sent,
        "summary_sent_at": p.summary_sent_at,
        "to": p.email,
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
    out = []
    for c in clips:
        # Surface selfies of candidate participants in this hole/window so
        # the reviewer can eyeball the right assignment.
        candidates = []
        if c.course_id:
            from datetime import timedelta
            course = db.get(Course, c.course_id)
            if course:
                mph = course.minutes_per_hole or 14
                lo = c.captured_at - timedelta(minutes=mph * 19)
                hi = c.captured_at
                from ..models import TeeTime as _TT
                tts = (
                    db.query(_TT)
                    .filter(
                        _TT.course_id == course.id,
                        _TT.starts_at <= hi,
                        _TT.starts_at >= lo,
                    )
                    .all()
                )
                for tt in tts:
                    for p in tt.participants:
                        candidates.append(
                            {
                                "id": p.id,
                                "name": p.name,
                                "selfie_url": f"/uploads/{p.selfie_path}" if p.selfie_path else None,
                            }
                        )
        out.append(
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
                "thumbnail_url": c.thumbnail_url,
                "candidates": candidates,
            }
        )
    return out


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

    course = db.get(Course, clip.course_id)
    notifications.maybe_send_round_summary(db, participant, course)

    db.commit()
    return {"ok": True, "summary_sent": participant.summary_sent_at is not None}


# --- Manual clip upload (proxy for Shot Tracer webhook in V0) ---------------

CLIPS_DIR = Path(__file__).resolve().parents[2] / settings.upload_dir / "clips"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/clips/upload")
async def upload_clip(
    course_id: int = Form(...),
    hole_number: int = Form(...),
    camera_type: str = Form("tee"),
    captured_at: str = Form(...),  # ISO datetime, e.g. "2026-04-17T14:32:11Z"
    carry_yards: int | None = Form(None),
    apex_feet: int | None = Form(None),
    ball_speed_mph: int | None = Form(None),
    ball_in_cup: bool = Form(False),
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Manual upload that mimics the Shot Tracer webhook.

    Saves the video to backend/uploads/clips/, creates a VideoClip row,
    runs the appearance matcher, and fires gallery-ready notifications
    just like the real webhook would. Useful for testing the pipeline
    end-to-end with phone or GoPro footage before any real cameras are
    deployed.
    """
    course = db.get(Course, course_id)
    if not course:
        raise HTTPException(404, "course not found")

    if not (video.content_type or "").startswith("video/"):
        raise HTTPException(400, "must be a video file")

    data = await video.read()
    if not data:
        raise HTTPException(400, "empty video upload")
    if len(data) > 500 * 1024 * 1024:
        raise HTTPException(413, "video too large (max 500MB)")

    ext = (video.filename or "").rsplit(".", 1)[-1].lower() if "." in (video.filename or "") else "mp4"
    if ext not in ("mp4", "mov", "webm", "m4v"):
        ext = "mp4"
    fname = f"{course_id}-h{hole_number}-{secrets.token_hex(6)}.{ext}"
    fpath = CLIPS_DIR / fname
    fpath.write_bytes(data)

    # Transcode for email delivery. Replaces the file in place, leaving
    # the same source_url valid. Falls back to the original on ffmpeg
    # failure so the gallery still works.
    compress_for_email(fpath)

    # Extract a JPG of the first frame so the video player has a poster
    # that matches the clip's opening — no black box before pressing play.
    thumb_path = extract_thumbnail(fpath)
    thumb_url = (
        f"{settings.app_base_url}/uploads/clips/{thumb_path.name}"
        if thumb_path else None
    )

    try:
        captured_dt = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        if captured_dt.tzinfo is not None:
            # Normalize to naive UTC since the column is TIMESTAMP WITHOUT TIME ZONE
            captured_dt = captured_dt.astimezone().replace(tzinfo=None)
    except ValueError:
        raise HTTPException(400, "invalid captured_at; use ISO 8601")

    clip = VideoClip(
        course_id=course_id,
        hole_number=hole_number,
        camera_type=camera_type,
        captured_at=captured_dt,
        source_url=f"{settings.app_base_url}/uploads/clips/{fname}",
        thumbnail_url=thumb_url,
        carry_yards=carry_yards,
        apex_feet=apex_feet,
        ball_speed_mph=ball_speed_mph,
        ball_in_cup=ball_in_cup,
        processing_status=ClipProcessingStatus.received.value,
    )
    db.add(clip)
    db.flush()

    participant = match_clip(db, clip)
    if participant and ball_in_cup:
        notifications.notify_hio_under_review(participant.name, participant.mobile, participant.email)

    db.commit()

    # Fire gallery-ready notification on first assigned clip
    if clip.participant_id and clip.processing_status == ClipProcessingStatus.assigned.value:
        p = db.get(Participant, clip.participant_id)
        if p and not p.gallery_ready_sent:
            p.gallery_ready_sent = True
            gallery_url = f"{settings.app_base_url}/g/{p.gallery_token}"
            notifications.notify_gallery_ready(p.name, p.mobile, p.email, gallery_url)
            db.commit()

    return {
        "clip_id": clip.id,
        "status": clip.processing_status,
        "participant_id": clip.participant_id,
        "participant_name": participant.name if participant else None,
        "source_url": clip.source_url,
        "issue_note": clip.issue_note,
    }


# --- Showcase (Home page 'Our videos in action') ----------------------------

SHOWCASE_DIR = Path(__file__).resolve().parents[2] / settings.upload_dir / "showcase"
SHOWCASE_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/showcase")
def list_showcase_admin(db: Session = Depends(get_db)):
    rows = db.query(Showcase).order_by(Showcase.position.asc()).all()
    return [
        {
            "position": s.position,
            "source_url": s.source_url,
            "thumbnail_url": s.thumbnail_url,
            "title": s.title,
            "caption": s.caption,
            "updated_at": s.updated_at,
        }
        for s in rows
    ]


@router.patch("/showcase/{position}")
def update_showcase(position: int, payload: dict, db: Session = Depends(get_db)):
    s = db.query(Showcase).filter(Showcase.position == position).first()
    if not s:
        raise HTTPException(404, "slot not found")
    for field in ("source_url", "thumbnail_url", "title", "caption"):
        if field in payload:
            setattr(s, field, payload[field])
    s.updated_at = datetime.utcnow()
    db.commit()
    return {"ok": True}


@router.post("/showcase/{position}/upload")
async def upload_showcase(
    position: int,
    video: UploadFile = File(...),
    title: str = Form(""),
    caption: str = Form(""),
    db: Session = Depends(get_db),
):
    s = db.query(Showcase).filter(Showcase.position == position).first()
    if not s:
        raise HTTPException(404, "slot not found")
    if not (video.content_type or "").startswith("video/"):
        raise HTTPException(400, "must be a video file")

    data = await video.read()
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > 500 * 1024 * 1024:
        raise HTTPException(413, "video too large (max 500MB)")

    ext = (video.filename or "").rsplit(".", 1)[-1].lower() if "." in (video.filename or "") else "mp4"
    if ext not in ("mp4", "mov", "webm", "m4v"):
        ext = "mp4"
    fname = f"slot{position}-{secrets.token_hex(6)}.{ext}"
    fpath = SHOWCASE_DIR / fname
    fpath.write_bytes(data)

    # Compress + extract first-frame thumbnail (same pipeline as clip uploads)
    compress_for_email(fpath)
    thumb = extract_thumbnail(fpath)

    s.source_url = f"{settings.app_base_url}/uploads/showcase/{fname}"
    s.thumbnail_url = (
        f"{settings.app_base_url}/uploads/showcase/{thumb.name}" if thumb else None
    )
    if title.strip():
        s.title = title.strip()
    if caption.strip():
        s.caption = caption.strip()
    s.updated_at = datetime.utcnow()
    db.commit()
    return {
        "ok": True,
        "source_url": s.source_url,
        "thumbnail_url": s.thumbnail_url,
        "title": s.title,
        "caption": s.caption,
    }


@router.delete("/showcase/{position}")
def clear_showcase(position: int, db: Session = Depends(get_db)):
    s = db.query(Showcase).filter(Showcase.position == position).first()
    if not s:
        raise HTTPException(404, "slot not found")
    s.source_url = None
    s.thumbnail_url = None
    s.title = None
    s.caption = None
    s.updated_at = datetime.utcnow()
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
