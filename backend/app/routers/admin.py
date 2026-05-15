from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger("golfreelz.admin")

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Response, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..config import settings
from ..database import SessionLocal, get_db
from ..deps import require_admin
from ..models import (
    AuditLog,
    Camera,
    CameraEvent,
    ClipProcessingStatus,
    Course,
    HIOStatus,
    HoleInOneEvent,
    LongVideoUpload,
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
from ..services.auth import hash_password
from ..services.stripe_service import refund_payment_intent
from ..services.tracer import have_tracer, render_tracer
from ..services.ai_tracer import (
    find_address_frame,
    detect_handedness_at_address,
    annotate_address_with_shaft,
    find_impact_frame_after_address,
    find_impact_via_audio,
    refine_impact_frame,
    track_ball_after_impact,
    render_tracer_video,
    run_full_ai_tracer_pipeline,
    detect_swings_from_audio,
    detect_swings_from_motion,
    detect_swings_combined,
)
from ..services.video import compress_for_email, concat_two_clips, cut_segment, extract_thumbnail, probe_fps, probe_source_device, probe_video_info
from ..services.intro_overlay import apply_intro_overlay_inplace

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


@router.post("/courses/{course_id}/operator-password")
def set_operator_password(course_id: int, payload: dict, db: Session = Depends(get_db)):
    """Set or clear the per-course operator portal password. Pass an empty
    string to disable operator login for that course."""
    course = db.get(Course, course_id)
    if not course:
        raise HTTPException(404, "course not found")
    pw = (payload or {}).get("password", "")
    if not pw:
        course.operator_password_hash = None
    else:
        if len(pw) < 6:
            raise HTTPException(400, "operator password must be at least 6 characters")
        course.operator_password_hash = hash_password(pw)
    db.commit()
    return {"ok": True, "configured": course.operator_password_hash is not None}


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
            "refunded_at": p.refunded_at,
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


@router.post("/participants/{participant_id}/refund")
def refund_participant(participant_id: int, db: Session = Depends(get_db)):
    p = db.get(Participant, participant_id)
    if not p:
        raise HTTPException(404, "participant not found")
    if p.refunded_at:
        return {"ok": True, "already_refunded": True, "refunded_at": p.refunded_at}
    if not p.paid:
        raise HTTPException(400, "participant was not paid")

    result = refund_payment_intent(p.stripe_payment_intent_id)
    if not result.get("ok"):
        raise HTTPException(502, f"refund failed: {result.get('error', 'unknown')}")

    p.refunded_at = datetime.utcnow()
    p.paid = False
    db.add(AuditLog(
        actor="admin", action="refund",
        target=f"participant:{p.id}",
        detail=f"mode={result.get('mode')} refund_id={result.get('refund_id')}",
    ))
    db.commit()
    return {"ok": True, "mode": result.get("mode"), "refund_id": result.get("refund_id")}


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


@router.get("/clips")
def list_all_clips(limit: int = 100, db: Session = Depends(get_db)):
    """All clips, newest first. Includes orphans (no participant_id).

    Powers the /admin/clips test/iteration page where we can rerun the
    tracer on any existing clip without re-uploading.
    """
    clips = (
        db.query(VideoClip)
        .order_by(VideoClip.created_at.desc())
        .limit(max(1, min(500, limit)))
        .all()
    )
    course_ids = {c.course_id for c in clips}
    participant_ids = {c.participant_id for c in clips if c.participant_id}
    courses = {c.id: c for c in db.query(Course).filter(Course.id.in_(course_ids)).all()} if course_ids else {}
    participants = (
        {p.id: p for p in db.query(Participant).filter(Participant.id.in_(participant_ids)).all()}
        if participant_ids else {}
    )
    out = []
    for c in clips:
        course = courses.get(c.course_id)
        participant = participants.get(c.participant_id) if c.participant_id else None
        # Lightweight header read of each source file to surface FPS and
        # the recording device (when discoverable). cv2 for FPS (no
        # subprocess), one ffprobe call for the device tags.
        fps_val: float | None = None
        source_device: str | None = None
        if c.source_url:
            fname = c.source_url.rstrip("/").rsplit("/", 1)[-1]
            if fname:
                source_path = CLIPS_DIR / fname
                if source_path.exists():
                    fps_val = probe_fps(source_path)
                    source_device = probe_source_device(source_path)
        out.append({
            "id": c.id,
            "course_id": c.course_id,
            "course_name": course.name if course else None,
            "hole_number": c.hole_number,
            "camera_type": c.camera_type,
            "captured_at": c.captured_at.isoformat() if c.captured_at else None,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "source_url": c.source_url,
            "tracer_url": c.tracer_url,
            "tee_clip_url": c.tee_clip_url,
            "thumbnail_url": c.thumbnail_url,
            "ball_in_cup": bool(c.ball_in_cup),
            "is_highlight": bool(c.is_highlight),
            "processing_status": c.processing_status,
            "participant_id": c.participant_id,
            "participant_name": participant.name if participant else None,
            "fps": round(fps_val, 1) if fps_val is not None else None,
            "source_device": source_device,
        })
    return out


@router.post("/clips/{clip_id}/broadcast")
def toggle_clip_broadcast(
    clip_id: int,
    payload: dict = Body(default={}),
    db: Session = Depends(get_db),
):
    """Mark a produced clip as eligible for the Broadcast channel by
    setting is_highlight. Optional body `{"broadcast": false}` clears
    the flag. With no body the call toggles the current state.
    """
    clip = db.get(VideoClip, clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    if "broadcast" in payload:
        clip.is_highlight = bool(payload["broadcast"])
    else:
        clip.is_highlight = not bool(clip.is_highlight)
    db.add(clip)
    db.add(AuditLog(
        actor="admin", action="toggle_clip_broadcast",
        target=f"clip:{clip.id}",
        detail=f"is_highlight={bool(clip.is_highlight)}",
    ))
    db.commit()
    db.refresh(clip)
    return {"clip_id": clip.id, "is_highlight": bool(clip.is_highlight)}


@router.get("/broadcast-clips")
def list_broadcast_clips(limit: int = 100, db: Session = Depends(get_db)):
    """List clips on the Broadcast channel — any clip an operator has
    flagged with is_highlight=True, plus the legacy dual-camera
    composites (source filename contains '_composite') so older
    long-upload runs keep showing up without a backfill. Newest
    first. Same per-clip shape as /clips so the existing FPS /
    source-device header line works without modification.
    """
    from sqlalchemy import or_  # local import — keeps the module
                                # header lean
    clips = (
        db.query(VideoClip)
        .filter(or_(
            VideoClip.is_highlight.is_(True),
            VideoClip.source_url.like("%_composite%"),
        ))
        .order_by(VideoClip.created_at.desc())
        .limit(max(1, min(500, limit)))
        .all()
    )
    course_ids = {c.course_id for c in clips}
    participant_ids = {c.participant_id for c in clips if c.participant_id}
    courses = {c.id: c for c in db.query(Course).filter(Course.id.in_(course_ids)).all()} if course_ids else {}
    participants = (
        {p.id: p for p in db.query(Participant).filter(Participant.id.in_(participant_ids)).all()}
        if participant_ids else {}
    )
    out = []
    for c in clips:
        course = courses.get(c.course_id)
        participant = participants.get(c.participant_id) if c.participant_id else None
        fps_val: float | None = None
        source_device: str | None = None
        if c.source_url:
            fname = c.source_url.rstrip("/").rsplit("/", 1)[-1]
            if fname:
                source_path = CLIPS_DIR / fname
                if source_path.exists():
                    fps_val = probe_fps(source_path)
                    source_device = probe_source_device(source_path)
        out.append({
            "id": c.id,
            "course_id": c.course_id,
            "course_name": course.name if course else None,
            "hole_number": c.hole_number,
            "camera_type": c.camera_type,
            "captured_at": c.captured_at.isoformat() if c.captured_at else None,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "source_url": c.source_url,
            "tracer_url": c.tracer_url,
            "thumbnail_url": c.thumbnail_url,
            "ball_in_cup": bool(c.ball_in_cup),
            "is_highlight": bool(c.is_highlight),
            "processing_status": c.processing_status,
            "participant_id": c.participant_id,
            "participant_name": participant.name if participant else None,
            "fps": round(fps_val, 1) if fps_val is not None else None,
            "source_device": source_device,
        })
    return out


@router.post("/clips/{clip_id}/retry-tracer")
def retry_tracer(
    clip_id: int,
    sensitivity: float = Form(1.0),
    db: Session = Depends(get_db),
):
    """Re-run the classical-CV tracer on an existing clip's source file.

    For iteration: lets us tune detector thresholds and re-render the
    overlay without uploading new footage. Updates clip.tracer_url on
    success. Returns the same info shape as /clips/upload so the
    AdminClips UI can render the new result inline.

    Doesn't work on dual-camera composite outputs (we'd need both raw
    halves and the composite logic, which is more than this V1 covers).
    """
    clip = db.get(VideoClip, clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    # Prefer the raw tee-only cut for composite clips — source_url for
    # dual-cam clips points at a composite that the classical CV tracer
    # can't make sense of (tracer overlay baked in + concatenated green
    # half).
    tracer_input_url = clip.tee_clip_url or clip.source_url
    if not tracer_input_url:
        raise HTTPException(400, "clip has no tracer input URL")
    # Pull the file name from the URL — we stored it as
    # {base}/uploads/clips/{fname}. Use the URL's last segment.
    fname = tracer_input_url.rstrip("/").rsplit("/", 1)[-1]
    if not fname:
        raise HTTPException(400, "could not parse filename from tracer input URL")
    if "_composite" in fname:
        raise HTTPException(
            400,
            "this clip is a composite and has no raw tee cut on file; "
            "re-process the long upload to populate tee_clip_url",
        )
    fpath = CLIPS_DIR / fname
    if not fpath.exists():
        raise HTTPException(404, f"source file missing on disk: {fname}")
    tracer_url, tracer_info, _, tracer_debug_url = _run_tracer(fpath, sensitivity=float(sensitivity))
    clip.tracer_url = tracer_url
    db.add(AuditLog(actor="admin", action="retry_tracer", target=f"clip:{clip.id}",
                    detail=str(tracer_info)))
    db.commit()
    return {
        "clip_id": clip.id,
        "source_url": clip.source_url,
        "tracer_url": clip.tracer_url,
        "tracer_info": tracer_info,
        "tracer_debug_url": tracer_debug_url,
    }


@router.post("/clips/{clip_id}/audio-impact-frame")
def audio_impact_frame(
    clip_id: int,
    min_ratio: float = Form(25.0),
    db: Session = Depends(get_db),
):
    """Run the audio impact detector on a clip and return a JPG of the
    frame it picked. Test endpoint: lets the operator verify the audio
    pipeline can replace the AI impact-pick / refine-impact steps
    before we wire it into the production AI tracer flow.

    Uses tee_clip_url for composites (same fallback as /ai-trace) so
    audio is read from a single-camera clip.
    """
    try:
        import cv2  # local import: keeps admin.py importable even on
                   # boxes where opencv-python isn't installed.
    except ImportError:
        raise HTTPException(500, "opencv required for frame grab")

    clip = db.get(VideoClip, clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    analysis_url = clip.tee_clip_url or clip.source_url
    if not analysis_url:
        raise HTTPException(400, "clip has no analyzable source")
    fname = analysis_url.rstrip("/").rsplit("/", 1)[-1]
    if not fname:
        raise HTTPException(400, "could not parse filename from analysis URL")
    if "_composite" in fname:
        raise HTTPException(
            400,
            "this clip is a composite and has no raw tee cut on file; "
            "re-process the long upload to populate tee_clip_url",
        )
    fpath = CLIPS_DIR / fname
    if not fpath.exists():
        raise HTTPException(404, f"source file missing on disk: {fname}")

    fps_val = probe_fps(fpath) or 30.0
    info = find_impact_via_audio(fpath, fps_val, min_ratio=float(min_ratio))
    if not info.get("ok"):
        return {
            "ok": False,
            "error": info.get("error") or "no impact found",
            "ratio": info.get("ratio"),
            "min_ratio": info.get("min_ratio_used", float(min_ratio)),
            "audio": info,
        }

    frame_idx = info.get("impact_frame")
    if frame_idx is None:
        raise HTTPException(500, "audio impact returned no frame_idx")

    # Derive the address frame using the same 1.5s-before-impact rule
    # the production pipeline uses, then grab both frames.
    address_offset_frames = int(round(1.5 * fps_val))
    address_idx = max(0, int(frame_idx) - address_offset_frames)

    cap = cv2.VideoCapture(str(fpath))
    if not cap.isOpened():
        raise HTTPException(500, "could not open source for frame grab")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok_impact, impact_frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(address_idx))
        ok_address, address_frame = cap.read()
    finally:
        cap.release()
    if not ok_impact or impact_frame is None:
        raise HTTPException(500, f"could not read frame {frame_idx}")

    impact_name = f"{fpath.stem}_audio_impact.jpg"
    impact_out = CLIPS_DIR / impact_name
    cv2.imwrite(str(impact_out), impact_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    impact_mtime = int(impact_out.stat().st_mtime)

    address_url: str | None = None
    if ok_address and address_frame is not None:
        address_name = f"{fpath.stem}_audio_address.jpg"
        address_out = CLIPS_DIR / address_name
        cv2.imwrite(str(address_out), address_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        address_mtime = int(address_out.stat().st_mtime)
        address_url = f"{settings.app_base_url}/uploads/clips/{address_name}?v={address_mtime}"

    return {
        "ok": True,
        "clip_id": clip.id,
        "impact_frame": int(frame_idx),
        "address_frame": int(address_idx),
        "address_offset_frames": address_offset_frames,
        "address_offset_sec": 1.5,
        "audio_peak_frame": info.get("audio_peak_frame"),
        "pre_peak_offset_frames": info.get("pre_peak_offset_frames"),
        "peak_time_sec": info.get("peak_time_sec"),
        "ratio": info.get("ratio"),
        "min_ratio": info.get("min_ratio_used"),
        "highpass_hz": info.get("highpass_hz"),
        "fps": fps_val,
        "image_url": f"{settings.app_base_url}/uploads/clips/{impact_name}?v={impact_mtime}",
        "address_image_url": address_url,
    }


@router.post("/clips/{clip_id}/ai-trace")
def ai_trace(
    clip_id: int,
    model: str | None = None,
    impact_frame_override: int | None = Form(None),
    ball_track_max_frames: int | None = Form(None),
    ball_at_rest_x: int | None = Form(None),
    ball_at_rest_y: int | None = Form(None),
    manual_ball_positions_json: str | None = Form(None),
    handedness_override: str | None = Form(None),
    db: Session = Depends(get_db),
):
    """AI analysis — full pipeline (address + handedness + impact +
    ball-track + tracer-render) on one clip.

    All overrides are optional. Supplying any of them swaps out the
    corresponding AI step with operator-supplied values:

    - impact_frame_override (int): bypasses audio + AI vision impact
      detection. Address frame is auto-derived (impact − 1.5 s).
    - ball_track_max_frames (int): replaces the per-fps default in
      track_ball_after_impact. Higher = longer tracer arc.
    - ball_at_rest_x, ball_at_rest_y (ints, native pixel coords):
      bypass the handedness Claude call; ball-at-address position
      is set from these.
    - manual_ball_positions_json (string, JSON of
      [{"frame":N,"x":X,"y":Y},…] in native pixel coords): merged
      into the ball-track output after AI tracking. Override
      existing frame entries OR insert new ones where AI missed.
    """
    clip = db.get(VideoClip, clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    # Prefer the raw tee-only cut for composite clips — source_url for
    # dual-cam clips points at a composite that has the tracer overlay
    # baked in and includes the green-side camera, which AI analysis
    # can't make sense of.
    analysis_url = clip.tee_clip_url or clip.source_url
    if not analysis_url:
        raise HTTPException(400, "clip has no analyzable source")
    fname = analysis_url.rstrip("/").rsplit("/", 1)[-1]
    if not fname:
        raise HTTPException(400, "could not parse filename from analysis URL")
    if "_composite" in fname:
        raise HTTPException(
            400,
            "this clip is a composite and has no raw tee cut on file; "
            "re-process the long upload to populate tee_clip_url",
        )
    fpath = CLIPS_DIR / fname
    if not fpath.exists():
        raise HTTPException(404, f"source file missing on disk: {fname}")

    # Parse + validate manual_ball_positions_json into the list-of-dicts
    # shape the pipeline expects.
    manual_positions: list[dict] | None = None
    if manual_ball_positions_json:
        try:
            parsed = json.loads(manual_ball_positions_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"manual_ball_positions_json invalid JSON: {exc}")
        if not isinstance(parsed, list):
            raise HTTPException(400, "manual_ball_positions_json must be a JSON array")
        manual_positions = []
        for entry in parsed:
            if not isinstance(entry, dict):
                raise HTTPException(400, "each manual position entry must be an object")
            try:
                manual_positions.append({
                    "frame": int(entry["frame"]),
                    "x": int(entry["x"]),
                    "y": int(entry["y"]),
                })
            except (KeyError, TypeError, ValueError):
                raise HTTPException(
                    400,
                    "each manual position entry needs integer frame, x, y",
                )

    ball_at_rest_override: tuple[float, float] | None = None
    if ball_at_rest_x is not None and ball_at_rest_y is not None:
        ball_at_rest_override = (float(ball_at_rest_x), float(ball_at_rest_y))

    handedness_clean: str | None = None
    if handedness_override:
        v = handedness_override.strip().lower()
        if v in ("right", "left", "unknown"):
            handedness_clean = v
        elif v not in ("", "auto", "ai"):
            raise HTTPException(
                400,
                "handedness_override must be 'right', 'left', or 'unknown'",
            )

    pipe = run_full_ai_tracer_pipeline(
        fpath, output_dir=CLIPS_DIR, output_prefix=fpath.stem, model=model,
        impact_frame_override=int(impact_frame_override) if impact_frame_override is not None else None,
        ball_track_max_frames_override=int(ball_track_max_frames) if ball_track_max_frames is not None else None,
        ball_at_rest_override=ball_at_rest_override,
        manual_ball_positions=manual_positions,
        handedness_override=handedness_clean,
    )

    def _public_url(p):
        if p is None:
            return None
        try:
            mtime = int(p.stat().st_mtime)
        except FileNotFoundError:
            return None
        return f"{settings.app_base_url}/uploads/clips/{p.name}?v={mtime}"

    image_url = _public_url(pipe.get("address_image_path"))
    impact_image_url = _public_url(pipe.get("impact_image_path"))

    # Transcode the tracer MP4 to H.264 + faststart for browser playback.
    tracer_video_url = None
    tracer_path = pipe.get("tracer_video_path")
    tracer_video_info = pipe.get("tracer_video_info")
    if tracer_path is not None and tracer_video_info and tracer_video_info.get("ok"):
        compressed = compress_for_email(tracer_path)
        if not compressed:
            log.warning(
                "ai_tracer: compress_for_email returned False for %s — "
                "browser playback may fail", tracer_path.name,
            )
        if tracer_path.exists() and tracer_path.stat().st_size > 0:
            tracer_video_url = _public_url(tracer_path)
        else:
            tracer_video_info = {
                **tracer_video_info,
                "ok": False,
                "error": "post-encode produced empty file",
            }

    # Resolve per-frame tracker image URLs.
    ball_track_frames_out = []
    for rec in pipe.get("ball_track_frames", []):
        filename = rec.get("image_filename")
        url = None
        if filename:
            fp = CLIPS_DIR / filename
            if fp.exists():
                mtime = int(fp.stat().st_mtime)
                url = f"{settings.app_base_url}/uploads/clips/{filename}?v={mtime}"
        ball_track_frames_out.append({
            "frame": rec.get("frame"),
            "found": rec.get("found"),
            "x": rec.get("x"),
            "y": rec.get("y"),
            "confidence": rec.get("confidence"),
            "notes": rec.get("notes"),
            "retry": rec.get("retry", False),
            "image_url": url,
        })

    ball_track_info = pipe.get("ball_track")
    ball_track_summary = None
    if ball_track_info is not None:
        ball_track_summary = {
            k: v for k, v in ball_track_info.items() if k != "frames"
        }

    db.add(AuditLog(
        actor="admin", action="ai_trace_address", target=f"clip:{clip.id}",
        detail=str({
            "address": pipe.get("address"),
            "handedness": pipe.get("handedness"),
            "impact": pipe.get("impact"),
            "impact_refined": pipe.get("impact_refined"),
            "ball_track": ball_track_summary,
            "n_track_frames_with_image": sum(
                1 for r in ball_track_frames_out if r.get("image_url")
            ),
            "tracer_video": tracer_video_info,
            "cutover_time_sec": pipe.get("cutover_time_sec"),
        }),
    ))
    db.commit()

    return {
        "clip_id": clip.id,
        "source_url": clip.source_url,
        "address": pipe.get("address"),
        "address_image_url": image_url,
        "handedness": pipe.get("handedness"),
        "impact": pipe.get("impact"),
        "impact_refined": pipe.get("impact_refined"),
        "impact_image_url": impact_image_url,
        "ball_track": ball_track_summary,
        "ball_track_frames": ball_track_frames_out,
        "tracer_video": tracer_video_info,
        "tracer_video_url": tracer_video_url,
        "cutover_time_sec": pipe.get("cutover_time_sec"),
    }


def _utcnow_naive() -> datetime:
    """Naive UTC datetime, matching how the model stores timestamps."""
    return datetime.utcnow()


def _run_long_upload_job(
    upload_id: int,
    seg_list: list[dict],
    auto_detect_swings: bool,
    starting_hole: int,
    ai_tracer_model: str | None,
    audio_min_peak_ratio: float = 5.0,
    motion_ratio: float = 2.0,
    combined_pair_window_sec: float = 3.0,
) -> None:
    """Background worker for the long-upload cut / splice / AI-tracer
    pipeline.

    Owns its own DB session so the calling HTTP request can return
    immediately. Flips the LongVideoUpload row's processing_status
    through processing → completed (or failed, with last_error set).
    Any caller-supplied segments take precedence; if seg_list is empty
    and auto_detect_swings is true, peaks are detected from the tee
    audio inside this worker.
    """
    db = SessionLocal()
    try:
        row = db.get(LongVideoUpload, upload_id)
        if not row:
            log.warning("long-upload worker: row %s vanished before start", upload_id)
            return
        row.processing_status = "processing"
        row.processing_started_at = _utcnow_naive()
        row.last_error = None
        db.commit()

        try:
            src_path = CLIPS_DIR / row.tee_filename if row.tee_filename else None
            if not src_path or not src_path.exists():
                raise RuntimeError(
                    f"tee source file missing on disk: {row.tee_filename}"
                )
            green_src_path: Path | None = None
            if row.green_filename:
                candidate = CLIPS_DIR / row.green_filename
                if not candidate.exists():
                    raise RuntimeError(
                        f"green source file missing on disk: {row.green_filename}"
                    )
                green_src_path = candidate

            segs = list(seg_list or [])
            auto_used = False
            if not segs and auto_detect_swings:
                auto_used = True
                tee_fps = probe_fps(src_path) or 30.0
                # Combined audio + motion detector: an audio impact only
                # counts when a motion burst peaks within ±3 s. Filters
                # the false positives each detector produces alone.
                detected = detect_swings_combined(
                    src_path,
                    fps=tee_fps,
                    audio_min_peak_ratio=float(audio_min_peak_ratio),
                    motion_ratio=float(motion_ratio),
                    pair_window_sec=float(combined_pair_window_sec),
                )
                for i, d in enumerate(detected):
                    segs.append({
                        "hole_number": starting_hole + i,
                        "start_sec": d["start_sec"],
                        "end_sec": d["end_sec"],
                    })
                if not segs:
                    raise RuntimeError(
                        "no swings detected (combined audio+motion found "
                        "no paired peaks within the 3s window)"
                    )

            durations = [s["end_sec"] - s["start_sec"] for s in segs]
            log.info(
                "long-upload worker: upload=%s segs=%d source=%s durations=%s",
                upload_id, len(segs), "auto-combined" if auto_used else "manual",
                ", ".join(f"{d:.1f}s" for d in durations[:30]),
            )

            # Publish total + reset progress so the polling UI can show
            # "X/Y processed" while the heavy work runs.
            row.last_n_segments = len(segs)
            row.last_n_succeeded = 0
            db.commit()

            results = _process_long_upload_segments(
                db,
                course_id=row.course_id,
                camera_type=row.camera_type,
                base_dt=row.base_captured_at,
                src_path=src_path,
                green_src_path=green_src_path,
                seg_list=segs,
                dual_camera=green_src_path is not None,
                ai_tracer_model=ai_tracer_model,
                progress_upload_id=upload_id,
            )

            # Re-fetch in case the session was rolled back during segment work.
            row = db.get(LongVideoUpload, upload_id)
            if row is None:
                log.warning("long-upload worker: row %s deleted mid-run", upload_id)
                return
            row.last_n_segments = len(segs)
            row.last_n_succeeded = sum(1 for r in results if r.get("ok"))
            row.processing_status = "completed"
            row.processing_completed_at = _utcnow_naive()
            row.last_error = None
            db.commit()
        except Exception as exc:
            log.exception("long-upload worker %s failed: %s", upload_id, exc)
            db.rollback()
            row = db.get(LongVideoUpload, upload_id)
            if row is not None:
                row.processing_status = "failed"
                row.processing_completed_at = _utcnow_naive()
                row.last_error = str(exc)[:2000]
                db.commit()
    finally:
        db.close()


def _process_long_upload_segments(
    db: Session,
    course_id: int,
    camera_type: str,
    base_dt: datetime,
    src_path: Path,
    green_src_path: Path | None,
    seg_list: list[dict],
    dual_camera: bool,
    ai_tracer_model: str | None,
    progress_upload_id: int | None = None,
) -> list[dict]:
    """Cut + process each swing segment from one (or two) source video(s).

    Shared between the initial /clips/long-upload endpoint and the
    /clips/long-uploads/{id}/reprocess endpoint so re-editing a stored
    long upload runs the exact same pipeline.

    When `progress_upload_id` is set, the matching LongVideoUpload
    row's `last_n_succeeded` is bumped after every segment commit so a
    polling UI sees "X/Y processed" in near-real-time.
    """
    def _bump_progress(done_so_far: int) -> None:
        if progress_upload_id is None:
            return
        row = db.get(LongVideoUpload, progress_upload_id)
        if row is not None:
            row.last_n_succeeded = done_so_far
            db.commit()

    # Cache the course once per call — _intro_overlay_for_clip needs
    # course.name / par3_holes / hole_yardages for every segment.
    _course_for_intro: Course | None = db.get(Course, course_id)

    def _intro_overlay_for_clip(clip: VideoClip, participant: Participant | None) -> None:
        """Best-effort: re-encode the clip's deliverable file in-place
        with the slide-in/out intro panels overlaid on the first ~3.5s.
        Any failure (PIL missing, ffmpeg fails, file gone) logs and
        moves on — the underlying clip ships either way."""
        if not clip.source_url:
            return
        fname = clip.source_url.rstrip("/").rsplit("/", 1)[-1]
        if not fname:
            return
        fpath = CLIPS_DIR / fname
        if not fpath.exists():
            return
        course = _course_for_intro
        course_name = course.name if course and course.name else ""
        # Default yardage when this hole isn't in course.hole_yardages
        # (e.g. fresh course setup or a newly added hole).
        yardage = 101
        if course and course.hole_yardages:
            raw_y = course.hole_yardages.get(str(int(clip.hole_number)))
            try:
                if raw_y is not None:
                    yardage = int(raw_y)
            except (TypeError, ValueError):
                pass
        try:
            apply_intro_overlay_inplace(
                fpath,
                # Default to 'Brent Baldwin' when the clip didn't match a
                # registered participant — keeps the on-screen graphic
                # populated instead of showing a blank '—'.
                player_name=(participant.name if participant else "Brent Baldwin"),
                course_name=course_name,
                hole_number=int(clip.hole_number),
                # GolfReelz is a par-3 challenge product; every hole is
                # treated as par 3 regardless of the course's par3_holes
                # list.
                par=3,
                yardage=yardage,
            )
        except Exception as exc:  # pragma: no cover
            log.warning("intro overlay failed for clip %s: %s", clip.id, exc)

    n_done = 0
    results: list[dict] = []
    for idx, seg in enumerate(seg_list):
        try:
            hole_number = int(seg["hole_number"])
            start_sec = float(seg["start_sec"])
            end_sec = float(seg["end_sec"])
        except (KeyError, TypeError, ValueError):
            results.append({"index": idx, "ok": False, "error": "missing or invalid hole_number / start_sec / end_sec"})
            continue
        if end_sec <= start_sec:
            results.append({"index": idx, "ok": False, "error": "end_sec must be > start_sec"})
            continue

        seg_name = f"{course_id}-h{hole_number}-{secrets.token_hex(6)}.mp4"
        seg_path = CLIPS_DIR / seg_name
        ok = cut_segment(src_path, seg_path, start_sec, end_sec)
        if not ok:
            results.append({"index": idx, "ok": False, "error": "ffmpeg cut failed (or ffmpeg not installed)"})
            continue

        # --- Dual-camera branch -----------------------------------
        if dual_camera and green_src_path is not None:
            green_seg_name = f"{course_id}-h{hole_number}-green-{secrets.token_hex(6)}.mp4"
            green_seg_path = CLIPS_DIR / green_seg_name
            green_cut_ok = cut_segment(green_src_path, green_seg_path, start_sec, end_sec)
            if not green_cut_ok:
                seg_path.unlink(missing_ok=True)
                results.append({"index": idx, "ok": False, "error": "green ffmpeg cut failed"})
                continue

            pipe = run_full_ai_tracer_pipeline(
                seg_path,
                output_dir=CLIPS_DIR,
                output_prefix=seg_path.stem,
                model=ai_tracer_model,
            )

            composite_url = None
            composite_info: dict | None = None
            composite_path: Path | None = None
            cutover = pipe.get("cutover_time_sec")
            tracer_path = pipe.get("tracer_video_path")
            if (
                pipe.get("ok") and tracer_path is not None
                and cutover is not None and cutover > 0
                and tracer_path.exists()
            ):
                tee_info = probe_video_info(tracer_path)
                green_info = probe_video_info(green_seg_path)
                tee_duration = float(tee_info.get("duration") or 0.0)
                green_duration = float(green_info.get("duration") or 0.0)
                switch_sec = max(0.0, min(cutover, tee_duration or cutover))
                end_green_sec = max(switch_sec + 0.1, green_duration or (end_sec - start_sec))
                composite_name = f"{seg_path.stem}_composite.mp4"
                composite_path = CLIPS_DIR / composite_name
                if concat_two_clips(
                    tracer_path, 0.0, switch_sec,
                    green_seg_path, switch_sec, end_green_sec,
                    composite_path,
                ):
                    compress_for_email(composite_path)
                    if composite_path.exists() and composite_path.stat().st_size > 0:
                        composite_url = f"{settings.app_base_url}/uploads/clips/{composite_name}"
                        composite_info = {
                            "switch_sec": round(switch_sec, 2),
                            "end_sec": round(end_green_sec, 2),
                            "fps": pipe.get("fps"),
                            "tracer_frame_range": (pipe.get("tracer_video_info") or {}).get("frame_range"),
                            "method": (pipe.get("impact") or {}).get("method"),
                        }

            thumb_source = tracer_path if (tracer_path and tracer_path.exists()) else seg_path
            thumb_path = extract_thumbnail(thumb_source)
            thumb_url = (
                f"{settings.app_base_url}/uploads/clips/{thumb_path.name}"
                if thumb_path else None
            )

            tee_clip_public_url: str | None = None
            if composite_url:
                public_source = composite_url
                public_tracer = composite_url
                # source_url points at the composite (tee-with-tracer + green),
                # which AI analysis can't use because the player isn't visible
                # in the green half and the tracer overlay is baked in. Keep a
                # pointer to the raw tee cut so the AI page has a single-camera
                # player-visible clip to target.
                if seg_path.exists():
                    compress_for_email(seg_path)
                    tee_clip_public_url = (
                        f"{settings.app_base_url}/uploads/clips/{seg_name}"
                    )
            elif tracer_path and tracer_path.exists():
                public_source = f"{settings.app_base_url}/uploads/clips/{tracer_path.name}"
                public_tracer = public_source
            else:
                compress_for_email(seg_path)
                public_source = f"{settings.app_base_url}/uploads/clips/{seg_name}"
                public_tracer = None

            captured_dt = base_dt + timedelta(seconds=start_sec)
            clip = VideoClip(
                course_id=course_id,
                hole_number=hole_number,
                camera_type=camera_type,
                captured_at=captured_dt,
                source_url=public_source,
                thumbnail_url=thumb_url,
                tracer_url=public_tracer,
                tee_clip_url=tee_clip_public_url,
                long_upload_id=progress_upload_id,
                carry_yards=_optional_int(seg.get("carry_yards")),
                apex_feet=_optional_int(seg.get("apex_feet")),
                ball_speed_mph=_optional_int(seg.get("ball_speed_mph")),
                distance_from_pin_feet=_optional_int(seg.get("distance_from_pin_feet")),
                ball_in_cup=bool(seg.get("ball_in_cup", False)),
                processing_status=ClipProcessingStatus.received.value,
            )
            db.add(clip)
            db.flush()
            participant = match_clip(db, clip)
            if participant and clip.ball_in_cup:
                notifications.notify_hio_under_review(participant.name, participant.mobile, participant.email)
            _intro_overlay_for_clip(clip, participant)
            db.commit()

            results.append({
                "index": idx,
                "ok": True,
                "clip_id": clip.id,
                "hole_number": hole_number,
                "captured_at": captured_dt.isoformat(),
                "status": clip.processing_status,
                "participant_id": clip.participant_id,
                "participant_name": participant.name if participant else None,
                "source_url": clip.source_url,
                "tracer_url": clip.tracer_url,
                "thumbnail_url": clip.thumbnail_url,
                "issue_note": clip.issue_note,
                "dual_camera": True,
                "composite": composite_info,
                "ai_tracer_error": pipe.get("error"),
            })
            n_done += 1
            _bump_progress(n_done)
            continue

        # --- Single-camera (original) branch ----------------------
        compress_for_email(seg_path)
        thumb_path = extract_thumbnail(seg_path)
        thumb_url = (
            f"{settings.app_base_url}/uploads/clips/{thumb_path.name}"
            if thumb_path else None
        )
        tracer_url, _, _, _ = _run_tracer(seg_path)

        captured_dt = base_dt + timedelta(seconds=start_sec)
        clip = VideoClip(
            course_id=course_id,
            hole_number=hole_number,
            camera_type=camera_type,
            captured_at=captured_dt,
            source_url=f"{settings.app_base_url}/uploads/clips/{seg_name}",
            thumbnail_url=thumb_url,
            tracer_url=tracer_url,
            long_upload_id=progress_upload_id,
            carry_yards=_optional_int(seg.get("carry_yards")),
            apex_feet=_optional_int(seg.get("apex_feet")),
            ball_speed_mph=_optional_int(seg.get("ball_speed_mph")),
            distance_from_pin_feet=_optional_int(seg.get("distance_from_pin_feet")),
            ball_in_cup=bool(seg.get("ball_in_cup", False)),
            processing_status=ClipProcessingStatus.received.value,
        )
        db.add(clip)
        db.flush()

        participant = match_clip(db, clip)
        if participant and clip.ball_in_cup:
            notifications.notify_hio_under_review(participant.name, participant.mobile, participant.email)
        _intro_overlay_for_clip(clip, participant)
        db.commit()

        results.append({
            "index": idx,
            "ok": True,
            "clip_id": clip.id,
            "hole_number": hole_number,
            "captured_at": captured_dt.isoformat(),
            "status": clip.processing_status,
            "participant_id": clip.participant_id,
            "participant_name": participant.name if participant else None,
            "source_url": clip.source_url,
            "tracer_url": clip.tracer_url,
            "issue_note": clip.issue_note,
        })
        n_done += 1
        _bump_progress(n_done)
    return results


@router.delete("/clips/{clip_id}")
def delete_clip(clip_id: int, db: Session = Depends(get_db)):
    """Delete a VideoClip row plus the underlying source / tracer /
    thumbnail / per-clip AI tracer files on disk. Used from the
    Broadcast review page when an auto-cut composite is bad.

    Per-swing tracer images / impact / address JPGs produced by the
    AI pipeline for this clip are also unlinked when the filename
    starts with the same stem. The stored LongVideoUpload row that
    this clip was derived from (if any) is preserved — reprocessing
    it would produce a new VideoClip.
    """
    clip = db.get(VideoClip, clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")

    freed = 0
    deleted_files: list[str] = []
    # Collect every URL we know about for this clip so we can resolve
    # them to filenames in CLIPS_DIR.
    candidate_urls = [clip.source_url, clip.tracer_url, clip.thumbnail_url]
    candidate_names: set[str] = set()
    for url in candidate_urls:
        if not url:
            continue
        # Strip any cache-buster query string and pull the basename.
        no_q = url.split("?", 1)[0]
        name = no_q.rstrip("/").rsplit("/", 1)[-1]
        if name:
            candidate_names.add(name)
    # Also unlink any per-clip AI tracer artifacts whose filenames
    # share the source stem (e.g. {stem}_address.jpg, {stem}_impact.jpg,
    # {stem}_track_f00000.jpg, {stem}_ai_tracer.mp4, {stem}_composite.mp4).
    stems: set[str] = set()
    for name in list(candidate_names):
        stem = name.rsplit(".", 1)[0]
        # Strip known suffixes so we get the original clip stem.
        for suffix in ("_composite", "_ai_tracer", "_traced"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        stems.add(stem)
    if stems:
        for fp in CLIPS_DIR.iterdir():
            if not fp.is_file():
                continue
            for stem in stems:
                if fp.name == stem or fp.name.startswith(stem + ".") or fp.name.startswith(stem + "_"):
                    candidate_names.add(fp.name)
                    break

    for name in candidate_names:
        fp = CLIPS_DIR / name
        try:
            if fp.exists():
                freed += fp.stat().st_size
                fp.unlink()
                deleted_files.append(name)
        except Exception as exc:
            log.warning("clip delete: failed to unlink %s: %s", fp, exc)

    db.add(AuditLog(
        actor="admin", action="delete_clip", target=f"clip:{clip.id}",
        detail=f"deleted {len(deleted_files)} files, freed {freed} bytes",
    ))
    db.delete(clip)
    db.commit()
    return {
        "deleted": True,
        "clip_id": clip_id,
        "freed_bytes": freed,
        "files_unlinked": deleted_files,
    }


@router.post("/clips/quick-upload")
async def quick_upload_videos(
    course_id: int = Form(...),
    base_captured_at: str = Form(...),
    video: UploadFile = File(...),
    video_green: UploadFile | None = File(None),
    swing_count: str = Form("multiple"),
    db: Session = Depends(get_db),
):
    """Simple operator-facing upload: save the tee video (plus optional
    green-side video) and create a LongVideoUpload row.

    `swing_count` drives the auto-produce decision:
      - 'multiple' (default): full round / many swings. The cut /
        AI-tracer / composite background job kicks off immediately
        with sane defaults (auto-detect swings, starting hole 1,
        combined detector at audio×5 / motion×2 thresholds).
      - 'single': one swing per video. Queue the row for manual
        editing on /admin/production before producing — no
        background thread spawns.

    Either way, this endpoint returns immediately so the upload UI
    isn't blocked on the multi-minute processing phase.
    """
    course = db.get(Course, course_id)
    if not course:
        raise HTTPException(404, "course not found")
    if not (video.content_type or "").startswith("video/"):
        raise HTTPException(400, "tee video must be a video file")
    dual_camera = video_green is not None
    if dual_camera and not (video_green.content_type or "").startswith("video/"):
        raise HTTPException(400, "video_green must be a video file")

    try:
        base_dt = datetime.fromisoformat(base_captured_at.replace("Z", "+00:00"))
        if base_dt.tzinfo is not None:
            base_dt = base_dt.astimezone().replace(tzinfo=None)
    except ValueError:
        raise HTTPException(400, "invalid base_captured_at; use ISO 8601")

    # Save the tee video.
    data = await video.read()
    if not data:
        raise HTTPException(400, "empty tee video upload")
    if len(data) > 1024 * 1024 * 1024:
        raise HTTPException(413, "tee video too large (max 1GB)")
    src_ext = (video.filename or "").rsplit(".", 1)[-1].lower() if "." in (video.filename or "") else "mp4"
    if src_ext not in ("mp4", "mov", "webm", "m4v"):
        src_ext = "mp4"
    src_name = f"long-{course_id}-{secrets.token_hex(6)}.{src_ext}"
    src_path = CLIPS_DIR / src_name
    src_path.write_bytes(data)

    # Save the green video if present.
    green_src_name: str | None = None
    green_original_filename: str | None = None
    if dual_camera:
        green_data = await video_green.read()
        if not green_data:
            src_path.unlink(missing_ok=True)
            raise HTTPException(400, "empty green video upload")
        if len(green_data) > 1024 * 1024 * 1024:
            src_path.unlink(missing_ok=True)
            raise HTTPException(413, "green video too large (max 1GB)")
        g_ext = (video_green.filename or "").rsplit(".", 1)[-1].lower() if "." in (video_green.filename or "") else "mp4"
        if g_ext not in ("mp4", "mov", "webm", "m4v"):
            g_ext = "mp4"
        green_src_name = f"long-{course_id}-green-{secrets.token_hex(6)}.{g_ext}"
        (CLIPS_DIR / green_src_name).write_bytes(green_data)
        green_original_filename = video_green.filename or None

    swing_count_norm = (swing_count or "multiple").strip().lower()
    if swing_count_norm not in ("single", "multiple"):
        swing_count_norm = "multiple"
    auto_process = swing_count_norm == "multiple"

    upload_row = LongVideoUpload(
        course_id=course_id,
        camera_type="tee",
        base_captured_at=base_dt,
        tee_filename=src_name,
        green_filename=green_src_name,
        tee_original_filename=(video.filename or None),
        green_original_filename=green_original_filename,
        processing_status="pending",
        swing_count=swing_count_norm,
    )
    db.add(upload_row)
    db.commit()
    db.refresh(upload_row)

    # Generate poster thumbnails so the Production page card has a
    # visual preview without re-probing the videos on every load.
    try:
        extract_thumbnail(src_path)
        if green_src_name:
            extract_thumbnail(CLIPS_DIR / green_src_name)
    except Exception as exc:  # pragma: no cover
        log.warning("quick-upload: thumbnail extraction failed: %s", exc)

    db.add(AuditLog(
        actor="admin", action="quick_upload_videos",
        target=f"long_upload:{upload_row.id}",
        detail=(
            f"course={course_id} tee={src_name} swing_count={swing_count_norm}"
            + (f" green={green_src_name}" if green_src_name else "")
        ),
    ))
    db.commit()

    log.info(
        "quick-upload: upload=%s course=%s tee=%s green=%s swing_count=%s",
        upload_row.id, course_id, src_name, green_src_name, swing_count_norm,
    )

    # 'multiple' swing-count kicks off the cut / AI-tracer / composite
    # background job immediately. 'single' queues for manual editing on
    # /admin/production first.
    if auto_process:
        threading.Thread(
            target=_run_long_upload_job,
            kwargs={
                "upload_id": upload_row.id,
                "seg_list": [],
                "auto_detect_swings": True,
                "starting_hole": 1,
                "ai_tracer_model": None,
            },
            daemon=True,
            name=f"long-upload-{upload_row.id}",
        ).start()
        message = (
            "Multiple-swing upload — processing started in the "
            "background. Produced clips will appear on Broadcast when "
            "ready."
        )
    else:
        message = (
            "Single-swing upload — queued for editing on Production. "
            "and click Reprocess when you're ready to produce."
        )

    return {
        "upload_id": upload_row.id,
        "processing_status": upload_row.processing_status,
        "dual_camera": dual_camera,
        "swing_count": swing_count_norm,
        "auto_processing": auto_process,
        "message": message,
    }


@router.post("/clips/long-upload")
async def upload_long_video(
    course_id: int = Form(...),
    camera_type: str = Form("tee"),
    base_captured_at: str = Form(...),
    segments: str = Form("[]"),
    auto_detect_swings: bool = Form(False),
    starting_hole: int = Form(1),
    video: UploadFile = File(...),
    video_green: UploadFile | None = File(None),
    ai_tracer_model: str | None = Form(None),
    audio_min_peak_ratio: float = Form(5.0),
    motion_ratio: float = Form(2.0),
    combined_pair_window_sec: float = Form(3.0),
    db: Session = Depends(get_db),
):
    """Cut a long video into multiple per-swing clips and run each through
    the standard match + deliver pipeline.

    Body (multipart):
      course_id: int
      camera_type: 'tee' | 'wide_green' | 'hole'
      base_captured_at: ISO 8601 — when the recording started. Each
                       segment's captured_at = base + start_sec.
      segments: JSON array of {hole_number, start_sec, end_sec, ...stats}
      video: long MP4 file
      video_green: (optional) second long MP4 file from a green-side
                   camera. Must be wall-clock-synchronized to `video`
                   (both started recording at the same moment). When
                   present, each segment is cut from BOTH videos, the
                   full AI tracer pipeline runs on the tee cut, and
                   the deliverable becomes a composite: tee-with-tracer
                   from t=0 up to 1 s after the tracer ends, then a
                   hard cut to the green clip for the ball landing.
      ai_tracer_model: (optional) override the AI tracer model used
                       on the tee cut in dual-camera mode. Falls back
                       to TRACER_AI_MODEL env / Opus 4.7 default.
    """
    course = db.get(Course, course_id)
    if not course:
        raise HTTPException(404, "course not found")
    if not (video.content_type or "").startswith("video/"):
        raise HTTPException(400, "must be a video file")
    dual_camera = video_green is not None
    if dual_camera and not (video_green.content_type or "").startswith("video/"):
        raise HTTPException(400, "video_green must be a video file")

    try:
        seg_list = json.loads(segments or "[]")
    except json.JSONDecodeError:
        raise HTTPException(400, "segments must be a JSON array")
    if not isinstance(seg_list, list):
        raise HTTPException(400, "segments must be a JSON array")
    # When the operator hasn't marked segments manually, we'll auto-
    # detect every club-on-ball impact from the tee audio after the
    # source video lands on disk — see the post-write block below.
    if not seg_list and not auto_detect_swings:
        raise HTTPException(
            400,
            "no segments supplied — pass auto_detect_swings=true to "
            "auto-detect swings from audio, or provide at least one "
            "segment manually",
        )

    try:
        base_dt = datetime.fromisoformat(base_captured_at.replace("Z", "+00:00"))
        if base_dt.tzinfo is not None:
            base_dt = base_dt.astimezone().replace(tzinfo=None)
    except ValueError:
        raise HTTPException(400, "invalid base_captured_at; use ISO 8601")

    data = await video.read()
    if not data:
        raise HTTPException(400, "empty video upload")
    if len(data) > 1024 * 1024 * 1024:  # 1GB cap on the long source
        raise HTTPException(413, "video too large (max 1GB)")

    src_ext = (video.filename or "").rsplit(".", 1)[-1].lower() if "." in (video.filename or "") else "mp4"
    if src_ext not in ("mp4", "mov", "webm", "m4v"):
        src_ext = "mp4"
    src_name = f"long-{course_id}-{secrets.token_hex(6)}.{src_ext}"
    src_path = CLIPS_DIR / src_name
    src_path.write_bytes(data)

    green_src_path: Path | None = None
    if dual_camera:
        green_data = await video_green.read()
        if not green_data:
            src_path.unlink(missing_ok=True)
            raise HTTPException(400, "empty green video upload")
        if len(green_data) > 1024 * 1024 * 1024:
            src_path.unlink(missing_ok=True)
            raise HTTPException(413, "green video too large (max 1GB)")
        g_ext = (video_green.filename or "").rsplit(".", 1)[-1].lower() if "." in (video_green.filename or "") else "mp4"
        if g_ext not in ("mp4", "mov", "webm", "m4v"):
            g_ext = "mp4"
        green_src_name = f"long-{course_id}-green-{secrets.token_hex(6)}.{g_ext}"
        green_src_path = CLIPS_DIR / green_src_name
        green_src_path.write_bytes(green_data)

    # Persist the long source(s) + queue the cut / splice / AI-tracer
    # pipeline as a background job. The HTTP request returns immediately
    # so the operator isn't blocked on the (multi-minute) processing
    # phase — the frontend polls /long-uploads to track progress.
    upload_row = LongVideoUpload(
        course_id=course_id,
        camera_type=camera_type,
        base_captured_at=base_dt,
        tee_filename=src_path.name,
        green_filename=(green_src_path.name if green_src_path is not None else None),
        tee_original_filename=(video.filename or None),
        green_original_filename=(video_green.filename if video_green is not None else None),
        processing_status="pending",
    )
    db.add(upload_row)
    db.commit()
    db.refresh(upload_row)

    upload_id = upload_row.id
    threading.Thread(
        target=_run_long_upload_job,
        kwargs={
            "upload_id": upload_id,
            "seg_list": list(seg_list),
            "auto_detect_swings": bool(auto_detect_swings),
            "starting_hole": int(starting_hole or 1),
            "ai_tracer_model": ai_tracer_model,
            "audio_min_peak_ratio": float(audio_min_peak_ratio),
            "motion_ratio": float(motion_ratio),
            "combined_pair_window_sec": float(combined_pair_window_sec),
        },
        daemon=True,
        name=f"long-upload-{upload_id}",
    ).start()

    return {
        "upload_id": upload_id,
        "processing_status": "pending",
        "dual_camera": dual_camera,
        "queued_segments": len(seg_list) if seg_list else None,
        "auto_detect_swings": bool(auto_detect_swings and not seg_list),
    }


@router.get("/long-uploads")
def list_long_uploads(limit: int = 100, db: Session = Depends(get_db)):
    """List previously-uploaded long videos so the operator can re-edit /
    reprocess them without re-uploading. Newest first."""
    rows = (
        db.query(LongVideoUpload)
        .order_by(LongVideoUpload.created_at.desc())
        .limit(max(1, min(500, limit)))
        .all()
    )
    course_ids = {r.course_id for r in rows}
    courses = (
        {c.id: c for c in db.query(Course).filter(Course.id.in_(course_ids)).all()}
        if course_ids else {}
    )
    def _quality_label(height: int | None) -> str | None:
        """Friendly resolution tier so the UI can show 'Quality: 720p HD'
        without re-deriving the mapping client-side."""
        if not height:
            return None
        if height >= 2160:
            return "4K UHD"
        if height >= 1440:
            return "1440p QHD"
        if height >= 1080:
            return "1080p HD"
        if height >= 720:
            return "720p HD"
        if height >= 480:
            return "480p SD"
        return f"{height}p"

    def _meta(path: Path | None, exists: bool) -> dict:
        """Bundle probe + thumbnail lookup for one source video. Skipping the
        probe entirely when the file is missing keeps list responses fast."""
        if not (path and exists):
            return {"size_mb": None, "duration_sec": None, "fps": None,
                    "nb_frames": None, "thumbnail_url": None,
                    "width": None, "height": None, "quality_label": None}
        size = path.stat().st_size
        info = probe_video_info(path)
        thumb = path.with_suffix(".jpg")
        thumb_url = (
            f"{settings.app_base_url}/uploads/clips/{thumb.name}"
            if thumb.exists() else None
        )
        return {
            "size_mb": round(size / 1024 / 1024, 1) if size else None,
            "duration_sec": round(info["duration"], 1) if info.get("duration") else None,
            "fps": round(info["fps"], 2) if info.get("fps") else None,
            "nb_frames": info.get("nb_frames"),
            "thumbnail_url": thumb_url,
            "width": info.get("width"),
            "height": info.get("height"),
            "quality_label": _quality_label(info.get("height")),
        }

    # Bulk-load every produced clip for the uploads we're about to return.
    # The Production card surfaces them as the third "Produced Video" tile.
    upload_ids = [r.id for r in rows]
    produced_by_upload: dict[int, list[VideoClip]] = {}
    if upload_ids:
        produced_rows = (
            db.query(VideoClip)
            .filter(VideoClip.long_upload_id.in_(upload_ids))
            .order_by(VideoClip.captured_at.asc())
            .all()
        )
        for clip in produced_rows:
            produced_by_upload.setdefault(clip.long_upload_id, []).append(clip)

    def _produced(upload_id: int) -> list[dict]:
        clips = produced_by_upload.get(upload_id, [])
        out = []
        for c in clips:
            # Prefer the tracer-rendered URL when we have one — it's the
            # final on-air output. Fallback to source_url so single-cam
            # clips still play before the tracer ships.
            play_url = c.tracer_url or c.source_url
            out.append({
                "id": c.id,
                "hole_number": c.hole_number,
                "captured_at": c.captured_at.isoformat() if c.captured_at else None,
                "video_url": play_url,
                "thumbnail_url": c.thumbnail_url,
                "ball_in_cup": bool(c.ball_in_cup),
                "is_highlight": bool(c.is_highlight),
            })
        return out

    out = []
    for r in rows:
        tee_path = CLIPS_DIR / r.tee_filename if r.tee_filename else None
        green_path = CLIPS_DIR / r.green_filename if r.green_filename else None
        tee_exists = bool(tee_path and tee_path.exists())
        green_exists = bool(green_path and green_path.exists())
        tee_meta = _meta(tee_path, tee_exists)
        green_meta = _meta(green_path, green_exists)
        course = courses.get(r.course_id)
        produced = _produced(r.id)
        out.append({
            "id": r.id,
            "course_id": r.course_id,
            "course_name": course.name if course else None,
            "course_hole_yardages": (course.hole_yardages or {}) if course else {},
            "camera_type": r.camera_type,
            "base_captured_at": r.base_captured_at.isoformat() if r.base_captured_at else None,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "swing_count": r.swing_count or "multiple",
            "tee_filename": r.tee_filename,
            "tee_original_filename": r.tee_original_filename,
            "tee_url": (
                f"{settings.app_base_url}/uploads/clips/{r.tee_filename}"
                if tee_exists else None
            ),
            "tee_thumbnail_url": tee_meta["thumbnail_url"],
            "tee_size_mb": tee_meta["size_mb"],
            "tee_duration_sec": tee_meta["duration_sec"],
            "tee_fps": tee_meta["fps"],
            "tee_nb_frames": tee_meta["nb_frames"],
            "tee_width": tee_meta["width"],
            "tee_height": tee_meta["height"],
            "tee_quality_label": tee_meta["quality_label"],
            "tee_missing": (r.tee_filename is not None and not tee_exists),
            "green_filename": r.green_filename,
            "green_original_filename": r.green_original_filename,
            "green_url": (
                f"{settings.app_base_url}/uploads/clips/{r.green_filename}"
                if green_exists else None
            ),
            "green_thumbnail_url": green_meta["thumbnail_url"],
            "green_size_mb": green_meta["size_mb"],
            "green_duration_sec": green_meta["duration_sec"],
            "green_fps": green_meta["fps"],
            "green_nb_frames": green_meta["nb_frames"],
            "green_width": green_meta["width"],
            "green_height": green_meta["height"],
            "green_quality_label": green_meta["quality_label"],
            "green_missing": (r.green_filename is not None and not green_exists),
            "dual_camera": r.green_filename is not None,
            "produced_clips": produced,
            "edit_metrics": r.edit_metrics,
            "last_n_segments": r.last_n_segments,
            "last_n_succeeded": r.last_n_succeeded,
            "processing_status": r.processing_status,
            "processing_started_at": (
                r.processing_started_at.isoformat()
                if r.processing_started_at else None
            ),
            "processing_completed_at": (
                r.processing_completed_at.isoformat()
                if r.processing_completed_at else None
            ),
            "last_error": r.last_error,
        })
    return out


@router.post("/long-uploads/{upload_id}/reprocess")
def reprocess_long_upload(
    upload_id: int,
    segments: str = Form("[]"),
    auto_detect_swings: bool = Form(True),
    starting_hole: int = Form(1),
    ai_tracer_model: str | None = Form(None),
    audio_min_peak_ratio: float = Form(5.0),
    motion_ratio: float = Form(2.0),
    combined_pair_window_sec: float = Form(3.0),
    db: Session = Depends(get_db),
):
    """Re-cut / re-process a previously-uploaded long video without
    re-uploading. Queues the same per-segment pipeline used by the
    initial POST /clips/long-upload as a background job; returns
    immediately with the upload_id and pending status so the frontend
    can poll /long-uploads."""
    row = db.get(LongVideoUpload, upload_id)
    if not row:
        raise HTTPException(404, "long upload not found")
    src_path = CLIPS_DIR / row.tee_filename if row.tee_filename else None
    if not src_path or not src_path.exists():
        raise HTTPException(
            404,
            f"tee source file missing on disk: {row.tee_filename}",
        )
    if row.green_filename:
        candidate = CLIPS_DIR / row.green_filename
        if not candidate.exists():
            raise HTTPException(
                404,
                f"green source file missing on disk: {row.green_filename}",
            )

    try:
        seg_list = json.loads(segments or "[]")
    except json.JSONDecodeError:
        raise HTTPException(400, "segments must be a JSON array")
    if not isinstance(seg_list, list):
        raise HTTPException(400, "segments must be a JSON array")
    if not seg_list and not auto_detect_swings:
        raise HTTPException(
            400,
            "no segments supplied — pass auto_detect_swings=true or "
            "provide segments manually",
        )
    if row.processing_status == "processing":
        raise HTTPException(409, "this upload is already being processed")

    row.processing_status = "pending"
    row.processing_started_at = None
    row.processing_completed_at = None
    row.last_error = None
    db.commit()

    threading.Thread(
        target=_run_long_upload_job,
        kwargs={
            "upload_id": row.id,
            "seg_list": list(seg_list),
            "auto_detect_swings": bool(auto_detect_swings),
            "starting_hole": int(starting_hole or 1),
            "ai_tracer_model": ai_tracer_model,
            "audio_min_peak_ratio": float(audio_min_peak_ratio),
            "motion_ratio": float(motion_ratio),
            "combined_pair_window_sec": float(combined_pair_window_sec),
        },
        daemon=True,
        name=f"long-upload-reprocess-{row.id}",
    ).start()

    return {
        "upload_id": row.id,
        "processing_status": "pending",
        "dual_camera": row.green_filename is not None,
        "queued_segments": len(seg_list) if seg_list else None,
        "auto_detect_swings": bool(auto_detect_swings and not seg_list),
    }


@router.post("/long-uploads/{upload_id}/auto-detect")
def auto_detect_long_upload(upload_id: int, db: Session = Depends(get_db)):
    """Run the lightweight per-swing detection on this upload's tee
    video and return: handedness, address frame (+ JPG), audio-detected
    impact frame, ball-at-rest position, a derived ball-detection ROI
    around that position, and a swing-direction "target" estimate.

    Used by the single-swing Edit wizard on /admin/production. This
    deliberately skips the per-frame ball-track Claude calls and the
    tracer-video render — those run only when the operator hits
    Produce. The cheap calls here (audio impact + one Claude
    handedness call) usually return in ~5–10 s.
    """
    row = db.get(LongVideoUpload, upload_id)
    if not row:
        raise HTTPException(404, "long upload not found")
    if not row.tee_filename:
        raise HTTPException(400, "upload has no tee video")
    src_path = CLIPS_DIR / row.tee_filename
    if not src_path.exists():
        raise HTTPException(404, f"tee source file missing on disk: {row.tee_filename}")

    fps_val = probe_fps(src_path) or 30.0

    # --- Step 1: audio impact frame ---
    audio_info = find_impact_via_audio(src_path, fps_val)
    impact_frame = audio_info.get("impact_frame") if audio_info.get("ok") else None

    # --- Step 2: address frame ---
    address_image_path = CLIPS_DIR / f"detect-{upload_id}_address.jpg"
    address_frame_idx: int | None = None
    if impact_frame is not None:
        # Address ≈ impact − 1.5s (matches run_full_ai_tracer_pipeline).
        address_frame_idx = max(0, int(impact_frame) - int(round(1.5 * fps_val)))
        try:
            import cv2  # type: ignore
            cap = cv2.VideoCapture(str(src_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, address_frame_idx)
            ok_read, frame = cap.read()
            cap.release()
            if ok_read and frame is not None:
                cv2.imwrite(str(address_image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        except Exception as exc:  # pragma: no cover
            log.warning("auto-detect: address-frame grab failed: %s", exc)
    else:
        # Audio didn't find a confident impact — fall back to the
        # Claude vision picker. Slower (one extra API call) but still
        # cheaper than running ball-track.
        addr_info = find_address_frame(src_path, output_image_path=address_image_path)
        if addr_info.get("ok") and addr_info.get("address_frame") is not None:
            address_frame_idx = int(addr_info["address_frame"])

    # --- Step 3: handedness + ball at rest ---
    handedness_info: dict = {}
    if address_frame_idx is not None:
        handedness_info = detect_handedness_at_address(src_path, address_frame_idx)

    def _public_url(p: Path | None) -> str | None:
        if not p or not p.exists():
            return None
        return f"{settings.app_base_url}/uploads/clips/{p.name}?v={int(p.stat().st_mtime)}"

    handedness = handedness_info.get("handedness") if handedness_info else None
    # Claude saw the address frame downscaled to image_width × image_height.
    # ball_x / ball_y come back in those scaled coords. Scale them up to
    # the source video's native pixel dimensions so every coordinate the
    # wizard saves (ball-at-rest, ROI, target) is in the same reference
    # frame as the ffmpeg overlay that produces the final clip.
    sent_w = handedness_info.get("image_width") if handedness_info else None
    sent_h = handedness_info.get("image_height") if handedness_info else None
    try:
        import cv2  # type: ignore
        _cap = cv2.VideoCapture(str(src_path))
        try:
            native_w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
            native_h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
        finally:
            _cap.release()
    except Exception:  # pragma: no cover
        native_w = native_h = None

    frame_w = native_w or sent_w
    frame_h = native_h or sent_h

    ball_x_raw = handedness_info.get("ball_x") if handedness_info else None
    ball_y_raw = handedness_info.get("ball_y") if handedness_info else None
    ball_x = ball_y = None
    if (
        ball_x_raw is not None and ball_y_raw is not None
        and sent_w and sent_h and frame_w and frame_h
    ):
        ball_x = int(round(float(ball_x_raw) * frame_w / sent_w))
        ball_y = int(round(float(ball_y_raw) * frame_h / sent_h))

    # --- Step 4: ball detection ROI ---
    # Square bbox centred on the ball-at-rest position, ~12% of frame
    # height on a side. Operator can resize/drag during the wizard.
    detection_area = None
    if (ball_x is not None and ball_y is not None
            and frame_w and frame_h):
        half = max(40, int(round(frame_h * 0.06)))
        detection_area = {
            "x": max(0, int(ball_x) - half),
            "y": max(0, int(ball_y) - half),
            "w": min(int(frame_w), int(ball_x) + half) - max(0, int(ball_x) - half),
            "h": min(int(frame_h), int(ball_y) + half) - max(0, int(ball_y) - half),
        }

    # --- Step 5: target (flag) point estimate ---
    # No green-detection model yet, so default the flag to the upper
    # quarter of the frame, centred horizontally. For the standard
    # behind-the-golfer tee angle this lands near the horizon — close
    # enough that the operator only has to nudge it in the wizard.
    target = None
    if frame_w and frame_h:
        target = {
            "x": int(round(frame_w * 0.5)),
            "y": int(round(frame_h * 0.25)),
            "method": "default par-3 flag estimate (upper-centre)",
        }

    db.add(AuditLog(
        actor="admin", action="auto_detect_long_upload",
        target=f"long_upload:{upload_id}",
        detail=(
            f"handedness={handedness} address_frame={address_frame_idx} "
            f"impact_frame={impact_frame} ball=({ball_x},{ball_y})"
        ),
    ))
    db.commit()

    return {
        "upload_id": upload_id,
        "fps": round(fps_val, 2) if fps_val else None,
        "tee_url": (
            f"{settings.app_base_url}/uploads/clips/{row.tee_filename}"
            if row.tee_filename else None
        ),
        "frame_width": frame_w,
        "frame_height": frame_h,
        "handedness": {
            "value": handedness,
            "confidence": handedness_info.get("confidence") if handedness_info else None,
            "notes": handedness_info.get("notes") if handedness_info else None,
        },
        "address": {
            "frame": address_frame_idx,
            "image_url": _public_url(address_image_path),
        },
        "impact": {
            "frame": impact_frame,
            "method": audio_info.get("method"),
            "ratio": audio_info.get("ratio"),
        },
        "ball_at_rest": (
            {"x": int(ball_x), "y": int(ball_y)}
            if ball_x is not None and ball_y is not None else None
        ),
        "ball_detection_area": detection_area,
        "target": target,
    }


@router.get("/long-uploads/{upload_id}/frame")
def long_upload_frame(
    upload_id: int, frame: int = 0, db: Session = Depends(get_db),
):
    """Grab a single frame from this upload's tee video as a JPG and
    return its public URL. The wizard pages through frames (±1, ±10)
    while the operator picks address / impact / ball-at-rest etc.

    Frames are cached on disk under `detect-{id}-frame-{N}.jpg` so
    re-visiting the same frame doesn't reseek. Use this for any
    frame-level UI; address auto-detect already writes
    `detect-{id}_address.jpg` via /auto-detect.
    """
    row = db.get(LongVideoUpload, upload_id)
    if not row:
        raise HTTPException(404, "long upload not found")
    if not row.tee_filename:
        raise HTTPException(400, "upload has no tee video")
    src_path = CLIPS_DIR / row.tee_filename
    if not src_path.exists():
        raise HTTPException(404, f"tee source file missing on disk: {row.tee_filename}")

    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(src_path))
    try:
        if not cap.isOpened():
            raise HTTPException(500, f"could not open {row.tee_filename}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        clamped = max(0, min(total - 1 if total else 0, int(frame)))
        out_path = CLIPS_DIR / f"detect-{upload_id}-frame-{clamped}.jpg"
        if not out_path.exists():
            cap.set(cv2.CAP_PROP_POS_FRAMES, clamped)
            ok, img = cap.read()
            if not ok or img is None:
                raise HTTPException(500, f"frame {clamped} unreadable")
            cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    finally:
        cap.release()

    return {
        "upload_id": upload_id,
        "frame": clamped,
        "total_frames": total,
        "width": width,
        "height": height,
        "image_url": (
            f"{settings.app_base_url}/uploads/clips/{out_path.name}"
            f"?v={int(out_path.stat().st_mtime)}"
        ),
    }


@router.post("/long-uploads/{upload_id}/edit-metrics")
def save_edit_metrics(
    upload_id: int,
    payload: dict = Body(...),
    db: Session = Depends(get_db),
):
    """Upsert the operator's saved wizard state on a single-swing
    upload. Body is merged into edit_metrics — pass only the fields
    that changed. Re-opening the Edit wizard reads from this so
    auto-detect only runs the very first time.

    Recognised top-level keys: handedness, address_frame,
    address_image_url, impact_frame, ball ({x,y}), roi ({x,y,w,h}),
    target ({x,y}), tracer_url, ball_track_frames (list).
    """
    row = db.get(LongVideoUpload, upload_id)
    if not row:
        raise HTTPException(404, "long upload not found")
    existing = dict(row.edit_metrics or {})
    for k, v in payload.items():
        existing[k] = v
    row.edit_metrics = existing
    db.add(row)
    db.add(AuditLog(
        actor="admin", action="save_edit_metrics",
        target=f"long_upload:{upload_id}",
        detail=str({k: (existing.get(k)) for k in payload.keys()}),
    ))
    db.commit()
    db.refresh(row)
    return {"upload_id": upload_id, "edit_metrics": row.edit_metrics}


@router.post("/long-uploads/{upload_id}/render-tracer")
def render_wizard_tracer(
    upload_id: int,
    payload: dict = Body(default={}),
    db: Session = Depends(get_db),
):
    """Run the full AI tracer pipeline (address + handedness + impact
    + ball-track + tracer render) on this upload's tee video, with
    operator overrides from the wizard's saved metrics. Persists the
    rendered tracer URL + per-frame ball positions to edit_metrics
    and returns them. Used by Step 2 of the Edit wizard.

    Optional `payload` body keys (all override the saved metrics):
      handedness ('right'|'left'), impact_frame (int),
      ball_at_rest {x,y}, manual_ball_positions (list of
      {frame,x,y}).
    """
    row = db.get(LongVideoUpload, upload_id)
    if not row:
        raise HTTPException(404, "long upload not found")
    if not row.tee_filename:
        raise HTTPException(400, "upload has no tee video")
    src_path = CLIPS_DIR / row.tee_filename
    if not src_path.exists():
        raise HTTPException(404, f"tee source file missing on disk: {row.tee_filename}")

    saved = dict(row.edit_metrics or {})

    def _pick(*keys):
        for k in keys:
            v = payload.get(k)
            if v is not None:
                return v
            v = saved.get(k)
            if v is not None:
                return v
        return None

    handedness = _pick("handedness")
    if handedness not in (None, "right", "left", "unknown"):
        handedness = None
    impact_override = _pick("impact_frame")
    ball_pt = _pick("ball", "ball_at_rest")
    manual_positions = _pick("manual_ball_positions") or []

    ball_at_rest_override = None
    if isinstance(ball_pt, dict) and ball_pt.get("x") is not None:
        ball_at_rest_override = (float(ball_pt["x"]), float(ball_pt["y"]))

    pipe = run_full_ai_tracer_pipeline(
        src_path,
        output_dir=CLIPS_DIR,
        output_prefix=f"wizard-{upload_id}",
        impact_frame_override=int(impact_override) if impact_override is not None else None,
        ball_at_rest_override=ball_at_rest_override,
        manual_ball_positions=manual_positions or None,
        handedness_override=handedness,
    )

    def _public_url(p):
        if p is None or not Path(p).exists():
            return None
        return f"{settings.app_base_url}/uploads/clips/{Path(p).name}?v={int(Path(p).stat().st_mtime)}"

    tracer_path = pipe.get("tracer_video_path")
    tracer_info = pipe.get("tracer_video_info") or {}
    tracer_url = None
    if tracer_path is not None and tracer_info.get("ok"):
        compress_for_email(tracer_path)
        if Path(tracer_path).exists() and Path(tracer_path).stat().st_size > 0:
            tracer_url = _public_url(tracer_path)

    # Surface per-frame ball-track entries with public image URLs (the
    # ai-trace pipeline writes a JPG per tracked frame next to the
    # source video — same convention used by /clips/{id}/ai-trace).
    ball_track_frames_out = []
    for rec in pipe.get("ball_track_frames", []) or []:
        filename = rec.get("image_filename")
        url = None
        if filename:
            fp = CLIPS_DIR / filename
            if fp.exists():
                url = f"{settings.app_base_url}/uploads/clips/{filename}?v={int(fp.stat().st_mtime)}"
        ball_track_frames_out.append({
            "frame": rec.get("frame"),
            "found": rec.get("found"),
            "x": rec.get("x"),
            "y": rec.get("y"),
            "confidence": rec.get("confidence"),
            "manual": rec.get("manual", False),
            "image_url": url,
        })

    saved.update({
        "handedness": pipe.get("handedness", {}).get("handedness") if pipe.get("handedness") else handedness,
        "address_frame": (pipe.get("address") or {}).get("address_frame"),
        "address_image_url": _public_url((pipe.get("address_image_path") or None)),
        "impact_frame": (pipe.get("impact_refined") or pipe.get("impact") or {}).get("impact_frame"),
        "impact_image_url": _public_url((pipe.get("impact_image_path") or None)),
        "ball": (
            {"x": int((pipe.get("handedness") or {}).get("ball_x") or 0),
             "y": int((pipe.get("handedness") or {}).get("ball_y") or 0)}
            if (pipe.get("handedness") or {}).get("ball_x") is not None else saved.get("ball")
        ),
        "tracer_url": tracer_url,
        "tracer_info": tracer_info,
        "ball_track_frames": ball_track_frames_out,
    })
    row.edit_metrics = saved
    db.add(row)
    db.add(AuditLog(
        actor="admin", action="render_wizard_tracer",
        target=f"long_upload:{upload_id}",
        detail=str({
            "tracer_ok": bool(tracer_url),
            "n_frames": len(ball_track_frames_out),
            "overrides": list(payload.keys()),
        }),
    ))
    db.commit()
    db.refresh(row)

    return {
        "upload_id": upload_id,
        "tracer_url": tracer_url,
        "ball_track_frames": ball_track_frames_out,
        "edit_metrics": row.edit_metrics,
        "pipeline_error": pipe.get("error"),
    }


@router.post("/long-uploads/{upload_id}/render-tracer-fast")
def render_tracer_fast(
    upload_id: int,
    payload: dict = Body(default={}),
    db: Session = Depends(get_db),
):
    """Re-render the tracer overlay using ONLY what's already been
    detected + any operator-supplied ball positions. No Claude calls,
    no audio probing — pure cv2. Used by Step 3 of the Edit wizard
    when the operator has added manual ball points and wants to bake
    them in without burning more API spend.

    Body (all optional, defaults pulled from edit_metrics):
      manual_positions: list of {frame, x, y} pixel coords. Merged
        into the existing ball_track_frames — overrides matching
        frames, inserts new ones for frames the AI never tracked.

    Returns the updated tracer URL + the merged ball_track_frames.
    """
    row = db.get(LongVideoUpload, upload_id)
    if not row:
        raise HTTPException(404, "long upload not found")
    if not row.tee_filename:
        raise HTTPException(400, "upload has no tee video")
    src_path = CLIPS_DIR / row.tee_filename
    if not src_path.exists():
        raise HTTPException(404, f"tee source file missing on disk: {row.tee_filename}")

    saved = dict(row.edit_metrics or {})
    existing = list(saved.get("ball_track_frames") or [])
    manual_positions = payload.get("manual_positions") or []

    # Merge: frame index → entry. Manual wins. Manual additions for
    # frames the AI never visited are flagged manual=True / found=True
    # so the renderer treats them as confirmed points.
    by_frame: dict[int, dict] = {}
    for rec in existing:
        try:
            f = int(rec.get("frame"))
        except (TypeError, ValueError):
            continue
        by_frame[f] = {
            "frame": f,
            "found": bool(rec.get("found")),
            "x": rec.get("x"),
            "y": rec.get("y"),
            "manual": bool(rec.get("manual", False)),
            "image_url": rec.get("image_url"),
        }
    for mp in manual_positions:
        try:
            f = int(mp["frame"]); x = int(mp["x"]); y = int(mp["y"])
        except (KeyError, TypeError, ValueError):
            continue
        prior = by_frame.get(f, {})
        by_frame[f] = {
            "frame": f,
            "found": True,
            "x": x,
            "y": y,
            "manual": True,
            "image_url": prior.get("image_url"),
        }
    merged = sorted(by_frame.values(), key=lambda e: e["frame"])

    # Pull anchor data from saved metrics. ball_at_rest seeds the
    # starting point of the tracer line; impact_frame anchors the
    # 'when the line begins' moment.
    ball = saved.get("ball") or {}
    ball_xy = None
    try:
        if ball and ball.get("x") is not None and ball.get("y") is not None:
            ball_xy = (float(ball["x"]), float(ball["y"]))
    except Exception:
        ball_xy = None
    try:
        impact_idx = int(saved.get("impact_frame") or 0)
    except (TypeError, ValueError):
        impact_idx = 0

    output_path = CLIPS_DIR / f"wizard-{upload_id}_tracer.mp4"
    info = render_tracer_video(
        src_path,
        output_path,
        ball_rest_xy_native=ball_xy,
        impact_frame_idx=impact_idx,
        # Forward the manual flag — the renderer pins manual anchors
        # so the parabola fit can't reject them and weights them so
        # they actually shape the rendered arc instead of getting
        # outvoted by the AI's earlier points.
        track_frames=[
            {
                "frame": e["frame"], "found": e["found"],
                "x": e["x"], "y": e["y"],
                "manual": e.get("manual", False),
            }
            for e in merged
        ],
        # Operator-confirmed points should always be drawn — never
        # truncate at apex when the wizard is driving the render.
        extend_to_last_anchor=True,
    )
    if not info.get("ok"):
        raise HTTPException(500, f"tracer render failed: {info.get('error') or 'unknown'}")

    compress_for_email(output_path)
    tracer_url = (
        f"{settings.app_base_url}/uploads/clips/{output_path.name}"
        f"?v={int(output_path.stat().st_mtime)}"
    )

    saved["tracer_url"] = tracer_url
    saved["tracer_info"] = info
    saved["ball_track_frames"] = merged
    # Re-finalizing was previously baked from the stale tracer — drop
    # the cached final URL so Step 3 knows to re-apply graphics.
    saved.pop("finalized_video_url", None)
    row.edit_metrics = saved
    db.add(row)
    db.add(AuditLog(
        actor="admin", action="render_tracer_fast",
        target=f"long_upload:{upload_id}",
        detail=f"merged_frames={len(merged)} manual={len(manual_positions)}",
    ))
    db.commit()
    db.refresh(row)
    return {
        "upload_id": upload_id,
        "tracer_url": tracer_url,
        "ball_track_frames": merged,
        "edit_metrics": row.edit_metrics,
    }


@router.post("/long-uploads/{upload_id}/finalize")
def finalize_wizard_video(
    upload_id: int,
    payload: dict = Body(default={}),
    db: Session = Depends(get_db),
):
    """Apply the same on-screen graphics the AI-clips page produces
    (player banner + course/hole/par/yardage) on top of the wizard's
    rendered tracer. Writes a separate `wizard-{id}_final.mp4` so
    re-finalizing doesn't overwrite the raw tracer. Persists the
    final URL into edit_metrics.finalized_video_url and returns it.

    Optional body keys (override per-call):
      player_name (str): defaults to None (no name shown).
      hole_number (int): defaults to 1.
    """
    row = db.get(LongVideoUpload, upload_id)
    if not row:
        raise HTTPException(404, "long upload not found")
    saved = dict(row.edit_metrics or {})
    tracer_url = saved.get("tracer_url")
    if not tracer_url:
        raise HTTPException(
            400, "no rendered tracer yet — finish Step 2 first")
    fname = tracer_url.rstrip("/").split("?")[0].rsplit("/", 1)[-1]
    tracer_path = CLIPS_DIR / fname
    if not tracer_path.exists():
        raise HTTPException(404, f"tracer file missing on disk: {fname}")

    final_path = CLIPS_DIR / f"wizard-{upload_id}_final.mp4"
    # Copy tracer → final, then apply intro overlay in-place on the
    # copy. Keeps the bare tracer available for re-finalize / debug.
    try:
        import shutil
        shutil.copyfile(tracer_path, final_path)
    except Exception as exc:
        raise HTTPException(500, f"copy failed: {exc}")

    course = db.get(Course, row.course_id) if row.course_id else None
    course_name = course.name if course else ""
    hole_number = int(payload.get("hole_number") or 1)
    # yardage: prefer an explicit operator override → otherwise the
    # course's hole_yardages entry → otherwise 101.
    yardage_override = payload.get("yardage")
    try:
        yardage = int(yardage_override) if yardage_override is not None else None
    except (TypeError, ValueError):
        yardage = None
    if yardage is None:
        yardage = 101
        if course and course.hole_yardages:
            raw_y = course.hole_yardages.get(str(hole_number))
            try:
                if raw_y is not None:
                    yardage = int(raw_y)
            except (TypeError, ValueError):
                pass
    player_name = payload.get("player_name") or "Brent Baldwin"

    # Target pixel for the 'TO HOLE / N YDS' stake overlay. Pulled
    # from the wizard's saved target on Step 1 (red flag pin). The
    # saved coords are in the source video's native dims, but the
    # file we're about to overlay is the wizard tracer — which the
    # tracer-render pipeline has already passed through
    # compress_for_email and scaled to 1280px long-edge. Probe both
    # so the target gets placed in the right spot on the final file.
    target_xy: tuple[int, int] | None = None
    target_saved = saved.get("target") or {}
    try:
        if target_saved.get("x") is not None and target_saved.get("y") is not None:
            target_xy = (int(target_saved["x"]), int(target_saved["y"]))
    except (TypeError, ValueError):
        target_xy = None

    if target_xy is not None:
        original_target = tuple(target_xy)
        # Source-coord reference: prefer the dims the wizard explicitly
        # saved alongside the target (so they're guaranteed to match
        # the coord system target.x/y was clicked into). Fall back to
        # a cv2 probe for old rows that pre-date the explicit save.
        src_w_native = saved.get("frame_width")
        src_h_native = saved.get("frame_height")
        try:
            src_w_native = int(src_w_native) if src_w_native else None
            src_h_native = int(src_h_native) if src_h_native else None
        except (TypeError, ValueError):
            src_w_native = src_h_native = None
        if not (src_w_native and src_h_native):
            try:
                import cv2  # type: ignore
                _cap_src = cv2.VideoCapture(str(src_path))
                try:
                    src_w_native = int(_cap_src.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
                    src_h_native = int(_cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
                finally:
                    _cap_src.release()
            except Exception:  # pragma: no cover
                pass
        # Final-file dims (what ffmpeg overlay will treat as the canvas).
        final_info = probe_video_info(final_path)
        final_w = final_info.get("width")
        final_h = final_info.get("height")
        log.info(
            "finalize: target_xy=%s src(cv2)=%sx%s final(ffprobe)=%sx%s "
            "saved=%s",
            original_target, src_w_native, src_h_native,
            final_w, final_h, target_saved,
        )
        if (src_w_native and src_h_native and final_w and final_h
                and (src_w_native != final_w or src_h_native != final_h)):
            sx = float(final_w) / float(src_w_native)
            sy = float(final_h) / float(src_h_native)
            target_xy = (
                int(round(target_xy[0] * sx)),
                int(round(target_xy[1] * sy)),
            )
            log.info(
                "finalize: scaled target_xy %s → %s (factor %.3f, %.3f)",
                original_target, target_xy, sx, sy,
            )
        # Final safety net: if the resulting target lies outside the
        # actual canvas, the saved coords were almost certainly in a
        # different reference frame than we expect (e.g. a stale
        # auto-detect from before the native-dim fix that referenced
        # ~1024px). Re-scale assuming the *largest* of (frame-w, x,
        # original-x) is the actual ref width — gives a sensible
        # fallback instead of off-screen placement.
        if (final_w and final_h
                and (target_xy[0] >= final_w or target_xy[1] >= final_h)):
            ref_w = max(final_w, target_xy[0] + 1, original_target[0] + 1)
            ref_h = max(final_h, target_xy[1] + 1, original_target[1] + 1)
            sx = float(final_w) / float(ref_w)
            sy = float(final_h) / float(ref_h)
            target_xy = (
                int(round(original_target[0] * sx)),
                int(round(original_target[1] * sy)),
            )
            log.warning(
                "finalize: target was off-canvas; rescaled assuming ref %sx%s "
                "→ %s",
                ref_w, ref_h, target_xy,
            )

    try:
        apply_intro_overlay_inplace(
            final_path,
            player_name=player_name,
            course_name=course_name,
            hole_number=hole_number,
            par=3,
            yardage=yardage,
            target_xy=target_xy,
        )
    except Exception as exc:  # pragma: no cover
        log.warning("finalize: intro overlay failed for upload %s: %s",
                    upload_id, exc)

    compress_for_email(final_path)
    final_url = (
        f"{settings.app_base_url}/uploads/clips/{final_path.name}"
        f"?v={int(final_path.stat().st_mtime)}"
    )

    saved["finalized_video_url"] = final_url
    saved["finalized_player_name"] = player_name
    saved["finalized_hole_number"] = hole_number
    saved["finalized_yardage"] = yardage
    row.edit_metrics = saved
    db.add(row)
    db.add(AuditLog(
        actor="admin", action="finalize_wizard_video",
        target=f"long_upload:{upload_id}",
        detail=f"final={final_path.name} hole={hole_number} player={player_name}",
    ))
    db.commit()
    db.refresh(row)
    return {
        "upload_id": upload_id,
        "final_video_url": final_url,
        "edit_metrics": row.edit_metrics,
    }


@router.post("/long-uploads/{upload_id}/commit")
def commit_wizard_clip(
    upload_id: int,
    db: Session = Depends(get_db),
):
    """Promote the wizard's finalized video to a real VideoClip so it
    shows up on Produced Clips and Broadcast. Idempotent: re-running
    after Save just updates the existing clip's tracer_url + sets
    delivered_at. Returns the clip id + URLs.
    """
    row = db.get(LongVideoUpload, upload_id)
    if not row:
        raise HTTPException(404, "long upload not found")
    saved = dict(row.edit_metrics or {})
    final_url = saved.get("finalized_video_url")
    tracer_url = saved.get("tracer_url")
    if not final_url:
        raise HTTPException(400, "no finalized video — run Step 3 first")

    # Re-use any existing wizard clip on this upload (avoids duplicates
    # when the operator Save's twice).
    clip = (
        db.query(VideoClip)
        .filter(VideoClip.long_upload_id == upload_id)
        .order_by(VideoClip.created_at.desc())
        .first()
    )
    hole_number = int(saved.get("finalized_hole_number") or 1)

    # Poster thumbnail for the Produced Video tile on /admin/production
    # (and any other "produced clip" listing). Extracts a JPG next to
    # the final video file; falls back to None so the UI shows the
    # "No preview" placeholder if ffmpeg isn't available.
    thumb_url: str | None = None
    final_fname = (final_url or "").rstrip("/").split("?")[0].rsplit("/", 1)[-1]
    if final_fname:
        final_path = CLIPS_DIR / final_fname
        if final_path.exists():
            try:
                thumb_path = extract_thumbnail(final_path)
                if thumb_path is not None:
                    thumb_url = (
                        f"{settings.app_base_url}/uploads/clips/{thumb_path.name}"
                        f"?v={int(thumb_path.stat().st_mtime)}"
                    )
            except Exception as exc:  # pragma: no cover
                log.warning("commit_wizard_clip: thumbnail extract failed: %s", exc)

    # The Broadcast page (and SMS / share / download buttons) play
    # clip.source_url — so that URL has to be the finalized video
    # with the intro overlay graphics baked in. The bare tracer
    # (no overlay) lives on tee_clip_url for the AI-tracer page to
    # re-iterate against.
    if not clip:
        clip = VideoClip(
            course_id=row.course_id,
            hole_number=hole_number,
            camera_type="tee",
            captured_at=row.base_captured_at or _utcnow_naive(),
            source_url=final_url,
            tracer_url=final_url,
            tee_clip_url=tracer_url,
            thumbnail_url=thumb_url,
            long_upload_id=upload_id,
            processing_status=ClipProcessingStatus.received.value,
        )
        db.add(clip)
    else:
        clip.tracer_url = final_url
        clip.source_url = final_url
        clip.tee_clip_url = tracer_url
        clip.hole_number = hole_number
        if thumb_url:
            clip.thumbnail_url = thumb_url
        if not clip.captured_at and row.base_captured_at:
            clip.captured_at = row.base_captured_at
    clip.delivered_at = _utcnow_naive()
    row.processing_status = "completed"
    row.processing_completed_at = _utcnow_naive()
    row.last_n_segments = 1
    row.last_n_succeeded = 1
    db.add(row)
    db.flush()

    db.add(AuditLog(
        actor="admin", action="commit_wizard_clip",
        target=f"long_upload:{upload_id}",
        detail=f"clip={clip.id} hole={hole_number}",
    ))
    db.commit()
    db.refresh(clip)
    return {
        "upload_id": upload_id,
        "clip_id": clip.id,
        "tracer_url": clip.tracer_url,
        "source_url": clip.source_url,
    }


@router.delete("/long-uploads/{upload_id}")
def delete_long_upload(upload_id: int, db: Session = Depends(get_db)):
    """Delete a stored long upload + its source file(s) from disk.
    Per-swing VideoClips produced from this upload are kept (they have
    their own VideoClip rows). Returns the freed disk bytes for
    confirmation."""
    row = db.get(LongVideoUpload, upload_id)
    if not row:
        raise HTTPException(404, "long upload not found")
    freed = 0
    for fname in (row.tee_filename, row.green_filename):
        if not fname:
            continue
        fp = CLIPS_DIR / fname
        try:
            if fp.exists():
                freed += fp.stat().st_size
                fp.unlink()
        except Exception as exc:
            log.warning("long-upload delete: failed to unlink %s: %s", fp, exc)
    db.delete(row)
    db.commit()
    return {"deleted": True, "freed_bytes": freed}


TESTCUTS_SUBDIR = "_testcuts"


@router.post("/long-uploads/{upload_id}/test-cut")
def test_cut_long_upload(
    upload_id: int,
    detector: str = Form("motion"),
    audio_min_peak_ratio: float = Form(10.0),
    motion_ratio: float = Form(4.0),
    combined_pair_window_sec: float = Form(3.0),
    cut_clips: bool = Form(True),
    db: Session = Depends(get_db),
):
    """Dry-run the swing-cutting half of the long-upload pipeline.

    Runs the chosen swing detector on the stored tee video, cuts each
    proposed window into a raw MP4 (no AI tracer, no matcher, no
    VideoClip row), and returns playable URLs so the operator can
    eyeball whether the cuts are right before paying for the full
    pipeline.

    Test cuts land in CLIPS_DIR/_testcuts/ so they don't pollute
    /admin/clips. Calling this again for the same upload wipes the
    previous test cuts first.

    detector: 'motion' (default) | 'audio'
    """
    row = db.get(LongVideoUpload, upload_id)
    if not row:
        raise HTTPException(404, "long upload not found")
    if not row.tee_filename:
        raise HTTPException(400, "long upload has no tee file")
    src_path = CLIPS_DIR / row.tee_filename
    if not src_path.exists():
        raise HTTPException(404, f"tee source file missing on disk: {row.tee_filename}")

    green_path: Path | None = None
    if row.green_filename:
        gp = CLIPS_DIR / row.green_filename
        if gp.exists():
            green_path = gp

    tee_fps = probe_fps(src_path) or 30.0
    detector = (detector or "motion").lower()
    debug: dict = {}
    if detector == "audio":
        windows = detect_swings_from_audio(
            src_path,
            fps=tee_fps,
            min_peak_ratio=float(audio_min_peak_ratio),
            debug=debug,
        )
    elif detector == "combined":
        windows = detect_swings_combined(
            src_path,
            fps=tee_fps,
            audio_min_peak_ratio=float(audio_min_peak_ratio),
            motion_ratio=float(motion_ratio),
            pair_window_sec=float(combined_pair_window_sec),
            debug=debug,
        )
    else:
        detector = "motion"
        windows = detect_swings_from_motion(
            src_path,
            fps=tee_fps,
            motion_ratio=float(motion_ratio),
            debug=debug,
        )

    # Wipe any previous test cuts for this upload so re-running the
    # detector doesn't accumulate stale files.
    testcuts_dir = CLIPS_DIR / TESTCUTS_SUBDIR
    testcuts_dir.mkdir(parents=True, exist_ok=True)
    for old in testcuts_dir.glob(f"testcut-{upload_id}-*.mp4"):
        try:
            old.unlink()
        except Exception:
            pass

    cuts: list[dict] = []
    t_loop_start = time.monotonic()
    n_windows_total = len(windows)
    for idx, w in enumerate(windows):
        start_sec = float(w["start_sec"])
        end_sec = float(w["end_sec"])
        peak_time_sec = (
            float(w.get("peak_time_sec"))
            if w.get("peak_time_sec") is not None else None
        )
        cut_entry: dict = {
            "index": idx,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": round(end_sec - start_sec, 2),
            "peak_time_sec": peak_time_sec,
            "ratio": w.get("ratio"),
            "confidence": w.get("confidence"),
            "burst_duration_sec": w.get("burst_duration_sec"),
        }
        if not cut_clips:
            # Detection-only mode: no ffmpeg, no files. Operator just
            # wants to see the peak times to debug the detector.
            cut_entry.update({
                "ok": None,
                "url": None,
                "green_url": None,
                "green_ok": None,
                "skipped": True,
            })
            cuts.append(cut_entry)
            continue

        tok = secrets.token_hex(4)
        tee_name = f"testcut-{upload_id}-{idx:02d}-tee-{tok}.mp4"
        tee_out = testcuts_dir / tee_name
        t_cut_start = time.monotonic()
        # Fast seek + stream copy: ~50x faster than the production
        # frame-accurate path. Snaps to nearest keyframe (~0-2s drift)
        # which is fine when the operator is just verifying that the
        # detector landed on real swings.
        tee_ok = cut_segment(src_path, tee_out, start_sec, end_sec, fast=True)

        green_url: str | None = None
        green_ok: bool | None = None
        if green_path is not None:
            green_name = f"testcut-{upload_id}-{idx:02d}-green-{tok}.mp4"
            green_out = testcuts_dir / green_name
            green_ok = cut_segment(green_path, green_out, start_sec, end_sec, fast=True)
            if green_ok:
                green_url = f"{settings.app_base_url}/uploads/clips/{TESTCUTS_SUBDIR}/{green_name}"

        log.info(
            "long-upload test-cut: upload=%s cut %d/%d [%.1f-%.1fs] tee=%s green=%s in %.1fs",
            upload_id, idx + 1, n_windows_total, start_sec, end_sec,
            "ok" if tee_ok else "fail",
            "ok" if green_ok else ("fail" if green_ok is False else "—"),
            time.monotonic() - t_cut_start,
        )

        cut_entry.update({
            "ok": bool(tee_ok),
            "url": (
                f"{settings.app_base_url}/uploads/clips/{TESTCUTS_SUBDIR}/{tee_name}"
                if tee_ok else None
            ),
            "green_url": green_url,
            "green_ok": green_ok,
        })
        cuts.append(cut_entry)

    log.info(
        "long-upload test-cut: upload=%s detector=%s windows=%d cut_clips=%s cuts_ok=%d total=%.1fs",
        upload_id, detector, len(windows), cut_clips,
        sum(1 for c in cuts if c.get("ok")),
        time.monotonic() - t_loop_start,
    )
    return {
        "upload_id": upload_id,
        "detector": detector,
        "tee_fps": tee_fps,
        "n_windows": len(windows),
        "dual_camera": green_path is not None,
        "cuts_skipped": not cut_clips,
        "debug": debug or None,
        "cuts": cuts,
    }


@router.post("/long-uploads/{upload_id}/process-segment")
def process_long_upload_segment(
    upload_id: int,
    hole_number: int = Form(...),
    start_sec: float = Form(...),
    end_sec: float = Form(...),
    ai_tracer_model: str | None = Form(None),
    db: Session = Depends(get_db),
):
    """Run the full per-segment pipeline on ONE detected window so the
    operator can promote individual test-cut previews to broadcast
    without re-processing the whole long upload.

    Runs synchronously (typically 30–90 s including AI tracer +
    composite) and returns the resulting VideoClip metadata.
    """
    row = db.get(LongVideoUpload, upload_id)
    if not row:
        raise HTTPException(404, "long upload not found")
    if not row.tee_filename:
        raise HTTPException(400, "long upload has no tee file")
    src_path = CLIPS_DIR / row.tee_filename
    if not src_path.exists():
        raise HTTPException(404, f"tee source file missing on disk: {row.tee_filename}")

    green_src_path: Path | None = None
    if row.green_filename:
        candidate = CLIPS_DIR / row.green_filename
        if not candidate.exists():
            raise HTTPException(
                404,
                f"green source file missing on disk: {row.green_filename}",
            )
        green_src_path = candidate

    if end_sec <= start_sec:
        raise HTTPException(400, "end_sec must be > start_sec")

    seg_list = [{
        "hole_number": int(hole_number),
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
    }]
    results = _process_long_upload_segments(
        db,
        course_id=row.course_id,
        camera_type=row.camera_type,
        base_dt=row.base_captured_at,
        src_path=src_path,
        green_src_path=green_src_path,
        seg_list=seg_list,
        dual_camera=green_src_path is not None,
        ai_tracer_model=ai_tracer_model,
    )
    if not results:
        raise HTTPException(500, "segment processing returned no result")
    return results[0]


def _optional_int(v):
    if v in (None, "", "null"):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _run_tracer(
    clip_path: Path,
    sensitivity: float = 1.0,
) -> tuple[str | None, dict | None, Path | None, str | None]:
    """Render the tracer overlay for clip_path.

    Returns (tracer_url, info, traced_path, debug_url). Best-effort: any
    failure here still returns a debug image URL so the operator can see
    what the detector is staring at (candidates circled in red, or a
    "0 candidates" overlay if the HSV/motion gates filtered everything).

    `sensitivity` (default 1.0) is forwarded to render_tracer so the
    AdminClips UI can iterate on detector strictness without redeploying.
    """
    if not have_tracer():
        return None, {"ok": False, "error": "opencv not installed"}, None, None
    traced_name = f"{clip_path.stem}_traced.mp4"
    traced_path = CLIPS_DIR / traced_name
    debug_name = f"{clip_path.stem}_candidates.jpg"
    debug_path = CLIPS_DIR / debug_name

    # Use the audio-based impact detector (peak/median ≥30, back off
    # AUDIO_IMPACT_PRE_PEAK_FRAMES) to anchor where ball flight starts.
    # The classical-CV tracer then drops every detection / track point
    # before that frame so the overlay only shows actual flight.
    fps_val = probe_fps(clip_path) or 30.0
    audio_impact = find_impact_via_audio(clip_path, fps_val)
    impact_hint = (
        int(audio_impact["impact_frame"])
        if audio_impact.get("ok") and audio_impact.get("impact_frame") is not None
        else None
    )
    log.info(
        "tracer: audio impact hint for %s — frame=%s (ratio=%.1f, ok=%s)",
        clip_path.name,
        impact_hint,
        audio_impact.get("ratio") or 0.0,
        audio_impact.get("ok"),
    )

    info = render_tracer(
        clip_path, traced_path, debug_path,
        impact_frame_hint=impact_hint,
        sensitivity=float(sensitivity),
    )
    info["audio_impact"] = audio_impact
    info["sensitivity"] = float(sensitivity)
    debug_url = (
        f"{settings.app_base_url}/uploads/clips/{debug_name}"
        if debug_path.exists() else None
    )
    if not info.get("ok"):
        traced_path.unlink(missing_ok=True)
        return None, info, None, debug_url
    # OpenCV writes mp4v; re-encode to H.264 + faststart for browser playback.
    compressed = compress_for_email(traced_path)
    if not compressed:
        log.warning(
            "tracer: compress_for_email returned False for %s — file likely still mp4v, browser playback may fail",
            traced_path.name,
        )
    if not traced_path.exists() or traced_path.stat().st_size == 0:
        return None, {"ok": False, "error": "post-encode produced empty file"}, None, debug_url
    # Probe the final file so we can spot mp4v-leftover or VFR-timestamp
    # bugs in the logs (symptoms: "still photo with claimed duration").
    probe = probe_video_info(traced_path)
    log.info(
        "tracer: traced output %s  codec=%s  fps=%s  nb_frames=%s  duration=%ss  size=%dB",
        traced_path.name, probe.get("codec"), probe.get("fps"),
        probe.get("nb_frames"), probe.get("duration"),
        traced_path.stat().st_size,
    )
    # Cache-bust the served URLs with the file mtime. Each retry rewrites
    # the same filename, and some browsers refuse to re-init the <video>
    # decoder when src stays identical — they sit on the old decoded
    # state and the new bytes never get rendered. Appending a version
    # query string forces the element to treat it as a new resource.
    traced_mtime = int(traced_path.stat().st_mtime)
    debug_mtime = int(debug_path.stat().st_mtime) if debug_path.exists() else traced_mtime
    debug_url = (
        f"{settings.app_base_url}/uploads/clips/{debug_name}?v={debug_mtime}"
        if debug_path.exists() else None
    )
    return (
        f"{settings.app_base_url}/uploads/clips/{traced_name}?v={traced_mtime}",
        info,
        traced_path,
        debug_url,
    )


@router.post("/clips/upload")
async def upload_clip(
    course_id: int = Form(...),
    hole_number: int = Form(...),
    camera_type: str = Form("tee"),
    captured_at: str = Form(...),  # ISO datetime, e.g. "2026-04-17T14:32:11Z"
    carry_yards: int | None = Form(None),
    apex_feet: int | None = Form(None),
    ball_speed_mph: int | None = Form(None),
    distance_from_pin_feet: int | None = Form(None),
    ball_in_cup: bool = Form(False),
    already_traced: bool = Form(False),
    video: UploadFile = File(...),
    video_green: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    """Manual upload that mimics the Shot Tracer webhook.

    Saves the video to backend/uploads/clips/, creates a VideoClip row,
    runs the appearance matcher, and fires gallery-ready notifications
    just like the real webhook would. Useful for testing the pipeline
    end-to-end with phone or GoPro footage before any real cameras are
    deployed.

    If `video_green` is also supplied, both cameras are assumed to have
    started recording at the same wall-clock moment. The tracer runs on
    each side independently. The deliverable clip is a composite:
    tee-cam shown from t=0 until 3 seconds after the ball leaves the
    tee, then a hard cut to green-cam at the matching wall-clock time
    until 3 seconds after the ball lands. The composite (with tracer
    overlay on both halves) becomes the gallery clip; tee + green raws
    are kept on disk for audit but not surfaced to the golfer.
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

    # When the operator says the clip is already traced (e.g. rendered
    # in the Shot Tracer iOS/Android app before upload), skip our
    # classical-CV pipeline entirely. The uploaded file IS the
    # deliverable — point both source_url and tracer_url at it.
    if already_traced:
        tracer_url = f"{settings.app_base_url}/uploads/clips/{fname}"
        tracer_info = {"ok": True, "source": "external", "note": "already_traced upload"}
        tee_traced_path = None
        tracer_debug_url = None
    else:
        # Render the classical-CV tracer overlay on the tee clip. Sync —
        # admin uploads are already a long-running request and the operator
        # wants to see the result. If detection fails, tracer_url stays null
        # and the original clip is still saved + delivered.
        tracer_url, tracer_info, tee_traced_path, tracer_debug_url = _run_tracer(fpath)

    # Dual-camera path: when a green-side clip is also uploaded, both
    # cameras are assumed to have started at the same moment. We run the
    # tracer on the green clip independently, then concat
    # tee[0..launch+3s] + green[launch+3s..landing+3s] into a single
    # composite. The composite replaces source_url / tracer_url so the
    # golfer's gallery shows the dual-angle deliverable.
    green_url = None
    green_tracer_url = None
    green_tracer_info = None
    green_debug_url = None
    composite_url = None
    composite_info: dict | None = None
    if video_green is not None and not already_traced:
        green_data = await video_green.read()
        if green_data:
            if len(green_data) > 500 * 1024 * 1024:
                raise HTTPException(413, "green video too large (max 500MB)")
            g_ext = (video_green.filename or "").rsplit(".", 1)[-1].lower() if "." in (video_green.filename or "") else "mp4"
            if g_ext not in ("mp4", "mov", "webm", "m4v"):
                g_ext = "mp4"
            green_name = f"{course_id}-h{hole_number}-{secrets.token_hex(6)}_green.{g_ext}"
            green_path = CLIPS_DIR / green_name
            green_path.write_bytes(green_data)
            compress_for_email(green_path)
            green_url = f"{settings.app_base_url}/uploads/clips/{green_name}"

            green_tracer_url, green_tracer_info, green_traced_path, green_debug_url = _run_tracer(green_path)

            if (
                tracer_info and tracer_info.get("ok")
                and green_tracer_info and green_tracer_info.get("ok")
                and tee_traced_path is not None and green_traced_path is not None
            ):
                tee_fps = float(tracer_info.get("fps") or 30.0) or 30.0
                green_fps = float(green_tracer_info.get("fps") or 30.0) or 30.0
                tee_launch_frame = tracer_info["frame_range"][0]
                green_land_frame = green_tracer_info["frame_range"][1]
                switch_sec = max(0.0, tee_launch_frame / tee_fps + 3.0)
                end_sec_in_green = green_land_frame / green_fps + 3.0
                # Sanity: the green cut window must have positive duration.
                if end_sec_in_green > switch_sec + 0.1:
                    composite_name = f"{fpath.stem}_composite.mp4"
                    composite_path = CLIPS_DIR / composite_name
                    if concat_two_clips(
                        tee_traced_path, 0.0, switch_sec,
                        green_traced_path, switch_sec, end_sec_in_green,
                        composite_path,
                    ):
                        composite_url = f"{settings.app_base_url}/uploads/clips/{composite_name}"
                        composite_info = {
                            "switch_sec": round(switch_sec, 2),
                            "end_sec": round(end_sec_in_green, 2),
                            "tee_fps": round(tee_fps, 2),
                            "green_fps": round(green_fps, 2),
                        }

    try:
        captured_dt = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        if captured_dt.tzinfo is not None:
            # Normalize to naive UTC since the column is TIMESTAMP WITHOUT TIME ZONE
            captured_dt = captured_dt.astimezone().replace(tzinfo=None)
    except ValueError:
        raise HTTPException(400, "invalid captured_at; use ISO 8601")

    # If we built a dual-camera composite, that IS the gallery clip —
    # use it for both source_url and tracer_url so it plays everywhere
    # the golfer or broadcast channel pulls clips from.
    final_source = composite_url or f"{settings.app_base_url}/uploads/clips/{fname}"
    final_tracer = composite_url or tracer_url
    clip = VideoClip(
        course_id=course_id,
        hole_number=hole_number,
        camera_type=camera_type,
        captured_at=captured_dt,
        source_url=final_source,
        thumbnail_url=thumb_url,
        tracer_url=final_tracer,
        carry_yards=carry_yards,
        apex_feet=apex_feet,
        ball_speed_mph=ball_speed_mph,
        distance_from_pin_feet=distance_from_pin_feet,
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
        "tracer_url": clip.tracer_url,
        "tracer_info": tracer_info,
        "tracer_debug_url": tracer_debug_url,
        "issue_note": clip.issue_note,
        "tee_raw_url": f"{settings.app_base_url}/uploads/clips/{fname}",
        "green_raw_url": green_url,
        "green_tracer_url": green_tracer_url,
        "green_tracer_info": green_tracer_info,
        "green_debug_url": green_debug_url,
        "composite_url": composite_url,
        "composite_info": composite_info,
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


# ----------------------------------------------------------------------
# Camera management (phase 1 of the on-course always-on hardware
# integration). Devices in the field auth via per-camera token on
# /api/cameras/{token}/... endpoints (phase 2); this section is the
# operator-facing CRUD that those tokens are minted by.
# ----------------------------------------------------------------------

_CAMERA_ROLES = ("tee", "green")


def _camera_to_dict(c: Camera, last_event: CameraEvent | None = None) -> dict:
    """Shape a Camera row for the admin UI. Includes the auth_token
    because operators need it to provision the Pi's SD card."""
    return {
        "id": c.id,
        "course_id": c.course_id,
        "assigned_hole": c.assigned_hole,
        "assigned_role": c.assigned_role,
        "paired_with_camera_id": c.paired_with_camera_id,
        "auth_token": c.auth_token,
        "name": c.name,
        "tee_box_roi": c.tee_box_roi,
        "last_seen_at": c.last_seen_at.isoformat() if c.last_seen_at else None,
        "firmware_version": c.firmware_version,
        "enabled": bool(c.enabled),
        "note": c.note,
        "created_at": c.created_at.isoformat() if c.created_at else None,
        "last_event_at": (
            last_event.triggered_at.isoformat()
            if last_event and last_event.triggered_at else None
        ),
        "last_event_status": last_event.status if last_event else None,
    }


@router.get("/cameras")
def list_cameras(db: Session = Depends(get_db)):
    """Every registered camera, newest first. Includes the
    most recent CameraEvent's status / timestamp for at-a-glance
    health visibility."""
    cams = db.query(Camera).order_by(Camera.created_at.desc()).all()
    out: list[dict] = []
    for c in cams:
        last_evt = (
            db.query(CameraEvent)
            .filter(
                (CameraEvent.tee_camera_id == c.id)
                | (CameraEvent.green_camera_id == c.id)
            )
            .order_by(CameraEvent.triggered_at.desc())
            .first()
        )
        out.append(_camera_to_dict(c, last_evt))
    return out


@router.post("/cameras")
def create_camera(
    course_id: int = Form(...),
    assigned_hole: int = Form(...),
    assigned_role: str = Form(...),
    name: str = Form(""),
    db: Session = Depends(get_db),
):
    """Mint a new camera + auth_token. The operator runs this once per
    physical device (or when rotating a stolen / lost device's token)
    and uses the returned auth_token to provision the Pi's SD card."""
    role = (assigned_role or "").strip().lower()
    if role not in _CAMERA_ROLES:
        raise HTTPException(400, f"assigned_role must be one of {_CAMERA_ROLES}")
    course = db.get(Course, int(course_id))
    if not course:
        raise HTTPException(404, "course not found")
    hole = int(assigned_hole)
    if hole < 1 or hole > 18:
        raise HTTPException(400, "assigned_hole must be 1..18")
    cam = Camera(
        course_id=course.id,
        assigned_hole=hole,
        assigned_role=role,
        name=(name or "").strip()[:120],
    )
    db.add(cam)
    db.flush()
    db.add(AuditLog(
        actor="admin", action="create_camera", target=f"camera:{cam.id}",
        detail=f"course={course.id} hole={hole} role={role}",
    ))
    db.commit()
    db.refresh(cam)
    return _camera_to_dict(cam)


@router.post("/cameras/{camera_id}/pair")
def pair_camera(
    camera_id: int,
    partner_id: int = Form(...),
    db: Session = Depends(get_db),
):
    """Mark two cameras as paired (one tee + one green, same course
    and hole). The relationship is mirrored — both rows point at each
    other — so the event-trigger relay can look up the partner from
    either side."""
    cam = db.get(Camera, camera_id)
    partner = db.get(Camera, int(partner_id))
    if not cam or not partner:
        raise HTTPException(404, "camera not found")
    if cam.id == partner.id:
        raise HTTPException(400, "camera cannot pair with itself")
    if cam.course_id != partner.course_id or cam.assigned_hole != partner.assigned_hole:
        raise HTTPException(400, "cameras must share course + hole to pair")
    if {cam.assigned_role, partner.assigned_role} != set(_CAMERA_ROLES):
        raise HTTPException(400, "pair must be exactly one tee + one green")
    # Mirror the link on both sides.
    cam.paired_with_camera_id = partner.id
    partner.paired_with_camera_id = cam.id
    db.add(AuditLog(
        actor="admin", action="pair_cameras",
        target=f"camera:{cam.id}", detail=f"partner={partner.id}",
    ))
    db.commit()
    return {"ok": True, "cameras": [_camera_to_dict(cam), _camera_to_dict(partner)]}


@router.post("/cameras/{camera_id}/unpair")
def unpair_camera(camera_id: int, db: Session = Depends(get_db)):
    """Clear pairing on both sides."""
    cam = db.get(Camera, camera_id)
    if not cam:
        raise HTTPException(404, "camera not found")
    partner = (
        db.get(Camera, cam.paired_with_camera_id)
        if cam.paired_with_camera_id else None
    )
    cam.paired_with_camera_id = None
    if partner is not None:
        partner.paired_with_camera_id = None
    db.commit()
    return {"ok": True}


@router.post("/cameras/{camera_id}/rotate-token")
def rotate_camera_token(camera_id: int, db: Session = Depends(get_db)):
    """Mint a new auth_token for this camera (invalidating the old
    one). Used when a device is lost / stolen / suspected compromised
    — re-provision the Pi with the new token."""
    cam = db.get(Camera, camera_id)
    if not cam:
        raise HTTPException(404, "camera not found")
    # Import locally so the module's lazy enough that test bench scripts
    # can import admin without pulling models' _token helper exposure.
    from ..models import _token as _make_token
    cam.auth_token = _make_token("cam_", 24)
    db.add(AuditLog(
        actor="admin", action="rotate_camera_token", target=f"camera:{cam.id}",
        detail="",
    ))
    db.commit()
    db.refresh(cam)
    return {"ok": True, "auth_token": cam.auth_token}


@router.post("/cameras/{camera_id}/update")
def update_camera(
    camera_id: int,
    name: str | None = Form(None),
    enabled: bool | None = Form(None),
    tee_box_roi: str | None = Form(None),  # JSON string {"x":N,"y":N,"w":N,"h":N}
    note: str | None = Form(None),
    course_id: int | None = Form(None),
    assigned_hole: int | None = Form(None),
    assigned_role: str | None = Form(None),
    db: Session = Depends(get_db),
):
    """Patch a camera's display name / enabled flag / tee-box ROI /
    note / course / hole / role. Each field is optional; only sent
    fields get applied.

    Moving a camera to a different course, hole, or role auto-unpairs
    it if the existing pair would no longer be valid (pair must share
    course + hole + have exactly one tee + one green). The partner
    side is unlinked too so it doesn't end up pointing at a camera
    that's not actually its pair anymore.
    """
    cam = db.get(Camera, camera_id)
    if not cam:
        raise HTTPException(404, "camera not found")
    if name is not None:
        cam.name = name.strip()[:120]
    if enabled is not None:
        cam.enabled = bool(enabled)
    if note is not None:
        cam.note = note.strip() or None
    if tee_box_roi is not None and tee_box_roi.strip():
        try:
            roi = json.loads(tee_box_roi)
        except json.JSONDecodeError:
            raise HTTPException(400, "tee_box_roi must be valid JSON")
        if not isinstance(roi, dict) or not all(k in roi for k in ("x", "y", "w", "h")):
            raise HTTPException(400, "tee_box_roi must be an object with x/y/w/h")
        cam.tee_box_roi = roi

    # Track placement changes (course / hole / role) so we can
    # auto-unpair if the existing pair would no longer be valid.
    placement_changed = False
    if course_id is not None:
        target_course = db.get(Course, int(course_id))
        if not target_course:
            raise HTTPException(404, "target course not found")
        if cam.course_id != target_course.id:
            cam.course_id = target_course.id
            placement_changed = True
    if assigned_hole is not None:
        hole = int(assigned_hole)
        if hole < 1 or hole > 18:
            raise HTTPException(400, "assigned_hole must be 1..18")
        if cam.assigned_hole != hole:
            cam.assigned_hole = hole
            placement_changed = True
    if assigned_role is not None:
        role = assigned_role.strip().lower()
        if role not in _CAMERA_ROLES:
            raise HTTPException(400, f"assigned_role must be one of {_CAMERA_ROLES}")
        if cam.assigned_role != role:
            cam.assigned_role = role
            placement_changed = True

    auto_unpaired = False
    if placement_changed and cam.paired_with_camera_id:
        partner = db.get(Camera, cam.paired_with_camera_id)
        if partner is None:
            cam.paired_with_camera_id = None  # stale link
        else:
            pair_still_valid = (
                cam.course_id == partner.course_id
                and cam.assigned_hole == partner.assigned_hole
                and {cam.assigned_role, partner.assigned_role} == set(_CAMERA_ROLES)
            )
            if not pair_still_valid:
                cam.paired_with_camera_id = None
                partner.paired_with_camera_id = None
                auto_unpaired = True

    if placement_changed:
        db.add(AuditLog(
            actor="admin", action="move_camera", target=f"camera:{cam.id}",
            detail=(
                f"course={cam.course_id} hole={cam.assigned_hole} "
                f"role={cam.assigned_role}"
                + (" (auto-unpaired)" if auto_unpaired else "")
            ),
        ))
    db.commit()
    db.refresh(cam)
    result = _camera_to_dict(cam)
    result["auto_unpaired"] = auto_unpaired
    return result


@router.delete("/cameras/{camera_id}")
def delete_camera(camera_id: int, db: Session = Depends(get_db)):
    """Hard-delete a camera row. Any partner's paired_with link is
    cleared first so the partner doesn't end up pointing at a missing
    row. CameraEvent rows referencing this camera are preserved (for
    audit), but the FK will fail if you try to recreate one with the
    same id — fine for normal use."""
    cam = db.get(Camera, camera_id)
    if not cam:
        raise HTTPException(404, "camera not found")
    if cam.paired_with_camera_id:
        partner = db.get(Camera, cam.paired_with_camera_id)
        if partner is not None:
            partner.paired_with_camera_id = None
    db.add(AuditLog(
        actor="admin", action="delete_camera", target=f"camera:{cam.id}",
        detail=f"course={cam.course_id} hole={cam.assigned_hole} role={cam.assigned_role}",
    ))
    db.delete(cam)
    db.commit()
    return {"deleted": True, "camera_id": camera_id}
