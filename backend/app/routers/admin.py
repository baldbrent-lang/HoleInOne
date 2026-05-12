from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger("golfreelz.admin")

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
from ..services.auth import hash_password
from ..services.stripe_service import refund_payment_intent
from ..services.tracer import have_tracer, render_tracer
from ..services.ai_tracer import (
    find_address_frame,
    detect_handedness_at_address,
    annotate_address_with_shaft,
    find_impact_frame_after_address,
    refine_impact_frame,
)
from ..services.video import compress_for_email, concat_two_clips, cut_segment, extract_thumbnail, probe_video_info

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
            "processing_status": c.processing_status,
            "participant_id": c.participant_id,
            "participant_name": participant.name if participant else None,
        })
    return out


@router.post("/clips/{clip_id}/retry-tracer")
def retry_tracer(clip_id: int, db: Session = Depends(get_db)):
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
    if not clip.source_url:
        raise HTTPException(400, "clip has no source_url")
    # Pull the file name from the URL — we stored it as
    # {base}/uploads/clips/{fname}. Use the URL's last segment.
    fname = clip.source_url.rstrip("/").rsplit("/", 1)[-1]
    if not fname:
        raise HTTPException(400, "could not parse filename from source_url")
    if "_composite" in fname:
        raise HTTPException(
            400, "retry-tracer doesn't yet support composite clips (need raw halves)",
        )
    fpath = CLIPS_DIR / fname
    if not fpath.exists():
        raise HTTPException(404, f"source file missing on disk: {fname}")
    tracer_url, tracer_info, _, tracer_debug_url = _run_tracer(fpath)
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


@router.post("/clips/{clip_id}/ai-trace")
def ai_trace(clip_id: int, db: Session = Depends(get_db)):
    """AI analysis — step 2: identify the golfer's address frame.

    Camera is always behind the golfer. This endpoint asks Claude
    which frame in the clip shows the golfer at address (set up
    over the ball, just before takeaway). The picked frame is
    saved as a JPEG so the page can display it.
    """
    clip = db.get(VideoClip, clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    if not clip.source_url:
        raise HTTPException(400, "clip has no source_url")
    fname = clip.source_url.rstrip("/").rsplit("/", 1)[-1]
    if not fname:
        raise HTTPException(400, "could not parse filename from source_url")
    if "_composite" in fname:
        raise HTTPException(
            400, "ai-trace doesn't support composite clips (need raw halves)",
        )
    fpath = CLIPS_DIR / fname
    if not fpath.exists():
        raise HTTPException(404, f"source file missing on disk: {fname}")

    image_name = f"{fpath.stem}_address.jpg"
    image_path = CLIPS_DIR / image_name
    address_info = find_address_frame(fpath, output_image_path=image_path)

    image_url = None
    if address_info.get("saved_image") and image_path.exists():
        mtime = int(image_path.stat().st_mtime)
        image_url = f"{settings.app_base_url}/uploads/clips/{image_name}?v={mtime}"

    # Step 3: once we have the address frame, ask Claude what handedness
    # the golfer is. Claude reports the (x, y) of hands + ball so we
    # can overlay the shaft on the displayed address frame and visually
    # verify what Claude saw.
    handedness_info: dict | None = None
    impact_info: dict | None = None
    refined_impact_info: dict | None = None
    impact_image_url: str | None = None
    addr_idx_int: int | None = None
    if address_info.get("ok") and address_info.get("address_frame") is not None:
        addr_idx_int = int(address_info["address_frame"])
        handedness_info = detect_handedness_at_address(fpath, addr_idx_int)
        if handedness_info.get("ok"):
            wrote = annotate_address_with_shaft(
                fpath, addr_idx_int, handedness_info, image_path,
            )
            if wrote and image_path.exists():
                # Refresh the cache-buster so the browser picks up the
                # annotated image instead of the clean one it may have
                # already cached.
                mtime = int(image_path.stat().st_mtime)
                image_url = f"{settings.app_base_url}/uploads/clips/{image_name}?v={mtime}"

        # Step 4: find the impact frame. Pass the ball's starting
        # position from the handedness pass (in its sent-image coords)
        # so the impact function can draw a blue circle at that spot
        # on each candidate frame.
        ball_xy_sent = None
        ball_sent_dims = None
        if handedness_info and handedness_info.get("ok"):
            bx = handedness_info.get("ball_x")
            by = handedness_info.get("ball_y")
            sw = handedness_info.get("image_width")
            sh = handedness_info.get("image_height")
            if bx is not None and by is not None and sw and sh:
                ball_xy_sent = (float(bx), float(by))
                ball_sent_dims = (int(sw), int(sh))

        impact_image_name = f"{fpath.stem}_impact.jpg"
        impact_image_path = CLIPS_DIR / impact_image_name
        # Step 4a: rough impact — 12 candidates evenly across [+1, +~2s].
        # We don't save the image here; the refinement step below
        # overwrites this path with the final pick (shaft overlay too).
        impact_info = find_impact_frame_after_address(
            fpath, addr_idx_int,
            ball_xy_sent=ball_xy_sent,
            ball_sent_dims=ball_sent_dims,
            output_image_path=None,
        )

        # Step 4b: refinement — ±5 around the rough pick. Same blue
        # ball-rest circle on every candidate so Claude can lock onto
        # the precise frame where the clubhead is back at the ball.
        # The refinement call also reports hands+clubhead landmarks on
        # the picked frame so we can overlay the shaft.
        refined_impact_info = None
        if impact_info.get("ok") and impact_info.get("impact_frame") is not None:
            refined_impact_info = refine_impact_frame(
                fpath, int(impact_info["impact_frame"]),
                ball_xy_sent=ball_xy_sent,
                ball_sent_dims=ball_sent_dims,
                output_image_path=impact_image_path,
            )
            if refined_impact_info.get("saved_image") and impact_image_path.exists():
                impact_mtime = int(impact_image_path.stat().st_mtime)
                impact_image_url = (
                    f"{settings.app_base_url}/uploads/clips/{impact_image_name}?v={impact_mtime}"
                )

    db.add(AuditLog(
        actor="admin", action="ai_trace_address", target=f"clip:{clip.id}",
        detail=str({
            "address": address_info,
            "handedness": handedness_info,
            "impact": impact_info,
            "impact_refined": refined_impact_info,
        }),
    ))
    db.commit()

    return {
        "clip_id": clip.id,
        "source_url": clip.source_url,
        "address": address_info,
        "address_image_url": image_url,
        "handedness": handedness_info,
        "impact": impact_info,
        "impact_refined": refined_impact_info,
        "impact_image_url": impact_image_url,
    }


@router.post("/clips/long-upload")
async def upload_long_video(
    course_id: int = Form(...),
    camera_type: str = Form("tee"),
    base_captured_at: str = Form(...),
    segments: str = Form(...),
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Cut a long video into multiple per-swing clips and run each through
    the standard match + deliver pipeline.

    Body (multipart):
      course_id: int
      camera_type: 'tee' | 'wide_green' | 'hole'
      base_captured_at: ISO 8601 — when the recording started. Each segment's
                       captured_at = base + start_sec.
      segments: JSON array of {hole_number, start_sec, end_sec, ...stats}
      video: long MP4 file
    """
    course = db.get(Course, course_id)
    if not course:
        raise HTTPException(404, "course not found")
    if not (video.content_type or "").startswith("video/"):
        raise HTTPException(400, "must be a video file")

    try:
        seg_list = json.loads(segments or "[]")
    except json.JSONDecodeError:
        raise HTTPException(400, "segments must be a JSON array")
    if not isinstance(seg_list, list) or not seg_list:
        raise HTTPException(400, "at least one segment is required")

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

    results = []
    try:
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
    finally:
        # Clean up the long source — we don't keep it after cutting.
        src_path.unlink(missing_ok=True)

    return {"results": results}


def _optional_int(v):
    if v in (None, "", "null"):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _run_tracer(clip_path: Path) -> tuple[str | None, dict | None, Path | None, str | None]:
    """Render the tracer overlay for clip_path.

    Returns (tracer_url, info, traced_path, debug_url). Best-effort: any
    failure here still returns a debug image URL so the operator can see
    what the detector is staring at (candidates circled in red, or a
    "0 candidates" overlay if the HSV/motion gates filtered everything).
    """
    if not have_tracer():
        return None, {"ok": False, "error": "opencv not installed"}, None, None
    traced_name = f"{clip_path.stem}_traced.mp4"
    traced_path = CLIPS_DIR / traced_name
    debug_name = f"{clip_path.stem}_candidates.jpg"
    debug_path = CLIPS_DIR / debug_name

    info = render_tracer(clip_path, traced_path, debug_path)
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
