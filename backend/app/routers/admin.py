from __future__ import annotations

import functools
import itertools
import json
import logging
import math
import os
import queue
import secrets
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger("golfreelz.admin")

# ---------------------------------------------------------------------------
# Background source-file rehydration
# ---------------------------------------------------------------------------
# After a redeploy the local CLIPS_DIR is empty; source files live in object
# storage.  When the list endpoint detects a file is in the bucket but not on
# local disk it schedules a background download so subsequent page loads show
# full metadata.  We keep a dedup set so the same file isn't downloaded twice
# concurrently.
_rehydrate_pending: set[str] = set()
_rehydrate_lock = threading.Lock()


def _rehydrate_background(clips_dir: Path, filename: str) -> None:
    """Ensure `filename` is on local disk, downloading from the bucket in a
    daemon thread.  Idempotent — ignores a request already in flight."""
    from ..services import storage  # local import avoids circular at module load

    with _rehydrate_lock:
        if filename in _rehydrate_pending:
            return
        _rehydrate_pending.add(filename)

    def _run() -> None:
        try:
            storage.ensure_local(clips_dir, filename)
        except Exception as exc:  # noqa: BLE001
            log.debug("rehydrate: could not download %s: %s", filename, exc)
        finally:
            with _rehydrate_lock:
                _rehydrate_pending.discard(filename)

    threading.Thread(target=_run, name=f"rehydrate-{filename}", daemon=True).start()

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
)
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..config import settings
from ..database import SessionLocal, get_db
from ..deps import require_admin
from .cameras import _LIVE_FRAMES, _WATCHERS, _LIVE_LOCK, WATCH_TTL, FRAME_TTL
from .cameras import battery_status as _battery_status
from ..models import (
    AuditLog,
    Camera,
    CameraEvent,
    ClipProcessingStatus,
    DeletedCameraSession,
    Course,
    HIOStatus,
    HoleInOneEvent,
    LongVideoUpload,
    BroadcastView,
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
from ..services import notifications, storage, tracer_examples
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
    judge_swing_heat_image,
    trace_ball_path_ai,
    render_tracer_video,
    run_full_ai_tracer_pipeline,
    detect_swings_from_audio,
    detect_swings_from_motion,
    detect_swings_from_ball,
    detect_swings_from_ai_ball,
    classify_swing_shot,
    find_resting_ball,
    compute_motion_trace,
    detect_swings_combined,
    filter_swings_by_ball_departure,
)
from ..services.video import (
    CLIP_SECONDS_BEFORE_IMPACT,
    CLIP_SECONDS_GREEN_AFTER_CUT,
    CLIP_SECONDS_TEE_AFTER_IMPACT,
    CLIP_SECONDS_TEE_ONLY_AFTER_IMPACT,
    compress_for_email,
    concat_two_clips,
    cut_segment,
    extract_thumbnail,
    make_vertical,
    make_vertical_pan,
    probe_fps,
    probe_source_device,
    probe_video_info,
    splice_impact_clip,
    transcode_for_web,
)
from ..services.intro_overlay import apply_intro_overlay_inplace

# ── produce phase clock ────────────────────────────────────────────────
# Stage 7 of the Debug3 panel — the real produce run — was 82.5% of a
# 240s report and a single opaque block, which is the wrong shape for the
# one number that decides how long an operator waits at the course. This
# breaks it down.
#
# THREAD-LOCAL on purpose. A produce job owns its thread (stage 7 runs
# synchronously on the Debug3 worker; /reprocess spawns its own), so each
# run accumulates into its own bucket with no locking and no chance of
# two concurrent uploads polluting each other's numbers.
_produce_clock = threading.local()


def _pt_add(name: str, secs: float) -> None:
    """Add `secs` to this thread's `name` bucket."""
    buckets = getattr(_produce_clock, "phases", None)
    if buckets is None:
        buckets = _produce_clock.phases = {}
    buckets[name] = round(buckets.get(name, 0.0) + float(secs), 3)


@contextmanager
def _pt(name: str):
    """Time a block into this thread's produce clock."""
    _t0 = time.perf_counter()
    try:
        yield
    finally:
        _pt_add(name, time.perf_counter() - _t0)


def _pt_reset() -> None:
    _produce_clock.phases = {}


def _pt_snapshot() -> dict:
    return dict(getattr(_produce_clock, "phases", None) or {})


def _timed(fn, name: str):
    """Wrap a helper so EVERY call site is timed, not the handful we
    remember to annotate. The video helpers are called from a dozen
    places across the produce path (and from inside _trace_segment);
    rebinding the imported name here catches all of them at once."""

    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        with _pt(name):
            return fn(*args, **kwargs)

    return _wrapped


# Rebind the video helpers to timed versions. Everything below this line
# in the module — including the produce loop and _trace_segment — picks
# up the wrapper, because it resolves these names at call time.
cut_segment = _timed(cut_segment, "ffmpeg_cut")
compress_for_email = _timed(compress_for_email, "ffmpeg_compress")
transcode_for_web = _timed(transcode_for_web, "ffmpeg_transcode")
make_vertical = _timed(make_vertical, "ffmpeg_vertical")
make_vertical_pan = _timed(make_vertical_pan, "ffmpeg_vertical")
extract_thumbnail = _timed(extract_thumbnail, "ffmpeg_thumbnail")
splice_impact_clip = _timed(splice_impact_clip, "ffmpeg_composite")
concat_two_clips = _timed(concat_two_clips, "ffmpeg_composite")
apply_intro_overlay_inplace = _timed(
    apply_intro_overlay_inplace, "ffmpeg_overlay",
)
render_tracer_video = _timed(render_tracer_video, "render_tracer")
detect_swings_combined = _timed(detect_swings_combined, "detect_swings")

router = APIRouter(
    prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)]
)


@router.get("/courses", response_model=list[CourseOut])
def list_courses(db: Session = Depends(get_db)):
    return db.query(Course).order_by(Course.created_at.desc()).all()


@router.post("/mirror-from-prod")
def mirror_from_prod():
    """Pull new camera clips from the configured source (prod) backend into
    this backend — the button version of scripts/mirror_prod_to_dev.py.
    Kicks a background pull and returns immediately; poll the status route."""
    from ..services import mirror

    return mirror.start_pull()


@router.get("/mirror-from-prod/status")
def mirror_from_prod_status():
    """Progress of the current/last 'Pull from prod' run + whether the
    feature is configured (so the UI can show/hide the button)."""
    from ..services import mirror

    return {"configured": bool(settings.mirror_course_id), **mirror.status()}


@router.post("/scan-non-golf")
def scan_non_golf(db: Session = Depends(get_db)):
    """Scan the Production queue's clips and flag the ones that don't look
    like a real golf shot (no person / indoor / no grass). Kicks a background
    scan; poll the status route for progress + the flagged upload ids. The UI
    only pre-checks those boxes — nothing is deleted. Dev-only tool."""
    if not settings.scan_non_golf_enabled:
        return {"ok": False, "error": "Scan is not enabled on this deployment."}
    from ..services import golf_scene

    # Scan the tee angle of every long-upload row (fall back to green), pulling
    # the file back from object storage first if the ephemeral disk lost it.
    rows = (
        db.query(LongVideoUpload)
        .order_by(LongVideoUpload.created_at.desc())
        .all()
    )
    items: list[tuple[int, Path]] = []
    for r in rows:
        name = r.tee_filename or r.green_filename
        if not name:
            continue
        storage.ensure_local(CLIPS_DIR, name)
        path = CLIPS_DIR / name
        if path.exists():
            items.append((r.id, path))
    if not golf_scene.start_scan(items):
        return {"ok": False, "error": "A scan is already running."}
    return {"ok": True, "total": len(items)}


@router.get("/scan-non-golf/status")
def scan_non_golf_status():
    """Progress of the current/last non-golf scan + whether the tool is
    enabled (so the UI can show/hide the button). `flagged` is the list of
    upload ids the UI should pre-check."""
    from ..services import golf_scene

    return {"enabled": bool(settings.scan_non_golf_enabled), **golf_scene.status()}


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


@router.post("/courses/{course_id}/ball-roi")
def set_ball_roi(course_id: int, payload: dict, db: Session = Depends(get_db)):
    """Set or clear the tee-box ROI that restricts ball detection.
    payload: {"roi": {"x","y","w","h"}} as fractions (0–1) of the frame, or
    {"roi": null} to clear. Drawn once per course (fixed camera)."""
    course = db.get(Course, course_id)
    if not course:
        raise HTTPException(404, "course not found")
    roi = (payload or {}).get("roi")
    if roi is None:
        course.ball_roi = None
    else:
        try:
            x = max(0.0, min(1.0, float(roi["x"])))
            y = max(0.0, min(1.0, float(roi["y"])))
            w = max(0.01, min(1.0 - x, float(roi["w"])))
            h = max(0.01, min(1.0 - y, float(roi["h"])))
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, "roi must be {x,y,w,h} fractions")
        course.ball_roi = {"x": x, "y": y, "w": w, "h": h}
    db.commit()
    return {"ok": True, "ball_roi": course.ball_roi}


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
            db.query(VideoClip.processing_status, func.count())
            .group_by(VideoClip.processing_status)
            .all()
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

    rows = (
        query.order_by(TeeTime.starts_at.desc(), Participant.id.desc())
        .limit(limit)
        .all()
    )

    # Pre-fetch clip counts
    ids = [p.id for (p, _tt, _c) in rows]
    clip_counts: dict[int, dict] = {i: {"total": 0, "assigned": 0} for i in ids}
    if ids:
        assigned_status = ClipProcessingStatus.assigned.value
        counts = (
            db.query(
                VideoClip.participant_id,
                VideoClip.processing_status,
                func.count(VideoClip.id),
            )
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
            "course": {
                "id": course.id,
                "name": course.name,
                "location": course.location,
            },
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
        "smtp"
        if (settings.smtp_host and settings.smtp_user and settings.smtp_password)
        else "sendgrid"
        if settings.sendgrid_api_key
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
def send_round_summary(
    participant_id: int, force: bool = False, db: Session = Depends(get_db)
):
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
    db.add(
        AuditLog(
            actor="admin",
            action="refund",
            target=f"participant:{p.id}",
            detail=f"mode={result.get('mode')} refund_id={result.get('refund_id')}",
        )
    )
    db.commit()
    return {
        "ok": True,
        "mode": result.get("mode"),
        "refund_id": result.get("refund_id"),
    }


@router.post("/participants/{participant_id}/resend-gallery")
def resend_gallery(participant_id: int, db: Session = Depends(get_db)):
    p = db.get(Participant, participant_id)
    if not p:
        raise HTTPException(404, "participant not found")
    gallery_url = f"{settings.app_base_url}/g/{p.gallery_token}"
    notifications.notify_gallery_ready(p.name, p.mobile, p.email, gallery_url)
    db.add(
        AuditLog(actor="admin", action="resend_gallery", target=f"participant:{p.id}")
    )
    db.commit()
    return {"ok": True, "gallery_url": gallery_url}


@router.get("/flagged-clips")
def flagged_clips(db: Session = Depends(get_db)):
    clips = (
        db.query(VideoClip)
        .filter(
            VideoClip.processing_status.in_(
                [
                    ClipProcessingStatus.flagged.value,
                    ClipProcessingStatus.unassigned.value,
                ]
            )
        )
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
                                "selfie_url": f"/uploads/{p.selfie_path}"
                                if p.selfie_path
                                else None,
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
def manually_assign_clip(
    clip_id: int, participant_id: int, db: Session = Depends(get_db)
):
    clip = db.get(VideoClip, clip_id)
    participant = db.get(Participant, participant_id)
    if not clip or not participant:
        raise HTTPException(404, "clip or participant missing")
    clip.participant_id = participant.id
    clip.processing_status = ClipProcessingStatus.assigned.value
    clip.issue_note = None
    db.add(
        AuditLog(
            actor="admin",
            action="assign_clip",
            target=f"clip:{clip.id}->p:{participant.id}",
        )
    )

    course = db.get(Course, clip.course_id)
    notifications.maybe_send_round_summary(db, participant, course)

    db.commit()
    return {"ok": True, "summary_sent": participant.summary_sent_at is not None}


# --- Manual clip upload (proxy for Shot Tracer webhook in V0) ---------------

CLIPS_DIR = Path(__file__).resolve().parents[2] / settings.upload_dir / "clips"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)


def _local_tee(row):
    """This upload's tee video on local disk, pulled back from object
    storage first if it is not there.

    THE DISK IS EPHEMERAL. Replit's filesystem is wiped on every deploy,
    so a clip uploaded an hour ago lives in object storage and not under
    CLIPS_DIR. The produce path has always rehydrated before reading;
    nine read-only endpoints did not, and simply 404'd. That is why the
    edit wizard showed "No frame" for a perfectly good upload — the
    frame grabber asked the local disk a question only object storage
    could answer.

    Returns the path (which may still not exist if the file is genuinely
    gone — callers keep their own existence check), or None when the row
    names no tee file at all.
    """
    if not getattr(row, "tee_filename", None):
        return None
    try:
        storage.ensure_local(CLIPS_DIR, row.tee_filename)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not rehydrate tee %s: %s", row.tee_filename, exc)
    return CLIPS_DIR / row.tee_filename


def _local_green(row):
    """The green half, same contract as `_local_tee`."""
    if not getattr(row, "green_filename", None):
        return None
    try:
        storage.ensure_local(CLIPS_DIR, row.green_filename)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not rehydrate green %s: %s", row.green_filename, exc)
    return CLIPS_DIR / row.green_filename


@router.get("/clips")
def list_all_clips(
    limit: int = 100, offset: int = 0, db: Session = Depends(get_db),
):
    """All clips, newest first. Includes orphans (no participant_id).

    Powers the /admin/clips test/iteration page where we can rerun the
    tracer on any existing clip without re-uploading.
    """
    clips = (
        db.query(VideoClip)
        .order_by(VideoClip.created_at.desc())
        .offset(max(0, offset))
        .limit(max(1, min(500, limit)))
        .all()
    )
    course_ids = {c.course_id for c in clips}
    participant_ids = {c.participant_id for c in clips if c.participant_id}
    courses = (
        {c.id: c for c in db.query(Course).filter(Course.id.in_(course_ids)).all()}
        if course_ids
        else {}
    )
    participants = (
        {
            p.id: p
            for p in db.query(Participant)
            .filter(Participant.id.in_(participant_ids))
            .all()
        }
        if participant_ids
        else {}
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
        out.append(
            {
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
                "vertical_url": c.vertical_url,
                "thumbnail_url": c.thumbnail_url,
                "ball_in_cup": bool(c.ball_in_cup),
                "is_highlight": bool(c.is_highlight),
                "processing_status": c.processing_status,
                "participant_id": c.participant_id,
                "participant_name": participant.name if participant else None,
                "fps": round(fps_val, 1) if fps_val is not None else None,
                "source_device": source_device,
            }
        )
    return out


@router.post("/clips/{clip_id}/vertical")
def make_clip_vertical(
    clip_id: int, force: int = 0, db: Session = Depends(get_db),
):
    """Generate (or return the existing) 9:16 vertical variant of a
    produced clip — full-frame crop aimed at the action. `force=1`
    re-renders even when one exists (e.g. after a style change).
    Lets clips produced before this feature get a vertical on demand."""
    clip = db.get(VideoClip, clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    if clip.vertical_url and not force:
        fname = clip.vertical_url.split("?")[0].rsplit("/", 1)[-1]
        if storage.ensure_local(CLIPS_DIR, fname) and (CLIPS_DIR / fname).exists():
            return {"clip_id": clip.id, "vertical_url": clip.vertical_url}
    src_url = clip.tracer_url or clip.source_url
    if not src_url:
        raise HTTPException(400, "clip has no video to convert")
    src_name = src_url.split("?")[0].rsplit("/", 1)[-1]
    if not storage.ensure_local(CLIPS_DIR, src_name):
        raise HTTPException(404, f"source file {src_name} not found")
    src_path = CLIPS_DIR / src_name
    out_path = CLIPS_DIR / f"{src_path.stem}_vertical.mp4"
    # Render from the CLEAN pre-overlay copy when one exists — the
    # baked landscape panels would otherwise drift through the pan.
    render_src = src_path
    _seg_stem = src_path.stem
    if _seg_stem.endswith("_composite"):
        _seg_stem = _seg_stem[: -len("_composite")]
    for _cand in (
        f"{src_path.stem}_clean{src_path.suffix}",
        # Old clips (pre clean-copy): the intermediate tracer render has
        # the tracer line but NO intro graphics baked in.
        f"{_seg_stem}_ai_mog2_tracer.mp4",
    ):
        if (
            storage.ensure_local(CLIPS_DIR, _cand)
            and (CLIPS_DIR / _cand).exists()
        ):
            render_src = CLIPS_DIR / _cand
            break
    # Prefer the follow-the-shot pan when this clip's produce run
    # persisted a ball track; fall back to a static crop aimed at the
    # golfer (or frame center) otherwise.
    _made = False
    _focus = 0.5
    try:
        if clip.long_upload_id:
            _up = db.get(LongVideoUpload, clip.long_upload_id)
            _em = (_up.edit_metrics or {}) if _up else {}
            _fw = float(_em.get("frame_width") or 0)
            for _sw in _em.get("swings") or []:
                if _sw.get("clip_id") != clip.id:
                    continue
                _fps = float(_sw.get("fps") or 0) or 30.0
                _off = int(_sw.get("start_frame") or 0)
                _ball = _sw.get("ball") or {}
                _rxy = (
                    [_ball.get("x"), _ball.get("y")]
                    if _ball.get("x") is not None else None
                )
                if _fw <= 0:
                    # Produce-only uploads never stored frame_width (the
                    # wizard save writes it) — probe the raw tee cut,
                    # whose native width the track coords live in.
                    _probe_src = None
                    if clip.tee_clip_url:
                        _tn = clip.tee_clip_url.split("?")[0].rsplit("/", 1)[-1]
                        if storage.ensure_local(CLIPS_DIR, _tn):
                            _probe_src = CLIPS_DIR / _tn
                    elif "_composite" not in src_name:
                        _probe_src = src_path
                    if _probe_src is not None and _probe_src.exists():
                        _fw = float(
                            (probe_video_info(_probe_src) or {}).get("width")
                            or 0
                        )
                if _rxy and _fw > 0:
                    _focus = max(0.15, min(0.85, float(_rxy[0]) / _fw))
                _trk = [
                    {"frame": int(r["frame"]) - _off, "x": r["x"]}
                    for r in _sw.get("ball_track_frames") or []
                    if r.get("found") and r.get("x") is not None
                    and int(r.get("frame") or 0) >= _off
                ]
                if render_src is src_path and _sw.get("tracer_url"):
                    _tn = (
                        _sw["tracer_url"].split("?")[0].rsplit("/", 1)[-1]
                    )
                    if (
                        _tn
                        and storage.ensure_local(CLIPS_DIR, _tn)
                        and (CLIPS_DIR / _tn).exists()
                    ):
                        render_src = CLIPS_DIR / _tn
                _gx = _probe_golfer_x_frac(render_src)
                if _gx is not None:
                    _focus = max(0.15, min(0.85, float(_gx)))
                if len(_trk) >= 3 and _fw > 0:
                    _imp_t = None
                    try:
                        _sif = _sw.get("impact_frame")
                        if _sif is not None:
                            _imp_t = max(0.0, (float(_sif) - _off) / _fps)
                    except (TypeError, ValueError):
                        _imp_t = None
                    _ppath = _vertical_pan_path(
                        _trk, _rxy, _fps, _fw, golfer_x=_gx,
                        impact_t=_imp_t,
                    )
                    _made = make_vertical_pan(render_src, out_path, _ppath)
                log.info(
                    "clip %s: vertical on-demand — track=%d fw=%s "
                    "golfer_x=%s src=%s -> %s",
                    clip.id, len(_trk), _fw,
                    round(_gx, 3) if _gx is not None else None,
                    render_src.name,
                    "PAN" if _made else "static fallback",
                )
                break
    except Exception as exc:  # noqa: BLE001
        log.warning("clip %s: vertical pan failed: %s", clip_id, exc)
    if not _made and not make_vertical(
        render_src, out_path, focus_x_frac=_focus,
    ):
        raise HTTPException(500, "vertical render failed (see server log)")
    # Portrait-geometry name plate + persistent logo on the vertical.
    try:
        _course = db.get(Course, clip.course_id)
        _pname = None
        if clip.participant_id:
            _p = db.get(Participant, clip.participant_id)
            _pname = _p.name if _p else None
        _yardage = 101
        if _course and _course.hole_yardages:
            try:
                _ry = _course.hole_yardages.get(str(int(clip.hole_number)))
                if _ry is not None:
                    _yardage = int(_ry)
            except (TypeError, ValueError):
                pass
        apply_intro_overlay_inplace(
            out_path,
            player_name=_pname or "Brent Baldwin",
            course_name=_course.name if _course else "",
            hole_number=int(clip.hole_number),
            par=3,
            yardage=_yardage,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("clip %s: vertical overlay failed: %s", clip_id, exc)
    clip.vertical_url = (
        f"{settings.app_base_url}/uploads/clips/{out_path.name}"
        f"?v={int(out_path.stat().st_mtime)}"
    )
    db.commit()
    log.info("clip %s: vertical variant rendered (%s)", clip.id, out_path.name)
    return {
        "clip_id": clip.id,
        "vertical_url": clip.vertical_url,
        "mode": "pan" if _made else "static",
    }


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
    db.add(
        AuditLog(
            actor="admin",
            action="toggle_clip_broadcast",
            target=f"clip:{clip.id}",
            detail=f"is_highlight={bool(clip.is_highlight)}",
        )
    )
    db.commit()
    db.refresh(clip)
    return {"clip_id": clip.id, "is_highlight": bool(clip.is_highlight)}


@router.post("/broadcast-clips/clear")
def clear_all_broadcast(db: Session = Depends(get_db)):
    """Un-flag every clip currently on the Broadcast channel — a clean
    slate. Sets is_highlight=False (and clears highlight_tag) on all clips
    that were flagged, so the Broadcast page empties and you re-promote
    only the good ones going forward. The produced clips themselves are
    untouched; this only removes them from Broadcast. Idempotent.
    """
    flagged = db.query(VideoClip).filter(VideoClip.is_highlight.is_(True)).all()
    n = len(flagged)
    for clip in flagged:
        clip.is_highlight = False
        clip.highlight_tag = None
        db.add(clip)
    db.add(
        AuditLog(
            actor="admin",
            action="clear_all_broadcast",
            target="broadcast",
            detail=f"cleared={n}",
        )
    )
    db.commit()
    log.info("admin: cleared %d clip(s) from Broadcast", n)
    return {"cleared": n}


@router.get("/broadcast-clips")
def list_broadcast_clips(
    limit: int = 100, offset: int = 0, db: Session = Depends(get_db),
):
    """List clips on the Broadcast channel — only clips an operator has
    explicitly flagged with is_highlight=True by pressing Broadcast.
    Newest first. Same per-clip shape as /clips so the existing FPS /
    source-device header line works without modification.

    NOTE: this used to ALSO include every dual-camera composite (source
    filename contains '_composite') as a convenience for old runs. That
    made every produced tee+green swing auto-appear here — hundreds of
    un-vetted, mostly-garbage clips — because every dual-camera session
    outputs a composite. Broadcast is a manual, operator-driven promotion
    (and the only quality gate the AI tracer learns from), so the list now
    shows strictly what was flagged. The public /broadcast/next playlist
    already filtered on is_highlight, so viewers never saw the garbage.
    """
    clips = (
        db.query(VideoClip)
        .filter(VideoClip.is_highlight.is_(True))
        .order_by(VideoClip.created_at.desc())
        .offset(max(0, offset))
        .limit(max(1, min(500, limit)))
        .all()
    )
    course_ids = {c.course_id for c in clips}
    participant_ids = {c.participant_id for c in clips if c.participant_id}
    courses = (
        {c.id: c for c in db.query(Course).filter(Course.id.in_(course_ids)).all()}
        if course_ids
        else {}
    )
    participants = (
        {
            p.id: p
            for p in db.query(Participant)
            .filter(Participant.id.in_(participant_ids))
            .all()
        }
        if participant_ids
        else {}
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
        out.append(
            {
                "id": c.id,
                "course_id": c.course_id,
                "course_name": course.name if course else None,
                "hole_number": c.hole_number,
                "camera_type": c.camera_type,
                "captured_at": c.captured_at.isoformat() if c.captured_at else None,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "source_url": c.source_url,
                "tracer_url": c.tracer_url,
                "vertical_url": c.vertical_url,
                "thumbnail_url": c.thumbnail_url,
                "ball_in_cup": bool(c.ball_in_cup),
                "is_highlight": bool(c.is_highlight),
                "processing_status": c.processing_status,
                "participant_id": c.participant_id,
                "participant_name": participant.name if participant else None,
                "fps": round(fps_val, 1) if fps_val is not None else None,
                "source_device": source_device,
            }
        )
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
    tracer_url, tracer_info, _, tracer_debug_url = _run_tracer(
        fpath, sensitivity=float(sensitivity)
    )
    clip.tracer_url = tracer_url
    db.add(
        AuditLog(
            actor="admin",
            action="retry_tracer",
            target=f"clip:{clip.id}",
            detail=str(tracer_info),
        )
    )
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
        cv2.imwrite(
            str(address_out), address_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92]
        )
        address_mtime = int(address_out.stat().st_mtime)
        address_url = (
            f"{settings.app_base_url}/uploads/clips/{address_name}?v={address_mtime}"
        )

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
                manual_positions.append(
                    {
                        "frame": int(entry["frame"]),
                        "x": int(entry["x"]),
                        "y": int(entry["y"]),
                    }
                )
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

    # Few-shot from prior broadcast clips on this clip's course/hole;
    # excluding the clip's source upload prevents self-reference.
    clip_examples = tracer_examples.fetch_all_kinds(
        db,
        course_id=clip.course_id,
        hole_number=int(clip.hole_number) if clip.hole_number else None,
        exclude_lvu_ids=({clip.long_upload_id} if clip.long_upload_id else None),
    )
    pipe = run_full_ai_tracer_pipeline(
        fpath,
        output_dir=CLIPS_DIR,
        output_prefix=fpath.stem,
        model=model,
        impact_frame_override=int(impact_frame_override)
        if impact_frame_override is not None
        else None,
        ball_track_max_frames_override=int(ball_track_max_frames)
        if ball_track_max_frames is not None
        else None,
        ball_at_rest_override=ball_at_rest_override,
        examples_by_kind=clip_examples,
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
                "browser playback may fail",
                tracer_path.name,
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
        ball_track_frames_out.append(
            {
                "frame": rec.get("frame"),
                "found": rec.get("found"),
                "x": rec.get("x"),
                "y": rec.get("y"),
                "confidence": rec.get("confidence"),
                "notes": rec.get("notes"),
                "retry": rec.get("retry", False),
                "image_url": url,
            }
        )

    ball_track_info = pipe.get("ball_track")
    ball_track_summary = None
    if ball_track_info is not None:
        ball_track_summary = {k: v for k, v in ball_track_info.items() if k != "frames"}

    db.add(
        AuditLog(
            actor="admin",
            action="ai_trace_address",
            target=f"clip:{clip.id}",
            detail=str(
                {
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
                }
            ),
        )
    )
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


def _ball_search_box(src_path, wrist_xy, feet_xy):
    """The square to look for a resting ball in, from the pose.

    Geometry per the operator's sketch: the hands are a REFERENCE POINT,
    not the ball. Take the top of the hands marker as the box's upper-left
    corner, run down to below the golfer's feet, and square it off from
    there. On a camera behind the golfer that region covers the ground in
    front of the stance, which is exactly where a ball at address sits —
    and it excludes the golfer's body, which is what kept dragging the
    search onto shoes and shorts.

    Returns (x, y, w, h) in native pixels, or None when there isn't enough
    pose to build it."""
    if not (wrist_xy and len(wrist_xy) == 2):
        return None
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(src_path))
        try:
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            cap.release()
        if not (fw and fh):
            return None
        wx, wy = int(wrist_xy[0]), int(wrist_xy[1])
        # Top edge: just above the hands marker (the ring drawn on the
        # debug frame), so the box starts level with the top of it.
        _r = max(10, int(fh * 0.02))
        top = max(0, wy - _r)
        # Bottom edge: below the feet. Falls back to a stance-height
        # estimate when pose lost the ankles at the blurred peak.
        if feet_xy and len(feet_xy) == 2:
            bottom = int(feet_xy[1]) + int(0.03 * fh)
        else:
            bottom = wy + int(0.22 * fh)
        bottom = min(fh, max(top + 32, bottom))
        size = bottom - top
        # Square, running from the hands toward the target side, then
        # grown 15% UP and AWAY from the golfer. The bottom and the
        # golfer-side edge stay put — the extra reach goes where the ball
        # can actually be (further from the stance, and higher up the
        # slope), not back across the body.
        # Nudged right, off the hands. The hands mark the golfer-side
        # edge of where the ball can be, so a box starting exactly on them
        # spends its left column on the club and the golfer's near leg.
        # A tenth of the box's width clears that without giving up reach.
        size = int(round(size * 1.15))
        left = max(0, min(fw - 16, wx + int(round(0.10 * size))))
        top = max(0, bottom - size)
        size = min(size, fw - left, fh - top)
        if size < 32:
            return None
        return (int(left), int(top), int(size), int(size))
    except Exception as exc:  # noqa: BLE001
        log.warning("ball search box failed: %s", exc)
        return None


def _save_assumed_impact_frame(
    src_path, frame_idx: int, fps: float, seed_xy, ai_points, name_stem: str,
    search_box=None, box_ball_xy=None,
) -> str | None:
    """Write the frame we ASSUMED impact on, annotated. Returns the
    filename, or None.

    When no ball departure is found the impact frame is a pose-peak
    estimate rather than something we watched happen, so the panel has to
    be able to show WHICH frame that is — cyan ring = the pose hands the
    launch plot was seeded from, magenta = each ball the AI then found."""
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(src_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, frame_idx)))
            ok, fr = cap.read()
        finally:
            cap.release()
        if not ok or fr is None:
            return None

        if search_box and len(search_box) == 4:
            _bx, _by, _bw, _bh = (int(v) for v in search_box)
            cv2.rectangle(
                fr, (_bx, _by), (_bx + _bw, _by + _bh),
                (0, 0, 255), 2, cv2.LINE_AA,
            )
        if box_ball_xy and len(box_ball_xy) == 2:
            cv2.circle(
                fr, (int(box_ball_xy[0]), int(box_ball_xy[1])),
                max(8, int(fr.shape[0] * 0.015)),
                (0, 255, 0), 3, cv2.LINE_AA,
            )
        for q in (ai_points or []):
            cv2.circle(
                fr, (int(q["x"]), int(q["y"])),
                max(6, int(fr.shape[0] * 0.012)),
                (255, 0, 255), 2, cv2.LINE_AA,
            )
        _txt = (
            f"ASSUMED impact f{int(frame_idx)} "
            f"({frame_idx / max(1.0, fps):.2f}s) - pose peak, no ball "
            f"departure; red=zoom square searched, green=ball found in "
            f"it, magenta=AI launch picks"
        )
        for _c, _w in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(
                fr, _txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                _c, _w, cv2.LINE_AA,
            )
        fname = f"{name_stem}.jpg"
        cv2.imwrite(
            str(CLIPS_DIR / fname), fr,
            [int(cv2.IMWRITE_JPEG_QUALITY), 85],
        )
        return fname if (CLIPS_DIR / fname).exists() else None
    except Exception as exc:  # noqa: BLE001
        log.warning("assumed-impact frame save failed: %s", exc)
        return None


def _seg_launch_points(seg: dict) -> list[dict]:
    """Every AI-plotted launch point we have for this swing, in SOURCE
    frame numbers.

    These are the magenta dots on the debug flight map — the model's
    frame-by-frame read of the first moments of flight. They were being
    DISPLAYED and IGNORED at the same time, because the map reads
    anchor_rec["ai_launch_points"] while the render reads
    seg["launch_points"], and only the latter sits behind the produce
    worker's chain of gates (practice filter found the ball AND the anchor
    verified AND the merged list came back non-empty). Miss any gate and
    the points are still computed, still drawn, and never handed over —
    so the renderer's nearest known ball position is hundreds of pixels up
    the arc. Take the union so a plotted point cannot be shown and
    discarded at once."""
    out: dict[int, dict] = {}
    for src in (
        seg.get("launch_points") or [],
        (seg.get("anchor_rec") or {}).get("ai_launch_points") or [],
    ):
        for pt in src:
            if not isinstance(pt, dict):
                continue
            f, x, y = pt.get("frame"), pt.get("x"), pt.get("y")
            if f is None or x is None or y is None:
                continue
            out.setdefault(int(f), {
                "frame": int(f), "x": float(x), "y": float(y),
            })
    return [out[k] for k in sorted(out)]


def _utcnow_naive() -> datetime:
    """Naive UTC datetime, matching how the model stores timestamps."""
    return datetime.utcnow()


def _run_long_upload_job(
    upload_id: int,
    seg_list: list[dict],
    auto_detect_swings: bool,
    starting_hole: int,
    ai_tracer_model: str | None,
    audio_min_peak_ratio: float = 3.0,
    motion_ratio: float = 2.0,
    combined_pair_window_sec: float = 3.0,
    tee_green_delta_sec: float = 0.0,
    single_hole: bool = False,
    motion_only: bool = False,
    debug_artifacts: bool = False,
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
    # Diagnostic film-strips (anchor walk, launch tracker, AI launch
    # plot) are only written when the run came from the Debug button —
    # plain Produce / Re-Produce skips them (they were the bulk of the
    # artifact spam). Product images (raw motion heat, MOG2 overlay,
    # click-to-plot dots) stay.
    _dbg_dir = CLIPS_DIR if debug_artifacts else None
    # Start this run's phase clock (thread-local — see _pt_add). Every
    # ffmpeg helper and AI call below accumulates into it; the totals are
    # logged at the end and read by the Debug3 panel for its stage 7.
    _pt_reset()
    _job_t0 = time.perf_counter()
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

        # Re-produce REPLACES, not accumulates: clear any clips from a prior
        # run of this upload so a re-detect (e.g. motion's 3 → pose's 1)
        # doesn't leave stale clips behind. Best-effort.
        try:
            _old = db.query(VideoClip).filter(VideoClip.long_upload_id == upload_id).all()
            for _c in _old:
                db.delete(_c)
            if _old:
                db.commit()
                log.info(
                    "long-upload worker: cleared %d prior clip(s) for upload %s",
                    len(_old), upload_id,
                )
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            log.warning("long-upload worker: could not clear prior clips: %s", exc)

        try:
            # Rehydrate the raw source(s) from object storage if the
            # ephemeral disk lost them (e.g. re-producing after a redeploy).
            if row.tee_filename:
                storage.ensure_local(CLIPS_DIR, row.tee_filename)
            if row.green_filename:
                storage.ensure_local(CLIPS_DIR, row.green_filename)
            src_path = CLIPS_DIR / row.tee_filename if row.tee_filename else None
            if not src_path or not src_path.exists():
                raise RuntimeError(
                    f"tee source file missing on disk: {row.tee_filename}"
                )
            green_src_path: Path | None = None
            if row.green_filename:
                candidate = _local_green(row)
                if not candidate.exists():
                    raise RuntimeError(
                        f"green source file missing on disk: {row.green_filename}"
                    )
                green_src_path = candidate

            segs = list(seg_list or [])
            auto_used = False
            used_pose = False
            if not segs and auto_detect_swings:
                auto_used = True
                tee_fps = probe_fps(src_path) or 30.0
                _detect_debug: dict = {}
                detected = None
                # Pose swing detector (dev, needs mediapipe). Falls back to
                # the motion/combined path when unavailable so we never
                # produce nothing.
                if settings.swing_detector == "pose":
                    try:
                        from ..services import pose_swing

                        if pose_swing.available():
                            detected = pose_swing.detect_swings_from_pose(
                                src_path, fps=tee_fps, debug=_detect_debug,
                            )
                            used_pose = True
                            log.info(
                                "long-upload worker: upload=%s pose detector -> "
                                "%d swing(s)", upload_id, len(detected),
                            )
                        else:
                            log.info(
                                "long-upload worker: swing_detector=pose but "
                                "mediapipe unavailable — falling back to motion",
                            )
                    except Exception as exc:  # noqa: BLE001
                        log.warning("pose detector failed (%s) — using motion", exc)
                        detected = None
                if detected is not None:
                    pass  # pose produced the swing list
                elif motion_only:
                    # Vision-only: key on the swing's motion burst (the
                    # downswing-through-impact + ball launch), then classify
                    # each by whether a ball was on the tee and left it.
                    # Used for camera sessions, detected from video alone
                    # (no audio "crack"). Each swing is tagged with a
                    # ball_verdict; whether non-shots are dropped depends on
                    # settings.camera_produce_unconfirmed_shots (permissive
                    # during course testing, strict once the check is tuned).
                    detected = detect_swings_from_motion(
                        src_path, fps=tee_fps, debug=_detect_debug,
                    )
                    detected = filter_swings_by_ball_departure(
                        src_path, detected, tee_fps, debug=_detect_debug,
                        keep_all=settings.camera_produce_unconfirmed_shots,
                        drop_garbage=settings.camera_drop_garbage_clips,
                    )
                else:
                    # Combined audio + motion detector: an audio impact
                    # only counts when a motion burst peaks within ±3 s.
                    detected = detect_swings_combined(
                        src_path,
                        fps=tee_fps,
                        audio_min_peak_ratio=float(audio_min_peak_ratio),
                        motion_ratio=float(motion_ratio),
                        pair_window_sec=float(combined_pair_window_sec),
                        debug=_detect_debug,
                    )

                # Rescueable near-miss bursts (dropped ONLY by the
                # speed-ratio gate, with a swing-posture bend) — used
                # both to VETO the non-golf delete below and to
                # resurrect candidates for the pipeline.
                _resc_all: list = []
                try:
                    _existing_ts0 = {
                        round(float(d.get("peak_time_sec") or 0.0), 2)
                        for d in (detected or [])
                    }
                    _resc_all = [
                        b for b in (
                            _detect_debug.get("bursts_detail") or []
                        )
                        if b.get("status") == "ratio_low"
                        and b.get("bend") is not None
                        and float(b["bend"]) >= 15.0
                        and round(float(b["t"]), 2) not in _existing_ts0
                    ]
                except Exception:  # noqa: BLE001
                    _resc_all = []

                # NON-GOLF: the detector doubles as the screen — a
                # video with zero swing candidates is a walk-by / pet /
                # empty capture. Delete it instead of failing the run.
                # A rescueable burst VETOES the delete: a distant real
                # swing can read 'below 5x' (landmark jitter eats the
                # ratio), and deletion is unrecoverable from the UI.
                if (
                    not detected and not _resc_all
                    and settings.auto_delete_non_golf
                ):
                    try:
                        db.rollback()  # release our txn before deleting
                    except Exception:  # noqa: BLE001
                        pass
                    _auto_delete_upload(
                        upload_id,
                        "no golf swing detected by the produce detector",
                    )
                    return

                # RATIO-RESCUE (distance fix): a candidate the pose
                # gate dropped ONLY for a low wrist-speed ratio, but
                # with a confirmed swing-posture bend, gets resurrected
                # — from a course-distance camera the landmark jitter
                # floor eats the speed ratio, so a real swing can read
                # "below 5x". Rescued candidates skip the heat-judge
                # veto and live or die purely on the ball-departure
                # check downstream (the one signal distance can't
                # dilute).
                try:
                    _resc = list(_resc_all)
                    _resc.sort(
                        key=lambda b: -float(b.get("ratio") or 0.0),
                    )
                    for b in _resc[:4]:
                        detected.append({
                            "peak_time_sec": float(b["t"]),
                            "start_sec": max(0.0, float(b["t"]) - 3.5),
                            "end_sec": float(b["t"]) + 5.0,
                            "ratio": b.get("ratio"),
                            "back_bend_deg": b.get("bend"),
                            "confidence": 0.3,
                            "rescued": True,
                        })
                    if _resc:
                        detected.sort(
                            key=lambda d: float(
                                d.get("peak_time_sec") or 0.0,
                            ),
                        )
                        log.info(
                            "long-upload worker: upload=%s ratio-rescue "
                            "resurrected %d candidate(s) at %s",
                            upload_id, min(len(_resc), 4),
                            [b["t"] for b in _resc[:4]],
                        )
                except Exception as exc:  # noqa: BLE001
                    log.warning("ratio-rescue failed: %s", exc)

                # Per-swing audit trail. Every filter decision lands here,
                # is logged as one JSON line, and is persisted to
                # edit_metrics.produce_decisions — so "debug said X but
                # produce did Y" is answerable from data, not guesswork.
                # _work collects the FULL record (pose data, heat images,
                # practice classifications) and is published to
                # _produce_work_state so the Debug report renders THIS
                # run's work instead of re-deriving its own.
                _all_detected = [dict(d) for d in (detected or [])]
                _work: dict = {
                    "run_started": time.time(),
                    "used_pose": bool(used_pose),
                    "pose_debug": _detect_debug,
                    "segments_all": _all_detected,
                    "heat": [],
                    "practice": [],
                }
                _decisions: list[dict] = []
                for d in detected:
                    _decisions.append({
                        "t": round(float(d.get("peak_time_sec") or 0.0), 2),
                        "ratio": d.get("ratio"),
                        "bend": d.get("back_bend_deg"),
                        "kept": True,
                        "dropped_by": None,
                    })

                def _dec_for(d):
                    _t = round(float(d.get("peak_time_sec") or 0.0), 2)
                    for _e in _decisions:
                        if _e["t"] == _t:
                            return _e
                    return {}

                # MOG2 + AI-judge swing confirmation FIRST (stronger and
                # cheaper than the ball-departure filter): each pose swing
                # must show a ball-flight chain, or Claude must recognise
                # the heat composite as a swing. Fail-safe: heuristic-only
                # rejections (no API key) can be resurrected if everything
                # was rejected — but a swing the AI judge positively called
                # "not a swing" STAYS dropped, even if that empties the
                # list. (Previously the keep-all fail-safe resurrected a
                # walking golfer that was the sole survivor of the
                # practice filter.)
                if used_pose and detected and settings.swing_heat_check_enabled:
                    try:
                        from ..services.tracer import swing_heat_check

                        _confirmed = []
                        _dropped_heuristic = []
                        for d in detected:
                            chk = swing_heat_check(
                                src_path,
                                float(d.get("peak_time_sec") or 0.0),
                                tee_fps,
                                ball_hint=d.get("impact_wrist_xy"),
                                debug_dir=CLIPS_DIR,
                                debug_prefix=(
                                    f"heatchk-prod-{upload_id}-"
                                    f"{secrets.token_hex(3)}"
                                ),
                            )
                            # The AI judge decides for EVERY swing (the
                            # ball-flight chain no longer short-circuits —
                            # it false-positived too often); the club-fan
                            # heuristic verdict only stands when there's
                            # no key.
                            _ai_seen = False
                            if (
                                chk.get("image_clean")
                                and os.environ.get("ANTHROPIC_API_KEY")
                            ):
                                _j = judge_swing_heat_image(
                                    CLIPS_DIR / chk["image_clean"],
                                )
                                if _j.get("is_swing") is True:
                                    chk["verdict"] = "club_swing"
                                    _ai_seen = True
                                elif _j.get("is_swing") is False:
                                    chk["verdict"] = "no_swing"
                                    _ai_seen = True
                                chk["ai_judge"] = _j.get("is_swing")
                                chk["ai_reason"] = _j.get("reason")
                            d["heat_check"] = {
                                "verdict": chk.get("verdict"),
                                "n_timed": chk.get("n_timed"),
                                "chain_len": chk.get("chain_len"),
                                "n_rays": chk.get("n_rays"),
                            }
                            # Full record (incl. evidence image names) for
                            # the debug report.
                            _work["heat"].append({
                                "t": round(
                                    float(d.get("peak_time_sec") or 0.0), 2,
                                ),
                                "verdict": chk.get("verdict"),
                                "n_timed": chk.get("n_timed"),
                                "chain_len": chk.get("chain_len"),
                                "chain_f0": chk.get("chain_f0"),
                                "chain_f1": chk.get("chain_f1"),
                                "chain_flight": chk.get("chain_flight"),
                                "n_rays": chk.get("n_rays"),
                                "n_angles": chk.get("n_angles"),
                                "fan": chk.get("fan"),
                                "ai_judge": chk.get("ai_judge"),
                                "ai_reason": chk.get("ai_reason"),
                                "reason": chk.get("reason"),
                                "image": chk.get("image"),
                            })
                            _e = _dec_for(d)
                            _e["heat"] = chk.get("verdict")
                            _e["ai_judge"] = chk.get("ai_judge")
                            _e["ai_reason"] = chk.get("ai_reason")
                            if d.get("rescued"):
                                _e["rescued"] = True
                            if (
                                d.get("rescued")
                                and chk.get("verdict") == "no_swing"
                            ):
                                # Rescued candidates bypass the judge's
                                # veto — the ball-departure check is
                                # their arbiter.
                                log.info(
                                    "long-upload worker: upload=%s "
                                    "rescued swing @ %.1fs kept past the "
                                    "judge — ball departure decides",
                                    upload_id,
                                    float(d.get("peak_time_sec") or 0.0),
                                )
                            elif (
                                chk.get("available")
                                and chk.get("verdict") == "no_swing"
                            ):
                                _e["kept"] = False
                                _e["dropped_by"] = (
                                    "heat_ai" if _ai_seen else "heat_heuristic"
                                )
                                if _ai_seen:
                                    log.info(
                                        "long-upload worker: upload=%s AI judge "
                                        "DROPPED swing @ %.1fs (%s)",
                                        upload_id,
                                        float(d.get("peak_time_sec") or 0.0),
                                        chk.get("ai_reason"),
                                    )
                                else:
                                    _dropped_heuristic.append(d)
                                    log.info(
                                        "long-upload worker: upload=%s heat "
                                        "heuristic dropped swing @ %.1fs "
                                        "(chain %s, %s dots)",
                                        upload_id,
                                        float(d.get("peak_time_sec") or 0.0),
                                        chk.get("chain_len"),
                                        chk.get("n_timed"),
                                    )
                            else:
                                _confirmed.append(d)
                        if _confirmed:
                            detected = _confirmed
                        elif _dropped_heuristic:
                            log.info(
                                "long-upload worker: upload=%s heat heuristic "
                                "rejected all %d — fail-safe keeping them "
                                "(no AI judge available)",
                                upload_id, len(_dropped_heuristic),
                            )
                            for d in _dropped_heuristic:
                                _e = _dec_for(d)
                                _e["kept"] = True
                                _e["dropped_by"] = None
                            detected = _dropped_heuristic
                        else:
                            detected = []
                    except Exception as exc:  # noqa: BLE001
                        log.warning(
                            "long-upload worker: heat check failed (%s) — "
                            "keeping all swings", exc,
                        )

                # Practice-swing filter (pose mode) on the CONFIRMED swings:
                # ball at rest before and gone after. Needs a key;
                # unknown/no-key keeps the swing so a real shot is never
                # dropped for lack of a key.
                if used_pose and detected and os.environ.get("ANTHROPIC_API_KEY"):
                    _real = []
                    for d in detected:
                        v = classify_swing_shot(
                            src_path, float(d.get("peak_time_sec") or 0.0), tee_fps,
                            hint_xy=d.get("impact_wrist_xy"),
                        )
                        _e = _dec_for(d)
                        _e["practice"] = v.get("verdict")
                        _e["practice_reason"] = v.get("reason")
                        # IMPACT BY DEPARTURE (MOG2/pixels, no AI): the
                        # classifier just FOUND the resting ball — watch
                        # that exact spot and pin impact to the frame the
                        # ball leaves it. A verified pin replaces the
                        # pose-peak estimate for the cut window AND lets
                        # the tracer skip its audio/vision impact +
                        # handedness calls entirely (overrides).
                        _anchor_rec = None
                        _t_orig = round(
                            float(d.get("peak_time_sec") or 0.0), 2,
                        )
                        _bf = v.get("before") or {}
                        if (
                            v.get("verdict") != "practice"
                            and _bf.get("present")
                            and _bf.get("x") is not None
                        ):
                            # The FOUND ball is the rest position — full
                            # stop. The tracer must look HERE, not
                            # wherever its own address-frame guess lands.
                            # (Pin verification below only tightens it.)
                            d["ball_rest_xy"] = [
                                float(_bf["x"]), float(_bf["y"]),
                            ]
                            try:
                                from ..services.ai_tracer import (
                                    verify_rest_and_impact,
                                    verify_rest_and_impact_ai,
                                )

                                _pk_f = int(round(
                                    float(d.get("peak_time_sec") or 0.0)
                                    * tee_fps,
                                ))
                                # AI-first anchor check: the film-strip
                                # vision lookup reads departure straight
                                # off the tiles (~2 cheap calls/swing),
                                # where the pixel presence gate flapped
                                # when the ring was slightly off the
                                # ball or shadows brightened the patch.
                                # Pixel check remains the fallback on
                                # API failure / no key.
                                # Sequential walk starts at the BEFORE
                                # frame (ball known present there).
                                _bf_t = _bf.get("t")
                                _start_f = (
                                    int(round(float(_bf_t) * tee_fps))
                                    if _bf_t is not None else None
                                )
                                # The classifier's AFTER frame bounds
                                # the walk — ball is gone by then.
                                _af_t = (v.get("after") or {}).get("t")
                                _end_f = (
                                    int(round(float(_af_t) * tee_fps))
                                    if _af_t is not None else None
                                )
                                with _pt("anchor_ai"):
                                    _anchor_rec = verify_rest_and_impact_ai(
                                        src_path,
                                        (float(_bf["x"]), float(_bf["y"])),
                                        _pk_f, tee_fps,
                                        start_frame=_start_f,
                                        end_frame=_end_f,
                                        debug_dir=_dbg_dir,
                                        debug_prefix=(
                                            f"anchorai-prod-{upload_id}-"
                                            f"{secrets.token_hex(3)}"
                                        ),
                                        window_sec=1.5,
                                    )
                                if _anchor_rec.get("api_error") or not (
                                    _anchor_rec.get("available")
                                ):
                                    _ai_fail = _anchor_rec.get("reason")
                                    with _pt("anchor_pixel"):
                                        _anchor_rec = verify_rest_and_impact(
                                            src_path,
                                            (float(_bf["x"]), float(_bf["y"])),
                                            _pk_f, tee_fps,
                                            debug_dir=_dbg_dir,
                                            debug_prefix=(
                                                f"anchorchk-prod-{upload_id}-"
                                                f"{secrets.token_hex(3)}"
                                            ),
                                            window_sec=1.5,
                                        )
                                    # Debug must SHOW that the AI check
                                    # bailed and why — a silent pixel
                                    # fallback looks like the AI answer.
                                    _anchor_rec["ai_fallback_reason"] = (
                                        _ai_fail
                                    )
                                if _anchor_rec.get("verified"):
                                    d["ball_rest_xy"] = list(
                                        _anchor_rec["rest_xy"],
                                    )
                                    d["impact_pinned"] = True
                                    # AI LAUNCH PLOT first: sequential
                                    # vision chase over the first 5
                                    # post-impact frames — the motion-
                                    # blur zone the pixel tracker
                                    # struggles in.
                                    _ai_pts: list = []
                                    try:
                                        from ..services.ai_tracer import (
                                            plot_launch_frames_ai,
                                        )

                                        with _pt("launch_plot_ai"):
                                            _alp = plot_launch_frames_ai(
                                                src_path,
                                                tuple(_anchor_rec["rest_xy"]),
                                                int(_anchor_rec["impact_frame"]),
                                                tee_fps,
                                                debug_dir=_dbg_dir,
                                                debug_prefix=(
                                                    f"ailaunch-{upload_id}-"
                                                    f"{secrets.token_hex(3)}"
                                                ),
                                            )
                                        _ai_pts = list(
                                            _alp.get("points") or [],
                                        )
                                        _anchor_rec["ai_launch_n"] = (
                                            _alp.get("n_found")
                                        )
                                        _anchor_rec["ai_launch_reason"] = (
                                            _alp.get("reason")
                                        )
                                        _anchor_rec["ai_launch_image"] = (
                                            _alp.get("image")
                                        )
                                        _anchor_rec["ai_launch_points"] = (
                                            _ai_pts
                                        )
                                        log.info(
                                            "long-upload worker: AI launch "
                                            "plot %s point(s) (%s)",
                                            _alp.get("n_found"),
                                            _alp.get("reason"),
                                        )
                                    except Exception as exc:  # noqa: BLE001
                                        log.warning(
                                            "ai launch plot failed: %s", exc,
                                        )
                                    # LAUNCH TRACKER (operator-designed,
                                    # pure pixels): adaptive square,
                                    # SEEDED from the AI's last point so
                                    # MOG2 takes over on the next frame
                                    # — one continuous AI→MOG2 chain in
                                    # the debug strips.
                                    try:
                                        from ..services.ai_tracer import (
                                            track_launch_from_rest,
                                        )

                                        with _pt("launch_track_pixel"):
                                            _lt = track_launch_from_rest(
                                                src_path,
                                                tuple(_anchor_rec["rest_xy"]),
                                                int(_anchor_rec["impact_frame"]),
                                                tee_fps,
                                                debug_dir=_dbg_dir,
                                                debug_prefix=(
                                                    f"launchtrk-{upload_id}-"
                                                    f"{secrets.token_hex(3)}"
                                                ),
                                                seed_points=_ai_pts or None,
                                            )
                                        _merged_lp = {
                                            int(p["frame"]): {
                                                "frame": int(p["frame"]),
                                                "x": p["x"], "y": p["y"],
                                            }
                                            for p in _ai_pts
                                        }
                                        for p in (_lt.get("points") or []):
                                            _merged_lp.setdefault(
                                                int(p["frame"]), {
                                                    "frame": int(p["frame"]),
                                                    "x": p["x"], "y": p["y"],
                                                },
                                            )
                                        if _merged_lp:
                                            d["launch_points"] = [
                                                _merged_lp[k]
                                                for k in sorted(_merged_lp)
                                            ]
                                        _anchor_rec["launch_n"] = _lt.get(
                                            "n_found",
                                        )
                                        _anchor_rec["launch_reason"] = (
                                            _lt.get("reason")
                                        )
                                        _anchor_rec["launch_image"] = (
                                            _lt.get("image")
                                        )
                                        _anchor_rec["launch_image_heat"] = (
                                            _lt.get("image_heat")
                                        )
                                        log.info(
                                            "long-upload worker: launch "
                                            "tracker found %s point(s) (%s)"
                                            "%s",
                                            _lt.get("n_found"),
                                            _lt.get("reason"),
                                            (
                                                f" [seeded from {len(_ai_pts)}"
                                                " AI points]"
                                                if _ai_pts else ""
                                            ),
                                        )
                                    except Exception as exc:  # noqa: BLE001
                                        log.warning(
                                            "launch tracker failed: %s", exc,
                                        )
                                        if _ai_pts and not d.get(
                                            "launch_points",
                                        ):
                                            d["launch_points"] = _ai_pts
                                    d["anchor_rec"] = {
                                        k: _anchor_rec.get(k)
                                        for k in (
                                            "verified", "snapped",
                                            "snap_px", "impact_frame",
                                            "impact_delta", "reason",
                                            "image", "image_mog2",
                                            "ai_fallback_reason",
                                            "launch_n",
                                            "launch_reason", "launch_image",
                                            "launch_image_heat",
                                            "ai_launch_n",
                                            "ai_launch_reason",
                                            "ai_launch_image",
                                            "ai_launch_points",
                                        )
                                    }
                                    d["peak_time_sec"] = (
                                        float(_anchor_rec["impact_frame"])
                                        / tee_fps
                                    )
                                    _e["impact_pinned_f"] = int(
                                        _anchor_rec["impact_frame"],
                                    )
                                    log.info(
                                        "long-upload worker: upload=%s "
                                        "impact PINNED by departure @ f%d "
                                        "(%+d vs pose peak), rest snapped "
                                        "%spx",
                                        upload_id,
                                        int(_anchor_rec["impact_frame"]),
                                        int(_anchor_rec["impact_delta"] or 0),
                                        _anchor_rec.get("snap_px"),
                                    )
                            except Exception as exc:  # noqa: BLE001
                                log.warning(
                                    "long-upload worker: departure pin "
                                    "failed: %s", exc,
                                )
                        elif v.get("verdict") != "practice":
                            # NO BALL DEPARTURE, BUT THE SWING PRODUCES.
                            # The pose burst passed the wrist-speed and
                            # bend gates and the AI judge called the heat
                            # a golf swing — the ball just never got found
                            # before/after, so the departure walk had
                            # nothing to anchor on and every AI stage was
                            # skipped. That left these swings on MOG2
                            # alone, which is the weakest tracer we have.
                            #
                            # Assume impact AT THE POSE PEAK and run the
                            # same chain as a departure-pinned swing: AI
                            # launch plot over impact..impact+5, then the
                            # pixel tracker seeded from its last point,
                            # then MOG2 for the rest of the flight. The
                            # verdict is recorded as ASSUMED so the debug
                            # panel never passes it off as a measured
                            # departure.
                            try:
                                from ..services.ai_tracer import (
                                    plot_launch_frames_ai,
                                    track_launch_from_rest,
                                )

                                _pk_f = int(round(
                                    float(d.get("peak_time_sec") or 0.0)
                                    * tee_fps,
                                ))
                                # The hands are a REFERENCE, never the
                                # ball. Build the search box from them:
                                # upper-left just above the hands, running
                                # down past the golfer's feet and squared
                                # off — that region is where a ball on the
                                # ground in front of the golfer has to be.
                                # Then LOOK IN IT, and whatever is found
                                # there is the rest position.
                                _seed = d.get("impact_wrist_xy")
                                _feet = d.get("impact_feet_xy")
                                _box = _ball_search_box(
                                    src_path, _seed, _feet,
                                )
                                if _box:
                                    _tok = secrets.token_hex(3)
                                    # The launch plot's FIRST tile is the
                                    # zoom square: look for the ball at
                                    # rest in there. Every frame after it
                                    # tracks from what was found, and the
                                    # square is never referred to again.
                                    # rest_xy is the box centre only so
                                    # the miss-widening has an origin —
                                    # the wrist is not used past building
                                    # the box's corner.
                                    _plot_from = (
                                        _box[0] + _box[2] / 2.0,
                                        _box[1] + _box[3] / 2.0,
                                    )
                                    with _pt("launch_plot_ai"):
                                        _alp = plot_launch_frames_ai(
                                            src_path,
                                            (float(_plot_from[0]),
                                             float(_plot_from[1])),
                                            _pk_f, tee_fps,
                                            first_rect=_box,
                                            debug_dir=_dbg_dir,
                                            debug_prefix=(
                                                f"ailaunch-assumed-{upload_id}-"
                                                f"{_tok}"
                                            ),
                                        )
                                    _ai_pts = [
                                        {"frame": int(pt["frame"]),
                                         "x": float(pt["x"]),
                                         "y": float(pt["y"])}
                                        for pt in (_alp.get("points") or [])
                                        if pt.get("frame") is not None
                                    ]
                                    log.info(
                                        "long-upload worker: upload=%s no "
                                        "departure — ASSUMED impact at pose "
                                        "peak f%d (%.2fs), AI launch plot "
                                        "found %d point(s) (%s)",
                                        upload_id, _pk_f,
                                        float(d.get("peak_time_sec") or 0.0),
                                        len(_ai_pts), _alp.get("reason"),
                                    )
                                    _found_rest = (
                                        [float(min(
                                            _ai_pts,
                                            key=lambda q: int(q["frame"]),
                                        )["x"]),
                                         float(min(
                                            _ai_pts,
                                            key=lambda q: int(q["frame"]),
                                        )["y"])]
                                        if _ai_pts else None
                                    )
                                    _anchor_rec = {
                                        "verified": bool(_ai_pts),
                                        "assumed_impact": True,
                                        "impact_frame": _pk_f,
                                        "impact_delta": 0,
                                        "snapped": False,
                                        "snap_px": None,
                                        "reason": (
                                            f"no ball departure found — "
                                            f"impact ASSUMED at the pose "
                                            f"peak (f{_pk_f}, "
                                            f"{float(d.get('peak_time_sec') or 0.0):.2f}s); "
                                            f"ball located by the AI launch "
                                            f"plot from there"
                                        ),
                                        "ai_launch_n": _alp.get("n_found"),
                                        "ai_launch_reason": _alp.get("reason"),
                                        "ai_launch_image": _alp.get("image"),
                                        "ai_launch_points": _ai_pts,
                                        "search_box": _box,
                                        "box_ball_xy": _found_rest,
                                    }
                                    if _found_rest:
                                        # A ball located inside the box
                                        # beats an inferred launch point.
                                        d["ball_rest_xy"] = list(_found_rest)
                                        d["impact_pinned"] = True
                                        _anchor_rec["rest_xy"] = list(
                                            _found_rest,
                                        )
                                        _e["impact_pinned_f"] = _pk_f
                                    # MOG2 ACROSS THE LAUNCH WINDOW - unconditionally.
                                    # This used to sit inside `if _ai_pts:`, so on the
                                    # swings that need it most - the AI seeing nothing
                                    # at all - the pixel tracker never ran either, and
                                    # the swing fell through to MOG2-only with no
                                    # anchor. Both detectors get the same six frames.
                                    _lt_early = None
                                    _tr_from = _found_rest or [
                                        _box[0] + _box[2] / 2.0,
                                        _box[1] + _box[3] * 0.75,
                                    ]
                                    try:
                                        with _pt("launch_track_pixel"):
                                            _lt_early = track_launch_from_rest(
                                                src_path,
                                                (float(_tr_from[0]), float(_tr_from[1])),
                                                _pk_f, tee_fps,
                                                debug_dir=_dbg_dir,
                                                debug_prefix=(
                                                    f"launchearly-{upload_id}-{_tok}"
                                                ),
                                                max_seconds=6.5 / max(1.0, tee_fps),
                                            )
                                        _anchor_rec["early_n"] = _lt_early.get("n_found")
                                        _anchor_rec["early_reason"] = _lt_early.get("reason")
                                        _anchor_rec["early_image"] = _lt_early.get("image")
                                        _anchor_rec["early_image_heat"] = (
                                            _lt_early.get("image_heat")
                                        )
                                        log.info(
                                            "long-upload worker: launch window - AI %d "
                                            "pt(s), MOG2 %s pt(s) over the same frames",
                                            len(_ai_pts), _lt_early.get("n_found"),
                                        )
                                    except Exception as exc:  # noqa: BLE001
                                        log.warning("early MOG2 pass failed: %s", exc)
                                    _early_pts = list((_lt_early or {}).get("points") or [])
                                    # Whichever detector saw the ball first IS the rest
                                    # anchor. The AI wins ties; MOG2 alone is still a
                                    # real anchor, and far better than none.
                                    _ai_frames = {int(a["frame"]) for a in _ai_pts}
                                    _all_early = sorted(
                                        _ai_pts + [
                                            q for q in _early_pts
                                            if int(q["frame"]) not in _ai_frames
                                        ],
                                        key=lambda q: int(q["frame"]),
                                    )
                                    if not _found_rest and _all_early:
                                        _found_rest = [
                                            float(_all_early[0]["x"]),
                                            float(_all_early[0]["y"]),
                                        ]
                                        d["ball_rest_xy"] = list(_found_rest)
                                        d["impact_pinned"] = True
                                        _anchor_rec["rest_xy"] = list(_found_rest)
                                        _anchor_rec["verified"] = True
                                        _e["impact_pinned_f"] = _pk_f
                                    if _all_early:
                                        # Continuation: the tracker picks up after the
                                        # last point either detector found.
                                        try:
                                            with _pt("launch_track_pixel"):
                                                _lt = track_launch_from_rest(
                                                    src_path,
                                                    (float(_found_rest[0]),
                                                     float(_found_rest[1])),
                                                    _pk_f, tee_fps,
                                                    debug_dir=_dbg_dir,
                                                    debug_prefix=(
                                                        f"launchtrk-assumed-"
                                                        f"{upload_id}-{_tok}"
                                                    ),
                                                    seed_points=_all_early,
                                                )
                                            _merged_lp = {
                                                int(q["frame"]): q for q in _ai_pts
                                            }
                                            for q in (
                                                _early_pts + (_lt.get("points") or [])
                                            ):
                                                _merged_lp.setdefault(int(q["frame"]), {
                                                    "frame": int(q["frame"]),
                                                    "x": q["x"], "y": q["y"],
                                                })
                                            d["launch_points"] = [
                                                _merged_lp[k] for k in sorted(_merged_lp)
                                            ]
                                            _anchor_rec["launch_n"] = _lt.get("n_found")
                                            _anchor_rec["launch_reason"] = _lt.get("reason")
                                            _anchor_rec["launch_image"] = _lt.get("image")
                                            _anchor_rec["launch_image_heat"] = (
                                                _lt.get("image_heat")
                                            )
                                        except Exception as exc:  # noqa: BLE001
                                            log.warning(
                                                "assumed-impact launch tracker failed: %s",
                                                exc,
                                            )
                                            d["launch_points"] = _all_early
                                    # The frame we ASSUMED impact on, saved
                                    # so the panel can show exactly which
                                    # one it is.
                                    if _dbg_dir is not None:
                                        _anchor_rec[
                                            "assumed_impact_image"
                                        ] = _save_assumed_impact_frame(
                                            src_path, _pk_f, tee_fps,
                                            _seed, _ai_pts,
                                            f"assumedimp-{upload_id}-{_tok}",
                                            search_box=_box,
                                            box_ball_xy=_found_rest,
                                        )
                                    d["anchor_rec"] = {
                                        k: _anchor_rec.get(k)
                                        for k in (
                                            "verified", "assumed_impact",
                                            "snapped", "snap_px",
                                            "impact_frame", "impact_delta",
                                            "reason", "rest_xy",
                                            "assumed_impact_image",
                                            "launch_n", "launch_reason",
                                            "launch_image",
                                            "launch_image_heat",
                                            "ai_launch_n",
                                            "ai_launch_reason",
                                            "ai_launch_image",
                                            "ai_launch_points",
                                            "search_box", "box_ball_xy",
                                            "early_n", "early_reason",
                                            "early_image",
                                            "early_image_heat",
                                        )
                                    }
                            except Exception as exc:  # noqa: BLE001
                                log.warning(
                                    "long-upload worker: assumed-impact "
                                    "chain failed: %s", exc,
                                )
                        _work["practice"].append({
                            # Keyed by the ORIGINAL pose peak — the debug
                            # report matches records on it; the pin above
                            # may have shifted d's peak_time_sec.
                            "t": _t_orig,
                            "verdict": v.get("verdict"),
                            "reason": v.get("reason"),
                            "before": v.get("before"),
                            "after": v.get("after"),
                            "anchor": (
                                {
                                    k: _anchor_rec.get(k)
                                    for k in (
                                        "verified", "snapped", "snap_px",
                                        "impact_frame", "impact_delta",
                                        "reason", "image", "image_mog2",
                                        "ai_fallback_reason",
                                        # assumed-impact path (no departure)
                                        "assumed_impact",
                                        "assumed_impact_image", "rest_xy",
                                        "search_box", "box_ball_xy",
                                        "early_n", "early_reason",
                                        "early_image", "early_image_heat",
                                        "launch_n",
                                        "launch_reason", "launch_image",
                                        "launch_image_heat",
                                        "ai_launch_n",
                                        "ai_launch_reason",
                                        "ai_launch_image",
                                        "ai_launch_points",
                                    )
                                }
                                if _anchor_rec else None
                            ),
                        })
                        if v.get("verdict") == "practice":
                            _e["kept"] = False
                            _e["dropped_by"] = "practice_filter"
                            log.info(
                                "long-upload worker: upload=%s dropping practice "
                                "swing @ %.1fs (%s)", upload_id,
                                float(d.get("peak_time_sec") or 0.0), v.get("reason"),
                            )
                        else:
                            _real.append(d)
                    detected = _real

                try:
                    import json as _json
                    log.info(
                        "long-upload worker: upload=%s produce decisions: %s",
                        upload_id, _json.dumps(_decisions),
                    )
                except Exception:  # noqa: BLE001
                    pass

                # Publish the run's full work record — the Debug report
                # renders from THIS, so debug and produce are one run.
                _work["decisions"] = _decisions
                _work["kept"] = [dict(d) for d in detected]
                _work["fps"] = tee_fps
                _work["published"] = time.time()
                with _produce_work_lock:
                    _produce_work_state[upload_id] = _work

                for i, d in enumerate(detected):
                    segs.append(
                        {
                            "hole_number": (
                                starting_hole if single_hole else starting_hole + i
                            ),
                            "start_sec": d["start_sec"],
                            "end_sec": d["end_sec"],
                            "peak_time_sec": d.get("peak_time_sec"),
                            "ball_verdict": d.get("ball_verdict"),
                            # Pose hands-at-impact position — the tracer's
                            # start anchor when the ball can't be seen.
                            "impact_wrist_xy": d.get("impact_wrist_xy"),
                            # Pixel-verified anchors (departure pin): the
                            # tracer skips its audio/vision impact and
                            # ball-rest calls when these are present.
                            "ball_rest_xy": d.get("ball_rest_xy"),
                            "impact_pinned": bool(d.get("impact_pinned")),
                            "anchor_rec": d.get("anchor_rec"),
                            "launch_points": d.get("launch_points"),
                        }
                    )
                if not segs:
                    if motion_only or used_pose:
                        # Nothing to produce (no motion burst / no ball / no
                        # pose swing). A valid outcome — produce nothing, don't
                        # fail. Fall through with empty segs → 0 produced clips.
                        log.info(
                            "long-upload worker: upload=%s nothing to produce "
                            "— 0 clips (detector=%s)", upload_id,
                            "pose" if used_pose else "motion",
                        )
                    else:
                        _comb = _detect_debug.get("combined") or {}
                        _n = _comb.get("n_audio_candidates", 0)
                        _rc = _comb.get("rejection_counts") or {}
                        if _n > 0 and any(_rc.values()):
                            _bd = ", ".join(
                                f"{g}={c}" for g, c in _rc.items() if c > 0
                            )
                            raise RuntimeError(
                                f"no swings detected: {_n} candidate(s) considered, "
                                f"rejected by: {_bd}"
                            )
                        raise RuntimeError(
                            "no swings detected (audio found no candidates; "
                            "motion_bursts=%d — check logs for details)"
                            % _comb.get("n_motion_windows", 0)
                        )

            durations = [s["end_sec"] - s["start_sec"] for s in segs]
            log.info(
                "long-upload worker: upload=%s segs=%d source=%s durations=%s",
                upload_id,
                len(segs),
                "auto-combined" if auto_used else "manual",
                ", ".join(f"{d:.1f}s" for d in durations[:30]),
            )

            # Publish total + reset progress so the polling UI can show
            # "X/Y processed" while the heavy work runs. Wrapped in the
            # dead-connection retry: minutes of pose/anchor work ran
            # since the last DB touch, and Postgres (Neon) kills a
            # connection that idled inside an open transaction — the
            # retry re-fetches on a fresh connection and re-applies.
            def _publish_segments(s):
                r2 = s.get(LongVideoUpload, upload_id)
                if r2 is None:
                    return
                r2.last_n_segments = len(segs)
                r2.last_n_succeeded = 0
                # Persist the swing windows into edit_metrics so the
                # Edit-wizard for multi-swing uploads can hydrate
                # without re-running detect-swings.
                saved_em = dict(r2.edit_metrics or {})
                existing_swings = saved_em.get("swings") or []
                by_idx = {
                    int(sw.get("idx", -1)): sw
                    for sw in existing_swings if isinstance(sw, dict)
                }
                for i, seg in enumerate(segs):
                    if i in by_idx:
                        continue
                    start_sec = float(seg.get("start_sec") or 0.0)
                    end_sec = float(seg.get("end_sec") or start_sec)
                    start_frame = int(round(start_sec * (tee_fps or 30.0)))
                    end_frame = int(round(end_sec * (tee_fps or 30.0)))
                    _peak_t = float(
                        seg.get("peak_time_sec")
                        or (start_sec + CLIP_SECONDS_BEFORE_IMPACT)
                    )
                    impact_frame = int(round(_peak_t * (tee_fps or 30.0)))
                    address_frame = max(
                        start_frame,
                        impact_frame - int(round(1.5 * (tee_fps or 30.0))),
                    )
                    by_idx[i] = {
                        "idx": i,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "address_frame": address_frame,
                        "impact_frame": impact_frame,
                        "fps": round(tee_fps, 2) if tee_fps else None,
                    }
                saved_em["swings"] = [by_idx[i] for i in sorted(by_idx)]
                # Audit trail of this produce run's per-swing filter
                # decisions — the ground truth for "why did produce cut
                # these clips?".
                try:
                    saved_em["produce_decisions"] = _decisions
                except NameError:
                    pass
                r2.edit_metrics = saved_em

            _commit_retry(db, _publish_segments, "publish segments")
            row = db.get(LongVideoUpload, upload_id)
            if row is None:
                log.warning(
                    "long-upload worker: row %s deleted mid-run", upload_id,
                )
                return

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
                tee_green_delta_sec=float(tee_green_delta_sec),
                # Pose mode uses a wider, fixed window around the swing.
                clip_before=(settings.pose_clip_before_sec if used_pose else None),
                clip_after=(settings.pose_clip_after_sec if used_pose else None),
            )

            # Final bookkeeping — dead-connection-retried like the
            # publish above (the segment loop can run for many minutes).
            _n_ok = sum(1 for r in results if r.get("ok"))

            def _mark_completed(s):
                r2 = s.get(LongVideoUpload, upload_id)
                if r2 is None:
                    return
                r2.last_n_segments = len(segs)
                r2.last_n_succeeded = _n_ok
                r2.processing_status = "completed"
                r2.processing_completed_at = _utcnow_naive()
                r2.last_error = None

            _commit_retry(db, _mark_completed, "mark completed")
        except Exception as exc:
            log.exception("long-upload worker %s failed: %s", upload_id, exc)
            db.rollback()
            _err_txt = str(exc)[:2000]

            def _mark_failed(s):
                r2 = s.get(LongVideoUpload, upload_id)
                if r2 is None:
                    return
                r2.processing_status = "failed"
                r2.processing_completed_at = _utcnow_naive()
                r2.last_error = _err_txt

            _commit_retry(db, _mark_failed, "mark failed")
    finally:
        db.close()
        # Where the produce run went. Logged on EVERY produce (not just
        # Debug3 runs) because this is the wait an operator actually
        # sits through at the course. `other` is whatever the wrapped
        # helpers and AI calls don't cover — if it dominates, the next
        # thing to instrument is inside it, not a guess.
        _phases = _pt_snapshot()
        _job_total = round(time.perf_counter() - _job_t0, 2)
        _phases["other"] = round(
            max(0.0, _job_total - sum(_phases.values())), 3,
        )
        _pct = (lambda v: 100.0 * v / _job_total) if _job_total > 0 else (
            lambda v: 0.0
        )
        log.info(
            "produce timing upload=%s total=%.1fs: %s",
            upload_id, _job_total,
            ", ".join(
                f"{k} {v}s ({_pct(v):.0f}%)"
                for k, v in sorted(_phases.items(), key=lambda kv: -kv[1])
                if v > 0
            ) or "nothing measured",
        )


def _build_tracer_diagnostics(
    pipe: dict, examples_by_kind: dict | None, ball_verdict: str | None = None,
) -> dict:
    """Collect what the AI tracer produced + which prior examples it
    was shown for this clip. Stored on VideoClip.tracer_diagnostics so
    we can measure whether few-shot examples improved picks. Cheap to
    construct; safe to call even when the pipeline errored out.

    `ball_verdict` is the tee-camera ball-departure classification for
    this swing ("departed" / "present" / "no_ball" / "uncertain"), carried
    through so we can review — and tune — why each swing was or wasn't
    treated as a confirmed shot, straight from the produced clip."""
    p = pipe or {}
    # `pipe` is either the full run_full_ai_tracer_pipeline result (nested
    # address/handedness/impact dicts) or _trace_segment's condensed info
    # (flat address_frame/impact_frame ints, handedness as a plain string).
    addr = p.get("address") or {}
    if not isinstance(addr, dict):
        addr = {}
    hand = p.get("handedness") or {}
    if not isinstance(hand, dict):
        hand = {"handedness": hand}
    impact = p.get("impact") or {}
    if not isinstance(impact, dict):
        impact = {}
    examples_summary = {}
    if examples_by_kind:
        for kind, exs in examples_by_kind.items():
            if not exs:
                continue
            examples_summary[kind] = [
                {"lvu_id": e.lvu_id, "hole": e.hole_number} for e in exs
            ]
    return {
        "examples": examples_summary,
        "ai_picks": {
            "address_frame": addr.get("address_frame", p.get("address_frame")),
            "address_method": addr.get("method"),
            "address_confidence": addr.get("confidence"),
            "handedness": hand.get("handedness"),
            "handedness_confidence": hand.get("confidence"),
            "impact_frame": impact.get("impact_frame", p.get("impact_frame")),
            "impact_method": impact.get("method"),
            "impact_confidence": impact.get("confidence"),
        },
        "model": addr.get("model") or impact.get("model"),
        "ball_verdict": ball_verdict,
        "ts": datetime.utcnow().isoformat(),
    }


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
    tee_green_delta_sec: float = 0.0,
    clip_before: float | None = None,
    clip_after: float | None = None,
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

        def _apply(s):
            r2 = s.get(LongVideoUpload, progress_upload_id)
            if r2 is not None:
                r2.last_n_succeeded = done_so_far

        _commit_retry(db, _apply, "bump progress")

    # Cache the course once per call — _intro_overlay_for_clip needs
    # course.name / par3_holes / hole_yardages for every segment.
    _course_for_intro: Course | None = db.get(Course, course_id)

    def _intro_overlay_for_clip(
        clip: VideoClip, participant: Participant | None
    ) -> None:
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
        # Preserve a CLEAN pre-overlay copy first — vertical re-renders
        # (and any future format) crop/pan the frame, so they must start
        # from footage without the landscape panels baked in.
        try:
            import shutil as _sh

            _clean = fpath.with_name(f"{fpath.stem}_clean{fpath.suffix}")
            if not _clean.exists():
                _sh.copy2(fpath, _clean)
        except Exception as exc:  # noqa: BLE001
            log.warning("clean-copy failed for %s: %s", fpath.name, exc)
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

    def _intro_overlay_for_vertical(
        clip: VideoClip, participant: Participant | None
    ) -> None:
        """Overlay the name plate + persistent logo onto the VERTICAL
        variant with portrait geometry (the landscape overlay lives in
        corners the 9:16 crop pans away from). Target sign is skipped —
        its coords are landscape-frame pixels. Best-effort."""
        if not clip.vertical_url:
            return
        fname = clip.vertical_url.split("?")[0].rsplit("/", 1)[-1]
        fpath = CLIPS_DIR / fname
        if not fpath.exists():
            return
        course = _course_for_intro
        course_name = course.name if course and course.name else ""
        yardage = 101
        if course and course.hole_yardages:
            raw_y = course.hole_yardages.get(str(int(clip.hole_number)))
            try:
                if raw_y is not None:
                    yardage = int(raw_y)
            except (TypeError, ValueError):
                pass
        try:
            if apply_intro_overlay_inplace(
                fpath,
                player_name=(
                    participant.name if participant else "Brent Baldwin"
                ),
                course_name=course_name,
                hole_number=int(clip.hole_number),
                par=3,
                yardage=yardage,
            ):
                clip.vertical_url = (
                    f"{settings.app_base_url}/uploads/clips/{fpath.name}"
                    f"?v={int(fpath.stat().st_mtime)}"
                )
        except Exception as exc:  # pragma: no cover
            log.warning(
                "vertical intro overlay failed for clip %s: %s",
                clip.id, exc,
            )

    # Effective clip window. clip_before/clip_after (pose mode) override the
    # defaults; the tee→green cutover (_tee_after) stays put and the green
    # portion is stretched/shrunk so the total post-swing coverage == after.
    _before = clip_before if clip_before is not None else CLIP_SECONDS_BEFORE_IMPACT
    _tee_after = CLIP_SECONDS_TEE_AFTER_IMPACT
    if clip_after is not None:
        _tee_only_after = float(clip_after)
        _green_after = max(0.5, float(clip_after) - _tee_after)
    else:
        _tee_only_after = CLIP_SECONDS_TEE_ONLY_AFTER_IMPACT
        _green_after = CLIP_SECONDS_GREEN_AFTER_CUT

    # Source fps, probed once — used to map each swing's segment-relative
    # ball track back into full-clip frame indices for the Edit wizard.
    _src_fps = probe_fps(src_path) or 30.0
    # Native TEE frame size, probed once. Every consumer of the persisted
    # pixel coords — click-to-plot (`p.x / frame_width`), the follow-the-
    # shot crop, the Edit wizard — normalizes against edit_metrics
    # .frame_width / tee_width, which are BOTH the native tee source size.
    # The pipeline, however, measures on the cut, and the cut is not
    # guaranteed to still be native (see _persist_swing_track).
    _src_info = probe_video_info(src_path) or {}
    _src_w = float(_src_info.get("width") or 0)
    _src_h = float(_src_info.get("height") or 0)

    def _persist_swing_track(
        swing_idx, tracer_info, tracer_url, cut_start_sec, cut_end_sec=None,
        clip_id=None, seg_dims=None,
    ):
        """Save everything this swing's production run figured out into
        edit_metrics.swings, so the Edit wizard opens fully pre-populated
        instead of re-running the AI pipeline on the whole multi-swing
        source: the actual cut window (start/end frame), the refined
        impact + address frames, handedness, the resting-ball position,
        the per-frame ball track, and the rendered tracer.

        The pipeline runs on the CUT segment (frame 0 = the cut's start),
        so every FRAME index is shifted by the cut's start offset to land
        in the full-clip frame space the wizard works in.

        PIXEL coords can need mapping too. The old comment here claimed
        "the cut trims time, not space", which is true of cut_segment but
        not of compress_for_email — that re-encodes IN PLACE through
        `scale=min(1280,iw)`, and the single-camera branch used to run it
        on the segment BEFORE tracing. On a 1920-wide tee source that left
        every point in 1280 space while click-to-plot divided by
        frame_width=1920, so a dot at the true centre landed at 33%
        instead of 50%: the whole plot shifted left, worsening across the
        frame.

        Both branches now trace BEFORE that re-encode, so the tracer reads
        native pixels and the factor below is 1.0. `seg_dims` is the size
        the caller saw at trace time and keeps that guarantee honest
        rather than assumed — but it has to be measured there, because by
        the time this runs the delivery re-encode has already shrunk the
        file on disk. (Probing the file here instead is what briefly
        pushed the plot off to the RIGHT: correct native points multiplied
        by a phantom 1.5.) An operator-marked ball (ball_manual) is never
        overwritten."""
        if progress_upload_id is None or not tracer_info:
            return
        offset = int(round((cut_start_sec or 0.0) * _src_fps))
        # Cut → native pixel scale. seg_dims MUST be the size the segment
        # was when the TRACER read it, captured by the caller right
        # before the trace — NOT probed here. By the time this runs the
        # caller has already sent the segment through compress_for_email
        # for delivery, so the file on disk is 1280-capped whatever the
        # tracer saw; probing it here read 1280 against a 1920 source and
        # scaled correct native points UP by 1.5, throwing the plot off
        # to the right exactly as far as it had previously been off to
        # the left.
        _sx = _sy = 1.0
        if seg_dims and _src_w > 0 and _src_h > 0:
            _seg_w, _seg_h = (float(seg_dims[0] or 0), float(seg_dims[1] or 0))
            if _seg_w > 0 and _seg_h > 0:
                _sx, _sy = _src_w / _seg_w, _src_h / _seg_h
        _scaled = abs(_sx - 1.0) > 1e-6 or abs(_sy - 1.0) > 1e-6
        if _scaled:
            log.info(
                "swing %s: cut is %.0fx%.0f vs native %.0fx%.0f — scaling "
                "detections by (%.4f, %.4f) into tee pixel space",
                swing_idx, _src_w / _sx, _src_h / _sy, _src_w, _src_h,
                _sx, _sy,
            )

        # Cut-space → native-space. None passes through, and when the cut
        # IS native the value is handed back untouched rather than
        # round-tripped through float — the unscaled path stays exactly
        # what it was before this mapping existed.
        def _mx(v):
            return v if (v is None or not _scaled) else float(v) * _sx

        def _my(v):
            return v if (v is None or not _scaled) else float(v) * _sy
        # AI engine emits ball_track_frames; the classical fallback emits
        # `track` ({frame,x,y} raw detections) — same shape after mapping.
        seg_frames = tracer_info.get("ball_track_frames") or []
        if not seg_frames:
            seg_frames = [
                {"frame": rec.get("frame"), "found": True,
                 "x": rec.get("x"), "y": rec.get("y")}
                for rec in (tracer_info.get("track") or [])
            ]
        mapped = []
        for rec in seg_frames:
            f = rec.get("frame")
            if f is None:
                continue
            mapped.append({
                "frame": int(f) + offset,
                "found": bool(rec.get("found")),
                "x": _mx(rec.get("x")),
                "y": _my(rec.get("y")),
                "confidence": rec.get("confidence"),
                "manual": bool(rec.get("manual", False)),
                # 'mog2' when the point came from the MOG2 layer-in
                # extension rather than an AI pick.
                "source": rec.get("source"),
                "image_url": None,
            })
        try:
            row = db.get(LongVideoUpload, progress_upload_id)
            if row is None:
                return
            em = dict(row.edit_metrics or {})
            swings = list(em.get("swings") or [])
            slot = None
            for i, sw in enumerate(swings):
                if isinstance(sw, dict) and int(sw.get("idx", -1)) == swing_idx:
                    slot = i
                    break
            nsw = dict(swings[slot]) if slot is not None else {"idx": swing_idx}
            # The window of the clip this run actually cut — source of
            # truth for the wizard's start/impact/end frames.
            nsw["start_frame"] = offset
            if cut_end_sec is not None:
                nsw["end_frame"] = int(round(float(cut_end_sec) * _src_fps))
            nsw["fps"] = round(_src_fps, 2)
            # The pixel space the coords above are in, stated rather than
            # implied. Always the native tee size after the scaling above;
            # recorded so a consumer can assert it instead of guessing,
            # and so a future mismatch shows up in the debug report.
            if _src_w > 0 and _src_h > 0:
                nsw["track_frame_width"] = int(_src_w)
                nsw["track_frame_height"] = int(_src_h)
                nsw["track_scaled_from_cut"] = _scaled
            # Stamp WHEN this record was written: the debug report's
            # production-tracer poll uses it to reject a previous run's
            # leftover entry (same swing idx, valid tracer_url) that
            # would otherwise satisfy the poll instantly and show stale
            # anchors/errors next to the fresh run's panels.
            nsw["persisted_at"] = round(time.time(), 2)
            # The VideoClip this swing produced into — per-swing editors
            # (click-to-plot, wizard Produce) commit against this id, so
            # the mapping survives individual clip deletions that would
            # break position-based matching.
            if clip_id is not None:
                nsw["clip_id"] = int(clip_id)
            _imp = tracer_info.get("impact_frame")
            if _imp is not None:
                nsw["impact_frame"] = int(_imp) + offset
            _addr = tracer_info.get("address_frame")
            if _addr is not None:
                nsw["address_frame"] = int(_addr) + offset
            elif _imp is not None:
                nsw["address_frame"] = max(
                    offset, int(_imp) + offset - int(round(1.5 * _src_fps)),
                )
            if tracer_info.get("handedness"):
                nsw["handedness"] = tracer_info["handedness"]
            # Resting-ball position — but not the pose-hands fallback
            # (that's the golfer's hands, not a ball) and never on top
            # of a ball the operator placed by hand.
            _rest = tracer_info.get("ball_rest_xy")
            if (
                _rest
                and len(_rest) == 2
                and tracer_info.get("ball_rest_source") != "pose_wrist_fallback"
                and not nsw.get("ball_manual")
            ):
                nsw["ball"] = {
                    "x": int(round(_mx(_rest[0]))),
                    "y": int(round(_my(_rest[1]))),
                }
            if mapped:
                nsw["ball_track_frames"] = mapped
            if tracer_url:
                nsw["tracer_url"] = tracer_url
            if mapped or tracer_url:
                nsw["tracer_engine"] = tracer_info.get("engine") or "ai"
            # Anchor check (rest snap + departure-frame impact) — the
            # pixel verification's verdict and film-strip, shown in the
            # debug report so the operator can SEE what was tried.
            _ac = tracer_info.get("anchor_check")
            if _ac:
                _ac_entry = {
                    k: _ac.get(k)
                    for k in (
                        "verified", "snapped", "snap_px", "impact_delta",
                        "present_ratio_pre", "reason",
                        "ai_fallback_reason", "ai_launch_points",
                        "ai_launch_n", "ai_launch_reason",
                    )
                }
                if _ac.get("ai_launch_image") and (
                    CLIPS_DIR / _ac["ai_launch_image"]
                ).exists():
                    _alp = CLIPS_DIR / _ac["ai_launch_image"]
                    _ac_entry["ai_launch_image_url"] = (
                        f"{settings.app_base_url}/uploads/clips/"
                        f"{_alp.name}?v={int(_alp.stat().st_mtime)}"
                    )
                if _ac.get("image") and (CLIPS_DIR / _ac["image"]).exists():
                    _acp = CLIPS_DIR / _ac["image"]
                    _ac_entry["image_url"] = (
                        f"{settings.app_base_url}/uploads/clips/"
                        f"{_acp.name}?v={int(_acp.stat().st_mtime)}"
                    )
                if _ac.get("image_mog2") and (
                    CLIPS_DIR / _ac["image_mog2"]
                ).exists():
                    _acp2 = CLIPS_DIR / _ac["image_mog2"]
                    _ac_entry["image_mog2_url"] = (
                        f"{settings.app_base_url}/uploads/clips/"
                        f"{_acp2.name}?v={int(_acp2.stat().st_mtime)}"
                    )
                nsw["anchor_check"] = _ac_entry
            # MOG2 layer-in evidence: overlay image (raw motion heat +
            # AI picks + MOG2 chain + added points) and the match/extend
            # stats — shown via the button under the produced video.
            _ovl = tracer_info.get("mog2_overlay_image")
            if _ovl and (CLIPS_DIR / _ovl).exists():
                nsw["mog2_overlay_url"] = (
                    f"{settings.app_base_url}/uploads/clips/{_ovl}"
                    f"?v={int((CLIPS_DIR / _ovl).stat().st_mtime)}"
                )
            if tracer_info.get("mog2"):
                nsw["mog2_stats"] = tracer_info["mog2"]
            if tracer_info.get("render_info") is not None:
                nsw["render_info"] = tracer_info["render_info"]
            # Timed transient dots (mapped to source frames) + the raw
            # motion heat image — the wizard's click-to-plot view opens
            # straight from these, no in-session re-render needed.
            _tp = tracer_info.get("timed_points") or []
            if _tp:
                nsw["timed_points"] = [
                    {
                        "frame": int(p["frame"]) + offset,
                        "x": int(round(_mx(p["x"]))),
                        "y": int(round(_my(p["y"]))),
                    }
                    for p in _tp
                    if p.get("frame") is not None
                    and (_imp is None or int(p["frame"]) >= int(_imp))
                ][:2000]
            # Denser candidate pool for click-to-plot's zoomed-in layer.
            # The classical fallback's tracer info carries "candidates"
            # natively; the AI path gets them from the MOG2 layer.
            # Flight-window only (impact − 2 on) — the fallback path's
            # pool isn't heat-windowed, and without the filter the 1500
            # cap would fill up with pre-swing body motion.
            _cp = tracer_info.get("candidates") or []
            if _cp:
                nsw["cand_points"] = [
                    {
                        "frame": int(p["frame"]) + offset,
                        "x": int(round(_mx(p["x"]))),
                        "y": int(round(_my(p["y"]))),
                    }
                    for p in _cp
                    if p.get("frame") is not None
                    and (_imp is None or int(p["frame"]) >= int(_imp))
                ][:1500]
            _rawm = tracer_info.get("raw_motion_image")
            if _rawm and (CLIPS_DIR / _rawm).exists():
                nsw["tracer_raw_motion_url"] = (
                    f"{settings.app_base_url}/uploads/clips/{_rawm}"
                    f"?v={int((CLIPS_DIR / _rawm).stat().st_mtime)}"
                )
            if slot is not None:
                swings[slot] = nsw
            else:
                swings.append(nsw)
                swings.sort(key=lambda s: int(s.get("idx", 0)))
            em["swings"] = swings
            row.edit_metrics = em
            db.commit()
            log.info(
                "long-upload: persisted swing idx=%s for the wizard — "
                "%d track points, window f%s–%s, impact=%s, ball=%s "
                "(offset=%d frames)",
                swing_idx, len(mapped), nsw.get("start_frame"),
                nsw.get("end_frame"), nsw.get("impact_frame"),
                nsw.get("ball"), offset,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("persist swing track failed (idx=%s): %s", swing_idx, exc)

    n_done = 0
    results: list[dict] = []
    for idx, seg in enumerate(seg_list):
        # Release any open transaction BEFORE this swing's minutes of
        # ffmpeg/tracer work — a connection idling inside a transaction
        # is exactly what Neon's idle-in-transaction timeout kills, and
        # the corpse then breaks the clip INSERT at the end of the
        # swing. (The insert also retries on a fresh connection; this
        # just stops the kills from happening at all.)
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass
        try:
            hole_number = int(seg["hole_number"])
            start_sec = float(seg["start_sec"])
            end_sec = float(seg["end_sec"])
        except (KeyError, TypeError, ValueError):
            results.append(
                {
                    "index": idx,
                    "ok": False,
                    "error": "missing or invalid hole_number / start_sec / end_sec",
                }
            )
            continue
        if end_sec <= start_sec:
            results.append(
                {"index": idx, "ok": False, "error": "end_sec must be > start_sec"}
            )
            continue

        # ── Impact anchor ────────────────────────────────────────────────────
        # Use the detector's peak_time_sec when available; fall back to
        # CLIP_SECONDS_BEFORE_IMPACT into the segment window.
        _raw_peak = seg.get("peak_time_sec")
        peak_time_sec = (
            float(_raw_peak) if _raw_peak is not None
            else start_sec + _before
        )

        # ── Tee cut bounds ───────────────────────────────────────────────────
        tee_cut_start = max(0.0, peak_time_sec - _before)
        if dual_camera and green_src_path is not None:
            tee_cut_end = peak_time_sec + _tee_after + _green_after
        else:
            tee_cut_end = peak_time_sec + _tee_only_after
        # Within-segment offset to the impact moment (≤ _before when the
        # impact is very close to the start of the recording).
        actual_before_sec = peak_time_sec - tee_cut_start
        # TEE video duration in the composite (before the hard cut to green).
        tee_video_dur = actual_before_sec + _tee_after
        log.info(
            "long-upload seg %d: peak_time=%.2fs tee cut [%.2f, %.2f] "
            "tee_video_dur=%.2fs dual_camera=%s",
            idx, peak_time_sec, tee_cut_start, tee_cut_end, tee_video_dur, dual_camera,
        )

        seg_name = f"{course_id}-h{hole_number}-{secrets.token_hex(6)}.mp4"
        seg_path = CLIPS_DIR / seg_name
        ok = cut_segment(src_path, seg_path, tee_cut_start, tee_cut_end)
        if not ok:
            results.append(
                {
                    "index": idx,
                    "ok": False,
                    "error": "ffmpeg cut failed (or ffmpeg not installed)",
                }
            )
            continue

        # --- Dual-camera branch -----------------------------------
        if dual_camera and green_src_path is not None:
            green_seg_name = (
                f"{course_id}-h{hole_number}-green-{secrets.token_hex(6)}.mp4"
            )
            green_seg_path = CLIPS_DIR / green_seg_name
            # ── Green cut bounds ─────────────────────────────────────────────
            # The green camera starts at a different wall-clock time than the
            # tee camera.  tee_green_delta_sec = green_start − tee_start.
            # A positive delta means green started LATER, so the same
            # real-world instant is EARLIER in the green clip.
            #
            #   green_impact_offset = peak in green coords
            #                       = peak_in_tee − delta
            #
            # Cut the green from (green_impact + TEE_AFTER) to
            # (green_impact + TEE_AFTER + GREEN_AFTER).
            green_impact_offset = peak_time_sec - tee_green_delta_sec
            green_cut_start = max(
                0.0, green_impact_offset + _tee_after
            )
            green_cut_end = (
                green_impact_offset + _tee_after + _green_after
            )
            if green_cut_end <= green_cut_start:
                green_cut_end = green_cut_start + _green_after
            log.info(
                "long-upload seg %d: green cut [%.2f, %.2f] (delta=%.3fs)",
                idx, green_cut_start, green_cut_end, tee_green_delta_sec,
            )
            green_cut_ok = cut_segment(
                green_src_path, green_seg_path, green_cut_start, green_cut_end
            )
            if not green_cut_ok:
                log.warning(
                    "long-upload seg %d: green cut failed — falling back to "
                    "tee-only composite",
                    idx,
                )
                green_seg_path = None  # splice_impact_clip handles the fallback

            # Ball-flight tracer — AI by default (settings.tracer_engine),
            # classical fallback. The cut to green is driven by tee_video_dur,
            # not the tracer, so the engine choice doesn't affect the cut.
            _seg_fps = probe_fps(seg_path) or 30.0
            # The pixel space the tracer is about to measure in. Probed
            # HERE because the segment gets re-encoded (and downscaled)
            # for delivery further down, so this is the last moment the
            # file still has the dimensions the detections will be in.
            _seg_i = probe_video_info(seg_path) or {}
            _seg_dims = (_seg_i.get("width"), _seg_i.get("height"))
            _cut_off = int(round(tee_cut_start * _src_fps))
            _lp_all = _seg_launch_points(seg)
            _lp_cut = [
                {**pt, "frame": int(pt["frame"]) - _cut_off}
                for pt in _lp_all
                if int(pt["frame"]) >= _cut_off
            ]
            log.info(
                "produce: swing hole=%s AI launch points -> render: "
                "%d of %d (cut_off=f%d)",
                seg.get("hole_number"), len(_lp_cut), len(_lp_all),
                _cut_off,
            )
            _tracer_url, tracer_info, traced_path, _debug_url = _trace_segment(
                seg_path,
                ball_at_rest_override=seg.get("impact_wrist_xy"),
                verified_rest_xy=seg.get("ball_rest_xy"),
                verified_impact_frame=(
                    int(round(actual_before_sec * _seg_fps))
                    if seg.get("impact_pinned") else None
                ),
                launch_points=_lp_cut or None,
            )
            tracer_ok = bool(tracer_info and tracer_info.get("ok"))

            composite_url = None
            composite_info: dict | None = None
            composite_path: Path | None = None
            tracer_path = traced_path
            # For the composite VIDEO use the tracer-overlaid tee clip when
            # available; fall back to the raw tee cut.  Either way the AUDIO
            # comes from the tee source (splice_impact_clip maps 0:a? from
            # the first input).
            tee_source_for_composite = (
                tracer_path
                if (tracer_ok and tracer_path is not None and tracer_path.exists())
                else seg_path
            )
            # Clamp the green video window to the clip's actual duration so we
            # never ask ffmpeg to trim past EOF.
            green_video_dur = _green_after
            if green_seg_path is not None and green_seg_path.exists():
                _ginfo = probe_video_info(green_seg_path)
                _gdur = float(_ginfo.get("duration") or 0.0)
                if _gdur > 0.1:
                    green_video_dur = min(_green_after, _gdur)
            composite_name = f"{seg_path.stem}_composite.mp4"
            composite_path = CLIPS_DIR / composite_name
            if splice_impact_clip(
                tee_source_for_composite,
                tee_video_dur,
                green_seg_path,
                green_video_dur,
                composite_path,
            ):
                compress_for_email(composite_path)
                if composite_path.exists() and composite_path.stat().st_size > 0:
                    composite_url = (
                        f"{settings.app_base_url}/uploads/clips/{composite_name}"
                    )
                    composite_info = {
                        "tee_video_dur_sec": round(tee_video_dur, 2),
                        "green_video_dur_sec": round(green_video_dur, 2),
                        "total_dur_sec": round(tee_video_dur + green_video_dur, 2),
                        "impact_offset_in_tee_sec": round(actual_before_sec, 2),
                        "tee_green_delta_sec": round(tee_green_delta_sec, 3),
                        "fps": probe_fps(seg_path),
                        "method": "classical-cv",
                    }

            thumb_source = (
                tracer_path if (tracer_path and tracer_path.exists()) else seg_path
            )
            thumb_path = extract_thumbnail(thumb_source)
            thumb_url = (
                f"{settings.app_base_url}/uploads/clips/{thumb_path.name}"
                if thumb_path
                else None
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
                public_source = (
                    f"{settings.app_base_url}/uploads/clips/{tracer_path.name}"
                )
                public_tracer = public_source
            else:
                compress_for_email(seg_path)
                public_source = f"{settings.app_base_url}/uploads/clips/{seg_name}"
                public_tracer = None

            captured_dt = base_dt + timedelta(seconds=tee_cut_start)

            # 9:16 vertical variant of the final video (blur-padded, so
            # the tracer + ball flight stay fully visible) for social /
            # phone. Best-effort: a failed render just leaves
            # vertical_url null and the on-demand endpoint can retry.
            vertical_url: str | None = None
            _vert_src = (
                composite_path
                if (composite_path and composite_path.exists())
                else (
                    tracer_path
                    if (tracer_path and tracer_path.exists())
                    else seg_path
                )
            )
            if _vert_src and _vert_src.exists():
                _vert_path = CLIPS_DIR / f"{_vert_src.stem}_vertical.mp4"
                _made = False
                _sw = 0.0
                _rxy = None
                _gx = None
                try:
                    _sinfo = probe_video_info(seg_path) or {}
                    _sw = float(_sinfo.get("width") or 0)
                    _rxy = (
                        (tracer_info or {}).get("ball_rest_xy")
                        or seg.get("ball_rest_xy")
                    )
                    # FOLLOW-THE-SHOT PAN: drive the 9:16 crop along
                    # the tracked ball flight (camera-operator style).
                    _trk = [
                        r
                        for r in (
                            (tracer_info or {}).get("ball_track_frames")
                            or []
                        )
                        if r.get("found") and r.get("x") is not None
                    ]
                    _gx = _probe_golfer_x_frac(_vert_src)
                    if len(_trk) >= 3 and _sw > 0 and _seg_fps:
                        _cut = None
                        if composite_url:
                            try:
                                _cut = float(tee_video_dur)
                            except NameError:
                                _cut = None
                        _imp_t = None
                        try:
                            _if = (tracer_info or {}).get("impact_frame")
                            if _if is not None:
                                _imp_t = float(_if) / float(_seg_fps)
                        except (TypeError, ValueError):
                            _imp_t = None
                        _ppath = _vertical_pan_path(
                            _trk, _rxy, float(_seg_fps), _sw, _cut,
                            golfer_x=_gx, impact_t=_imp_t,
                        )
                        _made = make_vertical_pan(
                            _vert_src, _vert_path, _ppath,
                        )
                except Exception as exc:  # noqa: BLE001
                    log.warning("vertical pan build failed: %s", exc)
                log.info(
                    "produce seg %d: vertical -> %s",
                    idx, "PAN" if _made else "static (no usable track)",
                )
                if not _made:
                    # Static crop aimed at the golfer (no usable track).
                    _focus = 0.5
                    try:
                        if _gx is not None:
                            _focus = max(0.15, min(0.85, float(_gx)))
                        elif _rxy and len(_rxy) == 2 and _sw > 0:
                            _focus = max(
                                0.15, min(0.85, float(_rxy[0]) / _sw),
                            )
                    except Exception:  # noqa: BLE001
                        _focus = 0.5
                    _made = make_vertical(
                        _vert_src, _vert_path, focus_x_frac=_focus,
                    )
                if _made and _vert_path.exists():
                    vertical_url = (
                        f"{settings.app_base_url}/uploads/clips/"
                        f"{_vert_path.name}?v={int(_vert_path.stat().st_mtime)}"
                    )

            # The session has idled through minutes of render/composite
            # work — Neon may have killed the connection (idle-in-
            # transaction timeout). Build + insert the clip through the
            # dead-connection retry: each attempt constructs a FRESH row
            # object so a half-flushed casualty of a dead session never
            # gets re-added.
            _clip_holder: dict = {}

            def _insert_clip(_db):
                _fields = dict(
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
                    distance_from_pin_feet=_optional_int(
                        seg.get("distance_from_pin_feet"),
                    ),
                    ball_in_cup=bool(seg.get("ball_in_cup", False)),
                    vertical_url=vertical_url,
                    processing_status=ClipProcessingStatus.received.value,
                    tracer_diagnostics=_build_tracer_diagnostics(
                        tracer_info, None,
                        ball_verdict=seg.get("ball_verdict"),
                    ),
                )
                _c = VideoClip(**_fields)
                _db.add(_c)
                _db.flush()
                participant = match_clip(_db, _c)
                if participant and _c.ball_in_cup:
                    notifications.notify_hio_under_review(
                        participant.name, participant.mobile,
                        participant.email,
                    )
                _intro_overlay_for_clip(_c, participant)
                _intro_overlay_for_vertical(_c, participant)
                _clip_holder["clip"] = _c
                # The matched participant is needed by the results
                # payload below; it lives in THIS closure now, so hand
                # it back out explicitly (reading it from the enclosing
                # scope raised UnboundLocalError after the clip insert
                # moved in here).
                _clip_holder["participant"] = participant

            if not _commit_retry(db, _insert_clip, "publish swing clip"):
                raise RuntimeError(
                    "could not save the produced clip row (DB connection "
                    "kept failing)",
                )
            clip = _clip_holder["clip"]
            participant = _clip_holder.get("participant")
            # Save the swing's detections for the Edit wizard (see helper).
            if seg.get("anchor_rec") and isinstance(tracer_info, dict):
                tracer_info.setdefault("anchor_check", seg["anchor_rec"])
            _persist_swing_track(
                idx, tracer_info, _tracer_url, tee_cut_start, tee_cut_end,
                clip_id=clip.id, seg_dims=_seg_dims,
            )

            results.append(
                {
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
                    "tracer_error": (tracer_info or {}).get("error"),
                }
            )
            n_done += 1
            _bump_progress(n_done)
            continue

        # --- Single-camera (original) branch ----------------------
        # NOTE: compress_for_email(seg_path) runs AFTER the tracer, not
        # before it. It re-encodes in place through scale=min(1280,iw),
        # so calling it first handed the tracer a downscaled segment while
        # the launch points fed IN (_lp_cut) and every consumer of the
        # points coming OUT stayed in native tee pixels — the mismatch
        # behind the left-shifted click-to-plot dots. The dual-camera
        # branch above already traces before it compresses.
        thumb_path = extract_thumbnail(seg_path)
        thumb_url = (
            f"{settings.app_base_url}/uploads/clips/{thumb_path.name}"
            if thumb_path
            else None
        )
        _seg_fps = probe_fps(seg_path) or 30.0
        # See the dual-camera branch: the tracer's pixel space, probed
        # before the delivery re-encode shrinks the file underneath us.
        _seg_i = probe_video_info(seg_path) or {}
        _seg_dims = (_seg_i.get("width"), _seg_i.get("height"))
        _cut_off = int(round(tee_cut_start * _src_fps))
        _lp_all = _seg_launch_points(seg)
        _lp_cut = [
            {**pt, "frame": int(pt["frame"]) - _cut_off}
            for pt in _lp_all
            if int(pt["frame"]) >= _cut_off
        ]
        log.info(
            "produce: swing hole=%s AI launch points -> render: "
            "%d of %d (cut_off=f%d)",
            seg.get("hole_number"), len(_lp_cut), len(_lp_all),
            _cut_off,
        )
        tracer_url, tracer_info, _, _ = _trace_segment(
            seg_path,
            ball_at_rest_override=seg.get("impact_wrist_xy"),
            verified_rest_xy=seg.get("ball_rest_xy"),
            verified_impact_frame=(
                int(round(actual_before_sec * _seg_fps))
                if seg.get("impact_pinned") else None
            ),
            launch_points=_lp_cut or None,
        )
        # Now that nothing measures against it, shrink the segment for
        # delivery (source_url below serves this exact file).
        compress_for_email(seg_path)

        captured_dt = base_dt + timedelta(seconds=tee_cut_start)
        _fields = dict(
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
            tracer_diagnostics=_build_tracer_diagnostics(
                None, None, ball_verdict=seg.get("ball_verdict"),
            ),
        )
        clip = VideoClip(**_fields)
        db.add(clip)
        db.flush()

        participant = match_clip(db, clip)
        if participant and clip.ball_in_cup:
            notifications.notify_hio_under_review(
                participant.name, participant.mobile, participant.email
            )
        _intro_overlay_for_clip(clip, participant)
        db.commit()
        # Save the swing's detections for the Edit wizard (see helper).
        if seg.get("anchor_rec") and isinstance(tracer_info, dict):
            tracer_info.setdefault("anchor_check", seg["anchor_rec"])
        _persist_swing_track(
            idx, tracer_info, tracer_url, tee_cut_start, tee_cut_end,
            clip_id=clip.id, seg_dims=_seg_dims,
        )

        results.append(
            {
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
            }
        )
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
    candidate_urls = [
        clip.source_url, clip.tracer_url, clip.thumbnail_url,
        clip.vertical_url,
    ]
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
                if (
                    fp.name == stem
                    or fp.name.startswith(stem + ".")
                    or fp.name.startswith(stem + "_")
                ):
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

    # NOTHING MAY STILL POINT AT THE ROW. Every one of these is a real
    # foreign key, so a single dangling reference turned the delete into
    # an IntegrityError -- the operator confirmed, the request 500'd, and
    # the clip stayed exactly where it was. Camera-sourced clips always
    # have at least one: camera_events.produced_clip_id is stamped the
    # moment the clip is produced.
    db.query(CameraEvent).filter(
        CameraEvent.produced_clip_id == clip_id,
    ).update({"produced_clip_id": None}, synchronize_session=False)
    for _col in (
        HoleInOneEvent.tee_clip_id,
        HoleInOneEvent.wide_clip_id,
        HoleInOneEvent.hole_clip_id,
    ):
        db.query(HoleInOneEvent).filter(_col == clip_id).update(
            {_col: None}, synchronize_session=False,
        )
    # A view log for a clip that no longer exists is meaningless, and the
    # column is NOT NULL, so these go rather than getting cleared.
    db.query(BroadcastView).filter(
        BroadcastView.clip_id == clip_id,
    ).delete(synchronize_session=False)

    # THE SWING GOES WITH THE CLIP. To the operator the trash icon on the
    # produced tile deletes a SWING -- that is what they are looking at
    # -- so leaving the swing in edit_metrics meant the edit wizard still
    # offered both swings after one had been deleted, and the next
    # produce would put the clip straight back. The swing list is the
    # wizard's source of truth, so the entry is removed outright, exactly
    # as the wizard's own per-swing ✕ does.
    #
    # Matched by the clip_id produce stamps on the swing. Positional
    # matching is the fallback for rows that predate that stamp, and
    # only when NO swing on the row carries one -- a mixed row means the
    # positions have already shifted and a guess would delete the wrong
    # swing's work.
    swing_removed = False
    if clip.long_upload_id:
        _up = db.get(LongVideoUpload, clip.long_upload_id)
        _em = dict(_up.edit_metrics or {}) if _up else {}
        _swings = _em.get("swings")
        if isinstance(_swings, list) and _swings:
            _keep = None
            if any(
                isinstance(s, dict) and s.get("clip_id") == clip_id
                for s in _swings
            ):
                _keep = [
                    s for s in _swings
                    if not (isinstance(s, dict) and s.get("clip_id") == clip_id)
                ]
            elif not any(
                isinstance(s, dict) and s.get("clip_id") is not None
                for s in _swings
            ):
                _order = (
                    db.query(VideoClip.id)
                    .filter(VideoClip.long_upload_id == clip.long_upload_id)
                    .order_by(VideoClip.captured_at.asc(), VideoClip.id.desc())
                    .all()
                )
                _pos = [r[0] for r in _order].index(clip_id) \
                    if clip_id in [r[0] for r in _order] else None
                if _pos is not None and _pos < len(_swings):
                    _keep = [s for i, s in enumerate(_swings) if i != _pos]
            if _keep is not None and len(_keep) != len(_swings):
                _em["swings"] = _keep
                # A fresh dict, so SQLAlchemy sees the assignment as
                # dirty without needing an explicit flag_modified.
                _up.edit_metrics = _em
                swing_removed = True
        # The card's "Produced · n/m clips" badge counts these, and a
        # stale count is how a row with one clip left reads as two.
        if _up is not None:
            _left = (
                db.query(VideoClip)
                .filter(
                    VideoClip.long_upload_id == clip.long_upload_id,
                    VideoClip.id != clip_id,
                )
                .count()
            )
            _up.last_n_succeeded = _left
            if (_up.last_n_segments or 0) > _left:
                _up.last_n_segments = _left

    db.add(
        AuditLog(
            actor="admin",
            action="delete_clip",
            target=f"clip:{clip.id}",
            detail=(
                f"deleted {len(deleted_files)} files, freed {freed} bytes"
                + (", removed its swing" if swing_removed else "")
            ),
        )
    )
    db.delete(clip)
    db.commit()
    return {
        "deleted": True,
        "clip_id": clip_id,
        "freed_bytes": freed,
        "files_unlinked": deleted_files,
        "swing_removed": swing_removed,
    }


def _commit_retry(db, apply_fn, what: str, retries: int = 2) -> bool:
    """Apply `apply_fn(db)` and commit, retrying on a dead connection.

    Long-running workers (produce spends minutes on tracer/AI work
    between DB touches) can find their connection killed by Postgres's
    idle-in-transaction timeout — the commit then dies with
    'SSL connection has been closed unexpectedly'. rollback() discards
    the dead connection; pool_pre_ping hands the retry a live one, and
    apply_fn re-fetches + re-applies so nothing depends on the dead
    session state. Returns True on success."""
    from sqlalchemy.exc import InterfaceError, OperationalError

    for attempt in range(retries + 1):
        try:
            apply_fn(db)
            db.commit()
            return True
        except (OperationalError, InterfaceError) as exc:
            try:
                db.rollback()
            except Exception:  # noqa: BLE001
                pass
            if attempt >= retries:
                log.error("%s: commit failed after retries: %s", what, exc)
                return False
            log.warning(
                "%s: dead DB connection (%s) — retrying on a fresh one",
                what, exc,
            )
            time.sleep(1.0 + attempt)
    return False


def _probe_golfer_x_frac(clip_path, sample_times=(0.3, 0.9, 1.5)):
    """Find the golfer's horizontal position (0..1) in the clip's first
    ~1.5s by running pose on a few frames — ground truth for where the
    vertical pan should OPEN, instead of inferring it from ball data
    (which framed empty background when the track started mid-flight).
    Returns the median hip x, or None (no mediapipe / nobody found)."""
    try:
        import cv2

        import mediapipe as mp  # type: ignore

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
        frames = []
        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
        cap.release()
        if not frames:
            return None

        def _body_x(res):
            """(normalized body-center x, visibility) or None. Body
            center = shoulders + hips averaged — centers the golfer
            visually (hips alone sat toward the rear of a side-on
            stance)."""
            lms = getattr(res, "pose_landmarks", None)
            if lms is None:
                return None
            _pts = [lms.landmark[i] for i in (11, 12, 23, 24)]
            vis = sum(float(q.visibility) for q in _pts) / len(_pts)
            hx = sum(float(q.x) for q in _pts) / len(_pts)
            if not (0.0 <= hx <= 1.0):
                return None
            return hx, vis

        xs: list[float] = []
        with mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.4,
        ) as pose:
            for frame in frames:
                r = _body_x(
                    pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                )
                if r is not None and r[1] >= 0.5:
                    xs.append(r[0])
            if not xs:
                # CROP-ZOOM SWEEP: a course-distance golfer is too
                # small for full-frame pose (same failure the swing
                # detector had). Slide a zoomed window across the
                # middle frame — bottom band first, golfers stand on
                # the ground — and keep the most confident hit.
                frame = frames[len(frames) // 2]
                h, w = frame.shape[:2]
                tw, th = max(64, w // 3), max(64, int(h * 0.6))
                best = None
                for ty0 in (h - th, 0):
                    for k in range(5):
                        tx0 = min(w - tw, k * tw // 2)
                        crop = frame[ty0:ty0 + th, tx0:tx0 + tw]
                        crop = cv2.resize(
                            crop, (tw * 2, th * 2),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        r = _body_x(
                            pose.process(
                                cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                            ),
                        )
                        if r is not None and r[1] >= 0.5:
                            gx = (tx0 + r[0] * tw) / float(w)
                            if best is None or r[1] > best[1]:
                                best = (gx, r[1])
                    if best is not None and best[1] >= 0.75:
                        break
                if best is not None:
                    xs.append(best[0])
        if not xs:
            return None
        xs.sort()
        return xs[len(xs) // 2]
    except Exception:  # noqa: BLE001
        return None


def _vertical_pan_path(
    track, rest_xy, fps, frame_w, cut_dur=None, golfer_x=None,
    impact_t=None,
):
    """Waypoints (time_sec, x_fraction) for the vertical follow-pan:
    hold on the golfer through address/swing, glide along the tracked
    ball flight, then return to center — at the tee->green cut when we
    know it (scene change), otherwise ~0.8s after the flight ends.
    `track` is CLIP-relative {frame, x} records; coords native px."""
    pts = sorted(
        (
            max(0.0, float(r["frame"]) / fps),
            min(1.0, max(0.0, float(r["x"]) / frame_w)),
        )
        for r in track
    )
    # CONTACT GATE: nothing moves the camera before the ball is
    # struck. Track entries earlier than impact are pre-swing noise
    # (waggle/club detections) — they were dragging the pan off the
    # golfer during the backswing. Drop them, and the hold below
    # extends to the impact moment.
    if impact_t is not None:
        pts = [(t, x) for t, x in pts if t >= float(impact_t) - 0.05]
    # OPENING FRAME = THE GOLFER. Rest-ball x is the best anchor (the
    # golfer stands at the ball); fall back to the median of the first
    # few track points — the launch happens at the golfer too, and the
    # median shrugs off a stray first detection.
    _launch_x = None
    if pts:
        _head = sorted(x for _, x in pts[:3])
        _launch_x = _head[len(_head) // 2]
    if golfer_x is not None:
        # Pose found the golfer in the actual opening frames — the
        # authoritative answer, no cross-checks needed.
        x0 = min(1.0, max(0.0, float(golfer_x)))
    elif rest_xy and len(rest_xy) == 2:
        x0 = min(1.0, max(0.0, float(rest_xy[0]) / frame_w))
        # A rest anchor that wildly disagrees with where the flight
        # starts is bad data (wrong scale / phantom) — the opener
        # would frame empty grass. Trust the launch instead.
        if _launch_x is not None and abs(x0 - _launch_x) > 0.35:
            x0 = _launch_x
    elif _launch_x is not None:
        x0 = _launch_x
    else:
        x0 = 0.5
    path = [(0.0, x0)]
    if pts:
        # HOLD on the golfer until contact: the pan's first movement
        # is the ball leaving. From there the interpolated targets ARE
        # the tracer's leading tip at each moment, so centering on the
        # targets keeps the tip centered while the ball is in the air;
        # after the last point the tip's END position stays centered.
        _hold_until = pts[0][0] - 0.05
        if impact_t is not None:
            _hold_until = max(_hold_until, float(impact_t) - 0.05)
        path.append((max(0.0, _hold_until), x0))
        path.extend(pts)
        t_last, x_last = pts[-1]
        if cut_dur is not None and cut_dur > t_last:
            # HOLD on the flight's end (the tracer lives there) until
            # the tee->green scene cut, then snap to center for the
            # landing view. Drifting home early dragged the tracer out
            # of frame while it was still on screen.
            path.append((max(t_last + 0.01, cut_dur - 0.01), x_last))
            path.append((float(cut_dur), 0.5))
        # No cut known: hold x_last to the end of the clip (implicit —
        # interpolation extends the final waypoint).
    return path


def _auto_delete_upload(upload_id: int, reason: str) -> None:
    """Delete a non-golf upload (row + files + linked camera event) with
    an audit entry. Owns its session; never raises."""
    try:
        db = SessionLocal()
        try:
            # Snapshot the source filenames BEFORE deleting the row so the
            # audit trail is enough to recover the clip from object storage
            # (scripts/recover_deleted_upload.py) if the screen was wrong.
            _row = db.get(LongVideoUpload, upload_id)
            _files = ""
            if _row is not None:
                _files = f" tee={_row.tee_filename}"
                if _row.green_filename:
                    _files += f" green={_row.green_filename}"
                _files += (
                    f" course={_row.course_id}"
                    f" captured_at={_row.base_captured_at.isoformat()}"
                )
            delete_long_upload(upload_id, db)
            db.add(
                AuditLog(
                    actor="system",
                    action="auto_delete_non_golf",
                    target=f"long_upload:{upload_id}",
                    detail=reason + _files,
                )
            )
            db.commit()
        finally:
            db.close()
        log.info(
            "auto-delete: removed non-golf upload %s (%s)",
            upload_id, reason,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "auto-delete failed for upload %s: %s", upload_id, exc,
        )


def _screen_non_golf_upload(upload_id: int, tee_name: str | None) -> bool:
    """Auto-screen an uploaded clip with the pose swing detector; when
    it contains NO golf swing, delete the upload (row + files + linked
    camera event) and return True. Runs on the upload's background
    thread before any processing spends time or API money on garbage
    (kitchen walk-bys, pets, empty frames).

    Fail-safe by construction: classification errors or a missing file
    KEEP the upload. Disable per-deployment with
    AUTO_DELETE_NON_GOLF=0."""
    if not settings.auto_delete_non_golf or not tee_name:
        return False
    try:
        from ..services import golf_scene

        path = CLIPS_DIR / tee_name
        if not path.exists():
            return False
        verdict = golf_scene.classify_clip(path)
        if verdict.get("is_golf", True):
            return False
        reason = verdict.get("reason") or "no golf swing detected"
        _auto_delete_upload(upload_id, reason)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "auto-screen failed for upload %s (%s) — keeping it",
            upload_id, exc,
        )
        return False


def _screen_then_run(upload_id, tee_name, target, kwargs):
    """Background-thread wrapper: non-golf screen first, then the real
    processing job (skipped entirely when the upload was deleted)."""
    if _screen_non_golf_upload(upload_id, tee_name):
        return
    target(**kwargs)


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

    Every upload auto-produces, whatever is in it. Debug3 finds however
    many swings the video contains and emits a clip per flight, so there
    is nothing for the operator to declare.

    `swing_count` is still ACCEPTED and still stored, because the Edit
    wizard reads it to pick its shape — flat `edit_metrics` for 'single',
    a per-swing `edit_metrics.swings[]` array for 'multiple'. It is no
    longer sent by the UI and no longer affects whether we produce; a
    stale cached frontend posting 'single' gets auto-produce like
    everything else. See the coercion below for why it is forced.

    Returns immediately so the upload UI isn't blocked on the
    multi-minute processing phase.
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
    src_ext = (
        (video.filename or "").rsplit(".", 1)[-1].lower()
        if "." in (video.filename or "")
        else "mp4"
    )
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
        g_ext = (
            (video_green.filename or "").rsplit(".", 1)[-1].lower()
            if "." in (video_green.filename or "")
            else "mp4"
        )
        if g_ext not in ("mp4", "mov", "webm", "m4v"):
            g_ext = "mp4"
        green_src_name = f"long-{course_id}-green-{secrets.token_hex(6)}.{g_ext}"
        (CLIPS_DIR / green_src_name).write_bytes(green_data)
        green_original_filename = video_green.filename or None

    # Forced to 'multiple', and NOT just because the UI stopped asking.
    # The Edit wizard picks its data shape from this field: 'single'
    # reads a flat edit_metrics, 'multiple' reads edit_metrics.swings[].
    # Debug3 — which now produces every upload — writes swings[] via
    # _d3_save_swing whatever the count. So a row stored as 'single'
    # would produce fine and then open an Edit wizard reading a shape
    # nothing ever wrote, showing no swing at all. Storing 'multiple'
    # keeps the produced data and the editor that reads it in agreement.
    _requested = (swing_count or "multiple").strip().lower()
    if _requested == "single":
        log.info(
            "quick-upload: swing_count='single' from a stale client — "
            "producing as 'multiple' (the only path there is now)",
        )
    swing_count_norm = "multiple"

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

    db.add(
        AuditLog(
            actor="admin",
            action="quick_upload_videos",
            target=f"long_upload:{upload_row.id}",
            detail=(
                f"course={course_id} tee={src_name} swing_count={swing_count_norm}"
                + (f" green={green_src_name}" if green_src_name else "")
            ),
        )
    )
    db.commit()

    log.info(
        "quick-upload: upload=%s course=%s tee=%s green=%s swing_count=%s",
        upload_row.id,
        course_id,
        src_name,
        green_src_name,
        swing_count_norm,
    )

    # EVERY upload auto-produces. There is no longer a one-swing /
    # multi-swing choice, because the distinction stopped being real:
    # Debug3 detects however many swings are in the video and emits a
    # clip per flight, so one swing in gives one clip out on the same
    # path. Asking the operator to declare it up front only created a
    # way to get it wrong.
    #
    # No pre-screen here: produce's own pose pass IS the non-golf
    # screen (zero swings detected -> the job auto-deletes), so a
    # separate mediapipe scan would just duplicate the work.
    enqueue_produce_job(upload_id=upload_row.id, hole_number=1)
    message = (
        "Upload received — producing in the background. Clips appear on "
        "Production as each swing finishes."
    )

    return {
        "upload_id": upload_row.id,
        "processing_status": upload_row.processing_status,
        "dual_camera": dual_camera,
        "swing_count": swing_count_norm,
        "auto_processing": True,
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
    audio_min_peak_ratio: float = Form(3.0),
    motion_ratio: float = Form(2.0),
    combined_pair_window_sec: float = Form(3.0),
    tee_green_delta_sec: float = Form(0.0),
    motion_only: bool = Form(False),
    db: Session = Depends(get_db),
):
    """Cut a long video into multiple per-swing clips and run each through
    the standard match + deliver pipeline.

    motion_only: when true, detect swings with the SAME vision-only
    detector the live camera events use (motion bursts + ball-departure
    filter), instead of the audio-first combined detector. Use this to
    faithfully test camera-captured clips (which have no 'crack' audio)
    that were mirrored in via this endpoint.

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

    src_ext = (
        (video.filename or "").rsplit(".", 1)[-1].lower()
        if "." in (video.filename or "")
        else "mp4"
    )
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
        g_ext = (
            (video_green.filename or "").rsplit(".", 1)[-1].lower()
            if "." in (video_green.filename or "")
            else "mp4"
        )
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
        green_original_filename=(
            video_green.filename if video_green is not None else None
        ),
        processing_status="pending",
    )
    db.add(upload_row)
    db.commit()
    db.refresh(upload_row)

    upload_id = upload_row.id

    # Generate the browser-friendly preview + poster thumbnail for the raw
    # source(s) in the background (same as the camera-upload path does), so
    # the production card shows a preview instead of "No preview". Best-
    # effort; failures just leave the card preview-less.
    raw_sources = [src_path] + ([green_src_path] if green_src_path else [])

    def _preview_raw_sources(paths: list[Path]) -> None:
        for p in paths:
            try:
                transcode_for_web(p)
                extract_thumbnail(p)
            except Exception as exc:  # pragma: no cover
                log.warning("long-upload preview gen failed for %s: %s", p.name, exc)

    threading.Thread(
        target=_preview_raw_sources, args=(raw_sources,),
        daemon=True, name=f"long-upload-preview-{upload_id}",
    ).start()

    # AUTO-PRODUCE runs THE produce path — the same one Debug3,
    # Re-Produce and the cameras run — so a clip is the same clip
    # however it was started.
    #
    # The exception is an operator who cut the swings by hand: Debug3
    # always auto-detects from pose and has nowhere to put a supplied
    # segment list, so a manual cut still goes through the old
    # per-segment pipeline. That is the only remaining caller of it, and
    # it is a deliberate one rather than a path nobody migrated.
    if seg_list:
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
                "tee_green_delta_sec": float(tee_green_delta_sec),
                # motion_only mirrors the live camera detector; when set,
                # all swings are on one hole (single_hole) as with
                # camera events.
                "motion_only": bool(motion_only),
                "single_hole": bool(motion_only),
            },
            daemon=True,
            name=f"long-upload-manual-{upload_id}",
        ).start()
    else:
        # An operator-supplied tee→green offset is real information the
        # cameras couldn't provide — persist it where the produce path's
        # sync lookup will find it.
        if tee_green_delta_sec:
            try:
                _r = db.get(LongVideoUpload, upload_id)
                if _r is not None:
                    _em = dict(_r.edit_metrics or {})
                    _em["tee_green_delta_sec"] = round(
                        float(tee_green_delta_sec), 4,
                    )
                    _r.edit_metrics = _em
                    db.commit()
            except Exception as exc:  # noqa: BLE001
                db.rollback()
                log.warning("could not persist green delta: %s", exc)
        enqueue_produce_job(
            upload_id=upload_id, hole_number=int(starting_hole or 1),
        )

    return {
        "upload_id": upload_id,
        "processing_status": "pending",
        "dual_camera": dual_camera,
        "queued_segments": len(seg_list) if seg_list else None,
        "auto_detect_swings": bool(auto_detect_swings and not seg_list),
    }


@router.get("/long-uploads")
def list_long_uploads(
    limit: int = 100,
    offset: int = 0,
    course: str = "",
    sort: str = "created",
    order: str = "desc",
    db: Session = Depends(get_db),
):
    """List previously-uploaded long videos so the operator can re-edit /
    reprocess them without re-uploading.

    Query params:
      course  case-insensitive substring match on the course name
              (blank = all courses).
      sort    "created" (upload date, default) or "course" (course name).
      order   "desc" (default) or "asc". For a course sort, "asc" is A→Z.
    """
    q = db.query(LongVideoUpload)

    # Search by course name — keep rows whose course name contains the
    # query (case-insensitive). Blank keeps everything.
    course_q = (course or "").strip()
    if course_q:
        q = q.join(Course, Course.id == LongVideoUpload.course_id).filter(
            Course.name.ilike(f"%{course_q}%")
        )

    descending = (order or "desc").lower() != "asc"
    if sort == "course":
        # Order alphabetically by course name; within a course, newest
        # upload first. Needs Course joined — reuse the search join when
        # present, else outer-join so course-less rows still appear.
        if not course_q:
            q = q.outerjoin(Course, Course.id == LongVideoUpload.course_id)
        name_col = Course.name
        q = q.order_by(
            name_col.desc() if descending else name_col.asc(),
            LongVideoUpload.created_at.desc(),
        )
    else:
        created_col = LongVideoUpload.created_at
        q = q.order_by(created_col.desc() if descending else created_col.asc())

    rows = (
        q.offset(max(0, offset))
        .limit(max(1, min(500, limit)))
        .all()
    )
    course_ids = {r.course_id for r in rows}
    courses = (
        {c.id: c for c in db.query(Course).filter(Course.id.in_(course_ids)).all()}
        if course_ids
        else {}
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

    def _meta(path: Path | None, exists: bool, in_bucket: bool = False) -> dict:
        """Bundle probe + thumbnail lookup for one source video. Skipping the
        probe entirely when the file is missing keeps list responses fast."""
        if not (path and exists):
            thumb_url = None
            if in_bucket and path:
                # Source video is in the bucket; check if its thumbnail JPG is
                # there too so the production card can show a preview image
                # even before the source is downloaded to local disk.
                thumb_name = path.stem + ".jpg"
                if storage.exists(thumb_name):
                    thumb_url = (
                        f"{settings.app_base_url}/uploads/clips/{thumb_name}"
                    )
                # Kick off background download of the source so the next page
                # load can probe and show full metadata.
                _rehydrate_background(CLIPS_DIR, path.name)
            return {
                "size_mb": None,
                "duration_sec": None,
                "fps": None,
                "nb_frames": None,
                "thumbnail_url": thumb_url,
                "width": None,
                "height": None,
                "quality_label": None,
            }
        size = path.stat().st_size
        info = probe_video_info(path)
        thumb = path.with_suffix(".jpg")
        if thumb.exists():
            thumb_url = f"{settings.app_base_url}/uploads/clips/{thumb.name}"
        elif storage.exists(thumb.name):
            # Thumbnail is in the bucket (generated before this redeploy)
            # but hasn't been rehydrated to local disk yet.
            thumb_url = f"{settings.app_base_url}/uploads/clips/{thumb.name}"
        else:
            thumb_url = None
        return {
            "size_mb": round(size / 1024 / 1024, 1) if size else None,
            "duration_sec": round(info["duration"], 1)
            if info.get("duration")
            else None,
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
            # captured_at alone is not a total order: it is derived from
            # the cut start, so two clips cut at the same offset tie. The
            # id tiebreak just makes the order deterministic — without it
            # produced_clips[0], which the card renders, is whichever row
            # the database felt like returning first.
            .order_by(VideoClip.captured_at.asc(), VideoClip.id.desc())
            .all()
        )
        for clip in produced_rows:
            produced_by_upload.setdefault(clip.long_upload_id, []).append(clip)

    # Bulk-load camera-source metadata for any rows whose
    # camera_event_id is set (Pi-sourced uploads). Surfaced as a "From
    # Camera #N · hole X" badge on the production card.
    cam_event_ids = {r.camera_event_id for r in rows if r.camera_event_id}
    cam_events_by_id: dict[int, CameraEvent] = (
        {
            ev.id: ev
            for ev in db.query(CameraEvent)
            .filter(CameraEvent.id.in_(cam_event_ids))
            .all()
        }
        if cam_event_ids
        else {}
    )
    tee_cam_ids = {
        ev.tee_camera_id
        for ev in cam_events_by_id.values()
        if ev.tee_camera_id
    }
    tee_cams_by_id: dict[int, Camera] = (
        {
            c.id: c
            for c in db.query(Camera).filter(Camera.id.in_(tee_cam_ids)).all()
        }
        if tee_cam_ids
        else {}
    )

    def _produced(upload_id: int) -> list[dict]:
        clips = produced_by_upload.get(upload_id, [])
        out = []
        for c in clips:
            # Prefer the tracer-rendered URL when we have one — it's the
            # final on-air output. Fallback to source_url so single-cam
            # clips still play before the tracer ships.
            play_url = c.tracer_url or c.source_url
            out.append(
                {
                    "id": c.id,
                    "hole_number": c.hole_number,
                    "captured_at": c.captured_at.isoformat() if c.captured_at else None,
                    "video_url": play_url,
                    "thumbnail_url": c.thumbnail_url,
                    "ball_in_cup": bool(c.ball_in_cup),
                    "is_highlight": bool(c.is_highlight),
                }
            )
        return out

    def _live_produce_stage(upload_id: int) -> dict:
        """Which Debug3 stage this upload is on, if it is producing right
        now. Reported only while running — a stale stage left on a
        finished row reads as stuck, which is the thing this fixes."""
        st = _debugx_get("produce", upload_id)
        if not st.get("running"):
            return {"produce_stage": None, "produce_done": None,
                    "produce_total": None}
        return {
            "produce_stage": st.get("stage"),
            "produce_done": st.get("done"),
            "produce_total": st.get("total"),
        }

    # Produce is serialised, so a card can sit at "pending" simply because
    # it is waiting its turn. Say which, rather than looking stuck. Queue
    # order is by upload time, so we can name the exact position.
    _q = produce_queue_status()
    _q_running = _q["running_upload_id"]
    _q_waiting = _q["queued_upload_ids"]
    _q_pos: dict[int, int] = {}
    if _q_waiting:
        try:
            _q_rows = (
                db.query(LongVideoUpload.id, LongVideoUpload.created_at)
                .filter(LongVideoUpload.id.in_(_q_waiting))
                .all()
            )
            _by_time = sorted(
                _q_rows, key=lambda t: (t[1] or _utcnow_naive(), t[0]),
            )
            _q_pos = {int(t[0]): i + 1 for i, t in enumerate(_by_time)}
        except Exception as exc:  # noqa: BLE001
            log.warning("could not order produce queue for display: %s", exc)
            _q_pos = {int(u): 0 for u in _q_waiting}

    out = []
    for r in rows:
        tee_path = CLIPS_DIR / r.tee_filename if r.tee_filename else None
        green_path = CLIPS_DIR / r.green_filename if r.green_filename else None
        tee_exists = bool(tee_path and tee_path.exists())
        green_exists = bool(green_path and green_path.exists())
        # If not on local disk, check object storage — files survive in the
        # bucket across redeploys even when the local CLIPS_DIR is fresh.
        # We only need the boolean here (metadata probing requires a local file);
        # ensure_local is called later by the produce job itself.
        tee_in_bucket = (
            bool(r.tee_filename and storage.exists(r.tee_filename))
            if not tee_exists else False
        )
        green_in_bucket = (
            bool(r.green_filename and storage.exists(r.green_filename))
            if not green_exists else False
        )
        tee_meta = _meta(tee_path, tee_exists, in_bucket=tee_in_bucket)
        green_meta = _meta(green_path, green_exists, in_bucket=green_in_bucket)
        course = courses.get(r.course_id)
        produced = _produced(r.id)
        cam_event = cam_events_by_id.get(r.camera_event_id) if r.camera_event_id else None
        tee_cam = (
            tee_cams_by_id.get(cam_event.tee_camera_id)
            if cam_event and cam_event.tee_camera_id
            else None
        )
        source = (
            {
                "kind": "camera",
                "camera_event_id": cam_event.id,
                "camera_id": tee_cam.id if tee_cam else cam_event.tee_camera_id,
                "camera_name": tee_cam.name if tee_cam else None,
                "hole_number": cam_event.hole_number,
                "triggered_at": (
                    cam_event.triggered_at.isoformat()
                    if cam_event.triggered_at
                    else None
                ),
            }
            if cam_event
            else {"kind": "upload"}
        )
        out.append(
            {
                "id": r.id,
                "course_id": r.course_id,
                "course_name": course.name if course else None,
                "course_hole_yardages": (course.hole_yardages or {}) if course else {},
                "camera_type": r.camera_type,
                "base_captured_at": r.base_captured_at.isoformat()
                if r.base_captured_at
                else None,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                # First-frame wall-clock times from the source camera
                # event (null for manual uploads). Used by the wizard to
                # show the cut frame's real-world timestamp and by the
                # production page's clock overlay.
                "tee_recording_started_at": (
                    cam_event.tee_recording_started_at.isoformat()
                    if cam_event and cam_event.tee_recording_started_at
                    else None
                ),
                "green_recording_started_at": (
                    cam_event.green_recording_started_at.isoformat()
                    if cam_event and cam_event.green_recording_started_at
                    else None
                ),
                "swing_count": r.swing_count or "multiple",
                "tee_filename": r.tee_filename,
                "tee_original_filename": r.tee_original_filename,
                "tee_url": (
                    f"{settings.app_base_url}/uploads/clips/{r.tee_filename}"
                    if tee_exists
                    else None
                ),
                "tee_thumbnail_url": tee_meta["thumbnail_url"],
                "tee_size_mb": tee_meta["size_mb"],
                "tee_duration_sec": tee_meta["duration_sec"],
                "tee_fps": tee_meta["fps"],
                "tee_nb_frames": tee_meta["nb_frames"],
                "tee_width": tee_meta["width"],
                "tee_height": tee_meta["height"],
                "tee_quality_label": tee_meta["quality_label"],
                "tee_missing": (r.tee_filename is not None and not tee_exists and not tee_in_bucket),
                "green_filename": r.green_filename,
                "green_original_filename": r.green_original_filename,
                "green_url": (
                    f"{settings.app_base_url}/uploads/clips/{r.green_filename}"
                    if green_exists
                    else None
                ),
                "green_thumbnail_url": green_meta["thumbnail_url"],
                "green_size_mb": green_meta["size_mb"],
                "green_duration_sec": green_meta["duration_sec"],
                "green_fps": green_meta["fps"],
                "green_nb_frames": green_meta["nb_frames"],
                "green_width": green_meta["width"],
                "green_height": green_meta["height"],
                "green_quality_label": green_meta["quality_label"],
                "green_missing": (r.green_filename is not None and not green_exists and not green_in_bucket),
                "dual_camera": r.green_filename is not None,
                "produced_clips": produced,
                "edit_metrics": r.edit_metrics,
                "last_n_segments": r.last_n_segments,
                "last_n_succeeded": r.last_n_succeeded,
                "processing_status": r.processing_status,
                "processing_started_at": (
                    r.processing_started_at.isoformat()
                    if r.processing_started_at
                    else None
                ),
                "processing_completed_at": (
                    r.processing_completed_at.isoformat()
                    if r.processing_completed_at
                    else None
                ),
                "last_error": r.last_error,
                "source": source,
                "queue_state": (
                    "running" if r.id == _q_running
                    else ("queued" if r.id in _q_pos else None)
                ),
                "queue_position": _q_pos.get(r.id),
                "queue_depth": len(_q_waiting),
                # Live stage of an in-flight produce, so the greyed card
                # can name what it is doing.
                **_live_produce_stage(r.id),
            }
        )
    return out


@router.get("/camera-events")
def list_camera_events(
    limit: int = 100, offset: int = 0, db: Session = Depends(get_db),
):
    """List recent CameraEvents for the production queue. Each row
    bundles the raw tee/green clips the Pis uploaded plus the
    produced VideoClip (if any), so the operator can review what was
    captured and re-run or delete it without leaving the page.

    Same shape conventions as /long-uploads so the production page
    can render both kinds of rows with the same building blocks."""
    rows = (
        db.query(CameraEvent)
        .order_by(CameraEvent.triggered_at.desc())
        .offset(max(0, offset))
        .limit(max(1, min(500, limit)))
        .all()
    )
    if not rows:
        return []

    course_ids = {r.course_id for r in rows}
    courses = {
        c.id: c
        for c in db.query(Course).filter(Course.id.in_(course_ids)).all()
    }
    camera_ids = {r.tee_camera_id for r in rows} | {
        r.green_camera_id for r in rows if r.green_camera_id
    }
    cameras_by_id = {
        c.id: c for c in db.query(Camera).filter(Camera.id.in_(camera_ids)).all()
    }
    clip_ids = {r.produced_clip_id for r in rows if r.produced_clip_id}
    clips_by_id = (
        {
            c.id: c
            for c in db.query(VideoClip).filter(VideoClip.id.in_(clip_ids)).all()
        }
        if clip_ids
        else {}
    )

    def _meta(fname: str | None) -> dict:
        """Probe one raw clip on disk for the production card. None
        fname means the Pi never uploaded; missing-on-disk means it
        uploaded once but the file was removed since."""
        if not fname:
            return {
                "url": None,
                "thumbnail_url": None,
                "size_mb": None,
                "duration_sec": None,
                "fps": None,
                "nb_frames": None,
                "width": None,
                "height": None,
                "missing": False,
            }
        path = CLIPS_DIR / fname
        if not path.exists():
            # Before declaring the file missing, check object storage — the
            # file may have survived a redeploy in the bucket even though the
            # local CLIPS_DIR was reset.
            if storage.exists(fname):
                # Also check for a pre-existing thumbnail JPG in the bucket.
                thumb_name = Path(fname).stem + ".jpg"
                thumb_url = (
                    f"{settings.app_base_url}/uploads/clips/{thumb_name}"
                    if storage.exists(thumb_name) else None
                )
                # Rehydrate source to local disk in the background so the next
                # page load can probe and show full metadata.
                _rehydrate_background(CLIPS_DIR, fname)
                return {
                    "url": f"{settings.app_base_url}/uploads/clips/{fname}",
                    "thumbnail_url": thumb_url,
                    "size_mb": None,
                    "duration_sec": None,
                    "fps": None,
                    "nb_frames": None,
                    "width": None,
                    "height": None,
                    "missing": False,
                }
            return {
                "url": None,
                "thumbnail_url": None,
                "size_mb": None,
                "duration_sec": None,
                "fps": None,
                "nb_frames": None,
                "width": None,
                "height": None,
                "missing": True,
            }
        size = path.stat().st_size
        info = probe_video_info(path) or {}
        thumb = path.with_suffix(".jpg")
        if thumb.exists():
            thumb_url = f"{settings.app_base_url}/uploads/clips/{thumb.name}"
        elif storage.exists(thumb.name):
            thumb_url = f"{settings.app_base_url}/uploads/clips/{thumb.name}"
        else:
            thumb_url = None
        return {
            "url": f"{settings.app_base_url}/uploads/clips/{fname}",
            "thumbnail_url": thumb_url,
            "size_mb": round(size / 1024 / 1024, 1) if size else None,
            "duration_sec": (
                round(info["duration"], 1) if info.get("duration") else None
            ),
            "fps": round(info["fps"], 2) if info.get("fps") else None,
            "nb_frames": info.get("nb_frames"),
            "width": info.get("width"),
            "height": info.get("height"),
            "missing": False,
        }

    out = []
    for r in rows:
        tee_meta = _meta(r.tee_clip_filename)
        green_meta = _meta(r.green_clip_filename)
        course = courses.get(r.course_id)
        tee_cam = cameras_by_id.get(r.tee_camera_id)
        green_cam = (
            cameras_by_id.get(r.green_camera_id) if r.green_camera_id else None
        )
        clip = clips_by_id.get(r.produced_clip_id) if r.produced_clip_id else None
        produced = None
        if clip is not None:
            produced = {
                "id": clip.id,
                "hole_number": clip.hole_number,
                "captured_at": (
                    clip.captured_at.isoformat() if clip.captured_at else None
                ),
                "video_url": clip.tracer_url or clip.source_url,
                "thumbnail_url": clip.thumbnail_url,
                "ball_in_cup": bool(clip.ball_in_cup),
                "is_highlight": bool(clip.is_highlight),
            }

        out.append(
            {
                "id": r.id,
                "session_id": r.session_id,
                "course_id": r.course_id,
                "course_name": course.name if course else None,
                "hole_number": r.hole_number,
                "triggered_at": (
                    r.triggered_at.isoformat() if r.triggered_at else None
                ),
                "status": r.status,
                "last_error": r.last_error,
                "tee_camera_id": r.tee_camera_id,
                "tee_camera_name": tee_cam.name if tee_cam else None,
                "green_camera_id": r.green_camera_id,
                "green_camera_name": green_cam.name if green_cam else None,
                "dual_camera": r.green_camera_id is not None,
                "tee_clip_filename": r.tee_clip_filename,
                "tee_url": tee_meta["url"],
                "tee_thumbnail_url": tee_meta["thumbnail_url"],
                "tee_size_mb": tee_meta["size_mb"],
                "tee_duration_sec": tee_meta["duration_sec"],
                "tee_fps": tee_meta["fps"],
                "tee_nb_frames": tee_meta["nb_frames"],
                "tee_width": tee_meta["width"],
                "tee_height": tee_meta["height"],
                "tee_missing": tee_meta["missing"],
                "green_clip_filename": r.green_clip_filename,
                "green_url": green_meta["url"],
                "green_thumbnail_url": green_meta["thumbnail_url"],
                "green_size_mb": green_meta["size_mb"],
                "green_duration_sec": green_meta["duration_sec"],
                "green_fps": green_meta["fps"],
                "green_nb_frames": green_meta["nb_frames"],
                "green_width": green_meta["width"],
                "green_height": green_meta["height"],
                "green_missing": green_meta["missing"],
                "produced_clip": produced,
            }
        )
    return out


@router.post("/camera-events/{event_id}/reprocess")
def reprocess_camera_event(event_id: int, db: Session = Depends(get_db)):
    """Re-run the production pipeline for a previously-uploaded
    camera event. Useful when an upstream change (tracer tweaks,
    overlay updates) means we want to regenerate the produced clip
    from the same raw inputs."""
    # Imported here to dodge the cameras.py ↔ admin.py circular.
    from .cameras import _process_camera_event_job

    event = db.get(CameraEvent, event_id)
    if event is None:
        raise HTTPException(404, "camera event not found")
    if not event.tee_clip_filename:
        raise HTTPException(409, "no tee clip on file to re-process")
    # Rehydrate from object storage if the file landed in the bucket but
    # the local CLIPS_DIR was reset by a redeploy.
    storage.ensure_local(CLIPS_DIR, event.tee_clip_filename)
    if event.green_clip_filename:
        storage.ensure_local(CLIPS_DIR, event.green_clip_filename)
    tee_path = CLIPS_DIR / event.tee_clip_filename
    if not tee_path.exists():
        raise HTTPException(
            404, f"tee clip missing on disk and in storage: {event.tee_clip_filename}",
        )

    # Reset status so the UI shows the run is restarting and any
    # stale last_error gets cleared.
    event.status = "paired_uploaded" if event.green_clip_filename else "tee_uploaded"
    event.last_error = None
    db.commit()

    threading.Thread(
        target=_process_camera_event_job,
        args=(event.id,),
        daemon=True,
        name=f"camera-event-reprocess-{event.id}",
    ).start()

    return {"event_id": event.id, "status": event.status, "queued": True}


@router.delete("/camera-events/{event_id}")
def delete_camera_event(event_id: int, db: Session = Depends(get_db)):
    """Permanently remove a camera event: deletes the raw tee/green
    MP4s from disk, the linked LongVideoUpload + every VideoClip
    produced from it, and the CameraEvent row itself. Use when a
    swing was a misfire (people walking past the tee box, false
    trigger, etc.) and you don't want it cluttering the queue."""
    event = db.get(CameraEvent, event_id)
    if event is None:
        raise HTTPException(404, "camera event not found")

    # TOMBSTONE FIRST. The Pi may still be holding this event's clip --
    # on a bad link a clip can sit spooled for hours -- and the
    # uploader's lost-trigger recovery re-registers any session the
    # server has no row for. Without this marker a deletion is undone
    # the moment the backlog drains; sixteen deleted events came back
    # as 502-518 that way. Recorded before anything else so a failure
    # later in the delete cannot leave the session resurrectable.
    if event.session_id and not db.query(DeletedCameraSession).filter(
        DeletedCameraSession.session_id == event.session_id,
    ).first():
        db.add(DeletedCameraSession(
            session_id=event.session_id, event_id=event.id,
        ))
        db.commit()

    for fname in (event.tee_clip_filename, event.green_clip_filename):
        if not fname:
            continue
        path = CLIPS_DIR / fname
        try:
            if path.exists():
                path.unlink()
            thumb = path.with_suffix(".jpg")
            if thumb.exists():
                thumb.unlink()
            marker = path.with_suffix(path.suffix + ".h264-ok")
            if marker.exists():
                marker.unlink()
        except OSError as exc:
            log.warning("delete camera event %s: could not unlink %s (%s)",
                        event_id, fname, exc)

    # Cascade to the linked long-upload (1:1 with the event under the
    # new production path) and the per-swing VideoClips it produced.
    lvu = (
        db.query(LongVideoUpload)
        .filter(LongVideoUpload.camera_event_id == event.id)
        .first()
    )
    if lvu is not None:
        for clip in (
            db.query(VideoClip).filter(VideoClip.long_upload_id == lvu.id).all()
        ):
            db.delete(clip)
        db.delete(lvu)

    # produced_clip_id may still reference a clip that wasn't linked
    # through long_upload_id (legacy data before the unification);
    # clean that up too.
    if event.produced_clip_id:
        clip = db.get(VideoClip, event.produced_clip_id)
        if clip is not None:
            db.delete(clip)

    db.delete(event)
    db.commit()
    return {"ok": True, "event_id": event_id}


@router.post("/long-uploads/{upload_id}/reprocess")
def reprocess_long_upload(
    upload_id: int,
    segments: str = Form("[]"),
    auto_detect_swings: bool = Form(True),
    starting_hole: int = Form(1),
    ai_tracer_model: str | None = Form(None),
    audio_min_peak_ratio: float = Form(3.0),
    motion_ratio: float = Form(2.0),
    combined_pair_window_sec: float = Form(3.0),
    tee_green_delta_sec: float = Form(0.0),
    db: Session = Depends(get_db),
):
    """Re-produce a previously-uploaded long video without re-uploading.

    Runs THE pipeline — the same `_debug3_run` the Debug3 panel runs,
    with the diagnostic artefacts turned off. Returns immediately with
    pending status so the frontend can poll /long-uploads.

    The detection Form fields (`segments`, `starting_hole`,
    `ai_tracer_model`, `audio_min_peak_ratio`, `motion_ratio`,
    `combined_pair_window_sec`) are still ACCEPTED so existing callers
    keep working, but no longer used: swings come from the pose
    detector, and the flight from Debug3. `tee_green_delta_sec` is the
    exception — it is a real operator input, so it is persisted for the
    produce to use.
    """
    row = db.get(LongVideoUpload, upload_id)
    if not row:
        raise HTTPException(404, "long upload not found")
    src_path = _local_tee(row)
    if not src_path or not src_path.exists():
        raise HTTPException(
            404,
            f"tee source file missing on disk: {row.tee_filename}",
        )
    if row.green_filename:
        candidate = _local_green(row)
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
    # An operator-supplied tee→green offset is real information the
    # cameras couldn't provide — persist it so _d3_green_delta_sec picks
    # it up (it ranks below the cameras' own clocks, above assuming 0).
    if tee_green_delta_sec:
        _em = dict(row.edit_metrics or {})
        _em["tee_green_delta_sec"] = round(float(tee_green_delta_sec), 4)
        row.edit_metrics = _em
    db.commit()

    # RE-PRODUCE IS DEBUG3, MINUS THE PANEL. They were two
    # implementations of one process and drifted the moment they existed
    # side by side: different swing detectors (pose vs audio+motion),
    # different clip windows, 35s versus 208s, and 40 AI calls per swing
    # on this side whose answers Debug3 overwrote anyway. Same function
    # now; `debug_artifacts=False` skips the film-strips, the whole-video
    # resting-ball scan and the preview clip — diagnostics nobody is
    # looking at on a plain Re-Produce. Nothing that decides what ships
    # differs between the two.
    #
    # _debugx_start owns the thread, the DB session and the progress
    # state, so the existing Debug3 status poll reports this run too.
    _debugx_start(
        "debug3", row.id,
        functools.partial(_debug3_run, debug_artifacts=False),
    )

    return {
        "upload_id": row.id,
        "processing_status": "pending",
        "dual_camera": row.green_filename is not None,
        "engine": "debug3",
        "queued_segments": None,
        "auto_detect_swings": True,
    }


@router.post("/long-uploads/{upload_id}/detect-swings")
def detect_swings_for_upload(
    upload_id: int,
    db: Session = Depends(get_db),
):
    """Run combined audio + motion swing detection on the upload's
    tee video and return the segments. Used by the multi-swing
    Edit wizard to populate its per-swing list. Caches the segments
    onto edit_metrics.swings so subsequent re-opens skip the probe.

    Each returned swing has:
      idx:           int (0-based)
      start_frame:   int (~before impact)
      end_frame:     int (~after impact)
      address_frame: int (impact - 1.5s, clamped)
      impact_frame:  int (audio peak)
      fps:           float
    """
    row = db.get(LongVideoUpload, upload_id)
    if not row:
        raise HTTPException(404, "long upload not found")
    if not row.tee_filename:
        raise HTTPException(400, "upload has no tee video")
    src_path = _local_tee(row)
    if not src_path.exists():
        raise HTTPException(404, f"tee source file missing on disk: {row.tee_filename}")

    fps_val = probe_fps(src_path) or 30.0
    segments = detect_swings_combined(src_path, fps=fps_val)

    swings: list[dict] = []
    for i, seg in enumerate(segments):
        start_sec = float(seg.get("start_sec") or 0.0)
        end_sec = float(seg.get("end_sec") or start_sec)
        peak_sec = float(seg.get("peak_sec") or (start_sec + end_sec) / 2)
        start_frame = int(round(start_sec * fps_val))
        end_frame = int(round(end_sec * fps_val))
        impact_frame = int(round(peak_sec * fps_val))
        address_frame = max(start_frame, impact_frame - int(round(1.5 * fps_val)))
        swings.append(
            {
                "idx": i,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "address_frame": address_frame,
                "impact_frame": impact_frame,
                "fps": round(fps_val, 2) if fps_val else None,
            }
        )

    saved = dict(row.edit_metrics or {})
    # Merge — don't blow away per-swing edits the operator already
    # made. We only seed swings that don't already exist.
    existing = saved.get("swings") or []
    by_idx = {int(s.get("idx", -1)): s for s in existing if isinstance(s, dict)}
    for sw in swings:
        if sw["idx"] not in by_idx:
            by_idx[sw["idx"]] = sw
    merged = [by_idx[i] for i in sorted(by_idx)]
    saved["swings"] = merged
    row.edit_metrics = saved
    db.add(row)
    db.add(
        AuditLog(
            actor="admin",
            action="detect_swings",
            target=f"long_upload:{upload_id}",
            detail=f"n_swings={len(swings)} fps={fps_val:.2f}",
        )
    )
    db.commit()
    db.refresh(row)
    return {
        "upload_id": upload_id,
        "fps": round(fps_val, 2) if fps_val else None,
        "swings": merged,
    }


def _run_auto_detect_seed(upload_id: int) -> dict | None:
    """Background-friendly wrapper around the auto-detect pipeline.

    Opens its own DB session, runs detection, persists every result
    into edit_metrics so the Edit wizard can hydrate without making
    any API calls. Spawned in a thread from /clips/quick-upload for
    single-swing uploads, so by the time the operator clicks Edit
    the wizard already has handedness / address / impact / ball /
    ROI / target ready to display.

    Returns the same dict shape /auto-detect serialises, or None on
    error. Never raises.
    """
    from ..database import SessionLocal  # local — avoids router-load cycle

    sess: Session = SessionLocal()
    try:
        try:
            return auto_detect_long_upload(upload_id, db=sess)
        except HTTPException as exc:
            log.warning("background auto-detect %s skipped: %s", upload_id, exc.detail)
            return None
        except Exception:  # pragma: no cover
            log.exception("background auto-detect %s failed", upload_id)
            return None
    finally:
        sess.close()


@router.post("/long-uploads/{upload_id}/auto-detect")
def auto_detect_long_upload(upload_id: int, db: Session = Depends(get_db)):
    """Run the lightweight per-swing detection on this upload's tee
    video and PERSIST every result onto edit_metrics. Returns the
    same data shape so callers (Edit wizard, or the background
    upload-time spawner) can use either path.

    The cheap calls here (audio impact + one Claude handedness call)
    usually return in ~5–10 s. The per-frame ball-track Claude calls
    and the tracer-video render run only when the operator hits
    Produce on Step 3.
    """
    row = db.get(LongVideoUpload, upload_id)
    if not row:
        raise HTTPException(404, "long upload not found")
    if not row.tee_filename:
        raise HTTPException(400, "upload has no tee video")
    src_path = _local_tee(row)
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
                cv2.imwrite(
                    str(address_image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92]
                )
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
        return (
            f"{settings.app_base_url}/uploads/clips/{p.name}?v={int(p.stat().st_mtime)}"
        )

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
        ball_x_raw is not None
        and ball_y_raw is not None
        and sent_w
        and sent_h
        and frame_w
        and frame_h
    ):
        ball_x = int(round(float(ball_x_raw) * frame_w / sent_w))
        ball_y = int(round(float(ball_y_raw) * frame_h / sent_h))

    # --- Step 4: ball detection ROI ---
    # Square bbox centred on the ball-at-rest position, ~12% of frame
    # height on a side. Operator can resize/drag during the wizard.
    detection_area = None
    if ball_x is not None and ball_y is not None and frame_w and frame_h:
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

    # Persist every detection result onto edit_metrics so the Edit
    # wizard never has to call /auto-detect again — it just hydrates
    # from edit_metrics on open. Idempotent: re-running auto-detect
    # overwrites these fields (the operator's own subsequent
    # adjustments via the per-field Apply buttons land here too).
    address_image_url = _public_url(address_image_path)
    saved = dict(row.edit_metrics or {})
    saved.update(
        {
            "handedness": handedness or saved.get("handedness") or "right",
            "address_frame": address_frame_idx
            if address_frame_idx is not None
            else saved.get("address_frame", 0),
            "address_image_url": address_image_url or saved.get("address_image_url"),
            "impact_frame": int(impact_frame)
            if impact_frame is not None
            else saved.get("impact_frame", 0),
            "ball": (
                {"x": int(ball_x), "y": int(ball_y)}
                if ball_x is not None and ball_y is not None
                else saved.get("ball")
            ),
            "roi": detection_area if detection_area is not None else saved.get("roi"),
            "target": (
                {"x": int(target["x"]), "y": int(target["y"])}
                if target
                else saved.get("target")
            ),
            "frame_width": frame_w or saved.get("frame_width"),
            "frame_height": frame_h or saved.get("frame_height"),
        }
    )
    row.edit_metrics = saved
    db.add(row)
    db.add(
        AuditLog(
            actor="admin",
            action="auto_detect_long_upload",
            target=f"long_upload:{upload_id}",
            detail=(
                f"handedness={handedness} address_frame={address_frame_idx} "
                f"impact_frame={impact_frame} ball=({ball_x},{ball_y})"
            ),
        )
    )
    db.commit()

    return {
        "upload_id": upload_id,
        "fps": round(fps_val, 2) if fps_val else None,
        "tee_url": (
            f"{settings.app_base_url}/uploads/clips/{row.tee_filename}"
            if row.tee_filename
            else None
        ),
        "frame_width": frame_w,
        "frame_height": frame_h,
        "handedness": {
            "value": handedness,
            "confidence": handedness_info.get("confidence")
            if handedness_info
            else None,
            "notes": handedness_info.get("notes") if handedness_info else None,
        },
        "address": {
            "frame": address_frame_idx,
            "image_url": address_image_url,
        },
        "impact": {
            "frame": impact_frame,
            "method": audio_info.get("method"),
            "ratio": audio_info.get("ratio"),
        },
        "ball_at_rest": (
            {"x": int(ball_x), "y": int(ball_y)}
            if ball_x is not None and ball_y is not None
            else None
        ),
        "ball_detection_area": detection_area,
        "target": target,
    }


def _frame_wall_clock(db, row, which: str, frame_idx: int, fps: float):
    """Real-world instant of a frame, as an ISO string (or None).

    A frame number and an offset into the clip both describe a position
    in a file; neither tells the operator WHEN the shot happened, which
    is what matches a clip to a group on the tee sheet. The Pis stamp the
    wall-clock instant of their first frame, so this is that instant plus
    frame/fps.

    Per camera, because the two recordings do not start together: the
    green's own stamp when it has one, otherwise the tee's shifted by the
    measured delta. Falls back to base_captured_at (the trigger time),
    which is close but not frame-exact -- better than showing nothing.
    """
    if fps is None or fps <= 0:
        return None
    try:
        start = None
        ev = None
        if getattr(row, "camera_event_id", None):
            ev = db.query(CameraEvent).filter(
                CameraEvent.id == row.camera_event_id,
            ).first()
        if str(which).lower() == "green":
            start = getattr(ev, "green_recording_started_at", None)
            if start is None:
                _tee = getattr(ev, "tee_recording_started_at", None) or getattr(
                    row, "base_captured_at", None)
                if _tee is not None:
                    delta, _ = _d3_green_delta_sec(db, row)
                    start = _tee + timedelta(seconds=float(delta))
        else:
            start = getattr(ev, "tee_recording_started_at", None)
        if start is None:
            start = getattr(row, "base_captured_at", None)
        if start is None:
            return None
        return (start + timedelta(seconds=float(frame_idx) / fps)).isoformat()
    except Exception as exc:  # noqa: BLE001
        log.debug("wall clock for upload %s frame %s failed: %s",
                  getattr(row, "id", None), frame_idx, exc)
        return None


def _default_end_frame(db, row, green_path, impact_frame, green_total):
    """The green frame produce would stop on, from a tee impact frame.

    None when there is nothing to reckon from -- the caller then shows
    the clip's last frame, which is what it did before.
    """
    if impact_frame is None:
        return None
    try:
        tee_path = _local_tee(row)
        tee_fps = float(probe_fps(tee_path) or 0.0) if tee_path else 0.0
        green_fps = float(probe_fps(green_path) or 0.0)
        if tee_fps <= 0 or green_fps <= 0:
            return None
        delta, _ = _d3_green_delta_sec(db, row)
        t_impact_green = (float(impact_frame) / tee_fps) - float(delta)
        n = int(round((t_impact_green + D3_GREEN_SEC) * green_fps))
        if green_total:
            n = max(0, min(int(green_total) - 1, n))
        return max(0, n)
    except Exception as exc:  # noqa: BLE001
        log.debug("default end frame for %s failed: %s",
                  getattr(row, "id", None), exc)
        return None


@router.get("/uploads-in-flight")
def uploads_in_flight():
    """Clips a Pi is part-way through sending, and how far it has got.

    A stuck upload used to be invisible from the app: the card said the
    tee clip "never arrived" whether the Pi had sent none of it or 90% of
    it, and the only way to tell them apart was to SSH in and read the
    agent's log. The server is holding the bytes -- it can just say.

    Sorted by staleness, so the one that has not moved in longest is
    first. `stale_seconds` is what distinguishes "climbing slowly" from
    "abandoned": a live upload's part file is touched every chunk.
    """
    from ..routers.cameras import PARTS_DIR

    out = []
    now = time.time()
    try:
        for part in sorted(PARTS_DIR.glob("*.part")):
            try:
                st = part.stat()
            except OSError:
                continue
            total = None
            meta = part.with_suffix(".meta")
            try:
                total = int(json.loads(meta.read_text()).get("total_size") or 0)
            except (OSError, ValueError, TypeError, AttributeError):
                total = None
            # cam{id}-{session}.part
            _stem = part.stem
            _cam, _, _session = _stem.partition("-")
            out.append({
                "camera": _cam,
                "session_id": _session or None,
                "received_bytes": st.st_size,
                "total_bytes": total,
                "percent": (
                    round(100.0 * st.st_size / total, 1)
                    if total else None
                ),
                "stale_seconds": int(max(0, now - st.st_mtime)),
            })
    except OSError as exc:
        log.warning("uploads-in-flight: cannot read %s: %s", PARTS_DIR, exc)
    out.sort(key=lambda r: -r["stale_seconds"])
    return {
        "in_flight": out,
        "count": len(out),
        # What the numbers mean, so the panel does not have to guess.
        "note": (
            "Bytes the server is holding for a clip still on its way. A "
            "percentage that climbs between refreshes is a slow link "
            "working; one that sits still is a Pi that has stopped "
            "trying, or a link that is down."
        ),
    }


@router.get("/long-uploads/{upload_id}/frame")
def long_upload_frame(
    upload_id: int,
    frame: int = 0,
    which: str = "tee",
    impact_frame: int | None = None,
    db: Session = Depends(get_db),
):
    """Grab a single frame from this upload as a JPG and return its
    public URL. The wizard pages through frames (±1, ±10) while the
    operator picks impact / end / ball-at-rest etc.

    `which` selects the camera. The END frame is a green-camera decision
    -- it is where the produced clip stops, and by then the cut is on the
    green -- so asking the operator to choose it from the tee view is
    asking about the wrong picture.

    Frames are cached on disk under `detect-{id}-frame-{N}.jpg` (green:
    `-green-frame-`) so re-visiting the same frame doesn't reseek.
    """
    row = db.get(LongVideoUpload, upload_id)
    if not row:
        raise HTTPException(404, "long upload not found")
    _green = str(which or "tee").lower() == "green"
    if _green and not row.green_filename:
        raise HTTPException(400, "upload has no green video")
    if not _green and not row.tee_filename:
        raise HTTPException(400, "upload has no tee video")
    src_path = _local_green(row) if _green else _local_tee(row)
    _name = row.green_filename if _green else row.tee_filename
    if not src_path.exists():
        raise HTTPException(404, f"source file missing on disk: {_name}")

    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(src_path))
    try:
        if not cap.isOpened():
            raise HTTPException(500, f"could not open {_name}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        clamped = max(0, min(total - 1 if total else 0, int(frame)))
        _tag = "green-frame" if _green else "frame"
        out_path = CLIPS_DIR / f"detect-{upload_id}-{_tag}-{clamped}.jpg"
        if not out_path.exists():
            cap.set(cv2.CAP_PROP_POS_FRAMES, clamped)
            ok, img = cap.read()
            if not ok or img is None:
                raise HTTPException(500, f"frame {clamped} unreadable")
            cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    finally:
        cap.release()

    _fps = float(probe_fps(src_path) or 0.0)
    return {
        "upload_id": upload_id,
        "frame": clamped,
        "which": "green" if _green else "tee",
        "fps": _fps,
        # When this frame was actually captured, not how far into the
        # file it is -- the operator matches clips to groups by time of
        # day, and an offset into a clip cannot do that.
        "wall_clock": _frame_wall_clock(
            db, row, "green" if _green else "tee", clamped, _fps,
        ),
        # What produce puts on the green side of the cut, so the wizard
        # can default the end frame to the same shape the pipeline uses
        # instead of hard-coding a number that quietly drifts from it.
        "green_seconds": D3_GREEN_SEC,
        # Where produce would end this clip, in GREEN frames. Computed
        # here because the tee->green offset lives here: a moment at tee
        # time T is at green time T - delta, and guessing that delta
        # client-side would put the default end seconds out on any pair
        # whose recordings did not start together.
        "default_end_frame": (
            _default_end_frame(db, row, src_path, impact_frame, total)
            if _green else None
        ),
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
    db.add(
        AuditLog(
            actor="admin",
            action="save_edit_metrics",
            target=f"long_upload:{upload_id}",
            detail=str({k: (existing.get(k)) for k in payload.keys()}),
        )
    )
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
    src_path = _local_tee(row)
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
    # True when the operator explicitly placed the resting ball (rest card
    # click on Step 2). An operator-set rest always beats the auto-derived
    # one on later renders.
    ball_manual = bool(_pick("ball_manual"))

    # Engine selector (A/B). "classical" runs the CV tracer (motion
    # detection + parabola fit, no API calls); default "ai" keeps the
    # Claude-vision pipeline below. The manual ball-editor and the
    # render-tracer-fast backup work with either engine, since both
    # produce the same ball_track_frames shape. Lets the operator run
    # both on the same clip and compare traces + point counts.
    engine = str(payload.get("engine") or saved.get("tracer_engine") or "ai").lower()
    # "classical" = the CV pipeline on MOG2 background subtraction;
    # "knn" = the SAME CV pipeline on the KNN subtractor (often separates
    # a small fast ball from drifting clouds / rippling water better);
    # "hybrid" = MOG2 detections cross-validated against a handful of AI
    # ball fixes — CV supplies density, AI supplies ground truth, and only
    # CV points that agree with the AI-anchored curve survive.
    if engine in ("classical", "knn", "hybrid"):
        # The classical tracer processes EVERY frame of its input. Running
        # it on the full multi-swing source (often minutes of video) blocked
        # this request past the HTTP proxy timeout — the wizard sat on
        # "Rendering tracer…" and then silently stopped. Cut a short window
        # around the swing's impact frame and trace just that; all frame
        # indices are shifted back into full-source coordinates so the
        # Step-2 ball editor still lines up with the source frames.
        src_for_trace = src_path
        offset_frames = 0
        fps_c = probe_fps(src_path) or 30.0
        start_override = payload.get("start_frame")
        end_override = payload.get("end_frame")
        _start_s = _end_s = None
        if (
            start_override is not None
            and end_override is not None
            and int(end_override) > int(start_override)
        ):
            # Operator-defined swing window from Step 1 — trace exactly
            # this. Extend the cut BACKWARD if needed so the background
            # subtractor still gets ~2.5s of pre-impact warmup (it needs
            # settled background + the address ball before flight starts);
            # frames before the operator's impact are dropped from the
            # track anyway via the impact cutoff, so the early extension
            # never adds points.
            _start_s = max(0.0, int(start_override) / fps_c)
            _end_s = int(end_override) / fps_c
            if impact_override is not None:
                _warm_s = max(0.0, int(impact_override) / fps_c - 2.5)
                _start_s = min(_start_s, _warm_s)
            # Clamp the render window to 30s. A wide Step-1 window (e.g.
            # end frame accidentally set 50s out) made the classical scan
            # process ~1700 frames and the trajectory scorer choke on the
            # noise — the request then died on the proxy timeout (502).
            # 30s comfortably covers any single swing + full ball flight.
            _MAX_WIN_S = 30.0
            if _end_s - _start_s > _MAX_WIN_S:
                _anchor_s = (
                    int(impact_override) / fps_c
                    if impact_override is not None else _start_s
                )
                _new_end = min(_end_s, max(_anchor_s + 27.0, _start_s + _MAX_WIN_S))
                log.info(
                    "wizard %s render: clamping window %.1fs-%.1fs -> end %.1fs "
                    "(30s cap)", engine, _start_s, _end_s, _new_end,
                )
                _end_s = _new_end
        elif impact_override is not None:
            _start_s = max(0.0, int(impact_override) / fps_c - 3.0)
            _end_s = int(impact_override) / fps_c + 9.0
        if _start_s is not None and _end_s is not None and _end_s > _start_s:
            _cut_name = f"wizard-classical-{upload_id}-{secrets.token_hex(4)}.mp4"
            _cut_path = CLIPS_DIR / _cut_name
            if cut_segment(src_path, _cut_path, _start_s, _end_s):
                src_for_trace = _cut_path
                offset_frames = int(round(_start_s * fps_c))
            else:
                log.warning(
                    "wizard classical render: cut failed — tracing full "
                    "source for upload %s (slow)", upload_id,
                )
        _dbg_prefix = f"wizdbg-{upload_id}-{secrets.token_hex(3)}"
        tracer_url_c, info_c, _traced_c, debug_url_c = _run_tracer(
            src_for_trace,
            frame_debug_dir=CLIPS_DIR,
            frame_debug_prefix=_dbg_prefix,
            bg_algo=("knn" if engine == "knn" else "mog2"),
            frame_label_offset=offset_frames,
            # Spatial launch anchor for chain selection + arbitration —
            # the operator/derived resting ball position.
            ball_rest_hint=ball_at_rest_override,
            # The operator's impact pick (mapped into cut-relative frames)
            # governs the pre-impact cutoff — not the audio re-detection.
            impact_frame_hint_override=(
                max(0, int(impact_override) - offset_frames)
                if impact_override is not None
                else None
            ),
        )
        info_c = info_c or {}
        track = info_c.get("track") or []
        try:
            import json as _json
            log.info(
                "wizard %s arc_debug: %s",
                engine, _json.dumps(info_c.get("arc_debug") or {}),
            )
        except Exception:  # noqa: BLE001
            pass

        # Rest-lock (the same launch detector production uses): cone up
        # from the operator's resting ball, 3-dot straight-line lock,
        # trail-follow over the ghost-filtered candidate pool. Wins when
        # it maps more flight than the classical chain — the exact
        # rescue for a "953 candidates, 0 points" run.
        rest_lock_info = None
        anchor_check_c = None
        _imp_cut = (
            max(0, int(impact_override) - offset_frames)
            if impact_override is not None else None
        )
        _lock_rest = ball_at_rest_override
        if ball_at_rest_override is not None and _imp_cut is not None:
            try:
                # ANCHOR CHECK first: AI film-strip departure lookup
                # (2 cheap vision calls), pixel presence check as the
                # fallback. Verified corrections feed the rest-lock
                # cone below.
                from ..services.ai_tracer import (
                    verify_rest_and_impact,
                    verify_rest_and_impact_ai,
                )

                anchor_check_c = verify_rest_and_impact_ai(
                    src_for_trace,
                    (
                        float(ball_at_rest_override[0]),
                        float(ball_at_rest_override[1]),
                    ),
                    _imp_cut, fps_c,
                    debug_dir=CLIPS_DIR,
                    debug_prefix=(
                        f"anchorai-wiz-{upload_id}-{secrets.token_hex(3)}"
                    ),
                )
                if anchor_check_c.get("api_error") or not (
                    anchor_check_c.get("available")
                ):
                    _ai_fail_c = anchor_check_c.get("reason")
                    anchor_check_c = verify_rest_and_impact(
                        src_for_trace,
                        (
                            float(ball_at_rest_override[0]),
                            float(ball_at_rest_override[1]),
                        ),
                        _imp_cut, fps_c,
                        debug_dir=CLIPS_DIR,
                        debug_prefix=(
                            f"anchorchk-wiz-{upload_id}-"
                            f"{secrets.token_hex(3)}"
                        ),
                    )
                    anchor_check_c["ai_fallback_reason"] = _ai_fail_c
                if anchor_check_c.get("verified"):
                    _lock_rest = (
                        float(anchor_check_c["rest_xy"][0]),
                        float(anchor_check_c["rest_xy"][1]),
                    )
                    _imp_cut = int(anchor_check_c["impact_frame"])
            except Exception as exc:  # noqa: BLE001
                log.warning("wizard %s: anchor check failed: %s", engine, exc)
            try:
                _chain_c, rest_lock_info = _flight_from_rest_lock(
                    _mog2_dot_pool(info_c),
                    (float(_lock_rest[0]), float(_lock_rest[1])),
                    _imp_cut, fps_c,
                    _imp_cut + int(
                        round(MOG2_LAYER_POST_IMPACT_SEC * fps_c),
                    ),
                )
                if rest_lock_info and rest_lock_info.get("seed_frames"):
                    # Report seeds in FULL-SOURCE frame space.
                    rest_lock_info["seed_frames"] = [
                        int(sf) + offset_frames
                        for sf in rest_lock_info["seed_frames"]
                    ]
                if len(_chain_c) > len(track):
                    log.info(
                        "wizard %s: rest-lock chain (%d dots, seed %s) "
                        "beats classical track (%d) — using it",
                        engine, len(_chain_c),
                        rest_lock_info.get("seed_frames"), len(track),
                    )
                    track = _chain_c
                rest_lock_info["used"] = len(_chain_c) > 0 and track is _chain_c
            except Exception as exc:  # noqa: BLE001
                log.warning("wizard %s: rest-lock failed: %s", engine, exc)
        # Per-frame MOG2 debug images — keyed by CUT-relative frame.
        # `debug_frame_images` = card crops zoomed on each chosen point;
        # `debug_frame_full_images` = whole annotated frames (source
        # coordinate space) for the editor's detector-view background.
        _dbg_imgs = info_c.get("debug_frame_images") or {}
        _dbg_full_imgs = info_c.get("debug_frame_full_images") or {}

        def _named_url(name: str | None) -> str | None:
            if not name:
                return None
            p = CLIPS_DIR / name
            if not p.exists():
                return None
            return (
                f"{settings.app_base_url}/uploads/clips/{name}"
                f"?v={int(p.stat().st_mtime)}"
            )

        def _dbg_url(seg_frame: int) -> str | None:
            return _named_url(_dbg_imgs.get(str(int(seg_frame))))

        def _dbg_full_url(seg_frame: int) -> str | None:
            return _named_url(_dbg_full_imgs.get(str(int(seg_frame))))

        # ── Hybrid: cross-validate the CV track with a few AI ball fixes ──
        # The CV picker sometimes latches onto a plausible-but-wrong point
        # cluster (tree noise chained into an "arc"). A handful of Claude
        # ball locations are sparse but trustworthy: fit a curve through
        # them, keep only CV points that agree with it, and pin the AI
        # anchors into the merged track. CV = density, AI = truth.
        n_ai_anchors = None
        if engine == "hybrid" and track:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                log.info("wizard hybrid: no ANTHROPIC_API_KEY — CV track kept")
            else:
                try:
                    import numpy as _np

                    _imp_cut = (
                        int(impact_override) - offset_frames
                        if impact_override is not None
                        else int(track[0]["frame"])
                    )
                    # Seed the AI tracker from, in order: an operator-placed
                    # rest ball (trusted), else the track's LAUNCH point —
                    # but only when the track starts within 8 frames of the
                    # operator's impact (at impact the club head IS at the
                    # ball, so that position is physically the ball
                    # regardless of which detector found it). Unseeded runs
                    # on a wide scene found 0 anchors, which disabled AI
                    # verification entirely. Anchors past the seed remain an
                    # independent check — they only validate/gap-fill the
                    # chain, never per-point rescue it.
                    _seed_pt = None
                    if ball_manual and ball_at_rest_override is not None:
                        _seed_pt = (
                            float(ball_at_rest_override[0]),
                            float(ball_at_rest_override[1]),
                        )
                    elif (
                        impact_override is not None
                        and abs(
                            int(track[0]["frame"])
                            - (int(impact_override) - offset_frames)
                        ) <= 8
                    ):
                        _seed_pt = (
                            float(track[0]["x"]), float(track[0]["y"]),
                        )
                    _seed_xy = _seed_dims = None
                    if _seed_pt is not None:
                        try:
                            _vinfo = probe_video_info(src_for_trace)
                            if _vinfo.get("width") and _vinfo.get("height"):
                                _seed_dims = (
                                    int(_vinfo["width"]), int(_vinfo["height"]),
                                )
                                _seed_xy = _seed_pt
                        except Exception:  # noqa: BLE001
                            _seed_xy = _seed_dims = None
                    ai_info = track_ball_after_impact(
                        src_for_trace,
                        max(0, _imp_cut),
                        output_dir=CLIPS_DIR,
                        output_prefix=f"hybrid-{upload_id}-{secrets.token_hex(3)}",
                        ball_xy_sent=_seed_xy,
                        ball_sent_dims=_seed_dims,
                        max_frames=12,
                    )
                    ai_pts = [
                        {
                            "frame": int(r["frame"]),
                            "x": float(r["x"]),
                            "y": float(r["y"]),
                            "ai_anchor": True,
                        }
                        for r in (ai_info.get("frames") or [])
                        if r.get("found") and r.get("x") is not None
                    ]
                    n_ai_anchors = len(ai_pts)
                    # Anchor SELF-consistency before the anchors may judge
                    # CV: fit the anchors alone, iteratively drop the worst
                    # outlier while it's >25px off, and require >=4
                    # survivors. A parabola through 3 unvalidated points has
                    # zero redundancy — one bad AI fix made the curve
                    # garbage, which then rejected CV's CORRECT points and
                    # kept noise riding the bad curve.
                    while len(ai_pts) > 4:
                        _fr = _np.array([p["frame"] for p in ai_pts], float)
                        _xs = _np.array([p["x"] for p in ai_pts], float)
                        _ys = _np.array([p["y"] for p in ai_pts], float)
                        _ycf = _np.polyfit(_fr, _ys, 2)
                        _xcf = _np.polyfit(_fr, _xs, 1 if len(ai_pts) < 6 else 2)
                        _res = _np.sqrt(
                            (_xs - _np.polyval(_xcf, _fr)) ** 2
                            + (_ys - _np.polyval(_ycf, _fr)) ** 2
                        )
                        _worst = int(_np.argmax(_res))
                        if float(_res[_worst]) <= 25.0:
                            break
                        log.info(
                            "wizard hybrid: dropped inconsistent AI anchor "
                            "f=%s (%.0fpx off the anchor fit)",
                            ai_pts[_worst]["frame"], float(_res[_worst]),
                        )
                        ai_pts.pop(_worst)
                    if len(ai_pts) >= 4:
                        _fr = _np.array([p["frame"] for p in ai_pts], float)
                        _xs = _np.array([p["x"] for p in ai_pts], float)
                        _ys = _np.array([p["y"] for p in ai_pts], float)
                        _ycf = _np.polyfit(_fr, _ys, 2)
                        _xcf = _np.polyfit(_fr, _xs, 1 if len(ai_pts) < 6 else 2)

                        # WHOLE-TRACK validation first (the operator's
                        # original rule: "if some of the MOG2 mappings are
                        # also AI mappings, it's correct"). If most anchors
                        # coincide with the CV track where they overlap in
                        # time, the ENTIRE track is confirmed — descent
                        # included — and no per-point curve gating runs.
                        # Curve gating extrapolated past the anchors' span
                        # kept re-rejecting the legitimate descent (an
                        # off-screen-apex flight has no anchors up there by
                        # definition).
                        def _near_track(a) -> bool:
                            best = 1e18
                            for p in track:
                                if abs(int(p["frame"]) - int(a["frame"])) <= 3:
                                    d = (
                                        (p["x"] - a["x"]) ** 2
                                        + (p["y"] - a["y"]) ** 2
                                    ) ** 0.5
                                    best = min(best, d)
                            return best <= 25.0

                        _have = {p["frame"] for p in ai_pts}
                        n_match = sum(1 for a in ai_pts if _near_track(a))
                        if n_match >= max(3, int(round(0.6 * len(ai_pts)))):
                            # Insert anchors ONLY near frames the chain
                            # actually covers (within 6 frames of a chain
                            # point). During an off-screen stretch the AI
                            # cannot see the ball either — "anchors" it
                            # reports there are fabricated and previously
                            # injected phantom points on no timed dot.
                            _chain_frames = sorted(
                                int(p["frame"]) for p in track
                            )

                            def _near_chain_frame(a) -> bool:
                                fa = int(a["frame"])
                                return any(
                                    abs(fa - cf) <= 6 for cf in _chain_frames
                                )

                            _add_anchors = [
                                a for a in ai_pts if _near_chain_frame(a)
                            ]
                            merged = _add_anchors + [
                                p for p in track
                                if int(p["frame"]) not in _have
                            ]
                            merged.sort(key=lambda p: int(p["frame"]))
                            log.info(
                                "wizard hybrid: track VALIDATED wholesale — "
                                "%d/%d anchors coincide; keeping all %d CV "
                                "pts + %d anchors (%d dropped as off-chain)",
                                n_match, len(ai_pts), len(track),
                                len(_add_anchors),
                                len(ai_pts) - len(_add_anchors),
                            )
                            track = merged
                        else:
                            # Anchors and CV disagree — fall back to curve
                            # gating: strict in-span, then refit-and-accept
                            # beyond span.
                            def _dist(p, xc, yc) -> float:
                                px = float(_np.polyval(xc, p["frame"]))
                                py = float(_np.polyval(yc, p["frame"]))
                                return (
                                    (p["x"] - px) ** 2 + (p["y"] - py) ** 2
                                ) ** 0.5

                            _a_lo = float(_fr.min()) - 5
                            _a_hi = float(_fr.max()) + 5
                            stage1 = [
                                p for p in track
                                if _a_lo <= p["frame"] <= _a_hi
                                and _dist(p, _xcf, _ycf) <= 25.0
                                and int(p["frame"]) not in _have
                            ]
                            merged = ai_pts + stage1
                            stage2: list = []
                            if len(merged) >= 5:
                                _f2 = _np.array([p["frame"] for p in merged], float)
                                _x2 = _np.array([p["x"] for p in merged], float)
                                _y2 = _np.array([p["y"] for p in merged], float)
                                _ycf2 = _np.polyfit(_f2, _y2, 2)
                                _xcf2 = _np.polyfit(
                                    _f2, _x2, 1 if len(merged) < 8 else 2,
                                )
                                _picked = {p["frame"] for p in merged}
                                stage2 = [
                                    p for p in track
                                    if int(p["frame"]) not in _picked
                                    and _dist(p, _xcf2, _ycf2) <= 30.0
                                ]
                                merged = merged + stage2
                            merged.sort(key=lambda p: int(p["frame"]))
                            log.info(
                                "wizard hybrid: anchors DISAGREE with CV "
                                "(%d/%d coincide) — curve gating: stage1=%d, "
                                "stage2=%d (of %d CV) -> merged %d pts",
                                n_match, len(ai_pts), len(stage1),
                                len(stage2), len(track), len(merged),
                            )
                            track = merged
                        # Re-draw the path-on-heat overlay from the MERGED
                        # track so the 🎯 view shows what actually renders.
                        _rm_name = info_c.get("raw_motion_image")
                        if _rm_name and len(track) >= 3:
                            try:
                                import cv2 as _cv2

                                _img = _cv2.imread(str(CLIPS_DIR / _rm_name))
                                if _img is not None:
                                    _f2 = _np.array([p["frame"] for p in track], float)
                                    _x2 = _np.array([p["x"] for p in track], float)
                                    _y2 = _np.array([p["y"] for p in track], float)
                                    _yc2 = _np.polyfit(_f2, _y2, 2)
                                    _xc2 = _np.polyfit(
                                        _f2, _x2, 1 if len(track) < 8 else 2
                                    )
                                    _ih, _iw = _img.shape[:2]
                                    _poly = []
                                    for _f in range(int(_f2.min()), int(_f2.max()) + 1):
                                        _px = int(round(float(_np.polyval(_xc2, _f))))
                                        _py = int(round(float(_np.polyval(_yc2, _f))))
                                        if 0 <= _px < _iw and 0 <= _py < _ih:
                                            _poly.append((_px, _py))
                                    if len(_poly) >= 2:
                                        _cv2.polylines(
                                            _img, [_np.array(_poly, _np.int32)],
                                            False, (0, 0, 255), 6, _cv2.LINE_AA,
                                        )
                                    for p in track:
                                        _c = (255, 200, 0) if p.get("ai_anchor") else (255, 255, 255)
                                        _cv2.circle(_img, (int(p["x"]), int(p["y"])), 5, _c, -1, _cv2.LINE_AA)
                                        _cv2.circle(_img, (int(p["x"]), int(p["y"])), 6, (0, 0, 255), 2, _cv2.LINE_AA)
                                    _arc_name = _rm_name.replace(".jpg", "-arc.jpg")
                                    _cv2.imwrite(
                                        str(CLIPS_DIR / _arc_name), _img,
                                        [int(_cv2.IMWRITE_JPEG_QUALITY), 85],
                                    )
                                    info_c["raw_motion_arc_image"] = _arc_name
                            except Exception as exc:  # noqa: BLE001
                                log.warning("wizard hybrid: arc redraw failed: %s", exc)
                    elif len(ai_pts) >= 2:
                        # Too few anchors to GATE anything — but verified
                        # ball fixes are still gold as references (operator:
                        # "AI found 2 points once the ball hit the sky —
                        # those should have been used"). Pin the ones that
                        # coincide with the track into it; ignore the rest.
                        def _near_track_few(a) -> bool:
                            best = 1e18
                            for p in track:
                                if abs(int(p["frame"]) - int(a["frame"])) <= 3:
                                    d = (
                                        (p["x"] - a["x"]) ** 2
                                        + (p["y"] - a["y"]) ** 2
                                    ) ** 0.5
                                    best = min(best, d)
                            return best <= 30.0

                        _coinc = [a for a in ai_pts if _near_track_few(a)]
                        if _coinc:
                            _have2 = {int(a["frame"]) for a in _coinc}
                            merged = _coinc + [
                                p for p in track
                                if int(p["frame"]) not in _have2
                            ]
                            merged.sort(key=lambda p: int(p["frame"]))
                            log.info(
                                "wizard hybrid: %d anchors (too few to "
                                "gate) — %d coincide with the track and "
                                "were pinned in",
                                len(ai_pts), len(_coinc),
                            )
                            track = merged
                        else:
                            log.info(
                                "wizard hybrid: %d anchors, none coincide "
                                "with the track — CV track kept unchanged",
                                len(ai_pts),
                            )
                    else:
                        log.info(
                            "wizard hybrid: only %d AI anchor(s) — CV "
                            "track kept unchanged",
                            len(ai_pts),
                        )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "wizard hybrid: AI verification failed (%s) — CV track kept",
                        exc,
                    )

        # Cards = the UNION of detected track points and the no-ball
        # flight-window frames the tracer emitted detector views for —
        # so the operator gets a clickable card for every flight frame
        # (found or not) and can plot the ball manually where detection
        # missed, exactly like the AI engine's cards.
        _track_by_frame = {int(p["frame"]): p for p in track}
        _all_card_frames = sorted(
            set(_track_by_frame) | {int(k) for k in _dbg_full_imgs}
        )
        ball_track_frames_out = []
        for _f in _all_card_frames:
            p = _track_by_frame.get(_f)
            if p is not None:
                ball_track_frames_out.append({
                    "frame": _f + offset_frames,
                    "found": True,
                    "x": int(round(p["x"])),
                    "y": int(round(p["y"])),
                    "confidence": None,
                    "manual": False,
                    "image_url": _dbg_url(_f),
                    # Whole annotated frame (same coords as the source) —
                    # the editor's zoomable detector-view background.
                    "overlay_image_url": _dbg_full_url(_f),
                    # Card image is ZOOMED on the ball with the ring baked
                    # in — the frontend skips its own full-frame-coord dot.
                    "zoomed": True,
                })
            else:
                ball_track_frames_out.append({
                    "frame": _f + offset_frames,
                    "found": False,
                    "x": None,
                    "y": None,
                    "confidence": None,
                    "manual": False,
                    "image_url": _dbg_full_url(_f),
                    "overlay_image_url": _dbg_full_url(_f),
                    "zoomed": False,
                })
        audio_impact = info_c.get("audio_impact") or {}
        classical_impact = (
            int(audio_impact["impact_frame"]) + offset_frames
            if audio_impact.get("ok") and audio_impact.get("impact_frame") is not None
            else saved.get("impact_frame")
        )

        # Rest position derived from the FLIGHT ITSELF: fit the recovered
        # track and evaluate at the operator's impact frame — the launch
        # point is where the ball sat. Replaces the old address-blob /
        # disappearance guess, which routinely latched onto non-ball
        # objects (trees, markers) and couldn't be corrected. The result
        # is returned as ball_at_rest so the wizard can show it on the
        # Step-2 rest card, where the operator can now click to move it.
        derived_rest = None
        if len(track) >= 3 and not ball_manual:
            try:
                import numpy as _np

                _fr = _np.array([p["frame"] for p in track], float)
                _xs = _np.array([p["x"] for p in track], float)
                _ys = _np.array([p["y"] for p in track], float)
                _imp = (
                    float(int(impact_override) - offset_frames)
                    if impact_override is not None
                    else float(_fr.min())
                )
                _ycf = _np.polyfit(_fr, _ys, 2)
                _xcf = _np.polyfit(_fr, _xs, 1 if len(track) < 8 else 2)
                derived_rest = (
                    float(_np.polyval(_xcf, _imp)),
                    float(_np.polyval(_ycf, _imp)),
                )
                log.info(
                    "wizard %s: rest derived from flight fit @ impact -> "
                    "(%.0f, %.0f)", engine, derived_rest[0], derived_rest[1],
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("wizard %s: rest derivation failed: %s", engine, exc)
                derived_rest = None

        # Re-render the overlay with the MODERN renderer (robust parabola
        # fit with outlier rejection + rest-constrained start + broadcast-
        # blue style) on the classical/KNN detections. The legacy dashed
        # renderer pins the first point to the ball and bridges to the fit
        # with a single straight segment — when the fit doesn't extrapolate
        # down to ball height, that bridge draws as a long straight
        # diagonal with a visible kink. The modern renderer constrains the
        # curve THROUGH the rest point instead, and its outlier rejection
        # also drops any bad backfilled points before they can bend the
        # line. Classical stays the detector; only the drawing changes.
        if track:
            try:
                _render_track = [
                    {
                        "frame": int(p["frame"]),
                        "found": True,
                        "x": int(round(p["x"])),
                        "y": int(round(p["y"])),
                    }
                    for p in track
                ]
                _rt_name = (
                    f"wizard-{engine}-{upload_id}-{secrets.token_hex(3)}_tracer.mp4"
                )
                _rt_path = CLIPS_DIR / _rt_name
                _rt_info = render_tracer_video(
                    src_for_trace,
                    _rt_path,
                    ball_rest_xy_native=(derived_rest or ball_at_rest_override),
                    impact_frame_idx=(
                        int(impact_override) - offset_frames
                        if impact_override is not None
                        else int(_render_track[0]["frame"])
                    ),
                    track_frames=_render_track,
                )
                if (
                    _rt_info.get("ok")
                    and _rt_path.exists()
                    and _rt_path.stat().st_size > 0
                ):
                    compress_for_email(_rt_path)
                    tracer_url_c = (
                        f"{settings.app_base_url}/uploads/clips/{_rt_name}"
                        f"?v={int(_rt_path.stat().st_mtime)}"
                    )
                else:
                    log.warning(
                        "wizard %s: modern re-render not ok (%s) — keeping "
                        "classical render", engine, _rt_info.get("error"),
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "wizard %s: modern re-render failed (%s) — keeping "
                    "classical render", engine, exc,
                )
        raw_motion_url = _named_url(info_c.get("raw_motion_image"))
        raw_motion_arc_url = _named_url(info_c.get("raw_motion_arc_image"))
        raw_motion_frames_url = _named_url(info_c.get("raw_motion_frames_image"))
        # Timed transient dots (source-frame mapped) — persisted so the
        # Step-2 click-to-plot view survives closing/reopening the wizard.
        timed_points_out = [
            {
                "frame": int(p["frame"]) + offset_frames,
                "x": int(round(p["x"])),
                "y": int(round(p["y"])),
            }
            for p in (info_c.get("timed_points") or [])
            if _imp_cut is None or int(p["frame"]) >= _imp_cut
        ][:2000]
        candidates_out = [
            {
                "frame": int(c["frame"]) + offset_frames,
                "x": int(round(c["x"])),
                "y": int(round(c["y"])),
            }
            for c in (info_c.get("candidates") or [])
            if _imp_cut is None or int(c["frame"]) >= _imp_cut
        ][:4000]
        saved.update(
            {
                "tracer_engine": engine,
                "tracer_url": tracer_url_c,
                "tracer_debug_url": debug_url_c,
                "tracer_raw_motion_url": raw_motion_url,
                "tracer_raw_motion_arc_url": raw_motion_arc_url,
                "tracer_raw_motion_frames_url": raw_motion_frames_url,
                "timed_points": timed_points_out,
                "cand_points": candidates_out[:1500],
                "tracer_info": {
                    "engine": engine,
                    "ok": bool(info_c.get("ok")),
                    "n_points": info_c.get("n_points"),
                    "n_candidates": info_c.get("n_candidates"),
                    "n_backfilled": info_c.get("n_backfilled"),
                    "residual_px": info_c.get("residual_px"),
                    "debug_url": debug_url_c,
                },
                "ball_track_frames": ball_track_frames_out,
            }
        )
        # Only let the audio-derived impact update the saved metrics when
        # the operator DIDN'T explicitly supply one in this request — an
        # explicit Step-1 pick must never be silently overwritten by the
        # render's own audio opinion.
        if classical_impact is not None and payload.get("impact_frame") is None:
            saved["impact_frame"] = int(classical_impact)
        # Persist the flight-derived rest (never over an operator-set one).
        rest_used = derived_rest or ball_at_rest_override
        if derived_rest is not None:
            saved["ball"] = {
                "x": int(round(derived_rest[0])),
                "y": int(round(derived_rest[1])),
            }
        row.edit_metrics = saved
        db.add(row)
        db.add(
            AuditLog(
                actor="admin",
                action="render_wizard_tracer_classical",
                target=f"long_upload:{upload_id}",
                detail=str(
                    {
                        "ok": bool(info_c.get("ok")),
                        "n_points": info_c.get("n_points"),
                        "n_candidates": info_c.get("n_candidates"),
                    }
                ),
            )
        )
        db.commit()
        db.refresh(row)
        return {
            "upload_id": upload_id,
            "engine": engine,
            "tracer_url": tracer_url_c,
            "ball_track_frames": ball_track_frames_out,
            "n_points": len(track),
            "n_candidates": info_c.get("n_candidates"),
            "n_backfilled": info_c.get("n_backfilled"),
            "n_ai_anchors": n_ai_anchors,
            "rest_lock": rest_lock_info,
            "anchor_check": (
                {
                    **{
                        k: anchor_check_c.get(k)
                        for k in (
                            "verified", "snapped", "snap_px",
                            "impact_delta", "present_ratio_pre", "reason",
                            "ai_fallback_reason",
                        )
                    },
                    "impact_frame": (
                        int(anchor_check_c["impact_frame"]) + offset_frames
                        if anchor_check_c.get("impact_frame") is not None
                        else None
                    ),
                    "rest_xy": anchor_check_c.get("rest_xy"),
                    "image_url": _named_url(anchor_check_c.get("image")),
                    "image_mog2_url": _named_url(
                        anchor_check_c.get("image_mog2"),
                    ),
                }
                if anchor_check_c else None
            ),
            "debug_url": debug_url_c,
            "raw_motion_url": raw_motion_url,
            "raw_motion_arc_url": raw_motion_arc_url,
            "raw_motion_frames_url": raw_motion_frames_url,
            "arc_debug": info_c.get("arc_debug"),
            # Per-frame MOG2 candidate detections (source-frame coords) so
            # the Step-2 editor can draw them as clickable snap targets —
            # click a dot to mark it as the ball for that frame. Flight
            # frames only (impact − 2 onward) to keep the golfer's
            # pre-swing body motion out of the editor.
            "candidates": candidates_out,
            # The timed transient dots — the SAME dots the frames-on-heat
            # image labels — so the Step-2 "click-to-plot" view can draw
            # them as clickable targets: one click queues the ball at that
            # dot for that dot's frame.
            "timed_points": timed_points_out,
            "ball_at_rest": (
                {"x": int(round(rest_used[0])), "y": int(round(rest_used[1]))}
                if rest_used is not None
                else None
            ),
            "ball_manual": ball_manual,
            "edit_metrics": row.edit_metrics,
        }

    # Few-shot prior from prior broadcast clips on the same course.
    # Excluding the upload itself so a re-produce doesn't reference
    # its own (possibly-stale) prior result.
    wizard_hole = None
    try:
        wizard_hole = int(saved.get("hole_number")) if saved.get("hole_number") else None
    except (TypeError, ValueError):
        pass
    examples_by_kind = tracer_examples.fetch_all_kinds(
        db,
        course_id=row.course_id,
        hole_number=wizard_hole,
        exclude_lvu_ids={row.id},
    )

    pipe = run_full_ai_tracer_pipeline(
        src_path,
        output_dir=CLIPS_DIR,
        output_prefix=f"wizard-{upload_id}",
        impact_frame_override=int(impact_override)
        if impact_override is not None
        else None,
        ball_at_rest_override=ball_at_rest_override,
        manual_ball_positions=manual_positions or None,
        handedness_override=handedness,
        examples_by_kind=examples_by_kind,
        # ai_mog2 renders its own WINDOWED video below (with the MOG2
        # additions merged in) — the pipeline's full-length render on a
        # long multi-swing source was the proxy-timeout killer.
        render_video=(engine != "ai_mog2"),
        ball_track_enabled=settings.ai_ball_track_enabled,
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

    # "ai_mog2": PRODUCE'S layer-in, in the wizard. After the AI
    # pipeline (its full-length render skipped — render_video=False),
    # cut the [impact-3s, impact+5s] window, run the MOG2 layer
    # (rest-lock + per-frame candidate trail), shift added points back
    # to full-source frames, then render ONE windowed video of the
    # final track. Windowing (write_start/write_end) keeps the render
    # to the swing instead of the whole multi-swing source — the
    # full-length passes were blowing the HTTP proxy timeout (502).
    mog2_overlay_url = None
    mog2_stats = None
    wiz_timed_points = []
    wiz_cand_points = []
    wiz_raw_motion_url = None
    if engine == "ai_mog2":
        _imp_full = (
            pipe.get("impact_refined") or pipe.get("impact") or {}
        ).get("impact_frame")
        _fps_w = float(pipe.get("fps") or probe_fps(src_path) or 30.0)
        try:
            if _imp_full is None:
                log.info("wizard ai_mog2: no impact frame — layer skipped")
            else:
                _off = max(0, int(_imp_full) - int(round(3.0 * _fps_w)))
                _cut_end_sec = (
                    int(_imp_full)
                    + int(round((MOG2_LAYER_POST_IMPACT_SEC + 1.0) * _fps_w))
                ) / _fps_w
                _cut_name = (
                    f"wizard-aimog2-{upload_id}-{secrets.token_hex(3)}.mp4"
                )
                _cut_path = CLIPS_DIR / _cut_name
                if cut_segment(
                    src_path, _cut_path, _off / _fps_w, _cut_end_sec,
                ):
                    _shifted = {
                        "ball_track_frames": [
                            {**r, "frame": int(r["frame"]) - _off}
                            for r in (pipe.get("ball_track_frames") or [])
                            if r.get("frame") is not None
                            and int(r["frame"]) >= _off
                        ],
                        "impact_refined": {
                            "impact_frame": int(_imp_full) - _off,
                        },
                        "ball_rest_xy_native": pipe.get(
                            "ball_rest_xy_native",
                        ),
                    }
                    _layer = _mog2_layer_for_ai_track(
                        _cut_path, _shifted, render_extended=False,
                    )
                    if _layer:
                        mog2_stats = _layer.get("stats")
                        _ac = (mog2_stats or {}).get("anchor_check")
                        if _ac and _ac.get("image"):
                            _acp = CLIPS_DIR / _ac["image"]
                            if _acp.exists():
                                _ac["image_url"] = _public_url(_acp)
                        _ovl = _layer.get("overlay_name")
                        if _ovl and (CLIPS_DIR / _ovl).exists():
                            mog2_overlay_url = _public_url(CLIPS_DIR / _ovl)
                        # Timed heat dots (shifted back to source frames)
                        # + raw-motion image for the click-to-plot view.
                        wiz_timed_points = [
                            {
                                "frame": int(p["frame"]) + _off,
                                "x": int(round(float(p["x"]))),
                                "y": int(round(float(p["y"]))),
                            }
                            for p in (_layer.get("timed_points") or [])
                        ][:2000]
                        wiz_cand_points = [
                            {
                                "frame": int(p["frame"]) + _off,
                                "x": int(round(float(p["x"]))),
                                "y": int(round(float(p["y"]))),
                            }
                            for p in (_layer.get("candidates") or [])
                        ][:1500]
                        _rawm = _layer.get("raw_motion_image")
                        if _rawm and (CLIPS_DIR / _rawm).exists():
                            wiz_raw_motion_url = _public_url(
                                CLIPS_DIR / _rawm,
                            )
                        _added_back = [
                            {**r, "frame": int(r["frame"]) + _off}
                            for r in (_layer.get("merged") or [])
                            if r.get("source") in ("mog2", "launch", "arc")
                        ]
                        if _added_back:
                            _addf = {r["frame"] for r in _added_back}
                            _base = [
                                r for r in (pipe.get("ball_track_frames") or [])
                                if r.get("found")
                                or int(r.get("frame") or -1) not in _addf
                            ]
                            pipe["ball_track_frames"] = sorted(
                                _base + _added_back,
                                key=lambda r: int(r.get("frame") or 0),
                            )
                        log.info(
                            "wizard ai_mog2: layer stats=%s extended=%s",
                            mog2_stats, bool(_added_back),
                        )
        except Exception as exc:  # noqa: BLE001
            log.warning("wizard ai_mog2: layer failed (%s) — AI-only kept", exc)
        # ONE windowed render of the final track (merged, or AI-only if
        # the layer added nothing / failed).
        try:
            if _imp_full is not None and (pipe.get("ball_track_frames")):
                _ext = CLIPS_DIR / f"wizard-{upload_id}_ai_mog2_tracer.mp4"
                _rr = render_tracer_video(
                    src_path, _ext,
                    ball_rest_xy_native=pipe.get("ball_rest_xy_native"),
                    impact_frame_idx=int(_imp_full),
                    track_frames=pipe.get("ball_track_frames") or [],
                    write_start=max(
                        0, int(_imp_full) - int(round(3.0 * _fps_w)),
                    ),
                    write_end=int(_imp_full) + int(
                        round((MOG2_LAYER_POST_IMPACT_SEC + 2.0) * _fps_w),
                    ),
                )
                if _rr.get("ok") and _ext.exists():
                    compress_for_email(_ext)
                    if _ext.exists() and _ext.stat().st_size > 0:
                        tracer_url = _public_url(_ext)
        except Exception as exc:  # noqa: BLE001
            log.warning("wizard ai_mog2: windowed render failed: %s", exc)

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
        ball_track_frames_out.append(
            {
                "frame": rec.get("frame"),
                "found": rec.get("found"),
                "x": rec.get("x"),
                "y": rec.get("y"),
                "confidence": rec.get("confidence"),
                "manual": rec.get("manual", False),
                "source": rec.get("source"),
                "image_url": url,
            }
        )

    saved.update(
        {
            "handedness": pipe.get("handedness", {}).get("handedness")
            if pipe.get("handedness")
            else handedness,
            "address_frame": (pipe.get("address") or {}).get("address_frame"),
            "address_image_url": _public_url((pipe.get("address_image_path") or None)),
            "impact_frame": (
                pipe.get("impact_refined") or pipe.get("impact") or {}
            ).get("impact_frame"),
            "impact_image_url": _public_url((pipe.get("impact_image_path") or None)),
            "ball": (
                {
                    "x": int((pipe.get("handedness") or {}).get("ball_x") or 0),
                    "y": int((pipe.get("handedness") or {}).get("ball_y") or 0),
                }
                if (pipe.get("handedness") or {}).get("ball_x") is not None
                else saved.get("ball")
            ),
            "tracer_url": tracer_url,
            "tracer_info": tracer_info,
            "tracer_engine": engine,
            "ball_track_frames": ball_track_frames_out,
            **(
                {
                    "mog2_overlay_url": mog2_overlay_url,
                    "mog2_stats": mog2_stats,
                    "timed_points": wiz_timed_points,
                    "cand_points": wiz_cand_points,
                    "tracer_raw_motion_url": wiz_raw_motion_url,
                }
                if engine == "ai_mog2" else {}
            ),
        }
    )
    row.edit_metrics = saved
    db.add(row)
    db.add(
        AuditLog(
            actor="admin",
            action="render_wizard_tracer",
            target=f"long_upload:{upload_id}",
            detail=str(
                {
                    "tracer_ok": bool(tracer_url),
                    "n_frames": len(ball_track_frames_out),
                    "overrides": list(payload.keys()),
                }
            ),
        )
    )
    db.commit()
    db.refresh(row)

    return {
        "upload_id": upload_id,
        "engine": engine,
        "tracer_url": tracer_url,
        "ball_track_frames": ball_track_frames_out,
        "n_points": len(ball_track_frames_out),
        "mog2_overlay_url": mog2_overlay_url,
        "mog2_stats": mog2_stats,
        "timed_points": wiz_timed_points,
        "candidates": wiz_cand_points,
        "raw_motion_url": wiz_raw_motion_url,
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
      cleared_frames: list of frame indices the operator rejected on
        the wizard. Dropped from the merged track entirely so the
        renderer doesn't re-anchor to AI detections the operator
        already said were wrong.

    Returns the updated tracer URL + the merged ball_track_frames.
    """
    row = db.get(LongVideoUpload, upload_id)
    if not row:
        raise HTTPException(404, "long upload not found")
    if not row.tee_filename:
        raise HTTPException(400, "upload has no tee video")
    src_path = _local_tee(row)
    if not src_path.exists():
        raise HTTPException(404, f"tee source file missing on disk: {row.tee_filename}")

    saved = dict(row.edit_metrics or {})
    # For a multi-swing upload the frontend passes the SELECTED swing's own
    # data (base track, impact, ball, target, window) so we render just that
    # swing instead of relying on stale top-level metrics + the whole clip.
    # Single-swing sends none of these and falls back to saved.
    existing = list(
        payload.get("base_track_frames") or saved.get("ball_track_frames") or []
    )
    manual_positions = payload.get("manual_positions") or []
    # Frames the operator explicitly cleared on the wizard. We drop
    # these from the merged track entirely so the renderer doesn't
    # anchor the tracer to a wrong AI detection the operator already
    # rejected.
    cleared_raw = payload.get("cleared_frames") or []
    cleared_frames: set[int] = set()
    for cf in cleared_raw:
        try:
            cleared_frames.add(int(cf))
        except (TypeError, ValueError):
            continue
    # Remember previously-cleared frames too, so a clear STAYS cleared
    # across re-renders and reopens. Without this the stored track (which
    # still holds the rejected detection) quietly re-introduces it on the
    # next merge — the "cleared frame reverts on re-render" bug.
    for cf in (saved.get("cleared_frames") or []):
        try:
            cleared_frames.add(int(cf))
        except (TypeError, ValueError):
            continue

    # Merge: frame index → entry. Manual wins. Manual additions for
    # frames the AI never visited are flagged manual=True / found=True
    # so the renderer treats them as confirmed points.
    by_frame: dict[int, dict] = {}
    for rec in existing:
        try:
            f = int(rec.get("frame"))
        except (TypeError, ValueError):
            continue
        if f in cleared_frames:
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
            f = int(mp["frame"])
            x = int(mp["x"])
            y = int(mp["y"])
        except (KeyError, TypeError, ValueError):
            continue
        # A re-mark on a previously-cleared frame is the operator
        # putting the ball back; the frontend already drops it from
        # the cleared set, but be defensive in case payloads cross.
        cleared_frames.discard(f)
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
    ball = payload.get("ball_at_rest") or saved.get("ball") or {}
    ball_xy = None
    try:
        if ball and ball.get("x") is not None and ball.get("y") is not None:
            ball_xy = (float(ball["x"]), float(ball["y"]))
    except Exception:
        ball_xy = None
    try:
        impact_idx = int(payload.get("impact_frame") or saved.get("impact_frame") or 0)
    except (TypeError, ValueError):
        impact_idx = 0
    # Target / landing spot (the flag the operator plots). Used to aim the
    # tracer's continuation past the last plotted point at the downrange
    # landing point instead of fabricating a descent into the foreground.
    target = payload.get("target") or saved.get("target") or {}
    target_xy = None
    try:
        if target and target.get("x") is not None and target.get("y") is not None:
            target_xy = (float(target["x"]), float(target["y"]))
    except Exception:
        target_xy = None

    # Output window: render ONLY the selected swing's frame span (from the
    # frontend), so a multi-swing / long source produces a short clip of
    # just that swing instead of re-rendering the whole video.
    win = payload.get("render_window") or {}
    write_start = write_end = None
    try:
        if win.get("start_frame") is not None:
            write_start = max(0, int(win["start_frame"]))
        if win.get("end_frame") is not None:
            write_end = int(win["end_frame"])
    except (TypeError, ValueError):
        write_start = write_end = None

    output_path = CLIPS_DIR / f"wizard-{upload_id}_tracer.mp4"
    info = render_tracer_video(
        src_path,
        output_path,
        ball_rest_xy_native=ball_xy,
        impact_frame_idx=impact_idx,
        target_xy=target_xy,
        write_start=write_start,
        write_end=write_end,
        # Forward the manual flag — the renderer pins manual anchors
        # so the parabola fit can't reject them and weights them so
        # they actually shape the rendered arc instead of getting
        # outvoted by the AI's earlier points.
        track_frames=[
            {
                "frame": e["frame"],
                "found": e["found"],
                "x": e["x"],
                "y": e["y"],
                "manual": e.get("manual", False),
            }
            for e in merged
        ],
    )
    if not info.get("ok"):
        raise HTTPException(
            500, f"tracer render failed: {info.get('error') or 'unknown'}"
        )

    compress_for_email(output_path)
    tracer_url = (
        f"{settings.app_base_url}/uploads/clips/{output_path.name}"
        f"?v={int(output_path.stat().st_mtime)}"
    )

    saved["tracer_url"] = tracer_url
    saved["tracer_info"] = info
    saved["ball_track_frames"] = merged
    # Record the source frame the tracer video now starts at (its frame 0),
    # or None when it's the whole clip. Step-3 finalize needs this to trim /
    # composite against the tracer's own timeline instead of source frames.
    saved["tracer_segment_start"] = write_start
    # Persist the accumulated cleared set (minus any frames just re-marked)
    # so the rejection is remembered on the next render / reopen.
    saved["cleared_frames"] = sorted(cleared_frames)
    # Re-finalizing was previously baked from the stale tracer — drop
    # the cached final URL so Step 3 knows to re-apply graphics.
    saved.pop("finalized_video_url", None)
    row.edit_metrics = saved
    db.add(row)
    db.add(
        AuditLog(
            actor="admin",
            action="render_tracer_fast",
            target=f"long_upload:{upload_id}",
            detail=(
                f"merged_frames={len(merged)} manual={len(manual_positions)} "
                f"cleared={len(cleared_frames)}"
            ),
        )
    )
    db.commit()
    db.refresh(row)
    return {
        "upload_id": upload_id,
        "tracer_url": tracer_url,
        "ball_track_frames": merged,
        "edit_metrics": row.edit_metrics,
    }


@router.post("/long-uploads/{upload_id}/scan-region")
def scan_plot_region(
    upload_id: int,
    payload: dict = Body(default={}),
    db: Session = Depends(get_db),
):
    """Deep-scan a region of the tee video for motion blobs — the
    click-to-plot "give me buttons where I'm zoomed" pass.

    Runs a plain frame-diff over the requested frame window, cropped to
    the region, with gates FAR looser than the tracer's candidate
    pipeline: every transient blob becomes a dot, because here the
    OPERATOR is the filter — they can see the ball on the heat and just
    need something clickable on it.

    Body: x, y, w, h (native px), start_frame, end_frame (source
    frames; end defaults to start+240, span hard-capped at 900), and
    `sensitivity` 1-3.

    SENSITIVITY EXISTS BECAUSE THE BALL IS SMALL AND FAST. At 720p50 a
    driven ball crossing frame is a handful of pixels, often dimmer than
    the foliage sparkle around it, and the level-1 gates (the originals)
    routinely came back with nothing in a region where the operator can
    plainly see it. Each level lowers the frame-diff threshold, widens
    the blob-size band and keeps more blobs per frame; level 3 will hand
    back leaves moving in the wind, which is fine — the operator clicks
    the ball and ignores the rest. Only the count is capped, so a noisy
    region cannot return a megabyte of dots.

    Returns {dots: [{frame, x, y}, ...]} in source coords.
    """
    import cv2  # type: ignore

    row = db.get(LongVideoUpload, upload_id)
    if not row:
        raise HTTPException(404, "long upload not found")
    if not row.tee_filename:
        raise HTTPException(400, "no tee video on this upload")
    src = _local_tee(row)
    if not src.exists():
        raise HTTPException(404, "tee video missing on disk")

    cap = cv2.VideoCapture(str(src))
    try:
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fw <= 0 or fh <= 0:
            raise HTTPException(500, "could not read video dimensions")
        try:
            x = max(0, min(fw - 2, int(payload.get("x") or 0)))
            y = max(0, min(fh - 2, int(payload.get("y") or 0)))
            w = max(8, min(fw - x, int(payload.get("w") or fw)))
            h = max(8, min(fh - y, int(payload.get("h") or fh)))
            _sens = int(payload.get("sensitivity") or 2)
            _sens = max(1, min(3, _sens))
            start = max(0, int(payload.get("start_frame") or 0))
            _end_raw = payload.get("end_frame")
            end = int(_end_raw) if _end_raw is not None else start + 240
        except (TypeError, ValueError):
            raise HTTPException(400, "bad region / frame values")
        if nb > 0:
            end = min(end, nb - 1)
        end = min(end, start + 900)
        if end <= start:
            raise HTTPException(400, "empty frame window")

        # (frame-diff threshold, max blobs kept per frame, biggest blob
        # area, total dot cap). Level 2 is the default: measurably more
        # generous than the original gates, still short of returning
        # every leaf.
        _THRESH, _PER_FRAME, _AREA_MAX, _CAP = {
            1: (12, 6, 600, 1200),
            2: (8, 10, 900, 2000),
            3: (5, 16, 1400, 3000),
        }[_sens]

        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
        prev = None
        dots: list[dict] = []
        for f in range(start, end + 1):
            ok, frame = cap.read()
            if not ok:
                break
            crop = frame[y:y + h, x:x + w]
            gray = cv2.GaussianBlur(
                cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (3, 3), 0,
            )
            if prev is not None:
                diff = cv2.absdiff(gray, prev)
                _, th = cv2.threshold(diff, _THRESH, 255, cv2.THRESH_BINARY)
                th = cv2.dilate(th, None, iterations=1)
                n, _lbl, stats, cents = cv2.connectedComponentsWithStats(th)
                frame_hits = []
                for i in range(1, n):
                    area = int(stats[i, cv2.CC_STAT_AREA])
                    if 1 <= area <= _AREA_MAX:
                        frame_hits.append(
                            (area, float(cents[i][0]), float(cents[i][1])),
                        )
                # Largest few per frame — keeps foliage sparkle from
                # burying the ball blob in a noisy region.
                frame_hits.sort(reverse=True)
                for _area, cx, cy in frame_hits[:_PER_FRAME]:
                    dots.append({
                        "frame": int(f),
                        "x": int(round(x + cx)),
                        "y": int(round(y + cy)),
                    })
            prev = gray
            if len(dots) >= _CAP:
                break
    finally:
        cap.release()
    log.info(
        "scan-region upload=%s region=(%d,%d %dx%d) f%d-%d sens=%d "
        "-> %d dots", upload_id, x, y, w, h, start, end, _sens, len(dots),
    )
    return {
        "dots": dots[:_CAP],
        "sensitivity": _sens,
        "n_frames": end - start + 1,
        "region": {"x": x, "y": y, "w": w, "h": h},
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
        raise HTTPException(400, "no rendered tracer yet — finish Step 2 first")
    fname = tracer_url.rstrip("/").split("?")[0].rsplit("/", 1)[-1]
    tracer_path = CLIPS_DIR / fname
    if not tracer_path.exists():
        raise HTTPException(404, f"tracer file missing on disk: {fname}")

    # Per-swing final file when the caller says which swing this is —
    # multi-swing uploads would otherwise share ONE final file, so
    # finalizing swing B silently replaced the video behind swing A's
    # committed clip URL. Single-swing callers keep the legacy name.
    _swing_no = payload.get("swing")
    final_path = CLIPS_DIR / (
        f"wizard-{upload_id}_final-s{int(_swing_no)}.mp4"
        if _swing_no is not None
        else f"wizard-{upload_id}_final.mp4"
    )

    # Resolve the operator's trim / cut frames. Prefer payload values —
    # for multi-swing uploads the frame indices live inside
    # edit_metrics.swings[i], not at the top level, so the frontend
    # sends the selected swing's values; fall back to top-level
    # edit_metrics for single-swing rows.
    def _pick_frame(key):
        v = payload.get(key)
        return saved.get(key) if v is None else v

    start_frame_saved = _pick_frame("start_frame")
    end_frame_saved = _pick_frame("end_frame")
    cut_frame_saved = _pick_frame("cut_frame")
    impact_frame_saved = _pick_frame("impact_frame")

    # Frame → seconds. Frame indices are in the tee SOURCE's space, so
    # use the source fps (fall back to tracer_info, then 30).
    tee_src = _local_tee(row)
    fps = float(probe_fps(tee_src) or 0) if (tee_src and tee_src.exists()) else 0.0
    if fps <= 0:
        fps = float((saved.get("tracer_info") or {}).get("fps") or 0)
    if fps <= 0:
        fps = 30.0

    # The Step-2 tracer may be pre-trimmed to just the selected swing (its
    # frame 0 = source frame `seg_start`). Tee-tracer times must then be in
    # the TRACER's own timeline (source_sec − seg_off); green stays in
    # SOURCE time (+Δ) because it's still the whole green clip. seg_off is 0
    # for a whole-clip tracer (single-swing), giving the old behaviour.
    _seg = saved.get("tracer_segment_start")
    seg_off_sec = (float(int(_seg)) / fps) if _seg is not None else 0.0

    # Source-space seconds of the operator's frames.
    start_src_sec = float(start_frame_saved or 0) / fps
    end_src_sec = (
        float(int(end_frame_saved) + 1) / fps if end_frame_saved is not None else None
    )

    green_path = _local_green(row)
    has_green = green_path is not None and green_path.exists()

    cut_src_sec = None
    if cut_frame_saved is not None:
        cut_src_sec = max(0.0, float(cut_frame_saved) / fps)
    elif has_green and impact_frame_saved is not None:
        cut_src_sec = max(0.0, float(impact_frame_saved) / fps + 2.5)

    # Tee-tracer-local seconds (0 at the start of the pre-trimmed tracer).
    start_sec = max(0.0, start_src_sec - seg_off_sec)
    end_sec = (end_src_sec - seg_off_sec) if end_src_sec is not None else None
    cut_sec = (cut_src_sec - seg_off_sec) if cut_src_sec is not None else None

    # Real-time offset between the two cameras. The tee and green start
    # recording a fraction of a second apart (trigger latency), so the
    # same frame index isn't the same real instant. Pull it from the
    # source CameraEvent's reported first-frame timestamps:
    #   green_local_time(M) = tee_local_time(M) + (tee_started − green_started)
    # delta < 0 when green started later (the usual case). Falls back to
    # 0 (frame-aligned) when timestamps are missing — old clips / Pis.
    green_delta = 0.0
    if row.camera_event_id:
        from ..models import CameraEvent
        cam_evt = db.get(CameraEvent, row.camera_event_id)
        if (
            cam_evt is not None
            and cam_evt.tee_recording_started_at is not None
            and cam_evt.green_recording_started_at is not None
        ):
            green_delta = (
                cam_evt.tee_recording_started_at
                - cam_evt.green_recording_started_at
            ).total_seconds()

    built = False
    # Dual-camera cut: tee tracer [start, cut] then green [cut+Δ, end+Δ],
    # where Δ aligns the green clip to the tee's real-world clock so the
    # switch lands on the same instant in both cameras.
    if has_green and cut_sec is not None and cut_sec > start_sec + 0.05:
        tracer_dur = float((probe_video_info(tracer_path) or {}).get("duration") or 0.0)
        green_dur = float((probe_video_info(green_path) or {}).get("duration") or 0.0)
        composite_cut = min(cut_sec, tracer_dur or cut_sec)
        composite_end = end_sec if end_sec is not None else (composite_cut + 10.0)
        composite_end = max(composite_cut + 0.1, composite_end)
        # Map the tee-local cut/end into green-local time. The tracer is
        # segment-relative (frame 0 = source seg_off), so add seg_off back to
        # recover SOURCE seconds, then +Δ to align the green clip's clock.
        green_cut = max(0.0, composite_cut + seg_off_sec + green_delta)
        green_end = composite_end + seg_off_sec + green_delta
        if green_dur:
            green_end = min(green_end, green_dur)
        green_end = max(green_cut + 0.1, green_end)
        if concat_two_clips(
            tracer_path, start_sec, composite_cut,
            green_path, green_cut, green_end,
            final_path,
        ) and final_path.exists() and final_path.stat().st_size > 0:
            built = True
            log.info(
                "finalize: composite upload=%s tee[%.2f-%.2f] green[%.2f-%.2f] Δ=%.3fs",
                upload_id, start_sec, composite_cut, green_cut, green_end, green_delta,
            )
        else:
            log.warning(
                "finalize: tee→green composite failed for upload %s; "
                "falling back to tee-only", upload_id,
            )

    if not built:
        # Tee-only: copy the tracer, then trim to [start, end].
        try:
            import shutil

            shutil.copyfile(tracer_path, final_path)
        except Exception as exc:
            raise HTTPException(500, f"copy failed: {exc}")
        if start_frame_saved or end_frame_saved:
            trim_end = end_sec
            if trim_end is None:
                dur = float((probe_video_info(final_path) or {}).get("duration") or 0.0)
                trim_end = dur if dur > 0 else start_sec + 60.0
            if trim_end > start_sec + 0.05:
                trimmed_path = final_path.with_name(
                    final_path.stem + ".trim" + final_path.suffix
                )
                if (
                    cut_segment(final_path, trimmed_path, start_sec, trim_end)
                    and trimmed_path.exists()
                    and trimmed_path.stat().st_size > 0
                ):
                    trimmed_path.replace(final_path)
                else:
                    log.warning(
                        "finalize: trim failed for upload %s "
                        "(start_sec=%.2f end_sec=%.2f); using untrimmed final",
                        upload_id, start_sec, trim_end,
                    )

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
                    src_w_native = (
                        int(_cap_src.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
                    )
                    src_h_native = (
                        int(_cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
                    )
                finally:
                    _cap_src.release()
            except Exception:  # pragma: no cover
                pass
        # Final-file dims (what ffmpeg overlay will treat as the canvas).
        final_info = probe_video_info(final_path)
        final_w = final_info.get("width")
        final_h = final_info.get("height")
        log.info(
            "finalize: target_xy=%s src(cv2)=%sx%s final(ffprobe)=%sx%s saved=%s",
            original_target,
            src_w_native,
            src_h_native,
            final_w,
            final_h,
            target_saved,
        )
        if (
            src_w_native
            and src_h_native
            and final_w
            and final_h
            and (src_w_native != final_w or src_h_native != final_h)
        ):
            sx = float(final_w) / float(src_w_native)
            sy = float(final_h) / float(src_h_native)
            target_xy = (
                int(round(target_xy[0] * sx)),
                int(round(target_xy[1] * sy)),
            )
            log.info(
                "finalize: scaled target_xy %s → %s (factor %.3f, %.3f)",
                original_target,
                target_xy,
                sx,
                sy,
            )
        # Final safety net: if the resulting target lies outside the
        # actual canvas, the saved coords were almost certainly in a
        # different reference frame than we expect (e.g. a stale
        # auto-detect from before the native-dim fix that referenced
        # ~1024px). Re-scale assuming the *largest* of (frame-w, x,
        # original-x) is the actual ref width — gives a sensible
        # fallback instead of off-screen placement.
        if final_w and final_h and (target_xy[0] >= final_w or target_xy[1] >= final_h):
            ref_w = max(final_w, target_xy[0] + 1, original_target[0] + 1)
            ref_h = max(final_h, target_xy[1] + 1, original_target[1] + 1)
            sx = float(final_w) / float(ref_w)
            sy = float(final_h) / float(ref_h)
            target_xy = (
                int(round(original_target[0] * sx)),
                int(round(original_target[1] * sy)),
            )
            log.warning(
                "finalize: target was off-canvas; rescaled assuming ref %sx%s → %s",
                ref_w,
                ref_h,
                target_xy,
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
        log.warning("finalize: intro overlay failed for upload %s: %s", upload_id, exc)

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
    db.add(
        AuditLog(
            actor="admin",
            action="finalize_wizard_video",
            target=f"long_upload:{upload_id}",
            detail=f"final={final_path.name} hole={hole_number} player={player_name}",
        )
    )
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
    payload: dict = Body(default={}),
    db: Session = Depends(get_db),
):
    """Promote the wizard's finalized video to a real VideoClip so it
    shows up on Produced Clips and Broadcast. Idempotent: re-running
    after Save just updates the existing clip's tracer_url + sets
    delivered_at. Returns the clip id + URLs.

    Optional body key `clip_id`: target EXACTLY that produced clip
    (must belong to this upload). Without it the most recent clip on
    the upload is updated — correct for single-swing rows, but on a
    multi-swing upload that's whichever swing produced last, so
    per-swing editors MUST pass the clip id.
    """
    row = db.get(LongVideoUpload, upload_id)
    if not row:
        raise HTTPException(404, "long upload not found")
    saved = dict(row.edit_metrics or {})
    final_url = saved.get("finalized_video_url")
    tracer_url = saved.get("tracer_url")
    if not final_url:
        raise HTTPException(400, "no finalized video — run Step 3 first")

    clip_id = payload.get("clip_id")
    if clip_id is not None:
        clip = (
            db.query(VideoClip)
            .filter(
                VideoClip.id == int(clip_id),
                VideoClip.long_upload_id == upload_id,
            )
            .first()
        )
        if clip is None:
            raise HTTPException(404, "clip not found on this upload")
    else:
        # Re-use any existing wizard clip on this upload (avoids
        # duplicates when the operator Save's twice).
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
        # Targeted commit: the clip already carries its swing's hole —
        # don't overwrite it with the top-level finalized default.
        clip.hole_number = (
            hole_number if clip_id is None
            else (clip.hole_number or hole_number)
        )
        if thumb_url:
            clip.thumbnail_url = thumb_url
        if not clip.captured_at and row.base_captured_at:
            clip.captured_at = row.base_captured_at
    clip.delivered_at = _utcnow_naive()
    row.processing_status = "completed"
    row.processing_completed_at = _utcnow_naive()
    if clip_id is None:
        # Single-swing wizard commit owns the whole upload's counts.
        # A targeted per-swing commit must NOT collapse a multi-swing
        # row's "Produced · 6/6 clips" down to 1/1.
        row.last_n_segments = 1
        row.last_n_succeeded = 1
    db.add(row)
    db.flush()

    db.add(
        AuditLog(
            actor="admin",
            action="commit_wizard_clip",
            target=f"long_upload:{upload_id}",
            detail=f"clip={clip.id} hole={hole_number}",
        )
    )
    db.commit()
    db.refresh(clip)
    return {
        "upload_id": upload_id,
        "clip_id": clip.id,
        "tracer_url": clip.tracer_url,
        "source_url": clip.source_url,
    }


@router.post("/long-uploads/{upload_id}/wizard-produce")
def wizard_produce(
    upload_id: int,
    payload: dict = Body(default={}),
    db: Session = Depends(get_db),
):
    """THE edit wizard's produce. Runs stages 4-8 on the operator's ball
    and impact frame and returns immediately so the wizard can close.

    Body: {ball: [x, y], impact_frame: int, hole_number?: int}

    Stages 1-3 only exist to find those two numbers. An operator who has
    opened the wizard has already found them by eye, so this starts at
    stage 4 — the same `find_flight` and the same renderer produce uses,
    with the operator's answer substituted for the detector's.

    Fire and forget: the job runs on a thread and reports progress to the
    same store the production card polls, so the operator watches it
    there rather than in a modal.
    """
    row = db.get(LongVideoUpload, upload_id)
    if not row:
        raise HTTPException(404, "long upload not found")

    ball = payload.get("ball")
    if (not isinstance(ball, (list, tuple)) or len(ball) < 2):
        raise HTTPException(400, "ball must be [x, y]")
    try:
        bx, by = float(ball[0]), float(ball[1])
    except (TypeError, ValueError):
        raise HTTPException(400, "ball must be two numbers")
    if payload.get("impact_frame") is None:
        raise HTTPException(400, "impact_frame is required")
    try:
        impact_frame = int(payload["impact_frame"])
    except (TypeError, ValueError):
        raise HTTPException(400, "impact_frame must be a whole number")
    if impact_frame < 0:
        raise HTTPException(400, "impact_frame must not be negative")

    # THE LANDING, on the GREEN camera. The frame the ball touches down
    # on and where on that frame it lands. Optional and travel together:
    # either both or neither is useful, so a spot without a frame is
    # ignored rather than half-applied. Absent means "let produce use its
    # own D3_GREEN_SEC and stop the tracer where the ball was last seen".
    landing_frame = payload.get("landing_frame")
    try:
        landing_frame = (
            int(landing_frame) if landing_frame is not None else None)
    except (TypeError, ValueError):
        raise HTTPException(400, "landing_frame must be a whole number")
    if landing_frame is not None and landing_frame < 0:
        raise HTTPException(400, "landing_frame must not be negative")

    landing_spot = payload.get("landing_spot")
    if landing_spot is not None:
        if (not isinstance(landing_spot, (list, tuple))
                or len(landing_spot) < 2):
            raise HTTPException(400, "landing_spot must be [x, y]")
        try:
            landing_spot = [float(landing_spot[0]), float(landing_spot[1])]
        except (TypeError, ValueError):
            raise HTTPException(400, "landing_spot must be two numbers")
    if landing_spot is not None and landing_frame is None:
        log.info("wizard produce: upload=%s sent a landing spot with no "
                 "landing frame — ignoring it", upload_id)
        landing_spot = None

    hole_number = payload.get("hole_number")
    try:
        hole_number = int(hole_number) if hole_number is not None else None
    except (TypeError, ValueError):
        hole_number = None

    # Remember what the operator chose, so re-opening the wizard shows the
    # ball where they put it rather than back at the detector's guess.
    try:
        saved = dict(row.edit_metrics or {})
        saved["wizard_ball"] = [bx, by]
        saved["wizard_impact_frame"] = impact_frame
        if landing_frame is not None:
            saved["landing_frame"] = landing_frame
        if landing_spot is not None:
            saved["landing_spot"] = {"x": landing_spot[0],
                                     "y": landing_spot[1]}
        row.edit_metrics = saved
        db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        log.warning("wizard produce: could not save the ball for %s: %s",
                    upload_id, exc)

    # CLAIM THE ROW BEFORE RETURNING. The production card greys on
    # processing_status, and the wizard closes the moment this responds --
    # so if the status were left to the worker thread, the operator would
    # watch the wizard vanish and the card sit there looking untouched.
    # Set here, synchronously, exactly as the queue worker does.
    try:
        row.processing_status = "processing"
        row.processing_started_at = _utcnow_naive()
        row.processing_completed_at = None
        row.last_error = None
        db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        log.warning("wizard produce: could not claim upload %s: %s",
                    upload_id, exc)

    _debugx_set("produce", upload_id, stage="Queued", done=0, total=0,
                running=True, error=None)
    threading.Thread(
        target=run_wizard_produce_job,
        args=(upload_id, (bx, by), impact_frame, hole_number,
              landing_frame, landing_spot),
        daemon=True,
        name=f"wizard-produce-{upload_id}",
    ).start()
    log.info(
        "wizard produce: upload=%s queued with ball=(%.0f,%.0f) impact=f%d",
        upload_id, bx, by, impact_frame,
    )
    return {"ok": True, "started": True, "upload_id": upload_id,
            "ball": [bx, by], "impact_frame": impact_frame,
            "landing_frame": landing_frame, "landing_spot": landing_spot}


@router.delete("/long-uploads/{upload_id}")
def delete_long_upload(upload_id: int, db: Session = Depends(get_db)):
    """Delete a stored long upload + its source file(s) from disk.
    Per-swing VideoClips produced from this upload are kept (they have
    their own VideoClip rows). Returns the freed disk bytes for
    confirmation.

    When this upload is linked to a CameraEvent (Pi-sourced capture),
    delete the event row too — the raw files belong to the pair and
    leaving the event around would point at unlinked files.
    """
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
    if row.camera_event_id is not None:
        event = db.get(CameraEvent, row.camera_event_id)
        if event is not None:
            db.delete(event)
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
    src_path = _local_tee(row)
    if not src_path.exists():
        raise HTTPException(404, f"tee source file missing on disk: {row.tee_filename}")

    green_path: Path | None = None
    if row.green_filename:
        gp = _local_green(row)
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
            if w.get("peak_time_sec") is not None
            else None
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
            cut_entry.update(
                {
                    "ok": None,
                    "url": None,
                    "green_url": None,
                    "green_ok": None,
                    "skipped": True,
                }
            )
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
            upload_id,
            idx + 1,
            n_windows_total,
            start_sec,
            end_sec,
            "ok" if tee_ok else "fail",
            "ok" if green_ok else ("fail" if green_ok is False else "—"),
            time.monotonic() - t_cut_start,
        )

        cut_entry.update(
            {
                "ok": bool(tee_ok),
                "url": (
                    f"{settings.app_base_url}/uploads/clips/{TESTCUTS_SUBDIR}/{tee_name}"
                    if tee_ok
                    else None
                ),
                "green_url": green_url,
                "green_ok": green_ok,
            }
        )
        cuts.append(cut_entry)

    log.info(
        "long-upload test-cut: upload=%s detector=%s windows=%d cut_clips=%s cuts_ok=%d total=%.1fs",
        upload_id,
        detector,
        len(windows),
        cut_clips,
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
    src_path = _local_tee(row)
    if not src_path.exists():
        raise HTTPException(404, f"tee source file missing on disk: {row.tee_filename}")

    green_src_path: Path | None = None
    if row.green_filename:
        candidate = _local_green(row)
        if not candidate.exists():
            raise HTTPException(
                404,
                f"green source file missing on disk: {row.green_filename}",
            )
        green_src_path = candidate

    if end_sec <= start_sec:
        raise HTTPException(400, "end_sec must be > start_sec")

    seg_list = [
        {
            "hole_number": int(hole_number),
            "start_sec": float(start_sec),
            "end_sec": float(end_sec),
        }
    ]
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
    frame_debug_dir: Path | None = None,
    frame_debug_prefix: str = "tracerdbg",
    bg_algo: str = "mog2",
    impact_frame_hint_override: int | None = None,
    frame_label_offset: int = 0,
    ball_rest_hint: tuple | None = None,
    heat_start_frame: int | None = None,
    heat_end_frame: int | None = None,
    render_video: bool = True,
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
    # An explicit caller-supplied impact (the operator's pick in the Edit
    # wizard) beats the audio-derived one — the hint controls where the
    # tracer starts keeping track points, and the operator's choice must
    # not be second-guessed by the audio peak.
    if impact_frame_hint_override is not None:
        impact_hint = max(0, int(impact_frame_hint_override))
    log.info(
        "tracer: audio impact hint for %s — frame=%s (ratio=%.1f, ok=%s)",
        clip_path.name,
        impact_hint,
        audio_impact.get("ratio") or 0.0,
        audio_impact.get("ok"),
    )

    info = render_tracer(
        clip_path,
        traced_path,
        debug_path,
        impact_frame_hint=impact_hint,
        sensitivity=float(sensitivity),
        frame_debug_dir=frame_debug_dir,
        frame_debug_prefix=frame_debug_prefix,
        bg_algo=bg_algo,
        frame_label_offset=frame_label_offset,
        ball_rest_hint=ball_rest_hint,
        heat_start_frame=heat_start_frame,
        heat_end_frame=heat_end_frame,
        render_video=render_video,
    )
    info["audio_impact"] = audio_impact
    info["sensitivity"] = float(sensitivity)
    debug_url = (
        f"{settings.app_base_url}/uploads/clips/{debug_name}"
        if debug_path.exists()
        else None
    )
    if not info.get("ok"):
        traced_path.unlink(missing_ok=True)
        return None, info, None, debug_url
    if not render_video or not traced_path.exists():
        return None, info, None, debug_url
    # OpenCV writes mp4v; re-encode to H.264 + faststart for browser playback.
    compressed = compress_for_email(traced_path)
    if not compressed:
        log.warning(
            "tracer: compress_for_email returned False for %s — file likely still mp4v, browser playback may fail",
            traced_path.name,
        )
    if not traced_path.exists() or traced_path.stat().st_size == 0:
        return (
            None,
            {"ok": False, "error": "post-encode produced empty file"},
            None,
            debug_url,
        )
    # Probe the final file so we can spot mp4v-leftover or VFR-timestamp
    # bugs in the logs (symptoms: "still photo with claimed duration").
    probe = probe_video_info(traced_path)
    log.info(
        "tracer: traced output %s  codec=%s  fps=%s  nb_frames=%s  duration=%ss  size=%dB",
        traced_path.name,
        probe.get("codec"),
        probe.get("fps"),
        probe.get("nb_frames"),
        probe.get("duration"),
        traced_path.stat().st_size,
    )
    # Cache-bust the served URLs with the file mtime. Each retry rewrites
    # the same filename, and some browsers refuse to re-init the <video>
    # decoder when src stays identical — they sit on the old decoded
    # state and the new bytes never get rendered. Appending a version
    # query string forces the element to treat it as a new resource.
    traced_mtime = int(traced_path.stat().st_mtime)
    debug_mtime = (
        int(debug_path.stat().st_mtime) if debug_path.exists() else traced_mtime
    )
    debug_url = (
        f"{settings.app_base_url}/uploads/clips/{debug_name}?v={debug_mtime}"
        if debug_path.exists()
        else None
    )
    return (
        f"{settings.app_base_url}/uploads/clips/{traced_name}?v={traced_mtime}",
        info,
        traced_path,
        debug_url,
    )


def _mog2_dot_pool(cv_info: dict) -> list:
    """Deduped, ghost-filtered MOG2 dot pool from a classical tracer run.

    Three signals merged:
     - per-frame surviving candidate detections (the yellow rings on the
       editor cards) — exact frame, exact position, motion-verified;
     - timed transient dots (median-of-hits frame — jittery, but
       survives when per-frame gates lose the ball);
     - the accepted chain (can lock onto club motion; never alone).

    Ghost-trail filter: MOG2 keeps firing a VACATED spot for several
    frames after the ball leaves it, so every early flight position
    re-appears as a stationary dot for ~5-10 frames — a ladder a trail
    follower could walk back down. A flying ball never occupies the
    same spot 3+ times in a short window: dots with >=2 prior
    appearances within 7px in the previous 12 frames are dropped.
    (Also thins pool-ripple / foliage repeats.)"""
    _pool_by_key: dict = {}
    for rec in (
        list(cv_info.get("candidates") or [])
        + list(cv_info.get("timed_points") or [])
        + list(cv_info.get("track") or [])
    ):
        if (
            rec.get("frame") is None
            or rec.get("x") is None or rec.get("y") is None
        ):
            continue
        k = (
            int(rec["frame"]),
            int(round(float(rec["x"]) / 4.0)),
            int(round(float(rec["y"]) / 4.0)),
        )
        _pool_by_key.setdefault(
            k, {"frame": int(rec["frame"]), "x": float(rec["x"]), "y": float(rec["y"])},
        )
    pool = sorted(_pool_by_key.values(), key=lambda r: r["frame"])
    _ghost_free: list = []
    _win: list = []
    for c in pool:
        f = int(c["frame"])
        while _win and f - _win[0][0] > 12:
            _win.pop(0)
        n_prior = sum(
            1 for (pf, px_, py_) in _win
            if pf < f
            and ((float(c["x"]) - px_) ** 2
                 + (float(c["y"]) - py_) ** 2) ** 0.5 < 7.0
        )
        if n_prior < 2:
            _ghost_free.append(c)
        _win.append((f, float(c["x"]), float(c["y"])))
    return _ghost_free


# MOG2 layer-in analysis window: nothing past this many seconds after
# impact accumulates heat or extends the arc (operator: "no more than 4
# seconds"). Shared by produce's layer and the wizard's ai_mog2 engine.
MOG2_LAYER_POST_IMPACT_SEC = 4.0


def _flight_from_rest_lock(
    pool: list, rest_xy: tuple, imp: int, fps: float,
    f_cap: int | None,
) -> tuple[list, dict]:
    """Operator-designed launch lock: search a variance cone opening UP
    from the RESTING BALL position in the frames after impact for a
    3-dot sequence with (a) strictly increasing frames, (b) each dot
    higher than the last, (c) all three on a straight line from the
    rest position (within tolerance), and (d) spacing that does not
    grow much dot-to-dot (a launched ball decelerates — its marks get
    slightly closer; club debris accelerating away does not). Once
    locked, follow the flight with the velocity-aware trail follower.

    Anchored at rest + impact ONLY — works even when the AI picks are
    few, clustered, or wrong. Returns (chain, lock_info); chain entries
    are {frame, found, x, y, source:'mog2'} in increasing frames."""
    import numpy as _np

    rx, ry = float(rest_xy[0]), float(rest_xy[1])
    lim_f = imp + int(round(1.5 * fps))
    cone = []
    for c in pool:
        f = int(c["frame"])
        if not (imp - 2 <= f <= lim_f):
            continue
        rise = ry - float(c["y"])
        if rise < 8:
            continue  # not above the ball
        if ((float(c["x"]) - rx) ** 2
                + (float(c["y"]) - ry) ** 2) ** 0.5 < 25.0:
            continue  # vacated-ball ghost at the rest spot
        if abs(float(c["x"]) - rx) > rise + 40.0:
            continue  # outside the cone (~45° half-angle + base slack)
        cone.append(c)
    cone.sort(key=lambda r: int(r["frame"]))
    info = {"locked": False, "n_cone": len(cone), "seed_frames": None}
    if len(cone) < 3:
        return [], info

    def _dline(p, bx, by):
        # distance of p from the line rest -> (bx, by)
        _l = ((bx - rx) ** 2 + (by - ry) ** 2) ** 0.5
        if _l < 1.0:
            return 1e9
        return abs(
            (bx - rx) * (ry - float(p["y"]))
            - (rx - float(p["x"])) * (by - ry)
        ) / _l

    def _d(a, b):
        return ((float(a["x"]) - float(b["x"])) ** 2
                + (float(a["y"]) - float(b["y"])) ** 2) ** 0.5

    # Collect up to 8 candidate seeds (one per starting dot) and follow
    # each — the LONGEST surviving chain wins. A seed built on a walker
    # or club fragment dies within a few dots; the real launch follows
    # for dozens of frames.
    seeds = []
    for d1 in cone:
        f1 = int(d1["frame"])
        found = None
        for d2 in cone:
            f2 = int(d2["frame"])
            if not (f1 < f2 <= f1 + 6):
                continue
            if float(d2["y"]) > float(d1["y"]) - 3.0:
                continue  # not moving upward
            for d3 in cone:
                f3 = int(d3["frame"])
                if not (f2 < f3 <= f2 + 6):
                    continue
                if float(d3["y"]) > float(d2["y"]) - 3.0:
                    continue
                # Straight line from REST through the sequence.
                if (
                    _dline(d1, float(d3["x"]), float(d3["y"])) > 25.0
                    or _dline(d2, float(d3["x"]), float(d3["y"])) > 25.0
                ):
                    continue
                # Per-frame spacing must not grow much.
                v12 = _d(d1, d2) / max(1, f2 - f1)
                v23 = _d(d2, d3) / max(1, f3 - f2)
                if v23 > v12 * 1.5 + 5.0:
                    continue
                found = (d1, d2, d3)
                break
            if found:
                break
        if found:
            seeds.append(found)
            if len(seeds) >= 8:
                break
    info["n_seeds"] = len(seeds)
    if not seeds:
        return [], info

    def _physics_break(chain):
        """First index whose SEQUENCE breaks ballistic physics —
        operator's rule: a ball never turns hard, and never suddenly
        slows or speeds up. Per-dot gates pass individually-plausible
        club/body dots; the sequence gives them away. Returns the index
        of the offending point, or None."""
        # Two-segment averaged velocities: raw consecutive triples are
        # jitter-dominated (±3px marks on 10px steps swing single-gap
        # ratios past 2x) and truncated GOOD chains. Averaging halves
        # the jitter; thresholds catch only egregious veers — branch
        # exploration handles the subtle ones by outcome.
        for i in range(4, len(chain)):
            a, m1, m2, c = (
                chain[i - 3], chain[i - 2], chain[i - 1], chain[i],
            )
            g1 = max(1, int(m2["frame"]) - int(a["frame"]))
            g2 = max(1, int(c["frame"]) - int(m1["frame"]))
            if g1 >= 12 or g2 >= 12:
                continue  # re-acquisition jump — not a kinematic read
            v1x = (m2["x"] - a["x"]) / g1
            v1y = (m2["y"] - a["y"]) / g1
            v2x = (c["x"] - m1["x"]) / g2
            v2y = (c["y"] - m1["y"]) / g2
            s1 = (v1x * v1x + v1y * v1y) ** 0.5
            s2 = (v2x * v2x + v2y * v2y) ** 0.5
            if s1 < 5.0 or s2 < 5.0:
                continue  # apex regime — direction/speed undefined
            _cos = (v1x * v2x + v1y * v2y) / (s1 * s2)
            if _cos < 0.35:  # > ~70 degrees — flight never does this
                return i
            r = s2 / s1
            if r > 2.5 or r < 0.4:
                return i
        return None

    def _follow(seed):
        d1, d2, d3 = seed
        chain = [
            {
                "frame": int(d["frame"]), "found": True,
                "x": float(d["x"]), "y": float(d["y"]), "source": "mog2",
            }
            for d in seed
        ]
        def _key(c):
            return (int(c["frame"]), int(round(float(c["x"]))),
                    int(round(float(c["y"]))))

        def _extend(chain, blacklist):
            """Trail-follow from the chain's current tail, skipping
            blacklisted dots."""
            _kf = [float(imp)] + [float(c["frame"]) for c in chain]
            _kx = [rx] + [c["x"] for c in chain]
            _ky = [ry] + [c["y"] for c in chain]

            def _fit():
                _deg_x = 2 if len(_kf) >= 4 else 1
                return (
                    _np.polyfit(_kf, _kx, _deg_x),
                    _np.polyfit(_kf, _ky, 2),
                )

            _cx, _cy = _fit()
            prev_f = int(chain[-1]["frame"])
            prev_x, prev_y = float(chain[-1]["x"]), float(chain[-1]["y"])
            lock_f = int(chain[min(2, len(chain) - 1)]["frame"])
            for c in pool:
                f = int(c["frame"])
                if f <= prev_f:
                    continue
                if f_cap is not None and f > f_cap:
                    break
                if f - prev_f > 45:
                    break  # trail went quiet
                if _key(c) in blacklist:
                    continue
                # Vacated-ball ghost at the rest spot.
                if ((float(c["x"]) - rx) ** 2
                        + (float(c["y"]) - ry) ** 2) ** 0.5 < 25.0:
                    continue
                gap = f - prev_f
                _vx = float(_np.polyval(_np.polyder(_cx), prev_f))
                _vy = float(_np.polyval(_np.polyder(_cy), prev_f))
                vel = float(_np.hypot(_vx, _vy))
                step = ((float(c["x"]) - prev_x) ** 2
                        + (float(c["y"]) - prev_y) ** 2) ** 0.5
                if step > 30.0 + gap * max(9.0, 1.6 * vel):
                    continue
                # Direction consistency: flight never u-turns.
                if vel > 6.0 and step > 1.0:
                    _dp = ((float(c["x"]) - prev_x) * _vx
                           + (float(c["y"]) - prev_y) * _vy)
                    if _dp < 0.2 * step * vel:
                        continue
                # Minimum progress (ghost drift defence).
                if f - lock_f <= int(round(1.5 * fps)) and step < 3.5 * gap:
                    continue
                if vel > 8.0 and step < 0.3 * vel * gap:
                    continue
                # Tight predictive corridor keyed to the gap.
                pred_x = float(_np.polyval(_cx, f))
                pred_y = float(_np.polyval(_cy, f))
                tol = 20.0 + 6.0 * gap
                d = ((float(c["x"]) - pred_x) ** 2
                     + (float(c["y"]) - pred_y) ** 2) ** 0.5
                if d > tol:
                    continue
                chain.append({
                    "frame": f, "found": True,
                    "x": float(c["x"]), "y": float(c["y"]),
                    "source": "mog2",
                })
                _kf.append(float(f))
                _kx.append(float(c["x"]))
                _ky.append(float(c["y"]))
                _cx, _cy = _fit()
                prev_f, prev_x, prev_y = f, float(c["x"]), float(c["y"])
            return chain

        # BRANCH-AND-PRUNE (operator-designed): follow with a physics
        # watchdog — on a sequence break (hard turn / sudden speed
        # change) roll back, blacklist the offender, re-follow. Then
        # BRANCH EXPLORATION: a straight club-shaft ladder at similar
        # speed is physically plausible dot-by-dot, so ALSO try the
        # path without each risky join (re-acquisition after a faint
        # gap) — the true flight outlives the ladder, so the furthest,
        # cleanest-fitting chain wins.
        seed_chain = list(chain)

        def _run(blacklist):
            ch = list(seed_chain)
            for _attempt in range(6):
                ch = _extend(ch, blacklist)
                bad_i = _physics_break(ch)
                if bad_i is None:
                    break
                blacklist.add(_key(ch[bad_i]))
                ch = ch[:bad_i]
                if len(ch) < 3:
                    break
            return ch

        def _score(ch):
            if len(ch) < 4:
                return (len(ch), 0.0)
            _fs = _np.array([float(c["frame"]) for c in ch])
            _ys = _np.array([float(c["y"]) for c in ch])
            try:
                _res = float(_np.mean(_np.abs(
                    _ys - _np.polyval(_np.polyfit(_fs, _ys, 2), _fs),
                )))
            except Exception:  # noqa: BLE001
                _res = 1e9
            return (len(ch), -_res)

        base = _run(set())
        best = base
        # Risky joins: dots accepted after a >=2 frame silence — where
        # a wrong branch (club ladder) can be grabbed. Try without each
        # branch (the join dot AND everything after it).
        _joins = [
            i for i in range(3, len(base))
            if int(base[i]["frame"]) - int(base[i - 1]["frame"]) >= 2
        ][:3]
        for ji in _joins:
            alt = _run({_key(c) for c in base[ji:]})
            if _score(alt) > _score(best):
                best = alt
        return best

    best_chain, best_seed = [], None
    for sd in seeds:
        ch = _follow(sd)
        if len(ch) > len(best_chain):
            best_chain, best_seed = ch, sd
    if len(best_chain) < 5:
        return [], info
    chain = best_chain
    info["locked"] = True
    info["seed_frames"] = [int(d["frame"]) for d in best_seed]
    info["chain_len"] = len(chain)
    return chain, info


def _mog2_layer_for_ai_track(
    clip_path: Path, pipe: dict, render_extended: bool = True,
) -> dict | None:
    """Post-produce MOG2 layer over a successful AI tracer run.

    Runs the classical MOG2 heatmap-arc pass on the SAME produced cut,
    checks whether its detected chain corresponds with the AI tracer's
    points (>=2 AI points within +/-3 frames and 25px of a chain point —
    the wizard-hybrid coincidence rule), and when it does, extends the
    ball-path arc with chain points BEYOND the last AI point. AI points
    are never moved or overridden — MOG2 only adds tail.

    Also writes an overlay JPEG on the raw-motion heat composite showing
    both point sets (yellow = AI picks, white = MOG2 chain, red = MOG2
    points actually added to the arc) for the button under the produced
    video.

    Returns {stats, overlay_name, merged, url, path} — merged/url/path
    only set when the arc was actually extended and the re-render
    succeeded. Never raises; returns None when MOG2 found nothing usable
    and no overlay could be drawn."""
    import cv2  # type: ignore

    ai_all = list(pipe.get("ball_track_frames") or [])
    ai_pts = [
        rec for rec in ai_all
        if rec.get("found") and rec.get("x") is not None and rec.get("y") is not None
    ]
    _imp = (pipe.get("impact_refined") or {}).get("impact_frame")
    _rest = pipe.get("ball_rest_xy_native")

    # Analysis window: impact-3 frames (margin for the strike) through
    # impact + 4s. Heat outside it never accumulates — the golfer
    # walking across the swing path BEFORE the shot, or wandering off
    # AFTER the ball lands, leaves no residue in the flight corridor.
    _fps = probe_fps(clip_path) or 30.0

    # ANCHOR CHECK (pixel-verify, no API): snap the rest position to
    # the bright-blob centroid and pin impact to the exact frame the
    # ball DEPARTS the rest patch. Both anchors feed the rest-lock's
    # cone, the heat window, and the 4s cap — a verified correction
    # here tightens everything downstream. Unverified = keep the
    # originals (never let a failed check break a working trace).
    anchor_check: dict | None = None
    if (
        _rest and len(_rest) == 2 and _imp is not None
        and pipe.get("ball_rest_source") != "pose_wrist_fallback"
        and not pipe.get("anchors_preverified")
    ):
        try:
            from ..services.ai_tracer import (
                verify_rest_and_impact,
                verify_rest_and_impact_ai,
            )

            anchor_check = verify_rest_and_impact_ai(
                clip_path,
                (float(_rest[0]), float(_rest[1])),
                int(_imp), _fps,
                debug_dir=CLIPS_DIR,
                debug_prefix=f"anchorai-{clip_path.stem}",
            )
            if anchor_check.get("api_error") or not anchor_check.get(
                "available",
            ):
                _ai_fail = anchor_check.get("reason")
                anchor_check = verify_rest_and_impact(
                    clip_path,
                    (float(_rest[0]), float(_rest[1])),
                    int(_imp), _fps,
                    debug_dir=CLIPS_DIR,
                    debug_prefix=f"anchorchk-{clip_path.stem}",
                )
                anchor_check["ai_fallback_reason"] = _ai_fail
            if anchor_check.get("verified"):
                _rest = (
                    float(anchor_check["rest_xy"][0]),
                    float(anchor_check["rest_xy"][1]),
                )
                _imp = int(anchor_check["impact_frame"])
                log.info(
                    "mog2 layer: anchors verified — rest snapped %spx, "
                    "impact -> f%d (%+d)",
                    anchor_check.get("snap_px"), _imp,
                    anchor_check.get("impact_delta") or 0,
                )
            else:
                log.info(
                    "mog2 layer: anchor check inconclusive (%s) — "
                    "keeping original anchors",
                    anchor_check.get("reason"),
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("mog2 layer: anchor check failed: %s", exc)

    _pfx = f"mog2layer-{clip_path.stem}"
    _cv_url, cv_info, _cv_traced, _cv_dbg = _run_tracer(
        clip_path,
        frame_debug_dir=CLIPS_DIR,
        frame_debug_prefix=_pfx,
        impact_frame_hint_override=(int(_imp) if _imp is not None else None),
        ball_rest_hint=(
            (float(_rest[0]), float(_rest[1]))
            if _rest and len(_rest) == 2 else None
        ),
        heat_start_frame=(max(0, int(_imp) - 3) if _imp is not None else None),
        heat_end_frame=(
            int(_imp) + int(round(MOG2_LAYER_POST_IMPACT_SEC * _fps))
            if _imp is not None else None
        ),
        # Points only — the AI render (possibly extended below) is the
        # deliverable; skipping the classical video write saves a full
        # read+write pass.
        render_video=False,
    )
    import numpy as _np

    cv_info = cv_info or {}
    pool = _mog2_dot_pool(cv_info)
    launch_pts = [
        {"frame": int(pt["frame"]), "x": float(pt["x"]), "y": float(pt["y"])}
        for pt in (pipe.get("launch_points") or [])
        if pt.get("frame") is not None and int(pt["frame"]) >= 0
    ]
    if launch_pts:
        # Adaptive-square tracker points: per-frame, pixel-exact,
        # already ball-verified — they join the dot pool AND go into
        # the arc directly (below); the lock/corridor phases only fill
        # what the tracker didn't cover.
        pool = sorted(pool + list(launch_pts), key=lambda rec: rec["frame"])

    def _near(a, b, df=3, dpx=25.0):
        return (
            abs(int(a["frame"]) - int(b["frame"])) <= df
            and ((float(a["x"]) - float(b["x"])) ** 2
                 + (float(a["y"]) - float(b["y"])) ** 2) ** 0.5 <= dpx
        )

    n_matched = sum(
        1 for ap in ai_pts if any(_near(ap, cp) for cp in pool)
    )

    _f_cap = (
        int(_imp) + int(round(MOG2_LAYER_POST_IMPACT_SEC * _fps))
        if _imp is not None else None
    )
    added_launch: list[dict] = []
    added_mid: list[dict] = []
    added_descent: list[dict] = []
    descent_debug: dict | None = None

    # PRIMARY: rest-lock flight chain (operator-designed). A cone opens
    # UP from the resting ball; a 3-dot straight-line, frames-increasing
    # sequence locks the launch; the trail follower takes it from there.
    # Anchored at rest + impact only, so it works even when the AI picks
    # are few, clustered at the apex, or plain wrong.
    lock_chain: list = []
    lock_info: dict = {"locked": False}
    if _rest and len(_rest) == 2 and _imp is not None:
        try:
            lock_chain, lock_info = _flight_from_rest_lock(
                pool, (float(_rest[0]), float(_rest[1])),
                int(_imp), _fps, _f_cap,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("mog2 layer: rest-lock failed: %s", exc)

    if lock_chain:
        _ai_ff = {int(p["frame"]) for p in ai_pts}
        _chain_add = [
            c for c in lock_chain if int(c["frame"]) not in _ai_ff
        ]
        if ai_pts:
            _ff = min(int(p["frame"]) for p in ai_pts)
            _lf = max(int(p["frame"]) for p in ai_pts)
            added_launch = [c for c in _chain_add if c["frame"] < _ff]
            added_mid = [
                c for c in _chain_add if _ff <= c["frame"] <= _lf
            ]
            added_descent = [c for c in _chain_add if c["frame"] > _lf]
        else:
            added_launch = _chain_add
        descent_debug = {
            "seen": len(lock_chain), "step_rej": 0, "corr_rej": 0,
            "samples": [], "rescue": False,
            "stopped": (
                f"rest-lock chain (seed f"
                f"{lock_info.get('seed_frames')})"
            ),
        }
    # FALLBACK (no lock): AI-anchored corridor phases. Even a SINGLE AI
    # pick anchors the launch corridor (impact position → that pick),
    # and the dots it fills then anchor the extrapolation fit.
    elif ai_pts:
        ai_sorted = sorted(ai_pts, key=lambda r: int(r["frame"]))
        first_ai, last_ai = ai_sorted[0], ai_sorted[-1]
        first_f, last_f = int(first_ai["frame"]), int(last_ai["frame"])
        _ai_found_frames = {int(p["frame"]) for p in ai_sorted}

        # ── Launch fill: impact position → first AI pick ─────────────
        # The ball demonstrably travelled from the strike to the first
        # AI point, so MOG2 dots inside that corridor with frames
        # between impact and the first pick are the flight. Corridor =
        # distance to the straight segment; frames must progress UP the
        # corridor (projection t increasing with frame).
        p0 = (
            (float(_rest[0]), float(_rest[1]))
            if _rest and len(_rest) == 2 else None
        )
        if p0 is not None and _imp is not None and first_f - int(_imp) > 2:
            _ax, _ay = float(first_ai["x"]), float(first_ai["y"])
            _vx, _vy = _ax - p0[0], _ay - p0[1]
            _seg2 = _vx * _vx + _vy * _vy
            _best_by_f: dict = {}  # frame -> (dist, t, c): best dot per frame
            if _seg2 > 1.0:
                for c in pool:
                    f = int(c["frame"])
                    if not (int(_imp) - 2 <= f < first_f):
                        continue
                    if f in _ai_found_frames:
                        continue
                    t = ((c["x"] - p0[0]) * _vx + (c["y"] - p0[1]) * _vy) / _seg2
                    if not (-0.05 <= t <= 1.05):
                        continue
                    _px, _py = p0[0] + t * _vx, p0[1] + t * _vy
                    d = ((c["x"] - _px) ** 2 + (c["y"] - _py) ** 2) ** 0.5
                    if d > 45.0:
                        continue
                    if f not in _best_by_f or d < _best_by_f[f][0]:
                        _best_by_f[f] = (d, t, c)
            _cands = [
                (f, t, c) for f, (d, t, c) in sorted(_best_by_f.items())
            ]
            _prev_t = -0.05
            for f, t, c in _cands:
                if t < _prev_t - 0.08:
                    continue  # steps back down the corridor — not flight
                _prev_t = max(_prev_t, t)
                added_launch.append({
                    "frame": f, "found": True,
                    "x": c["x"], "y": c["y"], "source": "mog2",
                })

        # ── Extension beyond the last AI pick ────────────────────────
        # Extrapolate the arc (quadratic x/y in frame) and accept pool
        # dots near the prediction, frames strictly and gradually
        # increasing. The fit anchors on the AI picks PLUS the launch-
        # fill dots — so even one AI pick, backed by a filled launch
        # corridor, extrapolates confidently. The corridor widens the
        # further out we go; a >45-frame silence ends it.
        _known = sorted(
            ai_sorted + added_launch, key=lambda r: int(r["frame"]),
        )
        if len(_known) >= 3:
            _kf0 = [float(p["frame"]) for p in _known]
            _kx0 = [float(p["x"]) for p in _known]
            _ky0 = [float(p["y"]) for p in _known]

            def _run_extension(corr_base, corr_slope, dbg):
                """One trail-following pass beyond the last AI pick.
                Gates per dot: velocity-aware STEP from the previous
                accepted dot (continuity — keeps off-chain dots from
                dragging the fit), then an arc corridor around the
                running fit (refit on every acceptance). Every
                rejection is counted, and the first few are sampled,
                so a 0-add result explains itself in mog2_stats."""
                picked: list[dict] = []
                kf, kx, ky = list(_kf0), list(_kx0), list(_ky0)

                def _fit():
                    _deg_x = 2 if len(kf) >= 4 else 1
                    return (
                        _np.polyfit(kf, kx, _deg_x),
                        _np.polyfit(kf, ky, 2),
                    )

                cx, cy = _fit()
                pf = last_f
                px = float(_known[-1]["x"])
                py = float(_known[-1]["y"])
                for c in pool:
                    f = int(c["frame"])
                    if f <= pf:
                        continue
                    if _f_cap is not None and f > _f_cap:
                        dbg["stopped"] = f"4s cap at f{f}"
                        break
                    if f - pf > 45:
                        dbg["stopped"] = (
                            f"gap: no accepted dot between f{pf} and f{f}"
                        )
                        break
                    dbg["seen"] += 1
                    gap = f - pf
                    _vx = float(_np.polyval(_np.polyder(cx), pf))
                    _vy = float(_np.polyval(_np.polyder(cy), pf))
                    vel = float(_np.hypot(_vx, _vy))
                    step = ((c["x"] - px) ** 2 + (c["y"] - py) ** 2) ** 0.5
                    # Direction consistency — flight never u-turns; a
                    # hop fighting the fit velocity is club/body motion.
                    if vel > 6.0 and step > 1.0:
                        _dp = ((c["x"] - px) * _vx + (c["y"] - py) * _vy)
                        if _dp < 0.2 * step * vel:
                            dbg["step_rej"] += 1
                            continue
                    allow = 30.0 + gap * max(9.0, 1.6 * vel)
                    pred_x = float(_np.polyval(cx, f))
                    pred_y = float(_np.polyval(cy, f))
                    tol = corr_base + corr_slope * (f - last_f)
                    d = ((c["x"] - pred_x) ** 2
                         + (c["y"] - pred_y) ** 2) ** 0.5
                    if step > allow or d > tol:
                        if step > allow:
                            dbg["step_rej"] += 1
                        else:
                            dbg["corr_rej"] += 1
                        if len(dbg["samples"]) < 5:
                            dbg["samples"].append({
                                "f": f,
                                "step": round(step),
                                "allow": round(allow),
                                "arc_d": round(d),
                                "tol": round(tol),
                            })
                        continue
                    picked.append({
                        "frame": f, "found": True,
                        "x": c["x"], "y": c["y"], "source": "mog2",
                    })
                    kf.append(float(f))
                    kx.append(float(c["x"]))
                    ky.append(float(c["y"]))
                    cx, cy = _fit()
                    pf, px, py = f, float(c["x"]), float(c["y"])
                if dbg.get("stopped") is None:
                    dbg["stopped"] = "end of pool"
                # Physics prune (same rule as the rest-lock follower):
                # truncate at the first sequence break — hard turn or
                # sudden speed change is club/body, not ball.
                _ctx = [
                    {"frame": last_f, "x": float(_known[-1]["x"]),
                     "y": float(_known[-1]["y"])}
                ] + picked
                for i in range(4, len(_ctx)):
                    a, m1, m2, c2 = (
                        _ctx[i - 3], _ctx[i - 2], _ctx[i - 1], _ctx[i],
                    )
                    g1 = max(1, int(m2["frame"]) - int(a["frame"]))
                    g2 = max(1, int(c2["frame"]) - int(m1["frame"]))
                    if g1 >= 12 or g2 >= 12:
                        continue
                    v1x = (m2["x"] - a["x"]) / g1
                    v1y = (m2["y"] - a["y"]) / g1
                    v2x = (c2["x"] - m1["x"]) / g2
                    v2y = (c2["y"] - m1["y"]) / g2
                    s1 = (v1x * v1x + v1y * v1y) ** 0.5
                    s2 = (v2x * v2x + v2y * v2y) ** 0.5
                    if s1 < 5.0 or s2 < 5.0:
                        continue
                    _cosv = (v1x * v2x + v1y * v2y) / (s1 * s2)
                    if _cosv < 0.35 or s2 / s1 > 2.5 or s2 / s1 < 0.4:
                        dbg["stopped"] = (
                            f"physics prune at f{int(c2['frame'])}"
                        )
                        picked = _ctx[1:i]
                        break
                return picked

            _desc_dbg = {
                "seen": 0, "step_rej": 0, "corr_rej": 0,
                "stopped": None, "samples": [], "rescue": False,
            }
            added_descent = _run_extension(60.0, 0.6, _desc_dbg)
            if not added_descent:
                # Rescue pass: the corridor near an apex-clustered fit
                # is the most likely mis-shape — double it; the step
                # gate still enforces chain continuity so noise stays
                # out.
                _desc_dbg2 = {
                    "seen": 0, "step_rej": 0, "corr_rej": 0,
                    "stopped": None, "samples": [], "rescue": True,
                }
                added_descent = _run_extension(120.0, 1.2, _desc_dbg2)
                if added_descent:
                    _desc_dbg = _desc_dbg2
            descent_debug = _desc_dbg

    # DIRECT adds: launch-tracker points are verified flight — into the
    # arc unconditionally (never dependent on the lock re-finding them).
    _ai_ff_all = {int(pp["frame"]) for pp in ai_pts}
    added_track: list[dict] = []
    _lt_frames: set = set()
    for pt in launch_pts:
        f = int(pt["frame"])
        if f in _ai_ff_all or f in _lt_frames:
            continue
        if _f_cap is not None and f > _f_cap:
            continue
        _lt_frames.add(f)
        added_track.append({
            "frame": f, "found": True,
            "x": pt["x"], "y": pt["y"], "source": "launch",
        })

    added = sorted(
        added_track + [
            a for a in (added_launch + added_mid + added_descent)
            if int(a["frame"]) not in _lt_frames
        ],
        key=lambda r: int(r["frame"]),
    )

    # ARC COMPLETION (operator's rule): the mapped arc tells us its
    # travel direction — continuation lives in a RECTANGLE from the
    # arc's end toward that side of the frame: up to the top (ascent /
    # apex, possibly exiting), down to the ground line (descent). Zoom
    # into that region (noise outside is ignored) and find the chain
    # of pool dots that rises along the trajectory with frame numbers
    # INCREASING, tops out, then descends with frames still increasing
    # — the completion pattern. Long frame gaps are allowed (the ball
    # hides behind trees); spatial steps scale with the gap.
    n_arc = 0
    arc_region = None
    try:
        _seenf: set = set()
        _fit_pts: list[dict] = []
        for p in sorted(
            [
                {"frame": int(pp["frame"]), "x": float(pp["x"]),
                 "y": float(pp["y"])}
                for pp in ai_pts
            ] + [
                {"frame": int(a["frame"]), "x": float(a["x"]),
                 "y": float(a["y"])}
                for a in added
            ],
            key=lambda p2: p2["frame"],
        ):
            if p["frame"] in _seenf:
                continue
            _seenf.add(p["frame"])
            _fit_pts.append(p)
        _vw = _vh = None
        if len(_fit_pts) >= 5:
            _capv = cv2.VideoCapture(str(clip_path))
            _vw = int(_capv.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            _vh = int(_capv.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            _capv.release()
        if len(_fit_pts) >= 5 and _vw and _vh:
            _r2 = max(10, int(round(0.015 * _vh)))
            _lastp = _fit_pts[-1]
            _lastf = int(_lastp["frame"])
            _dx_tot = _fit_pts[-1]["x"] - _fit_pts[0]["x"]
            _dirx = 1.0 if _dx_tot >= 0 else -1.0
            if abs(_dx_tot) < 3 * _r2:
                x_lo = _lastp["x"] - 20 * _r2
                x_hi = _lastp["x"] + 20 * _r2
            elif _dirx > 0:
                x_lo, x_hi = _lastp["x"] - 10 * _r2, float(_vw)
            else:
                x_lo, x_hi = 0.0, _lastp["x"] + 10 * _r2
            _gy = float(_rest[1]) if (_rest and len(_rest) == 2) else (
                _lastp["y"]
            )
            arc_region = [
                int(max(0, x_lo)), 0,
                int(min(_vw, x_hi)),
                int(min(_vh, _gy + 2 * _r2)),
            ]
            cands = sorted(
                [
                    d for d in pool
                    if int(d["frame"]) > _lastf
                    and int(d["frame"]) not in _seenf
                    and (_f_cap is None or int(d["frame"]) <= _f_cap)
                    and arc_region[0] <= float(d["x"]) <= arc_region[2]
                    and arc_region[1] <= float(d["y"]) <= arc_region[3]
                ],
                key=lambda d: int(d["frame"]),
            )
            # Was the mapped arc already descending at its end?
            _tail = _fit_pts[-3:]
            _phase0 = (
                "down"
                if len(_tail) == 3 and _tail[2]["y"] > _tail[0]["y"] + 4
                else "up"
            )
            # Ascent steepness of the mapped arc: vertical rise per
            # horizontal drift. The descent must be roughly the
            # INVERSE of the ascent — a chain that tops out and then
            # runs sideways (a physics-defying right-angle turn) must
            # never be used.
            _rise0 = max(
                1.0,
                float(_fit_pts[0]["y"])
                - min(float(q["y"]) for q in _fit_pts),
            )
            _drift0 = max(
                1.0,
                abs(float(_fit_pts[-1]["x"]) - float(_fit_pts[0]["x"])),
            )
            _A = max(0.35, min(6.0, _rise0 / _drift0))

            def _accept(prev, phase, d):
                fno, pf = int(d["frame"]), int(prev["frame"])
                if fno <= pf:
                    return None
                gap = fno - pf
                if gap > 45:  # ~1.5s hidden is the most we bridge
                    return None
                dx = float(d["x"]) - float(prev["x"])
                dy = float(d["y"]) - float(prev["y"])
                if (dx * _dirx) < -3 * _r2:
                    return None  # backward vs travel direction
                if (dx * dx + dy * dy) ** 0.5 > (6 + 2.5 * gap) * _r2:
                    return None  # too far for the frame gap
                if phase == "up":
                    if dy <= 2 * _r2:
                        return "up"
                    phase = "down"  # first drop = past the apex
                if dy >= -2 * _r2:
                    # INVERSE-SLOPE rule: a falling ball's horizontal
                    # drift per unit of drop is bounded by (a relaxed
                    # multiple of) the ascent's inverse slope. A
                    # near-horizontal step after the apex is the club /
                    # a bird / sparkle — never the ball.
                    if abs(dx) > (max(dy, 0.0) / _A) * 2.5 + 2 * _r2:
                        return None
                    return "down"
                return None  # descending chain must not jump back up

            def _greedy(first_i):
                prev = {"frame": _lastf, "x": _lastp["x"],
                        "y": _lastp["y"]}
                phase = _phase0
                first_ph = _accept(prev, phase, cands[first_i])
                if first_ph is None:
                    return []
                ch = [cands[first_i]]
                phase = first_ph
                prev = cands[first_i]
                for d in cands[first_i + 1:]:
                    nph = _accept(prev, phase, d)
                    if nph is None:
                        continue
                    ch.append(d)
                    phase = nph
                    prev = d
                return ch

            best: list = []
            for si in range(len(cands)):
                if len(cands) - si <= len(best):
                    break
                ch = _greedy(si)
                if len(ch) > len(best):
                    best = ch
            # 2/3-descent cap (operator's rule): trim chained dots that
            # sit lower than apex + 2/3 of the ascent height — the
            # renderer stops the line there anyway. EXCEPTION: an apex
            # hugging the top edge means the flight went off-screen —
            # the true apex is unknown, so keep the whole descent (the
            # renderer skips its cap in that case too).
            if best and _rest and len(_rest) == 2:
                _ys_all = [q["y"] for q in _fit_pts] + [
                    float(d["y"]) for d in best
                ]
                _apx_y = min(_ys_all)
                _asc_h = float(_rest[1]) - _apx_y
                if _asc_h > 60 and _apx_y > 0.04 * _vh:
                    _cap_y = _apx_y + (2.0 / 3.0) * _asc_h
                    best = [d for d in best if float(d["y"]) <= _cap_y]
            if len(best) >= 3:
                for d in best:
                    added.append({
                        "frame": int(d["frame"]), "found": True,
                        "x": int(round(float(d["x"]))),
                        "y": int(round(float(d["y"]))),
                        "source": "arc",
                    })
                    n_arc += 1
                added.sort(key=lambda rec: int(rec["frame"]))
                log.info(
                    "mog2 layer: arc completion chained %d region dot(s) "
                    "f%d-%d (region %s, dir %s)",
                    n_arc, int(best[0]["frame"]), int(best[-1]["frame"]),
                    arc_region, "right" if _dirx > 0 else "left",
                )
    except Exception as exc:  # noqa: BLE001
        log.warning("mog2 layer: arc completion failed: %s", exc)

    stats = {
        "n_ai": len(ai_pts),
        "n_cv": len(pool),
        "n_matched": n_matched,
        "n_added": len(added),
        # How many AI launch points the caller actually handed us — the
        # number that makes "plotted but ignored" visible.
        "n_launch_in": len(launch_pts),
        "n_added_track": len(added_track),
        "n_added_launch": len(added_launch),
        "n_added_mid": len(added_mid),
        "n_added_descent": len(added_descent),
        "n_arc_completed": n_arc,
        "arc_region": arc_region,
        "corresponds": bool(n_matched >= 2),
        "lock": lock_info,
        "anchor_check": anchor_check,
        "descent_debug": descent_debug,
    }
    try:
        import json as _json
        log.info(
            "produce: mog2 layer %s stats: %s",
            clip_path.name, _json.dumps(stats),
        )
    except Exception:  # noqa: BLE001
        pass

    # Overlay: raw-motion heat + both point sets, for the produced-video
    # button. Drawn even when nothing was added — seeing WHY (no chain,
    # no correspondence) is the point of the debug view.
    overlay_name = None
    _raw = cv_info.get("raw_motion_image")
    if _raw and (CLIPS_DIR / _raw).exists():
        try:
            img = cv2.imread(str(CLIPS_DIR / _raw))
            if img is not None:
                # Rest-lock visual: magenta line from the resting ball
                # through the 3-dot seed that locked the launch.
                if lock_info.get("locked") and _rest and lock_chain:
                    _s3 = lock_chain[min(2, len(lock_chain) - 1)]
                    cv2.line(
                        img,
                        (int(float(_rest[0])), int(float(_rest[1]))),
                        (int(_s3["x"]), int(_s3["y"])),
                        (255, 0, 255), 2, cv2.LINE_AA,
                    )
                if arc_region:
                    cv2.rectangle(
                        img, (arc_region[0], arc_region[1]),
                        (arc_region[2], arc_region[3]),
                        (0, 0, 255), 2,
                    )
                    cv2.putText(
                        img, "arc-completion search region",
                        (arc_region[0] + 6, arc_region[1] + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
                        cv2.LINE_AA,
                    )
                for cp in pool:
                    cv2.circle(
                        img, (int(cp["x"]), int(cp["y"])), 7,
                        (255, 255, 255), 2, cv2.LINE_AA,
                    )
                for ap in added:
                    cv2.circle(
                        img, (int(ap["x"]), int(ap["y"])), 6,
                        (0, 0, 255), -1, cv2.LINE_AA,
                    )
                    cv2.circle(
                        img, (int(ap["x"]), int(ap["y"])), 8,
                        (255, 255, 255), 1, cv2.LINE_AA,
                    )
                for ap in ai_pts:
                    cv2.circle(
                        img, (int(ap["x"]), int(ap["y"])), 5,
                        (0, 255, 255), -1, cv2.LINE_AA,
                    )
                _lbl = (
                    f"MOG2 vs AI - yellow=AI picks ({len(ai_pts)}), "
                    f"white=MOG2 dots ({len(pool)}), "
                    f"red=added to arc ({len(added)}"
                    + (
                        f", {len(added_track)} from launch tracker"
                        if added_track else ""
                    )
                    + f"), matched={n_matched}"
                    + (
                        f", LOCKED @ f{lock_info.get('seed_frames')}"
                        f" (magenta line, chain "
                        f"{lock_info.get('chain_len')})"
                        if lock_info.get("locked")
                        else ", no rest-lock"
                    )
                )
                cv2.putText(img, _lbl, (12, 56), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, _lbl, (12, 56), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62, (255, 255, 255), 1, cv2.LINE_AA)
                overlay_name = f"{clip_path.stem}_mog2_overlay.jpg"
                cv2.imwrite(
                    str(CLIPS_DIR / overlay_name), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("mog2 layer: overlay draw failed: %s", exc)
            overlay_name = None

    out = {
        "stats": stats, "overlay_name": overlay_name,
        "merged": None, "url": None, "path": None,
        # Corrected anchors (clip-relative) when the pixel check
        # verified them — callers persist these for the wizard.
        "anchors": (
            {
                "rest_xy": [float(_rest[0]), float(_rest[1])],
                "impact_frame": int(_imp),
            }
            if anchor_check and anchor_check.get("verified") else None
        ),
        # Timed transient dots (cut-relative frames, native coords) +
        # the raw-motion heat image — persisted per swing so the Edit
        # wizard's click-to-plot view works straight from produce,
        # without an in-session classical re-render.
        "timed_points": list(cv_info.get("timed_points") or []),
        # The denser per-frame candidate pool — click-to-plot reveals
        # these as extra clickable dots when zoomed in.
        "candidates": list(cv_info.get("candidates") or []),
        "raw_motion_image": cv_info.get("raw_motion_image"),
    }
    if not added:
        log.info(
            "produce: mog2 layer for %s — no extension (%s)",
            clip_path.name, stats,
        )
        return out if (overlay_name or pool) else None

    # FINAL physics pass over the assembled sequence (operator's rule,
    # applied at merge): stray dots — foliage flicker etc. — hijack the
    # render fit, which then dives to the junk and outlier-rejects the
    # REAL descent. A ballistic flight is quadratic in frame, so fit
    # x(f)/y(f) robustly and iteratively peel the worst outlier until
    # everything left sits on one arc. (Local triple checks could not
    # untangle strays interleaved with real descent dots.)
    if len(added) >= 8:
        _pts = sorted(added, key=lambda r: int(r["frame"]))
        _n_removed = 0
        for _ in range(20):
            _fs = _np.array([float(p2["frame"]) for p2 in _pts])
            _xs = _np.array([float(p2["x"]) for p2 in _pts])
            _ys = _np.array([float(p2["y"]) for p2 in _pts])
            _cx2 = _np.polyfit(_fs, _xs, 2 if len(_pts) >= 8 else 1)
            _cy2 = _np.polyfit(_fs, _ys, 2)
            _res = _np.hypot(
                _xs - _np.polyval(_cx2, _fs),
                _ys - _np.polyval(_cy2, _fs),
            )
            _w = int(_res.argmax())
            if float(_res[_w]) <= 60.0:
                break
            _pts.pop(_w)
            _n_removed += 1
            if len(_pts) < 6:
                break
        if _n_removed:
            log.info(
                "mog2 layer: merge arc-fit prune removed %d stray "
                "point(s)", _n_removed,
            )
            added = _pts
            stats["n_added"] = len(added)
            stats["n_pruned_tail"] = _n_removed

    # An added point beats an AI "not found" placeholder on the same
    # frame; found AI picks always win (added never lands on one).
    _added_frames = {int(a["frame"]) for a in added}
    _base = [
        rec for rec in ai_all
        if rec.get("found") or int(rec.get("frame") or -1) not in _added_frames
    ]
    merged = sorted(
        _base + added, key=lambda rec: int(rec.get("frame") or 0),
    )
    if not render_extended:
        # Caller renders its own video from `merged` (the wizard's
        # windowed render) — skip the cut-clip render here.
        out["merged"] = merged
        return out
    try:
        ext_path = CLIPS_DIR / f"{clip_path.stem}_ai_mog2_tracer.mp4"
        rr = render_tracer_video(
            clip_path, ext_path,
            ball_rest_xy_native=(
                (float(_rest[0]), float(_rest[1]))
                if _rest and len(_rest) == 2 else None
            ),
            impact_frame_idx=int(_imp) if _imp is not None else 0,
            track_frames=merged,
            # The departure walk watched the ball sit here and leave, and
            # that same walk is where _imp came from. Start point settled.
            rest_verified=bool(
                (anchor_check and anchor_check.get("verified"))
                or pipe.get("anchors_preverified")
            ),
        )
        if rr.get("ok") and ext_path.exists():
            compress_for_email(ext_path)
            if ext_path.exists() and ext_path.stat().st_size > 0:
                out["merged"] = merged
                out["path"] = ext_path
                out["url"] = (
                    f"{settings.app_base_url}/uploads/clips/{ext_path.name}"
                    f"?v={int(ext_path.stat().st_mtime)}"
                )
                log.info(
                    "produce: mog2 layer EXTENDED %s by %d points (%s)",
                    clip_path.name, len(added), stats,
                )
        else:
            log.warning(
                "produce: mog2 extended re-render failed for %s (%s) — "
                "keeping AI-only tracer", clip_path.name, rr.get("error"),
            )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "produce: mog2 extended re-render crashed for %s (%s)",
            clip_path.name, exc,
        )
    return out


def _trace_segment(
    clip_path: Path,
    ball_at_rest_override=None,
    verified_rest_xy=None,
    verified_impact_frame=None,
    launch_points=None,
):
    """Draw the ball-flight tracer on a cut segment for PRODUCTION.

    Uses the AI tracer by default (settings.tracer_engine == 'ai' and a key
    is set); falls back to the classical CV tracer when AI is unconfigured or
    fails. Same return shape as _run_tracer:
    (tracer_url, info, traced_path, debug_url).

    ball_at_rest_override (x, y in native pixels) is a FALLBACK anchor for
    where the tracer line starts: the AI still tries to see the resting ball
    first, but if it can't (backlit scene / dark ground) the line is anchored
    here instead of picking up mid-flight. Used to pass the golfer's hands-at-
    impact position from the pose detector.

    verified_rest_xy / verified_impact_frame are PIXEL-VERIFIED anchors
    (the departure pin: the classifier found the ball, and MOG2 watched
    it leave). When set they become hard overrides in the AI pipeline —
    which then SKIPS its audio impact, vision impact, refine, address
    and handedness calls entirely (frame indices are CUT-relative)."""
    # DEBUG3 IN PRODUCTION. The blob-and-track pipeline measures the same
    # three things the AI tracer does -- where the ball was, when it was
    # struck, and where it went -- and on the swings compared it did so more
    # accurately and without an API call. Its answers are fed in through the
    # existing pin: verified_rest_xy / verified_impact_frame are already
    # HARD overrides, and launch_points already flow through tracer_info
    # into edit_metrics, which is what click-to-plot reads.
    #
    # DEBUG3 WINS over the departure pin. This was written to defer to any
    # existing anchor, which meant a re-produce -- where seg["ball_rest_xy"]
    # is already filled in from the departure detector -- skipped Debug3
    # entirely and re-rendered an identical clip. Deferring was wrong on the
    # merits too: the departure detector is the component that put the ball
    # on a golfer's trainer, 120px from the real one. Operator marks are
    # applied later, through the Edit wizard and click-to-plot, so nothing
    # hand-made is overridden here.
    _d3 = None
    if settings.debug3_tracer:
        try:
            from ..services import debug3 as _d3mod
            from ..services import pose_swing as _ps

            _hd = _ft = None
            _imp = None
            if _ps.available():
                _c3 = list(_ps.detect_swings_from_pose(clip_path) or [])
                if _c3:
                    # The cut holds one swing; take the strongest burst.
                    _best = max(
                        _c3, key=lambda z: float(z.get("ratio") or 0.0))
                    _hd = _best.get("impact_head_xy")
                    _ft = _best.get("impact_feet_xy")
                    _pt = float(_best.get("peak_time_sec") or 0.0)
                    _imp = int(round(_pt * (probe_fps(clip_path) or 30.0)))
            _d3 = _d3mod.find_flight(
                clip_path, probe_fps(clip_path) or 30.0,
                impact_frame=_imp, head_xy=_hd, feet_xy=_ft,
            )
            if _d3.get("ok"):
                _was = (verified_rest_xy, verified_impact_frame)
                verified_rest_xy = tuple(_d3["ball"])
                verified_impact_frame = int(_d3["launch_frame"])
                launch_points = [
                    {"frame": int(z["frame"]), "x": float(z["x"]),
                     "y": float(z["y"])}
                    for z in _d3.get("points") or []
                ]
                log.info(
                    "debug3 tracer: %s (replaced rest=%s impact=%s)",
                    _d3.get("reason"), _was[0], _was[1],
                )
            # Same per-phase clock the Debug3 panel shows, logged for the
            # production path — this runs once per swing on every produce,
            # so it is the number that actually decides how long an
            # operator waits.
            _d3t = (_d3.get("debug") or {}).get("timing") or {}
            if _d3t:
                log.info(
                    "debug3 tracer timing: %s",
                    ", ".join(f"{k} {v}s" for k, v in _d3t.items()),
                )
            else:
                log.info("debug3 tracer: no flight (%s) -- falling back",
                         _d3.get("reason"))
        except Exception as exc:  # noqa: BLE001
            log.warning("debug3 tracer failed, falling back: %s", exc)
            _d3 = None

    # RENDER STRAIGHT FROM DEBUG3. Pinning the ball and impact was not
    # enough: the log for upload 495 shows the pin applied --
    #
    #   ai_tracer: ball-at-rest overridden -- (526, 640)
    #   ai_tracer: ball track SKIPPED (disabled by config)
    #
    # -- and then, because ai_ball_track_enabled is off, produce fell
    # through to the CLASSICAL tracer, which found its own ball at
    # (590, 611), its own impact frame 79, and its own 29 points. Debug3's 7
    # arrived as a minority in the merge (n_cv: 29, n_added: 7) and the
    # rendered line was the classical answer. That is the clip that kept
    # coming back unchanged.
    #
    # So when Debug3 has a flight, nothing else runs. Same renderer, same
    # three inputs, same output as the Debug3 panel shows -- because it is
    # now literally the same call.
    if _d3 and _d3.get("ok"):
        try:
            _o = clip_path.with_name(clip_path.stem + "_d3_tracer.mp4")
            _pts3 = [
                {"frame": int(z["frame"]), "found": True,
                 "x": float(z["x"]), "y": float(z["y"])}
                for z in (_d3.get("points") or [])
            ]
            _rv3 = render_tracer_video(
                clip_path, _o,
                (float(_d3["ball"][0]), float(_d3["ball"][1])),
                int(_d3["launch_frame"]), _pts3,
                rest_verified=True,
            )
            if _rv3.get("ok") and _o.exists() and _o.stat().st_size > 0:
                transcode_for_web(_o)
                compress_for_email(_o)
                # A BACKGROUND for click-to-plot. The editor draws its dots
                # over tracer_raw_motion_url, which the MOG2 layer used to
                # write -- and that layer no longer runs, so the modal opened
                # with the point list populated and nothing to draw on. Use
                # the frame at launch: for placing a ball by eye it is a
                # better canvas than a motion composite anyway, because it
                # shows the scene rather than an accumulation of it.
                _bgn = f"{clip_path.stem}_d3_plotbg.jpg"
                try:
                    import cv2 as _cv3

                    _cap3 = _cv3.VideoCapture(str(clip_path))
                    _cap3.set(_cv3.CAP_PROP_POS_FRAMES,
                              max(0, int(_d3["launch_frame"])))
                    _okf, _fr3 = _cap3.read()
                    _cap3.release()
                    if _okf and _fr3 is not None:
                        _cv3.imwrite(
                            str(CLIPS_DIR / _bgn), _fr3,
                            [int(_cv3.IMWRITE_JPEG_QUALITY), 88],
                        )
                    else:
                        _bgn = None
                except Exception as exc:  # noqa: BLE001
                    log.warning("debug3: plot background failed: %s", exc)
                    _bgn = None
                _info3 = {
                    "ok": True,
                    "engine": "debug3",
                    "ball_track_frames": _pts3,
                    # timed_points is what the card's click-to-plot button
                    # is gated on (hasEvidence in ProducedTile: an overlay
                    # url OR timed points). The MOG2 layer used to fill it,
                    # and that layer no longer runs -- so without this the
                    # button silently disappears on every clip Debug3
                    # produces. Same points, the shape the wizard expects.
                    "timed_points": [
                        {"frame": int(z["frame"]), "x": float(z["x"]),
                         "y": float(z["y"])}
                        for z in (_d3.get("points") or [])
                    ],
                    "n_points": len(_pts3),
                    "impact_frame": int(_d3["launch_frame"]),
                    "ball": {"x": float(_d3["ball"][0]),
                             "y": float(_d3["ball"][1]),
                             "source": _d3.get("ball_source")},
                    "render_info": _rv3,
                    "debug3": {"reason": _d3.get("reason")},
                }
                if _bgn:
                    _info3["raw_motion_image"] = _bgn
                # The clickable candidate layer. _persist_swing_track maps
                # these into edit_metrics.cand_points, which is what the
                # editor draws in amber and what the operator clicks to add
                # a point. Without it the modal is read-only.
                _cand3 = _d3.get("candidates") or []
                if _cand3:
                    _info3["candidates"] = _cand3
                _u3 = (
                    f"{settings.app_base_url}/uploads/clips/{_o.name}"
                    f"?v={int(_o.stat().st_mtime)}"
                )
                log.info(
                    "debug3 tracer: rendered %s directly (%d points) -- "
                    "classical tracer not run", _o.name, len(_pts3),
                )
                return _u3, _info3, _o, None
            log.warning(
                "debug3 tracer: direct render failed (%s) -- falling through",
                _rv3.get("error"),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("debug3 tracer: direct render crashed: %s", exc)

    # A departure-pinned swing with the AI ball track disabled makes
    # ZERO api calls in the pipeline — it may run without a key.
    _pinned_zero_ai = (
        verified_impact_frame is not None
        and bool(verified_rest_xy)
        and not settings.ai_ball_track_enabled
    )
    use_ai = settings.tracer_engine == "ai" and (
        bool(os.environ.get("ANTHROPIC_API_KEY")) or _pinned_zero_ai
    )
    if use_ai:
        try:
            prefix = f"{clip_path.stem}_ai-{secrets.token_hex(3)}"
            _fallback = None
            if ball_at_rest_override and len(ball_at_rest_override) == 2:
                _fallback = (
                    float(ball_at_rest_override[0]),
                    float(ball_at_rest_override[1]),
                )
            _rest_ovr = None
            if verified_rest_xy and len(verified_rest_xy) == 2:
                _rest_ovr = (
                    float(verified_rest_xy[0]), float(verified_rest_xy[1]),
                )
            r = run_full_ai_tracer_pipeline(
                clip_path, CLIPS_DIR, prefix,
                rest_anchor_fallback=_fallback,
                ball_at_rest_override=_rest_ovr,
                impact_frame_override=(
                    int(verified_impact_frame)
                    if verified_impact_frame is not None else None
                ),
                ball_track_enabled=settings.ai_ball_track_enabled,
                # With the track disabled there are no AI points to
                # draw — the layer's extended render is the deliverable,
                # so skip the pipeline's full-clip render pass entirely.
                render_video=settings.ai_ball_track_enabled,
            )
            # Departure-pinned anchors are already pixel-verified on the
            # FULL source — the layer must trust them, not re-check on
            # the re-encoded cut (a weaker look that once re-snapped a
            # dim ball onto the golfer's shoe 66px away).
            if verified_impact_frame is not None and _rest_ovr is not None:
                r["anchors_preverified"] = True
            if launch_points:
                # Adaptive-square tracker points (CUT-relative frames)
                # join the layer's dot pool — per-frame, pixel-exact.
                r["launch_points"] = launch_points
            # What the render decided about the line's START. Captured from
            # the PIPELINE's render as well as the layer's, because the
            # layer returns early — before it renders anything — whenever
            # it adds no points to the arc, and then the pipeline's render
            # is the deliverable. Capturing only the layer's meant the
            # panel showed nothing on those runs, which reads as "the debug
            # view stopped working".
            _render_info = {
                _k: (r.get("tracer_video_info") or {}).get(_k)
                for _k in (
                    "rest_anchor_relocated", "rest_anchor_dropped",
                    "rest_anchor_synthesized",
                )
                if (r.get("tracer_video_info") or {}).get(_k)
            }
            tvp = r.get("tracer_video_path")
            if r.get("ok"):
                # The pipeline's own render is optional now: with the AI
                # ball track disabled it produces anchors + no video,
                # and the MOG2 layer's extended render (launch tracker +
                # rest-lock points) is the deliverable.
                url = None
                p = None
                if tvp and Path(tvp).exists():
                    _p0 = Path(tvp)
                    compress_for_email(_p0)
                    if _p0.exists() and _p0.stat().st_size > 0:
                        p = _p0
                        url = (
                            f"{settings.app_base_url}/uploads/clips/"
                            f"{_p0.name}?v={int(_p0.stat().st_mtime)}"
                        )
                if True:
                    info = {
                        "ok": True, "engine": "ai",
                        "n_points": len(r.get("ball_track_frames") or []),
                        # Carry the per-frame ball track (segment-relative
                        # frame indices) so the caller can persist it into
                        # edit_metrics.swings — that lets the Edit wizard
                        # hydrate the found points instead of re-running the
                        # AI tracer on the whole multi-swing source.
                        "ball_track_frames": r.get("ball_track_frames") or [],
                        "impact_frame": (
                            (r.get("impact_refined") or {}).get("impact_frame")
                        ),
                        # Everything else the Edit wizard hydrates from —
                        # persisted per swing so Step 1/2 open pre-populated.
                        "address_frame": (
                            (r.get("address") or {}).get("address_frame")
                        ),
                        "handedness": (
                            (r.get("handedness") or {}).get("handedness")
                        ),
                        "ball_rest_xy": r.get("ball_rest_xy_native"),
                        "ball_rest_source": r.get("ball_rest_source"),
                    }
                    # MOG2 layer-in: after the AI tracer lands, run the
                    # classical MOG2 arc pass on the same cut; when its
                    # chain corresponds with the AI points, extend the
                    # arc with the chain's tail and swap in the
                    # re-rendered video. Best-effort — any failure keeps
                    # the AI-only tracer.
                    try:
                        _layer = _mog2_layer_for_ai_track(clip_path, r)
                    except Exception as exc:  # noqa: BLE001
                        log.warning(
                            "produce: mog2 layer crashed for %s: %s",
                            clip_path.name, exc,
                        )
                        _layer = None
                    if _layer:
                        info["mog2"] = _layer.get("stats")
                        for _k in (
                            "rest_anchor_relocated", "rest_anchor_dropped",
                            "rest_anchor_synthesized",
                        ):
                            if (_layer.get("stats") or {}).get(_k):
                                _render_info[_k] = _layer["stats"][_k]
                        if _layer.get("overlay_name"):
                            info["mog2_overlay_image"] = _layer["overlay_name"]
                        # Timed heat dots + candidate pool + raw-motion
                        # image from the layer's classical pass —
                        # persisted per swing so the wizard's
                        # click-to-plot opens pre-loaded.
                        if _layer.get("timed_points"):
                            info["timed_points"] = _layer["timed_points"]
                        if _layer.get("candidates"):
                            info["candidates"] = _layer["candidates"]
                        if _layer.get("raw_motion_image"):
                            info["raw_motion_image"] = _layer["raw_motion_image"]
                        # Pixel-verified anchors beat the vision
                        # estimates — persist the corrected rest ball +
                        # departure-frame impact for the wizard.
                        _anch = _layer.get("anchors")
                        if _anch:
                            info["impact_frame"] = _anch["impact_frame"]
                            info["ball_rest_xy"] = tuple(_anch["rest_xy"])
                            info["ball_rest_source"] = "pixel_verified"
                        _ac = (_layer.get("stats") or {}).get("anchor_check")
                        if _ac:
                            info["anchor_check"] = _ac
                        if _layer.get("merged") and _layer.get("url"):
                            info["ball_track_frames"] = _layer["merged"]
                            info["n_points"] = len(_layer["merged"])
                            url = _layer["url"]
                            p = _layer["path"]
                    if url:
                        info["ok"] = True
                        info["n_points"] = len(
                            info.get("ball_track_frames") or [],
                        )
                        # Present even when empty, so the panel can state
                        # "start kept on the anchor" rather than render
                        # nothing and look broken.
                        info["render_info"] = _render_info
                        log.info(
                            "produce: AI tracer ok for %s", clip_path.name,
                        )
                        return url, info, p, None
                    log.warning(
                        "produce: pipeline ok but nothing rendered for %s "
                        "(track disabled and layer added no points) — "
                        "classical fallback", clip_path.name,
                    )
            else:
                log.warning(
                    "produce: AI tracer failed for %s (%s) — classical "
                    "fallback", clip_path.name, r.get("error"),
                )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "produce: AI tracer crashed for %s (%s) — classical fallback",
                clip_path.name, exc,
            )
    url, info, traced, dbg = _run_tracer(clip_path)
    if isinstance(info, dict) and not info.get("engine"):
        info["engine"] = "classical"
    # Modern-render the fallback. The classical engine's own writer is
    # the LEGACY red-dot style — every produced clip should get the same
    # look as the AI+layer path, so re-draw its track (plus any launch-
    # tracker points, which the layer would normally contribute) through
    # render_tracer_video. Keeps the legacy file if the re-render fails.
    try:
        _rt = [
            {
                "frame": int(p["frame"]), "found": True,
                "x": int(round(p["x"])), "y": int(round(p["y"])),
            }
            for p in ((info or {}).get("track") or [])
        ]
        _seen_f = {rec["frame"] for rec in _rt}
        for lp in (launch_points or []):
            _f = int(lp["frame"])
            if _f not in _seen_f:
                _rt.append({
                    "frame": _f, "found": True,
                    "x": int(round(float(lp["x"]))),
                    "y": int(round(float(lp["y"]))),
                })
                _seen_f.add(_f)
        _rt.sort(key=lambda rec: rec["frame"])
        if len(_rt) >= 3:
            _rest_m = None
            if verified_rest_xy and len(verified_rest_xy) == 2:
                _rest_m = (
                    float(verified_rest_xy[0]), float(verified_rest_xy[1]),
                )
            elif ball_at_rest_override and len(ball_at_rest_override) == 2:
                _rest_m = (
                    float(ball_at_rest_override[0]),
                    float(ball_at_rest_override[1]),
                )
            _imp_m = (
                int(verified_impact_frame)
                if verified_impact_frame is not None
                else int(_rt[0]["frame"])
            )
            _mod_name = f"{clip_path.stem}_classical_modern.mp4"
            _mod_path = CLIPS_DIR / _mod_name
            _ri = render_tracer_video(
                clip_path, _mod_path,
                ball_rest_xy_native=_rest_m,
                impact_frame_idx=_imp_m,
                track_frames=_rt,
                # _rest_m is the departure-verified rest, or a ball the
                # operator placed themselves. Neither is a guess.
                rest_verified=bool(
                    verified_rest_xy or ball_at_rest_override,
                ),
            )
            if (
                _ri.get("ok")
                and _mod_path.exists()
                and _mod_path.stat().st_size > 0
            ):
                compress_for_email(_mod_path)
                if _mod_path.exists() and _mod_path.stat().st_size > 0:
                    url = (
                        f"{settings.app_base_url}/uploads/clips/{_mod_name}"
                        f"?v={int(_mod_path.stat().st_mtime)}"
                    )
                    traced = _mod_path
                    if isinstance(info, dict):
                        info["track"] = [
                            {"frame": rec["frame"], "x": rec["x"], "y": rec["y"]}
                            for rec in _rt
                        ]
            else:
                log.warning(
                    "produce: classical modern re-render not ok (%s) — "
                    "keeping legacy render", _ri.get("error"),
                )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "produce: classical modern re-render failed (%s) — keeping "
            "legacy render", exc,
        )
    # GUARANTEE the Debug3 points reach the Edit wizard. _persist_swing_track
    # reads tracer_info["ball_track_frames"] (or ["track"]), and that is what
    # click-to-plot renders. The AI path feeds launch_points into the MOG2
    # layer's dot pool and the classical path merges them into the render
    # track, so they usually arrive -- but "usually" is not good enough for
    # the points the tracer was actually drawn from. Merge them in here,
    # where there is exactly one exit and the answer is known.
    if _d3 and _d3.get("ok") and isinstance(info, dict):
        _have = info.get("ball_track_frames")
        _key = "ball_track_frames" if _have is not None else "track"
        _cur = list(info.get(_key) or [])
        _seen = {int(z["frame"]) for z in _cur if z.get("frame") is not None}
        for _q in _d3.get("points") or []:
            if int(_q["frame"]) in _seen:
                continue
            _cur.append({
                "frame": int(_q["frame"]), "found": True,
                "x": float(_q["x"]), "y": float(_q["y"]),
                "source": "debug3",
            })
            _seen.add(int(_q["frame"]))
        _cur.sort(key=lambda z: int(z.get("frame") or 0))
        info[_key] = _cur
        info["debug3"] = {
            "ball": _d3.get("ball"), "launch_frame": _d3.get("launch_frame"),
            "n_points": len(_d3.get("points") or []),
            "reason": _d3.get("reason"),
        }
    return url, info, traced, dbg


# ── Produce debug (dev course-testing tool) ────────────────────────────
# Per-upload diagnostic report. Runs ALONGSIDE a normal produce: for every
# detected swing it shows the classical-CV tracer's motion heatmap + whether
# it found the ball, and runs the AI tracer on the same swing so the two can
# be compared. Report lives in memory (transient, dev-only); the images and
# tracer videos it references are real files under uploads/clips.
_produce_debug_state: dict[int, dict] = {}
_produce_debug_lock = threading.Lock()

# The produce worker's recorded "work" — pose debug data, every heat-check
# verdict (with evidence image names), every practice classification, the
# decision trail and the kept swing list — published ONCE per produce run.
# The debug job renders its report FROM THIS RECORD instead of re-running
# the pipeline, so Debug and Produce can never diverge: debug literally
# shows the run that cut the clips. In-memory (same process, dev tool);
# decisions are also persisted to edit_metrics.produce_decisions.
_produce_work_state: dict[int, dict] = {}
_produce_work_lock = threading.Lock()


def _produce_debug_report(upload_id: int) -> dict:
    with _produce_debug_lock:
        rep = _produce_debug_state.get(upload_id)
        return dict(rep, swings=list(rep["swings"])) if rep else {
            "running": False, "total": 0, "done": 0, "swings": [], "finished_at": None,
        }


def _ball_debug_and_ref(src_path: Path, tee_fps: float, upload_id: int, roi):
    """Run the ball-departure detector and produce the images the debug UI
    needs: a clean reference frame (to draw the ROI on), a diagnostic overlay
    (ROI box + every white candidate the detector saw, so you can tell WHY it
    found nothing), and a ringed screenshot per departure. Returns
    (ball_dict, ref_frame_url, frame_w, frame_h)."""
    import cv2  # type: ignore

    ball_debug: dict = {}
    try:
        detect_swings_from_ball(src_path, fps=tee_fps, roi=roi, debug=ball_debug)
    except Exception as exc:  # noqa: BLE001
        ball_debug = {"reason": f"crashed: {exc}"}

    # Motion-gate: a real impact departure coincides with a swing motion
    # burst; the club settling at address does not. Using the FULL-RATE trace
    # (no 10 Hz aliasing), keep only departures with a motion spike within
    # motion_gate_sec, and snap the impact time to that local peak.
    motion_gate_sec = 1.5
    pre_gate = ball_debug.get("departures") or []
    try:
        _trace = compute_motion_trace(src_path, fps=tee_fps)
    except Exception:  # noqa: BLE001
        _trace = None
    gated_departures = pre_gate
    if _trace and _trace.get("series"):
        _series = _trace["series"]
        _n = len(_series)
        _dur = float(_trace.get("duration_sec") or (_n - 1)) or 1.0
        _thr = float(_trace.get("threshold") or 0.0)
        _idx = lambda tt: max(0, min(_n - 1, int(tt / _dur * (_n - 1))))
        _tof = lambda ii: (ii / (_n - 1)) * _dur if _n > 1 else 0.0
        gated_departures = []
        for dep in pre_gate:
            t = float(dep.get("t") or 0.0)
            lo, hi = _idx(t - motion_gate_sec), _idx(t + motion_gate_sec)
            window = _series[lo:hi + 1] or [0.0]
            peak_v = max(window)
            if peak_v > _thr:
                pk_i = lo + window.index(peak_v)
                gated_departures.append(
                    dict(dep, t=round(_tof(pk_i), 2), motion=round(peak_v, 3))
                )

    def _url(name: str) -> str | None:
        p = CLIPS_DIR / name
        if not p.exists():
            return None
        return f"{settings.app_base_url}/uploads/clips/{name}?v={int(p.stat().st_mtime)}"

    ref_frame_url = diag_url = None
    frame_w = frame_h = None
    ref = None
    try:
        _c = cv2.VideoCapture(str(src_path))
        _c.set(cv2.CAP_PROP_POS_FRAMES, int(2.0 * tee_fps))
        okr, ref = _c.read()
        _c.release()
    except Exception as exc:  # noqa: BLE001
        log.warning("produce-debug: ref grab failed: %s", exc)
        ref = None

    if ref is not None:
        frame_h, frame_w = ref.shape[:2]
        rname = f"debug-ref-{upload_id}-{secrets.token_hex(4)}.jpg"
        cv2.imwrite(str(CLIPS_DIR / rname), ref, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        ref_frame_url = _url(rname)
        # Diagnostic overlay: ROI box (orange) + every white candidate the
        # detector matched across the clip (green dots).
        diag = ref.copy()
        if roi:
            x0 = int(float(roi.get("x", 0)) * frame_w)
            y0 = int(float(roi.get("y", 0)) * frame_h)
            x1 = int((float(roi.get("x", 0)) + float(roi.get("w", 1))) * frame_w)
            y1 = int((float(roi.get("y", 0)) + float(roi.get("h", 1))) * frame_h)
            cv2.rectangle(diag, (x0, y0), (x1, y1), (0, 140, 255), 3)
        for cx, cy in (ball_debug.get("sample_cands") or []):
            cv2.circle(diag, (int(cx), int(cy)), 5, (0, 255, 0), -1, cv2.LINE_AA)
        dname = f"debug-balldiag-{upload_id}-{secrets.token_hex(4)}.jpg"
        cv2.imwrite(str(CLIPS_DIR / dname), diag, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        diag_url = _url(dname)

    # Ringed screenshot per (motion-gated) departure = confirmed swing.
    if gated_departures:
        try:
            _c = cv2.VideoCapture(str(src_path))
            for bi, dep in enumerate(gated_departures):
                rest = float(dep.get("rest_sec") or 1.0)
                snap_t = max(0.0, float(dep.get("t") or 0.0) - min(max(rest / 2.0, 0.3), 1.5))
                _c.set(cv2.CAP_PROP_POS_FRAMES, int(snap_t * tee_fps))
                okf, fr = _c.read()
                if not okf or fr is None:
                    continue
                x, y = int(dep.get("x") or 0), int(dep.get("y") or 0)
                rad = max(14, int(round(fr.shape[0] * 0.02)))
                cv2.circle(fr, (x, y), rad, (0, 255, 255), 3, cv2.LINE_AA)
                cv2.circle(fr, (x, y), 2, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    fr, f"ball @ {snap_t:.1f}s", (max(0, x - 50), max(22, y - rad - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
                )
                iname = f"debug-ball-{upload_id}-{bi}-{secrets.token_hex(4)}.jpg"
                cv2.imwrite(str(CLIPS_DIR / iname), fr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                dep["image_url"] = _url(iname)
            _c.release()
        except Exception as exc:  # noqa: BLE001
            log.warning("produce-debug: ball screenshot failed: %s", exc)

    ball_dict = {
        "n": len(gated_departures),
        "reason": ball_debug.get("reason"),
        "departures": gated_departures,
        "peaks": [round(float(d.get("t") or 0.0), 2) for d in gated_departures],
        "diag_url": diag_url,
        "n_cand_total": ball_debug.get("n_cand_total"),
        "n_cand_in_roi": ball_debug.get("n_cand_in_roi"),
        "n_tracks": ball_debug.get("n_tracks"),
        "n_rested": ball_debug.get("n_rested"),
        "min_rest_sec": ball_debug.get("min_rest_sec"),
        # v2: how many resting-ball departures the motion gate kept vs saw.
        "n_departures_pre_gate": len(pre_gate),
        "motion_gated": _trace is not None and bool(_trace.get("series")),
    }
    return ball_dict, ref_frame_url, frame_w, frame_h


def _ai_ball_report(src_path: Path, tee_fps: float, upload_id: int, aib: dict) -> dict:
    """Build the AI-resting-ball debug block: a diagnostic frame with every
    Claude-detected ball position ringed, plus a screenshot per departure."""
    import cv2  # type: ignore

    def _url(name: str):
        p = CLIPS_DIR / name
        return (
            f"{settings.app_base_url}/uploads/clips/{name}?v={int(p.stat().st_mtime)}"
            if p.exists() else None
        )

    samples = aib.get("samples") or []
    peaks = aib.get("peaks") or []
    diag_url = None
    try:
        c = cv2.VideoCapture(str(src_path))
        c.set(cv2.CAP_PROP_POS_FRAMES, int(2.0 * tee_fps))
        ok, ref = c.read()
        c.release()
        if ok and ref is not None:
            for s in samples:
                if s.get("present") and s.get("x") is not None:
                    cv2.circle(
                        ref, (int(s["x"]), int(s["y"])),
                        max(10, int(ref.shape[0] * 0.02)), (255, 255, 0), 3, cv2.LINE_AA,
                    )
            dname = f"debug-aiball-diag-{upload_id}-{secrets.token_hex(4)}.jpg"
            cv2.imwrite(str(CLIPS_DIR / dname), ref, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            diag_url = _url(dname)
    except Exception as exc:  # noqa: BLE001
        log.warning("produce-debug: ai-ball diag failed: %s", exc)

    shots = []
    try:
        c = cv2.VideoCapture(str(src_path))
        for bi, pk in enumerate(peaks):
            pos = next(
                (
                    (s["x"], s["y"]) for s in samples
                    if abs(s["t"] - pk) < 0.01 and s.get("x") is not None
                ),
                None,
            )
            c.set(cv2.CAP_PROP_POS_FRAMES, int(pk * tee_fps))
            ok, fr = c.read()
            if not ok or fr is None:
                continue
            if pos:
                cv2.circle(
                    fr, (int(pos[0]), int(pos[1])),
                    max(14, int(fr.shape[0] * 0.02)), (255, 255, 0), 3, cv2.LINE_AA,
                )
            cv2.putText(
                fr, f"AI ball @ {pk:.1f}s", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA,
            )
            iname = f"debug-aiball-{upload_id}-{bi}-{secrets.token_hex(3)}.jpg"
            cv2.imwrite(str(CLIPS_DIR / iname), fr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            u = _url(iname)
            if u:
                shots.append({"t": pk, "image_url": u})
        c.release()
    except Exception as exc:  # noqa: BLE001
        log.warning("produce-debug: ai-ball shots failed: %s", exc)

    return {
        "available": True,
        "reason": aib.get("reason"),
        "n_swings": aib.get("n_departures"),
        "n_ball_seen": aib.get("n_ball_seen"),
        "n_samples": aib.get("n_samples"),
        "peaks": peaks,
        "diag_url": diag_url,
        "screenshots": shots,
    }


def _run_produce_debug_job(
    upload_id: int, motion_only: bool, wait_after: float | None = None,
) -> None:
    """Render the debug report FROM THE PRODUCE RUN'S OWN WORK RECORD
    (single-run contract — see _produce_work_state). Adds the debug-only
    extras on top: pose screenshots, before/after ball frames drawn from
    the recorded coordinates, a fresh classical-CV comparison per kept
    swing, and the production tracer polled from edit_metrics. Never
    raises."""
    ai_available = bool(os.environ.get("ANTHROPIC_API_KEY"))
    with _produce_debug_lock:
        _produce_debug_state[upload_id] = {
            "running": True, "total": 0, "done": 0, "swings": [],
            "finished_at": None, "ai_available": ai_available, "error": None,
        }

    def _pub(name: str | None) -> str | None:
        if not name:
            return None
        p = CLIPS_DIR / Path(name).name
        if not p.exists():
            return None
        return f"{settings.app_base_url}/uploads/clips/{p.name}?v={int(p.stat().st_mtime)}"

    db = SessionLocal()
    try:
        row = db.get(LongVideoUpload, upload_id)
        if not row or not row.tee_filename:
            raise RuntimeError("upload has no tee video")
        storage.ensure_local(CLIPS_DIR, row.tee_filename)
        src_path = _local_tee(row)
        if not src_path.exists():
            raise RuntimeError(f"tee source missing: {row.tee_filename}")

        tee_fps = probe_fps(src_path) or 30.0

        # ── SINGLE-RUN CONTRACT ──────────────────────────────────────
        # This job does NOT run the detection pipeline. The produce
        # worker publishes its full work record (_produce_work_state)
        # as it filters; this report renders THAT record, so Debug
        # always shows the exact run that cut (or refused to cut) the
        # clips — they can never diverge. wait_after is the timestamp
        # just before this Debug click kicked produce; None means
        # analyze-only / produce already running, which uses the latest
        # record (waiting briefly if a run is in flight).
        work = None
        _deadline = time.time() + 30 * 60
        while time.time() < _deadline:
            with _produce_work_lock:
                _w = _produce_work_state.get(upload_id)
            if _w and _w.get("published") and (
                wait_after is None or _w["published"] >= wait_after
            ):
                work = _w
                break
            if wait_after is None:
                try:
                    db.expire_all()
                    _r = db.get(LongVideoUpload, upload_id)
                    _busy = _r and _r.processing_status in (
                        "pending", "processing",
                    )
                except Exception:  # noqa: BLE001
                    _busy = False
                if not _busy:
                    break
            time.sleep(2.0)
        if work is None:
            raise RuntimeError(
                "no recorded produce run to report on — "
                + (
                    "the produce worker never published its decisions "
                    "(it may have crashed before detection; check the "
                    "upload's error)"
                    if wait_after is not None
                    else "hit Produce (or Debug, which produces) first"
                )
            )

        pose_debug: dict = work.get("pose_debug") or {}
        pose_segments: list[dict] = work.get("segments_all") or []
        decisions: list[dict] = work.get("decisions") or []
        kept: list[dict] = work.get("kept") or []
        tee_fps = float(work.get("fps") or tee_fps)

        def _dec_at(i: int) -> dict:
            return decisions[i] if i < len(decisions) else {}

        # Pose screenshots — deterministic drawing from the recorded
        # peaks (no re-detection).
        pose_shots: list[dict] = []
        try:
            from ..services import pose_swing

            if pose_debug.get("available"):
                _pk_list = pose_debug.get("peaks") or []
                _bend_list = pose_debug.get("swing_bends") or []
                for _pi, pk in enumerate(_pk_list):
                    _bd = _bend_list[_pi] if _pi < len(_bend_list) else None
                    pname = (
                        f"debug-pose-{upload_id}-{int(float(pk) * 100)}-"
                        f"{secrets.token_hex(3)}.jpg"
                    )
                    if pose_swing.annotate_frame(
                        src_path, float(pk), tee_fps, CLIPS_DIR / pname,
                        bend_deg=_bd,
                    ):
                        pp = CLIPS_DIR / pname
                        pose_shots.append({
                            "t": pk,
                            "back_bend_deg": _bd,
                            "image_url": (
                                f"{settings.app_base_url}/uploads/clips/"
                                f"{pname}?v={int(pp.stat().st_mtime)}"
                            ),
                        })
        except Exception as exc:  # noqa: BLE001
            log.warning("produce-debug: pose screenshots failed: %s", exc)

        # Heat panel straight from the recorded checks — the SAME
        # evidence images the produce run's AI judge saw.
        heat_checks = []
        for i, h in enumerate(work.get("heat") or []):
            _img_url = None
            if h.get("image") and (CLIPS_DIR / h["image"]).exists():
                _hp = CLIPS_DIR / h["image"]
                _img_url = (
                    f"{settings.app_base_url}/uploads/clips/{_hp.name}"
                    f"?v={int(_hp.stat().st_mtime)}"
                )
            heat_checks.append({
                "swing": i + 1,
                **{k: h.get(k) for k in (
                    "t", "verdict", "n_timed", "chain_len", "chain_f0",
                    "chain_f1", "chain_flight", "n_rays", "n_angles",
                    "fan", "ai_judge", "ai_reason", "reason",
                )},
                "image_url": _img_url,
            })

        # Practice panel from the recorded classifications. Before/after
        # screenshots are drawn from the RECORDED ball coordinates — no
        # new vision calls; this is what the deciding run saw.
        _prac_by_t = {p.get("t"): p for p in (work.get("practice") or [])}
        ai_ball_swings: list[dict] = []
        n_real = 0
        try:
            import cv2  # type: ignore

            _cap = cv2.VideoCapture(str(src_path))
        except Exception:  # noqa: BLE001
            _cap = None

        def _drawn_shot(ball, tag, i):
            if not ball or _cap is None:
                return None
            try:
                t = float(ball.get("t") or 0.0)
                _cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * tee_fps))
                ok, fr = _cap.read()
                if not ok or fr is None:
                    return None
                present = bool(
                    ball.get("present") and ball.get("x") is not None
                )
                cb = ball.get("crop_box")
                if cb and len(cb) == 4:
                    cv2.rectangle(
                        fr, (int(cb[0]), int(cb[1])),
                        (int(cb[0] + cb[2]), int(cb[1] + cb[3])),
                        (255, 200, 0), 2, cv2.LINE_AA,
                    )
                if present:
                    rad = max(12, int(fr.shape[0] * 0.02))
                    cv2.circle(
                        fr, (int(ball["x"]), int(ball["y"])), rad,
                        (255, 255, 0), 3, cv2.LINE_AA,
                    )
                    cv2.circle(
                        fr, (int(ball["x"]), int(ball["y"])), 3,
                        (0, 0, 255), -1, cv2.LINE_AA,
                    )
                cv2.putText(
                    fr,
                    f"{tag} @ {t:.1f}s: {'ball' if present else 'no ball'}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 0) if present else (0, 0, 255), 2,
                    cv2.LINE_AA,
                )
                iname = (
                    f"debug-aiball-{upload_id}-s{i}-{tag}-"
                    f"{secrets.token_hex(3)}.jpg"
                )
                cv2.imwrite(
                    str(CLIPS_DIR / iname), fr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                )
                _sp = CLIPS_DIR / iname
                if not _sp.exists():
                    return None
                return {
                    "t": round(t, 2), "present": present,
                    "image_url": (
                        f"{settings.app_base_url}/uploads/clips/{iname}"
                        f"?v={int(_sp.stat().st_mtime)}"
                    ),
                }
            except Exception:  # noqa: BLE001
                return None

        for i, d in enumerate(pose_segments):
            _t = round(float(d.get("peak_time_sec") or 0.0), 2)
            _dec = _dec_at(i)
            _p = _prac_by_t.get(_t)
            if _p is not None:
                if _p.get("verdict") == "real":
                    n_real += 1
                _anc = _p.get("anchor")
                if _anc:
                    _anc = dict(_anc)
                    for _ik, _uk in (
                        ("image", "image_url"),
                        ("image_mog2", "image_mog2_url"),
                        ("launch_image", "launch_image_url"),
                        ("launch_image_heat", "launch_image_heat_url"),
                        ("ai_launch_image", "ai_launch_image_url"),
                        ("assumed_impact_image", "assumed_impact_image_url"),
                        ("early_image", "early_image_url"),
                        ("early_image_heat", "early_image_heat_url"),
                    ):
                        if _anc.get(_ik) and (
                            CLIPS_DIR / _anc[_ik]
                        ).exists():
                            _ap = CLIPS_DIR / _anc[_ik]
                            _anc[_uk] = (
                                f"{settings.app_base_url}/uploads/clips/"
                                f"{_ap.name}?v={int(_ap.stat().st_mtime)}"
                            )
                ai_ball_swings.append({
                    "swing": i + 1,
                    "verdict": _p.get("verdict"),
                    "reason": _p.get("reason"),
                    "before": _drawn_shot(_p.get("before"), "before", i),
                    "after": _drawn_shot(_p.get("after"), "after", i),
                    "anchor": _anc,
                })
            elif _dec.get("dropped_by") in ("heat_ai", "heat_heuristic"):
                ai_ball_swings.append({
                    "swing": i + 1,
                    "verdict": "skipped",
                    "reason": (
                        "eliminated by the MOG2/AI swing check — produce "
                        "never classified it"
                    ),
                    "before": None, "after": None,
                })
            else:
                ai_ball_swings.append({
                    "swing": i + 1,
                    "verdict": "unknown",
                    "reason": "not classified by the produce run (no key?)",
                    "before": None, "after": None,
                })
        if _cap is not None:
            try:
                _cap.release()
            except Exception:  # noqa: BLE001
                pass
        ai_ball_dict = {
            "available": bool(work.get("practice")),
            "pose_anchored": True,
            "n_swings": len(pose_segments),
            "n_real": n_real,
            "swings": ai_ball_swings,
            "reason": (
                None if work.get("practice")
                else "produce run classified nothing (no key / no survivors)"
            ),
        }

        # FINAL VERDICT — read straight off the recorded decision trail;
        # this IS what produce did, not a re-derivation.
        _stage_names = {
            "heat_ai": "MOG2 + AI swing judge",
            "heat_heuristic": "MOG2 swing check",
            "practice_filter": "ball-departure (practice) filter",
        }
        _fv_swings = []
        _kept_idxs: list[int] = []
        for i, d in enumerate(pose_segments):
            _t = round(float(d.get("peak_time_sec") or 0.0), 2)
            _dec = _dec_at(i)
            if _dec.get("kept"):
                _kept_idxs.append(i)
                _bits = []
                if _dec.get("heat") == "club_swing":
                    _bits.append(
                        "the AI judge recognised the heat composite as a "
                        "golf swing"
                        if _dec.get("ai_judge")
                        else "the club-fan heuristic recognised a swing"
                    )
                if _dec.get("practice") == "real":
                    _bits.append(
                        "the ball was there before impact and gone after "
                        "(real shot)"
                    )
                elif _dec.get("practice") == "unknown":
                    _bits.append(
                        "ball-departure was inconclusive"
                        + (
                            f' ("{_dec.get("practice_reason")}")'
                            if _dec.get("practice_reason") else ""
                        )
                        + " — kept, the filter only drops confirmed "
                        "practice swings"
                    )
                _fv_swings.append({
                    "swing": i + 1, "t": _t, "produced": True,
                    "stage": None,
                    "explanation": (
                        f"Pose burst @ {_t}s passed the wrist-speed and "
                        f"bend gates"
                        + ("; " + "; ".join(_bits) if _bits else "")
                        + ". Produced as a clip."
                    ),
                })
            else:
                _stage = _dec.get("dropped_by") or "unknown"
                if _stage == "heat_ai":
                    _why = (
                        "the AI judge looked at the motion-heat composite "
                        "and said it is not a swing"
                        + (
                            f' ("{_dec.get("ai_reason")}")'
                            if _dec.get("ai_reason") else ""
                        )
                    )
                elif _stage == "heat_heuristic":
                    _why = (
                        "no club fan in the motion heat (no AI judge "
                        "available)"
                    )
                else:
                    _why = (
                        "the ball never left its spot"
                        + (
                            f' ("{_dec.get("practice_reason")}")'
                            if _dec.get("practice_reason") else ""
                        )
                    )
                _fv_swings.append({
                    "swing": i + 1, "t": _t, "produced": False,
                    "stage": _stage,
                    "explanation": (
                        f"Pose flagged a candidate @ {_t}s, but the "
                        f"{_stage_names.get(_stage, _stage)} eliminated "
                        f"it: {_why}. Not produced — later stages skipped "
                        f"it."
                    ),
                })
        _prod_nums = [i + 1 for i in _kept_idxs]
        if not pose_segments:
            _fv_summary = "No pose swing candidates were found."
        elif _prod_nums:
            _fv_summary = (
                f"{len(_prod_nums)} of {len(pose_segments)} pose "
                f"candidate(s) survived every filter — producing swing"
                f"{'s' if len(_prod_nums) != 1 else ''} "
                f"{', '.join(str(n) for n in _prod_nums)}."
            )
        else:
            _fv_summary = (
                f"None of the {len(pose_segments)} pose candidates "
                f"survived the filters — nothing will be produced."
            )
        final_verdict = {
            "available": bool(pose_segments),
            "n_candidates": len(pose_segments),
            "n_produced": len(_kept_idxs),
            "produced": _prod_nums,
            "summary": _fv_summary,
            "swings": _fv_swings,
            # Rendered from the produce run's own record — not re-derived.
            "single_run": True,
        }

        with _produce_debug_lock:
            st = _produce_debug_state[upload_id]
            st["total"] = len(_kept_idxs)
            st["single_run"] = True
            st["heat_check"] = {
                "available": bool(heat_checks),
                "enabled": bool(settings.swing_heat_check_enabled),
                "swings": heat_checks,
            }
            st["pose"] = {
                "available": bool(pose_debug.get("available")),
                "reason": pose_debug.get("reason"),
                "series": pose_debug.get("series"),
                "duration_sec": pose_debug.get("duration_sec"),
                "threshold": pose_debug.get("threshold"),
                "n_pose_frames": pose_debug.get("n_pose_frames"),
                "n_samples": pose_debug.get("n_samples"),
                "coverage": pose_debug.get("coverage"),
                "n_swings": pose_debug.get("n_swings"),
                "n_bend_rejected": pose_debug.get("n_bend_rejected"),
                "back_bend_min_deg": pose_debug.get("back_bend_min_deg"),
                "strong_ratio": pose_debug.get("strong_ratio"),
                "peaks": pose_debug.get("peaks") or [],
                "bursts_detail": pose_debug.get("bursts_detail") or [],
                "screenshots": pose_shots,
            }
            st["ai_ball"] = ai_ball_dict
            st["final_verdict"] = final_verdict

        # ── Per-swing tracer panel ───────────────────────────────────
        # The PRODUCTION tracer (what produce actually rendered — polled
        # from edit_metrics as each segment finishes) next to a fresh
        # classical-CV run on the SAME window produce cut. No second AI
        # tracer run: what you see IS the produced render.
        _before_s = float(
            settings.pose_clip_before_sec if work.get("used_pose")
            else CLIP_SECONDS_BEFORE_IMPACT
        )
        _after_s = float(
            settings.pose_clip_after_sec if work.get("used_pose")
            else CLIP_SECONDS_TEE_ONLY_AFTER_IMPACT
        )
        for _done_n, i in enumerate(_kept_idxs, start=1):
            d = pose_segments[i]
            # Produce enumerates the surviving swings 0..n-1 in order —
            # that's the idx the wizard entries are persisted under.
            prod_idx = _done_n - 1
            peak = float(d.get("peak_time_sec") or 0.0)
            cut_start = max(0.0, peak - _before_s)
            cut_end = peak + _after_s
            seg_name = f"debug-{upload_id}-s{i}-{secrets.token_hex(6)}.mp4"
            seg_path = CLIPS_DIR / seg_name

            classical = {"ok": False, "error": "cut failed"}
            # Close any open read transaction before the minutes-long
            # classical run (Postgres idle-in-transaction timeout).
            try:
                db.rollback()
            except Exception:  # noqa: BLE001
                pass
            if cut_segment(src_path, seg_path, cut_start, cut_end):
                try:
                    c_url, c_info, _c_traced, c_debug_url = _run_tracer(
                        seg_path,
                    )
                    c_info = c_info or {}
                    classical = {
                        "ok": bool(c_info.get("ok")),
                        "n_points": c_info.get("n_points"),
                        "n_candidates": c_info.get("n_candidates"),
                        "error": c_info.get("error"),
                        "traced_url": c_url,
                        "heatmap_url": c_debug_url,
                    }
                except Exception as exc:  # noqa: BLE001
                    classical = {
                        "ok": False, "error": f"classical crashed: {exc}",
                    }

            # Production tracer: wait for produce to persist this
            # swing's entry (it's tracing in parallel).
            ai = {"ok": False, "error": "production tracer not ready"}
            _tr_deadline = time.time() + 20 * 60
            while time.time() < _tr_deadline:
                _done_row = None
                try:
                    db.expire_all()
                    _row2 = db.get(LongVideoUpload, upload_id)
                    _done_row = (
                        _row2 is not None
                        and _row2.processing_status in ("completed", "failed")
                    )
                    _sw = None
                    for s in ((_row2.edit_metrics or {}).get("swings") or []):
                        if (
                            isinstance(s, dict)
                            and int(s.get("idx", -1)) == prod_idx
                        ):
                            _sw = s
                            break
                    # Only accept an entry persisted by THIS produce run
                    # — a previous run's leftover satisfies the URL
                    # check instantly and shows stale anchors/errors.
                    # Entries with no stamp (pre-stamp data) are only
                    # trusted once produce has finished.
                    _run_t0 = float(work.get("run_started") or 0.0)
                    _fresh = (
                        float(_sw.get("persisted_at") or 0.0)
                        >= _run_t0 - 1.0
                        if _sw else False
                    ) or (_sw is not None and _done_row)
                    if _sw and _fresh and (
                        _sw.get("tracer_url") or _sw.get("ball_track_frames")
                    ):
                        ai = {
                            "ok": True,
                            "error": None,
                            "engine": _sw.get("tracer_engine"),
                            "address_frame": _sw.get("address_frame"),
                            "impact_frame": _sw.get("impact_frame"),
                            "handedness": _sw.get("handedness"),
                            "n_track": len(
                                _sw.get("ball_track_frames") or [],
                            ),
                            "traced_url": _sw.get("tracer_url"),
                            "mog2_overlay_url": _sw.get("mog2_overlay_url"),
                            "anchor_check": _sw.get("anchor_check"),
                            # Flight-map ingredients: heat background,
                            # labelled dots, rest ball, native dims.
                            "raw_motion_url": _sw.get(
                                "tracer_raw_motion_url",
                            ),
                            "timed_points": _sw.get("timed_points"),
                            "mog2_stats": _sw.get("mog2_stats"),
                            "render_info": _sw.get("render_info"),
                            # Full mapped track (all sources) so the
                            # flight map can draw the whole arc line.
                            "track_points": [
                                {
                                    "frame": rec.get("frame"),
                                    "x": rec.get("x"),
                                    "y": rec.get("y"),
                                    "source": rec.get("source"),
                                }
                                for rec in (
                                    _sw.get("ball_track_frames") or []
                                )
                                if rec.get("found")
                                and rec.get("x") is not None
                                and rec.get("y") is not None
                            ][:1200],
                            "ball": _sw.get("ball"),
                            "arc_region": (
                                (_sw.get("mog2_stats") or {}).get(
                                    "arc_region",
                                )
                            ),
                            # Native dims from edit_metrics (the model
                            # has no width/height columns); the
                            # frontend also falls back to the heat
                            # image's natural size.
                            "frame_w": (
                                (_row2.edit_metrics or {}).get(
                                    "frame_width",
                                ) if _row2 else None
                            ),
                            "frame_h": (
                                (_row2.edit_metrics or {}).get(
                                    "frame_height",
                                ) if _row2 else None
                            ),
                            "production": True,
                        }
                        break
                except Exception as exc:  # noqa: BLE001
                    ai = {
                        "ok": False,
                        "error": f"production tracer poll failed: {exc}",
                    }
                    break
                if _done_row:
                    ai = {
                        "ok": False,
                        "error": (
                            "produce finished without a tracer for this "
                            "swing"
                        ),
                    }
                    break
                # End the read transaction before sleeping — Postgres
                # kills connections that idle INSIDE a transaction, and
                # this poll can run for many minutes.
                try:
                    db.rollback()
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(5.0)

            entry = {
                "idx": i,
                "hole_number": d.get("hole_number"),
                "peak_time_sec": round(peak, 2),
                "ball_verdict": d.get("ball_verdict"),
                "classical": classical,
                "ai": ai,
            }
            with _produce_debug_lock:
                st = _produce_debug_state[upload_id]
                st["swings"].append(entry)
                st["done"] = _done_n
            log.info(
                "produce-debug: upload=%s swing %d classical_ok=%s "
                "production_tracer_ok=%s",
                upload_id, i, classical.get("ok"), ai.get("ok"),
            )

    except Exception as exc:  # noqa: BLE001
        log.exception("produce-debug %s failed: %s", upload_id, exc)
        with _produce_debug_lock:
            if upload_id in _produce_debug_state:
                _produce_debug_state[upload_id]["error"] = str(exc)[:500]
    finally:
        db.close()
        import time as _t
        with _produce_debug_lock:
            if upload_id in _produce_debug_state:
                _produce_debug_state[upload_id]["running"] = False
                _produce_debug_state[upload_id]["finished_at"] = _t.time()


@router.post("/long-uploads/{upload_id}/produce-debug")
def produce_debug(
    upload_id: int, analyze_only: bool = False, db: Session = Depends(get_db)
):
    """Dev tool: kick a normal produce (saves clips) AND a per-swing
    diagnostic that compares the classical-CV and AI tracers. Returns
    immediately; poll the status route for the report.

    analyze_only=true re-runs ONLY the diagnostic (e.g. after changing the
    tee-box ROI) without re-producing the clip."""
    if not settings.produce_debug_enabled:
        return {"ok": False, "error": "Produce debug is not enabled on this deployment."}
    row = db.get(LongVideoUpload, upload_id)
    if not row:
        raise HTTPException(404, "long upload not found")
    with _produce_debug_lock:
        if (_produce_debug_state.get(upload_id) or {}).get("running"):
            return {"ok": False, "error": "A debug run is already in progress."}
    motion_only = bool(getattr(row, "camera_event_id", None))

    # 1) Normal produce — saves the clips exactly like the Produce button.
    # The debug report renders from THIS run's published work record
    # (single-run contract), so wait_after marks the moment we kicked it.
    wait_after: float | None = None
    if not analyze_only and row.processing_status != "processing":
        row.processing_status = "pending"
        row.processing_started_at = None
        row.processing_completed_at = None
        row.last_error = None
        db.commit()
        wait_after = time.time()
        threading.Thread(
            target=_run_long_upload_job,
            kwargs={
                "upload_id": row.id, "seg_list": [], "auto_detect_swings": True,
                "starting_hole": 1, "ai_tracer_model": None,
                "debug_artifacts": True,
            },
            daemon=True, name=f"produce-debug-produce-{row.id}",
        ).start()

    # 2) Report renderer — reads the produce run's record; adds the
    # classical-CV comparison and shows the production tracer.
    threading.Thread(
        target=_run_produce_debug_job,
        kwargs={
            "upload_id": row.id, "motion_only": motion_only,
            "wait_after": wait_after,
        },
        daemon=True, name=f"produce-debug-analyze-{row.id}",
    ).start()
    return {"ok": True, "upload_id": row.id, "ai_available": bool(os.environ.get("ANTHROPIC_API_KEY"))}


@router.get("/long-uploads/{upload_id}/produce-debug/status")
def produce_debug_status(upload_id: int):
    """Progress + report of the current/last produce-debug run for this
    upload, plus whether the tool is enabled (so the UI can show the button)."""
    return {"enabled": bool(settings.produce_debug_enabled), **_produce_debug_report(upload_id)}


@router.post("/long-uploads/{upload_id}/rescan-ball")
def rescan_ball(upload_id: int, db: Session = Depends(get_db)):
    """Re-run ONLY the ball detector with the course's current tee-box ROI —
    fast + synchronous, used right after drawing/adjusting the ROI (the full
    produce-debug re-run is slow and re-does the tracer comparison, which the
    ROI doesn't affect). Returns the updated ball block + ref/diagnostic
    frames so the modal can refresh in place."""
    if not settings.produce_debug_enabled:
        return {"ok": False, "error": "Produce debug is not enabled on this deployment."}
    row = db.get(LongVideoUpload, upload_id)
    if not row or not row.tee_filename:
        raise HTTPException(404, "upload not found or has no tee video")
    storage.ensure_local(CLIPS_DIR, row.tee_filename)
    src_path = _local_tee(row)
    if not src_path.exists():
        raise HTTPException(404, "tee source missing on disk")
    tee_fps = probe_fps(src_path) or 30.0
    course = db.get(Course, row.course_id)
    ball_roi = course.ball_roi if course else None
    ball_dict, ref_frame_url, frame_w, frame_h = _ball_debug_and_ref(
        src_path, tee_fps, upload_id, ball_roi,
    )
    with _produce_debug_lock:
        st = _produce_debug_state.get(upload_id)
        if st is not None:
            st["ball"] = ball_dict
            st["ball_roi"] = ball_roi
            st["ref_frame_url"] = ref_frame_url
    return {
        "ok": True,
        "ball": ball_dict,
        "ball_roi": ball_roi,
        "ref_frame_url": ref_frame_url,
        "frame_w": frame_w,
        "frame_h": frame_h,
    }


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

    ext = (
        (video.filename or "").rsplit(".", 1)[-1].lower()
        if "." in (video.filename or "")
        else "mp4"
    )
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
        if thumb_path
        else None
    )

    # When the operator says the clip is already traced (e.g. rendered
    # in the Shot Tracer iOS/Android app before upload), skip our
    # classical-CV pipeline entirely. The uploaded file IS the
    # deliverable — point both source_url and tracer_url at it.
    if already_traced:
        tracer_url = f"{settings.app_base_url}/uploads/clips/{fname}"
        tracer_info = {
            "ok": True,
            "source": "external",
            "note": "already_traced upload",
        }
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
            g_ext = (
                (video_green.filename or "").rsplit(".", 1)[-1].lower()
                if "." in (video_green.filename or "")
                else "mp4"
            )
            if g_ext not in ("mp4", "mov", "webm", "m4v"):
                g_ext = "mp4"
            green_name = (
                f"{course_id}-h{hole_number}-{secrets.token_hex(6)}_green.{g_ext}"
            )
            green_path = CLIPS_DIR / green_name
            green_path.write_bytes(green_data)
            compress_for_email(green_path)
            green_url = f"{settings.app_base_url}/uploads/clips/{green_name}"

            green_tracer_url, green_tracer_info, green_traced_path, green_debug_url = (
                _run_tracer(green_path)
            )

            if (
                tracer_info
                and tracer_info.get("ok")
                and green_tracer_info
                and green_tracer_info.get("ok")
                and tee_traced_path is not None
                and green_traced_path is not None
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
                        tee_traced_path,
                        0.0,
                        switch_sec,
                        green_traced_path,
                        switch_sec,
                        end_sec_in_green,
                        composite_path,
                    ):
                        composite_url = (
                            f"{settings.app_base_url}/uploads/clips/{composite_name}"
                        )
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
        notifications.notify_hio_under_review(
            participant.name, participant.mobile, participant.email
        )

    db.commit()

    # Fire gallery-ready notification on first assigned clip
    if (
        clip.participant_id
        and clip.processing_status == ClipProcessingStatus.assigned.value
    ):
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

    ext = (
        (video.filename or "").rsplit(".", 1)[-1].lower()
        if "." in (video.filename or "")
        else "mp4"
    )
    if ext not in ("mp4", "mov", "webm", "m4v"):
        ext = "mp4"
    fname = f"slot{position}-{secrets.token_hex(6)}.{ext}"
    fpath = SHOWCASE_DIR / fname
    fpath.write_bytes(data)

    # Compress + extract first-frame thumbnail (same pipeline as clip uploads)
    compress_for_email(fpath)
    thumb = extract_thumbnail(fpath)

    # Persist to object storage immediately so the file survives a redeploy.
    # Showcase files don't go through CLIPS_DIR so the sweeper won't pick
    # them up — we push them here with a "showcase/" key prefix.
    from ..services import storage as _storage
    _storage.upload(f"showcase/{fname}", fpath)
    if thumb:
        _storage.upload(f"showcase/{thumb.name}", thumb)

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
    clip_ids = [
        cid for cid in (evt.tee_clip_id, evt.wide_clip_id, evt.hole_clip_id) if cid
    ]
    hole_clips = (
        db.query(VideoClip).filter(VideoClip.id.in_(clip_ids)).all() if clip_ids else []
    )
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


def _camera_to_dict(
    c: Camera,
    last_event: CameraEvent | None = None,
    course_name: str | None = None,
) -> dict:
    """Shape a Camera row for the admin UI. Includes the auth_token
    because operators need it to provision the Pi's SD card.

    `course_name` is passed in rather than looked up so list_cameras can
    batch the query. It matters: without it the card had no way to show
    where a camera actually is, and operators fell back to reading the
    free-text `name` — which goes stale the moment a camera is moved.
    """
    return {
        "id": c.id,
        "course_id": c.course_id,
        "course_name": course_name,
        "assigned_hole": c.assigned_hole,
        "assigned_role": c.assigned_role,
        "paired_with_camera_id": c.paired_with_camera_id,
        "auth_token": c.auth_token,
        "name": c.name,
        "tee_box_roi": c.tee_box_roi,
        "ball_side": c.ball_side,
        "last_seen_at": c.last_seen_at.isoformat() if c.last_seen_at else None,
        "firmware_version": c.firmware_version,
        "battery": _battery_status(
            c.battery_voltage, c.battery_current_a, c.battery_updated_at,
        ),
        "enabled": bool(c.enabled),
        "triggering_enabled": bool(c.triggering_enabled),
        "note": c.note,
        "created_at": c.created_at.isoformat() if c.created_at else None,
        "last_event_at": (
            last_event.triggered_at.isoformat()
            if last_event and last_event.triggered_at
            else None
        ),
        "last_event_status": last_event.status if last_event else None,
    }


def _course_name_for(db, cam: Camera) -> str | None:
    """Single-camera course name, for the endpoints that return one row.
    Without this the card blanks its course the moment you create, pair
    or edit a camera — until the next full list refresh."""
    c = db.get(Course, cam.course_id) if cam.course_id else None
    return c.name if c else None


@router.get("/cameras")
def list_cameras(db: Session = Depends(get_db)):
    """Every registered camera, newest first. Includes the
    most recent CameraEvent's status / timestamp for at-a-glance
    health visibility."""
    cams = db.query(Camera).order_by(Camera.created_at.desc()).all()
    # One query for every course rather than one per camera.
    course_names = {
        cid: nm
        for cid, nm in db.query(Course.id, Course.name)
        .filter(Course.id.in_({c.course_id for c in cams}))
        .all()
    } if cams else {}
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
        out.append(_camera_to_dict(c, last_evt, course_names.get(c.course_id)))
    return out


@router.get("/diagnostics")
def environment_diagnostics(db: Session = Depends(get_db)):
    """One-shot health readout of everything the produce pipeline needs,
    so a broken DEPLOYMENT environment (missing ffmpeg, mediapipe that
    can't load its native libs, absent API key, dead bucket) is visible
    in one request instead of being inferred from silent failures.
    Open /api/admin/diagnostics on the env that's misbehaving."""
    import shutil as _sh
    import subprocess as _sp

    out: dict = {
        "deployment": bool(os.environ.get("REPLIT_DEPLOYMENT")),
        "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "storage_bucket_enabled": storage.enabled(),
        "auto_delete_non_golf": settings.auto_delete_non_golf,
    }

    ffmpeg = _sh.which("ffmpeg")
    out["ffmpeg_path"] = ffmpeg
    if ffmpeg:
        try:
            v = _sp.run(
                [ffmpeg, "-version"], capture_output=True, text=True, timeout=10
            )
            out["ffmpeg_version"] = (v.stdout or "").splitlines()[0][:120]
        except Exception as exc:  # noqa: BLE001
            out["ffmpeg_version"] = f"ERROR: {exc}"
    out["ffprobe_path"] = _sh.which("ffprobe")

    try:
        import cv2 as _cv2

        out["opencv"] = _cv2.__version__
    except Exception as exc:  # noqa: BLE001
        out["opencv"] = f"ERROR: {type(exc).__name__}: {exc}"

    try:
        from ..services import pose_swing as _ps

        pose = _ps._get_pose()
        out["mediapipe_pose"] = (
            "ok" if pose is not None else f"UNAVAILABLE: {_ps._pose_error}"
        )
    except Exception as exc:  # noqa: BLE001
        out["mediapipe_pose"] = f"ERROR: {type(exc).__name__}: {exc}"

    try:
        usage = _sh.disk_usage(CLIPS_DIR)
        out["clips_dir"] = str(CLIPS_DIR)
        out["clips_count"] = sum(1 for f in CLIPS_DIR.glob("*") if f.is_file())
        out["disk_free_gb"] = round(usage.free / 1e9, 2)
    except Exception as exc:  # noqa: BLE001
        out["clips_dir"] = f"ERROR: {exc}"

    # The most recent uploads + how their produce runs ended — surfaces
    # the actual last_error instead of a silent "never completed".
    try:
        rows = (
            db.query(LongVideoUpload)
            .order_by(LongVideoUpload.id.desc())
            .limit(5)
            .all()
        )
        out["recent_uploads"] = [
            {
                "id": r.id,
                "status": r.processing_status,
                "started": (
                    r.processing_started_at.isoformat()
                    if r.processing_started_at
                    else None
                ),
                "error": (r.last_error or "")[:300] or None,
            }
            for r in rows
        ]
    except Exception as exc:  # noqa: BLE001
        out["recent_uploads"] = f"ERROR: {exc}"

    return out


@router.post("/cameras/{camera_id}/watch")
def start_watch_camera(camera_id: int, db: Session = Depends(get_db)):
    """Admin clicked Watch. Mark this camera as actively watched."""
    if not db.get(Camera, camera_id):
        raise HTTPException(404, "camera not found")
    with _LIVE_LOCK:
        _WATCHERS[camera_id] = datetime.utcnow()
    return {"ok": True}


@router.post("/cameras/{camera_id}/capture")
def capture_camera(
    camera_id: int, seconds: int = 30, db: Session = Depends(get_db),
):
    """Record `seconds` of footage on demand, without waiting for a
    golfer to walk into the tee box.

    Deliberately fires the SAME path a real trigger does: the tee posts
    /event-trigger, which wakes its paired green through poll-trigger,
    both upload, and _process_camera_event_job queues it for produce. So
    a Capture lands on /admin/production as an ordinary card and
    exercises the whole pipeline — a parallel "test" path would prove
    nothing about the real one.

    The request is delivered on the camera's next watch-status poll (a
    few seconds), so this returns immediately rather than blocking.
    """
    cam = db.get(Camera, camera_id)
    if not cam:
        raise HTTPException(404, "camera not found")
    if not cam.enabled:
        raise HTTPException(409, "camera is disabled")
    # Only a tee camera can start an event — the green half records
    # because its partner told it to. Asking a green to capture alone
    # would produce a clip with no tee footage, which cannot be produced.
    if (cam.assigned_role or "").lower() != "tee":
        raise HTTPException(
            409,
            "capture must be started from the TEE camera; its paired "
            "green records automatically",
        )
    seconds = max(5, min(120, int(seconds)))
    from .cameras import request_capture

    request_capture(camera_id, seconds)
    db.add(AuditLog(
        actor="admin", action="capture_camera",
        target=f"camera:{camera_id}", detail=f"seconds={seconds}",
    ))
    db.commit()
    return {
        "ok": True,
        "camera_id": camera_id,
        "seconds": seconds,
        "paired_green_camera_id": cam.paired_with_camera_id,
        "note": (
            "Queued. The camera picks this up on its next status poll "
            "(a few seconds), records, uploads, and the clip appears on "
            "Production."
        ),
    }


@router.post("/cameras/{camera_id}/calibrate")
def calibrate_green_camera(
    camera_id: int,
    payload: dict = Body(...),
    db: Session = Depends(get_db),
):
    """Save a camera's image -> green-plane mapping.

    Body: {"image_points": [[x,y], ...], "world_points": [[X,Y], ...],
           "pin": {"image": [x,y], "world": [X,Y]} | null}

    World coords are FEET on the plane of the green: X across, Y toward
    the back. The operator picks the origin; marking the pin makes
    distances readable without moving it.

    BOTH ROLES, for different jobs. On the GREEN camera this MEASURES --
    closest-to-the-pin -- so it is held to MAX_RMS_FT. On the TEE camera
    it AIMS: a landing marked on the green becomes feet on the green
    becomes a pixel in the tee frame, which is where the tracer has to
    finish. The tee's view of the green is small and far, so the same
    four clicks carry more world error -- but a few feet there is a few
    pixels on screen, and holding it to the measuring tolerance would
    reject a fit that is entirely good enough for the only job it has.
    Hence the looser limit, and `purpose` recorded so nothing downstream
    can mistake an aiming fit for a measuring one.
    """
    from ..services import green_calibration as gc

    cam = db.get(Camera, camera_id)
    if not cam:
        raise HTTPException(404, "camera not found")
    _role = (cam.assigned_role or "").lower()
    if _role not in ("green", "tee"):
        raise HTTPException(
            409, "calibration applies to green and tee cameras — it maps "
                 "the green's surface onto this camera's image",
        )
    _is_tee = _role == "tee"
    _limit = gc.MAX_RMS_FT_TEE if _is_tee else gc.MAX_RMS_FT
    try:
        H, rms, rms_meaningful = gc.compute_homography(
            payload.get("image_points"), payload.get("world_points"),
        )
    except gc.CalibrationError as exc:
        raise HTTPException(400, str(exc))

    if rms_meaningful and rms > _limit:
        raise HTTPException(
            400,
            f"fit is off by {rms} ft on your own marked points (limit "
            f"{_limit} ft). Check the measurements or re-click — a "
            f"bad mapping is worse than none, because it answers "
            f"confidently and wrongly.",
        )

    pin = payload.get("pin") or None
    cam.green_homography = {
        "image_points": payload.get("image_points"),
        "world_points": payload.get("world_points"),
        "homography": H,
        "pin": pin,
        "rms_error_ft": rms if rms_meaningful else None,
        "n_points": len(payload.get("image_points") or []),
        # What this fit is allowed to be used for. A tee fit aims the
        # tracer; it must never be read as a distance.
        "purpose": "tracer_aim" if _is_tee else "measure",
        "role": _role,
        "calibrated_at": _utcnow_naive().isoformat(),
    }
    db.add(AuditLog(
        actor="admin", action="calibrate_green_camera",
        target=f"camera:{camera_id}",
        detail=f"points={cam.green_homography['n_points']} rms={rms}",
    ))
    db.commit()
    return {
        "ok": True,
        "rms_error_ft": rms if rms_meaningful else None,
        "n_points": cam.green_homography["n_points"],
        # Four points fit exactly, so the residual proves nothing. Say so
        # rather than letting a 0.00 read as "perfect".
        "accuracy_note": (
            f"Fit residual {rms} ft across {cam.green_homography['n_points']} "
            f"points." if rms_meaningful else
            "Exact fit — 4 points always fit perfectly, so this cannot "
            "measure its own accuracy. Add a 5th marked point to get a real "
            "error estimate, or verify by clicking a ball you can pace out."
        ),
    }


@router.get("/cameras/{camera_id}/calibration")
def get_green_calibration(camera_id: int, db: Session = Depends(get_db)):
    """Stored calibration, or nulls if the camera has never been done."""
    cam = db.get(Camera, camera_id)
    if not cam:
        raise HTTPException(404, "camera not found")
    return {"camera_id": camera_id, "calibration": cam.green_homography}


@router.post("/cameras/{camera_id}/measure")
def measure_green_point(
    camera_id: int,
    payload: dict = Body(...),
    db: Session = Depends(get_db),
):
    """Where on the green is pixel (x, y)? The verification tool: click a
    ball you can pace out and check the answer before trusting the
    calibration with a contest."""
    from ..services import green_calibration as gc

    cam = db.get(Camera, camera_id)
    if not cam:
        raise HTTPException(404, "camera not found")
    if not cam.green_homography:
        raise HTTPException(409, "camera is not calibrated")
    try:
        x, y = float(payload["x"]), float(payload["y"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "body must be {\"x\": px, \"y\": px}")
    pos = gc.image_to_green(cam.green_homography, x, y)
    if pos is None:
        raise HTTPException(
            422, "that pixel doesn't land on the green's plane (the horizon "
                 "maps to infinity) — click on the putting surface",
        )
    # A TEE camera's fit exists to AIM the tracer, not to measure. It is
    # useful for checking the mapping looks sane; it is not a yardage.
    # Say so here rather than letting a number stand on its own.
    _purpose = (cam.green_homography or {}).get("purpose")
    return {**pos, "purpose": _purpose or "measure",
            "measures_distance": _purpose != "tracer_aim"}


@router.delete("/cameras/{camera_id}/watch")
def stop_watch_camera(camera_id: int):
    """Admin closed the live view. Clear watcher + cached frame."""
    with _LIVE_LOCK:
        _WATCHERS.pop(camera_id, None)
        _LIVE_FRAMES.pop(camera_id, None)
    return {"ok": True}


@router.get("/cameras/{camera_id}/live-frame")
def get_camera_live_frame(camera_id: int, db: Session = Depends(get_db)):
    """Admin pulls latest JPEG. Each pull renews the 10s watch TTL."""
    if not db.get(Camera, camera_id):
        raise HTTPException(404, "camera not found")
    with _LIVE_LOCK:
        _WATCHERS[camera_id] = datetime.utcnow()
        slot = _LIVE_FRAMES.get(camera_id)
    if not slot:
        return Response(status_code=204)
    frame_bytes, ts = slot
    if datetime.utcnow() - ts > FRAME_TTL:
        return Response(status_code=204)
    return Response(
        content=frame_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


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
    db.add(
        AuditLog(
            actor="admin",
            action="create_camera",
            target=f"camera:{cam.id}",
            detail=f"course={course.id} hole={hole} role={role}",
        )
    )
    db.commit()
    db.refresh(cam)
    return _camera_to_dict(cam, None, _course_name_for(db, cam))


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
    db.add(
        AuditLog(
            actor="admin",
            action="pair_cameras",
            target=f"camera:{cam.id}",
            detail=f"partner={partner.id}",
        )
    )
    db.commit()
    return {"ok": True, "cameras": [
        _camera_to_dict(cam, None, _course_name_for(db, cam)),
        _camera_to_dict(partner, None, _course_name_for(db, partner)),
    ]}


@router.post("/cameras/{camera_id}/unpair")
def unpair_camera(camera_id: int, db: Session = Depends(get_db)):
    """Clear pairing on both sides."""
    cam = db.get(Camera, camera_id)
    if not cam:
        raise HTTPException(404, "camera not found")
    partner = (
        db.get(Camera, cam.paired_with_camera_id) if cam.paired_with_camera_id else None
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
    db.add(
        AuditLog(
            actor="admin",
            action="rotate_camera_token",
            target=f"camera:{cam.id}",
            detail="",
        )
    )
    db.commit()
    db.refresh(cam)
    return {"ok": True, "auth_token": cam.auth_token}


@router.post("/cameras/{camera_id}/update")
def update_camera(
    camera_id: int,
    name: str | None = Form(None),
    enabled: bool | None = Form(None),
    triggering_enabled: bool | None = Form(None),
    tee_box_roi: str | None = Form(None),  # JSON string {"x":N,"y":N,"w":N,"h":N}
    note: str | None = Form(None),
    course_id: int | None = Form(None),
    assigned_hole: int | None = Form(None),
    assigned_role: str | None = Form(None),
    ball_side: str | None = Form(None),
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
    if triggering_enabled is not None:
        cam.triggering_enabled = bool(triggering_enabled)
    if note is not None:
        cam.note = note.strip() or None
    if ball_side is not None:
        _bs = (ball_side or "").strip().lower()
        if _bs in ("", "none", "auto"):
            cam.ball_side = None
        elif _bs in ("left", "right"):
            cam.ball_side = _bs
        else:
            raise HTTPException(400, "ball_side must be left, right or auto")
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
        db.add(
            AuditLog(
                actor="admin",
                action="move_camera",
                target=f"camera:{cam.id}",
                detail=(
                    f"course={cam.course_id} hole={cam.assigned_hole} "
                    f"role={cam.assigned_role}"
                    + (" (auto-unpaired)" if auto_unpaired else "")
                ),
            )
        )
    db.commit()
    db.refresh(cam)
    result = _camera_to_dict(cam, None, _course_name_for(db, cam))
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
    db.add(
        AuditLog(
            actor="admin",
            action="delete_camera",
            target=f"camera:{cam.id}",
            detail=f"course={cam.course_id} hole={cam.assigned_hole} role={cam.assigned_role}",
        )
    )
    db.delete(cam)
    db.commit()
    return {"deleted": True, "camera_id": camera_id}


# ── Debug2 ─────────────────────────────────────────────────────────────
# The operator's pipeline, separate from produce and from the original
# Debug: pose candidates -> impact + ball from the club arc -> AI judge on
# the heat composite -> windowed MOG2 heat -> a chain walked upward from
# the ball. Read-only: it writes images and returns a report, and touches
# neither edit_metrics nor any produced clip.

@router.post("/long-uploads/{upload_id}/debug2")
def debug2_start(upload_id: int):
    """Start Debug2. Poll /debug2/status — see _debugx_start for why this
    cannot run inside the request."""
    return _debugx_start("debug2", upload_id, _debug2_run)


@router.get("/long-uploads/{upload_id}/debug2/status")
def debug2_status(upload_id: int):
    # Same nan guard as debug3 — same numpy-built report, same failure.
    return _json_safe(_debugx_get("debug2", upload_id))


def _debug2_run(row, src_path, db, progress=None):
    upload_id = row.id
    from ..services import debug2 as d2
    from ..services import pose_swing
    from ..services.tracer import swing_heat_check

    tok = secrets.token_hex(3)
    fps = float(probe_fps(src_path) or 30.0)

    def _clip_url(name):
        """Filename under uploads/clips -> public URL. `_public_url` is a
        nested helper elsewhere and takes a Path, so this endpoint has its
        own. Named unambiguously: a short name here was shadowed by a
        tuple unpack further down and the helper became None mid-run."""
        if not name:
            return None
        p = CLIPS_DIR / name
        if not p.exists():
            return None
        return (
            f"{settings.app_base_url}/uploads/clips/{p.name}"
            f"?v={int(p.stat().st_mtime)}"
        )

    # Frame size: the height sets the ball scale the chain rules are
    # written in, and both are needed to map the AI's percentage points.
    _fw, _fh = 1280, 720
    try:
        import cv2 as _cv2

        _c = _cv2.VideoCapture(str(src_path))
        _fw = int(_c.get(_cv2.CAP_PROP_FRAME_WIDTH) or 1280) or 1280
        _fh = int(_c.get(_cv2.CAP_PROP_FRAME_HEIGHT) or 720) or 720
        _c.release()
    except Exception:  # noqa: BLE001
        pass
    rep: dict = {
        "ok": True, "upload_id": upload_id, "fps": round(fps, 2),
        "stages": [], "swings": [],
    }

    # 1. CANDIDATES — the same pose detector produce uses.
    if not pose_swing.available():
        return {
            "ok": False,
            "error": f"pose detector unavailable: {pose_swing.unavailable_reason()}",
        }
    try:
        db.rollback()          # the passes below are slow; don't hold a txn
    except Exception:  # noqa: BLE001
        pass
    # detect_swings_from_pose returns the SEGMENT LIST directly; the
    # per-burst diagnostics come back through the `debug` dict it fills in.
    pose_dbg: dict = {}
    cands = list(
        pose_swing.detect_swings_from_pose(src_path, fps=fps, debug=pose_dbg)
        or []
    )
    rep["pose_debug"] = {
        k: pose_dbg.get(k)
        for k in ("n_pose_frames", "n_samples", "coverage",
                  "n_bend_rejected", "back_bend_min_deg")
    }
    # Every burst the detector saw and what happened to it — the stage-1
    # working, including the ones that never became candidates.
    rep["bursts"] = list(pose_dbg.get("bursts_detail") or [])
    rep["stages"].append({
        "n": 1, "name": "Pose candidates",
        "detail": (
            f"{len(cands)} burst(s) passed the wrist-speed and spine-bend "
            f"gates"
            + (
                f" · {pose_dbg['n_bend_rejected']} rejected as upright"
                if pose_dbg.get("n_bend_rejected") else ""
            )
        ),
        "count": len(cands), "counts": "candidates",
    })

    if progress:
        progress("Scanning for the ball at rest", 0, len(cands))
    rest = _rest_ball_departures(src_path, fps, db, row)
    rep["rest_ball"] = {
        "reason": rest.get("reason"), "roi": rest.get("roi"),
        "departures": rest.get("deps"),
    }

    n_judged_out = 0
    for i, c in enumerate(cands):
        if progress:
            progress(f"Swing {i + 1} of {len(cands)}: finding the ball at impact",
                     i, len(cands))
        peak_t = float(c.get("peak_time_sec") or 0.0)
        imp_f = int(round(peak_t * fps))
        entry: dict = {
            "idx": i, "peak_time_sec": round(peak_t, 2),
            "impact_frame": imp_f,
            "back_bend_deg": c.get("back_bend_deg"),
            "ratio": c.get("ratio"),
            "wrist_xy": c.get("impact_wrist_xy"),
        }

        # 2. IMPACT + BALL from the bottom of the club's heat arc.
        club = d2.club_bottom_ball(
            src_path, imp_f, fps,
            hint_xy=c.get("impact_wrist_xy"),
            feet_xy=c.get("impact_feet_xy"),
            head_xy=c.get("impact_head_xy"),
            debug_dir=CLIPS_DIR,
            debug_prefix=f"d2club-{upload_id}-{tok}-{i}",
        )
        _resolve_ball(entry, rest, peak_t, club)
        entry["ball_side"] = club.get("side")
        entry["ball_offset_body"] = club.get("offset_body")
        entry["ball_image_url"] = _clip_url(club.get("image"))
        # Everything downstream -- the AI corridor, the band search, the
        # chain walks, the drawing -- must agree on ONE ball, or the aim
        # tests are measured against a point the panel never shows.
        _ball = entry.get("ball")

        # 3. AI JUDGE on the motion-heat composite.
        chk = swing_heat_check(
            src_path, peak_t, fps,
            ball_hint=c.get("impact_wrist_xy"),
            debug_dir=CLIPS_DIR,
            debug_prefix=f"d2heat-{upload_id}-{tok}-{i}",
        )
        entry["heat_image_url"] = _clip_url(
            chk.get("image") or chk.get("image_clean"),
        )
        verdict, reason = chk.get("verdict"), "club-fan heuristic (no API key)"
        if chk.get("image_clean") and os.environ.get("ANTHROPIC_API_KEY"):
            j = judge_swing_heat_image(CLIPS_DIR / chk["image_clean"])
            if j.get("is_swing") is True:
                verdict, reason = "swing", j.get("reason") or "AI judge: swing"
            elif j.get("is_swing") is False:
                verdict, reason = "not_swing", j.get("reason") or "AI judge: not a swing"
            else:
                reason = f"AI judge unavailable ({j.get('reason')})"
        entry["verdict"] = verdict
        entry["verdict_reason"] = reason

        if verdict == "not_swing":
            n_judged_out += 1
            rep["swings"].append(entry)
            continue

        # 4. WINDOWED MOG2 HEAT — impact-5 .. impact+100 and nothing else.
        f_lo = max(0, imp_f - d2.WIN_PRE)
        f_hi = imp_f + d2.WIN_POST
        entry["window"] = [f_lo, f_hi]
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass
        _trc_url, cv_info, _trc_path, _trc_dbg = _run_tracer(
            src_path,
            # The raw-motion heat is only WRITTEN when a debug dir is set
            # (tracer.py:690). Passing None meant stage 4 produced its dot
            # pool but no image — so the windowed heat was missing and the
            # chain had no background to draw on.
            frame_debug_dir=CLIPS_DIR,
            frame_debug_prefix=f"d2win-{upload_id}-{tok}-{i}",
            impact_frame_hint_override=imp_f,
            ball_rest_hint=(
                (float(club["xy"][0]), float(club["xy"][1]))
                if _ball else None
            ),
            heat_start_frame=f_lo,
            heat_end_frame=f_hi,
            render_video=False,
        )
        cv_info = cv_info or {}
        entry["heat_window_image_url"] = _clip_url(
            cv_info.get("raw_motion_image"),
        )
        pool = [
            p for p in _mog2_dot_pool(cv_info)
            if p.get("frame") is not None and f_lo <= int(p["frame"]) <= f_hi
        ]
        entry["n_dots"] = len(pool)

        # 4c. RE-SCAN THE CLEAN BAND WITH LOOSE GATES. The tracer's pool is
        # heavily filtered — per-frame candidate gates, a ghost-trail
        # filter, a hot-mask — all tuned to keep the ball out of the
        # golfer's body heat. Above the club fan none of that is needed:
        # the region is sky and treetops. Observed on a real swing: a run
        # of four ball dots plainly visible on the heat map was absent
        # from the pool entirely, so no chaining rule could have found it.
        # Scan that band directly and merge what it finds.
        _fan_pre = d2.fan_line_y(
            c.get("impact_head_xy"), c.get("impact_feet_xy"), _fh,
        )
        if _fan_pre is not None:
            _band = d2.scan_band(
                src_path, 0, 0, _fw, int(_fan_pre), f_lo, f_hi,
            )
            _have = {(int(p["frame"]), int(p["x"]) // 6, int(p["y"]) // 6)
                     for p in pool}
            _added = [
                b for b in _band
                if (int(b["frame"]), int(b["x"]) // 6, int(b["y"]) // 6)
                not in _have
            ]
            pool = pool + _added
            entry["n_band_scan"] = len(_band)
            entry["n_band_new"] = len(_added)
            log.info(
                "debug2: band scan above y=%d gave %d dot(s), %d new "
                "(pool %d -> %d)",
                int(_fan_pre), len(_band), len(_added),
                entry["n_dots"], len(pool),
            )

        # 4b. TRACE THE TRAIL — on the WINDOWED heat, not the swing-check
        # composite. The composite spans the whole swing, so the club fan
        # and the golfer's body dominate it and the ball's dots are a
        # faint extra; the windowed map holds impact-5..impact+100 and
        # almost nothing else, which is why the flight reads clearly on it
        # by eye. Ask the question against the picture where the answer is
        # actually visible. Falls back to the composite if the windowed
        # heat was not written.
        _heat_for_ai = cv_info.get("raw_motion_image") or chk.get("image_clean")
        if _heat_for_ai and (CLIPS_DIR / _heat_for_ai).exists():
            _tp = trace_ball_path_ai(CLIPS_DIR / _heat_for_ai)
            entry["ai_path_note"] = _tp.get("note") or _tp.get("error")
            entry["ai_path_confidence"] = _tp.get("confidence")
            entry["ai_path_source"] = (
                "windowed heat"
                if _heat_for_ai == cv_info.get("raw_motion_image")
                else "swing-check composite"
            )
            entry["ai_path"] = [
                {
                    "x": int(round(p["x_pct"] / 100.0 * _fw)),
                    "y": int(round(p["y_pct"] / 100.0 * _fh)),
                }
                for p in (_tp.get("points_pct") or [])
            ]
            if entry["ai_path"] and _ball:
                _a = entry["ai_path"][0]
                entry["ai_path_start_px"] = int(round((
                    (_a["x"] - club["xy"][0]) ** 2
                    + (_a["y"] - club["xy"][1]) ** 2
                ) ** 0.5))

        # 5. CHAIN. Lock on ABOVE THE HEAD, where the map is clean and
        # the ball's dots bunch as it slows, then walk back down toward
        # impact. Walking up from the ball starts in the worst place on
        # the map — body heat and motion blur — and one bad first link
        # poisons the rest. Falls back to the upward walk when there is no
        # head position or nothing above it to lock onto.
        _head = c.get("impact_head_xy")
        _head_y = float(_head[1]) if _head and len(_head) == 2 else None
        entry["head_y"] = _head_y
        _tries: list = []
        ch, method = None, None
        # BEST FIRST: the trail the AI traced. It knows the path's SHAPE
        # but has no frame numbers; MOG2 has exact frames but cannot tell
        # the ball from foliage. Intersecting them gives a path that is
        # both correctly shaped and correctly timed. Measured on a
        # synthetic flight buried in 120 noise dots: 100% of the flight
        # kept, none of the noise.
        _aim_gate = max(120.0, 0.20 * float(_fw))
        if entry.get("ai_path"):
            _c1 = d2.chain_along_ai_path(
                pool, entry["ai_path"], _ball, imp_f, _fh,
            )
            # The AI's trail gets NO free pass. It once traced a roughly
            # horizontal line across the treetops; the corridor collected
            # 43 dots along it and the real rising flight was discarded as
            # "off the trail". So its result must clear the same two tests
            # as everything else — it has to RISE, and it has to point back
            # at the ball — or we fall through to the thirds search, which
            # would have found the real chain.
            _rise = _c1.get("rise_px")
            _aim = _c1.get("aim_px")
            _ok = len(_c1.get("points") or []) >= 3
            if _ok and (_rise is None or _rise < 30):
                _ok = False
                _c1["reason"] += " — REJECTED: does not rise"
            if _ok and _aim is not None and _aim > _aim_gate:
                _ok = False
                _c1["reason"] += (
                    f" — REJECTED: aims {_aim:.0f}px from the ball "
                    f"(limit {_aim_gate:.0f})"
                )
            _tries.append(f"1 AI-traced trail: {_c1['reason']}")
            if _ok:
                ch, method = _c1, "1 · dots lying on the AI-traced trail"
        if ch is None:
            # THIRDS. Above the club fan the map is nearly empty, so split
            # that band into middle/right/left and hunt each for the
            # signature: 3+ dots in a straight line, higher meaning later,
            # whose line extended down points back at the ball. Every
            # third is searched and the best wins, so a fade or a pull is
            # not missed by the ordering.
            _fan_y = d2.fan_line_y(_head, c.get("impact_feet_xy"), _fh)
            entry["fan_y"] = _fan_y
            _c15 = d2.chain_by_thirds(
                pool, _ball, imp_f, _fh, _fw, _fan_y,
            )
            entry["bands"] = _c15.get("thirds")
            _tries.append(f"2 L/M/R bands above the club fan: {_c15['reason']}")
            if _c15.get("points"):
                ch, method = _c15, (
                    f"2 · straight run in the {_c15.get('zone')} band "
                    f"above the club fan"
                )
        if ch is None:
            _c2 = d2.chain_above_head(
                pool, _ball, imp_f, _fh, _head_y,
            )
            _tries.append(f"3 above-head lock-on: {_c2['reason']}")
            if _c2.get("points"):
                ch, method = _c2, "3 · above-head lock-on, walked back down"
        if ch is None:
            ch = d2.chain_from_ball(pool, _ball, imp_f, _fh)
            _tries.append(f"4 up-from-ball walk: {ch['reason']}")
            method = "4 · up-from-ball walk (last resort)"
        entry["chain_method"] = method
        entry["chain_tries"] = _tries
        entry["aim_px"] = ch.get("aim_px")
        entry["seed_frames"] = ch.get("seed")
        entry["chain"] = ch["points"]
        entry["chain_reason"] = ch["reason"]
        entry["n_rejected"] = len(ch["rejected"])
        # A count of rejects is not a diagnosis. Group them by reason so a
        # short chain says WHY it stopped growing.
        _why: dict = {}
        for rj in ch["rejected"]:
            _why[rj["why"]] = _why.get(rj["why"], 0) + 1
        entry["rejected_why"] = sorted(
            ({"why": k, "n": v} for k, v in _why.items()),
            key=lambda r: -r["n"],
        )
        _heat = cv_info.get("raw_motion_image")
        if _heat and (CLIPS_DIR / _heat).exists():
            name = f"d2chain-{upload_id}-{tok}-{i}.jpg"
            if d2.draw_chain(
                CLIPS_DIR / _heat, _ball, ch["points"],
                ch.get("rejected") or [], CLIPS_DIR / name,
                f"chain f{f_lo}-{f_hi}: {ch['reason']}  "
                f"(green=chain, magenta x=where it aims back to, "
                f"red x=rejected)",
                head_y=_head_y,
                aim_xy=ch.get("aim_xy"),
                ai_path=entry.get("ai_path"),
                # So a rejected AI trail is drawn as rejected. Method "1" is
                # the AI corridor; anything else means the trail lost.
                ai_used=str(method or "").startswith("1"),
                fan_y=entry.get("fan_y"),
                frame_w=_fw,
            ):
                entry["chain_image_url"] = _clip_url(name)
        rep["swings"].append(entry)

    n_real = sum(1 for s in rep["swings"] if s.get("verdict") != "not_swing")
    rep["stages"].extend([
        {"n": 2, "name": "Impact + ball from the club arc",
         "detail": "impact = peak wrist speed; ball = the bottom of the "
                   "club's heat arc through impact",
         "count": sum(1 for s in rep["swings"] if s.get("ball")),
         "counts": "balls located"},
        {"n": 3, "name": "AI judge on the heat composite",
         "detail": f"{n_real} kept, {n_judged_out} rejected as not a swing",
         "count": n_real, "counts": "real swings kept"},
        {"n": 4, "name": "Windowed MOG2 heat",
         "detail": f"impact−{d2.WIN_PRE} .. impact+{d2.WIN_POST} only",
         "count": sum(s.get("n_dots") or 0 for s in rep["swings"]),
         "counts": "motion dots in the window"},
        {"n": 5, "name": "Chain walked up from the ball",
         "detail": "each step advances in frame, rises while ascending, "
                   "and drifts sideways far less than it rises",
         "count": sum(len(s.get("chain") or []) for s in rep["swings"]),
         "counts": "tracer points linked"},
    ])
    return rep




# ── shared: the resting ball, from the detector the original Debug uses ──
# club_bottom_ball infers the ball from the club-arc vertex. That is an
# indirect argument and it is only as good as the arc. detect_swings_from_ball
# looks for the thing itself: a small, white, round blob that sits STILL for
# most of a second inside the course's hand-drawn tee-box ROI, and then
# leaves. When it fires it is the better answer, so Debug2 and Debug3 both
# prefer it and keep the club arc as the fallback.

def _rest_ball_departures(src_path, fps: float, db, row) -> dict:
    """Every resting-ball departure in this video, with the ROI used."""
    out = {"deps": [], "roi": None, "reason": None}
    try:
        course = db.get(Course, row.course_id) if row.course_id else None
        out["roi"] = course.ball_roi if course else None
    except Exception:  # noqa: BLE001
        pass
    dbg: dict = {}
    try:
        detect_swings_from_ball(src_path, fps=fps, roi=out["roi"], debug=dbg)
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"departure detector crashed: {exc}"
        return out
    out["deps"] = list(dbg.get("departures") or [])
    out["reason"] = dbg.get("reason") or (
        f"{len(out['deps'])} resting-ball departure(s)"
        + (" (no tee-box ROI set for this course)" if not out["roi"] else "")
    )
    return out


def _departure_for(deps: list, peak_t: float, tol_sec: float = 1.5):
    """The departure nearest this candidate's peak, or (None, None).

    Tolerance is generous because the two clocks measure different things:
    the pose peak is maximum wrist speed, the departure is when the ball is
    gone for good, and the detector samples at 15Hz.
    """
    best = None
    for d in deps or []:
        dt = abs(float(d.get("t") or 0.0) - float(peak_t))
        if dt <= tol_sec and (best is None or dt < best[0]):
            best = (dt, d)
    return (best[1], best[0]) if best else (None, None)


def _resolve_ball(entry: dict, rest: dict, peak_t: float, club: dict) -> None:
    """Choose between the departure detector and the club arc, and record
    BOTH so a disagreement is visible rather than silently resolved."""
    dep, dt = _departure_for(rest.get("deps") or [], peak_t)
    club_xy = club.get("xy")
    if dep:
        entry["ball"] = [int(dep["x"]), int(dep["y"])]
        entry["ball_source"] = "resting-ball departure"
        entry["ball_reason"] = (
            f"sat still {dep.get('rest_sec')}s, left at {dep.get('t')}s "
            f"({dt:.2f}s from the pose peak)"
        )
        entry["ball_alt"] = club_xy
        entry["ball_alt_source"] = "club-arc vertex"
        entry["ball_alt_reason"] = club.get("reason")
        if club_xy:
            entry["ball_disagree_px"] = round(
                math.hypot(club_xy[0] - dep["x"], club_xy[1] - dep["y"]), 1,
            )
    else:
        entry["ball"] = club_xy
        entry["ball_source"] = "club-arc vertex (no departure matched)"
        entry["ball_reason"] = club.get("reason")


# ── Debug2/Debug3 background runner ────────────────────────────────────
# Both pipelines run pose over the whole video and then several seconds of
# per-frame work PER CANDIDATE. Measured on upload 501: pose 26s, then ~82s
# for candidate 0 alone, with 3 candidates. Held inside the HTTP request
# that overruns Replit's proxy timeout, the connection is dropped (the
# operator sees a 502 "we couldn't reach this app") and the request is
# RETRIED from the top -- so the logs show pose running four times for one
# button press and no run ever finishes. Run them on a thread and let the
# UI poll instead.

_debugx_lock = threading.Lock()
_debugx_state: dict[tuple[str, int], dict] = {}


def _debugx_set(kind: str, upload_id: int, **fields) -> None:
    with _debugx_lock:
        st = _debugx_state.setdefault((kind, upload_id), {})
        st.update(fields)


def _debugx_get(kind: str, upload_id: int) -> dict:
    with _debugx_lock:
        return dict(_debugx_state.get((kind, upload_id)) or {"running": False})


def _debugx_start(kind: str, upload_id: int, runner) -> dict:
    """Kick `runner(row, src_path, db, progress)` on a thread."""
    with _debugx_lock:
        if (_debugx_state.get((kind, upload_id)) or {}).get("running"):
            return {"ok": True, "running": True, "upload_id": upload_id,
                    "note": "already running"}
        _debugx_state[(kind, upload_id)] = {
            "running": True, "stage": "starting", "done": 0, "total": 0,
            "report": None, "error": None,
        }

    def _job() -> None:
        # Same gate the produce queue holds: an operator hitting
        # Re-Produce while a batch of uploads is producing waits its
        # turn rather than adding a fifth job to the pile.
        with _produce_gate:
            _debugx_job(kind, upload_id, runner)

    def _debugx_job(kind: str, upload_id: int, runner) -> None:
        db = SessionLocal()
        try:
            row = db.get(LongVideoUpload, upload_id)
            if not row or not row.tee_filename:
                raise RuntimeError("upload not found or has no tee video")
            storage.ensure_local(CLIPS_DIR, row.tee_filename)
            src_path = _local_tee(row)
            if not src_path.exists():
                raise RuntimeError("tee source missing on disk")

            def _prog(stage: str, done: int = 0, total: int = 0) -> None:
                _debugx_set(kind, upload_id, stage=stage, done=done,
                            total=total)

            rep = runner(row, src_path, db, _prog)
            _debugx_set(kind, upload_id, running=False, report=rep,
                        stage="done",
                        error=(rep or {}).get("error")
                        if (rep or {}).get("ok") is False else None)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s failed for upload=%s: %s", kind, upload_id, exc,
                        exc_info=True)
            _debugx_set(kind, upload_id, running=False, error=f"{exc}",
                        stage="failed")
        finally:
            db.close()

    threading.Thread(target=_job, daemon=True,
                     name=f"{kind}-{upload_id}").start()
    return {"ok": True, "running": True, "upload_id": upload_id}


# ── Debug3 ─────────────────────────────────────────────────────────────
# A third method, and a deliberately different one. Debug2 reads the shape
# the swing draws in a motion COMPOSITE. Debug3 never looks at a composite:
# it asks per frame which connected blobs are ball-sized and off-body, then
# links those detections over time and fits a ballistic curve to them.
#
#   A-C  MOG2 -> connected components -> the big ones are the golfer (a
#        mask, not a detection) -> keep only ball-sized blobs
#   D    link across frames with a constant-velocity predictor and a
#        nearest-neighbour gate
#   E    RANSAC parabola (x linear in t, y quadratic), then the same two
#        flight tests Debug2 uses: it must rise, and run back to the impact
#        frame it must land near the ball
#
# Read-only, like Debug2: writes images, returns a report, touches neither
# edit_metrics nor any produced clip.

@router.post("/long-uploads/{upload_id}/debug3")
def debug3(upload_id: int):
    """Start the blob-and-track pipeline. Poll /debug3/status for the work."""
    return _debugx_start("debug3", upload_id, _debug3_run)


def _json_safe(obj):
    """Replace non-finite floats with None, recursively.

    The debug reports are built from numpy: means, divisions, polyfits.
    Any one of those can produce nan or inf on a degenerate input, and
    json.dumps refuses them — so a single bad number 500s the whole
    status response AFTER produce has already run and written its clips.
    The operator then sees "500: Internal Server Error" for a job that
    actually succeeded, which sends them looking in the wrong place.

    Individual nan sources are worth fixing where they are found (see the
    RANSAC refit guard in debug3.pick_flight), but the boundary should
    not be able to fail on the next one.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


@router.get("/long-uploads/{upload_id}/debug3/status")
def debug3_status(upload_id: int):
    return _json_safe(_debugx_get("debug3", upload_id))



# ── produce queue ──────────────────────────────────────────────────────
# Produce runs ONE AT A TIME, oldest upload first.
#
# Every path used to spawn its own unbounded thread, so four uploads
# landing together meant four jobs racing. The work is CPU- and
# memory-bound and cv2/ffmpeg release the GIL, so they genuinely compete
# — and debug3's own note says a single MOG2 pass at 1080p allocates
# ~25MB per frame, "enough to get a container killed mid-request, which
# surfaces as a 502 with no log line of its own". Multiply that by four
# on one container and the process dies, taking the web server and every
# in-flight upload with it. Serialised, four jobs also finish SOONER in
# wall-clock than four thrashing the same two cores.
#
# A semaphore would bound concurrency but not ordering — Python makes no
# promise about wake order — so this is a real priority queue keyed on
# the upload's created_at. Whoever was uploaded first produces first,
# regardless of which request happened to finish enqueuing first.
_produce_q: "queue.PriorityQueue[tuple]" = queue.PriorityQueue()
_produce_seq = itertools.count()
_produce_lock = threading.Lock()
_produce_pending: dict[int, dict] = {}   # upload_id -> task, for dedupe
_produce_running: int | None = None
_produce_worker: threading.Thread | None = None
# Held for the duration of a produce. The queue worker takes it, and so
# does Debug3 / Re-Produce (which run on their own thread so the panel
# can poll status) — so nothing produces concurrently, whatever started
# it, even though only queued uploads are ordered among themselves.
_produce_gate = threading.Lock()


def _produce_worker_loop() -> None:
    global _produce_running
    while True:
        try:
            _key, _seq, task = _produce_q.get()
        except Exception:  # noqa: BLE001
            continue
        uid = task["upload_id"]
        try:
            with _produce_lock:
                _produce_pending.pop(uid, None)
                _produce_running = uid
            waiting = _produce_q.qsize()
            log.info(
                "produce queue: starting upload %s (%d still waiting)",
                uid, waiting,
            )
            with _produce_gate:
                task["result"] = run_produce_job(
                    upload_id=uid,
                    hole_number=task.get("hole_number"),
                    debug_artifacts=task.get("debug_artifacts", False),
                )
        except Exception as exc:  # noqa: BLE001
            # A crash must never kill the worker — the whole queue would
            # stall silently behind it.
            log.exception("produce queue: upload %s crashed: %s", uid, exc)
            task["result"] = {"ok": False, "error": f"{exc}"}
        finally:
            with _produce_lock:
                _produce_running = None
            task["done"].set()
            _produce_q.task_done()


def _ensure_produce_worker() -> None:
    global _produce_worker
    with _produce_lock:
        if _produce_worker is not None and _produce_worker.is_alive():
            return
        _produce_worker = threading.Thread(
            target=_produce_worker_loop, daemon=True, name="produce-queue",
        )
        _produce_worker.start()


def enqueue_produce_job(
    upload_id: int,
    hole_number: int | None = None,
    debug_artifacts: bool = False,
    wait: bool = False,
    db=None,
) -> dict:
    """Queue an upload to be produced. Returns immediately unless `wait`.

    Ordering is by the upload's `created_at`, so a batch produces oldest
    first even if the requests finished out of order. `wait=True` blocks
    until this job is done — the camera path needs it, because it stamps
    the CameraEvent's terminal status from the job's outcome.

    Re-queuing an upload that is already queued is a no-op; re-queuing
    one that is currently RUNNING queues it again, since that is a
    genuine request to redo it with whatever changed.
    """
    # Sort key from the upload's own creation time, not from now.
    created = None
    try:
        _db = db or SessionLocal()
        try:
            _row = _db.get(LongVideoUpload, upload_id)
            created = getattr(_row, "created_at", None)
        finally:
            if db is None:
                _db.close()
    except Exception as exc:  # noqa: BLE001
        log.warning("produce queue: could not read created_at for %s: %s",
                    upload_id, exc)
    key = (created or _utcnow_naive()).timestamp()

    with _produce_lock:
        existing = _produce_pending.get(upload_id)
        if existing is not None:
            log.info("produce queue: upload %s already queued", upload_id)
            if wait:
                pass          # fall through and wait on the existing task
            else:
                return {"queued": True, "duplicate": True,
                        "position": _produce_q.qsize()}
            task = existing
        else:
            task = {
                "upload_id": upload_id,
                "hole_number": hole_number,
                "debug_artifacts": debug_artifacts,
                "done": threading.Event(),
                "result": None,
            }
            _produce_pending[upload_id] = task
            _produce_q.put((key, next(_produce_seq), task))

    _ensure_produce_worker()
    position = _produce_q.qsize()
    log.info(
        "produce queue: upload %s queued (%d ahead, running=%s)",
        upload_id, max(0, position - 1), _produce_running,
    )
    if wait:
        task["done"].wait()
        return task.get("result") or {"ok": False, "error": "no result"}
    return {"queued": True, "duplicate": False, "position": position}


def produce_queue_status() -> dict:
    """What the queue is doing — surfaced so a card sitting at 'pending'
    can say it is waiting its turn rather than looking stuck."""
    with _produce_lock:
        return {
            "running_upload_id": _produce_running,
            "queued_upload_ids": sorted(_produce_pending.keys()),
            "depth": _produce_q.qsize(),
        }


def run_wizard_produce_job(
    upload_id: int,
    ball_xy,
    impact_frame: int,
    hole_number: int | None = None,
    landing_frame: int | None = None,
    landing_spot=None,
) -> dict:
    """Stages 4-8, from the operator's ball and impact frame.

    THE EDIT WIZARD'S PRODUCE. Stages 1-3 exist to answer two questions:
    where is the ball at rest, and which frame is impact. When an operator
    has opened the wizard they have already answered both, by eye, better
    than the detector can — so this skips straight to the part that turns
    those two numbers into a clip.

    It is not a second pipeline. `find_flight` is the same function
    Debug3 and produce call, and it already accepts `impact_frame` and
    `rest_ball` from a caller; `_d3_fast_produce` is the same renderer
    stage 8 uses. This only packs the operator's answer into the shape
    those two already expect, which is why the wizard cannot drift from
    produce the way the old per-engine paths did.

    Synchronous. The HTTP handler runs it on a thread and returns at once
    so the wizard can close.
    """
    db = SessionLocal()
    try:
        row = db.get(LongVideoUpload, upload_id)
        if row is None:
            return {"ok": False, "error": f"upload {upload_id} not found"}
        if row.tee_filename:
            storage.ensure_local(CLIPS_DIR, row.tee_filename)
        if row.green_filename:
            storage.ensure_local(CLIPS_DIR, row.green_filename)
        src_path = CLIPS_DIR / row.tee_filename if row.tee_filename else None
        if not src_path or not src_path.exists():
            return {"ok": False,
                    "error": f"tee source missing on disk: {row.tee_filename}"}

        def _prog(stage: str, done: int = 0, total: int = 0) -> None:
            _debugx_set("produce", upload_id, stage=stage, done=done,
                        total=total, running=True)

        _debugx_set("produce", upload_id, stage="Starting", done=0, total=0,
                    running=True, error=None)
        try:
            from ..services import debug3 as d3

            fps = float(probe_fps(src_path) or 30.0)
            _fw, _fh = _probe_frame_size(src_path)
            _bx, _by = float(ball_xy[0]), float(ball_xy[1])
            _imp = int(impact_frame)

            # The shape club_bottom_ball returns, so find_flight treats the
            # operator's ball exactly as it treats a measured one — arming
            # the aim gate rather than leaving it dead.
            _rest = {
                "ok": True, "xy": [_bx, _by],
                "reason": "placed by the operator in the edit wizard",
            }
            _prog("Tracking the ball and fitting the flight", 0, 1)
            _ff = d3.find_flight(
                src_path, fps, impact_frame=_imp,
                frame_w=_fw, frame_h=_fh,
                ball_side=_wizard_ball_side(db, row),
                rest_ball=_rest,
                # THE OPERATOR PLACED IT. Without this the club-arc pass
                # re-measures the ball at the launch frame and its answer
                # wins -- so the placement moved, sometimes by tens of
                # pixels, and the tracer started somewhere the operator
                # did not put it. A control that gets silently overruled
                # is worse than no control.
                ball_locked=True,
            )
            if not _ff.get("ok"):
                _err = _ff.get("reason") or "no flight found"
                log.info("wizard produce: upload=%s no flight (%s)",
                         upload_id, _err)
                # NO FLIGHT IS WHEN THE OPERATOR NEEDS THE DOTS MOST.
                # The detections were computed either way; dropping them
                # left click-to-plot empty and the swing unrecoverable
                # except by another full produce. Save the pool, the
                # ball and the impact frame so the operator can plot the
                # flight by hand from this failure.
                _cands = _ff.get("candidates") or []
                if _cands:
                    try:
                        _d3_save_swing(db, row.id, 0, {
                            "idx": 0,
                            "ball": {"x": int(round(_bx)),
                                     "y": int(round(_by))},
                            "ball_manual": True,
                            "impact_frame": _imp,
                            "track_frame_width": _fw,
                            "track_frame_height": _fh,
                            "track_scaled_from_cut": False,
                            "cand_points": [
                                {"frame": int(c["frame"]),
                                 "x": int(round(float(c["x"]))),
                                 "y": int(round(float(c["y"])))}
                                for c in _cands
                            ][:1500],
                            "flight_error": _err,
                        }, 0.0)
                        log.info(
                            "wizard produce: upload=%s kept %d candidate "
                            "dots so click-to-plot can rescue it",
                            upload_id, min(len(_cands), 1500),
                        )
                    except Exception as _exc:  # noqa: BLE001
                        log.warning(
                            "wizard produce: could not keep the candidate "
                            "dots for upload %s: %s", upload_id, _exc)
                _finish_wizard_produce(db, upload_id, ok=False, n_ok=0,
                                       error=_err)
                return {"ok": False, "error": _err}

            # `_d3_fast_produce` reads swings off a Debug3 report, so hand
            # it one with the single swing the operator pointed at.
            rep = {"swings": [{
                "idx": 0,
                # Belt and braces: find_flight is locked to the operator's
                # ball above, so this is the same value -- but the clip is
                # rendered from THIS number, and it must not be able to
                # drift from what the operator set.
                "ball": [_bx, _by],
                "launch_frame": _ff.get("launch_frame"),
                "flight": _ff.get("points") or [],
                "impact_frame": _imp,
                # The clickable pool. Without it _d3_fast_produce writes
                # cand_points empty, and click-to-plot opens on a swing
                # with nothing to click -- read-only exactly where the
                # operator went to correct the tracer by hand.
                "candidates": _ff.get("candidates") or [],
            }]}
            # The landing frame is a GREEN frame index; the renderer
            # wants green-clock SECONDS, because every boundary in the
            # composite is kept in seconds so the two cameras' different
            # frame rates cannot introduce drift. The clip ends
            # LANDING_TAIL_SEC after the ball touches down.
            _end_sec = None
            _land_sec = None
            if landing_frame is not None:
                _gp = _local_green(row)
                _gfps = float(probe_fps(_gp) or 0.0) if _gp else 0.0
                if _gfps > 0:
                    _land_sec = float(landing_frame) / _gfps
                    _end_sec = _land_sec + LANDING_TAIL_SEC
                else:
                    log.warning(
                        "wizard produce: upload=%s has a landing frame but "
                        "no readable green fps — using the default length",
                        upload_id,
                    )
            _prog("Rendering the clip", 0, 1)
            out = _d3_fast_produce(
                row, src_path, db, rep, fps,
                progress=_prog, hole_number=hole_number,
                end_green_sec=_end_sec,
                landing=(
                    {"sec": _land_sec, "xy": list(landing_spot)}
                    if (_land_sec is not None and landing_spot) else None
                ),
            )
            log.info(
                "wizard produce: upload=%s ball=(%.0f,%.0f) impact=f%d -> "
                "ok=%s clips=%d", upload_id, _bx, _by, _imp,
                out.get("ok"), len(out.get("clips") or []),
            )
            _n_ok = len(out.get("clips") or [])
            _finish_wizard_produce(
                db, upload_id,
                ok=bool(out.get("ok")) and _n_ok > 0,
                n_ok=_n_ok,
                error=out.get("error"),
            )
            return out
        finally:
            _debugx_set("produce", upload_id, running=False, stage="done")
    except Exception as exc:  # noqa: BLE001
        log.exception("wizard produce %s crashed: %s", upload_id, exc)
        _debugx_set("produce", upload_id, running=False, stage="failed",
                    error=f"{exc}")
        _finish_wizard_produce(db, upload_id, ok=False, n_ok=0,
                               error=f"{exc}")
        return {"ok": False, "error": f"{exc}"}
    finally:
        db.close()


def _finish_wizard_produce(db, upload_id: int, ok: bool, n_ok: int,
                           error=None) -> None:
    """Release the row the wizard's produce claimed.

    A card greyed by `processing_status = "processing"` stays greyed until
    something sets a terminal status. Every exit path from the job goes
    through here -- success, no-flight, and crash -- because the one that
    does not is the one that leaves an upload looking permanently stuck.
    """
    try:
        row = db.get(LongVideoUpload, upload_id)
        if row is None:
            return
        row.processing_status = "completed" if ok else "failed"
        row.processing_completed_at = _utcnow_naive()
        row.last_error = None if ok else (str(error)[:500] if error else None)
        if ok:
            row.last_n_segments = n_ok
            row.last_n_succeeded = n_ok
        db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        log.warning("wizard produce: could not release upload %s: %s",
                    upload_id, exc)


def _wizard_ball_side(db, row) -> str | None:
    """Which side of the golfer the ball sits on, from the tee camera.
    Same lookup Debug3 does; None means search both sides."""
    try:
        if not getattr(row, "camera_event_id", None):
            return None
        ev = db.query(CameraEvent).filter(
            CameraEvent.id == row.camera_event_id,
        ).first()
        if ev is None or not ev.tee_camera_id:
            return None
        cam = db.get(Camera, ev.tee_camera_id)
        return getattr(cam, "ball_side", None) if cam else None
    except Exception as exc:  # noqa: BLE001
        log.debug("wizard produce: no ball_side for upload %s: %s",
                  getattr(row, "id", None), exc)
        return None


def _probe_frame_size(src_path) -> tuple[int, int]:
    """(width, height) of the source, defaulting to 1280x720."""
    try:
        import cv2 as _cv2

        cap = _cv2.VideoCapture(str(src_path))
        w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH) or 1280) or 1280
        h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT) or 720) or 720
        cap.release()
        return w, h
    except Exception:  # noqa: BLE001
        return 1280, 720


def run_produce_job(
    upload_id: int,
    hole_number: int | None = None,
    debug_artifacts: bool = False,
) -> dict:
    """Produce an upload. THE entry point — every caller uses this.

    Wraps `_debug3_run` with the row/source plumbing each caller was
    otherwise repeating: rehydrate the source from object storage,
    resolve it on disk, own a DB session, and never let an exception
    escape into a request handler or a camera callback.

    Synchronous. Callers that must not block (HTTP handlers) run it on a
    thread; the camera path deliberately blocks so it can stamp the
    CameraEvent's terminal status in one pass.

    Returns the Debug3 report, or {"ok": False, "error": ...}.
    """
    db = SessionLocal()
    try:
        row = db.get(LongVideoUpload, upload_id)
        if row is None:
            return {"ok": False, "error": f"upload {upload_id} not found"}
        if row.tee_filename:
            storage.ensure_local(CLIPS_DIR, row.tee_filename)
        if row.green_filename:
            storage.ensure_local(CLIPS_DIR, row.green_filename)
        src_path = CLIPS_DIR / row.tee_filename if row.tee_filename else None
        if not src_path or not src_path.exists():
            _err = f"tee source missing on disk: {row.tee_filename}"
            try:
                row.processing_status = "failed"
                row.processing_completed_at = _utcnow_naive()
                row.last_error = _err
                db.commit()
            except Exception:  # noqa: BLE001
                db.rollback()
            return {"ok": False, "error": _err}
        # Report the same stages Debug3's panel shows, so an auto-produce
        # card can say "Finding swing candidates" instead of just going
        # grey for three minutes. Same store the Debug3 poll reads, under
        # a "produce" kind — /long-uploads serves it to the card.
        def _prog(stage: str, done: int = 0, total: int = 0) -> None:
            _debugx_set("produce", upload_id, stage=stage, done=done,
                        total=total, running=True)

        _debugx_set("produce", upload_id, stage="Starting", done=0, total=0,
                    running=True, error=None)
        try:
            return _debug3_run(
                row, src_path, db,
                progress=_prog,
                debug_artifacts=debug_artifacts,
                hole_number=hole_number,
            )
        finally:
            # Clear it whatever happened, or the card keeps showing the
            # last stage it reached after the run is over.
            _debugx_set("produce", upload_id, running=False, stage="done")
    except Exception as exc:  # noqa: BLE001
        log.exception("produce job %s crashed: %s", upload_id, exc)
        _debugx_set("produce", upload_id, running=False, stage="failed",
                    error=f"{exc}")
        return {"ok": False, "error": f"{exc}"}
    finally:
        db.close()


# How long a d3clip-* preview stays on disk. They exist to be looked at
# in the Debug3 panel during the run that made them; nothing links to one
# afterwards, and a stale one is only ever a confusing older render.
D3_PREVIEW_MAX_AGE_HOURS = 6


def sweep_d3_previews(current_upload_id: int | None = None) -> int:
    """Delete stale `d3clip-*.mp4` previews from CLIPS_DIR.

    Stage 6 of a Debug3 run writes one preview clip PER SWING, named
    d3clip-<upload>-<token>-<i>.mp4, with a fresh random token every run
    — so re-running Debug3 on one upload left the previous run's files
    behind forever. They are pure diagnostics (the shipped clip comes
    from the produce path), nothing references them once the panel is
    closed, and each is a full H.264 encode, so on a busy day they were
    the largest thing in the directory.

    Two rules, both safe to run at any time:
      * anything older than D3_PREVIEW_MAX_AGE_HOURS goes;
      * every preview belonging to `current_upload_id` goes regardless
        of age, because the run about to start supersedes them.

    Returns the number of files removed. Never raises — a sweep that
    cannot delete must not take a Debug3 run or a boot down with it.
    """
    removed = 0
    try:
        cutoff = time.time() - D3_PREVIEW_MAX_AGE_HOURS * 3600
        prefix = (
            f"d3clip-{int(current_upload_id)}-"
            if current_upload_id is not None else None
        )
        for path in CLIPS_DIR.glob("d3clip-*.mp4"):
            try:
                superseded = bool(prefix and path.name.startswith(prefix))
                if not superseded and path.stat().st_mtime >= cutoff:
                    continue
                path.unlink(missing_ok=True)
                removed += 1
            except OSError as exc:
                log.warning("d3clip sweep: could not remove %s: %s",
                            path.name, exc)
        if removed:
            log.info(
                "d3clip sweep: removed %d stale preview(s) (older than %dh"
                "%s)", removed, D3_PREVIEW_MAX_AGE_HOURS,
                f", plus upload {current_upload_id}'s previous run"
                if current_upload_id is not None else "",
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("d3clip sweep failed: %s", exc)
    return removed


# ── Debug3 fast produce ────────────────────────────────────────────────
# The clip's shape, in SECONDS. Frames are a rendering detail; every
# boundary that has to line up with the real world is kept in float
# seconds so it survives the tee and green cameras running at different
# rates (49.76fps on the tee here — a frame is 20.1ms, and rounding that
# to whole frames is a fifth of a frame of drift per conversion).
D3_PRE_ROLL_SEC = 2.0        # lead-in before the strike
D3_POST_TRACER_SEC = 1.5     # tee tail after the tracer line stops
# Green-side coverage after the cutover. 6s rather than 4: at 4 the ball
# had barely settled before the clip ended, and the landing is the payoff
# — the tee half is the swing, the green half is the result. One constant
# drives all three consumers (the audio bed rendered under the green
# half, the green cut itself, and the splice), so they cannot drift.
D3_GREEN_SEC = 6.0
# Floor for an operator-trimmed green side. Below about a second the cut
# reads as a glitch rather than a shot landing, and an end frame that
# lands before the cutover would otherwise ask for a negative duration.
D3_MIN_GREEN_SEC = 1.0
# How long the clip runs past the landing when the operator marked one.
# Long enough to see the ball settle, short enough that it does not sit
# on an empty green. The wizard mirrors this so its copy matches the cut.
LANDING_TAIL_SEC = 1.5


def _d3_green_delta_sec(db, row) -> tuple[float, str]:
    """(green_start − tee_start) in seconds, and where it came from.

    This is the whole sync story. When both cameras stamp the real-world
    instant their recording began, the offset between the two timelines
    is a datetime subtraction — microsecond resolution, no frame
    quantisation, and correct even when the two cameras run at different
    frame rates. A positive delta means green started LATER, so a given
    real instant sits EARLIER in the green file:

        green_time = tee_time − delta

    Sources, best first:
      * `camera_event`  — the wall clocks. The real answer.
      * `edit_metrics`  — an offset a previous run or the operator
                          established for this upload.
      * `assumed_zero`  — no better information: treat the two files as
                          having started together, which is exactly what
                          the rest of the pipeline already defaults to
                          (`tee_green_delta_sec: float = 0.0`).

    The fallback matters: a manually uploaded pair has no camera event,
    and refusing to cut without one means dual-camera uploads silently
    ship tee-only. Returning the source alongside the number lets the
    caller log it and the panel show whether sync was MEASURED or
    ASSUMED, so a visibly wrong cut points at the offset rather than at
    the code."""
    if getattr(row, "camera_event_id", None):
        ev = db.get(CameraEvent, row.camera_event_id)
        if ev is not None:
            t_tee = ev.tee_recording_started_at
            t_green = ev.green_recording_started_at
            if t_tee is not None and t_green is not None:
                return (t_green - t_tee).total_seconds(), "camera_event"
    try:
        _saved = (row.edit_metrics or {}).get("tee_green_delta_sec")
        if _saved is not None:
            return float(_saved), "edit_metrics"
    except (TypeError, ValueError):
        pass
    return 0.0, "assumed_zero"


def _d3_plot_background(src_path: Path, frame_idx: int, name: str):
    """Write the frame the click-to-plot editor draws its dots over.

    The editor needs a canvas in the SAME pixel space as the points; the
    frame at launch is the natural one (it shows the scene rather than
    an accumulation of it). Returns the filename, or None."""
    try:
        import cv2 as _cv

        cap = _cv.VideoCapture(str(src_path))
        cap.set(_cv.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        out = CLIPS_DIR / name
        if not _cv.imwrite(str(out), frame,
                           [int(_cv.IMWRITE_JPEG_QUALITY), 88]):
            return None
        return name
    except Exception as exc:  # noqa: BLE001
        log.warning("d3 produce: plot background failed: %s", exc)
        return None


def _d3_save_swing(db, upload_id: int, idx: int, rec: dict,
                   delta_sec: float) -> None:
    """Merge one swing record into the upload's edit_metrics.

    Read-modify-write against a fresh row so a produce that ran for
    minutes doesn't clobber an operator's concurrent click-to-plot save
    with a stale copy. An operator-placed ball (ball_manual) is never
    overwritten — same rule _persist_swing_track follows."""
    try:
        r2 = db.get(LongVideoUpload, upload_id)
        if r2 is None:
            return
        em = dict(r2.edit_metrics or {})
        # Keep the offset that produced this cut, so a later run (or an
        # operator wondering why the green half is early) can see it.
        em["tee_green_delta_sec"] = round(float(delta_sec), 4)
        swings = [s for s in (em.get("swings") or []) if isinstance(s, dict)]
        prior = next(
            (s for s in swings if int(s.get("idx", -1)) == idx), None,
        )
        merged = dict(prior or {})
        merged.update(rec)
        if prior and prior.get("ball_manual"):
            merged["ball"] = prior.get("ball", rec.get("ball"))
            merged["ball_manual"] = True
        # The landing is the operator's too, and produce has nothing to
        # say about it -- a run that does not know about one must not
        # erase it from under the wizard.
        for _k in ("landing_frame", "landing_spot"):
            if prior and prior.get(_k) is not None and rec.get(_k) is None:
                merged[_k] = prior[_k]
        swings = [s for s in swings if int(s.get("idx", -1)) != idx]
        swings.append(merged)
        swings.sort(key=lambda s: int(s.get("idx", 0)))
        em["swings"] = swings
        r2.edit_metrics = em
        db.add(r2)
        db.commit()
        log.info(
            "d3 produce: saved swing %s for click-to-plot — %d track "
            "point(s), %d candidate(s), background=%s",
            idx, len(rec.get("ball_track_frames") or []),
            len(rec.get("cand_points") or []),
            bool(rec.get("tracer_raw_motion_url")),
        )
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        log.warning("d3 produce: could not save swing %s: %s", idx, exc)


def _d3_fast_produce(row, src_path, db, rep, fps, progress=None,
                     hole_number=None, end_green_sec=None,
                     landing=None) -> dict:
    """Build the shipped clip STRAIGHT from Debug3's numbers.

    Stage 8 used to re-run the whole production pipeline
    (`_run_long_upload_job` with auto-detect), which re-derived from
    scratch everything stages 1-6 had just measured: a whole-video
    audio+motion swing detection, the AI anchor walk, the AI launch
    plot, and a second `find_flight` inside `_trace_segment`. On upload
    543 that was 208s of a 245s run — and the AI half of it was
    discarded work, because `_trace_segment` overwrites the anchor's
    ball, impact frame and launch points with Debug3's own the moment
    Debug3 finds a flight.

    So this renders the answer Debug3 already has:

        [ t_launch − PRE_ROLL ] ───tracer───▶ [ tracer ends + 1.5s ]  tee
                                                        │
                                                        ▼  cut to green
                                              [ + GREEN_SEC ]        green

    The tee is rendered long enough to carry AUDIO across the whole
    composite (the green camera has no microphone), but only the
    pre-cutover portion is shown as video — same contract
    `splice_impact_clip` already expects.

    `end_green_sec` is the operator's end frame, in GREEN-clock seconds,
    from the edit wizard. It replaces D3_GREEN_SEC for this run only --
    and it must replace it in all three places at once (the tee render,
    which carries the audio bed past the cutover; the green cut; and the
    splice), or the composite ends up with a video half and an audio half
    of different lengths. None means "use the default", which is what
    every automatic produce does.

    `landing` is {"sec": green-clock seconds, "xy": [x, y] in GREEN
    pixels} when the operator marked where the ball came down. It sets
    the clip's end (via `end_green_sec`) and is recorded on the swing.
    It does NOT yet draw a tracer on the green half -- see the note in
    _d3_green_landing for why that is a separate problem and not a
    continuation of the tee's line."""
    out = {"ok": False, "clips": [], "error": None}
    swings = [
        s for s in (rep.get("swings") or [])
        if s.get("ball") and s.get("launch_frame") is not None
        and (s.get("flight") or [])
    ]
    if not swings:
        out["error"] = "no swing with a flight to produce"
        return out

    fps = float(fps or 30.0)
    delta, delta_src = _d3_green_delta_sec(db, row)
    out["green_delta_sec"] = round(delta, 4)
    out["green_delta_source"] = delta_src
    green_path = None
    if row.green_filename:
        storage.ensure_local(CLIPS_DIR, row.green_filename)
        _gp = CLIPS_DIR / row.green_filename
        if _gp.exists():
            green_path = _gp
        else:
            log.warning(
                "d3 produce: upload %s names green file %s but it is not "
                "on disk — tee-only", row.id, row.green_filename,
            )
    if green_path is not None:
        log.info(
            "d3 produce: upload %s cutting to green, delta=%.4fs (%s)",
            row.id, delta, delta_src,
        )

    # The pixel space the points are in — click-to-plot divides by
    # this, so it must be the source's own size (find_flight ran on
    # the source, so none of the cut-scaling in _persist_swing_track
    # applies here).
    _si = probe_video_info(src_path) or {}
    _src_w = int(_si.get("width") or 0) or None
    _src_h = int(_si.get("height") or 0) or None

    course = db.get(Course, row.course_id) if row.course_id else None
    # Re-produce REPLACES: drop this upload's prior clips first, same as
    # the full pipeline does.
    try:
        _old = db.query(VideoClip).filter(
            VideoClip.long_upload_id == row.id,
        ).all()
        # The caller's hole wins (a camera covers one par-3 and knows
        # which); otherwise inherit from the clips being replaced.
        _hole = int(hole_number) if hole_number else next(
            (int(c.hole_number) for c in _old if c.hole_number), 1,
        )
        for _c in _old:
            db.delete(_c)
        if _old:
            db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        _hole = int(hole_number) if hole_number else 1
        log.warning("d3 produce: could not clear prior clips: %s", exc)

    tok = secrets.token_hex(4)
    for i, sw in enumerate(swings):
        if progress:
            progress(f"Building clip {i + 1} of {len(swings)}",
                     i, len(swings))
        try:
            pts = [
                {"frame": int(p["frame"]), "found": True,
                 "x": float(p["x"]), "y": float(p["y"])}
                for p in (sw.get("flight") or [])
            ]
            ball = sw["ball"]
            launch_f = int(sw["launch_frame"])

            # ── the clip's boundaries, in seconds ──────────────────
            t_launch = launch_f / fps
            t0 = max(0.0, t_launch - D3_PRE_ROLL_SEC)
            t_tracer_end = max(p["frame"] for p in pts) / fps
            t1 = t_tracer_end + D3_POST_TRACER_SEC
            tee_video_dur = t1 - t0

            # How much green follows the cutover. Normally D3_GREEN_SEC;
            # when the operator set an end frame in the wizard, the
            # distance from the cutover to that frame instead. Worked out
            # HERE, before anything uses it, because three things below
            # have to agree on one number.
            _green_sec = D3_GREEN_SEC
            if end_green_sec is not None and green_path is not None:
                _want = float(end_green_sec) - (t1 - float(delta))
                if _want < D3_MIN_GREEN_SEC:
                    log.warning(
                        "d3 produce: swing %s end frame is %.2fs after the "
                        "cutover -- below the %.1fs floor, using the floor",
                        i, _want, D3_MIN_GREEN_SEC,
                    )
                    _green_sec = D3_MIN_GREEN_SEC
                else:
                    _green_sec = _want
                    log.info(
                        "d3 produce: swing %s green side set to %.2fs by "
                        "the operator's end frame (default %.1fs)",
                        i, _green_sec, D3_GREEN_SEC,
                    )

            # The tee file must also carry the audio bed under the green
            # half, so render past the cutover even though that footage
            # is never shown.
            t_render_end = t1 + (_green_sec if green_path else 0.0)

            _tee = CLIPS_DIR / f"d3prod-{row.id}-{tok}-{i}-tee.mp4"
            _rv = render_tracer_video(
                src_path, _tee,
                (float(ball[0]), float(ball[1])),
                launch_f, pts,
                write_start=int(round(t0 * fps)),
                write_end=int(round(t_render_end * fps)),
                rest_verified=True,
            )
            if not _rv.get("ok") or not _tee.exists():
                raise RuntimeError(
                    f"tracer render failed: {_rv.get('error')}")
            transcode_for_web(_tee)

            final = CLIPS_DIR / f"d3prod-{row.id}-{tok}-{i}.mp4"
            green_seg = None
            if green_path is not None:
                # The real instant at the cutover is t1 on the tee
                # clock; on the green clock that same instant is
                # t1 − delta. Seconds throughout — no frame rounding.
                g0 = t1 - float(delta)
                if g0 < 0:
                    log.warning(
                        "d3 produce: swing %s cutover lands %.3fs before "
                        "the green recording starts — tee-only", i, -g0,
                    )
                else:
                    green_seg = (
                        CLIPS_DIR / f"d3prod-{row.id}-{tok}-{i}-green.mp4"
                    )
                    if not cut_segment(
                        green_path, green_seg, g0, g0 + _green_sec,
                    ):
                        log.warning(
                            "d3 produce: green cut failed for swing %s "
                            "— tee-only", i,
                        )
                        green_seg = None

            if green_seg is not None:
                if not splice_impact_clip(
                    _tee, tee_video_dur, green_seg, _green_sec, final,
                ):
                    raise RuntimeError("composite splice failed")
            else:
                _tee.replace(final)

            # ── clip row + participant ────────────────────────────
            # Matched BEFORE the overlay so the name plate carries the
            # player's actual name rather than the placeholder.
            if landing:
                log.info(
                    "d3 produce: swing %s lands at %.2fs green-clock, "
                    "(%.0f, %.0f) in the green frame",
                    i, landing.get("sec") or 0.0,
                    (landing.get("xy") or [0, 0])[0],
                    (landing.get("xy") or [0, 0])[1],
                )
            _url = f"{settings.app_base_url}/uploads/clips/{final.name}"
            clip = VideoClip(
                course_id=row.course_id,
                hole_number=int(_hole),
                camera_type=row.camera_type,
                captured_at=(row.base_captured_at + timedelta(seconds=t0)),
                source_url=_url,
                tracer_url=_url,
                long_upload_id=row.id,
                processing_status=ClipProcessingStatus.received.value,
            )
            db.add(clip)
            db.flush()
            participant = match_clip(db, clip)

            # ── graphics ──────────────────────────────────────────
            _yardage = 101
            if course and course.hole_yardages:
                try:
                    _ry = course.hole_yardages.get(str(int(_hole)))
                    if _ry is not None:
                        _yardage = int(_ry)
                except (TypeError, ValueError):
                    pass
            apply_intro_overlay_inplace(
                final,
                player_name=(
                    participant.name if participant else "Brent Baldwin"
                ),
                course_name=(course.name if course and course.name else ""),
                hole_number=int(_hole),
                par=3,
                yardage=_yardage,
            )

            # Thumbnail AFTER the overlay so the card's still frame
            # matches the clip that ships.
            thumb = extract_thumbnail(final)
            if thumb:
                clip.thumbnail_url = (
                    f"{settings.app_base_url}/uploads/clips/{thumb.name}"
                )
            db.commit()

            # ── click-to-plot ─────────────────────────────────────
            # The old pipeline populated edit_metrics through
            # _persist_swing_track; this path bypasses it, so the editor
            # opened with a leftover candidate pool, no track, and — the
            # visible symptom — nothing to draw on. Everything below is
            # already in SOURCE pixel/frame space: find_flight ran on the
            # source, not on a cut, so there is no offset to apply and no
            # rescaling (see _persist_swing_track for why that matters).
            _bg = _d3_plot_background(
                src_path, launch_f, f"d3plotbg-{row.id}-{tok}-{i}.jpg",
            )
            _sw_rec = {
                "idx": i,
                "clip_id": clip.id,
                "fps": round(fps, 2),
                "start_frame": int(round(t0 * fps)),
                "end_frame": int(round(t1 * fps)),
                "impact_frame": launch_f,
                "ball": {"x": int(round(float(ball[0]))),
                         "y": int(round(float(ball[1])))},
                "tracer_url": _url,
                "tracer_engine": "debug3",
                "persisted_at": round(time.time(), 2),
                "track_frame_width": _src_w,
                "track_frame_height": _src_h,
                "track_scaled_from_cut": False,
                "ball_track_frames": [
                    {"frame": p["frame"], "found": True,
                     "x": p["x"], "y": p["y"]} for p in pts
                ],
                # The flight points are the primary clickable layer; the
                # full detection pool is the dense layer behind the zoom.
                "timed_points": [
                    {"frame": p["frame"], "x": int(round(p["x"])),
                     "y": int(round(p["y"]))} for p in pts
                ],
                "cand_points": [
                    {"frame": int(c["frame"]), "x": int(round(float(c["x"]))),
                     "y": int(round(float(c["y"])))}
                    for c in (sw.get("candidates") or [])
                ][:1500],
            }
            if _bg:
                _sw_rec["tracer_raw_motion_url"] = (
                    f"{settings.app_base_url}/uploads/clips/{_bg}"
                    f"?v={int((CLIPS_DIR / _bg).stat().st_mtime)}"
                )
            _d3_save_swing(db, row.id, i, _sw_rec, delta)

            out["clips"].append({
                "clip_id": clip.id,
                "tee_window_sec": [round(t0, 3), round(t1, 3)],
                "tee_video_dur_sec": round(tee_video_dur, 3),
                "green_delta_sec": round(float(delta), 4),
                "green_delta_source": delta_src,
                "green": green_seg is not None,
                "plot_background": bool(_bg),
                "url": _url,
            })
            log.info(
                "d3 produce: upload=%s swing=%s tee [%.3f, %.3f]s "
                "(%.3fs video) green=%s delta=%s",
                row.id, i, t0, t1, tee_video_dur,
                green_seg is not None,
                f"{delta:.4f}s" if delta is not None else "n/a",
            )
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            log.warning("d3 produce: swing %s failed: %s", i, exc)
            out["clips"].append({"error": f"{exc}"})

    out["ok"] = any(c.get("clip_id") for c in out["clips"])
    return out


def _judge_is_confident(confidence) -> bool:
    """Is the swing judge SURE? Only a sure 'no' is allowed to drop a clip.

    The judge answers with "high" | "medium" | "low". A medium or low "not
    a swing" is the judge saying the picture is ambiguous, and an
    ambiguous picture is not grounds for throwing away a golfer's shot —
    the cost of keeping a bad candidate is a clip nobody watches, and the
    cost of dropping a good one is a shot that never existed.

    Anything unrecognised reads as NOT confident. A judge that returns a
    confidence we cannot parse is exactly the case where we should not be
    acting on its answer. (A numeric confidence is accepted too, in case
    the prompt ever changes shape under us — 0.8 and up counts as high.)
    """
    if isinstance(confidence, bool):
        return False
    if isinstance(confidence, (int, float)):
        return float(confidence) >= 0.8
    if isinstance(confidence, str):
        return confidence.strip().lower() == "high"
    return False


def _debug3_run(row, src_path, db, progress=None, debug_artifacts=True,
                hole_number=None):
    """THE production pipeline. Analyse, then build the clip.

    Debug3 and Re-Produce are the same run — this function — because
    they were two implementations of one process and drifted apart the
    moment they existed side by side (different swing detectors,
    different clip windows, 35s versus 208s). `debug_artifacts` is the
    ONLY difference: the panel wants the film-strips and the preview
    clip to look at, a plain Re-Produce does not and skips the work.
    Everything that decides what ships is shared.
    """
    upload_id = row.id
    from ..services import debug2 as d2
    from ..services import debug3 as d3

    # Which side of the golfer's feet the ball sits on, from the TEE
    # camera that captured this. Set once per installation, because the
    # camera and the tee box do not move — a per-swing guess would be
    # strictly worse. None means search both sides, as before.
    _ball_side = None
    _ball_side_why = "this upload has no camera event, so no camera to ask"
    try:
        _ev = (
            db.query(CameraEvent)
            .filter(CameraEvent.id == row.camera_event_id)
            .first()
            if getattr(row, "camera_event_id", None) else None
        )
        if _ev is not None and _ev.tee_camera_id:
            _cam = db.get(Camera, _ev.tee_camera_id)
            _ball_side = getattr(_cam, "ball_side", None) if _cam else None
            _ball_side_why = (
                f"from tee camera {getattr(_cam, 'name', None) or _cam.id}"
                if _ball_side else
                f"tee camera {getattr(_cam, 'name', None) or (_cam and _cam.id)}"
                f" has no ball side set — set it on /admin/cameras"
            ) if _cam else "the event names a tee camera that no longer exists"
    except Exception as exc:  # noqa: BLE001
        log.debug("could not resolve ball_side for upload %s: %s", upload_id, exc)
        _ball_side_why = f"lookup failed: {exc}"
    from ..services import pose_swing

    # BALLS THE OPERATOR PLACED BY HAND. edit_metrics already refuses to
    # overwrite these when a produce saves its own numbers back
    # (_d3_save_swing); this is the other half of that promise -- the
    # render must USE them rather than re-detecting over the top. Keyed
    # by swing index, the same way _d3_save_swing pairs them.
    _manual_balls: dict = {}
    try:
        for _s in ((row.edit_metrics or {}).get("swings") or []):
            if isinstance(_s, dict) and _s.get("ball_manual") and _s.get("ball"):
                _b = _s["ball"]
                _xy = ([float(_b.get("x")), float(_b.get("y"))]
                       if isinstance(_b, dict) else
                       [float(_b[0]), float(_b[1])])
                _manual_balls[int(_s.get("idx", 0))] = _xy
        if _manual_balls:
            log.info("debug3: upload=%s has %d operator-placed ball(s) — "
                     "they will not be re-detected",
                     upload_id, len(_manual_balls))
    except Exception as exc:  # noqa: BLE001
        log.warning("debug3: could not read operator-placed balls: %s", exc)

    # Diagnostic images only when someone is going to look at them.
    _art_dir = CLIPS_DIR if debug_artifacts else None

    # The card polls processing_status — drive it from here so a run
    # shows as in-progress whichever button started it (and cannot stick
    # on "processing" if the analysis raises).
    def _set_status(status: str, err: str | None = None) -> None:
        try:
            r2 = db.get(LongVideoUpload, upload_id)
            if r2 is None:
                return
            r2.processing_status = status
            if status == "processing":
                r2.processing_started_at = _utcnow_naive()
                r2.last_error = None
            else:
                r2.processing_completed_at = _utcnow_naive()
                r2.last_error = err
            db.add(r2)
            db.commit()
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            log.warning("d3: could not set status=%s: %s", status, exc)

    _set_status("processing")

    # This run's previews replace the last run's — clear them (and
    # anything else that has aged out) before writing more.
    sweep_d3_previews(current_upload_id=upload_id)
    tok = secrets.token_hex(3)
    fps = float(probe_fps(src_path) or 30.0)

    def _clip_url(name):
        if not name:
            return None
        p = CLIPS_DIR / name
        if not p.exists():
            return None
        return (
            f"{settings.app_base_url}/uploads/clips/{p.name}"
            f"?v={int(p.stat().st_mtime)}"
        )

    _fw, _fh = 1280, 720
    try:
        import cv2 as _cv2

        _c = _cv2.VideoCapture(str(src_path))
        _fw = int(_c.get(_cv2.CAP_PROP_FRAME_WIDTH) or 1280) or 1280
        _fh = int(_c.get(_cv2.CAP_PROP_FRAME_HEIGHT) or 720) or 720
        _c.release()
    except Exception:  # noqa: BLE001
        pass
    # Same ball scale Debug2 works in, so the gates mean the same thing.
    _r = max(6.0, 0.012 * float(_fh))

    rep: dict = {
        "ok": True, "upload_id": upload_id, "fps": round(fps, 2),
        "frame": [_fw, _fh], "r_px": round(_r, 1),
        # Surfaced because it changes the ball search from ONE square in
        # front of the golfer to TWO mirrored ones, and a symmetric search
        # is how a shoe wins. Unset is easy to miss in a picture and
        # impossible to miss here.
        "ball_side": _ball_side, "ball_side_reason": _ball_side_why,
        "stages": [], "swings": [],
    }
    # WALL CLOCK. Every phase is timed so the panel can say where the run
    # went rather than leaving it to be guessed from the source. `_phase`
    # accumulates across the per-candidate loop; the stage table below
    # divides the total between the eight reported stages plus the
    # once-per-upload work that belongs to none of them.
    _t_run = time.perf_counter()
    _phase: dict[str, float] = {}

    def _add(name: str, secs: float) -> None:
        _phase[name] = round(_phase.get(name, 0.0) + float(secs), 3)

    if not pose_swing.available():
        _err = (
            f"pose detector unavailable: {pose_swing.unavailable_reason()}"
        )
        # Terminal, or the card sticks on 'processing' forever.
        _set_status("failed", _err)
        return {"ok": False, "error": _err}
    try:
        db.rollback()          # the passes below are slow; don't hold a txn
    except Exception:  # noqa: BLE001
        pass
    pose_dbg: dict = {}
    # Pose is the single longest stage (~26s on upload 501) and it ran
    # SILENT — the card sat grey with nothing on it until the first
    # candidate. Say what is happening before it starts, not after.
    if progress:
        progress("Finding swing candidates", 0, 0)
    _t0 = time.perf_counter()
    # STAGE 1 LABELS, IT NO LONGER ELIMINATES. The spine-bend and burst-
    # duration gates were dropping real swings, and a dropped burst was
    # gone for good: the non-max suppression ran over gate-PASSING bursts
    # only, so a swing rejected as upright never entered the pool and could
    # not be returned at all. Stages 2 and 3 — the ball at impact and the
    # club fan, both far better discriminators than a spine angle — never
    # got to look at it.
    #
    # So Debug3 takes the rejects too and runs 1, 2 and 3 on every one.
    # What the gates thought is recorded per candidate as a label, and the
    # panel shows it; nothing is removed on their say-so.
    cands = list(
        pose_swing.detect_swings_from_pose(
            src_path, fps=fps, keep_rejected=True,
            # Deliberately high. Debug3 is where we want to see everything;
            # this is a backstop against a pathological pose trace, not a
            # filter, and stage 1 says so out loud if it ever bites.
            max_rejected=40,
            debug=pose_dbg,
        )
        or []
    )
    _add("pose", time.perf_counter() - _t0)
    # STAGE 1'S WORKING. Zero candidates is the least informative number
    # in the panel: it cannot distinguish "pose never saw the golfer" from
    # "it saw the swing and the spine-bend gate rejected it" from "the
    # wrist speed never rose above the noise floor". All three are one
    # config change apart and they need completely different fixes.
    rep["pose_debug"] = {
        k: pose_dbg.get(k)
        for k in ("reason", "available", "n_pose_frames", "n_samples",
                  "coverage", "n_bridged", "duration_sec", "median",
                  "threshold", "n_raw_bursts", "n_bend_rejected",
                  "back_bend_min_deg", "back_bend_max_deg",
                  "strong_ratio", "ratio_min", "ratio_max", "reached_eof",
                  "n_crop_frames", "n_bootstrap_scans", "bootstrap_found_at",
                  "n_gate_passed", "n_rescued", "n_rescued_dropped")
    }
    # The wrist-speed trace itself, so a swing that never crossed the
    # threshold is visible as a shape rather than inferred from a count.
    rep["pose_series"] = [
        round(float(v), 4) for v in (pose_dbg.get("series") or [])
    ][:2000]
    # Every burst the detector saw and what happened to it, including the
    # ones that never became candidates.
    rep["bursts"] = list(pose_dbg.get("bursts_detail") or [])
    _pd = rep["pose_debug"]
    _why = []
    if _pd.get("available") is False:
        _why.append(_pd.get("reason") or "pose unavailable")
    elif not cands:
        if not _pd.get("n_pose_frames"):
            _why.append("pose never found a person in this clip")
        elif not _pd.get("n_raw_bursts"):
            _why.append(
                f"no wrist-speed burst cleared the threshold "
                f"({_pd.get('threshold')} vs a median of {_pd.get('median')})"
            )
        elif _pd.get("n_bend_rejected"):
            _why.append(
                f"{_pd['n_bend_rejected']} burst(s) rejected as upright — "
                f"the spine-bend gate wants "
                f"{_pd.get('back_bend_min_deg')}-{_pd.get('back_bend_max_deg')}"
                f" degrees"
            )
        else:
            _why.append("bursts were found but none passed the gates")
    _n_pass = sum(1 for c in cands if c.get("gate_ok", True))
    _n_resc = len(cands) - _n_pass
    _n_cut = int(pose_dbg.get("n_rescued_dropped") or 0)
    rep["stages"].append({
        "n": 1, "name": "Pose candidates",
        "detail": (
            "wrist speed + spine bend, the detector produce uses -- but "
            "the gates only LABEL here: every burst goes on to stages 2 "
            "and 3 whatever they think"
            + (f" · {_n_pass} passed, {_n_resc} carried despite failing"
               if _n_resc else "")
            + (f" · {_n_cut} weaker one(s) not carried (cap reached)"
               if _n_cut else "")
            + (" -- " + "; ".join(_why) if _why else "")
        ),
        "count": len(cands), "counts": "candidates",
        "seconds": _phase.get("pose", 0.0),
    })

    # Once per upload, not per candidate: it samples the whole video.
    if progress:
        progress("Scanning for the ball at rest", 0, len(cands))
    _t0 = time.perf_counter()
    rest = (
        _rest_ball_departures(src_path, fps, db, row)
        if debug_artifacts else {}
    )
    _add("rest_ball", time.perf_counter() - _t0)
    rep["rest_ball"] = {
        "reason": rest.get("reason"), "roi": rest.get("roi"),
        "departures": rest.get("deps"),
    }

    n_flights = 0
    n_produced = 0
    n_judged_out = 0
    # A flight the tests ACCEPTED but that did not survive to a usable
    # answer. Counted separately so stage 6 can never again read zero
    # while the table underneath it says "accepted".
    n_accepted_but_lost = 0
    for i, c in enumerate(cands):
        if progress:
            progress(f"Swing {i + 1} of {len(cands)}: finding the ball at impact",
                     i, len(cands))
        peak_t = float(c.get("peak_time_sec") or 0.0)
        imp_f = int(round(peak_t * fps))
        f_lo = max(0, imp_f - d3.WIN_PRE)
        f_hi = imp_f + d3.WIN_POST
        entry: dict = {
            "idx": i, "peak_time_sec": round(peak_t, 2),
            "impact_frame": imp_f, "window": [f_lo, f_hi],
            # What stage 1's gates thought — a label now, not a verdict.
            # "rescued" candidates reach stages 2 and 3 exactly like the
            # rest; this only says which ones to look at twice.
            "pose_gate": c.get("gate_status") or "swing",
            "pose_gate_ok": bool(c.get("gate_ok", True)),
        }

        # STAGE 2: WHERE IS THE BALL AT REST? The club-arc pass, at the
        # pose impact frame. It runs FIRST because two later stages want
        # its answer: the judge composites the heat around it (a fan
        # centred on the ball is a swing; the same fan centred on a bag
        # is not), and the flight picker needs it to arm the aim gate.
        # Measured once here and handed to both — find_flight takes it as
        # `rest_ball` rather than repeating the pass.
        _t0 = time.perf_counter()
        _rest_xy = None
        _pre: dict = {}
        if c.get("impact_feet_xy"):
            from ..services.debug2 import club_bottom_ball

            try:
                _pre = club_bottom_ball(
                    src_path, imp_f, fps,
                    feet_xy=c.get("impact_feet_xy"),
                    head_xy=c.get("impact_head_xy"),
                    ball_side=_ball_side,
                    debug_dir=_art_dir,
                    debug_prefix=f"d3-{upload_id}-{tok}-{i}-hint",
                ) or {}
            except Exception as _exc:  # noqa: BLE001
                log.warning("debug3 rest-ball pass failed: %s", _exc)
                _pre = {"reason": f"failed: {_exc}"}
            _rest_xy = _pre.get("xy") if _pre.get("ok") else None
        else:
            _pre = {"reason": "pose gave no feet — nothing to anchor on"}
        _add("club_arc", time.perf_counter() - _t0)
        # Reported here, not after the flight, so a candidate the judge
        # drops still shows where the ball was looked for — that picture
        # is drawn whether or not it was found.
        entry["ball_hint"] = _rest_xy
        entry["ball_hint_reason"] = _pre.get("reason")
        entry["ball_hint_image_url"] = _clip_url(_pre.get("image"))

        # STAGE 3: IS IT A SWING AT ALL? Pose fires on wrist speed and
        # spine bend, which a practice swing, a bag drop or someone
        # bending to tee up all produce. The club's fan — the wedge of
        # angles its head sweeps through around impact — is what
        # separates them, and that is what the judge is shown: a motion
        # heat composite over the impact window, centred on the ball
        # stage 2 just found (the wrist, which this used before the two
        # stages swapped, wanders with the swing; the ball does not).
        #
        # Deliberately narrow, twice over. Only the AI can veto — the
        # club-fan heuristic (the no-key fallback) is recorded and shown
        # but never drops anything — and only a HIGH-confidence "not a
        # swing" counts. Medium and low are the judge saying the picture
        # is ambiguous, and an ambiguous picture is not grounds for
        # throwing away a golfer's shot. Asymmetric on purpose: a bad
        # candidate kept costs a clip nobody watches; a real one dropped
        # costs a shot that never existed.
        _t0 = time.perf_counter()
        _heat: dict = {}
        _ai_seen = False
        _ai_sure = False
        if settings.swing_heat_check_enabled:
            try:
                from ..services.tracer import swing_heat_check

                # The composite is the judge's INPUT, not just a debug
                # picture, so it is written even when artifacts are off —
                # otherwise turning artifacts off would silently turn the
                # judge off with it.
                _heat_dir = _art_dir or CLIPS_DIR
                _heat = swing_heat_check(
                    src_path, peak_t, fps,
                    ball_hint=_rest_xy or c.get("impact_wrist_xy"),
                    debug_dir=_heat_dir,
                    debug_prefix=f"d3heat-{upload_id}-{tok}-{i}",
                ) or {}
                if _heat.get("image_clean") and os.environ.get(
                    "ANTHROPIC_API_KEY",
                ):
                    _j = judge_swing_heat_image(
                        _heat_dir / _heat["image_clean"],
                    )
                    _heat["ai_judge"] = _j.get("is_swing")
                    _heat["ai_reason"] = _j.get("reason")
                    _heat["ai_confidence"] = _j.get("confidence")
                    if _j.get("is_swing") is True:
                        _heat["verdict"] = "club_swing"
                        _ai_seen = True
                    elif _j.get("is_swing") is False:
                        _heat["verdict"] = "no_swing"
                        _ai_seen = True
                        _ai_sure = _judge_is_confident(_j.get("confidence"))
            except Exception as _exc:  # noqa: BLE001
                log.warning("debug3 heat judge failed: %s", _exc)
                _heat = {"reason": f"failed: {_exc}"}
        else:
            _heat = {"reason": "swing_heat_check_enabled is off"}
        _add("judge", time.perf_counter() - _t0)
        entry["judge"] = {
            "verdict": _heat.get("verdict"),
            "ai_judge": _heat.get("ai_judge"),
            "ai_reason": _heat.get("ai_reason"),
            "ai_confidence": _heat.get("ai_confidence"),
            "fan": _heat.get("fan"),
            "n_rays": _heat.get("n_rays"),
            "n_angles": _heat.get("n_angles"),
            "reason": _heat.get("reason"),
            "centred_on": ("the ball at rest" if _rest_xy
                           else "the wrist at impact (stage 2 found no ball)"),
            "decided_by": ("ai" if _ai_seen
                           else ("heuristic" if _heat.get("available")
                                 else "nothing")),
            # Whether this verdict was allowed to act. A 'not a swing' the
            # judge is not sure about is recorded and then ignored, and
            # that has to be legible or the panel looks like the judge was
            # overruled at random.
            "confident": _ai_sure,
        }
        entry["heat_image_url"] = _clip_url(
            _heat.get("image") or _heat.get("image_clean"))
        _no_swing = _ai_seen and _heat.get("verdict") == "no_swing"
        if _no_swing and not _ai_sure:
            # Recorded, not acted on. Keeping a bad candidate costs a clip
            # nobody watches; dropping a good one costs a shot.
            entry["judge_unsure"] = True
            log.info(
                "debug3: upload=%s judge said 'not a swing' at %s "
                "confidence for candidate %s @ %.1fs — KEPT (%s)",
                upload_id, _heat.get("ai_confidence"), i, peak_t,
                _heat.get("ai_reason"),
            )
        elif _no_swing:
            entry["dropped_by_judge"] = True
            entry["ok"] = False
            n_judged_out += 1
            log.info(
                "debug3: upload=%s AI judge dropped candidate %s @ %.1fs "
                "with high confidence (%s)",
                upload_id, i, peak_t, _heat.get("ai_reason"),
            )
            rep["swings"].append(entry)
            continue

        # ONE PIPELINE. This endpoint does not re-implement the stages -- it
        # calls the same function produce calls and asks it to show its work.
        # The two had already drifted once (the ball came from the club arc
        # here and from the extrapolation there, 3px versus 67px on the same
        # swing), which is what two implementations of one process always
        # eventually do.
        _manual = _manual_balls.get(i)
        _ff = d3.find_flight(
            src_path, fps, impact_frame=imp_f,
            head_xy=c.get("impact_head_xy"),
            feet_xy=c.get("impact_feet_xy"),
            frame_w=_fw, frame_h=_fh, ball_side=_ball_side,
            rest_ball=(
                {"ok": True, "xy": _manual,
                 "reason": "placed by the operator in the edit wizard"}
                if _manual else _pre
            ),
            ball_locked=_manual is not None,
            debug_dir=_art_dir,
            debug_prefix=f"d3-{upload_id}-{tok}-{i}-",
        )
        _dbg = _ff.get("debug") or {}
        # find_flight times its own phases; fold them into the run totals
        # and keep the per-swing breakdown so a single slow swing is
        # visible rather than averaged away.
        _ftime = dict(_dbg.get("timing") or {})
        for _k, _v in _ftime.items():
            if _k != "total":
                _add(_k, _v)
        entry["timing"] = _ftime
        _ball = _ff.get("ball")
        entry["ball"] = _ball
        entry["ball_source"] = _ff.get("ball_source")
        entry["ball_reason"] = (_dbg.get("club_arc") or {}).get("reason")
        entry["ball_alt"] = _dbg.get("ball_alt")
        entry["ball_alt_source"] = _dbg.get("ball_alt_source")
        entry["ball_disagree_px"] = (
            (_dbg.get("club_arc") or {}).get("vs_extrapolated_px"))
        entry["launch"] = _dbg.get("launch")
        entry["launch_frame"] = _ff.get("launch_frame")
        entry["launch_vs_pose_frames"] = (
            int(_ff["launch_frame"]) - imp_f
            if _ff.get("launch_frame") is not None else None
        )
        entry["body_box"] = _dbg.get("body_box")
        _d = _dbg.get("detect") or {}
        entry["detect_reason"] = _d.get("reason")
        entry["detect_stats"] = _d.get("stats")
        entry["max_area"] = _d.get("max_area")
        entry["max_side"] = _d.get("max_side")
        entry["n_at_strict_cap"] = _d.get("n_at_strict_cap")
        entry["area_summary"] = _d.get("area_summary")
        entry["frame_image_url"] = _clip_url((_d.get("images") or {}).get("frame"))
        entry["dets_image_url"] = _clip_url((_d.get("images") or {}).get("dets"))
        entry["n_tracks"] = _dbg.get("n_tracks")
        entry["tracks_preview"] = _dbg.get("tracks_preview")
        entry["tracks_image_url"] = _clip_url(_dbg.get("tracks_image"))
        entry["winner_not_shown"] = _dbg.get("winner_not_shown")
        _fl = _dbg.get("flight") or {}
        entry["flight_reason"] = _fl.get("reason")
        entry["tried"] = _fl.get("tried")
        entry["fit"] = {
            "n_inliers": _fl.get("n_inliers"), "rms_px": _fl.get("rms_px"),
            "at_impact": _fl.get("at_impact"), "x_degree": _fl.get("x_degree"),
            # Was missing, so the panel rendered "aims px from the ball"
            # with a hole where the number should be.
            "aim_px": _fl.get("aim_px"), "aim_basis": _fl.get("aim_basis"),
        }
        entry["flight"] = _ff.get("points") or []
        # find_flight's OWN verdict. Without this the panel could show a
        # track marked "accepted, score 15.89" beside a stage reading
        # "0 flights accepted" and offer no way to reconcile them.
        entry["find_flight_ok"] = bool(_ff.get("ok"))
        entry["find_flight_reason"] = _ff.get("reason")
        entry["find_flight_failed"] = bool(_ff.get("failed"))
        entry["images_error"] = _dbg.get("images_error")
        entry["find_flight_traceback"] = _dbg.get("traceback")
        entry["images_traceback"] = _dbg.get("images_traceback")
        if (_fl.get("n_inliers") or 0) and not _ff.get("ok"):
            n_accepted_but_lost += 1
        # The full ball-sized detection pool — click-to-plot's dense
        # layer. Carried on the entry so _d3_fast_produce can persist it,
        # then stripped from the report before it goes over the wire
        # (it is hundreds of points per swing and the panel never reads
        # it — the editor gets them from edit_metrics instead).
        entry["candidates"] = _ff.get("candidates") or []
        entry["ball_image_url"] = _clip_url(
            (_dbg.get("club_arc") or {}).get("image"))
        entry["rest_check_frame"] = _dbg.get("rest_check_frame")
        entry["rest_check_image_url"] = _clip_url(_dbg.get("rest_check_image"))
        entry["flight_image_url"] = _clip_url(_dbg.get("flight_image"))
        res = {"ok": _ff.get("ok"), "fit": {
            "inliers": [
                {"frame": q["frame"], "x": q["x"], "y": q["y"]}
                for q in (_ff.get("points") or [])
            ],
        }}
        if _ff.get("ok"):
            n_flights += 1

        # 6. PRODUCE. Everything above is measurement; this is the clip.
        # Debug3's own numbers go straight into the renderer produce uses --
        # the ball it settled on, the launch frame the flight derived, and
        # the RANSAC inliers as the tracer points. Written to its own file,
        # so nothing here replaces a produced clip or touches edit_metrics.
        if res.get("ok") and _ball and debug_artifacts:
            _t0 = time.perf_counter()
            try:
                _pts = [
                    {"frame": int(q["frame"]), "found": True,
                     "x": float(q["x"]), "y": float(q["y"])}
                    for q in ((res.get("fit") or {}).get("inliers") or [])
                ]
                _f_imp = int(entry.get("launch_frame") or imp_f)
                # A clip either side of the strike, not the whole window.
                _w0 = max(0, _f_imp - int(round(0.6 * fps)))
                _w1 = _f_imp + int(round(2.6 * fps))
                _out = CLIPS_DIR / f"d3clip-{upload_id}-{tok}-{i}.mp4"
                _rv = render_tracer_video(
                    src_path, _out,
                    (float(_ball[0]), float(_ball[1])),
                    _f_imp, _pts,
                    write_start=_w0, write_end=_w1,
                    # The ball came from the club arc measured at the
                    # launch frame and was eyeballed against the check
                    # frame; the renderer's relocation guards would only
                    # second-guess a better number.
                    rest_verified=True,
                )
                entry["produce"] = {
                    "ok": bool(_rv.get("ok")),
                    "error": _rv.get("error"),
                    "n_points": _rv.get("n_points"),
                    "frame_range": _rv.get("frame_range"),
                    "tracer_points": len(_pts),
                    "impact_frame": _f_imp,
                    "ball": _ball,
                }
                if _rv.get("ok") and _out.exists():
                    # render_tracer_video writes with the mp4v fourcc
                    # (ai_tracer.py:3917), which browsers refuse to play in a
                    # <video> tag -- the panel showed a rendered clip that
                    # sat black at 0:00. Produce runs every clip through this
                    # for exactly that reason; Debug3 was not.
                    entry["produce"]["h264"] = bool(transcode_for_web(_out))
                    entry["produce"]["clip_url"] = _clip_url(_out.name)
                    n_produced += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("debug3 produce failed on %s: %s", upload_id, exc)
                entry["produce"] = {"ok": False, "error": f"{exc}"}
            # Render + h264 transcode, whether or not it succeeded — a
            # failure that took 20s is the interesting kind.
            _pv = round(time.perf_counter() - _t0, 3)
            _add("preview", _pv)
            entry["timing"]["preview"] = _pv

        rep["swings"].append(entry)

    # 7. PRODUCE. Stage 6 renders a preview and replaces nothing -- that
    # was right when Debug3 was a read-only diagnostic, and wrong now
    # that it IS the process.
    #
    # This used to kick the full production job (auto-detect, whole-video
    # audio+motion detection, the AI anchor walk, the AI launch plot, and
    # a second find_flight inside _trace_segment). It re-derived from
    # scratch everything stages 1-6 had just measured, and _trace_segment
    # then THREW AWAY the AI's answers -- it overwrites the anchor's
    # ball, impact frame and launch points with Debug3's the moment
    # Debug3 has a flight. Measured on upload 543: 208s of a 245s run,
    # 62s of it AI calls whose results were discarded.
    #
    # Now the clip is built straight from the numbers above -- pre-roll,
    # tracer, tail, cut to green -- by _d3_fast_produce.
    #
    # Synchronous on purpose: we are already on the debug3 background
    # thread, and the status poll keeps the panel honest about it.
    if n_flights:
        if progress:
            progress("Building the clip", len(cands), len(cands))
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass
        _t0 = time.perf_counter()
        try:
            _fp = _d3_fast_produce(
                row, src_path, db, rep, fps, progress,
                hole_number=hole_number,
            )
            _n_made = sum(1 for c in _fp["clips"] if c.get("clip_id"))
            rep["produced"] = {
                "ok": bool(_fp["ok"]),
                "clips": _fp["clips"],
                "detail": (
                    f"built {_n_made} clip(s) straight from the numbers "
                    f"above -- pre-roll {D3_PRE_ROLL_SEC}s, tracer, "
                    f"+{D3_POST_TRACER_SEC}s tail, then the cut to green"
                ) if _fp["ok"] else (_fp.get("error") or "produce failed"),
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("debug3: produce failed for %s: %s", upload_id, exc)
            rep["produced"] = {"ok": False, "error": f"{exc}"}
        _add("produce_real", time.perf_counter() - _t0)
    # Consumed (or not needed) — keep them out of the panel payload.
    for _s in rep["swings"]:
        _s.pop("candidates", None)
        # The produce job ran synchronously on THIS thread, so its
        # thread-local phase clock is still readable — this is stage 8
        # broken down into ffmpeg / AI / detection instead of one
        # 198-second block.
        rep["produce_breakdown"] = _pt_snapshot()

    rep["stages"].extend([
        {"n": 2, "name": "Ball at impact",
         "detail": (
             f"run on all {len(cands)} candidate(s), gate-passing or not -- "
             f"club-arc vertex on the ground line at the feet, measured "
             f"first because stage 3 centres the judge's composite on it "
             f"and stage 6 needs it to arm the aim gate"
         ),
         "count": sum(1 for s in rep["swings"] if s.get("ball_hint")),
         "counts": "balls located",
         "seconds": _phase.get("club_arc", 0.0)},
        {"n": 3, "name": "AI judge on the club fan",
         "detail": (
             f"run on all {len(cands)} candidate(s) -- motion-heat "
             f"composite around that ball, judged on the club's angular "
             f"sweep. Only a HIGH-confidence 'not a swing' drops the "
             f"candidate, and it does so before any tracking runs"
         ),
         "count": len(cands) - n_judged_out, "counts": "candidates kept",
         "seconds": _phase.get("judge", 0.0)},
        {"n": 4, "name": "MOG2 + component + area filter",
         "detail": "big blobs become a golfer mask; only ball-sized "
                   "off-body blobs survive",
         "count": sum((s.get("detect_stats") or {}).get("kept", 0)
                      for s in rep["swings"]),
         "counts": "detections kept",
         "seconds": _phase.get("detect", 0.0)},
        {"n": 5, "name": "Nearest-neighbour tracking",
         "detail": "constant-velocity prediction with a gate that widens "
                   "on a missed frame",
         "count": sum(int(s.get("n_tracks") or 0)
                      for s in rep["swings"]),
         "counts": "tracks built",
         "seconds": _phase.get("tracks", 0.0)},
        {"n": 6, "name": "RANSAC parabola + flight tests",
         "detail": "x linear in t, y quadratic; must rise and must point "
                   "back at the ball"
                   + (f" -- WARNING: {n_accepted_but_lost} track(s) passed "
                      f"the tests but produced no usable ball or points; "
                      f"see the swing's flight reason"
                      if n_accepted_but_lost else ""),
         "count": n_flights, "counts": "flights usable",
         "seconds": _phase.get("flight", 0.0)},
        {"n": 7, "name": "Preview clip",
         "detail": "the same renderer produce uses, fed Debug3's ball, "
                   "launch frame and inliers -- rendered here so the answer "
                   "can be judged before it ships",
         "count": n_produced, "counts": "previews rendered",
         "seconds": _phase.get("preview", 0.0)},
        {"n": 8, "name": "Produced for real",
         "detail": ((rep.get("produced") or {}).get("detail")
                    or (rep.get("produced") or {}).get("error")
                    or "no flight, so nothing was re-produced"),
         "count": 1 if (rep.get("produced") or {}).get("ok") else 0,
         "counts": "uploads re-produced",
         "seconds": _phase.get("produce_real", 0.0)},
    ])

    # TIMING SUMMARY. The eight stages do not account for the whole run —
    # the resting-ball departure scan samples the entire video and belongs
    # to no stage, and the panel-only debug images (the rest-check frame,
    # the flight drawing) are overhead produce never pays. Report those
    # separately and carry an explicit `unattributed` remainder, so the
    # numbers reconcile against the wall clock instead of quietly missing
    # whatever is not on the list.
    # Stage 8 itemised, with its own remainder so the parts reconcile
    # against the 'Produced for real' figure rather than trailing off.
    _pb_raw = rep.pop("produce_breakdown", None) or {}
    _produce_breakdown = {
        k: round(v, 2)
        for k, v in sorted(_pb_raw.items(), key=lambda kv: -kv[1]) if v > 0
    }
    if _produce_breakdown:
        _pb_other = round(
            max(0.0, _phase.get("produce_real", 0.0)
                - sum(_pb_raw.values())), 2,
        )
        if _pb_other > 0:
            _produce_breakdown["other"] = _pb_other

    _total = round(time.perf_counter() - _t_run, 3)
    _staged = sum(
        float(st.get("seconds") or 0.0) for st in rep["stages"]
    )
    _overhead = {
        k: _phase[k] for k in ("rest_ball", "launch", "rest_check_image",
                               "draw_tracks", "draw_flight")
        if _phase.get(k)
    }
    rep["timing"] = {
        "total_sec": _total,
        "by_stage": {
            int(st["n"]): round(float(st.get("seconds") or 0.0), 2)
            for st in rep["stages"]
        },
        # Not one of the seven: whole-video scans and debug-image writes.
        "outside_stages": {k: round(v, 2) for k, v in _overhead.items()},
        "unattributed_sec": round(
            max(0.0, _total - _staged - sum(_overhead.values())), 2,
        ),
        "n_swings": len(cands),
        # Stage 7, itemised. `other` is the produce job's unmeasured
        # remainder — DB work, pose on the cuts, Debug3's own re-run
        # inside _trace_segment, and anything else not wrapped.
        "produce_breakdown": _produce_breakdown,
        # The per-swing cost of the analysis itself, which is what scales
        # with how many swings a round puts through the panel.
        "per_swing_sec": (
            round((_total - _phase.get("produce_real", 0.0)) / len(cands), 2)
            if cands else None
        ),
    }
    for st in rep["stages"]:
        st["seconds"] = round(float(st.get("seconds") or 0.0), 2)
        st["pct"] = round(100.0 * st["seconds"] / _total, 1) if _total else 0.0
    log.info(
        "debug3 timing upload=%s total=%.1fs over %d swing(s): %s | "
        "outside stages: %s | unattributed %.1fs",
        upload_id, _total, len(cands),
        ", ".join(f"{st['n']}.{st['name']} {st['seconds']}s "
                  f"({st['pct']}%)" for st in rep["stages"]),
        ", ".join(f"{k} {v}s" for k, v in _overhead.items()) or "none",
        rep["timing"]["unattributed_sec"],
    )
    # Terminal state for the card. A run that analysed but produced
    # nothing is a FAILURE from the operator's side — the upload still
    # has no clip — so say so rather than reporting completed.
    _prod = rep.get("produced") or {}
    if n_flights and _prod.get("ok"):
        _set_status("completed")
    else:
        # "no swings detected" is load-bearing: the camera path matches
        # on that prefix to tell a capture with nothing in it (not an
        # error — the Pi did its job, nobody hit a ball) from a real
        # failure, and flags the event red only for the latter.
        _set_status("failed", (
            _prod.get("error")
            or (f"no swings detected: {len(cands)} pose candidate(s), "
                f"no ball flight found"
                if not n_flights else "produce made no clip")
        )[:2000])
    return rep


# ── Email setup + test send ────────────────────────────────────────────

@router.get("/email-status")
def email_status():
    """Which transport email will actually use, without leaking secrets.

    Worth having as its own call because the failure mode is silent: with
    neither SMTP nor SendGrid configured, every send "succeeds" and only
    writes an EMAIL (mock) log line. This says so plainly."""
    from ..services import notifications as N

    smtp_ok = bool(
        settings.smtp_host and settings.smtp_user and settings.smtp_password
    )
    if smtp_ok:
        transport = "smtp"
        detail = (
            f"{settings.smtp_host}:{settings.smtp_port} as "
            f"{settings.smtp_user}"
            f"{' (SSL)' if settings.smtp_use_ssl or settings.smtp_port == 465 else ' (STARTTLS)'}"
        )
        sender = settings.smtp_from or settings.smtp_user
    elif settings.sendgrid_api_key:
        transport = "sendgrid"
        detail = f"API key set ({len(settings.sendgrid_api_key)} chars)"
        sender = settings.sendgrid_from_email
    else:
        transport = "mock"
        detail = (
            "NOTHING IS SENT — set SMTP_HOST / SMTP_USER / SMTP_PASSWORD, "
            "or SENDGRID_API_KEY"
        )
        sender = None
    _logo = N._logo_png()
    return {
        "transport": transport,
        "detail": detail,
        "from": sender,
        "logo_kb": round(len(_logo) / 1024.0, 1) if _logo else None,
        "missing": [
            k for k, v in (
                ("SMTP_HOST", settings.smtp_host),
                ("SMTP_USER", settings.smtp_user),
                ("SMTP_PASSWORD", settings.smtp_password),
            ) if not v
        ] if transport == "mock" else [],
    }


@router.post("/email-send-templates")
def email_send_templates(
    payload: dict = Body(default={}), db: Session = Depends(get_db),
):
    """Send one of each REAL email to an address so the templates can be seen.

    Distinct from /test-email, which sends a single generic "your wiring
    works" message. This one fires the actual notify_* functions.

    Uses the REAL notify_* functions, not copies — a preview that renders
    its own approximation of the template is a preview of the wrong thing.
    The clip email attaches a genuine produced clip when one is on disk,
    so the attachment path gets exercised too rather than only the body.
    """
    to = (payload.get("to") or "").strip()
    if "@" not in to:
        raise HTTPException(400, "a destination email address is required")
    which = payload.get("templates") or "all"
    from types import SimpleNamespace

    from ..models import Course, VideoClip
    from ..services import notifications as N

    gallery = f"{settings.app_base_url}/g/sample-token"
    results: list[dict] = []

    def _try(name: str, fn):
        if which != "all" and name not in which:
            return
        try:
            fn()
            results.append({"template": name, "ok": True})
        except Exception as exc:  # noqa: BLE001
            log.warning("email-test %s failed: %s", name, exc)
            results.append({"template": name, "ok": False, "error": str(exc)})

    _try("registration", lambda: N.notify_registration_confirmed(
        "Ben", None, to, gallery))
    _try("gallery_ready", lambda: N.notify_gallery_ready(
        "Ben", None, to, gallery))
    _try("hio_review", lambda: N.notify_hio_under_review("Ben", None, to))
    _try("hio_confirmed", lambda: N.notify_hio_confirmed(
        "Ben", None, to, gallery))

    # The clip email, with a real clip when we can find one — this is the
    # message golfers actually receive, and the one worth seeing.
    def _clip():
        clip_row = (
            db.query(VideoClip)
            .filter(VideoClip.source_url.isnot(None))
            .order_by(VideoClip.id.desc())
            .first()
        )
        fake_participant = SimpleNamespace(
            id=0, name="Ben", email=to, gallery_token="sample-token",
        )
        # VideoClip carries course_id, not a `course` relationship, so the
        # course has to be fetched. Use its real name and yardages when we
        # found a clip, so the preview matches what a golfer would get.
        course_row = (
            db.get(Course, clip_row.course_id) if clip_row else None
        )
        fake_course = SimpleNamespace(
            name=(course_row.name if course_row else "Baldwin Links"),
            hole_yardages=(course_row.hole_yardages or {}) if course_row
            else {"3": 173},
        )
        fake_clip = SimpleNamespace(
            id=0,
            delivered_at=None,          # so the idempotency guard lets it send
            hole_number=(clip_row.hole_number if clip_row else 3),
            carry_yards=168, ball_speed_mph=112,
            source_url=(clip_row.source_url if clip_row else None),
        )
        N.notify_clip_ready(fake_participant, fake_clip, fake_course)

    _try("clip_ready", _clip)

    st = email_status()
    return {
        "ok": all(r["ok"] for r in results),
        "to": to,
        "transport": st["transport"],
        "note": (
            "transport is 'mock' — nothing was actually delivered, check the "
            "server log for EMAIL (mock) lines"
            if st["transport"] == "mock" else None
        ),
        "results": results,
    }
