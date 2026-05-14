"""Device-facing endpoints for on-course capture (always-on Pi
cameras). All routes auth via the per-Camera auth_token in the URL —
the admin password and user JWTs are not in scope here. Endpoints
are designed to be called by an unattended Python agent running on a
Raspberry Pi in the field; see docs/field-deployment.md for the
deployment plan and routers/admin.py for the operator-facing CRUD
that mints the tokens these routes accept.

Flow per swing:

  1. Tee Pi detects a person in its tee-box ROI for >=2 s.
  2. Tee Pi POSTs /event-trigger with a session_id (UUID4) it
     generated. Backend creates a CameraEvent row and pushes the
     trigger into an asyncio.Queue keyed by the paired green camera's
     id. Tee Pi starts recording from its pre-roll buffer.
  3. Green Pi is sitting in /poll-trigger long-polling. It wakes up
     with the session_id, commits its pre-roll, keeps recording.
  4. Both Pis upload their MP4s via /upload-event with the shared
     session_id. As soon as the second clip lands (or the only clip,
     for unpaired-tee single-camera setups), a background thread
     runs the existing _process_long_upload_segments pipeline and
     produces a VideoClip row + composite.

State that lives in-memory only:
  _pending_triggers: dict[camera_id -> asyncio.Queue] — the wake-up
  queue for the long-poll. Lost on backend restart, which is fine:
  the tee Pi will retry; the green Pi reconnects to /poll-trigger on
  its next iteration. Single-instance deployment assumed (Replit).
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import threading
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..config import settings
from ..database import SessionLocal, get_db
from ..models import Camera, CameraEvent, Course, VideoClip
from ..services.video import probe_video_info

log = logging.getLogger("golfreelz.cameras")

router = APIRouter(prefix="/api/cameras", tags=["cameras"])

# Where uploaded MP4s land. Same directory the rest of the pipeline
# reads from / writes to. Resolved relative to the backend package
# root, matching the pattern in routers/admin.py.
CLIPS_DIR = Path(__file__).resolve().parents[2] / settings.upload_dir / "clips"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

# Per-camera wake-up queues. The tee Pi's /event-trigger writes into
# the paired green's queue; the green Pi's /poll-trigger awaits it.
_pending_triggers: dict[int, asyncio.Queue] = {}

# Max bytes per uploaded event clip (~100 MB). 1080p30 for 12 s is
# typically 30-50 MB; this catches malformed uploads / wrong files
# without rejecting legitimate captures.
MAX_EVENT_CLIP_BYTES = 100 * 1024 * 1024


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _utcnow_naive() -> datetime:
    """Match the timezone-naive convention the rest of the schema uses."""
    return datetime.utcnow()


def _get_camera_by_token(token: str, db: Session) -> Camera:
    """Resolve the auth_token in the URL to a Camera, or raise 404 /
    403. Bumps last_seen_at so any successful call counts as a
    heartbeat for free."""
    cam = db.query(Camera).filter(Camera.auth_token == token).first()
    if cam is None:
        raise HTTPException(404, "unknown camera token")
    if not cam.enabled:
        raise HTTPException(403, "camera is disabled")
    cam.last_seen_at = _utcnow_naive()
    return cam


def _queue_for(camera_id: int) -> asyncio.Queue:
    """Lazy-create the wake-up queue for a camera. maxsize=10 keeps a
    dead camera's queue from growing unbounded; old entries are
    dropped at trigger time (we only ever care about the latest
    pending trigger anyway)."""
    q = _pending_triggers.get(camera_id)
    if q is None:
        q = asyncio.Queue(maxsize=10)
        _pending_triggers[camera_id] = q
    return q


def _drain_queue(q: asyncio.Queue) -> None:
    """Clear any stale triggers before pushing a new one — we only
    want the green Pi to react to the most recent event."""
    while not q.empty():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            return


def _save_event_clip(
    data: bytes, event_id: int, role: str, original_filename: str | None,
) -> str:
    """Write the uploaded MP4 to disk under a recognizable name.
    Returns the bare filename (so it can be stored in
    CameraEvent.{tee,green}_clip_filename without absolute paths)."""
    ext = "mp4"
    if original_filename and "." in original_filename:
        candidate = original_filename.rsplit(".", 1)[-1].lower()
        if candidate in ("mp4", "mov", "m4v", "webm"):
            ext = candidate
    fname = f"event-{event_id}-{role}-{secrets.token_hex(4)}.{ext}"
    out_path = CLIPS_DIR / fname
    out_path.write_bytes(data)
    return fname


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------

@router.post("/{token}/heartbeat")
def heartbeat(
    token: str,
    firmware_version: str | None = Form(None),
    db: Session = Depends(get_db),
):
    """Cheap keepalive the Pi calls every ~60 s. Touches last_seen_at
    so the admin UI can flag offline cameras. Optionally captures the
    Pi's firmware/git-commit version for diagnostics."""
    cam = _get_camera_by_token(token, db)
    if firmware_version:
        cam.firmware_version = firmware_version.strip()[:40]
    db.commit()
    return {
        "ok": True,
        "camera_id": cam.id,
        "enabled": cam.enabled,
        "assigned_role": cam.assigned_role,
        "course_id": cam.course_id,
        "assigned_hole": cam.assigned_hole,
        "paired_with_camera_id": cam.paired_with_camera_id,
        "server_time": _utcnow_naive().isoformat(),
    }


@router.post("/{token}/event-trigger")
async def event_trigger(
    token: str,
    session_id: str = Form(...),
    db: Session = Depends(get_db),
):
    """Tee-Pi-only: signals that a person was detected on the tee box
    and the Pi is about to start uploading. Creates the CameraEvent
    row and wakes up the paired green Pi (if any) so it can commit
    its pre-roll buffer too.

    Returns immediately so the tee Pi can keep its capture loop
    responsive. session_id is the Pi's UUID4 for this event; the
    matching /upload-event call reuses it.
    """
    cam = _get_camera_by_token(token, db)
    if cam.assigned_role != "tee":
        raise HTTPException(400, "event-trigger is only valid for tee cameras")

    sid = (session_id or "").strip()[:80]
    if not sid:
        raise HTTPException(400, "session_id is required")

    # Idempotency: if this session_id has already been registered (e.g.
    # the Pi retried because the first response was lost), return the
    # existing event row instead of failing on the unique constraint.
    existing = db.query(CameraEvent).filter(CameraEvent.session_id == sid).first()
    if existing is not None:
        return {
            "ok": True,
            "event_id": existing.id,
            "session_id": existing.session_id,
            "duplicate": True,
        }

    event = CameraEvent(
        session_id=sid,
        tee_camera_id=cam.id,
        green_camera_id=cam.paired_with_camera_id,
        course_id=cam.course_id,
        hole_number=cam.assigned_hole,
        status="triggered",
    )
    db.add(event)
    db.commit()
    db.refresh(event)

    # Wake the paired green Pi if there is one. asyncio.Queue.put_nowait
    # is fire-and-forget — if no green is currently long-polling, the
    # message sits in the queue for the next iteration.
    partner_id = cam.paired_with_camera_id
    if partner_id is not None:
        q = _queue_for(partner_id)
        _drain_queue(q)
        try:
            q.put_nowait({
                "session_id": sid,
                "event_id": event.id,
                "triggered_at": event.triggered_at.isoformat() if event.triggered_at else None,
                "tee_camera_id": cam.id,
                "hole_number": cam.assigned_hole,
            })
        except asyncio.QueueFull:
            log.warning(
                "cameras: trigger queue full for camera %s, dropping; green "
                "Pi probably offline", partner_id,
            )

    log.info(
        "cameras: event-trigger upload=%s session=%s hole=%d tee=%s green=%s",
        event.id, sid, cam.assigned_hole, cam.id, partner_id,
    )
    return {
        "ok": True,
        "event_id": event.id,
        "session_id": sid,
        "duplicate": False,
        "paired_with_camera_id": partner_id,
    }


@router.get("/{token}/poll-trigger")
async def poll_trigger(
    token: str,
    timeout: int = 25,
    db: Session = Depends(get_db),
):
    """Green Pi long-poll. Returns immediately if there's a pending
    trigger for this camera; otherwise holds open for `timeout`
    seconds waiting for one. On timeout returns {trigger: null} and
    the Pi reconnects.

    `timeout` capped to 60 s server-side so a stuck client can't pin
    a worker forever. 25 s default is below most LTE / Replit proxy
    idle timeouts.
    """
    cam = _get_camera_by_token(token, db)
    db.commit()

    wait_seconds = max(1, min(60, int(timeout or 25)))
    q = _queue_for(cam.id)
    try:
        msg = await asyncio.wait_for(q.get(), timeout=float(wait_seconds))
        return {"trigger": msg, "server_time": _utcnow_naive().isoformat()}
    except asyncio.TimeoutError:
        return {"trigger": None, "server_time": _utcnow_naive().isoformat()}


@router.post("/{token}/upload-event")
async def upload_event(
    token: str,
    session_id: str = Form(...),
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Both Pis call this with their recorded MP4 after the event.
    The role (tee | green) is inferred from the camera's
    assigned_role and verified against the event's tee/green camera
    ids.

    When the second clip lands (or the only clip, on an unpaired tee),
    kicks off the per-segment processing pipeline in a background
    thread. The HTTP request returns immediately so the Pi isn't
    blocked on the multi-minute tracer / composite work.
    """
    cam = _get_camera_by_token(token, db)
    sid = (session_id or "").strip()[:80]
    if not sid:
        raise HTTPException(400, "session_id is required")

    event = db.query(CameraEvent).filter(CameraEvent.session_id == sid).first()
    if event is None:
        raise HTTPException(404, "no event for that session_id; trigger first")

    if cam.id == event.tee_camera_id:
        role = "tee"
    elif event.green_camera_id is not None and cam.id == event.green_camera_id:
        role = "green"
    else:
        raise HTTPException(403, "this camera is not part of that event")

    data = await video.read()
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_EVENT_CLIP_BYTES:
        raise HTTPException(413, f"clip exceeds {MAX_EVENT_CLIP_BYTES // (1024*1024)} MB cap")

    fname = _save_event_clip(data, event.id, role, video.filename)
    if role == "tee":
        event.tee_clip_filename = fname
    else:
        event.green_clip_filename = fname

    # Decide the new status + whether we're ready to process. A paired
    # event needs both clips; an unpaired tee event needs only its own.
    has_tee = event.tee_clip_filename is not None
    has_green = event.green_clip_filename is not None
    is_paired = event.green_camera_id is not None
    ready_to_process = (has_tee and has_green) if is_paired else has_tee
    if ready_to_process:
        event.status = "paired_uploaded"
    elif has_tee:
        event.status = "tee_uploaded"
    elif has_green:
        # Unusual: green came in before tee. Hold; the tee upload
        # (which is mandatory) will flip the status.
        event.status = "tee_uploaded"
    db.commit()

    log.info(
        "cameras: upload-event event=%s role=%s file=%s status=%s",
        event.id, role, fname, event.status,
    )

    if ready_to_process:
        threading.Thread(
            target=_process_camera_event_job,
            args=(event.id,),
            daemon=True,
            name=f"camera-event-{event.id}",
        ).start()

    return {
        "ok": True,
        "event_id": event.id,
        "role": role,
        "filename": fname,
        "status": event.status,
        "ready_to_process": ready_to_process,
    }


# ---------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------

def _process_camera_event_job(event_id: int) -> None:
    """Run the existing per-segment pipeline on a fully-uploaded
    CameraEvent. The whole uploaded clip is treated as one swing —
    no audio/motion detection needed (the Pi already triggered on a
    real person). Updates the event's status + produced_clip_id.

    Owns its own DB session so the HTTP request that kicked us off
    has long since returned. Best-effort: any failure marks the event
    'failed' with last_error set instead of bubbling out.
    """
    # Imported here to dodge a circular import (admin.py imports
    # heavy modules at top level).
    from .admin import _process_long_upload_segments

    db = SessionLocal()
    try:
        event = db.get(CameraEvent, event_id)
        if event is None:
            log.warning("cameras: event %s vanished before processing", event_id)
            return

        tee_path = (
            CLIPS_DIR / event.tee_clip_filename
            if event.tee_clip_filename else None
        )
        green_path = (
            CLIPS_DIR / event.green_clip_filename
            if event.green_clip_filename else None
        )
        if tee_path is None or not tee_path.exists():
            event.status = "failed"
            event.last_error = "tee clip missing on disk"
            db.commit()
            return

        course = db.get(Course, event.course_id)
        if course is None:
            event.status = "failed"
            event.last_error = f"course {event.course_id} not found"
            db.commit()
            return

        info = probe_video_info(tee_path) or {}
        duration = float(info.get("duration") or 10.0)
        if duration <= 0.1:
            duration = 10.0

        # Single segment covering the whole uploaded clip. start=0
        # because the Pi already trimmed to the swing window.
        seg_list = [{
            "hole_number": int(event.hole_number),
            "start_sec": 0.0,
            "end_sec": float(duration),
        }]

        # captured_at = the trigger moment. Approximate — the actual
        # impact is ~2 s into the clip — but good enough for the
        # match-clip-to-player window logic.
        base_dt = event.triggered_at or _utcnow_naive()

        results = _process_long_upload_segments(
            db,
            course_id=event.course_id,
            camera_type="tee",
            base_dt=base_dt,
            src_path=tee_path,
            green_src_path=green_path if (green_path and green_path.exists()) else None,
            seg_list=seg_list,
            dual_camera=(green_path is not None and green_path.exists()),
            ai_tracer_model=None,
        )

        event = db.get(CameraEvent, event_id)  # re-fetch in case session was rolled back
        if event is None:
            return
        if results and results[0].get("ok") and results[0].get("clip_id"):
            event.status = "processed"
            event.produced_clip_id = int(results[0]["clip_id"])
            event.last_error = None
        else:
            event.status = "failed"
            event.last_error = (
                (results[0].get("error") if results else None)
                or "processing returned no result"
            )[:2000]
        db.commit()
        log.info(
            "cameras: event %s processed — status=%s clip=%s",
            event.id, event.status, event.produced_clip_id,
        )
    except Exception as exc:  # pragma: no cover
        log.exception("cameras: event %s processing crashed: %s", event_id, exc)
        try:
            db.rollback()
            event = db.get(CameraEvent, event_id)
            if event is not None:
                event.status = "failed"
                event.last_error = str(exc)[:2000]
                db.commit()
        except Exception:
            pass
    finally:
        db.close()
