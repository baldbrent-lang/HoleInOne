"""Device-facing endpoints for on-course capture (always-on Pi
cameras). All routes auth via the per-Camera auth_token in the URL —
the admin password and user JWTs are not in scope here. Endpoints
are designed to be called by an unattended Python agent running on a
Raspberry Pi in the field; see docs/field-deployment.md for the
deployment plan and routers/admin.py for the operator-facing CRUD
that mints the tokens these routes accept.

Flow per session (a session covers a whole group's time on the tee
box — potentially many swings — not a single swing):

  1. Tee Pi detects a person in its tee-box ROI for >=2 s.
  2. Tee Pi POSTs /event-trigger with a session_id (UUID4) it
     generated. Backend creates a CameraEvent row and pushes the
     trigger into an asyncio.Queue keyed by the paired green camera's
     id. Tee Pi starts recording from its pre-roll buffer.
  3. Green Pi is sitting in /poll-trigger long-polling. It wakes up
     with the session_id, commits its pre-roll, keeps recording.
  4. Tee Pi keeps recording until its tee box has been empty for
     no_person_timeout_seconds (default 5 s). Then it POSTs
     /event-stop, releases the writer, and uploads. /event-stop sets
     stop_signal_at on the event row.
  5. Green Pi polls /event-status every ~1 s while recording. When
     stop_signal_at is non-null it releases its writer and uploads.
     There is a hard runaway-safety cap (default 10 min) on both Pis
     in case /event-stop never arrives (tee crash, network split).
  6. Both Pis upload their MP4s via /upload-event with the shared
     session_id. As soon as the second clip lands (or the only clip,
     for unpaired-tee single-camera setups), a background thread
     runs the existing _process_long_upload_segments pipeline. Since
     the raw clips can contain multiple swings, the worker first
     auto-detects them via the same audio+motion detector the long-
     upload flow uses and produces one VideoClip per detected swing.

State that lives in-memory only:
  _pending_triggers: dict[camera_id -> asyncio.Queue] — the wake-up
  queue for the long-poll. Lost on backend restart, which is fine:
  the tee Pi will retry; the green Pi reconnects to /poll-trigger on
  its next iteration. Single-instance deployment assumed (Replit).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from sqlalchemy.orm import Session

from ..config import settings
from ..database import SessionLocal, get_db
from ..models import Camera, CameraEvent, Course, VideoClip
from ..services import storage
from ..services.video import probe_video_info

log = logging.getLogger("golfreelz.cameras")

router = APIRouter(prefix="/api/cameras", tags=["cameras"])

# In-memory live-stream state. Single uvicorn worker assumed.
# Swap for Redis if you scale to multiple workers.
_LIVE_FRAMES: dict[int, tuple[bytes, datetime]] = {}
_WATCHERS: dict[int, datetime] = {}
_LIVE_LOCK = threading.Lock()
WATCH_TTL = timedelta(seconds=10)
FRAME_TTL = timedelta(seconds=5)

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
MAX_EVENT_CLIP_BYTES = 500 * 1024 * 1024

# RESUMABLE UPLOAD. The tee's uplink is a USB cellular modem that
# reboots itself every ~30 seconds under load: alive for ~23s at
# ~125 KB/s, then gone for ~8s while it re-enumerates. A 6.5 MB clip
# needs ~52s of live wire, so it can NEVER complete inside one of the
# modem's lifetimes — every whole-file POST died mid-flight and threw
# away everything it had sent. Partial uploads live here between
# chunks so a clip can cross as many modem lifetimes as it needs.
PARTS_DIR = CLIPS_DIR.parent / "partial"
PARTS_DIR.mkdir(parents=True, exist_ok=True)

# A part with no activity for this long is abandoned (the Pi gave up,
# or the clip was re-encoded and restarted under a fresh size).
PART_MAX_AGE_SECONDS = 24 * 3600

# Upper bound on a single chunk. Generous — the Pi picks the real size
# (512 KB by default, ~4s of wire at the modem's rate).
PART_MAX_CHUNK_BYTES = 8 * 1024 * 1024

# Appends are short and serialized; the read of the incoming body
# happens OUTSIDE this lock so the event loop is never blocked on the
# network while holding it.
_PARTS_LOCK = threading.Lock()
_next_part_prune = 0.0


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


def _post_process_raw_clip(filename: str) -> None:
    """Re-encode a Pi-uploaded raw clip to browser-friendly H.264 +
    faststart and generate a poster JPG sibling for the production
    preview tile. Both ops are best-effort: failures log a warning
    and leave the original file intact (production reads via cv2,
    which handles mp4v fine, so capture itself is unaffected).

    Designed to be called from a daemon thread so the Pi's
    upload-event HTTP response isn't blocked on a multi-second
    re-encode.
    """
    from ..services.video import extract_thumbnail, transcode_for_web

    src = CLIPS_DIR / filename
    if not src.exists():
        log.warning("post-process: %s vanished before processing", filename)
        return
    transcode_for_web(src)
    extract_thumbnail(src)


def _save_event_clip(
    data: bytes,
    event_id: int,
    role: str,
    original_filename: str | None,
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


def _adopt_event_clip(
    src: Path,
    event_id: int,
    role: str,
    original_filename: str | None,
) -> str:
    """Move an already-complete file into CLIPS_DIR under the standard
    name. Same result as _save_event_clip, minus reading the whole clip
    into memory — the resumable path already has it on disk, and
    PARTS_DIR is a sibling of CLIPS_DIR so the rename is atomic."""
    ext = "mp4"
    if original_filename and "." in original_filename:
        candidate = original_filename.rsplit(".", 1)[-1].lower()
        if candidate in ("mp4", "mov", "m4v", "webm"):
            ext = candidate
    fname = f"event-{event_id}-{role}-{secrets.token_hex(4)}.{ext}"
    src.replace(CLIPS_DIR / fname)
    return fname


# ---------------------------------------------------------------------
# Resumable upload state
# ---------------------------------------------------------------------


def _part_paths(camera_id: int, upload_id: str) -> tuple[Path, Path]:
    """Where this camera's in-progress upload lives. Namespaced by
    camera id so the tee and green can share a session_id as the
    upload_id without colliding."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "", upload_id or "")[:80]
    if not safe:
        raise HTTPException(400, "upload_id is required")
    base = PARTS_DIR / f"cam{int(camera_id)}-{safe}"
    return base.with_suffix(".part"), base.with_suffix(".meta")


def _part_received(part: Path, meta: Path, total_size: int) -> int:
    """Bytes already banked for this upload, or 0 if what's on disk
    belongs to a different file.

    The size check matters: when the link keeps failing, the uploader
    re-encodes the clip at a lower bitrate and retries. Those are
    different bytes under the same session_id, so resuming on top of
    them would splice two encodes into one corrupt MP4.
    """
    if not part.exists():
        return 0
    prev = None
    try:
        prev = json.loads(meta.read_text()).get("total_size")
    except (OSError, ValueError, AttributeError):
        prev = None
    if total_size > 0 and prev is not None and int(prev) != int(total_size):
        log.info(
            "cameras: %s restarted — clip is now %d bytes, not %d (re-encoded)",
            part.name, int(total_size), int(prev),
        )
        part.unlink(missing_ok=True)
        meta.unlink(missing_ok=True)
        return 0
    try:
        return part.stat().st_size
    except OSError:
        return 0


def _prune_parts() -> None:
    """Drop abandoned partials so a fortnight of dead uploads can't
    fill the disk. Cheap, and rate-limited to once a minute."""
    global _next_part_prune
    now = time.time()
    if now < _next_part_prune:
        return
    _next_part_prune = now + 60.0
    try:
        for p in PARTS_DIR.iterdir():
            try:
                if now - p.stat().st_mtime > PART_MAX_AGE_SECONDS:
                    p.unlink(missing_ok=True)
            except OSError:
                continue
    except OSError:
        pass


def _resolve_event_role(
    cam: Camera, session_id: str, db: Session,
) -> tuple[CameraEvent, str]:
    """Find the event this upload belongs to and which side it is."""
    event = db.query(CameraEvent).filter(
        CameraEvent.session_id == session_id,
    ).first()
    if event is None:
        raise HTTPException(404, "no event for that session_id; trigger first")
    if cam.id == event.tee_camera_id:
        return event, "tee"
    if event.green_camera_id is not None and cam.id == event.green_camera_id:
        return event, "green"
    raise HTTPException(403, "this camera is not part of that event")


def _record_event_clip(
    event: CameraEvent,
    role: str,
    fname: str,
    recording_started_at: float | None,
    db: Session,
) -> dict:
    """Attach a stored clip to its event, advance the status, and kick
    off post-processing. Shared by the whole-file and resumable upload
    paths so they cannot drift apart."""
    # First-frame wall-clock time (epoch seconds from the Pi). Stored
    # per role so the dual-camera cut can align the green clip to the
    # tee cut's real-world moment instead of trusting frame indices.
    started_dt = None
    if recording_started_at is not None:
        try:
            started_dt = datetime.utcfromtimestamp(float(recording_started_at))
        except (ValueError, OverflowError, OSError):
            started_dt = None
    if role == "tee":
        event.tee_clip_filename = fname
        if started_dt is not None:
            event.tee_recording_started_at = started_dt
    else:
        event.green_clip_filename = fname
        if started_dt is not None:
            event.green_recording_started_at = started_dt

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

    # Re-encode + thumbnail in the background so the admin preview
    # works. Doesn't block the Pi's HTTP response. Runs in parallel
    # with the production job for paired events; both touch the file
    # via os-atomic rename, so cv2 reading from one inode while
    # ffmpeg writes a new one is fine.
    threading.Thread(
        target=_post_process_raw_clip,
        args=(fname,),
        daemon=True,
        name=f"post-process-{event.id}-{role}",
    ).start()

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
# Endpoints
# ---------------------------------------------------------------------


def battery_status(
    voltage: float | None,
    current_a: float | None,
    updated_at=None,
) -> dict | None:
    """Coarse battery readout for the dashboard from one voltage/current
    sample. LiFePO4's discharge curve is famously flat mid-charge, so
    the percent is honest-but-approximate — the buckets are what matter
    (do I need to bring the charger?). None when no telemetry exists."""
    if voltage is None:
        return None
    v = float(voltage)
    # Resting-ish voltage -> approx % for a 4S LiFePO4 under light load.
    curve = [
        (13.35, 100), (13.25, 90), (13.18, 80), (13.10, 65),
        (13.05, 55), (13.00, 40), (12.90, 30), (12.80, 20),
        (12.60, 10), (12.10, 5),
    ]
    pct = 2
    for thresh, p in curve:
        if v >= thresh:
            pct = p
            break
    level = (
        "full" if pct >= 80
        else "good" if pct >= 40
        else "low" if pct >= 20
        else "critical"
    )
    watts = (
        round(v * float(current_a), 1)
        if current_a is not None and float(current_a) > 0.05
        else None
    )
    est_days = None
    cap = float(settings.battery_capacity_wh or 0)
    if cap > 0:
        draw_w = watts if (watts and watts > 1.0) else 7.0
        est_days = round(
            (cap * pct / 100.0)
            / (draw_w * max(1.0, float(settings.battery_active_hours))),
            1,
        )
    return {
        "voltage": round(v, 2),
        "current_a": round(float(current_a), 2) if current_a is not None else None,
        "watts": watts,
        "percent": pct,
        "level": level,
        "est_days": est_days,
        "low": bool(v < float(settings.battery_low_volts)),
        "updated_at": updated_at.isoformat() if updated_at else None,
    }


@router.post("/{token}/heartbeat")
def heartbeat(
    token: str,
    firmware_version: str | None = Form(None),
    battery_voltage: float | None = Form(None),
    battery_current_a: float | None = Form(None),
    db: Session = Depends(get_db),
):
    """Cheap keepalive the Pi calls every ~60 s. Touches last_seen_at
    so the admin UI can flag offline cameras. Optionally captures the
    Pi's firmware/git-commit version for diagnostics, plus battery
    telemetry (INA226) on battery-powered rigs."""
    cam = _get_camera_by_token(token, db)
    if firmware_version:
        cam.firmware_version = firmware_version.strip()[:40]
    if battery_voltage is not None:
        cam.battery_voltage = float(battery_voltage)
        cam.battery_current_a = (
            float(battery_current_a) if battery_current_a is not None else None
        )
        cam.battery_updated_at = _utcnow_naive()
        _thr = float(settings.battery_low_volts)
        if cam.battery_voltage >= _thr + 0.15:
            # Recharged / swapped — re-arm the one-shot alert.
            cam.battery_low_notified_at = None
        elif (
            cam.battery_voltage < _thr
            and cam.battery_low_notified_at is None
        ):
            cam.battery_low_notified_at = _utcnow_naive()
            try:
                from ..services import notifications

                _name = cam.name or f"camera #{cam.id}"
                notifications.send_email(
                    settings.admin_alert_email or None,
                    f"GolfReelz: low battery on {_name}",
                    (
                        f"{_name} (hole {cam.assigned_hole}, "
                        f"{cam.assigned_role}) reported "
                        f"{cam.battery_voltage:.2f}V — below the "
                        f"{_thr:.2f}V alert threshold. Plan a battery "
                        "swap in the next day or two."
                    ),
                )
                log.info(
                    "battery: LOW alert for camera %s (%.2fV)",
                    cam.id, cam.battery_voltage,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("battery: low alert send failed: %s", exc)
    db.commit()
    return {
        "ok": True,
        "camera_id": cam.id,
        "enabled": cam.enabled,
        # Pi can read this to skip its person-detection loop entirely
        # while paused. Older agents that ignore it are still covered:
        # /event-trigger refuses to create events when this is False.
        "triggering_enabled": cam.triggering_enabled,
        "assigned_role": cam.assigned_role,
        "course_id": cam.course_id,
        "assigned_hole": cam.assigned_hole,
        "paired_with_camera_id": cam.paired_with_camera_id,
        "server_time": _utcnow_naive().isoformat(),
    }


# Operator-requested captures, camera_id -> seconds. Set by the admin
# Capture button, consumed by the tee Pi's next watch-status poll.
# In-memory on purpose: a request that doesn't reach a camera within a
# poll interval is stale and should evaporate, not queue up.
_CAPTURE_LOCK = threading.Lock()
_CAPTURE_REQUESTS: dict[int, int] = {}


def request_capture(camera_id: int, seconds: int) -> None:
    """Queue a one-shot capture for a camera. Called from the admin
    router; delivered on the camera's next watch-status poll."""
    with _CAPTURE_LOCK:
        _CAPTURE_REQUESTS[camera_id] = int(seconds)


@router.get("/{token}/watch-status")
def watch_status(token: str, db: Session = Depends(get_db)):
    """Pi polls this. If watching=true, Pi should push live frames.

    Also the delivery channel for an operator-requested capture. The tee
    Pi already polls this every few seconds for the live view, so riding
    along here means no new endpoint on the device and no extra traffic.
    `capture_seconds` is CONSUMED on read — one poll, one capture.
    """
    cam = _get_camera_by_token(token, db)
    with _LIVE_LOCK:
        last = _WATCHERS.get(cam.id)
        watching = bool(last and _utcnow_naive() - last < WATCH_TTL)
    with _CAPTURE_LOCK:
        capture_seconds = _CAPTURE_REQUESTS.pop(cam.id, None)
    if capture_seconds:
        log.info(
            "cameras: delivering capture request to camera %s (%ss)",
            cam.id, capture_seconds,
        )
    return {"watching": watching, "capture_seconds": capture_seconds}


@router.post("/{token}/live-frame")
async def post_live_frame(
    token: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """Pi POSTs JPEG bytes as the request body."""
    cam = _get_camera_by_token(token, db)
    body = await request.body()
    if not body:
        raise HTTPException(400, "empty frame")
    if len(body) > 500_000:
        raise HTTPException(413, "frame too large (max 500KB)")
    with _LIVE_LOCK:
        _LIVE_FRAMES[cam.id] = (body, _utcnow_naive())
    return {"ok": True}


@router.post("/{token}/event-trigger")
async def event_trigger(
    token: str,
    session_id: str = Form(...),
    recover: bool = Form(False),
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

    # Operator kill-switch: camera is online but triggering is paused
    # (e.g. powered on indoors for testing). Touch last_seen_at (done
    # in _get_camera_by_token) but don't create an event — return a
    # benign signal so the Pi doesn't bother recording / uploading.
    if not cam.triggering_enabled:
        db.commit()  # persist the last_seen_at bump
        return {"ok": True, "triggering_disabled": True, "event_id": None}

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
    #
    # ...unless this is a RECOVERY. A trigger can be lost — the backend
    # was mid-deploy, the modem was between lives — and the tee then
    # records and uploads a clip for a session the server never heard
    # of. Re-registering the session lets that footage land instead of
    # being discarded, but the swing is minutes in the past: waking the
    # green now would only have it record an empty tee box and file it
    # under a session whose green half is long gone.
    partner_id = None if recover else cam.paired_with_camera_id
    if recover:
        # No green half is coming, so don't leave the event waiting for
        # one — the tee-only path can carry it.
        event.green_camera_id = None
        db.commit()
        log.warning(
            "cameras: recovered lost trigger session=%s as event=%s "
            "(tee-only; the green half cannot be recovered)", sid, event.id,
        )
    if partner_id is not None:
        q = _queue_for(partner_id)
        _drain_queue(q)
        try:
            q.put_nowait(
                {
                    "session_id": sid,
                    "event_id": event.id,
                    "triggered_at": event.triggered_at.isoformat()
                    if event.triggered_at
                    else None,
                    "tee_camera_id": cam.id,
                    "hole_number": cam.assigned_hole,
                }
            )
        except asyncio.QueueFull:
            log.warning(
                "cameras: trigger queue full for camera %s, dropping; green "
                "Pi probably offline",
                partner_id,
            )

    log.info(
        "cameras: event-trigger upload=%s session=%s hole=%d tee=%s green=%s",
        event.id,
        sid,
        cam.assigned_hole,
        cam.id,
        partner_id,
    )
    return {
        "ok": True,
        "event_id": event.id,
        "session_id": sid,
        "duplicate": False,
        "paired_with_camera_id": partner_id,
    }


@router.post("/{token}/event-stop")
def event_stop(
    token: str,
    session_id: str = Form(...),
    db: Session = Depends(get_db),
):
    """Tee-Pi-only: signals that recording has ended for this session.
    Called right after the tee Pi releases its VideoWriter, before the
    (potentially slow) upload, so the paired green Pi — which polls
    /event-status — can stop recording at roughly the same moment.

    Idempotent: repeat calls with the same session_id keep the first
    stop_signal_at and return ok=True. Cheap and fire-and-forget.
    """
    cam = _get_camera_by_token(token, db)
    if cam.assigned_role != "tee":
        raise HTTPException(400, "event-stop is only valid for tee cameras")

    sid = (session_id or "").strip()[:80]
    if not sid:
        raise HTTPException(400, "session_id is required")

    event = db.query(CameraEvent).filter(CameraEvent.session_id == sid).first()
    if event is None:
        raise HTTPException(404, "no event for that session_id; trigger first")
    if event.tee_camera_id != cam.id:
        raise HTTPException(403, "this camera is not the tee for that event")

    if event.stop_signal_at is None:
        event.stop_signal_at = _utcnow_naive()
        db.commit()
    return {
        "ok": True,
        "event_id": event.id,
        "stop_signal_at": event.stop_signal_at.isoformat(),
    }


@router.get("/{token}/event-status")
def event_status(
    token: str,
    session_id: str,
    db: Session = Depends(get_db),
):
    """Either Pi can poll this. Reports whether the tee has signalled
    end-of-session (via /event-stop) for `session_id`. The green Pi
    uses this to mirror the tee's stop decision instead of recording
    a hard-coded duration."""
    cam = _get_camera_by_token(token, db)
    db.commit()  # touch last_seen_at

    sid = (session_id or "").strip()[:80]
    if not sid:
        raise HTTPException(400, "session_id is required")

    event = db.query(CameraEvent).filter(CameraEvent.session_id == sid).first()
    if event is None:
        raise HTTPException(404, "no event for that session_id")
    # Auth: only the tee or paired green for this event may peek at it.
    if cam.id not in (event.tee_camera_id, event.green_camera_id):
        raise HTTPException(403, "this camera is not part of that event")

    return {
        "session_id": sid,
        "stop_signal": event.stop_signal_at is not None,
        "stop_signal_at": (
            event.stop_signal_at.isoformat() if event.stop_signal_at else None
        ),
        "server_time": _utcnow_naive().isoformat(),
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
    recording_started_at: float | None = Form(None),
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

    event, role = _resolve_event_role(cam, sid, db)

    data = await video.read()
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_EVENT_CLIP_BYTES:
        raise HTTPException(
            413, f"clip exceeds {MAX_EVENT_CLIP_BYTES // (1024 * 1024)} MB cap"
        )

    fname = _save_event_clip(data, event.id, role, video.filename)
    return _record_event_clip(event, role, fname, recording_started_at, db)


@router.get("/{token}/upload-status")
def upload_status(
    token: str,
    upload_id: str,
    total_size: int = 0,
    db: Session = Depends(get_db),
):
    """How many bytes of this clip the server already holds.

    The Pi calls this before every attempt so it resumes where the last
    one died rather than starting over. Also the recovery path when a
    chunk's response is lost in flight — the bytes may well have landed,
    and this is how the Pi finds out.
    """
    cam = _get_camera_by_token(token, db)
    db.commit()
    _prune_parts()
    part, meta = _part_paths(cam.id, upload_id)
    with _PARTS_LOCK:
        received = _part_received(part, meta, int(total_size or 0))
    return {
        "received": received,
        "total_size": int(total_size or 0),
        "max_chunk_bytes": PART_MAX_CHUNK_BYTES,
    }


@router.post("/{token}/upload-chunk")
async def upload_chunk(
    token: str,
    upload_id: str = Form(...),
    offset: int = Form(...),
    total_size: int = Form(...),
    chunk: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Append one slice of a clip.

    `offset` must match what the server already holds. When it doesn't
    the answer is `resync` plus the true offset rather than an error —
    a mismatch means the Pi's idea of progress is stale (a chunk landed
    but its response never made it back through a dying modem), which is
    routine here, not a fault.
    """
    cam = _get_camera_by_token(token, db)
    db.commit()
    if total_size <= 0:
        raise HTTPException(400, "total_size is required")
    if total_size > MAX_EVENT_CLIP_BYTES:
        raise HTTPException(
            413, f"clip exceeds {MAX_EVENT_CLIP_BYTES // (1024 * 1024)} MB cap"
        )
    part, meta = _part_paths(cam.id, upload_id)

    # Read the body BEFORE taking the lock — this awaits on the network
    # and must not hold up other cameras' appends.
    body = await chunk.read()
    if not body:
        raise HTTPException(400, "empty chunk")
    if len(body) > PART_MAX_CHUNK_BYTES:
        raise HTTPException(413, "chunk too large")

    with _PARTS_LOCK:
        received = _part_received(part, meta, total_size)
        if int(offset) != received:
            return {
                "ok": False,
                "resync": True,
                "received": received,
                "total_size": total_size,
            }
        if received + len(body) > total_size:
            raise HTTPException(400, "chunk runs past the declared total_size")
        with open(part, "ab") as fh:
            fh.write(body)
        received += len(body)
        meta.write_text(json.dumps({
            "total_size": int(total_size),
            "received": int(received),
            "updated_at": time.time(),
        }))

    return {
        "ok": True,
        "received": received,
        "total_size": total_size,
        "complete": received >= total_size,
    }


@router.post("/{token}/upload-complete")
def upload_complete(
    token: str,
    session_id: str = Form(...),
    upload_id: str = Form(...),
    total_size: int = Form(...),
    filename: str | None = Form(None),
    recording_started_at: float | None = Form(None),
    db: Session = Depends(get_db),
):
    """Seal a fully-transferred clip and hand it to the normal pipeline.

    Short of the declared size, this reports `resync` with what's
    actually banked instead of failing, so the Pi simply carries on
    sending from there.
    """
    cam = _get_camera_by_token(token, db)
    sid = (session_id or "").strip()[:80]
    if not sid:
        raise HTTPException(400, "session_id is required")
    event, role = _resolve_event_role(cam, sid, db)
    part, meta = _part_paths(cam.id, upload_id)

    with _PARTS_LOCK:
        received = _part_received(part, meta, int(total_size))
        if received < int(total_size):
            return {
                "ok": False,
                "resync": True,
                "received": received,
                "total_size": int(total_size),
            }
        fname = _adopt_event_clip(part, event.id, role, filename)
        meta.unlink(missing_ok=True)

    log.info(
        "cameras: resumable upload complete — event=%s role=%s %.1f MB",
        event.id, role, int(total_size) / (1024 * 1024),
    )
    return _record_event_clip(event, role, fname, recording_started_at, db)


# ---------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------


def _process_camera_event_job(event_id: int) -> None:
    """Hand off a fully-uploaded CameraEvent to the long-upload
    pipeline so it benefits from the existing multi-swing auto-detect
    + admin edit-wizard. Creates a LongVideoUpload row pointing at the
    raw files the Pis already saved, links it back via
    LongVideoUpload.camera_event_id, then kicks off the standard
    background job. The Production page reads it as a long-upload —
    same card, same buttons (Edit, Re-Produce, Broadcast, Delete).

    Idempotent: if the event already has a linked LongVideoUpload, we
    re-use it instead of creating a duplicate.

    Owns its own DB session so the HTTP request that kicked us off
    has long since returned. Best-effort: any failure marks the event
    'failed' with last_error set instead of bubbling out.
    """
    # Imported here to dodge a circular import (admin.py imports
    # heavy modules at top level).
    from .admin import enqueue_produce_job
    from ..models import LongVideoUpload

    db = SessionLocal()
    try:
        event = db.get(CameraEvent, event_id)
        if event is None:
            log.warning("cameras: event %s vanished before processing", event_id)
            return

        # Rehydrate raws from object storage if the ephemeral disk lost them
        # (e.g. reprocessing an event after a redeploy).
        if event.tee_clip_filename:
            storage.ensure_local(CLIPS_DIR, event.tee_clip_filename)
        if event.green_clip_filename:
            storage.ensure_local(CLIPS_DIR, event.green_clip_filename)

        tee_path = (
            CLIPS_DIR / event.tee_clip_filename if event.tee_clip_filename else None
        )
        if tee_path is None or not tee_path.exists():
            event.status = "failed"
            event.last_error = "tee clip missing on disk"
            db.commit()
            return

        # Re-encode the raws to browser-friendly H.264 + extract a
        # thumbnail in this thread before production opens them. Cheap
        # no-op when the upload-time hook already did it.
        if event.tee_clip_filename:
            _post_process_raw_clip(event.tee_clip_filename)
        if event.green_clip_filename:
            _post_process_raw_clip(event.green_clip_filename)

        green_filename = event.green_clip_filename
        green_path = CLIPS_DIR / green_filename if green_filename else None
        has_green = green_path is not None and green_path.exists()

        # Re-use a previously-linked LongVideoUpload row if any (handles
        # Re-Produce flowing back through this code path).
        lvu = (
            db.query(LongVideoUpload)
            .filter(LongVideoUpload.camera_event_id == event.id)
            .first()
        )
        if lvu is None:
            lvu = LongVideoUpload(
                course_id=event.course_id,
                camera_type="tee",
                base_captured_at=event.triggered_at or _utcnow_naive(),
                tee_filename=event.tee_clip_filename,
                green_filename=green_filename if has_green else None,
                tee_original_filename=f"camera-event-{event.id}-tee.mp4",
                green_original_filename=(
                    f"camera-event-{event.id}-green.mp4" if has_green else None
                ),
                swing_count="multiple",
                camera_event_id=event.id,
                processing_status="pending",
            )
            db.add(lvu)
            db.commit()
            db.refresh(lvu)
        elif has_green and not lvu.green_filename:
            # Re-processing an event that was produced TEE-ONLY (fallback)
            # and whose green half has since arrived — backfill it so this
            # run makes the paired composite instead of tee-only again.
            lvu.green_filename = green_filename
            lvu.green_original_filename = f"camera-event-{event.id}-green.mp4"
            db.commit()
            log.info(
                "cameras: event %s — green arrived late, upgrading upload %s "
                "to paired", event.id, lvu.id,
            )

        # No pre-screen: the produce job's own pose pass doubles as the
        # non-golf screen — zero swing candidates and the job deletes
        # the upload + this event itself (a separate mediapipe scan
        # here would just duplicate that work).
        log.info(
            "cameras: event %s -> long-upload %s (auto-detect swings)",
            event.id, lvu.id,
        )

        # Status flips to 'processed' (or 'failed') from inside the
        # produce job's LongVideoUpload bookkeeping; mirror that onto the
        # CameraEvent at the end. We block this thread on the job so the
        # camera-event row gets its terminal status set in one pass.
        try:
            # THE produce path — the same one Debug3 and Re-Produce run.
            # This used to call _run_long_upload_job with motion-only
            # audio+motion detection; swings come from the pose detector
            # now, and the flight from Debug3, so a capture behaves
            # identically however it was started. A camera covers one
            # par-3, so its hole is passed through rather than inferred.
            # wait=True: produce is queued and serialised with every
            # other upload, but this thread blocks until its turn is
            # done so the CameraEvent's terminal status is stamped from
            # the actual outcome rather than from "we started it".
            # Read what we need, then RELEASE THE POOLED CONNECTION
            # before blocking. The produce queue serialises jobs, so
            # this wait grows with queue depth — the Nth camera event
            # waits N produce durations. Holding a session across it
            # exhausts the pool (SQLAlchemy defaults to 5 + 10
            # overflow), and once it is dry EVERY endpoint 500s:
            # heartbeat, poll-trigger, upload-event, the admin UI. A
            # camera politely waiting its turn took the whole backend
            # down, which is how it presented in the field.
            _lvu_id = lvu.id
            _hole = int(event.hole_number)
            db.close()
            try:
                enqueue_produce_job(
                    upload_id=_lvu_id, hole_number=_hole, wait=True,
                )
            finally:
                db = SessionLocal()
        except Exception as exc:
            log.exception(
                "cameras: event %s produce job crashed: %s", event_id, exc,
            )
            try:
                db.close()
            except Exception:  # noqa: BLE001
                pass
            db = SessionLocal()

        # Fresh session, so no stale identity map to expire.
        event = db.get(CameraEvent, event_id)
        lvu = db.get(LongVideoUpload, _lvu_id)
        if event is None or lvu is None:
            return

        # The auto-swing detector treats "no paired audio+motion peaks"
        # as a hard failure, which is right for the long-upload UI but
        # wrong for the camera path: the Pi already captured + uploaded
        # the raws (visible on the production page), there just wasn't
        # an actual swing to produce a clip from. Treat this as a
        # successful capture so the event isn't flagged red.
        no_swings = (
            lvu.processing_status == "failed"
            and (lvu.last_error or "").startswith("no swings detected")
        )

        if lvu.processing_status == "completed" or no_swings:
            event.status = "processed"
            event.last_error = None
            # Pick the first produced clip for the back-compat
            # produced_clip_id pointer (the listing surfaces the full
            # set via the linked long-upload).
            from ..models import VideoClip
            first_clip = (
                db.query(VideoClip)
                .filter(VideoClip.long_upload_id == lvu.id)
                .order_by(VideoClip.captured_at.asc().nulls_last())
                .first()
            )
            if first_clip is not None:
                event.produced_clip_id = first_clip.id
        else:
            event.status = "failed"
            event.last_error = (lvu.last_error or "long-upload job did not complete")[:2000]
        db.commit()
        log.info(
            "cameras: event %s done — status=%s lvu=%s segs=%s produced=%s",
            event.id,
            event.status,
            lvu.id,
            lvu.last_n_segments,
            lvu.last_n_succeeded,
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


# Tee-only fallback ---------------------------------------------------------
# Events currently being force-produced tee-only, so the periodic sweep
# doesn't kick the same one twice while it's still running.
_fallback_inflight: set[int] = set()
_fallback_lock = threading.Lock()


def _tee_only_fallback_sweep() -> None:
    """Produce a TEE-ONLY clip for any paired event whose green half never
    arrived within the fallback window. _process_camera_event_job already
    handles a missing green (green_clip_filename is None -> tee-only), so we
    just kick it. If green shows up later, upload_event flips the event back
    to paired and re-produces with both."""
    secs = int(settings.camera_tee_only_fallback_seconds or 0)
    if secs <= 0:
        return
    cutoff = _utcnow_naive() - timedelta(seconds=secs)
    db = SessionLocal()
    try:
        ids = [
            e.id
            for e in db.query(CameraEvent)
            .filter(
                CameraEvent.status == "tee_uploaded",
                CameraEvent.tee_clip_filename.isnot(None),
                CameraEvent.triggered_at < cutoff,
            )
            .all()
        ]
    finally:
        db.close()

    for eid in ids:
        with _fallback_lock:
            if eid in _fallback_inflight:
                continue
            _fallback_inflight.add(eid)

        def _run(event_id: int = eid) -> None:
            try:
                log.info(
                    "cameras: tee-only fallback — green never arrived for "
                    "event %s, producing from tee alone", event_id,
                )
                _process_camera_event_job(event_id)
            finally:
                with _fallback_lock:
                    _fallback_inflight.discard(event_id)

        threading.Thread(
            target=_run, daemon=True, name=f"tee-only-fallback-{eid}"
        ).start()


def start_tee_only_fallback_sweeper(interval_sec: float = 60.0) -> None:
    """Periodically produce tee-only for events whose green half is overdue.
    No-op when the fallback is disabled (camera_tee_only_fallback_seconds=0).
    Idempotent to start."""
    if int(settings.camera_tee_only_fallback_seconds or 0) <= 0:
        return
    if getattr(start_tee_only_fallback_sweeper, "_started", False):
        return
    start_tee_only_fallback_sweeper._started = True  # type: ignore[attr-defined]

    def _loop() -> None:
        log.info(
            "cameras: tee-only fallback sweeper started (%ss window)",
            settings.camera_tee_only_fallback_seconds,
        )
        while True:
            try:
                _tee_only_fallback_sweep()
            except Exception as exc:  # noqa: BLE001
                log.warning("cameras: tee-only fallback sweep failed: %s", exc)
            time.sleep(interval_sec)

    threading.Thread(
        target=_loop, daemon=True, name="tee-only-fallback-sweeper"
    ).start()
