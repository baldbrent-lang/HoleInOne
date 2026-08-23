"""Server-side "Pull new from prod".

Lets the operator click a button on the dev Production page instead of
running scripts/mirror_prod_to_dev.py. The dev backend fetches new camera
clips from a source (prod) backend and imports them into the configured
local course, running the same produce pipeline as a live camera event.

Dedup + delete-persistence via the MirroredProdEvent ledger table: a
source event is imported at most once, and stays skipped even if the
operator later deletes it locally (a delete stays a delete).

Configured via settings.mirror_* (env: MIRROR_SOURCE_URL,
MIRROR_SOURCE_PASSWORD, MIRROR_COURSE_ID). No-op / button hidden when
MIRROR_COURSE_ID is 0.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from ..config import settings
from ..database import SessionLocal
from ..models import LongVideoUpload, MirroredProdEvent

log = logging.getLogger("golfreelz.mirror")

CLIPS_DIR = Path(__file__).resolve().parents[2] / settings.upload_dir / "clips"

# One pull at a time; progress is polled by the UI.
_state = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "total": 0,     # new events to import this run
    "done": 0,      # imported so far
    "failed": 0,
    "last_error": None,
}
_lock = threading.Lock()


def status() -> dict:
    with _lock:
        return dict(_state)


def _source_url() -> str:
    return (settings.mirror_source_url or "").rstrip("/")


def _source_pw() -> str:
    return settings.mirror_source_password or settings.admin_password


def _get_json(url: str):
    req = urllib.request.Request(url, headers={"X-Admin-Password": _source_pw()})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"X-Admin-Password": _source_pw()})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def _list_source_events() -> list:
    """Page through the source's camera-events (small pages — a big
    limit makes the source probe every clip at once and 500s)."""
    events: list = []
    offset, page = 0, 10
    while True:
        url = f"{_source_url()}/api/admin/camera-events?limit={page}&offset={offset}"
        try:
            batch = _get_json(url)
        except urllib.error.HTTPError as e:
            log.warning("mirror: events %s-%s HTTP %s", offset, offset + page, e.code)
            offset += page
            if offset > 3000:
                break
            continue
        if not batch:
            break
        events.extend(batch)
        if len(batch) < page:
            break
        offset += page
    return events


def _import_one(ev: dict) -> bool:
    """Download one source event's clip(s), record it in the ledger, and
    kick the produce pipeline. The ledger row is committed WITH the upload
    row, so a later produce failure can't cause a re-import."""
    # Lazy imports: _run_long_upload_job lives in admin.py (circular at
    # module load); video helpers are heavy.
    from ..routers.admin import run_produce_job
    from ..services.video import extract_thumbnail, transcode_for_web

    course_id = int(settings.mirror_course_id)
    eid = str(ev["id"])
    tee_url = ev.get("tee_url")
    if not tee_url:
        return False

    hole = int(ev.get("hole_number") or 1)
    tag = secrets.token_hex(6)
    tee_name = f"mirror-{course_id}-{eid}-tee-{tag}.mp4"
    tee_path = CLIPS_DIR / tee_name
    tee_path.parent.mkdir(parents=True, exist_ok=True)
    tee_path.write_bytes(_download(tee_url))

    green_name = None
    if ev.get("green_url"):
        green_name = f"mirror-{course_id}-{eid}-green-{tag}.mp4"
        (CLIPS_DIR / green_name).write_bytes(_download(ev["green_url"]))

    try:
        ts = ev.get("triggered_at")
        base_dt = datetime.fromisoformat(ts.replace("Z", "")) if ts else datetime.utcnow()
    except Exception:  # noqa: BLE001
        base_dt = datetime.utcnow()

    db = SessionLocal()
    try:
        lvu = LongVideoUpload(
            course_id=course_id,
            camera_type="tee",
            base_captured_at=base_dt,
            tee_filename=tee_name,
            green_filename=green_name,
            tee_original_filename=f"prod-event-{eid}-tee.mp4",
            green_original_filename=(
                f"prod-event-{eid}-green.mp4" if green_name else None
            ),
            swing_count="multiple",
            processing_status="pending",
            # THE HOLE, WRITTEN DOWN. A mirrored upload gets no
            # CameraEvent, so nothing on the row said which hole it was
            # of and `_hole_for_upload` had to fall back to 1 until a
            # produce stamped a clip. Anything filing data under the
            # answer -- the tee/green view map did -- pooled every
            # mirrored pair on the system into hole 1's slot, whatever
            # cameras shot them. The source told us the hole right here;
            # keeping it costs nothing.
            # WHEN EACH CAMERA STARTED ROLLING, carried over for the
            # same reason as the hole above. These are CameraEvent
            # columns and a mirrored upload gets no CameraEvent, so
            # without keeping them here a mirrored pair falls through to
            # an assumed zero offset: every green boundary in the clip
            # out by however far apart the two recordings really started,
            # and no time of day anywhere in the UI -- on footage whose
            # exact offset the source knew and handed over in the same
            # listing it handed the clips over in.
            edit_metrics={"source_hole_number": hole,
                          "source_event_id": eid,
                          "source_tee_started_at":
                              ev.get("tee_recording_started_at"),
                          "source_green_started_at":
                              ev.get("green_recording_started_at")},
        )
        db.add(lvu)
        # Mark imported in the SAME transaction — so if produce later fails,
        # we still never re-download/re-import this event.
        db.add(MirroredProdEvent(source_event_id=eid))
        db.commit()
        db.refresh(lvu)
        upload_id = lvu.id
    finally:
        db.close()

    # Browser preview + thumbnail (best-effort).
    for p in [tee_path] + ([CLIPS_DIR / green_name] if green_name else []):
        try:
            transcode_for_web(p)
            extract_thumbnail(p)
        except Exception as exc:  # noqa: BLE001
            log.warning("mirror: preview gen failed for %s: %s", p.name, exc)

    # Same produce path a live camera event uses -- `run_produce_job`,
    # which is the ball scan followed by the wizard's renderer. It used
    # to call `_run_long_upload_job` with the audio+motion detector,
    # which is a THIRD pipeline: a mirrored clip then behaved unlike the
    # same footage produced on prod, which defeats the point of mirroring
    # it. The source told us the hole; pass it rather than let the
    # produce infer one.
    try:
        run_produce_job(upload_id=upload_id, hole_number=hole)
    except Exception as exc:  # noqa: BLE001
        log.warning("mirror: produce failed for event %s: %s", eid, exc)
    return True


def _worker() -> None:
    try:
        events = _list_source_events()
        events = list(reversed(events))  # oldest first → dev order matches prod
        db = SessionLocal()
        try:
            done_ids = {r.source_event_id for r in db.query(MirroredProdEvent).all()}
        finally:
            db.close()
        todo = [e for e in events if e.get("tee_url") and str(e["id"]) not in done_ids]
        with _lock:
            _state.update({"total": len(todo), "done": 0, "failed": 0})
        log.info("mirror: %d source event(s), %d new to import", len(events), len(todo))
        for e in todo:
            try:
                ok = _import_one(e)
            except Exception as exc:  # noqa: BLE001
                log.warning("mirror: import event %s failed: %s", e.get("id"), exc)
                ok = False
            with _lock:
                if ok:
                    _state["done"] += 1
                else:
                    _state["failed"] += 1
    except Exception as exc:  # noqa: BLE001
        log.exception("mirror: pull failed: %s", exc)
        with _lock:
            _state["last_error"] = str(exc)[:300]
    finally:
        with _lock:
            _state["running"] = False
            _state["finished_at"] = datetime.utcnow().isoformat()


def start_pull() -> dict:
    """Kick a background pull if one isn't already running."""
    if not settings.mirror_course_id:
        return {"ok": False, "error": "mirror not configured (MIRROR_COURSE_ID unset)"}
    with _lock:
        if _state["running"]:
            return {"ok": True, "already_running": True, **_state}
        _state.update({
            "running": True,
            "started_at": datetime.utcnow().isoformat(),
            "finished_at": None,
            "total": 0, "done": 0, "failed": 0, "last_error": None,
        })
    threading.Thread(target=_worker, daemon=True, name="mirror-pull").start()
    return {"ok": True, "started": True}
