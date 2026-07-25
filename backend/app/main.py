from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Silence the libav decoder inside cv2 before any service module
# imports cv2. The Pi's cv2.VideoWriter-with-mp4v output contains
# minor bitstream quirks that libav's H.264 decoder logs as NAL-unit
# warnings on every frame. Production still produces the right
# composite, but the warnings flood uvicorn's stderr at 30 fps × N
# clips. -8 = AV_LOG_QUIET — caught at first FFmpeg backend init,
# so this has to land before any cv2.VideoCapture call.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import inspect, text
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import settings
from .database import Base, engine
from .routers import admin, auth, broadcast, cameras, gallery, operator, public, webhooks
from .services import storage

# Our internal loggers (`golfreelz.tracer`, `golfreelz.admin`, etc.) default to
# WARNING and uvicorn doesn't configure them, so INFO diagnostics were never
# making it to the Replit console. Attach a stdout handler at INFO once at
# import time so retry / encode / heatmap stats show up next to uvicorn's own
# request logs.
_glog = logging.getLogger("golfreelz")
_glog.setLevel(logging.INFO)
if not _glog.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _glog.addHandler(_h)

app = FastAPI(title="GolfReelz API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_methods=["*"],
    allow_headers=["*"],
)


def _migrate() -> None:
    """Lightweight idempotent column additions for in-place upgrades.

    SQLAlchemy's create_all only creates missing tables, not missing
    columns on existing ones. Replit's hosted DB persists across deploys,
    so we ALTER TABLE here for any new fields V0+ has added.
    """
    inspector = inspect(engine)
    if "participants" not in inspector.get_table_names():
        return
    cols_info = {c["name"]: c for c in inspector.get_columns("participants")}
    cols = set(cols_info)
    statements = []
    if "selfie_path" not in cols:
        statements.append("ALTER TABLE participants ADD COLUMN selfie_path VARCHAR(500)")
    if "appearance_embedding" not in cols:
        # Both SQLite and Postgres accept JSON; SQLite stores as TEXT.
        statements.append("ALTER TABLE participants ADD COLUMN appearance_embedding JSON")
    if "summary_sent_at" not in cols:
        statements.append("ALTER TABLE participants ADD COLUMN summary_sent_at TIMESTAMP")
    if "user_id" not in cols:
        statements.append("ALTER TABLE participants ADD COLUMN user_id INTEGER")
    if "refunded_at" not in cols:
        statements.append("ALTER TABLE participants ADD COLUMN refunded_at TIMESTAMP")
    # Old schema had playing_order NOT NULL. The matcher no longer uses it,
    # so drop the constraint on Postgres (SQLite can't ALTER nullability
    # and enforces it loosely anyway).
    if engine.dialect.name == "postgresql":
        po = cols_info.get("playing_order")
        if po is not None and po.get("nullable") is False:
            statements.append("ALTER TABLE participants ALTER COLUMN playing_order DROP NOT NULL")

    # Course additions
    if "courses" in inspector.get_table_names():
        course_cols = {c["name"] for c in inspector.get_columns("courses")}
        if "livestream_url" not in course_cols:
            statements.append("ALTER TABLE courses ADD COLUMN livestream_url VARCHAR(500)")
        if "hole_yardages" not in course_cols:
            statements.append("ALTER TABLE courses ADD COLUMN hole_yardages JSON")
            # Backfill existing rows so API responses don't surface NULL.
            statements.append("UPDATE courses SET hole_yardages = '{}' WHERE hole_yardages IS NULL")
        else:
            statements.append("UPDATE courses SET hole_yardages = '{}' WHERE hole_yardages IS NULL")
        if "operator_password_hash" not in course_cols:
            statements.append("ALTER TABLE courses ADD COLUMN operator_password_hash VARCHAR(200)")
        # tee-sheet integration fields. Added to the model but historically
        # missing from this migrator, which 500'd every GET /courses on
        # prod (SELECT of a column that didn't exist). Backfill the
        # provider so CourseOut's required `tee_sheet_provider: str` never
        # sees NULL on pre-existing rows.
        if "tee_sheet_provider" not in course_cols:
            statements.append(
                "ALTER TABLE courses ADD COLUMN tee_sheet_provider VARCHAR(40) DEFAULT 'mock'"
            )
            statements.append(
                "UPDATE courses SET tee_sheet_provider = 'mock' WHERE tee_sheet_provider IS NULL"
            )
        else:
            statements.append(
                "UPDATE courses SET tee_sheet_provider = 'mock' WHERE tee_sheet_provider IS NULL"
            )
        if "tee_sheet_config" not in course_cols:
            statements.append("ALTER TABLE courses ADD COLUMN tee_sheet_config JSON")
            statements.append(
                "UPDATE courses SET tee_sheet_config = '{}' WHERE tee_sheet_config IS NULL"
            )
        else:
            statements.append(
                "UPDATE courses SET tee_sheet_config = '{}' WHERE tee_sheet_config IS NULL"
            )

    # LongVideoUpload additions — background-job UX fields.
    if "long_video_uploads" in inspector.get_table_names():
        lvu_cols = {c["name"] for c in inspector.get_columns("long_video_uploads")}
        if "processing_status" not in lvu_cols:
            statements.append(
                "ALTER TABLE long_video_uploads ADD COLUMN processing_status VARCHAR(20) DEFAULT 'completed'"
            )
            # Existing rows predate the background-job flow — backfill as
            # completed so the UI doesn't flag them as stuck.
            statements.append(
                "UPDATE long_video_uploads SET processing_status = 'completed' WHERE processing_status IS NULL"
            )
        if "processing_started_at" not in lvu_cols:
            statements.append(
                "ALTER TABLE long_video_uploads ADD COLUMN processing_started_at TIMESTAMP"
            )
        if "processing_completed_at" not in lvu_cols:
            statements.append(
                "ALTER TABLE long_video_uploads ADD COLUMN processing_completed_at TIMESTAMP"
            )
        if "last_error" not in lvu_cols:
            statements.append(
                "ALTER TABLE long_video_uploads ADD COLUMN last_error TEXT"
            )
        if "swing_count" not in lvu_cols:
            statements.append(
                "ALTER TABLE long_video_uploads ADD COLUMN swing_count VARCHAR(20) DEFAULT 'multiple'"
            )
            statements.append(
                "UPDATE long_video_uploads SET swing_count = 'multiple' WHERE swing_count IS NULL"
            )
        if "edit_metrics" not in lvu_cols:
            statements.append("ALTER TABLE long_video_uploads ADD COLUMN edit_metrics JSON")
        if "camera_event_id" not in lvu_cols:
            statements.append(
                "ALTER TABLE long_video_uploads ADD COLUMN camera_event_id INTEGER"
            )

    # VideoClip additions
    if "video_clips" in inspector.get_table_names():
        clip_cols = {c["name"] for c in inspector.get_columns("video_clips")}
        if "delivered_at" not in clip_cols:
            statements.append("ALTER TABLE video_clips ADD COLUMN delivered_at TIMESTAMP")
        if "distance_from_pin_feet" not in clip_cols:
            statements.append("ALTER TABLE video_clips ADD COLUMN distance_from_pin_feet INTEGER")
        if "tracer_url" not in clip_cols:
            statements.append("ALTER TABLE video_clips ADD COLUMN tracer_url TEXT")
        if "tee_clip_url" not in clip_cols:
            statements.append("ALTER TABLE video_clips ADD COLUMN tee_clip_url TEXT")
        if "is_highlight" not in clip_cols:
            # SQLite allows BOOLEAN; Postgres treats it as BOOLEAN natively.
            statements.append("ALTER TABLE video_clips ADD COLUMN is_highlight BOOLEAN DEFAULT FALSE")
            statements.append("UPDATE video_clips SET is_highlight = FALSE WHERE is_highlight IS NULL")
        if "highlight_tag" not in clip_cols:
            statements.append("ALTER TABLE video_clips ADD COLUMN highlight_tag VARCHAR(60)")
        if "long_upload_id" not in clip_cols:
            statements.append("ALTER TABLE video_clips ADD COLUMN long_upload_id INTEGER")
        if "tracer_diagnostics" not in clip_cols:
            statements.append("ALTER TABLE video_clips ADD COLUMN tracer_diagnostics JSON")

    # CameraEvent additions
    if "camera_events" in inspector.get_table_names():
        ce_cols = {c["name"] for c in inspector.get_columns("camera_events")}
        if "stop_signal_at" not in ce_cols:
            statements.append(
                "ALTER TABLE camera_events ADD COLUMN stop_signal_at TIMESTAMP"
            )
        if "tee_recording_started_at" not in ce_cols:
            statements.append(
                "ALTER TABLE camera_events ADD COLUMN tee_recording_started_at TIMESTAMP"
            )
        if "green_recording_started_at" not in ce_cols:
            statements.append(
                "ALTER TABLE camera_events ADD COLUMN green_recording_started_at TIMESTAMP"
            )

    # Camera additions — per-camera trigger kill-switch.
    if "cameras" in inspector.get_table_names():
        cam_cols = {c["name"] for c in inspector.get_columns("cameras")}
        if "triggering_enabled" not in cam_cols:
            statements.append(
                "ALTER TABLE cameras ADD COLUMN triggering_enabled BOOLEAN DEFAULT TRUE"
            )
            # Existing cameras default to triggering-on so behavior is
            # unchanged until the operator flips the switch.
            statements.append(
                "UPDATE cameras SET triggering_enabled = TRUE WHERE triggering_enabled IS NULL"
            )

    # Generic backstop: add ANY column defined on a model but missing from
    # its existing table. The explicit ALTERs above cover special cases
    # (value backfills, dropping a NOT NULL); this catches plain additions
    # so the hand-maintained list can't silently drift from the models and
    # 500 a SELECT — which is exactly what happened when courses gained
    # tee_sheet_provider / tee_sheet_config without a matching entry here.
    # Added nullable: we can't backfill a value generically, and every
    # column added post-hoc is either nullable or has a Python-side default.
    import re

    already = set()
    for _s in statements:
        _m = re.search(r"ADD COLUMN (\w+)", _s)
        _t = re.search(r"ALTER TABLE (\w+)", _s)
        if _m and _t:
            already.add((_t.group(1), _m.group(1)))
    existing_tables = set(inspector.get_table_names())
    # .tables.values() (not sorted_tables) — ordering is irrelevant for
    # independent ADD COLUMNs, and sorted_tables warns/raises on the
    # camera_events↔video_clips FK cycle.
    for table in Base.metadata.tables.values():
        if table.name not in existing_tables:
            continue  # create_all makes whole missing tables correctly
        have = {c["name"] for c in inspector.get_columns(table.name)}
        for col in table.columns:
            if col.name in have or col.primary_key:
                continue
            if (table.name, col.name) in already:
                continue
            try:
                coltype = col.type.compile(dialect=engine.dialect)
            except Exception:
                continue
            statements.append(
                f"ALTER TABLE {table.name} ADD COLUMN {col.name} {coltype}"
            )

    if not statements:
        return
    # Run each statement in its own transaction, best-effort, so one bad
    # ALTER (e.g. a type the dialect won't add in place) can't roll back
    # every other fix and leave the schema stuck.
    mlog = logging.getLogger("golfreelz.migrate")
    ran = 0
    for stmt in statements:
        try:
            with engine.begin() as conn:
                conn.execute(text(stmt))
            ran += 1
        except Exception as exc:  # noqa: BLE001
            mlog.warning("migrate: skipped failing statement [%s]: %s", stmt, exc)

    # Drop every pooled connection after DDL. Postgres caches the result-type
    # plan of a prepared statement per connection; a SELECT compiled before
    # these ALTER TABLE ... ADD COLUMN runs will then fail with "cached plan
    # must not change result type" on any connection the migration reused —
    # which 500'd GET /courses right after the very deploy that added the
    # columns, even though the schema and data were correct. Disposing forces
    # fresh connections (and fresh plans) for all real request traffic.
    if ran:
        engine.dispose()
        mlog.info("migrate: applied %d statement(s); connection pool reset", ran)


def _reap_orphaned_jobs() -> None:
    """A backend restart kills any in-flight produce thread, but the
    LongVideoUpload row stays stuck at 'processing'/'pending' (the dead thread
    never flipped it) — so the UI shows "Production in Progress" forever and
    the Produce endpoint refuses to re-run it. On startup nothing is actually
    running, so mark any such row failed to clear the state."""
    from datetime import datetime

    from .database import SessionLocal
    from .models import LongVideoUpload

    db = SessionLocal()
    try:
        stuck = (
            db.query(LongVideoUpload)
            .filter(LongVideoUpload.processing_status.in_(["processing", "pending"]))
            .all()
        )
        for row in stuck:
            row.processing_status = "failed"
            row.processing_completed_at = datetime.utcnow()
            row.last_error = "interrupted (backend restarted)"
        if stuck:
            db.commit()
            _glog.info("startup: reaped %d orphaned in-progress upload(s)", len(stuck))
    except Exception as exc:  # noqa: BLE001
        _glog.warning("startup: reap orphaned jobs failed: %s", exc)
        db.rollback()
    finally:
        db.close()


def _heal_media_urls() -> None:
    """Rewrite stored /uploads/clips media URLs whose origin doesn't match
    this instance's app_base_url. Production historically stamped clips with
    the DEV workspace domain (APP_BASE_URL leak — see the settings
    validator), so thumbnails/videos only loaded while the dev workspace
    was awake. The files themselves are fine (same names, shared bucket);
    only the stored origins are wrong, and this makes them self-heal on
    every boot."""
    import json as _json
    import re as _re

    from .config import settings
    from .database import SessionLocal
    from .models import LongVideoUpload, VideoClip

    base = (settings.app_base_url or "").rstrip("/")
    if not base.startswith("https://"):
        return  # local/dev-sandbox runs: don't rewrite anything
    pat = _re.compile(r"https?://[^/\s\"']+/uploads/clips/")
    want = base + "/uploads/clips/"

    def _fix(text: str | None) -> str | None:
        if not text or "/uploads/clips/" not in text:
            return None
        healed = pat.sub(want, text)
        return healed if healed != text else None

    db = SessionLocal()
    try:
        n = 0
        for c in db.query(VideoClip).all():
            for col in ("source_url", "thumbnail_url", "tracer_url", "tee_clip_url"):
                healed = _fix(getattr(c, col))
                if healed is not None:
                    setattr(c, col, healed)
                    n += 1
        for u in db.query(LongVideoUpload).filter(
            LongVideoUpload.edit_metrics.isnot(None)
        ):
            healed = _fix(_json.dumps(u.edit_metrics))
            if healed is not None:
                u.edit_metrics = _json.loads(healed)
                n += 1
        if n:
            db.commit()
            _glog.info("startup: healed %d media URL(s) to %s", n, base)
    except Exception as exc:  # noqa: BLE001
        _glog.warning("startup: media URL heal failed: %s", exc)
        db.rollback()
    finally:
        db.close()


@app.on_event("startup")
def _startup() -> None:
    Base.metadata.create_all(bind=engine)
    _migrate()
    _heal_media_urls()
    _reap_orphaned_jobs()
    _remove_retired_courses()
    _seed_default_courses()
    _seed_showcase_slots()
    # Produce tee-only when a paired event's green half never arrives, so a
    # green cellular dropout doesn't cost the whole shot.
    cameras.start_tee_only_fallback_sweeper()


# Demo courses previously seeded into the DB that we no longer want shown
# anywhere. Safe to leave indefinitely — the deletion silently skips any
# course that still has dependent rows (bookings, cameras, etc).
_RETIRED_COURSE_NAMES = ("Kiawah Island",)


def _remove_retired_courses() -> None:
    from .database import SessionLocal
    from .models import Course

    # Leaves first, course last. Each statement is a no-op if there's
    # nothing to delete, so the whole block is idempotent.
    cascade_sql = [
        "DELETE FROM broadcast_views WHERE course_id = :c "
        "OR clip_id IN (SELECT id FROM video_clips WHERE course_id = :c)",
        "DELETE FROM camera_events WHERE course_id = :c",
        "DELETE FROM hole_in_one_events WHERE "
        "participant_id IN (SELECT p.id FROM participants p "
        "JOIN tee_times t ON p.tee_time_id = t.id WHERE t.course_id = :c) "
        "OR tee_clip_id  IN (SELECT id FROM video_clips WHERE course_id = :c) "
        "OR wide_clip_id IN (SELECT id FROM video_clips WHERE course_id = :c) "
        "OR hole_clip_id IN (SELECT id FROM video_clips WHERE course_id = :c)",
        "DELETE FROM video_clips WHERE course_id = :c",
        "DELETE FROM cameras WHERE course_id = :c",
        "DELETE FROM long_video_uploads WHERE course_id = :c",
        "DELETE FROM participants WHERE tee_time_id IN "
        "(SELECT id FROM tee_times WHERE course_id = :c)",
        "DELETE FROM tee_times WHERE course_id = :c",
        "DELETE FROM courses WHERE id = :c",
    ]

    db = SessionLocal()
    try:
        for name in _RETIRED_COURSE_NAMES:
            row = db.query(Course).filter(Course.name == name).first()
            if not row:
                continue
            cid = row.id
            try:
                for stmt in cascade_sql:
                    db.execute(text(stmt), {"c": cid})
                db.commit()
                _glog.info("removed retired course %r (id=%s) and dependents", name, cid)
            except Exception as exc:
                db.rollback()
                _glog.warning("failed to remove retired course %r: %s", name, exc)
    finally:
        db.close()


def _seed_showcase_slots() -> None:
    """Ensure positions 1/2/3 exist in the showcase table (empty by default)."""
    from .database import SessionLocal
    from .models import Showcase

    db = SessionLocal()
    try:
        existing = {s.position for s in db.query(Showcase).all()}
        for pos in (1, 2, 3):
            if pos not in existing:
                db.add(Showcase(position=pos))
        db.commit()
    finally:
        db.close()


def _seed_default_courses() -> None:
    """Ensure the demo course set exists. Adds courses by name if missing;
    never overwrites an existing course's data."""
    from .database import SessionLocal
    from .models import Course

    defaults = [
        {
            "name": "Maridoe Golf Club",
            "location": "Carrollton, TX",
            "par3_holes": [3, 8, 11, 14],
            "hole_yardages": {"3": 173, "8": 165, "11": 205, "14": 192},
            "minutes_per_hole": 14,
        },
        {
            "name": "Pebble Beach",
            "location": "Pebble Beach, CA",
            "par3_holes": [5, 7, 12, 17],
            "hole_yardages": {"5": 195, "7": 106, "12": 202, "17": 178},
            "minutes_per_hole": 14,
        },
    ]
    db = SessionLocal()
    try:
        for d in defaults:
            existing = db.query(Course).filter(Course.name == d["name"]).first()
            if existing:
                continue
            db.add(Course(**d))
        db.commit()
    finally:
        db.close()


@app.get("/health")
def health() -> dict:
    return {"ok": True, "app": "golfreelz"}


app.include_router(public.router)
app.include_router(gallery.router)
app.include_router(webhooks.router)
app.include_router(admin.router)
app.include_router(auth.router)
app.include_router(operator.router)
app.include_router(broadcast.router)
app.include_router(cameras.router)


# --- Uploads (selfies) -------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_ROOT = BACKEND_ROOT / settings.upload_dir
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
CLIPS_DIR = UPLOAD_ROOT / "clips"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)


# Serve clips through a route (registered BEFORE the /uploads mount so it
# wins for this path) that rehydrates from object storage when the local
# file is gone — e.g. after a redeploy wiped the ephemeral disk. FileResponse
# honours HTTP Range requests, which Safari needs for <video> playback.
@app.get("/uploads/clips/{name}", include_in_schema=False)
def serve_clip(name: str):
    safe = Path(name).name  # filenames only — no path traversal
    local = CLIPS_DIR / safe
    if not local.exists():
        storage.ensure_local(CLIPS_DIR, safe)
    if not local.exists():
        raise StarletteHTTPException(status_code=404)
    return FileResponse(local)


app.mount("/uploads", StaticFiles(directory=UPLOAD_ROOT), name="uploads")

# Background mirror: push new clips into object storage so a redeploy can't
# erase footage. No-op when object storage isn't configured (local dev).
storage.start_sweeper(CLIPS_DIR)


# --- Static SPA hosting (Replit / single-port deploy) ------------------------
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

if os.environ.get("SERVE_FRONTEND") == "1" and FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        if full_path.startswith(("api/", "uploads/", "docs", "openapi.json", "redoc", "health")):
            raise StarletteHTTPException(status_code=404)
        candidate = FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
