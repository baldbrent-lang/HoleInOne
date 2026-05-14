from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import inspect, text
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import settings
from .database import Base, engine
from .routers import admin, auth, broadcast, cameras, gallery, operator, public, webhooks

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

    if not statements:
        return
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


@app.on_event("startup")
def _startup() -> None:
    Base.metadata.create_all(bind=engine)
    _migrate()
    _seed_default_courses()
    _seed_showcase_slots()


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
        {
            "name": "Kiawah Island",
            "location": "Kiawah Island, SC",
            "par3_holes": [5, 8, 14, 17],
            "hole_yardages": {"5": 207, "8": 197, "14": 194, "17": 221},
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
app.mount("/uploads", StaticFiles(directory=UPLOAD_ROOT), name="uploads")


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
