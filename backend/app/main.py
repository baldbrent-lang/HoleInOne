from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import inspect, text
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import settings
from .database import Base, engine
from .routers import admin, gallery, public, webhooks

app = FastAPI(title="Par One API", version="0.1.0")

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
    # Old schema had playing_order NOT NULL. The matcher no longer uses it,
    # so drop the constraint on Postgres (SQLite can't ALTER nullability
    # and enforces it loosely anyway).
    if engine.dialect.name == "postgresql":
        po = cols_info.get("playing_order")
        if po is not None and po.get("nullable") is False:
            statements.append("ALTER TABLE participants ALTER COLUMN playing_order DROP NOT NULL")
    if not statements:
        return
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


@app.on_event("startup")
def _startup() -> None:
    Base.metadata.create_all(bind=engine)
    _migrate()


@app.get("/health")
def health() -> dict:
    return {"ok": True, "app": "par-one"}


app.include_router(public.router)
app.include_router(gallery.router)
app.include_router(webhooks.router)
app.include_router(admin.router)


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
