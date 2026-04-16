from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .database import Base, engine
from .routers import admin, gallery, public, webhooks

app = FastAPI(title="Par One API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "app": "par-one"}


app.include_router(public.router)
app.include_router(gallery.router)
app.include_router(webhooks.router)
app.include_router(admin.router)


# --- Static SPA hosting (Replit / single-port deploy) ------------------------
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

if os.environ.get("SERVE_FRONTEND") == "1" and FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        # Do not hijack API / docs routes.
        if full_path.startswith(("api/", "docs", "openapi.json", "redoc", "health")):
            raise StarletteHTTPException(status_code=404)
        candidate = FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
