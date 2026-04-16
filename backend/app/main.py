from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
