from __future__ import annotations

import secrets
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _token(prefix: str = "", nbytes: int = 16) -> str:
    return f"{prefix}{secrets.token_urlsafe(nbytes)}"


class ClipProcessingStatus(str, Enum):
    received = "received"
    processed = "processed"
    assigned = "assigned"
    unassigned = "unassigned"
    flagged = "flagged"


class CameraType(str, Enum):
    tee = "tee"
    wide_green = "wide_green"
    hole = "hole"


class HIOStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    needs_more = "needs_more"


def utcnow() -> datetime:
    return datetime.utcnow()


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Course(Base):
    __tablename__ = "courses"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    location: Mapped[str] = mapped_column(String(200), default="")
    par3_holes: Mapped[list] = mapped_column(JSON, default=list)  # e.g. [3, 7, 12, 16]
    hole_yardages: Mapped[dict] = mapped_column(JSON, default=dict)  # {"3": 173, "7": 165}
    minutes_per_hole: Mapped[int] = mapped_column(Integer, default=14)
    qr_token: Mapped[str] = mapped_column(String(64), unique=True, default=lambda: _token("c_"))
    tee_sheet_provider: Mapped[str] = mapped_column(String(40), default="mock")  # foreup|lightspeed|mock
    tee_sheet_config: Mapped[dict] = mapped_column(JSON, default=dict)
    livestream_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    tee_times: Mapped[list[TeeTime]] = relationship(back_populates="course", cascade="all, delete-orphan")


class TeeTime(Base):
    __tablename__ = "tee_times"

    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id"))
    starts_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    max_players: Mapped[int] = mapped_column(Integer, default=4)
    external_id: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)

    course: Mapped[Course] = relationship(back_populates="tee_times")
    participants: Mapped[list[Participant]] = relationship(back_populates="tee_time", cascade="all, delete-orphan")


class Participant(Base):
    __tablename__ = "participants"

    id: Mapped[int] = mapped_column(primary_key=True)
    tee_time_id: Mapped[int] = mapped_column(ForeignKey("tee_times.id"))
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    mobile: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    # Legacy field. We now match by appearance, not declared hitting order.
    playing_order: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    group_size: Mapped[int] = mapped_column(Integer, default=4)
    # Path under backend/uploads/ for the registration selfie.
    selfie_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    # Appearance embedding (CLIP/ReID-style). Stub mode stores a hash-derived
    # vector; real mode populated by services/appearance.py.
    appearance_embedding: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    stripe_payment_intent_id: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    paid: Mapped[bool] = mapped_column(Boolean, default=False)
    gallery_token: Mapped[str] = mapped_column(String(64), unique=True, default=lambda: _token("g_", 20))
    gallery_ready_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    summary_sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    tee_time: Mapped[TeeTime] = relationship(back_populates="participants")
    clips: Mapped[list[VideoClip]] = relationship(back_populates="participant")


class VideoClip(Base):
    __tablename__ = "video_clips"

    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id"))
    participant_id: Mapped[Optional[int]] = mapped_column(ForeignKey("participants.id"), nullable=True)
    hole_number: Mapped[int] = mapped_column(Integer)
    camera_type: Mapped[str] = mapped_column(String(20), default=CameraType.tee.value)
    captured_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    source_url: Mapped[str] = mapped_column(Text)  # Shot Tracer processed clip URL
    thumbnail_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    carry_yards: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    apex_feet: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ball_speed_mph: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    distance_from_pin_feet: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    processing_status: Mapped[str] = mapped_column(String(20), default=ClipProcessingStatus.received.value)
    ball_in_cup: Mapped[bool] = mapped_column(Boolean, default=False)
    issue_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    delivered_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    participant: Mapped[Optional[Participant]] = relationship(back_populates="clips")


class HoleInOneEvent(Base):
    __tablename__ = "hole_in_one_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id"))
    hole_number: Mapped[int] = mapped_column(Integer)
    tee_clip_id: Mapped[Optional[int]] = mapped_column(ForeignKey("video_clips.id"), nullable=True)
    wide_clip_id: Mapped[Optional[int]] = mapped_column(ForeignKey("video_clips.id"), nullable=True)
    hole_clip_id: Mapped[Optional[int]] = mapped_column(ForeignKey("video_clips.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default=HIOStatus.pending.value)
    reviewer: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    decision_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    actor: Mapped[str] = mapped_column(String(200))
    action: Mapped[str] = mapped_column(String(200))
    target: Mapped[str] = mapped_column(String(200))
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Showcase(Base):
    """Featured clips on the public Home page. Three slots (positions 1-3).
    Admin uploads or pastes a URL into each slot; appears in 'Our videos in
    action' on the marketing page."""
    __tablename__ = "showcase"

    id: Mapped[int] = mapped_column(primary_key=True)
    position: Mapped[int] = mapped_column(Integer, unique=True)  # 1, 2, 3
    source_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    thumbnail_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    caption: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
