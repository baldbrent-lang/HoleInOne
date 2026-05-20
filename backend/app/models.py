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
    operator_password_hash: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    tee_times: Mapped[list[TeeTime]] = relationship(back_populates="course", cascade="all, delete-orphan")

    @property
    def operator_password_set(self) -> bool:
        return bool(self.operator_password_hash)


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
    refunded_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
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
    # Set when this clip was cut from a LongVideoUpload by the production
    # pipeline. Lets the /admin/production page surface the produced
    # output next to the raw tee/green source thumbnails. NULL for
    # clips uploaded individually via /clips/upload.
    long_upload_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("long_video_uploads.id"), nullable=True, index=True,
    )
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
    # Rendered-with-tracer version of source_url. Populated by the tracer job
    # once the spike pipeline ships; until then the broadcast falls back to
    # source_url so the channel works without the tracer.
    tracer_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Raw tee-side cut without overlays or composite concatenation. Set when
    # the dual-cam composite pipeline runs so the AI-analysis page can target
    # a single-camera player-visible clip even though source_url points at the
    # composite. Null for single-cam clips (source_url is already the raw cut).
    tee_clip_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Marks a clip as eligible to play during dead time on the broadcast
    # channel ("highlights"). Auto-set for aces and CTP winners; operators
    # can toggle via the admin UI.
    is_highlight: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    highlight_tag: Mapped[Optional[str]] = mapped_column(String(60), nullable=True)
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


class LongVideoUpload(Base):
    """A persisted long video (or pair of synced long videos) so the
    operator can re-edit / re-process the same source on /admin/long-
    upload without re-uploading. Filenames are stored relative to the
    uploads/clips directory; the actual source files are kept on disk
    indefinitely until the operator deletes them.

    processing_status drives the background-job UX: the upload endpoint
    creates the row + saves the source file(s), then immediately
    returns with status='processing'. A worker thread runs the cut /
    splice / AI tracer pipeline and flips this to 'completed' (or
    'failed' with last_error) when done."""

    __tablename__ = "long_video_uploads"

    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id"))
    camera_type: Mapped[str] = mapped_column(String(40), default=CameraType.tee.value)
    base_captured_at: Mapped[datetime] = mapped_column(DateTime)
    tee_filename: Mapped[str] = mapped_column(String(200))
    green_filename: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    tee_original_filename: Mapped[Optional[str]] = mapped_column(String(400), nullable=True)
    green_original_filename: Mapped[Optional[str]] = mapped_column(String(400), nullable=True)
    last_n_segments: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    last_n_succeeded: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    processing_status: Mapped[str] = mapped_column(String(20), default="pending")  # pending|processing|completed|failed
    processing_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    processing_completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # 'single' = one swing, queue for manual editing before producing.
    # 'multiple' = a full round / many swings; auto-process on upload.
    # Drives the Production-page UI state and the quick-upload's
    # decision to spawn the background job immediately.
    swing_count: Mapped[str] = mapped_column(String(20), default="multiple")
    # Persisted state of the single-swing Edit wizard. Holds whatever
    # the operator has Saved so far — handedness, address/impact frame
    # indices, ball-at-rest pixel coords, detection-area ROI box,
    # target (flag) point, the rendered tracer URL, and the per-frame
    # ball-track positions. None until the wizard's first save; once
    # set, the wizard re-opens with these values instead of re-running
    # auto-detect.
    edit_metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # When this row was created by the Pi camera-event pipeline (instead
    # of an admin upload), points at the source CameraEvent. Drives the
    # "From Camera #N · hole X" badge on the production card and lets us
    # avoid double-listing the same physical capture.
    camera_event_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("camera_events.id"), nullable=True, index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class Camera(Base):
    """An on-course capture device — typically a Raspberry Pi paired
    with a USB / CSI camera, mounted on or near one hole. Tee cameras
    watch the tee box for a person and trigger; green cameras
    continuously buffer and persist their buffer when the paired tee
    camera fires.

    Authentication is via a per-camera token (sent in the URL of every
    event endpoint), not the admin password — that way devices in the
    field don't need the operator password baked into their SD cards.
    """

    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id"), index=True)
    assigned_hole: Mapped[int] = mapped_column(Integer)
    assigned_role: Mapped[str] = mapped_column(String(20))  # 'tee' | 'green'
    # Paired camera (tee's pair = green, and vice versa). Nullable so
    # a half-deployed hole still works; backend skips the dual-cam
    # pair-up step when this is null.
    paired_with_camera_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("cameras.id"), nullable=True,
    )
    # Per-camera secret used to auth /api/cameras/{token}/... requests.
    # Long random URL-safe string; rotate by editing the row.
    auth_token: Mapped[str] = mapped_column(
        String(80), unique=True,
        default=lambda: _token("cam_", 24),
    )
    # Display name the operator sets ("Hole 3 tee", "Hole 7 green").
    name: Mapped[str] = mapped_column(String(120), default="")
    # JSON-encoded {x, y, w, h} in the camera's native pixel coords —
    # the bounding box the tee-side person detector treats as "on the
    # tee". Green cameras leave this null.
    tee_box_roi: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # Last time the camera POSTed anything (heartbeat, event, upload).
    # Backend's offline alert watches this.
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    firmware_version: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class CameraEvent(Base):
    """One detected event from a tee camera, with optional uploaded
    clips from itself and its paired green camera. The row tracks
    which Pis have called in and gives the backend something to
    attach the produced VideoClip to.

    Lifecycle:
      1. Tee Pi POSTs /event-trigger → backend creates row,
         status='triggered'. Both Pis start recording.
      2. Tee Pi keeps recording until the tee box has been empty for
         no_person_timeout_seconds, then POSTs /event-stop → backend
         sets stop_signal_at=now. Green Pi (which has been polling
         /event-status) sees the flag and stops recording too.
      3. Tee Pi POSTs /upload-event → tee_clip_filename set,
         status='tee_uploaded'.
      4. Green Pi POSTs /upload-event → green_clip_filename set,
         status='paired_uploaded'.
      5. Worker picks up paired rows, detects swings inside the raw
         clip, runs the per-segment pipeline, flips to 'processed'
         with produced_clip_id set to the first emitted clip.

    An event now represents a group session (potentially many swings),
    not a single swing. produced_clip_id points at the first clip
    produced by the multi-swing detector; sibling clips are linked by
    long_upload_id on the VideoClip rows.
    """

    __tablename__ = "camera_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Operator-supplied session_id from the tee Pi (UUID4). Matches
    # the tee's clip with the green's clip when they upload.
    session_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    tee_camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"), index=True)
    green_camera_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("cameras.id"), nullable=True,
    )
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id"), index=True)
    hole_number: Mapped[int] = mapped_column(Integer)
    triggered_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    tee_clip_filename: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    green_clip_filename: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    # Set by the tee Pi's /event-stop call once the tee box has been
    # empty long enough. The green Pi polls /event-status looking for
    # this so both clips end at roughly the same wall-clock moment.
    stop_signal_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # triggered | tee_uploaded | paired_uploaded | processed | failed
    status: Mapped[str] = mapped_column(String(30), default="triggered")
    # VideoClip row produced from the pair, once the pipeline runs.
    produced_clip_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("video_clips.id"), nullable=True,
    )
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class BroadcastView(Base):
    """Per-viewer dedup for the /broadcast/next playlist.

    The viewer mints a UUID client-side and sends it as `viewer_id`. We log
    one row per (viewer, clip) so the same person doesn't see the same swing
    twice in quick succession. Old rows can be pruned after ~24h."""

    __tablename__ = "broadcast_views"

    id: Mapped[int] = mapped_column(primary_key=True)
    viewer_id: Mapped[str] = mapped_column(String(64), index=True)
    clip_id: Mapped[int] = mapped_column(ForeignKey("video_clips.id"))
    course_id: Mapped[Optional[int]] = mapped_column(ForeignKey("courses.id"), nullable=True)
    shown_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


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
