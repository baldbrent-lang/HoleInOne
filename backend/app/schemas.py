from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

class CourseCreate(BaseModel):
    name: str
    location: str = ""
    par3_holes: list[int] = Field(default_factory=list)
    hole_yardages: dict[str, int] = Field(default_factory=dict)
    minutes_per_hole: int = 14
    tee_sheet_provider: str = "mock"
    tee_sheet_config: dict = Field(default_factory=dict)
    livestream_url: Optional[str] = None


class CourseUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    par3_holes: Optional[list[int]] = None
    hole_yardages: Optional[dict[str, int]] = None
    minutes_per_hole: Optional[int] = None
    livestream_url: Optional[str] = None


class CourseOut(BaseModel):
    id: int
    name: str
    location: str
    par3_holes: list[int]
    hole_yardages: dict[str, int] = Field(default_factory=dict)
    minutes_per_hole: int
    qr_token: str
    tee_sheet_provider: str
    livestream_url: Optional[str] = None
    operator_password_set: bool = False

    @field_validator("hole_yardages", mode="before")
    @classmethod
    def _yardages_none_to_empty(cls, v):
        return v or {}

    @field_validator("par3_holes", mode="before")
    @classmethod
    def _holes_none_to_empty(cls, v):
        return v or []

    class Config:
        from_attributes = True


class PublicCourseOut(BaseModel):
    id: int
    name: str
    location: str
    qr_token: str
    livestream_url: Optional[str] = None
    hole_yardages: dict[str, int] = Field(default_factory=dict)

    @field_validator("hole_yardages", mode="before")
    @classmethod
    def _none_to_empty(cls, v):
        return v or {}

    class Config:
        from_attributes = True


class TeeTimeOut(BaseModel):
    id: int
    starts_at: datetime
    max_players: int
    spots_taken: int

    class Config:
        from_attributes = True


class RegistrationCreate(BaseModel):
    """Kept for OpenAPI documentation; the live endpoint uses multipart/form-data."""

    course_token: str
    tee_time_id: int
    name: str
    mobile: Optional[str] = None
    email: Optional[EmailStr] = None
    group_size: int = Field(ge=1, le=4)

    @field_validator("mobile", "email")
    @classmethod
    def _strip(cls, v):
        if isinstance(v, str):
            v = v.strip() or None
        return v


class RegistrationResult(BaseModel):
    participant_id: int
    gallery_url: str
    client_secret: Optional[str] = None
    paid: bool


class ParticipantOut(BaseModel):
    id: int
    name: str
    playing_order: Optional[int] = None
    paid: bool
    gallery_token: str

    class Config:
        from_attributes = True


class ClipOut(BaseModel):
    id: int
    hole_number: int
    camera_type: str
    captured_at: datetime
    source_url: str
    thumbnail_url: Optional[str] = None
    # Carry, apex and ball speed are no longer surfaced: we cannot measure
    # them reliably, so publishing them was publishing a guess. The
    # columns stay on the model, and the tracer still fills them, so
    # nothing in the pipeline changes and this can come back.
    # distance_from_pin_feet stays -- it is what Closest to the Pin IS.
    distance_from_pin_feet: Optional[int] = None
    processing_status: str
    ball_in_cup: bool

    class Config:
        from_attributes = True


class GalleryOut(BaseModel):
    participant: ParticipantOut
    course_name: str
    livestream_url: Optional[str] = None
    hole_yardages: dict[str, int] = Field(default_factory=dict)
    clips: list[ClipOut]


class IncomingClip(BaseModel):
    course_id: int
    hole_number: int
    camera_type: str
    captured_at: datetime
    source_url: str
    thumbnail_url: Optional[str] = None
    # Still ACCEPTED so an existing sender does not start failing
    # validation; simply not shown anywhere.
    carry_yards: Optional[int] = None
    apex_feet: Optional[int] = None
    ball_speed_mph: Optional[int] = None
    ball_in_cup: bool = False


class HIOReviewAction(BaseModel):
    action: str  # approve|reject|needs_more
    reviewer: str
    note: Optional[str] = None


class HIOEventOut(BaseModel):
    id: int
    participant_id: int
    hole_number: int
    status: str
    tee_clip_id: Optional[int]
    wide_clip_id: Optional[int]
    hole_clip_id: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True


class IssueFlag(BaseModel):
    note: str
