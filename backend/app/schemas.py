from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class CourseCreate(BaseModel):
    name: str
    location: str = ""
    par3_holes: list[int] = Field(default_factory=list)
    minutes_per_hole: int = 14
    tee_sheet_provider: str = "mock"
    tee_sheet_config: dict = Field(default_factory=dict)


class CourseOut(BaseModel):
    id: int
    name: str
    location: str
    par3_holes: list[int]
    minutes_per_hole: int
    qr_token: str
    tee_sheet_provider: str

    class Config:
        from_attributes = True


class PublicCourseOut(BaseModel):
    id: int
    name: str
    location: str

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
    playing_order: int
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
    carry_yards: Optional[int] = None
    apex_feet: Optional[int] = None
    ball_speed_mph: Optional[int] = None
    processing_status: str
    ball_in_cup: bool

    class Config:
        from_attributes = True


class GalleryOut(BaseModel):
    participant: ParticipantOut
    course_name: str
    clips: list[ClipOut]


class IncomingClip(BaseModel):
    course_id: int
    hole_number: int
    camera_type: str
    captured_at: datetime
    source_url: str
    thumbnail_url: Optional[str] = None
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
