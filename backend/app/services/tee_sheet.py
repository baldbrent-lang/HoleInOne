"""Tee sheet integrations. V0 ships a 'mock' provider + a stub for real providers.

ForeUP / Lightspeed real APIs are out of scope for V0; the mock provider
returns deterministic upcoming tee times so the flow is demoable.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from ..models import Course, TeeTime


def _next_interval(base: datetime, minutes: int) -> datetime:
    return base + timedelta(minutes=minutes)


def _ensure_mock_tee_times_for_date(db: Session, course: Course, on_date: datetime) -> list[TeeTime]:
    day_start = on_date.replace(hour=7, minute=0, second=0, microsecond=0)
    day_end = on_date.replace(hour=17, minute=0, second=0, microsecond=0)
    existing = (
        db.execute(
            select(TeeTime).where(
                and_(
                    TeeTime.course_id == course.id,
                    TeeTime.starts_at >= day_start,
                    TeeTime.starts_at < day_end,
                )
            )
        )
        .scalars()
        .all()
    )
    if existing:
        return existing

    times: list[TeeTime] = []
    t = day_start
    while t < day_end:
        tt = TeeTime(
            course_id=course.id,
            starts_at=t,
            max_players=4,
            external_id=f"mock-{course.id}-{t.isoformat()}",
        )
        db.add(tt)
        times.append(tt)
        t = _next_interval(t, 10)
    db.flush()
    return times


def list_available_tee_times(db: Session, course: Course, on_date: datetime | None = None) -> list[TeeTime]:
    on_date = on_date or datetime.now(timezone.utc).replace(tzinfo=None)
    if course.tee_sheet_provider == "mock":
        return _ensure_mock_tee_times_for_date(db, course, on_date)

    # Real providers intentionally unimplemented in V0 -- fall through to
    # whatever TeeTime rows already exist for the day (e.g. manually seeded).
    day_start = on_date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)
    return (
        db.execute(
            select(TeeTime).where(
                and_(
                    TeeTime.course_id == course.id,
                    TeeTime.starts_at >= day_start,
                    TeeTime.starts_at < day_end,
                )
            )
        )
        .scalars()
        .all()
    )
