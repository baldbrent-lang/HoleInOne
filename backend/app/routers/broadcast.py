"""Near-live "broadcast" channel.

Replaces continuous livestreaming with a server-curated playlist of
already-produced swing clips, intercut with highlight reels and leaderboard
slates during dead time. Costs ~10x less in cellular data than real live
streaming because we only move bytes during actual swings — the rest of the
viewing time reuses footage already on the server.

The frontend `/watch` page polls GET /broadcast/next?viewer_id=<uuid> and
plays whatever it gets back. The server picks:

  1. The freshest tracer'd swing from any par-3 in the last `FRESH_WINDOW`,
     that this viewer hasn't already seen.
  2. Otherwise, a highlight clip (`is_highlight=True`) the viewer hasn't
     seen recently, prioritized by tag (aces > stiffs > ctp winners > rest).
  3. Otherwise, a `kind=slate` response telling the client to render a
     full-screen leaderboard card for a few seconds before asking again.

`BroadcastView` rows are written each time a clip is served so the same
swing doesn't loop back to the same viewer; rows older than 24h are pruned.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import require_admin
from ..models import (
    BroadcastView,
    Course,
    HIOStatus,
    HoleInOneEvent,
    Participant,
    VideoClip,
)

router = APIRouter(prefix="/api/broadcast", tags=["broadcast"])

# How recent counts as "near-live" — anything newer wins over highlights.
FRESH_WINDOW = timedelta(minutes=15)
# After this, a viewer can see the same highlight again.
VIEW_DEDUP_WINDOW = timedelta(hours=6)
# Pruning threshold.
VIEW_TTL = timedelta(hours=24)


def _viewer_seen_ids(db: Session, viewer_id: str) -> set[int]:
    """Clip IDs this viewer has been served inside VIEW_DEDUP_WINDOW."""
    cutoff = datetime.utcnow() - VIEW_DEDUP_WINDOW
    rows = (
        db.query(BroadcastView.clip_id)
        .filter(BroadcastView.viewer_id == viewer_id, BroadcastView.shown_at >= cutoff)
        .all()
    )
    return {r[0] for r in rows}


def _prune_old_views(db: Session) -> None:
    """Best-effort cleanup; cheap because of the index on shown_at."""
    cutoff = datetime.utcnow() - VIEW_TTL
    db.query(BroadcastView).filter(BroadcastView.shown_at < cutoff).delete(synchronize_session=False)


def _clip_payload(clip: VideoClip, db: Session, course: Optional[Course]) -> dict:
    """Shape sent to the player. Includes overlay metadata so the frontend
    can render hole/golfer chrome without an extra round-trip."""
    participant = db.get(Participant, clip.participant_id) if clip.participant_id else None
    return {
        "kind": "swing",
        "clip_id": clip.id,
        "video_url": clip.tracer_url or clip.source_url,
        "tracer": bool(clip.tracer_url),
        "thumbnail_url": clip.thumbnail_url,
        "course_name": course.name if course else "",
        "hole_number": clip.hole_number,
        "yardage": (course.hole_yardages or {}).get(str(clip.hole_number)) if course else None,
        "camera_type": clip.camera_type,
        "golfer_name": participant.name if participant else None,
        "captured_at": clip.captured_at.isoformat() if clip.captured_at else None,
        "ball_in_cup": bool(clip.ball_in_cup),
        "distance_from_pin_feet": clip.distance_from_pin_feet,
        "is_highlight": bool(clip.is_highlight),
        "highlight_tag": clip.highlight_tag,
    }


@router.get("/next")
def broadcast_next(
    viewer_id: str = Query(..., min_length=8, max_length=64),
    course_id: Optional[int] = Query(default=None),
    db: Session = Depends(get_db),
):
    """Pick the next thing for this viewer to watch.

    `course_id` is optional — if provided, restrict the channel to one
    course (useful for clubhouse-TV kiosk mode). Omit it for the
    multi-course national feed.
    """
    _prune_old_views(db)
    seen = _viewer_seen_ids(db, viewer_id)

    # Pull the course up front for overlays.
    course = db.get(Course, course_id) if course_id else None

    # 1. Freshest unseen swing on any par-3 the configured courses care about.
    fresh_cutoff = datetime.utcnow() - FRESH_WINDOW
    fresh_q = (
        db.query(VideoClip)
        .filter(VideoClip.captured_at >= fresh_cutoff)
        .filter(VideoClip.participant_id.isnot(None))
        .order_by(desc(VideoClip.captured_at))
    )
    if course_id:
        fresh_q = fresh_q.filter(VideoClip.course_id == course_id)
    if seen:
        fresh_q = fresh_q.filter(~VideoClip.id.in_(seen))
    fresh = fresh_q.first()
    if fresh:
        course_for_clip = course or db.get(Course, fresh.course_id)
        _record_view(db, viewer_id, fresh)
        return _clip_payload(fresh, db, course_for_clip)

    # 2. Highlight reel — operator-tagged or auto-tagged.
    hi_q = db.query(VideoClip).filter(VideoClip.is_highlight.is_(True))
    if course_id:
        hi_q = hi_q.filter(VideoClip.course_id == course_id)
    if seen:
        hi_q = hi_q.filter(~VideoClip.id.in_(seen))
    # Crude priority: aces first (`01-ace`), then stiffs (`02-stiff`), then
    # whatever operators tagged manually. Untagged highlights sort last by
    # treating NULL as the highest string. Within a tier, newer wins.
    highlight = (
        hi_q.order_by(
            VideoClip.highlight_tag.is_(None),  # False (tagged) before True (untagged)
            VideoClip.highlight_tag.asc(),
            desc(VideoClip.created_at),
        ).first()
    )
    if highlight:
        course_for_clip = course or db.get(Course, highlight.course_id)
        _record_view(db, viewer_id, highlight)
        return _clip_payload(highlight, db, course_for_clip)

    # 3. Every highlight has been seen inside the dedup window — LOOP
    #    instead of parking on a slate: replay the highlight this viewer
    #    saw longest ago, so the channel runs clips on repeat forever.
    from sqlalchemy import func

    loop_q = db.query(VideoClip).filter(VideoClip.is_highlight.is_(True))
    if course_id:
        loop_q = loop_q.filter(VideoClip.course_id == course_id)
    loop_pool = loop_q.order_by(desc(VideoClip.created_at)).limit(200).all()
    if loop_pool:
        last_shown = dict(
            db.query(BroadcastView.clip_id, func.max(BroadcastView.shown_at))
            .filter(
                BroadcastView.viewer_id == viewer_id,
                BroadcastView.clip_id.in_([c.id for c in loop_pool]),
            )
            .group_by(BroadcastView.clip_id)
            .all()
        )
        replay = min(
            loop_pool,
            key=lambda c: last_shown.get(c.id, datetime.min),
        )
        course_for_clip = course or db.get(Course, replay.course_id)
        _record_view(db, viewer_id, replay)
        return _clip_payload(replay, db, course_for_clip)

    # 4. Genuinely nothing to play (no highlights exist at all) — tell the
    #    client to render a leaderboard slate for a few seconds and ask
    #    again. The data shape is intentionally minimal; /watch can fetch
    #    the full /contests payload separately.
    return {"kind": "slate", "duration_ms": 8000, "course_id": course_id}


# --- Channels ---------------------------------------------------------------
# A channel is a continuous, looping playlist of Broadcast (is_highlight)
# clips: one channel per course plus a cross-course "Best Shots" channel
# ordered closest-to-the-pin first. The /watch page plays a channel's
# playlist end to end and refetches on wrap, so freshly promoted clips
# join the loop without a page reload. Public (clubhouse TVs).

BEST_CHANNEL_LABEL = "Best Shots · Closest to the Pin"


@router.get("/channels")
def list_channels(db: Session = Depends(get_db)):
    """Every available channel + its clip count: one per course that has
    at least one Broadcast clip, plus the cross-course Best Shots
    channel (clips with a measured distance-from-pin)."""
    from sqlalchemy import func

    rows = (
        db.query(VideoClip.course_id, func.count(VideoClip.id))
        .filter(VideoClip.is_highlight.is_(True))
        .group_by(VideoClip.course_id)
        .all()
    )
    course_ids = [r[0] for r in rows]
    courses = (
        {c.id: c for c in db.query(Course).filter(Course.id.in_(course_ids)).all()}
        if course_ids
        else {}
    )
    channels = [
        {
            "key": f"course-{cid}",
            "course_id": cid,
            "label": courses[cid].name if cid in courses else f"Course {cid}",
            "count": int(n),
        }
        for cid, n in sorted(rows, key=lambda r: r[0])
        if n
    ]
    best_n = (
        db.query(func.count(VideoClip.id))
        .filter(
            VideoClip.is_highlight.is_(True),
            VideoClip.distance_from_pin_feet.isnot(None),
        )
        .scalar()
        or 0
    )
    channels.append({
        "key": "best",
        "course_id": None,
        "label": BEST_CHANNEL_LABEL,
        "count": int(best_n),
    })
    return {"channels": channels}


@router.get("/channels/{key}/playlist")
def channel_playlist(
    key: str,
    limit: int = Query(default=200, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Ordered playlist for one channel. Course channels play newest
    first; the Best channel plays closest-to-the-pin first (holed-out
    shots lead, then ascending distance). The player loops the list and
    refetches when it wraps."""
    base = db.query(VideoClip).filter(VideoClip.is_highlight.is_(True))
    if key == "best":
        label = BEST_CHANNEL_LABEL
        clips = (
            base.filter(VideoClip.distance_from_pin_feet.isnot(None))
            .order_by(
                desc(VideoClip.ball_in_cup),
                VideoClip.distance_from_pin_feet.asc(),
                desc(VideoClip.created_at),
            )
            .limit(limit)
            .all()
        )
    elif key.startswith("course-"):
        try:
            cid = int(key.split("-", 1)[1])
        except ValueError:
            raise HTTPException(404, "unknown channel")
        course = db.get(Course, cid)
        if course is None:
            raise HTTPException(404, "unknown channel")
        label = course.name
        clips = (
            base.filter(VideoClip.course_id == cid)
            .order_by(desc(VideoClip.created_at))
            .limit(limit)
            .all()
        )
    else:
        raise HTTPException(404, "unknown channel")

    course_ids = {c.course_id for c in clips}
    courses = (
        {c.id: c for c in db.query(Course).filter(Course.id.in_(course_ids)).all()}
        if course_ids
        else {}
    )
    return {
        "key": key,
        "label": label,
        "clips": [
            _clip_payload(c, db, courses.get(c.course_id)) for c in clips
        ],
    }


def _record_view(db: Session, viewer_id: str, clip: VideoClip) -> None:
    db.add(
        BroadcastView(
            viewer_id=viewer_id,
            clip_id=clip.id,
            course_id=clip.course_id,
        )
    )
    db.commit()


# --- Admin tagging ----------------------------------------------------------

@router.post(
    "/admin/clips/{clip_id}/highlight",
    dependencies=[Depends(require_admin)],
)
def tag_highlight(clip_id: int, tag: Optional[str] = None, db: Session = Depends(get_db)):
    """Mark a clip as a highlight, optionally with a tag string.

    Conventional tags (lower = higher priority for tie-breaking):
      `01-ace`, `02-stiff`, `03-ctp`, `04-feature`
    """
    clip = db.get(VideoClip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="clip not found")
    clip.is_highlight = True
    if tag:
        clip.highlight_tag = tag[:60]
    db.commit()
    return {"ok": True, "clip_id": clip.id, "tag": clip.highlight_tag}


@router.delete(
    "/admin/clips/{clip_id}/highlight",
    dependencies=[Depends(require_admin)],
)
def untag_highlight(clip_id: int, db: Session = Depends(get_db)):
    clip = db.get(VideoClip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="clip not found")
    clip.is_highlight = False
    clip.highlight_tag = None
    db.commit()
    return {"ok": True, "clip_id": clip.id}


@router.post(
    "/admin/auto-tag",
    dependencies=[Depends(require_admin)],
)
def auto_tag(db: Session = Depends(get_db)):
    """One-shot auto-tagger. Re-runnable; safe to call from a cron.

    Marks:
      - approved aces → tag `01-ace`
      - any clip with distance_from_pin_feet <= 3 → tag `02-stiff`
    Doesn't un-tag anything an operator added manually.
    """
    # Approved aces. We tag every clip linked to an approved HIO event.
    approved = (
        db.query(HoleInOneEvent)
        .filter(HoleInOneEvent.status == HIOStatus.approved.value)
        .all()
    )
    ace_clip_ids: set[int] = set()
    for ev in approved:
        for cid in (ev.tee_clip_id, ev.wide_clip_id, ev.hole_clip_id):
            if cid:
                ace_clip_ids.add(cid)
    aces_tagged = 0
    if ace_clip_ids:
        for clip in db.query(VideoClip).filter(VideoClip.id.in_(ace_clip_ids)).all():
            if not clip.is_highlight or not (clip.highlight_tag or "").startswith("01-"):
                clip.is_highlight = True
                clip.highlight_tag = "01-ace"
                aces_tagged += 1

    # Stiffs — inside 3 feet of the pin.
    stiff_q = (
        db.query(VideoClip)
        .filter(VideoClip.distance_from_pin_feet.isnot(None))
        .filter(VideoClip.distance_from_pin_feet <= 3)
    )
    stiffs_tagged = 0
    for clip in stiff_q.all():
        # Don't overwrite an ace tag.
        if (clip.highlight_tag or "").startswith("01-"):
            continue
        if not clip.is_highlight:
            stiffs_tagged += 1
        clip.is_highlight = True
        if not clip.highlight_tag:
            clip.highlight_tag = "02-stiff"

    db.commit()
    return {"aces_tagged": aces_tagged, "stiffs_tagged": stiffs_tagged}
