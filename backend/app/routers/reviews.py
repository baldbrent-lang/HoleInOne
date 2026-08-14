"""Review capture on our own site.

The thank-you email links here rather than to a third-party review site.
The gallery token in the URL is the whole authentication story: whoever
has it played the round, which is the same trust we already extend to the
gallery itself. That buys a form which already knows the golfer's name
and course, so leaving a review is a rating and a sentence rather than a
sign-up.

One review per participant. The form is reachable for as long as the
gallery is, and a golfer who follows the link twice should find what they
wrote, not a blank form inviting a duplicate.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Participant, Review

router = APIRouter(prefix="/api/reviews", tags=["reviews"])


class ReviewIn(BaseModel):
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = Field(default=None, max_length=4000)


class ReviewContext(BaseModel):
    name: str
    course_name: Optional[str] = None
    already_reviewed: bool = False
    rating: Optional[int] = None
    comment: Optional[str] = None


def _participant(db: Session, token: str) -> Participant:
    p = db.query(Participant).filter(Participant.gallery_token == token).first()
    if not p:
        raise HTTPException(404, "not found")
    return p


def _course_name(participant) -> Optional[str]:
    # Same defensive walk as notifications.course_name_for: a half-built
    # row must not be the reason a golfer cannot leave a review.
    try:
        return participant.tee_time.course.name
    except Exception:  # noqa: BLE001
        return None


@router.get("/{gallery_token}", response_model=ReviewContext)
def review_context(gallery_token: str, db: Session = Depends(get_db)):
    """Who is reviewing, and have they already."""
    p = _participant(db, gallery_token)
    existing = (
        db.query(Review).filter(Review.participant_id == p.id).first()
    )
    return ReviewContext(
        name=p.name,
        course_name=_course_name(p),
        already_reviewed=existing is not None,
        rating=existing.rating if existing else None,
        comment=existing.comment if existing else None,
    )


@router.post("/{gallery_token}", response_model=ReviewContext)
def submit_review(
    gallery_token: str, payload: ReviewIn, db: Session = Depends(get_db)
):
    """Record a review, or update the one already left.

    Updating rather than rejecting: someone who comes back to add detail
    to a bare 5 stars is doing us a favour, and a hard "you already
    reviewed" would throw their writing away.
    """
    p = _participant(db, gallery_token)
    comment = (payload.comment or "").strip() or None

    review = db.query(Review).filter(Review.participant_id == p.id).first()
    if review is None:
        review = Review(
            participant_id=p.id,
            name=p.name,
            course_name=_course_name(p),
            rating=payload.rating,
            comment=comment,
        )
        db.add(review)
    else:
        review.rating = payload.rating
        review.comment = comment
    db.commit()
    db.refresh(review)

    return ReviewContext(
        name=review.name,
        course_name=review.course_name,
        already_reviewed=True,
        rating=review.rating,
        comment=review.comment,
    )
