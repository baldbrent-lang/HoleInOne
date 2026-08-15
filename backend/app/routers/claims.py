"""Prize claims for confirmed holes-in-one.

Reached from the link in the confirmation email. The gallery token says
who is claiming, and an APPROVED HoleInOneEvent says whether there is
anything to claim -- the token alone is not enough, or anyone who played
a round could open the form and file for a prize they did not win.

No payment details are collected here. The claim records who won, how to
reach them, and where a physical prize should go; the payout itself is
arranged directly afterwards.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import HIOStatus, HoleInOneEvent, Participant, PrizeClaim

router = APIRouter(prefix="/api/claims", tags=["claims"])


class ClaimIn(BaseModel):
    email: Optional[str] = Field(default=None, max_length=200)
    mobile: Optional[str] = Field(default=None, max_length=40)
    mailing_address: Optional[str] = Field(default=None, max_length=2000)
    note: Optional[str] = Field(default=None, max_length=2000)


class ClaimContext(BaseModel):
    name: str
    course_name: Optional[str] = None
    hole_number: Optional[int] = None
    eligible: bool = False
    already_claimed: bool = False
    # Pre-fill only. What we already hold, so a winner is not made to
    # retype what they gave us at registration.
    email: Optional[str] = None
    mobile: Optional[str] = None
    status: Optional[str] = None
    # What they have won, worded exactly as the email worded it.
    prize_label: Optional[str] = None


def _participant(db: Session, token: str) -> Participant:
    p = db.query(Participant).filter(Participant.gallery_token == token).first()
    if not p:
        raise HTTPException(404, "not found")
    return p


def _approved_ace(db: Session, participant_id: int) -> Optional[HoleInOneEvent]:
    return (
        db.query(HoleInOneEvent)
        .filter(
            HoleInOneEvent.participant_id == participant_id,
            HoleInOneEvent.status == HIOStatus.approved.value,
        )
        .order_by(HoleInOneEvent.decided_at.desc())
        .first()
    )


def _course_name(participant) -> Optional[str]:
    try:
        return participant.tee_time.course.name
    except Exception:  # noqa: BLE001
        return None


def _context(db: Session, p: Participant) -> ClaimContext:
    ace = _approved_ace(db, p.id)
    claim = (
        db.query(PrizeClaim)
        .filter(PrizeClaim.participant_id == p.id)
        .order_by(PrizeClaim.created_at.desc())
        .first()
    )
    return ClaimContext(
        name=p.name,
        course_name=_course_name(p),
        hole_number=ace.hole_number if ace else None,
        eligible=ace is not None,
        already_claimed=claim is not None,
        email=(claim.email if claim else p.email),
        mobile=(claim.mobile if claim else p.mobile),
        status=claim.status if claim else None,
        prize_label=(settings.hio_prize_label or "").strip() or None,
    )


@router.get("/{gallery_token}", response_model=ClaimContext)
def claim_context(gallery_token: str, db: Session = Depends(get_db)):
    """Whether this golfer has a confirmed ace, and whether they claimed."""
    return _context(db, _participant(db, gallery_token))


@router.post("/{gallery_token}", response_model=ClaimContext)
def submit_claim(
    gallery_token: str, payload: ClaimIn, db: Session = Depends(get_db)
):
    """File the claim, or update the details on one already filed.

    Updating rather than rejecting: a winner who realises they gave the
    wrong address should be able to fix it themselves rather than having
    to find someone to email about it.
    """
    p = _participant(db, gallery_token)
    ace = _approved_ace(db, p.id)
    if ace is None:
        # Deliberately not 404 -- the link is valid, there is simply
        # nothing confirmed against it yet.
        raise HTTPException(403, "no confirmed hole-in-one on this account")

    def _clean(v: str | None) -> str | None:
        return (v or "").strip() or None

    claim = (
        db.query(PrizeClaim)
        .filter(PrizeClaim.participant_id == p.id)
        .order_by(PrizeClaim.created_at.desc())
        .first()
    )
    if claim is None:
        claim = PrizeClaim(
            participant_id=p.id,
            hio_event_id=ace.id,
            name=p.name,
            email=_clean(payload.email) or p.email,
            mobile=_clean(payload.mobile) or p.mobile,
            course_name=_course_name(p),
            hole_number=ace.hole_number,
            mailing_address=_clean(payload.mailing_address),
            note=_clean(payload.note),
        )
        db.add(claim)
    else:
        claim.email = _clean(payload.email) or claim.email
        claim.mobile = _clean(payload.mobile) or claim.mobile
        claim.mailing_address = _clean(payload.mailing_address)
        claim.note = _clean(payload.note)
    db.commit()

    return _context(db, p)
