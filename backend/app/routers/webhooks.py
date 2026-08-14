from __future__ import annotations

from collections import defaultdict
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import ClipProcessingStatus, Participant, VideoClip
from ..schemas import IncomingClip
from ..services import notifications
from ..services.matcher import match_clip
from ..services.stripe_service import verify_webhook

router = APIRouter(prefix="/api/webhooks", tags=["webhooks"])


@router.post("/shot-tracer")
def shot_tracer_ingest(
    payload: IncomingClip,
    db: Session = Depends(get_db),
    x_webhook_secret: str | None = Header(default=None),
):
    if settings.shot_tracer_webhook_secret and x_webhook_secret != settings.shot_tracer_webhook_secret:
        raise HTTPException(401, "bad webhook secret")

    clip = VideoClip(
        course_id=payload.course_id,
        hole_number=payload.hole_number,
        camera_type=payload.camera_type,
        captured_at=payload.captured_at,
        source_url=payload.source_url,
        thumbnail_url=payload.thumbnail_url,
        carry_yards=payload.carry_yards,
        apex_feet=payload.apex_feet,
        ball_speed_mph=payload.ball_speed_mph,
        ball_in_cup=payload.ball_in_cup,
        processing_status=ClipProcessingStatus.received.value,
    )
    db.add(clip)
    db.flush()

    participant = match_clip(db, clip)

    if participant and payload.ball_in_cup:
        notifications.notify_hio_under_review(participant.name, participant.mobile, participant.email)

    db.commit()
    _maybe_send_gallery_ready(db, clip)
    return {"clip_id": clip.id, "status": clip.processing_status, "participant_id": clip.participant_id}


def _maybe_send_gallery_ready(db: Session, clip: VideoClip) -> None:
    """Fire gallery-ready notification once a participant has at least one assigned clip.

    V0 rule: notify on the first assigned clip per participant. A production
    implementation would wait for round completion signals from the course.
    """
    if clip.participant_id is None or clip.processing_status != ClipProcessingStatus.assigned.value:
        return
    participant = db.get(Participant, clip.participant_id)
    if not participant or participant.gallery_ready_sent:
        return
    participant.gallery_ready_sent = True
    gallery_url = f"{settings.app_base_url}/g/{participant.gallery_token}"
    notifications.notify_gallery_ready(
        participant.name, participant.mobile, participant.email, gallery_url,
        course_name=notifications.course_name_for(participant),
    )
    db.commit()


@router.post("/stripe")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    body = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = verify_webhook(body, sig)
    except Exception as exc:  # pragma: no cover - passthrough of Stripe errors
        raise HTTPException(400, f"bad signature: {exc}")

    if event.get("type") == "payment_intent.succeeded":
        data = event.get("data", {}).get("object", {})
        pi_id = data.get("id")
        if pi_id:
            p = db.query(Participant).filter(Participant.stripe_payment_intent_id == pi_id).first()
            if p and not p.paid:
                p.paid = True
                gallery_url = f"{settings.app_base_url}/g/{p.gallery_token}"
                notifications.notify_registration_confirmed(
                    p.name, p.mobile, p.email, gallery_url,
                    # The subject names the course, so it has to be
                    # resolved here too rather than left blank on the one
                    # path that does not already have it in hand.
                    course_name=notifications.course_name_for(p),
                    # This fires on Stripe's confirmation of the lead's
                    # own payment, and the lead's selfie was required at
                    # registration -- but a member who somehow has no
                    # photo should still be asked for one.
                    invite_url=(
                        None if p.selfie_path
                        else f"{settings.app_base_url}/invite/{p.gallery_token}"
                    ),
                )
                db.commit()
    return {"ok": True}


# --- test/demo helper: bulk-ingest a synthetic round --------------------------
@router.post("/debug/simulate-round")
def simulate_round(
    payload: dict,
    db: Session = Depends(get_db),
    x_admin_password: str | None = Header(default=None),
):
    """Admin-only helper to inject synthetic clips for a tee_time_id.

    Body: {"tee_time_id": int, "include_hio": bool}
    """
    from datetime import timedelta
    from ..models import Course, TeeTime

    if x_admin_password != settings.admin_password:
        raise HTTPException(401, "admin password required")

    tt = db.get(TeeTime, payload.get("tee_time_id"))
    if not tt:
        raise HTTPException(404, "tee time not found")
    course = db.get(Course, tt.course_id)
    include_hio = bool(payload.get("include_hio"))
    par3_holes = course.par3_holes or [3, 7, 12, 16]

    # Generate one clip per registered participant per par-3 hole. The
    # appearance matcher picks who's who; in stub mode it round-robins.
    n_shots = max(len(tt.participants), 1)

    counts: dict = defaultdict(int)
    for idx, hole in enumerate(par3_holes):
        offset = timedelta(minutes=course.minutes_per_hole * (hole - 1) + 3)
        captured = tt.starts_at + offset
        for order in range(1, n_shots + 1):
            clip = VideoClip(
                course_id=course.id,
                hole_number=hole,
                camera_type="tee",
                captured_at=captured + timedelta(seconds=order * 90),
                source_url=f"https://cdn.example.com/mock/{course.id}/{hole}/{order}.mp4",
                thumbnail_url=f"https://cdn.example.com/mock/{course.id}/{hole}/{order}.jpg",
                carry_yards=150 + (hole + order) % 40,
                apex_feet=70 + (hole * order) % 30,
                ball_speed_mph=95 + (hole + order) % 20,
                ball_in_cup=(include_hio and idx == 0 and order == 1),
                processing_status=ClipProcessingStatus.received.value,
            )
            db.add(clip)
            db.flush()
            match_clip(db, clip)
            counts[clip.processing_status] += 1
    db.commit()
    return {"injected": sum(counts.values()), "by_status": dict(counts)}
