"""The delayed thank-you / review email.

The ask for a review cannot ride along with the gallery email. A golfer
who has just been told their videos are ready has not watched them yet,
and asking for a review before they have seen anything is asking them to
rate a thing they have not received. So the thank-you waits
(`settings.thanks_delay_hours`, four by default) and goes out once the
round has had time to land.

The delay is stored, not slept. A `time.sleep(4 * 3600)` in a thread is
gone the moment Replit redeploys or the container recycles, and a golfer
whose round finished before a deploy would simply never hear from us.
Instead the due time is written to `Participant.thanks_due_at` when the
gallery email fires, and the sweeper below asks the database what is due.
Restarts cost nothing: whatever was owed is still owed, and gets sent on
the next pass.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta

from ..config import settings
from . import notifications

log = logging.getLogger("golfreelz.thanks")

SWEEP_INTERVAL_SEC = 300.0   # 5 min; the window is hours, so this is ample
_BATCH = 50                  # cap per pass, so a backlog cannot stall boot


def schedule_thanks(participant) -> None:
    """Stamp when this golfer's thank-you should go out.

    Called at the moment the gallery email is sent. Never overwrites an
    existing due time: the gallery mail can be re-sent by an admin, and a
    re-send should not push the review ask further away (or, worse, queue
    a second one).
    """
    delay = float(settings.thanks_delay_hours or 0)
    if delay <= 0:
        return
    if participant.thanks_due_at is not None or participant.thanks_sent_at is not None:
        return
    participant.thanks_due_at = datetime.utcnow() + timedelta(hours=delay)


def sweep_due_thanks(db) -> int:
    """Send every thank-you whose time has come. Returns how many went.

    Each send is committed on its own. One golfer with a bad address must
    not roll back the sends that already succeeded in this pass, and a
    provider timeout halfway through a batch must not re-send the first
    half on the next sweep.
    """
    from ..models import Participant

    now = datetime.utcnow()
    due = (
        db.query(Participant)
        .filter(
            Participant.thanks_due_at.isnot(None),
            Participant.thanks_due_at <= now,
            Participant.thanks_sent_at.is_(None),
            Participant.email.isnot(None),
        )
        .order_by(Participant.thanks_due_at.asc())
        .limit(_BATCH)
        .all()
    )

    sent = 0
    for p in due:
        gallery_url = f"{settings.app_base_url}/g/{p.gallery_token}"
        try:
            notifications.notify_thanks_for_playing(
                p.name, p.mobile, p.email, gallery_url,
                course_name=notifications.course_name_for(p),
                review_url=notifications.review_url_for(p.gallery_token),
            )
        except Exception as exc:  # noqa: BLE001
            # Leave thanks_sent_at unset so the next pass retries. A
            # transient SendGrid blip should not silently cost the email.
            log.warning("thanks send failed for participant %s: %s", p.id, exc)
            continue
        # Stamped only after a clean send, and committed immediately.
        p.thanks_sent_at = datetime.utcnow()
        db.commit()
        sent += 1

    if sent:
        log.info("thanks: sent %s thank-you email(s)", sent)
    return sent


def start_thanks_sweeper(interval_sec: float = SWEEP_INTERVAL_SEC) -> None:
    """Run sweep_due_thanks on a loop. Idempotent to start.

    No-op when thanks_delay_hours is 0 -- that is the off switch for the
    automatic send, and it should not leave a thread spinning.
    """
    if not settings.thanks_delay_hours:
        log.info("thanks: automatic send disabled (thanks_delay_hours=0)")
        return
    if getattr(start_thanks_sweeper, "_started", False):
        return
    start_thanks_sweeper._started = True  # type: ignore[attr-defined]

    def _loop() -> None:
        from ..database import SessionLocal

        log.info(
            "thanks: sweeper started (%sh delay, %ss interval)",
            settings.thanks_delay_hours, interval_sec,
        )
        while True:
            db = SessionLocal()
            try:
                sweep_due_thanks(db)
            except Exception as exc:  # noqa: BLE001
                log.warning("thanks: sweep failed: %s", exc)
                try:
                    db.rollback()
                except Exception:  # noqa: BLE001
                    pass
            finally:
                db.close()
            time.sleep(interval_sec)

    threading.Thread(target=_loop, daemon=True, name="thanks-sweeper").start()
