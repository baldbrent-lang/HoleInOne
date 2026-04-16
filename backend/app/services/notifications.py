"""Notification service. No-ops when provider credentials are absent."""
from __future__ import annotations

import logging

from ..config import settings

log = logging.getLogger("parone.notify")


def send_sms(to: str | None, body: str) -> None:
    if not to:
        return
    if not (settings.twilio_account_sid and settings.twilio_auth_token and settings.twilio_from_number):
        log.info("SMS (mock) -> %s: %s", to, body)
        return
    from twilio.rest import Client  # type: ignore

    client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
    client.messages.create(from_=settings.twilio_from_number, to=to, body=body)


def send_email(to: str | None, subject: str, body: str) -> None:
    if not to:
        return
    if not settings.sendgrid_api_key:
        log.info("EMAIL (mock) -> %s | %s | %s", to, subject, body)
        return
    from sendgrid import SendGridAPIClient  # type: ignore
    from sendgrid.helpers.mail import Mail  # type: ignore

    message = Mail(
        from_email=settings.sendgrid_from_email,
        to_emails=to,
        subject=subject,
        plain_text_content=body,
    )
    SendGridAPIClient(settings.sendgrid_api_key).send(message)


def notify_registration_confirmed(name: str, mobile: str | None, email: str | None, gallery_url: str) -> None:
    msg = (
        f"You're registered for Par One, {name}! "
        f"We'll text you when your videos are ready. Your gallery: {gallery_url}"
    )
    send_sms(mobile, msg)
    send_email(email, "You're registered with Par One", msg)


def notify_gallery_ready(name: str, mobile: str | None, email: str | None, gallery_url: str) -> None:
    msg = f"Your Par One videos are ready, {name}! View + share: {gallery_url}"
    send_sms(mobile, msg)
    send_email(email, "Your Par One videos are ready", msg)


def notify_hio_under_review(name: str, mobile: str | None, email: str | None) -> None:
    msg = f"Par One flagged a possible hole-in-one for you, {name}! We're reviewing footage now."
    send_sms(mobile, msg)
    send_email(email, "Par One: possible hole-in-one under review", msg)


def notify_hio_confirmed(name: str, mobile: str | None, email: str | None, gallery_url: str) -> None:
    msg = f"HOLE IN ONE confirmed, {name}! Congrats. Clip + prize info: {gallery_url}"
    send_sms(mobile, msg)
    send_email(email, "Par One: hole-in-one CONFIRMED", msg)
