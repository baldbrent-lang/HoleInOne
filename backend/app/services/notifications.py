"""Notification service. No-ops when provider credentials are absent."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from ..config import settings

log = logging.getLogger("golfreelz.notify")

# Maximum attachment size for email. Most providers (Gmail, Yahoo, Outlook)
# cap inbound email at 25MB; SendGrid recommends <20MB to leave headroom
# for MIME encoding overhead.
MAX_ATTACH_BYTES = 20 * 1024 * 1024

# Resolve once so file lookups don't traverse on every call.
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_UPLOAD_ROOT = _BACKEND_ROOT / settings.upload_dir


def _local_path_from_url(url: str | None) -> Path | None:
    if not url:
        return None
    marker = "/uploads/"
    idx = url.find(marker)
    if idx < 0:
        return None
    rel = url[idx + len(marker):]
    return _UPLOAD_ROOT / rel


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


def send_email_with_attachment(
    to: str | None,
    subject: str,
    body: str,
    file_bytes: bytes,
    file_name: str,
    mime_type: str = "application/octet-stream",
) -> None:
    if not to:
        return
    if not settings.sendgrid_api_key:
        log.info(
            "EMAIL+ATTACH (mock) -> %s | %s | %s | %s (%d bytes, %s)",
            to, subject, body, file_name, len(file_bytes), mime_type,
        )
        return
    import base64
    from sendgrid import SendGridAPIClient  # type: ignore
    from sendgrid.helpers.mail import (  # type: ignore
        Attachment, Disposition, FileContent, FileName, FileType, Mail,
    )

    encoded = base64.b64encode(file_bytes).decode()
    attachment = Attachment(
        FileContent(encoded),
        FileName(file_name),
        FileType(mime_type),
        Disposition("attachment"),
    )
    message = Mail(
        from_email=settings.sendgrid_from_email,
        to_emails=to,
        subject=subject,
        plain_text_content=body,
    )
    message.attachment = attachment
    SendGridAPIClient(settings.sendgrid_api_key).send(message)


def notify_registration_confirmed(name: str, mobile: str | None, email: str | None, gallery_url: str) -> None:
    msg = (
        f"You're registered for GolfReelz, {name}! "
        f"We'll text you when your videos are ready. Your gallery: {gallery_url}"
    )
    send_sms(mobile, msg)
    send_email(email, "You're registered with GolfReelz", msg)


def notify_gallery_ready(name: str, mobile: str | None, email: str | None, gallery_url: str) -> None:
    msg = f"Your GolfReelz videos are ready, {name}! View + share: {gallery_url}"
    send_sms(mobile, msg)
    send_email(email, "Your GolfReelz videos are ready", msg)


def notify_hio_under_review(name: str, mobile: str | None, email: str | None) -> None:
    msg = f"GolfReelz flagged a possible hole-in-one for you, {name}! We're reviewing footage now."
    send_sms(mobile, msg)
    send_email(email, "GolfReelz: possible hole-in-one under review", msg)


def notify_hio_confirmed(name: str, mobile: str | None, email: str | None, gallery_url: str) -> None:
    msg = f"HOLE IN ONE confirmed, {name}! Congrats. Clip + prize info: {gallery_url}"
    send_sms(mobile, msg)
    send_email(email, "GolfReelz: hole-in-one CONFIRMED", msg)


def notify_clip_ready(participant, clip, course) -> bool:
    """Email a freshly-assigned clip to the golfer, attaching the video.

    Subject: "<Course Name> - Hole #<N>"
    Body: short summary + gallery link.
    Attachment: the clip itself (MP4) when it's a locally-stored upload
    smaller than the email size cap. Otherwise falls back to a download
    link in the body.

    Idempotent: sets clip.delivered_at on success and returns False on
    subsequent calls for the same clip.

    Returns True if a delivery attempt was made (so callers can persist
    delivered_at).
    """
    if not participant or not clip or not course:
        return False
    if clip.delivered_at is not None:
        return False
    if not participant.email:
        # Per spec: only email delivery for now. Skip silently.
        log.info("notify_clip_ready: no email on participant %s; skipping", participant.id)
        return False

    subject = f"{course.name} - Hole #{clip.hole_number}"
    yardage = (course.hole_yardages or {}).get(str(clip.hole_number))
    gallery_url = f"{settings.app_base_url}/g/{participant.gallery_token}"
    body_lines = [
        f"Your shot, {participant.name}.",
        "",
        f"Hole {clip.hole_number}"
        + (f" · {yardage} yds" if yardage else "")
        + (f" · {clip.carry_yards} yd carry" if clip.carry_yards else "")
        + (f" · {clip.ball_speed_mph} mph" if clip.ball_speed_mph else ""),
        "",
        f"Full gallery: {gallery_url}",
    ]

    file_path = _local_path_from_url(clip.source_url)
    if file_path and file_path.is_file():
        size = file_path.stat().st_size
        if size <= MAX_ATTACH_BYTES:
            try:
                data = file_path.read_bytes()
                send_email_with_attachment(
                    participant.email,
                    subject,
                    "\n".join(body_lines),
                    file_bytes=data,
                    file_name=file_path.name,
                    mime_type=_mime_for(file_path.suffix),
                )
                return True
            except Exception as exc:  # pragma: no cover - SendGrid passthrough
                log.warning(
                    "clip-ready attachment failed for clip %s, falling back to link: %s",
                    clip.id, exc,
                )
        else:
            log.info(
                "clip-ready clip %s is %.1fMB; over %.1fMB cap, sending link instead",
                clip.id, size / 1024 / 1024, MAX_ATTACH_BYTES / 1024 / 1024,
            )

    # Fallback: send body with download link only
    body_lines.append("")
    body_lines.append(f"Watch / download: {clip.source_url}")
    send_email(participant.email, subject, "\n".join(body_lines))
    return True


def _mime_for(ext: str) -> str:
    ext = (ext or "").lower().lstrip(".")
    return {
        "mp4": "video/mp4",
        "m4v": "video/mp4",
        "mov": "video/quicktime",
        "webm": "video/webm",
    }.get(ext, "application/octet-stream")


def mark_delivered(clip) -> None:
    """Record the delivery timestamp on a clip after a successful send."""
    clip.delivered_at = datetime.utcnow()
