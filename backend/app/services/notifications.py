"""Notification service. No-ops when provider credentials are absent."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from ..config import settings

log = logging.getLogger("golfreelz.notify")

# Hard email-server cap. Most providers (Gmail, Yahoo, Outlook) reject above
# ~25MB; SendGrid's API limit is 30MB including base64 overhead (~33%), so
# 22MB raw is the safe ceiling. Uploads are pre-compressed via ffmpeg in
# services/video.py to fit comfortably; this cap is a last-resort guard.
MAX_ATTACH_BYTES = 22 * 1024 * 1024

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


def _smtp_configured() -> bool:
    return bool(settings.smtp_host and settings.smtp_user and settings.smtp_password)


def _send_smtp(
    to: str,
    subject: str,
    body: str,
    attachments: list[tuple[str, bytes, str]] | None = None,
) -> None:
    """Plain-stdlib SMTP send with optional attachments. attachments is a list
    of (filename, bytes, mime_type)."""
    import smtplib
    import ssl
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["From"] = settings.smtp_from or settings.smtp_user
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    for fname, fbytes, mime in attachments or []:
        maintype, _, subtype = mime.partition("/")
        if not subtype:
            maintype, subtype = "application", "octet-stream"
        msg.add_attachment(fbytes, maintype=maintype, subtype=subtype, filename=fname)

    context = ssl.create_default_context()
    if settings.smtp_use_ssl or settings.smtp_port == 465:
        with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, context=context, timeout=30) as s:
            s.login(settings.smtp_user, settings.smtp_password)
            s.send_message(msg)
    else:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as s:
            s.ehlo()
            s.starttls(context=context)
            s.ehlo()
            s.login(settings.smtp_user, settings.smtp_password)
            s.send_message(msg)


def send_email(to: str | None, subject: str, body: str) -> None:
    if not to:
        return
    if _smtp_configured():
        _send_smtp(to, subject, body)
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
    if _smtp_configured():
        _send_smtp(to, subject, body, attachments=[(file_name, file_bytes, mime_type)])
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


def send_email_with_attachments(
    to: str | None,
    subject: str,
    body: str,
    attachments: list[tuple[str, bytes, str]],
) -> None:
    """attachments: list of (filename, bytes, mime_type)."""
    if not to:
        return
    if _smtp_configured():
        _send_smtp(to, subject, body, attachments=attachments)
        return
    if not settings.sendgrid_api_key:
        log.info(
            "EMAIL+ATTACHx%d (mock) -> %s | %s | %s | %s",
            len(attachments), to, subject, body,
            ", ".join(f"{n} ({len(b)}b)" for (n, b, _t) in attachments),
        )
        return
    import base64
    from sendgrid import SendGridAPIClient  # type: ignore
    from sendgrid.helpers.mail import (  # type: ignore
        Attachment, Disposition, FileContent, FileName, FileType, Mail,
    )

    message = Mail(
        from_email=settings.sendgrid_from_email,
        to_emails=to,
        subject=subject,
        plain_text_content=body,
    )
    sg_attachments = []
    for fname, fbytes, mime in attachments:
        sg_attachments.append(Attachment(
            FileContent(base64.b64encode(fbytes).decode()),
            FileName(fname),
            FileType(mime),
            Disposition("attachment"),
        ))
    message.attachment = sg_attachments
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


def maybe_send_round_summary(db, participant, course) -> bool:
    """Send a single email with all par-3 clips attached, once per round.

    Subject: 'Golf Reelz - <Course Name>'
    Body: per-hole stats + gallery link
    Attachments: one MP4 per par-3 hole, named 'Hole-<N>.mp4'

    Skips if:
    - participant has no email
    - course has no par-3 holes configured
    - any par-3 hole is missing an assigned clip (round not complete yet)
    - summary was already sent (Participant.summary_sent_at)

    If the combined attachments would exceed the email cap, falls back to
    a body containing per-hole download links instead.

    Returns True if a send was attempted.
    """
    from ..models import ClipProcessingStatus, VideoClip

    if not participant or not course or not participant.email:
        return False
    if participant.summary_sent_at is not None:
        return False
    par3s: list[int] = list(course.par3_holes or [])
    if not par3s:
        return False

    # Make sure pending in-flight changes (the clip we just assigned in
    # the matcher) are visible to the query below.
    db.flush()

    clips = (
        db.query(VideoClip)
        .filter(
            VideoClip.participant_id == participant.id,
            VideoClip.processing_status == ClipProcessingStatus.assigned.value,
        )
        .order_by(VideoClip.hole_number.asc(), VideoClip.captured_at.asc())
        .all()
    )

    # Pick one clip per hole (prefer the tee camera if multiple angles).
    by_hole: dict[int, VideoClip] = {}
    for c in clips:
        cur = by_hole.get(c.hole_number)
        if cur is None or (c.camera_type == "tee" and cur.camera_type != "tee"):
            by_hole[c.hole_number] = c

    missing = [h for h in par3s if h not in by_hole]
    if missing:
        log.info(
            "summary not yet sent for participant %s: missing holes %s",
            participant.id, missing,
        )
        return False

    selected = [by_hole[h] for h in par3s]

    yardages = course.hole_yardages or {}
    body_lines = [
        f"Your round at {course.name}, {participant.name}.",
        "",
        "Each clip is attached below — see the file name for which hole.",
        "",
    ]
    for c in selected:
        y = yardages.get(str(c.hole_number))
        bits = []
        if y: bits.append(f"{y} yds")
        if c.carry_yards: bits.append(f"{c.carry_yards} yd carry")
        if c.ball_speed_mph: bits.append(f"{c.ball_speed_mph} mph")
        suffix = ("  ·  " + "  ·  ".join(bits)) if bits else ""
        ace = "   HOLE-IN-ONE!" if c.ball_in_cup else ""
        body_lines.append(f"  Hole {c.hole_number}{suffix}{ace}")
    body_lines.append("")
    body_lines.append(f"Full gallery: {settings.app_base_url}/g/{participant.gallery_token}")

    subject = f"Golf Reelz - {course.name}"

    attachments: list[tuple[str, bytes, str]] = []
    total_bytes = 0
    for c in selected:
        path = _local_path_from_url(c.source_url)
        if not path or not path.is_file():
            continue
        data = path.read_bytes()
        ext = path.suffix.lstrip(".").lower() or "mp4"
        attachments.append((f"Hole-{c.hole_number}.{ext}", data, _mime_for(ext)))
        total_bytes += len(data)

    if attachments and total_bytes <= MAX_ATTACH_BYTES:
        try:
            send_email_with_attachments(
                participant.email, subject, "\n".join(body_lines), attachments,
            )
            participant.summary_sent_at = datetime.utcnow()
            for c in selected:
                if c.delivered_at is None:
                    c.delivered_at = participant.summary_sent_at
            return True
        except Exception as exc:  # pragma: no cover
            log.warning("summary send failed for participant %s: %s", participant.id, exc)
            # fall through to link-only body

    # Fallback: too big OR send failed — body with per-hole download links.
    body_lines.append("")
    body_lines.append("Clip downloads:")
    for c in selected:
        body_lines.append(f"  Hole {c.hole_number}: {c.source_url}")
    send_email(participant.email, subject, "\n".join(body_lines))
    participant.summary_sent_at = datetime.utcnow()
    return True
