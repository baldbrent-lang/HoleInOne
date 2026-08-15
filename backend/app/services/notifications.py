"""Notification service. No-ops when provider credentials are absent."""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import settings

log = logging.getLogger("golfreelz.notify")

# Resolve once so file lookups don't traverse on every call.
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_UPLOAD_ROOT = _BACKEND_ROOT / settings.upload_dir
_REPO_ROOT = _BACKEND_ROOT.parent


# ── branding ───────────────────────────────────────────────────────────
# The logo footer. Plain text cannot carry an image, so every email now
# goes out as multipart/alternative: the existing plain-text body
# unchanged (still what text-only clients and screen readers get), plus an
# HTML alternative that ends with the logo.
#
# The logo is attached INLINE by content-id rather than linked to a URL.
# A linked image is blocked by default in Gmail, Outlook and Apple Mail
# until the reader clicks "show images", which is exactly the wrong first
# impression; an inline part renders immediately.
_LOGO_CID = "golfreelz-logo"
# Anchored to the repo root, NOT the working directory: start.sh runs
# uvicorn from inside backend/, so a bare "frontend/dist/..." resolves to
# backend/frontend/dist/... and silently finds nothing — emails then go out
# text-only with no logo and only an INFO line to say so.
_LOGO_CANDIDATES = (
    _REPO_ROOT / "frontend" / "dist" / "golfreelz-logo.png",
    _REPO_ROOT / "frontend" / "public" / "golfreelz-logo.png",
)
_LOGO_WIDTH_PX = 280          # 2x the 140px display width, for retina
_logo_cache: tuple[bytes, str] | None | bool = False   # False = not tried


def _logo_png() -> bytes | None:
    """Small PNG of the logo for the email footer, cached after the first
    call.

    The source file is 661KB — fine for a web page, wasteful attached to
    every clip email, and enough to push a multi-clip round summary over
    the 22MB attachment cap on its own. Downscale once to footer size."""
    global _logo_cache
    if _logo_cache is not False:
        return _logo_cache[0] if _logo_cache else None
    _logo_cache = None
    for p in _LOGO_CANDIDATES:
        if not p.is_file():
            continue
        try:
            raw = p.read_bytes()
            try:
                import cv2  # type: ignore
                import numpy as np  # type: ignore

                # IMREAD_UNCHANGED keeps the alpha channel, so a logo with
                # a transparent background does not gain a black box.
                img = cv2.imdecode(
                    np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED,
                )
                if img is not None and img.shape[1] > _LOGO_WIDTH_PX:
                    scale = _LOGO_WIDTH_PX / float(img.shape[1])
                    img = cv2.resize(
                        img,
                        (_LOGO_WIDTH_PX, max(1, int(img.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                ok, buf = cv2.imencode(".png", img)
                if ok:
                    raw = bytes(buf)
            except Exception as exc:  # noqa: BLE001
                log.info("email logo: using full-size PNG (%s)", exc)
            _logo_cache = (raw, "image/png")
            log.info(
                "email logo: %s -> %.1fKB", p, len(raw) / 1024.0,
            )
            break
        except Exception as exc:  # noqa: BLE001
            log.warning("email logo: could not read %s: %s", p, exc)
    if _logo_cache is None:
        log.info("email logo: not found; emails will go out text-only")
    return _logo_cache[0] if _logo_cache else None


def _html_body(text: str) -> str:
    """The plain-text body as HTML, with the logo footer.

    Deliberately plain: inline styles only (mail clients strip <style>),
    a table-free single column, and no remote assets. Bare URLs in the
    text are made clickable, since that is the whole point of most of
    these messages."""
    import html as _html
    import re as _re

    esc = _html.escape(text)
    esc = _re.sub(
        r"(https?://[^\s<]+)",
        r'<a href="\1" style="color:#047857">\1</a>',
        esc,
    )
    lines = esc.split("\n")
    para = "<br>".join(lines)
    return (
        '<div style="font-family:-apple-system,Segoe UI,Roboto,Helvetica,'
        'Arial,sans-serif;font-size:15px;line-height:1.55;color:#111827;'
        'max-width:560px">'
        f"<div>{para}</div>"
        '<div style="margin-top:28px;padding-top:18px;'
        'border-top:1px solid #e5e7eb;text-align:center">'
        f'<img src="cid:{_LOGO_CID}" alt="GolfReelz" width="140" '
        'style="width:140px;height:auto;border:0;display:inline-block">'
        '<div style="margin-top:8px;font-size:12px;color:#6b7280">'
        "Every par-3 shot, tracked and delivered."
        "</div></div></div>"
    )


def _sg_logo_attachment():
    """The logo as an INLINE SendGrid attachment, or None.

    Same reasoning as the SMTP path: content-id inline rather than a
    linked image, so it renders without the reader allowing remote
    content."""
    _logo = _logo_png()
    if not _logo:
        return None
    import base64

    from sendgrid.helpers.mail import (  # type: ignore
        Attachment, ContentId, Disposition, FileContent, FileName, FileType,
    )

    a = Attachment(
        FileContent(base64.b64encode(_logo).decode()),
        FileName("golfreelz-logo.png"),
        FileType("image/png"),
        Disposition("inline"),
    )
    a.content_id = ContentId(_LOGO_CID)
    return a


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
    # HTML alternative carrying the logo footer. The plain-text part above
    # stays exactly as it was, so nothing is lost for text-only readers —
    # and if the logo file is missing we simply do not add the HTML part.
    _logo = _logo_png()
    if _logo:
        msg.add_alternative(_html_body(body), subtype="html")
        # Attach the logo to the HTML part (not the message), so it is a
        # multipart/related child of the alternative — which is what makes
        # cid: resolve instead of showing as a second attachment.
        # disposition="inline" matters: with a filename and no explicit
        # disposition the part is marked as an attachment, and Outlook then
        # shows a paperclip beside an image it is already displaying.
        msg.get_payload()[-1].add_related(
            _logo, maintype="image", subtype="png",
            cid=f"<{_LOGO_CID}>", filename="golfreelz-logo.png",
            disposition="inline",
        )
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
    _lg = _sg_logo_attachment()
    if _lg is not None:
        message.add_content(_html_body(body), "text/html")
        message.add_attachment(_lg)
    SendGridAPIClient(settings.sendgrid_api_key).send(message)


def notify_registration_confirmed(
    name: str,
    mobile: str | None,
    email: str | None,
    gallery_url: str,
    course_name: str | None = None,
    invite_url: str | None = None,
) -> None:
    """THE registration email. One template, every way in.

    There used to be three, all saying the same thing in different
    words: one for a golfer who signed themselves up, one for a group
    member whose photo was taken at the desk, and one for a group member
    whose photo was not. Three subject lines, three voices, one event.

    So there is one now, and the only thing that varies is a single
    block. `invite_url` is set when we do NOT have that golfer's photo
    yet -- which happens only when the lead registers someone who is not
    standing there, since a golfer registering themselves cannot get
    past the form without one. Without a photo the matcher has nothing
    to recognise them by and their clips go unassigned, so that case
    still has to ask, and it is the one thing in the email that is a
    request rather than a confirmation.
    """
    course = (course_name or "").strip()
    at_course = f" at {course}" if course else ""

    lines = [f"{name},", ""]
    lines.append(
        f"You are registered to participate in GolfReelz{at_course}.")
    lines.append("")
    lines.append(
        "Head out and enjoy your round. We will automatically track and "
        "record every one of your par-3 tee shots — nothing to set up, "
        "nothing to carry, and no need to do anything differently out "
        "there.")
    lines.append("")
    lines.append(
        "When you finish, keep an eye on your inbox. Your videos will be "
        "sent to you shortly after your round.")
    if invite_url:
        lines.append("")
        lines.append(
            "One more step first: we match videos to you by what you are "
            "wearing, and we do not have your photo yet. Add one here and "
            "you are all set:")
        lines.append(invite_url)
    lines.append("")
    lines.append("Your gallery, where every clip is collected:")
    lines.append(gallery_url)
    lines.append("")
    lines.append(
        "Best of luck out there — may this be the round you make an ace.")

    subject = f"{name}, you are registered for GolfReelz{at_course}"
    send_email(email, subject, "\n".join(lines))

    # The text message stays a text message: one line, and the link that
    # matters most -- the photo when we need it, the gallery otherwise.
    if invite_url:
        send_sms(mobile, (
            f"GolfReelz: you're registered{at_course}, {name}. One step "
            f"left — add a photo so we can match your shots: {invite_url}"))
    else:
        send_sms(mobile, (
            f"GolfReelz: you're registered{at_course}, {name}. We'll "
            f"record every par 3 and email your videos after the round. "
            f"Your gallery: {gallery_url}"))


def course_name_for(participant) -> str | None:
    """The course a participant played, or None.

    Every one of these messages wants to name the course, and every
    caller has a participant rather than a course to hand. The walk is
    guarded because a half-built or orphaned row must not be the reason
    a golfer never hears from us -- a missing course name costs the
    email three words, a raised exception costs the whole email.
    """
    try:
        return participant.tee_time.course.name
    except Exception:  # noqa: BLE001
        return None


def notify_gallery_ready(
    name: str,
    mobile: str | None,
    email: str | None,
    gallery_url: str,
    course_name: str | None = None,
) -> None:
    """The gallery is live and the golfer can go and watch it.

    This used to be a single sentence, sent identically to the phone and
    the inbox -- which made the email read like a text message that had
    wandered into the wrong medium. It is a real email now, in the same
    shape as the registration one, and the text stays a text.
    """
    course = (course_name or "").strip()
    at_course = f" from {course}" if course else ""

    lines = [
        f"{name},",
        "",
        f"Your videos{at_course} are ready.",
        "",
        "Check out your shots in your gallery here:",
        gallery_url,
        "",
        "Every par-3 tee shot we recorded is waiting for you there, "
        "ready to watch, download, or share.",
        "",
        "Thanks for playing — we hope to see you out there again soon.",
    ]
    send_email(
        email,
        f"{name}, your GolfReelz videos{at_course} are ready",
        "\n".join(lines),
    )
    send_sms(mobile, (
        f"GolfReelz: your videos{at_course} are ready, {name}. Check out "
        f"your shots in your gallery: {gallery_url}"))


def claim_url_for(gallery_token: str | None) -> str:
    """Where a confirmed ace goes to claim the prize."""
    if not gallery_token:
        return ""
    return f"{settings.app_base_url}/claim/{gallery_token}"


def notify_hio_confirmed(
    name: str,
    mobile: str | None,
    email: str | None,
    gallery_url: str,
    course_name: str | None = None,
    hole_number: int | None = None,
    claim_url: str | None = None,
    prize_label: str | None = None,
) -> None:
    """The ace is verified and there is a prize to collect.

    This was one line reused verbatim as both the text and the email,
    written to fit 160 characters -- so the biggest moment we ever report
    arrived as the shortest thing we ever sent. It is a real email now.

    The claim link is the point of it. A golfer told they have won and
    then handed a gallery link has to work out for themselves what
    happens next; one instruction, stated plainly, is the difference
    between a prize claimed and a prize wondered about.
    """
    course = (course_name or "").strip()
    at_course = f" at {course}" if course else ""
    hole = f" on hole {hole_number}" if hole_number else ""

    lines = [
        f"{name},",
        "",
        f"Congratulations — your hole-in-one{at_course}{hole} has been "
        f"confirmed. Our team has reviewed the footage, and the shot is "
        f"official.",
        "",
        "It is a rare thing to do and a rarer thing to have on video. "
        "Yours is waiting in your gallery, ready to watch and share:",
        gallery_url,
    ]

    claim_url = (claim_url or "").strip()
    prize = (
        prize_label if prize_label is not None else settings.hio_prize_label
    ).strip()
    # An unset prize falls back to naming no figure at all. Better a
    # vaguer sentence than one that promises an empty amount.
    prize_line = f"There is more. Your prize: {prize}." if prize else \
        "There is more — you have a prize waiting."

    if claim_url:
        lines += [
            "",
            prize_line,
            "",
            "Claim it here:",
            claim_url,
            "",
            "The form takes a minute — confirm how to reach you and we "
            "will be in touch to arrange everything.",
        ]
    else:
        # No claim link configured: say who acts next rather than
        # implying the golfer should do something we have not enabled.
        lines += [
            "",
            prize_line,
            "",
            "We will be in touch shortly to arrange it.",
        ]

    lines += [
        "",
        "Congratulations again from all of us. Enjoy it.",
        "",
        "The GolfReelz Team",
    ]

    send_email(
        email,
        f"Congratulations {name} — your hole-in-one is confirmed",
        "\n".join(lines),
    )

    if claim_url:
        send_sms(mobile, (
            f"GolfReelz: your hole-in-one{at_course} is confirmed, {name}! "
            f"Your prize: {prize}. Claim it here: {claim_url}" if prize else
            f"GolfReelz: your hole-in-one{at_course} is confirmed, {name}! "
            f"Claim your prize here: {claim_url}"))
    else:
        send_sms(mobile, (
            f"GolfReelz: your hole-in-one{at_course} is confirmed, {name}! "
            f"Watch it here: {gallery_url}"))


def review_url_for(gallery_token: str | None) -> str:
    """Where to send this golfer to leave a review.

    Our own review page by default, keyed by the same token as their
    gallery, so the form already knows who they are and the rating lands
    attached to a real round. `settings.review_url` overrides it when we
    would rather push people to Google or similar -- that link cannot be
    per-golfer, so it is used exactly as given.
    """
    override = (settings.review_url or "").strip()
    if override:
        return override
    if not gallery_token:
        return ""
    return f"{settings.app_base_url}/review/{gallery_token}"


def notify_thanks_for_playing(
    name: str,
    mobile: str | None,
    email: str | None,
    gallery_url: str,
    course_name: str | None = None,
    review_url: str | None = None,
) -> None:
    """The closing note: thanks for playing, and please leave a review.

    Sent after the golfer has had their videos, so it is the last thing
    they hear from us and the only one that asks for something back. The
    review ask is therefore the point of the message, but it is the third
    thing in it rather than the first -- the gratitude has to be real
    before the request is reasonable.

    `review_url` empty drops the ask entirely rather than sending a
    broken link, which leaves a short, honest thank-you. That is a fine
    email to send; a review request pointing nowhere is not.
    """
    course = (course_name or "").strip()
    at_course = f" at {course}" if course else ""

    lines = [
        f"{name},",
        "",
        f"Thank you for playing with GolfReelz{at_course}. It was a "
        f"pleasure to have you out there, and we hope the footage gave "
        f"you something worth keeping.",
        "",
        "Your gallery stays available, so you can revisit, download, or "
        "share your shots whenever you like:",
        gallery_url,
    ]

    review_url = (review_url or "").strip()
    if review_url:
        lines += [
            "",
            "If you have a moment, we would be grateful for your "
            "feedback. A short review helps other golfers find us and "
            "tells us where we can do better:",
            review_url,
        ]

    lines += [
        "",
        "Thank you again for your business. We look forward to "
        "recording your next round.",
        "",
        "The GolfReelz Team",
    ]

    send_email(
        email,
        f"Thank you for playing with GolfReelz{at_course}, {name}",
        "\n".join(lines),
    )

    if review_url:
        send_sms(mobile, (
            f"GolfReelz: thanks for playing{at_course}, {name}. If you "
            f"have a moment, we'd appreciate a review: {review_url}"))
    else:
        send_sms(mobile, (
            f"GolfReelz: thanks for playing{at_course}, {name}. Your "
            f"gallery stays up here: {gallery_url}"))


def _contest_win_email(
    *,
    name: str,
    mobile: str | None,
    email: str | None,
    subject: str,
    opening: str,
    flourish: str,
    gallery_url: str,
    prize: str,
    claim_url: str,
    sms_lead: str,
) -> None:
    """The shared body of the three contest-win emails.

    They are the same message with different news at the top: you won,
    here is the shot, here is the prize, here is how to collect. Writing
    that three times invites the three to drift apart -- and the one that
    drifts is the one nobody re-reads until a winner complains.

    Only `opening` and `flourish` differ per contest, which is exactly
    the part that should.
    """
    prize = (prize or "").strip()
    # "Your prize: X." rather than "X is waiting for you", because the
    # label is not always a noun -- a free round reads as a phrase, and
    # the older wording produced "a free round on us is waiting for you".
    prize_line = f"There is more. Your prize: {prize}." if prize else \
        "There is more — you have a prize waiting."

    lines = [f"{name},", "", opening, "", flourish, gallery_url]

    if claim_url:
        lines += [
            "",
            prize_line,
            "",
            "Claim it here:",
            claim_url,
            "",
            "The form takes a minute — confirm how to reach you and we "
            "will be in touch to arrange everything.",
        ]
    else:
        lines += [
            "",
            prize_line,
            "",
            "We will be in touch shortly to arrange it.",
        ]

    lines += ["", "Congratulations again from all of us.", "", "The GolfReelz Team"]

    send_email(email, subject, "\n".join(lines))

    if claim_url:
        tail = f"Your prize: {prize}. Claim it here: {claim_url}" if prize else \
               f"Claim your prize here: {claim_url}"
    else:
        tail = f"Your shots: {gallery_url}"
    send_sms(mobile, f"{sms_lead} {tail}")


def notify_ctp_win(
    name: str,
    mobile: str | None,
    email: str | None,
    gallery_url: str,
    course_name: str | None = None,
    hole_number: int | None = None,
    distance_feet: float | None = None,
    claim_url: str | None = None,
    prize_label: str | None = None,
) -> None:
    """Closest to the Pin — a single shot, on a single hole, on the day.

    The distance is the whole story here, so it goes in the subject line
    where the golfer will see it before opening anything.
    """
    course = (course_name or "").strip()
    at_course = f" at {course}" if course else ""
    hole = f" on hole {hole_number}" if hole_number else ""
    dist = f"{distance_feet:g} ft" if distance_feet is not None else None

    subject = (
        f"Congratulations {name} — closest to the pin{at_course}"
        + (f" at {dist}" if dist else "")
    )
    opening = (
        f"Congratulations — you won Closest to the Pin{at_course}{hole}"
        + (f", at {dist} from the cup." if dist else ".")
        + " Nobody got nearer all day."
    )
    _contest_win_email(
        name=name, mobile=mobile, email=email, subject=subject,
        opening=opening,
        flourish=(
            "The shot that did it is in your gallery, ready to watch and "
            "share:"),
        gallery_url=gallery_url,
        prize=(prize_label if prize_label is not None else settings.ctp_prize_label),
        claim_url=(claim_url or "").strip(),
        sms_lead=(
            f"GolfReelz: you won closest to the pin{at_course}"
            + (f" at {dist}" if dist else "") + f", {name}!"),
    )


def notify_shot_of_week(
    name: str,
    mobile: str | None,
    email: str | None,
    gallery_url: str,
    course_name: str | None = None,
    period_label: str | None = None,
    claim_url: str | None = None,
    prize_label: str | None = None,
) -> None:
    """Shot of the Week — chosen by us, out of everything recorded.

    Unlike the other contests this one is a judgement rather than a
    measurement, so the email says who chose it and why that is worth
    something.
    """
    course = (course_name or "").strip()
    at_course = f" at {course}" if course else ""
    period = (period_label or "").strip()
    of_period = f" for {period}" if period else " this week"

    subject = f"Congratulations {name} — your shot is our Shot of the Week"
    opening = (
        f"Congratulations — out of every shot recorded{at_course}"
        f"{of_period}, ours went to yours. Our team picked it as Shot of "
        f"the Week."
    )
    _contest_win_email(
        name=name, mobile=mobile, email=email, subject=subject,
        opening=opening,
        flourish=(
            "It is worth another look. Watch it, download it, send it to "
            "whoever you told about it:"),
        gallery_url=gallery_url,
        prize=(prize_label if prize_label is not None
               else settings.shot_of_week_prize_label),
        claim_url=(claim_url or "").strip(),
        sms_lead=f"GolfReelz: your shot is our Shot of the Week, {name}!",
    )


def notify_monthly_draw(
    name: str,
    mobile: str | None,
    email: str | None,
    gallery_url: str,
    period_label: str | None = None,
    claim_url: str | None = None,
    prize_label: str | None = None,
) -> None:
    """The monthly draw — won by entering, not by playing well.

    Nothing about the golf earned this one, so the email does not pretend
    otherwise. It says they were drawn, which is the honest and the more
    pleasant version: a stroke of luck on top of a round they already had.
    """
    period = (period_label or "").strip()
    for_period = f" for {period}" if period else ""

    subject = f"Congratulations {name} — you won our monthly draw{for_period}"
    opening = (
        f"Congratulations — your name came out of our monthly draw"
        f"{for_period}. Every golfer who played with us was entered, and "
        f"yours is the one that was drawn."
    )
    _contest_win_email(
        name=name, mobile=mobile, email=email, subject=subject,
        opening=opening,
        flourish=(
            "While you are here, your rounds are all still in your "
            "gallery:"),
        gallery_url=gallery_url,
        prize=(prize_label if prize_label is not None
               else settings.monthly_draw_prize_label),
        claim_url=(claim_url or "").strip(),
        sms_lead=f"GolfReelz: you won our monthly draw{for_period}, {name}!",
    )


def _money(cents: int | None) -> str | None:
    """Cents as $20 or $19.50. None when we do not know the amount."""
    if not cents:
        return None
    d = cents / 100.0
    return f"${d:,.0f}" if abs(d - round(d)) < 0.005 else f"${d:,.2f}"


def notify_refund_issued(
    name: str,
    mobile: str | None,
    email: str | None,
    amount_cents: int | None = None,
    course_name: str | None = None,
    reason: str | None = None,
) -> None:
    """We have refunded this golfer.

    Money moved and nothing said so. A refund the customer finds out
    about from their bank statement is a refund they email us about, and
    an unexplained one reads worse than the problem that caused it.

    Deliberately does not apologise twice or explain at length -- the
    refund IS the apology, and a golfer reading this mostly wants the
    amount and the timing.
    """
    course = (course_name or "").strip()
    at_course = f" at {course}" if course else ""
    amount = _money(amount_cents)

    lines = [f"{name},", ""]
    lines.append(
        f"We have refunded your GolfReelz registration{at_course}"
        + (f" — {amount}." if amount else ".")
    )
    if (reason or "").strip():
        lines += ["", reason.strip()]
    lines += [
        "",
        "The money goes back to the card you paid with. Most banks show "
        "it within five to ten business days; it can occasionally take a "
        "little longer, which is the bank's timing rather than ours.",
        "",
        "If anything about this does not look right, reply to this email "
        "and a person will pick it up.",
        "",
        "Thank you for giving us a try — we would be glad to record your "
        "next round.",
        "",
        "The GolfReelz Team",
    ]

    subject = (
        f"Your GolfReelz refund{f' of {amount}' if amount else ''} is on its way"
    )
    send_email(email, subject, "\n".join(lines))
    send_sms(mobile, (
        f"GolfReelz: we've refunded your registration{at_course}"
        + (f" ({amount})" if amount else "")
        + ". It should be back on your card within 5-10 business days."))


def notify_no_clips(
    name: str,
    mobile: str | None,
    email: str | None,
    course_name: str | None = None,
    refunded: bool = False,
    amount_cents: int | None = None,
    reason: str | None = None,
) -> None:
    """We recorded nothing for this golfer, and have to say so.

    The worst outcome in the system and, until now, the only one that
    said nothing at all: gallery-ready fires on the first assigned clip,
    so a golfer with no clips simply never heard from us again after
    registering. Silence after taking someone's money is the version of
    this that costs a customer permanently.

    So it leads with the failure rather than burying it, says plainly
    that it is our fault, and -- when a refund has been issued -- says so
    in the same breath rather than making them ask.
    """
    course = (course_name or "").strip()
    at_course = f" at {course}" if course else ""
    amount = _money(amount_cents)

    lines = [
        f"{name},",
        "",
        f"We have to apologise. We did not manage to record your par-3 "
        f"tee shots{at_course}, so there is nothing in your gallery from "
        f"your round. That is our failure, not anything you did.",
    ]
    if (reason or "").strip():
        lines += ["", reason.strip()]

    if refunded:
        lines += [
            "",
            "We have refunded your registration"
            + (f" — {amount}" if amount else "")
            + ". The money goes back to the card you paid with, and most "
            "banks show it within five to ten business days.",
        ]
    else:
        lines += [
            "",
            "We do not think you should pay for a round we did not "
            "capture. Reply to this email and we will put that right.",
        ]

    lines += [
        "",
        "We would genuinely like another go at this. If you play with us "
        "again, reply and we will make sure someone is watching your "
        "group's cameras that day.",
        "",
        "Sorry again — and thank you for the chance.",
        "",
        "The GolfReelz Team",
    ]

    send_email(
        email,
        f"{name}, we did not get your shots{at_course} — and we are sorry",
        "\n".join(lines),
    )
    send_sms(mobile, (
        f"GolfReelz: we're sorry — we didn't manage to record your shots"
        f"{at_course}. "
        + ("We've refunded your registration. " if refunded
           else "Reply and we'll put it right. ")
        + "Details are in your email."))
