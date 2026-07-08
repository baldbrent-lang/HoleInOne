import os

from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_app_base_url() -> str:
    # Replit sets REPLIT_DEV_DOMAIN automatically — use it so QR codes and
    # outbound links point at the public URL without manual config.
    replit_domain = os.environ.get("REPLIT_DEV_DOMAIN")
    if replit_domain:
        return f"https://{replit_domain}"
    return "http://localhost:5173"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./golfreelz.db"
    app_base_url: str = _default_app_base_url()
    admin_password: str = "Baldy123"
    # User-auth secret. Override in prod via env. Tokens are JWT HS256.
    jwt_secret: str = "dev-jwt-secret-change-in-prod"
    jwt_ttl_days: int = 30

    stripe_secret_key: str = ""
    stripe_publishable_key: str = ""
    stripe_webhook_secret: str = ""
    registration_price_cents: int = 2000

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""

    sendgrid_api_key: str = ""
    sendgrid_from_email: str = "noreply@golfreelz.test"

    # SMTP (Gmail, Outlook, Yahoo, etc). Used in preference to SendGrid
    # when set. Gmail wants smtp.gmail.com:587 + a 16-char App Password.
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""  # defaults to smtp_user when empty
    smtp_use_ssl: bool = False  # True for port 465; False for 587 STARTTLS

    shot_tracer_webhook_secret: str = ""

    default_minutes_per_hole: int = 14
    # How long after a golfer registers we'll match clips back to them.
    # Window starts at registration time (Participant.created_at) and ends
    # match_window_minutes later. Clips captured before registration are
    # never matched.
    match_window_minutes: int = 540  # 9 hours

    # Camera-event shot production.
    # When True (course-testing phase), produce a clip for EVERY detected
    # swing and tag each with the ball-departure verdict, instead of
    # silently dropping swings where the ball wasn't confirmed leaving the
    # tee. Prevents a real shot from vanishing when the camera can't clearly
    # see the ball. Flip to False (env: CAMERA_PRODUCE_UNCONFIRMED_SHOTS=0)
    # once the ball check is tuned and trustworthy, to keep non-shots out of
    # Production.
    camera_produce_unconfirmed_shots: bool = True

    # Garbage weeding. Even in permissive keep-everything mode (above), drop
    # clips where there was CLEARLY no golf: a motion burst fired but no golf
    # ball was ever on the tee (verdict "no_ball") — someone walking through
    # frame, kitchen/indoor motion, a dog, etc. This is a narrower cut than
    # camera_produce_unconfirmed_shots=False: it still keeps practice swings
    # (ball present) and shots where the camera just couldn't see the ball
    # ("uncertain"), so a real shot is never dropped on a "ball not visible"
    # glitch — only the no-ball-at-all garbage is removed. Set env
    # CAMERA_DROP_GARBAGE_CLIPS=0 to keep everything (raw course-test mode).
    camera_drop_garbage_clips: bool = True

    # "Pull new from prod" — the DEV backend fetches new camera clips from a
    # source (prod) backend and imports them, replacing the manual mirror
    # script. Set these on the DEV deployment only.
    #   MIRROR_SOURCE_URL       prod base URL to pull from
    #   MIRROR_SOURCE_PASSWORD  prod admin password (falls back to
    #                           admin_password if blank — fine when both are
    #                           the same)
    #   MIRROR_COURSE_ID        which local course to attach imports to
    #                           (0 = feature off / button hidden)
    mirror_source_url: str = "https://golf-reelz.replit.app"
    mirror_source_password: str = ""
    mirror_course_id: int = 0

    # Appearance matching
    embedding_provider: str = "stub"  # "stub" | "clip" | "replicate" | "fal"
    embedding_min_margin: float = 0.05  # min cosine margin between top-1 and top-2
    upload_dir: str = "uploads"


settings = Settings()
