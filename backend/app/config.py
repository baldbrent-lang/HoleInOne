import os

from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_app_base_url() -> str:
    # Explicit override wins — set APP_BASE_URL in production env vars.
    explicit = os.environ.get("APP_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    # Replit sets REPLIT_DEV_DOMAIN automatically in development.
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

    # Appearance matching
    embedding_provider: str = "stub"  # "stub" | "clip" | "replicate" | "fal"
    embedding_min_margin: float = 0.05  # min cosine margin between top-1 and top-2
    upload_dir: str = "uploads"


settings = Settings()
