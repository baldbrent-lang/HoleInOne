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

    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    registration_price_cents: int = 2000

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""

    sendgrid_api_key: str = ""
    sendgrid_from_email: str = "noreply@golfreelz.test"

    shot_tracer_webhook_secret: str = ""

    default_minutes_per_hole: int = 14
    # How long after a tee time we'll still try to match a clip to that
    # group. Real-world rounds run long, golfers play out of order, and
    # camera/upload pipelines back-fill late, so keep this generous.
    match_window_minutes: int = 300  # 5 hours

    # Appearance matching
    embedding_provider: str = "stub"  # "stub" | "clip" | "replicate" | "fal"
    embedding_min_margin: float = 0.05  # min cosine margin between top-1 and top-2
    upload_dir: str = "uploads"


settings = Settings()
