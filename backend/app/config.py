from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./parone.db"
    app_base_url: str = "http://localhost:5173"
    admin_api_key: str = "dev-admin-key"

    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    registration_price_cents: int = 2000

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""

    sendgrid_api_key: str = ""
    sendgrid_from_email: str = "noreply@parone.test"

    shot_tracer_webhook_secret: str = ""

    default_minutes_per_hole: int = 14


settings = Settings()
