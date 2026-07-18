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

    # Tee-only fallback. A paired camera event needs BOTH the tee and green
    # halves to produce. If the green half never arrives (dead cellular,
    # unplugged, weak signal), the event stalls forever at "tee_uploaded"
    # and you lose the whole shot. After this many seconds with only the
    # tee half, produce a TEE-ONLY clip so a green dropout costs you the
    # green angle, not the shot. If green shows up later, the event
    # re-produces with both. 0 disables the fallback.
    camera_tee_only_fallback_seconds: int = 180

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

    # "Scan for non-golf videos" — a dev course-testing tool that scans the
    # Production queue's clips and pre-checks the ones that don't look like a
    # real golf shot (no person, or an indoor/grass-less scene), so garbage
    # can be bulk-deleted. Only CHECKS boxes; never deletes. Set env
    # SCAN_NON_GOLF_ENABLED=1 on the DEV deployment to show the button.
    scan_non_golf_enabled: bool = False

    # "Produce (debug)" — a dev course-testing tool. Alongside a normal
    # produce, it runs a per-swing diagnostic that shows each step's result,
    # the classical-CV tracer's motion heatmap + whether it found the ball,
    # AND runs the AI tracer on the same swings so the two can be compared.
    # The AI half needs ANTHROPIC_API_KEY set (else it reports unavailable).
    # Set env PRODUCE_DEBUG_ENABLED=1 on the DEV deployment to show the button.
    produce_debug_enabled: bool = False

    # Swing detector for the produce pipeline. "pose" (default) = the
    # MediaPipe wrist-speed + back-bend detector; "motion" = the old
    # audio/motion burst detector. Pose needs mediapipe installed — when it
    # isn't (e.g. prod, which stays on OpenCV 4.10), the pipeline falls back
    # to motion automatically, so this default is safe everywhere: dev (with
    # mediapipe) uses pose; prod uses motion. Set SWING_DETECTOR=motion to
    # force the old detector.
    swing_detector: str = "pose"  # "pose" | "motion"
    # Produced-clip window around the detected swing (pose mode). The pose
    # spike sits at the downswing, so 2s before / 7s after captures setup
    # through ball flight + landing.
    pose_clip_before_sec: float = 2.0
    pose_clip_after_sec: float = 7.0

    # MOG2 swing confirmation for the produce pipeline (pose mode). Each
    # pose-detected swing gets a short background-subtraction pass around
    # its peak looking for a ball-flight launch chain in the transient
    # motion — physical proof a ball actually flew. Swings without one are
    # dropped from production (tee-planting, waggles). Fail-safe: if the
    # check rejects EVERY swing, all are kept (scene may be blind to the
    # ball, e.g. distant course framing). Env: SWING_HEAT_CHECK_ENABLED=0.
    swing_heat_check_enabled: bool = True
    # AI per-frame ball track (12 Opus vision calls/swing — the single
    # biggest cost item). OFF by default: the pixel launch tracker +
    # rest-lock supply the flight now. Set AI_BALL_TRACK_ENABLED=1 to
    # bring it back.
    ai_ball_track_enabled: bool = False

    # Ball-flight tracer engine for produced clips. "ai" (default) uses the
    # Claude vision tracer; "classical" uses the motion/parabola CV tracer.
    # AI needs ANTHROPIC_API_KEY — without a key (or on any AI failure) the
    # pipeline automatically falls back to the classical tracer, so a produce
    # never fails for lack of a key. Set TRACER_ENGINE=classical to force it.
    tracer_engine: str = "ai"  # "ai" | "classical"

    # Appearance matching
    embedding_provider: str = "stub"  # "stub" | "clip" | "replicate" | "fal"
    embedding_min_margin: float = 0.05  # min cosine margin between top-1 and top-2
    upload_dir: str = "uploads"


settings = Settings()
