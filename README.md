# Par One

Web platform that connects golfer self-registration to automatically processed
shot-tracer videos, delivering a per-golfer video gallery after their round.

Repo name: `HoleInOne`. Product name: **Par One**.

## Status

V0 scaffold. End-to-end flows wired with mocked external integrations
(Stripe, Twilio, SendGrid, Shot Tracer, ForeUP / Lightspeed).

## Layout

```
backend/          FastAPI + SQLAlchemy + PostgreSQL (SQLite fallback for dev)
frontend/         React + Vite single-page app
docker-compose.yml  Postgres + backend + frontend
```

## Running locally

### Option A — docker compose

```bash
docker compose up --build
```

- Backend: http://localhost:8000 (docs at `/docs`)
- Frontend: http://localhost:5173

### Option B — native

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# defaults to SQLite at ./parone.db if DATABASE_URL unset

# Frontend (separate shell)
cd frontend
npm install
npm run dev
```

## Key routes (frontend)

| Route                      | Purpose                              |
| -------------------------- | ------------------------------------ |
| `/r/:courseToken`          | Mobile registration (QR landing)     |
| `/confirm/:participantId`  | Post-registration confirmation       |
| `/g/:galleryToken`         | Golfer video gallery                 |
| `/admin`                   | Admin dashboard                      |
| `/admin/review`            | Hole-in-one verification queue       |

## Environment variables

Place in `backend/.env` (see `backend/.env.example`):

```
DATABASE_URL=postgresql+psycopg://parone:parone@db:5432/parone
STRIPE_SECRET_KEY=            # empty -> test/mock mode
STRIPE_WEBHOOK_SECRET=
TWILIO_ACCOUNT_SID=           # empty -> no-op
TWILIO_AUTH_TOKEN=
TWILIO_FROM_NUMBER=
SENDGRID_API_KEY=             # empty -> no-op
SENDGRID_FROM_EMAIL=
SHOT_TRACER_WEBHOOK_SECRET=
APP_BASE_URL=http://localhost:5173
ADMIN_API_KEY=dev-admin-key
```

## Explicitly out of scope for V0

- Real ForeUP / Lightspeed tee-sheet API wiring (mock endpoint only)
- Production auth for admin (API key header only)
- Pace-of-play heuristic tuning per course
- Mobile apps, i18n, live streaming, refund automation
- Deploy configs (Vercel / Railway / Cloudflare)
