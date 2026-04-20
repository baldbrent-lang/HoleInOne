# GolfReelz

Web platform that connects golfer self-registration to automatically processed
shot-tracer videos, delivering a per-golfer video gallery after their round.

Repo name: `HoleInOne`. Product name: **GolfReelz**.

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

### Option A — Replit (single Repl)

1. Import this repo into Replit. `.replit` + `replit.nix` are already in
   the project, so Replit will install Python 3.12 and Node 20.
2. Click **Run**. `start.sh` will:
   - `pip install` backend deps into `backend/.venv`
   - `npm install` + `npm run build` the frontend into `frontend/dist`
   - boot Uvicorn on `$PORT` (default 8000, mapped to port 80 publicly)
   - serve the API **and** the built SPA on the same origin
3. Open the webview URL. The SPA lives at `/`, the API at `/api/...`,
   OpenAPI docs at `/docs`.

Set Secrets (Replit → Tools → Secrets) for any external integrations
you want to light up:

- `DATABASE_URL` — e.g. Replit Postgres, Neon, Supabase. Omit to use
  SQLite at `backend/golfreelz.db`.
- `APP_BASE_URL` — your Repl's public URL
  (`https://<repl-name>.<user>.repl.co`)
- `ADMIN_PASSWORD` (default `Baldy123`)
- `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`
- `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`
- `SENDGRID_API_KEY`, `SENDGRID_FROM_EMAIL`
- `SHOT_TRACER_WEBHOOK_SECRET`

### Option B — docker compose

```bash
docker compose up --build
```

- Backend: http://localhost:8000 (docs at `/docs`)
- Frontend: http://localhost:5173

### Option C — native

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# defaults to SQLite at ./golfreelz.db if DATABASE_URL unset

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
| `/admin/participants`      | Filterable participants view         |
| `/admin/review`            | Hole-in-one verification queue       |

## Environment variables

Place in `backend/.env` (see `backend/.env.example`):

```
DATABASE_URL=postgresql+psycopg://golfreelz:golfreelz@db:5432/golfreelz
STRIPE_SECRET_KEY=            # empty -> test/mock mode
STRIPE_WEBHOOK_SECRET=
TWILIO_ACCOUNT_SID=           # empty -> no-op
TWILIO_AUTH_TOKEN=
TWILIO_FROM_NUMBER=
SENDGRID_API_KEY=             # empty -> no-op
SENDGRID_FROM_EMAIL=
SHOT_TRACER_WEBHOOK_SECRET=
APP_BASE_URL=http://localhost:5173
ADMIN_PASSWORD=Baldy123
```

## Explicitly out of scope for V0

- Real ForeUP / Lightspeed tee-sheet API wiring (mock endpoint only)
- Production auth for admin (API key header only)
- Pace-of-play heuristic tuning per course
- Mobile apps, i18n, live streaming, refund automation
- Deploy configs (Vercel / Railway / Cloudflare)
