# Personal Dashboard

A standalone dashboard app — completely separate from GolfReelz, it just
lives in this repo's `dashboard/` folder. One small FastAPI backend serves
the API and the built React frontend on a single port.

## Features

- **World clocks** (green bar on top): Chicago, New York, Phoenix, London,
  Sydney — live, ticking every second.
- **To-do list**: add / check off / delete; persisted in a local SQLite
  file (`backend/dashboard.db`).
- **AP News headlines** in three columns: Business, Macroeconomics (AP's
  Economy hub) and Politics. Scraped from apnews.com hub pages, with a
  Google News fallback that still returns AP stories. Refreshes every
  10 minutes.
- **Markets** from Yahoo Finance: Nvidia, QQQM, S&P 500, QBE.AX (shown in
  A$ with the USD equivalent alongside) and RTX — current price, day and
  week change, and an interactive one-week chart per symbol. Refreshes
  every 60 seconds.
- Installable on a phone (add to home screen — it has a PWA manifest and
  icons, and the layout is responsive).

## Run it

```bash
cd dashboard
./start.sh          # installs deps, builds the frontend, serves on :8100
```

Then open http://localhost:8100. Set `PORT` to use a different port.

## Development (hot reload)

```bash
# Terminal 1 — API on :8100
cd dashboard/backend
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn main:app --reload --port 8100

# Terminal 2 — frontend on :5180 (proxies /api to :8100)
cd dashboard/frontend
npm install
npm run dev
```

## API

| Route              | Purpose                                   |
| ------------------ | ----------------------------------------- |
| `GET /api/todos`   | List to-dos (`POST` to add, `PATCH`/`DELETE /api/todos/{id}`) |
| `GET /api/news`    | AP headlines, three sections, cached 10 min |
| `GET /api/quotes`  | Yahoo Finance prices + weekly series, cached 90 s |
| `GET /api/health`  | Liveness check                            |

No auth — it's a personal, single-user app. Put it behind a login or a
private network if you deploy it somewhere public.
