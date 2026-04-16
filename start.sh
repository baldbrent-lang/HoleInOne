#!/usr/bin/env bash
# Replit entrypoint. Installs deps on first run, builds the frontend,
# then starts FastAPI serving the API + the built SPA on a single port.
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d backend/.venv ] || [ ! -d frontend/node_modules ] || [ ! -d frontend/dist ]; then
  bash build.sh
fi

# shellcheck disable=SC1091
source backend/.venv/bin/activate

: "${PORT:=8000}"

export SERVE_FRONTEND=1

cd backend
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
