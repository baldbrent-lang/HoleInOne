#!/usr/bin/env bash
# Replit entrypoint. Installs deps on first run, rebuilds the frontend on
# every start (cheap ~1s vite build) so pulled code changes are picked up,
# then starts FastAPI serving the API + SPA on a single port.
set -euo pipefail

cd "$(dirname "$0")"

# First-time deps install
if [ ! -d backend/.venv ] || [ ! -d frontend/node_modules ]; then
  bash build.sh
fi

# shellcheck disable=SC1091
source backend/.venv/bin/activate

# Pull in any new Python deps that landed via git pull (idempotent + fast
# when nothing changed). Without this, new packages in requirements.txt
# don't get installed until someone wipes backend/.venv.
export PIP_USER=0
pip install --disable-pip-version-check --quiet -r backend/requirements.txt

# Always rebuild the SPA so new JSX is served after `git pull`.
echo "==> Rebuilding frontend"
(cd frontend && npm run build --silent)

: "${PORT:=8000}"

export SERVE_FRONTEND=1

cd backend
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
