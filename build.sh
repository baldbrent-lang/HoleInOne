#!/usr/bin/env bash
# One-time build: install deps and compile the React frontend into
# frontend/dist so FastAPI can serve it as static files.
set -euo pipefail

cd "$(dirname "$0")"

echo "==> Installing backend deps"
python -m venv backend/.venv
# shellcheck disable=SC1091
source backend/.venv/bin/activate
pip install --disable-pip-version-check --quiet -r backend/requirements.txt

echo "==> Installing frontend deps"
cd frontend
npm install --no-fund --no-audit --loglevel=error

echo "==> Building frontend"
npm run build

echo "==> Build complete"
