#!/usr/bin/env bash
# Build the frontend and run the dashboard on one port (default 8100).
set -euo pipefail
cd "$(dirname "$0")"

# Python deps (venv-local so nothing leaks into the system install)
if [ ! -d backend/.venv ]; then
  python3 -m venv backend/.venv
fi
backend/.venv/bin/pip install -q -r backend/requirements.txt

# Frontend build
(cd frontend && npm install --no-audit --no-fund && npm run build)

PORT="${PORT:-8100}"
echo "Dashboard running on http://localhost:${PORT}"
exec backend/.venv/bin/uvicorn main:app --app-dir backend --host 0.0.0.0 --port "$PORT"
