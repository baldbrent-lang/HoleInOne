# GolfReelz — one image: the built SPA plus the FastAPI app that serves it.
#
# Render's native Python runtime cannot apt-install anything, and this app
# shells out to ffmpeg for every produced clip, so Docker is required rather
# than preferred.

# ---------- stage 1: build the SPA ----------
FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-fund --no-audit
COPY frontend/ ./
RUN npm run build

# ---------- stage 2: runtime ----------
# Pinned to bookworm rather than plain -slim: the OpenCV and glib package
# names differ across Debian releases, and a base image that silently moves
# to the next one would fail at build time for no reason anybody would guess.
FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg          the audio swing detection and every clip cut/encode
# libgl1, libglib OpenCV / MediaPipe runtime deps
# postgresql-17   psql + pg_dump/pg_restore for the migration, from PGDG
#                 because Debian's own client is too old to dump a modern
#                 server and pg_dump refuses rather than degrading
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg \
 && install -d /usr/share/postgresql-common/pgdg \
 && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc \
      -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
 && echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt bookworm-pgdg main" \
      > /etc/apt/sources.list.d/pgdg.list \
 && apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libgl1 libglib2.0-0 postgresql-client-17 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ backend/
# main.py resolves FRONTEND_DIST as parents[2]/frontend/dist, which from
# /app/backend/app/main.py is /app/frontend/dist — exactly here.
COPY --from=web /web/dist frontend/dist

ENV SERVE_FRONTEND=1
WORKDIR /app/backend
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
