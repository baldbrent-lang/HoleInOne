#!/usr/bin/env bash
# GolfReelz Pi capture agent installer.
#
# Run this on a fresh Raspberry Pi OS Lite install (Debian Bookworm
# or later) as root, with this directory already copied to
# /opt/golfreelz-agent. The script is idempotent — safe to re-run
# after editing the agent code.
#
#   sudo bash /opt/golfreelz-agent/install.sh
#
# What it does:
#   1. Installs system packages (python3, venv, ffmpeg, libs that
#      OpenCV / MediaPipe link against).
#   2. Creates a 'golfreelz' service user (so the agent isn't root).
#   3. Sets up a Python venv at /opt/golfreelz-agent/venv and pip
#      installs the requirements.
#   4. Reminds you to drop your config.yaml in place.
#   5. Installs the systemd unit and enables it on boot.

set -euo pipefail

INSTALL_DIR="/opt/golfreelz-agent"
SERVICE_USER="golfreelz"
SERVICE_NAME="golfreelz-agent"

if [[ "$EUID" -ne 0 ]]; then
  echo "must be run as root (use sudo)" >&2
  exit 1
fi

cd "$INSTALL_DIR"

echo "==> apt: installing system deps"
apt-get update
apt-get install -y --no-install-recommends \
  python3 python3-venv python3-pip python3-dev \
  ffmpeg \
  alsa-utils \
  libatlas-base-dev libopenblas-dev \
  libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
  v4l-utils

echo "==> user: creating $SERVICE_USER if needed"
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
  useradd --system --create-home --shell /usr/sbin/nologin "$SERVICE_USER"
fi
# Service user needs access to /dev/video* via the 'video' group.
usermod -aG video "$SERVICE_USER"

echo "==> venv: creating $INSTALL_DIR/venv if needed"
if [[ ! -d "$INSTALL_DIR/venv" ]]; then
  python3 -m venv "$INSTALL_DIR/venv"
fi
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip wheel
"$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"

echo "==> ownership: $INSTALL_DIR -> $SERVICE_USER:$SERVICE_USER"
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"

if [[ ! -f "$INSTALL_DIR/config.yaml" ]]; then
  echo
  echo "!! NOTE: $INSTALL_DIR/config.yaml does not exist yet."
  echo "!! Copy config.example.yaml to config.yaml and edit the"
  echo "!! auth_token + backend_url before starting the service."
  echo
fi

echo "==> systemd: installing unit"
install -m 644 "$INSTALL_DIR/systemd/$SERVICE_NAME.service" \
  "/etc/systemd/system/$SERVICE_NAME.service"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

echo
echo "==> done. Next steps:"
echo "   1. Edit $INSTALL_DIR/config.yaml (auth_token, backend_url, ROI)"
echo "   2. sudo systemctl start $SERVICE_NAME"
echo "   3. sudo journalctl -u $SERVICE_NAME -f   # follow logs"
