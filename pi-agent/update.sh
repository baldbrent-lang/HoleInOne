#!/usr/bin/env bash
# Pull the latest agent code from GitHub and restart the service.
# Replaces the manual "stop / clone / cp / chown / start" dance with a
# single command. Touches only the Python files under agent/ and the
# pip requirements — config.yaml, the systemd unit, and any other
# local edits under /opt/golfreelz-agent are left alone.
#
# Usage:
#   sudo /opt/golfreelz-agent/update.sh                          # pulls main
#   sudo /opt/golfreelz-agent/update.sh <branch-or-tag>          # pulls a branch
#
# Examples:
#   sudo /opt/golfreelz-agent/update.sh
#   sudo /opt/golfreelz-agent/update.sh claude/connect-hotspot-cameras-h99lH

set -euo pipefail

BRANCH="${1:-main}"
INSTALL_DIR="/opt/golfreelz-agent"
SERVICE_USER="golfreelz"
SERVICE_NAME="golfreelz-agent"
REPO_URL="https://github.com/baldbrent-lang/HoleInOne.git"

if [[ "$EUID" -ne 0 ]]; then
  echo "must be run as root (use sudo)" >&2
  exit 1
fi

# Clone into a tmp dir FIRST — if the network is down or the branch
# name is wrong, we bail without ever touching the running service.
TMP_DIR="$(mktemp -d -t golfreelz-update-XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "==> branch: $BRANCH"
echo "==> cloning $REPO_URL"
git clone --quiet --depth 1 --branch "$BRANCH" "$REPO_URL" "$TMP_DIR/repo"

echo "==> stopping $SERVICE_NAME"
systemctl stop "$SERVICE_NAME"

echo "==> copying agent/*.py into $INSTALL_DIR/agent/"
install -m 644 -o "$SERVICE_USER" -g "$SERVICE_USER" \
  "$TMP_DIR/repo/pi-agent/agent/"*.py \
  "$INSTALL_DIR/agent/"

if compgen -G "$TMP_DIR/repo/pi-agent/models/*" >/dev/null; then
  echo "==> copying models/ (YOLO ONNX) into $INSTALL_DIR/models/"
  install -d -o "$SERVICE_USER" -g "$SERVICE_USER" "$INSTALL_DIR/models"
  install -m 644 -o "$SERVICE_USER" -g "$SERVICE_USER" \
    "$TMP_DIR/repo/pi-agent/models/"* \
    "$INSTALL_DIR/models/"
fi

echo "==> refreshing top-level files (agent, update.sh, point-at.sh, README)"
install -m 755 -o "$SERVICE_USER" -g "$SERVICE_USER" \
  "$TMP_DIR/repo/pi-agent/golfreelz_agent.py" \
  "$INSTALL_DIR/golfreelz_agent.py"
install -m 755 -o "$SERVICE_USER" -g "$SERVICE_USER" \
  "$TMP_DIR/repo/pi-agent/update.sh" \
  "$INSTALL_DIR/update.sh"
install -m 644 -o "$SERVICE_USER" -g "$SERVICE_USER" \
  "$TMP_DIR/repo/pi-agent/README.md" \
  "$INSTALL_DIR/README.md"
install -m 755 -o "$SERVICE_USER" -g "$SERVICE_USER" \
  "$TMP_DIR/repo/pi-agent/install.sh" \
  "$INSTALL_DIR/install.sh"
install -m 644 -o "$SERVICE_USER" -g "$SERVICE_USER" \
  "$TMP_DIR/repo/pi-agent/config.example.yaml" \
  "$INSTALL_DIR/config.example.yaml"
install -m 755 -o "$SERVICE_USER" -g "$SERVICE_USER" \
  "$TMP_DIR/repo/pi-agent/point-at.sh" \
  "$INSTALL_DIR/point-at.sh"

echo "==> pip: applying requirements.txt (failures non-fatal — see warnings)"
if ! "$INSTALL_DIR/venv/bin/pip" install --quiet \
    -r "$TMP_DIR/repo/pi-agent/requirements.txt"; then
  echo
  echo "    !! pip reported errors. The most common cause on a Pi is"
  echo "    !! mediapipe not having a wheel for this Python version —"
  echo "    !! the agent has a motion-density fallback for that."
  echo "    !! Continuing with the existing venv contents."
  echo
fi

echo "==> starting $SERVICE_NAME"
systemctl start "$SERVICE_NAME"

echo
echo "==> done. Tail logs with:"
echo "    sudo journalctl -u $SERVICE_NAME -f"
