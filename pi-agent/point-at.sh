#!/usr/bin/env bash
# Point this Pi's agent at a different backend, and restart it.
#
#   sudo /opt/golfreelz-agent/point-at.sh dev
#   sudo /opt/golfreelz-agent/point-at.sh prod
#   sudo /opt/golfreelz-agent/point-at.sh https://something-else.replit.dev
#   sudo /opt/golfreelz-agent/point-at.sh dev cam_abc123     # + a new token
#
# WHY THIS EXISTS. Publishing to production to test a backend change is a
# slow loop, and worse, it does it on the machine the course is relying
# on. Pointing the Pis at the dev workspace instead makes the loop a
# restart. Editing YAML by hand over a laggy SSH session every time is
# how you end up with a Pi silently pointed at nothing.
#
# The auth_token is a per-camera value that lives in the BACKEND'S
# DATABASE. If dev and production share a database the tokens match and
# only the URL changes; if they do not, pass the dev camera's token as
# the second argument and this stores it alongside. Either way the
# previous config is kept as config.yaml.bak so a bad switch is one
# `cp` from undone.
set -euo pipefail

CONFIG="${CONFIG:-/opt/golfreelz-agent/config.yaml}"
SERVICE="${SERVICE:-golfreelz-agent}"

# Edit these two to match your deployment.
PROD_URL="${GOLFREELZ_PROD_URL:-https://golf-reelz.replit.app}"
DEV_URL="${GOLFREELZ_DEV_URL:-}"

usage() {
  cat <<EOF
usage: $0 <dev|prod|https://...> [auth_token]

  dev     -> \$GOLFREELZ_DEV_URL  (${DEV_URL:-NOT SET — pass a URL instead})
  prod    -> $PROD_URL

Set GOLFREELZ_DEV_URL in this script, or pass the URL directly.
EOF
  exit 2
}

[[ $# -ge 1 ]] || usage
TARGET="$1"
TOKEN="${2:-}"

case "$TARGET" in
  dev)
    [[ -n "$DEV_URL" ]] || { echo "!! GOLFREELZ_DEV_URL is not set"; usage; }
    URL="$DEV_URL" ;;
  prod|production)
    URL="$PROD_URL" ;;
  https://*|http://*)
    URL="$TARGET" ;;
  *)
    usage ;;
esac

[[ -f "$CONFIG" ]] || { echo "!! no config at $CONFIG"; exit 1; }

CURRENT="$(grep -E '^backend_url:' "$CONFIG" | head -1 | sed 's/^backend_url:[[:space:]]*//' || true)"
echo "==> was: ${CURRENT:-<unset>}"
echo "==> now: \"$URL\""

cp -a "$CONFIG" "$CONFIG.bak"

# Rewrite in place. `|` as the delimiter because the value is a URL.
sed -i "s|^backend_url:.*|backend_url: \"$URL\"|" "$CONFIG"
if [[ -n "$TOKEN" ]]; then
  sed -i "s|^auth_token:.*|auth_token: \"$TOKEN\"|" "$CONFIG"
  echo "==> auth_token updated"
fi

# Fail loudly rather than restarting into a broken config.
if ! grep -qE "^backend_url: \"$URL\"" "$CONFIG"; then
  echo "!! rewrite did not take — restoring $CONFIG.bak"
  cp -a "$CONFIG.bak" "$CONFIG"
  exit 1
fi

echo "==> restarting $SERVICE"
systemctl restart "$SERVICE"
sleep 2
systemctl is-active "$SERVICE" || {
  echo "!! the agent did not come back — check: journalctl -u $SERVICE -n 50"
  exit 1
}

echo
echo "==> pointed at $URL"
echo "    previous config kept at $CONFIG.bak"
echo "    watch it: journalctl -u $SERVICE -f"
