#!/usr/bin/env python3
"""Pre-flight check before hitting test swings.

Verifies the backend is healthy and both cameras (tee + green) are online,
enabled, triggering, and recently seen — so you don't walk out to the tee
box and find a camera was asleep. Read-only; changes nothing.

Standard library only.

Usage:
    python3 scripts/preflight.py
    BASE_URL=https://<dev>.spock.replit.dev python3 scripts/preflight.py

Environment:
    BASE_URL        backend to check (default https://holeinone-s6qm.onrender.com)
    ADMIN_PASSWORD  admin password (default Baldy123)
    STALE_SECONDS   heartbeat older than this = OFFLINE (default 150)
"""

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

BASE = os.environ.get("BASE_URL", "https://holeinone-s6qm.onrender.com").rstrip("/")
PW = os.environ.get("ADMIN_PASSWORD", "Baldy123")
STALE = int(os.environ.get("STALE_SECONDS", "150"))

OK, WARN, BAD = "  ✅", "  ⚠️ ", "  ❌"
problems: list[str] = []


def _get(path: str):
    req = urllib.request.Request(BASE + path, headers={"X-Admin-Password": PW})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, json.loads(r.read())


def _age(iso: str | None):
    """Seconds since an ISO timestamp, or None if unparseable/missing."""
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except ValueError:
        return None


def _fmt_age(secs):
    if secs is None:
        return "never"
    if secs < 90:
        return f"{int(secs)}s ago"
    if secs < 5400:
        return f"{int(secs / 60)}m ago"
    return f"{secs / 3600:.1f}h ago"


print(f"\nPre-flight — {BASE}\n" + "=" * 52)

# 1. Backend health -----------------------------------------------------
try:
    st, body = _get("/health")
    if st == 200 and body.get("ok"):
        print(f"{OK} backend healthy ({body.get('app')})")
    else:
        print(f"{BAD} backend /health returned {st}: {body}")
        problems.append("backend health")
except Exception as e:  # noqa: BLE001
    print(f"{BAD} backend unreachable: {e}")
    print("\nStop here — nothing else can work until the backend is up.")
    sys.exit(1)

# 2. Courses (the endpoint that was 500ing) -----------------------------
try:
    st, courses = _get("/api/admin/courses")
    print(f"{OK} courses endpoint OK ({len(courses)} course(s))")
except Exception as e:  # noqa: BLE001
    print(f"{BAD} courses endpoint failed: {e}")
    problems.append("courses endpoint")

# 3. Cameras ------------------------------------------------------------
try:
    st, cams = _get("/api/admin/cameras")
except Exception as e:  # noqa: BLE001
    print(f"{BAD} cameras endpoint failed: {e}")
    print("\nCan't verify cameras — resolve the backend error above first.")
    sys.exit(1)

roles_seen = {}
print(f"\nCameras ({len(cams)} registered):")
for c in cams:
    role = c.get("assigned_role") or "?"
    roles_seen.setdefault(role, 0)
    roles_seen[role] += 1
    age = _age(c.get("last_seen_at"))
    online = age is not None and age <= STALE
    line = (
        f"  {role:<5} hole {c.get('assigned_hole')}  "
        f"'{c.get('name') or c.get('id')}'  "
        f"seen {_fmt_age(age)}  fw={c.get('firmware_version') or '?'}"
    )
    tag = OK if online else BAD
    print(f"{tag}{line}")
    if not online:
        problems.append(f"{role} camera offline (last seen {_fmt_age(age)})")
    if not c.get("enabled", True):
        print(f"{WARN}   ^ camera is DISABLED")
        problems.append(f"{role} camera disabled")
    if not c.get("triggering_enabled", True):
        print(f"{WARN}   ^ triggering is OFF — it won't record swings")
        problems.append(f"{role} triggering off")
    le = c.get("last_event_at")
    if le:
        print(f"       last capture: {_fmt_age(_age(le))} — {c.get('last_event_status')}")

# Need at least one tee + one green.
for want in ("tee", "green"):
    if not roles_seen.get(want):
        print(f"{BAD} no '{want}' camera registered")
        problems.append(f"missing {want} camera")

# 4. Summary ------------------------------------------------------------
print("\n" + "=" * 52)
if not problems:
    print("✅ READY — backend healthy, both cameras online + triggering.")
    print("   Go hit some swings.")
else:
    print(f"❌ NOT READY — {len(problems)} issue(s):")
    for p in problems:
        print(f"   • {p}")
    sys.exit(2)
