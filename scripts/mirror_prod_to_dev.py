#!/usr/bin/env python3
"""Mirror raw camera clips from the GolfReelz PRODUCTION backend into a DEV
backend, in one command, so you can test the producer/tracer on real
footage without touching production.

For each camera event in prod it downloads the tee (+ green) clip and
re-uploads it into dev's long-upload pipeline with auto-detect on, so the
event lands in dev's Production page ready to produce.

Repeatable / incremental: it records which prod events it has already
mirrored in `.mirrored_events.txt` and skips them next run, so you can run
it again anytime to pull only the NEW captures. Delete that file to
re-mirror everything.

Standard library only — nothing to install.

Environment variables:
  DEV_URL            (required) your dev site base URL, e.g.
                     https://<your-repl>.replit.dev
  DEV_COURSE_ID      (required) the course id in dev to attach clips to
  ADMIN_PASSWORD     admin password (used for BOTH prod and dev unless a
                     separate dev one is given). Default: Baldy123
  PROD_URL           default https://holeinone-s6qm.onrender.com
  DEV_ADMIN_PASSWORD (optional) dev admin password, if different from prod
  TEE_ONLY           set to 1 to mirror tee clips only (skip green) —
                     faster, and enough to test detection + the tracer
  LIMIT              mirror only the N most recent events. Prod has
                     hundreds going back to the first camera ever set
                     up, and pulling them oldest-first spends an hour on
                     footage nobody is looking at. LIMIT=25 is a working
                     set. Empty or 0 = all of them.
"""

import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

PROD = os.environ.get("PROD_URL", "https://holeinone-s6qm.onrender.com").rstrip("/")
DEV = (os.environ.get("DEV_URL") or "").rstrip("/")
PROD_PW = os.environ.get("ADMIN_PASSWORD", "Baldy123")
DEV_PW = os.environ.get("DEV_ADMIN_PASSWORD") or PROD_PW
DEV_COURSE_ID = os.environ.get("DEV_COURSE_ID")
TEE_ONLY = os.environ.get("TEE_ONLY")
# How many of the NEWEST events to mirror. Empty or 0 = all of them.
try:
    LIMIT = int(os.environ.get("LIMIT") or 0)
except ValueError:
    LIMIT = 0

# Ledger of prod event ids we've ALREADY mirrored. Append-only: deleting a
# clip in dev deliberately does NOT remove its id here, so a delete stays a
# delete and re-running won't drag it back in.
#
# Anchor to a FIXED absolute path (next to this script) instead of the CWD,
# so it doesn't matter which directory you launch from — a relative path
# meant that running from a different folder found no ledger, assumed nothing
# had ever been mirrored, and re-imported everything (including deleted ones).
# Override with MIRROR_STATE_FILE to pin it anywhere you like.
STATE = Path(
    os.environ.get("MIRROR_STATE_FILE")
    or (Path(__file__).resolve().parent / ".mirrored_events.txt")
)
# Legacy location (older runs wrote the ledger to the CWD). Read it too so any
# existing history is honored and those events aren't re-mirrored once.
_LEGACY_STATE = Path(".mirrored_events.txt")

if not DEV:
    sys.exit("Set DEV_URL to your dev site base URL (see header of this file).")
if not DEV_COURSE_ID:
    sys.exit("Set DEV_COURSE_ID to the course id in dev to attach clips to.")


def _get_json(url: str, pw: str):
    req = urllib.request.Request(url, headers={"X-Admin-Password": pw})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _download(url: str, pw: str) -> bytes:
    req = urllib.request.Request(url, headers={"X-Admin-Password": pw})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def _multipart(fields: dict, files: dict) -> tuple[str, bytes]:
    """Build a multipart/form-data body. files: name -> (filename, bytes)."""
    boundary = "----golfreelz" + uuid.uuid4().hex
    parts = []
    for name, val in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{val}\r\n'.encode()
        )
    for name, (fn, data) in files.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
            f'filename="{fn}"\r\nContent-Type: video/mp4\r\n\r\n'.encode()
        )
        parts.append(data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return boundary, b"".join(parts)


def _post_json(path: str, body: dict) -> int:
    req = urllib.request.Request(
        f"{DEV}{path}",
        data=json.dumps(body).encode(),
        headers={
            "X-Admin-Password": DEV_PW,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status


def _backfill_clocks(events: list) -> None:
    """Give already-mirrored uploads the two camera clocks.

    THE OFFSET IS NOT A DETAIL. Everything cross-camera is worked out
    from (green first frame - tee first frame): with no stamps, dev
    assumes the two recordings started together and every landing
    window lands in the wrong stretch of the green clip. Earlier runs of
    this script posted the clips without the stamps, so the uploads they
    made are all sitting on that assumption -- and re-mirroring them
    means re-downloading gigabytes to fix two datetimes.

    So: match dev's uploads back to their prod events by the filename
    this script uploaded them under, and merge the stamps in. Only rows
    that are missing them are touched, so this is safe to re-run and
    never overwrites an offset somebody established by hand.
    """
    by_name = {}
    for e in events:
        eid = str(e.get("id"))
        if e.get("tee_recording_started_at") or e.get("green_recording_started_at"):
            by_name[f"event-{eid}-tee.mp4"] = e
    if not by_name:
        return
    try:
        rows = _get_json(f"{DEV}/api/admin/long-uploads?limit=500", DEV_PW)
    except Exception as ex:  # noqa: BLE001
        print(f"(clock backfill skipped: could not list dev uploads: {ex})")
        return
    fixed = 0
    for r in rows or []:
        e = by_name.get(r.get("tee_original_filename") or "")
        if not e:
            continue
        _em = r.get("edit_metrics") or {}
        if _em.get("source_tee_started_at") and _em.get("source_green_started_at"):
            continue
        body = {}
        if e.get("tee_recording_started_at"):
            body["source_tee_started_at"] = e["tee_recording_started_at"]
        if e.get("green_recording_started_at"):
            body["source_green_started_at"] = e["green_recording_started_at"]
        if not body:
            continue
        try:
            _post_json(f"/api/admin/long-uploads/{r['id']}/edit-metrics", body)
            fixed += 1
        except Exception as ex:  # noqa: BLE001
            print(f"  (upload {r.get('id')}: clock backfill failed: {ex})")
    if fixed:
        print(f"clock backfill: gave {fixed} already-mirrored upload(s) the "
              f"tee/green start stamps they were uploaded without\n")


def _post_upload(fields: dict, files: dict) -> int:
    boundary, body = _multipart(fields, files)
    req = urllib.request.Request(
        f"{DEV}/api/admin/clips/long-upload",
        data=body,
        headers={
            "X-Admin-Password": DEV_PW,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.status


def _list_events() -> list:
    """Page through prod's camera-events in small chunks. A single big
    request (limit=500) makes prod probe every clip's video info at once
    and 500s; small pages keep each request light. A page that still
    errors is skipped with a warning rather than aborting the whole run."""
    events: list = []
    offset = 0
    page = 10
    while True:
        url = f"{PROD}/api/admin/camera-events?limit={page}&offset={offset}"
        try:
            batch = _get_json(url, PROD_PW)
        except urllib.error.HTTPError as e:
            print(f"  (skipping events {offset}-{offset + page}: HTTP {e.code})")
            offset += page
            if offset > 2000:  # safety stop
                break
            continue
        if not batch:
            break
        events.extend(batch)
        if len(batch) < page:
            break
        offset += page
    return events


done = set()
# Union the fixed ledger and the legacy CWD ledger so no history is lost.
for _ledger in {STATE.resolve(), _LEGACY_STATE.resolve()}:
    if _ledger.exists():
        done |= {ln.strip() for ln in _ledger.read_text().splitlines() if ln.strip()}
print(f"ledger: {STATE} ({len(done)} event(s) already mirrored, will be skipped)")

events = _list_events()
if not events:
    sys.exit("No events returned from prod (all pages failed, or none exist).")

# Newest first from the API; mirror oldest first so dev's ordering matches.
events = list(reversed(events))
todo = [e for e in events if e.get("tee_url") and str(e["id"]) not in done]
# THE RECENT ONES ARE THE ONES BEING WORKED ON. Four hundred events go
# back to the first camera ever pointed at a tee; a laptop pulling them
# oldest-first spends an hour on footage nobody is looking at before it
# reaches this week's. LIMIT takes the N most recent and still mirrors
# them oldest-first among themselves, so dev's ordering matches prod's.
if LIMIT:
    _skipped = max(0, len(todo) - LIMIT)
    todo = todo[-LIMIT:]
    if _skipped:
        print(f"LIMIT={LIMIT}: taking the {len(todo)} most recent, "
              f"leaving {_skipped} older one(s) unmirrored")
print(f"{len(events)} prod events, {len(todo)} new to mirror into dev course {DEV_COURSE_ID}\n")

# Repair what earlier runs uploaded without the clocks, before mirroring
# anything new -- it costs two API calls and nothing downloads.
_backfill_clocks(events)

ok = 0
for i, e in enumerate(todo, 1):
    eid = str(e["id"])
    tag = f"[{i}/{len(todo)}] event {eid} (hole {e.get('hole_number')})"
    try:
        tee = _download(e["tee_url"], PROD_PW)
        files = {"video": (f"event-{eid}-tee.mp4", tee)}
        if not TEE_ONLY and e.get("green_url"):
            files["video_green"] = (f"event-{eid}-green.mp4", _download(e["green_url"], PROD_PW))
        fields = {
            "course_id": str(DEV_COURSE_ID),
            "camera_type": "tee",
            "base_captured_at": e.get("triggered_at") or "2026-01-01T00:00:00",
            "auto_detect_swings": "true",
            "starting_hole": str(e.get("hole_number") or 1),
            "motion_ratio": "2.0",
            # Use the SAME vision-only detector the live camera events use,
            # so detection results in dev match what prod's camera path does.
            "motion_only": "true",
        }
        # THE TWO CAMERAS' WALL CLOCKS, CARRIED OVER. A mirrored pair
        # gets no CameraEvent in dev, and those two stamps -- the
        # instant each file's first frame was captured -- are the only
        # thing that says how far apart the recordings started. Without
        # them dev cuts on an ASSUMED zero offset, which is not a small
        # error: it lands directly on the frame the green half is
        # entered at, so every landing search looks in the wrong three
        # seconds and mirrored footage behaves unlike the same footage
        # on prod. Prod hands both stamps over in the very listing
        # these clips came from.
        for _f, _k in (("tee_started_at", "tee_recording_started_at"),
                       ("green_started_at", "green_recording_started_at")):
            if e.get(_k):
                fields[_f] = e[_k]
        status = _post_upload(fields, files)
        mb = sum(len(v[1]) for v in files.values()) / 1e6
        print(f"{tag}: uploaded {len(files)} clip(s), {mb:.1f} MB -> dev HTTP {status}")
        with STATE.open("a") as f:
            f.write(eid + "\n")
        ok += 1
    except urllib.error.HTTPError as ex:
        body = ex.read()[:200].decode(errors="replace")
        print(f"{tag}: FAILED HTTP {ex.code} {ex.reason}: {body}")
    except Exception as ex:  # noqa: BLE001
        print(f"{tag}: FAILED {ex}")

print(f"\nDone — mirrored {ok}/{len(todo)} new event(s) into dev. "
      f"Re-run anytime to pull new ones (state in {STATE}).")
