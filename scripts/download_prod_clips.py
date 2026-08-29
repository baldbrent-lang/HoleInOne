#!/usr/bin/env python3
"""Download all raw camera clips from the GolfReelz production backend.

Pulls every tee + green raw video the Pis uploaded so you can test/tune the
producer locally without re-recording on the course.

Run it on a machine that can reach production (e.g. your Mac):

    ADMIN_PASSWORD='your-admin-password' python3 download_prod_clips.py

Optional overrides (env vars):
    BASE_URL   default https://holeinone-s6qm.onrender.com
    OUT_DIR    default ./prod_clips

The admin password is the one you use to log in to /admin. It's only used
to LIST the clips; the files themselves are served statically. No extra
packages needed — standard library only.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get("BASE_URL", "https://holeinone-s6qm.onrender.com").rstrip("/")
PW = os.environ.get("ADMIN_PASSWORD")
OUT = Path(os.environ.get("OUT_DIR", "prod_clips"))

if not PW:
    sys.exit("Set ADMIN_PASSWORD (your /admin password). See the header of this file.")

OUT.mkdir(parents=True, exist_ok=True)


def _get(path: str):
    req = urllib.request.Request(BASE + path, headers={"X-Admin-Password": PW})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _collect_clip_urls(obj, acc: set):
    """Walk the JSON and grab anything that looks like a raw clip URL.
    Robust to the exact response shape — we just want the video URLs."""
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_clip_urls(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _collect_clip_urls(v, acc)
    elif isinstance(obj, str):
        low = obj.lower()
        if "/uploads/clips/" in low and low.endswith((".mp4", ".mov", ".m4v")):
            acc.add(obj)


urls: set = set()
for ep in ("/api/admin/camera-events?limit=500", "/api/admin/long-uploads?limit=500"):
    try:
        _collect_clip_urls(_get(ep), urls)
        print(f"{ep}: scanned — {len(urls)} clip URL(s) so far")
    except urllib.error.HTTPError as e:
        print(f"{ep}: HTTP {e.code} {e.reason} — skipping")
    except Exception as e:  # noqa: BLE001
        print(f"{ep}: {e} — skipping")

if not urls:
    sys.exit("No clip URLs found. Check ADMIN_PASSWORD / BASE_URL, or there may be no clips on disk.")

print(f"\n{len(urls)} unique clip(s). Downloading to {OUT}/ ...\n")
ok = 0
for i, url in enumerate(sorted(urls), 1):
    name = url.rsplit("/", 1)[-1].split("?", 1)[0]
    dest = OUT / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[{i}/{len(urls)}] skip (exists): {name}")
        ok += 1
        continue
    try:
        req = urllib.request.Request(url, headers={"X-Admin-Password": PW})
        with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
            f.write(r.read())
        print(f"[{i}/{len(urls)}] {name}  {dest.stat().st_size / 1e6:.1f} MB")
        ok += 1
    except Exception as e:  # noqa: BLE001
        print(f"[{i}/{len(urls)}] FAILED {name}: {e}")

print(f"\nDone — {ok}/{len(urls)} clip(s) in {OUT.resolve()}")
