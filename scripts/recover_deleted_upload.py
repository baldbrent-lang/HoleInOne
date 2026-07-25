#!/usr/bin/env python3
"""Recover a clip that the non-golf auto-screen deleted by mistake.

The auto-delete only removes the LOCAL files and the DB rows — the raw
tee/green videos survive in Replit Object Storage (the background sweeper
mirrors every clip to the bucket). This script pulls them back down and
recreates a LongVideoUpload row so the clip reappears on the Production
page, ready to Produce again.

MUST run inside the Replit workspace/deployment shell (it needs the app's
database and the object-storage bucket):

    # 1. See what was auto-deleted and whether the files are recoverable
    python3 scripts/recover_deleted_upload.py list

    # 2. Restore by the filename shown in the list output
    python3 scripts/recover_deleted_upload.py restore --tee <tee_filename> \
        [--green <green_filename>] [--course-id 1] \
        [--captured-at 2026-07-25T14:30:00]

Where the filenames come from:
  * Newer deletions: the auto_delete_non_golf audit entry itself records
    "tee=<name> green=<name> course=<id> captured_at=<iso>" — `list` shows
    them and `restore` can take everything from there.
  * Older deletions (before filenames were audited): `list` cross-references
    the quick_upload_videos audit entry for the same long_upload id, which
    always logged "tee=<name>". Pi camera-event uploads have no quick-upload
    audit — for those, find the file in the bucket by capture timestamp
    (filenames embed it) via `list --bucket`.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

# Run from the repo root; make `app` importable the way the backend sees it.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "backend"))

# The Replit shell's bare python3 has none of the app's packages — they
# live in backend/.venv (see start.sh). Re-exec through the venv's
# interpreter when we're not already inside it, so plain
# `python3 scripts/recover_deleted_upload.py` works from the shell.
_VENV_PY = _REPO / "backend" / ".venv" / "bin" / "python"
try:
    import pydantic  # noqa: F401
except ModuleNotFoundError:
    import os

    # NB: can't compare interpreter paths — the venv's bin/python is a
    # symlink to the same base interpreter, so resolve() looks identical
    # inside and outside the venv. An env marker guards against loops.
    if _VENV_PY.exists() and not os.environ.get("GR_RECOVER_REEXEC"):
        os.environ["GR_RECOVER_REEXEC"] = "1"
        os.execv(str(_VENV_PY), [str(_VENV_PY)] + sys.argv)
    raise SystemExit(
        "App packages not importable. Run via the backend venv:\n"
        "  backend/.venv/bin/python scripts/recover_deleted_upload.py ..."
    )

from app.config import settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import AuditLog, CameraEvent, LongVideoUpload  # noqa: E402
from app.services import storage  # noqa: E402

CLIPS_DIR = _REPO / "backend" / settings.upload_dir / "clips"


_EVENT_RE = re.compile(r"^event-(\d+)-(tee|green)-[0-9a-f]+\.\w+$")


def _orphan_event_files(db) -> list[tuple[int, str, str]]:
    """Bucket objects named like Pi event clips (event-<id>-<role>-*.mp4)
    that NO surviving DB row references — i.e. clips whose CameraEvent +
    LongVideoUpload rows were deleted. These are the recovery candidates
    for auto-deleted Pi captures. Returns (event_id, role, name) sorted
    newest-event first."""
    referenced: set[str] = set()
    for tee, green in db.query(
        LongVideoUpload.tee_filename, LongVideoUpload.green_filename
    ):
        referenced.update(filter(None, (tee, green)))
    for tee, green in db.query(
        CameraEvent.tee_clip_filename, CameraEvent.green_clip_filename
    ):
        referenced.update(filter(None, (tee, green)))
    out = []
    for name in storage.list_names():
        m = _EVENT_RE.match(name)
        if m and name not in referenced:
            out.append((int(m.group(1)), m.group(2), name))
    out.sort(key=lambda t: (-t[0], t[1]))
    return out


def _parse_kv(detail: str | None) -> dict:
    """Pull tee=/green=/course=/captured_at= tokens out of an audit detail."""
    out: dict[str, str] = {}
    for key in ("tee", "green", "course", "captured_at", "swing_count"):
        m = re.search(rf"\b{key}=(\S+)", detail or "")
        if m:
            out[key] = m.group(1)
    return out


def cmd_list(args: argparse.Namespace) -> int:
    db = SessionLocal()
    try:
        deletions = (
            db.query(AuditLog)
            .filter(AuditLog.action == "auto_delete_non_golf")
            .order_by(AuditLog.at.desc())
            .limit(args.limit)
            .all()
        )
        if not deletions:
            print("No auto_delete_non_golf audit entries found.")
        any_unnamed = False
        for d in deletions:
            info = _parse_kv(d.detail)
            upload_id = (d.target or "").split(":")[-1]
            # Older audit entries didn't record filenames — fall back to the
            # quick-upload audit for the same long_upload id.
            if "tee" not in info:
                up = (
                    db.query(AuditLog)
                    .filter(
                        AuditLog.action == "quick_upload_videos",
                        AuditLog.target == d.target,
                    )
                    .order_by(AuditLog.at.desc())
                    .first()
                )
                if up:
                    info.update({k: v for k, v in _parse_kv(up.detail).items() if k not in info})
            print(f"\n[{d.at:%Y-%m-%d %H:%M}] upload #{upload_id}")
            print(f"  reason: {d.detail}")
            if "tee" in info:
                for key in ("tee", "green"):
                    name = info.get(key)
                    if not name:
                        continue
                    local = (CLIPS_DIR / name).exists()
                    bucket = storage.exists(name)
                    state = (
                        "LOCAL" if local
                        else ("in bucket - recoverable" if bucket else "NOT FOUND")
                    )
                    print(f"  {key}: {name}  [{state}]")
                extra = " ".join(
                    f"--{k.replace('_', '-')} {info[k]}"
                    for k in ("green", "captured_at") if info.get(k)
                )
                cid = info.get("course")
                print(
                    "  restore: python3 scripts/recover_deleted_upload.py "
                    f"restore --tee {info['tee']}"
                    + (f" --course-id {cid}" if cid else "")
                    + (f" {extra}" if extra else "")
                )
            else:
                any_unnamed = True
                print(
                    "  no filename on record (Pi camera-event upload) — "
                    "see orphaned bucket candidates below."
                )
        if any_unnamed:
            orphans = _orphan_event_files(db)
            print(
                f"\n--- orphaned Pi event clips in bucket "
                f"({len(orphans)} candidate(s)) ---"
            )
            print(
                "These event files exist in object storage but no DB row "
                "references them —\nthe auto-deleted captures are among "
                "them (higher event id = more recent)."
            )
            by_event: dict[int, dict[str, str]] = {}
            for eid, role, name in orphans:
                by_event.setdefault(eid, {})[role] = name
            for eid in sorted(by_event, reverse=True):
                files = by_event[eid]
                print(f"\n  event {eid}:")
                for role, name in sorted(files.items(), reverse=True):
                    print(f"    {role}: {name}")
                if "tee" in files:
                    cmd = (
                        "python3 scripts/recover_deleted_upload.py "
                        f"restore --tee {files['tee']}"
                    )
                    if "green" in files:
                        cmd += f" --green {files['green']}"
                    print(f"    restore: {cmd}")
        if args.bucket:
            names = sorted(storage.list_names())
            print(f"\n--- bucket contents ({len(names)} objects) ---")
            for n in names:
                print(f"  {n}")
        return 0
    finally:
        db.close()


def cmd_restore(args: argparse.Namespace) -> int:
    tee = Path(args.tee).name
    green = Path(args.green).name if args.green else None

    for name in filter(None, [tee, green]):
        if (CLIPS_DIR / name).exists():
            print(f"already local: {name}")
        elif storage.ensure_local(CLIPS_DIR, name):
            print(f"downloaded from bucket: {name}")
        else:
            print(f"ERROR: {name} is neither on disk nor in the bucket — cannot recover.")
            return 1

    captured = (
        datetime.fromisoformat(args.captured_at)
        if args.captured_at
        else datetime.fromtimestamp((CLIPS_DIR / tee).stat().st_mtime)
    )

    db = SessionLocal()
    try:
        existing = (
            db.query(LongVideoUpload)
            .filter(LongVideoUpload.tee_filename == tee)
            .first()
        )
        if existing:
            print(f"upload row already exists (#{existing.id}) — nothing to insert.")
            return 0
        row = LongVideoUpload(
            course_id=args.course_id,
            camera_type="tee",
            base_captured_at=captured,
            tee_filename=tee,
            green_filename=green,
            swing_count="multiple",
            processing_status="pending",
            note="recovered after auto-delete (scripts/recover_deleted_upload.py)",
        )
        db.add(row)
        db.commit()
        print(
            f"restored as upload #{row.id} (course {args.course_id}, "
            f"captured {captured:%Y-%m-%d %H:%M}). It's back on the "
            "Production page — hit Produce to process it."
        )
        db.add(
            AuditLog(
                actor="admin",
                action="recover_deleted_upload",
                target=f"long_upload:{row.id}",
                detail=f"tee={tee}" + (f" green={green}" if green else ""),
            )
        )
        db.commit()
        return 0
    finally:
        db.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="show recent auto-deletions + recoverability")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.add_argument("--bucket", action="store_true", help="also dump bucket contents")
    p_list.set_defaults(fn=cmd_list)

    p_res = sub.add_parser("restore", help="pull files from the bucket + recreate the upload row")
    p_res.add_argument("--tee", required=True, help="tee video filename (as stored in clips/)")
    p_res.add_argument("--green", default=None, help="green video filename, if there was one")
    p_res.add_argument("--course-id", type=int, default=1)
    p_res.add_argument(
        "--captured-at", default=None,
        help="ISO timestamp (defaults to the file's mtime after download)",
    )
    p_res.set_defaults(fn=cmd_restore)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
