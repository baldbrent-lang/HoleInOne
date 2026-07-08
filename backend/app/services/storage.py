"""Persistent object storage for uploaded + produced clips.

Replit deployments have an EPHEMERAL filesystem — every redeploy wipes
`uploads/clips/`, so raw footage and produced videos vanish. This module
mirrors those files into **Replit Object Storage** (a GCS-backed bucket
that survives deploys), and rehydrates them back onto local disk on
demand, so a republish no longer deletes anything.

Design goals:
  * FAIL-SAFE. If the object-storage client can't initialise (bucket not
    added to the repl, package missing, running locally in dev/CI), every
    function becomes a no-op and the app behaves exactly as before —
    local disk only. Enabling persistence is purely additive.
  * NO write-path surgery. A background sweeper uploads any new local clip
    to the bucket, so we don't have to touch every place that writes a
    file. Serving + processing pull a file back from the bucket on demand
    when it's missing locally (post-redeploy).

The object key is just the filename (clips already have unique,
UUID/session-derived names), so there's no directory nesting to manage.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

log = logging.getLogger("golfreelz.storage")

_client = None
_enabled = False

try:  # pragma: no cover - depends on the deploy environment
    from replit.object_storage import Client  # type: ignore

    _client = Client()
    # Probe: the package can import + Client() can construct even when NO
    # bucket is configured for the repl (e.g. dev). Without this check we'd
    # think storage is "on" and then fail — and log — on every single file,
    # flooding the console. A cheap exists() call raises "no default bucket"
    # when unconfigured, so we disable cleanly and stay local-disk only.
    _client.exists("__golfreelz_probe__")
    _enabled = True
    log.info("storage: Replit Object Storage enabled")
except Exception as exc:  # noqa: BLE001
    _client = None
    log.info(
        "storage: object storage unavailable (%s) — running local-disk only",
        exc,
    )


def enabled() -> bool:
    return _enabled


def upload(name: str, path: Path) -> bool:
    """Mirror a local file into the bucket under key `name`. Best-effort."""
    if not _enabled or not path.exists():
        return False
    try:
        # Prefer the filename API; fall back to bytes for older clients.
        if hasattr(_client, "upload_from_filename"):
            _client.upload_from_filename(name, str(path))
        else:
            _client.upload_from_bytes(name, path.read_bytes())
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("storage: upload failed for %s: %s", name, exc)
        return False


def exists(name: str) -> bool:
    if not _enabled:
        return False
    try:
        return bool(_client.exists(name))
    except Exception:  # noqa: BLE001
        return False


def download(name: str, path: Path) -> bool:
    """Pull key `name` from the bucket onto local disk at `path`. Best-effort."""
    if not _enabled:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".dl")
        if hasattr(_client, "download_to_filename"):
            _client.download_to_filename(name, str(tmp))
        else:
            tmp.write_bytes(_client.download_as_bytes(name))
        tmp.replace(path)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("storage: download failed for %s: %s", name, exc)
        return False


def list_names() -> set[str]:
    if not _enabled:
        return set()
    try:
        out: set[str] = set()
        for obj in _client.list():
            # Objects expose their key as `.name`; be defensive about shape.
            n = getattr(obj, "name", None) or (obj if isinstance(obj, str) else None)
            if n:
                out.add(n)
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("storage: list failed: %s", exc)
        return set()


def ensure_local(clips_dir: Path, name: str) -> bool:
    """Guarantee `clips_dir/name` exists locally, pulling it from the bucket
    if the ephemeral disk lost it (e.g. after a redeploy). Returns True if the
    file is present locally afterward."""
    if not name:
        return False
    local = clips_dir / Path(name).name
    if local.exists():
        return True
    return download(Path(name).name, local)


# Files the sweeper should never push to the bucket (local-only artefacts).
_SKIP_SUFFIXES = (".dl", ".reencode.mp4", ".h264-ok")


def start_sweeper(clips_dir: Path, interval_sec: float = 30.0) -> None:
    """Background mirror: every `interval_sec`, upload any clip on local disk
    that isn't in the bucket yet. Avoids having to hook every write site.

    Skips files modified in the last ~20 s so we never upload one mid-write.
    Idempotent to start (guards against double-launch). No-op when storage is
    disabled."""
    if not _enabled:
        return
    if getattr(start_sweeper, "_started", False):
        return
    start_sweeper._started = True  # type: ignore[attr-defined]

    def _run() -> None:
        # Seed with what's already in the bucket so we don't re-upload.
        known = list_names()
        log.info("storage: sweeper started (%d object(s) already in bucket)", len(known))
        while True:
            try:
                now = time.time()
                for f in clips_dir.glob("*"):
                    if not f.is_file():
                        continue
                    if f.name in known:
                        continue
                    if f.name.endswith(_SKIP_SUFFIXES):
                        continue
                    # Skip very-recent files — may still be being written.
                    try:
                        if now - f.stat().st_mtime < 20:
                            continue
                    except OSError:
                        continue
                    if upload(f.name, f):
                        known.add(f.name)
                        log.info("storage: mirrored %s to bucket", f.name)
            except Exception as exc:  # noqa: BLE001
                log.warning("storage: sweeper pass failed: %s", exc)
            time.sleep(interval_sec)

    threading.Thread(target=_run, name="storage-sweeper", daemon=True).start()
