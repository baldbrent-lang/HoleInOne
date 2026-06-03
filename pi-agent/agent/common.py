"""Shared agent utilities: config loading, HTTP client, ring buffer,
heartbeat thread. Imported by both the tee and green role runners.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import requests
import yaml

log = logging.getLogger("golfreelz_agent.common")


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

def load_config(path: Path) -> dict:
    """Load YAML config + apply env-variable overrides for sensitive
    fields. `GOLFREELZ_AUTH_TOKEN` and `GOLFREELZ_BACKEND_URL` win
    over file values so the token doesn't have to be committed to the
    SD card if the operator prefers a secrets-manager flow."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if os.environ.get("GOLFREELZ_AUTH_TOKEN"):
        cfg["auth_token"] = os.environ["GOLFREELZ_AUTH_TOKEN"]
    if os.environ.get("GOLFREELZ_BACKEND_URL"):
        cfg["backend_url"] = os.environ["GOLFREELZ_BACKEND_URL"]
    if not cfg.get("auth_token"):
        raise RuntimeError(
            "auth_token missing — set it in config.yaml or via the "
            "GOLFREELZ_AUTH_TOKEN env var",
        )
    if not cfg.get("backend_url"):
        raise RuntimeError("backend_url missing")
    return cfg


# ---------------------------------------------------------------------
# Clip transcode (shrink uploads for slow cellular uplinks)
# ---------------------------------------------------------------------

# Target video bitrate for the H.264 re-encode. OpenCV writes raw mp4v
# at roughly 5 Mbps, which makes a multi-minute clip too big to push
# over a slow 4G uplink before the request times out. H.264 at this
# bitrate is a 3-5x size reduction at equal-or-better visual quality.
# Override per-Pi with the GOLFREELZ_UPLOAD_BITRATE env var (e.g.
# "1500k" on a very slow link, "4000k" on a fast one).
DEFAULT_UPLOAD_BITRATE = "2500k"


def _transcode_to_h264(src_path: Path) -> tuple[Path, bool]:
    """Re-encode a raw clip to efficient H.264 so the upload is small
    enough for a slow cellular uplink.

    Returns ``(path_to_upload, is_temp)``. On any problem — ffmpeg
    missing, encode failure, empty output — it logs a warning and
    returns the *original* clip (``is_temp=False``), so a transcode
    issue never blocks the upload; it just falls back to sending the
    bigger raw file. The caller deletes the temp file when is_temp."""
    if shutil.which("ffmpeg") is None:
        log.warning("ffmpeg not found on PATH — uploading raw clip")
        return src_path, False

    bitrate = os.environ.get("GOLFREELZ_UPLOAD_BITRATE", DEFAULT_UPLOAD_BITRATE)
    out_path = src_path.with_name(f"{src_path.stem}-h264.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src_path),
        "-c:v", "libx264", "-preset", "veryfast", "-b:v", bitrate,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",  # moov atom up front for fast server reads
        "-an",                      # cameras have no audio
        str(out_path),
    ]
    try:
        subprocess.run(
            cmd, check=True, timeout=600,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    except Exception as exc:  # FileNotFound / CalledProcess / Timeout
        log.warning("h264 transcode failed (%s) — uploading raw clip", exc)
        out_path.unlink(missing_ok=True)
        return src_path, False

    if not out_path.exists() or out_path.stat().st_size == 0:
        log.warning("h264 transcode produced an empty file — uploading raw clip")
        out_path.unlink(missing_ok=True)
        return src_path, False

    try:
        src_mb = src_path.stat().st_size / (1024 * 1024)
        out_mb = out_path.stat().st_size / (1024 * 1024)
        log.info(
            "h264 transcode: %.1f MB -> %.1f MB (bitrate %s)",
            src_mb, out_mb, bitrate,
        )
    except OSError:
        pass
    return out_path, True


# ---------------------------------------------------------------------
# Backend HTTP client
# ---------------------------------------------------------------------

class BackendClient:
    """Wraps the four /api/cameras/{token}/... endpoints. Retries
    transient network / 5xx errors with exponential backoff;
    auth/validation errors (4xx) fail fast."""

    def __init__(self, base_url: str, auth_token: str):
        self.base_url = base_url.rstrip("/")
        self.token = auth_token
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/cameras/{self.token}{path}"

    def _retry(self, method: str, path: str, *, retries: int = 4,
               timeout: int = 30, **kwargs) -> dict:
        delay = 1.0
        last_err: str = ""
        for attempt in range(retries):
            try:
                resp = self.session.request(
                    method, self._url(path), timeout=timeout, **kwargs,
                )
                if 200 <= resp.status_code < 300:
                    return resp.json() if resp.content else {}
                if resp.status_code in (400, 401, 403, 404):
                    # Hard fail — auth or validation problem, retry won't help.
                    raise RuntimeError(
                        f"{method} {path} -> {resp.status_code}: {resp.text[:200]}",
                    )
                last_err = f"{resp.status_code}: {resp.text[:120]}"
                log.warning(
                    "backend %s %s status %s (attempt %d/%d)",
                    method, path, resp.status_code, attempt + 1, retries,
                )
            except (requests.RequestException, socket.error) as e:
                last_err = str(e)
                log.warning(
                    "backend %s %s network error %s (attempt %d/%d)",
                    method, path, e, attempt + 1, retries,
                )
            time.sleep(delay)
            delay = min(delay * 2, 30)
        raise RuntimeError(f"backend {method} {path} failed: {last_err}")

    def heartbeat(self, firmware_version: str = "") -> dict:
        return self._retry(
            "POST", "/heartbeat",
            data={"firmware_version": firmware_version},
            retries=2,
        )

    def event_trigger(self, session_id: str) -> dict:
        return self._retry(
            "POST", "/event-trigger",
            data={"session_id": session_id},
        )

    def poll_trigger(self, timeout_seconds: int = 25) -> dict:
        # HTTP timeout slightly exceeds the server-side long-poll
        # timeout so the request actually gets the response.
        return self._retry(
            "GET", f"/poll-trigger?timeout={timeout_seconds}",
            timeout=timeout_seconds + 10,
            retries=2,
        )

    def upload_event(self, session_id: str, video_path: Path) -> dict:
        # Shrink the clip to H.264 before sending so long captures fit
        # through a slow uplink. Falls back to the raw clip if the
        # transcode can't run. Longer timeout than the small calls
        # because even a shrunk multi-minute clip takes a while on 4G.
        upload_path, is_temp = _transcode_to_h264(video_path)
        try:
            with open(upload_path, "rb") as fh:
                files = {"video": (upload_path.name, fh, "video/mp4")}
                return self._retry(
                    "POST", "/upload-event",
                    data={"session_id": session_id},
                    files=files,
                    timeout=600,
                    retries=3,
                )
        finally:
            if is_temp:
                upload_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------
# Heartbeat thread
# ---------------------------------------------------------------------

class HeartbeatThread(threading.Thread):
    """Daemon thread that pings /heartbeat at a fixed interval so the
    admin UI can see this camera as alive. Failures are warnings,
    not fatal — the main capture loop keeps running even if the
    backend is briefly unreachable."""

    def __init__(self, client: BackendClient, interval: int, firmware: str):
        super().__init__(daemon=True, name="heartbeat")
        self.client = client
        self.interval = max(15, int(interval))
        self.firmware = firmware
        self.stopping = threading.Event()

    def run(self) -> None:
        # First heartbeat immediately so admin UI sees the camera
        # come online without waiting a full interval.
        while not self.stopping.is_set():
            try:
                self.client.heartbeat(firmware_version=self.firmware)
                log.debug("heartbeat ok")
            except Exception as exc:  # pragma: no cover
                log.warning("heartbeat failed: %s", exc)
            if self.stopping.wait(self.interval):
                return

    def stop(self) -> None:
        self.stopping.set()


# ---------------------------------------------------------------------
# Frame ring buffer
# ---------------------------------------------------------------------

class FrameBuffer:
    """Thread-safe circular buffer of (timestamp, frame) tuples. Sized
    by `seconds * fps` so the operator can configure the pre-roll
    length in real time and have the buffer length match exactly.
    Frames are stored as numpy arrays — the buffer doesn't copy on
    push, so callers should pass already-cloned frames if they plan
    to keep mutating the original."""

    def __init__(self, seconds: float, fps: float):
        self.max_frames = max(1, int(round(seconds * fps)) + 1)
        self.lock = threading.Lock()
        self.frames = deque(maxlen=self.max_frames)

    def push(self, ts: float, frame) -> None:
        with self.lock:
            self.frames.append((ts, frame))

    def snapshot(self) -> list:
        """Return a shallow copy of the buffer's current contents.
        Callers can iterate freely; the underlying deque continues
        to mutate in the capture thread."""
        with self.lock:
            return list(self.frames)

    def latest_ts(self) -> Optional[float]:
        with self.lock:
            return self.frames[-1][0] if self.frames else None

    def clear(self) -> None:
        with self.lock:
            self.frames.clear()


# ---------------------------------------------------------------------
# Camera open helper
# ---------------------------------------------------------------------

def open_camera(cam_cfg: dict):
    """Open the configured camera with OpenCV. `device: "auto"` tries
    /dev/video0..3 and returns the first that opens; otherwise uses
    the supplied device string (path on Linux, index on Windows /
    macOS, or a v4l2 device path)."""
    import cv2

    device = cam_cfg.get("device", "auto")
    cap = None
    if device == "auto":
        for idx in range(4):
            candidate = cv2.VideoCapture(idx)
            if candidate.isOpened():
                log.info("opened camera at index %d", idx)
                cap = candidate
                break
            candidate.release()
    else:
        try:
            cap = cv2.VideoCapture(int(device))
        except (TypeError, ValueError):
            cap = cv2.VideoCapture(str(device))
    if cap is None or not cap.isOpened():
        raise RuntimeError(f"could not open camera device={device!r}")

    width = int(cam_cfg.get("width", 1920))
    height = int(cam_cfg.get("height", 1080))
    fps = int(cam_cfg.get("fps", 30))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps)
    log.info(
        "camera open: requested %dx%d@%d, got %dx%d@%.1f",
        width, height, fps, actual_w, actual_h, actual_fps,
    )
    return cap
