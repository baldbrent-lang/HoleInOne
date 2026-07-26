"""Shared agent utilities: config loading, HTTP client, ring buffer,
heartbeat thread. Imported by both the tee and green role runners.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import shlex
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

# Capture-mode presets for the Pi HQ Camera (IMX477). The sensor's
# native modes: full-width 2x2 binned 2028x1080 sustains ~50fps, and the
# cropped 1332x990 mode reaches 120fps. Frame rate is pure gold for the
# ball tracer — at 30fps a driven ball crosses hundreds of pixels
# between frames; at 50-60 you get ~2x the track points, half the
# per-frame motion, and less blur smearing the ball into the grass.
# Cost: bigger uploads (same bitrate, more even quality if you bump
# upload_bitrate_kbps) and a bigger RAM pre-roll buffer.
CAPTURE_MODES = {
    "1080p30": {"width": 1920, "height": 1080, "fps": 30},   # default
    "1080p50": {"width": 2028, "height": 1080, "fps": 50},   # binned, full FOV
    "990p120": {"width": 1332, "height": 990, "fps": 120},   # cropped, max fps
}
_MODE_ALIASES = {
    "default": "1080p30", "30": "1080p30", "30fps": "1080p30",
    "50": "1080p50", "50fps": "1080p50",
    "120": "990p120", "120fps": "990p120",
}


def _apply_capture_mode(cfg: dict) -> None:
    """Expand `camera.mode` into width/height/fps (mode wins over any
    explicit values), scale a 1920x1080-authored tee ROI to the new
    frame geometry, and log the pre-roll buffer's RAM appetite so a
    120fps experiment can't silently OOM a Pi."""
    cam = cfg.setdefault("camera", {})
    mode_raw = str(cam.get("mode") or "").strip().lower()
    if mode_raw:
        mode = _MODE_ALIASES.get(mode_raw, mode_raw)
        preset = CAPTURE_MODES.get(mode)
        if preset:
            cam.update(preset)
            log.info(
                "capture mode %r -> %dx%d@%d",
                mode, preset["width"], preset["height"], preset["fps"],
            )
            # The tee ROI was drawn in pixel coords against the old
            # frame. Scale it to the new geometry (approximate — the
            # binned/cropped modes shift FOV slightly; re-draw the ROI
            # if precision matters, this keeps detection working).
            roi = cfg.get("tee_box_roi")
            if (
                isinstance(roi, dict)
                and all(k in roi for k in ("x", "y", "w", "h"))
                and (preset["width"], preset["height"]) != (1920, 1080)
                and roi["x"] + roi["w"] <= 1920
                and roi["y"] + roi["h"] <= 1080
            ):
                sx = preset["width"] / 1920.0
                sy = preset["height"] / 1080.0
                scaled = {
                    "x": int(round(roi["x"] * sx)),
                    "y": int(round(roi["y"] * sy)),
                    "w": int(round(roi["w"] * sx)),
                    "h": int(round(roi["h"] * sy)),
                }
                cfg["tee_box_roi"] = scaled
                log.info(
                    "capture mode: tee ROI scaled %s -> %s", roi, scaled,
                )
        else:
            log.warning(
                "unknown camera.mode %r — valid: %s; keeping explicit "
                "width/height/fps",
                mode_raw, ", ".join(sorted(CAPTURE_MODES)),
            )
    # RAM appetite of the raw-frame pre-roll ring buffer. 5s of
    # 1080p30 ~= 930MB; 1080p50 ~= 1.6GB; 990p120 ~= 2.4GB. A Pi 5
    # 8GB survives all three, but log it loudly so nobody 120fps's a
    # 4GB Pi into the OOM killer.
    try:
        _w = int(cam.get("width", 1920))
        _h = int(cam.get("height", 1080))
        _f = float(cam.get("fps", 30))
        _sec = float(cfg.get("buffer_seconds", 5))
        _mb = _w * _h * 3 * _f * _sec / 1e6
        msg = (
            f"pre-roll buffer: ~{_mb:.0f}MB RAM "
            f"({_w}x{_h}@{_f:.0f} x {_sec:.1f}s)"
        )
        if _mb > 2600:
            log.warning(
                "%s — heavy! reduce buffer_seconds or fps if the Pi "
                "runs out of memory", msg,
            )
        else:
            log.info(msg)
    except (TypeError, ValueError):
        pass


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
    _apply_capture_mode(cfg)
    return cfg


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
               timeout: int = 30, make_files=None, **kwargs) -> dict:
        delay = 1.0
        last_err: str = ""
        for attempt in range(retries):
            # Rebuild any multipart file payload fresh for every attempt.
            # A single shared file handle gets consumed by the first
            # attempt; if that attempt times out, the retry would upload
            # an empty body and the server rejects it with 400 "empty
            # upload" — turning a transient stall into a fatal error.
            files = make_files() if make_files is not None else None
            try:
                call_kwargs = dict(kwargs)
                if files is not None:
                    call_kwargs["files"] = files
                resp = self.session.request(
                    method, self._url(path), timeout=timeout, **call_kwargs,
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
            finally:
                # Close handles opened for this attempt so the next attempt
                # re-reads the file from the start.
                if files:
                    for v in files.values():
                        fh = v[1] if isinstance(v, (tuple, list)) else v
                        try:
                            fh.close()
                        except Exception:
                            pass
            time.sleep(delay)
            delay = min(delay * 2, 30)
        raise RuntimeError(f"backend {method} {path} failed: {last_err}")

    def heartbeat(
        self, firmware_version: str = "", extra: dict | None = None,
    ) -> dict:
        data = {"firmware_version": firmware_version}
        if extra:
            data.update(extra)
        return self._retry(
            "POST", "/heartbeat",
            data=data,
            retries=2,
        )

    def event_trigger(self, session_id: str) -> dict:
        return self._retry(
            "POST", "/event-trigger",
            data={"session_id": session_id},
        )

    def event_stop(self, session_id: str) -> dict:
        # Called by the tee Pi the instant its writer.release()
        # returns, before the (potentially slow) upload. Short
        # retries — if it really can't get through, the green Pi
        # still bails out at its runaway-safety cap.
        return self._retry(
            "POST", "/event-stop",
            data={"session_id": session_id},
            retries=2,
            timeout=10,
        )

    def event_status(self, session_id: str) -> dict:
        # Polled by the green Pi every ~1s while recording so it can
        # mirror the tee's stop decision. Short timeout + retries —
        # the loop will try again next tick on transient failure.
        return self._retry(
            "GET", f"/event-status?session_id={session_id}",
            retries=1,
            timeout=8,
        )

    def poll_trigger(self, timeout_seconds: int = 25) -> dict:
        # HTTP timeout slightly exceeds the server-side long-poll
        # timeout so the request actually gets the response.
        return self._retry(
            "GET", f"/poll-trigger?timeout={timeout_seconds}",
            timeout=timeout_seconds + 10,
            retries=2,
        )

    def upload_event(
        self,
        session_id: str,
        video_path: Path,
        recording_started_at: float | None = None,
    ) -> dict:
        data = {"session_id": session_id}
        # Wall-clock epoch of this clip's first frame. The backend uses
        # the tee/green delta to align the dual-camera cut by real time.
        if recording_started_at is not None:
            data["recording_started_at"] = repr(float(recording_started_at))

        # Hand _retry a factory so it re-opens the file for each attempt.
        # The tee's clips are larger than the green's and can stall long
        # enough to trip the write timeout when the single-core backend is
        # busy transcoding a sibling upload; a clean re-open lets the next
        # attempt actually succeed instead of sending an empty body.
        def _make_files():
            return {"video": (video_path.name, open(video_path, "rb"), "video/mp4")}

        return self._retry(
            "POST", "/upload-event",
            data=data,
            make_files=_make_files,
            timeout=180,
            retries=5,
        )


# ---------------------------------------------------------------------
# Heartbeat thread
# ---------------------------------------------------------------------

class HeartbeatThread(threading.Thread):
    """Daemon thread that pings /heartbeat at a fixed interval so the
    admin UI can see this camera as alive. Failures are warnings,
    not fatal — the main capture loop keeps running even if the
    backend is briefly unreachable."""

    def __init__(
        self, client: BackendClient, interval: int, firmware: str,
        extra_fn=None,
    ):
        super().__init__(daemon=True, name="heartbeat")
        self.client = client
        self.interval = max(15, int(interval))
        self.firmware = firmware
        # Optional callable returning a dict of extra form fields to
        # ride along on each heartbeat (e.g. battery telemetry).
        # Failures inside it must never kill the heartbeat.
        self.extra_fn = extra_fn
        self.stopping = threading.Event()

    def run(self) -> None:
        # First heartbeat immediately so admin UI sees the camera
        # come online without waiting a full interval.
        while not self.stopping.is_set():
            extra = None
            if self.extra_fn is not None:
                try:
                    extra = self.extra_fn()
                except Exception as exc:  # noqa: BLE001
                    log.warning("heartbeat extra_fn failed: %s", exc)
            try:
                self.client.heartbeat(
                    firmware_version=self.firmware, extra=extra,
                )
                log.debug("heartbeat ok")
            except Exception as exc:  # pragma: no cover
                log.warning("heartbeat failed: %s", exc)
            if self.stopping.wait(self.interval):
                return

    def stop(self) -> None:
        self.stopping.set()


# ---------------------------------------------------------------------
# Background upload worker
# ---------------------------------------------------------------------

class BackgroundUploader(threading.Thread):
    """Serialized background upload worker.

    Capture loops enqueue a finished clip and return IMMEDIATELY, so the
    (slow, on cellular) compress + upload never blocks detecting/polling
    for the next event. That blocking was why the green — busy uploading
    the previous clip — missed the start of the next event and recorded a
    short clip out of sync with the tee. With this, the loop is back
    listening within milliseconds, so it catches every trigger and records
    the full window.

    Uploads are processed one at a time (a single worker) so we never run
    several ffmpeg encodes / uploads at once on the Pi. Optional
    `compress_kbps` re-encodes each clip to H.264 before sending.
    """

    def __init__(
        self,
        client: "BackendClient",
        *,
        compress_kbps: int = 0,
        scale_height: Optional[int] = None,
    ):
        super().__init__(daemon=True, name="uploader")
        self.client = client
        self.compress_kbps = int(compress_kbps)
        self.scale_height = scale_height
        self._q: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()

    def enqueue(self, session_id: str, clip_path: Path,
                recording_started_at: Optional[float],
                real_fps: Optional[float] = None) -> None:
        """Hand a finished clip to the worker and return at once.

        `real_fps` (measured delivered rate) re-clocks the clip during
        compression so a frame-dropping capture still plays in real time."""
        self._q.put((session_id, clip_path, recording_started_at, real_fps))
        depth = self._q.qsize()
        if depth > 1:
            log.info("uploader: %d clip(s) queued", depth)

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                session_id, clip_path, ts, real_fps = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if self.compress_kbps > 0:
                    compress_for_upload(
                        clip_path,
                        target_kbps=self.compress_kbps,
                        scale_height=(
                            int(self.scale_height) if self.scale_height else None
                        ),
                        force_input_fps=real_fps,
                    )
                result = self.client.upload_event(
                    session_id, clip_path, recording_started_at=ts,
                )
                log.info(
                    "uploaded: event=%s status=%s ready=%s",
                    result.get("event_id"), result.get("status"),
                    result.get("ready_to_process"),
                )
            except Exception as exc:  # pragma: no cover
                log.error("background upload failed for %s: %s", session_id, exc)
            finally:
                try:
                    clip_path.unlink(missing_ok=True)
                except Exception:
                    pass
                self._q.task_done()

    def stop(self, drain_timeout: float = 0.0) -> None:
        """Signal shutdown. Optionally wait up to drain_timeout for any
        queued uploads to finish before returning."""
        if drain_timeout > 0:
            deadline = time.time() + drain_timeout
            while not self._q.empty() and time.time() < deadline:
                time.sleep(0.2)
        self._stop.set()


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

    # Warm up the sensor before handing the camera to the capture loop.
    # The IMX477 (and most libcamera pipelines) deliver several all-black
    # frames right after open while auto-exposure / auto-white-balance
    # converge — measured ~6 black frames (~0.6s) on the Pi 5 / HQ cam.
    # If those leak through they (a) seed the motion detector's background
    # model with black, so the first real frame reads as a full-frame
    # "lighting shift" and detection is suppressed, and (b) become the
    # first thing the live view shows. Drain frames until the picture
    # actually comes up (non-trivial mean brightness) or a short timeout
    # elapses, so everything downstream only ever sees lit frames.
    warmup_timeout = float(cam_cfg.get("warmup_seconds", 3.0))
    warmup_deadline = time.time() + warmup_timeout
    discarded = 0
    while time.time() < warmup_deadline:
        ok, frame = cap.read()
        if ok and frame is not None and float(frame.mean()) > 5.0:
            break
        discarded += 1
        time.sleep(0.03)
    log.info("camera warmup: discarded %d black frame(s)", discarded)

    return cap


# ---------------------------------------------------------------------
# Audio capture (parallel arecord -> WAV -> ffmpeg mux into MP4)
# ---------------------------------------------------------------------

def _detect_audio_device() -> Optional[str]:
    """Auto-pick an ALSA capture device for the GoPro / USB-mic.

    Reads /proc/asound/cards and returns a plughw:CARD=Name string for
    the first card whose name looks like a real capture interface. Pi
    built-in HDMI cards (vc4hdmi*) and other output-only devices are
    excluded — they open as 'capture' but produce empty WAVs.

    Returns None when no plausible input is attached; callers should
    fall back to silent video rather than picking the wrong device.
    """
    try:
        with open("/proc/asound/cards") as f:
            text = f.read()
    except OSError:
        return None
    # Cards we know are output-only on a Raspberry Pi — never pick
    # these even as a last resort.
    excluded_prefixes = ("vc4hdmi", "headphones", "snd_rpi_")
    preferred = ("hero", "gopro", "uvc", "usb", "mic", "lavalier")
    candidates: list[str] = []
    # Each card is two lines; first line looks like:
    #   " 1 [HERO13Black    ]: USB-Audio - HERO13 Black"
    for m in re.finditer(r"^\s*\d+\s*\[([^\]]+?)\s*\]\s*:.*$", text, re.M):
        name = m.group(1).strip()
        if not name:
            continue
        if any(name.lower().startswith(p) for p in excluded_prefixes):
            continue
        candidates.append(name)
    if not candidates:
        return None
    for name in candidates:
        if any(p in name.lower() for p in preferred):
            return f"plughw:CARD={name}"
    return f"plughw:CARD={candidates[0]}"


class AudioRecorder:
    """Capture audio to a WAV file via `arecord` for the duration of a
    single recording. Designed to bracket the cv2.VideoWriter session
    in tee.py / green.py: start() at writer open, stop() right before
    writer.release(), then mux_audio_into_video() to fold the audio
    into the existing MP4.

    Best-effort: if `arecord` isn't installed, the audio device can't
    be opened, or ffmpeg muxing fails, all methods log a warning and
    return None so the video upload still goes through (just silent).
    """

    def __init__(
        self,
        work_dir: Path,
        device: Optional[str] = None,
        sample_rate: int = 44100,
        channels: int = 1,
        enabled: bool = True,
    ):
        self.work_dir = Path(work_dir)
        self.device = device  # ALSA name, e.g. "plughw:CARD=HERO13Black"
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.enabled = bool(enabled)
        self._proc: Optional[subprocess.Popen] = None
        self._wav_path: Optional[Path] = None

    def start(self, session_id: str) -> Optional[Path]:
        if not self.enabled:
            return None
        # Resolve device if not specified — do this each start so a Pi
        # that gets the GoPro re-plugged mid-session picks the new card.
        device = self.device or _detect_audio_device()
        if not device:
            log.debug("audio: no ALSA capture device found; skipping")
            return None

        wav_path = self.work_dir / f"{session_id}.wav"
        cmd = [
            "arecord",
            "-q",                       # quiet
            "-D", device,
            "-f", "S16_LE",
            "-c", str(self.channels),
            "-r", str(self.sample_rate),
            "-t", "wav",
            str(wav_path),
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            log.warning(
                "audio: arecord not installed — run "
                "`sudo apt install alsa-utils` to enable audio capture",
            )
            return None
        except Exception as exc:
            log.warning("audio: failed to launch arecord: %s", exc)
            return None
        self._wav_path = wav_path
        log.info("audio: capturing via %s -> %s", device, wav_path.name)
        return wav_path

    def stop(self) -> Optional[Path]:
        proc = self._proc
        wav = self._wav_path
        self._proc = None
        if proc is None:
            return None
        try:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        except Exception as exc:
            log.warning("audio: arecord stop failed: %s", exc)
            return None
        if wav is None or not wav.exists() or wav.stat().st_size < 1024:
            log.warning(
                "audio: WAV missing or too small after stop "
                "(%s) — uploading video without audio",
                wav.name if wav else "?",
            )
            return None
        return wav


def mux_audio_into_video(
    video_path: Path, audio_path: Path, audio_delay_seconds: float = 0.0,
) -> bool:
    """Re-encode audio with AAC and stream-copy the existing video into
    a new MP4, replacing the original on success. Returns True on
    success, False on any failure (caller can carry on uploading the
    original silent file).

    `audio_delay_seconds` pads the audio with leading silence so it
    lines up with the right moment in the video. The clip begins with a
    silent pre-roll (buffered frames captured before recording — and
    thus before `arecord` started), so without this the audio plays
    ~pre-roll-length seconds ahead of the picture.

    `-shortest` matches the end of the shorter stream so a slightly-
    longer audio recording doesn't pad the video with black frames.
    """
    if not (video_path.exists() and audio_path.exists()):
        return False
    tmp_out = video_path.with_suffix(".muxed.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
    ]
    # Pad the audio with leading silence so it aligns with the trigger
    # moment instead of the start of the silent pre-roll. adelay pads
    # all channels; requires the audio re-encode above (it does).
    delay_ms = int(round(max(0.0, audio_delay_seconds) * 1000))
    if delay_ms > 0:
        cmd += ["-af", f"adelay={delay_ms}:all=1"]
    cmd += [
        "-movflags", "+faststart",
        "-shortest",
        str(tmp_out),
    ]
    try:
        result = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=60,
        )
    except FileNotFoundError:
        log.warning("audio: ffmpeg not installed — leaving clip silent")
        return False
    except subprocess.TimeoutExpired:
        log.warning("audio: ffmpeg mux timed out — leaving clip silent")
        try:
            tmp_out.unlink(missing_ok=True)
        except Exception:
            pass
        return False
    if result.returncode != 0 or not tmp_out.exists():
        log.warning(
            "audio: ffmpeg mux failed (rc=%s): %s",
            result.returncode, (result.stderr or "")[:200],
        )
        try:
            tmp_out.unlink(missing_ok=True)
        except Exception:
            pass
        return False
    try:
        tmp_out.replace(video_path)
    except OSError as exc:
        log.warning("audio: replacing muxed file failed: %s", exc)
        return False
    try:
        audio_path.unlink(missing_ok=True)
    except Exception:
        pass
    return True


def compress_for_upload(
    video_path: Path,
    target_kbps: int = 2500,
    scale_height: Optional[int] = None,
    timeout: int = 240,
    force_input_fps: Optional[float] = None,
) -> bool:
    """Re-encode an MP4 to H.264 at a controlled bitrate to shrink it
    before upload. The raw mp4v clips the agent writes are tens to >100
    MB; on a metered/cellular link those are slow to send (they trip the
    upload write-timeout and lean on retries) and burn through an IoT SIM
    data plan fast. A 2.5 Mbps H.264 re-encode cuts a ~100 MB clip to
    ~10 MB with quality that's plenty for a cosmetic view.

    Replaces the file in place on success. Best-effort: any failure
    (ffmpeg missing, encode error, timeout) leaves the ORIGINAL file
    untouched and returns False, so the upload still happens — just with
    the larger file — rather than dropping the clip.

    `scale_height` (e.g. 720) optionally downscales; leave None to keep
    the capture resolution (safer for the dual-camera composite, which
    pairs this with the tee clip).
    """
    if not video_path.exists() or target_kbps <= 0:
        return False
    tmp_out = video_path.with_suffix(".h264.mp4")
    vf = ["-vf", f"scale=-2:{int(scale_height)}"] if scale_height else []
    # When the capture dropped frames, the clip's header fps overstates the
    # real rate, so it plays too fast / short (green fell to ~18 fps but was
    # stamped 30, ending 25 s before the tee). Re-interpret the input at the
    # measured delivered rate so the output plays in real time and stays
    # length-matched to the paired camera. `-r` BEFORE `-i` reclocks input.
    reclock = (
        ["-r", f"{force_input_fps:.3f}"]
        if force_input_fps and force_input_fps > 0
        else []
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        *reclock,
        "-i", str(video_path),
        *vf,
        "-c:v", "libx264", "-preset", "veryfast",
        "-b:v", f"{int(target_kbps)}k",
        "-maxrate", f"{int(target_kbps * 1.4)}k",
        "-bufsize", f"{int(target_kbps * 2)}k",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",  # preserve audio track if one was muxed in
        "-movflags", "+faststart",
        str(tmp_out),
    ]
    try:
        result = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        log.warning("compress: ffmpeg not installed — uploading original")
        return False
    except subprocess.TimeoutExpired:
        log.warning("compress: ffmpeg timed out — uploading original")
        tmp_out.unlink(missing_ok=True)
        return False
    if (
        result.returncode != 0
        or not tmp_out.exists()
        or tmp_out.stat().st_size == 0
    ):
        log.warning(
            "compress: ffmpeg failed (rc=%s): %s — uploading original",
            result.returncode, (result.stderr or "")[:200],
        )
        tmp_out.unlink(missing_ok=True)
        return False
    try:
        orig_mb = video_path.stat().st_size / (1024 * 1024)
        tmp_out.replace(video_path)
        new_mb = video_path.stat().st_size / (1024 * 1024)
        log.info(
            "compress: %s %.1f MB -> %.1f MB (H.264 %dk)",
            video_path.name, orig_mb, new_mb, target_kbps,
        )
    except OSError as exc:
        log.warning("compress: replacing file failed: %s", exc)
        return False
    return True


def build_audio_recorder(cfg: dict, work_dir: Path) -> AudioRecorder:
    """Construct an AudioRecorder from the config's `audio:` block.
    Missing block → enabled with auto-detected device + sensible
    defaults; explicit `enabled: false` disables it."""
    a = cfg.get("audio") or {}
    return AudioRecorder(
        work_dir=work_dir,
        device=a.get("device") or None,
        sample_rate=int(a.get("sample_rate", 44100)),
        channels=int(a.get("channels", 1)),
        enabled=bool(a.get("enabled", True)),
    )
