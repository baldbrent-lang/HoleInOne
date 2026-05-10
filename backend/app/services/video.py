"""Video compression for email delivery.

Real iPhone / GoPro footage is 30-100+ MB and gets rejected by every major
email provider. We transcode each uploaded clip down to a target file size
so it fits as an attachment.

Targets ~12MB (well below the ~22MB cap for SendGrid + Gmail). Uses ffmpeg
with bitrate-mode H.264 + AAC at 720p max width. Single-pass; if ffmpeg
isn't available the original file is left untouched.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger("golfreelz.video")

DEFAULT_TARGET_MB = 5
AUDIO_KBPS = 96
MAX_VIDEO_KBPS = 4000  # cap so short clips don't waste bytes
MIN_VIDEO_KBPS = 220   # floor so we don't produce unwatchable garbage


def cut_segment(input_path: Path, output_path: Path, start_sec: float, end_sec: float) -> bool:
    """Cut a [start_sec, end_sec] window out of input_path into a new MP4.

    Frame-accurate (`-ss` after `-i`) — slower than fast-seek but reliable
    on the few-minute scrubbed segments this is used for. Re-encodes via
    H.264 + AAC so the output is normalized for downstream compression.
    """
    if not have_ffmpeg():
        log.warning("ffmpeg missing; cannot cut %s", input_path)
        return False
    duration = max(0.1, float(end_sec) - float(start_sec))
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(input_path),
                "-ss", str(float(start_sec)),
                "-t", str(duration),
                "-c:v", "libx264", "-preset", "veryfast",
                "-c:a", "aac", "-b:a", "96k", "-ac", "2",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output_path),
            ],
            check=True,
            timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        log.warning("ffmpeg cut failed for %s: %s", input_path, exc)
        output_path.unlink(missing_ok=True)
        return False
    if not output_path.exists() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        return False
    return True


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def concat_two_clips(
    first_path: Path,
    first_start_sec: float,
    first_end_sec: float,
    second_path: Path,
    second_start_sec: float,
    second_end_sec: float,
    output_path: Path,
    target_height: int = 720,
    target_fps: int = 30,
) -> bool:
    """Stitch a window of one MP4 onto a window of another.

    Used by the dual-camera upload flow to compose tee-cam → green-cam
    into a single deliverable clip. Both segments are normalized to the
    same height / fps / pixel format before concat so the join is clean
    even if the source cameras shot at different framerates or
    resolutions. Audio is dropped.

    Returns True on success, False if ffmpeg is missing or the encode
    failed; output_path is removed on failure.
    """
    if not have_ffmpeg():
        log.warning("ffmpeg missing; cannot concat clips")
        return False
    if first_end_sec <= first_start_sec or second_end_sec <= second_start_sec:
        log.warning("concat: window has non-positive duration")
        return False
    filter_complex = (
        f"[0:v]trim=start={first_start_sec:.3f}:end={first_end_sec:.3f},"
        f"setpts=PTS-STARTPTS,"
        f"scale=-2:{target_height},fps={target_fps},setsar=1[a];"
        f"[1:v]trim=start={second_start_sec:.3f}:end={second_end_sec:.3f},"
        f"setpts=PTS-STARTPTS,"
        f"scale=-2:{target_height},fps={target_fps},setsar=1[b];"
        f"[a][b]concat=n=2:v=1:a=0[out]"
    )
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(first_path),
                "-i", str(second_path),
                "-filter_complex", filter_complex,
                "-map", "[out]",
                "-c:v", "libx264", "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output_path),
            ],
            check=True,
            timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        log.warning("ffmpeg concat failed: %s", exc)
        output_path.unlink(missing_ok=True)
        return False
    if not output_path.exists() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        return False
    return True


def extract_thumbnail(video_path: Path) -> Path | None:
    """Pull a JPG of the first frame so the player has a poster image
    that matches the clip's opening shot. Returns the path or None.
    """
    if not have_ffmpeg():
        return None
    out = video_path.with_suffix(".jpg")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(video_path),
                "-ss", "00:00:00.001",
                "-frames:v", "1",
                "-q:v", "2",
                "-vf", "scale='min(1280,iw)':-2",
                str(out),
            ],
            check=True,
            timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        log.warning("thumbnail extract failed for %s: %s", video_path.name, exc)
        out.unlink(missing_ok=True)
        return None
    if out.exists() and out.stat().st_size > 0:
        return out
    out.unlink(missing_ok=True)
    return None


def _probe_duration(path: Path) -> Optional[float]:
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json", str(path),
            ],
            stderr=subprocess.STDOUT,
            timeout=20,
        )
        data = json.loads(out)
        return float(data["format"]["duration"])
    except Exception as exc:  # pragma: no cover
        log.warning("ffprobe failed for %s: %s", path, exc)
        return None


def compress_for_email(input_path: Path, target_mb: int = DEFAULT_TARGET_MB) -> bool:
    """Replace input_path in place with a transcoded version targeting target_mb.

    Returns True on success, False if ffmpeg is unavailable or the encode
    failed (in which case the original file is preserved).
    """
    if not have_ffmpeg():
        log.warning("ffmpeg not on PATH; skipping compression for %s", input_path)
        return False

    duration = _probe_duration(input_path)
    if not duration or duration <= 0:
        # Constant-quality fallback if we can't probe duration.
        return _encode(input_path, video_kbps=1500, target_mb=target_mb)

    # Bits-per-second budget = target_size_bits / duration. Subtract audio.
    target_bits = target_mb * 8 * 1024 * 1024
    video_kbps = int(target_bits / duration / 1000) - AUDIO_KBPS
    video_kbps = max(MIN_VIDEO_KBPS, min(MAX_VIDEO_KBPS, video_kbps))

    return _encode(input_path, video_kbps=video_kbps, target_mb=target_mb)


def _encode(input_path: Path, video_kbps: int, target_mb: int) -> bool:
    tmp_out = input_path.with_suffix(input_path.suffix + ".tmp.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(input_path),
                # Cap the long edge at 1280px, preserving aspect, even dims.
                "-vf",
                "scale='if(gt(iw,ih),min(1280,iw),-2)':'if(gt(iw,ih),-2,min(1280,ih))'",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-b:v", f"{video_kbps}k",
                "-maxrate", f"{int(video_kbps * 1.5)}k",
                "-bufsize", f"{video_kbps * 2}k",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", f"{AUDIO_KBPS}k",
                "-ac", "2",
                "-movflags", "+faststart",
                str(tmp_out),
            ],
            check=True,
            timeout=600,
        )
    except subprocess.CalledProcessError as exc:
        log.warning("ffmpeg encode failed for %s: %s", input_path, exc)
        tmp_out.unlink(missing_ok=True)
        return False
    except subprocess.TimeoutExpired:
        log.warning("ffmpeg timed out for %s", input_path)
        tmp_out.unlink(missing_ok=True)
        return False

    if not tmp_out.exists() or tmp_out.stat().st_size == 0:
        tmp_out.unlink(missing_ok=True)
        return False

    # Atomic-ish replace
    tmp_out.replace(input_path)
    out_mb = input_path.stat().st_size / 1024 / 1024
    log.info("compressed %s to %.1fMB (target %dMB)", input_path.name, out_mb, target_mb)
    return True
