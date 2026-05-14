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
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger("golfreelz.video")

DEFAULT_TARGET_MB = 5
AUDIO_KBPS = 96
MAX_VIDEO_KBPS = 4000  # cap so short clips don't waste bytes
MIN_VIDEO_KBPS = 220   # floor so we don't produce unwatchable garbage


def cut_segment(
    input_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    fast: bool = False,
) -> bool:
    """Cut a [start_sec, end_sec] window out of input_path into a new MP4.

    Default mode is frame-accurate (`-ss` after `-i`) and re-encodes via
    H.264 + AAC so the output is normalized for downstream compression
    — slower but reliable on the few-minute scrubbed segments this is
    used for.

    `fast=True` switches to fast-seek + stream copy (`-ss` before `-i`,
    `-c copy`). The seek snaps to the nearest keyframe (usually within
    a GOP — ~0–2 s of the requested start), but the cut is 50–100×
    faster because no decode/re-encode happens. Use this for preview
    cuts where the operator is just eyeballing segmentation, not for
    deliverables.
    """
    if not have_ffmpeg():
        log.warning("ffmpeg missing; cannot cut %s", input_path)
        return False
    duration = max(0.1, float(end_sec) - float(start_sec))
    if fast:
        # Fast seek: -ss before -i jumps to nearest keyframe without
        # decoding, then -c copy streams bytes through.
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(float(start_sec)),
            "-i", str(input_path),
            "-t", str(duration),
            "-c", "copy",
            "-movflags", "+faststart",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(input_path),
            "-ss", str(float(start_sec)),
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "96k", "-ac", "2",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]
    try:
        subprocess.run(cmd, check=True, timeout=600)
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


def mux_audio_into_video(video_path: Path, audio_source: Path) -> bool:
    """Replace video_path's audio with the audio track from audio_source.

    Used after cv2.VideoWriter renders something (which strips audio):
    we mux the original clip's audio back in so the deliverable plays
    with sound. Video stream is copied (no re-encode) — only audio is
    transcoded to AAC. Idempotent — writes to a temp file then atomically
    renames over video_path. Returns False on any failure (ffmpeg
    missing, no audio in source, etc.) and leaves video_path untouched.
    """
    if not have_ffmpeg():
        return False
    if not video_path.exists() or not audio_source.exists():
        return False
    tmp = video_path.with_suffix(".audiomux.tmp.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(video_path),    # input 0 — video we want to keep
                "-i", str(audio_source),  # input 1 — audio we want to add
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "96k", "-ac", "2",
                "-map", "0:v:0",
                "-map", "1:a:0?",  # '?' = optional; OK if audio_source has no audio
                "-shortest",
                "-movflags", "+faststart",
                str(tmp),
            ],
            check=True,
            timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        log.warning("mux_audio_into_video failed for %s: %s", video_path, exc)
        tmp.unlink(missing_ok=True)
        return False
    if not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(video_path)
    return True


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
    resolutions.

    Tries audio-included concat first (both inputs must have an audio
    stream); on failure (e.g. one input is silent) falls back to a
    video-only concat. The dual-camera pipeline mixes audio in via
    mux_audio_into_video upstream, so by the time we get here both
    inputs should have audio.

    Returns True on success, False if ffmpeg is missing or every encode
    attempt failed; output_path is removed on failure.
    """
    if not have_ffmpeg():
        log.warning("ffmpeg missing; cannot concat clips")
        return False
    if first_end_sec <= first_start_sec or second_end_sec <= second_start_sec:
        log.warning("concat: window has non-positive duration")
        return False

    def _attempt(with_audio: bool) -> bool:
        if with_audio:
            filter_complex = (
                f"[0:v]trim=start={first_start_sec:.3f}:end={first_end_sec:.3f},"
                f"setpts=PTS-STARTPTS,"
                f"scale=-2:{target_height},fps={target_fps},setsar=1[v0];"
                f"[1:v]trim=start={second_start_sec:.3f}:end={second_end_sec:.3f},"
                f"setpts=PTS-STARTPTS,"
                f"scale=-2:{target_height},fps={target_fps},setsar=1[v1];"
                f"[0:a]atrim=start={first_start_sec:.3f}:end={first_end_sec:.3f},"
                f"asetpts=PTS-STARTPTS,aresample=44100[a0];"
                f"[1:a]atrim=start={second_start_sec:.3f}:end={second_end_sec:.3f},"
                f"asetpts=PTS-STARTPTS,aresample=44100[a1];"
                f"[v0][a0][v1][a1]concat=n=2:v=1:a=1[out_v][out_a]"
            )
            map_args = ["-map", "[out_v]", "-map", "[out_a]", "-c:a", "aac", "-b:a", "96k", "-ac", "2"]
        else:
            filter_complex = (
                f"[0:v]trim=start={first_start_sec:.3f}:end={first_end_sec:.3f},"
                f"setpts=PTS-STARTPTS,"
                f"scale=-2:{target_height},fps={target_fps},setsar=1[a];"
                f"[1:v]trim=start={second_start_sec:.3f}:end={second_end_sec:.3f},"
                f"setpts=PTS-STARTPTS,"
                f"scale=-2:{target_height},fps={target_fps},setsar=1[b];"
                f"[a][b]concat=n=2:v=1:a=0[out]"
            )
            map_args = ["-map", "[out]"]
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(first_path),
                    "-i", str(second_path),
                    "-filter_complex", filter_complex,
                    *map_args,
                    "-c:v", "libx264", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(output_path),
                ],
                check=True,
                timeout=600,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            log.warning(
                "ffmpeg concat (%s audio) failed: %s",
                "with" if with_audio else "without", exc,
            )
            output_path.unlink(missing_ok=True)
            return False
        if not output_path.exists() or output_path.stat().st_size == 0:
            output_path.unlink(missing_ok=True)
            return False
        return True

    if _attempt(with_audio=True):
        return True
    log.info("concat: retrying without audio (one input may be silent)")
    return _attempt(with_audio=False)


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


def probe_fps(path: Path) -> float | None:
    """Fast FPS lookup — reads the video container header via OpenCV
    (no subprocess) and returns the declared frame rate, or None if
    the file isn't openable or OpenCV isn't installed.

    Use this for cheap per-clip metadata on list views. For full
    codec/duration/frame-count diagnostics use probe_video_info()
    instead.
    """
    try:
        import cv2  # type: ignore  # local import keeps the module
                    # importable on systems without OpenCV.
    except Exception:
        return None
    try:
        cap = cv2.VideoCapture(str(path))
    except Exception:
        return None
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    try:
        fps_f = float(fps)
    except (TypeError, ValueError):
        return None
    return fps_f if fps_f > 0 else None


def probe_source_device(path: Path) -> str | None:
    """Best-effort guess at the recording device from container metadata.

    Returns a short human-readable label like 'iPhone 14 Pro' or
    'GoPro Hero 11', or None when no useful metadata is present.
    Uses ffprobe; if ffmpeg isn't on PATH, returns None silently.
    Cheap enough to call per-row on list endpoints — one ffprobe
    subprocess that only reads container headers.
    """
    if not have_ffmpeg():
        return None
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format_tags:stream_tags",
                "-of", "json", str(path),
            ],
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        data = json.loads(out)
    except Exception:
        return None

    # Flatten tags from the format and every stream into a single dict
    # so the order of iPhone-vs-GoPro probing below doesn't depend on
    # where the tag happens to live in the container.
    tags: dict[str, str] = {}
    fmt_tags = (data.get("format") or {}).get("tags") or {}
    for k, v in fmt_tags.items():
        tags[str(k).lower()] = str(v)
    for stream in data.get("streams") or []:
        for k, v in (stream.get("tags") or {}).items():
            tags.setdefault(str(k).lower(), str(v))

    # iPhone: Apple writes com.apple.quicktime.{make,model,software}.
    # The model is already nicely formatted ("iPhone 15 Pro"), no need
    # to splice with make.
    model = tags.get("com.apple.quicktime.model") or tags.get("model")
    if model:
        return model.strip()

    # GoPro: doesn't write a clean model tag the way Apple does. The
    # encoder / firmware string almost always contains "GoPro". Try to
    # pull out the model number if present, otherwise return "GoPro".
    for v in tags.values():
        if "gopro" in v.lower():
            m = re.search(
                r"GoPro\s+(?:Hero\s*\d+(?:\s+\w+)?|Max|Fusion)", v, re.IGNORECASE,
            )
            if m:
                return m.group(0).strip()
            return "GoPro"

    # Android: com.android.{manufacturer,model,version}
    android_make = tags.get("com.android.manufacturer")
    android_model = tags.get("com.android.model")
    if android_make and android_model:
        return f"{android_make} {android_model}".strip()
    if android_model:
        return android_model.strip()

    # DJI / Sony / Canon / generic — surface a sanitized encoder string
    # only if it doesn't look like a generic muxer (libav/lavf/x264).
    encoder = tags.get("com.apple.quicktime.software") or tags.get("encoder")
    if encoder:
        e = encoder.strip()
        bad = ("lavf", "libavformat", "libav", "x264", "x265", "ffmpeg")
        if not any(b in e.lower() for b in bad) and len(e) < 60:
            return e
    return None


def probe_video_info(path: Path) -> dict:
    """Return a small dict of video diagnostics: codec, fps, nb_frames,
    duration, width, height. Missing fields are None. Used to verify
    the output of the tracer encode pipeline — cv2's mp4v writer can
    produce files whose container duration looks right but whose
    timestamps are bunched up, so we need to see the actual codec +
    frame count to spot it.
    """
    info: dict = {
        "codec": None, "fps": None, "nb_frames": None, "duration": None,
        "width": None, "height": None,
    }
    if not have_ffmpeg():
        return info
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries",
                "stream=codec_name,r_frame_rate,nb_frames,width,height:format=duration",
                "-of", "json", str(path),
            ],
            stderr=subprocess.STDOUT,
            timeout=20,
        )
        data = json.loads(out)
        stream = (data.get("streams") or [{}])[0]
        info["codec"] = stream.get("codec_name")
        rfr = stream.get("r_frame_rate")
        if rfr and "/" in rfr:
            num, den = rfr.split("/", 1)
            try:
                d = float(den)
                info["fps"] = float(num) / d if d else None
            except (TypeError, ValueError):
                info["fps"] = None
        nb = stream.get("nb_frames")
        try:
            info["nb_frames"] = int(nb) if nb is not None else None
        except (TypeError, ValueError):
            info["nb_frames"] = None
        for key in ("width", "height"):
            v = stream.get(key)
            try:
                info[key] = int(v) if v is not None else None
            except (TypeError, ValueError):
                info[key] = None
        dur = (data.get("format") or {}).get("duration")
        try:
            info["duration"] = float(dur) if dur is not None else None
        except (TypeError, ValueError):
            info["duration"] = None
    except Exception as exc:  # pragma: no cover
        log.warning("ffprobe video-info failed for %s: %s", path, exc)
    return info


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
                # Force constant frame rate. cv2's mp4v intermediate
                # sometimes carries weird per-frame timestamps that
                # collapse into a "0→duration in a blink" player
                # experience if we let ffmpeg pass them through.
                "-fps_mode", "cfr",
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
