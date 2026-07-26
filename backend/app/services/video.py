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
import os
import re
import secrets
import shutil
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger("golfreelz.video")

DEFAULT_TARGET_MB = 5
AUDIO_KBPS = 96
MAX_VIDEO_KBPS = 4000  # cap so short clips don't waste bytes
MIN_VIDEO_KBPS = 220   # floor so we don't produce unwatchable garbage

# ── Impact-clip pacing ────────────────────────────────────────────────────────
# These constants define the cut structure for every produced clip.
# Adjust per round via constant edits; each can also be overridden
# at call-time via the corresponding kwargs on splice_impact_clip.
#
# Dual-camera output (12 s at defaults):
#   [← BEFORE_IMPACT →|impact|← TEE_AFTER →||←── GREEN_AFTER ──→]
#   |       TEE video + uninterrupted TEE audio                   |
#                                          | GREEN video          |
#
# Tee-only fallback (9 s at defaults):
#   [← BEFORE_IMPACT →|impact|←──── TEE_ONLY_AFTER ────→]
CLIP_SECONDS_BEFORE_IMPACT: float = 3.0       # tee footage before the impact
CLIP_SECONDS_TEE_AFTER_IMPACT: float = 3.0    # tee footage after impact (→ hard cut to green)
CLIP_SECONDS_GREEN_AFTER_CUT: float = 6.0     # green footage after the camera switch
CLIP_SECONDS_TEE_ONLY_AFTER_IMPACT: float = 6.0  # post-impact duration when green unavailable


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


def make_vertical(
    src: Path,
    out: Path,
    width: int = 1080,
    height: int = 1920,
    focus_x_frac: float = 0.5,
    style: str = "fill",
) -> bool:
    """Render a 9:16 vertical variant of a landscape clip for social /
    phone viewing.

    style="fill" (default): center-crop a 9:16 window out of the frame
    and scale it to fill the whole portrait canvas — no bars, only
    footage. `focus_x_frac` (0..1) aims the crop window horizontally
    (pass the golfer/ball x-fraction so the action stays centered);
    it's clamped so the window never leaves the frame.

    style="blur": the full landscape frame over a blurred zoomed
    backdrop (nothing cropped) — kept as an option.

    Audio passes through. Returns True on success; never raises."""
    if not have_ffmpeg() or not src.exists():
        return False
    if style == "blur":
        filt = (
            f"split[bg][fg];"
            f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},boxblur=22:4,eq=brightness=-0.06[b];"
            f"[fg]scale={width}:-2[f];"
            f"[b][f]overlay=(W-w)/2:(H-h)/2"
        )
    else:
        fx = max(0.0, min(1.0, float(focus_x_frac)))
        # Crop window: full height, width = ih*(W/H); x centered on the
        # focus fraction, clamped inside the frame.
        filt = (
            f"crop=w=ih*{width}/{height}:h=ih:"
            f"x='min(max({fx:.4f}*iw-ow/2,0),iw-ow)':y=0,"
            f"scale={width}:{height},setsar=1"
        )
    tmp = out.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-filter_complex", filt,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-c:a", "aac", "-b:a", "96k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(tmp),
    ]
    try:
        subprocess.run(
            cmd, check=True, timeout=600,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        err = ""
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            err = exc.stderr.decode(errors="replace")[:300]
        log.warning("make_vertical failed for %s: %s %s", src.name, exc, err)
        tmp.unlink(missing_ok=True)
        return False
    if not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(out)
    return True


def make_vertical_pan(
    src: Path,
    out: Path,
    pan_path: list,
    width: int = 1080,
    height: int = 1920,
) -> bool:
    """Full-screen 9:16 vertical with a FOLLOW-THE-ACTION pan: the crop
    window glides horizontally along `pan_path` — a list of
    (time_sec, x_fraction 0..1) waypoints (golfer at address, then the
    ball's tracked flight) — like a cameraman panning with the shot.
    Every frame fills the screen; across the clip the whole shot is
    seen. The raw path is speed-limited (~45% of frame width per
    second) and smoothed (~0.25s) so it reads as a camera move, not a
    twitch. Audio is muxed back in from the source. Returns True on
    success; never raises."""
    try:
        import cv2
        import numpy as np
    except Exception:  # noqa: BLE001
        return False
    if not src.exists() or not pan_path:
        return False
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return False
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cw = int(round(h * width / float(height)))
        if w <= 0 or h <= 0 or n <= 0 or cw >= w or cw < 16:
            return False

        pts = sorted(
            (max(0.0, float(t)), min(1.0, max(0.0, float(x))))
            for t, x in pan_path
        )
        ts = np.arange(n) / fps
        targets = np.interp(
            ts, [p[0] for p in pts], [p[1] for p in pts],
        ) * w
        # Speed limit: the pan chases the target like a camera operator
        # — fast enough to keep the tracer tip centered on a driven
        # ball, never a teleport.
        max_step = 0.75 * w / fps
        centers = np.empty(n)
        cur = float(targets[0])
        for i in range(n):
            cur += float(np.clip(targets[i] - cur, -max_step, max_step))
            centers[i] = cur
        # Smooth kinks (~0.18s box) then clamp inside the frame.
        k = max(1, int(round(0.18 * fps)))
        if k > 1:
            pad = np.concatenate(
                [np.full(k, centers[0]), centers, np.full(k, centers[-1])],
            )
            centers = np.convolve(pad, np.ones(k) / k, mode="same")[k:-k]
        centers = np.clip(centers, cw / 2.0, w - cw / 2.0)

        tmp_v = out.with_suffix(".pan.mp4")
        writer = cv2.VideoWriter(
            str(tmp_v), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (width, height),
        )
        if not writer.isOpened():
            return False
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            c = centers[min(i, n - 1)]
            x0 = int(round(c - cw / 2.0))
            x0 = max(0, min(w - cw, x0))
            crop = frame[:, x0:x0 + cw]
            writer.write(
                cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR),
            )
            i += 1
        writer.release()
        if i == 0 or not tmp_v.exists() or tmp_v.stat().st_size == 0:
            tmp_v.unlink(missing_ok=True)
            return False
    finally:
        cap.release()

    # H.264 + source audio for browser/social playback.
    if not have_ffmpeg():
        tmp_v.replace(out)
        return True
    tmp_o = out.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(tmp_v), "-i", str(src),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-c:a", "aac", "-b:a", "96k",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-shortest",
        str(tmp_o),
    ]
    try:
        subprocess.run(
            cmd, check=True, timeout=600,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        tmp_o.replace(out)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        log.warning("make_vertical_pan encode failed for %s: %s", src.name, exc)
        tmp_o.unlink(missing_ok=True)
        # Fall back to the raw mp4v render rather than nothing.
        tmp_v.replace(out)
        return True
    finally:
        tmp_v.unlink(missing_ok=True)


def transcode_for_web(path: Path) -> bool:
    """Re-encode an MP4 in-place to H.264 + faststart for browser
    playback. The Pi-agent writes its captures with the mp4v fourcc
    (MPEG-4 Part 2), which most modern browsers refuse to play in
    a <video> tag — this forces the file into the universally-
    supported H.264 mp4 form with the moov atom up front (so the
    browser can start playing before the whole file streams).

    Idempotent via a sibling marker file `<name>.h264-ok`. The first
    call transcodes and writes the marker; later calls find the
    marker and short-circuit. This is more reliable than probing the
    codec via ffprobe, which can mis-report cv2-mp4v output as
    something else and either skip a needed transcode or do a
    needless one. Marker has zero bytes — its existence is the signal.
    """
    if not have_ffmpeg():
        return False
    marker = path.with_suffix(path.suffix + ".h264-ok")
    if marker.exists():
        return True
    # Unique tmp name per call. Two concurrent transcodes for the
    # same source (upload_event background thread vs the sync call
    # inside _process_camera_event_job) used to share one tmp path
    # and collide on os.replace, surfacing as "[Errno 2] No such
    # file or directory" once the faster thread had moved the tmp
    # to the source. With a per-call random suffix, both threads
    # finish independently; the later os.replace just overwrites
    # the earlier one's output, which is fine because the content
    # is equivalent.
    tmp = path.with_suffix(f".{secrets.token_hex(4)}.reencode.mp4")
    try:
        # Capture stderr instead of letting it inherit — libav's
        # warnings while reading a Pi-mp4v source are very noisy and
        # would otherwise flood uvicorn's console at 30 fps.
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(path),
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-c:a", "aac", "-b:a", "128k",
                str(tmp),
            ],
            check=True,
            timeout=180,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        log.warning(
            "transcode_for_web failed for %s (exit %s): %s",
            path.name,
            exc.returncode,
            (exc.stderr or b"").decode(errors="replace")[:400],
        )
        tmp.unlink(missing_ok=True)
        return False
    except subprocess.TimeoutExpired as exc:
        log.warning("transcode_for_web timed out for %s: %s", path.name, exc)
        tmp.unlink(missing_ok=True)
        return False
    if not (tmp.exists() and tmp.stat().st_size > 0):
        tmp.unlink(missing_ok=True)
        # If our ffmpeg produced nothing but another concurrent call
        # already finished and dropped the marker, the source is
        # already good — report success.
        return marker.exists()
    try:
        os.replace(tmp, path)
    except FileNotFoundError:
        # Defense-in-depth: even with unique tmp names a stray
        # cleanup pass could have removed our tmp between the
        # exists() check above and the rename. If a marker exists,
        # someone else completed; we just lost the race harmlessly.
        tmp.unlink(missing_ok=True)
        return marker.exists()
    marker.touch()
    return True


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


def splice_impact_clip(
    tee_long_path: Path,
    tee_video_dur_sec: float,
    green_seg_path: "Path | None",
    green_video_dur_sec: float,
    output_path: Path,
    target_height: int = 720,
    target_fps: int = 30,
) -> bool:
    """Build the 12-second impact composite: TEE video then GREEN video
    with the TEE audio track playing uninterrupted for the full duration.

    tee_long_path
        The tee source (tracer-overlaid or raw cut) starting at
        ``impact − CLIP_SECONDS_BEFORE_IMPACT``.  Its audio is used
        for the entire output — including the green-video half — because
        the green camera has no microphone.  Must cover at least
        ``tee_video_dur_sec + green_video_dur_sec`` seconds for the
        audio track.

    tee_video_dur_sec
        How many seconds of tee VIDEO appear in the output.  Equals
        ``actual_before_sec + CLIP_SECONDS_TEE_AFTER_IMPACT`` under
        normal conditions; may be shorter when the impact is within
        CLIP_SECONDS_BEFORE_IMPACT of the recording start.

    green_seg_path
        Green cut starting at the real-world moment
        ``impact + CLIP_SECONDS_TEE_AFTER_IMPACT``.  Pass ``None``
        (or a missing/empty file) for the tee-only fallback.

    green_video_dur_sec
        How many seconds of green VIDEO to append.  Should be clamped
        to the green clip's actual duration before calling.

    Output layout
        Video  : tee_long[0 → tee_video_dur_sec] ++ green_seg[0 → green_video_dur_sec]
        Audio  : tee_long[0 → (tee_video_dur_sec + green_video_dur_sec)]
                 — completely uninterrupted across the camera cut.

    Tee-only fallback
        When green_seg_path is absent/missing, the output is a plain
        tee-only clip trimmed to ``tee_video_dur_sec + green_video_dur_sec``
        (total 9 s at defaults) so the clip still covers the full
        expected duration.

    Returns True on success, False otherwise.  ``output_path`` is
    removed on failure.  Never raises.
    """
    if not have_ffmpeg():
        log.warning("ffmpeg missing; cannot splice impact clip")
        return False

    tee_v = max(0.1, float(tee_video_dur_sec))
    green_v = max(0.1, float(green_video_dur_sec))

    green_available = (
        green_seg_path is not None
        and Path(green_seg_path).exists()
        and Path(green_seg_path).stat().st_size > 0
    )

    if not green_available:
        # Tee-only fallback: just trim the long tee source to cover the
        # full expected window.  The output length equals tee_v + green_v
        # (or the recording end, whichever comes first).
        total_dur = tee_v + green_v
        log.info(
            "splice_impact_clip: no green — tee-only (%.1f s tee + %.1f s extended "
            "= %.1f s total) → %s",
            tee_v, green_v, total_dur, output_path.name,
        )
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(tee_long_path),
            "-t", str(total_dur),
            "-c:v", "libx264", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "96k", "-ac", "2",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=600)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            log.warning("splice_impact_clip tee-only encode failed: %s", exc)
            output_path.unlink(missing_ok=True)
            return False
        if not output_path.exists() or output_path.stat().st_size == 0:
            output_path.unlink(missing_ok=True)
            return False
        return True

    # Dual-camera composite.
    #
    # Video : tee_long[0 → tee_v] ++ green_seg[0 → green_v]  (hard cut)
    # Audio : tee_long[0 → (tee_v + green_v)] — uninterrupted across the cut.
    #
    # The filter_complex handles only video (concat=a=0).  The tee audio
    # is mapped directly from input 0 without an atrim so it runs for the
    # full tee source duration, covering both the tee and green video
    # halves.  If the tee recording ends before the composite video does,
    # the audio simply stops — no padding, no crash.
    log.info(
        "splice_impact_clip: dual-cam %.1f s tee + %.1f s green → %s",
        tee_v, green_v, output_path.name,
    )
    filter_complex = (
        f"[0:v]trim=end={tee_v:.3f},setpts=PTS-STARTPTS,"
        f"scale=-2:{target_height},fps={target_fps},setsar=1[v0];"
        f"[1:v]trim=end={green_v:.3f},setpts=PTS-STARTPTS,"
        f"scale=-2:{target_height},fps={target_fps},setsar=1[v1];"
        f"[v0][v1]concat=n=2:v=1:a=0[out_v]"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(tee_long_path),
        "-i", str(green_seg_path),
        "-filter_complex", filter_complex,
        "-map", "[out_v]",
        "-map", "0:a?",   # tee audio; '?' = keep going if tee has no audio stream
        "-c:v", "libx264", "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "96k", "-ac", "2",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        log.warning("splice_impact_clip dual-cam encode failed: %s", exc)
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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        log.warning(
            "thumbnail extract failed for %s (exit %s): %s",
            video_path.name,
            exc.returncode,
            (exc.stderr or b"").decode(errors="replace")[:400],
        )
        out.unlink(missing_ok=True)
        return None
    except subprocess.TimeoutExpired as exc:
        log.warning("thumbnail extract timed out for %s: %s", video_path.name, exc)
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
