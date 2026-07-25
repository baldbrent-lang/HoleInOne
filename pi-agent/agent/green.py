"""Green-side capture agent. Continuously buffers frames in RAM and
long-polls the backend for triggers. When a trigger arrives (sent by
the paired tee's event-trigger call), commits the pre-roll buffer and
keeps recording until the tee's /event-stop hits the backend (polled
once per second via /event-status), then uploads.

The duration of one session is therefore variable — it matches what
the tee Pi recorded, which itself ends when the tee box has been
empty for no_person_timeout_seconds. A hard runaway cap
(max_clip_seconds, default 10 min) bounds the recording in case the
stop signal never arrives (tee crash, network split).

No person detection here — the trigger comes from the tee Pi, so the
green just has to be recording when the ball lands."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import cv2

from .common import (
    BackendClient,
    BackgroundUploader,
    FrameBuffer,
    HeartbeatThread,
    build_audio_recorder,
    mux_audio_into_video,
    open_camera,
)
from .livestream import LiveStreamer

log = logging.getLogger("golfreelz_agent.green")
FIRMWARE = "green-0.1.0"


class GreenAgent:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.client = BackendClient(cfg["backend_url"], cfg["auth_token"])
        self.cam_cfg = cfg.get("camera", {})
        self.buffer_seconds = float(cfg.get("buffer_seconds", 5))
        # Runaway-safety cap — recording normally ends on the tee's
        # /event-stop signal, this only kicks in if the signal never
        # arrives. 10 min handles a slow foursome.
        self.max_clip_seconds = float(cfg.get("max_clip_seconds", 600))
        # How often to ask the backend whether the tee has signalled
        # end-of-session. 1 s keeps the green's clip within ~1 s of
        # the tee's clip without spamming the endpoint.
        self.stop_poll_interval = float(cfg.get("stop_poll_interval_seconds", 1.0))
        self.poll_timeout = int(cfg.get("poll_timeout_seconds", 25))
        self.heartbeat_seconds = int(cfg.get("heartbeat_seconds", 60))
        # Compress each clip to H.264 before upload. Green runs on a
        # cellular SIM where raw clips are slow to send and eat the data
        # plan. Green is only the wide/landing angle AND the dual-camera
        # composite already downscales everything to 720p, so uploading
        # green at full 1080p / high bitrate was wasted bandwidth: default
        # to 720p @ 1.5 Mbps — a much smaller file, uploads far faster, and
        # no quality loss the composite would have kept anyway. Override in
        # config; set bitrate 0 to disable compression (fast wired/WiFi).
        self.upload_bitrate_kbps = int(cfg.get("upload_bitrate_kbps", 1500))
        self.upload_scale_height = cfg.get("upload_scale_height", 720)
        self.work_dir = Path(cfg.get("work_dir", "/tmp/golfreelz-green"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.stopping = threading.Event()
        self.buffer: Optional[FrameBuffer] = None
        self.frame_shape: Optional[tuple[int, int]] = None
        self.fps = float(self.cam_cfg.get("fps", 30))

    def stop(self) -> None:
        self.stopping.set()

    def run(self) -> None:
        cap = open_camera(self.cam_cfg)
        self.buffer = FrameBuffer(self.buffer_seconds, self.fps)

        # Start the livestream helper before the capture thread so the
        # very first frame can be forwarded if an admin happens to be
        # watching.
        self.streamer = LiveStreamer(self.client)
        self.streamer.start()

        # Background thread continuously drains the camera into the
        # ring buffer; the main thread does long-polling + recording
        # commits.
        capture_thread = threading.Thread(
            target=self._capture_loop, args=(cap,),
            daemon=True, name="capture",
        )
        capture_thread.start()

        from .battery import BatteryMonitor

        _batt = BatteryMonitor(self.cfg.get("battery"))
        hb = HeartbeatThread(
            self.client, self.heartbeat_seconds, FIRMWARE,
            extra_fn=lambda: (
                {
                    "battery_voltage": r["voltage"],
                    "battery_current_a": r["current_a"],
                }
                if (r := _batt.read_averaged()) else None
            ),
        )
        hb.start()

        # Uploads run on a background worker so a slow (cellular) upload
        # never blocks us from listening for the next trigger — that
        # blocking was making the green miss the start of an event and
        # record a short clip out of sync with the tee.
        self.uploader = BackgroundUploader(
            self.client,
            compress_kbps=self.upload_bitrate_kbps,
            scale_height=self.upload_scale_height,
        )
        self.uploader.start()

        log.info(
            "green agent running: buffer=%.1fs max=%.0fs poll=%ds",
            self.buffer_seconds, self.max_clip_seconds, self.poll_timeout,
        )
        try:
            while not self.stopping.is_set():
                try:
                    response = self.client.poll_trigger(self.poll_timeout)
                except Exception as exc:
                    log.warning("poll_trigger failed: %s — retry in 5s", exc)
                    time.sleep(5)
                    continue
                trigger = response.get("trigger") if response else None
                if trigger is None:
                    continue
                session_id = trigger.get("session_id")
                if not session_id:
                    log.warning("poll_trigger response missing session_id: %s", response)
                    continue
                log.info(
                    "trigger received: session=%s event=%s hole=%s",
                    session_id, trigger.get("event_id"), trigger.get("hole_number"),
                )
                self._record_and_upload(session_id)
        finally:
            self.stopping.set()
            capture_thread.join(timeout=2)
            cap.release()
            hb.stop()
            # Give any in-flight upload a moment to finish before exit.
            self.uploader.stop(drain_timeout=10.0)
            self.streamer.stop()

    # -----------------------------------------------------------------

    def _capture_loop(self, cap) -> None:
        """Drain the camera into the ring buffer as fast as it'll
        deliver frames. Runs until self.stopping is set."""
        while not self.stopping.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            if self.frame_shape is None:
                self.frame_shape = (frame.shape[0], frame.shape[1])
            self.buffer.push(time.time(), frame.copy())
            self.streamer.update_frame(frame)

    def _record_and_upload(self, session_id: str) -> None:
        """Commit the current ring buffer to an MP4 and keep writing
        new frames from the buffer until the tee Pi signals stop
        (via /event-stop, observed by polling /event-status) or the
        runaway-safety cap is hit. Then upload."""
        snapshot = self.buffer.snapshot()
        if not snapshot:
            log.warning("buffer empty at trigger; skipping session=%s", session_id)
            return
        height, width = snapshot[0][1].shape[:2]
        # First-frame wall-clock time (start of the committed pre-roll).
        # Reported on upload for dual-camera cut alignment.
        first_frame_ts = snapshot[0][0]
        clip_path = self.work_dir / f"{session_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(clip_path), fourcc, self.fps, (width, height))
        if not writer.isOpened():
            log.error("VideoWriter failed for %s", clip_path)
            return

        # Parallel audio capture; folded into the MP4 at release().
        # Best-effort — see tee.py for the error-handling shape.
        audio_recorder = build_audio_recorder(self.cfg, self.work_dir)
        audio_recorder.start(session_id)

        # Write the pre-roll first.
        for _ts, f in snapshot:
            writer.write(f)
        last_written_ts = snapshot[-1][0]
        frames_written = len(snapshot)
        start = time.time()
        deadline = start + self.max_clip_seconds
        next_stop_check = start + self.stop_poll_interval
        stop_reason = "unknown"

        while not self.stopping.is_set():
            current = self.buffer.snapshot()
            new_frames = [(ts, f) for ts, f in current if ts > last_written_ts]
            for ts, f in new_frames:
                writer.write(f)
                last_written_ts = ts
                frames_written += 1
            now = time.time()
            if now > deadline:
                stop_reason = "max_clip_seconds"
                log.warning(
                    "max_clip_seconds (%.0fs) hit before stop signal; stopping",
                    self.max_clip_seconds,
                )
                break
            if now >= next_stop_check:
                next_stop_check = now + self.stop_poll_interval
                try:
                    status = self.client.event_status(session_id)
                except Exception as exc:
                    # Transient network error — log at debug and try
                    # again next interval; we'd rather over-record by
                    # a second than stop early on a flaky link.
                    log.debug("event_status poll failed: %s", exc)
                else:
                    if status.get("stop_signal"):
                        stop_reason = "stop_signal"
                        log.info(
                            "stop signal received for %s after %.1fs",
                            session_id, now - start,
                        )
                        break
            time.sleep(0.05)
        writer.release()

        # Stop the parallel audio capture and fold the WAV into the MP4.
        wav_done = audio_recorder.stop()
        if wav_done is not None:
            # Delay audio by the silent pre-roll's playback length so it
            # lines up with the trigger moment, not the start of the
            # pre-roll (arecord only starts once recording begins).
            preroll_seconds = len(snapshot) / self.fps if self.fps else 0.0
            muxed = mux_audio_into_video(
                clip_path, wav_done, audio_delay_seconds=preroll_seconds,
            )
            if muxed:
                log.info(
                    "audio: muxed into %s (delay=%.2fs)",
                    clip_path.name, preroll_seconds,
                )

        size = clip_path.stat().st_size if clip_path.exists() else 0
        # Actual delivered rate over the whole recording, measured from
        # FRAME timestamps (first buffered frame → last written frame).
        # Green drops frames under load (livestream + cellular upload), so
        # stamping the nominal fps made clips play too fast; re-clock to the
        # measured rate on compress so it plays in real time.
        #
        # Measure against frame timestamps, NOT a wall-clock `start`: an
        # earlier version used `start`, which under-counted the span and
        # inflated the rate (e.g. 42 fps for a 30 fps camera → clips played
        # ~1.4x fast). Cap at nominal — the camera can't deliver faster than
        # its configured fps, so any higher reading is measurement error;
        # floor at 40% so a rare startup glitch can't stamp an absurdly slow
        # clip.
        real_span = last_written_ts - first_frame_ts
        real_fps = None
        if real_span > 1.0 and frames_written >= 5:
            measured = (frames_written - 1) / real_span
            real_fps = min(float(self.fps), max(0.4 * float(self.fps), measured))
        log.info(
            "recorded %s: %d frames, %.1f MB (reason=%s, real_fps=%s)",
            clip_path.name, frames_written, size / (1024 * 1024), stop_reason,
            f"{real_fps:.1f}" if real_fps else "n/a",
        )

        # Hand the clip to the background uploader and return AT ONCE, so
        # we're back listening for the next trigger immediately. The
        # worker does the (slow, cellular) compress + upload + cleanup off
        # the capture path, so we never miss the start of the next event.
        self.uploader.enqueue(session_id, clip_path, first_frame_ts, real_fps=real_fps)
