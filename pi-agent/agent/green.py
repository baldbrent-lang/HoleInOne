"""Green-side capture agent. Continuously buffers frames in RAM and
long-polls the backend for triggers. When a trigger arrives (sent by
the paired tee's event-trigger call), commits the pre-roll buffer and
keeps recording for `recording_seconds` more, then uploads.

No person detection here — the trigger comes from the tee Pi, so the
green just has to be recording when the ball lands."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import cv2

from .common import BackendClient, FrameBuffer, HeartbeatThread, open_camera
from .livestream import LiveStreamer

log = logging.getLogger("golfreelz_agent.green")
FIRMWARE = "green-0.1.0"


class GreenAgent:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.client = BackendClient(cfg["backend_url"], cfg["auth_token"])
        self.cam_cfg = cfg.get("camera", {})
        self.buffer_seconds = float(cfg.get("buffer_seconds", 5))
        self.recording_seconds = float(cfg.get("recording_seconds", 12))
        self.poll_timeout = int(cfg.get("poll_timeout_seconds", 25))
        self.heartbeat_seconds = int(cfg.get("heartbeat_seconds", 60))
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

        hb = HeartbeatThread(self.client, self.heartbeat_seconds, FIRMWARE)
        hb.start()

        log.info(
            "green agent running: buffer=%.1fs record=%.1fs poll=%ds",
            self.buffer_seconds, self.recording_seconds, self.poll_timeout,
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
        new frames from the buffer until `recording_seconds` of clip
        wall-clock has been captured. Then upload."""
        snapshot = self.buffer.snapshot()
        if not snapshot:
            log.warning("buffer empty at trigger; skipping session=%s", session_id)
            return
        height, width = snapshot[0][1].shape[:2]
        clip_path = self.work_dir / f"{session_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(clip_path), fourcc, self.fps, (width, height))
        if not writer.isOpened():
            log.error("VideoWriter failed for %s", clip_path)
            return

        # Write the pre-roll first.
        for _ts, f in snapshot:
            writer.write(f)
        last_written_ts = snapshot[-1][0]
        frames_written = len(snapshot)
        start = time.time()
        target_seconds = self.recording_seconds
        safety_deadline = start + target_seconds + 5  # +5 s cushion

        # Then continue draining only-new frames from the live buffer
        # until we've covered `target_seconds` of wall-clock from the
        # pre-roll's leading edge.
        target_end_ts = snapshot[0][0] + target_seconds
        while not self.stopping.is_set():
            current = self.buffer.snapshot()
            new_frames = [(ts, f) for ts, f in current if ts > last_written_ts]
            for ts, f in new_frames:
                writer.write(f)
                last_written_ts = ts
                frames_written += 1
            if last_written_ts >= target_end_ts:
                break
            if time.time() > safety_deadline:
                log.warning("recording safety deadline hit; stopping")
                break
            time.sleep(0.05)
        writer.release()
        size = clip_path.stat().st_size if clip_path.exists() else 0
        log.info(
            "recorded %s: %d frames, %.1f MB",
            clip_path.name, frames_written, size / (1024 * 1024),
        )

        try:
            result = self.client.upload_event(session_id, clip_path)
            log.info(
                "uploaded: event=%s status=%s ready=%s",
                result.get("event_id"), result.get("status"),
                result.get("ready_to_process"),
            )
        except Exception as exc:
            log.error("upload_event failed: %s", exc)
        finally:
            clip_path.unlink(missing_ok=True)
