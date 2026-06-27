"""Tee-side capture agent. Reads camera frames continuously into a
ring buffer, runs MediaPipe Pose at N fps on downsampled copies,
triggers a recording when a person dwells in the configured tee-box
ROI for `trigger_dwell_seconds`.

If MediaPipe isn't importable (lighter Pi OS install, ARMv7, etc.)
the agent falls back to a coarse motion-density detector — much less
selective, but lets the operator validate the network plumbing
without the full ML stack."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .common import (
    BackendClient,
    FrameBuffer,
    HeartbeatThread,
    build_audio_recorder,
    mux_audio_into_video,
    open_camera,
)
from .livestream import LiveStreamer

log = logging.getLogger("golfreelz_agent.tee")
FIRMWARE = "tee-0.1.0"

try:
    import mediapipe as mp  # type: ignore
    HAS_MEDIAPIPE = True
except Exception:  # pragma: no cover
    mp = None
    HAS_MEDIAPIPE = False


# ---------------------------------------------------------------------
# Person-in-ROI detectors
# ---------------------------------------------------------------------

class MediaPipePersonDetector:
    """MediaPipe Pose wrapper. Returns the person's bounding-box
    center in native-frame pixel coords on each detect() call, or
    None if no person is found."""

    def __init__(self, detect_width: int = 320):
        self.detect_width = detect_width
        self.pose = mp.solutions.pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.5,
        )

    def detect(self, frame) -> Optional[tuple[int, int]]:
        h, w = frame.shape[:2]
        if w > self.detect_width:
            scale = self.detect_width / float(w)
            small = cv2.resize(frame, (self.detect_width, int(round(h * scale))))
        else:
            scale = 1.0
            small = frame
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return None
        sh, sw = small.shape[:2]
        xs = [lm.x for lm in result.pose_landmarks.landmark]
        ys = [lm.y for lm in result.pose_landmarks.landmark]
        cx = (sum(xs) / len(xs)) * sw / scale
        cy = (sum(ys) / len(ys)) * sh / scale
        return int(round(cx)), int(round(cy))

    def close(self):
        try:
            self.pose.close()
        except Exception:
            pass


class MotionFallbackDetector:
    """Fallback when MediaPipe isn't installed: per-frame absolute
    difference against a running background mean, then the centroid
    of the high-motion mask. Much noisier than pose detection but
    keeps the rest of the loop testable on a stock Pi OS without
    the heavy ML deps."""

    def __init__(self, detect_width: int = 320):
        self.detect_width = detect_width
        self.bg: Optional[np.ndarray] = None
        self.alpha = 0.05  # background learning rate

    def detect(self, frame) -> Optional[tuple[int, int]]:
        h, w = frame.shape[:2]
        if w > self.detect_width:
            scale = self.detect_width / float(w)
            small = cv2.resize(frame, (self.detect_width, int(round(h * scale))))
        else:
            scale = 1.0
            small = frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.bg is None:
            self.bg = gray
            return None
        diff = cv2.absdiff(gray, self.bg)
        self.bg = (1 - self.alpha) * self.bg + self.alpha * gray
        # Higher per-pixel threshold ignores sensor noise; a pixel has to
        # really change, not just flicker.
        _, mask = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask.astype(np.uint8))
        n_changed = 0 if coords is None else len(coords)
        total_px = small.shape[0] * small.shape[1]
        # Reject two kinds of false positives that plagued a blank-wall
        # scene: (1) too FEW changed pixels = noise, not a person; (2) too
        # MANY = a global lighting / auto-exposure shift hitting the whole
        # frame at once, also not a person. A real person at the tee sits
        # comfortably between these bounds.
        if n_changed < 600 or n_changed > 0.40 * total_px:
            return None
        cx = float(coords[:, 0, 0].mean())
        cy = float(coords[:, 0, 1].mean())
        sh, sw = small.shape[:2]
        return int(round(cx / scale)), int(round(cy / scale))

    def close(self):
        pass


def _in_roi(pt: tuple[int, int], roi: dict) -> bool:
    x, y = pt
    return (
        roi["x"] <= x <= roi["x"] + roi["w"]
        and roi["y"] <= y <= roi["y"] + roi["h"]
    )


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------

class TeeAgent:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.client = BackendClient(cfg["backend_url"], cfg["auth_token"])
        self.roi = cfg["tee_box_roi"]
        self.cam_cfg = cfg.get("camera", {})
        self.det_cfg = cfg.get("detection", {})
        self.buffer_seconds = float(cfg.get("buffer_seconds", 5))
        # Runaway-safety cap. A normal session ends when the tee box
        # has been empty for no_person_timeout_seconds; this cap is
        # only ever hit if the detector misclassifies something (eg a
        # parked cart) as a person for an extended period. 10 min is
        # long enough for a foursome on a slow tee box.
        self.max_clip_seconds = float(cfg.get("max_clip_seconds", 600))
        self.heartbeat_seconds = int(cfg.get("heartbeat_seconds", 60))
        self.work_dir = Path(cfg.get("work_dir", "/tmp/golfreelz-tee"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.stopping = threading.Event()

    def stop(self) -> None:
        self.stopping.set()

    def _capture_loop(self, cap) -> None:
        """Dedicated capture thread — the ONLY place cap.read() runs.
        Drains the camera into the ring buffer at full frame rate and
        keeps the live view fresh. Because nothing in here blocks (no
        detection, no disk writes), the camera is never starved, so no
        frames get dropped — that starvation was the source of the
        recorded-clip stutter when detection ran inline."""
        prev = None
        while not self.stopping.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            fcopy = frame.copy()
            # Drop exact-duplicate reads. The GoPro in USB-webcam mode
            # hands cv2 the same frame again ~once/second (the read loop
            # outpaces the sensor's true ~29 fps). Buffered + written at
            # a fixed fps, each repeat is a 1-frame freeze then a
            # double-step jump. Genuine captures always differ by at
            # least sensor noise, so byte-identical means a re-read —
            # safe to skip so the clip stays smooth.
            if (
                prev is not None
                and fcopy.shape == prev.shape
                and np.array_equal(fcopy, prev)
            ):
                continue
            prev = fcopy
            ts = time.time()
            self.buffer.push(ts, fcopy)
            streamer = getattr(self, "streamer", None)
            if streamer is not None:
                streamer.update_frame(frame)

    def run(self) -> None:
        cap = open_camera(self.cam_cfg)
        fps = float(self.cam_cfg.get("fps", 30))
        self.fps = fps
        self.buffer = FrameBuffer(self.buffer_seconds, fps)

        det_width = int(self.det_cfg.get("detect_width", 320))
        det_fps = float(self.det_cfg.get("fps", 5))
        # Seconds between detection samples. Detection no longer gates
        # the capture rate, so this is purely how often we re-check for
        # a person — independent of the recorded frame rate.
        det_period = 1.0 / max(0.5, det_fps)
        dwell_seconds = float(self.det_cfg.get("trigger_dwell_seconds", 2))
        no_person_timeout = float(self.det_cfg.get("no_person_timeout_seconds", 5))

        if HAS_MEDIAPIPE:
            detector = MediaPipePersonDetector(detect_width=det_width)
            log.info("using MediaPipe Pose person detector")
        else:
            detector = MotionFallbackDetector(detect_width=det_width)
            log.warning(
                "mediapipe not importable — falling back to motion-density "
                "detector. Install mediapipe for production accuracy.",
            )

        hb = HeartbeatThread(self.client, self.heartbeat_seconds, FIRMWARE)
        hb.start()

        streamer = LiveStreamer(self.client)
        streamer.start()
        self.streamer = streamer

        # Start draining the camera into the buffer before we begin
        # detecting, so the pre-roll is already populated at trigger.
        capture_thread = threading.Thread(
            target=self._capture_loop, args=(cap,),
            daemon=True, name="tee-capture",
        )
        capture_thread.start()

        person_first_seen: Optional[float] = None
        log.info(
            "tee agent running: roi=%s buffer=%.1fs dwell=%.1fs",
            self.roi, self.buffer_seconds, dwell_seconds,
        )
        try:
            while not self.stopping.is_set():
                snap = self.buffer.snapshot()
                if not snap:
                    time.sleep(0.05)
                    continue
                # Detect on the most recent buffered frame. Runs in this
                # thread, but the capture thread keeps filling the buffer
                # meanwhile, so a slow detect() can't drop frames.
                frame = snap[-1][1]
                now = time.time()
                centroid = detector.detect(frame)
                in_roi = centroid is not None and _in_roi(centroid, self.roi)
                if in_roi:
                    if person_first_seen is None:
                        person_first_seen = now
                        log.info("person entered ROI at %s", centroid)
                    elif now - person_first_seen >= dwell_seconds:
                        session_id = str(uuid.uuid4())
                        log.info("trigger: session=%s", session_id)
                        try:
                            self.client.event_trigger(session_id)
                        except Exception as exc:
                            log.error(
                                "event_trigger failed (%s) — skipping", exc,
                            )
                            person_first_seen = None
                        else:
                            self._record_and_upload(
                                detector, no_person_timeout, det_period,
                                fps, session_id,
                            )
                            person_first_seen = None
                            self.buffer.clear()
                else:
                    if person_first_seen is not None:
                        log.debug("person left ROI before dwell threshold")
                    person_first_seen = None
                time.sleep(det_period)
        finally:
            self.stopping.set()
            capture_thread.join(timeout=2)
            cap.release()
            detector.close()
            hb.stop()
            streamer.stop()

    # -----------------------------------------------------------------

    def _record_and_upload(
        self, detector, no_person_timeout: float, det_period: float,
        fps: float, session_id: str,
    ) -> None:
        """Persist the pre-roll buffer + keep recording until no person
        is seen for `no_person_timeout` seconds (capped by
        `max_clip_seconds`), then upload. Frames come from the shared
        ring buffer (filled by the capture thread) — this method never
        touches the camera directly, so detection here can't stall
        capture and drop frames."""
        snapshot = self.buffer.snapshot()
        if not snapshot:
            log.warning("buffer empty at trigger; skipping session=%s", session_id)
            return
        height, width = snapshot[0][1].shape[:2]

        # Wall-clock time of the first frame in the clip (the start of
        # the committed pre-roll). Reported on upload so the backend can
        # align the dual-camera cut to the tee/green real-time delta.
        first_frame_ts = snapshot[0][0]

        # Write at the camera's REAL delivered rate, measured from the
        # (already de-duplicated) pre-roll timestamps, instead of the
        # nominal config fps. The GoPro webcam delivers ~29 unique fps,
        # so stamping a fixed 30 plays the clip slightly fast and — when
        # combined with the now-removed duplicates — was the source of
        # the periodic hitch. Measured rate => smooth, correct-duration
        # playback. Clamped to sane bounds; falls back to nominal when
        # the pre-roll is too short to measure.
        write_fps = fps
        if len(snapshot) >= 5:
            span = snapshot[-1][0] - snapshot[0][0]
            if span > 0.5:
                write_fps = max(1.0, min(120.0, (len(snapshot) - 1) / span))
        log.info(
            "record: nominal_fps=%.1f measured_fps=%.2f preroll_frames=%d",
            fps, write_fps, len(snapshot),
        )

        clip_path = self.work_dir / f"{session_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(clip_path), fourcc, write_fps, (width, height))
        if not writer.isOpened():
            log.error("VideoWriter failed to open for %s", clip_path)
            return

        # Kick off parallel audio capture. The WAV runs alongside the
        # video write and gets ffmpeg-muxed into the MP4 at release(),
        # so the uploaded clip has audio. Best-effort: if arecord
        # isn't installed or the device can't be opened, the agent
        # carries on with silent video.
        audio_recorder = build_audio_recorder(self.cfg, self.work_dir)
        audio_recorder.start(session_id)

        # Write the pre-roll, then keep draining newly-captured frames
        # from the buffer (the same pattern the green agent uses).
        for _ts, f in snapshot:
            writer.write(f)
        last_written_ts = snapshot[-1][0]
        n_frames_written = len(snapshot)
        recording_start = time.time()
        last_person_seen = recording_start
        next_det = recording_start + det_period

        while not self.stopping.is_set():
            now = time.time()
            current = self.buffer.snapshot()
            new_frames = [(ts, f) for ts, f in current if ts > last_written_ts]
            for ts, f in new_frames:
                writer.write(f)
                last_written_ts = ts
                n_frames_written += 1
            if now - recording_start > self.max_clip_seconds:
                log.warning("max_clip_seconds hit; stopping")
                break
            # Stop-detection on the latest frame, at the detection
            # cadence — off the capture path, so it can't drop frames.
            if now >= next_det and current:
                next_det = now + det_period
                centroid = detector.detect(current[-1][1])
                if centroid is not None:
                    last_person_seen = now
                elif now - last_person_seen > no_person_timeout:
                    log.info(
                        "no person for %.1fs; stopping (recorded %.1fs)",
                        no_person_timeout, now - recording_start,
                    )
                    break
            time.sleep(0.02)
        writer.release()
        # Tell the backend recording is done before starting the
        # (possibly multi-minute) upload, so the paired green Pi —
        # which is polling /event-status — can release its writer at
        # roughly the same wall-clock moment. Best-effort: if the
        # call fails the green Pi will eventually hit its safety cap.
        try:
            self.client.event_stop(session_id)
        except Exception as exc:
            log.warning("event_stop failed for %s: %s", session_id, exc)

        # Stop the parallel audio capture and fold the WAV into the
        # MP4. Each step is best-effort; failures leave the video
        # untouched (silent) so the upload still succeeds.
        wav_done = audio_recorder.stop()
        if wav_done is not None:
            # The clip opens with a silent pre-roll (buffered before
            # arecord started), so delay the audio by the pre-roll's
            # playback length to line it up with the trigger moment.
            preroll_seconds = len(snapshot) / write_fps if write_fps else 0.0
            muxed = mux_audio_into_video(
                clip_path, wav_done, audio_delay_seconds=preroll_seconds,
            )
            if muxed:
                log.info(
                    "audio: muxed into %s (delay=%.2fs)",
                    clip_path.name, preroll_seconds,
                )

        size = clip_path.stat().st_size if clip_path.exists() else 0
        log.info(
            "recorded %s: %d frames, %.1f MB",
            clip_path.name, n_frames_written, size / (1024 * 1024),
        )

        try:
            result = self.client.upload_event(
                session_id, clip_path, recording_started_at=first_frame_ts,
            )
            log.info(
                "uploaded: event=%s status=%s",
                result.get("event_id"), result.get("status"),
            )
        except Exception as exc:
            log.error("upload_event failed: %s", exc)
        finally:
            clip_path.unlink(missing_ok=True)
