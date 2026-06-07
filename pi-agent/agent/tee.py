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

from .common import BackendClient, FrameBuffer, HeartbeatThread, open_camera
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
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask.astype(np.uint8))
        if coords is None or len(coords) < 200:
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
        self.max_clip_seconds = float(cfg.get("max_clip_seconds", 30))
        self.heartbeat_seconds = int(cfg.get("heartbeat_seconds", 60))
        self.work_dir = Path(cfg.get("work_dir", "/tmp/golfreelz-tee"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.stopping = threading.Event()

    def stop(self) -> None:
        self.stopping.set()

    def run(self) -> None:
        cap = open_camera(self.cam_cfg)
        self.fps = float(self.cam_cfg.get("fps", 30))
        self.buffer = FrameBuffer(self.buffer_seconds, self.fps)

        det_width = int(self.det_cfg.get("detect_width", 320))
        self.det_fps = float(self.det_cfg.get("fps", 5))
        det_interval_sec = 1.0 / max(0.1, self.det_fps)
        dwell_seconds = float(self.det_cfg.get("trigger_dwell_seconds", 2))
        self.no_person_timeout = float(
            self.det_cfg.get("no_person_timeout_seconds", 5)
        )

        if HAS_MEDIAPIPE:
            self.detector = MediaPipePersonDetector(detect_width=det_width)
            log.info("using MediaPipe Pose person detector")
        else:
            self.detector = MotionFallbackDetector(detect_width=det_width)
            log.warning(
                "mediapipe not importable — falling back to motion-density "
                "detector. Install mediapipe for production accuracy.",
            )

        # Livestream + a dedicated capture thread come up first so frames
        # are flowing into the buffer before detection starts. Capture
        # runs on its own thread so that NOTHING the main loop does — a
        # slow event-trigger POST, an upload, an ffmpeg transcode — can
        # stall cap.read() and wedge the V4L2 pipeline. (The green agent
        # has always worked this way; the tee used to read inline, which
        # is why a hung backend call could kill its camera.)
        self.streamer = LiveStreamer(self.client)
        self.streamer.start()
        capture_thread = threading.Thread(
            target=self._capture_loop, args=(cap,),
            daemon=True, name="capture",
        )
        capture_thread.start()

        hb = HeartbeatThread(self.client, self.heartbeat_seconds, FIRMWARE)
        hb.start()

        person_first_seen: Optional[float] = None
        last_detect = 0.0
        log.info(
            "tee agent running: roi=%s buffer=%.1fs dwell=%.1fs",
            self.roi, self.buffer_seconds, dwell_seconds,
        )
        try:
            while not self.stopping.is_set():
                snap = self.buffer.snapshot()
                if not snap:
                    time.sleep(0.02)
                    continue
                ts, frame = snap[-1]
                if ts - last_detect < det_interval_sec:
                    time.sleep(0.01)
                    continue
                last_detect = ts

                centroid = self.detector.detect(frame)
                in_roi = centroid is not None and _in_roi(centroid, self.roi)
                if in_roi:
                    if person_first_seen is None:
                        person_first_seen = ts
                        log.info("person entered ROI at %s", centroid)
                    elif ts - person_first_seen >= dwell_seconds:
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
                            # Blocks this thread (network + record + upload),
                            # but the capture thread keeps the camera alive.
                            self._record_and_upload(session_id)
                            person_first_seen = None
                else:
                    if person_first_seen is not None:
                        log.debug("person left ROI before dwell threshold")
                    person_first_seen = None
        finally:
            self.stopping.set()
            capture_thread.join(timeout=2)
            cap.release()
            self.detector.close()
            hb.stop()
            self.streamer.stop()

    # -----------------------------------------------------------------

    def _capture_loop(self, cap) -> None:
        """Drain the camera into the ring buffer as fast as it delivers
        frames, on its own thread. Decoupling capture from the main
        loop is the whole point: a 30 s hang on a slow backend call no
        longer starves cap.read(), so the camera never wedges."""
        prev = None
        while not self.stopping.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            fcopy = frame.copy()
            # Drop exact-duplicate reads. The sensor's true rate (~29
            # fps) is below this read loop's, so cv2 re-hands the same
            # frame ~once a second. Buffered and written at a fixed fps,
            # each repeat is a 1-frame freeze then a forward jump — the
            # periodic stutter. Real frames always differ by at least
            # sensor noise, so a byte-identical read is a re-read and is
            # safe to skip.
            if (
                prev is not None
                and fcopy.shape == prev.shape
                and np.array_equal(fcopy, prev)
            ):
                continue
            prev = fcopy
            self.buffer.push(time.time(), fcopy)
            self.streamer.update_frame(frame)

    def _record_and_upload(self, session_id: str) -> None:
        """Persist the pre-roll buffer, then keep appending new frames
        drained from the capture thread's buffer until no person is
        seen for `no_person_timeout` seconds (capped by
        `max_clip_seconds`), then upload. Reads frames from the shared
        buffer — never from the camera directly — so it cooperates with
        the capture thread instead of competing with it."""
        snapshot = self.buffer.snapshot()
        if not snapshot:
            log.warning("buffer empty at trigger; skipping session=%s", session_id)
            return
        height, width = snapshot[0][1].shape[:2]

        # Write at the camera's REAL delivered rate, measured from the
        # (now de-duplicated) pre-roll timestamps, not the nominal config
        # fps. The sensor delivers ~29 unique fps; stamping a fixed 30
        # plays the clip slightly fast and is the other half of the
        # periodic stutter. Falls back to nominal when the pre-roll is
        # too short to measure.
        write_fps = self.fps
        if len(snapshot) >= 5:
            span = snapshot[-1][0] - snapshot[0][0]
            if span > 0.5:
                write_fps = max(1.0, min(120.0, (len(snapshot) - 1) / span))
        log.info(
            "record: nominal_fps=%.1f measured_fps=%.2f preroll_frames=%d",
            self.fps, write_fps, len(snapshot),
        )

        clip_path = self.work_dir / f"{session_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(clip_path), fourcc, write_fps, (width, height))
        if not writer.isOpened():
            log.error("VideoWriter failed to open for %s", clip_path)
            return

        for _ts, f in snapshot:
            writer.write(f)
        last_written_ts = snapshot[-1][0]
        n_frames_written = len(snapshot)
        recording_start = time.time()
        last_person_seen = recording_start
        last_detect = 0.0
        det_interval_sec = 1.0 / max(0.1, self.det_fps)

        while not self.stopping.is_set():
            current = self.buffer.snapshot()
            new_frames = [(ts, f) for ts, f in current if ts > last_written_ts]
            for ts, f in new_frames:
                writer.write(f)
                last_written_ts = ts
                n_frames_written += 1

            now = time.time()
            if now - recording_start > self.max_clip_seconds:
                log.warning("max_clip_seconds hit; stopping")
                break
            if new_frames and now - last_detect >= det_interval_sec:
                last_detect = now
                centroid = self.detector.detect(new_frames[-1][1])
                if centroid is not None:
                    last_person_seen = now
                elif now - last_person_seen > self.no_person_timeout:
                    log.info(
                        "no person for %.1fs; stopping (recorded %.1fs)",
                        self.no_person_timeout, now - recording_start,
                    )
                    break
            if not new_frames:
                time.sleep(0.02)
        writer.release()
        size = clip_path.stat().st_size if clip_path.exists() else 0
        log.info(
            "recorded %s: %d frames, %.1f MB",
            clip_path.name, n_frames_written, size / (1024 * 1024),
        )

        try:
            result = self.client.upload_event(session_id, clip_path)
            log.info(
                "uploaded: event=%s status=%s",
                result.get("event_id"), result.get("status"),
            )
        except Exception as exc:
            log.error("upload_event failed: %s", exc)
        finally:
            clip_path.unlink(missing_ok=True)
