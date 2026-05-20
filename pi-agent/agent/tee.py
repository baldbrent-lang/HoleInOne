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

    def run(self) -> None:
        cap = open_camera(self.cam_cfg)
        fps = float(self.cam_cfg.get("fps", 30))
        buffer = FrameBuffer(self.buffer_seconds, fps)

        det_width = int(self.det_cfg.get("detect_width", 320))
        det_fps = float(self.det_cfg.get("fps", 5))
        det_interval = max(1, int(round(fps / det_fps)))
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

        person_first_seen: Optional[float] = None
        frame_idx = 0
        log.info(
            "tee agent running: roi=%s buffer=%.1fs dwell=%.1fs",
            self.roi, self.buffer_seconds, dwell_seconds,
        )
        try:
            while not self.stopping.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue
                ts = time.time()
                buffer.push(ts, frame.copy())
                streamer.update_frame(frame)

                if frame_idx % det_interval == 0:
                    centroid = detector.detect(frame)
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
                                    "event_trigger failed (%s) — skipping",
                                    exc,
                                )
                                person_first_seen = None
                            else:
                                self._record_and_upload(
                                    cap, buffer, detector, det_interval,
                                    no_person_timeout, fps, session_id,
                                )
                                person_first_seen = None
                                buffer.clear()
                    else:
                        if person_first_seen is not None:
                            log.debug("person left ROI before dwell threshold")
                        person_first_seen = None
                frame_idx += 1
        finally:
            cap.release()
            detector.close()
            hb.stop()
            streamer.stop()

    # -----------------------------------------------------------------

    def _record_and_upload(
        self, cap, buffer: FrameBuffer, detector, det_interval: int,
        no_person_timeout: float, fps: float, session_id: str,
    ) -> None:
        """Persist the pre-roll buffer + keep recording until no
        person is seen for `no_person_timeout` seconds (capped by
        `max_clip_seconds`), then upload."""
        snapshot = buffer.snapshot()
        if not snapshot:
            log.warning("buffer empty at trigger — recording from now")
            ok, sample = cap.read()
            if not ok or sample is None:
                log.error("can't read frame; aborting record")
                return
            height, width = sample.shape[:2]
            snapshot = [(time.time(), sample.copy())]
        else:
            height, width = snapshot[0][1].shape[:2]

        clip_path = self.work_dir / f"{session_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            log.error("VideoWriter failed to open for %s", clip_path)
            return

        for _ts, f in snapshot:
            writer.write(f)
        last_person_seen = time.time()
        recording_start = time.time()
        frames_since_check = 0
        n_frames_written = len(snapshot)

        while not self.stopping.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            writer.write(frame)
            n_frames_written += 1
            now = time.time()
            if now - recording_start > self.max_clip_seconds:
                log.warning("max_clip_seconds hit; stopping")
                break
            frames_since_check += 1
            if frames_since_check >= det_interval:
                frames_since_check = 0
                centroid = detector.detect(frame)
                if centroid is not None:
                    last_person_seen = now
                elif now - last_person_seen > no_person_timeout:
                    log.info(
                        "no person for %.1fs; stopping (recorded %.1fs)",
                        no_person_timeout, now - recording_start,
                    )
                    break
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
