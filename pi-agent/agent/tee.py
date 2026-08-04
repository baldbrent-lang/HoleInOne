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
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .common import (
    BackendClient,
    BackgroundUploader,
    ClipWriter,
    FrameBuffer,
    HeartbeatThread,
    build_audio_recorder,
    mux_audio_into_video,
    open_camera,
)
from .livestream import LiveStreamer

log = logging.getLogger("golfreelz_agent.tee")

# Touch this file to force one capture without a person in the ROI:
#     touch /tmp/golfreelz-trigger
# Exists so the record + upload path can be exercised over SSH from
# anywhere — the alternative is someone standing in front of the camera,
# which is a site visit every time you want to test an upload fix.
FORCE_TRIGGER_PATH = Path("/tmp/golfreelz-trigger")


_last_trigger_mtime = 0.0


def _take_force_trigger() -> bool:
    """True once per `touch`, keyed on MTIME rather than on deleting the
    file.

    Deleting it does not work: the operator touches it as `pi`, the agent
    runs as its own service user, and /tmp is sticky — so only the owner
    may unlink. The first version unlinked, caught the PermissionError,
    logged, and left the file in place: the condition stayed true, so it
    re-fired and re-logged every loop, several times a second, forever.

    Watching mtime is ownership-proof. Each touch bumps it, we fire once,
    and a file we cannot remove is simply ignored until it is touched
    again. Deletion is still attempted, but only as tidying.
    """
    global _last_trigger_mtime
    try:
        mtime = FORCE_TRIGGER_PATH.stat().st_mtime
    except FileNotFoundError:
        return False
    except OSError:
        return False        # unreadable /tmp — stay silent, never spam
    if mtime <= _last_trigger_mtime:
        return False
    _last_trigger_mtime = mtime
    try:
        FORCE_TRIGGER_PATH.unlink()
    except OSError:
        pass                # not ours to delete; mtime already guards us
    return True
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


class YoloPersonDetector:
    """YOLOv8n person detector running on OpenCV's built-in DNN module.

    This is the production detector. It loads a pre-exported ONNX model
    (shipped in the repo at pi-agent/models/) and runs it with cv2.dnn —
    so it needs NO extra Python packages beyond the opencv we already
    depend on. That deliberately sidesteps the torch / ncnn / mediapipe
    wheel problems on newer Pi OS (Trixie / Python 3.13), which is what
    kept knocking us back to the motion detector.

    detect() returns the pixel center of the highest-confidence person
    in native-frame coords, or None. Same contract as the other
    detectors so the main loop doesn't care which one is active.
    """

    # COCO class 0 is "person". The model outputs 4 bbox + 80 class
    # scores per anchor; we only ever look at the person column.
    PERSON_CLASS = 0

    def __init__(
        self,
        model_path,
        input_size: int = 320,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        min_box_area_frac: float = 0.0,
    ):
        self.input_size = int(input_size)
        self.conf = float(conf_threshold)
        self.iou = float(iou_threshold)
        # Reject persons whose box is smaller than this fraction of the
        # frame — lets the operator ignore golfers on a *different* tee
        # far in the background. 0.0 = accept any person in the ROI.
        self.min_box_area_frac = float(min_box_area_frac)
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, frame, conf_override: Optional[float] = None) -> Optional[tuple[int, int]]:
        conf = self.conf if conf_override is None else float(conf_override)
        h, w = frame.shape[:2]
        n = self.input_size
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (n, n), swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        out = self.net.forward()          # (1, 84, anchors)
        out = out[0].T                    # (anchors, 84)
        # Person score is column 4 (first class after the 4 bbox coords).
        person_scores = out[:, 4 + self.PERSON_CLASS]
        keep = person_scores >= conf
        if not np.any(keep):
            return None
        rows = out[keep]
        scores = person_scores[keep]
        min_area_px = self.min_box_area_frac * (w * h)
        boxes, confs = [], []
        for row, sc in zip(rows, scores):
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            bw_px = bw / n * w
            bh_px = bh / n * h
            if bw_px * bh_px < min_area_px:
                continue
            x = (cx - bw / 2) / n * w
            y = (cy - bh / 2) / n * h
            boxes.append([int(x), int(y), int(bw_px), int(bh_px)])
            confs.append(float(sc))
        if not boxes:
            return None
        idxs = cv2.dnn.NMSBoxes(boxes, confs, conf, self.iou)
        if idxs is None or len(idxs) == 0:
            return None
        idxs = np.array(idxs).flatten()
        # Among surviving boxes, return the most confident person's center.
        best = max(idxs, key=lambda i: confs[i])
        x, y, bw_px, bh_px = boxes[best]
        return int(x + bw_px / 2), int(y + bh_px / 2)

    def close(self):
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
        _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        # Open kills isolated speckle (sensor noise); close merges a
        # person's separate motion patches (torso, arms, club) into one
        # coherent blob.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        total_px = mask.shape[0] * mask.shape[1]
        # A global lighting / auto-exposure shift lights up most of the
        # frame at once — never a person.
        if int(cv2.countNonZero(mask)) > 0.40 * total_px:
            return None
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None
        # Key idea: a person is ONE coherent blob; sensor noise is many
        # scattered specks. Take the single largest connected region and
        # require it to be person-sized. Catches a golfer standing back at
        # the tee (small but solid) while ignoring the speckle that made a
        # blank wall false-trigger.
        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) < 300:
            return None
        m = cv2.moments(biggest)
        if m["m00"] == 0:
            return None
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
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
        # Operator-requested capture, seconds, set from the
        # live-stream poll thread and consumed by the main loop.
        self._pending_capture: float | None = None
        self._capture_lock = threading.Lock()
        self.det_cfg = cfg.get("detection", {})
        # Lenient confidence used ONLY to keep an in-progress recording
        # alive. A golfer bent over the ball reads at lower confidence than
        # the strict trigger threshold wants, so keying the "still present"
        # check on the strict threshold dropped them mid-shot (clip ended
        # while they were clearly on the mat). Starting a clip still needs a
        # confident person; staying recorded only needs a faint one.
        self.keepalive_conf = float(
            self.det_cfg.get("keepalive_conf_threshold", 0.2)
        )
        self.buffer_seconds = float(cfg.get("buffer_seconds", 5))
        # Runaway-safety cap. A normal session ends when the tee box
        # has been empty for no_person_timeout_seconds; this cap is
        # only ever hit if the detector misclassifies something (eg a
        # parked cart) as a person for an extended period. 10 min is
        # long enough for a foursome on a slow tee box.
        # CLIP LENGTH IS UPLOAD SIZE. A tee clip runs at ~1500 kbps after
        # compression, so 30s is about 5.5 MB and 143s is 27 MB — and on
        # the link at Snee Farm (measured: 164 KB/s down one minute, 0 the
        # next, 270ms RTT) nothing above about 10 MB completes inside the
        # 180s write timeout. This is not a safety cap any more; it is how
        # long a clip is. A group standing on the tee produces a series of
        # 30s clips that each arrive, instead of one 3-minute clip that
        # never does — and one swing per clip is what produce wants anyway.
        self.max_clip_seconds = float(cfg.get("max_clip_seconds", 30))
        self.heartbeat_seconds = int(cfg.get("heartbeat_seconds", 60))
        # Compress each clip to H.264 at this bitrate (kbps) before upload.
        # Makes clips play in any browser (mp4v won't play in desktop
        # Chrome) and lighter on a cellular SIM. Higher default than the
        # green (3.5 Mbps vs 1.5) because the tee feeds the ball tracer and
        # needs the extra detail. Dropped from 5 Mbps: measured cellular
        # showed the tee is the flakier link (1.5% loss, 450 ms jitter vs
        # green's 0%/204 ms), so the tee's oversized clip was landing last
        # and stalling the pairing. 3.5 Mbps keeps ample tracer detail while
        # cutting the file ~30% so it clears the tee's jittery link sooner.
        # Set to 0 to disable.
        self.upload_bitrate_kbps = int(cfg.get("upload_bitrate_kbps", 3500))
        self.upload_scale_height = cfg.get("upload_scale_height")
        self.work_dir = Path(cfg.get("work_dir", "/tmp/golfreelz-tee"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.stopping = threading.Event()

    def stop(self) -> None:
        self.stopping.set()

    def _resolve_model_path(self) -> Optional[Path]:
        """Find the YOLO ONNX model. Honors detection.model_path in the
        config, else looks for models/yolov8n.onnx alongside the agent
        install (../models relative to this file). Returns None if absent."""
        configured = self.det_cfg.get("model_path")
        if configured:
            p = Path(configured)
            return p if p.exists() else None
        # agent/tee.py -> agent/ -> install root -> models/
        default = Path(__file__).resolve().parent.parent / "models" / "yolov8n.onnx"
        return default if default.exists() else None

    def _build_detector(self, det_width: int):
        """Pick the best available detector: YOLOv8n (OpenCV DNN) first —
        no extra deps, OS-independent — then MediaPipe if installed, then
        the crude motion detector as a last resort."""
        model_path = self._resolve_model_path()
        if model_path is not None:
            try:
                detector = YoloPersonDetector(
                    model_path,
                    input_size=int(self.det_cfg.get("input_size", 320)),
                    conf_threshold=float(self.det_cfg.get("conf_threshold", 0.4)),
                    iou_threshold=float(self.det_cfg.get("iou_threshold", 0.5)),
                    min_box_area_frac=float(
                        self.det_cfg.get("min_box_area_frac", 0.0)
                    ),
                )
                log.info(
                    "using YOLOv8n person detector (OpenCV DNN) — model=%s",
                    model_path,
                )
                return detector
            except Exception as exc:
                log.warning(
                    "YOLO model load failed (%s) — trying next detector", exc,
                )
        if HAS_MEDIAPIPE:
            log.info("using MediaPipe Pose person detector")
            return MediaPipePersonDetector(detect_width=det_width)
        log.warning(
            "no YOLO model and mediapipe not importable — falling back to "
            "motion-density detector (less selective; ship models/yolov8n.onnx "
            "for production accuracy).",
        )
        return MotionFallbackDetector(detect_width=det_width)

    def _request_capture(self, seconds: float) -> None:
        """Operator pressed Capture. Records `seconds` regardless of
        whether anyone is in the ROI, then follows the ordinary trigger
        path — so the clip reaches Production exactly like a real swing
        would. Called from the live-stream poll thread."""
        with self._capture_lock:
            self._pending_capture = float(seconds)

    def _take_pending_capture(self) -> float | None:
        """Consume a pending operator capture, if any."""
        with self._capture_lock:
            secs, self._pending_capture = self._pending_capture, None
        return secs

    def _capture_loop(self, cap) -> None:
        """Dedicated capture thread — the ONLY place cap.read() runs.
        Drains the camera into the ring buffer at full frame rate and
        keeps the live view fresh. Because nothing in here blocks (no
        detection, no disk writes), the camera is never starved, so no
        frames get dropped — that starvation was the source of the
        recorded-clip stutter when detection ran inline."""
        prev = None
        # Watchdog. A wedged V4L2 pipeline (libcamerify losing its buffer
        # queue on open — 'select() timeout' every 10s) returns failure
        # FOREVER, and this loop used to spin on it silently until
        # someone SSHed in. The camera itself is fine: rpicam-still
        # captures normally while the agent sees nothing. So don't
        # diagnose the cause here, just refuse to sit in the broken
        # state — release and reopen until frames come back.
        stall_after = float(self.cam_cfg.get("stall_reopen_seconds", 15.0))
        last_good = time.time()
        reopens = 0
        while not self.stopping.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                if time.time() - last_good > stall_after:
                    reopens += 1
                    log.error(
                        "camera delivered no frames for %.0fs — reopening "
                        "(attempt %d)", stall_after, reopens,
                    )
                    try:
                        cap.release()
                    except Exception:  # noqa: BLE001
                        pass
                    time.sleep(1.0)
                    try:
                        cap = open_camera(self.cam_cfg)
                        self.cap = cap          # run()'s finally releases this one
                        log.info("camera reopened after stall")
                    except Exception as exc:  # noqa: BLE001
                        log.error("camera reopen failed: %s", exc)
                    # Restart the clock either way, so a failed reopen
                    # retries on the next interval instead of hammering.
                    last_good = time.time()
                    prev = None
                time.sleep(0.02)
                continue
            if reopens:
                log.info("camera recovered after %d reopen(s)", reopens)
                reopens = 0
            last_good = time.time()
            # Camera-side delivery stats. Recorded independently of the
            # writer so the timing log can say whether a stall was the
            # CAMERA not delivering frames (thermal/driver) or OUR
            # pipeline failing to keep up — two different fixes.
            _cnow = time.time()
            if self._cap_last is not None:
                _cgap = _cnow - self._cap_last
                if _cgap > 1.8 / max(1.0, self.fps):
                    self._cap_gaps += 1
                    self._cap_worst = max(self._cap_worst, _cgap)
            self._cap_last = _cnow
            self._cap_frames += 1
            fcopy = frame.copy()
            # Drop exact-duplicate reads. The GoPro in USB-webcam mode
            # hands cv2 the same frame again ~once/second (the read loop
            # outpaces the sensor's true ~29 fps). Buffered + written at
            # a fixed fps, each repeat is a 1-frame freeze then a
            # double-step jump. Genuine captures always differ by at
            # least sensor noise, so byte-identical means a re-read —
            # safe to skip so the clip stays smooth.
            # TWO-STAGE duplicate check. Stage 1: sparse grid (every
            # 16th pixel, ~1/256th the cost) — different frames almost
            # always differ here, so the fast path stays fast. Stage 2:
            # only when the sparse grid matches, confirm with the FULL
            # compare — a static scene can look identical on the grid
            # while genuinely differing by sensor noise, and dropping
            # those real frames showed up as ~11% frame loss + ~1s
            # gaps even on a cool, unthrottled Pi. Only true re-reads
            # (byte-identical everywhere) are skipped.
            if (
                prev is not None
                and fcopy.shape == prev.shape
                and np.array_equal(fcopy[::16, ::16], prev[::16, ::16])
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
        # Camera-side delivery counters (see _capture_loop).
        self._cap_frames = 0
        self._cap_gaps = 0
        self._cap_worst = 0.0
        self._cap_last = None
        # NB: do NOT cap cv2.setNumThreads here. Capping it to 2
        # starved the mp4v ENCODER — measured 9.9 fps of throughput
        # versus 31 fps with the full pool — and the encoder, not the
        # camera, is the pipeline's bottleneck.

        det_width = int(self.det_cfg.get("detect_width", 320))
        det_fps = float(self.det_cfg.get("fps", 5))
        # Seconds between detection samples. Detection no longer gates
        # the capture rate, so this is purely how often we re-check for
        # a person — independent of the recorded frame rate.
        det_period = 1.0 / max(0.5, det_fps)
        dwell_seconds = float(self.det_cfg.get("trigger_dwell_seconds", 2))
        no_person_timeout = float(self.det_cfg.get("no_person_timeout_seconds", 5))

        detector = self._build_detector(det_width)

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

        streamer = LiveStreamer(
            self.client, on_capture_request=self._request_capture,
        )
        streamer.start()
        self.streamer = streamer

        # Uploads (compress + send) run on a background worker so a slow
        # cellular upload never blocks us from detecting the next group.
        self.uploader = BackgroundUploader(
            self.client,
            compress_kbps=self.upload_bitrate_kbps,
            scale_height=self.upload_scale_height,
            # Under work_dir on purpose: the spool move must stay on one
            # filesystem or the rename fails across devices.
            spool_dir=self.work_dir / "pending",
            spool_max_mb=int(self.cfg.get("upload_spool_max_mb", 2048)),
            spool_max_age_hours=float(
                self.cfg.get("upload_spool_max_age_hours", 24)),
            fresh_timeout=int(self.cfg.get("upload_fresh_timeout", 120)),
            idle_timeout=int(self.cfg.get("upload_idle_timeout", 300)),
            patient_timeout=int(self.cfg.get("upload_patient_timeout", 300)),
            backoff_base=float(self.cfg.get("upload_backoff_base", 20)),
            backoff_max=float(self.cfg.get("upload_backoff_max", 600)),
        )
        self.uploader.start()

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
                # Manual trigger, for testing the record+upload path
                # without anyone walking into frame:
                #     touch /tmp/golfreelz-trigger
                # Consumed on sight, so it fires exactly once.
                # Operator-requested Capture (admin Cameras page) carries
                # an explicit duration; the SSH sentinel does not.
                pending_secs = self._take_pending_capture()
                forced = _take_force_trigger() or pending_secs is not None

                if not forced:
                    if in_roi:
                        if person_first_seen is None:
                            person_first_seen = now
                            log.info("person entered ROI at %s", centroid)
                    else:
                        if person_first_seen is not None:
                            log.debug("person left ROI before dwell threshold")
                        person_first_seen = None

                dwell_met = (
                    person_first_seen is not None
                    and now - person_first_seen >= dwell_seconds
                )
                if forced or dwell_met:
                    session_id = str(uuid.uuid4())
                    log.info(
                        "trigger: session=%s%s", session_id,
                        (" (CAPTURE %.0fs)" % pending_secs)
                        if pending_secs is not None
                        else (" (MANUAL)" if forced else ""),
                    )
                    try:
                        self.client.event_trigger(session_id)
                    except Exception as exc:
                        log.error("event_trigger failed (%s) — skipping", exc)
                    else:
                        _why = self._record_and_upload(
                            detector, no_person_timeout, det_period,
                            fps, session_id, fixed_seconds=pending_secs,
                        )
                        if _why == "length_cap":
                            # A SPLIT, not an ending. Keep the ring buffer
                            # so the next clip opens with its full pre-roll
                            # and OVERLAPS this one — otherwise a swing
                            # that happens to land on the boundary is cut
                            # in half and neither clip can be produced.
                            # The golfer is still in the ROI, so re-arm the
                            # dwell from now and the next clip starts in
                            # about two seconds.
                            person_first_seen = now
                        else:
                            self.buffer.clear()
                            person_first_seen = None
                time.sleep(det_period)
        finally:
            self.stopping.set()
            capture_thread.join(timeout=2)
            cap.release()
            detector.close()
            hb.stop()
            self.uploader.stop(drain_timeout=10.0)
            streamer.stop()

    # -----------------------------------------------------------------

    def _record_and_upload(
        self, detector, no_person_timeout: float, det_period: float,
        fps: float, session_id: str, fixed_seconds: float | None = None,
    ) -> str:
        """Persist the pre-roll buffer + keep recording until no person
        is seen for `no_person_timeout` seconds (capped by
        `max_clip_seconds`), then upload. Frames come from the shared
        ring buffer (filled by the capture thread) — this method never
        touches the camera directly, so detection here can't stall
        capture and drop frames."""
        snapshot = self.buffer.snapshot()
        if not snapshot:
            log.warning("buffer empty at trigger; skipping session=%s", session_id)
            return "no_buffer"
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
        #
        # Guard against a bogus measurement: the first capture after boot
        # (or after a long idle) can have a pre-roll of frames the camera
        # delivered slowly during warmup — e.g. 151 frames spanning 150 s
        # reads as 1 fps. Stamping the whole clip at that rate plays the
        # real ~30 fps action in slow motion (and reports an 8-minute
        # duration). Only trust the measured rate when it's within a sane
        # band of nominal (±40%); otherwise keep nominal.
        # Use the MEDIAN inter-frame interval, not the mean: a single
        # stall inside the pre-roll drags the mean down and would stamp
        # the whole clip slow. The median is the camera's true cadence.
        write_fps = fps
        if len(snapshot) >= 5:
            _iv = sorted(
                snapshot[i][0] - snapshot[i - 1][0]
                for i in range(1, len(snapshot))
            )
            _med = _iv[len(_iv) // 2]
            if _med > 1e-4:
                measured = 1.0 / _med
                if 0.6 * fps <= measured <= 1.4 * fps:
                    write_fps = max(1.0, min(120.0, measured))
                else:
                    log.warning(
                        "record: ignoring implausible measured_fps=%.2f "
                        "(nominal=%.1f) — using nominal", measured, fps,
                    )
        log.info(
            "record: nominal_fps=%.1f measured_fps=%.2f preroll_frames=%d",
            fps, write_fps, len(snapshot),
        )

        clip_path = self.work_dir / f"{session_id}.mp4"
        clip_writer = ClipWriter(clip_path, write_fps, (width, height))
        if not clip_writer.ok:
            log.error("VideoWriter failed to open for %s", clip_path)
            return "writer_failed"

        # Kick off parallel audio capture. The WAV runs alongside the
        # video write and gets ffmpeg-muxed into the MP4 at release(),
        # so the uploaded clip has audio. Best-effort: if arecord
        # isn't installed or the device can't be opened, the agent
        # carries on with silent video.
        audio_recorder = build_audio_recorder(self.cfg, self.work_dir)
        audio_recorder.start(session_id)

        # Hand the pre-roll to the encoder thread, then keep feeding it
        # newly-captured frames. Submitting is a queue put — the drain
        # loop never waits on encoding, so the ring buffer can't wrap
        # and lose frames while a hitch works itself out. The writer
        # thread owns gap filling (see _ClipWriter).
        _cap_at_start = self._cap_frames
        _cap_t_start = time.time()
        for _ts, f in snapshot:
            clip_writer.submit(_ts, f)
        last_written_ts = snapshot[-1][0]
        # Playback length of the pre-roll == its wall-clock span, since
        # gap filling keeps playback time locked to real time.
        preroll_span = snapshot[-1][0] - snapshot[0][0]
        recording_start = time.time()
        last_person_seen = recording_start
        next_det = recording_start + det_period
        # Why this clip ended decides whether the NEXT one keeps its
        # pre-roll — see the caller.
        stop_reason = "stopping"

        while not self.stopping.is_set():
            now = time.time()
            current = self.buffer.snapshot()
            new_frames = [(ts, f) for ts, f in current if ts > last_written_ts]
            for ts, f in new_frames:
                clip_writer.submit(ts, f)
                last_written_ts = ts
            if fixed_seconds is not None:
                # Operator Capture: record exactly this long. The
                # person-presence stop below is skipped entirely — there
                # is nobody in frame, and stopping on that would give a
                # clip a few seconds long instead of the requested one.
                if now - recording_start >= fixed_seconds:
                    log.info(
                        "capture complete (%.1fs requested)", fixed_seconds,
                    )
                    stop_reason = "fixed"
                    break
                time.sleep(0.02)
                continue
            if now - recording_start > self.max_clip_seconds:
                # NOT an error any more. This is the normal way a clip
                # ends when a group is on the tee: cut here, upload a
                # file small enough to survive the link, and re-trigger
                # for the next stretch. The golfer is still in frame, so
                # the next clip starts within a couple of seconds.
                log.info(
                    "clip length cap (%.0fs) reached — splitting here and "
                    "starting a new clip; the golfer is still in the ROI",
                    self.max_clip_seconds,
                )
                stop_reason = "length_cap"
                break
            # Stop-detection on the latest frame, at the detection
            # cadence — off the capture path, so it can't drop frames.
            if now >= next_det and current:
                next_det = now + det_period
                # Keep-alive uses the LENIENT confidence so a bent-over
                # golfer (low YOLO confidence) isn't dropped mid-shot — the
                # strict trigger confidence already gated starting the clip.
                # Fall back for detectors that don't take the override.
                try:
                    centroid = detector.detect(
                        current[-1][1], conf_override=self.keepalive_conf
                    )
                except TypeError:
                    centroid = detector.detect(current[-1][1])
                if centroid is not None:
                    last_person_seen = now
                elif now - last_person_seen > no_person_timeout:
                    log.info(
                        "no person for %.1fs; stopping (recorded %.1fs)",
                        no_person_timeout, now - recording_start,
                    )
                    stop_reason = "no_person"
                    break
            time.sleep(0.02)
        clip_writer.close()
        # No re-timing on upload: gap filling already made the file a
        # correct constant-rate clip whose duration equals wall-clock.
        real_fps = None
        real_span = last_written_ts - first_frame_ts
        n_frames_written = clip_writer.n_written
        _captured = n_frames_written - clip_writer.n_filled
        # Two rates, deliberately separate:
        #   camera  = what the SENSOR delivered to the capture thread
        #   written = what survived into the clip
        # Camera low  -> the camera/driver stalled (thermal, hardware).
        # Camera fine but written low -> our pipeline lost frames
        # (queue overflow) — a software problem. Never guess again.
        log.info(
            "timing: %d frames over %.2fs (%d captured @ %.1f fps eff, "
            "%d gap-filled, stamped %.2f) — %d gap(s), worst %.0f ms | "
            "camera delivered %d frames (%.1f fps, %d gap(s), worst "
            "%.0f ms), queue drops %d, fills skipped %d",
            n_frames_written, real_span, _captured,
            (_captured / real_span) if real_span > 0.5 else 0.0,
            clip_writer.n_filled, write_fps, clip_writer.n_gaps,
            clip_writer.worst_gap * 1000.0,
            self._cap_frames - _cap_at_start,
            (self._cap_frames - _cap_at_start)
            / max(0.001, time.time() - _cap_t_start),
            self._cap_gaps, self._cap_worst * 1000.0,
            clip_writer.n_dropped, clip_writer.n_fills_skipped,
        )
        # Reset camera counters for the next recording.
        self._cap_frames = 0
        self._cap_gaps = 0
        self._cap_worst = 0.0
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
            preroll_seconds = preroll_span
            muxed = mux_audio_into_video(
                clip_path, wav_done, audio_delay_seconds=preroll_seconds,
            )
            if muxed:
                log.info(
                    "audio: muxed into %s (delay=%.2fs)",
                    clip_path.name, preroll_seconds,
                )

        size = clip_path.stat().st_size if clip_path.exists() else 0
        # The CLIP's length, not the wall clock to this line. Measuring
        # wall clock here included the encoder drain in clip_writer.close()
        # — 40s on a busy Pi — and reported a 35s clip as 59s, which made
        # the 30s cap look broken when it was working exactly right.
        # Frames written at the write rate IS the clip's playback length.
        _dur = (n_frames_written / write_fps) if write_fps else 0.0
        _wall = time.time() - recording_start
        log.info(
            "recorded %s: %d frames, %.1fs of video, %.1f MB "
            "(%.0fs wall clock incl. encoder drain)",
            clip_path.name, n_frames_written, _dur, size / (1024 * 1024),
            _wall,
        )
        # LENGTH IS WHAT DRIVES SIZE, and length here is however long a
        # person stood in the ROI — on a tee box a waiting group can hold
        # it open for minutes. A clip this long will not clear a weak
        # uplink inside the upload timeout however well it compresses, so
        # say so at the moment it is created rather than leaving it to be
        # inferred from an upload that quietly failed five times.
        if _dur > 60.0:
            log.warning(
                "recorded %.0fs of tee video — that is %.0fx a swing and "
                "will be ~%.0f MB after compression at %d kbps. Expect the "
                "upload to struggle: it needs ~%.0f kbps sustained.",
                _dur, _dur / 12.0,
                self.upload_bitrate_kbps * _dur / 8 / 1024,
                self.upload_bitrate_kbps,
                self.upload_bitrate_kbps * _dur / 180.0,
            )

        # Hand off to the background uploader (compress + send + cleanup)
        # and return to detection AT ONCE, so a slow cellular upload can't
        # make us miss the next group arriving at the tee.
        self.uploader.enqueue(
            session_id, clip_path, first_frame_ts, real_fps=real_fps,
        )
        return stop_reason
