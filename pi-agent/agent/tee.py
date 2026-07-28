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
    BackgroundUploader,
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
        self.max_clip_seconds = float(cfg.get("max_clip_seconds", 600))
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

        streamer = LiveStreamer(self.client)
        streamer.start()
        self.streamer = streamer

        # Uploads (compress + send) run on a background worker so a slow
        # cellular upload never blocks us from detecting the next group.
        self.uploader = BackgroundUploader(
            self.client,
            compress_kbps=self.upload_bitrate_kbps,
            scale_height=self.upload_scale_height,
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
            self.uploader.stop(drain_timeout=10.0)
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
        #
        # Guard against a bogus measurement: the first capture after boot
        # (or after a long idle) can have a pre-roll of frames the camera
        # delivered slowly during warmup — e.g. 151 frames spanning 150 s
        # reads as 1 fps. Stamping the whole clip at that rate plays the
        # real ~30 fps action in slow motion (and reports an 8-minute
        # duration). Only trust the measured rate when it's within a sane
        # band of nominal (±40%); otherwise keep nominal.
        write_fps = fps
        if len(snapshot) >= 5:
            span = snapshot[-1][0] - snapshot[0][0]
            if span > 0.5:
                measured = (len(snapshot) - 1) / span
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
        # Frame-gap instrumentation: count capture gaps > 1.8x the
        # nominal period so choppiness is measurable in the journal
        # instead of only visible in playback.
        _period = 1.0 / max(1.0, fps)
        _gap_count = 0
        _worst_gap = 0.0
        _prev_ts = None
        for _t, _f in snapshot:
            if _prev_ts is not None and (_t - _prev_ts) > 1.8 * _period:
                _gap_count += 1
                _worst_gap = max(_worst_gap, _t - _prev_ts)
            _prev_ts = _t
        recording_start = time.time()
        last_person_seen = recording_start
        next_det = recording_start + det_period

        while not self.stopping.is_set():
            now = time.time()
            current = self.buffer.snapshot()
            new_frames = [(ts, f) for ts, f in current if ts > last_written_ts]
            for ts, f in new_frames:
                writer.write(f)
                if _prev_ts is not None and (ts - _prev_ts) > 1.8 * _period:
                    _gap_count += 1
                    _worst_gap = max(_worst_gap, ts - _prev_ts)
                _prev_ts = ts
                last_written_ts = ts
                n_frames_written += 1
            if now - recording_start > self.max_clip_seconds:
                log.warning("max_clip_seconds hit; stopping")
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
                    break
            time.sleep(0.02)
        writer.release()
        # TRUE average rate over the WHOLE recording (the stamped
        # write_fps was measured from the 5s pre-roll only — if the
        # capture rate drifted during a minutes-long clip, constant-
        # rate playback of uneven frames is the 'skipping' judder).
        # The uploader re-times the encode to this rate, same as the
        # green agent has always done — which also keeps the tee's
        # time axis truthful, so the tee->green composite alignment
        # lands where the clocks say it should.
        real_span = last_written_ts - first_frame_ts
        real_fps = None
        if n_frames_written >= 10 and real_span > 0.5:
            _measured_all = (n_frames_written - 1) / real_span
            if 0.5 * fps <= _measured_all <= 1.5 * fps:
                real_fps = _measured_all
        log.info(
            "timing: %d frames over %.2fs (avg %.2f fps, stamped %.2f) — "
            "%d dropped-frame gap(s), worst %.0f ms",
            n_frames_written, real_span,
            ((n_frames_written - 1) / real_span) if real_span > 0.5 else 0.0,
            write_fps, _gap_count, _worst_gap * 1000.0,
        )
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

        # Hand off to the background uploader (compress + send + cleanup)
        # and return to detection AT ONCE, so a slow cellular upload can't
        # make us miss the next group arriving at the tee.
        self.uploader.enqueue(
            session_id, clip_path, first_frame_ts, real_fps=real_fps,
        )
