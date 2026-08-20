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
    ClipWriter,
    BackendClient,
    BackgroundUploader,
    FrameBuffer,
    HeartbeatThread,
    build_audio_recorder,
    mux_audio_into_video,
    open_camera,
)
from .focus_meter import FocusMeter
from .livestream import LiveStreamer

log = logging.getLogger("golfreelz_agent.green")
FIRMWARE = "green-0.1.0"


class GreenAgent:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.client = BackendClient(cfg["backend_url"], cfg["auth_token"])
        self.cam_cfg = cfg.get("camera", {})
        self.buffer_seconds = float(cfg.get("buffer_seconds", 5))
        # SAME HOLD AS THE TEE, and for the same reason: the trigger
        # says a golfer has walked onto the tee box, not that anyone is
        # about to hit. Nothing reaches the green for another twenty
        # seconds at least, so the front of the clip was pure cost. The
        # green measures the hold from when the trigger REACHES it,
        # which is a long-poll return away from the tee's own start
        # line — close enough that both halves begin together, and the
        # backend aligns the cut on wall clock regardless.
        self.record_delay = float(cfg.get("record_delay_seconds", 10))
        # Runaway-safety cap — recording normally ends on the tee's
        # /event-stop signal, this only kicks in if the signal never
        # arrives. 10 min handles a slow foursome.
        # A BACKSTOP, not the clip length — the green stops when the tee
        # signals it. Kept above the tee's own cap so the tee's stop is
        # always what ends a normal clip, with enough headroom that a
        # slow stop-poll doesn't truncate the green half; if the signal
        # never arrives, this bounds the damage. Moves with the tee's cap:
        # at 120s there, 45s here would have the GREEN cutting every long
        # clip short, which is the one thing this must never do.
        self.max_clip_seconds = float(cfg.get("max_clip_seconds", 150))
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
        # NOT /tmp. The spool (work_dir/pending) holds clips part-way
        # through an upload, sometimes for tens of minutes across
        # link outages -- and /tmp does not survive a reboot. The
        # unit's StateDirectory= creates this owned by the service
        # user, so a config that never mentions work_dir still gets
        # somewhere durable.
        self.work_dir = Path(
            cfg.get("work_dir", "/var/lib/golfreelz-agent"))
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
        self._cap_frames = 0
        self._cap_gaps = 0
        self._cap_worst = 0.0
        self._cap_last = None
        # NB: do NOT cap cv2.setNumThreads here. Capping it to 2
        # starved the mp4v ENCODER — measured 9.9 fps of throughput
        # versus 31 fps with the full pool — and the encoder, not the
        # camera, is the pipeline's bottleneck.

        # Start the livestream helper before the capture thread so the
        # very first frame can be forwarded if an admin happens to be
        # watching.
        self.streamer = LiveStreamer(
            self.client, on_focus_mode=self._on_focus_mode,
        )
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
        # No ROI on the green: it has no tee box, so the meter falls
        # back to the middle-lower half of the frame -- the green and
        # its surrounds rather than the sky.
        _focus = FocusMeter()
        self._focus = _focus

        def _hb_extra():
            out = {}
            if (r := _batt.read_averaged()):
                out["battery_voltage"] = r["voltage"]
                out["battery_current_a"] = r["current_a"]
            if (f := _focus.read()):
                out.update(f)
            return out or None

        hb = HeartbeatThread(
            self.client, self.heartbeat_seconds, FIRMWARE,
            extra_fn=_hb_extra,
        )
        self._hb = hb
        hb.start()

        # Uploads run on a background worker so a slow (cellular) upload
        # never blocks us from listening for the next trigger — that
        # blocking was making the green miss the start of an event and
        # record a short clip out of sync with the tee.
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
            fresh_timeout=int(self.cfg.get("upload_fresh_timeout", 60)),
            idle_timeout=int(self.cfg.get("upload_idle_timeout", 90)),
            patient_timeout=int(self.cfg.get("upload_patient_timeout", 120)),
            # DELIBERATELY LONGER THAN THE TEE'S. Both Pis record the same
            # event and stop within a second of each other, so an equal
            # settle would have them wake into the SAME window and upload
            # simultaneously — turning a fix for contention into a
            # metronome for it. The tee goes first because its clip is
            # the one the tracer needs.
            settle_seconds=float(self.cfg.get("upload_settle_seconds", 240)),
            backoff_base=float(self.cfg.get("upload_backoff_base", 20)),
            backoff_max=float(self.cfg.get("upload_backoff_max", 600)),
            chunked=bool(self.cfg.get("upload_chunked", True)),
            chunk_kb=int(self.cfg.get("upload_chunk_kb", 512)),
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
                _not_before = None
                if self.record_delay > 0:
                    log.info(
                        "holding %.0fs before recording — the ball is not "
                        "coming this way yet", self.record_delay,
                    )
                    _not_before = time.time() + self.record_delay
                    while (
                        not self.stopping.is_set()
                        and time.time() < _not_before
                    ):
                        time.sleep(min(0.2, max(0.0, _not_before - time.time())))
                    if self.stopping.is_set():
                        break
                self._record_and_upload(session_id, not_before=_not_before)
        finally:
            self.stopping.set()
            capture_thread.join(timeout=2)
            cap.release()
            hb.stop()
            # Give any in-flight upload a moment to finish before exit.
            self.uploader.stop(drain_timeout=10.0)
            self.streamer.stop()

    # -----------------------------------------------------------------

    def _on_focus_mode(self, seconds: float) -> None:
        """Backend says focus mode is armed for `seconds` (0 = off).

        Called on every watch-status poll while armed, so the deadline
        keeps sliding forward and the mode ends a beat after the backend
        stops asking -- no explicit "off" message to lose.

        Guarded on both attributes because this arrives from the
        livestream thread, which starts before the meter and the
        heartbeat exist on the green runner.
        """
        deadline = time.monotonic() + float(seconds or 0)
        meter = getattr(self, "_focus", None)
        if meter is not None:
            meter.set_fast_until(deadline)
        hb = getattr(self, "_hb", None)
        if hb is not None:
            hb.set_fast_until(deadline)

    def _capture_loop(self, cap) -> None:
        """Drain the camera into the ring buffer as fast as it'll
        deliver frames. Runs until self.stopping is set."""
        while not self.stopping.is_set():
            ok, frame = cap.read()
            if ok and frame is not None:
                self._focus.sample(frame)
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            # Camera-side delivery stats — see tee.py. Separates "the
            # camera stalled" from "we lost frames" in the log.
            _cnow = time.time()
            if self._cap_last is not None:
                _cgap = _cnow - self._cap_last
                if _cgap > 1.8 / max(1.0, self.fps):
                    self._cap_gaps += 1
                    self._cap_worst = max(self._cap_worst, _cgap)
            self._cap_last = _cnow
            self._cap_frames += 1
            if self.frame_shape is None:
                self.frame_shape = (frame.shape[0], frame.shape[1])
            self.buffer.push(time.time(), frame.copy())
            self.streamer.update_frame(frame)

    def _record_and_upload(
        self, session_id: str, not_before: float | None = None,
    ) -> None:
        """Commit the current ring buffer to an MP4 and keep writing
        new frames from the buffer until the tee Pi signals stop
        (via /event-stop, observed by polling /event-status) or the
        runaway-safety cap is hit. Then upload."""
        snapshot = self.buffer.snapshot()
        if not snapshot:
            log.warning("buffer empty at trigger; skipping session=%s", session_id)
            return
        # The buffer kept filling through the hold, so committing it
        # would put back the seconds we just decided not to record.
        # Never drop it all: the writer sizes itself from a frame.
        if not_before is not None:
            _kept = [f for f in snapshot if f[0] >= not_before]
            _dropped = len(snapshot) - len(_kept)
            snapshot = _kept if len(_kept) >= 2 else snapshot[-2:]
            log.info(
                "record: held %.0fs, dropped %d pre-roll frames",
                self.record_delay, _dropped,
            )
        height, width = snapshot[0][1].shape[:2]
        # First-frame wall-clock time (start of the committed pre-roll).
        # Reported on upload for dual-camera cut alignment.
        first_frame_ts = snapshot[0][0]
        clip_path = self.work_dir / f"{session_id}.mp4"
        clip_writer = ClipWriter(clip_path, self.fps, (width, height))
        if not clip_writer.ok:
            log.error("VideoWriter failed for %s", clip_path)
            return

        # Parallel audio capture; folded into the MP4 at release().
        # Best-effort — see tee.py for the error-handling shape.
        audio_recorder = build_audio_recorder(self.cfg, self.work_dir)
        audio_recorder.start(session_id)
        # One thing at a time — see BackgroundUploader.capture_started.
        self.uploader.capture_started()

        # Hand the pre-roll to the encoder thread. Encoding runs off
        # this loop (see ClipWriter) so a hitch can't stall the drain
        # and wrap the ring buffer — that overflow is how frames get
        # destroyed and clips come out choppy.
        _cap_at_start = self._cap_frames
        _cap_t_start = time.time()
        for _ts, f in snapshot:
            clip_writer.submit(_ts, f)
        last_written_ts = snapshot[-1][0]
        preroll_span = snapshot[-1][0] - snapshot[0][0]
        start = time.time()
        deadline = start + self.max_clip_seconds
        next_stop_check = start + self.stop_poll_interval
        stop_reason = "unknown"

        while not self.stopping.is_set():
            current = self.buffer.snapshot()
            new_frames = [(ts, f) for ts, f in current if ts > last_written_ts]
            for ts, f in new_frames:
                clip_writer.submit(ts, f)
                last_written_ts = ts
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
        clip_writer.close()

        # Stop the parallel audio capture and fold the WAV into the MP4.
        wav_done = audio_recorder.stop()
        if wav_done is not None:
            # Delay audio by the silent pre-roll's playback length so it
            # lines up with the trigger moment, not the start of the
            # pre-roll (arecord only starts once recording begins).
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
        # NO average re-timing. Frame loss is BURSTY: one stall drags
        # the average down, and re-clocking the whole clip to it plays
        # the healthy frames slow — the "slow motion" artifact. The
        # ClipWriter instead fills gaps by holding the previous frame,
        # so playback time already equals wall-clock time.
        real_fps = None
        real_span = last_written_ts - first_frame_ts
        frames_written = clip_writer.n_written
        _captured = frames_written - clip_writer.n_filled
        log.info(
            "recorded %s: %d frames, %.1f MB (reason=%s) | timing: "
            "%d captured @ %.1f fps eff, %d gap-filled, %d gap(s), "
            "worst %.0f ms | camera delivered %d frames (%.1f fps, "
            "%d gap(s), worst %.0f ms), queue drops %d, fills skipped %d",
            clip_path.name, frames_written, size / (1024 * 1024),
            stop_reason, _captured,
            (_captured / real_span) if real_span > 0.5 else 0.0,
            clip_writer.n_filled, clip_writer.n_gaps,
            clip_writer.worst_gap * 1000.0,
            self._cap_frames - _cap_at_start,
            (self._cap_frames - _cap_at_start)
            / max(0.001, time.time() - _cap_t_start),
            self._cap_gaps, self._cap_worst * 1000.0,
            clip_writer.n_dropped, clip_writer.n_fills_skipped,
        )
        self._cap_frames = 0
        self._cap_gaps = 0
        self._cap_worst = 0.0

        # Hand the clip to the background uploader and return AT ONCE, so
        # we're back listening for the next trigger immediately. The
        # worker does the (slow, cellular) compress + upload + cleanup off
        # the capture path, so we never miss the start of the next event.
        self.uploader.capture_ended()
        self.uploader.enqueue(session_id, clip_path, first_frame_ts, real_fps=real_fps)
