"""Tap-to-watch live stream for the on-course Pi agent.

Polls the backend's /api/cameras/{token}/watch-status endpoint. When the
operator clicks Watch on /admin/cameras, the backend flips that flag
to true. The Pi then JPEG-encodes its most recent frame and POSTs it
to /api/cameras/{token}/live-frame at ~10 fps until the operator closes
the view (the backend stops returning watching=true after ~10s of no
admin poll, which is the natural stop signal).

Integration points (in tee.py / green.py main loops):

    from .livestream import LiveStreamer

    streamer = LiveStreamer(self.client)
    streamer.start()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None: continue
            streamer.update_frame(frame)   # ← single new line
            # ... existing capture logic
    finally:
        streamer.stop()

The streamer thread is daemonized and self-throttling: zero frame
encoding cost when no one is watching, so it's safe to leave running
forever in the background.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2

log = logging.getLogger("golfreelz_agent.livestream")


class LiveStreamer:
    def __init__(
        self,
        client,
        *,
        fps: int = 10,
        jpeg_quality: int = 65,
        idle_poll_seconds: float = 1.0,
        watched_poll_seconds: float = 5.0,
        on_capture_request=None,
    ) -> None:
        self.client = client
        # Called with (seconds) when the backend asks for an on-demand
        # capture. Delivered on the watch-status poll this thread already
        # makes, so the operator's Capture button needs no new endpoint
        # on the device and no extra polling. Tee cameras pass a handler;
        # green ignores it, because a green records only when its paired
        # tee tells it to.
        self.on_capture_request = on_capture_request
        self.frame_interval = 1.0 / max(1, fps)
        self.jpeg_quality = max(20, min(95, jpeg_quality))
        self.idle_poll = idle_poll_seconds
        self.watched_poll = watched_poll_seconds

        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._watching = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def update_frame(self, frame) -> None:
        """Capture loop calls this on every frame it processes."""
        with self._frame_lock:
            self._latest_frame = frame

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="livestream",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    # -------- internal -----------------------------------------------------

    def _watch_status_url(self) -> str:
        # Reuse BackendClient's tokenized URL builder so we stay in lock-
        # step with the auth scheme used by every other Pi endpoint.
        return self.client._url("/watch-status")

    def _live_frame_url(self) -> str:
        return self.client._url("/live-frame")

    def _run(self) -> None:
        last_poll = 0.0
        last_push = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            poll_every = self.watched_poll if self._watching else self.idle_poll
            if now - last_poll >= poll_every:
                self._poll_watch_status()
                last_poll = now

            if self._watching and now - last_push >= self.frame_interval:
                self._push_frame()
                last_push = now

            time.sleep(0.05)

    def _poll_watch_status(self) -> None:
        try:
            r = self.client.session.get(self._watch_status_url(), timeout=5)
            r.raise_for_status()
            payload = r.json()
            new_state = bool(payload.get("watching"))
            if new_state != self._watching:
                log.info("live-stream %s", "started" if new_state else "stopped")
            self._watching = new_state
            # Consumed server-side on read, so it arrives exactly once.
            secs = payload.get("capture_seconds")
            if secs and self.on_capture_request:
                log.info("capture requested by operator: %ss", secs)
                try:
                    self.on_capture_request(float(secs))
                except Exception as exc:  # noqa: BLE001
                    log.error("capture request handler failed: %s", exc)
        except Exception as e:
            log.debug("watch-status poll failed: %s", e)
            if self._watching:
                log.info("live-stream stopped (poll failed)")
            self._watching = False

    def _push_frame(self) -> None:
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            return
        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            return
        try:
            self.client.session.post(
                self._live_frame_url(),
                data=buf.tobytes(),
                headers={"Content-Type": "image/jpeg"},
                timeout=2,
            )
        except Exception as e:
            # Don't let stream failures interrupt recording.
            log.debug("frame push failed: %s", e)
