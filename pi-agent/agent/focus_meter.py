"""Focus score, measured from the frames the agent is already reading.

Focusing an IMX477 is a measurement, not a look: there is no autofocus,
the ring is physical, and a preview squinted at on a laptop in sunlight
is how a camera ends up half a turn off. `focus.py` does that job well
and is the right tool while standing at the mount with a screwdriver.

It has one problem, and it is the reason this exists: the agent holds
the camera, so running focus.py means stopping the service. Every check
is a deliberate act on a machine someone has to be logged into. Nobody
does that routinely, so a camera knocked out of focus -- by a lens swap,
a knock, a temperature swing on a long lens -- stays that way until
somebody notices the clips look soft, which is days.

So the agent measures it in passing. The capture loop already has every
frame in hand; this takes the variance of the Laplacian over a region of
one frame every few seconds, which is cheap enough to be invisible next
to the detection already running, and hands it to the heartbeat. The
score then sits on the cameras page next to battery voltage, where a
soft camera is something you notice rather than something you go and
check.

The number is a RELATIVE measure. High is sharper, but its absolute
value depends entirely on what the camera is pointed at -- turf alone
scores low however sharp it is, a tree line scores high. It is useful
two ways: watched over time on one camera, where a drop means something
moved; and turned against in real time at the mount, where the peak is
the answer.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

try:
    import cv2
except ImportError:  # pragma: no cover - the Pi always has it
    cv2 = None  # type: ignore[assignment]

log = logging.getLogger("golfreelz_agent.focus")


class FocusMeter:
    """Rolling sharpness score off the live capture loop.

    `sample(frame)` is called with whatever the loop just read and is a
    no-op except every `interval_seconds`, so the cost is a Laplacian on
    a sub-region a few times a minute rather than per frame.

    The region defaults to the middle-lower half of the frame -- the
    ground and the tree line, not the sky. Sky is featureless and would
    report every camera as hopelessly soft. A camera with a meaningful
    ROI (the tee box) should pass it instead: that is the patch whose
    sharpness actually decides whether a golfer is detected.
    """

    def __init__(
        self,
        interval_seconds: float = 5.0,
        roi: Optional[dict] = None,
    ):
        self.interval = max(1.0, float(interval_seconds))
        self.roi = roi if isinstance(roi, dict) else None
        self._last_at = 0.0
        self._score: Optional[float] = None
        self._bright: Optional[float] = None
        # Focus mode: measure every frame-ish while someone is at the
        # mount turning the ring. Off by default -- the cost is trivial
        # per sample but not worth paying continuously for a number
        # nobody is watching.
        self.fast_interval = 1.0
        self._fast_until = 0.0

    def set_fast_until(self, monotonic_deadline: float) -> None:
        self._fast_until = float(monotonic_deadline)

    def _due(self, now: float) -> bool:
        gap = (self.fast_interval if self._fast_until > now else self.interval)
        return (now - self._last_at) >= gap

    def _region(self, frame):
        h, w = frame.shape[:2]
        r = self.roi
        if r and all(k in r for k in ("x", "y", "w", "h")):
            x, y = int(r["x"]), int(r["y"])
            rw, rh = int(r["w"]), int(r["h"])
            # A stale ROI -- drawn for a different lens, or for a
            # capture mode that has since changed -- must not crash the
            # meter or silently measure a one-pixel sliver.
            x = max(0, min(x, w - 2))
            y = max(0, min(y, h - 2))
            rw = max(16, min(rw, w - x))
            rh = max(16, min(rh, h - y))
            return frame[y:y + rh, x:x + rw]
        return frame[h // 2:h // 2 + h // 3, w // 4:w // 4 + w // 2]

    def sample(self, frame) -> None:
        """Offer a frame. Cheap to call on every one; measures rarely."""
        if cv2 is None or frame is None:
            return
        now = time.monotonic()
        if not self._due(now):
            return
        self._last_at = now
        try:
            region = self._region(frame)
            if region.size == 0:
                return
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            self._score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            self._bright = float(gray.mean())
        except Exception as exc:  # noqa: BLE001
            # A focus reading is a nicety. It must never take the
            # capture loop down with it.
            log.debug("focus: sample failed: %s", exc)

    def read(self) -> Optional[dict]:
        """Latest reading for the heartbeat, or None if nothing yet."""
        if self._score is None:
            return None
        out = {"focus_score": round(self._score, 1)}
        if self._bright is not None:
            out["focus_brightness"] = round(self._bright, 1)
        return out
