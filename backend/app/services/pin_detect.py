"""Find the flagstick's BASE in a green-camera frame.

The base, not the flag. The flag is what we can see; the base is where
the stick meets the putting surface, and that is the only point of the
whole assembly that sits on the ground plane -- which is what makes it
the cup's position and what every distance is then measured from. Aim at
the cloth and you are measuring from a point seven feet in the air,
which the homography will happily convert into a confident, wrong answer.

Two facts about this installation make it tractable:

  * the flag is ALWAYS the same colour, so colour is a strong first cut
  * the camera is FIXED, so yesterday's answer is a good prior for today

The pin moves daily, which is the entire reason this exists, but it
moves within the green -- tens of feet, not hundreds. So a detection far
from the previous one is more likely to be a yellow golf bag than a
genuinely relocated cup, and is reported with low confidence rather than
silently accepted.

Nothing here decides anything. It returns a point and a confidence, and
the caller chooses whether that is good enough to use. A wrong pin does
not produce a visibly wrong picture -- it produces a plausible distance
that hands the prize to the wrong golfer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("golfreelz.pin")

# Yellow flag, in OpenCV HSV (H is 0-179, not 0-359). Deliberately wide:
# the same flag reads very differently at 7am and at noon, and a range
# that only works in flat light is a range that fails every morning.
# Overridable per call so these can be tuned against real frames rather
# than guessed at.
DEFAULT_HSV_LO = (20, 90, 90)
DEFAULT_HSV_HI = (38, 255, 255)

# A flag is a small object in a wide frame. These are fractions of total
# frame area -- anything bigger is a bunker, a roof or the sun on grass.
MIN_AREA_FRAC = 0.000_02      # ~40 px in a 1920x1080 frame
MAX_AREA_FRAC = 0.010

# A standard flagstick is 7 ft; the cloth is roughly the top fifth of it.
# So from the bottom of the flag to the ground is about four times the
# flag's own height. Used only as a FALLBACK when the stick itself
# cannot be traced, and reported at low confidence when it is.
STICK_TO_FLAG_RATIO = 4.0


@dataclass
class PinDetection:
    """Where the base of the flagstick is, and how much to trust it.

    Coordinates are FRACTIONS of frame width/height, matching how
    tee_boxes and ball_sizes are stored -- so a camera swapped to another
    resolution does not silently invalidate every stored pin.
    """

    x: float
    y: float
    confidence: float
    # How the base was arrived at: "stick" means it was traced down the
    # pole to the ground, "estimated" means it was inferred from the
    # flag's height. The first is worth trusting; the second is a hint.
    method: str = "stick"
    flag_box: Optional[tuple[int, int, int, int]] = None   # x,y,w,h in px
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "x": round(self.x, 5),
            "y": round(self.y, 5),
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "flag_box": list(self.flag_box) if self.flag_box else None,
            "notes": self.notes,
        }


def detect_pin(
    frame,
    roi: Optional[tuple[float, float, float, float]] = None,
    prev: Optional[tuple[float, float]] = None,
    hsv_lo: tuple[int, int, int] = DEFAULT_HSV_LO,
    hsv_hi: tuple[int, int, int] = DEFAULT_HSV_HI,
) -> Optional[PinDetection]:
    """Find the flagstick base in one BGR frame.

    `roi` is an (x, y, w, h) rectangle in FRACTIONS, limiting the search
    to the putting surface. Strongly recommended: it is what stops a
    yellow bag on the cart path from becoming today's cup.

    `prev` is yesterday's base, also in fractions. Used only to score
    candidates -- never to override what is actually in the picture.

    Returns None when nothing plausible is found, which is a normal
    outcome (a golfer standing in front of the flag, a frame taken while
    the pin was out) and the reason callers should sample many frames
    rather than trusting one.
    """
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    if not h or not w:
        return None

    # Search window. Everything below is computed inside it and shifted
    # back into full-frame coordinates at the end.
    x0 = y0 = 0
    view = frame
    if roi:
        rx, ry, rw, rh = roi
        x0, y0 = int(rx * w), int(ry * h)
        x1, y1 = int((rx + rw) * w), int((ry + rh) * h)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        view = frame[y0:y1, x0:x1]

    hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8))
    # Open then close: drop single-pixel speckle, then heal the flag back
    # into one blob after the cloth's folds and shadows split it.
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None

    frame_area = float(h * w)
    best = None
    best_score = -1.0
    for i in range(1, n):
        bx, by, bw, bh, area = (
            stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
            stats[i, cv2.CC_STAT_AREA],
        )
        frac = area / frame_area
        if frac < MIN_AREA_FRAC or frac > MAX_AREA_FRAC:
            continue
        # A flag is a compact blob, not a long thin streak of glare on
        # grass. Allow a generous range: wind stretches it sideways.
        ar = bw / float(bh or 1)
        if ar < 0.25 or ar > 4.0:
            continue

        score = 1.0
        # Prefer candidates near the previous pin. A cup moves within a
        # green; it does not cross the frame overnight.
        if prev is not None:
            cx = (x0 + centroids[i][0]) / w
            cy = (y0 + centroids[i][1]) / h
            d = ((cx - prev[0]) ** 2 + (cy - prev[1]) ** 2) ** 0.5
            score *= max(0.15, 1.0 - min(1.0, d / 0.35))
        # Prefer the larger of two similar candidates -- the flag is
        # bigger than whatever yellow speck shares its hue.
        score *= min(1.0, frac / (MIN_AREA_FRAC * 8)) ** 0.25

        if score > best_score:
            best_score = score
            best = (int(bx), int(by), int(bw), int(bh), int(area), i)

    if best is None:
        return None

    bx, by, bw, bh, area, idx = best
    notes: list[str] = []

    base_y = _trace_stick_base(view, mask, bx, by, bw, bh)
    if base_y is None:
        # Could not see the pole. Fall back to flagstick proportions,
        # and say so -- this is the answer worth a human glance.
        base_y = int(by + bh + bh * STICK_TO_FLAG_RATIO)
        method = "estimated"
        notes.append("stick not traced; base inferred from flag height")
    else:
        method = "stick"

    base_y = min(base_y, view.shape[0] - 1)
    # The stick's own x, not the flag's centroid: wind pushes the cloth
    # sideways off the pole, and half a flag-width at this range is feet
    # on the ground.
    base_x = _stick_x(mask, bx, by, bw, bh)

    fx = (x0 + base_x) / w
    fy = (y0 + base_y) / h

    conf = 0.85 if method == "stick" else 0.45
    conf *= min(1.0, max(0.2, best_score))
    if prev is not None:
        d = ((fx - prev[0]) ** 2 + (fy - prev[1]) ** 2) ** 0.5
        if d > 0.25:
            conf *= 0.4
            notes.append(f"far from previous pin ({d:.2f} of frame)")

    return PinDetection(
        x=fx, y=fy, confidence=round(min(1.0, conf), 3), method=method,
        flag_box=(x0 + bx, y0 + by, bw, bh), notes=notes,
    )


def _stick_x(mask, bx: int, by: int, bw: int, bh: int) -> int:
    """Which column the pole is in.

    The cloth hangs to one side of the pole, so the flag's centre is not
    the pole's. The pole is at the flag's attached edge -- the side whose
    column of flag pixels reaches HIGHEST, since the cloth tapers away
    downwind from the top of the stick.
    """
    import numpy as np

    band = mask[by:by + bh, bx:bx + bw]
    if band.size == 0:
        return bx + bw // 2
    # For each column, the topmost lit row. The attached edge is where
    # the flag starts highest.
    cols = []
    for c in range(band.shape[1]):
        col = np.nonzero(band[:, c])[0]
        cols.append(col[0] if col.size else band.shape[0] + 1)
    return bx + int(np.argmin(cols))


def _trace_stick_base(view, mask, bx: int, by: int, bw: int, bh: int) -> Optional[int]:
    """Walk down the pole from under the flag and find where it ends.

    The pole is thin, near-vertical, and contrasts against the green --
    usually white, sometimes dark. Rather than guess its colour, look for
    a narrow column of pixels that differ from their immediate
    neighbours, and follow it until that stops being true. Where it stops
    is the ground.

    Returns None when the pole cannot be followed -- a golfer in front of
    it, a low-contrast frame, a pole the same tone as the grass behind.
    That is a normal outcome, not an error.
    """
    import cv2
    import numpy as np

    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    sx = _stick_x(mask, bx, by, bw, bh)
    start = by + bh                       # just under the cloth
    if start >= h - 4:
        return None

    # Half-width of the window we search for the pole in, per row. The
    # pole leans and the camera is not perfectly square to it, so allow
    # some drift, but not enough to wander onto a bunker edge.
    win = max(3, bw)
    last_good = None
    x_cursor = sx

    for y in range(start, h):
        lo = max(0, x_cursor - win)
        hi = min(w, x_cursor + win + 1)
        if hi - lo < 3:
            break
        row = gray[y, lo:hi].astype(np.int16)
        # Contrast of each pixel against the row's own median: the pole
        # is whatever stands out locally, whichever way it stands out.
        med = float(np.median(row))
        dev = np.abs(row - med)
        j = int(np.argmax(dev))
        if dev[j] < 18:                   # pole no longer distinguishable
            break
        x_cursor = lo + j
        last_good = y

    if last_good is None or last_good - start < 4:
        return None
    return last_good
