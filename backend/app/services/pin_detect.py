"""Find the flagstick's BASE in a green-camera frame.

The base, not the flag. The base is the only part of the assembly on the
ground plane, and it is the ground plane the homography maps -- aim at
the cloth and you are measuring from a point seven feet in the air,
which the calibration converts into a confident, wrong answer.

WHY NOT COLOUR. The first version of this looked for a yellow flag,
because the first course we looked at flies yellow ones. Two things
killed it. Snee Farm flies WHITE flags, so the primary signal was simply
absent; and a yellow threshold on a golf course in August catches the
entire dried rough -- 75,000 pixels of it on the frame this was tested
against. It locked onto a patch of brush and reported 0.85 confidence.
Colour is a property of one course on one day. It cannot be the thing
this stands on.

WHAT IT USES INSTEAD. A flagstick is a thin, near-vertical object
standing on a large, smooth, uniformly-coloured surface, and it rises
off that surface into a background that is almost always darker (trees,
brush, shadow). That geometry is true of every flagstick on every course
regardless of what colour the cloth is, and it is what this looks for:

  1. Segment the putting surface -- green hue AND low texture, which is
     what separates a mown green from rough that is just as green but
     visually noisy.
  2. In the band just above that surface, score every column by how much
     it stands out from its immediate left and right neighbours. A pole
     is a narrow ridge; measured on the test frame the flagstick column
     scored 775 against a frame mean of 160.
  3. Walk the winning column down to where the ridge dies. That is the
     ground.

Colour is still used, but only at the end and only to raise confidence:
a saturated or bright blob at the top of the winning column looks like a
flag, and a column with one is more likely to be the flagstick than a
palm trunk. It can never create a detection on its own.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("golfreelz.pin")

# Putting-surface segmentation. Hue range is wide because greens read
# very differently in flat morning light and hard noon sun; the TEXTURE
# gate is what actually separates green from rough.
GREEN_HUE_LO, GREEN_HUE_HI = 32, 90
GREEN_SAT_MIN, GREEN_VAL_MIN = 55, 55
GREEN_TEXTURE_MAX = 6.0     # blurred |Laplacian| — higher is rougher
GREEN_MIN_AREA_FRAC = 0.010  # below this it is not a putting surface

# How far above the green's top edge to look for the pole. A flagstick
# is 7 ft and greens are seen obliquely, so the visible rise above the
# surface is short -- tens of pixels, not hundreds.
STICK_BAND_PX = 80
# A column must beat the band's own noise by this many sigma to count.
RIDGE_SIGMA = 2.5


@dataclass
class PinDetection:
    """Where the base of the flagstick is, and how much to trust it.

    Coordinates are FRACTIONS of frame width/height, matching how
    tee_boxes and ball_sizes are stored, so a camera swapped to another
    resolution does not silently invalidate a stored pin.
    """

    x: float
    y: float
    confidence: float
    method: str = "ridge"
    # Top of the pole, in pixels. Carried because a flagstick is 7 ft of
    # known vertical object, so base-to-top IS the scale of the frame at
    # that spot -- which is what lets a camera calibrate itself from a
    # picture of a pin. Without it the height would have to be guessed,
    # and the horizon it fixes moves every distance in the frame.
    # A PROPOSAL, not a measurement. Tuned tight the walk loses the
    # pole; tuned loose it runs past the flag into the treeline behind.
    # It exists so an operator has a number to accept or correct, and
    # the horizon it implies must be eyeballed against the real one
    # before any distance is built on it. Fortunately this is a
    # once-per-camera question: the mount is fixed, so the horizon does
    # not move when the pin does.
    stick_top_y: Optional[float] = None
    green_box: Optional[tuple[int, int, int, int]] = None
    ridge_score: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "x": round(self.x, 5),
            "y": round(self.y, 5),
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "stick_top_y": self.stick_top_y,
            "stick_px": (
                None if self.stick_top_y is None else round(self.stick_top_y, 1)
            ),
            "green_box": list(self.green_box) if self.green_box else None,
            "ridge_score": round(self.ridge_score, 1),
            "notes": self.notes,
        }


def find_green(frame):
    """The putting surface: (mask, (x, y, w, h)) or (None, None).

    Hue alone finds every blade of grass in shot. The texture gate is
    the useful half: a mown green is smooth, and rough, brush and fringe
    are not, however green they are.
    """
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    hue_ok = (
        (H >= GREEN_HUE_LO) & (H <= GREEN_HUE_HI)
        & (S > GREEN_SAT_MIN) & (V > GREEN_VAL_MIN)
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tex = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)),
                           (0, 0), 3)
    m = (hue_ok & (tex < GREEN_TEXTURE_MAX)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None, None
    i = max(range(1, n), key=lambda k: st[k, cv2.CC_STAT_AREA])
    area = st[i, cv2.CC_STAT_AREA]
    if area < GREEN_MIN_AREA_FRAC * h * w:
        return None, None
    box = (int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP]),
           int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT]))
    return (lab == i).astype(np.uint8), box


def _ridge_scores(gray, y0: int, y1: int):
    """Per-column "narrow vertical thing" score across a band of rows.

    Subtracting a horizontal median leaves what is narrower than the
    median window -- a pole survives, a tree canopy or a bunker edge does
    not. Both polarities are summed so a dark pole against bright sky
    scores as well as a bright one against trees.
    """
    import cv2
    import numpy as np

    band = gray[y0:y1].astype(np.float32)
    if band.size == 0:
        return None
    med = cv2.medianBlur(band.astype(np.uint8), 11).astype(np.float32)
    ridge = np.abs(band - med)
    score = ridge.sum(axis=0)
    return cv2.GaussianBlur(score.reshape(1, -1), (0, 0), 1.5).ravel()


def _walk_base(gray, x: int, y_from: int, y_max: int, win: int = 6,
               min_dev: float = 10.0, tolerate: int = 4) -> Optional[int]:
    """Follow the pole down from `y_from` and return where it ends.

    Tolerates a few weak rows rather than stopping at the first. The
    pole crosses the green's own edge, a mown stripe and its own shadow
    on the way down, and any of those can wash out one or two rows --
    stopping there put the "base" a foot above the ground, which the
    homography turns into feet of error in the direction that matters.
    """
    import numpy as np

    h, w = gray.shape[:2]
    x_cur, last, misses = x, None, 0
    for y in range(max(0, y_from), min(y_max, h)):
        lo, hi = max(0, x_cur - win), min(w, x_cur + win + 1)
        if hi - lo < 3:
            break
        row = gray[y, lo:hi].astype(np.int16)
        med = float(np.median(row))
        dev = np.abs(row - med)
        j = int(np.argmax(dev))
        if dev[j] < min_dev:
            misses += 1
            if misses > tolerate:
                break
            continue
        misses = 0
        x_cur, last = lo + j, y
    return last


def _walk_top(gray, x: int, y_from: int, y_min: int, win: int = 8,
              min_dev: float = 10.0, tolerate: int = 10,
              max_drift: int = 6) -> Optional[int]:
    """Follow the pole UP from the base to the top of the assembly.

    More tolerant than the downward walk: on the way up the pole crosses
    the green's edge, the fringe, whatever is behind it, and finally the
    cloth, which is wider than the pole and can sit off to one side in
    wind. Stopping early here understates the stick's height, which
    pushes the horizon down and makes every distance in the frame read
    short.
    """
    import numpy as np

    h, w = gray.shape[:2]
    x0_anchor = x
    x_cur, last, misses = x, None, 0
    for y in range(min(y_from, h - 1), max(0, y_min) - 1, -1):
        lo = max(0, max(x_cur - win, x0_anchor - max_drift))
        hi = min(w, min(x_cur + win + 1, x0_anchor + max_drift + 1))
        if hi - lo < 3:
            break
        row = gray[y, lo:hi].astype(np.int16)
        med = float(np.median(row))
        dev = np.abs(row - med)
        j = int(np.argmax(dev))
        if dev[j] < min_dev:
            misses += 1
            if misses > tolerate:
                break
            continue
        misses = 0
        x_cur, last = lo + j, y
    return last


def _flaggy(frame, x: int, y0: int, y1: int, win: int = 14) -> bool:
    """Is there something flag-like at the top of this column?

    Any strongly saturated OR notably bright/dark patch beside the top of
    the pole. Deliberately colour-agnostic: it answers "does this look
    like it has a flag on it", not "is it yellow".
    """
    import numpy as np

    h, w = frame.shape[:2]
    y0, y1 = max(0, y0), min(h, y1)
    lo, hi = max(0, x - win), min(w, x + win + 1)
    if y1 - y0 < 3 or hi - lo < 3:
        return False
    import cv2

    patch = cv2.cvtColor(frame[y0:y1, lo:hi], cv2.COLOR_BGR2HSV)
    S, V = patch[:, :, 1].astype(int), patch[:, :, 2].astype(int)
    return bool((S > 110).mean() > 0.06 or (V > 200).mean() > 0.06)


def detect_pin(
    frame,
    roi: Optional[tuple[float, float, float, float]] = None,
    prev: Optional[tuple[float, float]] = None,
    **_ignored,
) -> Optional[PinDetection]:
    """Find the flagstick base. Returns None when nothing convincing.

    `roi` (x, y, w, h in fractions) limits the search. `prev` is
    yesterday's base, used ONLY to break ties between candidates -- never
    to override the picture, or a pin that genuinely moved would be
    dragged back to where it used to be.

    Extra keyword arguments are accepted and ignored so callers written
    against the old colour-based signature keep working.
    """
    import numpy as np

    h, w = frame.shape[:2]
    x0 = y0 = 0
    view = frame
    if roi:
        rx, ry, rw, rh = roi
        x0, y0 = max(0, int(rx * w)), max(0, int(ry * h))
        x1, y1 = min(w, int((rx + rw) * w)), min(h, int((ry + rh) * h))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        view = frame[y0:y1, x0:x1]

    import cv2

    green_mask, box = find_green(view)
    if box is None:
        return None
    gx, gy, gw, gh = box

    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    band_top = max(0, gy - STICK_BAND_PX)
    band_bot = min(view.shape[0], gy + max(8, gh // 4))
    score = _ridge_scores(gray, band_top, band_bot)
    if score is None:
        return None

    # Only columns over the green itself. A pole outside it is a tree.
    lo_c, hi_c = gx, gx + gw
    window = score[lo_c:hi_c]
    if window.size < 5:
        return None
    thresh = float(score.mean() + RIDGE_SIGMA * score.std())

    # Candidate peaks, most prominent first.
    order = np.argsort(window)[::-1]
    cands: list[int] = []
    for j in order[:40]:
        cx = lo_c + int(j)
        if window[j] < thresh:
            break
        if all(abs(cx - c) > 12 for c in cands):   # one per pole
            cands.append(cx)
        if len(cands) >= 5:
            break
    if not cands:
        return None

    best = None
    for cx in cands:
        base = _walk_base(gray, cx, max(band_top, gy - 30), gy + gh + 20)
        if base is None or base < gy - 30:
            continue
        s = float(score[cx])
        conf = min(0.9, 0.35 + 0.55 * min(1.0, (s - thresh) / max(thresh, 1e-6)))
        if _flaggy(view, cx, max(0, band_top - 10), band_top + 30):
            conf = min(0.95, conf + 0.12)
        if prev is not None:
            fx, fy = (x0 + cx) / w, (y0 + base) / h
            d = ((fx - prev[0]) ** 2 + (fy - prev[1]) ** 2) ** 0.5
            conf *= max(0.5, 1.0 - min(1.0, d / 0.6))
        if best is None or conf > best[0]:
            best = (conf, cx, base, s)

    if best is None:
        return None
    conf, cx, base, s = best
    top = _walk_top(gray, cx, base - 2, max(0, band_top - 40))
    notes: list[str] = []
    if len(cands) > 1:
        notes.append(f"{len(cands)} pole-like columns; took the strongest")
    fx, fy = (x0 + cx) / w, (y0 + base) / h
    if prev is not None:
        d = ((fx - prev[0]) ** 2 + (fy - prev[1]) ** 2) ** 0.5
        if d > 0.2:
            notes.append(f"moved {d:.2f} of frame from the last pin")

    return PinDetection(
        x=fx, y=fy, confidence=round(conf, 3), method="ridge",
        stick_top_y=(None if top is None else float(y0 + top)),
        green_box=(x0 + gx, y0 + gy, gw, gh), ridge_score=s, notes=notes,
    )


def detect_pin_stable(
    frames,
    roi: Optional[tuple[float, float, float, float]] = None,
    prev: Optional[tuple[float, float]] = None,
    min_hits: int = 3,
) -> Optional[PinDetection]:
    """Run the detector over several frames and take the median.

    One frame is a guess. A golfer walks in front of the pin, the flag
    blows across its own pole, a cloud changes the contrast the ridge
    scan depends on -- any single frame can be the bad one, and there is
    no way to tell which from inside it.

    The median is chosen over the mean deliberately: a wrong detection
    is not a small error around the right answer, it is a completely
    different object somewhere else in the frame, and a mean would drag
    the answer toward it. A median ignores it entirely.

    Spread across the surviving frames is reported, and a wide spread
    lowers confidence -- detections that disagree with each other should
    not add up to certainty.
    """
    import statistics

    hits = []
    for f in frames:
        try:
            d = detect_pin(f, roi=roi, prev=prev)
        except Exception as exc:  # noqa: BLE001
            log.warning("pin detect failed on a frame: %s", exc)
            continue
        if d is not None:
            hits.append(d)
    if len(hits) < min_hits:
        return None

    mx = statistics.median([d.x for d in hits])
    my = statistics.median([d.y for d in hits])

    # REJECT OUTLIERS BEFORE MEASURING SPREAD. Standard deviation is as
    # vulnerable to a wrong detection as the mean was: on the test clip
    # ONE frame locked onto the tracer line drawn over the green, and
    # its distance from the median inflated the spread enough to crush
    # confidence on the eighteen frames that were right. A detection
    # that far out is a different object, not a noisy reading of the
    # same one, so it is dropped rather than averaged in.
    inliers = [d for d in hits
               if ((d.x - mx) ** 2 + (d.y - my) ** 2) ** 0.5 <= 0.05]
    rejected = len(hits) - len(inliers)
    if len(inliers) < min_hits:
        return None
    xs = [d.x for d in inliers]
    ys = [d.y for d in inliers]
    mx, my = statistics.median(xs), statistics.median(ys)
    spread = max(
        statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        statistics.pstdev(ys) if len(ys) > 1 else 0.0,
    )

    conf = statistics.median([d.confidence for d in inliers])
    # 1% of the frame is a couple of feet at this range; 4% is a
    # different object. Scale confidence down across that band.
    conf *= max(0.25, 1.0 - min(1.0, spread / 0.04))
    notes = [f"median of {len(inliers)} frames", f"spread {spread:.4f} of frame"]
    if rejected:
        notes.append(f"{rejected} outlier frame(s) dropped")
    if spread > 0.02:
        notes.append("frames disagree — worth a look before trusting this")

    return PinDetection(
        x=mx, y=my, confidence=round(min(0.95, conf), 3), method="ridge-median",
        green_box=inliers[len(inliers) // 2].green_box,
        ridge_score=statistics.median([d.ridge_score for d in inliers]),
        notes=notes,
    )
