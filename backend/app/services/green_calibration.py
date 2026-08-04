"""Green-camera calibration: image pixels -> position on the green, in feet.

Pixels are not yards, and the conversion changes across the frame — a ball
30 ft from the pin but further from the camera covers fewer pixels than one
30 ft away and near. So a flat "pixels per foot" is wrong everywhere except
at one distance. What is needed is a mapping from the image to the PLANE of
the green, which is a homography: the green is flat, the camera is a
pinhole, and a plane viewed by a pinhole camera is related to its image by
a single 3x3.

Four point correspondences determine it. The operator clicks four features
they can identify in the still AND locate on the yardage book — front edge,
back edge, left and right extremes — and types what those are in feet.

Everything downstream depends on this: closest-to-the-pin distances, and
finishing the tee-side tracer where the ball actually landed. Neither may
guess a conversion without it, which is why an uncalibrated camera returns
None rather than a plausible number.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, Optional, Sequence

log = logging.getLogger("golfreelz.green_calibration")

# Refuse a fit worse than this. At 1080p from a typical green-side mount
# the honest accuracy is +-1-3 ft; a residual above 5 ft means points were
# mis-clicked or mis-measured, and a bad homography is worse than none
# because it produces confident wrong answers.
MAX_RMS_FT = 5.0

Point = Sequence[float]


class CalibrationError(ValueError):
    """Bad input, phrased for the operator rather than the log."""


def _as_pairs(pts: Iterable, label: str) -> list[list[float]]:
    out: list[list[float]] = []
    for p in pts or []:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            raise CalibrationError(f"{label} must be [x, y] number pairs")
        if not (math.isfinite(x) and math.isfinite(y)):
            raise CalibrationError(f"{label} contains a non-finite value")
        out.append([x, y])
    return out


def _collinear(pts: list[list[float]], tol: float = 1e-6) -> bool:
    """Any 3 of 4 on a line makes the homography degenerate. Catch it here
    with a clear message instead of letting OpenCV return a matrix that
    maps everything to nonsense."""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                (x1, y1), (x2, y2), (x3, y3) = pts[i], pts[j], pts[k]
                area2 = abs(
                    (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                )
                if area2 <= tol:
                    return True
    return False


def compute_homography(
    image_points: Iterable, world_points: Iterable,
) -> tuple[list[list[float]], float, bool]:
    """Solve image -> world (feet).

    Returns (3x3, rms_error_ft, residual_is_meaningful).

    That third value matters. FOUR points always fit a homography
    EXACTLY — eight equations, eight unknowns — so the residual comes
    back 0.00 however badly they were clicked or measured. Reporting
    that as "accuracy: 0 ft" would be a lie told at exactly the moment
    the operator is deciding whether to trust the calibration.

    The residual only becomes real evidence at FIVE or more points,
    where the fit is over-determined and has to compromise. So four is
    accepted (it is what contests.md asks for and it is genuinely
    enough to define the mapping), but the caller is told the check
    could not be performed rather than being handed a flattering zero.
    """
    import numpy as np

    img = _as_pairs(image_points, "image_points")
    wld = _as_pairs(world_points, "world_points")
    if len(img) != len(wld):
        raise CalibrationError(
            f"got {len(img)} image points and {len(wld)} world points — "
            "they must pair up",
        )
    if len(img) < 4:
        raise CalibrationError("need at least 4 points to fit a homography")
    if _collinear(img):
        raise CalibrationError(
            "three or more image points are on a line — spread them around "
            "the green (front, back, and both sides)",
        )
    if _collinear(wld):
        raise CalibrationError(
            "three or more world points are on a line — check the measured "
            "positions",
        )

    import cv2

    src = np.array(img, dtype=np.float64)
    dst = np.array(wld, dtype=np.float64)
    if len(img) == 4:
        H = cv2.getPerspectiveTransform(
            src.astype(np.float32), dst.astype(np.float32),
        )
    else:
        H, _mask = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise CalibrationError("could not fit a homography to those points")
    H = np.asarray(H, dtype=np.float64)
    if not np.all(np.isfinite(H)):
        raise CalibrationError("the fit produced a degenerate matrix")

    # Residual: push the operator's own image points through and compare.
    errs = []
    for (px, py), (wx, wy) in zip(img, wld):
        mapped = _apply(H, px, py)
        if mapped is None:
            raise CalibrationError("the fit maps a marked point to infinity")
        errs.append(math.hypot(mapped[0] - wx, mapped[1] - wy))
    rms = math.sqrt(sum(e * e for e in errs) / len(errs))
    return H.tolist(), round(rms, 2), len(img) >= 5


def _apply(H, x: float, y: float) -> Optional[tuple[float, float]]:
    """Project one image pixel through a homography. None when the point
    maps to the horizon (w ~ 0) — which is what happens if you click the
    sky, and is a real answer, not an error."""
    a = H[0] if not hasattr(H, "tolist") else H[0]
    b, c = H[1], H[2]
    w = c[0] * x + c[1] * y + c[2]
    if abs(w) < 1e-12:
        return None
    return (
        (a[0] * x + a[1] * y + a[2]) / w,
        (b[0] * x + b[1] * y + b[2]) / w,
    )


def image_to_green(calibration: dict, x: float, y: float) -> Optional[dict]:
    """Where on the green is this pixel? Feet, plus distance from the pin
    when the pin was marked. None if the camera isn't calibrated or the
    pixel doesn't land on the plane."""
    if not calibration:
        return None
    H = calibration.get("homography")
    if not H:
        return None
    pos = _apply(H, float(x), float(y))
    if pos is None:
        return None
    out = {"x_ft": round(pos[0], 2), "y_ft": round(pos[1], 2)}
    pin = (calibration.get("pin") or {}).get("world")
    if pin:
        d = math.hypot(pos[0] - float(pin[0]), pos[1] - float(pin[1]))
        out["distance_from_pin_ft"] = round(d, 1)
        # Report in the units a golfer uses, and NEVER to a precision the
        # measurement doesn't have: +-1-3 ft is the honest accuracy, so
        # feet-and-inches would be a lie. contests.md makes this a rule.
        out["distance_from_pin_display"] = _feet_display(d)
    return out


def _feet_display(d: float) -> str:
    if d < 1:
        return "inside 1 ft"
    return f"{int(round(d))} ft"
