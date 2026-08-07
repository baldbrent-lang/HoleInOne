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
# A TEE camera's view of the green is small and far, so the same four
# clicks carry more world error -- but the tee calibration is not used to
# MEASURE anything. It aims the end of the tracer, where a few feet on
# the green is a few pixels on screen. Holding it to the measuring
# tolerance would reject a fit that is entirely good enough for the only
# job it has.
MAX_RMS_FT_TEE = 20.0

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
    src_label: str = "image_points", dst_label: str = "world_points",
    src_hint: str = "spread them around the green (front, back, and both "
                    "sides)",
    dst_hint: str = "check the measured positions",
    dst_tol: float = 1e-6,
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

    img = _as_pairs(image_points, src_label)
    wld = _as_pairs(world_points, dst_label)
    if len(img) != len(wld):
        raise CalibrationError(
            f"got {len(img)} {src_label} and {len(wld)} {dst_label} — "
            "they must pair up",
        )
    if len(img) < 4:
        raise CalibrationError("need at least 4 points to fit a homography")
    if _collinear(img):
        raise CalibrationError(
            f"three or more {src_label} are on a line — {src_hint}",
        )
    # `dst_tol` is loosened for the world-feet case and tightened almost
    # to nothing for a view map. There the destination points ARE nearly
    # collinear -- the tee camera sees the green as a sliver a few pixels
    # tall, and no amount of clicking changes that -- so the guard meant
    # to catch a degenerate arrangement instead fires at random on a
    # perfectly good one. What protects a view map is the pair count.
    if _collinear(wld, dst_tol):
        raise CalibrationError(
            f"three or more {dst_label} are on a line — {dst_hint}",
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


def green_to_image(calibration: dict, X: float, Y: float):
    """Feet on the green -> pixel in THIS camera's image. The inverse of
    `image_to_green`, and the piece that lets a landing marked on the
    green camera be drawn in the tee camera's frame.

    Returns (x, y) or None when the camera has no calibration, the matrix
    will not invert, or the point projects behind the camera.
    """
    H = (calibration or {}).get("homography")
    if not H:
        return None
    try:
        import numpy as np

        Hi = np.linalg.inv(np.asarray(H, dtype=float))
    except Exception as exc:  # noqa: BLE001
        log.warning("calibration will not invert: %s", exc)
        return None
    pt = _apply(Hi.tolist(), float(X), float(Y))
    return pt


# --------------------------------------------------------------------
# green view -> tee view, straight across
# --------------------------------------------------------------------
# The tracer has to finish where the ball landed, and the landing is
# marked on the GREEN camera. Going via world feet works and is what
# `green_to_image` is for, but it needs the tee camera calibrated
# against measured distances -- a tape measure, a yardage book, and a
# walk out to the green.
#
# Aiming does not need any of that. Both cameras look at the same flat
# ground, so one homography takes green pixels to tee pixels directly,
# and the operator fits it by clicking the same four ground features in
# both pictures: the bunker's corners, the flagstick's BASE, a bend in
# the cart path. No measurement, done once per hole from a desk.
#
# The plane matters. A homography is exact only for points ON the
# surface it was fitted to. A ball at rest is on the ground, so it maps
# correctly; the TOP of the flagstick is not, which is why the hint says
# base.

# Residual limit, in TEE pixels. Generous on purpose: the tee camera's
# view of the green is small and near the horizon, so a click a few
# pixels out there is many feet on the ground and the fit cannot be held
# to a tight number. This is aiming, not measuring.
MAX_RMS_PX = 40.0

# FOUR PAIRS ARE NOT ENOUGH HERE, and the reason is the geometry rather
# than the algebra. Four determine a homography exactly -- which means
# they also reproduce every click error exactly, with nothing left over
# to average it away. That is survivable when the destination points are
# spread across a frame. They are not: the tee camera is ~180 m back, so
# the whole green and everything round it projects into a sliver about
# 200 px wide and FIVE px tall. Fitting a projective transform to
# destination points that nearly lie on a line is ill-conditioned, and
# the noise it cannot average away comes out magnified.
#
# Measured on a simulated pair of cameras with 1.5 px of click error in
# each view, mapping points on and behind the green:
#
#     pairs   median err   90th pct   worst
#       4       20.0 px     55.6 px   1215 px
#       5        2.0 px      5.0 px     40 px
#       6        1.3 px      2.8 px     13 px
#       8        1.1 px      2.5 px      —
#
# So six, and the UI asks for eight. The jump from four to five is the
# whole story: it is the first fit that is over-determined.
MIN_VIEW_MAP_POINTS = 6


def fit_view_map(points: Iterable, green_size, tee_size) -> dict:
    """Fit green view -> tee view from clicked pairs.

    `points` is [{"green": [x,y], "tee": [x,y]}, ...], four or more, in
    the pixel coordinates of frames whose sizes are given.

    THE FIT IS STORED NORMALISED. Clicks arrive in whatever resolution
    the operator happened to be looking at, and the videos this is later
    applied to are not always that: a cut re-encoded through
    compress_for_email is capped at 1280 on the long edge, and a camera
    can be swapped for one with a different sensor. Dividing through by
    the frame size makes the mapping a property of the two VIEWS rather
    than of two particular files.
    """
    gw, gh = (float(v) for v in (green_size or (0, 0)))
    tw, th = (float(v) for v in (tee_size or (0, 0)))
    if min(gw, gh, tw, th) <= 0:
        raise CalibrationError(
            "frame sizes are required — the mapping is stored independent "
            "of resolution and cannot be without them",
        )
    pts = list(points or [])
    g, t = [], []
    for p in pts:
        try:
            g.append([float(p["green"][0]) / gw, float(p["green"][1]) / gh])
            t.append([float(p["tee"][0]) / tw, float(p["tee"][1]) / th])
        except (KeyError, TypeError, ValueError, IndexError):
            raise CalibrationError(
                'each point must be {"green": [x, y], "tee": [x, y]}',
            )
    if len(pts) < MIN_VIEW_MAP_POINTS:
        raise CalibrationError(
            f"need at least {MIN_VIEW_MAP_POINTS} pairs — you have "
            f"{len(pts)}. The tee camera sees the whole green as a sliver "
            f"a few pixels tall, and a four-pair fit reproduces every "
            f"click error exactly instead of averaging it out; measured, "
            f"that is ~20px of error against ~1px at six pairs.",
        )
    H, rms_norm, meaningful = compute_homography(
        g, t,
        src_label="green points", dst_label="tee points",
        src_hint="spread them around the green — its front and back edges "
                 "and both sides, not all along one edge",
        dst_hint="two of them are the same point in the tee view",
        dst_tol=1e-12,
    )
    # How much room the fit had on the tee side, in that view's own
    # pixels. Reported rather than enforced: it is what it is, decided by
    # where the cameras are bolted, and the operator can only respond to
    # it by clicking more pairs.
    _tw = [p[0] for p in t]
    _th = [p[1] for p in t]
    _spread = [round((max(_tw) - min(_tw)) * tw, 1),
               round((max(_th) - min(_th)) * th, 1)]

    # THE RESIDUAL IS NOT THE ACCURACY HERE, and reporting it as such
    # would be the most misleading number on the screen. With the
    # destination points strung along a line the fit has enough freedom
    # left to absorb the click noise almost exactly -- it comes back
    # near zero on a fit that is metres out in the middle.
    #
    # So measure what is actually wanted: leave each pair out, fit on
    # the rest, and see how far the fit misses the pair it never saw.
    # That is the error on a point the model was not shown, which is
    # precisely the question being asked of it every time a landing is
    # mapped.
    _cv = None
    if len(pts) >= 5:
        _errs = []
        for i in range(len(pts)):
            _g = [v for j, v in enumerate(g) if j != i]
            _t = [v for j, v in enumerate(t) if j != i]
            try:
                _H, _, _ = compute_homography(
                    _g, _t, src_label="green points", dst_label="tee points",
                    dst_tol=1e-12,
                )
            except CalibrationError:
                continue
            _m = _apply(_H, g[i][0], g[i][1])
            if _m is None:
                continue
            _errs.append(math.hypot((_m[0] - t[i][0]) * tw,
                                    (_m[1] - t[i][1]) * th))
        if _errs:
            _errs.sort()
            _cv = round(_errs[len(_errs) // 2], 1)
    return {
        "points": [
            {"green": [float(p["green"][0]), float(p["green"][1])],
             "tee": [float(p["tee"][0]), float(p["tee"][1])]}
            for p in pts
        ],
        "homography": H,
        "green_size": [gw, gh],
        "tee_size": [tw, th],
        # Back into tee pixels at the resolution it was clicked on, which
        # is the only number the operator can judge.
        "rms_px": round(rms_norm * tw, 1) if meaningful else None,
        # The number to trust and to show: median error on a pair the
        # fit was not given. See above for why rms_px is not it.
        "cv_px": _cv,
        "tee_spread_px": _spread,
        "n_points": len(pts),
    }


def map_to_tee(view_map: dict, x: float, y: float,
               green_size=None, tee_size=None):
    """A green-camera pixel -> the tee camera's pixel for the same spot.

    `green_size` / `tee_size` are the frames the input and output are in;
    they default to the sizes the mapping was fitted on. Returns (x, y),
    or None when there is no mapping or the point projects to the horizon
    -- which is the honest answer for a click in the sky, not an error.
    """
    H = (view_map or {}).get("homography")
    if not H:
        return None
    gw, gh = view_map.get("green_size") or (0, 0)
    tw, th = view_map.get("tee_size") or (0, 0)
    if green_size and min(green_size) > 0:
        gw, gh = float(green_size[0]), float(green_size[1])
    if tee_size and min(tee_size) > 0:
        tw, th = float(tee_size[0]), float(tee_size[1])
    if min(gw, gh, tw, th) <= 0:
        return None
    pt = _apply(H, float(x) / float(gw), float(y) / float(gh))
    if pt is None:
        return None
    return (pt[0] * float(tw), pt[1] * float(th))


def _feet_display(d: float) -> str:
    if d < 1:
        return "inside 1 ft"
    return f"{int(round(d))} ft"
