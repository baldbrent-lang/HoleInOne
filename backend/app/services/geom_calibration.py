"""Calibrate a green camera from its own geometry, not from a tape measure.

green_calibration fits a homography to four or more points the operator
identified in the frame AND located on the ground in feet. That is the
accurate way and it needs somebody standing on the green with a
rangefinder.

This is the other way. Three things determine the same mapping:

  * how high the camera is mounted        (measured once, per camera)
  * the lens's horizontal field of view   (a spec: Pi HQ + 6mm is ~54 deg)
  * where the horizon is                  (derived, see below)

and the third comes free. A flagstick is 7 ft of known vertical object
standing on the very plane we are trying to map, so its height in pixels
at a known image row fixes the horizon:

    pixels = 7ft * (row_of_base - row_of_horizon) / camera_height

One detected flagstick, one subtraction, and the geometry closes. The
pin moves daily and gets re-detected daily, so the calibration renews
itself instead of going stale.

THE MODEL, and what it assumes. Treating the green as a level plane
seen by a camera with no roll:

    ground distance  d = f * Hc / (v - horizon)
    lateral offset   X = (u - cx) * d / f

which rearranges into a homography, so the output slots into
image_to_green unchanged and every consumer downstream -- the closing
plate, closest-to-the-pin, the tracer's aim -- works without knowing
where the numbers came from.

It assumes no camera roll, no lens distortion, and a green level with
the camera's base. A fitted homography measures reality and absorbs all
three; this derives from a model and does not. Validated against a hand
measurement on a real frame it came within half a foot, which is well
inside what this system claims -- but it is a model, and the honest
place for it is `source: geometric` on the record so nothing downstream
can mistake it for a measured fit.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

log = logging.getLogger("golfreelz.geom_calibration")

# A flagstick is 7 ft by the Rules of Golf. This is the ruler.
FLAGSTICK_FT = 7.0

# Sanity rails. A camera 3 ft off the ground or 40 ft up is a typo, and
# a horizon computed from one is a mapping that fails quietly.
MIN_MOUNT_FT, MAX_MOUNT_FT = 4.0, 30.0
MIN_HFOV_DEG, MAX_HFOV_DEG = 5.0, 120.0


class GeomError(ValueError):
    """Bad input, phrased for the operator rather than the log."""


def horizon_row(
    stick_base_y: float, stick_px: float, mount_height_ft: float,
) -> float:
    """Image row of the horizon, from one flagstick.

    A vertical object of height H standing at image row v spans
    H * (v - horizon) / Hc pixels, so the horizon follows from a single
    observation once the mount height is known.
    """
    if stick_px <= 0:
        raise GeomError("the flagstick must have a positive pixel height")
    return stick_base_y - stick_px * mount_height_ft / FLAGSTICK_FT


def build_calibration(
    frame_w: int,
    frame_h: int,
    mount_height_ft: float,
    hfov_deg: float,
    stick_base_xy: tuple[float, float],
    stick_px: float,
    pin_is_stick_base: bool = True,
) -> dict:
    """A green_calibration-shaped record derived from geometry.

    `stick_base_xy` is the base of the flagstick in image pixels and
    `stick_px` its full height in pixels. The base doubles as the pin,
    because the base of the stick IS the hole -- which is why this can
    calibrate and locate the pin in one pass.
    """
    if not (MIN_MOUNT_FT <= mount_height_ft <= MAX_MOUNT_FT):
        raise GeomError(
            f"mount height {mount_height_ft} ft is outside {MIN_MOUNT_FT}-"
            f"{MAX_MOUNT_FT} ft — check the value before trusting a mapping "
            f"built on it",
        )
    if not (MIN_HFOV_DEG <= hfov_deg <= MAX_HFOV_DEG):
        raise GeomError(f"horizontal FOV {hfov_deg} deg is implausible")
    if frame_w <= 0 or frame_h <= 0:
        raise GeomError("frame size is required")

    f_px = (frame_w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    cx = frame_w / 2.0
    Hc = float(mount_height_ft)
    bx, by = float(stick_base_xy[0]), float(stick_base_xy[1])
    y_h = horizon_row(by, float(stick_px), Hc)

    if by - y_h < 8:
        raise GeomError(
            "the flagstick sits essentially on the horizon — either the "
            "detection is wrong or the camera is looking at the sky",
        )

    # X = Hc(u - cx)/(v - y_h),  Y = f*Hc/(v - y_h)
    # In homogeneous form [X', Y', W] with W = (v - y_h):
    H = [
        [Hc, 0.0, -Hc * cx],
        [0.0, 0.0, f_px * Hc],
        [0.0, 1.0, -y_h],
    ]

    out: dict = {
        "homography": H,
        # No clicked correspondences exist; say so rather than leaving
        # keys that imply somebody measured something.
        "image_points": None,
        "world_points": None,
        "n_points": 0,
        "rms_error_ft": None,
        "purpose": "measure",
        "role": "green",
        # The provenance that matters. A consumer that wants to know
        # whether a number rests on a tape measure or on a model can ask.
        "source": "geometric",
        "geometry": {
            "mount_height_ft": Hc,
            "hfov_deg": float(hfov_deg),
            "focal_px": round(f_px, 1),
            "horizon_y": round(y_h, 1),
            "frame_size": [int(frame_w), int(frame_h)],
            "flagstick_px": round(float(stick_px), 1),
        },
    }
    if pin_is_stick_base:
        from . import green_calibration as gc

        world = gc.image_to_green(out, bx, by)
        if world is None:
            raise GeomError("the flagstick base does not project onto the plane")
        out["pin"] = {"image": [round(bx, 1), round(by, 1)],
                      "world": [world["x_ft"], world["y_ft"]]}
    return out


def stick_px_from_detection(det, frame_h: int) -> Optional[float]:
    """Flagstick pixel height from a PinDetection, when it recorded one.

    The detector reports the base; the height comes from how far the
    ridge it followed extended above that. Returns None when the
    detection did not carry enough to say, which must not be guessed --
    a wrong stick height moves the horizon, and the horizon moves every
    distance in the frame.
    """
    top = getattr(det, "stick_top_y", None)
    if top is None:
        return None
    base = float(det.y) * frame_h
    px = base - float(top)
    return px if px > 8 else None
