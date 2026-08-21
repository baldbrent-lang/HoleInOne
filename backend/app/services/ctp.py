"""Closest to the pin: turn a resting ball into a distance.

Every part of this already existed separately and none of them talked.
debug3's green follow returns `rest_xy` -- where the ball actually
stopped, in green-camera pixels. green_calibration turns a green-camera
pixel into feet on the plane of the green, and into a distance from the
pin when the pin is marked. intro_overlay can put a number on the end of
the clip. This is the joint.

It answers with None far more readily than with a number. A distance is
the one thing in this system that decides who wins a prize, and every
way of not knowing it -- uncalibrated camera, unmarked pin, a ball that
never came to rest in view, a pixel that projects off the green's plane
-- has to end in silence rather than in an estimate. A clip with no
plate is a clip that says nothing. A clip with a wrong plate says
something false to the person it is about.
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger("golfreelz.ctp")


def measure_rest(calibration: dict | None, rest_xy) -> Optional[dict]:
    """Where the resting ball is, in feet, and how far from the pin.

    `rest_xy` is a GREEN-CAMERA pixel -- debug3's `rest_xy`, not the
    tee-side `ball_rest_xy`, which is the ball sitting on the tee before
    the swing and would produce a confident answer to a different
    question entirely.

    Returns None unless the camera is calibrated FOR MEASURING, the pin
    is marked, and the pixel lands on the green's plane.
    """
    from . import green_calibration as gc

    if not calibration or not calibration.get("homography"):
        return None
    # A tee camera's fit aims the tracer; it is not a yardage, and
    # calibrate_green_camera records which job a fit was accepted for.
    if (calibration.get("purpose") or "measure") != "measure":
        log.info("ctp: calibration is for %s, not measuring",
                 calibration.get("purpose"))
        return None
    if not (calibration.get("pin") or {}).get("world"):
        log.info("ctp: no pin marked on this camera; nothing to measure from")
        return None
    try:
        x, y = float(rest_xy[0]), float(rest_xy[1])
    except (TypeError, ValueError, IndexError):
        return None

    pos = gc.image_to_green(calibration, x, y)
    if not pos or pos.get("distance_from_pin_ft") is None:
        return None
    return pos


def plate_text(pos: dict | None) -> Optional[str]:
    """What the closing plate should read, or None for no plate.

    Uses the display string green_calibration produced rather than
    re-formatting the raw feet. That string is deliberately coarse --
    the honest accuracy is a foot or three, so feet-and-inches would be
    a precision the homography does not have -- and re-rounding here
    would quietly undo that decision in the one place a golfer reads it.
    """
    if not pos:
        return None
    disp = (pos.get("distance_from_pin_display") or "").strip()
    if disp:
        return disp
    d = pos.get("distance_from_pin_ft")
    return None if d is None else f"{round(float(d))} FEET"


def distance_plate_for_clip(calibration: dict | None, rest_xy) -> Optional[str]:
    """measure_rest + plate_text, for callers that only want the string."""
    return plate_text(measure_rest(calibration, rest_xy))
