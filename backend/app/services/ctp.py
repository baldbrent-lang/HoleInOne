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


def measure_pair(calibration: dict | None, pin_xy, ball_xy) -> Optional[dict]:
    """Distance between TWO operator-chosen green pixels, in feet.

    `measure_rest` answers "how far is this ball from the pin the camera
    was calibrated against". This answers "how far apart are these two
    points I just clicked", and the difference matters: the pin marked
    at calibration time is where the hole was THAT day. Pins move. When
    an operator can see both the ball and the flag in the frame in front
    of them, their two clicks describe this swing, and a stored pin from
    another session does not.

    So this deliberately does NOT require `calibration["pin"]["world"]` —
    the operator supplies the pin. Everything else it still refuses on:
    an uncalibrated camera, a fit accepted only for aiming a tracer, or
    a pixel that does not land on the green's plane all return None,
    because a hand-placed click cannot rescue a missing scale.
    """
    from . import green_calibration as gc

    if not calibration or not calibration.get("homography"):
        return None
    if (calibration.get("purpose") or "measure") != "measure":
        log.info("ctp: calibration is for %s, not measuring",
                 calibration.get("purpose"))
        return None
    try:
        px, py = float(pin_xy[0]), float(pin_xy[1])
        bx, by = float(ball_xy[0]), float(ball_xy[1])
    except (TypeError, ValueError, IndexError):
        return None

    pin = gc.image_to_green(calibration, px, py)
    ball = gc.image_to_green(calibration, bx, by)
    if not pin or not ball:
        return None

    import math as _math
    d = _math.hypot(ball["x_ft"] - pin["x_ft"], ball["y_ft"] - pin["y_ft"])
    return {
        "distance_from_pin_ft": round(d, 1),
        # gc's own formatter, not a local one: the coarse rounding is a
        # measurement decision that green_calibration owns, and a second
        # copy here would be free to drift into a false precision.
        "distance_from_pin_display": gc._feet_display(d),
        "pin_world": [pin["x_ft"], pin["y_ft"]],
        "ball_world": [ball["x_ft"], ball["y_ft"]],
        "pin_green": [round(px, 1), round(py, 1)],
        "ball_green": [round(bx, 1), round(by, 1)],
    }
