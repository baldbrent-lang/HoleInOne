"""Debug2 — the operator's swing pipeline, built to show its work.

A deliberately separate path from produce. Nothing here writes to
edit_metrics or replaces a produced clip; it exists so the reasoning
behind a swing can be inspected stage by stage, and so a different set of
rules can be tried without disturbing the pipeline that ships clips.

The stages, in order:

  1. CANDIDATES     pose bursts gated on wrist speed and spine bend —
                    the same detector produce uses, so the candidate list
                    is directly comparable.
  2. IMPACT + BALL  impact is the peak-wrist-speed frame. The ball is
                    found at the BOTTOM OF THE CLUB'S HEAT ARC: through
                    impact the club sweeps a bright arc whose lowest
                    point is where it meets the ground, which is where
                    the ball is sitting. No vision call, no rest-ball
                    detector — just the shape the swing draws.
  3. AI JUDGE       the motion-heat composite goes to the vision judge:
                    a standing figure in warm heat, a blue fan of club
                    streaks above it, a dotted trail leaving the frame.
                    Candidates it rejects are dropped.
  4. WINDOWED HEAT  for survivors, MOG2 heat over impact-5 .. impact+100
                    ONLY, so the map holds the flight and nothing else.
  5. CHAIN          walk a chain of MOG2 spots upward from the ball: each
                    step rises, drifts only a little sideways, and the
                    frames increase. The ball goes up and slightly over
                    before it comes down, so the chain accepts a gentle
                    arc and rejects anything that turns hard or reverses.

Every stage returns its numbers and an image, so a wrong answer can be
attributed to a stage rather than guessed at.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

log = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    HAS_CV = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    np = None  # type: ignore
    HAS_CV = False

# Flight window, in frames either side of impact. The 5 frames of lead-in
# cover an impact frame estimated a touch late; 100 after is ~2s at 50fps,
# by which point the ball is gone and everything left is the golfer
# walking off or wind in the trees.
WIN_PRE = 5
WIN_POST = 100


# ── stage 2: the ball, from the bottom of the club's heat arc ──────────

def club_bottom_ball(
    input_path: Path,
    impact_frame: int,
    fps: float,
    hint_xy=None,
    debug_dir: Path | None = None,
    debug_prefix: str = "d2club",
) -> dict:
    """Find the ball as the LOWEST POINT OF THE CLUB'S SWEEP.

    Through impact the club draws a bright arc in the motion heat, and the
    bottom of that arc is where it meets the ground — which is where the
    ball is. This is a different question from "find a small white sphere"
    and it fails in different places, which is the point: it needs no
    vision call and does not care whether the ball is bright, shadowed, or
    a few pixels across.

    Returns {ok, xy, arc_px, reason, image}. Never raises."""
    out = {"ok": False, "xy": None, "arc_px": 0, "reason": None, "image": None}
    if not HAS_CV:
        out["reason"] = "opencv not installed"
        return out
    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            out["reason"] = "could not open video"
            return out
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # The downswing only. Starting earlier drags in the backswing and
        # the takeaway, whose arc bottoms out somewhere else entirely.
        f0 = max(0, int(impact_frame) - 8)
        f1 = int(impact_frame) + 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        prev = None
        acc = np.zeros((h, w), np.float32)
        base = None
        for _f in range(f0, f1 + 1):
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            if base is None:
                base = fr.copy()
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            g = cv2.GaussianBlur(g, (5, 5), 0)
            if prev is not None:
                d = cv2.absdiff(g, prev)
                acc = np.maximum(acc, d.astype(np.float32))
            prev = g
        cap.release()
        if base is None:
            out["reason"] = "could not read the impact window"
            return out

        # Keep the strong movers. The club head is the fastest thing in
        # frame, so it sits at the top of this distribution.
        thr = max(24.0, float(np.percentile(acc, 99.2)))
        mask = (acc >= thr).astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8),
        )

        # Confine to the golfer's neighbourhood. Without this the lowest
        # bright pixel in frame is a shadow at the bottom edge, every time.
        if hint_xy and len(hint_xy) == 2:
            hx, hy = int(hint_xy[0]), int(hint_xy[1])
            rx = int(0.22 * w)
            ry_lo = max(0, hy - int(0.10 * h))
            ry_hi = min(h, hy + int(0.35 * h))
            box = np.zeros_like(mask)
            box[ry_lo:ry_hi, max(0, hx - rx):min(w, hx + rx)] = 255
            mask = cv2.bitwise_and(mask, box)

        ys, xs = np.nonzero(mask)
        out["arc_px"] = int(xs.size)
        if xs.size < 30:
            out["reason"] = f"club arc too faint ({xs.size}px)"
            return out

        # The bottom of the arc: take the lowest band of arc pixels and
        # use its median x. A single lowest pixel is noise; the band is
        # the club at the bottom of its sweep.
        y_bot = int(np.percentile(ys, 99.0))
        band = ys >= (y_bot - max(3, int(0.006 * h)))
        bx = int(np.median(xs[band]))
        by = int(np.median(ys[band]))
        out["ok"] = True
        out["xy"] = [bx, by]
        out["reason"] = f"bottom of a {xs.size}px club arc"

        if debug_dir is not None:
            img = base.copy()
            img[mask > 0] = (0.35 * img[mask > 0] + 0.65
                             * np.array([255, 120, 0])).astype(np.uint8)
            cv2.circle(img, (bx, by), max(10, int(0.02 * h)),
                       (0, 255, 0), 3, cv2.LINE_AA)
            _label(
                img,
                f"club arc f{f0}-{f1} ({xs.size}px); green = its bottom "
                f"= ball at impact ({bx},{by})",
            )
            name = f"{debug_prefix}.jpg"
            cv2.imwrite(str(Path(debug_dir) / name), img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 86])
            out["image"] = name
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("debug2 club_bottom_ball failed: %s", exc)
        out["reason"] = f"failed: {exc}"
        return out


# ── stage 5: walk a chain of MOG2 spots up from the ball ───────────────

def chain_from_ball(
    dots: list,
    ball_xy,
    impact_frame: int,
    frame_h: int,
    max_frames: int = WIN_POST,
) -> dict:
    """Link MOG2 spots into a flight, upward from the ball.

    The rules are the shape a struck ball actually draws, not a generic
    tracker: it leaves the ground going UP, keeps going up for a while,
    drifts sideways far less than it rises, and only then comes down. So
    each accepted step must advance in frame, and while ascending must
    rise; sideways drift is bounded relative to the rise, which is what
    rejects a chain that turns and runs along the treeline. Once the
    chain has topped out, descending steps are accepted on the same drift
    rule but may not climb again.

    Returns {points, reason, rejected} — rejected carries why each
    considered dot lost, so the map can show the near-misses."""
    out = {"points": [], "reason": None, "rejected": []}
    r = max(6.0, 0.012 * float(frame_h))       # ball scale in this frame
    cand = sorted(
        [
            d for d in (dots or [])
            if d.get("frame") is not None
            and int(d["frame"]) > int(impact_frame)
            and int(d["frame"]) <= int(impact_frame) + int(max_frames)
        ],
        key=lambda d: int(d["frame"]),
    )
    if not cand:
        out["reason"] = "no dots in the flight window"
        return out
    if not (ball_xy and len(ball_xy) == 2):
        out["reason"] = "no ball position to start from"
        return out

    px, py = float(ball_xy[0]), float(ball_xy[1])
    pf = int(impact_frame)
    phase = "up"
    chain: list = []
    for d in cand:
        f, x, y = int(d["frame"]), float(d["x"]), float(d["y"])
        gap = f - pf
        if gap <= 0:
            continue
        if gap > 25:                    # >0.5s unseen: the trail is cold
            continue
        dx, dy = x - px, y - py         # dy < 0 is UP in image coords
        rise, drift = -dy, abs(dx)
        step = math.hypot(dx, dy)
        why = None
        if step > (7.0 + 3.0 * gap) * r:
            why = "too far for the frame gap"
        elif phase == "up":
            if rise < 0.5 * r:
                # Not climbing. If the chain is established this is the
                # apex; if it has barely started, it is not the ball.
                if len(chain) >= 3:
                    phase = "down"
                else:
                    why = "not rising"
            if why is None and phase == "up" and drift > 2.2 * max(rise, r):
                why = "drifts sideways more than it rises"
        if why is None and phase == "down":
            if rise > 1.0 * r:
                why = "climbs again after the apex"
            elif drift > 2.6 * max(-rise, r):
                why = "drifts sideways more than it falls"
        if why is not None:
            out["rejected"].append(
                {"frame": f, "x": int(x), "y": int(y), "why": why},
            )
            continue
        chain.append({"frame": f, "x": int(x), "y": int(y), "phase": phase})
        px, py, pf = x, y, f

    out["points"] = chain
    if not chain:
        out["reason"] = (
            f"nothing linked upward from the ball "
            f"({len(out['rejected'])} dot(s) considered and rejected)"
        )
    else:
        n_up = sum(1 for c in chain if c["phase"] == "up")
        out["reason"] = (
            f"{len(chain)} linked f{chain[0]['frame']}-{chain[-1]['frame']} "
            f"({n_up} ascending, {len(chain) - n_up} descending); "
            f"{len(out['rejected'])} rejected"
        )
    return out


# ── drawing ────────────────────────────────────────────────────────────

def _label(img, text: str) -> None:
    for colour, weight in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(
            img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
            colour, weight, cv2.LINE_AA,
        )


def draw_chain(
    heat_path: Path,
    ball_xy,
    chain: list,
    rejected: list,
    out_path: Path,
    header: str,
) -> bool:
    """The chain over the windowed heat: the ball, the links, and the
    dots that were considered and thrown out. The rejects are drawn
    because a chain that stops early is usually explained by what it
    refused, not by what it took."""
    if not HAS_CV:
        return False
    try:
        img = cv2.imread(str(heat_path))
        if img is None:
            return False
        for rj in (rejected or []):
            cv2.drawMarker(
                img, (int(rj["x"]), int(rj["y"])), (60, 60, 220),
                cv2.MARKER_TILTED_CROSS, 11, 2, cv2.LINE_AA,
            )
        pts = [(int(c["x"]), int(c["y"])) for c in (chain or [])]
        if ball_xy and len(ball_xy) == 2:
            pts = [(int(ball_xy[0]), int(ball_xy[1]))] + pts
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, (255, 220, 0), 2, cv2.LINE_AA)
        for c in (chain or []):
            colour = (0, 220, 0) if c["phase"] == "up" else (0, 170, 255)
            cv2.circle(img, (int(c["x"]), int(c["y"])), 6, colour, -1, cv2.LINE_AA)
            cv2.circle(img, (int(c["x"]), int(c["y"])), 8, (255, 255, 255), 1,
                       cv2.LINE_AA)
        if ball_xy and len(ball_xy) == 2:
            cv2.circle(img, (int(ball_xy[0]), int(ball_xy[1])), 11,
                       (0, 255, 0), 3, cv2.LINE_AA)
        _label(img, header)
        cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        return out_path.exists()
    except Exception as exc:  # noqa: BLE001
        log.warning("debug2 draw_chain failed: %s", exc)
        return False
