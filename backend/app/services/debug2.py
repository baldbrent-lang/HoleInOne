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

def _label2(img, text: str, y: int) -> None:
    for colour, weight in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(img, text, (10, max(12, y)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, colour, weight, cv2.LINE_AA)


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
    head_y: float | None = None,
    aim_xy=None,
    ai_path=None,
    fan_y: float | None = None,
    frame_w: int | None = None,
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
        if fan_y is not None:
            # The clean band starts here, and the thirds it was searched
            # in. Above this line the map is sky and treetops.
            cv2.line(img, (0, int(fan_y)), (img.shape[1], int(fan_y)),
                     (0, 0, 255), 2, cv2.LINE_AA)
            _label2(img, "club-fan line — searched in thirds above here",
                    int(fan_y) - 6)
            _w = frame_w or img.shape[1]
            for k in (1, 2):
                cv2.line(img, (int(_w * k / 3), 0),
                         (int(_w * k / 3), int(fan_y)),
                         (0, 0, 255), 1, cv2.LINE_AA)
        if ai_path and len(ai_path) >= 2:
            # The trail the AI traced off the heat map — the corridor the
            # dots were matched against.
            for a, b in zip(ai_path, ai_path[1:]):
                cv2.line(img, (int(a["x"]), int(a["y"])),
                         (int(b["x"]), int(b["y"])),
                         (255, 255, 255), 1, cv2.LINE_AA)
            for a in ai_path:
                cv2.circle(img, (int(a["x"]), int(a["y"])), 3,
                           (255, 255, 255), -1, cv2.LINE_AA)
        if head_y is not None:
            # The line the lock-on happens above. Everything over it is
            # sky and trees, which is why the dots up there are clean.
            cv2.line(img, (0, int(head_y)), (img.shape[1], int(head_y)),
                     (0, 200, 255), 1, cv2.LINE_AA)
            _label2(img, "head line — lock on above here", int(head_y) - 6)
        if aim_xy and len(aim_xy) == 2:
            # Where the chain, run BACK to the impact frame, says the ball
            # was. Its distance from the green ball marker is the check on
            # both the chain and the ball position.
            ax, ay = int(aim_xy[0]), int(aim_xy[1])
            cv2.drawMarker(img, (ax, ay), (255, 0, 255),
                           cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
            if ball_xy and len(ball_xy) == 2:
                cv2.line(img, (ax, ay),
                         (int(ball_xy[0]), int(ball_xy[1])),
                         (255, 0, 255), 1, cv2.LINE_AA)
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


# ── stage 5b: lock on above the head, then walk back down to the ball ──

def chain_above_head(
    dots: list,
    ball_xy,
    impact_frame: int,
    frame_h: int,
    head_y: float | None,
    max_frames: int = WIN_POST,
) -> dict:
    """Find the flight where it is EASY, then walk back to where it is hard.

    Walking up from the ball starts in the worst place on the map: the
    first frames after impact sit inside the golfer's body heat, the ball
    is a motion-blurred smear, and one bad first link poisons everything
    after it. Above head height the opposite holds — no body heat, and the
    ball is slowing down so its dots bunch closer together and read as an
    obvious line.

    So: lock a chain onto the clean dots above the head, then extend it
    BACKWARD, frame by frame, down toward impact. A correct chain points
    at the ball when you run it back, which also gives a free check on the
    ball position itself — reported as aim_px.

    Returns {points, seed, aim_px, reason, rejected}."""
    out = {
        "points": [], "seed": None, "aim_px": None,
        "reason": None, "rejected": [],
    }
    r = max(6.0, 0.012 * float(frame_h))
    f0, f1 = int(impact_frame), int(impact_frame) + int(max_frames)
    inwin = [
        {"frame": int(d["frame"]), "x": float(d["x"]), "y": float(d["y"])}
        for d in (dots or [])
        if d.get("frame") is not None and f0 < int(d["frame"]) <= f1
        and d.get("x") is not None and d.get("y") is not None
    ]
    if head_y is None:
        out["reason"] = "no head position — cannot pick the clean region"
        return out
    above = sorted(
        [d for d in inwin if d["y"] <= float(head_y)],
        key=lambda d: d["frame"],
    )
    if len(above) < 3:
        out["reason"] = (
            f"only {len(above)} dot(s) above head level (y<={int(head_y)}) "
            f"— nothing to lock onto"
        )
        return out

    # Seed on the straightest 3-dot run: constant-velocity triples, scored
    # by how little the second step turns from the first. A ball in flight
    # barely turns between adjacent frames; foliage flicker does nothing
    # else BUT turn.
    best = None
    for i in range(len(above)):
        a = above[i]
        for j in range(i + 1, len(above)):
            b = above[j]
            df1 = b["frame"] - a["frame"]
            if df1 <= 0 or df1 > 8:
                break
            v1 = ((b["x"] - a["x"]) / df1, (b["y"] - a["y"]) / df1)
            if math.hypot(*v1) < 0.35 * r:
                continue                      # stationary: not a ball
            for k in range(j + 1, len(above)):
                c = above[k]
                df2 = c["frame"] - b["frame"]
                if df2 <= 0 or df2 > 8:
                    break
                v2 = ((c["x"] - b["x"]) / df2, (c["y"] - b["y"]) / df2)
                turn = math.hypot(v2[0] - v1[0], v2[1] - v1[1])
                spd = math.hypot(*v1)
                if turn > 0.55 * spd + 0.5 * r:
                    continue                  # turns too hard to be flight
                # Prefer TIGHT, STRAIGHT triples. As the ball climbs it
                # slows, so its dots bunch up — consecutive frames are the
                # signal. Rewarding a long span instead picks three
                # widely-spaced dots that line up by coincidence, which is
                # exactly what foliage flicker offers.
                score = -(turn / max(spd, 1e-3)) - 0.10 * (df1 + df2)
                if best is None or score > best[0]:
                    best = (score, a, b, c, v2)
    if best is None:
        out["reason"] = (
            f"{len(above)} dot(s) above head level but no straight "
            f"3-dot run among them"
        )
        return out
    _s, a, b, c, vel = best
    chain = [a, b, c]
    out["seed"] = [a["frame"], b["frame"], c["frame"]]

    # Forward along the flight, then backward down toward the ball. The
    # backward walk is the point of the exercise.
    def _walk(anchor, v, direction):
        got, px, py, pf = [], anchor["x"], anchor["y"], anchor["frame"]
        vx, vy = v
        while True:
            nxt = None
            for d in inwin:
                df = (d["frame"] - pf) * direction
                if df <= 0 or df > 10:
                    continue
                ex, ey = px + vx * df * direction, py + vy * df * direction
                dist = math.hypot(d["x"] - ex, d["y"] - ey)
                tol = (2.5 + 0.9 * df) * r
                if dist <= tol and (nxt is None or dist < nxt[0]):
                    nxt = (dist, d, df)
            if nxt is None:
                break
            _dist, d, df = nxt
            got.append(d)
            # Blend the measured step into the velocity so a gently
            # curving flight is followed rather than fought.
            nvx = (d["x"] - px) / (df * direction)
            nvy = (d["y"] - py) / (df * direction)
            vx, vy = 0.5 * vx + 0.5 * nvx, 0.5 * vy + 0.5 * nvy
            px, py, pf = d["x"], d["y"], d["frame"]
        return got, (vx, vy), (px, py, pf)

    fwd, _v_f, _end = _walk(c, vel, +1)
    back, v_b, (bx, by, bf) = _walk(a, vel, -1)

    allpts = sorted(
        {int(d["frame"]): d for d in (back + chain + fwd)}.values(),
        key=lambda d: d["frame"],
    )
    out["points"] = [
        {"frame": int(d["frame"]), "x": int(d["x"]), "y": int(d["y"]),
         "phase": "up"}
        for d in allpts
    ]

    # Does the chain, run back to the impact frame, arrive at the ball?
    # This is the check that makes the whole approach worth it: it tests
    # the chain AND the ball position against each other.
    if ball_xy and len(ball_xy) == 2:
        # BALLISTIC back-extrapolation, not linear. The ball is fastest at
        # launch and slows as it climbs, so running the chain's LATE
        # velocity straight back overshoots badly — measured 186px on a
        # synthetic flight whose quadratic fit lands within a few. Fit
        # x and y against frame and evaluate at impact, which is the same
        # model the renderer uses.
        ex = ey = None
        pts = out["points"]
        if HAS_CV and len(pts) >= 4:
            try:
                fs = np.array([p["frame"] for p in pts], float)
                xs = np.array([p["x"] for p in pts], float)
                ys = np.array([p["y"] for p in pts], float)
                deg = 2 if len(pts) >= 6 else 1
                ex = float(np.polyval(np.polyfit(fs, xs, deg), impact_frame))
                ey = float(np.polyval(np.polyfit(fs, ys, 2), impact_frame))
            except Exception:  # noqa: BLE001
                ex = ey = None
        if ex is None:
            df = bf - int(impact_frame)
            ex, ey = bx - v_b[0] * df, by - v_b[1] * df
        out["aim_px"] = round(
            math.hypot(ex - float(ball_xy[0]), ey - float(ball_xy[1])), 1,
        )
        out["aim_xy"] = [int(ex), int(ey)]

    out["reason"] = (
        f"{len(out['points'])} linked f{out['points'][0]['frame']}-"
        f"{out['points'][-1]['frame']} — locked on {len(above)} clean dots "
        f"above the head (seed f{a['frame']}/{b['frame']}/{c['frame']}), "
        f"walked back {len(back)} and forward {len(fwd)}"
        + (
            f"; back-extension lands {out['aim_px']:.0f}px from the ball"
            if out.get("aim_px") is not None else ""
        )
    )
    return out


# ── stage 5c: chain along the corridor the AI traced ───────────────────

def chain_along_ai_path(
    dots: list,
    ai_path: list,
    ball_xy,
    impact_frame: int,
    frame_h: int,
    max_frames: int = WIN_POST,
    corridor_px: float | None = None,
) -> dict:
    """Keep the dots that sit ON the trail the AI traced, in frame order.

    The AI reads the trail's SHAPE off the heat map but has no frame
    numbers — it is looking at an image where the whole flight is
    superimposed. MOG2 has exact frames but cannot tell the ball's dots
    from foliage. Intersecting the two gives a path that is both correctly
    shaped and correctly timed, and neither source can supply that alone.

    Returns {points, reason, rejected}."""
    out = {"points": [], "reason": None, "rejected": []}
    if not ai_path or len(ai_path) < 2:
        out["reason"] = "no AI trail to follow"
        return out
    r = max(6.0, 0.012 * float(frame_h))
    corr = corridor_px if corridor_px is not None else 6.0 * r

    def _dist_to_path(x, y):
        best = None
        for a, b in zip(ai_path, ai_path[1:]):
            ax, ay = float(a["x"]), float(a["y"])
            bx, by = float(b["x"]), float(b["y"])
            vx, vy = bx - ax, by - ay
            L2 = vx * vx + vy * vy
            t = 0.0 if L2 <= 0 else max(
                0.0, min(1.0, ((x - ax) * vx + (y - ay) * vy) / L2),
            )
            d = math.hypot(x - (ax + t * vx), y - (ay + t * vy))
            if best is None or d < best:
                best = d
        return best if best is not None else 1e9

    f0, f1 = int(impact_frame), int(impact_frame) + int(max_frames)
    kept = []
    for d in sorted(
        [
            d for d in (dots or [])
            if d.get("frame") is not None and f0 < int(d["frame"]) <= f1
            and d.get("x") is not None and d.get("y") is not None
        ],
        key=lambda d: int(d["frame"]),
    ):
        dd = _dist_to_path(float(d["x"]), float(d["y"]))
        if dd > corr:
            out["rejected"].append({
                "frame": int(d["frame"]), "x": int(d["x"]), "y": int(d["y"]),
                "why": "off the AI trail",
            })
            continue
        kept.append({
            "frame": int(d["frame"]), "x": int(d["x"]), "y": int(d["y"]),
            "phase": "up", "path_px": round(dd, 1),
        })

    # One dot per frame — the nearest to the trail wins, since MOG2 often
    # fires several times around a moving object.
    byf: dict = {}
    for k in kept:
        cur = byf.get(k["frame"])
        if cur is None or k["path_px"] < cur["path_px"]:
            byf[k["frame"]] = k
    out["points"] = [byf[f] for f in sorted(byf)]
    out["reason"] = (
        f"{len(out['points'])} dot(s) on the AI trail "
        f"(corridor {corr:.0f}px), {len(out['rejected'])} off it"
        + (
            f", f{out['points'][0]['frame']}-{out['points'][-1]['frame']}"
            if out["points"] else ""
        )
    )
    return out


# ── stage 5d: search the clean band in thirds ──────────────────────────

def fan_line_y(head_xy, feet_xy, frame_h: int) -> float | None:
    """The line above which the map is clean.

    Not the head — the CLUB FAN reaches well above it, and its streaks are
    the last strong non-ball feature going up. Half a body-height above
    the head clears the fan, leaving only sky, treetops and the ball."""
    if not (head_xy and len(head_xy) == 2):
        return None
    hy = float(head_xy[1])
    if feet_xy and len(feet_xy) == 2:
        body = max(20.0, float(feet_xy[1]) - hy)
        return max(0.0, hy - 0.5 * body)
    return max(0.0, hy - 0.15 * float(frame_h))


def chain_by_thirds(
    dots: list,
    ball_xy,
    impact_frame: int,
    frame_h: int,
    frame_w: int,
    top_y: float | None,
    max_frames: int = WIN_POST,
) -> dict:
    """Hunt the flight in the clean band, one third of the frame at a time.

    Above the club fan there is very little left but sky and the ball, so
    the search space is small. Split that band into left/middle/right and
    look in each for the signature: three or more dots in a near-straight
    line, HIGHER MEANING LATER, whose line extended downward points back
    at the ball. The middle is searched first because a shot down the
    target line lives there, then right, then left — but every third is
    searched and the best candidate wins, so a fade or a pull is not
    missed by the ordering.

    Returns {points, reason, thirds, rejected}."""
    out = {"points": [], "reason": None, "thirds": [], "rejected": []}
    if top_y is None:
        out["reason"] = "no fan line — need a head position"
        return out
    r = max(6.0, 0.012 * float(frame_h))
    f0, f1 = int(impact_frame), int(impact_frame) + int(max_frames)
    band = [
        {"frame": int(d["frame"]), "x": float(d["x"]), "y": float(d["y"])}
        for d in (dots or [])
        if d.get("frame") is not None and f0 < int(d["frame"]) <= f1
        and d.get("x") is not None and d.get("y") is not None
        and float(d["y"]) <= float(top_y)
    ]
    if len(band) < 3:
        out["reason"] = (
            f"only {len(band)} dot(s) above the fan line (y<={int(top_y)})"
        )
        return out

    third = float(frame_w) / 3.0
    zones = [
        ("middle", third, 2 * third),
        ("right", 2 * third, float(frame_w)),
        ("left", 0.0, third),
    ]
    best_overall = None
    for order, (name, x0, x1) in enumerate(zones):
        pool = sorted(
            [d for d in band if x0 <= d["x"] < x1], key=lambda d: d["frame"],
        )
        rec = {"zone": name, "n_dots": len(pool), "chain": [], "aim_px": None}
        cand = _best_line_in(
            pool, ball_xy, r, impact_frame,
            aim_gate=max(120.0, 0.20 * float(frame_w)),
        )
        if cand:
            rec["chain"] = cand["points"]
            rec["aim_px"] = cand["aim_px"]
            rec["straight_px"] = cand["straight_px"]
            # More points is better, aiming near the ball is much better,
            # and a straighter line is better. Zone order only breaks ties.
            score = (
                2.0 * len(cand["points"])
                - cand["straight_px"] / 10.0
                - 0.01 * order
            )
            rec["score"] = round(score, 2)
            if best_overall is None or score > best_overall[0]:
                best_overall = (score, name, cand)
        out["thirds"].append(rec)

    if best_overall is None:
        out["reason"] = (
            "no run of 3+ dots in a straight line pointing back at the "
            "ball, in any third of the clean band"
        )
        return out
    _sc, zone, cand = best_overall
    out["points"] = [
        {**p, "phase": "up"} for p in cand["points"]
    ]
    out["aim_px"] = cand["aim_px"]
    out["aim_xy"] = cand["aim_xy"]
    out["zone"] = zone
    out["reason"] = (
        f"{len(cand['points'])} dot(s) in the {zone} third, "
        f"f{cand['points'][0]['frame']}-{cand['points'][-1]['frame']}, "
        f"straightness {cand['straight_px']:.0f}px; extended down it lands "
        f"{cand['aim_px']:.0f}px from the ball"
    )
    return out


def _best_line_in(pool: list, ball_xy, r: float, impact_frame: int,
                  aim_gate: float = 260.0):
    """Best run of 3+ dots that reads as a ball in flight: higher means
    later, near-collinear, and pointing back at the ball when extended
    down. Returns None when nothing qualifies."""
    n = len(pool)
    if n < 3:
        return None
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            a, b = pool[i], pool[j]
            if b["frame"] <= a["frame"]:
                continue
            # HIGHER = LATER. A ball climbing away from the camera gains
            # height as the frames advance; anything falling here is the
            # club coming down, a bird, or noise.
            if b["y"] >= a["y"] - 0.5 * r:
                continue
            run = [a, b]
            vx = (b["x"] - a["x"]) / (b["frame"] - a["frame"])
            vy = (b["y"] - a["y"]) / (b["frame"] - a["frame"])
            for k in range(j + 1, n):
                c = pool[k]
                df = c["frame"] - run[-1]["frame"]
                if df <= 0:
                    continue
                ex = run[-1]["x"] + vx * df
                ey = run[-1]["y"] + vy * df
                if math.hypot(c["x"] - ex, c["y"] - ey) <= (2.0 + 0.8 * df) * r:
                    run.append(c)
            if len(run) < 3:
                continue
            # Straightness: mean distance from the best-fit line.
            xs = [p["x"] for p in run]
            ys = [p["y"] for p in run]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            if sxx <= 1e-6:
                straight = sum(abs(x - mx) for x in xs) / len(xs)
                slope = None
            else:
                slope = sxy / sxx
                straight = sum(
                    abs(y - (my + slope * (x - mx)))
                    for x, y in zip(xs, ys)
                ) / len(run) / math.sqrt(1 + slope * slope)
            # Extend the run DOWN to the impact frame and see where it
            # lands. This is the test that separates a ball from a bird:
            # a ball's line points back at where it was struck.
            aim_px = aim_xy = None
            if ball_xy and len(ball_xy) == 2:
                ex = ey = None
                # BALLISTIC back-projection when the run is long enough.
                # A straight line run back from a CURVING arc always
                # overshoots — measured 127px on a synthetic flight whose
                # quadratic fit lands far closer — and that penalises the
                # late, longest part of the flight for being real.
                if HAS_CV and len(run) >= 4:
                    try:
                        _f = np.array([p["frame"] for p in run], float)
                        _x = np.array([p["x"] for p in run], float)
                        _y = np.array([p["y"] for p in run], float)
                        ex = float(np.polyval(np.polyfit(_f, _x, 1), impact_frame))
                        ey = float(np.polyval(np.polyfit(_f, _y, 2), impact_frame))
                    except Exception:  # noqa: BLE001
                        ex = ey = None
                if ex is None:
                    df = run[0]["frame"] - int(impact_frame)
                    ex = run[0]["x"] - vx * df
                    ey = run[0]["y"] - vy * df
                aim_px = math.hypot(
                    ex - float(ball_xy[0]), ey - float(ball_xy[1]),
                )
                aim_xy = [int(ex), int(ey)]
            # AIM IS A GATE, not a tiebreak. "Points back at the ball" is
            # the thing that separates a ball from a bird, so a run that
            # misses by more than a fifth of the frame is not a ball at
            # all — no amount of length or straightness redeems it. Among
            # what survives, longer and straighter wins.
            if aim_px is not None and aim_px > aim_gate:
                continue
            score = 2.0 * len(run) - straight / 10.0
            if best is None or score > best[0]:
                best = (score, {
                    "points": [
                        {"frame": p["frame"], "x": int(p["x"]),
                         "y": int(p["y"])}
                        for p in run
                    ],
                    "straight_px": straight,
                    "aim_px": aim_px,
                    "aim_xy": aim_xy,
                })
    return best[1] if best else None
