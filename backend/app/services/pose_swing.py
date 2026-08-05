"""Pose-based swing detector (dev course-testing tool).

An alternative to the motion and ball detectors that reads the GOLFER, not
the ball — so it's immune to the club occluding the ball and to whole-frame
motion aliasing. Uses MediaPipe Pose to track the wrists frame-by-frame: a
golf swing is a big, fast wrist excursion (backswing up, downswing through
impact), so a burst of high wrist speed = one swing, and the speed peak is
~impact.

MediaPipe is an OPTIONAL dependency. It pulls in OpenCV 5.0 (replacing our
pinned 4.10), so it is deliberately NOT in requirements.txt — install it on
the DEV deployment only to keep prod on 4.10. Use a version that still ships
the legacy Solutions API with a BUNDLED pose model (no external download,
which the deploy proxy blocks): `pip install "mediapipe==0.10.14"`. Newer
0.10.35 removed `mp.solutions` in favour of the Tasks API (needs a separate
.task model file), so it won't work here. When mediapipe isn't importable
every function is a no-op that reports the reason, so nothing else breaks.
"""

from __future__ import annotations

import logging
import threading
import warnings

log = logging.getLogger("golfreelz.pose_swing")

# MediaPipe's Pose object is NOT thread-safe, but we cache a single global
# instance. Produce + debug + the non-golf scan can each run pose on their
# own thread at the same time; concurrent .process() calls on the shared
# instance deadlock MediaPipe. Serialize every .process() with this lock.
_pose_lock = threading.Lock()

# MediaPipe's bundled protobuf emits this UserWarning on every frame it
# processes, which floods the console during a scan. It's a harmless
# deprecation notice inside a dependency — silence just that one.
warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated",
    category=UserWarning,
)

# MediaPipe pose landmark indices.
_LEFT_WRIST = 15
_RIGHT_WRIST = 16
# Feet — the bottom of the golfer. The ball sits on the ground near this
# line, so it bounds the search box for a resting ball.
_NOSE = 0
_LEFT_ANKLE = 27
_RIGHT_ANKLE = 28
_LEFT_FOOT = 31
_RIGHT_FOOT = 32
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24

_pose = None
_pose_tried = False
# The actual import/construction error, surfaced to the debug UI so
# "mediapipe not installed" isn't reported when the real problem is a
# missing system lib (libGL), a numpy ABI clash, etc.
_pose_error: str | None = None


def available() -> bool:
    return _get_pose() is not None


def unavailable_reason() -> str | None:
    """Why mediapipe couldn't be loaded (None when it loaded fine)."""
    _get_pose()
    return _pose_error


def _get_pose():
    """Lazily construct a MediaPipe Pose. Returns None if mediapipe isn't
    importable on this deployment; the error is kept in _pose_error."""
    global _pose, _pose_tried, _pose_error
    if _pose is not None:
        return _pose
    if _pose_tried:
        return None
    _pose_tried = True
    try:
        import mediapipe as mp  # type: ignore

        _pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
        )
        log.info("pose_swing: mediapipe Pose ready")
        return _pose
    except Exception as exc:  # noqa: BLE001
        _pose_error = f"{type(exc).__name__}: {exc}"
        log.warning("pose_swing: mediapipe unavailable — %s", _pose_error)
        return None


def annotate_frame(
    input_path, time_sec: float, fps: float, out_path, bend_deg=None,
) -> bool:
    """Grab the frame at time_sec, draw the detected pose skeleton on it, and
    write it to out_path (for a per-swing verification screenshot). When
    bend_deg is given, it's shown in the label. Returns True on success.
    No-op / False when mediapipe is unavailable."""
    pose = _get_pose()
    if pose is None:
        return False
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore

        cap = cv2.VideoCapture(str(input_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0.0, time_sec) * (fps or 30.0)))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return False
        with _pose_lock:
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        lm = getattr(res, "pose_landmarks", None)
        if lm is not None:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, lm, mp.solutions.pose.POSE_CONNECTIONS,
            )
            # Extra-visible ring on each wrist (the swing signal).
            h, w = frame.shape[:2]
            for wi in (_LEFT_WRIST, _RIGHT_WRIST):
                p = lm.landmark[wi]
                if getattr(p, "visibility", 0.0) >= 0.3:
                    cv2.circle(
                        frame, (int(p.x * w), int(p.y * h)),
                        max(10, int(h * 0.02)), (155, 89, 182), 3, cv2.LINE_AA,
                    )
        _lbl = f"pose swing @ {time_sec:.1f}s"
        if bend_deg is not None:
            _lbl += f"  ·  back bend {bend_deg:.0f}deg"
        cv2.putText(
            frame, _lbl, (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (155, 89, 182), 2, cv2.LINE_AA,
        )
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("pose_swing: annotate failed: %s", exc)
        return False


# How many consecutive full-frame misses before the tiled bootstrap
# starts sweeping, and how often it sweeps once it has. A sweep is one
# pose call per tile, so it is the expensive path — bounded to "only
# while we have found nobody at all, and only every Nth sample".
_BOOTSTRAP_AFTER = 6
_BOOTSTRAP_EVERY = 3


def detect_swings_from_pose(
    input_path,
    fps: float | None = None,
    sample_hz: float = 15.0,
    min_separation_sec: float = 4.0,
    before_sec: float = 3.5,
    after_sec: float = 5.0,
    speed_ratio: float = 3.0,
    min_burst_sec: float = 0.25,
    max_burst_sec: float = 3.0,
    back_bend_min_deg: float = 15.0,
    back_bend_max_deg: float = 40.0,
    strong_ratio: float = 6.0,
    ratio_min: float = 5.0,
    ratio_max: float = 25.0,
    start_sec: float = 0.0,
    max_scan_sec: float | None = None,
    keep_rejected: bool = False,
    max_rejected: int = 12,
    debug: dict | None = None,
) -> list[dict]:
    """Find swings from wrist-speed bursts, gated by back bend.

    A wrist-speed burst only counts as a swing if the golfer's spine is bent
    over at the peak (a real swing has the torso tilted toward the ball;
    someone standing straight who just moves their hands does not). Rejects
    fast-hands-but-upright false positives. Returns the same segment shape as
    the other detectors. Empty + a debug reason when mediapipe is missing or
    nothing is found.

    `keep_rejected` returns the gate-rejected bursts too, each tagged with
    `gate_status` / `gate_ok`, so a caller can judge them by other evidence
    instead of losing them here. Debug3 uses it: the ball at impact and the
    AI judge are far better discriminators than a spine-angle threshold, but
    they never got a chance because this function had already thrown the
    burst away. Bounded by `max_rejected` (weakest bursts dropped first, and
    the count is reported in `debug`) so a long clip cannot hand the caller
    a hundred candidates to run vision models over."""
    if debug is not None:
        debug.update({"reason": None, "method": "pose_wrist_speed", "available": False})

    pose = _get_pose()
    if pose is None:
        if debug is not None:
            debug["reason"] = (
                f"mediapipe failed to load: {_pose_error}"
                if _pose_error
                else "mediapipe not installed on this deployment"
            )
        return []
    if debug is not None:
        debug["available"] = True

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:  # noqa: BLE001
        if debug is not None:
            debug["reason"] = "opencv/numpy not installed"
        return []

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        if debug is not None:
            debug["reason"] = "could not open video"
        return []
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

    wrist: list[tuple[float, float] | None] = []  # per sample: (x,y) normalized or None
    # Lowest visible foot/ankle per sample — the ground line under
    # the golfer, which bounds where a resting ball can be.
    feet: list[tuple[float, float] | None] = []
    # Head (nose) per sample. Everything above this line in the frame is
    # sky/trees — the clean part of the map, where the ball's trail has no
    # body heat to compete with.
    head: list[tuple[float, float] | None] = []
    bend: list[float | None] = []  # per sample: spine angle from vertical (deg)
    times: list[float] = []
    n_pose = 0

    def _spine_deg(pts) -> float | None:
        """Angle (deg) of the hip→shoulder spine from vertical. ~0 standing
        straight; larger when bent over the ball. None if hips/shoulders
        aren't confidently visible."""
        def _mid(a, b):
            pa, pb = pts[a], pts[b]
            if getattr(pa, "visibility", 0.0) < 0.3 or getattr(pb, "visibility", 0.0) < 0.3:
                return None
            return ((pa.x + pb.x) / 2.0, (pa.y + pb.y) / 2.0)

        sh = _mid(_LEFT_SHOULDER, _RIGHT_SHOULDER)
        hp = _mid(_LEFT_HIP, _RIGHT_HIP)
        if sh is None or hp is None:
            return None
        vx, vy = sh[0] - hp[0], sh[1] - hp[1]  # points toward the shoulders
        mag = (vx * vx + vy * vy) ** 0.5
        if mag < 1e-6:
            return None
        import math
        # vertical-up is (0,-1); cos = (-vy)/mag
        cos = max(-1.0, min(1.0, -vy / mag))
        return math.degrees(math.acos(cos))

    reached_eof = True
    try:
        src_fps = float(fps) if fps else float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if src_fps <= 0:
            src_fps = 30.0
        step = max(1, int(round(src_fps / sample_hz)))
        eff_hz = src_fps / step
        # Optional time window (chunked early-exit scanning). Seek to start,
        # stop after max_scan_sec; reached_eof tells the caller if there's more.
        start_frame = int(max(0.0, start_sec) * src_fps)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        max_scan_frames = int(max_scan_sec * src_fps) if max_scan_sec else None
        idx = start_frame - 1

        # PERSON-CROP ZOOM (distance fix): once the golfer is found, run
        # pose on a crop around them instead of the full frame. From a
        # course-distance camera the golfer is a small figure — landmark
        # jitter stays constant while true wrist motion shrinks, so the
        # swing's speed ratio collapses below the gate. Cropping makes
        # the golfer full-size again for the model; landmarks are mapped
        # back to FULL-FRAME normalized coords so every downstream
        # number (speed ratio, spine angle, wrist anchor) keeps its
        # exact meaning. static_image_mode=True means per-frame crops
        # are safe (no internal tracker to confuse).
        crop = None            # (x0, y0, x1, y1) px, or None = full frame
        crop_misses = 0
        n_crop = 0
        # BOOTSTRAP. The crop above is the distance fix, but it only ever
        # engages ONCE THE GOLFER HAS ALREADY BEEN FOUND on the full
        # frame — and at course distance that first find is exactly what
        # fails. MediaPipe wants a prominent subject; a golfer occupying
        # 2% of a 1280x720 frame is not one, so pose returns nothing, the
        # crop never starts, and the run ends "pose found in only 0
        # frame(s)" with the golfer plainly visible in the clip.
        #
        # So when the full frame has come up empty for a while, sweep a
        # grid of overlapping tiles until one of them finds a person, and
        # hand that to the normal crop tracking. Only until the first
        # hit, and only every few samples, so the cost is bounded.
        misses_full = 0
        n_boot_scans = 0
        boot_hit_at = None

        def _tiles():
            """Overlapping thirds across, halves down. A golfer standing
            anywhere lands whole inside at least one of them."""
            tw, th = int(vid_w * 0.45), int(vid_h * 0.55)
            out = []
            for fy in (0.0, 0.45):
                for fx in (0.0, 0.275, 0.55):
                    x0 = int(fx * vid_w)
                    y0 = int(fy * vid_h)
                    out.append((x0, y0, min(vid_w, x0 + tw),
                                min(vid_h, y0 + th)))
            return [t for t in out if t[2] - t[0] > 64 and t[3] - t[1] > 64]

        class _P:
            __slots__ = ("x", "y", "visibility")

            def __init__(self, x, y, v):
                self.x = x
                self.y = y
                self.visibility = v

        _KEY_LMS = (0, 11, 12, 15, 16, 23, 24, 25, 26, 27, 28)

        def _crop_from(ptsl):
            xs, ys = [], []
            for i in _KEY_LMS:
                pp = ptsl[i]
                if getattr(pp, "visibility", 0.0) >= 0.3:
                    xs.append(pp.x * vid_w)
                    ys.append(pp.y * vid_h)
            if len(xs) < 4:
                return None
            bx0, bx1 = min(xs), max(xs)
            by0, by1 = min(ys), max(ys)
            bw = max(bx1 - bx0, 1.0)
            bh = max(by1 - by0, 1.0)
            ccx = (bx0 + bx1) / 2.0
            ccy = (by0 + by1) / 2.0
            # Generous pad: the wrists + club sweep well outside the
            # standing bbox during a swing.
            half_w = max(1.5 * bw, 0.9 * bh, 128.0)
            half_h = max(1.3 * bh, 128.0)
            cx0 = max(0, int(ccx - half_w))
            cx1 = min(vid_w, int(ccx + half_w))
            cy0 = max(0, int(ccy - half_h))
            cy1 = min(vid_h, int(ccy + half_h))
            if cx1 - cx0 < 64 or cy1 - cy0 < 64:
                return None
            # Person already fills the frame — cropping buys nothing.
            if (cx1 - cx0) > 0.85 * vid_w and (cy1 - cy0) > 0.85 * vid_h:
                return None
            return (cx0, cy0, cx1, cy1)

        while True:
            idx += 1
            if max_scan_frames is not None and (idx - start_frame) >= max_scan_frames:
                reached_eof = False  # stopped at the chunk limit, not EOF
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx % step != 0:
                continue
            times.append(idx / src_fps)
            pts = None
            if crop is not None:
                cx0, cy0, cx1, cy1 = crop
                sub = frame[cy0:cy1, cx0:cx1]
                with _pose_lock:
                    res = pose.process(cv2.cvtColor(sub, cv2.COLOR_BGR2RGB))
                lm = getattr(res, "pose_landmarks", None)
                if lm is not None:
                    cw = float(cx1 - cx0)
                    ch = float(cy1 - cy0)
                    pts = [
                        _P(
                            (cx0 + pp.x * cw) / vid_w,
                            (cy0 + pp.y * ch) / vid_h,
                            getattr(pp, "visibility", 0.0),
                        )
                        for pp in lm.landmark
                    ]
                    n_crop += 1
            if pts is None:
                with _pose_lock:
                    res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                lm = getattr(res, "pose_landmarks", None)
                if lm is not None:
                    pts = [
                        _P(pp.x, pp.y, getattr(pp, "visibility", 0.0))
                        for pp in lm.landmark
                    ]
                    crop_misses = 0
                    misses_full = 0
                else:
                    crop_misses += 1
                    misses_full += 1
                    if crop_misses >= 2:
                        crop = None
            # Nothing on the full frame for a while: sweep the tiles.
            if (pts is None and boot_hit_at is None
                    and misses_full >= _BOOTSTRAP_AFTER
                    and misses_full % _BOOTSTRAP_EVERY == 0):
                n_boot_scans += 1
                for tx0, ty0, tx1, ty1 in _tiles():
                    with _pose_lock:
                        tres = pose.process(cv2.cvtColor(
                            frame[ty0:ty1, tx0:tx1], cv2.COLOR_BGR2RGB))
                    tlm = getattr(tres, "pose_landmarks", None)
                    if tlm is None:
                        continue
                    _tw = float(tx1 - tx0)
                    _th = float(ty1 - ty0)
                    _cand = [
                        _P((tx0 + pp.x * _tw) / vid_w,
                           (ty0 + pp.y * _th) / vid_h,
                           getattr(pp, "visibility", 0.0))
                        for pp in tlm.landmark
                    ]
                    # Only accept a tile hit that looks like a whole
                    # person; a hand or a tree branch scores landmarks too.
                    _vis = sum(1 for i in _KEY_LMS
                               if _cand[i].visibility >= 0.3)
                    if _vis < 6:
                        continue
                    pts = _cand
                    boot_hit_at = round(idx / src_fps, 2)
                    misses_full = 0
                    break
            if pts is None:
                wrist.append(None)
                feet.append(None)
                head.append(None)
                bend.append(None)
                continue
            crop = _crop_from(pts)
            bend.append(_spine_deg(pts))
            # Lowest visible foot/ankle — the ground line under the golfer.
            _fc = [
                (p.x, p.y) for p in (
                    pts[_LEFT_ANKLE], pts[_RIGHT_ANKLE],
                    pts[_LEFT_FOOT], pts[_RIGHT_FOOT],
                )
                if getattr(p, "visibility", 0.0) >= 0.3
            ]
            feet.append(max(_fc, key=lambda q: q[1]) if _fc else None)
            _nose = pts[_NOSE]
            head.append(
                (_nose.x, _nose.y)
                if getattr(_nose, "visibility", 0.0) >= 0.3 else None
            )
            cands = []
            for wi in (_LEFT_WRIST, _RIGHT_WRIST):
                p = pts[wi]
                if getattr(p, "visibility", 0.0) >= 0.3:
                    cands.append((p.x, p.y))
            if not cands:
                wrist.append(None)
                continue
            wx = sum(c[0] for c in cands) / len(cands)
            wy = sum(c[1] for c in cands) / len(cands)
            wrist.append((wx, wy))
            n_pose += 1
    finally:
        cap.release()

    if n_pose < 4:
        if debug is not None:
            debug["reason"] = (
                f"pose found in only {n_pose} frame(s)"
                + (f" — the tiled bootstrap swept {n_boot_scans} time(s) "
                   f"and never found a whole person either, so the golfer "
                   f"is likely too small or too occluded for the model"
                   if n_boot_scans else "")
            )
            debug["n_pose_frames"] = int(n_pose)
            debug["n_bootstrap_scans"] = int(n_boot_scans)
            debug["reached_eof"] = reached_eof
        return []

    # Bridge short tracking dropouts by linear interpolation (up to ~0.6s).
    # MediaPipe routinely loses the wrists for a few samples during the
    # motion-blurred downswing; without bridging, the speed signal zeroes
    # out mid-swing and one real swing fragments into slivers that each
    # fail the min-burst-duration gate ("burst too short"). Long gaps
    # (golfer walked off / pose fully lost) are left as None — inventing
    # motion across those would fabricate bursts.
    max_gap = max(1, int(round(0.6 * eff_hz)))
    n_bridged = 0
    last_known = None  # (index, (x, y))
    for i in range(len(wrist)):
        if wrist[i] is None:
            continue
        if last_known is not None:
            gap = i - last_known[0]
            if 1 < gap <= max_gap:
                (x0, y0), (x1, y1) = last_known[1], wrist[i]
                for k in range(1, gap):
                    t = k / gap
                    wrist[last_known[0] + k] = (
                        x0 + (x1 - x0) * t, y0 + (y1 - y0) * t,
                    )
                    n_bridged += 1
        last_known = (i, wrist[i])

    # Wrist speed between consecutive samples (0 when either endpoint missing).
    speed = np.zeros(len(wrist), dtype=np.float32)
    for i in range(1, len(wrist)):
        a, b = wrist[i - 1], wrist[i]
        if a is not None and b is not None:
            speed[i] = float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)
    # Smooth ~200 ms so one swing is one burst.
    win = max(1, int(round(0.2 * eff_hz)))
    if 1 < win < speed.size:
        speed = np.convolve(speed, np.ones(win, np.float32) / win, mode="same")

    median = float(np.median(speed[speed > 0])) if np.any(speed > 0) else 0.0
    if median <= 1e-6:
        median = 1e-6
    threshold = median * speed_ratio
    above = speed > threshold

    duration = times[-1] if times else 0.0
    bursts = []  # (start_i, end_i, peak_i, peak_v)
    i, n = 0, above.size
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        seg = speed[i:j]
        pk = int(np.argmax(seg))
        bursts.append((i, j - 1, i + pk, float(seg[pk])))
        i = j

    def _bend_near(p_i):
        """Max spine bend from ~1.5s BEFORE the peak through 0.4s after.

        The wrist-speed peak is the downswing — fast motion blurs the torso
        and pose often drops out there, so the bend is frequently unknown at
        the exact peak. But ~1-1.5s earlier the golfer is at address: set up
        over the ball, stationary, clearly bent, and reliably tracked. Look
        back to catch that address bend so a real swing isn't rejected just
        because the downswing frame lost the pose. A trotting/upright false
        positive is never bent anywhere in the window, so it stays rejected.
        None if no pose landmarks in the window at all.

        Bends above back_bend_max_deg (40°) are ignored: a real golf address
        sits in the ~15–40° band. Larger means the golfer is bent fully over
        (planting a tee / picking up a ball) OR — commonly — MediaPipe
        flipped the torso and put the shoulders below the hips (e.g. a bogus
        177°). Either way it's not a swing posture, so it must not count as a
        valid bend. Filtering per-frame (not just the max) means a real ~30°
        address bend is still used even if a garbage 61° or 177° sits in the
        same window."""
        lo = max(0, p_i - int(round(1.5 * eff_hz)))
        hi = min(len(bend), p_i + int(round(0.4 * eff_hz)) + 1)
        vals = [
            bend[i] for i in range(lo, hi)
            if bend[i] is not None and bend[i] <= back_bend_max_deg
        ]
        return max(vals) if vals else None

    # Duration-gate each raw burst; remember WHY each dropped one was dropped
    # so the debug view can explain a missed swing instead of it vanishing.
    #
    # Short-burst RESCUE: when pose coverage is poor, even the bridged burst
    # can come out a sample or two long — killing a real swing on duration
    # alone. A burst that's too short but (a) towers over baseline
    # (>= strong_ratio) AND (b) has a confirmed swing-posture bend is a real
    # swing whose neighbours got dropped, not noise — keep it. Weak or
    # posture-less slivers still die here.
    accepted = []
    burst_status: dict[int, str] = {}  # peak sample index -> outcome
    for s_i, e_i, p_i, p_v in bursts:
        dur = (e_i - s_i) / eff_hz if eff_hz > 0 else 0.0
        if dur < min_burst_sec:
            ratio_pk = (p_v / median) if median > 0 else 0.0
            b_pk = _bend_near(p_i)
            if (
                ratio_min <= ratio_pk <= ratio_max
                and b_pk is not None
                and b_pk >= back_bend_min_deg
            ):
                accepted.append((s_i, e_i, p_i, p_v))
            else:
                burst_status[p_i] = "too_short"
        elif dur > max_burst_sec:
            burst_status[p_i] = "too_long"
        else:
            accepted.append((s_i, e_i, p_i, p_v))

    # Posture gate BEFORE non-max suppression. A swing needs a fast-hands
    # burst AND a bent-over spine — but an UNKNOWN bend must not veto an
    # obviously-real swing (pose routinely drops the torso during the
    # blurred downswing). Reject only when we can POSITIVELY see it's not a
    # swing: a measured-but-upright spine, or a weak burst whose posture we
    # couldn't confirm. Order matters: gating first means a rejected burst
    # (e.g. a taller-but-upright spike 3s away) can no longer win the NMS
    # merge and then die, silently taking the real swing down with it.
    n_bend_rejected = 0
    gated = []  # (s_i, e_i, p_i, p_v, bend, ratio)
    for s_i, e_i, p_i, p_v in accepted:
        b = _bend_near(p_i)
        ratio = p_v / median if median > 0 else 0.0
        # Ratio BAND (operator-tuned): a real swing's wrist-speed peak
        # sits between ~5x and ~25x the resting-hands median. Below =
        # waggle / walk noise; above = tracking glitch (a landmark jump
        # reads as impossible speed).
        if ratio < ratio_min:
            burst_status[p_i] = "ratio_low"
            continue
        if ratio > ratio_max:
            burst_status[p_i] = "ratio_high"
            continue
        if b is not None and b < back_bend_min_deg:
            burst_status[p_i] = "upright"
            n_bend_rejected += 1
            continue
        if b is None and ratio < strong_ratio:
            burst_status[p_i] = "bend_unknown_weak"
            n_bend_rejected += 1
            continue
        gated.append((s_i, e_i, p_i, p_v, b, ratio))

    # Non-max suppression by peak separation.
    #
    # Normally only gate-passing bursts compete. With `keep_rejected` EVERY
    # burst does, because the gates are the thing being second-guessed: a
    # real swing rejected as upright or too-short is gone before this point,
    # and no later stage can rescue what was never returned. NMS still runs
    # over the combined pool so the caller gets ONE candidate per swing
    # rather than a duplicate for every burst inside the same 4 seconds.
    _pool = [(s_i, e_i, p_i, p_v, b, ratio, "swing")
             for s_i, e_i, p_i, p_v, b, ratio in gated]
    if keep_rejected:
        _passed = {t[2] for t in gated}
        for s_i, e_i, p_i, p_v in bursts:
            if p_i in _passed:
                continue
            _b = _bend_near(p_i)
            _r = p_v / median if median > 0 else 0.0
            _pool.append((s_i, e_i, p_i, p_v, _b, _r,
                          burst_status.get(p_i) or "rejected"))
    _pool.sort(key=lambda t: -t[3])
    chosen, keep = [], []
    min_sep = int(min_separation_sec * eff_hz)
    for _rec in _pool:
        p_i = _rec[2]
        if any(abs(p_i - c) < min_sep for c in chosen):
            if _rec[6] == "swing":
                burst_status[p_i] = "nms_suppressed"
            continue
        chosen.append(p_i)
        keep.append(_rec)

    # Bound the rescued set: weakest first, so a long clip cannot hand the
    # caller a hundred candidates to run vision models over. Never silent —
    # what was dropped is reported.
    n_rescued_dropped = 0
    if keep_rejected:
        _resc = [r for r in keep if r[6] != "swing"]
        if len(_resc) > max_rejected:
            _resc.sort(key=lambda t: -t[5])          # by ratio, best first
            _cut = {id(r) for r in _resc[max_rejected:]}
            n_rescued_dropped = len(_cut)
            keep = [r for r in keep if id(r) not in _cut]
    keep.sort(key=lambda t: t[2])

    def _wrist_native(p_i):
        """Mean-wrist position (native pixels) at/near the swing peak — the
        golfer's hands where the ball is struck. Pose often drops out at the
        exact peak (motion blur), so search outward up to ~0.5s for the
        nearest tracked wrist. Returns [x, y] or None. Used as the tracer's
        start anchor so the line begins at the strike point even when the
        ball itself can't be seen (backlit / dark ground)."""
        if not (vid_w and vid_h):
            return None
        span = int(round(0.5 * eff_hz)) + 1
        for off in range(span):
            for j in (p_i - off, p_i + off):
                if 0 <= j < len(wrist) and wrist[j] is not None:
                    wx, wy = wrist[j]
                    return [int(round(wx * vid_w)), int(round(wy * vid_h))]
        return None

    def _head_native(p_i):
        """Nose position (native px) at/near the peak — the top of the
        golfer. Same outward search as the wrist."""
        if not (vid_w and vid_h):
            return None
        span = int(round(0.5 * eff_hz)) + 1
        for off in range(span):
            for j in (p_i - off, p_i + off):
                if 0 <= j < len(head) and head[j] is not None:
                    hx, hy = head[j]
                    return [int(round(hx * vid_w)), int(round(hy * vid_h))]
        return None

    def _feet_native(p_i):
        """Lowest visible foot/ankle (native pixels) at/near the peak — the
        ground line under the golfer. Same outward search as the wrist,
        since pose drops out at the blurred peak. Returns [x, y] or None."""
        if not (vid_w and vid_h):
            return None
        span = int(round(0.5 * eff_hz)) + 1
        for off in range(span):
            for j in (p_i - off, p_i + off):
                if 0 <= j < len(feet) and feet[j] is not None:
                    fx, fy = feet[j]
                    return [int(round(fx * vid_w)), int(round(fy * vid_h))]
        return None

    segments = []
    for s_i, e_i, p_i, p_v, b, ratio, gate in keep:
        if gate == "swing":
            burst_status[p_i] = "swing"
        peak_t = times[p_i] if p_i < len(times) else (p_i / eff_hz)
        conf = "high" if ratio >= 10 else ("medium" if ratio >= 6 else "low")
        segments.append({
            # Which gate this burst cleared, or the one it failed. Callers
            # that did not ask for rejects only ever see "swing".
            "gate_status": gate,
            "gate_ok": gate == "swing",
            "peak_time_sec": float(peak_t),
            "start_sec": float(max(0.0, peak_t - before_sec)),
            "end_sec": float(min(duration, peak_t + after_sec)),
            "confidence": conf,
            "ratio": round(float(ratio), 1),
            "back_bend_deg": (round(float(b), 1) if b is not None else None),
            # Hands position at the strike — the tracer's start anchor.
            "impact_wrist_xy": _wrist_native(p_i),
            # Ground line under the golfer — bounds the ball search box.
            "impact_feet_xy": _feet_native(p_i),
            # Head line — above it the map is sky and trees.
            "impact_head_xy": _head_native(p_i),
        })

    # Decimate the wrist-speed waveform for plotting (peak-preserving).
    series = speed
    max_pts = 600
    if series.size > max_pts:
        b = int(np.ceil(series.size / max_pts))
        pad = (-series.size) % b
        if pad:
            series = np.concatenate([series, np.full(pad, series[-1], series.dtype)])
        series = series.reshape(-1, b).max(axis=1)

    log.info(
        "pose_swing: %d swings (bursts=%d, bend-rejected=%d, rescued=%d, "
        "rescued-dropped=%d) pose_frames=%d hz=%.1f",
        len(segments), len(bursts), n_bend_rejected,
        sum(1 for s in segments if not s.get("gate_ok")), n_rescued_dropped,
        n_pose, eff_hz,
    )
    if debug is not None:
        debug.update({
            "duration_sec": float(duration),
            "n_pose_frames": int(n_pose),
            "n_samples": int(len(times)),
            "coverage": round(n_pose / len(times), 2) if times else 0.0,
            "n_bridged": int(n_bridged),
            "n_crop_frames": int(n_crop),
            "n_bootstrap_scans": int(n_boot_scans),
            "bootstrap_found_at": boot_hit_at,
            "median": float(median),
            "threshold": float(threshold),
            "n_raw_bursts": len(bursts),
            "n_bend_rejected": int(n_bend_rejected),
            "back_bend_min_deg": float(back_bend_min_deg),
            "back_bend_max_deg": float(back_bend_max_deg),
            "strong_ratio": float(strong_ratio),
            "ratio_min": float(ratio_min),
            "ratio_max": float(ratio_max),
            "n_swings": len(segments),
            "n_gate_passed": sum(1 for s in segments if s.get("gate_ok")),
            "n_rescued": sum(1 for s in segments if not s.get("gate_ok")),
            "n_rescued_dropped": int(n_rescued_dropped),
            "reached_eof": reached_eof,
            "series": [round(float(v), 5) for v in series],
            "peaks": [round(float(s["peak_time_sec"]), 2) for s in segments],
            "swing_bends": [s.get("back_bend_deg") for s in segments],
        })
        # Per-burst breakdown: every candidate the detector saw, with why it
        # was kept or dropped, so a missed swing is diagnosable at a glance.
        detail = []
        for s_i, e_i, p_i, p_v in bursts:
            t = times[p_i] if p_i < len(times) else p_i / eff_hz
            bd = _bend_near(p_i)
            detail.append({
                "t": round(float(t), 2),
                "ratio": round(float(p_v / median) if median > 0 else 0.0, 1),
                "dur": round((e_i - s_i) / eff_hz, 2) if eff_hz > 0 else 0.0,
                "bend": (round(float(bd), 0) if bd is not None else None),
                "status": burst_status.get(p_i, "?"),
            })
        detail.sort(key=lambda d: d["t"])
        debug["bursts_detail"] = detail
    return segments
