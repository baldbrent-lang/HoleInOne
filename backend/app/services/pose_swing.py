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
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24

_pose = None
_pose_tried = False


def available() -> bool:
    return _get_pose() is not None


def _get_pose():
    """Lazily construct a MediaPipe Pose. Returns None if mediapipe isn't
    installed on this deployment (dev-only)."""
    global _pose, _pose_tried
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
        log.info("pose_swing: mediapipe unavailable (%s)", exc)
        return None


def annotate_frame(input_path, time_sec: float, fps: float, out_path) -> bool:
    """Grab the frame at time_sec, draw the detected pose skeleton on it, and
    write it to out_path (for a per-swing verification screenshot). Returns
    True on success. No-op / False when mediapipe is unavailable."""
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
        cv2.putText(
            frame, f"pose swing @ {time_sec:.1f}s", (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (155, 89, 182), 2, cv2.LINE_AA,
        )
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("pose_swing: annotate failed: %s", exc)
        return False


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
    start_sec: float = 0.0,
    max_scan_sec: float | None = None,
    debug: dict | None = None,
) -> list[dict]:
    """Find swings from wrist-speed bursts, gated by back bend.

    A wrist-speed burst only counts as a swing if the golfer's spine is bent
    over at the peak (a real swing has the torso tilted toward the ball;
    someone standing straight who just moves their hands does not). Rejects
    fast-hands-but-upright false positives. Returns the same segment shape as
    the other detectors. Empty + a debug reason when mediapipe is missing or
    nothing is found."""
    if debug is not None:
        debug.update({"reason": None, "method": "pose_wrist_speed", "available": False})

    pose = _get_pose()
    if pose is None:
        if debug is not None:
            debug["reason"] = "mediapipe not installed on this deployment"
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

    wrist: list[tuple[float, float] | None] = []  # per sample: (x,y) normalized or None
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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with _pose_lock:
                res = pose.process(rgb)
            lm = getattr(res, "pose_landmarks", None)
            if lm is None:
                wrist.append(None)
                bend.append(None)
                continue
            pts = lm.landmark
            bend.append(_spine_deg(pts))
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
            debug["reason"] = f"pose found in only {n_pose} frame(s)"
            debug["n_pose_frames"] = int(n_pose)
            debug["reached_eof"] = reached_eof
        return []

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

    accepted = []
    for s_i, e_i, p_i, p_v in bursts:
        dur = (e_i - s_i) / eff_hz if eff_hz > 0 else 0.0
        if min_burst_sec <= dur <= max_burst_sec:
            accepted.append((s_i, e_i, p_i, p_v))

    # Non-max suppression by peak separation.
    accepted.sort(key=lambda t: -t[3])
    chosen, keep = [], []
    min_sep = int(min_separation_sec * eff_hz)
    for s_i, e_i, p_i, p_v in accepted:
        if any(abs(p_i - c) < min_sep for c in chosen):
            continue
        chosen.append(p_i)
        keep.append((s_i, e_i, p_i, p_v))
    keep.sort(key=lambda t: t[2])

    def _bend_near(p_i):
        """Max spine bend within the burst around p_i (the golfer is bent over
        somewhere in the swing even if the exact peak frame lost the pose).
        None if no pose landmarks nearby."""
        lo, hi = max(0, p_i - int(round(0.4 * eff_hz))), min(len(bend), p_i + int(round(0.4 * eff_hz)) + 1)
        vals = [bend[i] for i in range(lo, hi) if bend[i] is not None]
        return max(vals) if vals else None

    segments = []
    n_bend_rejected = 0
    for s_i, e_i, p_i, p_v in keep:
        b = _bend_near(p_i)
        # A swing requires BOTH a fast-hands burst AND a bent-over spine.
        # Reject when the golfer is standing upright, AND when the spine bend
        # can't be confirmed at all (pose lost the torso) — both conditions
        # must be positively present, not merely "not disproven".
        if b is None or b < back_bend_min_deg:
            n_bend_rejected += 1
            continue
        peak_t = times[p_i] if p_i < len(times) else (p_i / eff_hz)
        ratio = p_v / median if median > 0 else 0.0
        conf = "high" if ratio >= 10 else ("medium" if ratio >= 6 else "low")
        segments.append({
            "peak_time_sec": float(peak_t),
            "start_sec": float(max(0.0, peak_t - before_sec)),
            "end_sec": float(min(duration, peak_t + after_sec)),
            "confidence": conf,
            "ratio": round(float(ratio), 1),
            "back_bend_deg": round(float(b), 1),
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
        "pose_swing: %d swings (bursts=%d, bend-rejected=%d) pose_frames=%d hz=%.1f",
        len(segments), len(bursts), n_bend_rejected, n_pose, eff_hz,
    )
    if debug is not None:
        debug.update({
            "duration_sec": float(duration),
            "n_pose_frames": int(n_pose),
            "n_samples": int(len(times)),
            "coverage": round(n_pose / len(times), 2) if times else 0.0,
            "median": float(median),
            "threshold": float(threshold),
            "n_raw_bursts": len(bursts),
            "n_bend_rejected": int(n_bend_rejected),
            "back_bend_min_deg": float(back_bend_min_deg),
            "n_swings": len(segments),
            "reached_eof": reached_eof,
            "series": [round(float(v), 5) for v in series],
            "peaks": [round(float(s["peak_time_sec"]), 2) for s in segments],
        })
    return segments
