"""How much CPU this container actually has, and telling OpenCV about it.

WHY THIS EXISTS. `os.cpu_count()` reports the HOST's cores, not the
share this container is allowed to use. On a throttled deployment those
two numbers can be wildly different -- eight cores visible, one core's
worth of quota -- and OpenCV believes the first one. It then runs every
`cv2` operation across eight threads that are collectively descheduled
whenever they exceed the quota, so a decode spends its time being
throttled and context-switched rather than decoding.

The effect is worst on exactly the work the produce does: long runs of
per-frame OpenCV calls, which is a thread pool started and stopped
thousands of times. Matching the pool to the real quota is not a
micro-optimisation on a box like that; it is the difference between
using the CPU you have and thrashing it.

On an unthrottled machine the quota is unlimited, this reads the real
core count, and nothing changes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("golfreelz.cpu")


def cgroup_quota() -> float | None:
    """CPU cores this container may use, or None when unlimited/unknown.

    Both cgroup layouts, because which one a host uses is not something
    the application gets to choose:
      v2  /sys/fs/cgroup/cpu.max      -> "<quota> <period>" or "max ..."
      v1  cpu.cfs_quota_us / cpu.cfs_period_us, quota -1 when unlimited
    """
    try:
        p2 = Path("/sys/fs/cgroup/cpu.max")
        if p2.exists():
            parts = p2.read_text().split()
            if len(parts) >= 2 and parts[0] != "max":
                q, per = float(parts[0]), float(parts[1])
                if q > 0 and per > 0:
                    return q / per
            return None
        pq = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        pp = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if pq.exists() and pp.exists():
            q, per = float(pq.read_text().strip()), float(pp.read_text().strip())
            if q > 0 and per > 0:
                return q / per
    except (OSError, ValueError) as exc:
        log.debug("could not read the cgroup CPU quota: %s", exc)
    return None


def effective_cpus() -> int:
    """Cores this process can actually use, at least 1.

    The smaller of the quota and the visible core count -- a quota above
    the core count does not conjure cores, and a core count above the
    quota is the lie this module exists to stop believing.
    """
    host = os.cpu_count() or 1
    q = cgroup_quota()
    if q is None:
        return max(1, host)
    return max(1, min(host, int(q)))


def describe() -> dict:
    """What the machine is, for a timing report to carry.

    Reported rather than only logged, because "why is production six
    times slower than dev on the same video" is answerable from two
    timing tables side by side ONLY if each one says what it ran on.
    """
    out = {
        "host_cpus": os.cpu_count(),
        "cgroup_quota_cpus": cgroup_quota(),
        "effective_cpus": effective_cpus(),
        "cv2_threads": None,
    }
    try:
        import cv2  # type: ignore

        out["cv2_threads"] = cv2.getNumThreads()
    except Exception as exc:  # noqa: BLE001
        log.debug("cv2 thread count unavailable: %s", exc)
    return out


def tune_opencv() -> dict:
    """Point OpenCV's thread pool at the quota. Safe to call twice."""
    info = describe()
    want = info["effective_cpus"]
    try:
        import cv2  # type: ignore

        before = cv2.getNumThreads()
        if before != want:
            cv2.setNumThreads(want)
            info["cv2_threads"] = cv2.getNumThreads()
            log.info(
                "opencv threads %s -> %s (host reports %s core(s), cgroup "
                "allows %s)", before, info["cv2_threads"], info["host_cpus"],
                info["cgroup_quota_cpus"],
            )
        else:
            log.info("opencv threads already %s (host %s, quota %s)",
                     before, info["host_cpus"], info["cgroup_quota_cpus"])
    except Exception as exc:  # noqa: BLE001
        log.warning("could not tune opencv threads: %s", exc)
    return info
