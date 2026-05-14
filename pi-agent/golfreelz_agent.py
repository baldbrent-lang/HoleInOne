#!/usr/bin/env python3
"""GolfReelz Pi capture agent — entry point.

Loads config.yaml (or the path passed as the first arg), dispatches
to the tee or green runner based on the role the backend assigns to
this camera's auth_token.

Usage:
    python3 golfreelz_agent.py [/path/to/config.yaml]

The role is determined at runtime: the first heartbeat returns the
camera's assigned_role, and the agent picks tee.run() or green.run()
based on that. So a single SD-card image works for both deployment
positions — only the config.yaml differs.
"""
from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path

from agent.common import BackendClient, load_config


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str]) -> int:
    cfg_path = Path(argv[1]) if len(argv) > 1 else Path("config.yaml")
    if not cfg_path.exists():
        print(f"config not found at {cfg_path}; see config.example.yaml", file=sys.stderr)
        return 2
    cfg = load_config(cfg_path)
    setup_logging(cfg.get("log_level", "INFO"))
    log = logging.getLogger("golfreelz_agent")

    # First heartbeat tells us our assigned role.
    client = BackendClient(cfg["backend_url"], cfg["auth_token"])
    try:
        hb = client.heartbeat(firmware_version="bootstrap")
    except Exception as exc:
        log.error("initial heartbeat failed: %s", exc)
        return 1
    role = (hb.get("assigned_role") or "").lower()
    if role not in ("tee", "green"):
        log.error("backend returned unknown assigned_role=%r", role)
        return 1
    log.info(
        "this camera = #%s on course %s hole %s role=%s",
        hb.get("camera_id"), hb.get("course_id"),
        hb.get("assigned_hole"), role,
    )
    if not hb.get("enabled"):
        log.error("camera is disabled on the backend — exiting")
        return 1

    if role == "tee":
        from agent.tee import TeeAgent
        if not cfg.get("tee_box_roi"):
            log.error("tee_box_roi missing in config.yaml — required for tee role")
            return 2
        agent = TeeAgent(cfg)
    else:
        from agent.green import GreenAgent
        agent = GreenAgent(cfg)

    # Graceful Ctrl-C / systemd stop.
    def _shutdown(signum, _frame):
        log.info("signal %d received — shutting down", signum)
        agent.stop()
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        agent.run()
    except KeyboardInterrupt:
        agent.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
