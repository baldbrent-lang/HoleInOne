"""Where the brand assets live on disk, in one place.

ONE LOOKUP, because there were two and they disagreed. `notifications`
found the logo and `intro_overlay` did not, so the same build sent emails
with the wordmark on them and produced every clip with the drawn
fallback badge instead -- and nothing said so, because the fallback is a
legitimate-looking graphic rather than an error.

THE ONE THAT WORKED CHECKED `dist`. That is the difference and it is not
a detail: `frontend/public/` is where the file is COMMITTED, and Vite
copies it into `frontend/dist/` at build time. The Dockerfile ships
`COPY --from=web /web/dist frontend/dist` and never copies `public/` at
all, so in a deployed container `dist` is the only one of the two that
exists. A lookup that knows only about `public` finds the logo on a
developer's machine and never in production, which is the worst possible
place for the difference to show up.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("golfreelz.branding")

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _BACKEND_ROOT.parent

# Anchored to the repo root, NOT the working directory: start.sh runs
# uvicorn from inside backend/, so a bare "frontend/dist/..." resolves to
# backend/frontend/dist/... and silently finds nothing.
#
# `dist` first: it is what a deployed container has, and on a developer's
# machine it is the more recently built of the two. `public` second, so a
# checkout that has never been built still finds the committed file.
LOGO_CANDIDATES: tuple[Path, ...] = (
    _REPO_ROOT / "frontend" / "dist" / "golfreelz-logo.png",
    _REPO_ROOT / "frontend" / "public" / "golfreelz-logo.png",
    _BACKEND_ROOT / "app" / "assets" / "golfreelz-logo.png",
)


def logo_path() -> Path | None:
    """The brand logo PNG, or None with a WARNING naming what was tried.

    Warned rather than logged quietly. Every caller has a fallback that
    still produces something -- a drawn badge on a clip, a text-only
    email -- so a missing file never fails loudly on its own, and the
    only symptom is branding that silently reverts. That is worth a line
    in the log with the paths in it.
    """
    for p in LOGO_CANDIDATES:
        try:
            if p.exists() and p.stat().st_size > 0:
                return p
        except OSError as exc:  # pragma: no cover
            log.debug("branding: could not stat %s: %s", p, exc)
    log.warning(
        "branding: the GolfReelz logo is missing — every clip produced "
        "from this build will carry the drawn fallback badge instead of "
        "the wordmark. Looked in: %s",
        ", ".join(str(p) for p in LOGO_CANDIDATES),
    )
    return None
