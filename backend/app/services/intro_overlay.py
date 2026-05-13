"""Intro overlay for processed swing clips.

Renders two panels onto the first ~3.5s of a clip:
  - Left: player name, course name, hole / par / yardage
  - Right: GolfReelz brand wordmark + flag icon

Both panels slide in from off-screen at t=0, hold for ~2.5s, then slide
back out by t=3.4s. The rest of the clip plays normally underneath.

Implementation:
  1. Render each panel as a transparent PNG via PIL (one-shot, ~50 ms
     of work).
  2. ffmpeg overlay filter composites both onto the input video with
     time-based x-position expressions for the slide animation.

Best-effort: every public function returns False / a no-op on failure
so the caller can keep shipping the underlying video even if the
overlay step breaks.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger("golfreelz.intro_overlay")

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
    HAS_PIL = True
except Exception:  # pragma: no cover
    Image = ImageDraw = ImageFont = None  # type: ignore
    HAS_PIL = False


# Brand palette
BRAND_GREEN = (28, 168, 92, 255)
PANEL_TOP_GRAY = (230, 232, 236, 255)   # light gray for the player-name section
PANEL_BOTTOM_BLUE = (10, 26, 70, 245)   # deep navy for the course/info section
TEXT_BLACK = (12, 16, 22, 255)
TEXT_PRIMARY = (255, 255, 255, 255)
TEXT_MUTED = (210, 218, 230, 255)

# Animation timing (seconds)
SLIDE_IN_SEC = 0.4
HOLD_SEC = 3.6
SLIDE_OUT_SEC = 0.4

# Font search paths — most Linux containers ship DejaVu. Try a couple of
# fallbacks in order; the renderer falls back to PIL's default if none of
# these are present (still readable, just not as polished).
FONT_CANDIDATES_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
]
FONT_CANDIDATES_REGULAR = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
]


def _find_font(candidates: list[str], size: int):
    if not HAS_PIL:
        return None
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:  # pragma: no cover
        return None


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _measure(text: str, font) -> int:
    """Best-effort horizontal text measurement for PIL fonts. Handles
    both modern (getlength) and legacy (font default) Pillow APIs."""
    if not text:
        return 0
    if hasattr(font, "getlength"):
        try:
            return int(round(font.getlength(text)))
        except Exception:
            pass
    # Fallback when getlength isn't available (very old Pillow / load_default).
    return int(round(len(text) * (getattr(font, "size", 14) * 0.55)))


def render_left_panel(
    player_name: str | None,
    course_name: str,
    hole_number: int | None,
    par: int | None,
    yardage: int | None,
) -> Image.Image | None:
    """Two-section left panel: light-gray top (player name in black)
    over a blue bottom (course + hole/par/yardage in white). Width is
    computed dynamically from the rendered text widths so the panel is
    only as wide as it needs to be."""
    if not HAS_PIL:
        return None

    bold_big = _find_font(FONT_CANDIDATES_BOLD, 30)
    bold_med = _find_font(FONT_CANDIDATES_BOLD, 18)
    reg_med = _find_font(FONT_CANDIDATES_REGULAR, 17)

    player_text = (player_name or "Brent Baldwin").upper()
    course_text = course_name or ""
    par_value = 3 if par is None else int(par)
    yards_value = int(yardage) if yardage is not None else 101
    info_parts = []
    if hole_number is not None:
        info_parts.append(f"HOLE {int(hole_number)}")
    info_parts.append(f"PAR {par_value}")
    info_parts.append(f"{yards_value} YDS")
    info_text = "  ·  ".join(info_parts)

    pad_x = 18
    pad_y_top = 12
    pad_y_bot = 14
    line_gap = 6

    w_player = _measure(player_text, bold_big)
    w_course = _measure(course_text, reg_med)
    w_info = _measure(info_text, bold_med)
    content_w = max(w_player, w_course, w_info)
    width = content_w + 2 * pad_x

    # Section heights — top holds the player name, bottom stacks course
    # name (if any) above the info row.
    top_h = (bold_big.size if hasattr(bold_big, "size") else 30) + 2 * pad_y_top
    info_h = (bold_med.size if hasattr(bold_med, "size") else 18)
    course_h = (reg_med.size if hasattr(reg_med, "size") else 17) + line_gap if course_text else 0
    bottom_h = pad_y_bot + course_h + info_h + pad_y_bot
    height = top_h + bottom_h

    # Build the colored sections first, then clip to a rounded outer
    # mask. PIL's rounded_rectangle doesn't support per-corner radius,
    # so an outer mask is cleaner than stitching two rounded shapes.
    sections = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sections)
    sd.rectangle((0, 0, width, top_h), fill=PANEL_TOP_GRAY)
    sd.rectangle((0, top_h, width, height), fill=PANEL_BOTTOM_BLUE)

    mask = Image.new("L", (width, height), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle((0, 0, width - 1, height - 1), radius=12, fill=255)

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    img.paste(sections, (0, 0), mask)

    draw = ImageDraw.Draw(img)
    # Player name centered vertically in the top section
    draw.text(
        (pad_x, pad_y_top), player_text, fill=TEXT_BLACK, font=bold_big,
    )

    # Bottom section: course (optional) + info row
    y = top_h + pad_y_bot
    if course_text:
        draw.text((pad_x, y), course_text, fill=TEXT_MUTED, font=reg_med)
        y += course_h
    draw.text((pad_x, y), info_text, fill=TEXT_PRIMARY, font=bold_med)
    return img


def _flag_icon(img: Image.Image, x: int, y: int, size: int, fill):
    """Tiny vector flag icon, scaled to fit `size`. Matches the React
    FlagIcon component's geometry — stick + triangle flag + base ball.
    Drawn into the supplied image at (x, y)."""
    draw = ImageDraw.Draw(img)
    # Stick (vertical line at the left side of the icon box)
    stick_x = x + int(size * 0.30)
    draw.line(
        [(stick_x, y + int(size * 0.10)), (stick_x, y + int(size * 0.90))],
        fill=fill, width=max(2, size // 12),
    )
    # Flag (right-pointing triangle)
    flag_top = (stick_x, y + int(size * 0.10))
    flag_tip = (x + int(size * 0.85), y + int(size * 0.30))
    flag_bot = (stick_x, y + int(size * 0.50))
    draw.polygon([flag_top, flag_tip, flag_bot], fill=fill)
    # Base ball
    r = max(2, size // 10)
    draw.ellipse(
        (stick_x - r, y + int(size * 0.90) - r, stick_x + r, y + int(size * 0.90) + r),
        fill=fill,
    )


def _find_brand_logo() -> Path | None:
    """Locate the GolfReelz brand logo on disk. The canonical location
    is `frontend/public/golfreelz-logo.png` at the repo root; we also
    check a backend-local copy in case the frontend tree isn't shipped
    with the backend deploy."""
    # services/intro_overlay.py → ../../../ = repo root.
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    candidates = [
        repo_root / "frontend" / "public" / "golfreelz-logo.png",
        here.parent.parent / "assets" / "golfreelz-logo.png",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def render_right_panel(
    height: int = 70,
) -> Image.Image | None:
    """PIL Image of the right brand panel. Uses the GolfReelz logo PNG
    when it exists on disk; falls back to a drawn flag-icon + wordmark
    otherwise so the overlay still ships during initial deploys."""
    if not HAS_PIL:
        return None

    logo_path = _find_brand_logo()
    if logo_path is not None:
        try:
            logo = Image.open(logo_path).convert("RGBA")
        except Exception as exc:  # pragma: no cover
            log.warning("intro_overlay: failed to open logo %s: %s", logo_path, exc)
            logo = None
    else:
        logo = None

    if logo is not None:
        # Source PNG has ~20% blank space top and bottom. Crop to the
        # middle 60% vertically so the logo content fills the panel
        # without visible buffer above and below.
        crop_top = int(round(logo.height * 0.20))
        crop_bottom = int(round(logo.height * 0.80))
        logo = logo.crop((0, crop_top, logo.width, crop_bottom))

        pad_x = 16
        pad_y = 6
        # Scale the cropped logo to fit the panel height (minus padding).
        target_h = height - 2 * pad_y
        scale = target_h / float(logo.height)
        logo_w = int(round(logo.width * scale))
        logo_h = target_h
        logo_scaled = logo.resize((logo_w, logo_h), Image.LANCZOS)
        width = logo_w + 2 * pad_x
        # White-ish rounded background — the logo is dark-on-light by
        # design, so we need a light fill underneath so it's legible
        # against sky / grass video.
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        bg = Image.new("RGBA", (width, height), (255, 255, 255, 240))
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            (0, 0, width - 1, height - 1), radius=12, fill=255,
        )
        img.paste(bg, (0, 0), mask)
        img.paste(logo_scaled, (pad_x, pad_y), logo_scaled)
        return img

    # Fallback: drawn flag icon + 'GolfReelz' wordmark, used until the
    # PNG is dropped into frontend/public/.
    width = 220
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        (0, 0, width - 1, height - 1), radius=10, fill=BRAND_GREEN,
    )
    icon_size = height - 26
    _flag_icon(img, 14, (height - icon_size) // 2, icon_size, TEXT_PRIMARY)
    bold = _find_font(FONT_CANDIDATES_BOLD, 26)
    text_x = 14 + icon_size + 10
    text_y = (height - 32) // 2
    draw.text((text_x, text_y), "GolfReelz", fill=TEXT_PRIMARY, font=bold)
    return img


def _x_expr_left(panel_w: int, video_w: int, anchor_x: int) -> str:
    """ffmpeg overlay x= expression for the left panel.

    Slide in from x=-w → x=anchor_x over SLIDE_IN_SEC, hold, slide out
    back to x=-w over SLIDE_OUT_SEC starting at SLIDE_IN_SEC + HOLD_SEC.
    """
    t_in = SLIDE_IN_SEC
    t_hold_end = SLIDE_IN_SEC + HOLD_SEC
    t_out = t_hold_end + SLIDE_OUT_SEC
    # In: x interpolates from -w to anchor_x
    # Hold: x = anchor_x
    # Out: x interpolates from anchor_x back to -w
    # After: x = -w (off-screen)
    return (
        f"if(lt(t,{t_in}), -w+(w+{anchor_x})*t/{t_in}, "
        f"if(lt(t,{t_hold_end}), {anchor_x}, "
        f"if(lt(t,{t_out}), {anchor_x} - (w+{anchor_x})*(t-{t_hold_end})/{SLIDE_OUT_SEC}, "
        f"-w)))"
    )


def _x_expr_right(panel_w: int, video_w: int, anchor_right_pad: int) -> str:
    """ffmpeg overlay x= expression for the right panel.

    Right panel slides in from x=W → x=W-w-pad, holds, slides back out
    to x=W.
    """
    t_in = SLIDE_IN_SEC
    t_hold_end = SLIDE_IN_SEC + HOLD_SEC
    t_out = t_hold_end + SLIDE_OUT_SEC
    anchor = f"(W-w-{anchor_right_pad})"
    return (
        f"if(lt(t,{t_in}), W - (W-{anchor})*t/{t_in}, "
        f"if(lt(t,{t_hold_end}), {anchor}, "
        f"if(lt(t,{t_out}), {anchor} + (W-{anchor})*(t-{t_hold_end})/{SLIDE_OUT_SEC}, "
        f"W)))"
    )


def apply_intro_overlay(
    input_video: Path,
    output_video: Path,
    player_name: str | None,
    course_name: str,
    hole_number: int | None,
    par: int | None,
    yardage: int | None,
) -> bool:
    """Composite the slide-in/out intro panels onto input_video and
    write the result to output_video. Returns True on success, False
    on any failure (PIL missing, ffmpeg missing, panel render failure,
    ffmpeg failure). Leaves input_video untouched on failure.
    """
    if not HAS_PIL:
        log.warning("intro_overlay: PIL not installed; skipping")
        return False
    if not _have_ffmpeg():
        log.warning("intro_overlay: ffmpeg not on PATH; skipping")
        return False
    if not input_video.exists():
        log.warning("intro_overlay: input %s missing", input_video)
        return False

    left_img = render_left_panel(
        player_name=player_name,
        course_name=course_name,
        hole_number=hole_number,
        par=par,
        yardage=yardage,
    )
    right_img = render_right_panel()
    if left_img is None or right_img is None:
        log.warning("intro_overlay: panel render returned None")
        return False

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        left_png = td_path / "intro_left.png"
        right_png = td_path / "intro_right.png"
        left_img.save(left_png)
        right_img.save(right_png)

        # 24 px padding off the top + sides; panels keep their rendered
        # size, x slides across the video width via the expressions.
        pad = 24
        left_x_expr = _x_expr_left(left_img.width, 0, pad)
        right_x_expr = _x_expr_right(right_img.width, 0, pad)
        filter_complex = (
            f"[0:v][1:v]overlay=x='{left_x_expr}':y={pad}:eval=frame[v1];"
            f"[v1][2:v]overlay=x='{right_x_expr}':y={pad}:eval=frame"
        )
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(input_video),
            "-i", str(left_png),
            "-i", str(right_png),
            "-filter_complex", filter_complex,
            "-c:v", "libx264", "-preset", "veryfast",
            "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_video),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=600)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            log.warning("intro_overlay: ffmpeg failed: %s", exc)
            output_video.unlink(missing_ok=True)
            return False
    if not output_video.exists() or output_video.stat().st_size == 0:
        output_video.unlink(missing_ok=True)
        return False
    log.info(
        "intro_overlay: rendered %s → %s (player=%s hole=%s par=%s yds=%s)",
        input_video.name, output_video.name, player_name, hole_number, par, yardage,
    )
    return True


def apply_intro_overlay_inplace(
    video_path: Path,
    player_name: str | None,
    course_name: str,
    hole_number: int | None,
    par: int | None,
    yardage: int | None,
) -> bool:
    """Same as apply_intro_overlay but writes to a temp file and
    atomically renames over `video_path`. Best-effort: returns False
    on any failure and leaves video_path untouched."""
    tmp = video_path.with_suffix(video_path.suffix + ".intro.tmp.mp4")
    ok = apply_intro_overlay(
        video_path, tmp,
        player_name, course_name, hole_number, par, yardage,
    )
    if not ok:
        tmp.unlink(missing_ok=True)
        return False
    try:
        tmp.replace(video_path)
    except Exception as exc:  # pragma: no cover
        log.warning("intro_overlay: rename failed: %s", exc)
        tmp.unlink(missing_ok=True)
        return False
    return True
