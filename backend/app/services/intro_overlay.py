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
import random
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


# ── placeholder player names ─────────────────────────────────────────
#
# WHY THERE IS A NAME AT ALL when nobody has been identified. The intro
# panel is a layout, and a blank where the name goes does not read as
# "unknown player", it reads as broken software. So a clip that matched
# no registered participant gets a stand-in.
#
# THE STAND-IN USED TO BE THE OWNER'S OWN NAME, hard-coded in five
# places, which meant every demo clip of every golfer on the range went
# out under one person's name. Plainly wrong on a shared clip and
# confusing on a review screen.
#
# Ordinary invented names, deliberately not the names of real touring
# professionals: a placeholder that reads as a real golfer somebody
# could look up is a worse placeholder, not a better one.
_PLACEHOLDER_FIRST = (
    "Alex", "Jordan", "Casey", "Morgan", "Riley", "Taylor", "Drew",
    "Quinn", "Avery", "Reese", "Blake", "Cameron", "Hayden", "Emerson",
    "Rowan", "Sawyer", "Parker", "Finley", "Marlow", "Ellis",
)
_PLACEHOLDER_LAST = (
    "Hollis", "Vance", "Ashby", "Merritt", "Calloway", "Renfro",
    "Thorne", "Bexley", "Waverly", "Lockhart", "Danforth", "Ainsley",
    "Croft", "Pemberton", "Halloway", "Whitfield", "Stanhope",
    "Marchetti", "Okonkwo", "Lindqvist",
)


def placeholder_player_name(seed=None) -> str:
    """A stand-in name for a clip that matched no registered participant.

    STABLE FOR A GIVEN CLIP, random across clips. `seed` should be
    something that identifies the clip -- its id, its filename -- and
    the same seed always returns the same name. That matters because
    re-rendering happens: an overlay is reapplied when a clip is
    re-produced or its graphics are edited, and a name that changed
    every time would make the same swing look like a different golfer
    on every pass, with no way to tell a re-render from a real change.

    Called with no seed it is genuinely random, which is the right
    behaviour only where nothing identifies the clip.
    """
    rnd = random.Random(seed) if seed is not None else random.Random()
    return f"{rnd.choice(_PLACEHOLDER_FIRST)} {rnd.choice(_PLACEHOLDER_LAST)}"


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

    # No seed here: by the time a name reaches the renderer the
    # caller has had every chance to supply one, and this floor
    # exists so the panel is never blank.
    player_text = (player_name or placeholder_player_name()).upper()
    course_text = course_name or ""
    # WHAT IS NOT KNOWN IS NOT PRINTED. These used to fall back to
    # "PAR 3" and "101 YDS", and 101 is the number that shipped to
    # players on every hole whose yardage nobody had entered -- it
    # looks like a measurement and cannot be told from one. A hole with
    # no yardage on the course record now simply says HOLE 4 · PAR 3,
    # and the log says which hole to go and fill in.
    info_parts = []
    if hole_number is not None:
        info_parts.append(f"HOLE {int(hole_number)}")
    if par is not None:
        info_parts.append(f"PAR {int(par)}")
    if yardage is not None:
        info_parts.append(f"{int(yardage)} YDS")
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
    """The brand logo, from the one lookup both callers share.

    This used to keep its own candidate list, and that list checked
    `frontend/public/` and a `backend/app/assets/` directory that does
    not exist. Vite copies `public/` into `dist/` at build time and the
    Dockerfile ships only `dist`, so in a deployed container neither path
    was ever there -- every produced clip fell back to the drawn green
    badge while `notifications`, which checked `dist`, put the real
    wordmark on every email out of the same build.
    """
    from .branding import logo_path
    return logo_path()


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


def render_target_sign(yardage: int | None, pole_height: int = 130) -> Image.Image | None:
    """Stake-style 'TO HOLE / N YDS' marker drawn into a transparent
    PNG. The visible sign sits at the top of the image and a thin
    pole extends down to the bottom — when composited the bottom of
    the image is anchored at the target pixel coordinate, so the pole
    appears to plant into the green at the flag.
    """
    if not HAS_PIL:
        return None

    # No yardage, no stake: a sign reading "0 YDS" is worse than no
    # sign, and the caller already knows to skip it.
    if yardage is None:
        return None
    yards = int(yardage)

    sign_w = 168
    header_h = 32
    body_h = 38
    sign_h = header_h + body_h
    pole_h = max(20, int(pole_height))
    height = sign_h + pole_h

    bold_header = _find_font(FONT_CANDIDATES_BOLD, 16)
    bold_body = _find_font(FONT_CANDIDATES_BOLD, 24)

    img = Image.new("RGBA", (sign_w, height), (0, 0, 0, 0))

    # Sign body: navy header above white body, both clipped to a
    # rounded outer rectangle for the same visual language as the
    # other intro panels.
    sections = Image.new("RGBA", (sign_w, sign_h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sections)
    sd.rectangle((0, 0, sign_w, header_h), fill=PANEL_BOTTOM_BLUE)
    sd.rectangle((0, header_h, sign_w, sign_h), fill=(255, 255, 255, 245))

    mask = Image.new("L", (sign_w, sign_h), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle((0, 0, sign_w - 1, sign_h - 1), radius=6, fill=255)
    img.paste(sections, (0, 0), mask)

    draw = ImageDraw.Draw(img)
    header_text = "TO HOLE"
    w_h = _measure(header_text, bold_header)
    header_font_size = getattr(bold_header, "size", 16)
    draw.text(
        ((sign_w - w_h) // 2,
         (header_h - header_font_size) // 2),
        header_text, fill=TEXT_PRIMARY, font=bold_header,
    )

    yds_text = f"{yards} YDS"
    w_y = _measure(yds_text, bold_body)
    body_font_size = getattr(bold_body, "size", 24)
    draw.text(
        ((sign_w - w_y) // 2,
         header_h + (body_h - body_font_size) // 2),
        yds_text, fill=PANEL_BOTTOM_BLUE, font=bold_body,
    )

    # Thin white pole with a dark outline so it reads against grass
    # and sky alike. Plants into the target pixel when the image is
    # anchored bottom-aligned at target.y.
    pole_x = sign_w // 2
    pole_w = 3
    draw.rectangle(
        (pole_x - pole_w - 1, sign_h, pole_x + pole_w + 1, height),
        fill=(0, 0, 0, 200),
    )
    draw.rectangle(
        (pole_x - pole_w, sign_h, pole_x + pole_w, height),
        fill=(255, 255, 255, 235),
    )
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
    """ffmpeg overlay x= expression for the right (brand logo) panel.

    Slides in from x=W → x=W-w-pad and then STAYS for the whole clip —
    the GolfReelz logo is persistent branding, start to finish, on both
    the landscape and vertical outputs. (The name plate and target sign
    keep their slide-out.)
    """
    t_in = SLIDE_IN_SEC
    anchor = f"(W-w-{anchor_right_pad})"
    return f"if(lt(t,{t_in}), W - (W-{anchor})*t/{t_in}, {anchor})"


def _y_expr_target(anchor_y: int) -> str:
    """ffmpeg overlay y= expression for the target sign.

    Drops down from y=-h (entirely above frame) to y=anchor_y during
    slide-in, holds, then retracts back above the frame on slide-out.
    Matches the timing of the left/right panel slides so all three
    intro graphics appear and disappear together.
    """
    t_in = SLIDE_IN_SEC
    t_hold_end = SLIDE_IN_SEC + HOLD_SEC
    t_out = t_hold_end + SLIDE_OUT_SEC
    return (
        f"if(lt(t,{t_in}), -h+(h+{anchor_y})*t/{t_in}, "
        f"if(lt(t,{t_hold_end}), {anchor_y}, "
        f"if(lt(t,{t_out}), {anchor_y} - (h+{anchor_y})*(t-{t_hold_end})/{SLIDE_OUT_SEC}, "
        f"-h)))"
    )


def apply_intro_overlay(
    input_video: Path,
    output_video: Path,
    player_name: str | None,
    course_name: str,
    hole_number: int | None,
    par: int | None,
    yardage: int | None,
    target_xy: tuple[int, int] | None = None,
) -> bool:
    """Composite the slide-in/out intro panels onto input_video and
    write the result to output_video. Returns True on success, False
    on any failure (PIL missing, ffmpeg missing, panel render failure,
    ffmpeg failure). Leaves input_video untouched on failure.

    When `target_xy` is supplied (native pixel coords on the source
    video) an additional 'TO HOLE / N YDS' marker is rendered, with
    its pole planting at that pixel. It slides in / holds / slides
    out in sync with the left and right panels.
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

    # Optional: target-pin sign. The sign body sits just above the
    # target pixel with a short pole that plants into the green where
    # the operator dropped the flag. When the target sits near the top
    # of the frame the image is clamped to y=0 so the sign body never
    # gets cut off.
    target_sign = None
    target_anchor_x: int | None = None
    target_anchor_y: int | None = None
    if target_xy is not None:
        tx, ty = int(target_xy[0]), int(target_xy[1])
        short_pole = 40
        sign_img = render_target_sign(yardage, pole_height=short_pole)
        if sign_img is not None:
            target_sign = sign_img
            # Anchor: image bottom (= pole tip) at ty, image center at tx.
            target_anchor_x = tx - sign_img.width // 2
            target_anchor_y = ty - sign_img.height
            if target_anchor_y < 0:
                target_anchor_y = 0

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        left_png = td_path / "intro_left.png"
        right_png = td_path / "intro_right.png"
        left_img.save(left_png)
        right_img.save(right_png)
        target_png = None
        if target_sign is not None:
            target_png = td_path / "intro_target.png"
            target_sign.save(target_png)

        # 24 px padding off the top + sides; panels keep their rendered
        # size, x slides across the video width via the expressions.
        pad = 24
        left_x_expr = _x_expr_left(left_img.width, 0, pad)
        right_x_expr = _x_expr_right(right_img.width, 0, pad)

        inputs = [
            "-i", str(input_video),
            "-i", str(left_png),
            "-i", str(right_png),
        ]
        filter_complex = (
            f"[0:v][1:v]overlay=x='{left_x_expr}':y={pad}:eval=frame[v1];"
            f"[v1][2:v]overlay=x='{right_x_expr}':y={pad}:eval=frame"
        )
        if target_png is not None and target_anchor_y is not None:
            inputs += ["-i", str(target_png)]
            target_y_expr = _y_expr_target(target_anchor_y)
            filter_complex += (
                f"[v2];[v2][3:v]overlay="
                f"x={target_anchor_x}:y='{target_y_expr}':eval=frame"
            )
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            *inputs,
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
        "intro_overlay: rendered %s → %s (player=%s hole=%s par=%s yds=%s target=%s)",
        input_video.name, output_video.name, player_name, hole_number, par, yardage,
        target_xy,
    )
    return True


def apply_intro_overlay_inplace(
    video_path: Path,
    player_name: str | None,
    course_name: str,
    hole_number: int | None,
    par: int | None,
    yardage: int | None,
    target_xy: tuple[int, int] | None = None,
) -> bool:
    """Same as apply_intro_overlay but writes to a temp file and
    atomically renames over `video_path`. Best-effort: returns False
    on any failure and leaves video_path untouched."""
    tmp = video_path.with_suffix(video_path.suffix + ".intro.tmp.mp4")
    ok = apply_intro_overlay(
        video_path, tmp,
        player_name, course_name, hole_number, par, yardage,
        target_xy=target_xy,
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


# ── closing plate: distance to pin ─────────────────────────────────────
# How long the distance plate stays up at the end, and how long it takes
# to drop in. It holds to the last frame rather than sliding away: it is
# the answer the clip was building to, and a viewer who scrubs back to
# the end should find it there rather than watch it leave.
DIST_DROP_SEC = 0.45
DIST_LEAD_SEC = 2.6          # how far before the end it starts dropping


def render_distance_panel(
    distance_display: str,
    label: str = "DISTANCE TO PIN",
) -> "Image.Image | None":
    """The closing plate: 'DISTANCE TO PIN — 41 FEET'.

    Takes a pre-formatted display string rather than a number, because
    the rounding is a measurement decision, not a drawing one --
    green_calibration already refuses to print a precision the
    homography does not have, and this must not quietly re-round it into
    something more confident.
    """
    if not HAS_PIL:
        return None
    text = (distance_display or "").strip()
    if not text:
        return None

    bold_big = _find_font(FONT_CANDIDATES_BOLD, 34)
    bold_lbl = _find_font(FONT_CANDIDATES_BOLD, 16)

    label_text = label.upper()
    value_text = text.upper()

    pad_x, pad_y = 26, 14
    gap = 10
    w_lbl = _measure(label_text, bold_lbl)
    w_val = _measure(value_text, bold_big)
    content_w = max(w_lbl, w_val)
    width = content_w + 2 * pad_x
    lbl_h = (bold_lbl.size if hasattr(bold_lbl, "size") else 16)
    val_h = (bold_big.size if hasattr(bold_big, "size") else 34)
    height = pad_y * 2 + lbl_h + gap + val_h

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, width, height), fill=PANEL_BOTTOM_BLUE)
    # A thin brand rule under the value, the same green the mark uses.
    d.rectangle((0, height - 4, width, height), fill=BRAND_GREEN)

    d.text(((width - w_lbl) // 2, pad_y), label_text,
           fill=TEXT_MUTED, font=bold_lbl)
    d.text(((width - w_val) // 2, pad_y + lbl_h + gap), value_text,
           fill=TEXT_PRIMARY, font=bold_big)
    return img


def _y_expr_distance(anchor_y: int, start_t: float) -> str:
    """Drop the plate in from above at `start_t` and hold to the end."""
    return (
        f"if(lt(t,{start_t}), -h, "
        f"if(lt(t,{start_t + DIST_DROP_SEC}), "
        f"-h+(h+{anchor_y})*(t-{start_t})/{DIST_DROP_SEC}, {anchor_y}))"
    )


def apply_distance_plate(
    input_video: Path,
    output_video: Path,
    distance_display: str,
    label: str = "DISTANCE TO PIN",
    rest_at_sec: float | None = None,
) -> bool:
    """Drop the distance plate into the top of the frame at the end.

    Separate from the intro overlay on purpose: the intro is about the
    hole and can be drawn before anything is known, while this is the
    RESULT and only exists once the ball has come to rest and a
    calibrated camera has turned that into feet. Bolting it onto the
    intro would mean an unmeasurable shot could not get its intro either.

    Returns False and leaves the input alone on any failure -- including
    an empty distance, which is the normal case for an uncalibrated
    camera and must never become a blank or guessed plate.
    """
    if not (distance_display or "").strip():
        return False
    if not HAS_PIL or not _have_ffmpeg() or not input_video.exists():
        return False

    plate = render_distance_panel(distance_display, label=label)
    if plate is None:
        return False

    dur = _probe_seconds(input_video)
    if not dur or dur <= 0:
        return False
    # WHEN THE BALL STOPPED, not a fixed lead off the end. Timing this
    # from the tail alone announced the distance while the ball was
    # still in the air on a clip whose green angle ran long -- the
    # answer arriving before the thing it answers, which reads as a
    # spoiler and invites the obvious "how does it know yet?".
    if rest_at_sec is not None and 0 <= rest_at_sec < dur:
        start_t = min(rest_at_sec + 0.25, max(0.0, dur - DIST_DROP_SEC))
    else:
        start_t = max(0.0, dur - DIST_LEAD_SEC)

    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "distance_plate.png"
        plate.save(png)
        # Top-centre, a little below the top edge -- clear of the
        # GolfReelz mark in the corner.
        y_expr = _y_expr_distance(28, start_t)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(input_video), "-i", str(png),
            "-filter_complex",
            f"[0:v][1:v]overlay=x=(W-w)/2:y='{y_expr}':eval=frame",
            "-c:v", "libx264", "-preset", "veryfast",
            "-c:a", "copy", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_video),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=600)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            log.warning("distance plate: ffmpeg failed: %s", exc)
            output_video.unlink(missing_ok=True)
            return False

    if not output_video.exists() or output_video.stat().st_size == 0:
        output_video.unlink(missing_ok=True)
        return False
    log.info("distance plate: %s -> %s (%s)",
             input_video.name, output_video.name, distance_display)
    return True


def _probe_seconds(path: Path) -> float | None:
    """Clip duration in seconds, or None.

    ffprobe first, cv2 second. They ship together often enough to assume
    and not always enough to rely on -- a box with ffmpeg but no ffprobe
    would otherwise lose the plate silently, which looks exactly like
    "we had no distance" and is a much harder thing to notice.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        d = float((out.stdout or "").strip())
        if d > 0:
            return d
    except Exception:  # noqa: BLE001
        pass
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if fps > 0 and n > 0:
            return float(n) / float(fps)
    except Exception as exc:  # noqa: BLE001
        log.warning("distance plate: could not probe duration: %s", exc)
    return None


def apply_distance_plate_inplace(
    video_path: Path, distance_display: str,
    label: str = "DISTANCE TO PIN",
    rest_at_sec: float | None = None,
) -> bool:
    """As above, writing to a temp file and renaming over the original."""
    tmp = video_path.with_suffix(video_path.suffix + ".dist.tmp.mp4")
    ok = apply_distance_plate(video_path, tmp, distance_display, label=label,
                              rest_at_sec=rest_at_sec)
    if not ok:
        tmp.unlink(missing_ok=True)
        return False
    try:
        tmp.replace(video_path)
        return True
    except OSError as exc:
        log.warning("distance plate: rename failed: %s", exc)
        tmp.unlink(missing_ok=True)
        return False
