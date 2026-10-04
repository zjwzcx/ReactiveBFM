"""Video frame header overlay helpers for MuJoCo rollout recording.

Implementation taken from the ReactiveBFM ``play_rbfm_npz_mujoco`` playback
tool so that ScaleBridge video annotation matches the original tooling and no
longer depends on the planner data package at import time.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    # macOS fallbacks (development machines)
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return draw.textsize(text, font=font)


def load_font(font_size: int) -> ImageFont.ImageFont:
    """Load the first available TrueType font, falling back to PIL's default."""
    for font_path in FONT_CANDIDATES:
        if Path(font_path).is_file():
            try:
                return ImageFont.truetype(font_path, max(10, font_size))
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap_text_by_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = text.strip().split()
    if not words:
        return [""]
    lines: list[str] = []
    cur = words[0]
    for word in words[1:]:
        test = f"{cur} {word}"
        w, _ = _text_size(draw, test, font)
        if w <= max_width:
            cur = test
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def overlay_header(
    rgb: np.ndarray,
    *,
    frame_idx: int,
    total_frames: int,
    fps: float,
    text_label: str,
    font: ImageFont.ImageFont,
) -> np.ndarray:
    """Draw the header bar (frame counter + wrapped prompt text) on a frame."""
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image, "RGBA")

    line1 = f"Frame: {frame_idx + 1}/{total_frames}    FPS: {fps:.2f}"
    wrapped = _wrap_text_by_width(draw, text_label, font, max_width=max(120, image.width - 24))
    lines = [line1]
    if wrapped:
        lines.append(f"Text: {wrapped[0]}")
        for extra in wrapped[1:]:
            lines.append(f"      {extra}")

    padding_x = 12
    padding_y = 8
    line_gap = 4

    line_sizes = [_text_size(draw, line, font) for line in lines]
    text_h = sum(h for _, h in line_sizes) + line_gap * (len(lines) - 1)
    box_h = text_h + 2 * padding_y

    draw.rectangle([(0, 0), (image.width, box_h)], fill=(0, 0, 0, 170))

    y = padding_y
    for line, (w, h) in zip(lines, line_sizes):
        x = padding_x
        if w > image.width - 2 * padding_x:
            trunc = textwrap.shorten(line, width=max(12, image.width // 10), placeholder="...")
            line = trunc
        draw.text(
            (x, y),
            line,
            fill=(255, 255, 255, 255),
            font=font,
            stroke_width=1,
            stroke_fill=(0, 0, 0, 255),
        )
        y += h + line_gap

    return np.asarray(image)
