"""TTF text for the overlay, with bounded glyph and string caches.

Fonts load once per size/weight. Changing telemetry reuses rasterized
glyphs; drawing blends only the text's small rectangle into the frame.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=24)
def _font(size: int, bold: bool):
    win = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    candidates = [
        win / ("segoeuib.ttf" if bold else "segoeui.ttf"),
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


@lru_cache(maxsize=2048)
def _glyph(char: str, size: int, bold: bool):
    font = _font(size, bold)
    advance = float(font.getlength(char))
    ascent, descent = font.getmetrics()
    tile = Image.new("L", (math.ceil(advance) + 4, ascent + descent))
    ImageDraw.Draw(tile).text((2, ascent), char, font=font, fill=255, anchor="ls")
    return np.asarray(tile), advance


@lru_cache(maxsize=256)
def _mask(text: str, size: int, bold: bool):
    glyphs = [_glyph(char, size, bold) for char in text]
    height = _font(size, bold).getmetrics()
    width = math.ceil(sum(advance for _, advance in glyphs))
    mask = np.zeros((sum(height), width + 4), np.uint16)
    cursor = 0.0
    for tile, advance in glyphs:
        x = round(cursor)
        part = mask[:, x : x + tile.shape[1]]
        np.maximum(part, tile[:, : part.shape[1]], out=part)
        cursor += advance
    return mask[:, :, None], width


@lru_cache(maxsize=1024)
def text_width(text: str, size: int = 14, bold: bool = False) -> int:
    # Layout (wrapping/truncation) must not rasterize every candidate string.
    return math.ceil(sum(_glyph(char, size, bold)[1] for char in text))


def fit_text(text: str, width: int, size: int = 14, bold: bool = False) -> str:
    if text_width(text, size, bold) <= width:
        return text
    while text and text_width(text + "…", size, bold) > width:
        text = text[:-1]
    return text + "…" if text else ""


def draw_text(
    img,
    text: str,
    x: int,
    y: int,
    color=(240, 240, 240),
    size: int = 14,
    bold: bool = False,
    max_width: int | None = None,
):
    """Top-aligned BGR text; clips safely even on tiny replay frames."""
    if max_width is not None:
        text = fit_text(text, max_width, size, bold)
    if not text:
        return
    alpha, _ = _mask(text, size, bold)
    h, w = alpha.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    a = alpha[y0 - y : y1 - y, x0 - x : x1 - x]
    roi = img[y0:y1, x0:x1]
    roi[:] = (
        (roi.astype(np.uint16) * (255 - a) + np.asarray(color, np.uint16) * a + 127) // 255
    ).astype(np.uint8)
