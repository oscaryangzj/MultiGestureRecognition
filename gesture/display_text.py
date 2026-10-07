from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=64)
def _font(path, size):
    return ImageFont.truetype(path, size)


def draw_text(frame, text, position, font_size, color, font_path, max_width=None):
    """Draw one Unicode line in place on a BGR frame; position is its top-left corner."""
    if not text:
        return frame
    size = max(1, round(font_size))
    font = _font(str(font_path), size)
    left, top, right, bottom = font.getbbox(text)
    if max_width is not None and right - left > max_width:
        size = max(1, int(size * max_width / (right - left)))
        font = _font(str(font_path), size)
        left, top, right, bottom = font.getbbox(text)
        while right - left > max_width and size > 1:
            size -= 1
            font = _font(str(font_path), size)
            left, top, right, bottom = font.getbbox(text)
    width, height = right - left, bottom - top
    if not width or not height:
        return frame
    mask = Image.new("L", (width, height))
    ImageDraw.Draw(mask).text((-left, -top), text, font=font, fill=255)
    x, y = map(int, position)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.shape[1], x + width), min(frame.shape[0], y + height)
    if x0 < x1 and y0 < y1:
        alpha = np.asarray(mask, np.float32)[y0 - y:y1 - y, x0 - x:x1 - x, None] / 255
        patch = frame[y0:y1, x0:x1]
        patch[:] = (patch * (1 - alpha) + np.asarray(color, np.float32) * alpha).astype(np.uint8)
    return frame
