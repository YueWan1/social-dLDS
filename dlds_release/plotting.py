"""Small plotting helpers shared by figure scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from PIL import ImageFont


def load_font(
    size: int,
    *,
    preferred: Iterable[str | Path] = (),
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a portable sans-serif font, with an explicit environment override.

    ``SOCIAL_DLDS_FONT`` is useful when exact typography matters.  Otherwise
    Pillow's bundled DejaVu Sans is preferred before common Linux font paths.
    """
    candidates: list[str | Path] = []
    if configured := os.environ.get("SOCIAL_DLDS_FONT"):
        candidates.append(configured)
    candidates.extend(preferred)
    candidates.extend(
        [
            "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf",
        ]
    )

    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            continue

    # A bitmap fallback keeps the script runnable on minimal systems. Exact
    # publication typography should instead set SOCIAL_DLDS_FONT.
    return ImageFont.load_default()
