"""Procedural wall-like backgrounds so no background photo set is required."""

from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from collective_ml.files import is_image_file


def _wall_colour(rng: np.random.Generator) -> tuple[int, int, int]:
    hue = float(rng.random())
    saturation = float(rng.uniform(0.02, 0.2))
    value = float(rng.uniform(0.55, 0.98))
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(r * 255), int(g * 255), int(b * 255)


def procedural_wall(size: int, rng: np.random.Generator) -> Image.Image:
    """
    A square RGB canvas resembling an interior wall: muted colour, soft directional
    lighting, optional vignette and panel lines, and faint sensor noise.
    """
    base = np.empty((size, size, 3), dtype=np.float32)
    base[:] = _wall_colour(rng)

    gy, gx = np.mgrid[0:size, 0:size].astype(np.float32) / max(1, size - 1)
    angle = float(rng.uniform(0, 2 * np.pi))
    direction = np.cos(angle) * gx + np.sin(angle) * gy
    lighting = 1.0 + float(rng.uniform(-0.12, 0.12)) * (direction - 0.5) * 2
    base *= lighting[:, :, None]

    if rng.random() < 0.6:
        dist = np.sqrt((gx - 0.5) ** 2 + (gy - 0.5) ** 2)
        vignette = 1.0 - np.clip(dist * float(rng.uniform(0.3, 0.9)), 0.0, 1.0) * 0.25
        base *= vignette[:, :, None]

    if rng.random() < 0.25:
        # Faint vertical panelling or wainscot lines.
        spacing = int(rng.integers(size // 6, size // 2))
        offset = int(rng.integers(0, spacing))
        for x in range(offset, size, spacing):
            base[:, x : x + 2] *= float(rng.uniform(0.85, 0.95))

    noise = rng.normal(0.0, float(rng.uniform(0.5, 2.5)), (size, size, 1)).astype(np.float32)
    base = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(base, mode="RGB")


class BackgroundSource:
    """Serves backgrounds from a folder when available, else procedurally."""

    def __init__(self, size: int, rng: np.random.Generator, directory: Path | None = None) -> None:
        self._size = size
        self._rng = rng
        self._files: list[Path] = []
        if directory and directory.is_dir():
            self._files = sorted(p for p in directory.rglob("*") if is_image_file(p))

    def next(self) -> Image.Image:
        """Return a square RGB background."""
        if self._files and self._rng.random() < 0.8:
            path = self._files[int(self._rng.integers(0, len(self._files)))]
            image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
            return image.resize((self._size, self._size), Image.Resampling.LANCZOS)
        return procedural_wall(self._size, self._rng)
