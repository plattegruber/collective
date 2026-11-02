"""Small filesystem helpers."""

from __future__ import annotations

from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def is_image_file(path: Path) -> bool:
    """True for files the compositor can read."""
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def list_images(directory: Path) -> list[Path]:
    """Image files directly inside or below `directory`, sorted for reproducibility."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.rglob("*") if is_image_file(p))


def scan_raw_dir(raw_dir: Path) -> dict[str, list[Path]]:
    """Discover `<raw_dir>/<artworkId>/*.jpg` folders. Folders without images are ignored."""
    pieces: dict[str, list[Path]] = {}
    if not raw_dir.is_dir():
        return pieces
    for folder in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        images = sorted(p for p in folder.iterdir() if is_image_file(p))
        if images:
            pieces[folder.name] = images
    return pieces
