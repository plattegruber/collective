"""Mapping between detector class indices and artwork ids."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def is_image_file(path: Path) -> bool:
    """True for files the synthesizer can read."""
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def build_label_map(artwork_ids: Iterable[str]) -> dict[str, str]:
    """
    Assign class indices 1..N to artwork ids in sorted order; 0 is reserved for background.

    The result is the exact object the web app expects in `labels.json`.
    """
    unique = sorted(set(artwork_ids))
    return {str(index): artwork_id for index, artwork_id in enumerate(unique, start=1)}


def class_index_of(label_map: dict[str, str]) -> dict[str, int]:
    """Invert a label map: artwork id → class index."""
    return {artwork_id: int(index) for index, artwork_id in label_map.items()}


def scan_raw_dir(raw_dir: Path) -> dict[str, list[Path]]:
    """
    Discover `<raw_dir>/<artworkId>/*.jpg` folders. Folders without images are ignored.
    """
    pieces: dict[str, list[Path]] = {}
    if not raw_dir.is_dir():
        return pieces
    for folder in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        images = sorted(p for p in folder.iterdir() if is_image_file(p))
        if images:
            pieces[folder.name] = images
    return pieces


def save_label_map(label_map: dict[str, str], path: Path) -> None:
    """Write `labels.json`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(label_map, indent=2, sort_keys=True))


def load_label_map(path: Path) -> dict[str, str]:
    """Read `labels.json`, validating its shape."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and k.isdigit() and isinstance(v, str) for k, v in data.items()
    ):
        raise ValueError(f"{path} must map numeric class indices to artwork ids.")
    return {str(k): str(v) for k, v in data.items()}
