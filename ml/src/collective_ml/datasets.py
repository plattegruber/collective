"""Open image sets used to teach the detector what "a picture on a wall" looks like."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from collective_ml.files import list_images

log = logging.getLogger(__name__)

FOREGROUNDS_DATASET = "huggan/wikiart"
# Parquet-backed (no dataset scripts). The first is small; LSUN bedrooms tops it up.
ROOM_SOURCES: tuple[tuple[str, str], ...] = (
    ("Voxel51/IndoorSceneRecognition", "train"),
    ("Voxel51/IndoorSceneRecognition", "test"),
    ("pcuenq/lsun-bedrooms", "train"),
)


def _load_streaming(name: str, split: str = "train", **kwargs: object) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise SystemExit("Install the training extras: pip install -e './ml[train]'") from error
    return load_dataset(name, split=split, streaming=True, **kwargs)


def fetch_foregrounds(out_dir: Path, count: int = 400, max_edge: int = 640) -> int:
    """Downloads `count` paintings from WikiArt as detector foregrounds. Skips when present."""
    existing = list_images(out_dir)
    if len(existing) >= count:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = len(existing)
    for row in _load_streaming(FOREGROUNDS_DATASET):
        if saved >= count:
            break
        image = row["image"].convert("RGB")
        image.thumbnail((max_edge, max_edge))
        image.save(out_dir / f"wikiart-{saved:05d}.jpg", quality=88)
        saved += 1
    log.info("Foregrounds: %d images in %s", saved, out_dir)
    return saved - len(existing)


def fetch_rooms(out_dir: Path, count: int = 300, max_edge: int = 800) -> int:
    """Downloads `count` indoor scenes as real backgrounds. Skips when present."""
    existing = list_images(out_dir)
    if len(existing) >= count:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = len(existing)
    for name, split in ROOM_SOURCES:
        if saved >= count:
            break
        for row in _load_streaming(name, split):
            if saved >= count:
                break
            image = row["image"].convert("RGB")
            image.thumbnail((max_edge, max_edge))
            image.save(out_dir / f"room-{saved:04d}.jpg", quality=88)
            saved += 1
    log.info("Rooms: %d images in %s", saved, out_dir)
    return saved - len(existing)
