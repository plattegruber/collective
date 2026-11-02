"""End-to-end orchestration used by the CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from collective_ml.api import CollectiveClient, pull_collection
from collective_ml.composites import ARTWORK_CATEGORY, CompositeOptions, compose
from collective_ml.config import ApiSettings, Workspace
from collective_ml.files import list_images

log = logging.getLogger(__name__)


def pull(settings: ApiSettings, workspace: Workspace) -> int:
    """Download a collection's photos into `raw/<artworkId>/` for evaluation."""
    workspace.ensure()
    with CollectiveClient(settings) as client:
        collection, downloaded, skipped = pull_collection(
            client, workspace.raw_dir, workspace.collection_json
        )
    usable = len(collection.with_images())
    log.info(
        "Pulled %d new photos (%d already present); %d artworks have photos.",
        downloaded,
        skipped,
        usable,
    )
    return usable


def fetch_data(workspace: Workspace, foregrounds: int, rooms: int) -> None:
    """Open images that teach the detector: paintings as foregrounds, rooms as backgrounds."""
    from collective_ml.datasets import fetch_foregrounds, fetch_rooms

    workspace.ensure()
    fetch_foregrounds(workspace.foregrounds_dir, foregrounds)
    fetch_rooms(workspace.rooms_dir, rooms)


def compose_detector_dataset(workspace: Workspace, options: CompositeOptions) -> int:
    """Every painting (plus any pulled photos) on every kind of wall, one class: artwork."""
    foregrounds = list_images(workspace.foregrounds_dir) + list_images(workspace.raw_dir)
    if not foregrounds:
        raise SystemExit(
            f"No foregrounds under {workspace.foregrounds_dir}. Run `collective-ml fetch-data`."
        )
    rooms = workspace.rooms_dir if list_images(workspace.rooms_dir) else None
    dataset = compose(
        dict.fromkeys(foregrounds, ARTWORK_CATEGORY), workspace.detector_dataset_dir, options, rooms
    )
    return len(dataset.images)


def train_detector(workspace: Workspace, **train_kwargs: Any) -> dict[str, Any]:
    """Trains, exports and verifies the detector; returns the metrics written next to it."""
    from collective_ml.export import export_onnx, verify_onnx
    from collective_ml.train import TrainOptions, train

    options = TrainOptions(**train_kwargs)
    result = train(workspace.detector_dataset_dir, workspace.detector_dir, options)
    export_onnx(result.checkpoint, workspace.detector_onnx)
    verify_onnx(workspace.detector_onnx, options.img_size)
    log.info(
        "Detector: best epoch %d, val loss %.3f, mAP@50 %.3f → %s",
        result.best_epoch,
        result.best_val_loss,
        result.map50,
        workspace.detector_onnx,
    )
    return load_json(workspace.detector_metrics_json) or {}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())
