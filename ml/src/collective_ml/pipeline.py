"""End-to-end orchestration used by the CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from collective_ml.api import CollectiveClient, PublishedModel, pull_collection
from collective_ml.config import ApiSettings, Workspace
from collective_ml.labels import load_label_map

log = logging.getLogger(__name__)


def pull(settings: ApiSettings, workspace_root: Path) -> Workspace:
    """Download the collection into `<workspace_root>/<userId>/`."""
    with CollectiveClient(settings) as client:
        collection = client.fetch_collection()
        workspace = Workspace(workspace_root / collection.user_id).ensure()
        _, downloaded, skipped = pull_collection(
            client, workspace.raw_dir, workspace.collection_json
        )
    usable = len(collection.with_images())
    log.info(
        "Pulled %d new photos (%d already present); %d artworks have photos.",
        downloaded,
        skipped,
        usable,
    )
    if usable == 0:
        raise SystemExit("No artwork has a photo yet. Add photos in the app, then pull again.")
    return workspace


def resolve_workspace(workspace_root: Path, user_id: str | None) -> Workspace:
    """Pick the per-user workspace; infer the user when only one exists."""
    if user_id:
        return Workspace(workspace_root / user_id)
    candidates = (
        sorted(p for p in workspace_root.iterdir() if p.is_dir()) if workspace_root.is_dir() else []
    )
    if len(candidates) == 1:
        return Workspace(candidates[0])
    if not candidates:
        raise SystemExit(f"No workspace under {workspace_root}. Run `collective-ml pull` first.")
    names = ", ".join(p.name for p in candidates)
    raise SystemExit(f"Several workspaces found ({names}); pass --user to choose one.")


def load_metrics(workspace: Workspace) -> dict[str, Any] | None:
    """Read `metrics.json` if training wrote one."""
    if not workspace.metrics_json.exists():
        return None
    data = json.loads(workspace.metrics_json.read_text())
    # Keep the payload small: history is useful locally, not in the app.
    data.pop("history", None)
    return data


def publish(settings: ApiSettings, workspace: Workspace) -> PublishedModel:
    """Upload `model.onnx` + `labels.json` (+ metrics) and activate it."""
    if not workspace.onnx_path.exists():
        raise SystemExit(f"{workspace.onnx_path} not found. Run `collective-ml export` first.")
    labels = load_label_map(workspace.labels_json)
    metrics = load_metrics(workspace)
    input_size = int(metrics["imgSize"]) if metrics and "imgSize" in metrics else 320
    with CollectiveClient(settings) as client:
        published = client.publish_model(workspace.onnx_path, labels, input_size, metrics)
    log.info(
        "Published model %s (%d classes) → %s", published.id, len(published.labels), published.url
    )
    return published
