"""Runtime settings and workspace layout."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_URL = "COLLECTIVE_URL"
ENV_TOKEN = "COLLECTIVE_TOKEN"
ENV_RELEASE_TOKEN = "COLLECTIVE_RELEASE_TOKEN"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


def _origin(url: str | None) -> str:
    resolved = (url or os.environ.get(ENV_URL, "")).strip().rstrip("/")
    if not resolved:
        raise ConfigError(f"Set {ENV_URL} (e.g. https://collective.example.com) or pass --url.")
    return resolved


@dataclass(frozen=True)
class ApiSettings:
    """A collector's API token: reads their collection and photos."""

    base_url: str
    token: str

    @classmethod
    def from_env(cls, url: str | None = None, token: str | None = None) -> ApiSettings:
        resolved_token = (token or os.environ.get(ENV_TOKEN, "")).strip()
        if not resolved_token:
            raise ConfigError(
                f"Set {ENV_TOKEN} to an API token from Settings → API tokens, or pass --token."
            )
        return cls(base_url=_origin(url), token=resolved_token)


@dataclass(frozen=True)
class ReleaseSettings:
    """The maintainers' release token: publishes global recognizer releases."""

    base_url: str
    token: str

    @classmethod
    def from_env(cls, url: str | None = None, token: str | None = None) -> ReleaseSettings:
        resolved_token = (token or os.environ.get(ENV_RELEASE_TOKEN, "")).strip()
        if not resolved_token:
            raise ConfigError(
                f"Set {ENV_RELEASE_TOKEN} (the app's RELEASE_TOKEN secret) or pass --release-token."
            )
        return cls(base_url=_origin(url), token=resolved_token)


def default_workspace_root() -> Path:
    """`ml/workspace` when run from the repo root, otherwise `./workspace`."""
    return Path("ml/workspace") if Path("ml").is_dir() else Path("workspace")


@dataclass(frozen=True)
class Workspace:
    """
    Filesystem layout of one build:

        raw/<artworkId>/<imageId>.jpg   photos pulled from a collection (evaluation only)
        data/foregrounds, data/rooms    open images used to teach the detector
        detector/                       dataset, checkpoint, ONNX graph, metrics
        embedder/                       ONNX graph and its metadata
        eval/                           held-out composites and the evaluation report
        release/                        what `collective-ml release` publishes
    """

    root: Path

    @property
    def collection_json(self) -> Path:
        return self.root / "collection.json"

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def foregrounds_dir(self) -> Path:
        return self.root / "data" / "foregrounds"

    @property
    def rooms_dir(self) -> Path:
        return self.root / "data" / "rooms"

    @property
    def detector_dir(self) -> Path:
        return self.root / "detector"

    @property
    def detector_dataset_dir(self) -> Path:
        return self.detector_dir / "dataset"

    @property
    def detector_checkpoint(self) -> Path:
        return self.detector_dir / "best.pth"

    @property
    def detector_onnx(self) -> Path:
        return self.detector_dir / "model.onnx"

    @property
    def detector_metrics_json(self) -> Path:
        return self.detector_dir / "metrics.json"

    @property
    def embedder_dir(self) -> Path:
        return self.root / "embedder"

    @property
    def embedder_onnx(self) -> Path:
        return self.embedder_dir / "embedder.onnx"

    @property
    def embedder_json(self) -> Path:
        return self.embedder_dir / "embedder.json"

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"

    @property
    def release_dir(self) -> Path:
        return self.root / "release"

    def ensure(self) -> Workspace:
        for path in (
            self.raw_dir,
            self.foregrounds_dir,
            self.rooms_dir,
            self.detector_dir,
            self.embedder_dir,
            self.eval_dir,
            self.release_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self
