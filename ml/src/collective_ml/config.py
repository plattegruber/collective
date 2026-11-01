"""Runtime settings and workspace layout."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_URL = "COLLECTIVE_URL"
ENV_TOKEN = "COLLECTIVE_TOKEN"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


@dataclass(frozen=True)
class ApiSettings:
    """Where the Collective app lives and how to authenticate against it."""

    base_url: str
    token: str

    @classmethod
    def from_env(cls, url: str | None = None, token: str | None = None) -> ApiSettings:
        """Build settings from explicit values, falling back to environment variables."""
        resolved_url = (url or os.environ.get(ENV_URL, "")).strip().rstrip("/")
        resolved_token = (token or os.environ.get(ENV_TOKEN, "")).strip()
        if not resolved_url:
            raise ConfigError(f"Set {ENV_URL} (e.g. https://collective.example.com) or pass --url.")
        if not resolved_token:
            raise ConfigError(
                f"Set {ENV_TOKEN} to an API token from Settings → API tokens, or pass --token."
            )
        return cls(base_url=resolved_url, token=resolved_token)


def default_workspace_root() -> Path:
    """`ml/workspace` when run from the repo root, otherwise `./workspace`."""
    return Path("ml/workspace") if Path("ml").is_dir() else Path("workspace")


@dataclass(frozen=True)
class Workspace:
    """Filesystem layout for one collector's training run."""

    root: Path

    @property
    def collection_json(self) -> Path:
        return self.root / "collection.json"

    @property
    def raw_dir(self) -> Path:
        """`raw/<artworkId>/<imageId>.<ext>` - one folder per artwork."""
        return self.root / "raw"

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def model_dir(self) -> Path:
        return self.root / "model"

    @property
    def labels_json(self) -> Path:
        return self.model_dir / "labels.json"

    @property
    def checkpoint(self) -> Path:
        return self.model_dir / "best.pth"

    @property
    def onnx_path(self) -> Path:
        return self.model_dir / "model.onnx"

    @property
    def metrics_json(self) -> Path:
        return self.model_dir / "metrics.json"

    def ensure(self) -> Workspace:
        """Create the directory tree."""
        for path in (self.raw_dir, self.dataset_dir, self.model_dir):
            path.mkdir(parents=True, exist_ok=True)
        return self
