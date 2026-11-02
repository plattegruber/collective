"""Bundle a detector and an embedder into a release and publish it to the app."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx

from collective_ml.config import ReleaseSettings, Workspace
from collective_ml.embedder import EmbedderInfo

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Thresholds:
    detect: float = 0.45
    match: float = 0.6
    strong: float = 0.72
    # A match must also beat the runner-up piece by this much (a ratio test), which keeps
    # unknown art out without raising the absolute threshold as collections grow.
    margin: float = 0.08
    # The absolute thresholds were calibrated on a collection of `match_reference` pieces;
    # the browser raises them by `match_slope` * ln(pieces / reference) for larger ones,
    # because the best impostor among more pieces scores higher.
    match_reference: int = 7
    match_slope: float = 0.034
    memory_novelty: float = 0.92


@dataclass(frozen=True)
class ReleaseConfig:
    """Mirrors `RecognizerConfig` in the app (camelCase on the wire)."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    pooling: str
    attribution: str
    crop_margin: float = 0.06
    thresholds: Thresholds = field(default_factory=Thresholds)
    max_memories: int = 8

    def to_payload(self) -> dict[str, Any]:
        return {
            "mean": list(self.mean),
            "std": list(self.std),
            "pooling": self.pooling,
            "cropMargin": self.crop_margin,
            "thresholds": {
                "detect": self.thresholds.detect,
                "match": self.thresholds.match,
                "strong": self.thresholds.strong,
                "margin": self.thresholds.margin,
                "matchReference": self.thresholds.match_reference,
                "matchSlope": self.thresholds.match_slope,
                "memoryNovelty": self.thresholds.memory_novelty,
            },
            "maxMemories": self.max_memories,
            "attribution": self.attribution,
        }


def assemble_release(
    workspace: Workspace,
    version: str,
    detector_input_size: int = 320,
    thresholds: Thresholds | None = None,
    notes: str | None = None,
    out_dir: Path | None = None,
    overrides: dict[str, float] | None = None,
) -> Path:
    """
    Copies `detector.onnx` and `embedder.onnx` into the release folder and writes
    `release.json` with everything the publish endpoint needs. Thresholds default to the
    ones the last evaluation suggested, when a report exists.
    """
    target = out_dir or workspace.release_dir
    target.mkdir(parents=True, exist_ok=True)
    if not workspace.detector_onnx.exists():
        raise SystemExit(
            f"{workspace.detector_onnx} not found. Run `collective-ml build-detector`."
        )
    if not workspace.embedder_onnx.exists():
        raise SystemExit(
            f"{workspace.embedder_onnx} not found. Run `collective-ml build-embedder`."
        )
    info = EmbedderInfo.load(workspace.embedder_json)
    shutil.copyfile(workspace.detector_onnx, target / "detector.onnx")
    shutil.copyfile(workspace.embedder_onnx, target / "embedder.onnx")

    metrics: dict[str, Any] = {}
    report_path = workspace.eval_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        metrics = {k: v for k, v in report.items() if k != "samples"}
        if thresholds is None and "suggested" in report:
            suggested = report["suggested"]
            thresholds = Thresholds(
                match=float(suggested["match"]),
                strong=float(suggested["strong"]),
                margin=float(suggested.get("margin", Thresholds.margin)),
                match_reference=int(report.get("pieces", Thresholds.match_reference)),
            )
    if workspace.detector_metrics_json.exists():
        detector_metrics = json.loads(workspace.detector_metrics_json.read_text())
        detector_metrics.pop("history", None)
        metrics["detector"] = detector_metrics
    metrics["embedder"] = {
        "model": info.model,
        "quantized": info.quantized,
        "cosineVsFp32": info.cosine_vs_fp32,
    }

    if overrides:
        thresholds = replace(thresholds or Thresholds(), **overrides)
    config = ReleaseConfig(
        mean=info.mean,
        std=info.std,
        pooling=info.pooling,
        attribution=info.attribution,
        thresholds=thresholds or Thresholds(),
    )
    manifest = {
        "version": version,
        "detectorInputSize": detector_input_size,
        "embedderInputSize": info.input_size,
        "embedderDim": info.dim,
        "config": config.to_payload(),
        "metrics": metrics,
        "notes": notes or "",
    }
    (target / "release.json").write_text(json.dumps(manifest, indent=2))
    log.info("Release %s assembled in %s", version, target)
    return target


def publish_release(
    settings: ReleaseSettings,
    release_dir: Path,
    activate: bool = True,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """POSTs the release folder to `/api/ml/releases`; returns the manifest the app serves."""
    manifest = json.loads((release_dir / "release.json").read_text())
    data = {
        "version": manifest["version"],
        "detectorInputSize": str(manifest["detectorInputSize"]),
        "embedderInputSize": str(manifest["embedderInputSize"]),
        "embedderDim": str(manifest["embedderDim"]),
        "config": json.dumps(manifest["config"]),
        "metrics": json.dumps(manifest.get("metrics") or {}),
        "notes": manifest.get("notes") or "",
        "activate": "true" if activate else "false",
    }
    with (
        (release_dir / "detector.onnx").open("rb") as detector,
        (release_dir / "embedder.onnx").open("rb") as embedder,
        httpx.Client(
            base_url=settings.base_url,
            headers={"Authorization": f"Bearer {settings.token}", "User-Agent": "collective-ml"},
            timeout=httpx.Timeout(60.0, write=600.0, read=300.0),
            transport=transport,
        ) as client,
    ):
        response = client.post(
            "/api/ml/releases",
            data=data,
            files={
                "detector": ("detector.onnx", detector, "application/octet-stream"),
                "embedder": ("embedder.onnx", embedder, "application/octet-stream"),
            },
        )
    if not response.is_success:
        detail = ""
        try:
            detail = str(response.json().get("message") or "")
        except ValueError:
            detail = response.text[:300]
        raise SystemExit(f"Publish failed (HTTP {response.status_code}): {detail}")
    payload = response.json()
    log.info("Published release %s (%s)", payload.get("version"), payload.get("id"))
    return payload
