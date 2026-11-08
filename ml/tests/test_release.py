import json
from pathlib import Path

import httpx

from collective_ml.config import ReleaseSettings, Workspace
from collective_ml.embedder import EmbedderInfo
from collective_ml.release import Thresholds, assemble_release, publish_release


def _workspace(tmp_path: Path) -> Workspace:
    workspace = Workspace(tmp_path / "ws").ensure()
    workspace.detector_onnx.write_bytes(b"detector")
    workspace.embedder_onnx.write_bytes(b"embedder")
    EmbedderInfo(
        model="vit_small_patch16_dinov3.lvd1689m",
        input_size=224,
        dim=768,
        patch_size=16,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        pooling="cls+mean",
        bytes=8,
        quantized=True,
        cosine_vs_fp32=0.999,
        attribution="Built with DINOv3",
    ).save(workspace.embedder_json)
    return workspace


def test_assemble_release_uses_evaluation_thresholds(tmp_path: Path):
    workspace = _workspace(tmp_path)
    (workspace.eval_dir / "report.json").write_text(
        json.dumps(
            {
                "top1Overall": 0.98,
                "pieces": 7,
                "suggested": {"match": 0.63, "strong": 0.74, "margin": 0.08},
                "sweep": [],
            }
        )
    )
    release_dir = assemble_release(workspace, "2026.11.02-1", notes="first")
    manifest = json.loads((release_dir / "release.json").read_text())
    assert manifest["version"] == "2026.11.02-1"
    assert manifest["embedderDim"] == 768 and manifest["detectorInputSize"] == 320
    assert manifest["config"]["thresholds"]["match"] == 0.63
    assert manifest["config"]["thresholds"]["strong"] == 0.74
    assert manifest["config"]["thresholds"]["margin"] == 0.08
    assert manifest["config"]["thresholds"]["matchReference"] == 7
    assert manifest["config"]["attribution"] == "Built with DINOv3"
    assert manifest["metrics"]["top1Overall"] == 0.98
    assert (release_dir / "detector.onnx").read_bytes() == b"detector"


def test_assemble_release_explicit_thresholds_win(tmp_path: Path):
    workspace = _workspace(tmp_path)
    release_dir = assemble_release(workspace, "v2", thresholds=Thresholds(match=0.5, strong=0.9))
    manifest = json.loads((release_dir / "release.json").read_text())
    assert manifest["config"]["thresholds"] == {
        "detect": 0.45,
        "match": 0.5,
        "strong": 0.9,
        "margin": 0.08,
        "matchReference": 7,
        "matchSlope": 0.034,
        "memoryNovelty": 0.92,
    }


def test_publish_release_sends_multipart(tmp_path: Path):
    workspace = _workspace(tmp_path)
    release_dir = assemble_release(workspace, "v3")
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["origin"] = request.headers["origin"]
        seen["path"] = request.url.path
        body = request.read()
        seen["has_files"] = b'name="detector"' in body and b'name="embedder"' in body
        seen["version"] = b"v3" in body
        return httpx.Response(201, json={"id": "rel1", "version": "v3"})

    payload = publish_release(
        ReleaseSettings(base_url="https://collective.test", token="secret"),
        release_dir,
        transport=httpx.MockTransport(handler),
    )
    assert payload["id"] == "rel1"
    assert seen == {
        "auth": "Bearer secret",
        "origin": "https://collective.test",
        "path": "/api/ml/releases",
        "has_files": True,
        "version": True,
    }


def test_assemble_release_overrides_win(tmp_path: Path):
    workspace = _workspace(tmp_path)
    release_dir = assemble_release(workspace, "v4", overrides={"margin": 0.12})
    manifest = json.loads((release_dir / "release.json").read_text())
    assert manifest["config"]["thresholds"]["margin"] == 0.12
