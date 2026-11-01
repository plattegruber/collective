"""Export the trained detector to ONNX in the shape the browser scanner expects."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from collective_ml.train import build_model

log = logging.getLogger(__name__)

INPUT_NAME = "images"
OUTPUT_NAMES = ("boxes", "scores", "labels")


class BatchedDetections(nn.Module):
    """
    Wraps a torchvision detector so a single-image batch yields
    `boxes [1,N,4]`, `scores [1,N]`, `labels [1,N]` instead of a list of dicts.
    """

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.detector = detector

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.detector(images)[0]
        return (
            output["boxes"].unsqueeze(0),
            output["scores"].unsqueeze(0),
            output["labels"].unsqueeze(0),
        )


@dataclass(frozen=True)
class ExportResult:
    onnx_path: Path
    input_size: int
    output_names: tuple[str, ...]
    output_shapes: tuple[tuple[int, ...], ...]


def load_checkpoint(checkpoint: Path) -> tuple[nn.Module, int]:
    """Rebuild the model from `best.pth`; returns (model, img_size)."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(int(state["num_classes"]))
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, int(state["img_size"])


def export_onnx(checkpoint: Path, onnx_path: Path, opset: int = 17) -> ExportResult:
    """Export and verify with onnxruntime on a random letterboxed frame."""
    model, img_size = load_checkpoint(checkpoint)
    wrapped = BatchedDetections(model).eval()
    dummy = torch.rand(1, 3, img_size, img_size)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs: dict[str, object] = {
        "input_names": [INPUT_NAME],
        "output_names": list(OUTPUT_NAMES),
        "opset_version": opset,
        "do_constant_folding": True,
        "dynamic_axes": {
            "boxes": {1: "detections"},
            "scores": {1: "detections"},
            "labels": {1: "detections"},
        },
    }
    # Newer torch defaults to the dynamo exporter, which does not handle torchvision's
    # detection post-processing; request the TorchScript tracer when the flag exists.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    with torch.no_grad():
        torch.onnx.export(wrapped, (dummy,), str(onnx_path), **export_kwargs)

    onnx.checker.check_model(str(onnx_path))
    return verify_onnx(onnx_path, img_size)


def verify_onnx(onnx_path: Path, img_size: int) -> ExportResult:
    """Run the graph once with onnxruntime and assert the I/O contract."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]
    if input_names != [INPUT_NAME]:
        raise RuntimeError(f"Expected input {INPUT_NAME!r}, got {input_names}")
    output_names = tuple(o.name for o in session.get_outputs())
    if output_names != OUTPUT_NAMES:
        raise RuntimeError(f"Expected outputs {OUTPUT_NAMES}, got {output_names}")

    frame = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
    outputs = session.run(None, {INPUT_NAME: frame})
    shapes = tuple(tuple(int(d) for d in o.shape) for o in outputs)
    boxes, scores, labels = outputs
    if boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[2] != 4:
        raise RuntimeError(f"boxes must be [1,N,4], got {boxes.shape}")
    if scores.shape != boxes.shape[:2] or labels.shape != boxes.shape[:2]:
        raise RuntimeError(f"scores/labels must be [1,N]; got {scores.shape} / {labels.shape}")
    if labels.dtype != np.int64:
        raise RuntimeError(f"labels must be int64, got {labels.dtype}")
    log.info(
        "ONNX ok: input %s [1,3,%d,%d]; outputs %s with shapes %s",
        INPUT_NAME,
        img_size,
        img_size,
        output_names,
        shapes,
    )
    described = ", ".join(
        f"{name}{shape}" for name, shape in zip(output_names, shapes, strict=True)
    )
    print(f"outputs: {described}")
    return ExportResult(
        onnx_path=onnx_path, input_size=img_size, output_names=output_names, output_shapes=shapes
    )
