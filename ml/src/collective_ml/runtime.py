"""
The recognizer as the browser runs it, in Python: same letterboxing, same normalisation,
same pooling. Used to evaluate releases end to end with the exact ONNX graphs that ship.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from collective_ml.geometry import Box, letterbox

Pooling = str  # "cls" | "mean" | "cls+mean"


@dataclass(frozen=True)
class Region:
    score: float
    box: Box


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def letterbox_image(
    image: Image.Image, size: int, fill: tuple[int, int, int]
) -> tuple[np.ndarray, object]:
    """Fits the image into a `size` by `size` canvas exactly like the browser's FrameBuffer."""
    box = letterbox(image.width, image.height, size)
    canvas = Image.new("RGB", (size, size), fill)
    resized = image.resize((box.width, box.height), Image.Resampling.BILINEAR)
    canvas.paste(resized, (box.left, box.top))
    return np.asarray(canvas, dtype=np.float32) / 255.0, box


class OnnxDetector:
    """Class-agnostic artwork detector: `images [1,3,S,S]` RGB 0..1 → boxes/scores/labels."""

    def __init__(self, path: Path, input_size: int) -> None:
        self.input_size = input_size
        self._session = _session(path)
        self._input = self._session.get_inputs()[0].name

    def detect(self, image: Image.Image, threshold: float, min_edge: int = 8) -> list[Region]:
        pixels, box = letterbox_image(image.convert("RGB"), self.input_size, (0, 0, 0))
        tensor = np.transpose(pixels, (2, 0, 1))[None]
        boxes, scores, labels = self._session.run(None, {self._input: tensor})
        regions: list[Region] = []
        for i in range(scores.shape[1]):
            score = float(scores[0, i])
            if score < threshold or int(labels[0, i]) <= 0:
                continue
            x1, y1, x2, y2 = (float(v) for v in boxes[0, i])
            source = box.to_source((x1, y1, x2, y2))
            source = (
                min(max(0.0, source[0]), image.width),
                min(max(0.0, source[1]), image.height),
                min(max(0.0, source[2]), image.width),
                min(max(0.0, source[3]), image.height),
            )
            if source[2] - source[0] < min_edge or source[3] - source[1] < min_edge:
                continue
            regions.append(Region(score, source))
        return sorted(regions, key=lambda r: r.score, reverse=True)


def expand_box(box: Box, margin: float, width: int, height: int) -> Box:
    dx = (box[2] - box[0]) * margin
    dy = (box[3] - box[1]) * margin
    return (
        max(0.0, box[0] - dx),
        max(0.0, box[1] - dy),
        min(float(width), box[2] + dx),
        min(float(height), box[3] + dy),
    )


def pool_tokens(cls: np.ndarray, patches: np.ndarray, pooling: Pooling) -> np.ndarray:
    """Turns `cls [C]` and `patches [N, C]` into the vector the index stores, L2-normalised."""
    if pooling == "cls":
        vector = cls
    elif pooling == "mean":
        vector = patches.mean(axis=0)
    elif pooling == "cls+mean":
        vector = np.concatenate([cls, patches.mean(axis=0)])
    else:
        raise ValueError(f"Unknown pooling {pooling!r}")
    norm = float(np.linalg.norm(vector))
    return (vector / norm).astype(np.float32) if norm > 0 else vector.astype(np.float32)


class OnnxEmbedder:
    """DINO-style embedder: `image [1,3,S,S]` standardised RGB → `cls [1,C]`, `patches [1,N,C]`."""

    def __init__(
        self,
        path: Path,
        input_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        pooling: Pooling = "cls+mean",
        crop_margin: float = 0.06,
    ) -> None:
        self.input_size = input_size
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.pooling = pooling
        self.crop_margin = crop_margin
        self._session = _session(path)
        self._input = self._session.get_inputs()[0].name

    @property
    def dim(self) -> int:
        channels = int(self._session.get_outputs()[0].shape[-1])
        return channels * (2 if self.pooling == "cls+mean" else 1)

    def embed(self, image: Image.Image, box: Box | None = None) -> np.ndarray:
        image = image.convert("RGB")
        if box is not None:
            x1, y1, x2, y2 = expand_box(box, self.crop_margin, image.width, image.height)
            image = image.crop(
                (int(x1), int(y1), max(int(x1) + 1, int(x2)), max(int(y1) + 1, int(y2)))
            )
        fill = tuple(round(v * 255) for v in self.mean)
        pixels, _ = letterbox_image(image, self.input_size, fill)  # type: ignore[arg-type]
        tensor = np.transpose((pixels - self.mean) / self.std, (2, 0, 1))[None].astype(np.float32)
        cls, patches = self._session.run(None, {self._input: tensor})
        return pool_tokens(cls[0], patches[0], self.pooling)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator > 0 else 0.0
