"""Export the embedding backbone (DINOv3 ViT-S/16 by default) to a browser-sized ONNX graph."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from collective_ml.runtime import OnnxEmbedder, cosine, letterbox_image, pool_tokens

log = logging.getLogger(__name__)

DEFAULT_EMBEDDER = "vit_small_patch16_dinov3.lvd1689m"


@dataclass(frozen=True)
class EmbedderSpec:
    model: str = DEFAULT_EMBEDDER
    input_size: int = 224
    pooling: str = "cls+mean"
    # Weight-only 8-bit block quantisation: ~4x smaller, cosine > 0.999 to fp32 in our tests.
    quantize: bool = True
    opset: int = 17


@dataclass(frozen=True)
class EmbedderInfo:
    """What a release needs to know about the exported embedder."""

    model: str
    input_size: int
    dim: int
    patch_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    pooling: str
    bytes: int
    quantized: bool
    cosine_vs_fp32: float
    attribution: str

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> EmbedderInfo:
        data = json.loads(path.read_text())
        data["mean"] = tuple(data["mean"])
        data["std"] = tuple(data["std"])
        return cls(**data)


def attribution_for(model_name: str) -> str:
    if "dinov3" in model_name:
        return "Built with DINOv3"
    if "dinov2" in model_name:
        return "Built with DINOv2"
    return f"Embeddings by {model_name}"


def _quantize_weights(source: Path, target: Path, bits: int = 8, block_size: int = 128) -> bool:
    try:
        import onnx
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            DefaultWeightOnlyQuantConfig,
            MatMulNBitsQuantizer,
        )
    except ImportError:
        log.warning("onnxruntime lacks MatMulNBitsQuantizer; shipping fp32 weights.")
        return False
    model = onnx.load(str(source))
    quantizer = MatMulNBitsQuantizer(
        model,
        algo_config=DefaultWeightOnlyQuantConfig(
            block_size=block_size, is_symmetric=True, bits=bits
        ),
    )
    quantizer.process()
    quantizer.model.save_model_to_file(str(target), False)
    return True


def export_embedder(spec: EmbedderSpec, out_dir: Path) -> EmbedderInfo:
    """
    Downloads the backbone through timm, exports `embedder.onnx` with outputs `cls` and
    `patches`, quantises it, verifies it against PyTorch, and writes `embedder.json`.
    """
    import timm
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    model = timm.create_model(
        spec.model, pretrained=True, num_classes=0, img_size=spec.input_size
    ).eval()
    config = timm.data.resolve_data_config({}, model=model)
    prefix = int(model.num_prefix_tokens)
    patch = int(model.patch_embed.patch_size[0])

    class Tokens(torch.nn.Module):
        def __init__(self, backbone: torch.nn.Module) -> None:
            super().__init__()
            self.backbone = backbone

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            tokens = self.backbone.forward_features(x)
            return tokens[:, 0], tokens[:, prefix:]

    wrapped = Tokens(model)
    fp32_path = out_dir / "embedder.fp32.onnx"
    example = torch.randn(1, 3, spec.input_size, spec.input_size)
    torch.onnx.export(
        wrapped,
        example,
        str(fp32_path),
        input_names=["image"],
        output_names=["cls", "patches"],
        opset_version=spec.opset,
        dynamo=False,
        do_constant_folding=True,
    )
    final_path = out_dir / "embedder.onnx"
    quantized = spec.quantize and _quantize_weights(fp32_path, final_path)
    if not quantized:
        final_path.write_bytes(fp32_path.read_bytes())

    # Parity check: the shipped graph against PyTorch on the same letterboxed pixels, so
    # the number reflects quantisation alone.
    rng = np.random.default_rng(0)
    probe = Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))
    mean = tuple(float(v) for v in config["mean"])
    std = tuple(float(v) for v in config["std"])
    runtime = OnnxEmbedder(final_path, spec.input_size, mean, std, spec.pooling)  # type: ignore[arg-type]
    onnx_vector = runtime.embed(probe)
    fill = tuple(round(v * 255) for v in mean)
    pixels, _ = letterbox_image(probe, spec.input_size, fill)  # type: ignore[arg-type]
    with torch.no_grad():
        tensor = torch.from_numpy(
            np.transpose((pixels - np.asarray(mean)) / np.asarray(std), (2, 0, 1))[None]
        ).float()
        cls, patches = wrapped(tensor)
    torch_vector = pool_tokens(cls[0].numpy(), patches[0].numpy(), spec.pooling)
    parity = cosine(onnx_vector, torch_vector)
    log.info("Embedder parity (ONNX vs PyTorch, same input): cosine %.4f", parity)

    info = EmbedderInfo(
        model=spec.model,
        input_size=spec.input_size,
        dim=runtime.dim,
        patch_size=patch,
        mean=mean,  # type: ignore[arg-type]
        std=std,  # type: ignore[arg-type]
        pooling=spec.pooling,
        bytes=final_path.stat().st_size,
        quantized=quantized,
        cosine_vs_fp32=parity,
        attribution=attribution_for(spec.model),
    )
    info.save(out_dir / "embedder.json")
    log.info(
        "Exported %s → %s (%.1f MB, dim %d)", spec.model, final_path, info.bytes / 1e6, info.dim
    )
    return info
