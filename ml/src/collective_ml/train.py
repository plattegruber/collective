"""Fine-tune SSDLite320-MobileNetV3 as a class-agnostic artwork detector on composites."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from collective_ml.coco import CocoDataset
from collective_ml.geometry import coco_to_voc, letterbox
from collective_ml.metrics import Detection, GroundTruth, mean_average_precision

log = logging.getLogger(__name__)

Target = dict[str, torch.Tensor]


@dataclass(frozen=True)
class TrainOptions:
    epochs: int = 12
    batch_size: int = 16
    lr: float = 8e-4
    weight_decay: float = 1e-4
    img_size: int = 320
    workers: int = 4
    seed: int = 1337
    device: str | None = None
    val_fraction: float = 0.1
    freeze_backbone: bool = False
    amp: bool = False


@dataclass(frozen=True)
class TrainResult:
    checkpoint: Path
    num_classes: int
    best_epoch: int
    best_val_loss: float
    map50: float
    train_samples: int
    val_samples: int


def pick_device(requested: str | None = None) -> torch.device:
    """CUDA, then Apple MPS, then CPU — unless overridden."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LetterboxedCoco(Dataset[tuple[torch.Tensor, Target]]):
    """Serves square, letterboxed tensors with boxes mapped into canvas space."""

    def __init__(
        self, dataset: CocoDataset, images_dir: Path, image_ids: list[int], img_size: int
    ) -> None:
        self._dataset = dataset
        self._images_dir = images_dir
        self._ids = image_ids
        self._by_id = {image.id: image for image in dataset.images}
        self._img_size = img_size

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Target]:
        image_id = self._ids[index]
        meta = self._by_id[image_id]
        bgr = cv2.imread(str(self._images_dir / meta.file_name))
        if bgr is None:
            raise FileNotFoundError(self._images_dir / meta.file_name)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        box = letterbox(w, h, self._img_size)
        resized = cv2.resize(rgb, (box.width, box.height), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self._img_size, self._img_size, 3), dtype=np.uint8)
        canvas[box.top : box.top + box.height, box.left : box.left + box.width] = resized

        boxes: list[list[float]] = []
        labels: list[int] = []
        for ann in self._dataset.annotations_for(image_id):
            boxes.append(list(box.to_canvas(coco_to_voc(ann.bbox))))
            labels.append(ann.category_id)
        if not boxes:
            boxes, labels = [[0.0, 0.0, 1.0, 1.0]], [1]
        return to_tensor(canvas), {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
        }


def _collate(batch: list[tuple[torch.Tensor, Target]]) -> tuple[list[torch.Tensor], list[Target]]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def split_image_ids(
    dataset: CocoDataset, val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Deterministic train/val split by image id."""
    ids = [image.id for image in dataset.images]
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction)) if len(ids) > 1 else 0
    return ids[n_val:], ids[:n_val]


def build_model(num_classes: int) -> nn.Module:
    """
    SSDLite320 with `num_classes` outputs (background included).

    The ImageNet backbone weights are always requested: torchvision narrows the backbone's
    tail when they are omitted, so a checkpoint only loads into an identically built model.
    """
    return ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        num_classes=num_classes,
    )


def _validation_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    was_training = model.training
    model.train()
    total, count = 0.0, 0
    with torch.no_grad():
        for images, targets in loader:
            if len(images) < 2:
                continue  # BatchNorm in train mode needs more than one sample.
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            total += float(sum(losses.values()).detach().cpu())
            count += 1
    model.train(was_training)
    return total / count if count else float("inf")


def evaluate_map50(
    model: nn.Module, loader: DataLoader, device: torch.device, score_threshold: float = 0.05
) -> float:
    """Run inference on a loader and compute mAP@0.5 in canvas coordinates."""
    model.eval()
    detections: list[Detection] = []
    truths: list[GroundTruth] = []
    with torch.no_grad():
        for images, targets in loader:
            outputs = model([im.to(device) for im in images])
            for output, target in zip(outputs, targets, strict=True):
                image_id = int(target["image_id"].item())
                for box, label in zip(
                    target["boxes"].tolist(), target["labels"].tolist(), strict=True
                ):
                    truths.append(GroundTruth(image_id, int(label), tuple(box)))  # type: ignore[arg-type]
                for box, score, label in zip(
                    output["boxes"].cpu().tolist(),
                    output["scores"].cpu().tolist(),
                    output["labels"].cpu().tolist(),
                    strict=True,
                ):
                    if score >= score_threshold:
                        detections.append(Detection(image_id, int(label), float(score), tuple(box)))  # type: ignore[arg-type]
    return mean_average_precision(detections, truths).map50


def train(dataset_dir: Path, out_dir: Path, options: TrainOptions) -> TrainResult:
    """Train, keep the best checkpoint by val loss, and report mAP@50 on the held-out split."""
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    device = pick_device(options.device)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = CocoDataset.load(dataset_dir / "annotations.json")
    images_dir = dataset_dir / "images"
    train_ids, val_ids = split_image_ids(coco, options.val_fraction, options.seed)
    ds_train = LetterboxedCoco(coco, images_dir, train_ids, options.img_size)
    ds_val = LetterboxedCoco(coco, images_dir, val_ids, options.img_size)
    num_classes = max(coco.categories) + 1

    use_cuda = device.type == "cuda"
    loader_kwargs: dict[str, Any] = {
        "num_workers": options.workers,
        "collate_fn": _collate,
        "pin_memory": use_cuda,
        "persistent_workers": options.workers > 0,
    }
    train_loader = DataLoader(
        ds_train,
        batch_size=options.batch_size,
        shuffle=True,
        drop_last=len(ds_train) >= 2,
        **loader_kwargs,
    )
    val_loader = DataLoader(ds_val, batch_size=options.batch_size, shuffle=False, **loader_kwargs)

    model = build_model(num_classes).to(device)
    if options.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=options.lr, weight_decay=options.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=options.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and options.amp)

    best_val, best_epoch = float("inf"), 0
    history: list[dict[str, float]] = []
    checkpoint = out_dir / "best.pth"

    for epoch in range(1, options.epochs + 1):
        model.train()
        running, steps = 0.0, 0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{options.epochs}", leave=False)
        for images, targets in bar:
            if len(images) < 2:
                continue
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda and options.amp):
                losses = model(images, targets)
                loss = sum(losses.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().cpu())
            steps += 1
            bar.set_postfix(loss=f"{running / max(1, steps):.3f}")
        scheduler.step()

        train_loss = running / max(1, steps)
        val_loss = _validation_loss(model, val_loader, device) if len(ds_val) else train_loss
        if val_loss == float("inf"):
            val_loss = train_loss  # Validation split too small for a loss pass; fall back.
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        log.info("epoch %d: train %.3f | val %.3f", epoch, train_loss, val_loss)

        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_classes": num_classes,
                    "img_size": options.img_size,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                checkpoint,
            )

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state"])
    map50 = evaluate_map50(model, val_loader, device) if len(ds_val) else 0.0
    log.info("best epoch %d (val %.3f), mAP@50 %.3f", best_epoch, best_val, map50)

    result = TrainResult(
        checkpoint=checkpoint,
        num_classes=num_classes,
        best_epoch=best_epoch,
        best_val_loss=best_val,
        map50=map50,
        train_samples=len(ds_train),
        val_samples=len(ds_val),
    )
    metrics = {
        "map50": round(map50, 4),
        "bestEpoch": best_epoch,
        "bestValLoss": round(best_val, 4),
        "epochs": options.epochs,
        "imgSize": options.img_size,
        "samples": len(ds_train) + len(ds_val),
        "classes": num_classes - 1,
        "history": history,
        "options": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(options).items()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return result
