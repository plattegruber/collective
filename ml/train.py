#!/usr/bin/env python3
"""
train.py — train a tiny single-stage detector (SSDLite320 MobileNetV3) on the
COCO-style dataset produced by make_dataset.py, then export ONNX for TF.js/ONNX-Web.

Why SSDLite320_MobileNetV3:
- Lightweight, mobile-friendly backbone
- Supported by torchvision, easy to fine-tune
- ONNX export works reliably with opset >= 12

Usage:
  python train.py \
    --data artifacts/datasets \
    --out artifacts/models/ssdlite320 \
    --epochs 20 \
    --batch-size 16 \
    --lr 0.0008 \
    --img-size 320 \
    --workers 8

Dependencies:
  pip install torch torchvision opencv-python pillow numpy tqdm pycocotools onnx onnxruntime

Notes:
- Expects {data}/images/*.jpg and {data}/annotations.json (COCO-lite).
- Category IDs in annotations.json should start at 1 (as written by make_dataset.py).
- Exports: best.pth, best.onnx, labels.json, training_log.json
- For TF.js: convert ONNX -> TF.js with your export step.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


# -----------------------------
# Dataset
# -----------------------------
class CocoLiteDet(Dataset):
    def __init__(self, data_dir: Path, img_size: int = 320, split: str = "train", val_fraction: float = 0.1, seed: int = 1337):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "images"
        with open(self.data_dir / "annotations.json", "r") as f:
            coco = json.load(f)
        # Build indices
        self.images = {img["id"]: img for img in coco["images"]}
        self.cats = {cat["id"]: cat["name"] for cat in coco["categories"]}
        anns_by_img: Dict[int, List[Dict[str, Any]]] = {img_id: [] for img_id in self.images}
        for ann in coco["annotations"]:
            anns_by_img[ann["image_id"]].append(ann)
        self.records: List[Tuple[int, Dict[str, Any]]] = []
        for img_id, img_meta in self.images.items():
            self.records.append((img_id, {"meta": img_meta, "anns": anns_by_img[img_id]}))
        # Train/val split
        rng = np.random.default_rng(seed)
        rng.shuffle(self.records)
        n_val = max(1, int(len(self.records) * val_fraction))
        if split == "train":
            self.records = self.records[n_val:]
        else:
            self.records = self.records[:n_val]
        self.img_size = img_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_id, rec = self.records[idx]
        meta = rec["meta"]
        path = self.img_dir / meta["file_name"]
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        # Resize to square (letterbox-like pad to preserve aspect)
        scale = min(self.img_size / h, self.img_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        top = (self.img_size - new_h) // 2
        left = (self.img_size - new_w) // 2
        canvas[top:top+new_h, left:left+new_w] = img_resized
        img_tensor = to_tensor(canvas)  # [0,1], CxHxW

        # Transform bboxes (COCO xywh -> resize+pad -> tensor)
        boxes: List[List[float]] = []
        labels: List[int] = []
        for ann in rec["anns"]:
            x, y, bw, bh = ann["bbox"]
            x2, y2 = x + bw, y + bh
            # scale + pad
            x = x * scale + left
            y = y * scale + top
            x2 = x2 * scale + left
            y2 = y2 * scale + top
            boxes.append([x, y, x2, y2])
            labels.append(int(ann["category_id"]))
        if len(boxes) == 0:
            # shouldn’t happen with synthetic builder, but guard anyway
            boxes = [[0.0, 0.0, 1.0, 1.0]]
            labels = [1]  # put into a foreground class to avoid SSD complaints
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }
        return img_tensor, target


def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)


# -----------------------------
# Training & Eval Utilities
# -----------------------------
def evaluate_loss(model, loader, device):
    """Compute validation *loss* for torchvision detection models.
    In eval() they return detections, not losses, so temporarily switch to train()
    but keep gradients disabled.
    """
    was_training = model.training
    model.train()
    total = 0.0
    n = 0
    with torch.no_grad():
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total += float(losses.detach().cpu())
            n += 1
    if not was_training:
        model.eval()
    return total / max(1, n)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('artifacts/datasets'), help='Path to dataset (with images/ and annotations.json)')
    ap.add_argument('--out', type=Path, default=Path('artifacts/models/ssdlite320'))
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=8e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--img-size', type=int, default=320)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--freeze-backbone', action='store_true')
    ap.add_argument('--amp', action='store_true', help='Enable CUDA mixed precision')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device selection (CUDA > MPS > CPU)
    use_cuda = torch.cuda.is_available()
    use_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    device = torch.device('cuda') if use_cuda else (torch.device('mps') if use_mps else torch.device('cpu'))

    # AMP only for CUDA
    scaler = torch.amp.GradScaler('cuda', enabled=(use_cuda and args.amp))
    autocast = (lambda: torch.amp.autocast('cuda', enabled=(use_cuda and args.amp)))

    args.out.mkdir(parents=True, exist_ok=True)

    # Datasets & loaders
    ds_train = CocoLiteDet(args.data, img_size=args.img_size, split='train')
    ds_val   = CocoLiteDet(args.data, img_size=args.img_size, split='val')

    N = len(ds_train.cats)  # foreground classes

    pin = use_cuda  # pin_memory helps only on CUDA
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn,
                              pin_memory=pin, drop_last=True)   # avoid BN with batch=1
    val_loader   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, collate_fn=collate_fn,
                              pin_memory=pin, drop_last=False)

    # ---- Model (clean, version-safe) ----
    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        num_classes=N + 1  # include background class as index 0
    )

    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)

    model.to(device)

    # Optimizer & scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float('inf')
    log = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        running = 0.0
        for images, targets in pbar:
            # Skip tiny batches (BN needs >1 sample)
            if len(images) < 2:
                continue
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(losses.detach().cpu())
            pbar.set_postfix({"loss": f"{running / (pbar.n or 1):.3f}"})
        lr_sched.step()

        # Eval
        val_loss = evaluate_loss(model, val_loader, device)
        train_loss = running / max(1, len(train_loader))
        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)
        tqdm.write(f"epoch {epoch}: train {train_loss:.3f} | val {val_loss:.3f}")

        # Save best
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'num_classes': N + 1,
                'img_size': args.img_size,
                'labels': ds_train.cats,
            }, args.out / 'best.pth')

        # periodic checkpoint
        if epoch % 5 == 0:
            torch.save(model.state_dict(), args.out / f'ckpt_epoch{epoch}.pth')

    with open(args.out / 'training_log.json', 'w') as f:
        json.dump(log, f)

    # Save labels
    with open(args.out / 'labels.json', 'w') as f:
        json.dump(ds_train.cats, f)

    # Export ONNX
    model.eval()
    dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    onnx_path = args.out / 'best.onnx'
    # NOTE: output_names are labels for convenience; actual graph may name them differently.
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=['images'], output_names=['boxes', 'scores', 'labels'],
        opset_version=17, do_constant_folding=True,
        dynamic_axes={'images': {0: 'batch'}}
    )
    print(f"Exported ONNX to {onnx_path}")

    with open(args.out / 'export_note.json', 'w') as f:
        json.dump({
            "onnx": str(onnx_path),
            "hint": "Browser will decode raw heads + run NMS; tensor names may differ — inspect session outputs.",
        }, f)


if __name__ == '__main__':
    main()
