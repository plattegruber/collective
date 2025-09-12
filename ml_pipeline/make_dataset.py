#!/usr/bin/env python3
"""
make_dataset.py — build a synthetic object-detection dataset for your gallery pieces.

Outputs a COCO-style dataset with heavy, realistic augmentations tailored for
museum/home viewing: perspective, motion blur, glare, shadows, occlusion, color/contrast,
JPEG artifacts, and noise — all on top of randomized wall-like backgrounds.

Usage:
  python make_dataset.py \
      --assets-dir assets \
      --out out_dataset \
      --img-size 640 \
      --samples-per-piece 200 \
      --backgrounds-dir optional/backgrounds

Dependencies:
  pip install pillow opencv-python numpy albumentations tqdm

Notes:
- We infer class IDs from folder names under assets/2d/* and assets/3d/*.
- Each source image can be any photograph (PNG/JPG). Transparent PNGs are supported.
- BBoxes are Pascal VOC during augmentation, converted to COCO on write.
- We resize the final synthetic to a square (img-size x img-size).
"""
import argparse
import os
import json
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2  # not strictly needed, but common
from tqdm import tqdm

# -----------------------------
# Utilities
# -----------------------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}


def load_rgba(image_path: Path) -> Image.Image:
    img = Image.open(image_path).convert("RGBA")
    return img


def solid_or_texture_bg(size: int) -> Image.Image:
    """Generate a procedural wall-like background (solid + subtle noise/gradient)."""
    w = h = size
    base = np.ones((h, w, 3), dtype=np.uint8)
    # choose a neutral wall-like color
    hue = random.random()
    sat = random.uniform(0.05, 0.18)
    val = random.uniform(0.80, 0.98)
    # convert HSV to RGB
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    base[:] = (int(r*255), int(g*255), int(b*255))
    # add gentle vignette/gradient
    gy, gx = np.mgrid[0:h, 0:w]
    cx, cy = w/2, h/2
    dist = np.sqrt(((gx-cx)/w)**2 + ((gy-cy)/h)**2)
    falloff = (1 - np.clip(dist*1.7, 0, 1))[:, :, None]
    base = (base.astype(np.float32) * (0.92 + 0.08*falloff)).clip(0,255).astype(np.uint8)
    # subtle noise
    noise = np.random.normal(0, 1.0, (h, w, 3)).astype(np.float32)
    base = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(base)


def choose_background(size: int, backgrounds_dir: Path | None) -> Image.Image:
    if backgrounds_dir and backgrounds_dir.exists():
        candidates = [p for p in backgrounds_dir.rglob('*') if is_image_file(p)]
        if candidates:
            p = random.choice(candidates)
            bg = Image.open(p).convert('RGB')
            bg = ImageOps.exif_transpose(bg)
            bg = bg.resize((size, size), Image.LANCZOS)
            return bg
    return solid_or_texture_bg(size)


def paste_rgba_on(bg: Image.Image, fg: Image.Image, xy: Tuple[int, int]) -> Tuple[Image.Image, Tuple[int,int,int,int]]:
    """Paste RGBA foreground onto RGB background at (x,y). Return new image and bbox (x1,y1,x2,y2)."""
    bg = bg.copy()
    x, y = xy
    fw, fh = fg.size
    # Clamp paste coords
    x = int(np.clip(x, 0, bg.width - fw))
    y = int(np.clip(y, 0, bg.height - fh))
    bg.paste(fg, (x, y), fg)
    bbox = (x, y, x + fw, y + fh)
    return bg, bbox


def random_resize_keep_aspect(img: Image.Image, target_short: int, min_scale=0.35, max_scale=0.9) -> Image.Image:
    """Resize so that the foreground occupies a reasonable fraction of the canvas."""
    w, h = img.size
    short = min(w, h)
    scale = random.uniform(min_scale, max_scale)
    new_short = int(target_short * scale)
    if short == w:
        new_w = new_short
        new_h = int(h * (new_w / w))
    else:
        new_h = new_short
        new_w = int(w * (new_h / h))
    return img.resize((new_w, new_h), Image.LANCZOS)


def rand_xy_for(bg_size: int, fg_size: Tuple[int, int]) -> Tuple[int, int]:
    bw = bh = bg_size
    fw, fh = fg_size
    pad = int(0.03 * bg_size)
    x = random.randint(pad, max(pad, bw - fw - pad))
    y = random.randint(pad, max(pad, bh - fh - pad))
    return x, y


def rgba_from_rgb_with_alpha(rgb: Image.Image) -> Image.Image:
    """If source lacks alpha, synth a soft alpha that keeps rectangular shape but rounded edges."""
    rgb = rgb.convert('RGB')
    w, h = rgb.size
    alpha = Image.new('L', (w, h), 255)
    # round the corners slightly to mimic frames/sculpture silhouettes forgivingly
    rad = max(2, min(w, h)//50)
    alpha = ImageOps.expand(Image.new('L', (w-2*rad, h-2*rad), 255), border=rad, fill=255)
    alpha = alpha.resize((w, h), Image.LANCZOS)
    rgba = rgb.copy()
    rgba.putalpha(alpha)
    return rgba


# -----------------------------
# Albumentations pipeline
# -----------------------------

def build_aug_pipeline(img_size: int) -> A.Compose:
    """
    Version-agnostic Albumentations pipeline:
    - Handles API differences across releases (Cutout/CoarseDropout, ImageCompression/JpegCompression, GaussianNoise/GaussNoise).
    - Skips transforms your local version doesn't support instead of crashing.
    """
    def try_add(tlist, name, **kwargs):
        cls = getattr(A, name, None)
        if cls is None:
            print(f"[augment] skip {name}: not in Albumentations {getattr(A, '__version__', '?')}")
            return
        try:
            tlist.append(cls(**kwargs))
        except TypeError as e:
            print(f"[augment] skip {name}: bad args for this version -> {e}")

    ts = []

    # Geometric
    try_add(ts, "Affine",
            scale=(0.9, 1.12),
            translate_percent=(0.0, 0.035),   # was 0.05
            rotate=(-6, 6),
            shear=(-3, 3),
            p=0.8)
    try_add(ts, "Perspective", scale=(0.02, 0.06), p=0.6)

    # Photometric / sensor
    try_add(ts, "RandomBrightnessContrast", brightness_limit=0.15, contrast_limit=0.2, p=0.8)
    try_add(ts, "HueSaturationValue", hue_shift_limit=10, sat_shift_limit=12, val_shift_limit=10, p=0.5)

    # Noise (prefer GaussianNoise; fallback to GaussNoise with conservative defaults)
    if hasattr(A, "GaussianNoise"):
        try_add(ts, "GaussianNoise", var_limit=(5.0, 18.0), p=0.4)
    else:
        # Some old versions reject var_limit param; call with no kwargs
        try_add(ts, "GaussNoise", p=0.4)

    # Blur / motion
    try_add(ts, "MotionBlur", blur_limit=5, p=0.3)
    try_add(ts, "GaussianBlur", blur_limit=3, p=0.2)

    # Glare / lighting
    try_add(ts, "RandomSunFlare", src_radius=80, flare_roi=(0.0, 0.0, 1.0, 0.5), p=0.25)

    # Occlusion (prefer Cutout; fallback to CoarseDropout with minimal args)
    if hasattr(A, "Cutout"):
        try_add(ts, "Cutout",
                num_holes=2,
                max_h_size=int(0.18 * img_size),
                max_w_size=int(0.18 * img_size),
                fill_value=0, p=0.35)
    else:
        # Older CoarseDropout signatures vary a lot; keep args minimal
        try_add(ts, "CoarseDropout", max_holes=2, p=0.35)

    # Compression artifacts
    if hasattr(A, "ImageCompression"):
        try_add(ts, "ImageCompression", quality_range=(55, 92), p=0.7)
    else:
        try_add(ts, "JpegCompression", quality_lower=55, quality_upper=92, p=0.7)

    # Final resize/pad to square
    try_add(ts, "LongestMaxSize", max_size=img_size, interpolation=cv2.INTER_AREA)
    try_add(ts, "PadIfNeeded", min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT)

    return A.Compose(
        ts,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_area=16,
            min_visibility=0.3,
            clip=True,                   # <-- important: clamp to image after each transform
            check_each_transform=False   # <-- avoid early hard failures; we still filter by visibility
        )
    )


# -----------------------------
# COCO writer
# -----------------------------

def voc_to_coco_bbox(voc_bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = voc_bbox
    return float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))


@dataclass
class CocoAnn:
    image_id: int
    category_id: int
    bbox: Tuple[float, float, float, float]
    area: float
    iscrowd: int = 0


# -----------------------------
# Main synthesis loop
# -----------------------------

def gather_pieces(assets_dir: Path) -> Dict[str, List[Path]]:
    pieces: Dict[str, List[Path]] = {}
    for sub in ("2d", "3d"):
        d = assets_dir / sub
        if not d.exists():
            continue
        for piece_dir in sorted(d.iterdir()):
            if not piece_dir.is_dir():
                continue
            pid = f"{sub}:{piece_dir.name}"
            imgs = [p for p in piece_dir.iterdir() if is_image_file(p)]
            if imgs:
                pieces[pid] = imgs
    return pieces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assets-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--backgrounds-dir', type=Path, default=None)
    ap.add_argument('--img-size', type=int, default=640)
    ap.add_argument('--samples-per-piece', type=int, default=200)
    ap.add_argument('--seed', type=int, default=1337)
    args = ap.parse_args()

    seed_everything(args.seed)

    out_images = args.out / 'images'
    out_images.mkdir(parents=True, exist_ok=True)
    anns: List[CocoAnn] = []
    images_meta = []

    pieces = gather_pieces(args.assets_dir)
    if not pieces:
        raise SystemExit(f"No images found under {args.assets_dir}/(2d|3d)/*")

    # category mapping
    categories = []
    cat_id_of: Dict[str, int] = {}
    for i, pid in enumerate(sorted(pieces.keys()), start=1):
        cat_id_of[pid] = i
        categories.append({"id": i, "name": pid})

    aug = build_aug_pipeline(args.img_size)

    image_id = 1
    ann_id = 1

    for pid, src_list in pieces.items():
        cat_id = cat_id_of[pid]
        n_samples = args.samples_per_piece
        pbar = tqdm(range(n_samples), desc=f"{pid}")
        for _ in pbar:
            # 1) Build background
            bg = choose_background(args.img_size, args.backgrounds_dir)
            bg = ImageOps.exif_transpose(bg)

            # 2) Choose a source shot
            src_path = random.choice(src_list)
            fg_rgba = load_rgba(src_path)
            # If no alpha channel content (fully opaque), synth a mild alpha
            if np.array(fg_rgba.split()[-1]).mean() > 250:
                fg_rgba = rgba_from_rgb_with_alpha(fg_rgba.convert('RGB'))

            # 3) Randomly scale + paste
            fg_scaled = random_resize_keep_aspect(fg_rgba, target_short=args.img_size,
                                                  min_scale=0.35, max_scale=0.92)
            x, y = rand_xy_for(args.img_size, fg_scaled.size)
            comp, bbox_voc = paste_rgba_on(bg, fg_scaled, (x, y))

            # 4) Albumentations transforms with bbox tracking
            comp_np = np.array(comp.convert('RGB'))
            x1, y1, x2, y2 = bbox_voc
            bboxes = [(x1, y1, x2, y2)]
            class_labels = [pid]

            transformed = aug(image=comp_np, bboxes=bboxes, class_labels=class_labels)
            img_aug = transformed['image']
            bboxes_aug = transformed['bboxes']
            class_labels_aug = transformed['class_labels']

            if not bboxes_aug:
                # too aggressive transform removed visible bbox; skip
                continue

            # 5) Write image
            file_stem = hashlib.sha1(f"{pid}-{src_path.name}-{random.random()}".encode()).hexdigest()[:16]
            img_name = f"{file_stem}.jpg"
            img_path = out_images / img_name
            # Additional save-time JPEG artifacts (already in aug but reinforce)
            cv2.imwrite(str(img_path), cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(70, 92)])

            h, w = img_aug.shape[:2]
            images_meta.append({
                'id': image_id,
                'file_name': img_name,
                'width': int(w),
                'height': int(h),
            })

            # 6) Convert bboxes (usually single) and record annotations
            for (bx1, by1, bx2, by2), cls_lbl in zip(bboxes_aug, class_labels_aug):
                coco_bbox = voc_to_coco_bbox((bx1, by1, bx2, by2))
                area = coco_bbox[2] * coco_bbox[3]
                anns.append(CocoAnn(image_id=image_id, category_id=cat_id_of[cls_lbl], bbox=coco_bbox, area=area, iscrowd=0))
                ann_id += 1

            image_id += 1

    # 7) Write COCO JSON
    out_json = args.out / 'annotations.json'
    coco = {
        'images': images_meta,
        'annotations': [
            {
                'id': i+1,
                'image_id': ann.image_id,
                'category_id': ann.category_id,
                'bbox': list(map(float, ann.bbox)),
                'area': float(ann.area),
                'iscrowd': ann.iscrowd,
            } for i, ann in enumerate(anns)
        ],
        'categories': categories,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(coco, f)

    print(f"\nWrote {len(images_meta)} images and {len(anns)} annotations to {args.out}")


if __name__ == '__main__':
    main()
