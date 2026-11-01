"""Synthesise a detection dataset by compositing artwork photos onto wall backgrounds."""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

# Albumentations phones home for version checks on import; training should never need the network.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from collective_ml.backgrounds import BackgroundSource
from collective_ml.coco import CocoDataset
from collective_ml.geometry import Box, clip_box
from collective_ml.labels import build_label_map, class_index_of, save_label_map, scan_raw_dir

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SynthOptions:
    img_size: int = 640
    samples_per_piece: int = 160
    seed: int = 1337
    backgrounds_dir: Path | None = None
    min_scale: float = 0.35
    max_scale: float = 0.92


def _maybe(name: str, **kwargs: object) -> A.BasicTransform | None:
    """Instantiate an Albumentations transform if this version supports it."""
    cls = getattr(A, name, None)
    if cls is None:
        log.debug("Albumentations %s lacks %s; skipping.", A.__version__, name)
        return None
    try:
        return cls(**kwargs)
    except TypeError as error:
        log.debug("Skipping %s: %s", name, error)
        return None


def build_augmentations(img_size: int) -> A.Compose:
    """Perspective, lighting, blur, glare, occlusion, compression - realistic home viewing."""
    candidates = [
        _maybe(
            "Affine",
            scale=(0.9, 1.12),
            translate_percent=(0.0, 0.035),
            rotate=(-6, 6),
            shear=(-3, 3),
            p=0.8,
        ),
        _maybe("Perspective", scale=(0.02, 0.06), p=0.6),
        _maybe("RandomBrightnessContrast", brightness_limit=0.15, contrast_limit=0.2, p=0.8),
        _maybe(
            "HueSaturationValue", hue_shift_limit=10, sat_shift_limit=12, val_shift_limit=10, p=0.5
        ),
        _maybe("GaussNoise", p=0.4),
        _maybe("MotionBlur", blur_limit=5, p=0.3),
        _maybe("GaussianBlur", blur_limit=3, p=0.2),
        _maybe("RandomSunFlare", src_radius=80, flare_roi=(0.0, 0.0, 1.0, 0.5), p=0.25),
        _maybe("CoarseDropout", p=0.35),
        _maybe("ImageCompression", quality_range=(55, 92), p=0.7),
        _maybe("LongestMaxSize", max_size=img_size, interpolation=cv2.INTER_AREA),
        _maybe(
            "PadIfNeeded", min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT
        ),
    ]
    transforms = [t for t in candidates if t is not None]
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_area=16,
            min_visibility=0.3,
            clip=True,
            check_each_transform=False,
        ),
    )


def _load_rgba(path: Path) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    alpha = np.asarray(image.getchannel("A"))
    if alpha.mean() > 250:
        # Opaque photo: soften corners so composites don't look like hard rectangles every time.
        w, h = image.size
        radius = max(2, min(w, h) // 50)
        mask = Image.new("L", (w - 2 * radius, h - 2 * radius), 255)
        mask = ImageOps.expand(mask, border=radius, fill=255).resize(
            (w, h), Image.Resampling.LANCZOS
        )
        image.putalpha(mask)
    return image


def _scaled_foreground(
    fg: Image.Image, canvas: int, rng: np.random.Generator, lo: float, hi: float
) -> Image.Image:
    w, h = fg.size
    short = min(w, h)
    target_short = max(8, int(canvas * float(rng.uniform(lo, hi))))
    scale = target_short / short
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    if new_w > canvas or new_h > canvas:
        fit = min(canvas / new_w, canvas / new_h)
        new_w, new_h = max(1, int(new_w * fit)), max(1, int(new_h * fit))
    return fg.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _paste(bg: Image.Image, fg: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, Box]:
    pad = int(0.03 * bg.width)
    x = int(rng.integers(pad, max(pad + 1, bg.width - fg.width - pad)))
    y = int(rng.integers(pad, max(pad + 1, bg.height - fg.height - pad)))
    x = min(max(0, x), bg.width - fg.width)
    y = min(max(0, y), bg.height - fg.height)
    out = bg.copy()
    out.paste(fg, (x, y), fg)
    return out, clip_box((x, y, x + fg.width, y + fg.height), bg.width, bg.height)


def synthesize(
    raw_dir: Path, out_dir: Path, labels_path: Path, options: SynthOptions
) -> CocoDataset:
    """
    Build `<out_dir>/images/*.jpg` + `<out_dir>/annotations.json` and write `labels.json`.

    Raises `ValueError` when no artwork folder contains images.
    """
    pieces = scan_raw_dir(raw_dir)
    if not pieces:
        raise ValueError(
            f"No artwork photos found under {raw_dir}. Run `collective-ml pull` first."
        )

    label_map = build_label_map(pieces.keys())
    save_label_map(label_map, labels_path)
    class_of = class_index_of(label_map)

    rng = np.random.default_rng(options.seed)
    backgrounds = BackgroundSource(options.img_size, rng, options.backgrounds_dir)
    augment = build_augmentations(options.img_size)
    dataset = CocoDataset(categories={index: artwork_id for artwork_id, index in class_of.items()})

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for stale in images_dir.glob("*.jpg"):
        stale.unlink()

    for artwork_id, sources in pieces.items():
        foregrounds = [_load_rgba(path) for path in sources]
        produced = 0
        attempts = 0
        with tqdm(total=options.samples_per_piece, desc=artwork_id[:12], leave=False) as bar:
            while produced < options.samples_per_piece and attempts < options.samples_per_piece * 3:
                attempts += 1
                fg = foregrounds[int(rng.integers(0, len(foregrounds)))]
                scaled = _scaled_foreground(
                    fg, options.img_size, rng, options.min_scale, options.max_scale
                )
                composite, box = _paste(backgrounds.next(), scaled, rng)
                result = augment(
                    image=np.asarray(composite.convert("RGB")),
                    bboxes=[box],
                    class_labels=[artwork_id],
                )
                if not result["bboxes"]:
                    continue
                image = result["image"]
                stem = hashlib.sha1(f"{artwork_id}:{produced}:{options.seed}".encode()).hexdigest()[
                    :16
                ]
                file_name = f"{stem}.jpg"
                quality = int(rng.integers(70, 93))
                cv2.imwrite(
                    str(images_dir / file_name),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), quality],
                )
                h, w = image.shape[:2]
                image_id = dataset.add_image(file_name, int(w), int(h))
                for voc_box, label in zip(result["bboxes"], result["class_labels"], strict=True):
                    dataset.add_box(
                        image_id, class_of[str(label)], tuple(float(v) for v in voc_box[:4])
                    )  # type: ignore[arg-type]
                produced += 1
                bar.update(1)
        if produced < options.samples_per_piece:
            log.warning(
                "%s: produced %d/%d samples.", artwork_id, produced, options.samples_per_piece
            )

    dataset.save(out_dir / "annotations.json")
    log.info(
        "Wrote %d images / %d boxes to %s", len(dataset.images), len(dataset.annotations), out_dir
    )
    return dataset
