"""
Composite artworks onto walls to build detection datasets.

Two uses share the code: teaching the class-agnostic detector ("there is a picture here")
from hundreds of unrelated artworks, and building held-out views of a specific collection
to evaluate the whole recognizer. Pieces get a synthetic mat and frame most of the time;
the augmentations model phones in living rooms: perspective, lighting, blur, glare,
occlusion, compression, and a little sensor noise.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

# Albumentations phones home for version checks on import; builds should never need the network.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from collective_ml.backgrounds import BackgroundSource
from collective_ml.coco import CocoDataset
from collective_ml.geometry import Box, clip_box

log = logging.getLogger(__name__)

ARTWORK_CATEGORY = 1


@dataclass(frozen=True)
class CompositeOptions:
    size: int = 640
    count: int = 3000
    seed: int = 7
    min_scale: float = 0.12
    max_scale: float = 0.75
    max_pieces: int = 3
    # Fraction of images that are a bare background, so the detector learns silence too.
    negative_prob: float = 0.08
    real_background_prob: float = 0.6
    # Chance a piece gets a synthetic mat/frame; 0 when the foregrounds already show their frames.
    frame_prob: float = 1.0


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
    """Perspective, lighting, blur, glare, occlusion, compression: realistic home viewing."""
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
        # Albumentations 2.x defaults to std 0.2-0.44 (of 1.0), which buries the image;
        # phone sensors add roughly 1-5%.
        _maybe("GaussNoise", std_range=(0.01, 0.05), p=0.4),
        _maybe("MotionBlur", blur_limit=5, p=0.3),
        _maybe("GaussianBlur", blur_limit=3, p=0.2),
        _maybe(
            "RandomSunFlare",
            src_radius=40,
            num_flare_circles_range=(2, 4),
            flare_roi=(0.0, 0.0, 1.0, 0.5),
            p=0.1,
        ),
        _maybe("CoarseDropout", fill="random_uniform", p=0.35),
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


FRAME_COLOURS = (
    (20, 20, 20),
    (245, 243, 238),
    (120, 80, 40),
    (160, 120, 60),
    (200, 170, 90),
    (90, 90, 95),
    (60, 40, 30),
    (230, 200, 120),
)


def load_foreground(path: Path, max_edge: int = 512) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image.thumbnail((max_edge, max_edge))
    return image


def add_frame(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Adds a mat and/or a frame most of the time; sometimes a bare canvas with a dark edge."""
    short = min(image.size)
    if rng.random() < 0.15:
        edge = tuple(int(v * 0.6) for v in np.asarray(image).reshape(-1, 3).mean(0))
        return ImageOps.expand(image, border=max(1, short // 120), fill=edge)  # type: ignore[arg-type]
    if rng.random() < 0.6:
        mat = int(short * rng.uniform(0.02, 0.14))
        if rng.random() < 0.8:
            colour = (int(rng.integers(225, 255)),) * 3
        else:
            colour = (
                int(rng.integers(200, 240)),
                int(rng.integers(190, 230)),
                int(rng.integers(170, 215)),
            )
        image = ImageOps.expand(image, border=mat, fill=colour)
    frame = max(2, int(short * rng.uniform(0.012, 0.07)))
    colour = FRAME_COLOURS[int(rng.integers(0, len(FRAME_COLOURS)))]
    framed = np.asarray(ImageOps.expand(image, border=frame, fill=colour)).astype(np.float32)
    # A simple bevel: lighter top/left edge, darker bottom/right edge.
    f = frame
    framed[f - 1 : f, f - 1 : -f + 1] *= 1.25
    framed[-f : -f + 1, f - 1 : -f + 1] *= 0.8
    framed[f - 1 : -f + 1, f - 1 : f] *= 1.15
    framed[f - 1 : -f + 1, -f : -f + 1] *= 0.85
    return Image.fromarray(np.clip(framed, 0, 255).astype(np.uint8))


def overlap_ratio(a: Box, b: Box) -> float:
    """Intersection over the smaller box, so a piece nested inside another counts as overlap."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def place_pieces(
    background: Image.Image,
    pieces: list[Image.Image],
    rng: np.random.Generator,
    min_scale: float,
    max_scale: float,
) -> tuple[Image.Image, list[Box]]:
    """Pastes each piece at a random non-overlapping spot; pieces that do not fit are skipped."""
    canvas = background.copy()
    size = background.width
    boxes: list[Box] = []
    for piece in pieces:
        for _ in range(12):
            target_short = int(size * rng.uniform(min_scale, max_scale))
            scale = target_short / min(piece.size)
            width, height = int(piece.width * scale), int(piece.height * scale)
            if width > size * 0.95 or height > size * 0.95 or width < 8 or height < 8:
                continue
            x = int(rng.integers(0, size - width + 1))
            y = int(rng.integers(0, size - height + 1))
            box = clip_box((x, y, x + width, y + height), size, size)
            if all(overlap_ratio(box, other) < 0.03 for other in boxes):
                canvas.paste(piece.resize((width, height), Image.Resampling.LANCZOS), (x, y))
                boxes.append(box)
                break
    return canvas, boxes


def compose(
    foregrounds: dict[Path, int],
    out_dir: Path,
    options: CompositeOptions,
    rooms_dir: Path | None = None,
    categories: dict[int, str] | None = None,
) -> CocoDataset:
    """
    Write `<out_dir>/images/*.jpg` and `<out_dir>/annotations.json`.

    `foregrounds` maps each image to its category id; pass everything as `ARTWORK_CATEGORY`
    for the detector, or one category per artwork to evaluate recognition.
    """
    if not foregrounds:
        raise ValueError("No foreground images to compose.")
    paths = sorted(foregrounds)
    rng = np.random.default_rng(options.seed)
    rooms = BackgroundSource(options.size, rng, rooms_dir)
    plain = BackgroundSource(options.size, rng, None)
    augment = build_augmentations(options.size)
    dataset = CocoDataset(categories=categories or {ARTWORK_CATEGORY: "artwork"})
    cache: dict[Path, Image.Image] = {}

    def foreground(path: Path) -> Image.Image:
        if path not in cache:
            cache[path] = load_foreground(path)
        return cache[path]

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for stale in images_dir.glob("*.jpg"):
        stale.unlink()

    produced = attempts = 0
    with tqdm(total=options.count, desc="compose", leave=False) as bar:
        while produced < options.count and attempts < options.count * 3:
            attempts += 1
            use_room = rooms_dir is not None and rng.random() < options.real_background_prob
            background = rooms.next() if use_room else plain.next()
            if rng.random() < options.negative_prob:
                composite, boxes, labels = background, [], []
            else:
                n = int(rng.integers(1, options.max_pieces + 1))
                chosen = [paths[int(rng.integers(0, len(paths)))] for _ in range(n)]
                pieces = [
                    add_frame(foreground(path), rng)
                    if rng.random() < options.frame_prob
                    else foreground(path)
                    for path in chosen
                ]
                composite, boxes = place_pieces(
                    background, pieces, rng, options.min_scale, options.max_scale
                )
                labels = [foregrounds[path] for path in chosen[: len(boxes)]]
                if not boxes:
                    continue
            result = augment(
                image=np.asarray(composite.convert("RGB")), bboxes=boxes, class_labels=labels
            )
            if boxes and not result["bboxes"]:
                continue
            image = result["image"]
            stem = hashlib.sha1(f"{produced}:{options.seed}".encode()).hexdigest()[:16]
            cv2.imwrite(
                str(images_dir / f"{stem}.jpg"),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), int(rng.integers(70, 93))],
            )
            height, width = image.shape[:2]
            image_id = dataset.add_image(f"{stem}.jpg", int(width), int(height))
            for voc_box, label in zip(result["bboxes"], result["class_labels"], strict=True):
                dataset.add_box(image_id, int(label), tuple(float(v) for v in voc_box[:4]))  # type: ignore[arg-type]
            produced += 1
            bar.update(1)
    dataset.save(out_dir / "annotations.json")
    log.info(
        "Composed %d images with %d boxes into %s", produced, len(dataset.annotations), out_dir
    )
    return dataset
