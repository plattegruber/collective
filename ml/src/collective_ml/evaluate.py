"""
End-to-end evaluation of a release against a real collection.

Photos pulled from the app become the gallery (prototypes are built exactly as the browser
builds them), held-out composites of the same photos on rooms become the queries, and
unrelated paintings become distractors that must stay unrecognised. The report includes a
threshold sweep so the release ships with calibrated match thresholds.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from collective_ml.coco import CocoDataset
from collective_ml.composites import ARTWORK_CATEGORY, CompositeOptions, compose
from collective_ml.files import list_images, scan_raw_dir
from collective_ml.geometry import Box, coco_to_voc, iou
from collective_ml.runtime import OnnxDetector, OnnxEmbedder, Region

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalOptions:
    per_piece: int = 40
    distractor_images: int = 150
    seed: int = 4242
    size: int = 640
    # How many of the strongest regions in a distractor image are embedded.
    regions_per_distractor: int = 2


def _load(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def union_box(boxes: list[Box]) -> Box:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def locate_artwork(
    detector: OnnxDetector,
    image: Image.Image,
    threshold: float,
    min_region_fraction: float = 0.02,
    min_coverage: float = 0.2,
) -> Box | None:
    """
    The browser's rule for an uploaded photo (`chooseArtworkRegion` in the app): the union
    of every confident, non-trivial detection, when that union covers a meaningful part of
    the photo; otherwise the whole photo. The union keeps multi-panel works together.
    """
    area = image.width * image.height
    boxes = [
        r.box
        for r in detector.detect(image, threshold)
        if (r.box[2] - r.box[0]) * (r.box[3] - r.box[1]) >= min_region_fraction * area
    ]
    if not boxes:
        return None
    union = union_box(boxes)
    if (union[2] - union[0]) * (union[3] - union[1]) < min_coverage * area:
        return None
    return union


def crop_to_artwork(detector: OnnxDetector, image: Image.Image, threshold: float) -> Image.Image:
    box = locate_artwork(detector, image, threshold)
    if box is None:
        return image
    return image.crop(tuple(round(v) for v in box))


def build_prototypes(
    detector: OnnxDetector, embedder: OnnxEmbedder, raw_dir: Path, threshold: float
) -> dict[str, list[np.ndarray]]:
    prototypes: dict[str, list[np.ndarray]] = {}
    for artwork_id, photos in scan_raw_dir(raw_dir).items():
        for photo in photos:
            image = _load(photo)
            box = locate_artwork(detector, image, threshold)
            prototypes.setdefault(artwork_id, []).append(embedder.embed(image, box))
    return prototypes


def similarities(prototypes: dict[str, list[np.ndarray]], vector: np.ndarray) -> dict[str, float]:
    return {
        artwork_id: max(float(p @ vector) for p in vectors)
        for artwork_id, vectors in prototypes.items()
    }


def sweep_thresholds(
    genuine: np.ndarray,
    correct: np.ndarray,
    impostors: np.ndarray,
    far_targets: tuple[float, float] = (0.02, 0.005),
    grid: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    For each threshold: true-positive rate (genuine similarity above it *and* the right
    piece) and false-accept rate on impostors. Suggests the lowest thresholds that keep
    false accepts at or below each target; falls back to the strictest grid value.
    """
    grid = np.round(np.arange(0.30, 0.96, 0.01), 2) if grid is None else grid
    rows = []
    for t in grid:
        tpr = float(np.mean((genuine >= t) & correct)) if genuine.size else 0.0
        far = float(np.mean(impostors >= t)) if impostors.size else 0.0
        rows.append({"threshold": float(t), "tpr": tpr, "far": far})
    suggestions = []
    for target in far_targets:
        chosen = next((row for row in rows if row["far"] <= target), rows[-1])
        suggestions.append(chosen["threshold"])
    return {"rows": rows, "match": suggestions[0], "strong": max(suggestions)}


def _best_region(regions: list[Region], truth: Box) -> tuple[Region | None, float]:
    best, best_iou = None, 0.0
    for region in regions:
        overlap = iou(region.box, truth)
        if overlap > best_iou:
            best, best_iou = region, overlap
    return best, best_iou


def evaluate_release(
    release_dir: Path,
    raw_dir: Path,
    foregrounds_dir: Path,
    rooms_dir: Path | None,
    eval_dir: Path,
    options: EvalOptions | None = None,
) -> dict[str, Any]:
    options = options or EvalOptions()
    manifest = json.loads((release_dir / "release.json").read_text())
    config = manifest["config"]
    detector = OnnxDetector(release_dir / "detector.onnx", int(manifest["detectorInputSize"]))
    embedder = OnnxEmbedder(
        release_dir / "embedder.onnx",
        int(manifest["embedderInputSize"]),
        tuple(config["mean"]),
        tuple(config["std"]),
        config["pooling"],
        float(config["cropMargin"]),
    )
    detect_threshold = float(config["thresholds"]["detect"])

    pieces = scan_raw_dir(raw_dir)
    if not pieces:
        raise SystemExit(f"No photos under {raw_dir}. Run `collective-ml pull` first.")
    artwork_ids = sorted(pieces)
    categories = {index: artwork_id for index, artwork_id in enumerate(artwork_ids, start=1)}
    category_of = {artwork_id: index for index, artwork_id in categories.items()}

    log.info("Building prototypes for %d pieces", len(pieces))
    prototypes = build_prototypes(detector, embedder, raw_dir, detect_threshold)

    log.info("Composing %d held-out views per piece", options.per_piece)
    crops_dir = eval_dir / "crops"
    query_fg: dict[Path, int] = {}
    for artwork_id, photos in pieces.items():
        for photo in photos:
            crop = crop_to_artwork(detector, _load(photo), detect_threshold)
            target = crops_dir / artwork_id / f"{photo.stem}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            crop.save(target, quality=92)
            query_fg[target] = category_of[artwork_id]
    queries = compose(
        query_fg,
        eval_dir / "queries",
        CompositeOptions(
            size=options.size,
            count=options.per_piece * len(pieces),
            seed=options.seed,
            max_pieces=1,
            negative_prob=0.0,
            min_scale=0.2,
            max_scale=0.85,
            frame_prob=0.0,
        ),
        rooms_dir,
        categories,
    )
    distractor_paths = list_images(foregrounds_dir)
    distractors: CocoDataset | None = None
    if distractor_paths:
        distractors = compose(
            dict.fromkeys(distractor_paths, ARTWORK_CATEGORY),
            eval_dir / "distractors",
            CompositeOptions(
                size=options.size,
                count=options.distractor_images,
                seed=options.seed + 1,
                max_pieces=2,
                negative_prob=0.1,
            ),
            rooms_dir,
        )

    genuine: list[float] = []
    correct: list[bool] = []
    impostor_known: list[float] = []
    ious: list[float] = []
    detected = total = 0
    for image_meta in queries.images:
        image = _load(eval_dir / "queries" / "images" / image_meta.file_name)
        regions = detector.detect(image, detect_threshold)
        for ann in queries.annotations_for(image_meta.id):
            total += 1
            truth = coco_to_voc(ann.bbox)
            region, overlap = _best_region(regions, truth)
            ious.append(overlap)
            if region is None or overlap < 0.5:
                continue
            detected += 1
            scores = similarities(prototypes, embedder.embed(image, region.box))
            truth_id = categories[ann.category_id]
            predicted = max(scores, key=lambda k: scores[k])
            genuine.append(scores[truth_id])
            correct.append(predicted == truth_id)
            others = [v for k, v in scores.items() if k != truth_id]
            if others:
                impostor_known.append(max(others))

    impostor_unknown: list[float] = []
    distractor_regions = 0
    if distractors is not None:
        for image_meta in distractors.images:
            image = _load(eval_dir / "distractors" / "images" / image_meta.file_name)
            for region in detector.detect(image, detect_threshold)[
                : options.regions_per_distractor
            ]:
                distractor_regions += 1
                scores = similarities(prototypes, embedder.embed(image, region.box))
                impostor_unknown.append(max(scores.values()))

    genuine_arr = np.asarray(genuine, dtype=np.float32)
    correct_arr = np.asarray(correct, dtype=bool)
    unknown_arr = np.asarray(impostor_unknown, dtype=np.float32)
    known_arr = np.asarray(impostor_known, dtype=np.float32)
    impostors = np.concatenate([unknown_arr, known_arr]) if known_arr.size else unknown_arr
    sweep = sweep_thresholds(genuine_arr, correct_arr, impostors)

    def pct(values: np.ndarray, q: float) -> float:
        return float(np.percentile(values, q)) if values.size else float("nan")

    report: dict[str, Any] = {
        "pieces": len(pieces),
        "prototypes": sum(len(v) for v in prototypes.values()),
        "queries": total,
        "detectionRecall": detected / total if total else 0.0,
        "meanIoU": float(np.mean(ious)) if ious else 0.0,
        "top1AmongDetected": float(correct_arr.mean()) if correct_arr.size else 0.0,
        "top1Overall": float(correct_arr.sum() / total) if total else 0.0,
        "genuine": {
            "p5": pct(genuine_arr, 5),
            "p50": pct(genuine_arr, 50),
            "mean": float(genuine_arr.mean()) if genuine_arr.size else 0.0,
        },
        "impostorKnown": {
            "p95": pct(known_arr, 95),
            "max": float(known_arr.max()) if known_arr.size else 0.0,
        },
        "impostorUnknown": {
            "regions": distractor_regions,
            "p95": pct(unknown_arr, 95),
            "max": float(unknown_arr.max()) if unknown_arr.size else 0.0,
        },
        "suggested": {"match": sweep["match"], "strong": sweep["strong"]},
        "sweep": sweep["rows"],
        "options": options.__dict__,
    }
    (eval_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def format_report(report: dict[str, Any]) -> str:
    g, k, u = report["genuine"], report["impostorKnown"], report["impostorUnknown"]
    at = next((r for r in report["sweep"] if r["threshold"] == report["suggested"]["match"]), None)
    lines = [
        f"pieces {report['pieces']} · prototypes {report['prototypes']} · "
        f"queries {report['queries']}",
        f"detection recall@0.5 {report['detectionRecall']:.3f} · mean IoU {report['meanIoU']:.3f}",
        f"top-1 among detected {report['top1AmongDetected']:.3f} · "
        f"overall {report['top1Overall']:.3f}",
        f"genuine similarity p5 {g['p5']:.3f} · median {g['p50']:.3f}",
        f"impostor (other piece) p95 {k['p95']:.3f} max {k['max']:.3f}",
        f"impostor (unknown art, {u['regions']} regions) p95 {u['p95']:.3f} max {u['max']:.3f}",
        f"suggested thresholds: match {report['suggested']['match']:.2f} · "
        f"strong {report['suggested']['strong']:.2f}"
        + (f" (TPR {at['tpr']:.3f}, FAR {at['far']:.3f} at match)" if at else ""),
    ]
    return "\n".join(lines)
