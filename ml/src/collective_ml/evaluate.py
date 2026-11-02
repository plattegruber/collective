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
    genuine_margin: np.ndarray | None = None,
    impostor_margin: np.ndarray | None = None,
    margins: tuple[float, ...] = (0.0, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15),
) -> dict[str, Any]:
    """
    A region is accepted when its best similarity clears a threshold *and* beats the
    runner-up piece by a margin. For every margin, find the lowest threshold whose
    false-accept rate on impostors is at or below each target; keep the margin that
    recognises the most genuine views at the first target. Without margin arrays the
    sweep is purely on similarity.
    """
    # Never let the absolute floor drop below 0.45: the margin may do the work on a large
    # collection, but a weak absolute similarity should still never pass.
    grid = np.round(np.arange(0.45, 0.96, 0.01), 2)
    if genuine_margin is None or impostor_margin is None:
        margins = (0.0,)
        genuine_margin = np.full(genuine.shape, np.inf)
        impostor_margin = np.full(impostors.shape, np.inf)

    def rate(values: np.ndarray, mask: np.ndarray) -> float:
        return float(np.mean(mask)) if values.size else 0.0

    best: dict[str, Any] | None = None
    by_margin: list[dict[str, float]] = []
    for margin in margins:
        rows = [
            {
                "threshold": float(t),
                "tpr": rate(genuine, (genuine >= t) & (genuine_margin >= margin) & correct),
                "far": rate(impostors, (impostors >= t) & (impostor_margin >= margin)),
            }
            for t in grid
        ]
        picks = [next((r for r in rows if r["far"] <= target), rows[-1]) for target in far_targets]
        candidate = {
            "margin": float(margin),
            "rows": rows,
            "match": picks[0]["threshold"],
            "strong": max(p["threshold"] for p in picks),
            "tpr": picks[0]["tpr"],
        }
        by_margin.append(
            {"margin": float(margin), "match": candidate["match"], "tpr": candidate["tpr"]}
        )
        if best is None or candidate["tpr"] > best["tpr"]:
            best = candidate
    assert best is not None
    best["byMargin"] = by_margin
    return best


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
    genuine_margin: list[float] = []
    correct: list[bool] = []
    impostor_known: list[float] = []
    impostor_known_margin: list[float] = []
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
            ranked = sorted(scores.values(), reverse=True)
            others = [v for k, v in scores.items() if k != truth_id]
            genuine.append(scores[truth_id])
            genuine_margin.append(scores[truth_id] - max(others) if others else float("inf"))
            correct.append(predicted == truth_id)
            if predicted != truth_id:
                # A wrong piece won: that is what a false accept would look like.
                impostor_known.append(ranked[0])
                impostor_known_margin.append(ranked[0] - ranked[1])

    impostor_unknown: list[float] = []
    impostor_unknown_margin: list[float] = []
    distractor_regions = 0
    if distractors is not None:
        for image_meta in distractors.images:
            image = _load(eval_dir / "distractors" / "images" / image_meta.file_name)
            for region in detector.detect(image, detect_threshold)[
                : options.regions_per_distractor
            ]:
                distractor_regions += 1
                scores = similarities(prototypes, embedder.embed(image, region.box))
                ranked = sorted(scores.values(), reverse=True)
                impostor_unknown.append(ranked[0])
                impostor_unknown_margin.append(
                    ranked[0] - ranked[1] if len(ranked) > 1 else float("inf")
                )

    genuine_arr = np.asarray(genuine, dtype=np.float32)
    correct_arr = np.asarray(correct, dtype=bool)
    unknown_arr = np.asarray(impostor_unknown, dtype=np.float32)
    known_arr = np.asarray(impostor_known, dtype=np.float32)
    impostors = np.concatenate([unknown_arr, known_arr])
    impostor_margins = np.concatenate(
        [
            np.asarray(impostor_unknown_margin, dtype=np.float32),
            np.asarray(impostor_known_margin, dtype=np.float32),
        ]
    )
    sweep = sweep_thresholds(
        genuine_arr,
        correct_arr,
        impostors,
        genuine_margin=np.asarray(genuine_margin, dtype=np.float32),
        impostor_margin=impostor_margins,
    )

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
        "suggested": {
            "match": sweep["match"],
            "strong": sweep["strong"],
            "margin": sweep["margin"],
        },
        "sweep": sweep["rows"],
        "byMargin": sweep["byMargin"],
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
        f"strong {report['suggested']['strong']:.2f} · margin {report['suggested']['margin']:.2f}"
        + (f" (TPR {at['tpr']:.3f}, FAR {at['far']:.3f} at match)" if at else ""),
        "by margin: "
        + ", ".join(
            f"{row['margin']:.2f}→{row['match']:.2f}/{row['tpr']:.3f}"
            for row in report.get("byMargin", [])
        ),
    ]
    return "\n".join(lines)
