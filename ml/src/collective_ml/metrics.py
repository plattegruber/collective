"""Detection metrics (VOC-style mAP@0.5) with no framework dependencies."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from collective_ml.geometry import Box, iou


@dataclass(frozen=True)
class Detection:
    image_id: int
    label: int
    score: float
    box: Box


@dataclass(frozen=True)
class GroundTruth:
    image_id: int
    label: int
    box: Box


@dataclass
class MapResult:
    map50: float
    per_class: dict[int, float] = field(default_factory=dict)


def average_precision(recalls: list[float], precisions: list[float]) -> float:
    """All-point interpolated AP over a monotone recall curve."""
    if not recalls:
        return 0.0
    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def mean_average_precision(
    detections: list[Detection],
    ground_truths: list[GroundTruth],
    iou_threshold: float = 0.5,
) -> MapResult:
    """Compute mAP@`iou_threshold` across every class present in the ground truth."""
    gts_by_class: dict[int, dict[int, list[Box]]] = defaultdict(lambda: defaultdict(list))
    for gt in ground_truths:
        gts_by_class[gt.label][gt.image_id].append(gt.box)
    dets_by_class: dict[int, list[Detection]] = defaultdict(list)
    for det in detections:
        dets_by_class[det.label].append(det)

    per_class: dict[int, float] = {}
    for label, gts in gts_by_class.items():
        total = sum(len(boxes) for boxes in gts.values())
        dets = sorted(dets_by_class.get(label, []), key=lambda d: -d.score)
        matched: dict[int, list[bool]] = {img: [False] * len(boxes) for img, boxes in gts.items()}
        tp = fp = 0
        recalls: list[float] = []
        precisions: list[float] = []
        for det in dets:
            candidates = gts.get(det.image_id, [])
            best_iou, best_index = 0.0, -1
            for index, box in enumerate(candidates):
                overlap = iou(det.box, box)
                if overlap > best_iou:
                    best_iou, best_index = overlap, index
            if (
                best_index >= 0
                and best_iou >= iou_threshold
                and not matched[det.image_id][best_index]
            ):
                matched[det.image_id][best_index] = True
                tp += 1
            else:
                fp += 1
            recalls.append(tp / total if total else 0.0)
            precisions.append(tp / (tp + fp))
        per_class[label] = average_precision(recalls, precisions)

    map50 = sum(per_class.values()) / len(per_class) if per_class else 0.0
    return MapResult(map50=map50, per_class=per_class)
