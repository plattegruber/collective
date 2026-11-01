"""Pure box arithmetic shared by synthesis, training, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Letterbox:
    """How an image was scaled and padded into a square canvas."""

    scale: float
    left: int
    top: int
    width: int
    height: int

    def to_canvas(self, box: Box) -> Box:
        """Map a box in source pixels to canvas pixels."""
        x1, y1, x2, y2 = box
        return (
            x1 * self.scale + self.left,
            y1 * self.scale + self.top,
            x2 * self.scale + self.left,
            y2 * self.scale + self.top,
        )

    def to_source(self, box: Box) -> Box:
        """Map a box in canvas pixels back to source pixels."""
        x1, y1, x2, y2 = box
        return (
            (x1 - self.left) / self.scale,
            (y1 - self.top) / self.scale,
            (x2 - self.left) / self.scale,
            (y2 - self.top) / self.scale,
        )


def letterbox(width: int, height: int, size: int) -> Letterbox:
    """Fit a `width x height` image into a `size x size` square, centred, aspect preserved."""
    scale = min(size / width, size / height)
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    return Letterbox(
        scale=scale,
        left=(size - new_w) // 2,
        top=(size - new_h) // 2,
        width=new_w,
        height=new_h,
    )


def voc_to_coco(box: Box) -> Box:
    """(x1, y1, x2, y2) → (x, y, w, h)."""
    x1, y1, x2, y2 = box
    return (float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1)))


def coco_to_voc(box: Box) -> Box:
    """(x, y, w, h) → (x1, y1, x2, y2)."""
    x, y, w, h = box
    return (float(x), float(y), float(x + w), float(y + h))


def clip_box(box: Box, width: float, height: float) -> Box:
    """Clamp a VOC box to the image bounds."""
    x1, y1, x2, y2 = box
    return (
        min(max(x1, 0.0), width),
        min(max(y1, 0.0), height),
        min(max(x2, 0.0), width),
        min(max(y2, 0.0), height),
    )


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two VOC boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
