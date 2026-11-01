"""Assemble and persist a minimal COCO-style dataset manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from collective_ml.geometry import Box, voc_to_coco


@dataclass(frozen=True)
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class CocoAnnotation:
    id: int
    image_id: int
    category_id: int
    bbox: Box  # x, y, w, h
    area: float
    iscrowd: int = 0


@dataclass
class CocoDataset:
    """In-memory manifest; call `add_image` / `add_box` while synthesising, then `save`."""

    categories: dict[int, str]
    images: list[CocoImage] = field(default_factory=list)
    annotations: list[CocoAnnotation] = field(default_factory=list)

    def add_image(self, file_name: str, width: int, height: int) -> int:
        image_id = len(self.images) + 1
        self.images.append(CocoImage(image_id, file_name, width, height))
        return image_id

    def add_box(self, image_id: int, category_id: int, voc_box: Box) -> None:
        if category_id not in self.categories:
            raise KeyError(f"Unknown category {category_id}")
        bbox = voc_to_coco(voc_box)
        self.annotations.append(
            CocoAnnotation(
                id=len(self.annotations) + 1,
                image_id=image_id,
                category_id=category_id,
                bbox=bbox,
                area=bbox[2] * bbox[3],
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "images": [vars(image) for image in self.images],
            "annotations": [{**vars(ann), "bbox": list(ann.bbox)} for ann in self.annotations],
            "categories": [
                {"id": cid, "name": name} for cid, name in sorted(self.categories.items())
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: Path) -> CocoDataset:
        data = json.loads(path.read_text())
        dataset = cls(categories={int(c["id"]): str(c["name"]) for c in data["categories"]})
        dataset.images = [
            CocoImage(int(i["id"]), str(i["file_name"]), int(i["width"]), int(i["height"]))
            for i in data["images"]
        ]
        dataset.annotations = [
            CocoAnnotation(
                id=int(a["id"]),
                image_id=int(a["image_id"]),
                category_id=int(a["category_id"]),
                bbox=tuple(float(v) for v in a["bbox"]),  # type: ignore[arg-type]
                area=float(a["area"]),
                iscrowd=int(a.get("iscrowd", 0)),
            )
            for a in data["annotations"]
        ]
        return dataset

    def annotations_for(self, image_id: int) -> list[CocoAnnotation]:
        return [ann for ann in self.annotations if ann.image_id == image_id]
