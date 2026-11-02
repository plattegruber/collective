from pathlib import Path

import numpy as np
from PIL import Image

from collective_ml.coco import CocoDataset
from collective_ml.composites import (
    ARTWORK_CATEGORY,
    CompositeOptions,
    add_frame,
    compose,
    overlap_ratio,
    place_pieces,
)


def _painting(path: Path, colour: tuple[int, int, int], size=(90, 60)) -> Path:
    Image.new("RGB", size, colour).save(path)
    return path


def test_add_frame_grows_the_piece_and_keeps_it_rgb():
    rng = np.random.default_rng(1)
    piece = Image.new("RGB", (80, 50), (200, 30, 30))
    framed = add_frame(piece, rng)
    assert framed.mode == "RGB"
    assert framed.width > 80 and framed.height > 50


def test_overlap_ratio_counts_nesting_as_overlap():
    outer = (0.0, 0.0, 100.0, 100.0)
    inner = (10.0, 10.0, 30.0, 30.0)
    assert overlap_ratio(outer, inner) == 1.0
    assert overlap_ratio(outer, (200.0, 200.0, 300.0, 300.0)) == 0.0


def test_place_pieces_never_overlaps():
    rng = np.random.default_rng(3)
    background = Image.new("RGB", (640, 640), (240, 240, 240))
    pieces = [Image.new("RGB", (120, 90), (c, c, c)) for c in (10, 120, 200)]
    _, boxes = place_pieces(background, pieces, rng, 0.15, 0.4)
    assert 1 <= len(boxes) <= 3
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            assert overlap_ratio(a, b) < 0.03


def test_compose_writes_a_coco_dataset(tmp_path: Path):
    foregrounds = {
        _painting(tmp_path / "a.png", (200, 20, 20)): 1,
        _painting(tmp_path / "b.png", (20, 20, 200), (60, 90)): 2,
    }
    out = tmp_path / "dataset"
    dataset = compose(
        foregrounds,
        out,
        CompositeOptions(size=256, count=6, seed=1, negative_prob=0.0),
        categories={1: "art-a", 2: "art-b"},
    )
    assert len(dataset.images) == 6
    assert len(dataset.annotations) >= 6
    assert {ann.category_id for ann in dataset.annotations} <= {1, 2}
    reloaded = CocoDataset.load(out / "annotations.json")
    assert reloaded.categories == {1: "art-a", 2: "art-b"}
    assert len(list((out / "images").glob("*.jpg"))) == 6


def test_compose_single_class_with_negatives(tmp_path: Path):
    foregrounds = {_painting(tmp_path / "a.png", (90, 90, 90)): ARTWORK_CATEGORY}
    dataset = compose(
        foregrounds,
        tmp_path / "det",
        CompositeOptions(size=256, count=5, seed=2, negative_prob=1.0),
    )
    assert len(dataset.images) == 5 and dataset.annotations == []
    assert dataset.categories == {ARTWORK_CATEGORY: "artwork"}
