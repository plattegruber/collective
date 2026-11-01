from pathlib import Path

import pytest

from collective_ml.coco import CocoDataset


def test_dataset_assembly_and_round_trip(tmp_path: Path):
    ds = CocoDataset(categories={1: "art-a", 2: "art-b"})
    first = ds.add_image("a.jpg", 640, 640)
    second = ds.add_image("b.jpg", 640, 640)
    ds.add_box(first, 1, (10, 20, 110, 220))
    ds.add_box(second, 2, (0, 0, 50, 50))

    assert first == 1 and second == 2
    assert ds.annotations[0].bbox == (10.0, 20.0, 100.0, 200.0)
    assert ds.annotations[0].area == 20000.0

    path = tmp_path / "annotations.json"
    ds.save(path)
    loaded = CocoDataset.load(path)
    assert loaded.categories == ds.categories
    assert [i.file_name for i in loaded.images] == ["a.jpg", "b.jpg"]
    assert loaded.annotations_for(2)[0].category_id == 2
    assert loaded.to_dict()["categories"] == [
        {"id": 1, "name": "art-a"},
        {"id": 2, "name": "art-b"},
    ]


def test_unknown_category_rejected():
    ds = CocoDataset(categories={1: "art-a"})
    image_id = ds.add_image("a.jpg", 10, 10)
    with pytest.raises(KeyError):
        ds.add_box(image_id, 7, (0, 0, 1, 1))
