from pathlib import Path

import pytest

from collective_ml.labels import (
    build_label_map,
    class_index_of,
    load_label_map,
    save_label_map,
    scan_raw_dir,
)


def test_build_label_map_is_sorted_and_one_based():
    labels = build_label_map(["zeta", "alpha", "mid", "alpha"])
    assert labels == {"1": "alpha", "2": "mid", "3": "zeta"}
    assert class_index_of(labels) == {"alpha": 1, "mid": 2, "zeta": 3}


def test_scan_raw_dir_ignores_empty_folders_and_non_images(tmp_path: Path):
    (tmp_path / "art-b").mkdir()
    (tmp_path / "art-b" / "one.JPG").write_bytes(b"x")
    (tmp_path / "art-b" / "notes.txt").write_bytes(b"x")
    (tmp_path / "art-a").mkdir()
    (tmp_path / "art-a" / "a.png").write_bytes(b"x")
    (tmp_path / "empty").mkdir()

    pieces = scan_raw_dir(tmp_path)
    assert list(pieces) == ["art-a", "art-b"]
    assert [p.name for p in pieces["art-b"]] == ["one.JPG"]


def test_label_map_round_trip(tmp_path: Path):
    path = tmp_path / "model" / "labels.json"
    save_label_map({"1": "abc", "2": "def"}, path)
    assert load_label_map(path) == {"1": "abc", "2": "def"}


def test_load_label_map_rejects_bad_shape(tmp_path: Path):
    path = tmp_path / "labels.json"
    path.write_text('{"a": "b"}')
    with pytest.raises(ValueError):
        load_label_map(path)
