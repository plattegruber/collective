import pytest

from collective_ml.geometry import clip_box, coco_to_voc, iou, letterbox, voc_to_coco


def test_letterbox_landscape_centres_vertically():
    box = letterbox(640, 480, 320)
    assert box.scale == pytest.approx(0.5)
    assert (box.width, box.height) == (320, 240)
    assert (box.left, box.top) == (0, 40)


def test_letterbox_round_trip():
    box = letterbox(1000, 400, 320)
    original = (100.0, 50.0, 900.0, 350.0)
    canvas = box.to_canvas(original)
    assert canvas[0] >= 0 and canvas[3] <= 320
    back = box.to_source(canvas)
    assert back == pytest.approx(original)


def test_voc_coco_conversions():
    assert voc_to_coco((10, 20, 30, 50)) == (10.0, 20.0, 20.0, 30.0)
    assert coco_to_voc((10, 20, 20, 30)) == (10.0, 20.0, 30.0, 50.0)
    assert voc_to_coco((10, 10, 5, 5))[2:] == (0.0, 0.0)


def test_clip_box():
    assert clip_box((-5, -5, 50, 500), 100, 100) == (0.0, 0.0, 50.0, 100.0)


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
