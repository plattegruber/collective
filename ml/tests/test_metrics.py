import pytest

from collective_ml.metrics import Detection, GroundTruth, average_precision, mean_average_precision


def test_average_precision_perfect():
    assert average_precision([0.5, 1.0], [1.0, 1.0]) == pytest.approx(1.0)


def test_map50_rewards_correct_and_penalises_wrong():
    truths = [GroundTruth(1, 1, (0, 0, 10, 10)), GroundTruth(2, 2, (0, 0, 10, 10))]
    perfect = [Detection(1, 1, 0.9, (0, 0, 10, 10)), Detection(2, 2, 0.9, (0, 0, 10, 10))]
    assert mean_average_precision(perfect, truths).map50 == pytest.approx(1.0)

    wrong = [Detection(1, 2, 0.9, (0, 0, 10, 10)), Detection(2, 1, 0.9, (0, 0, 10, 10))]
    assert mean_average_precision(wrong, truths).map50 == pytest.approx(0.0)

    half = [Detection(1, 1, 0.9, (0, 0, 10, 10)), Detection(2, 2, 0.9, (50, 50, 60, 60))]
    result = mean_average_precision(half, truths)
    assert result.per_class[1] == pytest.approx(1.0)
    assert result.per_class[2] == pytest.approx(0.0)
    assert result.map50 == pytest.approx(0.5)


def test_duplicate_detections_count_once():
    truths = [GroundTruth(1, 1, (0, 0, 10, 10))]
    dets = [Detection(1, 1, 0.9, (0, 0, 10, 10)), Detection(1, 1, 0.8, (0, 0, 10, 10))]
    assert mean_average_precision(dets, truths).map50 == pytest.approx(1.0)
