import numpy as np

from collective_ml.evaluate import format_report, sweep_thresholds


def test_sweep_picks_lowest_threshold_meeting_each_false_accept_target():
    genuine = np.array([0.9, 0.85, 0.8, 0.7, 0.65])
    correct = np.array([True, True, True, True, False])
    impostors = np.array([0.5, 0.55, 0.6, 0.62, 0.7] + [0.3] * 95)
    sweep = sweep_thresholds(genuine, correct, impostors, far_targets=(0.02, 0.005))
    # 2% of 100 impostors: the threshold must exclude all but two → above 0.6.
    assert sweep["match"] == 0.61
    assert sweep["strong"] == 0.71
    row = next(r for r in sweep["rows"] if r["threshold"] == 0.61)
    assert row["tpr"] == 0.8 and row["far"] == 0.02


def test_sweep_falls_back_to_strictest_threshold_when_impostors_saturate():
    sweep = sweep_thresholds(np.array([0.9]), np.array([True]), np.array([0.99, 0.99]))
    assert sweep["match"] == 0.95 and sweep["strong"] == 0.95


def test_format_report_is_human_readable():
    report = {
        "pieces": 2,
        "prototypes": 2,
        "queries": 10,
        "detectionRecall": 1.0,
        "meanIoU": 0.9,
        "top1AmongDetected": 1.0,
        "top1Overall": 1.0,
        "genuine": {"p5": 0.8, "p50": 0.9, "mean": 0.88},
        "impostorKnown": {"p95": 0.4, "max": 0.5},
        "impostorUnknown": {"regions": 20, "p95": 0.3, "max": 0.45},
        "suggested": {"match": 0.6, "strong": 0.7},
        "sweep": [{"threshold": 0.6, "tpr": 1.0, "far": 0.0}],
    }
    text = format_report(report)
    assert "detection recall@0.5 1.000" in text and "match 0.60" in text
