import numpy as np

from collective_ml.runtime import auto_levels


def test_auto_levels_stretches_a_dim_gray_ramp_to_full_range():
    ramp = np.array([[[20, 20, 20], [40, 40, 40]], [[60, 60, 60], [80, 80, 80]]], dtype=np.uint8)
    out = auto_levels(ramp)
    assert out[0, 0, 0] == 0.0
    assert out[1, 1, 0] == 1.0
    assert abs(out[0, 1, 0] - 1 / 3) < 0.01


def test_auto_levels_removes_a_mild_colour_cast():
    cast = np.array([[[120, 100, 80], [60, 50, 40]]], dtype=np.uint8)
    out = auto_levels(cast)
    assert np.allclose(out[0, 0], out[0, 0, 0])
    assert np.allclose(out[0, 1], out[0, 1, 0])


def test_auto_levels_clamps_extreme_gains():
    cast = np.array([[[200, 100, 50], [100, 50, 25]]], dtype=np.uint8)
    out = auto_levels(cast)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert not np.allclose(out[0, 0], out[0, 0, 0])
