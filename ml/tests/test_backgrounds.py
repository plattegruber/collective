import numpy as np

from collective_ml.backgrounds import BackgroundSource, procedural_wall


def test_procedural_wall_shape_and_determinism():
    a = procedural_wall(64, np.random.default_rng(1))
    b = procedural_wall(64, np.random.default_rng(1))
    assert a.size == (64, 64) and a.mode == "RGB"
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_background_source_without_directory():
    source = BackgroundSource(32, np.random.default_rng(0), directory=None)
    assert source.next().size == (32, 32)
