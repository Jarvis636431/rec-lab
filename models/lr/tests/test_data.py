import numpy as np

from models.lr.data import prepare_data


def test_time_split_sizes_and_standardization() -> None:
    data = prepare_data(samples=1_000, seed=7)

    assert data.train_x.shape == (700, 5)
    assert data.val_x.shape == (150, 5)
    assert data.test_x.shape == (150, 5)
    assert np.allclose(data.train_x.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(data.train_x.std(axis=0), 1.0, atol=1e-5)


def test_cross_feature_is_optional() -> None:
    baseline = prepare_data(samples=1_000, seed=7, include_cross=False)
    with_cross = prepare_data(samples=1_000, seed=7, include_cross=True)

    assert baseline.feature_names[-1] == "is_evening"
    assert with_cross.feature_names[-1] == "sports_match"
    assert with_cross.train_x.shape[1] == baseline.train_x.shape[1] + 1
