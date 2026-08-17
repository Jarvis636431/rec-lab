import numpy as np

from models.lr.data import generate_exposures, prepare_data


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


def test_default_generator_remains_reproducible() -> None:
    original = generate_exposures(samples=1_000, seed=7)
    explicit_default = generate_exposures(samples=1_000, seed=7, base_ctr=None)

    assert np.array_equal(original["label"], explicit_default["label"])
    assert np.array_equal(original["probability"], explicit_default["probability"])


def test_lower_base_ctr_reduces_observed_ctr() -> None:
    original = generate_exposures(samples=20_000, seed=7)
    low_ctr = generate_exposures(samples=20_000, seed=7, base_ctr=0.005)

    assert low_ctr["label"].mean() < original["label"].mean()
    assert low_ctr["label"].mean() < 0.02


def test_base_ctr_must_be_a_probability() -> None:
    for invalid_value in (0.0, 1.0, -0.1, 1.1):
        try:
            generate_exposures(samples=1_000, seed=7, base_ctr=invalid_value)
        except ValueError as error:
            assert str(error) == "base_ctr must be between 0 and 1"
        else:
            raise AssertionError("invalid base_ctr should raise ValueError")
