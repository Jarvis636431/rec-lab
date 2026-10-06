import numpy as np

from models.sparse_ctr.data import CATEGORICAL_FIELDS, prepare_data


def test_split_vocab_oov_and_numeric_stats():
    data = prepare_data(samples=50_000, seed=42)
    assert data.train_cat.shape == (35_000, 6)
    assert data.val_cat.shape == (7_500, 6)
    assert data.test_cat.shape == (7_500, 6)
    assert data.train_num.shape == (35_000, 2)
    assert np.allclose(data.train_num.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(data.train_num.std(axis=0), 1, atol=1e-4)
    assert data.oov_counts["train"] == dict.fromkeys(CATEGORICAL_FIELDS, 0)
    assert data.oov_counts["validation"]["user_id"] > 0

    for col, name in enumerate(CATEGORICAL_FIELDS):
        known = set(data.raw_categories[0][:, col])
        for raw, encoded, split in (
            (data.raw_categories[1], data.val_cat, "validation"),
            (data.raw_categories[2], data.test_cat, "test"),
        ):
            unseen = np.array([value not in known for value in raw[:, col]])
            assert unseen.sum() == data.oov_counts[split][name]
            assert np.array_equal(
                encoded[unseen, col], np.full(unseen.sum(), data.offsets[col])
            )


def test_data_is_reproducible():
    first = prepare_data(samples=1000, seed=7)
    second = prepare_data(samples=1000, seed=7)
    np.testing.assert_array_equal(first.train_cat, second.train_cat)
    np.testing.assert_array_equal(first.train_y, second.train_y)
