"""Synthetic multi-field impressions with train-only categorical vocabularies."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


CATEGORICAL_FIELDS: Tuple[str, ...] = (
    "user_id", "item_id", "user_segment", "item_category", "device", "hour_bucket"
)
NUMERIC_FIELDS: Tuple[str, ...] = ("log_price", "position")


@dataclass(frozen=True)
class SparseSplit:
    train_cat: np.ndarray
    train_num: np.ndarray
    train_y: np.ndarray
    val_cat: np.ndarray
    val_num: np.ndarray
    val_y: np.ndarray
    test_cat: np.ndarray
    test_num: np.ndarray
    test_y: np.ndarray
    vocab_values: Tuple[np.ndarray, ...]
    vocab_sizes: Tuple[int, ...]
    offsets: Tuple[int, ...]
    numeric_mean: np.ndarray
    numeric_std: np.ndarray
    oov_counts: dict
    raw_categories: Tuple[np.ndarray, np.ndarray, np.ndarray]


def _draw_ids(rng: np.random.Generator, count: int, size: int) -> np.ndarray:
    ranks = np.arange(1, size + 1, dtype=np.float64)
    probabilities = ranks ** -1.3
    probabilities /= probabilities.sum()
    return rng.choice(size, count, p=probabilities)


def generate_exposures(samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw categorical fields, numeric fields, and click labels."""
    if samples < 100:
        raise ValueError("samples must be at least 100")
    rng = np.random.default_rng(seed)
    users, items, segments, categories = 600, 400, 8, 12
    user_segment = rng.integers(0, segments, size=users)
    item_category = rng.integers(0, categories, size=items)
    user_bias = rng.normal(0, 0.25, size=users)
    item_bias = rng.normal(0, 0.25, size=items)
    segment_factors = rng.normal(0, 0.8, size=(segments, 4))
    category_factors = rng.normal(0, 0.8, size=(categories, 4))

    user_id = _draw_ids(rng, samples, users)
    item_id = _draw_ids(rng, samples, items)
    device = rng.choice(3, samples, p=[0.65, 0.25, 0.10])
    hour_bucket = rng.integers(0, 4, size=samples)
    price_by_item = rng.lognormal(np.log(250), 0.55, size=items)
    price = price_by_item[item_id] * rng.lognormal(0, 0.12, size=samples)
    position = rng.integers(1, 11, size=samples)
    log_price = np.log1p(price)

    affinity = (
        segment_factors[user_segment[user_id]] * category_factors[item_category[item_id]]
    ).sum(axis=1) / 2.0
    logits = (
        -2.65 + 1.35 * affinity + user_bias[user_id] + item_bias[item_id]
        - 0.45 * (log_price - np.log1p(250))
        - 0.65 * (position - 1) / 9
        + 0.18 * (device == 0) + 0.16 * (hour_bucket == 3)
    )
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    labels = rng.binomial(1, probabilities).astype(np.float32)
    categorical = np.column_stack((
        user_id, item_id, user_segment[user_id], item_category[item_id], device, hour_bucket
    )).astype(np.int64)
    numeric = np.column_stack((log_price, position)).astype(np.float32)
    return categorical, numeric, labels


def _encode(values: np.ndarray, vocab_values: Tuple[np.ndarray, ...], offsets: Tuple[int, ...]):
    encoded = np.empty(values.shape, dtype=np.int64)
    oov = {}
    for col, field in enumerate(CATEGORICAL_FIELDS):
        known = vocab_values[col]
        positions = np.searchsorted(known, values[:, col])
        in_vocab = positions < len(known)
        in_vocab[in_vocab] = known[positions[in_vocab]] == values[in_vocab, col]
        encoded[:, col] = np.where(in_vocab, positions + 1, 0) + offsets[col]
        oov[field] = int((~in_vocab).sum())
    return encoded, oov


def prepare_data(samples: int = 50_000, seed: int = 42) -> SparseSplit:
    raw_cat, raw_num, labels = generate_exposures(samples, seed)
    train_end, val_end = int(samples * 0.7), int(samples * 0.85)
    cat_parts = (raw_cat[:train_end], raw_cat[train_end:val_end], raw_cat[val_end:])
    num_parts = (raw_num[:train_end], raw_num[train_end:val_end], raw_num[val_end:])
    vocab_values = tuple(np.unique(cat_parts[0][:, col]) for col in range(len(CATEGORICAL_FIELDS)))
    vocab_sizes = tuple(len(values) + 1 for values in vocab_values)
    offsets = tuple(int(sum(vocab_sizes[:col])) for col in range(len(vocab_sizes)))
    encoded = tuple(_encode(part, vocab_values, offsets) for part in cat_parts)
    mean = num_parts[0].mean(axis=0)
    std = num_parts[0].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    standardized = tuple(((part - mean) / std).astype(np.float32) for part in num_parts)
    return SparseSplit(
        train_cat=encoded[0][0], train_num=standardized[0], train_y=labels[:train_end],
        val_cat=encoded[1][0], val_num=standardized[1], val_y=labels[train_end:val_end],
        test_cat=encoded[2][0], test_num=standardized[2], test_y=labels[val_end:],
        vocab_values=vocab_values, vocab_sizes=vocab_sizes, offsets=offsets,
        numeric_mean=mean, numeric_std=std,
        oov_counts={name: result[1] for name, result in zip(("train", "validation", "test"), encoded)},
        raw_categories=cat_parts,
    )
