"""Synthetic exposure data for the first CTR experiment."""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np


BASE_FEATURES: Tuple[str, ...] = (
    "user_sports",
    "item_sports",
    "price",
    "position",
    "is_evening",
)
CROSS_FEATURE = "sports_match"


@dataclass(frozen=True)
class DataSplit:
    """Time-ordered train, validation, and test arrays."""

    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    feature_names: Tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    true_ctr: float


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def generate_exposures(samples: int, seed: int) -> Dict[str, np.ndarray]:
    """Generate chronological impressions from a known click process."""

    if samples < 100:
        raise ValueError("samples must be at least 100")

    rng = np.random.default_rng(seed)
    user_sports = rng.binomial(1, 0.42, samples).astype(np.float32)
    item_sports = rng.binomial(1, 0.35, samples).astype(np.float32)
    price = rng.lognormal(mean=np.log(280), sigma=0.65, size=samples).astype(
        np.float32
    )
    price = np.clip(price, 30, 2500)
    position = rng.integers(1, 11, samples).astype(np.float32)
    is_evening = rng.binomial(1, 0.38, samples).astype(np.float32)
    sports_match = user_sports * item_sports

    price_scale = np.log1p(price) - np.log1p(280)
    position_scale = (position - 1) / 9
    logits = (
        -2.6
        + 0.35 * user_sports
        + 0.20 * item_sports
        + 1.55 * sports_match
        - 0.70 * price_scale
        - 0.90 * position_scale
        + 0.25 * is_evening
    )
    probabilities = _sigmoid(logits)
    labels = rng.binomial(1, probabilities).astype(np.float32)

    return {
        "user_sports": user_sports,
        "item_sports": item_sports,
        "price": np.log1p(price).astype(np.float32),
        "position": position,
        "is_evening": is_evening,
        "sports_match": sports_match,
        "probability": probabilities.astype(np.float32),
        "label": labels,
    }


def prepare_data(
    samples: int = 20_000,
    seed: int = 42,
    include_cross: bool = False,
) -> DataSplit:
    """Generate, time-split, and standardize features without leakage."""

    exposures = generate_exposures(samples=samples, seed=seed)
    feature_names: Sequence[str] = BASE_FEATURES
    if include_cross:
        feature_names = (*BASE_FEATURES, CROSS_FEATURE)

    features = np.column_stack([exposures[name] for name in feature_names]).astype(
        np.float32
    )
    labels = exposures["label"]

    train_end = int(samples * 0.70)
    val_end = int(samples * 0.85)

    train_x = features[:train_end]
    val_x = features[train_end:val_end]
    test_x = features[val_end:]

    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    def standardize(values: np.ndarray) -> np.ndarray:
        return ((values - mean) / std).astype(np.float32)

    return DataSplit(
        train_x=standardize(train_x),
        train_y=labels[:train_end],
        val_x=standardize(val_x),
        val_y=labels[train_end:val_end],
        test_x=standardize(test_x),
        test_y=labels[val_end:],
        feature_names=tuple(feature_names),
        mean=mean.squeeze(0),
        std=std.squeeze(0),
        true_ctr=float(labels.mean()),
    )
