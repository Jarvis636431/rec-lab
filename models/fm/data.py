"""Use the exact LR exposure generation and split for a paired comparison."""

from models.lr.data import DataSplit, prepare_data as prepare_lr_data


def prepare_data(samples: int = 20_000, seed: int = 42) -> DataSplit:
    """Return the five original features, without the manual sports cross."""
    return prepare_lr_data(samples=samples, seed=seed, include_cross=False)
