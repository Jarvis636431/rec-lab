"""DeepFM uses the exact sparse CTR data and train-only vocabulary."""

from models.sparse_ctr.data import (  # noqa: F401
    CATEGORICAL_FIELDS,
    NUMERIC_FIELDS,
    SparseSplit,
    prepare_data,
)
