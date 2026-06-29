"""
Paper2Fig-2026 Retrieval Training
Module:
    Dataset Loader
Responsibilities
----------------
- Load retrieval pair dataset.
- Load retrieval triplet dataset.
- Validate dataset integrity.
- Provide reusable dataset loading APIs.
Author: Nguyen Khanh
Project: Paper2Fig-2026
"""
from __future__ import annotations
import pandas as pd
from logging import Logger
from retrieval.configs import (
    RETRIEVAL_PAIR_FILE,
    RETRIEVAL_TRIPLET_FILE,
    FIGURE_METADATA_FILE,
)
from retrieval.constants import (
    PAIR_COLUMNS,
    TRIPLET_COLUMNS,
    METADATA_COLUMNS,
)
# ============================================================
# Exceptions
# ============================================================
class DatasetFormatError(Exception):
    """Dataset format is invalid."""
class MissingColumnError(Exception):
    """Required columns are missing."""
# ============================================================
# Validation
# ============================================================
def validate_pairs(
    pairs: pd.DataFrame,
) -> None:
    """
    Validate retrieval pair dataset.
    Parameters
    ----------
    pairs : pd.DataFrame
    """
    missing = [
        col
        for col in PAIR_COLUMNS
        if col not in pairs.columns
    ]
    if missing:
        raise MissingColumnError(
            f"Missing columns: {missing}"
        )
    if pairs.empty:
        raise DatasetFormatError(
            "Pair dataset is empty."
        )
    if pairs.isna().any().any():
        raise DatasetFormatError(
            "Pair dataset contains NaN values."
        )
    duplicated = pairs.duplicated().sum()
    if duplicated > 0:
        raise DatasetFormatError(
            f"Pair dataset contains {duplicated} duplicated rows."
        )
    labels = set(pairs["label"].unique())
    if not labels.issubset({0, 1}):
        raise DatasetFormatError(
            "Labels must be either 0 or 1."
        )
def validate_triplets(
    triplets: pd.DataFrame,
) -> None:
    """
    Validate retrieval triplet dataset.
    """
    missing = [
        col
        for col in TRIPLET_COLUMNS
        if col not in triplets.columns
    ]
    if missing:
        raise MissingColumnError(
            f"Missing columns: {missing}"
        )
    if triplets.empty:
        raise DatasetFormatError(
            "Triplet dataset is empty."
        )
    if triplets.isna().any().any():
        raise DatasetFormatError(
            "Triplet dataset contains NaN values."
        )
    duplicated = triplets.duplicated().sum()
    if duplicated > 0:
        raise DatasetFormatError(
            f"Triplet dataset contains {duplicated} duplicated rows."
        )
    invalid = (
        (triplets["anchor_id"] == triplets["positive_id"])
        |
        (triplets["anchor_id"] == triplets["negative_id"])
        |
        (triplets["positive_id"] == triplets["negative_id"])
    )
    if invalid.any():
        raise DatasetFormatError(
            "Triplet dataset contains invalid triplets."
        )
# ============================================================
# Loader
# ============================================================
def load_pairs(
    logger: Logger,
) -> pd.DataFrame:
    """
    Load retrieval pair dataset.
    """
    logger.info(
        "Loading retrieval pairs..."
    )
    pairs = pd.read_csv(
        RETRIEVAL_PAIR_FILE,
    )
    validate_pairs(
        pairs,
    )
    logger.info(
        "Pairs : %s",
        f"{len(pairs):,}",
    )
    return pairs
def load_triplets(
    logger: Logger,
) -> pd.DataFrame:
    """
    Load retrieval triplet dataset.
    """
    logger.info(
        "Loading retrieval triplets..."
    )
    triplets = pd.read_csv(
        RETRIEVAL_TRIPLET_FILE,
    )
    validate_triplets(
        triplets,
    )
    logger.info(
        "Triplets : %s",
        f"{len(triplets):,}",
    )
    return triplets
# ============================================================
# Convenience API
# ============================================================
def load_training_dataset(
    logger: Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load pair and triplet datasets.
    Returns
    -------
    tuple
        (pairs, triplets)
    """
    logger.info("=" * 60)
    logger.info("Loading Retrieval Training Dataset")
    logger.info("=" * 60)
    pairs = load_pairs(logger)
    triplets = load_triplets(logger)
    logger.info("=" * 60)
    logger.info("Dataset validation passed.")
    logger.info("=" * 60)
    return (
        pairs,
        triplets,
    )
def validate_metadata(
    metadata: pd.DataFrame,
) -> None:
    missing = [
        col
        for col in METADATA_COLUMNS
        if col not in metadata.columns
    ]
    if missing:
        raise MissingColumnError(
            f"Missing metadata columns: {missing}"
        )
    if metadata.empty:
        raise DatasetFormatError(
            "Metadata is empty."
        )
    if metadata.isna().any().any():
        raise DatasetFormatError(
            "Metadata contains NaN."
        )
def load_figures_metadata(
    logger: Logger,
) -> pd.DataFrame:
    logger.info(
        "Loading figures metadata..."
    )
    metadata = pd.read_csv(
        FIGURE_METADATA_FILE,
    )
    validate_metadata(
        metadata,
    )
    logger.info(
        "Figures : %s",
        f"{len(metadata):,}",
    )
    return metadata
