"""
Paper2Fig-2026 Retrieval Training

Dataset Factory
---------------

Responsibilities
----------------
- Build retrieval datasets from configuration.
- Load metadata.
- Load pair/triplet annotations.
- Create PairDataset, TripletDataset or CombinedDataset.
- Hide dataset construction from Trainer.

Author
------
Nguyen Khanh
"""

from __future__ import annotations


import pandas as pd
from torch.utils.data import Dataset

from retrieval.configs import (
    LOSS_TYPE,
    METADATA_FILE,
    RETRIEVAL_PAIR_FILE,
    RETRIEVAL_TRIPLET_FILE,
)

from retrieval.training.dataset import (
    build_image_lookup,
    PairDataset,
    TripletDataset,
    CombinedDataset,
)

# ============================================================
# Metadata
# ============================================================

def _load_metadata() -> pd.DataFrame:
    """
    Load metadata file.

    Returns
    -------
    pd.DataFrame
        Figure metadata.
    """
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_FILE}"
        )

    return pd.read_csv(
        METADATA_FILE,
    )


# ============================================================
# Pair Dataset
# ============================================================

def _load_pairs() -> pd.DataFrame:
    """
    Load retrieval pair annotations.

    Returns
    -------
    pd.DataFrame
        Pair annotations.
    """
    if not RETRIEVAL_PAIR_FILE.exists():
        raise FileNotFoundError(
            f"Pair file not found:\n{RETRIEVAL_PAIR_FILE}"
        )

    return pd.read_csv(
        RETRIEVAL_PAIR_FILE,
    )


# ============================================================
# Triplet Dataset
# ============================================================

def _load_triplets() -> pd.DataFrame:
    """
    Load retrieval triplet annotations.

    Returns
    -------
    pd.DataFrame
        Triplet annotations.
    """
    if not RETRIEVAL_TRIPLET_FILE.exists():
        raise FileNotFoundError(
            f"Triplet file not found:\n{RETRIEVAL_TRIPLET_FILE}"
        )

    return pd.read_csv(
        RETRIEVAL_TRIPLET_FILE,
    )

# ============================================================
# Dataset Builder
# ============================================================

def _build_dataset(
    transform=None,
) -> Dataset:
    """
    Build retrieval dataset.

    Parameters
    ----------
    transform : callable, optional
        Image transform.

    Returns
    -------
    Dataset
        PairDataset, TripletDataset or CombinedDataset
        depending on LOSS_TYPE.
    """
    metadata = _load_metadata()

    image_lookup = build_image_lookup(
        metadata,
    )

    # --------------------------------------------------------
    # Pair Dataset
    # --------------------------------------------------------
    if LOSS_TYPE == "contrastive":

        pairs = _load_pairs()

        return PairDataset(
            pairs=pairs,
            image_lookup=image_lookup,
            transform=transform,
        )

    # --------------------------------------------------------
    # Triplet Dataset
    # --------------------------------------------------------
    if LOSS_TYPE == "triplet":

        triplets = _load_triplets()

        return TripletDataset(
            triplets=triplets,
            image_lookup=image_lookup,
            transform=transform,
        )

    # --------------------------------------------------------
    # Hybrid Dataset
    # --------------------------------------------------------
    if LOSS_TYPE == "hybrid":

        pairs = _load_pairs()

        triplets = _load_triplets()

        pair_dataset = PairDataset(
            pairs=pairs,
            image_lookup=image_lookup,
            transform=transform,
        )

        triplet_dataset = TripletDataset(
            triplets=triplets,
            image_lookup=image_lookup,
            transform=transform,
        )

        return CombinedDataset(
            pair_dataset=pair_dataset,
            triplet_dataset=triplet_dataset,
        )

    raise ValueError(
        f"Unsupported LOSS_TYPE: {LOSS_TYPE}"
    )

# ============================================================
# Public API
# ============================================================

def build_train_dataset(
    transform=None,
) -> Dataset:
    """
    Build training dataset.

    Parameters
    ----------
    transform : callable, optional
        Image transform.

    Returns
    -------
    Dataset
        Training dataset.
    """
    return _build_dataset(
        transform=transform,
    )


def build_validation_dataset(
    transform=None,
) -> Dataset:
    """
    Build validation dataset.

    Temporary:
        use the training dataset until
        validation split is implemented.
    """

    return _build_dataset(
        transform=transform,
    )


def build_test_dataset(
    transform=None,
) -> Dataset:
    """
    Build test dataset.

    Notes
    -----
    Test split is not implemented yet.

    Parameters
    ----------
    transform : callable, optional
        Image transform.

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError(
        "Test dataset is not implemented yet."
    )
