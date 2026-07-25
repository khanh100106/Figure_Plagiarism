"""
Paper2Fig-2026 Retrieval Training

DataLoader Factory
------------------

Responsibilities
----------------
- Create PyTorch DataLoaders.
- Hide DataLoader construction from Trainer.

Author
------
Nguyen Khanh
"""

from __future__ import annotations

from torch.utils.data import (
    DataLoader,
    Dataset,
)

from retrieval.configs import (
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)


# ============================================================
# Internal Builder
# ============================================================

def _build_dataloader(
    dataset: Dataset,
    shuffle: bool,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    """
    Build a PyTorch DataLoader.

    Parameters
    ----------
    dataset : Dataset
        PyTorch dataset.

    shuffle : bool
        Whether to shuffle the dataset.

    batch_size : int, default=BATCH_SIZE
        Batch size.

    Returns
    -------
    DataLoader
        Configured DataLoader.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

# ============================================================
# Public API
# ============================================================

def build_train_dataloader(
    dataset: Dataset,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    """
    Build training DataLoader.

    Parameters
    ----------
    dataset : Dataset
        Training dataset.

    batch_size : int, default=BATCH_SIZE
        Batch size.

    Returns
    -------
    DataLoader
        Training DataLoader.
    """
    return _build_dataloader(
        dataset=dataset,
        shuffle=True,
        batch_size=batch_size,
    )


def build_validation_dataloader(
    dataset: Dataset,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    """
    Build validation DataLoader.

    Parameters
    ----------
    dataset : Dataset
        Validation dataset.

    batch_size : int, default=BATCH_SIZE
        Batch size.

    Returns
    -------
    DataLoader
        Validation DataLoader.
    """
    return _build_dataloader(
        dataset=dataset,
        shuffle=False,
        batch_size=batch_size,
    )


def build_test_dataloader(
    dataset: Dataset,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    """
    Build test DataLoader.

    Parameters
    ----------
    dataset : Dataset
        Test dataset.

    batch_size : int, default=BATCH_SIZE
        Batch size.

    Returns
    -------
    DataLoader
        Test DataLoader.
    """
    return _build_dataloader(
        dataset=dataset,
        shuffle=False,
        batch_size=batch_size,
    )

