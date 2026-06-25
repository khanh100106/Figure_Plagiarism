"""
Retrieval Dataset Saver

Author: Nguyen Khanh
"""

import pandas as pd

from retrieval.configs import (
    RETRIEVAL_PAIR_FILE,
    RETRIEVAL_TRIPLET_FILE,
)

def save_pair_dataset(
    pair_dataset,
    logger,
):
    """
    Save retrieval pair dataset.
    """

    pair_dataset.to_csv(
        RETRIEVAL_PAIR_FILE,
        index=False,
    )

    logger.info(
        f"Saved : {RETRIEVAL_PAIR_FILE}"
    )

def save_triplet_dataset(
    triplet_dataset,
    logger,
):
    """
    Save retrieval triplet dataset.
    """

    triplet_dataset.to_csv(
        RETRIEVAL_TRIPLET_FILE,
        index=False,
    )

    logger.info(
        f"Saved : {RETRIEVAL_TRIPLET_FILE}"
    )

def save_all_outputs(
    pair_dataset,
    triplet_dataset,
    logger,
):
    """
    Save all retrieval datasets.
    """

    save_pair_dataset(
        pair_dataset,
        logger,
    )

    save_triplet_dataset(
        triplet_dataset,
        logger,
    )

