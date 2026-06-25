"""
Retrieval Dataset Statistics

Author: Nguyen Khanh
"""

import json

from retrieval.configs import (
    RETRIEVAL_STATISTICS_FILE,
)

def compute_retrieval_statistics(
    pair_dataset,
    triplet_dataset,
):
    """
    Compute retrieval dataset statistics.
    """

    positive_pairs = (
        pair_dataset["label"] == 1
    ).sum()

    negative_pairs = (
        pair_dataset["label"] == 0
    ).sum()

    statistics = {

        "total_pairs":
            int(len(pair_dataset)),

        "positive_pairs":
            int(positive_pairs),

        "negative_pairs":
            int(negative_pairs),

        "total_triplets":
            int(len(triplet_dataset)),

        "unique_pair_anchors":
            int(
                pair_dataset[
                    "anchor_id"
                ].nunique()
            ),

        "unique_triplet_anchors":
            int(
                triplet_dataset[
                    "anchor_id"
                ].nunique()
            ),

    }

    return statistics

def save_retrieval_statistics(
    statistics,
):
    """
    Save retrieval statistics.
    """

    with open(
        RETRIEVAL_STATISTICS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            statistics,
            f,
            indent=4,
        )

def log_retrieval_statistics(
    statistics,
    logger,
):

    logger.info(
        "=" * 60
    )

    for key, value in statistics.items():

        logger.info(
            f"{key}: {value:,}"
        )

    logger.info(
        "=" * 60
    )