"""
Retrieval Dataset Loader

Author: Nguyen Khanh
"""

import pandas as pd

from retrieval.configs import (
    POSITIVE_PAIR_FILE,
    HARD_NEGATIVE_FILE,
)

def load_positive_pairs(
    logger,
):

    positives = pd.read_csv(
        POSITIVE_PAIR_FILE
    )

    logger.info(
        f"Positive pairs : {len(positives):,}"
    )

    return positives

def load_hard_negatives(
    logger,
):

    negatives = pd.read_csv(
        HARD_NEGATIVE_FILE
    )

    logger.info(
        f"Hard negatives : {len(negatives):,}"
    )

    return negatives

def validate_inputs(
    positives,
    negatives,
):

    required_columns = [

        "query_figure_id",

        "candidate_figure_id",

    ]

    for column in required_columns:

        if column not in positives.columns:

            raise ValueError(
                f"Missing column: {column}"
            )

    for column in required_columns:

        if column not in negatives.columns:

            raise ValueError(
                f"Missing column: {column}"
            )

