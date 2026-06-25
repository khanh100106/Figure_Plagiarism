"""
Hard Negative Mining Loader

Author: Nguyen Khanh
"""

import pandas as pd

from retrieval.configs import (
    TOPK_CANDIDATE_FILE,
    POSITIVE_PAIR_FILE,
)

def load_candidates(
    logger,
):
    """
    Load Top-K candidate pairs.
    """

    candidates = pd.read_csv(
        TOPK_CANDIDATE_FILE
    )

    logger.info(
        f"Candidates : {len(candidates):,}"
    )

    return candidates

def load_positive_pairs(
    logger,
):
    """
    Load positive pairs.
    """

    positives = pd.read_csv(
        POSITIVE_PAIR_FILE
    )

    logger.info(
        f"Positive pairs : {len(positives):,}"
    )

    return positives

def validate_inputs(
    candidates,
    positives,
):
    """
    Validate mining inputs.
    """

    required_columns = [

        "query_figure_id",

        "candidate_figure_id",

        "similarity",

    ]

    for column in required_columns:

        if column not in candidates.columns:

            raise ValueError(
                f"Missing column: {column}"
            )

    for column in [

        "query_figure_id",

        "candidate_figure_id",

    ]:

        if column not in positives.columns:

            raise ValueError(
                f"Missing column: {column}"
            )

