"""
Positive utilities.

Author: Nguyen Khanh
Project: Paper2Fig-2026
"""

from retrieval.configs import (
    POSITIVE_THRESHOLD,
    POSITIVE_PAIR_FILE,
)

# ============================================================
# Filter Positive Pairs
# ============================================================

def filter_positive_pairs(
    candidates,
    logger,
):
    """
    Filter positive pairs using similarity threshold.
    """

    logger.info(
        "Filtering positive pairs..."
    )

    positive_pairs = candidates[

        candidates["similarity"]

        >= POSITIVE_THRESHOLD

    ].copy()

    logger.info(

        f"Positive pairs : {len(positive_pairs):,}"

    )

    return positive_pairs

# ============================================================
# Save Positive Pairs
# ============================================================

def save_positive_pairs(
    positive_pairs,
    logger,
):

    positive_pairs.to_csv(

        POSITIVE_PAIR_FILE,

        index=False,

    )

    logger.info(

        f"Saved : {POSITIVE_PAIR_FILE}"

    )