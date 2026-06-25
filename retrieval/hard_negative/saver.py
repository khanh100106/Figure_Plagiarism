"""
Hard Negative Mining Saver

Author: Nguyen Khanh
"""

from retrieval.configs import (
    HARD_NEGATIVE_FILE,
    INTRA_NEGATIVE_FILE,
    INTER_NEGATIVE_FILE,
)


def save_hard_negatives(
    hard_negatives,
    logger,
):
    """
    Save all hard negatives.
    """

    hard_negatives.to_csv(
        HARD_NEGATIVE_FILE,
        index=False,
    )

    logger.info(
        f"Saved : {HARD_NEGATIVE_FILE}"
    )


def save_intra_negatives(
    hard_negatives,
    logger,
):
    """
    Save intra-paper negatives.
    """

    intra = hard_negatives[

        hard_negatives["negative_type"]

        == "intra_paper"

    ]

    intra.to_csv(
        INTRA_NEGATIVE_FILE,
        index=False,
    )

    logger.info(
        f"Saved : {INTRA_NEGATIVE_FILE}"
    )


def save_inter_negatives(
    hard_negatives,
    logger,
):
    """
    Save inter-paper negatives.
    """

    inter = hard_negatives[

        hard_negatives["negative_type"]

        == "inter_paper"

    ]

    inter.to_csv(
        INTER_NEGATIVE_FILE,
        index=False,
    )

    logger.info(
        f"Saved : {INTER_NEGATIVE_FILE}"
    )


def save_all_outputs(
    hard_negatives,
    logger,
):
    """
    Save all CSV outputs.
    """

    save_hard_negatives(
        hard_negatives,
        logger,
    )

    save_intra_negatives(
        hard_negatives,
        logger,
    )

    save_inter_negatives(
        hard_negatives,
        logger,
    )