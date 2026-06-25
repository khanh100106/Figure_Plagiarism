"""
Hard Negative Filtering

Author: Nguyen Khanh
"""

from retrieval.configs import (
    HARD_NEGATIVE_MIN,
    HARD_NEGATIVE_MAX,
)

def build_positive_lookup(
    positives,
):
    """
    Build positive pair lookup table.
    """

    positive_lookup = set()

    for _, row in positives.iterrows():

        pair = tuple(

            sorted(

                [

                    row["query_figure_id"],
                    row["candidate_figure_id"],

                ]

            )

        )

        positive_lookup.add(
            pair
        )

    return positive_lookup

def remove_positive_pairs(
    candidates,
    positive_lookup,
    logger,
):
    """
    Remove positive pairs from candidates.
    """

    logger.info(
        "Removing positive pairs..."
    )

    keep_rows = []

    for _, row in candidates.iterrows():

        pair = tuple(

            sorted(

                [

                    row["query_figure_id"],
                    row["candidate_figure_id"],

                ]

            )

        )

        if pair not in positive_lookup:

            keep_rows.append(
                row
            )

    filtered = candidates.__class__(
        keep_rows
    )

    logger.info(

        f"Remaining candidates : "
        f"{len(filtered):,}"

    )

    return filtered

def filter_similarity_range(
    candidates,
    logger,
):
    """
    Keep only hard negative range.
    """

    logger.info(
        "Filtering similarity range..."
    )

    filtered = candidates[

        (candidates["similarity"]
         >= HARD_NEGATIVE_MIN)

        &

        (candidates["similarity"]
         < HARD_NEGATIVE_MAX)

    ].copy()

    logger.info(

        f"Hard negatives : "
        f"{len(filtered):,}"

    )

    return filtered

def split_intra_inter(
    hard_negatives,
    logger,
):
    """
    Label intra/inter paper negatives.
    """

    logger.info(
        "Assigning negative types..."
    )

    hard_negatives = (
        hard_negatives.copy()
    )

    hard_negatives[
        "negative_type"
    ] = (

        hard_negatives[
            "query_paper_id"
        ]

        ==

        hard_negatives[
            "candidate_paper_id"
        ]

    )

    hard_negatives[
        "negative_type"
    ] = (

        hard_negatives[
            "negative_type"
        ]

        .map(
            {
                True:
                    "intra_paper",

                False:
                    "inter_paper",
            }
        )

    )

    return hard_negatives

def build_hard_negative_table(
    candidates,
    positives,
    logger,
):
    """
    Complete hard negative mining pipeline.
    """

    positive_lookup = (
        build_positive_lookup(
            positives
        )
    )

    candidates = (
        remove_positive_pairs(
            candidates,
            positive_lookup,
            logger,
        )
    )

    hard_negatives = (
        filter_similarity_range(
            candidates,
            logger,
        )
    )

    hard_negatives = (
        split_intra_inter(
            hard_negatives,
            logger,
        )
    )

    logger.info(

        f"Final hard negatives : "
        f"{len(hard_negatives):,}"

    )

    return hard_negatives