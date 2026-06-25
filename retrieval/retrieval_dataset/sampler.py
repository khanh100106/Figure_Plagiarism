"""
Retrieval Dataset Sampler

Author: Nguyen Khanh
"""

from retrieval.configs import (
    MAX_POSITIVES_PER_ANCHOR,
    MAX_NEGATIVES_PER_ANCHOR,
)

def sample_positive_pairs(
    positives,
    max_positives=MAX_POSITIVES_PER_ANCHOR,
):
    """
    Limit number of positives per anchor.
    """

    sampled = (
        positives
        .groupby(
            "query_figure_id",
            group_keys=False,
        )
        .head(
            max_positives
        )
        .reset_index(
            drop=True
        )
    )

    return sampled

def sample_negative_pairs(
    negatives,
    max_negatives=MAX_NEGATIVES_PER_ANCHOR,
):
    """
    Keep hardest negatives per anchor.
    """

    sampled = (
        negatives
        .groupby(
            "query_figure_id",
            group_keys=False,
        )
        .head(
            max_negatives
        )
        .reset_index(
            drop=True
        )
    )

    return sampled

def sample_training_pairs(
    positives,
    negatives,
):
    """
    Sample positives and negatives.
    """

    positives = (
        sample_positive_pairs(
            positives
        )
    )

    negatives = (
        sample_negative_pairs(
            negatives
        )
    )

    return (
        positives,
        negatives,
    )

def summarize_sampling(
    positives_before,
    positives_after,
    negatives_before,
    negatives_after,
):
    """
    Sampling summary.
    """

    return {

        "positives_before":
            len(positives_before),

        "positives_after":
            len(positives_after),

        "negatives_before":
            len(negatives_before),

        "negatives_after":
            len(negatives_after),

    }

