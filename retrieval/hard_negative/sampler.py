"""
Hard Negative Sampler

Author: Nguyen Khanh
"""

import pandas as pd


def sample_random_negatives(
    hard_negatives,
    n_samples,
    seed=42,
):
    """
    Random sample.
    """

    n_samples = min(
        n_samples,
        len(hard_negatives),
    )

    return hard_negatives.sample(
        n=n_samples,
        random_state=seed,
    )


def sample_top_negatives(
    hard_negatives,
    n_samples,
):
    """
    Hardest negatives.
    """

    return (

        hard_negatives

        .sort_values(
            "similarity",
            ascending=False,
        )

        .head(n_samples)

    )


def sample_mixed_negatives(
    hard_negatives,
    n_samples,
    seed=42,
):
    """
    50% hard
    50% random
    """

    hard_n = n_samples // 2

    random_n = n_samples - hard_n

    hard_part = sample_top_negatives(
        hard_negatives,
        hard_n,
    )

    remaining = hard_negatives.drop(
        hard_part.index
    )

    random_part = sample_random_negatives(
        remaining,
        random_n,
        seed,
    )

    return pd.concat(

        [

            hard_part,

            random_part,

        ],

        ignore_index=True,

    )


def sample_negatives_per_anchor(
    hard_negatives,
    max_negatives,
    seed=42,
):
    """
    Sample negatives per query figure.
    """

    sampled = []

    grouped = (

        hard_negatives

        .groupby(
            "query_figure_id"
        )

    )

    for _, group in grouped:

        n = min(
            max_negatives,
            len(group),
        )

        sampled.append(

            group.sample(
                n=n,
                random_state=seed,
            )

        )

    return pd.concat(
        sampled,
        ignore_index=True,
    )