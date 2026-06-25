"""
Retrieval Dataset Builder

Author: Nguyen Khanh
"""

import pandas as pd

def build_pair_dataset(
    positives,
    negatives,
):
    """
    Build pair dataset for contrastive learning.

    Returns
    -------
    DataFrame

    Columns
    -------
    anchor_id
    target_id
    label
    """

    positive_pairs = pd.DataFrame({

        "anchor_id":
            positives["query_figure_id"],

        "target_id":
            positives["candidate_figure_id"],

        "label":
            1,

    })

    negative_pairs = pd.DataFrame({

        "anchor_id":
            negatives["query_figure_id"],

        "target_id":
            negatives["candidate_figure_id"],

        "label":
            0,

    })

    pair_dataset = pd.concat(
        [
            positive_pairs,
            negative_pairs,
        ],
        ignore_index=True,
    )

    return pair_dataset

def build_triplet_dataset(
    positives,
    negatives,
):
    """
    Build triplet dataset.

    Returns
    -------
    DataFrame

    Columns
    -------
    anchor_id
    positive_id
    negative_id
    """

    triplets = []

    positive_groups = (
        positives.groupby(
            "query_figure_id"
        )
    )

    negative_groups = (
        negatives.groupby(
            "query_figure_id"
        )
    )

    anchors = (
        set(
            positive_groups.groups.keys()
        )
        &
        set(
            negative_groups.groups.keys()
        )
    )

    for anchor_id in anchors:

        anchor_positives = (
            positive_groups
            .get_group(anchor_id)
        )

        anchor_negatives = (
            negative_groups
            .get_group(anchor_id)
        )

        positive_ids = (
            anchor_positives[
                "candidate_figure_id"
            ]
            .tolist()
        )

        negative_ids = (
            anchor_negatives[
                "candidate_figure_id"
            ]
            .tolist()
        )

        for positive_id in positive_ids:

            for negative_id in negative_ids:

                triplets.append({

                    "anchor_id":
                        anchor_id,

                    "positive_id":
                        positive_id,

                    "negative_id":
                        negative_id,

                })

    return pd.DataFrame(
        triplets
    )

def count_triplets(
    positives,
    negatives,
):
    """
    Estimate number of generated triplets.
    """

    total = 0

    positive_counts = (
        positives.groupby(
            "query_figure_id"
        ).size()
    )

    negative_counts = (
        negatives.groupby(
            "query_figure_id"
        ).size()
    )

    anchors = (
        set(positive_counts.index)
        &
        set(negative_counts.index)
    )

    for anchor in anchors:

        total += (
            positive_counts[anchor]
            *
            negative_counts[anchor]
        )

    return total
