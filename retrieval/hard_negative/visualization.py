"""
Hard Negative Visualizations

Author: Nguyen Khanh
"""

import matplotlib.pyplot as plt

from retrieval.configs import (
    HARD_NEGATIVE_HISTOGRAM_FILE,
    NEGATIVE_TYPE_PLOT_FILE,
)

from retrieval.configs import (
    SIMILARITY_BY_TYPE_FILE,
)

def plot_similarity_histogram(
    hard_negatives,
):
    """
    Similarity histogram.
    """

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        hard_negatives["similarity"],
        bins=30,
    )

    plt.xlabel(
        "Similarity"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        "Hard Negative Similarity Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        HARD_NEGATIVE_HISTOGRAM_FILE
    )

    plt.close()

def plot_negative_type_distribution(
    hard_negatives,
):
    """
    Negative type distribution.
    """

    counts = (

        hard_negatives[
            "negative_type"
        ]

        .value_counts()

    )

    plt.figure(
        figsize=(6, 4)
    )

    counts.plot(
        kind="bar"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        "Negative Type Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        NEGATIVE_TYPE_PLOT_FILE
    )

    plt.close()

def plot_similarity_by_type(
    hard_negatives,
):
    """
    Similarity distribution by type.
    """

    plt.figure(
        figsize=(8, 5)
    )

    for negative_type in [

        "intra_paper",

        "inter_paper",

    ]:

        subset = hard_negatives[

            hard_negatives[
                "negative_type"
            ]

            == negative_type

        ]

        plt.hist(
            subset["similarity"],
            bins=20,
            alpha=0.5,
            label=negative_type,
        )

    plt.xlabel(
        "Similarity"
    )

    plt.ylabel(
        "Count"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        SIMILARITY_BY_TYPE_FILE
    )

    plt.close()