"""
Visualization utilities.

Author: Nguyen Khanh
Project: Paper2Fig-2026
"""

import numpy as np
import matplotlib.pyplot as plt

from retrieval.configs import (
    SIMILARITY_HISTOGRAM_FILE,
    SIMILARITY_CDF_FILE,
    THRESHOLD_CURVE_FILE,
)

def plot_similarity_histogram(
    candidates,
):
    """
    Plot similarity histogram.
    """

    similarities = candidates[
        "similarity"
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        similarities,
        bins=50,
    )

    plt.xlabel(
        "Similarity"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Similarity Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        SIMILARITY_HISTOGRAM_FILE,
        dpi=300,
    )

    plt.close()

def plot_similarity_cdf(
    candidates,
):
    """
    Plot cumulative distribution.
    """

    similarities = np.sort(

        candidates[
            "similarity"
        ].values

    )

    cdf = np.arange(
        len(similarities)
    ) / float(
        len(similarities)
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        similarities,
        cdf,
    )

    plt.xlabel(
        "Similarity"
    )

    plt.ylabel(
        "CDF"
    )

    plt.title(
        "Similarity CDF"
    )

    plt.tight_layout()

    plt.savefig(
        SIMILARITY_CDF_FILE,
        dpi=300,
    )

    plt.close()

def plot_threshold_curve(
    threshold_df,
):
    """
    Plot threshold vs positive pairs.
    """

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(

        threshold_df[
            "threshold"
        ],

        threshold_df[
            "positive_pairs"
        ],

        marker="o",

    )

    plt.xlabel(
        "Threshold"
    )

    plt.ylabel(
        "Positive Pairs"
    )

    plt.title(
        "Threshold Analysis"
    )

    plt.tight_layout()

    plt.savefig(
        THRESHOLD_CURVE_FILE,
        dpi=300,
    )

    plt.close()

