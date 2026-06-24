"""
Mining statistics utilities.

Author: Nguyen Khanh
Project: Paper2Fig-2026
"""

import json

import numpy as np
import pandas as pd

from retrieval.configs import (
    SIMILARITY_STATISTICS_FILE,
    THRESHOLD_ANALYSIS_FILE,
)


# ============================================================
# Similarity Statistics
# ============================================================

def compute_similarity_statistics(
    candidates,
):
    """
    Compute similarity statistics.
    """

    similarities = candidates[
        "similarity"
    ]

    stats = {

        "count":
            int(len(similarities)),

        "mean":
            float(similarities.mean()),

        "median":
            float(similarities.median()),

        "std":
            float(similarities.std()),

        "min":
            float(similarities.min()),

        "max":
            float(similarities.max()),

        "p90":
            float(
                similarities.quantile(0.90)
            ),

        "p95":
            float(
                similarities.quantile(0.95)
            ),

        "p97":
            float(
                similarities.quantile(0.97)
            ),

        "p99":
            float(
                similarities.quantile(0.99)
            ),
    }

    return stats


# ============================================================
# Save Similarity Statistics
# ============================================================

def save_similarity_statistics(
    stats,
):
    """
    Save similarity statistics to JSON.
    """

    with open(
        SIMILARITY_STATISTICS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            stats,
            f,
            indent=4,
        )


# ============================================================
# Threshold Analysis
# ============================================================

def evaluate_thresholds(
    candidates,
):
    """
    Evaluate number of positive pairs
    under multiple thresholds.
    """

    thresholds = np.arange(
        0.70,
        1.00,
        0.02,
    )

    results = []

    for threshold in thresholds:

        num_pairs = len(

            candidates[
                candidates["similarity"]
                >= threshold
            ]

        )

        results.append(

            {

                "threshold":
                    round(
                        float(threshold),
                        2,
                    ),

                "positive_pairs":
                    int(num_pairs),

            }

        )

    return pd.DataFrame(
        results
    )


# ============================================================
# Save Threshold Analysis
# ============================================================

def save_threshold_analysis(
    threshold_df,
):
    """
    Save threshold analysis.
    """

    threshold_df.to_csv(
        THRESHOLD_ANALYSIS_FILE,
        index=False,
    )


# ============================================================
# Recommended Threshold
# ============================================================

def recommend_threshold(
    stats,
):
    """
    Recommend threshold.

    Current strategy:
    P95 similarity.
    """

    return stats["p95"]