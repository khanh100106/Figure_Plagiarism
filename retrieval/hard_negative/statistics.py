"""
Hard Negative Statistics

Author: Nguyen Khanh
"""

import json
from datetime import datetime

from retrieval.configs import (
    HARD_NEGATIVE_STATISTICS_FILE,
)

def compute_hard_negative_statistics(
    hard_negatives,
):
    """
    Compute hard negative statistics.
    """

    similarities = hard_negatives[
        "similarity"
    ]

    intra_count = int(

        (
            hard_negatives[
                "negative_type"
            ]
            ==
            "intra_paper"
        ).sum()

    )

    inter_count = int(

        (
            hard_negatives[
                "negative_type"
            ]
            ==
            "inter_paper"
        ).sum()

    )

    stats = {

        "total_hard_negatives":

            int(
                len(hard_negatives)
            ),

        "intra_paper":

            intra_count,

        "inter_paper":

            inter_count,

        "mean_similarity":

            float(
                similarities.mean()
            ),

        "median_similarity":

            float(
                similarities.median()
            ),

        "std_similarity":

            float(
                similarities.std()
            ),

        "min_similarity":

            float(
                similarities.min()
            ),

        "max_similarity":

            float(
                similarities.max()
            ),

        "p90":

            float(
                similarities.quantile(
                    0.90
                )
            ),

        "p95":

            float(
                similarities.quantile(
                    0.95
                )
            ),

        "p99":

            float(
                similarities.quantile(
                    0.99
                )
            ),

    }

    return stats

def save_hard_negative_statistics(
    stats,
):
    """
    Save statistics JSON.
    """

    output = {

        **stats,

        "created_at":

            datetime.now().isoformat(),

    }

    with open(

        HARD_NEGATIVE_STATISTICS_FILE,

        "w",

        encoding="utf-8",

    ) as f:

        json.dump(

            output,

            f,

            indent=4,

        )

