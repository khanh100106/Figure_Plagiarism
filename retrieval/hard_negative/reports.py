"""
Hard Negative Mining Reports

Author: Nguyen Khanh
"""

import json
from datetime import datetime

from retrieval.configs import (
    HARD_NEGATIVE_INFO_FILE,
    HARD_NEGATIVE_MIN,
    HARD_NEGATIVE_MAX,
)


def save_hard_negative_info(
    hard_negatives,
):
    """
    Save hard negative mining summary.
    """

    info = {

        "hard_negatives":
            len(hard_negatives),

        "intra_paper":
            int(
                (
                    hard_negatives["negative_type"]
                    == "intra_paper"
                ).sum()
            ),

        "inter_paper":
            int(
                (
                    hard_negatives["negative_type"]
                    == "inter_paper"
                ).sum()
            ),

        "hard_negative_min":
            HARD_NEGATIVE_MIN,

        "hard_negative_max":
            HARD_NEGATIVE_MAX,

        "created_at":
            datetime.now().isoformat(),

    }

    with open(
        HARD_NEGATIVE_INFO_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            info,
            f,
            indent=4,
        )