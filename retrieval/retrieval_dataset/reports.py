"""
Retrieval Dataset Reports

Author: Nguyen Khanh
"""

import json
from datetime import datetime

from retrieval.configs import (
    DATASET_INFO_FILE,
)

def create_dataset_report(
    statistics,
):
    """
    Create retrieval dataset report.
    """

    report = {

        "created_at":
            datetime.now().isoformat(),

        "total_pairs":
            statistics["total_pairs"],

        "positive_pairs":
            statistics["positive_pairs"],

        "negative_pairs":
            statistics["negative_pairs"],

        "total_triplets":
            statistics["total_triplets"],

        "unique_pair_anchors":
            statistics["unique_pair_anchors"],

        "unique_triplet_anchors":
            statistics["unique_triplet_anchors"],

    }

    return report

def save_dataset_report(
    report,
):
    """
    Save dataset report.
    """

    with open(
        DATASET_INFO_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
        )

def generate_dataset_report(
    statistics,
):
    """
    Generate and save report.
    """

    report = (
        create_dataset_report(
            statistics
        )
    )

    save_dataset_report(
        report
    )

    return report