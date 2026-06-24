"""
Module 04

Hard Negative Mining

Author: Nguyen Khanh
"""

import json
import time
from datetime import datetime
import numpy as np
import pandas as pd

from retrieval.configs import (
    LOG_DIR,
    HARD_NEGATIVE_MIN,
    HARD_NEGATIVE_MAX,
    TOPK_CANDIDATE_FILE,
    HARD_NEGATIVE_PAIR_FILE,
    HARD_NEGATIVE_INFO_FILE,
    HARD_NEGATIVE_COMPLETED_FLAG,
    HARD_NEGATIVE_CHECKPOINT_FILE,
)

from retrieval.logger import (
    setup_logger,
)

from retrieval.checkpoints import (
    CheckpointManager,
)

# ============================================================
# Load Candidates
# ============================================================

def load_candidates(
    logger,
):
    """
    Load Top-K candidates.
    """

    logger.info(
        "Loading Top-K candidates..."
    )

    candidates = pd.read_csv(
        TOPK_CANDIDATE_FILE
    )

    logger.info(
        f"Candidates : {len(candidates):,}"
    )

    return candidates


# ============================================================
# Validate Inputs
# ============================================================

def validate_inputs(
    candidates,
):

    if len(candidates) == 0:

        raise RuntimeError(
            "No candidates found."
        )

# ============================================================
# Filter Hard Negatives
# ============================================================

def filter_hard_negatives(
    candidates,
    logger,
):
    """
    Filter hard negative pairs from Top-K candidates.
    """

    logger.info(
        "Filtering hard negatives..."
    )

    hard_negatives = candidates[
        (
            candidates["similarity"] >= HARD_NEGATIVE_MIN
        )
        &
        (
            candidates["similarity"] <= HARD_NEGATIVE_MAX
        )
    ].copy()

    hard_negatives["negative_type"] = np.where(
        hard_negatives["query_paper_id"]
        ==
        hard_negatives["candidate_paper_id"],
        "intra_paper",
        "inter_paper",
    )

    num_intra = len(
        hard_negatives[
            hard_negatives["negative_type"] == "intra_paper"
        ]
    )

    num_inter = len(
        hard_negatives[
            hard_negatives["negative_type"] == "inter_paper"
        ]
    )

    logger.info(
        f"Hard negatives : {len(hard_negatives):,}"
    )

    logger.info(
        f"Intra-paper : {num_intra:,}"
    )

    logger.info(
        f"Inter-paper : {num_inter:,}"
    )

    return hard_negatives


# ============================================================
# Save Hard Negatives
# ============================================================

def save_hard_negatives(
    hard_negatives,
    logger,
):
    """
    Save hard negative pairs.
    """

    hard_negatives.to_csv(

        HARD_NEGATIVE_PAIR_FILE,

        index=False,

    )

    logger.info(

        f"Saved : {HARD_NEGATIVE_PAIR_FILE}"

    )

# ============================================================
# Save Hard Negative Information
# ============================================================

def save_hard_negative_info(
    hard_negatives,
):
    """
    Save mining statistics.
    """
    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    num_intra = len(
        hard_negatives[
            hard_negatives["negative_type"] == "intra_paper"
            ]
    )

    num_inter = len(
        hard_negatives[
            hard_negatives["negative_type"] == "inter_paper"
            ]
    )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    info = {

        "num_hard_negatives": len(
            hard_negatives
        ),

        "hard_negative_min":
            HARD_NEGATIVE_MIN,

        "hard_negative_max":
            HARD_NEGATIVE_MAX,

        "num_intra_paper":
            num_intra,

        "num_inter_paper":
            num_inter,

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

# ============================================================
# Main
# ============================================================

def main():

    start_time = time.time()

    # ---------------------------------------------------------
    # Logger
    # ---------------------------------------------------------

    logger = setup_logger(
        LOG_DIR / "hard_negative_mining.log"
    )

    logger.info("=" * 60)
    logger.info("Paper2Fig-2026 Hard Negative Mining")
    logger.info("=" * 60)

    # ---------------------------------------------------------
    # Checkpoint
    # ---------------------------------------------------------

    checkpoint = CheckpointManager(
        HARD_NEGATIVE_CHECKPOINT_FILE
    )

    # ---------------------------------------------------------
    # Load candidates
    # ---------------------------------------------------------

    candidates = load_candidates(
        logger
    )

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------

    validate_inputs(
        candidates
    )

    # ---------------------------------------------------------
    # Filter
    # ---------------------------------------------------------

    hard_negatives = filter_hard_negatives(
        candidates,
        logger,
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    save_hard_negatives(
        hard_negatives,
        logger,
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    save_hard_negative_info(
        hard_negatives,
    )

    # ---------------------------------------------------------
    # Checkpoint
    # ---------------------------------------------------------

    checkpoint.save(
        completed=True,
    )

    HARD_NEGATIVE_COMPLETED_FLAG.touch()

    # ---------------------------------------------------------
    # Finish
    # ---------------------------------------------------------

    elapsed = time.time() - start_time

    logger.info("=" * 60)

    logger.info(

        f"Finished in {elapsed:.2f} seconds"

    )

    logger.info("=" * 60)


if __name__ == "__main__":

    main()