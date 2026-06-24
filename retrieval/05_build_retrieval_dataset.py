"""
Module 05

Build Retrieval Dataset

Author: Nguyen Khanh
"""

import json
import random
import time
from datetime import datetime

import pandas as pd

from retrieval.configs import (
    LOG_DIR,
    EMBEDDING_DIR,
    MINING_DIR,
    RETRIEVAL_TRIPLET_FILE,
    DATASET_INFO_FILE,
    DATASET_COMPLETED_FLAG,
    DATASET_CHECKPOINT_FILE,
    MAX_POSITIVES_PER_ANCHOR,
    MAX_NEGATIVES_PER_ANCHOR,
    SEED,
)

from retrieval.logger import (
    setup_logger,
)

from retrieval.checkpoints import (
    CheckpointManager,
)

# ============================================================
# Load Positive Pairs
# ============================================================

def load_positive_pairs(
    logger,
):
    """
    Load positive pairs.
    """

    file = MINING_DIR / "positive_pairs.csv"

    logger.info(
        "Loading positive pairs..."
    )

    positives = pd.read_csv(file)

    logger.info(
        f"Positive pairs : {len(positives):,}"
    )

    return positives

# ============================================================
# Load Hard Negatives
# ============================================================

def load_hard_negative_pairs(
    logger,
):
    """
    Load hard negative pairs.
    """

    file = MINING_DIR / "hard_negative_pairs.csv"

    logger.info(
        "Loading hard negatives..."
    )

    negatives = pd.read_csv(file)

    logger.info(
        f"Hard negatives : {len(negatives):,}"
    )

    return negatives

# ============================================================
# Load Embedding Index
# ============================================================

def load_embedding_index(
    logger,
):
    """
    Load embedding index.
    """

    file = EMBEDDING_DIR / "embedding_index.csv"

    logger.info(
        "Loading embedding index..."
    )

    embedding_index = pd.read_csv(file)

    logger.info(
        f"Figures : {len(embedding_index):,}"
    )

    return embedding_index

# ============================================================
# Validate Inputs
# ============================================================

def validate_inputs(
    positives,
    negatives,
    embedding_index,
):

    if len(positives) == 0:
        raise RuntimeError(
            "No positive pairs found."
        )

    if len(negatives) == 0:
        raise RuntimeError(
            "No hard negatives found."
        )

    if len(embedding_index) == 0:
        raise RuntimeError(
            "Embedding index is empty."
        )

# ============================================================
# Build Lookup Tables
# ============================================================

def build_lookup_tables(
    positives,
    negatives,
    embedding_index,
    logger,
):
    """
    Build lookup tables for retrieval dataset generation.
    """

    logger.info(
        "Building lookup tables..."
    )

    # --------------------------------------------------------
    # Figure lookup
    # --------------------------------------------------------

    figure_lookup = {}

    for _, row in embedding_index.iterrows():

        figure_lookup[
            row["figure_id"]
        ] = row.to_dict()

    # ============================================================
    # Positive lookup
    # ============================================================

    positive_lookup = {}

    for _, row in positives.iterrows():
        anchor = row["query_figure_id"]

        positive_lookup.setdefault(
            anchor,
            []
        ).append(
            {
                "candidate": row["candidate_figure_id"],
                "similarity": row["similarity"],
                "rank": row["rank"],
            }
        )

    # ============================================================
    # Negative lookup
    # ============================================================

    negative_lookup = {}

    for _, row in negatives.iterrows():
        anchor = row["query_figure_id"]

        negative_lookup.setdefault(
            anchor,
            []
        ).append(
            {
                "candidate": row["candidate_figure_id"],
                "similarity": row["similarity"],
                "rank": row["rank"],
                "negative_type": row["negative_type"],
            }
        )

    logger.info(
        f"Anchors with positives : {len(positive_lookup):,}"
    )

    logger.info(
        f"Anchors with negatives : {len(negative_lookup):,}"
    )

    return (
        figure_lookup,
        positive_lookup,
        negative_lookup,
    )

# ============================================================
# Generate Triplets
# ============================================================

def generate_triplets(
    figure_lookup,
    positive_lookup,
    negative_lookup,
    logger,
):
    """
    Generate balanced retrieval triplets.
    """

    logger.info(
        "Generating retrieval triplets..."
    )

    random.seed(SEED)

    triplets = []

    # --------------------------------------------------------
    # Iterate anchors
    # --------------------------------------------------------

    for anchor in positive_lookup.keys():

        if anchor not in negative_lookup:
            continue

        positives = sorted(
            positive_lookup[anchor],
            key=lambda x: x["similarity"],
            reverse=True,
        )

        negatives = sorted(
            negative_lookup[anchor],
            key=lambda x: x["similarity"],
            reverse=True,
        )

        if (
            len(positives) == 0
            or
            len(negatives) == 0
        ):
            continue

        # ---------------------------------------------
        # Limit positives
        # ---------------------------------------------

        positives = positives[
            :MAX_POSITIVES_PER_ANCHOR
        ]

        # ---------------------------------------------
        # Limit negatives
        # ---------------------------------------------

        negatives = negatives[
            :MAX_NEGATIVES_PER_ANCHOR
        ]


        # --------------------------------------------------------
        # Iterate anchors
        # --------------------------------------------------------

        for anchor in positive_lookup.keys():

            if anchor not in negative_lookup:
                continue

            positives = sorted(
                positive_lookup[anchor],
                key=lambda x: x["similarity"],
                reverse=True,
            )

            negatives = sorted(
                negative_lookup[anchor],
                key=lambda x: x["similarity"],
                reverse=True,
            )

            # ---------------------------------------------
            # Keep Top-K Positives
            # ---------------------------------------------

            positives = positives[
                :MAX_POSITIVES_PER_ANCHOR
            ]

            # ---------------------------------------------
            # Keep Top-K Hard Negatives
            # ---------------------------------------------

            negatives = negatives[
                :MAX_NEGATIVES_PER_ANCHOR
            ]

            if (
                    len(positives) == 0
                    or
                    len(negatives) == 0
            ):
                continue

            # ---------------------------------------------
            # Cartesian Product
            # ---------------------------------------------

            for positive in positives:

                positive_id = positive["candidate"]

                for negative in negatives:

                    negative_id = negative["candidate"]

                    if (
                            anchor not in figure_lookup
                            or
                            positive_id not in figure_lookup
                            or
                            negative_id not in figure_lookup
                    ):
                        continue

                    triplets.append(

                        {

                            "anchor_id":
                                anchor,

                            "positive_id":
                                positive_id,

                            "negative_id":
                                negative_id,

                            "positive_similarity":
                                positive["similarity"],

                            "negative_similarity":
                                negative["similarity"],

                            "anchor_path":
                                figure_lookup[anchor][
                                    "figure_image_path"
                                ],

                            "positive_path":
                                figure_lookup[positive_id][
                                    "figure_image_path"
                                ],

                            "negative_path":
                                figure_lookup[negative_id][
                                    "figure_image_path"
                                ],

                        }

                    )

    triplets = pd.DataFrame(
        triplets
    )

    logger.info(
        f"Triplets : {len(triplets):,}"
    )

    return triplets

# ============================================================
# Save Triplets
# ============================================================

def save_triplets(
    triplets,
    logger,
):
    """
    Save retrieval triplets.
    """

    triplets.to_csv(

        RETRIEVAL_TRIPLET_FILE,

        index=False,

    )

    logger.info(

        f"Saved : {RETRIEVAL_TRIPLET_FILE}"

    )

# ============================================================
# Save Dataset Information
# ============================================================

def save_dataset_info(
    triplets,
):
    """
    Save dataset statistics.
    """

    info = {

        "num_triplets": len(
            triplets
        ),

        "max_positive_per_anchor":
            MAX_POSITIVES_PER_ANCHOR,

        "max_negative_per_anchor":
            MAX_NEGATIVES_PER_ANCHOR,

        "created_at":
            datetime.now().isoformat(),

    }

    with open(

        DATASET_INFO_FILE,

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
        LOG_DIR / "build_retrieval_dataset.log"
    )

    logger.info("=" * 60)
    logger.info("Paper2Fig-2026 Retrieval Dataset Builder")
    logger.info("=" * 60)

    # ---------------------------------------------------------
    # Checkpoint
    # ---------------------------------------------------------

    checkpoint = CheckpointManager(
        DATASET_CHECKPOINT_FILE
    )

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    positives = load_positive_pairs(
        logger
    )

    negatives = load_hard_negative_pairs(
        logger
    )

    embedding_index = load_embedding_index(
        logger
    )

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------

    validate_inputs(
        positives,
        negatives,
        embedding_index,
    )

    # ---------------------------------------------------------
    # Lookup
    # ---------------------------------------------------------

    (
        figure_lookup,
        positive_lookup,
        negative_lookup,
    ) = build_lookup_tables(

        positives,
        negatives,
        embedding_index,
        logger,

    )

    # ---------------------------------------------------------
    # Generate Triplets
    # ---------------------------------------------------------

    triplets = generate_triplets(

        figure_lookup,
        positive_lookup,
        negative_lookup,
        logger,

    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    save_triplets(
        triplets,
        logger,
    )

    save_dataset_info(
        triplets,
    )

    # ---------------------------------------------------------
    # Checkpoint
    # ---------------------------------------------------------

    checkpoint.save(
        completed=True,
    )

    DATASET_COMPLETED_FLAG.touch()

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
