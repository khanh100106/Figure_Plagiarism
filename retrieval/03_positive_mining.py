"""
Module 03

Positive Pair Mining

Author: Nguyen Khanh
"""

import json
import time
from datetime import datetime

import faiss
import numpy as np
import pandas as pd

from retrieval.configs import (
    EMBEDDING_DIR,
    FAISS_DIR,
    MINING_DIR,
    LOG_DIR,

    POSITIVE_PAIR_FILE,
    MINING_INFO_FILE,
    MINING_COMPLETED_FLAG,
    MINING_CHECKPOINT_FILE,

    TOP_K,
    POSITIVE_THRESHOLD,
    SEARCH_BATCH_SIZE,
    TOPK_CANDIDATE_FILE,
)

from retrieval.logger import (
    setup_logger,
)

from retrieval.checkpoints import (
    CheckpointManager,
)

# ============================================================
# Load Data
# ============================================================

def load_data(
    logger,
):
    """
    Load embeddings, metadata and FAISS index.
    """

    embeddings = np.load(

        EMBEDDING_DIR /
        "embeddings.npy"

    ).astype(np.float32)

    metadata = pd.read_csv(

        EMBEDDING_DIR /
        "embedding_index.csv"

    )

    index = faiss.read_index(

        str(
            FAISS_DIR /
            "index.faiss"
        )

    )

    logger.info(

        f"Embeddings : {embeddings.shape}"

    )

    logger.info(

        f"Metadata   : {len(metadata):,}"

    )

    logger.info(

        f"Index size : {index.ntotal:,}"

    )

    return (

        embeddings,

        metadata,

        index,

    )

# ============================================================
# Validation
# ============================================================

def validate_inputs(

    embeddings,

    metadata,

    index,

):

    if len(embeddings) != len(metadata):

        raise ValueError(

            "Embedding / metadata mismatch."

        )

    if index.ntotal != len(metadata):

        raise ValueError(

            "FAISS index mismatch."

        )



# ============================================================
# Generate Top-K Candidates
# ============================================================

def generate_topk_candidates(
    distances,
    indices,
    metadata,
    logger,
):
    """
    Generate Top-K candidates from FAISS search.
    """

    logger.info(
        "Generating Top-K candidates..."
    )

    metadata_records = metadata.to_dict(
        "records"
    )

    candidates = []

    for query_idx in range(
        len(metadata_records)
    ):

        query = metadata_records[
            query_idx
        ]

        neighbors = indices[
            query_idx
        ]

        scores = distances[
            query_idx
        ]

        rank = 1

        for candidate_idx, score in zip(

            neighbors[1:],

            scores[1:],

        ):

            if query_idx >= candidate_idx:
                continue

            candidate = metadata_records[
                candidate_idx
            ]

            candidates.append(

                {

                    "query_index":
                        query_idx,

                    "candidate_index":
                        candidate_idx,

                    "query_figure_id":
                        query["figure_id"],

                    "candidate_figure_id":
                        candidate["figure_id"],

                    "query_paper_id":
                        query["paper_id"],

                    "candidate_paper_id":
                        candidate["paper_id"],

                    "query_field":
                        query["field"],

                    "candidate_field":
                        candidate["field"],

                    "similarity":
                        float(score),

                    "rank":
                        rank,

                }

            )

            rank += 1

    candidates = pd.DataFrame(
        candidates
    )

    logger.info(

        f"Candidates : {len(candidates):,}"

    )

    return candidates

# ============================================================
# Save Top-K Candidates
# ============================================================

def save_topk_candidates(
    candidates,
    logger,
):
    """
    Save Top-K candidates.
    """

    candidates.to_csv(

        TOPK_CANDIDATE_FILE,

        index=False,

    )

    logger.info(

        f"Saved : {TOPK_CANDIDATE_FILE}"

    )

# ============================================================
# Filter Positive Pairs
# ============================================================

def filter_positive_pairs(
    candidates,
    logger,
):
    """
    Filter positive pairs using similarity threshold.
    """

    logger.info(
        "Filtering positive pairs..."
    )

    positive_pairs = candidates[

        candidates["similarity"]

        >= POSITIVE_THRESHOLD

    ].copy()

    logger.info(

        f"Positive pairs : {len(positive_pairs):,}"

    )

    return positive_pairs

# ============================================================
# Save Positive Pairs
# ============================================================

def save_positive_pairs(
    positive_pairs,
    logger,
):

    positive_pairs.to_csv(

        POSITIVE_PAIR_FILE,

        index=False,

    )

    logger.info(

        f"Saved : {POSITIVE_PAIR_FILE}"

    )

# ============================================================
# Save Mining Information
# ============================================================

def save_mining_info(
    positive_pairs,
):
    """
    Save mining information.
    """

    info = {

        "positive_pairs":

            len(positive_pairs),

        "threshold":

            POSITIVE_THRESHOLD,

        "top_k":

            TOP_K,

        "created_at":

            datetime.now().isoformat(),

    }

    with open(

        MINING_INFO_FILE,

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
        LOG_DIR / "positive_mining.log"
    )

    logger.info("=" * 60)
    logger.info("Paper2Fig-2026 Positive Pair Mining")
    logger.info("=" * 60)

    # ---------------------------------------------------------
    # Checkpoint
    # ---------------------------------------------------------

    checkpoint = CheckpointManager(
        MINING_CHECKPOINT_FILE
    )

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    embeddings, metadata, index = load_data(
        logger
    )

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------

    validate_inputs(
        embeddings,
        metadata,
        index,
    )

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    distances, indices = batch_search(

        embeddings,

        index,

        logger,

    )

    # ---------------------------------------------------------
    # Generate Top-K
    # ---------------------------------------------------------

    candidates = generate_topk_candidates(

        distances,

        indices,

        metadata,

        logger,

    )

    # ---------------------------------------------------------
    # Save Top-K
    # ---------------------------------------------------------

    save_topk_candidates(

        candidates,

        logger,

    )

    # ---------------------------------------------------------
    # Positive Mining
    # ---------------------------------------------------------

    positive_pairs = filter_positive_pairs(

        candidates,

        logger,

    )

    # ---------------------------------------------------------
    # Save Positive Pairs
    # ---------------------------------------------------------

    save_positive_pairs(

        positive_pairs,

        logger,

    )

    # ---------------------------------------------------------
    # Save Mining Information
    # ---------------------------------------------------------

    save_mining_info(

        positive_pairs,

    )

    # ---------------------------------------------------------
    # Save Checkpoint
    # ---------------------------------------------------------

    checkpoint.save(

        completed=True,

    )

    MINING_COMPLETED_FLAG.touch()

    # ---------------------------------------------------------
    # Finish
    # ---------------------------------------------------------

    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info(
        f"Positive pairs : {len(positive_pairs):,}"
    )
    logger.info(
        f"Finished in {elapsed:.2f} seconds"
    )
    logger.info("=" * 60)


if __name__ == "__main__":

    main()

