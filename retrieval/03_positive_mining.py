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
    LOG_DIR,
    MINING_INFO_FILE,
    MINING_COMPLETED_FLAG,
    MINING_CHECKPOINT_FILE,
    TOP_K,
    POSITIVE_THRESHOLD,
)
from retrieval.mining.search import (
    search_topk,
)
from retrieval.mining.candidate import (
    generate_topk_candidates,
    save_topk_candidates,
)
from retrieval.mining.positive import (
    filter_positive_pairs,
    save_positive_pairs,
)

from retrieval.mining.statistics import (
    compute_similarity_statistics,
    save_similarity_statistics,
    evaluate_thresholds,
    save_threshold_analysis,
    recommend_threshold,
)

from retrieval.mining.visualization import (
    plot_similarity_histogram,
    plot_similarity_cdf,
    plot_threshold_curve,
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

    distances, indices = search_topk(

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
    # Similarity Statistics
    # ---------------------------------------------------------

    stats = compute_similarity_statistics(
        candidates
    )

    save_similarity_statistics(
        stats
    )

    recommended_threshold = recommend_threshold(
        stats
    )

    logger.info(
        f"Recommended threshold: "
        f"{recommended_threshold:.4f}"
    )

    # ---------------------------------------------------------
    # Threshold Analysis
    # ---------------------------------------------------------

    threshold_df = evaluate_thresholds(
        candidates
    )

    save_threshold_analysis(
        threshold_df
    )

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------

    plot_similarity_histogram(
        candidates
    )

    plot_similarity_cdf(
        candidates
    )

    plot_threshold_curve(
        threshold_df
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

