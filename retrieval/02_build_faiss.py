"""
Module 02

Build FAISS index for Paper2Fig-2026.

Author: Nguyen Khanh
"""

import json
import time
from datetime import datetime

import faiss
import numpy as np

from retrieval.configs import (
    EMBEDDING_DIR,
    LOG_DIR,
    EMBEDDING_DIM,
    FAISS_INDEX_FILE,
    FAISS_INFO_FILE,
    FAISS_COMPLETED_FLAG,
    FAISS_CHECKPOINT_FILE,
    FAISS_INDEX_TYPE,
    FAISS_METRIC,
)

from retrieval.logger import (
    setup_logger,
)

from retrieval.checkpoints import (
    CheckpointManager,
)

def load_embeddings(logger):
    """
    Load extracted embeddings.
    """

    embedding_file = (
        EMBEDDING_DIR / "embeddings.npy"
    )

    embeddings = np.load(
        embedding_file
    ).astype(np.float32)

    logger.info(
        f"Loaded embeddings: {embeddings.shape}"
    )

    return embeddings

def validate_embeddings(
    embeddings,
    logger,
):
    """
    Validate embedding matrix.
    """

    if embeddings.ndim != 2:
        raise ValueError(
            "Embedding matrix must be 2D."
        )

    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            "Embedding dimension mismatch."
        )

    if np.isnan(embeddings).any():
        raise ValueError(
            "NaN detected in embeddings."
        )

    if np.isinf(embeddings).any():
        raise ValueError(
            "Inf detected in embeddings."
        )

    logger.info(
        "Embedding validation passed."
    )

# ============================================================
# Build FAISS Index
# ============================================================

def build_index(
    embeddings,
    logger,
):
    """
    Build FAISS IndexFlatIP.
    """

    logger.info(
        f"Building {FAISS_INDEX_TYPE}..."
    )

    index = faiss.IndexFlatIP(
        EMBEDDING_DIM
    )

    index.add(
        embeddings
    )

    logger.info(
        f"Indexed vectors : {index.ntotal:,}"
    )

    return index

# ============================================================
# Save FAISS Index
# ============================================================

def save_index(
    index,
    logger,
):
    """
    Save FAISS index.
    """

    faiss.write_index(
        index,
        str(FAISS_INDEX_FILE)
    )

    logger.info(
        f"Index saved : {FAISS_INDEX_FILE}"
    )

# ============================================================
# Save FAISS Information
# ============================================================

def save_faiss_info(
    index,
):
    """
    Save FAISS metadata.
    """

    info = {

        "index_type": FAISS_INDEX_TYPE,

        "metric": FAISS_METRIC,

        "dimension": EMBEDDING_DIM,

        "vectors": index.ntotal,

        "created_at": datetime.now().isoformat(),

        "faiss_version": faiss.__version__,

    }

    with open(
        FAISS_INFO_FILE,
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

    logger = setup_logger(
        LOG_DIR / "build_faiss.log"
    )

    logger.info("=" * 60)
    logger.info("Paper2Fig-2026 FAISS Builder")
    logger.info("=" * 60)

    checkpoint = CheckpointManager(
        FAISS_CHECKPOINT_FILE
    )

    embeddings = load_embeddings(
        logger
    )

    validate_embeddings(
        embeddings,
        logger,
    )

    index = build_index(
        embeddings,
        logger,
    )

    save_index(
        index,
        logger,
    )

    save_faiss_info(
        index,
    )

    checkpoint.save(
        completed=True,
    )

    FAISS_COMPLETED_FLAG.touch()

    elapsed = time.time() - start_time

    logger.info(
        f"Finished in {elapsed:.2f} seconds"
    )


if __name__ == "__main__":

    main()