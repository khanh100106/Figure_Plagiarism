"""
FAISS search utilities.

Author: Nguyen Khanh
Project: Paper2Fig-2026
"""

import numpy as np

from retrieval.configs import (
    SEARCH_BATCH_SIZE,
    TOP_K,
)


def search_topk(
    embeddings,
    index,
    logger,
):
    """
    Search Top-K nearest neighbours for all embeddings.

    Parameters
    ----------
    embeddings : np.ndarray

    index : faiss.Index

    logger : logging.Logger

    Returns
    -------
    distances : np.ndarray

    indices : np.ndarray
    """

    logger.info(
        f"Searching Top-{TOP_K} nearest neighbours..."
    )

    all_distances = []

    all_indices = []

    total = len(
        embeddings
    )

    for start in range(
        0,
        total,
        SEARCH_BATCH_SIZE,
    ):

        end = min(
            start + SEARCH_BATCH_SIZE,
            total,
        )

        batch = embeddings[
            start:end
        ]

        distances, indices = index.search(

            batch,

            TOP_K + 1,

        )

        all_distances.append(
            distances
        )

        all_indices.append(
            indices
        )

        logger.debug(
            f"Processed {end:,}/{total:,}"
        )

    distances = np.vstack(
        all_distances
    )

    indices = np.vstack(
        all_indices
    )

    logger.info(
        "Nearest neighbour search completed."
    )

    return (
        distances,
        indices,
    )