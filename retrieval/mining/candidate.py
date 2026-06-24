"""
FAISS candidates.

Author: Nguyen Khanh
Project: Paper2Fig-2026
"""

import pandas as pd

from retrieval.configs import (
    TOPK_CANDIDATE_FILE,
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