"""
Paper2Fig-2026 Retrieval Dataset Builder

Author: Nguyen Khanh
"""

import time

from retrieval.logger import setup_logger

from retrieval.configs import (
    LOG_DIR,
    DATASET_COMPLETED_FLAG,
)

from retrieval.retrieval_dataset.loader import (
    load_positive_pairs,
    load_hard_negatives,
)

from retrieval.retrieval_dataset.sampler import (
    sample_training_pairs,
)

from retrieval.retrieval_dataset.builder import (
    build_pair_dataset,
    build_triplet_dataset,
)

from retrieval.retrieval_dataset.saver import (
    save_all_outputs,
)

from retrieval.retrieval_dataset.statistics import (
    compute_retrieval_statistics,
    save_retrieval_statistics,
    log_retrieval_statistics,
)

from retrieval.retrieval_dataset.reports import (
    generate_dataset_report,
)

def main():

    start_time = time.time()

    logger = setup_logger(
        LOG_DIR / "retrieval_dataset.log"
    )

    logger.info(
        "=" * 60
    )

    logger.info(
        "Paper2Fig-2026 Retrieval Dataset Builder"
    )

    logger.info(
        "=" * 60
    )

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    positives = (
        load_positive_pairs(
            logger
        )
    )

    negatives = (
        load_hard_negatives(
            logger
        )
    )

    # --------------------------------------------------
    # Sampling
    # --------------------------------------------------

    sampled_positives, sampled_negatives = (
        sample_training_pairs(
            positives,
            negatives,
        )
    )

    logger.info(
        f"Sampled positives : "
        f"{len(sampled_positives):,}"
    )

    logger.info(
        f"Sampled negatives : "
        f"{len(sampled_negatives):,}"
    )

    # --------------------------------------------------
    # Build
    # --------------------------------------------------

    pair_dataset = (
        build_pair_dataset(
            sampled_positives,
            sampled_negatives,
        )
    )

    triplet_dataset = (
        build_triplet_dataset(
            sampled_positives,
            sampled_negatives,
        )
    )

    logger.info(
        f"Pair dataset : "
        f"{len(pair_dataset):,}"
    )

    logger.info(
        f"Triplet dataset : "
        f"{len(triplet_dataset):,}"
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    save_all_outputs(
        pair_dataset,
        triplet_dataset,
        logger,
    )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    statistics = (
        compute_retrieval_statistics(
            pair_dataset,
            triplet_dataset,
        )
    )

    save_retrieval_statistics(
        statistics
    )

    log_retrieval_statistics(
        statistics,
        logger,
    )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    generate_dataset_report(
        statistics
    )

    # --------------------------------------------------
    # Completed Flag
    # --------------------------------------------------

    DATASET_COMPLETED_FLAG.touch()

    elapsed = (
        time.time()
        - start_time
    )

    logger.info(
        "=" * 60
    )

    logger.info(
        f"Finished in "
        f"{elapsed:.2f} seconds"
    )

    logger.info(
        "=" * 60
    )

if __name__ == "__main__":
    main()
