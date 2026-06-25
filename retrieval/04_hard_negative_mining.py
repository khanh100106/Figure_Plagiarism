"""
Module 04

Hard Negative Mining

Author: Nguyen Khanh
"""

import time

from retrieval.configs import (
    LOG_DIR,

    HARD_NEGATIVE_CHECKPOINT_FILE,
    HARD_NEGATIVE_COMPLETED_FLAG,
)

from retrieval.logger import (
    setup_logger,
)

from retrieval.checkpoints import (
    CheckpointManager,
)

from retrieval.hard_negative.loader import (
    load_candidates,
    load_positive_pairs,
    validate_inputs,
)

from retrieval.hard_negative.filter import (
    build_hard_negative_table,
)

from retrieval.hard_negative.saver import (
    save_all_outputs,
)

from retrieval.hard_negative.statistics import (
    compute_hard_negative_statistics,
    save_hard_negative_statistics,
)

from retrieval.hard_negative.reports import (
    save_hard_negative_info,
)

from retrieval.hard_negative.visualization import (
    plot_similarity_histogram,
    plot_negative_type_distribution,
    plot_similarity_by_type,
)

def main():

    start_time = time.time()

    logger = setup_logger(
        LOG_DIR / "hard_negative.log"
    )

    logger.info("=" * 60)
    logger.info(
        "Paper2Fig-2026 Hard Negative Mining"
    )
    logger.info("=" * 60)

    checkpoint = CheckpointManager(
        HARD_NEGATIVE_CHECKPOINT_FILE
    )

    candidates = load_candidates(
        logger
    )

    positives = load_positive_pairs(
        logger
    )

    validate_inputs(
        candidates,
        positives,
    )

    hard_negatives = (
        build_hard_negative_table(
            candidates,
            positives,
            logger,
        )
    )

    logger.info(
        f"Final hard negatives : "
        f"{len(hard_negatives):,}"
    )

    save_all_outputs(
        hard_negatives,
        logger,
    )

    stats = (
        compute_hard_negative_statistics(
            hard_negatives
        )
    )

    save_hard_negative_statistics(
        stats
    )

    save_hard_negative_info(
        hard_negatives
    )

    plot_similarity_histogram(
        hard_negatives
    )

    plot_negative_type_distribution(
        hard_negatives
    )

    plot_similarity_by_type(
        hard_negatives
    )

    checkpoint.save(
        completed=True
    )

    HARD_NEGATIVE_COMPLETED_FLAG.touch()

    elapsed = (
        time.time()
        - start_time
    )

    logger.info("=" * 60)

    logger.info(
        f"Hard negatives : "
        f"{len(hard_negatives):,}"
    )

    logger.info(
        f"Finished in "
        f"{elapsed:.2f} seconds"
    )

    logger.info("=" * 60)

if __name__ == "__main__":

    main()