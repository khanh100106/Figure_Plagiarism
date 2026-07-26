"""
Paper2Fig-2026 Retrieval Framework

Training Entry
--------------

Entry point for model training.

Author
------
Nguyen Khanh
"""

from __future__ import annotations

from retrieval.configs import (
    NUM_EPOCHS,
    RESUME_TRAINING,
)

from retrieval.training.builder import (
    build_trainer,
)

def main() -> None:
    """
    Training entry.
    """

    trainer = build_trainer()

    # ------------------------------------------
    # Resume
    # ------------------------------------------

    if RESUME_TRAINING:

        try:

            trainer.resume()

        except FileNotFoundError:

            print(
                "Checkpoint not found. "
                "Training from scratch."
            )

    trainer.fit(
        NUM_EPOCHS,
    )

if __name__ == "__main__":
    main()