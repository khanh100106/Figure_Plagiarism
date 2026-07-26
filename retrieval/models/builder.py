"""
Paper2Fig-2026 Retrieval Framework

Model Builder

Responsibilities
----------------
- Build retrieval model from configuration.
- Hide model implementation from the training pipeline.

Author
------
Nguyen Khanh
"""

from __future__ import annotations

from retrieval.configs import (
    BACKBONE_NAME,
)

from retrieval.models.model import (
    RetrievalModel,
)


# ============================================================
# Build Model
# ============================================================

def build_model() -> RetrievalModel:
    """
    Build retrieval model.
    """

    #
    # Current implementation
    #

    if BACKBONE_NAME == "dinov2_vitb14":

        return RetrievalModel()

    raise ValueError(
        f"Unsupported backbone: {BACKBONE_NAME}"
    )

