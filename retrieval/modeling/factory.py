"""
Paper2Fig-2026 Retrieval Framework

Model Factory
-------------

Build retrieval models.

Author
------
Nguyen Khanh
"""

from __future__ import annotations

from retrieval.modeling.retrieval_model import (
    RetrievalModel,
)


def build_model() -> RetrievalModel:
    """
    Build retrieval model.

    Returns
    -------
    RetrievalModel
    """

    return RetrievalModel()
