"""
Paper2Fig-2026 Retrieval Framework

Head Factory
------------

Build embedding heads.

Author
------
Nguyen Khanh
"""

from __future__ import annotations

from retrieval.configs import (
    HEAD,
    PROJECTION_DIM,
    DROPOUT,
)

from retrieval.modeling.heads.heads_base import (
    BaseHead,
)

from retrieval.modeling.heads.projection import (
    ProjectionHead,
)


def build_head(
    input_dim: int,
) -> BaseHead:
    """
    Build embedding head.

    Parameters
    ----------
    input_dim : int

    Returns
    -------
    BaseHead
    """

    if HEAD == "projection":

        return ProjectionHead(
            input_dim=input_dim,
            embedding_dim=PROJECTION_DIM,
            dropout=DROPOUT,
        )

    raise ValueError(
        f"Unsupported head: {HEAD}"
    )