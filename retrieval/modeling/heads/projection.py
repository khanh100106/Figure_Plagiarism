"""
Paper2Fig-2026 Retrieval Framework

Projection Head
---------------

Projection head for retrieval learning.

Author
------
Nguyen Khanh
"""

from __future__ import annotations

import torch
from torch import nn

from retrieval.modeling.heads.base import BaseHead


class ProjectionHead(BaseHead):
    """
    MLP projection head.
    """

    def __init__(
            self,
            input_dim: int,
            embedding_dim: int,
            hidden_dim: int | None = None,
            dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
        )

        if hidden_dim is None:
            hidden_dim = input_dim


        self.layers = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout,
            ),

            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),

        )


    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.layers(
            features,
        )

        embeddings = nn.functional.normalize(
            embeddings,
            p=2,
            dim=1,
        )

        return embeddings

    @property
    def name(
            self,
    ) -> str:
        """
        Head name.
        """

        return "projection"

    def __repr__(
            self,
    ) -> str:
        """
        String representation.
        """

        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"embedding_dim={self.embedding_dim})"
        )
