"""
Paper2Fig-2026 Retrieval Framework

Base Head
---------

Abstract interface for embedding heads.

Author
------
Nguyen Khanh
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)

import torch
from torch import nn


class BaseHead(
    nn.Module,
    ABC,
):
    """
    Abstract embedding head.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
    ) -> None:

        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

    @property
    @abstractmethod
    def name(
        self,
    ) -> str:
        """
        Head name.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert features into retrieval embeddings.
        """
        raise NotImplementedError

    @property
    def output_dim(
            self,
    ) -> int:
        """
        Output embedding dimension.
        """
        return self.embedding_dim