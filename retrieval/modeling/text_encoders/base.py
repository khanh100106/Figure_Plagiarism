"""
Paper2Fig-2026 Retrieval Framework

Base Text Encoder
-----------------

Abstract interface for all text encoders.

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


class BaseTextEncoder(
    nn.Module,
    ABC,
):
    """
    Abstract text encoder.
    """

    def __init__(
        self,
    ) -> None:

        super().__init__()

    @property
    @abstractmethod
    def feature_dim(
        self,
    ) -> int:
        """
        Output feature dimension.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(
        self,
    ) -> str:
        """
        Encoder name.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        captions: list[str],
    ) -> torch.Tensor:
        """
        Encode captions.

        Parameters
        ----------
        captions : list[str]

        Returns
        -------
        torch.Tensor
            Caption features.
        """
        raise NotImplementedError