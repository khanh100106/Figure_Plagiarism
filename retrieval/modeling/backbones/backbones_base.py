"""
Paper2Fig-2026 Retrieval Framework

Base Backbone
-------------

Abstract interface for all image backbones.

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


class BaseBackbone(
    nn.Module,
    ABC,
):
    """
    Abstract backbone interface.
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
        Feature dimension produced by the backbone.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract image features.

        Parameters
        ----------
        images : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        raise NotImplementedError

    def freeze(
            self,
    ) -> None:
        """
        Freeze all backbone parameters.
        """

        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze(
            self,
    ) -> None:
        """
        Unfreeze all backbone parameters.
        """

        for parameter in self.parameters():
            parameter.requires_grad = True

    @property
    def num_parameters(
            self,
    ) -> int:
        """
        Number of trainable parameters.
        """

        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )

