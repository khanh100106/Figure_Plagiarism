"""
Paper2Fig-2026
Base Task
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import torch
import torch.nn as nn


class BaseTask(ABC):
    """
    Base task interface.

    Every training task must implement

    - forward
    - compute_loss
    - compute_metrics

    Trainer never needs to know task details.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        criterion: nn.Module,
    ) -> None:

        self.model = model

        self.criterion = criterion

    @abstractmethod
    def forward(
        self,
        batch: dict,
    ) -> dict:
        """
        Forward one batch.
        """

    @abstractmethod
    def compute_loss(
        self,
        outputs: dict,
    ) -> dict:
        """
        Compute task loss.
        """

    @abstractmethod
    def compute_metrics(
        self,
        outputs: dict,
    ) -> dict:
        """
        Compute metrics from outputs.
        """