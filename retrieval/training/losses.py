"""
Paper2Fig-2026 Retrieval Training
Loss Functions
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from retrieval.configs import (
    LOSS_TYPE,
    CONTRASTIVE_MARGIN,
    TRIPLET_MARGIN,
    HYBRID_CONTRASTIVE_WEIGHT,
    HYBRID_TRIPLET_WEIGHT,
)
# ============================================================
# Contrastive Loss
# ============================================================
class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        margin: float = CONTRASTIVE_MARGIN,
    ) -> None:
        super().__init__()
        self.margin = margin
    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        distance = F.pairwise_distance(
            embedding1,
            embedding2,
            p=2,
        )
        positive = (
            labels.float()
            * distance.pow(2)
        )
        negative = (
            (1 - labels.float())
            * torch.clamp(
                self.margin - distance,
                min=0.0,
            ).pow(2)
        )
        loss = (
            positive + negative
        ).mean()
        return loss
# ============================================================
# Triplet Loss
# ============================================================
class TripletLoss(nn.Module):
    def __init__(
        self,
        margin: float = TRIPLET_MARGIN,
    ) -> None:
        super().__init__()
        self.margin = margin
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        positive_distance = F.pairwise_distance(
            anchor,
            positive,
            p=2,
        )
        negative_distance = F.pairwise_distance(
            anchor,
            negative,
            p=2,
        )
        loss = torch.relu(
            positive_distance
            - negative_distance
            + self.margin
        ).mean()
        return (
            loss,
            positive_distance.mean(),
            negative_distance.mean(),
        )
# ============================================================
# Factory
# ============================================================
def build_loss():
    losses = {}
    if LOSS_TYPE in (
        "contrastive",
        "hybrid",
    ):
        losses["contrastive"] = ContrastiveLoss()
    if LOSS_TYPE in (
        "triplet",
        "hybrid",
    ):
        losses["triplet"] = TripletLoss()
    return losses
# ============================================================
# Summary
# ============================================================
def summarize_loss(
) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append(
        "Loss Configuration"
    )
    lines.append("=" * 60)
    lines.append(
        "Loss Type : {}".format(
            LOSS_TYPE,
        )
    )
    lines.append(
        "Contrastive Margin : {}".format(
            CONTRASTIVE_MARGIN,
        )
    )
    lines.append(
        "Triplet Margin : {}".format(
            TRIPLET_MARGIN,
        )
    )
    lines.append(
        "Contrastive Weight : {}".format(
            HYBRID_CONTRASTIVE_WEIGHT,
        )
    )
    lines.append(
        "Triplet Weight : {}".format(
            HYBRID_TRIPLET_WEIGHT,
        )
    )
    return "\n".join(
        lines,
    )
