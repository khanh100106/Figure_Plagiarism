"""
Paper2Fig-2026 Retrieval Training
Loss Functions
"""
from __future__ import annotations
import torch
from torch import Tensor
from typing import Optional
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
# Unified Retrieval Loss
# ============================================================

class RetrievalLoss(nn.Module):
    """
    Unified retrieval loss.

    This wrapper encapsulates all supported retrieval losses and
    provides a consistent interface for the Trainer.

    Returns
    -------
    dict
        {
            "loss": Tensor,
            "contrastive_loss": Tensor | None,
            "triplet_loss": Tensor | None,
            "positive_distance": Tensor | None,
            "negative_distance": Tensor | None,
        }
    """

    def __init__(
        self,
    ) -> None:

        super().__init__()

        self.loss_type = LOSS_TYPE

        self.contrastive = ContrastiveLoss()

        self.triplet = TripletLoss()
    def forward(
        self,
        *,
        pair_anchor: Optional[torch.Tensor] = None,
        pair_target: Optional[torch.Tensor] = None,
        pair_label: Optional[torch.Tensor] = None,
        triplet_anchor: Optional[torch.Tensor] = None,
        triplet_positive: Optional[torch.Tensor] = None,
        triplet_negative: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        contrastive_loss = None

        triplet_loss = None

        positive_distance = None

        negative_distance = None
        if self.loss_type in (
                "contrastive",
                "hybrid",
        ):
            if (
                    pair_anchor is None
                    or pair_target is None
                    or pair_label is None
            ):
                raise ValueError(
                    "Pair inputs are required for contrastive loss."
                )

        if self.loss_type in (
                "triplet",
                "hybrid",
        ):
            if (
                    triplet_anchor is None
                    or triplet_positive is None
                    or triplet_negative is None
            ):
                raise ValueError(
                    "Triplet inputs are required for triplet loss."
                )
        if self.loss_type in (

            "contrastive",

            "hybrid",

        ):

            contrastive_loss = self.contrastive(

                pair_anchor,

                pair_target,

                pair_label,

            )
        if self.loss_type in (

            "triplet",

            "hybrid",

        ):

            (

                triplet_loss,

                positive_distance,

                negative_distance,

            ) = self.triplet(

                triplet_anchor,

                triplet_positive,

                triplet_negative,

            )
        if self.loss_type == "contrastive":

            total_loss = contrastive_loss

        elif self.loss_type == "triplet":

            total_loss = triplet_loss

        else:

            total_loss = (

                HYBRID_CONTRASTIVE_WEIGHT

                * contrastive_loss

                +

                HYBRID_TRIPLET_WEIGHT

                * triplet_loss

            )
        return {

            "loss": total_loss,

            "contrastive_loss": contrastive_loss,

            "triplet_loss": triplet_loss,

            "positive_distance": positive_distance,

            "negative_distance": negative_distance,

        }
# ============================================================
# Factory
# ============================================================
def build_loss(
) -> RetrievalLoss:
    """
    Build retrieval loss.
    """

    return RetrievalLoss()
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
