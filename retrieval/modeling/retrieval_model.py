"""
Paper2Fig-2026 Retrieval Framework

Figure Plagiarism Retrieval Model
----------------------------------

Siamese image encoder for figure-to-figure retrieval.

Author
------
Nguyen Khanh
"""

from __future__ import annotations

import torch
from torch import nn

from retrieval.modeling.backbones.backbones_factory import (
    build_backbone,
)

from retrieval.modeling.heads.heads_factory import (
    build_head,
)


class RetrievalModel(nn.Module):
    """
    Siamese image retrieval model.

    Architecture
    ------------
    Image
        -> DINOv2 Backbone
        -> Projection Head
        -> L2-normalized Embedding

    The same encoder is shared by anchor and target images.

    This model is designed for figure plagiarism retrieval.
    """

    def __init__(
        self,
    ) -> None:

        super().__init__()

        # --------------------------------------------------
        # Shared image encoder
        # --------------------------------------------------

        self.backbone = build_backbone()

        self.head = build_head(
            input_dim=self.backbone.feature_dim,
        )

    # ======================================================
    # Encode image
    # ======================================================

    def encode_image(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode images into retrieval embeddings.

        Parameters
        ----------
        images : torch.Tensor
            Shape:
                [B, 3, H, W]

        Returns
        -------
        torch.Tensor
            Shape:
                [B, embedding_dim]
        """

        features = self.backbone(
            images,
        )

        embeddings = self.head(
            features,
        )

        return embeddings

    # ======================================================
    # Forward
    # ======================================================

    def forward(
        self,
        anchor_images: torch.Tensor,
        target_images: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Encode anchor and target figures.

        Parameters
        ----------
        anchor_images : torch.Tensor
            Anchor figures.

        target_images : torch.Tensor
            Target figures.

        Returns
        -------
        tuple
            (
                anchor_embeddings,
                target_embeddings,
            )
        """

        anchor_embeddings = self.encode_image(
            anchor_images,
        )

        target_embeddings = self.encode_image(
            target_images,
        )

        return (
            anchor_embeddings,
            target_embeddings,
        )

    # ======================================================
    # Embedding dimension
    # ======================================================

    @property
    def embedding_dim(
        self,
    ) -> int:

        return self.head.embedding_dim

    # ======================================================
    # Representation
    # ======================================================

    def __repr__(
        self,
    ) -> str:

        return (
            f"{self.__class__.__name__}(\n"
            f"  backbone={self.backbone.name},\n"
            f"  embedding_dim={self.embedding_dim}\n"
            f")"
        )