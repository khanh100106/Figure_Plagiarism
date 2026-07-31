"""
Paper2Fig-2026 Retrieval Framework

Dual Encoder Retrieval Model
----------------------------

Author
------
Nguyen Khanh
"""

from __future__ import annotations

import torch
from torch import nn

from retrieval.modeling.backbones.factory import (
    build_backbone,
)

from retrieval.modeling.text_encoders.factory import (
    build_text_encoder,
)

from retrieval.modeling.heads.factory import (
    build_head,
)


class RetrievalModel(
    nn.Module,
):
    """
    Dual-encoder retrieval model.

    Image
        -> Backbone
        -> Projection Head
        -> Image Embedding

    Caption
        -> Text Encoder
        -> Projection Head
        -> Text Embedding
    """

    def __init__(
        self,
    ) -> None:

        super().__init__()

        # --------------------------------------------------
        # Image branch
        # --------------------------------------------------

        self.image_backbone = build_backbone()

        self.image_head = build_head(
            input_dim=self.image_backbone.feature_dim,
        )

        # --------------------------------------------------
        # Text branch
        # --------------------------------------------------

        self.text_encoder = build_text_encoder()

        self.text_head = build_head(
            input_dim=self.text_encoder.feature_dim,
        )

    def encode_image(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode images into retrieval embeddings.
        """

        image_features = self.image_backbone(
            images,
        )

        image_embeddings = self.image_head(
            image_features,
        )

        return image_embeddings

    def encode_text(
        self,
        captions: list[str],
    ) -> torch.Tensor:
        """
        Encode captions into retrieval embeddings.
        """

        text_features = self.text_encoder(
            captions,
        )

        text_embeddings = self.text_head(
            text_features,
        )

        return text_embeddings

    def forward(
        self,
        images: torch.Tensor,
        captions: list[str],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Encode images and captions.

        Returns
        -------
        tuple
            (image_embeddings, text_embeddings)
        """

        image_embeddings = self.encode_image(
            images,
        )

        text_embeddings = self.encode_text(
            captions,
        )

        return (
            image_embeddings,
            text_embeddings,
        )

    @property
    def embedding_dim(
        self,
    ) -> int:
        """
        Retrieval embedding dimension.
        """

        return self.image_head.embedding_dim

    def __repr__(
        self,
    ) -> str:

        return (
            f"{self.__class__.__name__}(\n"
            f"  image_backbone={self.image_backbone.name},\n"
            f"  text_encoder={self.text_encoder.name},\n"
            f"  embedding_dim={self.embedding_dim}\n"
            f")"
        )