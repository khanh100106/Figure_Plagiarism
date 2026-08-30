"""
Paper2Fig-2026 Retrieval Framework

DINOv2 Backbone
---------------

Author
------
Nguyen Khanh
"""

from __future__ import annotations

import torch

from retrieval.modeling.backbones.backbones_base import (
    BaseBackbone,
)

from retrieval.configs import (
    BACKBONE_MODEL,
    PRETRAINED,
    FREEZE_BACKBONE,
)

class DINOv2Backbone(
    BaseBackbone,
):
    """
    DINOv2 image backbone.
    """

    def __init__(
            self,
            model_name: str = BACKBONE_MODEL,
            pretrained: bool = PRETRAINED,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.model = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2",
            model=model_name,
            pretrained=pretrained,
        )
        if FREEZE_BACKBONE:
            self.freeze()

    @property
    def feature_dim(
            self,
    ) -> int:
        """
        Backbone output feature dimension.
        """

        return self.model.embed_dim

    def forward(
            self,
            images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract image features.
        """

        return self.model(
            images,
        )

    @property
    def name(
            self,
    ) -> str:
        """
        Backbone name.
        """

        return self.model_name

    def __repr__(
            self,
    ) -> str:
        """
        String representation.
        """

        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"feature_dim={self.feature_dim})"
        )
