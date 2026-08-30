"""
Paper2Fig-2026 Retrieval Framework

Embedding Extractor
-------------------

Wrapper for image feature extraction.

Author
------
Nguyen Khanh
"""

from __future__ import annotations

import torch

from retrieval.modeling.backbones.backbones_factory import (
    build_backbone,
)


class EmbeddingExtractor:
    """
    Extract image features using the configured backbone.
    """

    def __init__(
        self,
    ) -> None:

        self.backbone = build_backbone()

        self.backbone.eval()

    @torch.no_grad()
    def extract(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract backbone features.

        Parameters
        ----------
        images : torch.Tensor

        Returns
        -------
        torch.Tensor
        """

        return self.backbone(
            images,
        )