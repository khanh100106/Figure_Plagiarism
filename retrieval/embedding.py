"""
Embedding Factory

Paper2Fig-2026 Retrieval Framework

Author: Nguyen Khanh
"""

from retrieval.configs import BACKBONE

from retrieval.backbones import (
    DINOv2Extractor,
)


class EmbeddingExtractor:
    """
    Factory class for embedding backbones.

    Usage
    -----
    extractor = EmbeddingExtractor()

    embeddings = extractor.extract(images)
    """

    def __new__(cls):

        backbone = BACKBONE.lower()

        if backbone == "dinov2":

            return DINOv2Extractor()

        raise ValueError(
            f"Unsupported backbone: {BACKBONE}"
        )