"""
Paper2Fig-2026 Retrieval Framework

Backbone Factory
----------------

Build image backbones.

Author
------
Nguyen Khanh
"""

from __future__ import annotations

from retrieval.configs import (
    BACKBONE,
)

from retrieval.modeling.backbones.backbones_base import (
    BaseBackbone,
)

from retrieval.modeling.backbones.dinov2 import (
    DINOv2Backbone,
)

# Future
#
# from retrieval.modeling.backbones.clip import ClipBackbone
# from retrieval.modeling.backbones.siglip import SigLIPBackbone


def build_backbone() -> BaseBackbone:
    """
    Build image backbone.

    Returns
    -------
    BaseBackbone
        Initialized backbone.
    """

    if BACKBONE == "dinov2":

        return DINOv2Backbone()

    #
    # Future
    #
    # if BACKBONE == "clip":
    #     return ClipBackbone()
    #
    # if BACKBONE == "siglip":
    #     return SigLIPBackbone()

    raise ValueError(
        f"Unsupported backbone: {BACKBONE}"
    )