"""
Paper2Fig-2026 Retrieval Framework

Text Encoder Factory
--------------------

Build text encoders from configuration.

Author
------
Nguyen Khanh
"""

from __future__ import annotations

from retrieval.configs import (
    TEXT_ENCODER,
)

from retrieval.modeling.text_encoders.base import (
    BaseTextEncoder,
)

from retrieval.modeling.text_encoders.scibert import (
    SciBERTEncoder,
)

# from retrieval.modeling.text_encoders.specter2 import (
#     Specter2Encoder,
# )


def build_text_encoder() -> BaseTextEncoder:
    """
    Build text encoder.

    Returns
    -------
    BaseTextEncoder
        Initialized text encoder.
    """

    if TEXT_ENCODER == "scibert":

        return SciBERTEncoder()

    # --------------------------------------------------------
    # Future Models
    # --------------------------------------------------------

    # if TEXT_ENCODER == "specter2":
    #
    #     return Specter2Encoder()

    raise ValueError(
        f"Unsupported text encoder: {TEXT_ENCODER}"
    )