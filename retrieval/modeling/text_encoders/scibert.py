"""
Paper2Fig-2026 Retrieval Framework

SciBERT Text Encoder
--------------------

Author
------
Nguyen Khanh
"""

from __future__ import annotations

import torch
from retrieval.configs import (
    TEXT_MODEL,
    FREEZE_TEXT_ENCODER,
)
from transformers import (
    AutoTokenizer,
    AutoModel,
)

from retrieval.modeling.text_encoders.text_encoders_base import (
    BaseTextEncoder,
)

from retrieval.configs import (
    TEXT_MODEL,
)

class SciBERTEncoder(
    BaseTextEncoder,
):
    """
    SciBERT text encoder.
    """

    def __init__(
        self,
        model_name: str = TEXT_MODEL,
    ) -> None:
        super().__init__()

        self.model_name = model_name


        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

        self.model = AutoModel.from_pretrained(
            model_name,
        )

        if FREEZE_TEXT_ENCODER:
            self.freeze()

    @property
    def feature_dim(
        self,
    ) -> int:
        """
        Output feature dimension.
        """
        return self.model.config.hidden_size

    @property
    def name(
        self,
    ) -> str:
        """
        Encoder name.
        """
        return self.model_name

    def forward(
            self,
            captions: list[str],
    ) -> torch.Tensor:
        """
        Encode scientific captions.

        Parameters
        ----------
        captions : list[str]
            Caption texts.

        Returns
        -------
        torch.Tensor
            Text features.
        """

        inputs = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(
                self.model.device,
            )
            for key, value in inputs.items()
        }

        outputs = self.model(
            **inputs,
        )

        last_hidden_state = outputs.last_hidden_state

        attention_mask = inputs[
            "attention_mask"
        ]

        mask = attention_mask.unsqueeze(
            -1
        ).expand(
            last_hidden_state.size()
        ).float()

        features = (
                last_hidden_state * mask
        ).sum(
            dim=1,
        )

        features = features / mask.sum(
            dim=1,
        ).clamp(
            min=1e-9,
        )

        return features

    def freeze(
            self,
    ) -> None:
        """
        Freeze encoder parameters.
        """

        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def __repr__(
            self,
    ) -> str:
        """
        String representation.
        """

        return (

            f"{self.__class__.__name__}("

            f"model='{self.model_name}', "

            f"feature_dim={self.feature_dim})"

        )