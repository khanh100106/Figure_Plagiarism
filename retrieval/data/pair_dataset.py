"""
Paper2Fig-2026 Retrieval Framework

Pair Dataset
------------

Dataset used for contrastive retrieval training.

Author
------
Nguyen Khanh
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from torch.utils.data import Dataset

from retrieval.configs import (
    DATASET_ROOT,
    METADATA_FILE,
    RETRIEVAL_PAIR_FILE,
)

from retrieval.data.transforms import (
    build_train_transform,
    build_eval_transform,
)


class PairDataset(
    Dataset,
):
    """
    Pair retrieval dataset.

    Returns
    -------
    dict

    {
        "image": Tensor,
        "caption": str,
        "label": int,
        "anchor_id": str,
        "target_id": str,
    }
    """

    def __init__(
        self,
        train: bool = True,
    ) -> None:

        super().__init__()

        self.train = train

        self.transform = (
            build_train_transform()
            if train
            else build_eval_transform()
        )

        #
        # pair table
        #

        self.pairs = pd.read_csv(
            RETRIEVAL_PAIR_FILE,
        )

        #
        # metadata
        #

        metadata = pd.read_csv(
            METADATA_FILE,
        )

        #
        # figure_id -> metadata
        #

        self.metadata = (
            metadata
            .set_index(
                "figure_id",
            )
            .to_dict(
                orient="index",
            )
        )

    def __len__(
        self,
    ) -> int:

        return len(
            self.pairs,
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict:

        row = self.pairs.iloc[index]

        anchor_id = row["anchor_id"]

        target_id = row["target_id"]

        label = int(
            row["label"],
        )

        #
        # anchor image
        #

        anchor = self.metadata[
            anchor_id
        ]

        image_path = (
            DATASET_ROOT
            / anchor["figure_image_path"]
        )

        image = Image.open(
            image_path,
        ).convert(
            "RGB",
        )

        image = self.transform(
            image,
        )

        #
        # target caption
        #

        target = self.metadata[
            target_id
        ]

        caption = target[
            "caption_text"
        ]

        return {

            "image": image,

            "caption": caption,

            "label": label,

            "anchor_id": anchor_id,

            "target_id": target_id,

        }

    def __repr__(
        self,
    ) -> str:

        return (

            f"{self.__class__.__name__}("

            f"samples={len(self)}, "

            f"train={self.train})"

        )