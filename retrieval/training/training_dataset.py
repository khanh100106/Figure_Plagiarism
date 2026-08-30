"""
Paper2Fig-2026 Retrieval Framework

Training Dataset
----------------

Pair dataset for Siamese figure retrieval training.

Author
------
Nguyen Khanh
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from PIL import Image

from torch import Tensor
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


class PairTrainingDataset(
    Dataset,
):
    """
    Pair dataset for Siamese image retrieval training.

    Each sample contains:

        anchor_image
        target_image
        caption
        label
        anchor_id
        target_id

    The caption is retained as metadata but is not encoded
    by the current image-only retrieval model.
    """

    def __init__(
        self,
        train: bool = True,
    ) -> None:

        super().__init__()

        self.train = train

        # --------------------------------------------------
        # Transform
        # --------------------------------------------------

        if self.train:

            self.transform = (
                build_train_transform()
            )

        else:

            self.transform = (
                build_eval_transform()
            )

        # --------------------------------------------------
        # Pair table
        # --------------------------------------------------

        self.pairs = pd.read_csv(
            RETRIEVAL_PAIR_FILE,
        )

        # --------------------------------------------------
        # Metadata
        # --------------------------------------------------

        metadata = pd.read_csv(
            METADATA_FILE,
        )

        self.metadata = (
            metadata
            .set_index(
                "figure_id",
            )
            .to_dict(
                orient="index",
            )
        )

        # --------------------------------------------------
        # Validate IDs
        # --------------------------------------------------

        missing_anchor = [
            figure_id
            for figure_id in self.pairs["anchor_id"]
            if figure_id not in self.metadata
        ]

        missing_target = [
            figure_id
            for figure_id in self.pairs["target_id"]
            if figure_id not in self.metadata
        ]

        if missing_anchor:

            raise ValueError(
                "Missing anchor IDs in metadata: "
                f"{len(missing_anchor)}"
            )

        if missing_target:

            raise ValueError(
                "Missing target IDs in metadata: "
                f"{len(missing_target)}"
            )

    # ======================================================
    # Length
    # ======================================================

    def __len__(
        self,
    ) -> int:

        return len(
            self.pairs,
        )

    # ======================================================
    # Image loading
    # ======================================================

    def _load_image(
        self,
        figure_id: str,
    ) -> Tensor:
        """
        Load and transform one figure.
        """

        record = self.metadata[
            figure_id
        ]

        image_path = (
            DATASET_ROOT
            / record[
                "figure_image_path"
            ]
        )

        image = Image.open(
            image_path,
        ).convert(
            "RGB",
        )

        image = self.transform(
            image,
        )

        return image

    # ======================================================
    # Get sample
    # ======================================================

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:

        row = self.pairs.iloc[
            index
        ]

        anchor_id = str(
            row["anchor_id"]
        )

        target_id = str(
            row["target_id"]
        )

        label = int(
            row["label"]
        )

        # --------------------------------------------------
        # Anchor image
        # --------------------------------------------------

        anchor_image = self._load_image(
            anchor_id,
        )

        # --------------------------------------------------
        # Target image
        # --------------------------------------------------

        target_image = self._load_image(
            target_id,
        )

        # --------------------------------------------------
        # Caption
        #
        # Keep caption in dataset.
        # It is NOT used by the current image-only model.
        # --------------------------------------------------

        target_record = self.metadata[
            target_id
        ]

        caption = str(
            target_record[
                "caption_text"
            ]
        )

        return {

            "anchor_image": anchor_image,

            "target_image": target_image,

            "caption": caption,

            "label": label,

            "anchor_id": anchor_id,

            "target_id": target_id,

        }

    # ======================================================
    # Representation
    # ======================================================

    def __repr__(
        self,
    ) -> str:

        return (
            f"{self.__class__.__name__}("
            f"samples={len(self)}, "
            f"train={self.train})"
        )