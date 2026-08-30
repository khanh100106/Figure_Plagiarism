"""
Paper2Fig-2026 Retrieval Framework

Base Dataset
------------

Abstract retrieval dataset.

Author
------
Nguyen Khanh
"""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any

import pandas as pd

from PIL import Image

from torch.utils.data import Dataset

from retrieval.configs import (
    METADATA_FILE,
)

from retrieval.data.transforms import (
    build_train_transform,
    build_eval_transform,
)


class BaseDataset(
    Dataset,
    ABC,
):
    """
    Base retrieval dataset.

    This dataset is responsible only for reading metadata,
    loading images and captions, and returning a unified sample.

    PairDataset, TripletDataset and HybridDataset should inherit
    from this class.
    """

    def __init__(
        self,
        train: bool = True,
    ) -> None:

        super().__init__()

        self.train = train

        self.metadata = pd.read_csv(
            METADATA_FILE,
        )

        if self.train:

            self.transform = build_train_transform()

        else:

            self.transform = build_eval_transform()

    def __len__(
        self,
    ) -> int:
        """
        Number of figures.

        Returns
        -------
        int
        """

        return len(
            self.metadata,
        )

    def _load_image(
        self,
        image_path: str,
    ) -> Any:
        """
        Load figure image.

        Parameters
        ----------
        image_path : str

        Returns
        -------
        torch.Tensor
        """

        image = Image.open(
            image_path,
        ).convert(
            "RGB",
        )

        return self.transform(
            image,
        )

    def _build_sample(
        self,
        row: pd.Series,
    ) -> dict[str, Any]:
        """
        Build one retrieval sample.

        Parameters
        ----------
        row : pd.Series

        Returns
        -------
        dict
        """

        image = self._load_image(
            row["figure_image_path"],
        )

        sample = {

            # -------------------------
            # IDs
            # -------------------------

            "figure_id": row["figure_id"],

            "caption_id": row["caption_id"],

            "paper_id": row["paper_id"],

            # -------------------------
            # Metadata
            # -------------------------

            "field": row["field"],

            "category": row["category"],

            "published": row["published"],

            "title": row["title"],

            "page": row["page"],

            # -------------------------
            # Retrieval
            # -------------------------

            "image": image,

            "caption": row["caption_text"],

            # -------------------------
            # Paths
            # -------------------------

            "image_path": row["figure_image_path"],

            "caption_image_path": row["caption_image_path"],

            # -------------------------
            # Bounding Boxes
            # -------------------------

            "figure_bbox": (

                row["figure_x1"],

                row["figure_y1"],

                row["figure_x2"],

                row["figure_y2"],

            ),

            "caption_bbox": (

                row["caption_x1"],

                row["caption_y1"],

                row["caption_x2"],

                row["caption_y2"],

            ),

        }

        return sample

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        """
        Get one figure sample.

        Parameters
        ----------
        index : int

        Returns
        -------
        dict
        """

        row = self.metadata.iloc[
            index
        ]

        return self._build_sample(
            row,
        )

    def __repr__(
        self,
    ) -> str:

        return (

            f"{self.__class__.__name__}("

            f"samples={len(self)}, "

            f"train={self.train})"

        )
