"""
Paper2Fig-2026 Retrieval Training
Module
------
Dataset
Responsibilities
----------------
- Build image lookup table.
- Load figure images.
- Support image caching.
- Provide reusable dataset base class.
Author
------
Nguyen Khanh
"""
from __future__ import annotations
import torch
import pandas as pd
from pathlib import Path
from typing import Callable
from typing import Optional
from PIL import Image
from torch.utils.data import Dataset
from retrieval.constants import (
    METADATA_COLUMNS,
)
from retrieval.configs import DATASET_ROOT
# ============================================================
# Exceptions
# ============================================================
class ImageNotFoundError(FileNotFoundError):
    """
    Raised when an image cannot be found.
    """
# ============================================================
# Utilities
# ============================================================
def build_image_lookup(
    metadata: pd.DataFrame,
) -> dict[str, Path]:
    missing = [
        col
        for col in METADATA_COLUMNS
        if col not in metadata.columns
    ]
    if missing:
        raise ValueError(
            "Metadata missing columns: {}".format(missing)
        )
    lookup = {}
    for row in metadata.itertuples():
        image_path = DATASET_ROOT / Path(row.image_path)
        lookup[row.figure_id] = image_path
    return lookup
# ============================================================
# Base Dataset
# ============================================================
class BaseDataset(Dataset):
    """
    Base dataset shared by all retrieval datasets.
    """
    def __init__(
        self,
        image_lookup: dict[str, Path],
        transform: Optional[Callable] = None,
        cache_images: bool = False,
    ) -> None:
        self.image_lookup = image_lookup
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache: dict[str, Image.Image] = {}
    # --------------------------------------------------------
    def __len__(
        self,
    ) -> int:
        raise NotImplementedError
    # --------------------------------------------------------
    def load_image(
        self,
        figure_id: str,
    ) -> Image.Image:
        """
        Load figure image.
        """
        if (
            self.cache_images
            and figure_id in self.image_cache
        ):
            return self.image_cache[
                figure_id
            ].copy()
        image_path = self.image_lookup.get(
            figure_id
        )
        if image_path is None:
            raise ImageNotFoundError(
                f"Unknown figure_id: {figure_id}"
            )
        if not image_path.exists():
            raise ImageNotFoundError(
                f"Image does not exist:\n{image_path}"
            )
        image = (
            Image
            .open(image_path)
            .convert("RGB")
        )
        if self.cache_images:
            self.image_cache[
                figure_id
            ] = image.copy()
        return image
    # --------------------------------------------------------
    def apply_transform(
        self,
        image: Image.Image,
    ):
        """
        Apply transform if available.
        """
        if self.transform is None:
            return image
        return self.transform(
            image
        )
# ============================================================
# Pair Dataset
# ============================================================
class PairDataset(BaseDataset):
    """
    Retrieval Pair Dataset.
    Returns
    -------
    {
        "anchor": Tensor,
        "target": Tensor,
        "label": Tensor,
        "anchor_id": str,
        "target_id": str,
    }
    """
    REQUIRED_COLUMNS = [
        "anchor_id",
        "target_id",
        "label",
    ]
    def __init__(
        self,
        pairs: pd.DataFrame,
        image_lookup: dict[str, Path],
        transform: Optional[Callable] = None,
        cache_images: bool = False,
    ) -> None:
        super().__init__(
            image_lookup=image_lookup,
            transform=transform,
            cache_images=cache_images,
        )
        missing = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in pairs.columns
        ]
        if missing:
            raise ValueError(
                f"Pair dataset missing columns: {missing}"
            )
        self.pairs = (
            pairs
            .reset_index(drop=True)
            .copy()
        )
    # --------------------------------------------------------
    def __len__(
        self,
    ) -> int:
        return len(
            self.pairs
        )
    # --------------------------------------------------------
    def __getitem__(
        self,
        index: int,
    ) -> dict:
        row = self.pairs.iloc[index]
        anchor_id = row.anchor_id
        target_id = row.target_id
        label = int(
            row.label
        )
        anchor = self.load_image(
            anchor_id
        )
        target = self.load_image(
            target_id
        )
        anchor = self.apply_transform(
            anchor
        )
        target = self.apply_transform(
            target
        )
        return {
            "anchor": anchor,
            "target": target,
            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),
            "anchor_id": anchor_id,
            "target_id": target_id,
        }
class TripletDataset(
    BaseDataset,
):
    """
    Retrieval Triplet Dataset.
    """
    def __init__(
        self,
        triplets: pd.DataFrame,
        image_lookup: dict[str, Path],
        transform=None,
    ) -> None:
        super().__init__(
            image_lookup=image_lookup,
            transform=transform,
        )
        self.triplets = triplets.reset_index(
            drop=True,
        )
    def __len__(
        self,
    ) -> int:
        return len(
            self.triplets,
        )
    def __getitem__(
        self,
        index: int,
    ) -> dict:
        row = self.triplets.iloc[index]
        anchor = self.load_image(
            row.anchor_id,
        )
        positive = self.load_image(
            row.positive_id,
        )
        negative = self.load_image(
            row.negative_id,
        )
        if self.transform is not None:
            anchor = self.transform(
                anchor,
            )
            positive = self.transform(
                positive,
            )
            negative = self.transform(
                negative,
            )
        return {
            "anchor": anchor,
            "positive": positive,
            "negative": negative,
            "anchor_id": row.anchor_id,
            "positive_id": row.positive_id,
            "negative_id": row.negative_id,
        }
# ============================================================
# Combined Dataset
# ============================================================
class CombinedDataset(Dataset):
    """
    Combined dataset used by the Trainer.

    The dataset automatically cycles through the
    smaller dataset so that every epoch always
    covers the larger dataset completely.

    This prevents wasting samples when the number
    of pairs and triplets are different.
    """

    def __init__(
        self,
        pair_dataset: PairDataset,
        triplet_dataset: TripletDataset,
    ) -> None:

        self.pair_dataset = pair_dataset

        self.triplet_dataset = triplet_dataset

        self.length = max(
            len(pair_dataset),
            len(triplet_dataset),
        )

    def __len__(
        self,
    ) -> int:

        return self.length

    def __getitem__(
        self,
        index: int,
    ) -> dict:

        pair = self.pair_dataset[
            index % len(self.pair_dataset)
        ]

        triplet = self.triplet_dataset[
            index % len(self.triplet_dataset)
        ]

        return {

            "pair": pair,

            "triplet": triplet,

        }