"""
Dataset utilities for the Paper2Fig-2026 Retrieval Framework.

Author: Nguyen Khanh
Project: Paper2Fig-2026
"""

from pathlib import Path
from typing import Dict

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from retrieval.configs import (
    DATASET_ROOT,
    METADATA_FILE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    DEBUG,
    DEBUG_SAMPLES,
)

from utils import load_image


# ============================================================
# Figure Dataset
# ============================================================

class FigureDataset(Dataset):
    """
    Dataset for scientific figures.

    Each sample contains:

    - PIL Image
    - Complete metadata (dictionary)
    """

    def __init__(
        self,
        metadata_file: Path = METADATA_FILE,
        start_index: int = 0,
    ) -> None:

        self.metadata_file = metadata_file

        self.df = pd.read_csv(metadata_file)

        if DEBUG:

            end_index = start_index + DEBUG_SAMPLES

            self.df = (
                self.df
                .iloc[start_index:end_index]
                .reset_index(drop=True)
            )

        else:

            self.df = (
                self.df
                .iloc[start_index:]
                .reset_index(drop=True)
            )

    # --------------------------------------------------------

    def __len__(self) -> int:

        return len(self.df)

    # --------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]

        image_path = (
                DATASET_ROOT /
                row["figure_image_path"]
        )

        image_path = image_path.resolve()

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        image = load_image(image_path)

        return {

            "image": image,

            "figure_id": row["figure_id"],

            "paper_id": row["paper_id"],

            "field": row["field"],

            "figure_image_path": row["figure_image_path"]

        }

# ============================================================
# Collate Function
# ============================================================

def collate_fn(batch):
    """
    Keep the batch as a list of samples.
    """
    return batch


# ============================================================
# DataLoader
# ============================================================

def create_dataloader(
    dataset: Dataset,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
) -> DataLoader:
    """
    Create PyTorch DataLoader.
    """

    return DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        num_workers=num_workers,

        pin_memory=pin_memory,

        collate_fn=collate_fn,

    )


# ============================================================
# Dataset Information
# ============================================================

def print_dataset_info(
    dataset: FigureDataset,
) -> None:
    """
    Print dataset summary.
    """

    print("=" * 60)

    print("Paper2Fig Dataset")

    print("=" * 60)

    print(f"Metadata : {dataset.metadata_file}")

    print(f"Samples  : {len(dataset):,}")

    print("=" * 60)


# ============================================================
# Verify Images
# ============================================================

def verify_images(
    dataset: FigureDataset,
) -> None:
    """
    Verify all image files exist.
    """

    print()

    print("Verifying image paths...")

    missing = 0

    for _, row in dataset.df.iterrows():

        image_path = (
            DATASET_ROOT /
            row["figure_image_path"]
        )

        if not image_path.exists():

            print(image_path)

            missing += 1

    print()

    print(f"Missing images : {missing}")


# ============================================================
# Debug
# ============================================================

if __name__ == "__main__":

    dataset = FigureDataset()

    print_dataset_info(dataset)

    verify_images(dataset)

    sample = dataset[0]

    print()

    print("First Sample")

    print(sample["metadata"]["figure_id"])

    print(sample["metadata"]["paper_id"])

    print(sample["metadata"]["field"])

    print(sample["image"])