"""
Stage 1 - Image Plagiarism Pretraining
Dataset
--------------------------------------------------------------
Doc pairs_with_hard_negatives.csv (tao boi build_pairs.py), tra
ve triplet (anchor, positive, negative) da qua transform.

- pair_type="real": positive_path la anh that, doc truc tiep.
- pair_type="synthetic": positive_path rong - sinh on-the-fly
  bang cach bien doi anchor_path theo tier_or_plagtype (tranh
  phai luu them anh ra dia).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from synthetic_pairs import generate_synthetic_copy


class TripletPlagiarismDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Khong tim thay anh: {path}")
        return Image.open(path).convert("RGB")

    def __getitem__(self, index: int) -> dict:
        row = self.dataframe.iloc[index]

        anchor_image = self._load_image(row["anchor_path"])

        positive_path = row["positive_path"]
        if pd.isna(positive_path) or str(positive_path).strip() == "":
            positive_image, _ = generate_synthetic_copy(
                anchor_image,
                tier=row["tier_or_plagtype"],
            )
        else:
            positive_image = self._load_image(positive_path)

        negative_image = self._load_image(row["negative_path"])

        if self.transform is not None:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return {
            "anchor": anchor_image,
            "positive": positive_image,
            "negative": negative_image,
            "pair_type": str(row["pair_type"]),
        }


def load_pairs_csv(csv_path: Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay file pairs: {csv_path}\n"
            f"-> Chay 'python build_pairs.py' truoc de tao file nay."
        )
    return pd.read_csv(csv_path)


def split_train_val(
    dataframe: pd.DataFrame,
    train_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = dataframe.sample(frac=1.0, random_state=seed).reset_index(
        drop=True
    )
    cutoff = int(len(shuffled) * train_ratio)
    train_df = shuffled.iloc[:cutoff].reset_index(drop=True)
    val_df = shuffled.iloc[cutoff:].reset_index(drop=True)
    return train_df, val_df
