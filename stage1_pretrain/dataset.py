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


def split_by_anchor(
    dataframe: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chia train/val/test theo anchor_path (khong theo tung dong).

    Ly do khong chia theo dong: SYNTHETIC_COPIES_PER_FIGURE > 1
    nghia la CUNG 1 anchor_path xuat hien nhieu dong (nhieu ban
    synthetic khac nhau). Chia theo dong ngau nhien co the khien
    2 ban cua CUNG 1 anh goc roi vao 2 tap khac nhau -> ro ri nhe.

    Tap test KHONG duoc dung trong qua trinh tuning - chi dung 1
    lan de danh gia cuoi cung.
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio + val_ratio phai nho hon 1.0 de con lai "
            f"phan cho tap test (hien tai: {train_ratio} + "
            f"{val_ratio} = {train_ratio + val_ratio})"
        )

    anchors = dataframe["anchor_path"].unique().tolist()

    shuffled_anchors = (
        pd.Series(anchors).sample(frac=1.0, random_state=seed).tolist()
    )

    n_total = len(shuffled_anchors)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_anchors = set(shuffled_anchors[:n_train])
    val_anchors = set(shuffled_anchors[n_train : n_train + n_val])
    test_anchors = set(shuffled_anchors[n_train + n_val :])

    train_df = dataframe[
        dataframe["anchor_path"].isin(train_anchors)
    ].reset_index(drop=True)

    val_df = dataframe[
        dataframe["anchor_path"].isin(val_anchors)
    ].reset_index(drop=True)

    test_df = dataframe[
        dataframe["anchor_path"].isin(test_anchors)
    ].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Mot trong 3 tap train/val/test bi rong. Dataset qua "
            "nho hoac so luong anchor duy nhat qua it de chia lam "
            "3 phan. Hay dieu chinh TRAIN_RATIO/VAL_RATIO trong "
            "config.py."
        )

    return train_df, val_df, test_df
