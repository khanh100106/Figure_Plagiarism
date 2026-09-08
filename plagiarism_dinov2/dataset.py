"""
Figure-Caption Plagiarism Pretraining
Dataset
--------------------------------------------------------------
Cot bat buoc trong CSV: figure_id, caption_id, paper_id,
figure_image_path.
- Neu modality="image": can them cot caption_image_path.
- Neu modality="text": can them cot caption_text.

Moi dong duoc coi la 1 cap (figure, caption) THAT (cung xuat hien
trong cung 1 paper). Khi train, dung contrastive trong batch:
figure[i] phai khop voi caption[i], cac caption con lai trong
batch dong vai tro negative.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


BASE_REQUIRED_COLUMNS = {
    "figure_id",
    "caption_id",
    "paper_id",
    "figure_image_path",
}

MODALITY_REQUIRED_COLUMNS = {
    "image": {"caption_image_path"},
    "text": {"caption_text"},
}


def load_metadata(
    csv_path: Path,
    modality: str = "text",
) -> pd.DataFrame:
    """
    Doc file CSV/TSV, tu dong nhan dien dau phan cach (',' hoac '\t').
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay file metadata: {csv_path}\n"
            f"-> Sua DATA_CSV trong config.py cho dung duong dan."
        )

    with open(csv_path, "r", encoding="utf-8") as file:
        first_line = file.readline()

    separator = "\t" if "\t" in first_line else ","

    dataframe = pd.read_csv(csv_path, sep=separator)

    required = BASE_REQUIRED_COLUMNS | MODALITY_REQUIRED_COLUMNS.get(
        modality,
        set(),
    )
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"File metadata thieu cot bat buoc cho modality="
            f"'{modality}': {missing}"
        )

    return dataframe


def split_by_paper(
    dataframe: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chia train/val/test theo paper_id (khong theo tung dong), de
    dam bao cac figure/caption cua CUNG mot paper khong bi chia
    vao nhieu hon 1 tap -> tranh data leakage.

    Tap test KHONG duoc dung trong suot qua trinh tuning (chon
    tham so, so sanh cau hinh, early stopping...) - chi dung 1
    lan duy nhat sau khi da chot cau hinh cuoi cung, de danh gia
    khach quan (val da bi dung lien tuc de chon "best model" nen
    khong con khach quan cho muc dich nay nua).
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio + val_ratio phai nho hon 1.0 de con lai "
            f"phan cho tap test (hien tai: {train_ratio} + "
            f"{val_ratio} = {train_ratio + val_ratio})"
        )

    paper_ids = dataframe["paper_id"].unique().tolist()

    shuffled_papers = (
        pd.Series(paper_ids)
        .sample(frac=1.0, random_state=seed)
        .tolist()
    )

    n_total = len(shuffled_papers)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_papers = set(shuffled_papers[:n_train])
    val_papers = set(shuffled_papers[n_train : n_train + n_val])
    test_papers = set(shuffled_papers[n_train + n_val :])

    train_df = dataframe[
        dataframe["paper_id"].isin(train_papers)
    ].reset_index(drop=True)

    val_df = dataframe[
        dataframe["paper_id"].isin(val_papers)
    ].reset_index(drop=True)

    test_df = dataframe[
        dataframe["paper_id"].isin(test_papers)
    ].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Mot trong 3 tap train/val/test bi rong. Dataset qua "
            "nho hoac so luong paper_id duy nhat qua it de chia "
            "lam 3 phan. Hay dieu chinh TRAIN_RATIO/VAL_RATIO "
            "trong config.py hoac kiem tra lai cot paper_id."
        )

    return train_df, val_df, test_df


class FigureCaptionDataset(Dataset):
    """
    Tra ve 1 mau (anh figure, caption) da qua transform.

    modality="text": caption tra ve la chuoi van ban tho
        (row["caption_text"]). Viec tokenize duoc lam trong
        collate_fn (xem collate.py) de padding dong theo tung
        batch, hieu qua hon la pad co dinh tu dataset.
    modality="image": caption tra ve la anh da crop (giong
        figure), giu nguyen hanh vi ban dau (Siamese DINOv2).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_root: Path,
        transform=None,
        modality: str = "text",
    ) -> None:
        if modality not in ("text", "image"):
            raise ValueError(
                f"modality phai la 'text' hoac 'image', nhan duoc: "
                f"{modality}"
            )

        self.dataframe = dataframe.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.modality = modality

    def __len__(self) -> int:
        return len(self.dataframe)

    def _load_image(self, relative_path: str) -> Image.Image:
        path = Path(relative_path)

        if not path.is_absolute() and str(self.image_root) != "":
            path = self.image_root / path

        if not path.exists():
            raise FileNotFoundError(
                f"Khong tim thay anh: {path}\n"
                f"-> Kiem tra lai IMAGE_ROOT trong config.py hoac "
                f"duong dan trong CSV."
            )

        return Image.open(path).convert("RGB")

    def __getitem__(self, index: int) -> dict:
        row = self.dataframe.iloc[index]

        figure_image = self._load_image(row["figure_image_path"])
        if self.transform is not None:
            figure_image = self.transform(figure_image)

        item = {
            "figure": figure_image,
            "figure_id": str(row["figure_id"]),
            "caption_id": str(row["caption_id"]),
        }

        if self.modality == "text":
            caption_text = row["caption_text"]
            item["caption_text"] = (
                "" if pd.isna(caption_text) else str(caption_text)
            )
        else:
            caption_image = self._load_image(row["caption_image_path"])
            if self.transform is not None:
                caption_image = self.transform(caption_image)
            item["caption"] = caption_image

        return item
