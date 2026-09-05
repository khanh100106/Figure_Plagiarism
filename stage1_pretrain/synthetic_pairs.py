"""
Stage 1 - Image Plagiarism Pretraining
Synthetic Pairs
--------------------------------------------------------------
Sinh cap (source, plagiarized) GIA LAP tu figure that trong
paper2fig2026 (ca 7 category), mo phong 3 muc do dao hinh trong
corpus flowchart that:
  - tier "exact_copy":    resize/nen/crop nhe/chinh mau nhe
  - tier "text_plagiarism": che 1-2 vung chu nhat ngau nhien
  - tier "structure":     xoay/lat/crop manh/doi mau manh (gan
                          nhat voi dao hinh bang AI, cau truc
                          thay doi nhieu hon)

Day la ky thuat pho bien khi khong co du lieu dao hinh that: tu
sinh cap positive bang augmentation (giong cach lam cua SSCD -
Meta AI copy detection).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


TIERS = ("exact_copy", "text_plagiarism", "structure")


def list_figure_paths(
    metadata_csv: Path,
    image_root: Path,
    categories: list[str],
) -> list[Path]:
    """
    Doc paper2fig2026_metadata.csv, tra ve danh sach duong dan
    anh figure (khong trung lap) cho cac category duoc chi dinh.
    """
    metadata_csv = Path(metadata_csv)
    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"Khong tim thay metadata: {metadata_csv}"
        )

    with open(metadata_csv, "r", encoding="utf-8") as file:
        first_line = file.readline()
    separator = "\t" if "\t" in first_line else ","

    dataframe = pd.read_csv(metadata_csv, sep=separator)

    if "category" in dataframe.columns:
        filter_column = "category"
    elif "field" in dataframe.columns:
        filter_column = "field"
    else:
        raise ValueError(
            "Metadata khong co cot 'category' hoac 'field' de loc "
            "theo CATEGORIES trong config.py."
        )

    filtered = dataframe[
        dataframe[filter_column].astype(str).str.contains(
            "|".join(categories),
            case=False,
            na=False,
        )
    ]

    image_root = Path(image_root)
    paths = []
    for relative_path in filtered["figure_image_path"].dropna().unique():
        full_path = Path(relative_path)
        if not full_path.is_absolute():
            full_path = image_root / full_path
        if full_path.exists():
            paths.append(full_path)

    if not paths:
        raise ValueError(
            "Khong tim thay figure nao khop CATEGORIES trong "
            "paper2fig2026 - kiem tra lai PAPER2FIG_ROOT/METADATA "
            "va ten category trong config.py."
        )

    return paths


def _adjust_hue(image: Image.Image, hue_shift: float) -> Image.Image:
    """
    hue_shift trong khoang [-0.5, 0.5] (ty le vong mau, giong don
    vi cua torchvision.adjust_hue) - dung numpy tren khong gian HSV.
    """
    hsv = np.array(image.convert("HSV"), dtype=np.uint8)
    shift = int(hue_shift * 255)
    hsv[..., 0] = (hsv[..., 0].astype(np.int32) + shift) % 256
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def _apply_exact_copy(image: Image.Image) -> Image.Image:
    width, height = image.size

    scale = random.uniform(0.85, 1.0)
    image = image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale)))
    )

    if random.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.5))

    brightness = random.uniform(0.9, 1.1)
    image = ImageEnhance.Brightness(image).enhance(brightness)

    return image


def _apply_text_plagiarism(image: Image.Image) -> Image.Image:
    image = _apply_exact_copy(image)
    image = image.copy()

    width, height = image.size
    draw = ImageDraw.Draw(image)

    num_boxes = random.randint(1, 2)
    for _ in range(num_boxes):
        box_width = random.uniform(0.1, 0.3) * width
        box_height = random.uniform(0.05, 0.15) * height
        x1 = random.uniform(0, max(1, width - box_width))
        y1 = random.uniform(0, max(1, height - box_height))
        color = tuple(random.randint(200, 255) for _ in range(3))
        draw.rectangle(
            [x1, y1, x1 + box_width, y1 + box_height],
            fill=color,
        )

    return image


def _apply_structure(image: Image.Image) -> Image.Image:
    image = _apply_text_plagiarism(image)

    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    angle = random.uniform(-15, 15)
    image = image.rotate(angle, expand=True, fillcolor=(255, 255, 255))

    width, height = image.size
    crop_ratio = random.uniform(0.8, 0.95)
    crop_width = int(width * crop_ratio)
    crop_height = int(height * crop_ratio)
    left = random.randint(0, max(0, width - crop_width))
    top = random.randint(0, max(0, height - crop_height))
    image = image.crop(
        (left, top, left + crop_width, top + crop_height)
    )

    saturation = random.uniform(0.6, 1.4)
    image = ImageEnhance.Color(image).enhance(saturation)
    hue = random.uniform(-0.05, 0.05)
    image = _adjust_hue(image, hue)

    return image


_TIER_FUNCTIONS = {
    "exact_copy": _apply_exact_copy,
    "text_plagiarism": _apply_text_plagiarism,
    "structure": _apply_structure,
}


def generate_synthetic_copy(
    image: Image.Image,
    tier: str | None = None,
) -> tuple[Image.Image, str]:
    """
    Sinh 1 ban "dao hinh gia lap" tu 1 anh goc.

    Neu tier=None, chon ngau nhien 1 trong 3 muc do.
    Tra ve (anh_da_bien_doi, tier_da_dung).
    """
    if tier is None:
        tier = random.choice(TIERS)

    if tier not in _TIER_FUNCTIONS:
        raise ValueError(f"Tier khong hop le: {tier}")

    transformed = _TIER_FUNCTIONS[tier](image.convert("RGB"))
    return transformed, tier
