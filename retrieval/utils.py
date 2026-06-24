"""
Common utility functions for the Paper2Fig-2026 Retrieval Framework.

Author: Nguyen Khanh
Project: Paper2Fig-2026
"""

from pathlib import Path
from typing import List

import random
import numpy as np
import torch
from PIL import Image


# ============================================================
# Random Seed
# ============================================================

def seed_everything(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Image
# ============================================================

def load_image(image_path: Path) -> Image.Image:
    """
    Load image as RGB.
    """

    return Image.open(image_path).convert("RGB")


# ============================================================
# Directory
# ============================================================

def ensure_dir(directory: Path) -> None:
    """
    Create directory if not exists.
    """

    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# Embedding
# ============================================================

def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """
    L2 normalize embeddings.
    """

    norm = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    return embeddings / np.maximum(norm, 1e-12)


# ============================================================
# Similarity
# ============================================================

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Compute cosine similarity.
    """

    return float(
        np.dot(a, b)
        /
        (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )
    )


# ============================================================
# Batch
# ============================================================

def chunk_list(
    items: List,
    chunk_size: int,
):
    """
    Yield chunks from list.
    """

    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


# ============================================================
# GPU
# ============================================================

def gpu_memory() -> float:
    """
    Current GPU memory usage (GB).
    """

    if not torch.cuda.is_available():
        return 0.0

    return (
        torch.cuda.memory_allocated()
        / 1024**3
    )


# ============================================================
# Time
# ============================================================

def format_seconds(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS.
    """

    seconds = int(seconds)

    h = seconds // 3600

    m = (seconds % 3600) // 60

    s = seconds % 60

    return f"{h:02}:{m:02}:{s:02}"