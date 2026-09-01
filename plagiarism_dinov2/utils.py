"""
Figure-Caption Plagiarism Pretraining
Utils
--------------------------------------------------------------
Reproducibility, retrieval metrics, ve bieu do training.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def recall_at_k(
    figure_embeddings: torch.Tensor,
    caption_embeddings: torch.Tensor,
    k_values: tuple[int, ...] = (1, 5),
) -> dict:
    """
    Voi moi figure, kiem tra caption dung (cung index) co nam
    trong top-k caption gan nhat (theo cosine similarity) khong.
    """
    similarity = figure_embeddings @ caption_embeddings.t()
    targets = torch.arange(similarity.size(0), device=similarity.device)

    max_k = min(max(k_values), similarity.size(1))
    _, top_indices = similarity.topk(max_k, dim=1)

    results = {}
    for k in k_values:
        k_eff = min(k, max_k)
        hits = (top_indices[:, :k_eff] == targets.unsqueeze(1)).any(dim=1)
        results[f"recall@{k}"] = hits.float().mean().item()

    return results


def plot_training_curves(history: dict, output_path: Path) -> None:
    """
    history can co cac key:
    "train_loss", "val_loss", "val_recall1", "val_recall5"
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_recall1"], label="Recall@1")
    axes[1].plot(epochs, history["val_recall5"], label="Recall@5")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Recall")
    axes[1].set_title("Validation Retrieval Recall")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
