"""
Stage 1 - Image Plagiarism Pretraining
Hard Negative Mining
--------------------------------------------------------------
Buoc 1 (build_embedding_index): trich embedding cho toan bo pool
anh bang 1 backbone DINOv2 NHO va DONG BANG - chi de do luong do
giong nhau, hoan toan tach biet voi model dang train o Stage 1.

Buoc 2 (rank_hard_negatives): voi moi anchor, tim trong pool
nhung anh GIONG NHAT NHUNG KHONG PHAI chinh no/dung positive -
do la hard negative that su (khac voi in-batch negative ngau
nhien).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def rank_hard_negatives(
    anchor_embeddings: np.ndarray,
    pool_embeddings: np.ndarray,
    exclude_indices: list[int],
    top_k: int,
) -> list[list[int]]:
    """
    anchor_embeddings: (N, D) da L2-normalize.
    pool_embeddings:   (M, D) da L2-normalize.
    exclude_indices:   voi moi anchor i, exclude_indices[i] la 1
        chi so trong pool_embeddings can LOAI TRU (thuong la chinh
        no hoac dung positive - khong duoc chon lam negative).
    top_k: so luong hard negative candidate giu lai cho moi anchor.

    Tra ve: list N phan tu, moi phan tu la list toi da top_k chi
    so trong pool_embeddings, xep theo do giong giam dan, KHONG
    bao gom exclude_indices[i].
    """
    if anchor_embeddings.shape[1] != pool_embeddings.shape[1]:
        raise ValueError(
            "So chieu embedding cua anchor va pool khac nhau."
        )

    if len(exclude_indices) != anchor_embeddings.shape[0]:
        raise ValueError(
            "exclude_indices phai co do dai bang so luong anchor."
        )

    similarity = anchor_embeddings @ pool_embeddings.T  # (N, M)
    pool_size = pool_embeddings.shape[0]
    effective_k = min(top_k, max(pool_size - 1, 0))

    results = []
    for i in range(similarity.shape[0]):
        row = similarity[i].copy()

        exclude = exclude_indices[i]
        if exclude is not None:
            row[exclude] = -np.inf

        if effective_k <= 0:
            results.append([])
            continue

        # argpartition de lay top-k nhanh (O(M)), roi sap xep rieng
        # top-k do giam dan.
        top_indices = np.argpartition(-row, effective_k - 1)[:effective_k]
        top_indices = top_indices[np.argsort(-row[top_indices])]

        # Loai bo cac candidate co similarity = -inf (truong hop
        # pool qua nho, khong du top_k candidate hop le).
        top_indices = [
            int(index) for index in top_indices if row[index] != -np.inf
        ]
        results.append(top_indices)

    return results


def build_embedding_index(
    image_paths: list[Path],
    backbone_name: str,
    image_size: int,
    batch_size: int,
    device,
) -> np.ndarray:
    """
    Trich embedding L2-normalize cho toan bo image_paths bang 1
    backbone DINOv2 NHO va DONG BANG (chi de do similarity, khong
    lien quan gi den model dang train o Stage 1).

    Can torch/torchvision/PIL that de chay - khong the smoke-test
    trong moi truong khong co torch.
    """
    import torch
    from PIL import Image
    from torchvision import transforms

    backbone = torch.hub.load(
        repo_or_dir="facebookresearch/dinov2",
        model=backbone_name,
    ).to(device)
    backbone.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    all_embeddings = []

    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            batch_tensors = []
            for path in batch_paths:
                image = Image.open(path).convert("RGB")
                batch_tensors.append(transform(image))

            batch = torch.stack(batch_tensors, dim=0).to(device)

            features = backbone(batch)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            all_embeddings.append(features.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)
