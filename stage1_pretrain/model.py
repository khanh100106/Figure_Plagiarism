"""
Stage 1 - Image Plagiarism Pretraining
Model
--------------------------------------------------------------
1 encoder DINOv2 duy nhat, dung chung cho ca anchor/positive/
negative (Siamese). Checkpoint sau khi train se duoc nap lai vao
image_backbone cua ImageTextDualEncoder o plagiarism_dinov2/
(xem export_backbone.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, projection_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embeddings = self.net(features)
        return F.normalize(embeddings, p=2, dim=-1)


class PlagiarismEncoder(nn.Module):
    """
    1 backbone DINOv2 duy nhat + projection head, dung cho ca 3
    nhanh (anchor/positive/negative) cua triplet.
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vitb14",
        projection_dim: int = 256,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone_name
        self.freeze_backbone = freeze_backbone

        self.backbone = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2",
            model=backbone_name,
        )

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()

        self.head = ProjectionHead(
            input_dim=self.backbone.embed_dim,
            projection_dim=projection_dim,
            dropout=dropout,
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.head(features)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.encode(anchor),
            self.encode(positive),
            self.encode(negative),
        )

    def train(self, mode: bool = True) -> "PlagiarismEncoder":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    @property
    def trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def __repr__(self) -> str:
        return (
            f"PlagiarismEncoder(backbone={self.backbone_name}, "
            f"embed_dim={self.backbone.embed_dim}, "
            f"projection_dim={self.head.net[-1].out_features}, "
            f"trainable_params={self.trainable_parameters:,})"
        )
