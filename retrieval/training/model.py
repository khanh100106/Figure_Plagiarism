"""
Paper2Fig-2026 Retrieval Training
Model
"""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
from retrieval.configs import (
    BACKBONE_NAME,
    EMBEDDING_DIM,
    PROJECTION_DIM,
    DROPOUT,
    FREEZE_BACKBONE,
)
# ============================================================
# Projection Head
# ============================================================
class ProjectionHead(nn.Module):
    """
    Projection Head for Metric Learning.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden_dim = input_dim
        self.layers = nn.Sequential(
            nn.LayerNorm(
                input_dim,
            ),
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout,
            ),
            nn.Linear(
                hidden_dim,
                output_dim,
            ),
        )
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.layers(
            x,
        )
# ============================================================
# Retrieval Model
# ============================================================
class RetrievalModel(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            BACKBONE_NAME,
        )
        self.projection = ProjectionHead(
            input_dim=EMBEDDING_DIM,
            output_dim=PROJECTION_DIM,
            dropout=DROPOUT,
        )
        if FREEZE_BACKBONE:
            self.freeze_backbone()
    # ============================================================
    # Freeze
    # ============================================================
    def freeze_backbone(
        self,
    ) -> None:
        """
        Freeze all backbone parameters.
        """
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
    # ============================================================
    # Unfreeze
    # ============================================================
    def unfreeze_backbone(
        self,
    ) -> None:
        """
        Unfreeze all backbone parameters.
        """
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True
    # ============================================================
    # Encode
    # ============================================================
    def encode(
            self,
            images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract normalized embeddings.
        """
        outputs = self.backbone.forward_features(
            images,
        )
        features = outputs[
            "x_norm_clstoken"
        ]
        embeddings = self.projection(
            features,
        )
        embeddings = nn.functional.normalize(
            embeddings,
            p=2,
            dim=1,
        )
        return embeddings
    # ============================================================
    # Extract Backbone Features
    # ============================================================
    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract normalized DINOv2 CLS features.
        """
        outputs = self.backbone.forward_features(
            images,
        )
        features = outputs[
            "x_norm_clstoken"
        ]
        return nn.functional.normalize(
            features,
            p=2,
            dim=1,
        )
    # ============================================================
    # Embedding Dimension
    # ============================================================
    @property
    def embedding_dim(
        self,
    ) -> int:
        return PROJECTION_DIM
    # ============================================================
    # Forward
    # ============================================================
    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode(
            images,
        )
    # ============================================================
    # Load Checkpoint
    # ============================================================
    def load_checkpoint(
        self,
        checkpoint: str | Path,
        strict: bool = True,
    ) -> None:
        """
        Load model checkpoint.
        """
        checkpoint = torch.load(
            checkpoint,
            map_location="cpu",
        )
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        self.load_state_dict(
            state_dict,
            strict=strict,
        )
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.load_state_dict(
            state_dict,
            strict=strict,
        )
    # ============================================================
    # Save Checkpoint
    # ============================================================
    def save_checkpoint(
        self,
        checkpoint: str | Path,
    ) -> None:
        """
        Save model checkpoint.
        """
        torch.save(
            {
                "model": self.state_dict(),
            },
            checkpoint,
        )
    # ============================================================
    # Number of Parameters
    # ============================================================
    @property
    def num_parameters(
        self,
    ) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
        )
    @property
    def trainable_parameters(
        self,
    ) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )