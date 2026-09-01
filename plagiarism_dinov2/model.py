"""
Figure-Caption Plagiarism Pretraining
Model
--------------------------------------------------------------
DINOv2 backbone dung chung (Siamese) + projection head, ap dung
cho ca anh figure va anh caption.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ProjectionHead(nn.Module):
    """
    MLP nho, chieu dac trung backbone ve khong gian embedding
    va chuan hoa L2 (de dung cosine similarity / contrastive loss).
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, projection_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embeddings = self.net(features)
        return F.normalize(embeddings, p=2, dim=-1)


class TwinDinoV2Encoder(nn.Module):
    """
    Mot backbone DINOv2 duy nhat, dung chung de encode ca figure
    va caption (kieu Siamese network).
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vitb14",
        projection_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone_name

        self.backbone = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2",
            model=backbone_name,
        )

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.head = ProjectionHead(
            input_dim=self.backbone.embed_dim,
            projection_dim=projection_dim,
            dropout=dropout,
        )

        # Nhiet do hoc duoc (giong CLIP), khoi tao ~ 1/0.07
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1.0 / 0.07))
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Dung ham nay khi can trich embedding cho 1 loai anh
        (vi du: trich embedding cua toan bo tap figure de tim
        kiem/so trung sau khi train xong).
        """
        features = self.backbone(images)
        return self.head(features)

    def forward(
        self,
        figure: torch.Tensor,
        caption: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        figure_embeddings = self.encode(figure)
        caption_embeddings = self.encode(caption)
        return figure_embeddings, caption_embeddings

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def __repr__(self) -> str:
        return (
            f"TwinDinoV2Encoder(backbone={self.backbone_name}, "
            f"embed_dim={self.backbone.embed_dim}, "
            f"projection_dim={self.head.net[-1].out_features}, "
            f"trainable_params={self.trainable_parameters:,})"
        )


class TextEncoder(nn.Module):
    """
    Bao mot model HuggingFace (BERT/SciBERT/...) va mean-pool cac
    token embedding (co mask) thanh 1 vector cho ca cau.
    """

    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        freeze: bool = False,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)

        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        token_embeddings = outputs.last_hidden_state  # (batch, seq, hidden)

        mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)

        return summed / counts


class ImageTextDualEncoder(nn.Module):
    """
    DINOv2 (anh figure) + text encoder HuggingFace (caption_text),
    moi nhanh co projection head rieng, chieu ve chung 1 khong
    gian embedding (kieu CLIP).
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vitb14",
        text_model_name: str = "allenai/scibert_scivocab_uncased",
        projection_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        freeze_text_encoder: bool = False,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone_name
        self.text_model_name = text_model_name

        self.image_backbone = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2",
            model=backbone_name,
        )

        if freeze_backbone:
            for parameter in self.image_backbone.parameters():
                parameter.requires_grad = False

        self.text_backbone = TextEncoder(
            model_name=text_model_name,
            freeze=freeze_text_encoder,
        )

        self.image_head = ProjectionHead(
            input_dim=self.image_backbone.embed_dim,
            projection_dim=projection_dim,
            dropout=dropout,
        )

        self.text_head = ProjectionHead(
            input_dim=self.text_backbone.hidden_size,
            projection_dim=projection_dim,
            dropout=dropout,
        )

        # Nhiet do hoc duoc (giong CLIP), khoi tao ~ 1/0.07
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1.0 / 0.07))
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        features = self.image_backbone(images)
        return self.image_head(features)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        features = self.text_backbone(input_ids, attention_mask)
        return self.text_head(features)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(input_ids, attention_mask)
        return image_embeddings, text_embeddings

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def __repr__(self) -> str:
        return (
            f"ImageTextDualEncoder(\n"
            f"  image_backbone={self.backbone_name} "
            f"(dim={self.image_backbone.embed_dim}),\n"
            f"  text_backbone={self.text_model_name} "
            f"(dim={self.text_backbone.hidden_size}),\n"
            f"  projection_dim={self.image_head.net[-1].out_features},\n"
            f"  trainable_params={self.trainable_parameters:,}\n"
            f")"
        )
