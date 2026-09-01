"""
Figure-Caption Plagiarism Pretraining
Losses
--------------------------------------------------------------
Symmetric InfoNCE (kieu CLIP): figure[i] phai khop caption[i]
trong cung 1 batch, cac cap khac trong batch la negative.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def contrastive_loss(
    figure_embeddings: torch.Tensor,
    caption_embeddings: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    figure_embeddings, caption_embeddings: (batch, dim), da L2-normalize.
    """
    scale = logit_scale.exp()

    logits_per_figure = scale * figure_embeddings @ caption_embeddings.t()
    logits_per_caption = logits_per_figure.t()

    targets = torch.arange(
        figure_embeddings.size(0),
        device=figure_embeddings.device,
    )

    loss_figure_to_caption = F.cross_entropy(logits_per_figure, targets)
    loss_caption_to_figure = F.cross_entropy(logits_per_caption, targets)

    return (loss_figure_to_caption + loss_caption_to_figure) / 2.0
