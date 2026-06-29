"""
Paper2Fig-2026 Retrieval Training
Optimizer
"""
from __future__ import annotations
import torch
from torch import nn
from torch.optim import Optimizer
from retrieval.configs import (
    OPTIMIZER,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BETAS,
    EPS,
    SGD_MOMENTUM,
)
# ============================================================
# Factory
# ============================================================
def build_optimizer(
    model: nn.Module,
) -> Optimizer:
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer_name = OPTIMIZER.lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=BETAS,
            eps=EPS,
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=BETAS,
            eps=EPS,
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            momentum=SGD_MOMENTUM,
        )
    raise ValueError(
        "Unknown optimizer: {}".format(
            OPTIMIZER,
        )
    )
# ============================================================
# Summary
# ============================================================
def summarize_optimizer(
) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append(
        "Optimizer Configuration"
    )
    lines.append("=" * 60)
    lines.append(
        "Optimizer : {}".format(
            OPTIMIZER,
        )
    )
    lines.append(
        "Learning Rate : {}".format(
            LEARNING_RATE,
        )
    )
    lines.append(
        "Weight Decay : {}".format(
            WEIGHT_DECAY,
        )
    )
    lines.append(
        "Betas : {}".format(
            BETAS,
        )
    )
    lines.append(
        "EPS : {}".format(
            EPS,
        )
    )
    return "\n".join(
        lines,
    )
