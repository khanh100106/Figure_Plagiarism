"""
Paper2Fig-2026 Retrieval Training
Learning Rate Scheduler
"""
from __future__ import annotations
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from retrieval.configs import (
    LEARNING_RATE,
    MIN_LEARNING_RATE,
    NUM_EPOCHS,
    WARMUP_EPOCHS,
)
def learning_rate_lambda(
    epoch: int,
) -> float:
    """
    Linear warmup followed by cosine decay.
    """
    if epoch < WARMUP_EPOCHS:
        return float(
            epoch + 1
        ) / float(
            WARMUP_EPOCHS
        )
    denominator = max(
        1,
        NUM_EPOCHS - WARMUP_EPOCHS,
    )

    progress = (
                       epoch - WARMUP_EPOCHS
               ) / denominator
    cosine = 0.5 * (
        1.0
        + math.cos(
            math.pi * progress
        )
    )
    min_factor = (
        MIN_LEARNING_RATE
        / LEARNING_RATE
    )
    return (
        min_factor
        + (1.0 - min_factor)
        * cosine
    )
def build_scheduler(
    optimizer: Optimizer,
) -> LambdaLR:
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=learning_rate_lambda,
    )
    return scheduler
def summarize_scheduler(
) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append(
        "Scheduler Configuration"
    )
    lines.append("=" * 60)
    lines.append(
        "Scheduler : Warmup + Cosine"
    )
    lines.append(
        "Epochs : {}".format(
            NUM_EPOCHS,
        )
    )
    lines.append(
        "Warmup Epochs : {}".format(
            WARMUP_EPOCHS,
        )
    )
    lines.append(
        "Initial LR : {}".format(
            LEARNING_RATE,
        )
    )
    lines.append(
        "Minimum LR : {}".format(
            MIN_LEARNING_RATE,
        )
    )
    return "\n".join(
        lines,
    )
