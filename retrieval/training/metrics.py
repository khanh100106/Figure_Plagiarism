"""
Paper2Fig-2026 Retrieval Training
Metrics
"""
from __future__ import annotations
from dataclasses import (
    dataclass,
    field,
)
import torch
import torch.nn.functional as F
# ============================================================
# Average Meter
# ============================================================
class AverageMeter:
    def __init__(
        self,
    ) -> None:
        self.reset()
    def reset(
        self,
    ) -> None:
        self.value = 0.0
        self.sum = 0.0
        self.count = 0
        self.average = 0.0
    def update(
        self,
        value: float,
        n: int = 1,
    ) -> None:
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count
# ============================================================
# Metric Tracker
# ============================================================
@dataclass
class MetricTracker:
    train_loss: AverageMeter = field(
        default_factory=AverageMeter,
    )
    val_loss: AverageMeter = field(
        default_factory=AverageMeter,
    )
    recall1: AverageMeter = field(
        default_factory=AverageMeter,
    )
    recall5: AverageMeter = field(
        default_factory=AverageMeter,
    )
    positive_distance: AverageMeter = field(
        default_factory=AverageMeter,
    )
    negative_distance: AverageMeter = field(
        default_factory=AverageMeter,
    )
    def reset(
        self,
    ) -> None:
        self.train_loss.reset()
        self.val_loss.reset()
        self.recall1.reset()
        self.recall5.reset()
        self.positive_distance.reset()
        self.negative_distance.reset()
    def reset_train(
            self,
    ) -> None:
        """
        Reset training metrics.
        """
        self.train_loss.reset()
    def reset_validation(
            self,
    ) -> None:
        """
        Reset validation metrics.
        """
        self.val_loss.reset()
        self.recall1.reset()
        self.recall5.reset()
        self.positive_distance.reset()
        self.negative_distance.reset()
    def update_train_loss(
        self,
        loss: float,
        batch_size: int,
    ) -> None:
        self.train_loss.update(
            loss,
            batch_size,
        )
    def update_val_loss(
        self,
        loss: float,
        batch_size: int,
    ) -> None:
        self.val_loss.update(
            loss,
            batch_size,
        )

    def update_recall(
            self,
            recall1: float,
            recall5: float,
    ) -> None:
        """
        Update retrieval metrics.

        Recall is computed once over the entire
        validation set, therefore it does not
        require batch-size weighting.
        """

        self.recall1.value = recall1
        self.recall1.average = recall1

        self.recall5.value = recall5
        self.recall5.average = recall5
    def update_distances(
        self,
        positive_distance: float,
        negative_distance: float,
        batch_size: int,
    ) -> None:
        self.positive_distance.update(
            positive_distance,
            batch_size,
        )
        self.negative_distance.update(
            negative_distance,
            batch_size,
        )
    def to_dict(
        self,
    ) -> dict:
        return {
            "train_loss":
                self.train_loss.average,
            "val_loss":
                self.val_loss.average,
            "recall1":
                self.recall1.average,
            "recall5":
                self.recall5.average,
            "positive_distance":
                self.positive_distance.average,
            "negative_distance":
                self.negative_distance.average,
        }
    def get(
            self,
            name: str,
    ) -> float:
        """
        Get metric value by name.
        """
        if not hasattr(
                self,
                name,
        ):
            raise AttributeError(
                f"Unknown metric: {name}"
            )
        meter = getattr(
            self,
            name,
        )
        return meter.average

# ============================================================
# Summary
# ============================================================
def summarize_metrics(
    tracker: MetricTracker,
) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append(
        "Validation Metrics"
    )
    lines.append("=" * 60)
    metrics = tracker.to_dict()
    for key, value in metrics.items():
        lines.append(
            f"{key:20s}: {value:.6f}"
        )
    return "\n".join(
        lines,
    )