"""
Paper2Fig-2026 Retrieval Framework

Training Entry Point

Responsibilities
----------------
- Build training pipeline.
- Launch training.

Author
------
Nguyen Khanh
"""

from __future__ import annotations

import torch

from retrieval.configs import (
    DEVICE,
    NUM_EPOCHS,
    EXPERIMENT_DIR,
)

from retrieval.models import RetrievalModel

from retrieval.training.loss import RetrievalLoss

from retrieval.training.optimizer import (
    build_optimizer,
)

from retrieval.training.scheduler import (
    build_scheduler,
)

from retrieval.training.logger import Logger

from retrieval.training.metrics import (
    MetricTracker,
)

from retrieval.training.evaluator import (
    RetrievalEvaluator,
)

from retrieval.training.early_stopping import (
    EarlyStopping,
)

from retrieval.training.trainer import (
    Trainer,
)

from retrieval.training.dataset_factory import (
    build_train_dataset,
    build_validation_dataset,
)

from retrieval.training.dataloader_factory import (
    build_train_dataloader,
    build_validation_dataloader,
)

# ============================================================
# Build DataLoader
# ============================================================

def build_dataloaders():
    """
    Build training and validation dataloaders.
    """

    train_dataset = build_train_dataset()

    validation_dataset = build_validation_dataset()

    train_loader = build_train_dataloader(
        train_dataset,
    )

    validation_loader = build_validation_dataloader(
        validation_dataset,
    )

    return (
        train_loader,
        validation_loader,
    )

# ============================================================
# Build Trainer
# ============================================================

def build_trainer() -> Trainer:
    """
    Build the complete training pipeline.
    """

    #
    # Data
    #

    train_loader, validation_loader = (
        build_dataloaders()
    )

    #
    # Model
    #

    model = RetrievalModel()

    #
    # Loss
    #

    criterion = RetrievalLoss()

    #
    # Optimizer
    #

    optimizer = build_optimizer(
        model,
    )

    #
    # Scheduler
    #

    scheduler = build_scheduler(
        optimizer,
    )

    #
    # Logger
    #

    logger = Logger(
        experiment_dir=EXPERIMENT_DIR,
    )

    #
    # Metrics
    #

    metric_tracker = MetricTracker()

    #
    # Evaluator
    #

    evaluator = RetrievalEvaluator()

    #
    # Early Stopping
    #

    early_stopping = EarlyStopping()

    #
    # Trainer
    #

    trainer = Trainer(

        model=model,

        criterion=criterion,

        optimizer=optimizer,

        scheduler=scheduler,

        logger=logger,

        metric_tracker=metric_tracker,

        evaluator=evaluator,

        early_stopping=early_stopping,

        train_loader=train_loader,

        val_loader=validation_loader,

        experiment_dir=EXPERIMENT_DIR,

        device=torch.device(
            DEVICE,
        ),

    )

    return trainer

