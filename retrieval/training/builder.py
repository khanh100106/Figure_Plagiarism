"""
Paper2Fig-2026 Retrieval Framework
Training Builder
----------------
Responsibilities
----------------
- Build every component required for training.
- Assemble the complete training pipeline.
- Keep train.py clean.
Author
------
Nguyen Khanh
"""
from __future__ import annotations
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from retrieval.configs import (
    DEVICE,
    EXPERIMENT_DIR,
)
from retrieval.training.dataloader_factory import (
    build_train_dataloader,
    build_validation_dataloader,
)
from retrieval.training.dataset_factory import (
    build_train_dataset,
    build_validation_dataset,
)
from retrieval.modeling.model import (
    RetrievalModel,
)
from retrieval.training.loss import (
    RetrievalLoss,
)
from retrieval.training.optimizer import (
    build_optimizer,
)
from retrieval.training.scheduler import (
    build_scheduler,
)
from retrieval.training.logger import (
    Logger,
)
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
# ============================================================
# Model
# ============================================================
def build_model() -> RetrievalModel:
    """
    Build retrieval model.
    Returns
    -------
    RetrievalModel
        Initialized retrieval model.
    """
    return RetrievalModel()
# ============================================================
# Loss
# ============================================================
def build_loss() -> RetrievalLoss:
    """
    Build retrieval loss.
    Returns
    -------
    RetrievalLoss
        Initialized retrieval loss.
    """
    return RetrievalLoss()
# ============================================================
# Optimizer
# ============================================================
def build_training_optimizer(
    model: RetrievalModel,
) -> Optimizer:
    """
    Build optimizer.
    Parameters
    ----------
    model : RetrievalModel
    Returns
    -------
    Optimizer
    """
    return build_optimizer(
        model,
    )
# ============================================================
# Scheduler
# ============================================================
def build_training_scheduler(
    optimizer: Optimizer,
) -> LRScheduler:
    """
    Build learning rate scheduler.
    Parameters
    ----------
    optimizer : Optimizer
    Returns
    -------
    LRScheduler
    """
    return build_scheduler(
        optimizer,
    )
# ============================================================
# Dataset
# ============================================================
def build_datasets() -> tuple[Dataset, Dataset]:
    """
    Build training and validation datasets.
    """
    train_dataset = build_train_dataset()
    validation_dataset = build_validation_dataset()
    return (
        train_dataset,
        validation_dataset,
    )
# ============================================================
# DataLoader
# ============================================================
def build_dataloaders() -> tuple[DataLoader, DataLoader]:
    """
    Build training and validation dataloaders.
    """
    train_dataset, validation_dataset = build_datasets()
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
# Logger
# ============================================================
def build_logger() -> Logger:
    """
    Build training logger.
    """
    return Logger(
        experiment_dir=EXPERIMENT_DIR,
    )
# ============================================================
# Metric Tracker
# ============================================================
def build_metric_tracker() -> MetricTracker:
    """
    Build metric tracker.
    """
    return MetricTracker()
# ============================================================
# Evaluator
# ============================================================
def build_evaluator() -> RetrievalEvaluator:
    """
    Build retrieval evaluator.
    """
    return RetrievalEvaluator()
# ============================================================
# Early Stopping
# ============================================================
def build_early_stopping() -> EarlyStopping:
    """
    Build early stopping.
    """
    return EarlyStopping()
# ============================================================
# Trainer
# ============================================================
def build_trainer() -> Trainer:
    """
    Build the complete training pipeline.
    Returns
    -------
    Trainer
    """
    model = build_model()
    criterion = build_loss()
    optimizer = build_training_optimizer(
        model,
    )
    scheduler = build_training_scheduler(
        optimizer,
    )
    train_loader, validation_loader = build_dataloaders()
    logger = build_logger()
    metric_tracker = build_metric_tracker()
    evaluator = build_evaluator()
    early_stopping = build_early_stopping()
    return Trainer(
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
        device=DEVICE,
    )
