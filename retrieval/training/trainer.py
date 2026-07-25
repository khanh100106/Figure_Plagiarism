"""
Paper2Fig-2026
Trainer
"""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from torch.amp import GradScaler
from typing import Any, Dict, Optional
from retrieval.training.logger import Logger
from retrieval.training.metrics import (
    MetricTracker,
)

from retrieval.training.evaluator import (
    RetrievalEvaluator,
)
from retrieval.training.early_stopping import EarlyStopping
from retrieval.training.checkpoint import (
    save_checkpoint,
    resume_checkpoint,
)
from retrieval.training.status import (
    update_status,
)
class Trainer:
    """
    Generic Retrieval Trainer.
    This class is responsible for
    - training
    - validation
    - checkpoint
    - logging
    - scheduler
    - early stopping
    It DOES NOT contain any model-specific logic.
    """
    def __init__(
            self,
            *,
            model: nn.Module,
            criterion: nn.Module,
            optimizer,
            scheduler,
            logger: Logger,
            metric_tracker: MetricTracker,
            evaluator: RetrievalEvaluator,
            early_stopping: EarlyStopping,
            train_loader,
            val_loader,
            experiment_dir: Path,
            device: torch.device,
            use_amp: bool = True,
            gradient_clip: float | None = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.metric_tracker = metric_tracker
        self.evaluator = evaluator
        self.early_stopping = early_stopping
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment_dir = experiment_dir
        #
        # Device
        #
        self.device = device
        self.amp_device = self.device.type
        #
        # Automatic Mixed Precision
        #
        self.use_amp = (
                use_amp
                and self.device.type == "cuda"
        )
        self.scaler = (
            GradScaler(
                self.amp_device,
            )
            if self.use_amp
            else None
        )
        #
        # Gradient
        #
        self.gradient_clip = gradient_clip
        #
        # Training State
        #
        self.current_epoch = 0
        self.best_epoch = 0
        self.best_score = 0.0
        self.start_epoch = 1
        #
        # Move model
        #
        self.model.to(
            self.device,
        )
        self.logger.log(
            "Trainer initialized.",
        )
    def resume(
        self,
    ) -> None:
        """
        Resume training from checkpoint.
        """
        checkpoint = resume_checkpoint(
            self.experiment_dir,
            self.device,
        )
        self.model.load_state_dict(
            checkpoint["model"],
        )
        self.optimizer.load_state_dict(
            checkpoint["optimizer"],
        )
        if (
            self.scheduler is not None
            and
            checkpoint["scheduler"] is not None
        ):
            self.scheduler.load_state_dict(
                checkpoint["scheduler"],
            )
        if (
            self.scaler is not None
            and
            "scaler" in checkpoint
        ):
            self.scaler.load_state_dict(
                checkpoint["scaler"],
            )
        self.early_stopping.load_state_dict(
            checkpoint["early_stopping"],
        )
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_score = checkpoint["best_score"]
        self.best_epoch = self.early_stopping.best_epoch
        self.logger.log(
            f"Resume from epoch {checkpoint['epoch']}",
        )
    # =====================================================
    # Helpers
    # =====================================================
    def _move_to_device(
        self,
        batch: Any,
    ):
        """
        Move batch recursively to target device.
        """
        if torch.is_tensor(batch):
            return batch.to(
                self.device,
                non_blocking=True,
            )
        if isinstance(
            batch,
            dict,
        ):
            return {
                key: self._move_to_device(value)
                for key, value in batch.items()
            }
        if isinstance(
            batch,
            (list, tuple),
        ):
            return type(batch)(
                self._move_to_device(item)
                for item in batch
            )
        return batch
    def _set_train_mode(
        self,
    ) -> None:
        """
        Switch model to training mode.
        """
        self.model.train()
    def _set_eval_mode(
        self,
    ) -> None:
        """
        Switch model to evaluation mode.
        """
        self.model.eval()
    # =====================================================
    # Forward
    # =====================================================
    def _forward(
            self,
            batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Forward one batch.
        Supports:
            - pair only
            - triplet only
            - pair + triplet
        """
        outputs: Dict[str, Any] = {}
        #
        # Pair
        #
        if "pair" in batch:
            pair = batch["pair"]
            outputs["pair"] = {
                "anchor": self.model(
                    pair["anchor"],
                ),
                "target": self.model(
                    pair["target"],
                ),
                "label": pair["label"],
            }
        #
        # Triplet
        #
        if "triplet" in batch:
            triplet = batch["triplet"]
            outputs["triplet"] = {
                "anchor": self.model(
                    triplet["anchor"],
                ),
                "positive": self.model(
                    triplet["positive"],
                ),
                "negative": self.model(
                    triplet["negative"],
                ),
            }
        return outputs
    # =====================================================
    # Compute Loss
    # =====================================================
    def _compute_loss(
            self,
            outputs: Dict[str, Any],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Compute retrieval loss.
        """
        pair = outputs.get("pair")
        triplet = outputs.get("triplet")
        return self.criterion(
            pair_anchor=(
                pair["anchor"]
                if pair is not None
                else None
            ),
            pair_target=(
                pair["target"]
                if pair is not None
                else None
            ),
            pair_label=(
                pair["label"]
                if pair is not None
                else None
            ),
            triplet_anchor=(
                triplet["anchor"]
                if triplet is not None
                else None
            ),
            triplet_positive=(
                triplet["positive"]
                if triplet is not None
                else None
            ),
            triplet_negative=(
                triplet["negative"]
                if triplet is not None
                else None
            ),
        )
    def _train_step(
        self,
        batch: dict,
    ) -> dict:
        """
        Train one batch.
        """
        batch = self._move_to_device(
            batch,
        )
        self.optimizer.zero_grad(
            set_to_none=True,
        )
        if self.use_amp:
            with torch.autocast(
                device_type="cuda",
            ):
                outputs = self._forward(
                    batch,
                )
                loss_dict = self._compute_loss(
                    outputs,
                )
                loss = loss_dict["loss"]
            self.scaler.scale(
                loss,
            ).backward()
            if self.gradient_clip is not None:
                self.scaler.unscale_(
                    self.optimizer,
                )
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip,
                )
            self.scaler.step(
                self.optimizer,
            )
            self.scaler.update()
        else:
            outputs = self._forward(
                batch,
            )
            loss_dict = self._compute_loss(
                outputs,
            )
            loss = loss_dict["loss"]
            loss.backward()
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip,
                )
            self.optimizer.step()
        return {
            "loss": loss,
            "loss_dict": loss_dict,
            "batch_size": self._batch_size(
                batch,
            ),
        }
    @torch.no_grad()
    def _validate_step(
        self,
        batch: dict,
    ) -> dict:
        """
        Validate one batch.
        """
        batch = self._move_to_device(
            batch,
        )
        outputs = self._forward(
            batch,
        )
        loss_dict = self._compute_loss(
            outputs,
        )
        return {
            "outputs": outputs,
            "loss_dict": loss_dict,
            "batch_size": self._batch_size(
                batch,
            ),
        }
    def _train_one_epoch(
            self,
    ) -> dict:
        """
        Train one epoch.
        """
        self._set_train_mode()
        self.metric_tracker.reset()
        for batch in self.train_loader:
            result = self._train_step(
                batch,
            )
            loss = result["loss"]
            loss_dict = result["loss_dict"]
            batch_size = result["batch_size"]
            self.metric_tracker.update_train_loss(
                loss.item(),
                batch_size,
            )
            if loss_dict["positive_distance"] is not None:
                self.metric_tracker.update_distances(
                    positive_distance=loss_dict[
                        "positive_distance"
                    ].item(),
                    negative_distance=loss_dict[
                        "negative_distance"
                    ].item(),
                    batch_size=batch_size,
                )
        return self.metric_tracker.to_dict()
    @torch.no_grad()
    def validate_one_epoch(
        self,
    ) -> dict:
        """
        Validate one epoch.
        Returns
        -------
        dict
            Validation metrics.
        """
        self._set_eval_mode()
        self.metric_tracker.reset_validation()
        query_embeddings = []
        target_embeddings = []
        for batch in self.val_loader:
            result = self._validate_step(
                batch,
            )
            outputs = result["outputs"]
            loss_dict = result["loss_dict"]
            batch_size = result["batch_size"]
            self.metric_tracker.update_val_loss(
                loss_dict["loss"].item(),
                batch_size,
            )
            if loss_dict["positive_distance"] is not None:
                self.metric_tracker.update_distances(
                    positive_distance=loss_dict[
                        "positive_distance"
                    ].item(),
                    negative_distance=loss_dict[
                        "negative_distance"
                    ].item(),
                    batch_size=batch_size,
                )
            query_embeddings.append(
                outputs["pair"]["anchor"],
            )
            target_embeddings.append(
                outputs["pair"]["target"],
            )
        #
        # Retrieval evaluation
        #
        query_embeddings = torch.cat(
            query_embeddings,
            dim=0,
        )

        target_embeddings = torch.cat(
            target_embeddings,
            dim=0,
        )
        retrieval_metrics = self.evaluator.evaluate(
            query_embeddings=query_embeddings,
            target_embeddings=target_embeddings,
        )

        self.metric_tracker.update_recall(
            recall1=retrieval_metrics["recall1"],
            recall5=retrieval_metrics["recall5"],
        )
        return self.metric_tracker.to_dict()
    def fit(
        self,
        num_epochs: int,
    ) -> None:
        """
        Full training loop.
        """
        if self.start_epoch > num_epochs:
            self.logger.log(
                "Training already finished.",
            )
            return
        self.logger.log(
            "=" * 80,
        )
        self.logger.log(
            "Start Training",
        )
        self.logger.log(
            "=" * 80,
        )
        for epoch in range(
            self.start_epoch,
            num_epochs + 1,
        ):
            self.current_epoch = epoch
            #
            # Train
            #
            train_metrics = self._train_one_epoch()
            #
            # Validation
            #
            val_metrics = self.validate_one_epoch()
            #
            # Scheduler
            #
            if self.scheduler is not None:
                self.scheduler.step()
            learning_rate = self.optimizer.param_groups[0]["lr"]
            #
            # Early Stopping
            #
            early_stop = self.early_stopping.step(
                epoch=epoch,
                metrics=val_metrics,
            )
            is_best = early_stop["improved"]
            if is_best:
                self.best_epoch = epoch
                self.best_score = early_stop["best_score"]
            #
            # Save checkpoint
            #
            checkpoint = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": (
                    self.scheduler.state_dict()
                    if self.scheduler is not None
                    else None
                ),
                "best_score": self.best_score,
                "early_stopping": self.early_stopping.state_dict(),
            }
            if self.scaler is not None:
                checkpoint["scaler"] = self.scaler.state_dict()
            save_checkpoint(
                self.experiment_dir,
                checkpoint,
                is_best=is_best,
            )
            #
            # Update status
            #
            update_status(
                self.experiment_dir,
                current_epoch=epoch,
                best_epoch=self.best_epoch,
                best_recall1=self.best_score,
            )
            #
            # Logging
            #
            self.logger.log_epoch(
                epoch=epoch,
                total_epoch=num_epochs,
                train_loss=train_metrics["train_loss"],
                val_loss=val_metrics["val_loss"],
                recall1=val_metrics["recall1"],
                recall5=val_metrics["recall5"],
                positive_distance=val_metrics["positive_distance"],
                negative_distance=val_metrics["negative_distance"],
                learning_rate=learning_rate,
                is_best=is_best,
            )
            #
            # Stop
            #
            if early_stop["stop"]:
                self.logger.log(
                    f"Early stopping at epoch {epoch}.",
                )
                break
        self.logger.log(
            "=" * 80,
        )
        self.logger.log(
            "Training Finished.",
        )
        self.logger.log(
            "=" * 80,
        )
    def _batch_size(
        self,
        batch: dict,
    ) -> int:
        """
        Infer batch size automatically.
        """
        if "pair" in batch:
            return batch["pair"]["anchor"].size(
                0,
            )
        if "triplet" in batch:
            return batch["triplet"]["anchor"].size(
                0,
            )
        raise RuntimeError(
            "Cannot determine batch size."
        )
