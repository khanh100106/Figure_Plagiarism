"""
Paper2Fig-2026 Retrieval Training
Training Logger
"""
from __future__ import annotations
import csv
from pathlib import Path
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ModuleNotFoundError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False
from retrieval.constants import (
    LOG_DIR,
    LOG_FILE,
    METRIC_DIR,
    METRICS_FILE,
    CHECKPOINT_EPOCH,
    METRIC_TRAIN_LOSS,
    METRIC_VAL_LOSS,
    METRIC_RECALL1,
    METRIC_RECALL5,
    METRIC_POSITIVE_DISTANCE,
    METRIC_NEGATIVE_DISTANCE,
    METRIC_LEARNING_RATE,
)
CSV_FIELDS = [
    CHECKPOINT_EPOCH,
    METRIC_TRAIN_LOSS,
    METRIC_VAL_LOSS,
    METRIC_RECALL1,
    METRIC_RECALL5,
    METRIC_POSITIVE_DISTANCE,
    METRIC_NEGATIVE_DISTANCE,
    METRIC_LEARNING_RATE,
]
class Logger:
    """
    Training logger.
    """
    def __init__(
            self,
            experiment_dir: Path,
            use_tensorboard: bool = True,
    ):
        self.use_tensorboard = (
                use_tensorboard
                and
                TENSORBOARD_AVAILABLE
        )
        self.experiment_dir = experiment_dir
        self.log_directory = (
            experiment_dir
            / LOG_DIR
        )
        self.metric_directory = (
            experiment_dir
            / METRIC_DIR
        )
        self.log_path = (
            self.log_directory
            / LOG_FILE
        )
        self.metric_path = (
            self.metric_directory
            / METRICS_FILE
        )
        if self.use_tensorboard:
            self.writer = SummaryWriter(
                log_dir=self.log_directory,
            )
        else:
            self.writer = None
        self.log_file = open(
            self.log_path,
            "w",
            encoding="utf-8",
        )
        self.metric_file = open(
            self.metric_path,
            "w",
            newline="",
            encoding="utf-8",
        )
        self.metric_writer = csv.DictWriter(
            self.metric_file,
            fieldnames=CSV_FIELDS,
        )
        self.metric_writer.writeheader()
    def log(
            self,
            message: str,
    ) -> None:
        print(
            message,
        )
        self.log_file.write(
            message + "\n",
        )
        self.log_file.flush()

    def info(
            self,
            message: str,
    ) -> None:
        self.log(message)

    def warning(
            self,
            message: str,
    ) -> None:
        self.log(f"[WARNING] {message}")

    def error(
            self,
            message: str,
    ) -> None:
        self.log(f"[ERROR] {message}")
    def write_metrics(
            self,
            *,
            epoch: int,
            train_loss: float,
            val_loss: float,
            recall1: float,
            recall5: float,
            positive_distance: float,
            negative_distance: float,
            learning_rate: float,
    ) -> None:
        """
        Write one row into metrics.csv.
        """
        self.metric_writer.writerow(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "recall1": recall1,
                "recall5": recall5,
                "positive_distance": positive_distance,
                "negative_distance": negative_distance,
                "learning_rate": learning_rate,
            }
        )
        self.metric_file.flush()
    def write_tensorboard(
            self,
            *,
            epoch: int,
            train_loss: float,
            val_loss: float,
            recall1: float,
            recall5: float,
            positive_distance: float,
            negative_distance: float,
            learning_rate: float,
    ) -> None:
        """
        Write metrics into TensorBoard.
        """
        if self.writer is None:
            return
        self.writer.add_scalar(
            "Loss/Train",
            train_loss,
            epoch,
        )
        self.writer.add_scalar(
            "Loss/Validation",
            val_loss,
            epoch,
        )
        self.writer.add_scalar(
            "Recall/Recall@1",
            recall1,
            epoch,
        )
        self.writer.add_scalar(
            "Recall/Recall@5",
            recall5,
            epoch,
        )
        self.writer.add_scalar(
            "Distance/Positive",
            positive_distance,
            epoch,
        )
        self.writer.add_scalar(
            "Distance/Negative",
            negative_distance,
            epoch,
        )
        self.writer.add_scalar(
            "LearningRate",
            learning_rate,
            epoch,
        )
        self.writer.flush()
    def log_epoch(
            self,
            *,
            epoch: int,
            total_epoch: int,
            train_loss: float,
            val_loss: float,
            recall1: float,
            recall5: float,
            positive_distance: float,
            negative_distance: float,
            learning_rate: float,
            is_best: bool,
    ) -> None:
        """
        Log one training epoch.
        """
        self.log("=" * 60)
        self.log(
            f"Epoch {epoch}/{total_epoch}"
        )
        self.log("=" * 60)
        self.log(
            f"Train Loss          : {train_loss:.6f}"
        )
        self.log(
            f"Validation Loss     : {val_loss:.6f}"
        )
        self.log(
            f"Recall@1            : {recall1:.6f}"
        )
        self.log(
            f"Recall@5            : {recall5:.6f}"
        )
        self.log(
            f"Positive Distance   : {positive_distance:.6f}"
        )
        self.log(
            f"Negative Distance   : {negative_distance:.6f}"
        )
        self.log(
            f"Learning Rate       : {learning_rate:.6e}"
        )
        self.log(
            f"Checkpoint          : {'Best' if is_best else 'Last'}"
        )
        self.write_metrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            recall1=recall1,
            recall5=recall5,
            positive_distance=positive_distance,
            negative_distance=negative_distance,
            learning_rate=learning_rate,
        )
        self.write_tensorboard(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            recall1=recall1,
            recall5=recall5,
            positive_distance=positive_distance,
            negative_distance=negative_distance,
            learning_rate=learning_rate,
        )
    def close(
            self,
    ) -> None:
        self.metric_file.flush()
        self.log_file.flush()
        self.metric_file.close()
        self.log_file.close()
        if self.writer is not None:
            self.writer.close()
    def __del__(
            self,
    ):
        try:
            self.close()
        except Exception:
            pass