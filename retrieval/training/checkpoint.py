"""
Paper2Fig-2026 Retrieval Training
Checkpoint Management
"""
from __future__ import annotations
import shutil
from pathlib import Path
import torch
from retrieval.configs import (
    EXPERIMENT_DIR,
)
from retrieval.constants import (
    BEST_CHECKPOINT,
    LAST_CHECKPOINT,
    CHECKPOINT_DIR,
    CONFIG_DIR,
    LOG_DIR,
    METRIC_DIR,
    SUMMARY_FILE,
)
# ============================================================
# Helpers
# ============================================================
def experiment_name(
    index: int,
) -> str:
    return f"exp_{index:04d}"
def list_experiments(
) -> list[Path]:
    if not EXPERIMENT_DIR.exists():
        return []
    experiments = sorted(
        [
            path
            for path in EXPERIMENT_DIR.iterdir()
            if path.is_dir()
        ]
    )
    return experiments
def next_experiment_name(
) -> str:
    experiments = list_experiments()
    if len(experiments) == 0:
        return experiment_name(1)
    last = experiments[-1].name
    number = int(
        last.split("_")[1]
    )
    return experiment_name(
        number + 1
    )
# ============================================================
# Experiments
# ============================================================
def create_experiment(
) -> Path:
    experiment = (
        EXPERIMENT_DIR
        / next_experiment_name()
    )
    (
            experiment
            / CHECKPOINT_DIR
    ).mkdir(
        parents=True,
    )
    (
            experiment
            / LOG_DIR
    ).mkdir()
    (
            experiment
            / METRIC_DIR
    ).mkdir()
    (
            experiment
            / CONFIG_DIR
    ).mkdir()
    return experiment
def resume_checkpoint(
    experiment_dir: Path,
    device: str | torch.device = "cpu",
) -> dict:
    """
    Resume from last checkpoint.
    """
    checkpoint_path = get_last_checkpoint(
        experiment_dir,
    )
    checkpoint = load_checkpoint(
        checkpoint_path,
        device=device,
    )
    return checkpoint
def fork_experiment(
    source_experiment: Path,
) -> Path:
    """
    Fork an existing experiment.
    Parameters
    ----------
    source_experiment : Path
    Returns
    -------
    Path
        New experiment directory.
    """
    target_experiment = create_experiment()
    source_checkpoint = get_checkpoint_directory(
        source_experiment,
    )
    target_checkpoint = get_checkpoint_directory(
        target_experiment,
    )
    for file in source_checkpoint.glob("*.pt"):
        shutil.copy2(
            file,
            target_checkpoint / file.name,
        )
    return target_experiment
# ============================================================
# Paths
# ============================================================
def get_checkpoint_directory(
    experiment_dir: Path,
) -> Path:
    """
    Return checkpoint directory.
    """
    return (
        experiment_dir
        / CHECKPOINT_DIR
    )
def get_last_checkpoint(
    experiment_dir: Path,
) -> Path:
    """
    Return last checkpoint path.
    """
    return (
        get_checkpoint_directory(
            experiment_dir,
        )
        / LAST_CHECKPOINT
    )
def get_best_checkpoint(
    experiment_dir: Path,
) -> Path:
    """
    Return best checkpoint path.
    """
    return (
        get_checkpoint_directory(
            experiment_dir,
        )
        / BEST_CHECKPOINT
    )
# ============================================================
# Checkpoint IO
# ============================================================
def save_checkpoint(
    experiment_dir: Path,
    checkpoint: dict,
    is_best: bool = False,
) -> None:
    """
    Save training checkpoint.
    Parameters
    ----------
    experiment_dir : Path
    checkpoint : dict
    is_best : bool
    """
    checkpoint_path = get_last_checkpoint(
        experiment_dir,
    )
    torch.save(
        checkpoint,
        checkpoint_path,
    )
    if is_best:
        best_path = get_best_checkpoint(
            experiment_dir,
        )
        shutil.copy2(
            checkpoint_path,
            best_path,
        )
def load_checkpoint(
    checkpoint_path: Path,
    device: str | torch.device = "cpu",
) -> dict:
    """
    Load checkpoint.
    Parameters
    ----------
    checkpoint_path : Path
    device : str | torch.device
    Returns
    -------
    dict
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )
    if not isinstance(
            checkpoint,
            dict,
    ):
        raise TypeError(
            "Checkpoint must be a dictionary."
        )
    required_keys = [
        "epoch",
        "model",
    ]
    missing = [
        key
        for key in required_keys
        if key not in checkpoint
    ]
    if missing:
        raise KeyError(
            f"Missing checkpoint keys: {missing}"
        )
    return checkpoint
# ============================================================
# Summary
# ============================================================
def save_summary(
    experiment_dir: Path,
    text: str,
) -> None:
    """
    Save experiment summary.
    """
    summary_path = (
        experiment_dir
        / SUMMARY_FILE
    )
    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            text,
        )
def load_summary(
    experiment_dir: Path,
) -> str:
    """
    Load experiment summary.
    """
    summary_path = (
        experiment_dir
        / SUMMARY_FILE
    )
    if not summary_path.exists():
        return ""
    with open(
        summary_path,
        "r",
        encoding="utf-8",
    ) as file:
        return file.read()
# ============================================================
# Helper
# ============================================================
def checkpoint_exists(
    experiment_dir: Path,
) -> bool:
    return get_last_checkpoint(
        experiment_dir,
    ).exists()
def best_checkpoint_exists(
    experiment_dir: Path,
) -> bool:
    return get_best_checkpoint(
        experiment_dir,
    ).exists()
