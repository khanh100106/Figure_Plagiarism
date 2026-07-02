"""
Paper2Fig-2026 Retrieval Training
Experiment Status
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from retrieval.constants import (
    STATUS_FILE,
    STATUS_CURRENT_EPOCH,
    STATUS_BEST_EPOCH,
    STATUS_BEST_RECALL1,
    STATUS_CREATED_AT,
    STATUS_UPDATED_AT,
)
# ============================================================
# Status
# ============================================================
def save_status(
    experiment_dir: Path,
    status: dict,
) -> None:
    """
    Save experiment status.
    """
    status_path = (
        experiment_dir
        / STATUS_FILE
    )
    with open(
        status_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            status,
            file,
            indent=4,
        )
def load_status(
    experiment_dir: Path,
) -> dict:
    """
    Load experiment status.
    """
    status_path = (
        experiment_dir
        / STATUS_FILE
    )
    if not status_path.exists():
        return {}
    with open(
        status_path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file,
        )
def create_status(
    mode: str,
) -> dict:
    """
    Create default experiment status.
    """
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S",
    )
    return {
        STATUS_CURRENT_EPOCH: 0,
        STATUS_BEST_EPOCH: 0,
        STATUS_BEST_RECALL1: 0.0,
        STATUS_CREATED_AT: timestamp,
        STATUS_UPDATED_AT: timestamp,
    }
def update_status(
    experiment_dir: Path,
    **kwargs,
) -> None:
    """
    Update experiment status.
    """
    status = load_status(
        experiment_dir,
    )
    if not status:
        raise FileNotFoundError(
            STATUS_FILE,
        )
    for key, value in kwargs.items():
        status[key] = value
    status[STATUS_UPDATED_AT] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S",
    )
    save_status(
        experiment_dir,
        status,
    )