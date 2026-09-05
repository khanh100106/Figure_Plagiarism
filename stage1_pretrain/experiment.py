"""
Figure-Caption Plagiarism Pretraining
Experiment
--------------------------------------------------------------
Moi lan chay train.py se tao 1 thu muc experiment rieng
(outputs/exp_0001, outputs/exp_0002, ...) thay vi ghi de len
ket qua cua lan chay truoc. Moi thu muc luu kem 1 file
config_used.json de sau con biet lan chay do dung cau hinh gi
(hop khi ban thu nhieu bo tham so / backbone / dataset khac nhau).
"""
from __future__ import annotations

import json
from pathlib import Path


def _existing_experiment_indices(base_dir: Path) -> list[int]:
    if not base_dir.exists():
        return []

    indices = []
    for path in base_dir.iterdir():
        if path.is_dir() and path.name.startswith("exp_"):
            try:
                indices.append(int(path.name.split("_")[1]))
            except (IndexError, ValueError):
                continue

    return indices


def create_experiment_dir(base_dir: Path) -> Path:
    """
    Tao thu muc <base_dir>/exp_XXXX moi, danh so tang dan tu dong
    (khong bi trung/ghi de len experiment cu).
    """
    base_dir = Path(base_dir)
    indices = _existing_experiment_indices(base_dir)
    next_index = (max(indices) + 1) if indices else 1

    experiment_dir = base_dir / f"exp_{next_index:04d}"
    experiment_dir.mkdir(parents=True, exist_ok=False)

    return experiment_dir


def save_config_snapshot(experiment_dir: Path, config_module) -> Path:
    """
    Luu toan bo tham so (cac bien VIET_HOA trong config.py) da
    dung cho experiment nay thanh 1 file JSON.
    """
    snapshot = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(config_module).items()
        if key.isupper()
    }

    config_path = experiment_dir / "config_used.json"
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(snapshot, file, indent=2, ensure_ascii=False)

    return config_path
