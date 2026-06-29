"""
Paper2Fig-2026

Shared Constants
"""

# ============================================================
# Dataset
# ============================================================

PAIR_COLUMNS = [
    "anchor_id",
    "target_id",
    "label",
]

TRIPLET_COLUMNS = [
    "anchor_id",
    "positive_id",
    "negative_id",
]

METADATA_COLUMNS = [
    "figure_id",
    "image_path",
]

# ============================================================
# Experiment Files
# ============================================================

BEST_CHECKPOINT = "best.pt"

LAST_CHECKPOINT = "last.pt"

STATUS_FILE = "status.json"

SUMMARY_FILE = "summary.txt"

CONFIG_FILE = "config.json"

METRICS_FILE = "metrics.csv"

LOG_FILE = "train.log"

# ============================================================
# Experiment Modes
# ============================================================

MODE_NEW = "new"

MODE_RESUME = "resume"

MODE_FORK = "fork"

# ============================================================
# Experiment Directories
# ============================================================

CHECKPOINT_DIR = "checkpoints"

CONFIG_DIR = "configs"

LOG_DIR = "logs"

METRIC_DIR = "metrics"

# ============================================================
# Experiment status
# ============================================================

STATUS_RUNNING = "running"

STATUS_COMPLETED = "completed"

STATUS_INTERRUPTED = "interrupted"

STATUS_FORKED = "forked"
