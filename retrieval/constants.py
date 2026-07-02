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
# ============================================================
# Metrics
# ============================================================
METRIC_TRAIN_LOSS = "train_loss"
METRIC_VAL_LOSS = "val_loss"
METRIC_RECALL1 = "recall1"
METRIC_RECALL5 = "recall5"
METRIC_POSITIVE_DISTANCE = "positive_distance"
METRIC_NEGATIVE_DISTANCE = "negative_distance"
METRIC_LEARNING_RATE = "learning_rate"
# ============================================================
# Direction
# ============================================================
DIRECTION_MAX = "max"
DIRECTION_MIN = "min"
# ============================================================
# Early Stopping
# ============================================================
EARLY_STOP_REASON_PATIENCE = "patience"
EARLY_STOP_REASON_THRESHOLD = "threshold"
# ============================================================
# Checkpoint Keys
# ============================================================
CHECKPOINT_MODEL = "model"
CHECKPOINT_OPTIMIZER = "optimizer"
CHECKPOINT_SCHEDULER = "scheduler"
CHECKPOINT_EPOCH = "epoch"
CHECKPOINT_BEST_SCORE = "best_score"
CHECKPOINT_EARLY_STOPPING = "early_stopping"
CHECKPOINT_SCALER = "scaler"
# ============================================================
# Status Keys
# ============================================================
STATUS_CURRENT_EPOCH = "current_epoch"
STATUS_BEST_EPOCH = "best_epoch"
STATUS_BEST_RECALL1 = "best_recall1"
STATUS_CREATED_AT = "created_at"
STATUS_UPDATED_AT = "updated_at"
