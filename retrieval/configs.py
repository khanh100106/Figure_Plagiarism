"""
Global configuration for the Paper2Fig-2026 Retrieval Framework.
Author: Nguyen Khanh
Project: Paper2Fig-2026
"""
from pathlib import Path
#import torch
# ============================================================
# Project Directories
# ============================================================
# train/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# Dataset Selection
# ============================================================

TRAIN_DATASET = "paper2fig"

SUPPORTED_DATASETS = {

    "paper2fig": {

        "root": PROJECT_ROOT / "paper2fig2026",

        "figure_dir": "figures",

        "caption_dir": "captions",

        "metadata": "metadata/paper2fig2026_metadata.csv",

        "analysis_dir": "analysis",

    },

    # Future datasets
    #
    # "docfigure": {
    #     "root": PROJECT_ROOT / "docfigure",
    #     "figure_dir": "figures",
    #     "caption_dir": "captions",
    #     "metadata": "metadata/docfigure.csv",
    #     "analysis_dir": "analysis",
    # },

}

if TRAIN_DATASET not in SUPPORTED_DATASETS:
    raise ValueError(
        f"Unsupported dataset: {TRAIN_DATASET}"
    )

DATASET_CONFIG = SUPPORTED_DATASETS[
    TRAIN_DATASET
]

DATASET_ROOT = DATASET_CONFIG["root"]

FIGURE_DIR = (
    DATASET_ROOT /
    DATASET_CONFIG["figure_dir"]
)

CAPTION_DIR = (
    DATASET_ROOT /
    DATASET_CONFIG["caption_dir"]
)

METADATA_FILE = (
    DATASET_ROOT /
    DATASET_CONFIG["metadata"]
)

ANALYSIS_DIR = (
    DATASET_ROOT /
    DATASET_CONFIG["analysis_dir"]
)
# ============================================================
# Output Directories
# ============================================================
EMBEDDING_DIR = ANALYSIS_DIR / "embeddings"
FAISS_DIR = ANALYSIS_DIR / "faiss"
MINING_DIR = ANALYSIS_DIR / "mining"
DATASET_OUTPUT_DIR = ANALYSIS_DIR / "datasets"
METADATA_DIR = DATASET_ROOT / "metadata"
CHECKPOINT_DIR = ANALYSIS_DIR / "checkpoints"
LOG_DIR = ANALYSIS_DIR / "logs"
# Create folders automatically
for folder in [
    EMBEDDING_DIR,
    FAISS_DIR,
    MINING_DIR,
    DATASET_OUTPUT_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)
# ============================================================
# Hardware
# ============================================================

import torch

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_WORKERS = 8
PIN_MEMORY = True
# ============================================================
# Model
# ============================================================
BACKBONE = "dinov2"
MODEL_NAME = "dinov2_vitb14"
IMAGE_SIZE = 518
EMBEDDING_DIM = 768
# ============================================================
# Embedding Extraction
# ============================================================
BATCH_SIZE = 64
SAVE_EVERY = 1000
CHECKPOINT_EVERY = 500
# ============================================================
# FAISS
# ============================================================
TOP_K = 20
USE_GPU_FAISS = True
FAISS_INDEX_FILE = FAISS_DIR / "index.faiss"
FAISS_INFO_FILE = FAISS_DIR / "faiss_info.json"
FAISS_COMPLETED_FLAG = FAISS_DIR / "completed.flag"
FAISS_CHECKPOINT_FILE = (
    CHECKPOINT_DIR / "faiss_checkpoint.json"
)
FAISS_INDEX_TYPE = "IndexFlatIP"
FAISS_METRIC = "InnerProduct"
# ============================================================
# Similarity Threshold
# (Initial values, will be tuned later)
# ============================================================
POSITIVE_THRESHOLD = 0.95
# ============================================================
# Random Seed
# ============================================================
SEED = 42
# ============================================================
# Logging
# ============================================================
LOG_LEVEL = "INFO"
# ============================================================
# Supported Image Extensions
# ============================================================
IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
)
# ============================================================
# Debug
# ============================================================
DEBUG = False
DEBUG_SAMPLES = 100
# ============================================================
# Positive Mining
# ============================================================
POSITIVE_PAIR_FILE = (
    MINING_DIR / "positive_pairs.csv"
)
MINING_INFO_FILE = (
    MINING_DIR / "mining_info.json"
)
MINING_COMPLETED_FLAG = (
    MINING_DIR / "completed.flag"
)
MINING_CHECKPOINT_FILE = (
    CHECKPOINT_DIR /
    "positive_mining_checkpoint.json"
)
SEARCH_BATCH_SIZE = 1000
# ============================================================
# Mining Outputs
# ============================================================
TOPK_CANDIDATE_FILE = (
    MINING_DIR / "topk_candidates.csv"
)
# ============================================================
# Retrieval Dataset
# ============================================================
RETRIEVAL_TRIPLET_FILE = (
    DATASET_OUTPUT_DIR / "retrieval_triplets.csv"
)
RETRIEVAL_PAIR_FILE = (
    DATASET_OUTPUT_DIR / "retrieval_pairs.csv"
)
DATASET_INFO_FILE = (
    DATASET_OUTPUT_DIR / "dataset_info.json"
)
DATASET_COMPLETED_FLAG = (
    DATASET_OUTPUT_DIR / "completed.flag"
)
DATASET_CHECKPOINT_FILE = (
    CHECKPOINT_DIR / "dataset_checkpoint.json"
)
MAX_POSITIVES_PER_ANCHOR = 3
MAX_NEGATIVES_PER_ANCHOR = 5
# ============================================================
# Mining statistic
# ============================================================
SIMILARITY_STATISTICS_FILE = (
    MINING_DIR / "similarity_statistics.json"
)
THRESHOLD_ANALYSIS_FILE = (
    MINING_DIR / "threshold_analysis.csv"
)
# ============================================================
# Mining visualization
# ============================================================
SIMILARITY_HISTOGRAM_FILE = (
    MINING_DIR / "similarity_histogram.png"
)
SIMILARITY_CDF_FILE = (
    MINING_DIR / "similarity_cdf.png"
)
THRESHOLD_CURVE_FILE = (
    MINING_DIR / "threshold_curve.png"
)
# ============================================================
# Hard Negative Mining
# ============================================================
HARD_NEGATIVE_DIR = (
    ANALYSIS_DIR / "hard_negative"
)
HARD_NEGATIVE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
HARD_NEGATIVE_FILE = (
    HARD_NEGATIVE_DIR / "hard_negatives.csv"
)
INTRA_NEGATIVE_FILE = (
    HARD_NEGATIVE_DIR / "intra_paper_negatives.csv"
)
INTER_NEGATIVE_FILE = (
    HARD_NEGATIVE_DIR / "inter_paper_negatives.csv"
)
HARD_NEGATIVE_STATISTICS_FILE = (
    HARD_NEGATIVE_DIR /
    "hard_negative_statistics.json"
)
HARD_NEGATIVE_INFO_FILE = (
    HARD_NEGATIVE_DIR /
    "hard_negative_info.json"
)
HARD_NEGATIVE_HISTOGRAM_FILE = (
    HARD_NEGATIVE_DIR /
    "hard_negative_histogram.png"
)
NEGATIVE_TYPE_PLOT_FILE = (
    HARD_NEGATIVE_DIR /
    "negative_type_distribution.png"
)
HARD_NEGATIVE_COMPLETED_FLAG = (
    HARD_NEGATIVE_DIR /
    "completed.flag"
)
HARD_NEGATIVE_CHECKPOINT_FILE = (
    CHECKPOINT_DIR /
    "hard_negative_checkpoint.json"
)
# ============================================================
# Mining Thresholds
# ============================================================
HARD_NEGATIVE_MIN = 0.75
HARD_NEGATIVE_MAX = 0.90
SIMILARITY_BY_TYPE_FILE = (
    HARD_NEGATIVE_DIR /
    "similarity_by_type.png"
)
RETRIEVAL_STATISTICS_FILE = (
    DATASET_OUTPUT_DIR /
    "retrieval_statistics.json"
)
FIGURE_METADATA_FILE = (
    METADATA_DIR / "figures_metadata.csv"
)
# ============================================================
# Augmentation
# ============================================================
ENABLE_HORIZONTAL_FLIP = True
HORIZONTAL_FLIP_PROB = 0.5
ENABLE_ROTATION = True
ROTATION_DEGREES = 10
ENABLE_COLOR_JITTER = True
COLOR_BRIGHTNESS = 0.2
COLOR_CONTRAST = 0.2
COLOR_SATURATION = 0.2
COLOR_HUE = 0.05
ENABLE_GAUSSIAN_BLUR = True
GAUSSIAN_KERNEL_SIZE = 3
GAUSSIAN_SIGMA = (0.1, 2.0)
ENABLE_RANDOM_ERASING = True
ERASING_PROBABILITY = 0.5
ERASING_SCALE = (0.02, 0.15)
ERASING_RATIO = (0.3, 3.3)
# ------------------------------------------------------------
# Partial Occlusion (Paper2Fig custom augmentation)
# ------------------------------------------------------------
ENABLE_PARTIAL_OCCLUSION = True
PARTIAL_OCCLUSION_PROBABILITY = 0.30
PARTIAL_OCCLUSION_MIN = 0.10
PARTIAL_OCCLUSION_MAX = 0.30
# ============================================================
# Model
# ============================================================
BACKBONE_NAME = "dinov2_vitb14"
PROJECTION_DIM = 512
DROPOUT = 0.2
FREEZE_BACKBONE = False
# ============================================================
# Training
# ============================================================

TRAINING_MODE = "hybrid"

# pair
# triplet
# hybrid
# ============================================================
# Loss
# ============================================================
LOSS_TYPE = "hybrid"
# contrastive
# triplet
# hybrid
CONTRASTIVE_MARGIN = 0.5
TRIPLET_MARGIN = 0.2
HYBRID_CONTRASTIVE_WEIGHT = 0.5
HYBRID_TRIPLET_WEIGHT = 0.5
# ============================================================
# Optimizer
# ============================================================
OPTIMIZER = "adamw"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BETAS = (
    0.9,
    0.999,
)
EPS = 1e-8
SGD_MOMENTUM = 0.9
# ============================================================
# Scheduler
# ============================================================
SCHEDULER = "warmup_cosine"
NUM_EPOCHS = 50
WARMUP_EPOCHS = 5
MIN_LEARNING_RATE = 1e-6
# ============================================================
# Experiments Directories
# ============================================================
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
# ============================================================
# Resume
# ============================================================

RESUME_TRAINING = False
