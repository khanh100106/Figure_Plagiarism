"""
Global configuration for the Paper2Fig-2026 Retrieval Framework.

Author: Nguyen Khanh
Project: Paper2Fig-2026
"""

from pathlib import Path
import torch

# ============================================================
# Project Directories
# ============================================================

# train/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset
DATASET_ROOT = PROJECT_ROOT / "paper2fig2026"

FIGURE_DIR = DATASET_ROOT / "figures"
CAPTION_DIR = DATASET_ROOT / "captions"

METADATA_FILE = (
    DATASET_ROOT
    / "metadata"
    / "paper2fig2026_metadata.csv"
)

ANALYSIS_DIR = DATASET_ROOT / "analysis"

# ============================================================
# Output Directories
# ============================================================

EMBEDDING_DIR = ANALYSIS_DIR / "embeddings"
FAISS_DIR = ANALYSIS_DIR / "faiss"
MINING_DIR = ANALYSIS_DIR / "mining"
DATASET_OUTPUT_DIR = ANALYSIS_DIR / "datasets"

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

HARD_NEGATIVE_MIN = 0.75

HARD_NEGATIVE_MAX = 0.90

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

DEBUG = True

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
# Hard Negative Mining
# ============================================================

HARD_NEGATIVE_PAIR_FILE = (
    MINING_DIR / "hard_negative_pairs.csv"
)

HARD_NEGATIVE_INFO_FILE = (
    MINING_DIR / "hard_negative_info.json"
)

HARD_NEGATIVE_COMPLETED_FLAG = (
    MINING_DIR / "hard_negative_completed.flag"
)

HARD_NEGATIVE_CHECKPOINT_FILE = (
    CHECKPOINT_DIR /
    "hard_negative_checkpoint.json"
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

SEED = 42