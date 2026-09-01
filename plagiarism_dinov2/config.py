"""
Figure-Caption Plagiarism Pretraining
Config
--------------------------------------------------------------
"""
from pathlib import Path

# ----------------------------------------------------------------
# Duong dan
# ----------------------------------------------------------------
# File CSV/TSV chua cac cot:
# figure_id, caption_id, paper_id, field, category, published, title,
# page, figure_image_path, caption_image_path, caption_text,
# figure_x1, figure_y1, figure_x2, figure_y2,
# caption_x1, caption_y1, caption_x2, caption_y2
DATA_CSV = Path("D:/Research/PhD/Plagiarism/train/paper2fig2026/metadata/paper2fig2026_metadata.csv")

IMAGE_ROOT = Path("D:/Research/PhD/Plagiarism/train/paper2fig2026")

# Noi luu checkpoint, log, bieu do
OUTPUT_DIR = Path("outputs")

# ----------------------------------------------------------------
# Caption Modality
# ----------------------------------------------------------------
# "text":  dung caption_text + text encoder (BERT/SciBERT...)
#          -> dual-encoder kieu CLIP (image-text). MAC DINH.
# "image": dung caption_image_path + DINOv2 dung chung (Siamese)
#          -> cach image-image ban dau.
CAPTION_MODALITY = "text"

# ----------------------------------------------------------------
# Text Encoder (chi dung khi CAPTION_MODALITY == "text")
# ----------------------------------------------------------------
# SciBERT phu hop vi caption trong day la tu paper khoa hoc.
# Co the doi sang "bert-base-uncased", "roberta-base",
# "allenai/specter2", ... (ten model tren HuggingFace Hub).
TEXT_MODEL_NAME = "allenai/scibert_scivocab_uncased"

# So token toi da cho 1 caption (cat bot neu dai hon).
MAX_TEXT_LENGTH = 128

# True: chi train text projection head, dong bang text encoder.
FREEZE_TEXT_ENCODER = False

# ----------------------------------------------------------------
# Backbone (DINOv2)
# ----------------------------------------------------------------
# Cac lua chon: dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14 | dinov2_vitg14
BACKBONE_NAME = "dinov2_vitb14"

# True: chi train projection head, dong bang DINOv2 (nhanh, it VRAM,
#       phu hop khi dataset nho).
# False: fine-tune ca DINOv2 (can nhieu du lieu + VRAM hon).
FREEZE_BACKBONE = False

# DINOv2 dung patch 14x14 nen IMAGE_SIZE phai la boi so cua 14.
IMAGE_SIZE = 518

# ----------------------------------------------------------------
# Projection Head
# ----------------------------------------------------------------
PROJECTION_DIM = 256
DROPOUT = 0.1

# ----------------------------------------------------------------
# Training
# ----------------------------------------------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Ty le du lieu dung de train (phan con lai dung de validate).
# Split theo paper_id de tranh leakage (cung 1 paper khong bi
# chia vao ca train lan val).
TRAIN_RATIO = 0.85

NUM_WORKERS = 4
SEED = 42

# Se tu dong fallback ve "cpu" neu may khong co GPU.
DEVICE = "cuda"

# Dung training som neu recall@1 tren val khong cai thien sau
# tung nay epoch lien tiep.
EARLY_STOPPING_PATIENCE = 7
