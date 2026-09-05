"""
Figure-Caption Plagiarism Pretraining
Config
--------------------------------------------------------------
Sua cac gia tri o day cho phu hop voi may/dataset cua ban.
Khong can sua code o cac file khac.
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
MAX_TEXT_LENGTH = 64

# True: chi train text projection head, dong bang text encoder.
# -> Dat True truoc (mac dinh): batch nho (16-64) khong du negative
#    de fine-tune ca SciBERT (110M) ma khong bi overfit - SciBERT se
#    "hoc thuoc" tung caption thay vi hoc ngu nghia that.
FREEZE_TEXT_ENCODER = True

# ----------------------------------------------------------------
# Backbone (DINOv2)
# ----------------------------------------------------------------
# Cac lua chon: dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14 | dinov2_vitg14
BACKBONE_NAME = "dinov2_vitb14"

# True: chi train projection head, dong bang DINOv2 (nhanh, it VRAM,
#       phu hop khi dataset nho HOAC GPU it VRAM nhu 8GB).
# False: fine-tune ca DINOv2 (can nhieu du lieu + VRAM hon rat nhieu -
#        voi GPU 8GB va IMAGE_SIZE lon se bi OOM).
# -> Dat True truoc de chay on dinh, sau khi on roi neu du VRAM/du
#    lieu, co the thu dat False de fine-tune sau cho chat luong tot hon.
FREEZE_BACKBONE = False

# DINOv2 dung patch 14x14 nen IMAGE_SIZE phai la boi so cua 14.
# 518 la kich thuoc "chuan" cua DINOv2 nhung rat nang (attention
# tang binh phuong theo so patch) -> voi GPU <=8GB de 224 hoac 252.
# Neu GPU >=16GB va da FREEZE_BACKBONE=True, co the tang len 336/518.
IMAGE_SIZE = 224

# ----------------------------------------------------------------
# Projection Head
# ----------------------------------------------------------------
PROJECTION_DIM = 256

# Tang dropout de chong overfit (train_loss ve gan 0 trong khi
# val_loss tang la dau hieu can dropout/regularization manh hon).
DROPOUT = 0.4

# ----------------------------------------------------------------
# Training
# ----------------------------------------------------------------
# Voi FREEZE_BACKBONE=True va FREEZE_TEXT_ENCODER=True, chi con 2
# projection head nho la trainable -> VRAM con lai rat nhieu, co
# the tang BATCH_SIZE de co nhieu negative hon (tot cho contrastive
# loss). Neu van du VRAM, thu tang tiep len 128.
BATCH_SIZE = 64

# Khong can cong don gradient nhieu nua vi BATCH_SIZE da tang.
GRAD_ACCUM_STEPS = 1

# Mixed precision (fp16 tren GPU) - giam ~40-50% VRAM va tang toc,
# gan nhu khong doi chat luong. Tu dong tat neu chay CPU.
USE_AMP = True

NUM_EPOCHS = 60
LEARNING_RATE = 1e-5

# Tang weight decay de chong overfit khi model hoc rat nhanh
# (train_loss ve gan 0 chi sau vai chuc epoch).
WEIGHT_DECAY = 0.1

# LR danh cho backbone/text-encoder KHI ban dat FREEZE_BACKBONE
# hoac FREEZE_TEXT_ENCODER = False (fine-tune). LR thuc te se la
# LEARNING_RATE * BACKBONE_LR_MULTIPLIER cho phan encoder da
# pretrain (can LR nho de khong pha vo feature da hoc), con
# projection head (khoi tao ngau nhien) van dung LEARNING_RATE
# day du. Khong co tac dung gi neu ca 2 dang bi dong bang.
BACKBONE_LR_MULTIPLIER = 0.02

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
EARLY_STOPPING_PATIENCE = 6

# Recall@1 phai tang toi thieu tung nay thi moi tinh la "cai
# thien that" (reset patience). Neu khong co nguong nay, cac dao
# dong nhieu (vd 0.0221 -> 0.0229 -> 0.0206) se lien tuc lam
# early stopping tuong nham la van dang cai thien, khong bao gio
# dung som duoc.
EARLY_STOPPING_MIN_DELTA = 0.0005
