"""
Stage 1 - Image Plagiarism Pretraining
Config
--------------------------------------------------------------
Muc tieu: pretrain DINOv2 nhan dien dao hinh (image-image) tren
2 nguon du lieu gop lai:
  1. Corpus dao hinh that (flowchart, co nhan that) - hoc cac
     kieu dao hinh that su nhung domain hep (chi flowchart).
  2. Cap synthetic sinh tu paper2fig2026 (ca 7 category) - bu
     domain con thieu (figure khoa hoc mau, da dang loai).

Checkpoint sau giai doan nay se duoc nap vao image_backbone cua
ImageTextDualEncoder o thu muc plagiarism_dinov2/ (Giai doan 2).
"""
from pathlib import Path

# ----------------------------------------------------------------
# Nguon 1: Corpus dao hinh that (flowchart)
# ----------------------------------------------------------------
# Sua duong dan nay tro toi thu muc goc "Figure Plagiarism corpus".
FLOWCHART_ROOT = Path("D:/Research/PhD/Plagiarism/train/figure_plagiarism_corpus")

# Ten cac thu muc con - SUA LAI neu ten thuc te cua ban khac (vd
# viet hoa/thuong, dau cach khac).
FLOWCHART_GROUPS = [
    {
        "name": "shape_based",
        "root": "Shape_Based_Figure Plagiarism",
        "annotations": "Annotations",
        "plagiarised": "Plagiarised figures",
        "source": "Source figures",
        "has_xml": True,
    },
    {
        "name": "hybrid",
        "root": "Textual features based Plus shape based(Hybrid)",
        "annotations": None,
        "plagiarised": "Plagiarised figures",
        "source": "Source figures",
        "has_xml": False,
    },
    {
        "name": "textual_reference",
        "root": "Textual reference based figure plagiarism",
        # XML nhom nay (feature name="Real plagiarism") KHONG co
        # source_reference - khong the lien ket chinh xac tung cap
        # tu XML. Ghep theo so thu tu trong ten file thay the
        # (suspicious 00.pdf <-> source 00.pdf), giong nhom Hybrid.
        "annotations": None,
        "plagiarised": "plagiarised figures",
        "source": "Source figures",
        "has_xml": False,
    },
]

# ----------------------------------------------------------------
# Nguon 2: paper2fig2026 (de sinh cap synthetic)
# ----------------------------------------------------------------
PAPER2FIG_ROOT = Path("D:/Research/PhD/Plagiarism/train/paper2fig2026")
PAPER2FIG_METADATA = PAPER2FIG_ROOT / "metadata" / "paper2fig2026_metadata.csv"
CATEGORIES = ["AI", "BIOINFO", "CV", "IMAGING", "ML", "ROBOTICS", "SIGNAL"]

# So ban sao synthetic sinh ra tren moi figure goc (moi ban dung
# ngau nhien 1 trong 3 muc do bien doi, xem synthetic_pairs.py).
SYNTHETIC_COPIES_PER_FIGURE = 2

# ----------------------------------------------------------------
# Output
# ----------------------------------------------------------------
OUTPUT_DIR = Path("stage1_outputs")
PAIRS_FILE = OUTPUT_DIR / "pairs_with_hard_negatives.csv"

# ----------------------------------------------------------------
# Hard Negative Mining
# ----------------------------------------------------------------
# Dung backbone NHO va DONG BANG chi de tinh similarity (khong
# train o buoc nay) - nhanh hon nhieu so voi dung backbone lon.
MINING_BACKBONE_NAME = "dinov2_vits14"
MINING_IMAGE_SIZE = 224
MINING_BATCH_SIZE = 64

# Voi moi anchor, lay top-K anh giong nhat trong pool (SAU KHI da
# loai bo dung positive that), roi chon ngau nhien 1 trong so do
# lam hard negative cho moi lan train (da dang hoa, tranh hoc
# thuoc dung 1 negative co dinh).
HARD_NEGATIVE_TOP_K = 10

# ----------------------------------------------------------------
# Training (Stage 1)
# ----------------------------------------------------------------
BACKBONE_NAME = "dinov2_vitb14"
IMAGE_SIZE = 224
PROJECTION_DIM = 256
DROPOUT = 0.3

# Muc dich Stage 1 la de DINOv2 THAT SU hoc dac trung dao hinh,
# nen fine-tune toan bo (khac voi Stage 2 dang dong bang mac dinh).
FREEZE_BACKBONE = False

BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 1
USE_AMP = True

NUM_EPOCHS = 20
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-2

# Trong triplet loss: khoang cach(anchor,negative) phai lon hon
# khoang cach(anchor,positive) it nhat TRIPLET_MARGIN.
TRIPLET_MARGIN = 0.2

# Ty le du lieu train/val/test, chia theo anchor_path.
# Tap test khong dung trong qua trinh tuning, chi de danh gia cuoi.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # = 1 - TRAIN_RATIO - VAL_RATIO
SEED = 42
DEVICE = "cuda"
NUM_WORKERS = 4

EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.0005

# True: neu experiment gan nhat trong OUTPUT_DIR co checkpoint
# "last.pt", tiep tuc train tu do (dung khi bi ngat ngang - mat
# dien, crash, dong may...). False: luon tao experiment moi.
RESUME_TRAINING = False
