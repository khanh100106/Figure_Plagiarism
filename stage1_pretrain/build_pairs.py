"""
Stage 1 - Image Plagiarism Pretraining
Build Pairs (Orchestrator)
--------------------------------------------------------------
Chay: python build_pairs.py

1. Doc cap dao hinh THAT tu corpus flowchart (flowchart_corpus.py).
2. Lay danh sach figure THAT tu paper2fig2026 (ca 7 category) de
   lam nguon sinh cap SYNTHETIC.
3. Gop toan bo anh (Source that + figure paper2fig2026) thanh 1
   pool, trich embedding 1 lan bang backbone nho dong bang.
4. Voi moi anchor (source that / figure goc dung sinh synthetic),
   mining hard negative that su tu pool (loai tru chinh no).
5. Ghi ra CSV: anchor_path, positive_path, negative_path,
   pair_type, tier_or_plagtype.
   - pair_type="real": positive_path la anh suspicious that.
   - pair_type="synthetic": positive_path de trong - se duoc sinh
     on-the-fly luc train (xem dataset.py) tu chinh anchor_path
     theo tier_or_plagtype, de khong phai luu them anh ra dia.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import torch

import config
from flowchart_corpus import load_flowchart_pairs
from synthetic_pairs import list_figure_paths, TIERS
from mining import build_embedding_index, rank_hard_negatives


CSV_FIELDS = [
    "anchor_path",
    "positive_path",
    "negative_path",
    "pair_type",
    "tier_or_plagtype",
]


def _build_real_rows(
    real_pairs,
    pool_paths: list[Path],
    pool_path_to_index: dict,
    pool_embeddings,
    top_k: int,
) -> list[dict]:
    """
    Anchor = source that. Loai tru chinh no khoi ung vien negative
    (pool chi gom Source that + figure paper2fig2026, KHONG gom
    anh suspicious, nen dung positive khong bao gio bi chon nham).
    """
    anchor_indices = [
        pool_path_to_index[pair.source_path] for pair in real_pairs
    ]
    anchor_embeddings = pool_embeddings[anchor_indices]

    hard_negative_candidates = rank_hard_negatives(
        anchor_embeddings=anchor_embeddings,
        pool_embeddings=pool_embeddings,
        exclude_indices=anchor_indices,
        top_k=top_k,
    )

    rows = []
    for pair, candidates in zip(real_pairs, hard_negative_candidates):
        if not candidates:
            print(
                f"[Canh bao] Khong tim duoc hard negative cho "
                f"{pair.source_path.name} - bo qua dong nay."
            )
            continue

        negative_index = random.choice(candidates)
        rows.append(
            {
                "anchor_path": str(pair.source_path),
                "positive_path": str(pair.suspicious_path),
                "negative_path": str(pool_paths[negative_index]),
                "pair_type": "real",
                "tier_or_plagtype": pair.plag_type,
            }
        )

    return rows


def _build_synthetic_rows(
    figure_paths: list[Path],
    copies_per_figure: int,
    pool_paths: list[Path],
    pool_path_to_index: dict,
    pool_embeddings,
    top_k: int,
) -> list[dict]:
    """
    Anchor = chinh figure goc. Positive se duoc sinh on-the-fly
    luc train (dataset.py se doc tier_or_plagtype va tu bien doi
    anchor_path), nen o day chi can mining hard negative cho
    chinh figure goc.
    """
    records = []
    for figure_path in figure_paths:
        for _ in range(copies_per_figure):
            tier = random.choice(TIERS)
            records.append((figure_path, tier))

    anchor_indices = [
        pool_path_to_index[figure_path] for figure_path, _ in records
    ]
    anchor_embeddings = pool_embeddings[anchor_indices]

    hard_negative_candidates = rank_hard_negatives(
        anchor_embeddings=anchor_embeddings,
        pool_embeddings=pool_embeddings,
        exclude_indices=anchor_indices,
        top_k=top_k,
    )

    rows = []
    for (figure_path, tier), candidates in zip(records, hard_negative_candidates):
        if not candidates:
            continue

        negative_index = random.choice(candidates)
        rows.append(
            {
                "anchor_path": str(figure_path),
                "positive_path": "",
                "negative_path": str(pool_paths[negative_index]),
                "pair_type": "synthetic",
                "tier_or_plagtype": tier,
            }
        )

    return rows


def main() -> None:
    random.seed(config.SEED)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )
    print(f"Su dung thiet bi: {device}")

    # --------------------------------------------------------
    # 1. Cap that tu corpus flowchart
    # --------------------------------------------------------
    real_pairs = load_flowchart_pairs(
        config.FLOWCHART_ROOT,
        config.FLOWCHART_GROUPS,
    )
    print(f"Tong so cap that (flowchart): {len(real_pairs)}")

    real_source_pool = sorted(
        {pair.source_path for pair in real_pairs},
        key=str,
    )

    # --------------------------------------------------------
    # 2. Nguon figure that tu paper2fig2026 (de sinh synthetic)
    # --------------------------------------------------------
    p2f_figures = list_figure_paths(
        config.PAPER2FIG_METADATA,
        config.PAPER2FIG_ROOT,
        config.CATEGORIES,
    )
    print(
        f"So figure paper2fig2026 dung lam nguon synthetic: "
        f"{len(p2f_figures)}"
    )

    # --------------------------------------------------------
    # 3. Gop pool + trich embedding 1 lan
    # --------------------------------------------------------
    full_pool = real_source_pool + p2f_figures
    pool_path_to_index = {path: i for i, path in enumerate(full_pool)}

    print(
        f"Trich embedding cho {len(full_pool)} anh trong pool "
        f"(co the mat vai phut)..."
    )
    pool_embeddings = build_embedding_index(
        image_paths=full_pool,
        backbone_name=config.MINING_BACKBONE_NAME,
        image_size=config.MINING_IMAGE_SIZE,
        batch_size=config.MINING_BATCH_SIZE,
        device=device,
    )

    # --------------------------------------------------------
    # 4. Mining hard negative + ghep dong CSV
    # --------------------------------------------------------
    real_rows = _build_real_rows(
        real_pairs,
        full_pool,
        pool_path_to_index,
        pool_embeddings,
        config.HARD_NEGATIVE_TOP_K,
    )

    synthetic_rows = _build_synthetic_rows(
        p2f_figures,
        config.SYNTHETIC_COPIES_PER_FIGURE,
        full_pool,
        pool_path_to_index,
        pool_embeddings,
        config.HARD_NEGATIVE_TOP_K,
    )

    all_rows = real_rows + synthetic_rows
    random.shuffle(all_rows)

    with open(config.PAIRS_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDa ghi {len(all_rows)} dong vao {config.PAIRS_FILE}")
    print(f"  - Cap that (flowchart):        {len(real_rows)}")
    print(f"  - Cap synthetic (paper2fig2026): {len(synthetic_rows)}")


if __name__ == "__main__":
    main()
