"""
Stage 1 -> Stage 2 Bridge
--------------------------------------------------------------
Trich rieng trong so cua image backbone tu 1 checkpoint Stage 1
(vd stage1_outputs/exp_0001/best.pt), luu thanh 1 file nho gon de
nap vao image_backbone cua ImageTextDualEncoder o plagiarism_dinov2/
(Giai doan 2) - xem PRETRAINED_IMAGE_BACKBONE trong
plagiarism_dinov2/config.py.

Chay:
    python export_backbone.py --checkpoint stage1_outputs/exp_0001/best.pt --output stage1_backbone.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


BACKBONE_PREFIX = "backbone."


def export_backbone(checkpoint_path: Path, output_path: Path) -> None:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay checkpoint: {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint["model_state_dict"]

    backbone_state = {
        key[len(BACKBONE_PREFIX):]: value
        for key, value in model_state.items()
        if key.startswith(BACKBONE_PREFIX)
    }

    if not backbone_state:
        raise ValueError(
            "Khong tim thay tham so nao co tien to 'backbone.' trong "
            "checkpoint - kiem tra lai file checkpoint co dung dinh "
            "dang tu train_stage1.py khong."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "backbone_state_dict": backbone_state,
            "backbone_name": checkpoint.get("backbone_name", "unknown"),
            "source_checkpoint": str(checkpoint_path),
            "source_epoch": checkpoint.get("epoch"),
            "source_val_metrics": checkpoint.get("val_metrics"),
        },
        output_path,
    )

    print(f"Da xuat {len(backbone_state)} tensor backbone vao {output_path}")
    print(
        f"-> Dat PRETRAINED_IMAGE_BACKBONE = "
        f"Path(\"{output_path}\") trong plagiarism_dinov2/config.py"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Xuat backbone tu checkpoint Stage 1 sang Stage 2."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Duong dan checkpoint Stage 1 (vd stage1_outputs/exp_0001/best.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stage1_backbone.pt",
        help="Duong dan file backbone se xuat ra",
    )
    args = parser.parse_args()

    export_backbone(Path(args.checkpoint), Path(args.output))


if __name__ == "__main__":
    main()
