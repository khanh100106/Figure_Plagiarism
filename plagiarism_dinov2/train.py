"""
Figure-Caption Plagiarism Pretraining
Train
--------------------------------------------------------------
Chay: python train.py

Quy trinh:
  - Doc CSV metadata, chia train/val theo paper_id.
  - Moi epoch: train (contrastive loss) -> validate (loss +
    recall@1/@5) -> in ra man hinh -> luu checkpoint.
  - Sau khi train xong: luu bieu do loss/recall theo epoch.
"""
from __future__ import annotations

import json
import time
import math

import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import config
from dataset import load_metadata, split_by_paper, FigureCaptionDataset
from model import TwinDinoV2Encoder, ImageTextDualEncoder
from collate import TextCollator
from losses import contrastive_loss
from utils import seed_everything, recall_at_k, plot_training_curves
from experiment import create_experiment_dir, save_config_snapshot


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform


def build_param_groups(
    model,
    base_lr: float,
    backbone_lr_multiplier: float,
) -> list[dict]:
    """
    Nhom tham so thanh 2 nhom LR khac nhau:
      - backbone/text-encoder (neu dang duoc fine-tune, tuc
        khong bi dong bang): LR = base_lr * backbone_lr_multiplier.
        Cac encoder nay da duoc pretrain rat tot, fine-tune voi
        LR lon nhu head se de lam hong feature/overfit nhanh.
      - phan con lai (projection head, logit_scale): base_lr day
        du, vi la tham so khoi tao ngau nhien, can hoc nhanh hon.

    Neu backbone/text-encoder dang bi dong bang (requires_grad=False)
    thi nhom nay se rong, khong anh huong gi.
    """
    backbone_modules = []
    for attribute_name in ("backbone", "image_backbone", "text_backbone"):
        module = getattr(model, attribute_name, None)
        if module is not None:
            backbone_modules.append(module)

    backbone_param_ids = set()
    backbone_params = []
    for module in backbone_modules:
        for parameter in module.parameters():
            if parameter.requires_grad:
                backbone_params.append(parameter)
                backbone_param_ids.add(id(parameter))

    other_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_param_ids
    ]

    param_groups = []
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": base_lr * backbone_lr_multiplier,
            }
        )
    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": base_lr,
            }
        )

    return param_groups


def build_forward_fn(modality: str):
    """
    Tra ve ham forward_fn(model, batch, device) -> (emb_a, emb_b).

    Tach rieng phan "lay du lieu tu batch + goi model" khoi vong
    lap train/validate, de mot bo train_one_epoch/validate dung
    chung duoc cho ca 2 modality ("text" va "image").
    """
    if modality == "text":

        def forward_fn(model, batch, device):
            figure = batch["figure"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(
                device,
                non_blocking=True,
            )
            return model(figure, input_ids, attention_mask)

        return forward_fn

    if modality == "image":

        def forward_fn(model, batch, device):
            figure = batch["figure"].to(device, non_blocking=True)
            caption = batch["caption"].to(device, non_blocking=True)
            return model(figure, caption)

        return forward_fn

    raise ValueError(
        f"CAPTION_MODALITY phai la 'text' hoac 'image', nhan duoc: "
        f"{modality}"
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    forward_fn,
    scaler: GradScaler,
    use_amp: bool,
    grad_accum_steps: int = 1,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0

    optimizer.zero_grad()
    num_batches = len(loader)

    progress = tqdm(loader, desc="train", leave=False)
    for step, batch in enumerate(progress):
        with autocast(device_type=device.type, enabled=use_amp):
            embeddings_a, embeddings_b = forward_fn(model, batch, device)
            loss = contrastive_loss(
                embeddings_a,
                embeddings_b,
                model.logit_scale,
            )
            loss_to_backward = loss / grad_accum_steps

        scaler.scale(loss_to_backward).backward()

        is_last_batch = (step + 1) == num_batches
        if (step + 1) % grad_accum_steps == 0 or is_last_batch:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100))

        batch_size = embeddings_a.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(total, 1)


@torch.no_grad()
def validate(model, loader, device, forward_fn, use_amp: bool) -> dict:
    model.eval()
    running_loss = 0.0
    total = 0

    all_embeddings_a = []
    all_embeddings_b = []

    for batch in tqdm(loader, desc="val", leave=False):
        with autocast(device_type=device.type, enabled=use_amp):
            embeddings_a, embeddings_b = forward_fn(model, batch, device)
            loss = contrastive_loss(
                embeddings_a,
                embeddings_b,
                model.logit_scale,
            )

        batch_size = embeddings_a.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

        all_embeddings_a.append(embeddings_a.float().cpu())
        all_embeddings_b.append(embeddings_b.float().cpu())

    all_embeddings_a = torch.cat(all_embeddings_a, dim=0)
    all_embeddings_b = torch.cat(all_embeddings_b, dim=0)

    metrics = recall_at_k(
        all_embeddings_a,
        all_embeddings_b,
        k_values=(1, 5),
    )
    metrics["loss"] = running_loss / max(total, 1)
    return metrics


def main() -> None:
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    seed_everything(config.SEED)

    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )
    print(f"Su dung thiet bi: {device}")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    experiment_dir = create_experiment_dir(config.OUTPUT_DIR)
    save_config_snapshot(experiment_dir, config)
    print(f"Experiment: {experiment_dir}")

    dataframe = load_metadata(config.DATA_CSV, modality=config.CAPTION_MODALITY)
    train_df, val_df = split_by_paper(
        dataframe,
        config.TRAIN_RATIO,
        config.SEED,
    )
    print(f"So dong train: {len(train_df)} | So dong val: {len(val_df)}")

    train_transform, eval_transform = build_transforms()

    train_dataset = FigureCaptionDataset(
        train_df,
        config.IMAGE_ROOT,
        train_transform,
        modality=config.CAPTION_MODALITY,
    )
    val_dataset = FigureCaptionDataset(
        val_df,
        config.IMAGE_ROOT,
        eval_transform,
        modality=config.CAPTION_MODALITY,
    )

    if config.CAPTION_MODALITY == "text":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        collate_fn = TextCollator(tokenizer, config.MAX_TEXT_LENGTH)

        model = ImageTextDualEncoder(
            backbone_name=config.BACKBONE_NAME,
            text_model_name=config.TEXT_MODEL_NAME,
            projection_dim=config.PROJECTION_DIM,
            dropout=config.DROPOUT,
            freeze_backbone=config.FREEZE_BACKBONE,
            freeze_text_encoder=config.FREEZE_TEXT_ENCODER,
        ).to(device)
    else:
        collate_fn = None

        model = TwinDinoV2Encoder(
            backbone_name=config.BACKBONE_NAME,
            projection_dim=config.PROJECTION_DIM,
            dropout=config.DROPOUT,
            freeze_backbone=config.FREEZE_BACKBONE,
        ).to(device)

    print(model)
    forward_fn = build_forward_fn(config.CAPTION_MODALITY)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(config.NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(config.NUM_WORKERS > 0),
    )

    param_groups = build_param_groups(
        model,
        base_lr=config.LEARNING_RATE,
        backbone_lr_multiplier=config.BACKBONE_LR_MULTIPLIER,
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.NUM_EPOCHS,
    )

    # Mixed precision: giam dang ke VRAM + tang toc tren GPU,
    # tu dong tat neu chay tren CPU (khong ho tro/khong can thiet).
    use_amp = config.USE_AMP and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    print(
        f"Mixed precision (AMP): {'BAT' if use_amp else 'TAT'} | "
        f"Grad accumulation steps: {config.GRAD_ACCUM_STEPS}"
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_recall1": [],
        "val_recall5": [],
    }

    best_recall1 = -1.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            forward_fn,
            scaler,
            use_amp,
            grad_accum_steps=config.GRAD_ACCUM_STEPS,
        )
        val_metrics = validate(model, val_loader, device, forward_fn, use_amp)
        scheduler.step()

        elapsed = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_recall1"].append(val_metrics["recall@1"])
        history["val_recall5"].append(val_metrics["recall@5"])

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{config.NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"recall@1={val_metrics['recall@1']:.4f} | "
            f"recall@5={val_metrics['recall@5']:.4f} | "
            f"lr={current_lr:.2e} | "
            f"time={elapsed:.1f}s"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, experiment_dir / "last.pt")

        if val_metrics["recall@1"] > best_recall1 + config.EARLY_STOPPING_MIN_DELTA:
            best_recall1 = val_metrics["recall@1"]
            epochs_without_improvement = 0
            torch.save(checkpoint, experiment_dir / "best.pt")
            print(f"  -> Luu model tot nhat theo recall@1 (recall@1={best_recall1:.4f})")
        else:
            epochs_without_improvement += 1

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(checkpoint, experiment_dir / "best_val_loss.pt")
            print(f"  -> Luu model tot nhat theo val_loss (val_loss={best_val_loss:.4f})")

        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            print(f"Dung som (early stopping) tai epoch {epoch}.")
            break

    with open(experiment_dir / "history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    plot_path = experiment_dir / "training_curves.png"
    plot_training_curves(history, plot_path)
    print(f"Da luu bieu do train/val vao: {plot_path}")
    print(f"Toan bo ket qua experiment nay nam trong: {experiment_dir}")


if __name__ == "__main__":
    main()
