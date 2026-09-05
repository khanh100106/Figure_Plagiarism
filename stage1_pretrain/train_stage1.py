"""
Stage 1 - Image Plagiarism Pretraining
Train
--------------------------------------------------------------
Chay (SAU KHI da chay build_pairs.py de tao pairs CSV):
    python train_stage1.py

Dung triplet margin loss: khoang_cach(anchor, negative) phai lon
hon khoang_cach(anchor, positive) it nhat TRIPLET_MARGIN.
"""
from __future__ import annotations

import json
import time

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import config
from dataset import load_pairs_csv, split_train_val, TripletPlagiarismDataset
from model import PlagiarismEncoder
from experiment import create_experiment_dir, save_config_snapshot


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    transform = transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform, transform


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive_distance = F.pairwise_distance(anchor, positive, p=2)
    negative_distance = F.pairwise_distance(anchor, negative, p=2)
    loss = F.relu(positive_distance - negative_distance + margin)
    return loss.mean(), positive_distance.mean(), negative_distance.mean()


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler: GradScaler,
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0

    progress = tqdm(loader, desc="train", leave=False)
    for batch in progress:
        anchor = batch["anchor"].to(device, non_blocking=True)
        positive = batch["positive"].to(device, non_blocking=True)
        negative = batch["negative"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=use_amp):
            anchor_emb, positive_emb, negative_emb = model(
                anchor, positive, negative
            )
            loss, _, _ = triplet_loss(
                anchor_emb, positive_emb, negative_emb, config.TRIPLET_MARGIN
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = anchor.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(total, 1)


@torch.no_grad()
def validate(model, loader, device, use_amp: bool) -> dict:
    model.eval()
    running_loss = 0.0
    running_pos_dist = 0.0
    running_neg_dist = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="val", leave=False):
        anchor = batch["anchor"].to(device, non_blocking=True)
        positive = batch["positive"].to(device, non_blocking=True)
        negative = batch["negative"].to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            anchor_emb, positive_emb, negative_emb = model(
                anchor, positive, negative
            )
            loss, pos_dist, neg_dist = triplet_loss(
                anchor_emb, positive_emb, negative_emb, config.TRIPLET_MARGIN
            )

        batch_size = anchor.size(0)
        running_loss += loss.item() * batch_size
        running_pos_dist += pos_dist.item() * batch_size
        running_neg_dist += neg_dist.item() * batch_size
        total += batch_size

        # "Do chinh xac": ty le mau ma khoang_cach(pos) < khoang_cach(neg)
        # (tuc model xep dung positive gan hon negative).
        distances_pos = F.pairwise_distance(anchor_emb, positive_emb, p=2)
        distances_neg = F.pairwise_distance(anchor_emb, negative_emb, p=2)
        correct += (distances_pos < distances_neg).sum().item()

    return {
        "loss": running_loss / max(total, 1),
        "positive_distance": running_pos_dist / max(total, 1),
        "negative_distance": running_neg_dist / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


def main() -> None:
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(config.SEED)

    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )
    print(f"Su dung thiet bi: {device}")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    experiment_dir = create_experiment_dir(config.OUTPUT_DIR)
    save_config_snapshot(experiment_dir, config)
    print(f"Experiment: {experiment_dir}")

    dataframe = load_pairs_csv(config.PAIRS_FILE)
    train_df, val_df = split_train_val(dataframe, config.TRAIN_RATIO, config.SEED)
    print(f"So dong train: {len(train_df)} | So dong val: {len(val_df)}")
    print(
        f"  Ty le real/synthetic (train): "
        f"{(train_df['pair_type'] == 'real').mean():.2%} / "
        f"{(train_df['pair_type'] == 'synthetic').mean():.2%}"
    )

    train_transform, eval_transform = build_transforms()

    train_dataset = TripletPlagiarismDataset(train_df, train_transform)
    val_dataset = TripletPlagiarismDataset(val_df, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        drop_last=True,
        pin_memory=True,
        persistent_workers=(config.NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(config.NUM_WORKERS > 0),
    )

    model = PlagiarismEncoder(
        backbone_name=config.BACKBONE_NAME,
        projection_dim=config.PROJECTION_DIM,
        dropout=config.DROPOUT,
        freeze_backbone=config.FREEZE_BACKBONE,
    ).to(device)
    print(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS
    )

    use_amp = config.USE_AMP and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_positive_distance": [],
        "val_negative_distance": [],
    }

    best_accuracy = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, scaler, use_amp
        )
        val_metrics = validate(model, val_loader, device, use_amp)
        scheduler.step()

        elapsed = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_positive_distance"].append(val_metrics["positive_distance"])
        history["val_negative_distance"].append(val_metrics["negative_distance"])

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{config.NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"pos_dist={val_metrics['positive_distance']:.4f} | "
            f"neg_dist={val_metrics['negative_distance']:.4f} | "
            f"lr={current_lr:.2e} | "
            f"time={elapsed:.1f}s"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "backbone_name": config.BACKBONE_NAME,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, experiment_dir / "last.pt")

        if val_metrics["accuracy"] > best_accuracy + config.EARLY_STOPPING_MIN_DELTA:
            best_accuracy = val_metrics["accuracy"]
            epochs_without_improvement = 0
            torch.save(checkpoint, experiment_dir / "best.pt")
            print(f"  -> Luu model tot nhat (val_acc={best_accuracy:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            print(f"Dung som (early stopping) tai epoch {epoch}.")
            break

    with open(experiment_dir / "history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    print(f"Toan bo ket qua experiment nay nam trong: {experiment_dir}")
    print(
        f"-> O Giai doan 2 (plagiarism_dinov2/config.py), dat: "
        f"STAGE1_CHECKPOINT = Path(\"{experiment_dir / 'best.pt'}\")"
    )
    print(
        "   (khong bat buoc phai chay export_backbone.py - Giai doan 2 "
        "doc thang duoc checkpoint nay; export_backbone.py chi de tuy "
        "chon tach rieng 1 file backbone gon nhe hon neu muon.)"
    )


if __name__ == "__main__":
    main()
