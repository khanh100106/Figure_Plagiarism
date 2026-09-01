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

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import config
from dataset import load_metadata, split_by_paper, FigureCaptionDataset
from model import TwinDinoV2Encoder, ImageTextDualEncoder
from collate import TextCollator
from losses import contrastive_loss
from utils import seed_everything, recall_at_k, plot_training_curves


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


def train_one_epoch(model, loader, optimizer, device, forward_fn) -> float:
    model.train()
    running_loss = 0.0
    total = 0

    progress = tqdm(loader, desc="train", leave=False)
    for batch in progress:
        optimizer.zero_grad()
        embeddings_a, embeddings_b = forward_fn(model, batch, device)
        loss = contrastive_loss(
            embeddings_a,
            embeddings_b,
            model.logit_scale,
        )
        loss.backward()
        optimizer.step()

        batch_size = embeddings_a.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(total, 1)


@torch.no_grad()
def validate(model, loader, device, forward_fn) -> dict:
    model.eval()
    running_loss = 0.0
    total = 0

    all_embeddings_a = []
    all_embeddings_b = []

    for batch in tqdm(loader, desc="val", leave=False):
        embeddings_a, embeddings_b = forward_fn(model, batch, device)
        loss = contrastive_loss(
            embeddings_a,
            embeddings_b,
            model.logit_scale,
        )

        batch_size = embeddings_a.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

        all_embeddings_a.append(embeddings_a.cpu())
        all_embeddings_b.append(embeddings_b.cpu())

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
    seed_everything(config.SEED)

    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )
    print(f"Su dung thiet bi: {device}")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.NUM_EPOCHS,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_recall1": [],
        "val_recall5": [],
    }

    best_recall1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, forward_fn,
        )
        val_metrics = validate(model, val_loader, device, forward_fn)
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
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, config.OUTPUT_DIR / "last.pt")

        if val_metrics["recall@1"] > best_recall1:
            best_recall1 = val_metrics["recall@1"]
            epochs_without_improvement = 0
            torch.save(checkpoint, config.OUTPUT_DIR / "best.pt")
            print(f"  -> Luu model tot nhat (recall@1={best_recall1:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            print(f"Dung som (early stopping) tai epoch {epoch}.")
            break

    with open(config.OUTPUT_DIR / "history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    plot_path = config.OUTPUT_DIR / "training_curves.png"
    plot_training_curves(history, plot_path)
    print(f"Da luu bieu do train/val vao: {plot_path}")


if __name__ == "__main__":
    main()
