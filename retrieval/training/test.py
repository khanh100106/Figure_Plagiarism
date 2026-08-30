from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap


PYTHON = sys.executable


def run_phase(name: str, code: str):

    print("=" * 70, flush=True)
    print(f"PHASE {name}", flush=True)
    print("=" * 70, flush=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    ) as f:

        f.write(textwrap.dedent(code))
        script = f.name

    try:

        result = subprocess.run(
            [PYTHON, script],
            capture_output=True,
            text=True,
        )

        print(
            result.stdout,
            end="",
            flush=True,
        )

        if result.stderr:

            print(
                result.stderr,
                end="",
                flush=True,
            )

        print(
            f"\nReturn code: {result.returncode}",
            flush=True,
        )

        if result.returncode == 0:

            print(
                f"[PASS] PHASE {name}",
                flush=True,
            )

            return True

        print(
            f"[FAIL] PHASE {name}",
            flush=True,
        )

        return False

    finally:

        try:
            os.remove(script)
        except OSError:
            pass


def main():

    print(
        "STEP 4C-6 ISOLATED NATIVE CRASH DIAGNOSTIC",
        flush=True,
    )

    # ======================================================
    # A — imports
    # ======================================================

    if not run_phase(
        "A — IMPORTS",
        """
        print("START", flush=True)

        import torch
        print("[PASS] torch", flush=True)

        from retrieval.configs import DEVICE
        print("[PASS] configs", flush=True)

        from retrieval.training.training_dataset import PairTrainingDataset
        print("[PASS] training_dataset", flush=True)

        from retrieval.modeling.retrieval_model import RetrievalModel
        print("[PASS] retrieval_model", flush=True)

        from retrieval.training.loss import build_loss
        print("[PASS] loss", flush=True)

        print("DONE", flush=True)
        """,
    ):
        return

    # ======================================================
    # B — dataset + batch
    # ======================================================

    if not run_phase(
        "B — REAL BATCH",
        """
        import torch
        from torch.utils.data import DataLoader
        from retrieval.training.training_dataset import PairTrainingDataset

        print("Creating dataset", flush=True)

        dataset = PairTrainingDataset(train=True)

        print(
            f"[PASS] dataset = {len(dataset)}",
            flush=True,
        )

        print("Creating DataLoader", flush=True)

        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        print("[PASS] DataLoader", flush=True)

        print("Loading batch", flush=True)

        batch = next(iter(loader))

        print("[PASS] batch", flush=True)

        print(
            batch["anchor_image"].shape,
            flush=True,
        )

        print(
            batch["target_image"].shape,
            flush=True,
        )

        print("DONE", flush=True)
        """,
    ):
        return

    # ======================================================
    # C — model CUDA
    # ======================================================

    if not run_phase(
        "C — MODEL CUDA",
        """
        import torch
        from retrieval.configs import DEVICE
        from retrieval.modeling.retrieval_model import RetrievalModel

        print("Creating model", flush=True)

        model = RetrievalModel()

        print("[PASS] model", flush=True)

        print("Moving model to CUDA", flush=True)

        model = model.to(DEVICE)

        torch.cuda.synchronize()

        print("[PASS] CUDA", flush=True)

        print("DONE", flush=True)
        """,
    ):
        return

    # ======================================================
    # D — REAL BATCH + FORWARD
    # ======================================================

    if not run_phase(
        "D — REAL FORWARD",
        """
        import torch
        from torch.utils.data import DataLoader

        from retrieval.configs import DEVICE
        from retrieval.training.training_dataset import PairTrainingDataset
        from retrieval.modeling.retrieval_model import RetrievalModel

        print("Dataset", flush=True)

        dataset = PairTrainingDataset(train=True)

        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        print("Loading real batch", flush=True)

        batch = next(iter(loader))

        print("[PASS] real batch", flush=True)

        model = RetrievalModel()

        print("[PASS] model", flush=True)

        model = model.to(DEVICE)

        model.train()

        torch.cuda.synchronize()

        print("[PASS] model CUDA", flush=True)

        anchor = batch["anchor_image"].to(
            DEVICE,
            non_blocking=False,
        )

        target = batch["target_image"].to(
            DEVICE,
            non_blocking=False,
        )

        torch.cuda.synchronize()

        print("[PASS] batch CUDA", flush=True)

        print("ANCHOR FORWARD START", flush=True)

        anchor_embedding = model.encode_image(anchor)

        torch.cuda.synchronize()

        print(
            "[PASS] anchor forward",
            anchor_embedding.shape,
            flush=True,
        )

        print("TARGET FORWARD START", flush=True)

        target_embedding = model.encode_image(target)

        torch.cuda.synchronize()

        print(
            "[PASS] target forward",
            target_embedding.shape,
            flush=True,
        )

        print("DONE", flush=True)
        """,
    ):
        return

    # ======================================================
    # E — LOSS
    # ======================================================

    if not run_phase(
        "E — CONTRASTIVE LOSS",
        """
        import torch
        from torch.utils.data import DataLoader

        from retrieval.configs import DEVICE
        from retrieval.training.training_dataset import PairTrainingDataset
        from retrieval.modeling.retrieval_model import RetrievalModel
        from retrieval.training.loss import build_loss

        dataset = PairTrainingDataset(train=True)

        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        batch = next(iter(loader))

        model = RetrievalModel().to(DEVICE)

        model.train()

        anchor = batch["anchor_image"].to(DEVICE)
        target = batch["target_image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        torch.cuda.synchronize()

        print("Forward", flush=True)

        anchor_embedding = model.encode_image(anchor)

        target_embedding = model.encode_image(target)

        torch.cuda.synchronize()

        print("[PASS] forward", flush=True)

        criterion = build_loss()

        print("[PASS] loss created", flush=True)

        print("LOSS FORWARD START", flush=True)

        result = criterion(
            pair_anchor=anchor_embedding,
            pair_target=target_embedding,
            pair_label=labels,
        )

        torch.cuda.synchronize()

        print("[PASS] loss", flush=True)

        print(
            "loss =",
            result["loss"].item(),
            flush=True,
        )

        print("DONE", flush=True)
        """,
    ):
        return

    # ======================================================
    # F — BACKWARD
    # ======================================================

    if not run_phase(
        "F — BACKWARD",
        """
        import torch
        from torch.utils.data import DataLoader

        from retrieval.configs import DEVICE
        from retrieval.training.training_dataset import PairTrainingDataset
        from retrieval.modeling.retrieval_model import RetrievalModel
        from retrieval.training.loss import build_loss

        dataset = PairTrainingDataset(train=True)

        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        batch = next(iter(loader))

        model = RetrievalModel().to(DEVICE)

        model.train()

        anchor = batch["anchor_image"].to(DEVICE)
        target = batch["target_image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        torch.cuda.synchronize()

        anchor_embedding = model.encode_image(anchor)

        target_embedding = model.encode_image(target)

        criterion = build_loss()

        result = criterion(
            pair_anchor=anchor_embedding,
            pair_target=target_embedding,
            pair_label=labels,
        )

        loss = result["loss"]

        torch.cuda.synchronize()

        print(
            "LOSS READY:",
            loss.item(),
            flush=True,
        )

        print("BACKWARD START", flush=True)

        loss.backward()

        torch.cuda.synchronize()

        print("[PASS] backward", flush=True)

        print("DONE", flush=True)
        """,
    ):
        return

    # ======================================================
    # G — OPTIMIZER
    # ======================================================

    run_phase(
        "G — OPTIMIZER STEP",
        """
        import torch
        from torch.utils.data import DataLoader

        from retrieval.configs import (
            DEVICE,
            LEARNING_RATE,
            WEIGHT_DECAY,
            BETAS,
            EPS,
        )

        from retrieval.training.training_dataset import PairTrainingDataset
        from retrieval.modeling.retrieval_model import RetrievalModel
        from retrieval.training.loss import build_loss

        dataset = PairTrainingDataset(train=True)

        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        batch = next(iter(loader))

        model = RetrievalModel().to(DEVICE)

        model.train()

        criterion = build_loss()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            betas=BETAS,
            eps=EPS,
            weight_decay=WEIGHT_DECAY,
        )

        anchor = batch["anchor_image"].to(DEVICE)
        target = batch["target_image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        torch.cuda.synchronize()

        anchor_embedding = model.encode_image(anchor)

        target_embedding = model.encode_image(target)

        result = criterion(
            pair_anchor=anchor_embedding,
            pair_target=target_embedding,
            pair_label=labels,
        )

        loss = result["loss"]

        torch.cuda.synchronize()

        print(
            "LOSS:",
            loss.item(),
            flush=True,
        )

        optimizer.zero_grad(set_to_none=True)

        print("BACKWARD START", flush=True)

        loss.backward()

        torch.cuda.synchronize()

        print("[PASS] backward", flush=True)

        print("OPTIMIZER STEP START", flush=True)

        optimizer.step()

        torch.cuda.synchronize()

        print("[PASS] optimizer.step()", flush=True)

        print("DONE", flush=True)
        """,
    )


if __name__ == "__main__":
    main()