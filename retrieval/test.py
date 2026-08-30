"""
Paper2Fig-2026 Retrieval Framework

STEP 3 — MODEL TEST
-------------------

Test:
    PairDataset
        -> DataLoader
        -> Siamese DINOv2
        -> Projection Head
        -> Figure embeddings

Author
------
Nguyen Khanh
"""

from __future__ import annotations

import time

import torch

from retrieval.configs import (
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)

from retrieval.dataset import PairDataset

from retrieval.modeling.factory import (
    build_model,
)

from torch.utils.data import DataLoader


# ============================================================
# Helpers
# ============================================================

def check(
    condition: bool,
    message: str,
) -> None:

    if condition:
        print(f"[PASS] {message}")

    else:
        raise RuntimeError(
            f"[FAIL] {message}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Paper2Fig-2026")
    print("STEP 3 — RETRIEVAL MODEL TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    print("\n[1] Check device")

    print(
        f"DEVICE       : {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA version : "
            f"{torch.version.cuda}"
        )

    else:

        print(
            "CUDA is not available."
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\n[2] Build PairDataset")

    dataset = PairDataset(
        train=True,
    )

    print(
        f"Dataset size : "
        f"{len(dataset):,}"
    )

    check(
        len(dataset) > 0,
        "Dataset is not empty",
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    print("\n[3] Build DataLoader")

    print(
        f"Batch size   : {BATCH_SIZE}"
    )

    print(
        f"Num workers  : {NUM_WORKERS}"
    )

    print(
        f"Pin memory   : {PIN_MEMORY}"
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    check(
        dataloader.batch_size == BATCH_SIZE,
        "DataLoader batch size is correct",
    )

    check(
        dataloader.num_workers == NUM_WORKERS,
        "DataLoader num_workers is correct",
    )

    check(
        dataloader.pin_memory == PIN_MEMORY,
        "DataLoader pin_memory is correct",
    )

    # --------------------------------------------------------
    # Fetch batch
    # --------------------------------------------------------

    print("\n[4] Fetch first batch")

    start_time = time.time()

    batch = next(
        iter(dataloader)
    )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"Elapsed      : "
        f"{elapsed:.2f} sec"
    )

    check(
        isinstance(batch, dict),
        "Batch is dictionary",
    )

    print(
        f"Batch keys   : "
        f"{list(batch.keys())}"
    )

    # --------------------------------------------------------
    # Input tensors
    # --------------------------------------------------------

    print("\n[5] Validate input tensors")

    anchor = batch["image"]

    target = batch["image"]

    label = batch["label"]

    check(
        isinstance(anchor, torch.Tensor),
        "Anchor is Tensor",
    )

    check(
        isinstance(target, torch.Tensor),
        "Target is Tensor",
    )

    check(
        isinstance(label, torch.Tensor),
        "Label is Tensor",
    )

    print(
        f"Image shape  : "
        f"{anchor.shape}"
    )

    print(
        f"Label shape  : "
        f"{label.shape}"
    )

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # The current PairDataset returns ONE image per sample.
    #
    # Therefore the current PairDataset is NOT yet compatible
    # with the Siamese model that requires:
    #
    #     anchor_images
    #     target_images
    #
    # We stop here intentionally.
    #

    print("\n" + "=" * 70)
    print("MODEL TEST STOPPED")
    print("=" * 70)

    print()
    print(
        "The current PairDataset returns:"
    )

    print(
        "    image"
    )

    print(
        "    caption"
    )

    print(
        "    label"
    )

    print(
        "    anchor_id"
    )

    print(
        "    target_id"
    )

    print()
    print(
        "But the Figure Plagiarism model requires:"
    )

    print(
        "    anchor image"
    )

    print(
        "    target image"
    )

    print()
    print(
        "Therefore PairDataset must be corrected "
        "before testing the model."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()