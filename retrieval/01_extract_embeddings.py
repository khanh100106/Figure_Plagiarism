"""
Module 01

Extract DINOv2 embeddings for every figure in Paper2Fig-2026.

Author: Nguyen Khanh
"""
import json
from datetime import datetime
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm import tqdm

from retrieval.configs import (
    EMBEDDING_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    SAVE_EVERY,
    BACKBONE,
    MODEL_NAME,
    EMBEDDING_DIM,
)

from retrieval.dataset import (
    FigureDataset,
    create_dataloader,
)

from retrieval.embedding import (
    EmbeddingExtractor,
)

from retrieval.logger import (
    setup_logger,
)

from retrieval.checkpoints import (
    CheckpointManager,
)

# ============================================================
# Save Embedding Chunk
# ============================================================

def save_chunk(
    embedding_buffer,
    metadata_buffer,
    chunk_id,
    logger,
):
    """
    Save one embedding chunk to disk.

    Parameters
    ----------
    embedding_buffer : list
        List of embedding vectors.

    metadata_buffer : list
        List of metadata dictionaries.

    chunk_id : int
        Current chunk index.

    logger : logging.Logger
        Logger instance.

    Returns
    -------
    int
        Next chunk id.
    """

    if len(embedding_buffer) == 0:
        return chunk_id

    chunk_dir = EMBEDDING_DIR / "chunks"
    chunk_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embedding_file = (
        chunk_dir /
        f"embeddings_{chunk_id:05d}.npy"
    )

    metadata_file = (
        chunk_dir /
        f"metadata_{chunk_id:05d}.csv"
    )

    embeddings = np.vstack(
        embedding_buffer
    )

    np.save(
        embedding_file,
        embeddings,
    )

    pd.DataFrame(
        metadata_buffer
    ).to_csv(
        metadata_file,
        index=False,
    )

    logger.info(
        f"Chunk {chunk_id:05d} saved "
        f"({len(embedding_buffer):,} embeddings)"
    )

    embedding_buffer.clear()
    metadata_buffer.clear()

    return chunk_id + 1

def main():

    start_time = time.time()

    # ---------------------------------------------------------
    # Logger
    # ---------------------------------------------------------

    logger = setup_logger(
        LOG_DIR / "extract_embeddings.log"
    )

    logger.info("=" * 60)
    logger.info("Paper2Fig-2026 Embedding Extraction")
    logger.info("=" * 60)

    # ---------------------------------------------------------
    # Checkpoint
    # ---------------------------------------------------------

    checkpoint = CheckpointManager(
        CHECKPOINT_DIR /
        "embedding_checkpoint.json"
    )

    state = checkpoint.load()

    start_index = state.get(
        "processed",
        0,
    )

    logger.info(
        f"Resume from sample: {start_index:,}"
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = FigureDataset(
        start_index=start_index
    )

    loader = create_dataloader(
        dataset
    )

    logger.info(
        f"Remaining figures: {len(dataset):,}"
    )

    # ---------------------------------------------------------
    # Backbone
    # ---------------------------------------------------------

    extractor = EmbeddingExtractor()

    logger.info(
        "Backbone loaded successfully."
    )

    # ---------------------------------------------------------
    # Buffers
    # ---------------------------------------------------------

    embedding_buffer = []

    metadata_buffer = []

    chunk_id = state.get(
        "chunk_id",
        0,
    )

    processed = start_index

    logger.info(
        f"Start chunk: {chunk_id}"
    )
    # ---------------------------------------------------------
    # Embedding Extraction Loop
    # ---------------------------------------------------------

    for batch in tqdm(
        loader,
        desc="Extracting Embeddings",
    ):

        # ---------------------------------------------
        # Images
        # ---------------------------------------------

        images = [
            sample["image"]
            for sample in batch
        ]

        # ---------------------------------------------
        # Extract embeddings
        # ---------------------------------------------

        embeddings = extractor.extract(
            images
        )

        # ---------------------------------------------
        # Store embeddings
        # ---------------------------------------------


        for sample, embedding in zip(
                batch,
                embeddings,
        ):
            embedding_buffer.append(
                embedding
            )

            metadata = {
                key: value
                for key, value in sample.items()
                if key != "image"
            }

            metadata_buffer.append(
                metadata
            )

            processed += 1

        # ---------------------------------------------
        # Save every SAVE_EVERY samples
        # ---------------------------------------------

        if len(embedding_buffer) >= SAVE_EVERY:
            chunk_id = save_chunk(
                embedding_buffer,
                metadata_buffer,
                chunk_id,
                logger,
            )

            checkpoint.save(
                processed=processed,
                chunk_id=chunk_id,
            )

    # ---------------------------------------------------------
    # Save Remaining Embeddings
    # ---------------------------------------------------------

    if len(embedding_buffer) > 0:
        chunk_id = save_chunk(
            embedding_buffer,
            metadata_buffer,
            chunk_id,
            logger,
        )

        checkpoint.save(
            processed=processed,
            chunk_id=chunk_id,
        )

    logger.info(
        "Embedding extraction finished."
    )

    # ---------------------------------------------------------
    # Merge all chunks
    # ---------------------------------------------------------

    merge_chunks(
        logger
    )

    # ---------------------------------------------------------
    # Save embedding information
    # ---------------------------------------------------------

    save_embedding_info()

    # ---------------------------------------------------------
    # Create completed flag
    # ---------------------------------------------------------

    completed = EMBEDDING_DIR / "completed.flag"
    completed.touch()

    # ---------------------------------------------------------
    # Finish
    # ---------------------------------------------------------
    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info(
        f"Finished in {elapsed / 60:.2f} minutes"
    )
    logger.info("=" * 60)



# ============================================================
# Merge Embedding Chunks
# ============================================================

def merge_chunks(logger):
    """
    Merge all embedding chunks into one file.

    Outputs
    -------
    embeddings.npy

    embedding_index.csv
    """

    chunk_dir = EMBEDDING_DIR / "chunks"

    embedding_files = sorted(
        chunk_dir.glob("embeddings_*.npy")
    )

    metadata_files = sorted(
        chunk_dir.glob("metadata_*.csv")
    )

    if len(embedding_files) == 0:

        logger.warning(
            "No embedding chunks found."
        )

        return

    logger.info(
        f"Merging {len(embedding_files)} chunks..."
    )

    # --------------------------------------------------------
    # Merge embeddings
    # --------------------------------------------------------

    embedding_list = []

    for file in embedding_files:

        embedding_list.append(
            np.load(file)
        )

    embeddings = np.vstack(
        embedding_list
    )

    np.save(
        EMBEDDING_DIR / "embeddings.npy",
        embeddings,
    )

    # --------------------------------------------------------
    # Merge metadata
    # --------------------------------------------------------

    metadata_list = []

    for file in metadata_files:

        metadata_list.append(
            pd.read_csv(file)
        )

    metadata = pd.concat(
        metadata_list,
        ignore_index=True,
    )

    metadata.to_csv(

        EMBEDDING_DIR / "embedding_index.csv",

        index=False,

    )

    logger.info(
        f"Merged embeddings : {embeddings.shape}"
    )

    logger.info(
        f"Embedding index    : {len(metadata):,}"
    )

# ============================================================
# Save Embedding Information
# ============================================================

def save_embedding_info():

    info = {

        "dataset": "Paper2Fig-2026",

        "backbone": BACKBONE,

        "model_name": MODEL_NAME,

        "embedding_dim": EMBEDDING_DIM,

        "num_embeddings": len(
            pd.read_csv(
                EMBEDDING_DIR / "embedding_index.csv"
            )
        ),

        "created_at": datetime.now().isoformat(),

    }

    with open(

        EMBEDDING_DIR / "embedding_info.json",

        "w",

        encoding="utf-8",

    ) as f:

        json.dump(
            info,
            f,
            indent=4,
        )

if __name__ == "__main__":

    main()