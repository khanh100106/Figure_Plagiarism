import numpy as np

from retrieval.configs import (
    EMBEDDING_DIR,
)

embeddings = np.load(
    EMBEDDING_DIR / "embeddings.npy"
)

print(embeddings.shape)