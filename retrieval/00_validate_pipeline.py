import pandas as pd

from retrieval.configs import (
    MINING_DIR,
)

candidate = pd.read_csv(
    MINING_DIR / "topk_candidates.csv"
)

print(candidate["similarity"].describe())

print(
    candidate["similarity"].quantile(
        [0.90, 0.95, 0.97, 0.99]
    )
)