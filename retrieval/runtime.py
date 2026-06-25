"""
Runtime configuration.

Author: Nguyen Khanh
"""

try:
    import torch

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

except Exception:

    DEVICE = "cpu"