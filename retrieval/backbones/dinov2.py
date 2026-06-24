"""
DINOv2 Backbone

Main embedding model for Paper2Fig-2026.

Author: Nguyen Khanh
"""

from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from retrieval.configs import (
    DEVICE,
    MODEL_NAME,
    IMAGE_SIZE,
)


class DINOv2Extractor:
    """
    DINOv2 embedding extractor.

    Output:
        L2-normalized float32 embeddings.
    """

    def __init__(self):

        self.device = DEVICE

        print("=" * 60)
        print("Loading DINOv2 Backbone")
        print(f"Model  : {MODEL_NAME}")
        print(f"Device : {self.device}")
        print("=" * 60)

        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            MODEL_NAME,
        )

        self.model.eval()

        self.model.to(self.device)

        for p in self.model.parameters():
            p.requires_grad = False

        self.transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    # -------------------------------------------------------

    def preprocess(
        self,
        image: Image.Image,
    ) -> torch.Tensor:

        if image.mode != "RGB":
            image = image.convert("RGB")

        return self.transform(image)

    # -------------------------------------------------------

    @torch.inference_mode()
    def extract(
        self,
        images: List[Image.Image],
    ) -> np.ndarray:

        batch = torch.stack(

            [
                self.preprocess(img)
                for img in images
            ]

        )

        batch = batch.to(
            self.device,
            non_blocking=True,
        )

        embeddings = self.model(batch)

        embeddings = embeddings.float()

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1,
        )

        return embeddings.cpu().numpy()

    # -------------------------------------------------------

    def embedding_dim(self):

        dummy = Image.new(
            "RGB",
            (IMAGE_SIZE, IMAGE_SIZE),
        )

        emb = self.extract(
            [dummy]
        )

        return emb.shape[1]