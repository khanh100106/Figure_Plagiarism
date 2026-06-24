"""
Base class for embedding backbones.

Paper2Fig-2026
"""

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class BaseExtractor(ABC):

    @abstractmethod
    def extract(
        self,
        images: list[Image.Image],
    ) -> np.ndarray:
        """
        Return L2-normalized embeddings.
        """
        pass