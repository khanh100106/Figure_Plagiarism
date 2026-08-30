"""
Paper2Fig-2026 Retrieval Framework

Partial Occlusion
-----------------

Custom augmentation for scientific figures.

Author
------
Nguyen Khanh
"""

from __future__ import annotations

import random

from PIL import (
    Image,
    ImageDraw,
)


class PartialOcclusion:
    """
    Randomly occlude one or more rectangular regions.
    """

    def __init__(
        self,
        probability: float,
        min_width: float,
        max_width: float,
        min_height: float,
        max_height: float,
        fill: str = "black",
        max_regions: int = 1,
    ) -> None:

        self.probability = probability

        self.min_width = min_width
        self.max_width = max_width

        self.min_height = min_height
        self.max_height = max_height

        self.fill = fill

        self.max_regions = max_regions

    def __call__(
        self,
        image: Image.Image,
    ) -> Image.Image:

        if random.random() > self.probability:

            return image

        image = image.copy()

        width, height = image.size

        draw = ImageDraw.Draw(
            image,
        )

        num_regions = random.randint(
            1,
            self.max_regions,
        )

        for _ in range(
            num_regions,
        ):

            occ_width = int(

                width *

                random.uniform(
                    self.min_width,
                    self.max_width,
                )

            )

            occ_height = int(

                height *

                random.uniform(
                    self.min_height,
                    self.max_height,
                )

            )

            occ_width = max(
                occ_width,
                1,
            )

            occ_height = max(
                occ_height,
                1,
            )

            x = random.randint(
                0,
                max(
                    width - occ_width,
                    0,
                ),
            )

            y = random.randint(
                0,
                max(
                    height - occ_height,
                    0,
                ),
            )

            draw.rectangle(

                [

                    x,
                    y,
                    x + occ_width,
                    y + occ_height,

                ],

                fill=self._fill_color(),

            )

        return image

    def _fill_color(
        self,
    ):

        if self.fill == "black":

            return (
                0,
                0,
                0,
            )

        if self.fill == "white":

            return (
                255,
                255,
                255,
            )

        if self.fill == "random":

            return (

                random.randint(
                    0,
                    255,
                ),

                random.randint(
                    0,
                    255,
                ),

                random.randint(
                    0,
                    255,
                ),

            )

        raise ValueError(
            f"Unsupported fill mode: {self.fill}"
        )

    def __repr__(
        self,
    ) -> str:

        return (

            f"{self.__class__.__name__}("

            f"probability={self.probability}, "

            f"fill={self.fill}, "

            f"max_regions={self.max_regions})"

        )