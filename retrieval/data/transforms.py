"""
Paper2Fig-2026 Retrieval Framework

Image Transforms
----------------

Training and evaluation transforms.

Author
------
Nguyen Khanh
"""

from __future__ import annotations

from typing import Any

from torchvision import transforms

from retrieval.data.partial_occlusion import (
    PartialOcclusion,
)

from retrieval.configs import (
    IMAGE_SIZE,
    IMAGE_MEAN,
    IMAGE_STD,
    ENABLE_HORIZONTAL_FLIP,
    HORIZONTAL_FLIP_PROB,
    ENABLE_ROTATION,
    ROTATION_DEGREES,
    ENABLE_COLOR_JITTER,
    COLOR_BRIGHTNESS,
    COLOR_CONTRAST,
    COLOR_SATURATION,
    COLOR_HUE,
    ENABLE_GAUSSIAN_BLUR,
    GAUSSIAN_KERNEL_SIZE,
    GAUSSIAN_SIGMA,
    ENABLE_RANDOM_ERASING,
    ERASING_PROBABILITY,
    ERASING_SCALE,
    ERASING_RATIO,
    ENABLE_PARTIAL_OCCLUSION,
    PARTIAL_OCCLUSION_PROBABILITY,
    PARTIAL_OCCLUSION_MIN_WIDTH,
    PARTIAL_OCCLUSION_MAX_WIDTH,
    PARTIAL_OCCLUSION_MIN_HEIGHT,
    PARTIAL_OCCLUSION_MAX_HEIGHT,
    PARTIAL_OCCLUSION_FILL,
    PARTIAL_OCCLUSION_MAX_REGIONS,
)


def build_train_transform() -> transforms.Compose:
    """
    Build training transform.

    Returns
    -------
    transforms.Compose
        Image augmentation pipeline.
    """

    transform_list: list[Any] = []

    # ----------------------------------------------------------
    # Resize
    # ----------------------------------------------------------

    transform_list.append(

        transforms.Resize(

            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )

        )

    )

    # ----------------------------------------------------------
    # Data augmentation (PIL Image)
    # ----------------------------------------------------------

    if ENABLE_HORIZONTAL_FLIP:

        transform_list.append(

            transforms.RandomHorizontalFlip(

                p=HORIZONTAL_FLIP_PROB,

            )

        )

    if ENABLE_ROTATION:

        transform_list.append(

            transforms.RandomRotation(

                degrees=ROTATION_DEGREES,

            )

        )

    if ENABLE_COLOR_JITTER:

        transform_list.append(

            transforms.ColorJitter(

                brightness=COLOR_BRIGHTNESS,

                contrast=COLOR_CONTRAST,

                saturation=COLOR_SATURATION,

                hue=COLOR_HUE,

            )

        )

    if ENABLE_GAUSSIAN_BLUR:

        transform_list.append(

            transforms.GaussianBlur(

                kernel_size=GAUSSIAN_KERNEL_SIZE,

                sigma=GAUSSIAN_SIGMA,

            )

        )

    if ENABLE_PARTIAL_OCCLUSION:

        transform_list.append(

            PartialOcclusion(

                probability=PARTIAL_OCCLUSION_PROBABILITY,

                min_width=PARTIAL_OCCLUSION_MIN_WIDTH,

                max_width=PARTIAL_OCCLUSION_MAX_WIDTH,

                min_height=PARTIAL_OCCLUSION_MIN_HEIGHT,

                max_height=PARTIAL_OCCLUSION_MAX_HEIGHT,

                fill=PARTIAL_OCCLUSION_FILL,

                max_regions=PARTIAL_OCCLUSION_MAX_REGIONS,

            )

        )

    # ----------------------------------------------------------
    # Tensor conversion
    # ----------------------------------------------------------

    transform_list.append(

        transforms.ToTensor()

    )

    # ----------------------------------------------------------
    # Normalization
    # ----------------------------------------------------------

    transform_list.append(

        transforms.Normalize(

            mean=IMAGE_MEAN,

            std=IMAGE_STD,

        )

    )

    # ----------------------------------------------------------
    # Tensor augmentation
    # ----------------------------------------------------------

    if ENABLE_RANDOM_ERASING:

        transform_list.append(

            transforms.RandomErasing(

                p=ERASING_PROBABILITY,

                scale=ERASING_SCALE,

                ratio=ERASING_RATIO,

            )

        )

    return transforms.Compose(

        transform_list,

    )


def build_eval_transform() -> transforms.Compose:
    """
    Build validation/test transform.

    Returns
    -------
    transforms.Compose
        Evaluation preprocessing pipeline.
    """

    return transforms.Compose(

        [

            transforms.Resize(

                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                )

            ),

            transforms.ToTensor(),

            transforms.Normalize(

                mean=IMAGE_MEAN,

                std=IMAGE_STD,

            ),

        ]

    )