"""
Paper2Fig-2026 Retrieval Training
Transforms
"""
from __future__ import annotations
import random
from PIL import (
    Image,
    ImageDraw,
)
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from retrieval.configs import (
    IMAGE_SIZE,
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
    PARTIAL_OCCLUSION_MIN,
    PARTIAL_OCCLUSION_MAX,
)

# ============================================================
# DINOv2 Normalization
# ============================================================
DINO_MEAN = (
    0.485,
    0.456,
    0.406,
)
DINO_STD = (
    0.229,
    0.224,
    0.225,
)
# ============================================================
# Partial Occlusion
# ============================================================
class PartialOcclusion:
    """
    Simulate partially plagiarized figures by
    covering one border or corner.
    """
    def __init__(
        self,
        probability: float,
        min_ratio: float,
        max_ratio: float,
    ) -> None:
        self.probability = probability
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
    def __call__(
        self,
        image: Image.Image,
    ) -> Image.Image:
        if random.random() > self.probability:
            return image
        image = image.copy()
        draw = ImageDraw.Draw(image)
        width, height = image.size
        ratio = random.uniform(
            self.min_ratio,
            self.max_ratio,
        )
        occ_w = int(width * ratio)
        occ_h = int(height * ratio)
        mode = random.choice(
            [
                "top",
                "bottom",
                "left",
                "right",
                "top_left",
                "top_right",
                "bottom_left",
                "bottom_right",
            ]
        )
        fill = tuple(
            random.randint(180, 255)
            for _ in range(3)
        )
        if mode == "top":
            box = (
                0,
                0,
                width,
                occ_h,
            )
        elif mode == "bottom":
            box = (
                0,
                height - occ_h,
                width,
                height,
            )
        elif mode == "left":
            box = (
                0,
                0,
                occ_w,
                height,
            )
        elif mode == "right":
            box = (
                width - occ_w,
                0,
                width,
                height,
            )
        elif mode == "top_left":
            box = (
                0,
                0,
                occ_w,
                occ_h,
            )
        elif mode == "top_right":
            box = (
                width - occ_w,
                0,
                width,
                occ_h,
            )
        elif mode == "bottom_left":
            box = (
                0,
                height - occ_h,
                occ_w,
                height,
            )
        else:
            box = (
                width - occ_w,
                height - occ_h,
                width,
                height,
            )
        draw.rectangle(
            box,
            fill=fill,
        )
        return image
# ============================================================
# Train Transform
# ============================================================
def build_train_transform(
    dataset_name: str = "paper2fig",
) -> T.Compose:
    transforms = []
    #
    # Dataset-specific augmentation
    #

    enable_rotation = ENABLE_ROTATION

    enable_color = ENABLE_COLOR_JITTER

    enable_occlusion = ENABLE_PARTIAL_OCCLUSION

    if dataset_name.lower() == "shape":
        enable_rotation = False

        enable_color = False
    transforms.append(
        T.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            interpolation=InterpolationMode.BICUBIC,
        )
    )
    if ENABLE_HORIZONTAL_FLIP:
        transforms.append(
            T.RandomHorizontalFlip(
                p=HORIZONTAL_FLIP_PROB,
            )
        )
    if enable_rotation:
        transforms.append(
            T.RandomRotation(
                degrees=ROTATION_DEGREES,
            )
        )
    if enable_color:
        transforms.append(
            T.ColorJitter(
                brightness=COLOR_BRIGHTNESS,
                contrast=COLOR_CONTRAST,
                saturation=COLOR_SATURATION,
                hue=COLOR_HUE,
            )
        )
    if ENABLE_GAUSSIAN_BLUR:
        transforms.append(
            T.GaussianBlur(
                kernel_size=GAUSSIAN_KERNEL_SIZE,
                sigma=GAUSSIAN_SIGMA,
            )
        )
    if enable_occlusion:
        transforms.append(
            PartialOcclusion(
                probability=PARTIAL_OCCLUSION_PROBABILITY,
                min_ratio=PARTIAL_OCCLUSION_MIN,
                max_ratio=PARTIAL_OCCLUSION_MAX,
            )
        )
    transforms.append(
        T.ToTensor()
    )
    if ENABLE_RANDOM_ERASING:
        transforms.append(
            T.RandomErasing(
                p=ERASING_PROBABILITY,
                scale=ERASING_SCALE,
                ratio=ERASING_RATIO,
            )
        )
    transforms.append(
        T.Normalize(
            mean=DINO_MEAN,
            std=DINO_STD,
        )
    )
    return T.Compose(
        transforms,
    )
# ============================================================
# Validation Transform
# ============================================================
def build_validation_transform(
    dataset_name: str = "paper2fig",
) -> T.Compose:
    return T.Compose(
        [
            T.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(
                mean=DINO_MEAN,
                std=DINO_STD,
            ),
        ]
    )
# ============================================================
# Inference Transform
# ============================================================
def build_inference_transform() -> T.Compose:
    return build_validation_transform()
# ============================================================
# Summary
# ============================================================
def summarize_transform(
    train: bool = True,
) -> str:
    lines = []
    lines.append("=" * 60)
    if train:
        lines.append(
            "Training Transform Pipeline"
        )
    else:
        lines.append(
            "Inference Transform Pipeline"
        )
    lines.append("=" * 60)
    lines.append(
        "Resize({})".format(
            IMAGE_SIZE,
        )
    )
    if train:
        if ENABLE_HORIZONTAL_FLIP:
            lines.append(
                "RandomHorizontalFlip"
            )
        if ENABLE_ROTATION:
            lines.append(
                "RandomRotation({})".format(
                    ROTATION_DEGREES,
                )
            )
        if ENABLE_COLOR_JITTER:
            lines.append(
                "ColorJitter"
            )
        if ENABLE_GAUSSIAN_BLUR:
            lines.append(
                "GaussianBlur"
            )
        if ENABLE_PARTIAL_OCCLUSION:
            lines.append(
                "PartialOcclusion"
            )
        if ENABLE_RANDOM_ERASING:
            lines.append(
                "RandomErasing"
            )
    lines.append(
        "Normalize(DINOv2)"
    )
    return "\n".join(
        lines,
    )
