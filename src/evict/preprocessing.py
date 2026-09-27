"""
EviCT preprocessing utilities.

Deterministic source preprocessing contract:

* CoronaCases:
    clip [-1250, 250] and map to [0, 1]

* Radiopaedia:
    clip [0, 255] and map to [0, 1]

* Aspect-ratio preserving resize to fit 336 x 336.
* Symmetric padding to 336 x 336.
* Bilinear interpolation for images.
* Nearest-neighbor interpolation for masks.
* Explicit valid-pixel mask.
* Reversible crop/resize metadata.

No training augmentation is performed here.
"""

from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F


TARGET_H = 336
TARGET_W = 336


@dataclass(frozen=True)
class ResizePadTransform:

    original_h: int
    original_w: int

    resized_h: int
    resized_w: int

    target_h: int
    target_w: int

    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int

    scale_y: float
    scale_x: float


    def to_dict(self) -> Dict[str, Any]:

        return asdict(self)


def normalize_intensity(
    image: np.ndarray,
    provenance: str,
) -> np.ndarray:

    """
    Convert one source CT slice to float32 [0,1].
    """

    image = np.asarray(
        image,
        dtype=np.float32,
    )


    if provenance == "coronacases_named":

        image = np.clip(
            image,
            -1250.0,
            250.0,
        )

        image = (
            image
            + 1250.0
        ) / 1500.0


    elif provenance == "radiopaedia_named":

        image = np.clip(
            image,
            0.0,
            255.0,
        )

        image = (
            image
            / 255.0
        )


    else:

        raise ValueError(
            f"Unsupported provenance: {provenance}"
        )


    image = np.clip(
        image,
        0.0,
        1.0,
    )


    if not np.isfinite(
        image
    ).all():

        raise ValueError(
            "Non-finite image after normalization."
        )


    return image.astype(
        np.float32,
        copy=False,
    )


def build_resize_pad_transform(
    height: int,
    width: int,
    target_h: int = TARGET_H,
    target_w: int = TARGET_W,
) -> ResizePadTransform:

    if height <= 0 or width <= 0:

        raise ValueError(
            "Invalid source geometry."
        )


    scale = min(
        target_h / height,
        target_w / width,
    )


    # Explicit deterministic half-up style rounding.
    resized_h = max(
        1,
        int(
            np.floor(
                height
                * scale
                + 0.5
            )
        ),
    )


    resized_w = max(
        1,
        int(
            np.floor(
                width
                * scale
                + 0.5
            )
        ),
    )


    resized_h = min(
        resized_h,
        target_h,
    )

    resized_w = min(
        resized_w,
        target_w,
    )


    remaining_h = (
        target_h
        - resized_h
    )

    remaining_w = (
        target_w
        - resized_w
    )


    pad_top = (
        remaining_h
        // 2
    )

    pad_bottom = (
        remaining_h
        - pad_top
    )

    pad_left = (
        remaining_w
        // 2
    )

    pad_right = (
        remaining_w
        - pad_left
    )


    return ResizePadTransform(
        original_h=int(
            height
        ),

        original_w=int(
            width
        ),

        resized_h=int(
            resized_h
        ),

        resized_w=int(
            resized_w
        ),

        target_h=int(
            target_h
        ),

        target_w=int(
            target_w
        ),

        pad_top=int(
            pad_top
        ),

        pad_bottom=int(
            pad_bottom
        ),

        pad_left=int(
            pad_left
        ),

        pad_right=int(
            pad_right
        ),

        scale_y=float(
            resized_h
            / height
        ),

        scale_x=float(
            resized_w
            / width
        ),
    )


def _resize_2d(
    array: np.ndarray,
    new_h: int,
    new_w: int,
    mode: str,
) -> np.ndarray:

    x = torch.from_numpy(
        np.asarray(
            array,
            dtype=np.float32,
        )
    )[
        None,
        None,
    ]


    if mode == "bilinear":

        y = F.interpolate(
            x,
            size=(
                new_h,
                new_w,
            ),
            mode="bilinear",
            align_corners=False,
        )

    elif mode == "nearest":

        y = F.interpolate(
            x,
            size=(
                new_h,
                new_w,
            ),
            mode="nearest",
        )

    else:

        raise ValueError(
            f"Unsupported interpolation: {mode}"
        )


    return (
        y[
            0,
            0,
        ]
        .cpu()
        .numpy()
    )


def forward_image(
    image: np.ndarray,
    transform: ResizePadTransform,
) -> np.ndarray:

    resized = _resize_2d(
        image,
        transform.resized_h,
        transform.resized_w,
        mode="bilinear",
    )


    output = np.zeros(
        (
            transform.target_h,
            transform.target_w,
        ),
        dtype=np.float32,
    )


    y0 = transform.pad_top
    y1 = (
        y0
        + transform.resized_h
    )

    x0 = transform.pad_left
    x1 = (
        x0
        + transform.resized_w
    )


    output[
        y0:y1,
        x0:x1,
    ] = resized


    return output


def forward_mask(
    mask: np.ndarray,
    transform: ResizePadTransform,
) -> np.ndarray:

    mask = (
        np.asarray(
            mask
        )
        > 0.5
    ).astype(
        np.float32
    )


    resized = _resize_2d(
        mask,
        transform.resized_h,
        transform.resized_w,
        mode="nearest",
    )


    output = np.zeros(
        (
            transform.target_h,
            transform.target_w,
        ),
        dtype=np.uint8,
    )


    y0 = transform.pad_top
    y1 = (
        y0
        + transform.resized_h
    )

    x0 = transform.pad_left
    x1 = (
        x0
        + transform.resized_w
    )


    output[
        y0:y1,
        x0:x1,
    ] = (
        resized
        > 0.5
    ).astype(
        np.uint8
    )


    return output


def build_valid_mask(
    transform: ResizePadTransform,
) -> np.ndarray:

    valid = np.zeros(
        (
            transform.target_h,
            transform.target_w,
        ),
        dtype=np.uint8,
    )


    y0 = transform.pad_top
    y1 = (
        y0
        + transform.resized_h
    )

    x0 = transform.pad_left
    x1 = (
        x0
        + transform.resized_w
    )


    valid[
        y0:y1,
        x0:x1,
    ] = 1


    return valid


def inverse_mask(
    padded_mask: np.ndarray,
    transform: ResizePadTransform,
) -> np.ndarray:

    padded_mask = (
        np.asarray(
            padded_mask
        )
        > 0.5
    ).astype(
        np.float32
    )


    y0 = transform.pad_top
    y1 = (
        y0
        + transform.resized_h
    )

    x0 = transform.pad_left
    x1 = (
        x0
        + transform.resized_w
    )


    cropped = padded_mask[
        y0:y1,
        x0:x1,
    ]


    restored = _resize_2d(
        cropped,
        transform.original_h,
        transform.original_w,
        mode="nearest",
    )


    return (
        restored
        > 0.5
    ).astype(
        np.uint8
    )


def binary_iou(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    a = (
        np.asarray(
            a
        )
        > 0.5
    )

    b = (
        np.asarray(
            b
        )
        > 0.5
    )


    intersection = np.logical_and(
        a,
        b,
    ).sum()


    union = np.logical_or(
        a,
        b,
    ).sum()


    if union == 0:
        return 1.0


    return float(
        intersection
        / union
    )
