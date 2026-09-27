
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Two 3x3 convolutions with GroupNorm and residual projection.

    GroupNorm is used instead of BatchNorm so the model remains stable
    with small medical-imaging micro-batches.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 8,
    ):
        super().__init__()

        g1 = min(groups, out_channels)

        while out_channels % g1 != 0:
            g1 -= 1

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.norm1 = nn.GroupNorm(
            g1,
            out_channels,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.norm2 = nn.GroupNorm(
            g1,
            out_channels,
        )

        if in_channels != out_channels:

            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            )

        else:

            self.skip = nn.Identity()

        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = self.skip(
            x
        )

        x = self.conv1(
            x
        )

        x = self.norm1(
            x
        )

        x = self.activation(
            x
        )

        x = self.conv2(
            x
        )

        x = self.norm2(
            x
        )

        x = x + residual

        x = self.activation(
            x
        )

        return x


class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.fuse = ResidualBlock(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:

        x = self.up(
            x
        )

        if x.shape[-2:] != skip.shape[-2:]:

            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat(
            [
                x,
                skip,
            ],
            dim=1,
        )

        return self.fuse(
            x
        )


class EviCTResidualUNet2D(nn.Module):
    """
    Competitive 2D U-Net reference baseline.

    Input:
        B x 3 x 336 x 336

    Output:
        B x 1 x 336 x 336 lesion logits

    Initialization:
        Random / no external pretraining.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
    ):
        super().__init__()

        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2

        self.enc1 = ResidualBlock(
            in_channels,
            c1,
        )

        self.enc2 = ResidualBlock(
            c1,
            c2,
        )

        self.enc3 = ResidualBlock(
            c2,
            c3,
        )

        self.enc4 = ResidualBlock(
            c3,
            c4,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.bottleneck = ResidualBlock(
            c4,
            c5,
        )

        self.dec4 = UpBlock(
            c5,
            c4,
            c4,
        )

        self.dec3 = UpBlock(
            c4,
            c3,
            c3,
        )

        self.dec2 = UpBlock(
            c3,
            c2,
            c2,
        )

        self.dec1 = UpBlock(
            c2,
            c1,
            c1,
        )

        self.head = nn.Conv2d(
            c1,
            1,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):

        e1 = self.enc1(
            x
        )

        e2 = self.enc2(
            self.pool(
                e1
            )
        )

        e3 = self.enc3(
            self.pool(
                e2
            )
        )

        e4 = self.enc4(
            self.pool(
                e3
            )
        )

        b = self.bottleneck(
            self.pool(
                e4
            )
        )

        d4 = self.dec4(
            b,
            e4,
        )

        d3 = self.dec3(
            d4,
            e3,
        )

        d2 = self.dec2(
            d3,
            e2,
        )

        d1 = self.dec1(
            d2,
            e1,
        )

        logits = self.head(
            d1
        )

        return {
            "logits": logits
        }


def masked_supervised_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Exact Notebook-04 supervised objective:

        0.5 * soft Dice loss
      + 0.5 * BCE-with-logits

    Padding is excluded from both terms.

    Reduction:
        per image first, then batch mean.
    """

    logits = logits.float()
    targets = targets.float()
    valid_mask = valid_mask.float()

    if logits.shape != targets.shape:
        raise ValueError(
            "logits and targets must have equal shape."
        )

    if logits.shape != valid_mask.shape:
        raise ValueError(
            "logits and valid mask must have equal shape."
        )

    valid_count = valid_mask.sum(
        dim=(
            1,
            2,
            3,
        )
    )

    if torch.any(
        valid_count <= 0
    ):
        raise ValueError(
            "Every image requires valid pixels."
        )

    bce_map = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    bce_per_image = (
        (
            bce_map
            * valid_mask
        )
        .sum(
            dim=(
                1,
                2,
                3,
            )
        )
        / valid_count
    )

    probabilities = torch.sigmoid(
        logits
    )

    probabilities = (
        probabilities
        * valid_mask
    )

    target_valid = (
        targets
        * valid_mask
    )

    intersection = (
        probabilities
        * target_valid
    ).sum(
        dim=(
            1,
            2,
            3,
        )
    )

    denominator = (
        probabilities.sum(
            dim=(
                1,
                2,
                3,
            )
        )
        +
        target_valid.sum(
            dim=(
                1,
                2,
                3,
            )
        )
    )

    soft_dice = (
        (
            2.0
            * intersection
            + eps
        )
        /
        (
            denominator
            + eps
        )
    )

    dice_loss_per_image = (
        1.0
        - soft_dice
    )

    total_per_image = (
        0.5
        * dice_loss_per_image
        +
        0.5
        * bce_per_image
    )

    return {
        "loss":
            total_per_image.mean(),

        "dice_loss":
            dice_loss_per_image.mean(),

        "bce_loss":
            bce_per_image.mean(),

        "per_image_loss":
            total_per_image,
    }
