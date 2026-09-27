from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvGNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = 8 if out_channels >= 8 else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SegCTClipVisualDecoder(nn.Module):
    """
    Explicit EviCT adaptation of the documented SegCT-CLIP visual pathway.

    This is NOT an exact SegCT-CLIP reproduction.

    Inputs
    ------
    low_feature, high_feature:
        Frozen CLIP ViT-L/14-336 patch feature maps with shape
        [B, 1024, 24, 24].

    Design
    ------
    - low/high 1x1 projections to 128 channels;
    - concatenation and 3x3 fusion;
    - four bilinear upsampling/refinement stages;
    - 1x1 binary lesion head at 336x336.

    The exact layer indices, fusion operator and decoder topology were not
    disclosed by the base paper; these choices are therefore recorded as an
    adaptation in config/notebook05_segct_clip_visual_adaptation.json.
    """

    def __init__(
        self,
        in_channels: int = 1024,
        projection_channels: int = 128,
    ):
        super().__init__()

        self.low_projection = nn.Conv2d(
            in_channels,
            projection_channels,
            kernel_size=1,
            bias=False,
        )
        self.high_projection = nn.Conv2d(
            in_channels,
            projection_channels,
            kernel_size=1,
            bias=False,
        )

        fused_channels = projection_channels * 2

        self.fusion = nn.Sequential(
            ConvGNAct(fused_channels, 256),
            ConvGNAct(256, 256),
        )

        self.up1 = ConvGNAct(256, 128)
        self.up2 = ConvGNAct(128, 64)
        self.up3 = ConvGNAct(64, 32)
        self.up4 = ConvGNAct(32, 16)

        self.head = nn.Conv2d(
            16,
            1,
            kernel_size=1,
        )

    def forward(
        self,
        low_feature: torch.Tensor,
        high_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        low = self.low_projection(low_feature)
        high = self.high_projection(high_feature)

        x = torch.cat(
            [low, high],
            dim=1,
        )

        x = self.fusion(x)

        x = F.interpolate(
            x,
            size=(48, 48),
            mode="bilinear",
            align_corners=False,
        )
        x = self.up1(x)

        x = F.interpolate(
            x,
            size=(96, 96),
            mode="bilinear",
            align_corners=False,
        )
        x = self.up2(x)

        x = F.interpolate(
            x,
            size=(192, 192),
            mode="bilinear",
            align_corners=False,
        )
        x = self.up3(x)

        x = F.interpolate(
            x,
            size=(336, 336),
            mode="bilinear",
            align_corners=False,
        )
        x = self.up4(x)

        logits = self.head(x)

        return {
            "logits": logits,
        }


def masked_soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    epsilon: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """
    Per-image masked soft Dice loss.

    Padding is excluded using valid_mask. The base paper explicitly reports
    Dice as the segmentation component of SegCT-CLIP's objective. Because the
    caption bank is unavailable, this visual-pathway adaptation does not invent
    the missing contrastive supervision and optimizes only the documented
    segmentation component.
    """

    if logits.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: logits={logits.shape}, target={target.shape}"
        )

    if valid_mask.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: valid={valid_mask.shape}, target={target.shape}"
        )

    probabilities = torch.sigmoid(logits)

    probabilities = probabilities * valid_mask
    target = target * valid_mask

    dims = tuple(
        range(
            1,
            target.ndim,
        )
    )

    intersection = (
        probabilities
        * target
    ).sum(
        dim=dims
    )

    denominator = (
        probabilities.sum(
            dim=dims
        )
        + target.sum(
            dim=dims
        )
    )

    dice = (
        2.0 * intersection
        + epsilon
    ) / (
        denominator
        + epsilon
    )

    dice_loss = (
        1.0
        - dice
    ).mean()

    return {
        "loss": dice_loss,
        "dice_loss": dice_loss,
    }
