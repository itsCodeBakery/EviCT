from __future__ import annotations

from typing import Sequence, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import SegformerModel


class SegFormerMLPProjection(nn.Module):
    """
    Per-level SegFormer linear projection.

    Input:
        BCHW encoder feature

    Output:
        BCHW decoder feature
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.proj = nn.Linear(
            in_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 4:

            raise ValueError(
                "Expected BCHW encoder feature, "
                f"received {tuple(x.shape)}."
            )

        b, c, h, w = x.shape

        x = (
            x
            .flatten(2)
            .transpose(1, 2)
        )

        x = self.proj(
            x
        )

        x = (
            x
            .transpose(1, 2)
            .reshape(
                b,
                -1,
                h,
                w,
            )
        )

        return x


class SegFormerMLPDecoder(nn.Module):
    """
    Four-level SegFormer MLP decoder.

    All four MiT features are projected to decoder_dim,
    resized to stage-1 spatial resolution, concatenated,
    and fused to decoder_dim channels.
    """

    def __init__(
        self,
        hidden_sizes: Sequence[int],
        decoder_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        if len(
            hidden_sizes
        ) != 4:

            raise ValueError(
                "Expected four MiT encoder feature levels."
            )

        self.hidden_sizes = list(
            hidden_sizes
        )

        self.decoder_dim = int(
            decoder_dim
        )

        self.projections = nn.ModuleList([
            SegFormerMLPProjection(
                in_channels=int(c),
                out_channels=self.decoder_dim,
            )
            for c
            in hidden_sizes
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(
                self.decoder_dim * 4,
                self.decoder_dim,
                kernel_size=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                self.decoder_dim
            ),

            nn.ReLU(
                inplace=True
            ),
        )

        self.dropout = nn.Dropout(
            p=float(
                dropout
            )
        )

    def forward(
        self,
        features,
    ) -> torch.Tensor:

        if len(
            features
        ) != 4:

            raise ValueError(
                f"Expected four encoder levels; "
                f"received {len(features)}."
            )

        target_size = (
            features[
                0
            ].shape[
                -2:
            ]
        )

        projected = []

        for feature, projection in zip(
            features,
            self.projections,
        ):

            x = projection(
                feature
            )

            if (
                x.shape[
                    -2:
                ]
                != target_size
            ):

                x = F.interpolate(
                    x,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )

            projected.append(
                x
            )

        # Deep-to-shallow SegFormer-style fusion.
        x = torch.cat(
            projected[
                ::-1
            ],
            dim=1,
        )

        x = self.fuse(
            x
        )

        x = self.dropout(
            x
        )

        return x


class EviCTSegFormerB1(nn.Module):
    """
    EviCT supervised visual segmentation baseline.

    Encoder:
        ImageNet-initialized MiT-B1.

    Decoder:
        Four-level SegFormer MLP decoder.

    Decoder representation:
        256 channels at 1/4 input spatial resolution.

    Output:
        One lesion logit per pixel.
    """

    def __init__(
        self,
        checkpoint_path: str,
        decoder_dim: int = 256,
    ):
        super().__init__()

        self.encoder = (
            SegformerModel
            .from_pretrained(
                checkpoint_path,
                local_files_only=True,
            )
        )

        hidden_sizes = list(
            self.encoder
            .config
            .hidden_sizes
        )

        dropout = float(
            getattr(
                self.encoder.config,
                "classifier_dropout_prob",
                0.1,
            )
        )

        self.decoder = SegFormerMLPDecoder(
            hidden_sizes=hidden_sizes,
            decoder_dim=decoder_dim,
            dropout=dropout,
        )

        self.visual_head = nn.Conv2d(
            decoder_dim,
            1,
            kernel_size=1,
        )

        self.decoder_dim = int(
            decoder_dim
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        encoder_output = self.encoder(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        features = encoder_output.hidden_states

        if features is None:

            raise RuntimeError(
                "MiT encoder returned no hidden states."
            )

        if len(
            features
        ) != 4:

            raise RuntimeError(
                "Expected four MiT encoder feature levels; "
                f"received {len(features)}."
            )

        for level, feature in enumerate(
            features,
            start=1,
        ):

            if feature.ndim != 4:

                raise RuntimeError(
                    f"Encoder feature level {level} "
                    f"is not BCHW: {tuple(feature.shape)}"
                )

        decoder_features = self.decoder(
            features
        )

        quarter_logits = self.visual_head(
            decoder_features
        )

        full_logits = F.interpolate(
            quarter_logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return {
            "logits":
                full_logits,

            "quarter_logits":
                quarter_logits,

            "decoder_features":
                decoder_features,

            "encoder_features":
                features,
        }


def masked_supervised_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
):
    """
    EviCT supervised baseline objective:

        0.5 × Soft Dice loss
      + 0.5 × BCE-with-logits

    Padding is removed from BOTH terms using valid_mask.

    Loss is reduced per image first, then averaged over batch.
    Empty-reference slices retain BCE supervision.
    """

    logits = logits.float()
    targets = targets.float()
    valid_mask = valid_mask.float()

    if (
        logits.shape
        != targets.shape
    ):

        raise ValueError(
            f"logits shape = {tuple(logits.shape)}, "
            f"target shape = {tuple(targets.shape)}"
        )

    if (
        logits.shape
        != valid_mask.shape
    ):

        raise ValueError(
            f"logits shape = {tuple(logits.shape)}, "
            f"valid-mask shape = {tuple(valid_mask.shape)}"
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
            "Every image requires at least one valid pixel."
        )

    # --------------------------------------------------------
    # BCE
    # --------------------------------------------------------

    bce_map = (
        F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
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

    # --------------------------------------------------------
    # SOFT DICE
    # --------------------------------------------------------

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
