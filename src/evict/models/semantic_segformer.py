from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import SegformerModel


class SegFormerMLPProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW feature, got {tuple(x.shape)}")
        b, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x.transpose(1, 2).reshape(b, -1, h, w)


class SegFormerMLPDecoder(nn.Module):
    def __init__(self, hidden_sizes: Sequence[int], decoder_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        if len(hidden_sizes) != 4:
            raise ValueError("Expected four MiT feature levels")
        self.projections = nn.ModuleList([
            SegFormerMLPProjection(int(c), int(decoder_dim)) for c in hidden_sizes
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(int(decoder_dim) * 4, int(decoder_dim), 1, bias=False),
            nn.BatchNorm2d(int(decoder_dim)),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, features) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError(f"Expected four features, got {len(features)}")
        target_size = features[0].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            x = projection(feature)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            projected.append(x)
        return self.dropout(self.fuse(torch.cat(projected[::-1], dim=1)))


class EviCTSemanticSegFormerB1(nn.Module):
    """SegFormer-B1 plus a fixed-prototype semantic branch."""

    def __init__(
        self,
        checkpoint_path: str,
        foreground_prototype: torch.Tensor,
        background_prototype: torch.Tensor,
        decoder_dim: int = 256,
        tau: float = 0.1,
        alpha0: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()

        fg = torch.as_tensor(foreground_prototype, dtype=torch.float32).flatten()
        bg = torch.as_tensor(background_prototype, dtype=torch.float32).flatten()
        if fg.shape != bg.shape or fg.ndim != 1:
            raise ValueError("Foreground/background prototypes must be 1-D and same shape")
        if fg.numel() < 2:
            raise ValueError("Text embedding dimension is unexpectedly small")

        fg = F.normalize(fg, dim=0)
        bg = F.normalize(bg, dim=0)
        self.register_buffer("foreground_prototype", fg, persistent=True)
        self.register_buffer("background_prototype", bg, persistent=True)

        self.encoder = SegformerModel.from_pretrained(
            checkpoint_path,
            local_files_only=True,
        )
        hidden_sizes = list(self.encoder.config.hidden_sizes)
        dropout = float(getattr(self.encoder.config, "classifier_dropout_prob", 0.1))
        self.decoder = SegFormerMLPDecoder(hidden_sizes, decoder_dim=decoder_dim, dropout=dropout)
        self.visual_head = nn.Conv2d(int(decoder_dim), 1, kernel_size=1)
        self.semantic_projection = nn.Conv2d(
            int(decoder_dim), int(fg.numel()), kernel_size=1, bias=True
        )

        if not (0.0 < float(alpha0) < 1.0):
            raise ValueError("alpha0 must lie strictly between 0 and 1")
        self.alpha_logit = nn.Parameter(
            torch.tensor(math.log(float(alpha0) / (1.0 - float(alpha0))), dtype=torch.float32)
        )
        self.tau = float(tau)
        self.eps = float(eps)
        if self.tau <= 0:
            raise ValueError("tau must be positive")

    @property
    def text_dim(self) -> int:
        return int(self.foreground_prototype.numel())

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoder_output = self.encoder(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        features = encoder_output.hidden_states
        if features is None or len(features) != 4:
            raise RuntimeError("MiT-B1 did not return four hidden-state feature maps")

        decoder_features = self.decoder(features)
        visual_quarter_logits = self.visual_head(decoder_features)

        z = self.semantic_projection(decoder_features)
        z = F.normalize(z.float(), p=2, dim=1, eps=self.eps)
        fg = self.foreground_prototype.view(1, -1, 1, 1)
        bg = self.background_prototype.view(1, -1, 1, 1)
        semantic_quarter_logits = (
            (z * fg).sum(dim=1, keepdim=True)
            - (z * bg).sum(dim=1, keepdim=True)
        ) / self.tau

        alpha = torch.sigmoid(self.alpha_logit)
        fused_quarter_logits = visual_quarter_logits.float() + alpha * semantic_quarter_logits
        full_logits = F.interpolate(
            fused_quarter_logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        visual_full_logits = F.interpolate(
            visual_quarter_logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return {
            "logits": full_logits,
            "fused_quarter_logits": fused_quarter_logits,
            "visual_logits": visual_full_logits,
            "visual_quarter_logits": visual_quarter_logits,
            "semantic_quarter_logits": semantic_quarter_logits,
            "decoder_features": decoder_features,
            "alpha": alpha,
        }


def masked_supervised_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
):
    logits = logits.float()
    targets = targets.float()
    valid_mask = valid_mask.float()
    if logits.shape != targets.shape or logits.shape != valid_mask.shape:
        raise ValueError("logits, targets and valid_mask must have identical shapes")

    valid_count = valid_mask.sum(dim=(1, 2, 3))
    if torch.any(valid_count <= 0):
        raise ValueError("Every image must contain valid pixels")

    bce_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce = (bce_map * valid_mask).sum(dim=(1, 2, 3)) / valid_count

    probs = torch.sigmoid(logits) * valid_mask
    target_valid = targets * valid_mask
    intersection = (probs * target_valid).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + target_valid.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - (2.0 * intersection + eps) / (denominator + eps)
    total = 0.5 * dice_loss + 0.5 * bce
    return {
        "loss": total.mean(),
        "dice_loss": dice_loss.mean(),
        "bce_loss": bce.mean(),
        "per_image_loss": total,
    }


def semantic_patch_auxiliary_loss(
    semantic_quarter_logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    low_fraction: float = 0.1,
    high_fraction: float = 0.9,
    require_full_valid_patch: bool = True,
):
    """Auxiliary text loss on unambiguous decoder-grid patches only."""
    semantic_quarter_logits = semantic_quarter_logits.float()
    targets = targets.float()
    valid_mask = valid_mask.float()

    if targets.shape != valid_mask.shape:
        raise ValueError("targets and valid_mask must have identical shapes")
    qh, qw = semantic_quarter_logits.shape[-2:]
    h, w = targets.shape[-2:]
    if h % qh != 0 or w % qw != 0:
        raise ValueError("Input dimensions must divide exactly into decoder grid")
    kh, kw = h // qh, w // qw

    valid_fraction = F.avg_pool2d(valid_mask, kernel_size=(kh, kw), stride=(kh, kw))
    lesion_fraction = (
        F.avg_pool2d(targets * valid_mask, kernel_size=(kh, kw), stride=(kh, kw))
        / valid_fraction.clamp_min(1e-6)
    )

    usable = valid_fraction >= (1.0 - 1e-6) if require_full_valid_patch else valid_fraction > 0
    confident_bg = usable & (lesion_fraction <= float(low_fraction))
    confident_fg = usable & (lesion_fraction >= float(high_fraction))
    selected = confident_bg | confident_fg
    patch_target = confident_fg.float()

    if selected.any():
        loss_map = F.binary_cross_entropy_with_logits(
            semantic_quarter_logits, patch_target, reduction="none"
        )
        loss = loss_map[selected].mean()
    else:
        loss = semantic_quarter_logits.sum() * 0.0

    return {
        "loss": loss,
        "selected_patches": int(selected.sum().detach().cpu().item()),
        "foreground_patches": int(confident_fg.sum().detach().cpu().item()),
        "background_patches": int(confident_bg.sum().detach().cpu().item()),
    }
