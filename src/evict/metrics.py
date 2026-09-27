from __future__ import annotations

from typing import Iterable, Dict, Optional, Any
import math
import numpy as np


def _to_numpy(x):
    """
    Convert NumPy arrays or detached PyTorch tensors to NumPy.
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()

    return np.asarray(x)


def _prepare_binary_inputs(
    reference,
    prediction,
    valid_mask=None,
):
    """
    Validate shapes and return flattened uint8 arrays restricted
    to valid pixels only.
    """

    ref = _to_numpy(reference)
    pred = _to_numpy(prediction)

    if ref.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch: reference={ref.shape}, prediction={pred.shape}"
        )

    ref = (ref > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)

    if valid_mask is None:
        valid = np.ones_like(ref, dtype=bool)
    else:
        valid = _to_numpy(valid_mask)

        if valid.shape != ref.shape:
            raise ValueError(
                f"Valid-mask shape mismatch: valid={valid.shape}, "
                f"reference={ref.shape}"
            )

        valid = valid > 0

    if valid.sum() == 0:
        raise ValueError("No valid pixels remain after applying valid_mask.")

    ref = ref[valid]
    pred = pred[valid]

    return ref, pred


def threshold_probabilities(
    probabilities,
    threshold: float = 0.5,
):
    """
    Convert probabilities to a binary prediction.
    """

    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must lie in [0,1].")

    p = _to_numpy(probabilities)

    return (p >= threshold).astype(np.uint8)


def binary_confusion(
    reference,
    prediction,
    valid_mask=None,
) -> Dict[str, int]:
    """
    Compute binary segmentation confusion counts on valid pixels.
    """

    ref, pred = _prepare_binary_inputs(
        reference,
        prediction,
        valid_mask,
    )

    tp = int(np.logical_and(ref == 1, pred == 1).sum())
    tn = int(np.logical_and(ref == 0, pred == 0).sum())
    fp = int(np.logical_and(ref == 0, pred == 1).sum())
    fn = int(np.logical_and(ref == 1, pred == 0).sum())

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "valid_pixels": int(ref.size),
    }


def metrics_from_confusion(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
) -> Dict[str, Any]:
    """
    EviCT frozen segmentation-metric policy.

    Dice / IoU:
        empty-reference + empty-prediction -> 1.0

    Sensitivity:
        NaN when no positive reference pixels exist.

    Specificity:
        NaN when no negative reference pixels exist.
    """

    tp = int(tp)
    tn = int(tn)
    fp = int(fp)
    fn = int(fn)

    if min(tp, tn, fp, fn) < 0:
        raise ValueError("Confusion counts must be non-negative.")

    dice_den = 2 * tp + fp + fn

    if dice_den == 0:
        dice = 1.0
    else:
        dice = (2.0 * tp) / dice_den

    iou_den = tp + fp + fn

    if iou_den == 0:
        iou = 1.0
    else:
        iou = tp / iou_den

    positive_reference = tp + fn

    if positive_reference == 0:
        sensitivity = float("nan")
    else:
        sensitivity = tp / positive_reference

    negative_reference = tn + fp

    if negative_reference == 0:
        specificity = float("nan")
    else:
        specificity = tn / negative_reference

    return {
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),

        "empty_reference": bool(
            (tp + fn) == 0
        ),

        "empty_prediction": bool(
            (tp + fp) == 0
        ),
    }


def binary_segmentation_metrics(
    reference,
    prediction,
    valid_mask=None,
) -> Dict[str, Any]:
    """
    Compute binary segmentation metrics on valid pixels.
    """

    counts = binary_confusion(
        reference,
        prediction,
        valid_mask,
    )

    metrics = metrics_from_confusion(
        counts["tp"],
        counts["tn"],
        counts["fp"],
        counts["fn"],
    )

    return {
        **counts,
        **metrics,
    }


class CaseMetricAccumulator:
    """
    Accumulate confusion counts across slices of one case.

    Primary per-case metrics are calculated after accumulation.
    This deliberately avoids averaging slice-level Dice scores.
    """

    def __init__(self, case_id: str):
        self.case_id = str(case_id)

        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.valid_pixels = 0
        self.n_slices = 0

    def update(
        self,
        reference,
        prediction,
        valid_mask=None,
    ):
        counts = binary_confusion(
            reference,
            prediction,
            valid_mask,
        )

        self.tp += counts["tp"]
        self.tn += counts["tn"]
        self.fp += counts["fp"]
        self.fn += counts["fn"]
        self.valid_pixels += counts["valid_pixels"]
        self.n_slices += 1

    def compute(self):
        metrics = metrics_from_confusion(
            self.tp,
            self.tn,
            self.fp,
            self.fn,
        )

        return {
            "case_id": self.case_id,
            "n_slices": int(self.n_slices),

            "tp": int(self.tp),
            "tn": int(self.tn),
            "fp": int(self.fp),
            "fn": int(self.fn),

            "valid_pixels": int(self.valid_pixels),

            **metrics,
        }


def macro_case_summary(
    case_records: Iterable[Dict[str, Any]],
):
    """
    Macro-average case-level metrics.

    Undefined sensitivity/specificity values remain NaN and are
    excluded from their corresponding macro means.
    """

    rows = list(case_records)

    if len(rows) == 0:
        raise ValueError("No case records supplied.")

    output = {
        "n_cases": len(rows),
    }

    for metric_name in [
        "dice",
        "iou",
        "sensitivity",
        "specificity",
    ]:
        values = np.asarray(
            [
                float(row[metric_name])
                for row in rows
            ],
            dtype=np.float64,
        )

        finite = values[np.isfinite(values)]

        output[f"macro_{metric_name}"] = (
            float(finite.mean())
            if len(finite) > 0
            else float("nan")
        )

        output[f"n_valid_{metric_name}_cases"] = int(
            len(finite)
        )

    return output
