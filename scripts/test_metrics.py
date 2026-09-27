import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/kaggle/working/EviCT")

if str(ROOT / "src") not in sys.path:
    sys.path.insert(
        0,
        str(ROOT / "src"),
    )


from evict.metrics import (
    binary_segmentation_metrics,
    threshold_probabilities,
    CaseMetricAccumulator,
    macro_case_summary,
)


results = []


def record(name, passed, details=""):
    results.append(
        {
            "test": name,
            "passed": bool(passed),
            "details": details,
        }
    )

    if not passed:
        raise AssertionError(
            f"{name} failed: {details}"
        )


# ------------------------------------------------------------
# TEST 1 — PERFECT POSITIVE SEGMENTATION
# ------------------------------------------------------------

ref = np.array([
    [0, 1],
    [1, 0],
])

pred = ref.copy()

m = binary_segmentation_metrics(
    ref,
    pred,
)

record(
    "perfect_positive_dice",
    np.isclose(m["dice"], 1.0),
    str(m),
)

record(
    "perfect_positive_iou",
    np.isclose(m["iou"], 1.0),
    str(m),
)

record(
    "perfect_positive_sensitivity",
    np.isclose(m["sensitivity"], 1.0),
    str(m),
)

record(
    "perfect_positive_specificity",
    np.isclose(m["specificity"], 1.0),
    str(m),
)


# ------------------------------------------------------------
# TEST 2 — COMPLETELY DISJOINT FOREGROUND
# ------------------------------------------------------------

ref = np.array([
    [1, 0],
    [0, 0],
])

pred = np.array([
    [0, 1],
    [0, 0],
])

m = binary_segmentation_metrics(
    ref,
    pred,
)

record(
    "disjoint_dice_zero",
    np.isclose(m["dice"], 0.0),
    str(m),
)

record(
    "disjoint_iou_zero",
    np.isclose(m["iou"], 0.0),
    str(m),
)

record(
    "disjoint_sensitivity_zero",
    np.isclose(m["sensitivity"], 0.0),
    str(m),
)


# ------------------------------------------------------------
# TEST 3 — EMPTY REFERENCE + EMPTY PREDICTION
# ------------------------------------------------------------

ref = np.zeros(
    (4, 4),
    dtype=np.uint8,
)

pred = np.zeros_like(
    ref
)

m = binary_segmentation_metrics(
    ref,
    pred,
)

record(
    "empty_empty_dice_one",
    np.isclose(m["dice"], 1.0),
    str(m),
)

record(
    "empty_empty_iou_one",
    np.isclose(m["iou"], 1.0),
    str(m),
)

record(
    "empty_empty_sensitivity_nan",
    math.isnan(m["sensitivity"]),
    str(m),
)

record(
    "empty_empty_specificity_one",
    np.isclose(m["specificity"], 1.0),
    str(m),
)

record(
    "empty_reference_flag",
    m["empty_reference"] is True,
    str(m),
)

record(
    "empty_prediction_flag",
    m["empty_prediction"] is True,
    str(m),
)


# ------------------------------------------------------------
# TEST 4 — EMPTY REFERENCE + FALSE POSITIVE
# ------------------------------------------------------------

ref = np.zeros(
    (2, 2),
    dtype=np.uint8,
)

pred = np.array([
    [1, 0],
    [0, 0],
])

m = binary_segmentation_metrics(
    ref,
    pred,
)

record(
    "empty_ref_false_positive_dice_zero",
    np.isclose(m["dice"], 0.0),
    str(m),
)

record(
    "empty_ref_sensitivity_nan",
    math.isnan(m["sensitivity"]),
    str(m),
)

record(
    "empty_ref_specificity_expected",
    np.isclose(
        m["specificity"],
        3 / 4,
    ),
    str(m),
)


# ------------------------------------------------------------
# TEST 5 — ALL-POSITIVE REFERENCE
# specificity is undefined because no negative reference exists
# ------------------------------------------------------------

ref = np.ones(
    (2, 2),
    dtype=np.uint8,
)

pred = np.ones_like(
    ref
)

m = binary_segmentation_metrics(
    ref,
    pred,
)

record(
    "all_positive_sensitivity_one",
    np.isclose(m["sensitivity"], 1.0),
    str(m),
)

record(
    "all_positive_specificity_nan",
    math.isnan(m["specificity"]),
    str(m),
)


# ------------------------------------------------------------
# TEST 6 — PADDING MUST NOT AFFECT METRICS
# ------------------------------------------------------------

ref = np.array([
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
])

pred = ref.copy()

# Deliberately introduce false positives ONLY in padding.
pred[:, 0] = 1
pred[:, 3] = 1

valid = np.array([
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
])

m_valid = binary_segmentation_metrics(
    ref,
    pred,
    valid_mask=valid,
)

m_unmasked = binary_segmentation_metrics(
    ref,
    pred,
)

record(
    "padding_excluded_dice_one",
    np.isclose(
        m_valid["dice"],
        1.0,
    ),
    str(m_valid),
)

record(
    "padding_would_hurt_without_mask",
    m_unmasked["dice"] < 1.0,
    str(m_unmasked),
)


# ------------------------------------------------------------
# TEST 7 — THRESHOLDING
# ------------------------------------------------------------

probs = np.array([
    0.29,
    0.30,
    0.49,
    0.50,
    0.70,
])

pred = threshold_probabilities(
    probs,
    threshold=0.5,
)

expected = np.array([
    0,
    0,
    0,
    1,
    1,
])

record(
    "probability_threshold",
    np.array_equal(
        pred,
        expected,
    ),
    f"{pred}",
)


# ------------------------------------------------------------
# TEST 8 — PER-CASE AGGREGATION
# ------------------------------------------------------------

acc = CaseMetricAccumulator(
    "case_test"
)

# Slice 1: perfect foreground
acc.update(
    np.array([
        [1, 0],
        [0, 0],
    ]),
    np.array([
        [1, 0],
        [0, 0],
    ]),
)

# Slice 2: one false negative
acc.update(
    np.array([
        [1, 0],
        [0, 0],
    ]),
    np.array([
        [0, 0],
        [0, 0],
    ]),
)

case = acc.compute()

record(
    "case_two_slices",
    case["n_slices"] == 2,
    str(case),
)

record(
    "case_counts",
    (
        case["tp"] == 1
        and case["fn"] == 1
        and case["fp"] == 0
        and case["tn"] == 6
    ),
    str(case),
)

record(
    "case_dice_from_aggregated_counts",
    np.isclose(
        case["dice"],
        2 / 3,
    ),
    str(case),
)

record(
    "case_iou_from_aggregated_counts",
    np.isclose(
        case["iou"],
        1 / 2,
    ),
    str(case),
)


# ------------------------------------------------------------
# TEST 9 — MACRO CASE SUMMARY
# ------------------------------------------------------------

case_a = {
    "dice": 1.0,
    "iou": 1.0,
    "sensitivity": 1.0,
    "specificity": 1.0,
}

case_b = {
    "dice": 0.5,
    "iou": 1 / 3,
    "sensitivity": float("nan"),
    "specificity": 0.8,
}

macro = macro_case_summary(
    [
        case_a,
        case_b,
    ]
)

record(
    "macro_dice",
    np.isclose(
        macro["macro_dice"],
        0.75,
    ),
    str(macro),
)

record(
    "macro_sensitivity_ignores_nan",
    (
        np.isclose(
            macro["macro_sensitivity"],
            1.0,
        )
        and macro[
            "n_valid_sensitivity_cases"
        ] == 1
    ),
    str(macro),
)


# ------------------------------------------------------------
# TEST 10 — SHAPE MISMATCH MUST FAIL
# ------------------------------------------------------------

shape_error_raised = False

try:
    binary_segmentation_metrics(
        np.zeros((2, 2)),
        np.zeros((3, 3)),
    )

except ValueError:
    shape_error_raised = True

record(
    "shape_mismatch_raises",
    shape_error_raised,
)


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

report = {
    "n_tests": len(results),
    "n_passed": sum(
        x["passed"]
        for x in results
    ),
    "all_passed": all(
        x["passed"]
        for x in results
    ),
    "tests": results,
}

OUTPUT = (
    ROOT
    / "artifacts"
    / "audit"
    / "metrics_unit_tests.json"
)

OUTPUT.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)

print("=" * 78)
print("EVICT NOTEBOOK 04A — METRIC UNIT TESTS")
print("=" * 78)

for item in results:
    print(
        "✓",
        item["test"]
    )

print()
print(
    f"Passed: "
    f"{report['n_passed']} / "
    f"{report['n_tests']}"
)

assert report["all_passed"]

print()
print(
    "METRICS UNIT TESTS — PASS"
)
