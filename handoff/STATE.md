# EviCT Execution State

## Current stage

NOTEBOOK_04C_TINY_OVERFIT_PASS_FULL_BASELINE_PENDING

## Timestamp

2026-09-27T07:48:08.200352+00:00

## Completed stages

- Notebook 00: COMPLETE
- Notebook 01: COMPLETE
- Notebook 02: COMPLETE
- Notebook 03 preprocessing/geometry: FROZEN
- Notebook 04A metrics/unit tests: PASS
- Notebook 04B MiT-B1 forward/backward smoke: PASS
- Notebook 04C tiny-set overfit sanity: PASS

## Notebook 04C

Purpose:

PIPELINE SANITY / DEBUGGING ONLY

Research result:

NO

Training images:

2

Unique cases:

2

Source provenance groups:

2

Small lesion included:

YES

Representative lesion included:

YES

## Geometry criterion

Required round-trip IoU:

>= 0.85

Threshold lowered:

NO

Reference round-trip IoUs:

[0.8888888888888888, 0.9020356234096693]

Prediction round-trip IoUs:

[0.9037267080745341, 0.9057635675220866]

## Empty-mask metrics

PASS

## Tiny-set memorization

Optimizer steps:

100

Final mean Dice:

0.987520

Final minimum Dice:

0.984595

Small-lesion Dice:

0.990446

Representative-lesion Dice:

0.984595

## Audit recovery

A previous finalization assertion failed because a correct metadata value:

geometry_threshold_lowered = False

was incorrectly included inside all(checks.values()).

No scientific criterion failed.

Training was NOT rerun.

Geometry threshold was NOT changed.

## Figure policy

Required manuscript font:

Times New Roman

Exact Times New Roman available in Kaggle:

NO

Substitute font used:

NO

Text-free debugging overlays:

GENERATED

Labeled manuscript figure:

PENDING

## Target lock

ACTIVE

No MedSeg images, labels, prompts or performance metrics accessed.

## Important

Notebook 04C deliberately memorizes known training images.

Its Dice values are debugging evidence only and must never be
reported as model performance.

## Next

Notebook 04D — full-label source supervised SegFormer-B1 baseline.

Before launch:

1. restore normal stochastic regularization;
2. train only on the frozen 12 fitting cases;
3. use only the 4 source-selection cases for model selection;
4. keep the 4 source-calibration cases untouched;
5. use resumable best.pt and last.pt checkpoints;
6. checkpoint every 250 optimizer updates;
7. store structured training logs;
8. store raw source-selection logits;
9. preserve the target-performance lock.


## Figure font policy update

Preferred font:

Times New Roman

Fallback fonts:

Calibri -> Arial Narrow -> Arial -> Liberation Sans -> DejaVu Sans

Fallback allowed:

YES

Labels may be removed because preferred font is unavailable:

NO

Notebook 04C labeled figure:

GENERATED

Actual font used:

Liberation Sans

PNG:

figures/notebook04c_tiny_overfit.png

PDF:

figures/notebook04c_tiny_overfit.pdf

Caption embedded in image:

NO
