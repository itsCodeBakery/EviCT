# EviCT Execution State

## Current stage

NOTEBOOK_04A_METRICS_PASS_MODEL_SMOKE_PENDING

## Timestamp

2026-09-27T07:18:15.477351+00:00

## Completed stages

- Notebook 00: COMPLETE
- Notebook 01: COMPLETE
- Notebook 02: COMPLETE
- Notebook 03 preprocessing/geometry: FROZEN
- Notebook 04A segmentation metrics: PASS

## Notebook 04A metric policy

Dice empty-reference + empty-prediction:

1.0

IoU empty-reference + empty-prediction:

1.0

Sensitivity with no positive reference:

NaN

Specificity with no negative reference:

NaN

Padding:

Excluded using valid-pixel mask.

Primary aggregation:

Accumulate TP/TN/FP/FN across all valid pixels of each complete case,
then compute case-level metrics.

Cross-case reporting:

Macro-average case-level metrics.

Debug probability threshold:

0.5

Final threshold:

NOT YET SELECTED.

Later selection grid:

0.3, 0.4, 0.5, 0.6, 0.7

Target-driven threshold tuning:

PROHIBITED

## Unit tests

Passed:

28 / 28

## Target lock

ACTIVE

No target metrics inspected.

## Figure standard

Times New Roman
Bold readable labels
600-dpi PNG
Vector PDF
No captions embedded in figures

## Next

Notebook 04B:

1. inspect GPU/runtime;
2. pin/load ImageNet SegFormer MiT-B1;
3. construct standard segmentation decoder;
4. forward-pass shape test;
5. backward-pass finite-gradient test;
6. then overfit 2–4 fitting images before any long run.
