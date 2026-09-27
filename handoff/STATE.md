# EviCT Execution State

## Current stage

NOTEBOOK_05B_UNET_LR1E4_RUNNING_STEP_1250

## Timestamp

2026-09-27T15:56:31.946850+00:00

## Notebook 04

Status:

COMPLETE_SOURCE_BASELINE_FROZEN

Notebook 04 retraining:

NO

## Notebook 05B

Baseline:

Competitive residual 2D U-Net

Pilot seed:

17

Candidate:

lr1e4

Learning rate:

0.00010000

Precision:

FP32

Optimizer step:

1250

Best source-selection macro case Dice:

0.65699226

Best checkpoint step:

1250

Patience:

0 / 8

Image exposures:

20000

Selection threshold:

0.5 fixed

Architecture tuning:

NO

## Isolation

Fitting cases:

12 frozen source fitting cases

Selection cases:

4 frozen complete source-selection cases

Calibration accessed:

NO

Target / MedSeg accessed:

NO

Target lock:

ACTIVE

## Pilot contract

Learning-rate candidates:

0.0001, 0.0003

Candidate comparison:

PENDING until both candidates complete the full declared budget or early stopping.

## Next

Continue the declared U-Net learning-rate pilot without changing architecture,
data, loss, selection metric, threshold, or target lock.
