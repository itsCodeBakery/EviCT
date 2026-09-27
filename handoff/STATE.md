# EviCT Execution State

## Current stage

NOTEBOOK_05C_UNET_SEED_42_RUNNING_STEP_1250

## Timestamp

2026-09-27T17:58:54.649241+00:00

## Baseline

Competitive residual 2D U-Net

## Frozen protocol

Architecture:

Residual 2D U-Net with GroupNorm

Learning rate:

0.00030000

Learning-rate status:

FROZEN FROM SEED-17 PILOT

Training seed:

42

Split seed:

17

Precision:

FP32

Optimizer step:

1250

Best source-selection macro case Dice:

0.67127798

Best checkpoint step:

500

Patience:

3 / 8

Image exposures:

20000

Selection threshold:

0.5 fixed

Architecture tuning:

NO

Threshold tuning:

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

## Next

Complete the frozen U-Net primary-seed runs for seeds 42 and 2026,
then aggregate seeds 17, 42, and 2026 without changing the protocol.
