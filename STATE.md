# EviCT Execution State

## Current stage

NOTEBOOK_05C_UNET_SEED_2026_RUNNING_STEP_2000

## Timestamp

2026-09-27T19:22:09.448191+00:00

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

2026

Split seed:

17

Precision:

FP32

Optimizer step:

2000

Best source-selection macro case Dice:

0.70793484

Best checkpoint step:

1500

Patience:

2 / 8

Image exposures:

32000

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
