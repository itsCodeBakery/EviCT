# EviCT Execution State

## Current stage

NOTEBOOK_05C_UNET_SEED_42_RUNNING_STEP_3500

## Timestamp

2026-09-27T18:38:08.783093+00:00

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

3500

Best source-selection macro case Dice:

0.70757153

Best checkpoint step:

2000

Patience:

6 / 8

Image exposures:

56000

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
