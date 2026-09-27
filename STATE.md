# EviCT Execution State

## Current stage

NOTEBOOK_05A_UNET_PROTOCOL_AND_GPU_SMOKE_PASS

## Timestamp

2026-09-27T14:48:42.452422+00:00

## Notebook 04

Status:

COMPLETE_SOURCE_BASELINE_FROZEN

Three-seed source-selection macro case Dice:

0.75405094 ± 0.01236482

Notebook 04 retraining allowed:

NO

## Notebook 05

Current baseline:

Competitive residual 2D U-Net

Baseline type:

Independent strong source reference

Pretraining:

NONE

Precision:

FP32

Input:

336 x 336 grayscale replicated to 3 channels

Normalization:

Same ImageNet normalization as frozen SegFormer baseline

Loss:

0.5 soft Dice + 0.5 BCE

Loss-equivalence audit:

PASS

GPU smoke micro-batch:

4

Planned gradient accumulation:

4

Planned effective batch:

16

Peak GPU allocated:

1.8252 GiB

Peak GPU reserved:

2.3516 GiB

Total parameters:

8111297

Trainable parameters:

8111297

Forward:

PASS

Backward:

PASS

Finite gradients:

PASS

## Tuning allowance

Architecture tuning:

NO

Pilot training seed:

17

Allowed learning-rate candidates:

0.0001, 0.0003

Selection criterion:

Frozen source-selection macro case Dice

Threshold:

0.5 fixed

## Isolation

Fitting cases:

12 frozen source cases

Selection cases:

4 frozen source cases

Calibration accessed:

NO

Target / MedSeg accessed:

NO

Target lock:

ACTIVE

## Next

Run Notebook 05B U-Net pilot training on seed 17 for the two predeclared
learning-rate candidates. Select the learning rate using source-selection
macro case Dice only. Then freeze that setting before seeds 42 and 2026.
