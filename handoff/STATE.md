# EviCT Execution State

## Current stage

NOTEBOOK_05_UNET_THREE_SEED_SOURCE_BASELINE_FROZEN

## Timestamp

2026-09-27T20:06:52.278575+00:00

## Baseline

Competitive residual 2D U-Net

Architecture:

Residual 2D U-Net with GroupNorm

Frozen learning rate:

0.00030000

Primary seeds:

17, 42, 2026

Three-seed source-selection macro case Dice:

0.72220701

Sample standard deviation:

0.01805858

Selection threshold:

0.5 fixed

Architecture tuning:

NO

Threshold tuning:

NO

## Isolation

Calibration accessed:

NO

Target / MedSeg accessed:

NO

Target lock:

ACTIVE

## Durability

Seed 17 final recovery:

https://github.com/itsCodeBakery/EviCT/releases/tag/evict-nb05b-unet-seed17-lr3e4-final-step3500

Seed 42 final recovery:

https://github.com/itsCodeBakery/EviCT/releases/tag/evict-nb05c-unet-seed42-lr3e4-final-step4000

Seed 2026 final recovery:

https://github.com/itsCodeBakery/EviCT/releases/tag/evict-nb05c-unet-seed2026-lr3e4-final-step4500

## Next

The competitive U-Net reference baseline is frozen.
Proceed to the remaining E03 comparison component (SegCT-CLIP reproduction/reimplementation)
before starting the proposed semantic branch.
