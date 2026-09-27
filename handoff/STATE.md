# EviCT Execution State

## Current stage

NOTEBOOK_05B_UNET_LEARNING_RATE_FROZEN

## Timestamp

2026-09-27T17:20:46.222354+00:00

## Baseline

Competitive residual 2D U-Net

Pilot seed:

17

Architecture tuning:

NO

Candidate initialization:

IDENTICAL

Initialization SHA-256:

0698a4bf77cbdf2f0ee99f080eab4677f778d8bfbb15b244f00695dc6f0b19b1

## Candidate 1

Learning rate:

0.00010000

Final optimizer step:

3500

Stop reason:

EARLY_STOPPING_PATIENCE_8

Best source-selection macro case Dice:

0.74123800

Best checkpoint step:

1500

Durable release:

https://github.com/itsCodeBakery/EviCT/releases/tag/evict-nb05b-unet-seed17-lr1e4-final-step3500

## Candidate 2

Learning rate:

0.00030000

Final optimizer step:

3500

Stop reason:

EARLY_STOPPING_PATIENCE_8

Best source-selection macro case Dice:

0.74238804

Best checkpoint step:

1500

Durable release:

https://github.com/itsCodeBakery/EviCT/releases/tag/evict-nb05b-unet-seed17-lr3e4-final-step3500

## Frozen learning rate

Selected candidate:

lr3e4

Selected learning rate:

0.00030000

Selection metric:

Frozen source-selection macro case Dice

Selection margin:

0.00115005

Threshold:

0.5 fixed

## Isolation

Calibration accessed:

NO

Target / MedSeg accessed:

NO

Target lock:

ACTIVE

## Next

Keep the selected U-Net architecture and learning rate fixed.
Run the remaining primary training seeds 42 and 2026 under the
same source-only protocol before final U-Net aggregation.
