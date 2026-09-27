# EviCT Execution State

## Current stage

NOTEBOOK_05B_UNET_LR_PILOT_BOTH_STEP1000_DURABLE

## Timestamp

2026-09-27T15:30:48.333433+00:00

## Notebook 05B

Baseline:

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

0.0001

Step:

1000

Best source-selection macro case Dice so far:

0.63906136

Best step so far:

750

Durable release:

https://github.com/itsCodeBakery/EviCT/releases/tag/evict-nb05b-unet-seed17-lr1e4-step1000

## Candidate 2

Learning rate:

0.0003

Step:

1000

Best source-selection macro case Dice so far:

0.69771367

Best step so far:

1000

Durable release:

https://github.com/itsCodeBakery/EviCT/releases/tag/evict-nb05b-unet-seed17-lr3e4-step1000

## Selection rule

No winner is selected at step 1000.

Both predeclared candidates must complete the same full declared training
budget or early-stopping rule before the learning rate is frozen.

## Isolation

Calibration accessed:

NO

Target / MedSeg accessed:

NO

Target lock:

ACTIVE

## Next

Run 20-update recovery audits for both candidates, then continue both under
the identical frozen protocol to early stopping or step 5000.
