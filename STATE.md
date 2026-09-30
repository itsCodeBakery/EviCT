# EviCT Execution State

## Current stage

NOTEBOOK_07_B050_SEED17_WARMUP_FROZEN

## Notebook 07 warm-up

Budget:

50 percent visible fitting masks (b050)

Seed:

17

Method:

Common supervised real-text warm-up

Optimizer step:

1000

Warm-up terminal step:

1000

Best source-selection macro-case Dice:

0.71837039

Best validation step:

1000

Labeled image exposures:

8000

Hidden fitting masks used:

NO

## Branching contract

The terminal step-1000 student checkpoint is the common initialization for:

1. supervised continuation
2. confidence-only EMA
3. agreement-filtered EMA

No branch may replace this warm-up with the 100%-label Notebook-06 trained checkpoint.

## Isolation

Calibration accessed:

NO

Target / MedSeg accessed:

NO

Target lock:

ACTIVE

## Next

Complete the common step-1000 warm-up, then branch the three Notebook-07 methods
without changing the patient split, optimizer schedule, text prototypes, or warm-up state.
