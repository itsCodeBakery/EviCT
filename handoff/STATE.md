# EviCT Execution State

## Current stage

NOTEBOOK_04D_SEED17_COMPLETE_AWAITING_KAGGLE_SAVE_VERSION

## Timestamp

2026-09-27T09:24:15.065449+00:00

## Notebook 04D

Model:

Supervised SegFormer MiT-B1

Seed:

17

Precision:

FP32

## Completion

Final optimizer step:

5000

Completion reason:

MAXIMUM_5000_UPDATES

Maximum allowed step:

5000

## Training exposures

Images per optimizer update:

16

Total image exposures:

80000

## Validation

Every:

250 optimizer updates

Last validation step:

5000

## Best source-selection checkpoint

Macro case Dice:

0.75258052

Best step:

3500

Final patience:

6 / 8

Checkpoint-selection threshold:

0.5

Threshold optimization performed:

NO

## Durable artifacts

best.pt:

YES

last.pt:

YES

last_known_good.pt:

YES

best raw source-selection logits:

YES

## Resume integrity

CPU-first checkpoint loading:

PASS

CPU RNG restore:

PASS

CUDA RNG restore:

PASS

Patient sampler restore:

PASS

Optimizer restore:

PASS

Scheduler restore:

PASS

## Isolation

Training data:

12 fitting cases only

Selection data:

4 complete source-selection cases only

Calibration accessed:

NO

Target / MedSeg accessed:

NO

## Target lock

ACTIVE

## Next

SAVE THIS COMPLETED SEED-17 RUN AS A KAGGLE VERSION.

After durable save, continue with the next protocol stage.
