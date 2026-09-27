# EviCT Execution State

## Current stage

NOTEBOOK_04D_SEED17_RESUME_AUDIT_PASS_STEP1020_FULL_CONTINUATION_PENDING

## Timestamp

2026-09-27T08:36:48.850101+00:00

## Notebook 04D

Model:

Supervised SegFormer MiT-B1

Seed:

17

Precision:

FP32

## Durable Kaggle recovery point

Step:

1000

Archive SHA-256:

26dc931c9b5465c663ef3e3b3b21e1a99525252dedd1bbfc5a1651727ce6f8f6

## Resume audit

Source checkpoint:

last.pt at step 1000

Checkpoint loaded with:

map_location=cpu

Reason:

CPU PyTorch RNG state must remain a CPU torch.ByteTensor.

CPU RNG restore:

PASS

CUDA RNG restore:

PASS

Sampler restore:

PASS

Optimizer restore:

PASS

Scheduler restore:

PASS

## Recovery verification

Successful resumed updates:

20

Current optimizer step:

1020

Last source-selection validation:

1000

Best source-selection macro case Dice:

0.68108677

Best step:

250

Patience:

3 / 8

Step-1020 losses finite:

YES

Step-1020 gradients finite:

YES

## Working recovery checkpoint

recovery.pt:

STEP 1020

SHA-256:

b8dc3aebabff5a00aa6687b696490ad2c86793e3cbae58257d6476710064a24a

## Isolation

Training:

12 fitting cases only

Calibration accessed:

NO

MedSeg / target accessed:

NO

## Target lock

ACTIVE

## Next

Resume from recovery.pt at optimizer step 1020 and
continue the supervised seed-17 run.

Next source-selection validation:

step 1250
