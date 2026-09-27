# EviCT Execution State

## Current stage

NOTEBOOK_04D_SEED42_RUNNING_STEP_1000

## Timestamp

2026-09-27T09:45:07.131395+00:00

## Completed primary baseline seeds

Seed 17:

COMPLETE — 5000 / 5000

Best source-selection macro case Dice:

0.75258052

Best seed-17 checkpoint:

step 3500

Seed-17 final Kaggle recovery:

USER-CONFIRMED DURABLE

## Current run

Model:

Supervised SegFormer MiT-B1

Training seed:

42

Precision:

FP32

Current optimizer step:

1000 / 5000

Current durable segment:

0 -> 1000

Images per optimizer update:

16

Total seed-42 image exposures:

16000

## Validation

Frequency:

250 optimizer updates

Last completed validation:

1000

Current best source-selection macro case Dice:

0.6996857137077372

Current best step:

750

Patience:

1 / 8

## Recovery

recovery.pt:

every 50 successful optimizer updates

last.pt:

every 250 optimizer updates

best.pt:

on source-selection improvement

Resume method:

CPU-first checkpoint load + RNG/sampler restoration

## GitHub synchronization

Config:

config/notebook04d_seed42_config.json

Hashes:

config/notebook04d_seed42_hashes.json

Training log:

artifacts/audit/notebook04d_seed42_train_log.csv

Selection metrics:

artifacts/audit/notebook04d_seed42_selection_metrics.csv

Case metrics:

artifacts/audit/notebook04d_seed42_selection_case_metrics.csv

Handoff:

handoff/notebook04d_seed42_resume.json

Large checkpoints committed to Git:

NO

Reason:

Repository intentionally ignores artifacts/large and predictions.

Large recovery location:

Kaggle output archive

## Isolation

Training:

12 frozen fitting cases only

Model selection:

4 frozen complete source-selection cases only

Calibration cases accessed:

NO

Target / MedSeg accessed:

NO

## Target lock

ACTIVE

## Next

Continue seed 42 until step 1000, then save a Kaggle
version containing the seed-42 recovery TAR and SHA-256.
