# EviCT Execution State

## Current stage

NOTEBOOK_04D_SEED42_STEP1000_AWAITING_KAGGLE_SAVE_VERSION

## Timestamp

2026-09-27T09:45:14.283062+00:00

## Primary baseline progress

Seed 17:

COMPLETE

Final optimizer step:

5000

Best source-selection macro case Dice:

0.75258052

Best checkpoint:

step 3500

Final seed-17 Kaggle recovery:

USER-CONFIRMED DURABLE

---

Seed 42:

IN PROGRESS

Current optimizer step:

1000 / 5000

Precision:

FP32

Images per update:

16

Total seed-42 image exposures:

16000

## Seed-42 source-selection validation

Completed validations:

4

Validation steps:

250, 500, 750, 1000

Best source-selection macro case Dice:

0.69968571

Best step:

750

Current patience:

1 / 8

Threshold used only for checkpoint selection:

0.5

Threshold optimization performed:

NO

## Seed-42 recovery artifacts

last.pt:

VERIFIED

best.pt:

VERIFIED

Raw best selection logits:

VERIFIED

Recovery archive:

EviCT_Notebook04D_seed42_step1000_Recovery.tar

Recovery archive SHA-256:

93c040b55f9d0e858bbf5d3075c6b9bc3116f7460972ed54df3d9d30ed0f1ac0

## GitHub synchronization

Git-safe configuration, hashes, logs, metrics,
audit records, STATE.md and handoff records:

SYNCHRONIZED

Large checkpoints committed to GitHub:

NO

Reason:

artifacts/large and predictions are intentionally excluded
from ordinary Git to avoid unsafe large-binary commits.

Their hashes and locations are recorded in GitHub.

The checkpoint binaries and raw logits are preserved in the
Kaggle recovery archive.

## Resume procedure

1. Restore the seed-42 recovery TAR if the Kaggle runtime is new.
2. Verify archive SHA-256.
3. Load seed-42 checkpoint on CPU first.
4. Restore model.
5. Restore optimizer state to CUDA.
6. Restore scheduler.
7. Restore sampler.
8. Restore CPU/CUDA RNG.
9. Continue from optimizer step 1000.

## Isolation

Training:

12 frozen fitting cases only

Model selection:

4 frozen complete source-selection cases only

Calibration accessed:

NO

Target / MedSeg accessed:

NO

## Target lock

ACTIVE

## Next required action

SAVE A KAGGLE VERSION WITH OUTPUTS.

Confirm these two files are visible:

1. EviCT_Notebook04D_seed42_step1000_Recovery.tar
2. EviCT_Notebook04D_seed42_step1000_Recovery.sha256

After durable save, continue seed 42 from optimizer step 1000.


## GitHub Release Durable Backup

Reason:

Kaggle notebook source exceeded the notebook commit size limit.

Seed:

42

Optimizer step:

1000

Recovery state:

DURABLE

Release tag:

evict-nb04d-seed42-step1000

Release:

https://github.com/itsCodeBakery/EviCT/releases/tag/evict-nb04d-seed42-step1000

Recovery TAR:

EviCT_Notebook04D_seed42_step1000_Recovery.tar

Recovery TAR SHA-256:

93c040b55f9d0e858bbf5d3075c6b9bc3116f7460972ed54df3d9d30ed0f1ac0

SHA file:

EviCT_Notebook04D_seed42_step1000_Recovery.sha256

Remote TAR size:

920606720 bytes

Remote SHA-file size:

65 bytes

Normal Git synchronization:

COMPLETE

Large binary synchronization:

COMPLETE VIA GITHUB RELEASE

Resume source preference:

GitHub Release seed-42 step-1000 recovery TAR

Calibration accessed:

NO

Target / MedSeg accessed:

NO
