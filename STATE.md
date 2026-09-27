# EviCT Execution State

## Current stage

NOTEBOOK_04D_SEED2026_STEP1000_DURABLE

## Timestamp

2026-09-27T12:23:11.055120+00:00

## Notebook 04D — Seed 2026

Model:

Supervised SegFormer MiT-B1

Precision:

FP32

Training seed:

2026

Frozen split seed:

17

Current optimizer step:

1000 / 5000

Last validation step:

1000

Best source-selection macro case Dice:

0.72659239

Best checkpoint step:

500

Patience:

2 / 8

Images seen:

16000

## Initialization

Pinned ImageNet MiT-B1:

YES

Seed-17 checkpoint used:

NO

Seed-42 checkpoint used:

NO

## Durable recovery

Recovery TAR:

EviCT_Notebook04D_seed2026_step1000_Recovery.tar

SHA-256:

6d0edd2635f4093f854a5b7b2fd0f2254039a5438f5c670c4e972069f48737fa

GitHub Release:

https://github.com/itsCodeBakery/EviCT/releases/tag/evict-nb04d-seed2026-step1000

Remote TAR:

VERIFIED

Remote SHA file:

VERIFIED

## Isolation

Training:

12 frozen fitting cases only

Selection:

4 frozen complete source-selection cases only

Calibration accessed:

NO

Target / MedSeg accessed:

NO

## Target lock

ACTIVE

## Seed status

Seed 17:

COMPLETE

Seed 42:

COMPLETE

Seed 2026:

STEP 1000 DURABLE — CONTINUATION PENDING

## Next

Perform 20-update recovery audit from seed-2026 step 1000, then continue to early stop or step 5000.
