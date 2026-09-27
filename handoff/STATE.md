# EviCT Execution State

## Current stage

NOTEBOOK_04D_SEED2026_COMPLETE_DURABLE

## Timestamp

2026-09-27T13:10:12.381254+00:00

## Notebook 04D — Seed 2026

Model:

Supervised SegFormer MiT-B1

Precision:

FP32

Training seed:

2026

Frozen split seed:

17

Final optimizer step:

5000

Last validation step:

5000

Best source-selection macro case Dice:

0.74248709

Best checkpoint step:

4250

Final patience:

3 / 8

Total image exposures:

80000

## Recovery

Model:

PRESERVED

Optimizer:

PRESERVED

Scheduler:

PRESERVED

Sampler:

PRESERVED

CPU RNG:

PRESERVED

CUDA RNG:

PRESERVED

## Final durable backup

Recovery TAR:

EviCT_Notebook04D_seed2026_final_step5000_Recovery.tar

SHA-256:

f32656901ed12ae30b5fbe2c524d114bc9b816ea3f1667d610ea2b3489894fd4

GitHub Release:

https://github.com/itsCodeBakery/EviCT/releases/tag/evict-nb04d-seed2026-final-step5000

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

Best macro case Dice:

0.75258052

Seed 42:

COMPLETE

Best macro case Dice:

0.76708522

Seed 2026:

COMPLETE

Best macro case Dice:

0.74248709

## Next

Perform three-seed baseline aggregation, validation-overlay generation,
checkpoint/logit audit, and freeze Notebook 04D.
