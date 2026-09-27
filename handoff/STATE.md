# EviCT Execution State

## Current stage

NOTEBOOK_04D_SEED17_FP32_RUNNING_STEP_250

## Timestamp

2026-09-27T08:18:52.046232+00:00

## Notebook 04D

Model:

Supervised SegFormer MiT-B1

Seed:

17

Precision:

FP32

AMP:

DISABLED

Reason:

Previous abandoned FP16 attempt produced a non-finite
scaled gradient before optimizer update 119. No durable
checkpoint had been reached.

Scientific training protocol changed:

NO

## Split

Fitting:

12 cases

Selection:

4 cases

Calibration:

4 cases — UNTOUCHED

## Current optimizer step

250

## Planned maximum

5000

## Effective labeled images per update

16

## Validation frequency

250 optimizer updates

## Current best source-selection macro case Dice

0.68108677

## Best step

250

## Patience

0 / 8

## Checkpoints

recovery.pt:

every 50 successful optimizer updates

last.pt:

every 250 optimizer updates

best.pt:

when source-selection macro case Dice improves

## Target lock

ACTIVE

MedSeg accessed:

NO

Calibration accessed:

NO

## Figure font policy

Preferred:

Times New Roman

Fallback:

Calibri -> Arial Narrow -> Arial -> Liberation Sans -> DejaVu Sans

Labels remain present.
