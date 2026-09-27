# EviCT Execution State

## Current stage

NOTEBOOK_04D_SEED17_STEP1000_AWAITING_KAGGLE_SAVE_VERSION

## Timestamp

2026-09-27T08:25:47.659861+00:00

## Notebook 04D

Model:

Supervised SegFormer MiT-B1

Seed:

17

Precision:

FP32

AMP:

DISABLED

## Numerical recovery note

The original pre-checkpoint FP16 attempt was abandoned after
a non-finite scaled gradient before optimizer update 119.

No durable checkpoint existed.

The run was restarted deterministically from optimizer step 0.

Scientific protocol changed:

NO

Only execution precision changed from FP16 to FP32.

## Training schedule

Current optimizer step:

1000 / 5000

Warm-up:

200 updates

Post-warmup:

Cosine learning-rate decay

Encoder base LR:

1e-4

Decoder base LR:

3e-4

Weight decay:

0.01

Images per successful optimizer update:

16

Total image exposures:

16000

## Split

Training:

12 fitting cases only

Model selection:

4 complete source-selection cases only

Calibration:

4 cases untouched

## Validation

Every:

250 optimizer updates

Completed validations:

4

Fixed checkpoint-selection threshold:

0.5

Threshold tuning performed:

NO

## Best source-selection checkpoint

Macro case Dice:

0.68108677

Best step:

250

Patience:

3 / 8

## Recovery

recovery.pt:

written every 50 successful updates

last.pt:

written every 250 updates

best.pt:

written when source-selection macro case Dice improves

last.pt reload verification:

PASS

## Raw validation logits

Best source-selection raw logits:

SAVED

Dtype:

float32

Sigmoid applied:

NO

Threshold applied:

NO

## Target lock

ACTIVE

Calibration data accessed:

NO

MedSeg / target accessed:

NO

## Figure font policy

Preferred:

Times New Roman

Fallback:

Calibri -> Arial Narrow -> Arial -> Liberation Sans -> DejaVu Sans

Labels must remain present.

## Next

SAVE A KAGGLE VERSION WITH OUTPUTS BEFORE CONTINUING
BEYOND OPTIMIZER STEP 1000.
