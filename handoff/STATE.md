# EviCT Execution State

## Current stage

NOTEBOOK_05E_SEGCT_CLIP_VISUAL_SEED_17_RUNNING_STEP_4000

## Timestamp

2026-09-29T06:18:07.369399+00:00

## Baseline

SegCT-CLIP visual-pathway adaptation

Claim status:

EXPLICIT ADAPTATION — NOT EXACT SEGCT-CLIP REPRODUCTION

Backbone:

Frozen CLIP ViT-L/14-336

Caption bank:

NOT USED — exact author caption bank unavailable

Contrastive loss:

NOT USED — exact caption supervision unavailable

Visual design:

Dual-level frozen CLIP patch features + explicit EviCT lightweight decoder adaptation

Training seed:

17

Split seed:

17

Optimizer step:

4000

Best source-selection macro case Dice:

0.73094277

Best checkpoint step:

2000

Patience:

8 / 8

Image exposures:

64000

Selection threshold:

0.5 fixed

## Isolation

Calibration accessed:

NO

Target / MedSeg accessed:

NO

Target lock:

ACTIVE

## Next

Complete seeds 17, 42 and 2026 under the same frozen adaptation protocol,
aggregate the source-selection results, then proceed to Notebook 06.
