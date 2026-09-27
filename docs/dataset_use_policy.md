# EviCT Frozen Dataset-Use Policy

## Status

**FROZEN**

Policy SHA-256:

`d23e3bd4941d3884be354ca319fddf086c2153d54198489ddc67e7387227362f`

## Controlled experiment

The currently permitted controlled direction is:

**SegDB-2 → SegDB-1 / MedSeg**

## SegDB-2

Role:

**SOURCE**

Verified volumes:

**20**

Future partition:

- fitting: 12
- selection: 4
- calibration: 4

The fixed partition will be created in Notebook 02.

Nested fitting-label budgets:

- 25% = 3 / 12 fitting volumes
- 50% = 6 / 12 fitting volumes
- 100% = 12 / 12 fitting volumes

Primary experiment seeds:

- 17
- 42
- 2026

## SegDB-1 / MedSeg

Role:

**EXTERNAL IMAGE-LEVEL TARGET**

Training slices:

**100**

Verified patient mapping:

**No**

Therefore MedSeg cannot currently be used as a source-training
dataset.

The reverse direction:

**MedSeg → SegDB-2**

is blocked.

The core lesion target is:

**ground-glass OR consolidation**

corresponding to Kaggle mask channels 0 and 1.

## Radiopaedia competition arrays

Status:

**QUARANTINED**

They must not enter training, validation, calibration, retrieval,
or evaluation until provenance overlap with SegDB-2 is explicitly
resolved.

## Target lock

MedSeg target performance must not influence:

- architecture choice
- hyperparameters
- checkpoint choice
- threshold choice
- calibration
- uncertainty threshold
- text prompt selection
- pseudo-label rules
- augmentation choices
- post-processing choices

Target evaluation becomes available only after the later
source-side protocol freeze.

## Track R

Matched CT-Insight reproduction:

**NOT READY**

Published CT-Insight results remain historical reference values
rather than EviCT rerun results.
