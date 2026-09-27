# EviCT Execution State

## Current stage

NOTEBOOK_01_DATASET_POLICY_FROZEN

## Timestamp

2026-09-27T05:39:43.712896+00:00

## Completed stages

- Bootstrap: COMPLETE
- Notebook 00: COMPLETE
- Notebook 01A SegDB-2 audit: COMPLETE
- Notebook 01B SegDB-1 / MedSeg audit: COMPLETE
- Notebook 01C dataset-use policy: FROZEN

## Frozen controlled direction

SegDB-2 -> SegDB-1 / MedSeg

## Source dataset

SegDB-2

Verified volumes:

20

Future source partition:

- fitting: 12
- selection: 4
- calibration: 4

## External target

SegDB-1 / MedSeg

100 slices.

Patient grouping unresolved.

Source-training eligibility:

BLOCKED

External evaluation:

Allowed only after later source-side protocol freeze and must be
reported explicitly as image-level / selected-slice evaluation.

## Reverse direction

MedSeg -> SegDB-2:

BLOCKED until legitimate patient/group mapping is recovered.

## Radiopaedia competition arrays

QUARANTINED

## Target leakage lock

ACTIVE

No target performance may be used for model selection, threshold
selection, calibration, prompts, uncertainty rules, augmentation,
post-processing, or hyperparameter tuning.

## Dataset policy

Path:

config/dataset_use_policy.json

SHA-256:

d23e3bd4941d3884be354ca319fddf086c2153d54198489ddc67e7387227362f

## Scientific status

No preprocessing performed.

No source split created yet.

No model trained.

No target evaluation performed.

## Next stage

Notebook 02 — create fixed source-only SegDB-2 12/4/4 split and nested label budgets.
