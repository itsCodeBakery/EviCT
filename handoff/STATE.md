# EviCT Execution State

## Current stage

NOTEBOOK_01B_SEGDB1_AUDIT_COMPLETE

## Timestamp

2026-09-27T05:35:03.059640+00:00

## Completed stages

- Bootstrap: COMPLETE
- Notebook 00 environment audit: COMPLETE
- Notebook 01A SegDB-2 audit: COMPLETE
- Notebook 01B SegDB-1 / MedSeg audit: COMPLETE

## SegDB-2

20/20 CT-mask pairs verified.

20/20 geometry-valid.

Eligible as controlled source dataset.

## SegDB-1 / MedSeg

Training slices:

100

Mask channels:

4

Observed values:

{'0': [0.0, 1.0], '1': [0.0, 1.0], '2': [0.0, 1.0], '3': [0.0, 1.0]}

Core EviCT target:

GGO channel 0 OR consolidation channel 1

Patient grouping recovered:

False

Eligible as source-training dataset:

False

Reason:

The Kaggle arrays do not provide verified patient mapping for the
100 slices. No random slice split will be used.

Eligible as image-level external target:

True

## Direction policy

SegDB-2 -> MedSeg:

Allowed later as explicitly image-level external evaluation after
protocol freeze.

MedSeg -> SegDB-2:

Blocked until legitimate MedSeg patient/group mapping is recovered.

## Radiopaedia competition arrays

QUARANTINED because of potential provenance overlap with SegDB-2.

## Scientific status

No preprocessing performed.

No split created yet.

No model trained.

No target evaluation performed.

Track R remains NOT READY.

## Next

Freeze Notebook 01 dataset-use policy, then proceed to source-only SegDB-2 split/budget design.
