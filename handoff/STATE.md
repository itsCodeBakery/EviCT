# EviCT Execution State

## Current stage

NOTEBOOK_02_SPLITS_AND_BUDGETS_FROZEN

## Timestamp

2026-09-27T05:47:11.175766+00:00

## Completed stages

- Bootstrap: COMPLETE
- Notebook 00 environment audit: COMPLETE
- Notebook 01 source audits: COMPLETE
- Notebook 01 dataset-use policy: FROZEN
- Notebook 02 source split and budgets: FROZEN

## Controlled direction

SegDB-2 -> SegDB-1 / MedSeg

## Frozen SegDB-2 source split

Seed:

17

Fitting:

12

Selection:

4

Calibration:

4

## Provenance balance

Fitting:

6 CoronaCases + 6 Radiopaedia

Selection:

2 CoronaCases + 2 Radiopaedia

Calibration:

2 CoronaCases + 2 Radiopaedia

## Nested training-label budgets

Seeds:

17, 42, 2026

25%:

3 / 12 fitting masks visible

Total source annotation access:

11 / 20

50%:

6 / 12 fitting masks visible

Total source annotation access:

14 / 20

100%:

12 / 12 fitting masks visible

Total source annotation access:

20 / 20

## Leakage controls

Unlabeled fitting manifests contain no mask paths.

Unlabeled fitting manifests contain no derived label statistics.

Selection and calibration masks remain separate source-only
supervision pools.

MedSeg target remains locked.

No target performance has been inspected.

## Core artifacts

- manifests/splits.csv
- manifests/budgets.csv
- manifests/annotation_ledger.csv
- manifests/source_selection_cases.csv
- manifests/source_calibration_cases.csv
- manifests/loaders/
- config/split_protocol.json
- config/split_manifest_hashes.json

## Dataset policy SHA256

d23e3bd4941d3884be354ca319fddf086c2153d54198489ddc67e7387227362f

## Split manifest SHA256

a3abd2a275f21d2ec3ad5395abbd5326705ddd5ac62b9a52cd3acd8687dc039f

## Budget manifest SHA256

b0b1eb002986a0b70a25f8b58e68e19d51524339a3d94af12983497ba4a4d11d

## Annotation ledger SHA256

8e154b38de870fad7e90258cddfe370ecf31b0ffee9dd0812e99a7811f6bee6d

## Hash-record SHA256

dbf3b206634f3c21326193f9b78a7039d39c47958184ca8c91c9e1fbbd6f0cba

## Scientific status

No preprocessing performed.

No image resized.

No model trained.

No source-selection result inspected.

No target evaluation performed.

## Next stage

Notebook 03 — provenance-specific preprocessing, geometry transforms, caches, slice manifests and visual audit.
