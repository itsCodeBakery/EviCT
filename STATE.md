# EviCT Execution State

## Current stage

NOTEBOOK_03_FROZEN_AWAITING_KAGGLE_SAVE_VERSION

## Timestamp

2026-09-27T07:05:22.750242+00:00

## Completed stages

- Notebook 00: COMPLETE
- Notebook 01: COMPLETE
- Notebook 02: COMPLETE
- Notebook 03A preprocessing smoke/manual QC: COMPLETE
- Notebook 03B full source cache: COMPLETE
- Notebook 03C source intensity QC: COMPLETE
- Notebook 03 preprocessing/geometry: FROZEN

## Source cache

SegDB-2 volumes:

20

Source slices:

3520

Cache size:

1.48 GiB

Verified cache/transform artifacts:

100

## Cache recovery archive

Path:

/kaggle/working/EviCT_Notebook03_Cache.tar

SHA-256:

d88e786cd467816d6cb446333385918016946a83cdd3c258b3011f458fbd3fc4

Checksum file:

/kaggle/working/EviCT_Notebook03_Cache.sha256

## Figure standard

Times New Roman

Bold readable labels

Minimum 600-dpi PNG

Vector PDF required

No caption embedded inside figure

No explanatory footnote embedded inside figure

Captions belong in LaTeX.

## Target lock

ACTIVE

MedSeg has not been used for performance-guided development.

## GitHub

All Git-trackable Notebook 03 metadata, manifests, configurations,
hashes, transforms, audit records and figures are synchronized.

Large .npy cache files remain intentionally outside ordinary Git.

## Required next action

SAVE A KAGGLE NOTEBOOK VERSION WITH OUTPUTS.

Verify that these files appear in the saved output:

- EviCT_Notebook03_Cache.tar
- EviCT_Notebook03_Cache.sha256

## After durable Kaggle save

Proceed to Notebook 04:

1. implement and unit-test metrics;
2. build SegFormer MiT-B1 supervised baseline;
3. forward/backward smoke test;
4. overfit 2-4 fitting images;
5. begin the full-label source baseline only after the smoke tests pass.
