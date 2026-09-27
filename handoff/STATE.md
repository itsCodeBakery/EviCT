# EviCT Execution State

## Current stage

NOTEBOOK_03B_AUTOMATED_PASS_EXPANDED_VISUAL_QC_PENDING

## Timestamp

2026-09-27T06:24:31.135133+00:00

## Completed stages

- Notebook 00: COMPLETE
- Notebook 01: COMPLETE
- Notebook 02: COMPLETE
- Notebook 03A preprocessing smoke/manual QC: COMPLETE
- Notebook 03B full SegDB-2 source cache: AUTOMATED PASS

## Full source cache

Cases:

20

Slices:

3520

Image cache:

float16 [0,1], Zx336x336

Infection GT vault:

uint8 values 0/1

Lung GT vault:

uint8 values 0/1/2

## Label security

The common slices.csv contains no lesion-mask paths or lesion-derived
statistics.

Fitting unlabeled slice manifests contain no GT-vault paths.

All hidden-case slices are retained without lesion-based filtering.

Source selection and calibration masks are available only through
their explicit held-out manifests.

## Geometry

Aspect-ratio preserving resize/pad:

336x336

Image interpolation:

bilinear

Label interpolation:

nearest

Real non-square source cases exercise the valid-pixel mask.

Every case has a Git-tracked reversible transform JSON.

## Target lock

ACTIVE

MedSeg target was not opened.

## Cache durability

Large cache .npy files are intentionally excluded from ordinary Git.

Their SHA-256 hashes and regeneration metadata are committed.

A Kaggle Save Version/output checkpoint is required after final visual
QC to make these generated cache files durable outside the live runtime.

## Visual QC pending

Inspect:

figures/source_preprocessing_qc_gallery.png

and:

figures/non_square_padding_qc.png

The 20-panel overlay gallery contains ten source-heldout slices from
each provenance stratum.

## Next

After expanded visual QC passes, freeze Notebook 03 and preserve the
generated cache with a Kaggle notebook version before Notebook 04.
