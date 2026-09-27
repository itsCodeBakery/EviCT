# EviCT Execution State

## Current stage

NOTEBOOK_03A_PREPROCESSING_SMOKE_COMPLETE

## Timestamp

2026-09-27T06:14:12.085969+00:00

## Completed stages

- Notebook 00: COMPLETE
- Notebook 01: COMPLETE
- Notebook 02: COMPLETE
- Notebook 03A automated preprocessing smoke: PASS
- Notebook 03A manual visual QC: PASS

## Manual visual QC

Reviewed:

- coronacases_003
- radiopaedia_10_85902_1

Image-mask alignment:

PASS

Orientation:

PASS

Aspect-ratio preservation:

PASS

Inverse-transform behavior:

PASS

## Valid-pixel mask note

Both reviewed images are square and required no padding after
aspect-ratio-preserving resize to 336x336.

Their valid-pixel masks are therefore all ones.

The black appearance in the QC gallery is a Matplotlib display
normalization artifact for a constant-valued array, not an invalid
mask.

Future valid-mask figures must use:

vmin=0, vmax=1

## Target lock

ACTIVE

MedSeg was not loaded or evaluated.

## Next

Notebook 03B — build complete deterministic SegDB-2 source image
cache, geometry records, slice manifest, restricted label caches,
and expanded source-only QC.
