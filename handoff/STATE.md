# EviCT Execution State

## Current stage

NOTEBOOK_03A_AUTOMATED_PASS_MANUAL_QC_PENDING

## Timestamp

2026-09-27T06:01:48.114811+00:00

## Completed

- Notebook 00: COMPLETE
- Notebook 01: COMPLETE
- Notebook 02: COMPLETE
- Notebook 03A automated preprocessing smoke: PASS

## Frozen source preprocessing

CoronaCases:

[-1250,250] benchmark lung-window protocol -> [0,1]

Radiopaedia:

documented windowed display [0,255] -> [0,1]

Geometry:

aspect-ratio preserving resize and symmetric padding to 336x336

Images:

bilinear interpolation

Masks:

nearest-neighbor interpolation

Padding:

excluded using explicit valid-pixel masks

## Automated geometry QC

Synthetic roundtrip IoU:

0.9845668186601193

Source-selection QC cases:

[{'case_id': 'coronacases_003', 'provenance_stratum': 'coronacases_named', 'roundtrip_mask_iou': 0.9246123607774623}, {'case_id': 'radiopaedia_10_85902_1', 'provenance_stratum': 'radiopaedia_named', 'roundtrip_mask_iou': 0.9374563730280608}]

## Target lock

ACTIVE

MedSeg target was not loaded.

## Manual visual QC

PENDING

Inspect:

figures/preprocessing_smoke_gallery.png

Confirm:

1. CT orientation looks anatomically plausible.
2. Infection masks visibly align with opacities.
3. No transpose/rotation mismatch is visible.
4. 336x336 resize preserves aspect ratio.
5. Padding appears only outside the resized image.

## Preprocessing config SHA256

51a6259ebe818087309abfc8600996281b972ec23618e0051e8c91a66ed684fe

## Preprocessing module SHA256

9316263daa60fe00e0d58b624da646554ad7564820ffec0b7934d4ece6f0c017

## Next

After manual visual QC is confirmed, build the full deterministic
SegDB-2 source cache and slices.csv.
