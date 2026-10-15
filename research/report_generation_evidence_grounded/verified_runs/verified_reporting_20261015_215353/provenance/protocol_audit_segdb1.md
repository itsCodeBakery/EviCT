# EviCT Notebook 01B — SegDB-1 / MedSeg Audit

Timestamp: 2026-09-27T05:35:03.058820+00:00

## Execution source

Kaggle competition:

`covid-segmentation`

Mounted at:

`/kaggle/input/competitions/covid-segmentation`

## Verified arrays

Training images:

`(100, 512, 512, 1)`

Training masks:

`(100, 512, 512, 4)`

Competition test images:

`(10, 512, 512, 1)`

Radiopaedia images:

`(829, 512, 512, 1)`

Radiopaedia masks:

`(829, 512, 512, 4)`

All five files have SHA-256 hashes recorded in
`manifests/segdb1_source_inventory.csv`.

## Kaggle MedSeg mask representation

Channel 0: ground glass

Channel 1: consolidation

Channel 2: lungs other

Channel 3: background

Observed values:

`{'0': [0.0, 1.0], '1': [0.0, 1.0], '2': [0.0, 1.0], '3': [0.0, 1.0]}`

All channels binary:

**True**

All mask values finite:

**True**

## EviCT core target

The controlled EviCT binary target is:

**ground glass OR consolidation**

That is:

**channel 0 OR channel 1**

Channel 2 is excluded from the core lesion target.

## Original-source terminology warning

The original MedSeg Dataset 1 description documents three radiologist
labels:

1. ground-glass
2. consolidation
3. pleural effusion

The Kaggle competition representation instead describes its third
foreground-style channel as `lungs other`.

EviCT does not automatically claim that Kaggle channel 2 is equivalent
to the original pleural-effusion label.

## Patient grouping

The 100 slices are documented as originating from more than 40 patients.

The Kaggle `.npy` representation does not provide a verified per-slice
patient identifier in the attached files.

No patient ID has been invented.

No grouping has been inferred from adjacent array indices.

Therefore:

### MedSeg -> SegDB-2 training direction

**BLOCKED**

until valid MedSeg patient/group mapping is recovered.

### SegDB-2 -> MedSeg evaluation direction

**PERMITTED ONLY AS EXPLICIT IMAGE-LEVEL / SELECTED-SLICE EXTERNAL
EVALUATION**

after the model and evaluation protocol have been frozen using source
data.

It must not be described as patient-level volumetric evaluation.

## Kaggle Radiopaedia arrays

The competition also provides Radiopaedia-derived arrays.

These are currently:

**QUARANTINED**

because SegDB-2 already contains Radiopaedia-derived CT volumes and
there is a material risk of provenance overlap.

They are not included in the MedSeg training source.

## Core scientific status

No image preprocessing performed.

No resizing performed.

No random train/validation split created.

No model trained.

No threshold selected.

No target results inspected.

