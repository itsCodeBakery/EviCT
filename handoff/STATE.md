# EviCT Execution State

## Current stage

NOTEBOOK_04B_MODEL_SMOKE_PASS_TINY_OVERFIT_PENDING

## Timestamp

2026-09-27T07:31:41.612509+00:00

## Completed stages

- Notebook 00: COMPLETE
- Notebook 01: COMPLETE
- Notebook 02: COMPLETE
- Notebook 03 preprocessing/geometry: FROZEN
- Notebook 04A segmentation metrics: PASS
- Notebook 04B MiT-B1 forward/backward smoke: PASS

## Supervised visual baseline

Encoder:

MiT-B1

Checkpoint repository:

nvidia/mit-b1

Pinned revision:

13ddceec4e8bdf401e7cd7acf5aebc526222518c

Checkpoint SHA-256:

980b86b60db37b1b1528086f6c53d253d879e9e09ebe07397b40fd275738a3bf

Initialization:

ImageNet-1k encoder only

Encoder hidden sizes:

[64, 128, 320, 512]

Decoder:

SegFormer four-level MLP decoder

Decoder channels:

256

Decoder resolution for 336x336 input:

84x84

Visual head:

1x1 binary lesion-logit convolution

Final output:

336x336

## Frozen cache contract used

Image cache:

[Z,336,336]

Infection-mask cache:

[Z,336,336]

Valid-pixel mask:

[336,336]

Valid-pixel mask is case-level and shared across slices.

## Smoke batch

Real SegDB-2 fitting slices only.

Samples:

[
  {
    "case_id": "coronacases_001",
    "provenance": "coronacases_named",
    "z_index": 118,
    "lesion_pixels": 6004,
    "valid_pixels": 112896,
    "valid_fraction": 1.0,
    "image_min": 0.1527099609375,
    "image_max": 1.0
  },
  {
    "case_id": "radiopaedia_14_85914_0",
    "provenance": "radiopaedia_named",
    "z_index": 43,
    "lesion_pixels": 7151,
    "valid_pixels": 71904,
    "valid_fraction": 0.636904776096344,
    "image_min": 0.0,
    "image_max": 1.0
  }
]

## Supervised objective

0.5 soft Dice + 0.5 BCE

Padding:

Excluded through valid-pixel masks.

## Forward/backward result

Total loss:

0.84930503

Soft Dice loss:

0.85314131

BCE loss:

0.84546876

Encoder gradients finite/non-zero:

True / True

Decoder gradients finite/non-zero:

True / True

Visual-head gradients finite/non-zero:

True / True

Optimizer update verified:

True

## GPU

Tesla T4

FP16 smoke:

PASS

Peak allocated GPU memory:

0.438 GiB

Peak reserved GPU memory:

0.547 GiB

## Target lock

ACTIVE

No MedSeg images, masks, metrics or prompts accessed.

## Figure standard

Times New Roman
Bold readable labels
600-dpi PNG
Vector PDF
No captions embedded in figures

## Next

Notebook 04C — deliberate tiny-set overfit.

Use 2-4 visible source-fitting images, including a small lesion.

PASS requires:

- finite optimization;
- strong deliberate memorization of the tiny training set;
- prediction-mask alignment;
- empty-mask metric sanity;
- no target access.

Only after Notebook 04C passes may the full supervised baseline begin.
