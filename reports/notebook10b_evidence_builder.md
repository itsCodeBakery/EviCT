# EViCT Notebook 10B — Prediction-Only Evidence Builder

## Status

**PASS**

## Records

- Corrected Notebook-09 predictions: 900
- Evidence records generated: 900
- Unique MedSeg images: 100
- Seeds: 17, 42, 2026
- Methods: supervised, confidence-only EMA, agreement-filtered EMA

## Leakage control

The evidence constructor did **not** access:

- `masks_medseg.npy`
- `notebook09_corrected_target_case_metrics.csv`
- target Dice
- target IoU
- target ground truth

Eligibility was determined only from the source-derived Notebook-10A p80 gates.

## Supported evidence

- predicted target-opacity presence;
- predicted lesion extent in pixels on the original 512 × 512 image grid;
- case uncertainty;
- source-derived p80 eligibility;
- aggregate four-view lesion extent when preserved;
- true four-view stability range only when individual aligned view outputs are preserved.

## Unsupported evidence

- physical lesion area in mm²;
- lesion volume;
- whole-lung burden;
- patient-level burden;
- laterality;
- lobar localization.

## Four-view stability

Records with aggregate four-view masks:

`900 / 900`

Records with enough preserved individual-view information to calculate the true four-view lesion-extent stability range:

`0 / 900`

If this number is zero, no stability interval was invented.

## Evidence manifest SHA256

`b4b5d53ad80f2a84198fe5588e0840406b1ee922c93e80cfef8ddc5f1149b9f9`

## Measurement contract SHA256

`f76d556c5a95966448ecc47df01ebfaf6c78727476ef24c1486aa4e05304eb01`

