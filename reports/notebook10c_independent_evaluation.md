# EViCT Notebook 10C — Independent Measurement and Eligibility Evaluation

## Status

**PASS**

## Separation from evidence construction

Notebook 10B constructed 900 evidence records without target ground truth.

Notebook 10C is a separate evaluation layer and accesses the locked MedSeg
reference masks only to evaluate the correctness of those prediction-derived
records.

No model, checkpoint, segmentation threshold, temperature, prediction, or
eligibility gate was changed.

## Evidence fidelity

- Evidence records checked: 900
- Pixel-extent mismatches: 0
- Uncertainty-copy mismatches: 0
- Source-p80 eligibility mismatches: 0

## Measurement evaluation

Supported external-target measurement evaluation:

- lesion extent in pixels;
- lesion image fraction;
- extent bias;
- extent absolute error;
- Dice and IoU for evaluation only.

Not evaluated as deployable measurements:

- physical area in mm²;
- lesion volume;
- whole-lung involvement;
- patient-level burden;
- laterality;
- lobar localization.

## Eligibility analysis

The external target was evaluated using the source-derived uncertainty
thresholds:

- p50
- p70
- p80
- p90
- p100

The default eligibility gate remains source p80.

These gates were not adjusted using target results.

## Four-view stability

Notebook 10B found:

`0 / 900`

records with the individual aligned four-view outputs needed to calculate a
genuine lesion-extent stability range.

Therefore no stability-range correctness result is reported.

No stability interval was fabricated.

## Target reference mask SHA256

`517aa4c81400152d7b9ea7375a8603ff3544be2dd9281d069dd8fab7c674fb9d`

## Measurement table SHA256

`20918c999bb5b26451e6393af6346ad399ba73599ca7f24fdc20abd58f0c5f79`

## Source-gate evaluation SHA256

`98e794800db9cc66f8f6047dfc2d802f45d5b71a9e6d88e93b02a5add21b2cc7`
