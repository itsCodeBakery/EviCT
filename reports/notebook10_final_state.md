# EViCT Notebook 10 — Final State

## Status

**COMPLETE**

Notebook 10A, 10B, 10C, and 10D passed their prespecified checks.

## Evidence integrity

- Prediction-derived evidence records: **900 / 900**
- Independently verified records: **900 / 900**
- Lesion-extent fidelity mismatches: **0**
- Uncertainty-copy mismatches: **0**
- p80 eligibility mismatches: **0**

The evidence constructor did not access target ground truth.

Ground truth entered only in the independent Notebook-10C evaluator.

## Default source-p80 gate on MedSeg

Three-seed mean external coverage:

- Supervised: **5.00%**
- Confidence EMA: **3.67%**
- Agreement EMA: **4.00%**

Three-seed mean retained-set Dice:

- Supervised: **0.675**
- Confidence EMA: **0.740**
- Agreement EMA: **0.658**

These retained-set Dice values are descriptive. The retained target cases differ
across methods and seeds, so they must not be presented as a matched ranking.

## Main reliability finding

The source-derived p80 uncertainty threshold is highly conservative under
external-domain shift. Instead of retaining approximately 80% of MedSeg cases,
the observed mean coverage is only about 4–5%.

This does not invalidate the frozen uncertainty protocol. It demonstrates that
a source acceptance target does not imply equivalent coverage after domain
shift.

The gate was not changed after observing MedSeg.

## Measurement scope

Supported:

- predicted target-opacity presence;
- lesion extent in pixels;
- lesion image fraction;
- uncertainty;
- external eligibility coverage.

Unsupported:

- physical lesion area;
- lesion volume;
- whole-lung involvement;
- patient-level burden;
- laterality;
- lobar localization.

## Four-view stability limitation

Individual aligned four-view predictions were not preserved in Notebook 09.

True lesion-extent stability ranges available:

**0 / 900**

No stability range was fabricated.

## Next stage

**Notebook 11 — Structured Evidence-Grounded Reporting**

The validated prediction-only evidence JSON records now become the sole
numerical input to the strict reporting pipeline.
