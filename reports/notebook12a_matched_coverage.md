# EViCT Notebook 12A — Matched-Coverage Reliability Analysis

## Status

**PASS**

## Purpose

Notebook 12A evaluates whether the frozen case-uncertainty score is useful for
selective reporting on the external MedSeg target.

The source p80 threshold is not modified.

Instead, every method is compared at exactly the same retained coverage.

## Protocol

Each seed/method contains 100 external-target images.

Cases are ranked from lowest to highest frozen case uncertainty.

Matched coverage is evaluated from 5% to 100% in 5-percentage-point steps.

No Dice, IoU, lesion-extent error, or other target-performance metric is used
to rank cases.

Methods may retain different image identities at the same coverage.

## Selective risk

- Agreement EMA: selective-risk AUC 0.3174 ± 0.0384; risk reduction vs full-set baseline +0.0064
- Confidence EMA: selective-risk AUC 0.3087 ± 0.0305; risk reduction vs full-set baseline +0.0201
- Supervised: selective-risk AUC 0.3355 ± 0.0369; risk reduction vs full-set baseline +0.0189

Positive risk reduction relative to the full target set indicates that
low-uncertainty ranking concentrates lower-error cases on average.

Negative values indicate that uncertainty ranking does not provide the desired
selective-risk behavior.

## Uncertainty–error association

- Agreement EMA: mean uncertainty–segmentation-error rho +0.149; uncertainty–extent-error rho +0.425
- Confidence EMA: mean uncertainty–segmentation-error rho +0.239; uncertainty–extent-error rho +0.398
- Supervised: mean uncertainty–segmentation-error rho +0.173; uncertainty–extent-error rho +0.463

A positive Spearman coefficient indicates that larger uncertainty tends to
correspond to larger error.

## Interpretation boundary

This is a post-hoc reliability diagnostic.

It does not tune or replace the frozen source-derived p80 gate.

Matched coverage means equal retained case counts, not necessarily identical
retained target cases.

The MedSeg unit remains a selected 2D image rather than a verified independent
patient-level cohort.

## Software correction

The initial Notebook-12A execution completed the scientific calculations but
stopped while serializing the audit because NumPy integer scalar objects are not
directly supported by Python's JSON encoder.

The audit was repaired by conversion to native Python scalar types only.

No scientific value, prediction, ranking, model, threshold, or analysis result
was changed.

## Next stage

Notebook 12B — robustness / perturbation stress testing.
