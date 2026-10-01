# EViCT Notebook 12B0 — Synthetic Stress-Test Protocol Freeze

## Status

**PASS**

## Frozen perturbations

1. Clean identity control
2. Gaussian noise, sigma = 0.03
3. Gaussian blur, sigma = 1.0, kernel = 5
4. Half-resolution degradation, 512→256→512

Perturbations are applied after the locked CT normalization and before the
existing 336×336 model preprocessing.

## Scope

- Methods: supervised, confidence-only EMA, agreement-filtered EMA
- Training seeds: 17, 42, 2026
- Prediction mode: frozen primary single-view inference
- Target cases: complete 100-image MedSeg target
- Scientific thresholds: unchanged
- Temperatures: unchanged
- Postprocessing: unchanged
- Retraining: none

## Required clean reproduction

Stress results are accepted only if a clean rerun reproduces the preserved
Notebook-09 corrected predictions within the frozen tolerance.

## Stratification

Lesion burden will be reported as:

- empty
- small
- medium
- large

using tertiles of nonempty reference lesion pixels.

No within-MedSeg acquisition-provenance subgroup is fabricated because verified
provenance strata are unavailable.

## Claim boundary

These results are synthetic stress tests only.

They must not be described as proof of robustness to real clinical acquisition
variation.

## Local checkpoint status

Ready:

**0 / 9**

Missing or invalid:

**9 / 9**

Notebook 12B1 may restore missing files from the pre-existing Notebook-07
GitHub release archives, but every restored checkpoint must exactly match its
Notebook-08 frozen SHA256.

## Next

Notebook 12B1 — clean reproduction gate + locked synthetic stress inference.
