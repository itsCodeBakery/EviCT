# EViCT — Verified Evidence-Grounded Reporting

Run ID: `verified_reporting_20261015_215353`

## Experimental status

The EViCT segmentation and evidence-grounded reporting
experiments were completed previously.

This run recovers and verifies those results.

No model retraining or new Qwen report generation was performed.

## Verified reporting performance

| Method | Records | Direct acceptance | Safe delivery |
|---|---:|---:|---:|
| Deterministic | 900 | 100.00% | 100.00% |
| Strict sentence-ID | 900 | 99.56% | 100.00% |
| Controlled free text | 900 | 99.56% | 100.00% |

## Correct lesion definition

- Channel 0: ground-glass opacity
- Channel 1: consolidation
- Channel 2: lungs other
- Channel 3: background

Binary EViCT target = channel 0 OR channel 1.

## Supported contributions

- Segmentation-derived evidence records
- Prediction-derived lesion extent
- Numerical-integrity protection
- Structured sentence-ID reporting
- Controlled language generation
- Safe fallback mechanisms
- Uncertainty-based selective reporting analysis

## Scientific boundaries

- No independent clinical reader study
- No verified diagnostic accuracy
- No physical lesion area or volume
- No validated laterality or lobar localization
- No autonomous etiological diagnosis
- Target percentile analyses must not be confused
  with source-calibrated external reliability

## Provenance

The original frozen experiment remains authoritative.

All quantitative claims should be checked against
the repository's final result source map.
