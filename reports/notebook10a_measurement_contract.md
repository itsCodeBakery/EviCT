# EViCT Notebook 10A — Measurement and Evidence Contract

## Status

**PASS — contract frozen**

Notebook 10A defines which measurements may and may not be emitted before
constructing any evidence records.

## External target scope

The MedSeg target used in the locked Notebook-09 evaluation consists of 100
selected 2D CT images.

The core evaluation does not establish:

- verified patient grouping;
- complete-volume geometry;
- verified physical pixel spacing;
- verified slice spacing;
- orientation sufficient for laterality;
- a deployable lung-mask denominator.

Therefore the core evidence layer may report predicted lesion extent in pixels
on the original 512 × 512 image grid, but it may not report physical lesion area,
lesion volume, whole-lung burden, patient-level burden, laterality, or lobar
location.

## Uncertainty eligibility

The predeclared research plan uses the 80th percentile of source calibration
uncertainty as the default case eligibility gate and requests the
50/70/80/90/100 percentiles for coverage analysis.

Notebook 08 preserved four source calibration uncertainty scores per seed/method.
The frozen protocol stored p50/p75/p90/p95.

Notebook 10A first reproduced all frozen p50/p75/p90/p95 values using NumPy's
linear percentile rule.

Maximum absolute reproduction error:

`5.551e-17`

The predeclared p70/p80/p100 values were then recovered from the same frozen
source-only scores. No target performance, target label, or target metric was
used.

## Default eligibility rule

A numerical evidence field that requires uncertainty gating is eligible when:

`case_uncertainty_score <= method-and-seed source p80`

This is a source acceptance target, not an external accuracy guarantee.

## Ground-truth separation

The evidence constructor is prediction-only.

Target ground-truth masks and target metric tables are prohibited from the
evidence constructor.

Ground truth may be accessed only afterward by a separate evaluation function.

## Hashes

Measurement contract SHA256:

`f76d556c5a95966448ecc47df01ebfaf6c78727476ef24c1486aa4e05304eb01`

Evidence schema SHA256:

`cb2b2299b4fd3d84e64b95d2282f76739a804ccff4a003579094fc5b2f0c1f38`
