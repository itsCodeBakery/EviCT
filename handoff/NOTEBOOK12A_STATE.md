# NOTEBOOK 12A HANDOFF

Status: PASS

Analysis:
Matched external-target coverage using frozen uncertainty ranking.

Coverage grid:
[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

Cases per seed/method:
100

Ranking:
ascending case_uncertainty_score
tie break = case_id

Target metric used for ranking:
NO

Source-p80 gate changed:
NO

Model/inference rerun:
NO

Serialization correction:
JSON-safe conversion of NumPy scalar types only.

Main files:
- tables/notebook12a_matched_coverage_per_seed.csv
- tables/notebook12a_matched_coverage_method_summary.csv
- tables/notebook12a_key_coverage_summary.csv
- tables/notebook12a_uncertainty_error_correlation.csv
- tables/notebook12a_selective_risk_auc.csv
- artifacts/audit/notebook12a_matched_coverage_audit.json

Figures:
- matched_coverage_dice
- matched_coverage_segmentation_risk
- matched_coverage_extent_mae
- matched_coverage_risk_reduction

Important:
Matched coverage means the same retained COUNT, not necessarily the same cases.

Next:
Notebook 12B — robustness / perturbation stress tests.
