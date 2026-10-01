# Notebook 12B0-R1

Status: PASS

The stress protocol was corrected before stress inference.

Removed endpoints:
['case_uncertainty_score', 'source_p80_coverage_change_vs_clean', 'source_p80_eligibility_coverage', 'uncertainty_change_vs_clean']

Reason:
The frozen uncertainty estimator requires four views, whereas the robustness
experiment uses primary single-view inference.

No surrogate uncertainty measure was introduced.

Original protocol SHA256:
65a29ad93ff403bcdf299059e1243b79528a1340a3683aaa9471d7598d29ac9d

Corrected protocol SHA256:
d88f46e025874c09884898f2af848ceaebb762f5833b15f4531e886dcc413f35
