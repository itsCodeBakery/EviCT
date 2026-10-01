# NOTEBOOK 12B0 HANDOFF

Status:
PASS

Protocol:
config/notebook12b_stress_protocol.json

Protocol SHA256:
65a29ad93ff403bcdf299059e1243b79528a1340a3683aaa9471d7598d29ac9d

Perturbations:
- clean
- gaussian_noise_sigma_0p03
- gaussian_blur_sigma_1p0
- resolution_half

Methods:
['supervised', 'confidence_only_ema', 'agreement_filtered_ema']

Seeds:
[17, 42, 2026]

Inference:
single frozen primary view

Expected total inference passes:
3600

Stress-only passes:
2700

Local checkpoints ready:
0/9

Missing/invalid:
9/9

Important:
1. Restore only frozen selected_best checkpoints.
2. Verify SHA256 after restoration.
3. Clean inference must pass reproduction gate first.
4. Do not alter temperature or segmentation threshold.
5. Stress tests are synthetic, not clinical acquisition robustness.

Next:
Notebook 12B1.
