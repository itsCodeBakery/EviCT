# EviCT Execution State

## Project
EviCT — Uncertainty-Aware Vision-Language Segmentation and Evidence-Grounded Lung CT Reporting

## Current stage
BOOTSTRAP_COMPLETE

## Timestamp
2026-09-27T04:36:49.724149+00:00

## GitHub
Repository: https://github.com/itsCodeBakery/EviCT.git
Branch: main

## Kaggle project root
/kaggle/working/EviCT

## Attached dataset
COVID-19 CT scans

Kaggle path:
/kaggle/input/datasets/andrewmvd/covid19-ct-scans

Bootstrap verification:
- dataset root exists: YES
- ct_scans exists: YES
- infection_mask exists: YES
- lung_mask exists: YES
- lung_and_infection_mask exists: YES
- metadata.csv exists: YES

## Scientific status
No preprocessing performed.
No dataset provenance assumptions made.
No label mapping assumed.
No train/validation/test split created.
No model trained.
No target evaluation performed.
No research result produced.

## Next stage
Notebook 00 — Environment and Resource Audit

Next requirements:
1. Record GPU and VRAM.
2. Record CPU/RAM/disk.
3. Record Python/CUDA/PyTorch/package versions.
4. Test FP16 forward/backward execution.
5. Save environment.json.
6. Save requirements-lock.txt.
7. Update STATE.md.

## Recovery rule
Clone the GitHub repository into:

/kaggle/working/EviCT

Then attach the required Kaggle datasets and execute the current stage.
