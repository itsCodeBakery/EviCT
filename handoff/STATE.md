# EviCT Execution State

## Project

EviCT — Uncertainty-Aware Vision-Language Segmentation and Evidence-Grounded Lung CT Reporting

## Current stage

NOTEBOOK_00_COMPLETE

## Timestamp

2026-09-27T04:40:03.656343+00:00

## Repository

https://github.com/itsCodeBakery/EviCT

Branch: main

## Kaggle project root

/kaggle/working/EviCT

## Dataset currently attached

COVID-19 CT scans

Path:

/kaggle/input/datasets/andrewmvd/covid19-ct-scans

## Notebook 00 — Environment audit

GPU:
GPU 0: Tesla T4, 14.562 GB; GPU 1: Tesla T4, 14.562 GB

CUDA available:
True

PyTorch:
2.10.0+cu128

PyTorch CUDA:
12.8

BF16 supported:
True

FP16 smoke test:
PASS

Working disk free:
19.502 GB

Internet available during audit:
True

## Validation checks

{
  "project_repository_exists": true,
  "dataset_exists": true,
  "working_directory_writable": true,
  "torch_imported": true,
  "environment_json_written": true,
  "requirements_lock_written": true,
  "gpu_fp16_smoke_test": true
}

## Outputs

- config/environment.json
- config/gpu_smoke_test.json
- requirements-lock.txt
- STATE.md
- handoff/STATE.md

## Scientific status

No image preprocessing performed.

No label mapping assumed.

No train/selection/calibration split created.

No segmentation model trained.

No model-selection decision made.

No target dataset evaluated.

No research result has been produced.

## Next stage

Notebook 01 — Source and Reproduction Audit

Notebook 01 must audit:

1. Original dataset provenance.
2. Exact case inventory.
3. CT/mask pairing.
4. NIfTI dimensions and geometry.
5. Label values.
6. Infection/lung mask semantics.
7. Intensity provenance.
8. Original-source correspondence.
9. Potential duplicate/overlap issues.
10. Track R versus Track C eligibility.

