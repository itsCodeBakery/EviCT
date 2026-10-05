# EViCT — vlmDiagnosis

This directory contains the isolated diagnostic extension of EViCT.

## Core policy

The original EViCT research study is frozen and read-only.

Frozen core commit: `a717ae6e51b1dd51d6040d063fb3ebbebcc7b751`

All new diagnostic work, recovery state, manifests, checkpoints, and execution metadata must remain under `vlmDiagnosis/`.

The extension may read frozen EViCT assets but may not overwrite them.

## Planned capabilities

- COVID-19 / CAP / Normal classification
- left/right lung segmentation
- lobe segmentation
- GGO segmentation
- consolidation segmentation
- pleural-effusion analysis
- left/right/total lung involvement quantification
- diagnostic uncertainty and abstention
- evidence-grounded VLM reporting
- radiologist-review application

## Validation rule

A report field is enabled only after quantitative validation.

## Current status

- STEP 01 — core lock: COMPLETE
- STEP 02 — dataset, target, split, metric and report-field protocol: FROZEN
- STEP 03A — dataset identity and availability audit: COMPLETE
- STEP 03B — 20-case COVID-19 CT segmentation dataset validation: COMPLETE
- STEP 03C — 20-case geometry, labels and five-fold split: FROZEN
- COVID-CT-MD archive: acquired in the current Kaggle workflow
- GDCM JPEG-lossless decoding: VERIFIED on sampled COVID-19, CAP and Normal DICOMs
- EViCT-Dx training: NOT YET STARTED

## Interruption-safe execution

Long-running EViCT-Dx work uses the isolated recovery system in:

- `vlmDiagnosis/runtime/recovery.py`
- `vlmDiagnosis/scripts/kaggle_recovery.py`
- `vlmDiagnosis/config/recovery_policy.json`
- `vlmDiagnosis/docs/KAGGLE_RECOVERY.md`

Run state is stored under `vlmDiagnosis/runs/<RUN_ID>/`. Large checkpoints are stored as rolling GitHub Release assets, while small state/manifests/logs stay under `vlmDiagnosis/`.

## Next

Complete the COVID-CT-MD DICOM inventory, geometry/intensity audit, and patient-level diagnosis manifest/split freeze before starting diagnosis training.
