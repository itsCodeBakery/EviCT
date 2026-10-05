# EViCT — vlmDiagnosis

This directory contains the isolated diagnostic extension of EViCT.

## Core policy

The original EViCT research study is frozen and read-only.

Frozen core commit: `a717ae6e51b1dd51d6040d063fb3ebbebcc7b751`

All new diagnostic work must remain under `vlmDiagnosis/`.

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

STEP 01 — Core isolation and lock.

No EViCT-Dx training has been performed yet.

Next: STEP 02 — Dataset and diagnostic target protocol freeze.
