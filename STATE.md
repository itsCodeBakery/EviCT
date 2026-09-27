# EviCT Execution State

## Project

EviCT — Uncertainty-Aware Vision-Language Segmentation and Evidence-Grounded Lung CT Reporting

## Current stage

NOTEBOOK_01A_BLOCKED

## Timestamp

2026-09-27T04:48:37.077232+00:00

## Repository

https://github.com/itsCodeBakery/EviCT

Branch: main

---

## Completed stages

- Bootstrap: COMPLETE
- Notebook 00 environment/resource audit: COMPLETE
- Notebook 01A SegDB-2 source audit: BLOCKED

---

## SegDB-2 execution copy

Path:

`/kaggle/input/datasets/andrewmvd/covid19-ct-scans`

Case identifiers found:

40

Complete CT/mask sets:

0/40

Header-level geometry valid:

0/40

Exact duplicate records:

0

Observed infection-mask labels:

[]

Observed lung-mask labels:

[]

Observed combined-mask labels:

[]

---

## Base paper repository

URL:

https://github.com/Owais-CodeHub/CT-Insight-VLM.git

Reachable during audit:

False

Matched reproduction status:

NOT READY

See `protocol_audit.md`.

---

## Generated artifacts

- manifests/segdb2_source_inventory.csv
- manifests/segdb2_file_pairing.csv
- manifests/segdb2_exact_duplicates.csv
- manifests/segdb2_metadata_summary.json
- manifests/segdb2_audit_summary.json
- manifests/cases.csv
- manifests/exclusions.csv
- config/source_registry_segdb2.json
- config/label_mapping.json
- config/ct_insight_repository_status.json
- artifacts/audit/segdb2_metadata_preview.csv
- protocol_audit.md
- STATE.md
- handoff/STATE.md

---

## Scientific status

No preprocessing performed.

No HU assumption made from file format alone.

No images resized.

No train/selection/calibration split created.

No model trained.

No target dataset evaluated.

No published result has been entered as an executed result.

No claim of improvement has been made.

---

## Next stage

Resolve failed SegDB-2 source-audit checks.

