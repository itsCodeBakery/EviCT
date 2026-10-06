# EViCT-Dx Final Reliability and Traceability Freeze

Run: DX_step15_final_reliability_traceability_freeze_v1  
Execution base commit: 87eb9393450406e76758e089445110053af23d9e  
Execution branch: main  
Frozen EViCT-Core: verified unchanged at Step15 execution.

## Final validated scope

The final EViCT-Dx reporting scope is deliberately narrower than the full experimental scope. Only **left lung involvement percentage**, **right lung involvement percentage**, and **total lung involvement percentage** passed their pre-specified validation gates and may be emitted in the **Provisional AI Diagnostic Report**, always with mandatory radiologist review.

Disease names, GGO, consolidation, bilateral involvement, pleural effusion, lobar localization, and physical area/volume remain withheld or deferred. Treatment, antibiotic, and clinical-management recommendations remain prohibited.

## Diagnosis

The COVID-CT-MD baseline achieved macro-F1 **0.6325**, balanced accuracy **0.6271**, macro-AUROC **0.7626**, and ECE **0.1566** on the 61-case held-out test. The frozen disease unlock criteria were not met. Later Step06/07 diagnosis candidates were evaluated only on train/validation and did not justify another held-out-test access.

## OOF segmentation and quantification

Across 20 strict OOF cases, left/right lung Dice were **0.9726 / 0.9769**, with generic infection Dice **0.7050**.

Lung involvement quantification:

| Field | MAE (pp) | CCC | Final status |
|---|---:|---:|---|
| Left | 3.693 | 0.8977 | Unlocked with radiologist review |
| Right | 3.272 | 0.8956 | Unlocked with radiologist review |
| Total | 3.305 | 0.9041 | Unlocked with radiologist review |

Bilateral involvement remained locked: balanced accuracy **0.5667**, sensitivity **0.9333**, specificity **0.2000**.

## Subtype validation

Internal NCP testing was strong:

| Subtype | Internal Dice | Presence AUROC | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| GGO | 0.7800 | 0.9064 | 0.8727 | 0.8000 |
| Consolidation | 0.7353 | 0.9898 | 0.9688 | 0.9302 |

However, corrected independent LongCIU STAPLE validation failed:

| Subtype | External Dice | IoU | Sensitivity | Specificity | Gate |
|---|---:|---:|---:|---:|---|
| GGO | 0.2446 | 0.1393 | 0.4214 | 0.9021 | FAIL |
| Consolidation | 0.0047 | 0.0024 | 0.0052 | 0.9972 | FAIL |

The valid external result is **Step12B**, which corrected the NIfTI array convention to the official LongCIU SimpleITK-compatible [z,y,x] convention without any performance-based orientation search. The earlier Step12 v1 run is preserved as an invalidated technical run.

## Permission-aware reporting

Step13 created **20** structured evidence records and **20** deterministic reports. All **20/20** passed validation. Ground truth was not used in evidence construction.

Step14 used **Qwen/Qwen2.5-VL-3B-Instruct** strictly as a **text-only evidence realizer**. No CT image and no ground truth were provided. Raw model outputs were accepted in **11/20 (55.0%)** cases. Deterministic fallback was required in **9/20 (45.0%)**. Exact raw numeric-copy compliance was **85.0%**. Raw prohibited recommendation/management-language violations occurred in **35.0%** of cases. The validator/fallback pipeline produced **20/20 (100.0%)** valid final reports.

This reporting result supports the value of permission-aware deterministic validation and fallback; it does **not** support unrestricted clinical free-text generation.

## Mandatory report footer

> This AI-generated report must be reviewed, assessed, and finalized by a qualified radiologist before it is used for medical assessment, diagnosis, treatment planning, or any other clinical decision.

## Manuscript positioning

The strongest defensible contribution is a **reliability-gated evidence-to-language CT framework** in which each candidate report field must independently earn permission through pre-specified validation. The negative diagnosis and external subtype results are part of the contribution: they demonstrate that high in-domain performance is not sufficient for safe report-field emission and that downstream language generation requires explicit permission control and deterministic fallback.

See:
- vlmDiagnosis/tables/step15_component_summary.csv
- vlmDiagnosis/tables/step15_manuscript_results.csv
- vlmDiagnosis/tables/step15_report_permissions.csv
- vlmDiagnosis/artifacts/audit/step15_supported_and_prohibited_claims.json
- vlmDiagnosis/artifacts/manifests/step15_traceability_sha256.csv
