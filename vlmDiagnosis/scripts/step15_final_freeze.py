from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DX = ROOT / "vlmDiagnosis"

CFG_PATH = DX / "config" / "step15_final_freeze.json"
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))

RUN_ID = CFG["run_id"]
RUN = DX / "runs" / RUN_ID
TABLES = DX / "tables"
AUDIT_DIR = DX / "artifacts" / "audit"
MANIFEST_DIR = DX / "artifacts" / "manifests"
DOCS = DX / "docs"
for p in [RUN, TABLES, AUDIT_DIR, MANIFEST_DIR, DOCS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_COMPONENT = ROOT / CFG["outputs"]["component_summary"]
OUT_MANUSCRIPT = ROOT / CFG["outputs"]["manuscript_results"]
OUT_PERMISSIONS = ROOT / CFG["outputs"]["report_permissions"]
OUT_CLAIMS = ROOT / CFG["outputs"]["claims"]
OUT_AUDIT = ROOT / CFG["outputs"]["final_audit"]
OUT_MANIFEST = ROOT / CFG["outputs"]["traceability_manifest"]
OUT_DOC = ROOT / CFG["outputs"]["final_document"]
OUT_STATE = ROOT / CFG["outputs"]["state"]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(rel: str):
    p = ROOT / rel
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def atomic_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_text(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, df: pd.DataFrame):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sync_git(message):
    helper = DX / "scripts" / "git_sync_dx.py"
    r = subprocess.run([sys.executable, str(helper), message], cwd=str(ROOT), check=False)
    return r.returncode == 0


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
    ).strip()


def git_branch():
    return subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=str(ROOT), text=True
    ).strip()


def fmt(v, d=4):
    if v is None:
        return ""
    try:
        x = float(v)
        if math.isnan(x):
            return "NA"
        return f"{x:.{d}f}"
    except Exception:
        return str(v)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    print("=" * 118)
    print("EViCT-Dx STEP 15 — FINAL RELIABILITY + MANUSCRIPT TRACEABILITY FREEZE")
    print("NO TRAINING — NO TEST REINFERENCE — NO THRESHOLD SEARCH — NO MODEL CHANGES")
    print("=" * 118)

    subprocess.run(
        [sys.executable, str(DX / "scripts" / "verify_core_lock.py")],
        cwd=str(ROOT),
        check=True,
    )

    if OUT_STATE.exists():
        prior = json.loads(OUT_STATE.read_text())
        if prior.get("status") == "COMPLETE":
            print("✓ Step15 is already COMPLETE; refusing to regenerate the final freeze.")
            print(json.dumps(prior, indent=2))
            return

    execution_base_commit = git_head()
    execution_branch = git_branch()

    s04 = load_json("vlmDiagnosis/artifacts/audit/step04_diagnosis_baseline_results.json")
    s05 = load_json("vlmDiagnosis/artifacts/audit/step05_diagnosis_unlock_audit.json")
    s06 = load_json("vlmDiagnosis/artifacts/audit/step06_strong_diagnosis_development.json")
    s07 = load_json("vlmDiagnosis/artifacts/audit/step07_frozen_dual_encoder_development.json")
    s08 = load_json("vlmDiagnosis/artifacts/audit/step08_oof_segmentation_baseline_results.json")
    s09 = load_json("vlmDiagnosis/artifacts/audit/step09_segmentation_quantification_unlock_audit.json")
    s11 = load_json("vlmDiagnosis/artifacts/audit/step11_ncp_ggo_consolidation_results.json")
    s12b = load_json("vlmDiagnosis/artifacts/audit/step12b_longciu_external_results.json")
    s13 = load_json("vlmDiagnosis/artifacts/audit/step13_permission_aware_reporting.json")
    s14 = load_json("vlmDiagnosis/artifacts/audit/step14_controlled_vlm_reporting.json")
    permissions = load_json("vlmDiagnosis/config/report_field_permissions_step12b.json")
    contract = load_json("vlmDiagnosis/config/report_field_contract_step02.json")

    require(s05["diagnostic_disease_names_unlocked"] is False,
            "Diagnosis disease-name state drift: expected locked.")
    require(s05["all_global_unlock_criteria_pass"] is False,
            "Diagnosis unlock audit unexpectedly passes globally.")
    require(s06["held_out_test_accessed"] is False and s07["held_out_test_accessed"] is False,
            "Later diagnosis development unexpectedly accessed held-out test.")
    require(int(s08["strict_oof_cases"]) == 20,
            "Step08 strict OOF case count is not 20.")
    require(s09["quantification_gates"]["left"]["field_gate_pass"] is True,
            "Left involvement gate is not pass.")
    require(s09["quantification_gates"]["right"]["field_gate_pass"] is True,
            "Right involvement gate is not pass.")
    require(s09["quantification_gates"]["total"]["field_gate_pass"] is True,
            "Total involvement gate is not pass.")
    require(s09["bilateral_gate"]["field_gate_pass"] is False,
            "Bilateral involvement unexpectedly passes.")
    require(s11["internal_unlock_gate"]["GGO"]["internal_gate_pass"] is True,
            "Step11 GGO internal gate not pass.")
    require(s11["internal_unlock_gate"]["consolidation"]["internal_gate_pass"] is True,
            "Step11 consolidation internal gate not pass.")
    require(s12b["external_confirmatory_gate"]["GGO"]["final_subtype_validation_pass"] is False,
            "GGO unexpectedly passes final subtype validation.")
    require(s12b["external_confirmatory_gate"]["consolidation"]["final_subtype_validation_pass"] is False,
            "Consolidation unexpectedly passes final subtype validation.")
    require(s12b["technical_correction"]["orientation_search"] is False,
            "Step12B reports orientation search.")
    require(s12b["threshold_search_performed"] is False,
            "Step12B reports threshold search.")
    require(int(s13["reports_valid"]) == 20 and float(s13["validation_rate"]) == 1.0,
            "Step13 deterministic report baseline is not 20/20.")
    require(s13["ground_truth_used_for_evidence_construction"] is False,
            "Ground-truth leakage reported by Step13.")
    require(int(s14["cases"]) == 20,
            "Step14 case count is not 20.")
    require(int(s14["raw_accepted"]) + int(s14["fallback_cases"]) == 20,
            "Step14 accepted + fallback count is not 20.")
    require(int(s14["final_safe_reports"]) == 20 and float(s14["final_safe_rate"]) == 1.0,
            "Step14 final safe report rate is not 20/20.")
    require(s14["ground_truth_given_to_model"] is False and s14["CT_image_given_to_model"] is False,
            "Step14 input provenance drift: CT or ground truth was supplied.")
    require(s14["all_raw_outputs_preserved"] is True and s14["all_final_outputs_validated"] is True,
            "Step14 output preservation/validation invariant failed.")

    expected_unlocked = {
        "Left_lung_involvement_percent",
        "Right_lung_involvement_percent",
        "Total_lung_involvement_percent",
    }
    actually_unlocked = {
        k for k, v in permissions["fields"].items()
        if str(v).startswith("UNLOCKED")
    }
    require(expected_unlocked.issubset(actually_unlocked),
            "Final involvement permissions are not all unlocked.")

    component_rows = [
        {
            "component": "Etiologic diagnosis",
            "dataset_or_reference": "COVID-CT-MD held-out test",
            "evaluation": "Step04 baseline + Step05 frozen unlock audit",
            "headline_result": (
                f"macro-F1={fmt(s04['test_metrics']['macro_F1'])}; "
                f"balanced_accuracy={fmt(s04['test_metrics']['balanced_accuracy'])}; "
                f"macro-AUROC={fmt(s04['test_metrics']['macro_AUROC'])}; "
                f"ECE={fmt(s04['test_metrics']['ECE'])}"
            ),
            "final_status": "LOCKED_FAILED_VALIDATION",
            "manuscript_role": "Negative/limitation result; do not emit COVID-19/CAP/Normal in reports.",
        },
        {
            "component": "Diagnosis development after Step04",
            "dataset_or_reference": "COVID-CT-MD train/validation only",
            "evaluation": "Step06 ConvNeXt MIL and Step07 frozen dual-encoder development",
            "headline_result": (
                f"Step06 val macro-F1={fmt(s06['strong_validation_metrics']['macro_F1'])}; "
                f"Step07 val macro-F1={fmt(s07['validation_metrics']['macro_F1'])}; "
                "held-out test not accessed"
            ),
            "final_status": "REJECTED_DEVELOPMENT_CANDIDATES",
            "manuscript_role": "Documents leakage-safe stopping decision; Step04 remains diagnostic baseline.",
        },
        {
            "component": "Left/right lung segmentation",
            "dataset_or_reference": "COVID19_CT_SEG_20 strict 5-fold OOF",
            "evaluation": "Step08",
            "headline_result": (
                f"left Dice={fmt(s08['mean_Dice_left'])}; "
                f"right Dice={fmt(s08['mean_Dice_right'])}"
            ),
            "final_status": "EVALUATED_DESCRIPTIVE",
            "manuscript_role": "Strong lung-mask support for quantification; no separately frozen report-field gate.",
        },
        {
            "component": "Generic infection segmentation",
            "dataset_or_reference": "COVID19_CT_SEG_20 strict 5-fold OOF",
            "evaluation": "Step08/09",
            "headline_result": (
                f"Dice={fmt(s08['mean_Dice_infection'])}; "
                f"IoU={fmt(s08['mean_IoU_infection'])}; "
                f"sensitivity={fmt(s08['mean_infection_sensitivity'])}; "
                f"specificity={fmt(s08['mean_infection_specificity'])}"
            ),
            "final_status": "DESCRIPTIVE_ONLY",
            "manuscript_role": "Supports quantification; no independent generic-infection report-field unlock.",
        },
        {
            "component": "Lung involvement quantification",
            "dataset_or_reference": "COVID19_CT_SEG_20 strict OOF",
            "evaluation": "Step08/09",
            "headline_result": (
                f"left MAE={fmt(s08['quantification']['left']['MAE_percentage_points'])} pp, "
                f"CCC={fmt(s08['quantification']['left']['CCC'])}; "
                f"right MAE={fmt(s08['quantification']['right']['MAE_percentage_points'])} pp, "
                f"CCC={fmt(s08['quantification']['right']['CCC'])}; "
                f"total MAE={fmt(s08['quantification']['total']['MAE_percentage_points'])} pp, "
                f"CCC={fmt(s08['quantification']['total']['CCC'])}"
            ),
            "final_status": "UNLOCKED_WITH_RADIOLOGIST_REVIEW",
            "manuscript_role": "Only validated case-specific quantitative report content.",
        },
        {
            "component": "Bilateral involvement",
            "dataset_or_reference": "COVID19_CT_SEG_20 strict OOF",
            "evaluation": "Step08/09",
            "headline_result": (
                f"balanced_accuracy={fmt(s08['bilateral']['balanced_accuracy'])}; "
                f"sensitivity={fmt(s08['bilateral']['sensitivity'])}; "
                f"specificity={fmt(s08['bilateral']['specificity'])}; "
                f"F1={fmt(s08['bilateral']['F1'])}"
            ),
            "final_status": "LOCKED_FAILED_VALIDATION",
            "manuscript_role": "Do not emit bilateral classification.",
        },
        {
            "component": "GGO subtype segmentation/presence",
            "dataset_or_reference": "Public NCP internal test",
            "evaluation": "Step11",
            "headline_result": (
                f"Dice={fmt(s11['internal_test_segmentation']['GGO']['Dice'])}; "
                f"presence AUROC={fmt(s11['internal_test_presence']['GGO']['AUROC'])}; "
                f"sensitivity={fmt(s11['internal_test_presence']['GGO']['sensitivity'])}; "
                f"specificity={fmt(s11['internal_test_presence']['GGO']['specificity'])}"
            ),
            "final_status": "INTERNAL_GATE_PASS_EXTERNAL_CONFIRMATION_REQUIRED",
            "manuscript_role": "Strong in-domain performance, not sufficient for report-field unlock.",
        },
        {
            "component": "Consolidation subtype segmentation/presence",
            "dataset_or_reference": "Public NCP internal test",
            "evaluation": "Step11",
            "headline_result": (
                f"Dice={fmt(s11['internal_test_segmentation']['consolidation']['Dice'])}; "
                f"presence AUROC={fmt(s11['internal_test_presence']['consolidation']['AUROC'])}; "
                f"sensitivity={fmt(s11['internal_test_presence']['consolidation']['sensitivity'])}; "
                f"specificity={fmt(s11['internal_test_presence']['consolidation']['specificity'])}"
            ),
            "final_status": "INTERNAL_GATE_PASS_EXTERNAL_CONFIRMATION_REQUIRED",
            "manuscript_role": "Strong in-domain performance, not sufficient for report-field unlock.",
        },
        {
            "component": "GGO external generalization",
            "dataset_or_reference": "LongCIU 90-slice STAPLE consensus",
            "evaluation": "Corrected Step12B",
            "headline_result": (
                f"Dice={fmt(s12b['primary_STAPLE_metrics']['GGO']['Dice'])}; "
                f"IoU={fmt(s12b['primary_STAPLE_metrics']['GGO']['IoU'])}; "
                f"sensitivity={fmt(s12b['primary_STAPLE_metrics']['GGO']['sensitivity'])}; "
                f"specificity={fmt(s12b['primary_STAPLE_metrics']['GGO']['specificity'])}"
            ),
            "final_status": "LOCKED_FAILED_EXTERNAL_VALIDATION",
            "manuscript_role": "Cross-domain failure; preserve as reliability/abstention evidence.",
        },
        {
            "component": "Consolidation external generalization",
            "dataset_or_reference": "LongCIU 90-slice STAPLE consensus",
            "evaluation": "Corrected Step12B",
            "headline_result": (
                f"Dice={fmt(s12b['primary_STAPLE_metrics']['consolidation']['Dice'])}; "
                f"IoU={fmt(s12b['primary_STAPLE_metrics']['consolidation']['IoU'])}; "
                f"sensitivity={fmt(s12b['primary_STAPLE_metrics']['consolidation']['sensitivity'])}; "
                f"specificity={fmt(s12b['primary_STAPLE_metrics']['consolidation']['specificity'])}"
            ),
            "final_status": "LOCKED_FAILED_EXTERNAL_VALIDATION",
            "manuscript_role": "Cross-domain failure; preserve as reliability/abstention evidence.",
        },
        {
            "component": "Deterministic evidence-to-report baseline",
            "dataset_or_reference": "20 strict-OOF cases",
            "evaluation": "Step13",
            "headline_result": "20/20 reports valid; prediction-only evidence; no ground truth in builder",
            "final_status": "PASS",
            "manuscript_role": "Safety baseline and deterministic fallback.",
        },
        {
            "component": "Controlled VLM realization",
            "dataset_or_reference": "20 Step13 evidence records",
            "evaluation": "Step14 Qwen2.5-VL-3B-Instruct text-only",
            "headline_result": (
                f"raw acceptance={100*s14['raw_acceptance_rate']:.1f}%; "
                f"fallback={100*s14['fallback_rate']:.1f}%; "
                f"numeric-copy={100*s14['raw_exact_numeric_copy_rate']:.1f}%; "
                f"prohibited-recommendation raw violation={100*s14['raw_prohibited_recommendation_rate']:.1f}%; "
                f"final safe={100*s14['final_safe_rate']:.1f}%"
            ),
            "final_status": "PASS_WITH_DETERMINISTIC_FALLBACK",
            "manuscript_role": "Demonstrates why validator/fallback is required for language realization.",
        },
    ]
    component_df = pd.DataFrame(component_rows)
    atomic_csv(OUT_COMPONENT, component_df)

    manuscript_rows = []

    def add(section, component, metric, value, unit="", split="", interpretation=""):
        manuscript_rows.append({
            "section": section,
            "component": component,
            "metric": metric,
            "value": value,
            "unit": unit,
            "evaluation_split_or_reference": split,
            "interpretation": interpretation,
        })

    add("Diagnosis", "COVID-CT-MD baseline", "Macro-F1", s04["test_metrics"]["macro_F1"], "", "held-out test n=61", "Below unlock criterion")
    add("Diagnosis", "COVID-CT-MD baseline", "Balanced accuracy", s04["test_metrics"]["balanced_accuracy"], "", "held-out test n=61", "Below unlock criterion")
    add("Diagnosis", "COVID-CT-MD baseline", "Macro-AUROC", s04["test_metrics"]["macro_AUROC"], "", "held-out test n=61", "Below unlock criterion")
    add("Diagnosis", "COVID-CT-MD baseline", "ECE", s04["test_metrics"]["ECE"], "", "held-out test n=61", "Above maximum unlock criterion")
    add("Segmentation", "Left lung", "Dice", s08["mean_Dice_left"], "", "20-case strict OOF", "Descriptive lung-mask performance")
    add("Segmentation", "Right lung", "Dice", s08["mean_Dice_right"], "", "20-case strict OOF", "Descriptive lung-mask performance")
    add("Segmentation", "Generic infection", "Dice", s08["mean_Dice_infection"], "", "20-case strict OOF", "Supports quantification; no independent report field")
    for side in ["left", "right", "total"]:
        label = side.capitalize()
        add("Quantification", f"{label} lung involvement", "MAE", s08["quantification"][side]["MAE_percentage_points"], "percentage points", "20-case strict OOF", "Pass <=5 pp")
        add("Quantification", f"{label} lung involvement", "CCC", s08["quantification"][side]["CCC"], "", "20-case strict OOF", "Pass >=0.85")
    add("Quantification", "Bilateral involvement", "Balanced accuracy", s08["bilateral"]["balanced_accuracy"], "", "20-case strict OOF", "Fail <0.80")
    add("Quantification", "Bilateral involvement", "Sensitivity", s08["bilateral"]["sensitivity"], "", "20-case strict OOF", "Pass >=0.80")
    add("Quantification", "Bilateral involvement", "Specificity", s08["bilateral"]["specificity"], "", "20-case strict OOF", "Fail <0.80")

    for key, label in [("GGO", "GGO"), ("consolidation", "Consolidation")]:
        add("Subtype internal", label, "Segmentation Dice", s11["internal_test_segmentation"][key]["Dice"], "", "NCP internal test 75 slices / 15 groups", "Internal gate pass")
        add("Subtype internal", label, "Presence AUROC", s11["internal_test_presence"][key]["AUROC"], "", "NCP internal test 75 slices / 15 groups", "Internal gate pass")
        add("Subtype internal", label, "Presence sensitivity", s11["internal_test_presence"][key]["sensitivity"], "", "NCP internal test 75 slices / 15 groups", "Internal gate pass")
        add("Subtype internal", label, "Presence specificity", s11["internal_test_presence"][key]["specificity"], "", "NCP internal test 75 slices / 15 groups", "Internal gate pass")
        add("Subtype external", label, "STAPLE Dice", s12b["primary_STAPLE_metrics"][key]["Dice"], "", "LongCIU 90 selected slices", "External gate fail")

    add("Reporting", "Step13 deterministic baseline", "Validated final report rate", s13["validation_rate"], "", "20 strict-OOF cases", "20/20 pass")
    add("Reporting", "Step14 controlled VLM", "Raw acceptance rate", s14["raw_acceptance_rate"], "", "20 evidence records", "11/20 accepted")
    add("Reporting", "Step14 controlled VLM", "Fallback rate", s14["fallback_rate"], "", "20 evidence records", "9/20 deterministic fallback")
    add("Reporting", "Step14 controlled VLM", "Exact numeric copy rate", s14["raw_exact_numeric_copy_rate"], "", "20 raw generations", "85%")
    add("Reporting", "Step14 controlled VLM", "Raw prohibited-recommendation violation rate", s14["raw_prohibited_recommendation_rate"], "", "20 raw generations", "35%; blocked by validator")
    add("Reporting", "Step14 controlled VLM", "Final safe report rate", s14["final_safe_rate"], "", "20 delivered reports", "20/20 after deterministic fallback")
    add("Reporting", "Step14 controlled VLM", "Raw locked-field violation rate", s14["raw_locked_field_violation_rate"], "", "20 raw generations", "0%")

    manuscript_df = pd.DataFrame(manuscript_rows)
    atomic_csv(OUT_MANUSCRIPT, manuscript_df)

    permission_rows = [{"field": k, "status": v} for k, v in permissions["fields"].items()]
    permission_df = pd.DataFrame(permission_rows).sort_values("field").reset_index(drop=True)
    atomic_csv(OUT_PERMISSIONS, permission_df)

    claims = {
        "project": "EViCT-Dx",
        "stage": "STEP_15_SUPPORTED_AND_PROHIBITED_CLAIMS",
        "created_utc": now(),
        "supported_claims": [
            "The diagnostic baseline did not satisfy the pre-specified disease-name unlock criteria, and COVID-19/CAP/Normal remain withheld from generated reports.",
            "Strict five-fold OOF evaluation on the 20-volume segmentation cohort produced high left/right lung Dice and unlocked left, right, and total lung-involvement percentages under the pre-specified MAE/CCC gates.",
            "Bilateral-involvement classification failed its pre-specified OOF gate and remains withheld.",
            "GGO and consolidation passed the frozen internal NCP subtype gates but failed independent corrected LongCIU STAPLE external validation, demonstrating a substantial cross-domain generalization gap.",
            "The external LongCIU evaluation used the frozen Step11 model and thresholds without training, fine-tuning, checkpoint selection, threshold search, or performance-based orientation search.",
            "Step13 generated 20/20 valid deterministic provisional reports from prediction-only structured evidence, with reference columns excluded before evidence construction.",
            "Qwen2.5-VL-3B-Instruct was used in Step14 as a text-only structured-evidence realizer; it did not receive CT images or ground truth.",
            "Step14 accepted 11/20 raw VLM reports, used deterministic fallback for 9/20, and achieved 20/20 validated final reports.",
            "Raw VLM generations had 85% exact numeric-copy compliance and a 35% prohibited recommendation/management-language violation rate under the frozen validator.",
            "Only left, right, and total lung-involvement percentages are currently unlocked for the Provisional AI Diagnostic Report, and mandatory radiologist review remains required."
        ],
        "prohibited_or_unsupported_claims": [
            "Do not claim validated etiologic diagnosis for COVID-19, CAP, or Normal.",
            "Do not claim externally validated GGO or consolidation detection/segmentation.",
            "Do not claim bilateral-involvement classification is validated.",
            "Do not claim validated pleural-effusion reporting.",
            "Do not claim lobar localization, physical lesion area, or physical lesion volume.",
            "Do not claim autonomous clinical decision support, treatment recommendation, antibiotic recommendation, or management recommendation.",
            "Do not describe Step14 as direct image-to-text diagnosis or direct VLM interpretation of CT images; no CT image was provided to the Step14 language model.",
            "Do not claim expert-level or radiologist-equivalent report quality without a reader study.",
            "Do not hide the corrected LongCIU external-domain failure or replace it with tuned LongCIU results.",
            "Do not use Step08 OOF labels, Step11 internal-test labels, or LongCIU labels for post-hoc threshold tuning.",
            "Do not treat the invalidated Step12 v1 orientation run as the valid external evaluation; Step12B is the corrected reference-convention result."
        ],
    }
    atomic_json(OUT_CLAIMS, claims)

    md = f"""# EViCT-Dx Final Reliability and Traceability Freeze

Run: {RUN_ID}  
Execution base commit: {execution_base_commit}  
Execution branch: {execution_branch}  
Frozen EViCT-Core: verified unchanged at Step15 execution.

## Final validated scope

The final EViCT-Dx reporting scope is deliberately narrower than the full experimental scope. Only **left lung involvement percentage**, **right lung involvement percentage**, and **total lung involvement percentage** passed their pre-specified validation gates and may be emitted in the **{contract['mandatory_title']}**, always with mandatory radiologist review.

Disease names, GGO, consolidation, bilateral involvement, pleural effusion, lobar localization, and physical area/volume remain withheld or deferred. Treatment, antibiotic, and clinical-management recommendations remain prohibited.

## Diagnosis

The COVID-CT-MD baseline achieved macro-F1 **{s04['test_metrics']['macro_F1']:.4f}**, balanced accuracy **{s04['test_metrics']['balanced_accuracy']:.4f}**, macro-AUROC **{s04['test_metrics']['macro_AUROC']:.4f}**, and ECE **{s04['test_metrics']['ECE']:.4f}** on the 61-case held-out test. The frozen disease unlock criteria were not met. Later Step06/07 diagnosis candidates were evaluated only on train/validation and did not justify another held-out-test access.

## OOF segmentation and quantification

Across 20 strict OOF cases, left/right lung Dice were **{s08['mean_Dice_left']:.4f} / {s08['mean_Dice_right']:.4f}**, with generic infection Dice **{s08['mean_Dice_infection']:.4f}**.

Lung involvement quantification:

| Field | MAE (pp) | CCC | Final status |
|---|---:|---:|---|
| Left | {s08['quantification']['left']['MAE_percentage_points']:.3f} | {s08['quantification']['left']['CCC']:.4f} | Unlocked with radiologist review |
| Right | {s08['quantification']['right']['MAE_percentage_points']:.3f} | {s08['quantification']['right']['CCC']:.4f} | Unlocked with radiologist review |
| Total | {s08['quantification']['total']['MAE_percentage_points']:.3f} | {s08['quantification']['total']['CCC']:.4f} | Unlocked with radiologist review |

Bilateral involvement remained locked: balanced accuracy **{s08['bilateral']['balanced_accuracy']:.4f}**, sensitivity **{s08['bilateral']['sensitivity']:.4f}**, specificity **{s08['bilateral']['specificity']:.4f}**.

## Subtype validation

Internal NCP testing was strong:

| Subtype | Internal Dice | Presence AUROC | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| GGO | {s11['internal_test_segmentation']['GGO']['Dice']:.4f} | {s11['internal_test_presence']['GGO']['AUROC']:.4f} | {s11['internal_test_presence']['GGO']['sensitivity']:.4f} | {s11['internal_test_presence']['GGO']['specificity']:.4f} |
| Consolidation | {s11['internal_test_segmentation']['consolidation']['Dice']:.4f} | {s11['internal_test_presence']['consolidation']['AUROC']:.4f} | {s11['internal_test_presence']['consolidation']['sensitivity']:.4f} | {s11['internal_test_presence']['consolidation']['specificity']:.4f} |

However, corrected independent LongCIU STAPLE validation failed:

| Subtype | External Dice | IoU | Sensitivity | Specificity | Gate |
|---|---:|---:|---:|---:|---|
| GGO | {s12b['primary_STAPLE_metrics']['GGO']['Dice']:.4f} | {s12b['primary_STAPLE_metrics']['GGO']['IoU']:.4f} | {s12b['primary_STAPLE_metrics']['GGO']['sensitivity']:.4f} | {s12b['primary_STAPLE_metrics']['GGO']['specificity']:.4f} | FAIL |
| Consolidation | {s12b['primary_STAPLE_metrics']['consolidation']['Dice']:.4f} | {s12b['primary_STAPLE_metrics']['consolidation']['IoU']:.4f} | {s12b['primary_STAPLE_metrics']['consolidation']['sensitivity']:.4f} | {s12b['primary_STAPLE_metrics']['consolidation']['specificity']:.4f} | FAIL |

The valid external result is **Step12B**, which corrected the NIfTI array convention to the official LongCIU SimpleITK-compatible [z,y,x] convention without any performance-based orientation search. The earlier Step12 v1 run is preserved as an invalidated technical run.

## Permission-aware reporting

Step13 created **{s13['evidence_records']}** structured evidence records and **{s13['deterministic_reports']}** deterministic reports. All **{s13['reports_valid']}/{s13['deterministic_reports']}** passed validation. Ground truth was not used in evidence construction.

Step14 used **{s14['model']}** strictly as a **text-only evidence realizer**. No CT image and no ground truth were provided. Raw model outputs were accepted in **{s14['raw_accepted']}/20 ({100*s14['raw_acceptance_rate']:.1f}%)** cases. Deterministic fallback was required in **{s14['fallback_cases']}/20 ({100*s14['fallback_rate']:.1f}%)**. Exact raw numeric-copy compliance was **{100*s14['raw_exact_numeric_copy_rate']:.1f}%**. Raw prohibited recommendation/management-language violations occurred in **{100*s14['raw_prohibited_recommendation_rate']:.1f}%** of cases. The validator/fallback pipeline produced **{s14['final_safe_reports']}/20 ({100*s14['final_safe_rate']:.1f}%)** valid final reports.

This reporting result supports the value of permission-aware deterministic validation and fallback; it does **not** support unrestricted clinical free-text generation.

## Mandatory report footer

> {contract['mandatory_footer']}

## Manuscript positioning

The strongest defensible contribution is a **reliability-gated evidence-to-language CT framework** in which each candidate report field must independently earn permission through pre-specified validation. The negative diagnosis and external subtype results are part of the contribution: they demonstrate that high in-domain performance is not sufficient for safe report-field emission and that downstream language generation requires explicit permission control and deterministic fallback.

See:
- vlmDiagnosis/tables/step15_component_summary.csv
- vlmDiagnosis/tables/step15_manuscript_results.csv
- vlmDiagnosis/tables/step15_report_permissions.csv
- vlmDiagnosis/artifacts/audit/step15_supported_and_prohibited_claims.json
- vlmDiagnosis/artifacts/manifests/step15_traceability_sha256.csv
"""
    atomic_text(OUT_DOC, md)

    audit = {
        "project": "EViCT-Dx",
        "stage": "STEP_15_FINAL_RELIABILITY_AND_MANUSCRIPT_TRACEABILITY_FREEZE",
        "run_id": RUN_ID,
        "completed_utc": now(),
        "execution_base_commit": execution_base_commit,
        "execution_branch": execution_branch,
        "training_performed": False,
        "test_reinference_performed": False,
        "threshold_search_performed": False,
        "model_changed": False,
        "final_permissions": permissions["fields"],
        "diagnosis": {
            "fields_unlocked": False,
            "test_macro_F1": s04["test_metrics"]["macro_F1"],
            "test_balanced_accuracy": s04["test_metrics"]["balanced_accuracy"],
            "test_macro_AUROC": s04["test_metrics"]["macro_AUROC"],
            "test_ECE": s04["test_metrics"]["ECE"],
            "later_development_test_accessed": False,
        },
        "oof_segmentation_quantification": {
            "cases": 20,
            "left_lung_Dice": s08["mean_Dice_left"],
            "right_lung_Dice": s08["mean_Dice_right"],
            "infection_Dice": s08["mean_Dice_infection"],
            "left_involvement_unlocked": True,
            "right_involvement_unlocked": True,
            "total_involvement_unlocked": True,
            "bilateral_involvement_unlocked": False,
        },
        "subtypes": {
            "GGO_internal_gate_pass": True,
            "consolidation_internal_gate_pass": True,
            "GGO_external_gate_pass": False,
            "consolidation_external_gate_pass": False,
            "valid_external_run": "DX_step12b_longciu_reference_orientation_v1",
            "invalidated_external_run_retained": "DX_step12_longciu_external_ggo_consolidation_v1",
        },
        "reporting": {
            "step13_deterministic_safe_rate": s13["validation_rate"],
            "step14_model": s14["model"],
            "step14_model_role": s14["model_role"],
            "CT_image_given_to_step14_model": False,
            "ground_truth_given_to_step14_model": False,
            "raw_acceptance_rate": s14["raw_acceptance_rate"],
            "fallback_rate": s14["fallback_rate"],
            "raw_numeric_copy_rate": s14["raw_exact_numeric_copy_rate"],
            "raw_prohibited_recommendation_rate": s14["raw_prohibited_recommendation_rate"],
            "final_safe_rate": s14["final_safe_rate"],
        },
        "supported_claims_file": str(OUT_CLAIMS.relative_to(ROOT)).replace("\\", "/"),
        "final_document": str(OUT_DOC.relative_to(ROOT)).replace("\\", "/"),
        "status": "FINAL_EVICT_DX_EXPERIMENTAL_FREEZE_COMPLETE",
        "next_action": "MANUSCRIPT_WRITING_AND_OPTIONAL_FUTURE_NEW_PROTOCOLS_ONLY",
    }
    atomic_json(OUT_AUDIT, audit)

    manifest_paths = [ROOT / p for p in CFG["source_artifacts"]] + [
        CFG_PATH,
        OUT_COMPONENT,
        OUT_MANUSCRIPT,
        OUT_PERMISSIONS,
        OUT_CLAIMS,
        OUT_DOC,
        OUT_AUDIT,
    ]

    manifest_rows = []
    seen = set()
    for p in manifest_paths:
        p = p.resolve()
        if p in seen:
            continue
        seen.add(p)
        if not p.exists():
            raise FileNotFoundError(p)
        manifest_rows.append({
            "relative_path": str(p.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(p),
            "bytes": p.stat().st_size,
        })
    manifest_df = pd.DataFrame(manifest_rows).sort_values("relative_path").reset_index(drop=True)
    atomic_csv(OUT_MANIFEST, manifest_df)

    state = {
        "run_id": RUN_ID,
        "status": "COMPLETE",
        "updated_utc": now(),
        "execution_base_commit": execution_base_commit,
        "training_performed": False,
        "test_reinference_performed": False,
        "threshold_search_performed": False,
        "model_changed": False,
        "traceability_files_hashed": int(len(manifest_df)),
        "final_unlocked_report_fields": sorted(expected_unlocked),
        "final_locked_or_deferred_report_fields": sorted(
            k for k, v in permissions["fields"].items()
            if not str(v).startswith("UNLOCKED")
        ),
        "next_action": "MANUSCRIPT_WRITING_AND_OPTIONAL_FUTURE_NEW_PROTOCOLS_ONLY",
    }
    atomic_json(OUT_STATE, state)

    synced = sync_git("Complete EViCT-Dx Step15 final reliability and traceability freeze")

    print("\n" + "=" * 118)
    print("✅ STEP 15 COMPLETE — FINAL EViCT-Dx RELIABILITY + TRACEABILITY FREEZE")
    print("Diagnosis disease fields             : LOCKED")
    print("Left involvement %                   : UNLOCKED")
    print("Right involvement %                  : UNLOCKED")
    print("Total involvement %                  : UNLOCKED")
    print("Bilateral involvement                : LOCKED")
    print("GGO                                   : LOCKED — external validation failed")
    print("Consolidation                         : LOCKED — external validation failed")
    print(f"Step13 deterministic reports         : {s13['reports_valid']}/20 valid")
    print(f"Step14 raw VLM accepted              : {s14['raw_accepted']}/20 ({100*s14['raw_acceptance_rate']:.1f}%)")
    print(f"Step14 deterministic fallback        : {s14['fallback_cases']}/20 ({100*s14['fallback_rate']:.1f}%)")
    print(f"Step14 final safe reports            : {s14['final_safe_reports']}/20 ({100*s14['final_safe_rate']:.1f}%)")
    print(f"Traceability files hashed            : {len(manifest_df)}")
    print(f"GitHub metadata sync                 : {'SUCCESS' if synced else 'CHECK REQUIRED'}")
    print("Experimental status                  : FINAL EViCT-Dx FREEZE COMPLETE")
    print("NEXT                                 : MANUSCRIPT WRITING / FIGURES / TABLES")
    print("=" * 118)


if __name__ == "__main__":
    main()
