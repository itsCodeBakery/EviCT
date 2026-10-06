from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DX = ROOT / "vlmDiagnosis"

CFG_PATH = DX / "config" / "step13_permission_aware_reporting.json"
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))

RUN_ID = CFG["run_id"]
RUN = DX / "runs" / RUN_ID
TABLES = DX / "tables"
AUDIT_DIR = DX / "artifacts" / "audit"
EVIDENCE_DIR = ROOT / CFG["outputs"]["evidence_dir"]
REPORTS_DIR = ROOT / CFG["outputs"]["reports_dir"]

for p in [RUN, TABLES, AUDIT_DIR, EVIDENCE_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

PREDICTIONS = ROOT / CFG["inputs"]["quantitative_predictions"]
PERMISSIONS_PATH = ROOT / CFG["inputs"]["permissions"]
REPORT_CONTRACT_PATH = ROOT / CFG["inputs"]["report_contract"]
STEP12B_RESULTS = ROOT / CFG["inputs"]["step12b_results"]

VALIDATION_CSV = ROOT / CFG["outputs"]["validation_table"]
REPORT_INDEX_CSV = ROOT / CFG["outputs"]["report_index"]
AUDIT_JSON = ROOT / CFG["outputs"]["audit"]
STATE_JSON = ROOT / CFG["outputs"]["state"]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
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
    script = DX / "scripts" / "git_sync_dx.py"
    r = subprocess.run([sys.executable, str(script), message], cwd=str(ROOT), check=False)
    if r.returncode != 0:
        print(f"⚠ metadata Git sync returned {r.returncode}")
    return r.returncode == 0


def verify_permissions(permissions):
    required = {
        "Left_lung_involvement_percent",
        "Right_lung_involvement_percent",
        "Total_lung_involvement_percent",
    }
    unlocked = {
        k for k, v in permissions["fields"].items()
        if str(v).startswith("UNLOCKED")
    }
    if not required.issubset(unlocked):
        raise RuntimeError(
            "The three lung-involvement percentage fields are not all unlocked."
        )

    forbidden_to_emit = [
        "COVID-19", "CAP", "Normal", "GGO", "Consolidation",
        "Bilateral_involvement", "Pleural_effusion_slice_level",
    ]
    for field in forbidden_to_emit:
        status = permissions["fields"].get(field, "")
        if str(status).startswith("UNLOCKED"):
            raise RuntimeError(
                f"Unexpected permission drift: {field} is unlocked in Step13 input."
            )


def sanitize_prediction_table(df: pd.DataFrame):
    required = {
        "case_id",
        "predicted_left_percent",
        "predicted_right_percent",
        "predicted_total_percent",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Step08 OOF prediction table missing columns: {sorted(missing)}")

    # IMPORTANT: select prediction-only fields explicitly.
    # Reference columns are never copied into evidence/report generation.
    out = df[
        [
            "case_id",
            "predicted_left_percent",
            "predicted_right_percent",
            "predicted_total_percent",
        ]
    ].copy()

    if len(out) != 20 or out.case_id.nunique() != 20:
        raise RuntimeError(
            f"Expected 20 unique strict-OOF cases; got rows={len(out)}, unique={out.case_id.nunique()}."
        )

    for col in [
        "predicted_left_percent",
        "predicted_right_percent",
        "predicted_total_percent",
    ]:
        vals = pd.to_numeric(out[col], errors="coerce")
        if vals.isna().any():
            raise RuntimeError(f"Non-numeric value found in {col}.")
        if ((vals < 0) | (vals > 100)).any():
            raise RuntimeError(f"Out-of-range percentage found in {col}.")
        out[col] = vals.astype(float)

    return out.sort_values("case_id").reset_index(drop=True)


def rounded(v: float):
    d = int(CFG["numeric_policy"]["report_precision_decimals"])
    return round(float(v), d)


def build_evidence(row, provenance):
    left = rounded(row.predicted_left_percent)
    right = rounded(row.predicted_right_percent)
    total = rounded(row.predicted_total_percent)

    return {
        "project": "EViCT-Dx",
        "stage": "STEP_13_PERMISSION_AWARE_STRUCTURED_EVIDENCE",
        "case_id": str(row.case_id),
        "evaluation_unit": "complete_CT_volume_case",
        "provenance": provenance,
        "validated_measurements": {
            "Left_lung_involvement_percent": {
                "value": left,
                "unit": "percent",
                "permission": "UNLOCKED_FOR_PROVISIONAL_AI_REPORT_WITH_MANDATORY_RADIOLOGIST_REVIEW",
            },
            "Right_lung_involvement_percent": {
                "value": right,
                "unit": "percent",
                "permission": "UNLOCKED_FOR_PROVISIONAL_AI_REPORT_WITH_MANDATORY_RADIOLOGIST_REVIEW",
            },
            "Total_lung_involvement_percent": {
                "value": total,
                "unit": "percent",
                "permission": "UNLOCKED_FOR_PROVISIONAL_AI_REPORT_WITH_MANDATORY_RADIOLOGIST_REVIEW",
            },
        },
        "withheld_fields": {
            "etiologic_diagnosis": "WITHHELD_FAILED_DIAGNOSIS_VALIDATION",
            "pathology_subtype": "WITHHELD_FAILED_EXTERNAL_SUBTYPE_VALIDATION",
            "bilateral_involvement": "WITHHELD_FAILED_OOF_VALIDATION",
            "pleural_effusion": "WITHHELD_PENDING_VALIDATION",
            "lobar_localization": "WITHHELD_DEFERRED",
            "physical_area_volume": "WITHHELD_DEFERRED",
        },
        "prohibited_fields": [
            "Treatment_recommendation",
            "Antibiotic_recommendation",
            "Clinical_management_decision",
        ],
        "generation_constraints": {
            "may_emit_disease_name": False,
            "may_emit_subtype_name_as_case_finding": False,
            "may_emit_bilateral_classification": False,
            "may_emit_treatment_recommendation": False,
            "may_emit_only_validated_numeric_fields": True,
            "radiologist_review_required": True,
        },
        "ground_truth_used_to_construct_this_evidence": False,
    }


def render_report(evidence, title, footer):
    m = evidence["validated_measurements"]
    left = m["Left_lung_involvement_percent"]["value"]
    right = m["Right_lung_involvement_percent"]["value"]
    total = m["Total_lung_involvement_percent"]["value"]

    lines = [
        title,
        "",
        "Findings",
        (
            "AI-estimated lung involvement: "
            f"left lung {left:.1f}%, right lung {right:.1f}%, "
            f"total lung {total:.1f}%."
        ),
        "",
        "Validation-aware limitations",
        (
            "Etiologic disease classification is withheld because the frozen "
            "diagnostic validation criteria were not met."
        ),
        (
            "Pathology subtype classification is withheld because independent "
            "external subtype validation criteria were not met."
        ),
        (
            "Bilateral-involvement classification is withheld because its "
            "pre-specified validation criteria were not met."
        ),
        (
            "Pleural-effusion classification, lobar localization, and physical "
            "area or volume measurements are not emitted because their required "
            "validation or geometry prerequisites are incomplete."
        ),
        "",
        footer,
    ]
    return "\n".join(lines).strip() + "\n"


def validate_report(report: str, evidence, title: str, footer: str):
    errors = []

    if not report.startswith(title + "\n"):
        errors.append("TITLE_MISSING_OR_CHANGED")

    if footer not in report:
        errors.append("MANDATORY_FOOTER_MISSING_OR_CHANGED")

    # Exact three protected percentages only.
    expected = [
        evidence["validated_measurements"]["Left_lung_involvement_percent"]["value"],
        evidence["validated_measurements"]["Right_lung_involvement_percent"]["value"],
        evidence["validated_measurements"]["Total_lung_involvement_percent"]["value"],
    ]
    seen = [float(x) for x in re.findall(r"(?<![\d.])(\d{1,3}\.\d)%", report)]

    if seen != [float(x) for x in expected]:
        errors.append(
            f"NUMERIC_COPY_MISMATCH expected={expected} seen={seen}"
        )

    # Case-level disease/subtype findings must not appear. These checks focus on
    # affirmative clinical statements, not on abstract limitation language.
    prohibited_assertion_patterns = [
        r"\bdiagnosis\s*:\s*covid",
        r"\bdiagnosis\s*:\s*cap\b",
        r"\bdiagnosis\s*:\s*normal\b",
        r"\bground[- ]glass opacity\s+(is|are|present|detected)",
        r"\bggo\s+(is|are|present|detected)",
        r"\bconsolidation\s+(is|are|present|detected)",
        r"\bpleural effusion\s+(is|are|present|detected)",
        r"\bbilateral involvement\s*:\s*(yes|no|present|absent)",
        r"\brecommend(ed|ation)?\b.*\b(antibiotic|treatment|therapy)\b",
    ]

    lower = report.lower()
    for pattern in prohibited_assertion_patterns:
        if re.search(pattern, lower, flags=re.I | re.S):
            errors.append(f"PROHIBITED_ASSERTION:{pattern}")

    if "radiologist" not in lower:
        errors.append("RADIOLOGIST_REVIEW_REQUIREMENT_MISSING")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "protected_numeric_values": expected,
        "seen_percent_values": seen,
    }


def main():
    print("=" * 112)
    print("EViCT-Dx STEP 13 — PERMISSION-AWARE STRUCTURED EVIDENCE + DETERMINISTIC REPORTING")
    print("PREDICTION-ONLY EVIDENCE — LOCKED FIELDS WITHHELD — NO LLM YET")
    print("=" * 112)

    subprocess.run(
        [sys.executable, str(DX / "scripts" / "verify_core_lock.py")],
        cwd=str(ROOT),
        check=True,
    )

    if STATE_JSON.exists():
        prior = json.loads(STATE_JSON.read_text())
        if prior.get("status") == "COMPLETE":
            print("✓ Step13 is already COMPLETE; refusing to regenerate frozen reports.")
            print(json.dumps(prior, indent=2))
            return

    required_paths = [
        PREDICTIONS,
        PERMISSIONS_PATH,
        REPORT_CONTRACT_PATH,
        STEP12B_RESULTS,
    ]
    for p in required_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    permissions = json.loads(PERMISSIONS_PATH.read_text())
    contract = json.loads(REPORT_CONTRACT_PATH.read_text())
    step12b = json.loads(STEP12B_RESULTS.read_text())

    verify_permissions(permissions)

    if step12b["external_confirmatory_gate"]["GGO"]["final_subtype_validation_pass"]:
        raise RuntimeError("Step13 expects GGO to remain locked after Step12B.")
    if step12b["external_confirmatory_gate"]["consolidation"]["final_subtype_validation_pass"]:
        raise RuntimeError("Step13 expects consolidation to remain locked after Step12B.")

    title = contract["mandatory_title"]
    footer = contract["mandatory_footer"]

    source_full = pd.read_csv(PREDICTIONS)
    pred = sanitize_prediction_table(source_full)

    sanitized_csv = RUN / "prediction_only_input.csv"
    atomic_csv(sanitized_csv, pred)

    provenance = {
        "source_run_id": "DX_step08_oof_segmentation_resnet34_unet_seed1705_v1",
        "prediction_only_input_file": str(sanitized_csv.relative_to(ROOT)).replace("\\", "/"),
        "prediction_only_input_sha256": sha256(sanitized_csv),
        "source_step08_table_sha256": sha256(PREDICTIONS),
        "step12b_permissions_sha256": sha256(PERMISSIONS_PATH),
        "report_contract_sha256": sha256(REPORT_CONTRACT_PATH),
        "reference_columns_excluded_before_evidence_construction": True,
    }

    validation_rows = []
    index_rows = []

    for row in pred.itertuples(index=False):
        evidence = build_evidence(row, provenance)

        evidence_path = EVIDENCE_DIR / f"{row.case_id}.json"
        report_path = REPORTS_DIR / f"{row.case_id}.txt"

        atomic_json(evidence_path, evidence)

        report = render_report(evidence, title, footer)
        atomic_text(report_path, report)

        check = validate_report(report, evidence, title, footer)

        validation_rows.append({
            "case_id": row.case_id,
            "valid": check["valid"],
            "errors": "|".join(check["errors"]),
            "left_percent": evidence["validated_measurements"]["Left_lung_involvement_percent"]["value"],
            "right_percent": evidence["validated_measurements"]["Right_lung_involvement_percent"]["value"],
            "total_percent": evidence["validated_measurements"]["Total_lung_involvement_percent"]["value"],
            "seen_percent_values": "|".join(str(x) for x in check["seen_percent_values"]),
        })

        index_rows.append({
            "case_id": row.case_id,
            "evidence_file": str(evidence_path.relative_to(ROOT)).replace("\\", "/"),
            "evidence_sha256": sha256(evidence_path),
            "report_file": str(report_path.relative_to(ROOT)).replace("\\", "/"),
            "report_sha256": sha256(report_path),
        })

    validation_df = pd.DataFrame(validation_rows)
    index_df = pd.DataFrame(index_rows)
    atomic_csv(VALIDATION_CSV, validation_df)
    atomic_csv(REPORT_INDEX_CSV, index_df)

    if not bool(validation_df["valid"].all()):
        bad = validation_df[~validation_df["valid"]]
        raise RuntimeError(
            "At least one deterministic report failed validation:\n"
            + bad[["case_id", "errors"]].to_string(index=False)
        )

    audit = {
        "project": "EViCT-Dx",
        "stage": "STEP_13_PERMISSION_AWARE_STRUCTURED_EVIDENCE_AND_REPORTING",
        "run_id": RUN_ID,
        "completed_utc": now(),
        "cases": int(len(pred)),
        "strict_oof_cases": int(pred.case_id.nunique()),
        "evidence_records": int(len(index_df)),
        "deterministic_reports": int(len(index_df)),
        "reports_valid": int(validation_df.valid.sum()),
        "validation_rate": float(validation_df.valid.mean()),
        "ground_truth_used_for_evidence_construction": False,
        "reference_columns_excluded_before_evidence_construction": True,
        "llm_used": False,
        "validated_numeric_fields_emitted": [
            "Left_lung_involvement_percent",
            "Right_lung_involvement_percent",
            "Total_lung_involvement_percent",
        ],
        "locked_fields_withheld": [
            "etiologic_diagnosis",
            "pathology_subtype",
            "bilateral_involvement",
            "pleural_effusion",
            "lobar_localization",
            "physical_area_volume",
        ],
        "prohibited_fields_not_generated": [
            "Treatment_recommendation",
            "Antibiotic_recommendation",
            "Clinical_management_decision",
        ],
        "mandatory_title_exact": title,
        "mandatory_footer_exact": footer,
        "permissions_sha256": sha256(PERMISSIONS_PATH),
        "report_contract_sha256": sha256(REPORT_CONTRACT_PATH),
        "prediction_only_input_sha256": sha256(sanitized_csv),
        "next_action": CFG["next_stage"],
    }
    atomic_json(AUDIT_JSON, audit)

    state = {
        "run_id": RUN_ID,
        "status": "COMPLETE",
        "updated_utc": now(),
        "cases": int(len(pred)),
        "reports_valid": int(validation_df.valid.sum()),
        "llm_used": False,
        "ground_truth_used_for_evidence_construction": False,
        "GGO_emitted_as_case_finding": False,
        "consolidation_emitted_as_case_finding": False,
        "disease_name_emitted": False,
        "next_action": CFG["next_stage"],
    }
    atomic_json(STATE_JSON, state)

    synced = sync_git("Complete EViCT-Dx Step13 permission-aware evidence and reporting")

    print("\n" + "=" * 112)
    print("✅ STEP 13 COMPLETE — PERMISSION-AWARE REPORTING BASELINE")
    print(f"Strict OOF cases                  : {len(pred)}")
    print(f"Structured evidence records       : {len(index_df)}")
    print(f"Deterministic reports             : {len(index_df)}")
    print(f"Reports passing validator         : {int(validation_df.valid.sum())}/{len(validation_df)}")
    print("Ground truth used in builder      : NO")
    print("Disease names emitted             : NO")
    print("GGO/consolidation findings emitted: NO")
    print("Bilateral classification emitted  : NO")
    print("Treatment recommendation emitted  : NO")
    print("Unlocked numeric fields emitted   : Left / Right / Total involvement %")
    print(f"GitHub metadata sync              : {'SUCCESS' if synced else 'CHECK REQUIRED'}")
    print(f"NEXT                              : {CFG['next_stage']}")
    print("=" * 112)


if __name__ == "__main__":
    main()
