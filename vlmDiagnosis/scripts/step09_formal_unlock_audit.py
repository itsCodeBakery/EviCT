from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DX = ROOT / "vlmDiagnosis"
CFG = json.loads((DX / "config" / "step09_unlock_audit.json").read_text(encoding="utf-8"))

STEP08_JSON = DX / "artifacts" / "audit" / "step08_oof_segmentation_baseline_results.json"
CASE_CSV = DX / "tables" / "step08_oof_case_metrics.csv"
QUANT_CSV = DX / "tables" / "step08_oof_involvement_predictions.csv"
METRICS_JSON = DX / "config" / "metrics_and_unlock_step02.json"
REPORT_CONTRACT_JSON = DX / "config" / "report_field_contract_step02.json"
STEP05_JSON = DX / "artifacts" / "audit" / "step05_diagnosis_unlock_audit.json"

OUT_AUDIT = DX / "artifacts" / "audit" / "step09_segmentation_quantification_unlock_audit.json"
OUT_PERMISSIONS = DX / "config" / "report_field_permissions_step09.json"
OUT_REVIEW = DX / "tables" / "step09_case_failure_review.csv"
OUT_HASHES = DX / "artifacts" / "manifests" / "step09_source_sha256.csv"
RUN = DX / "runs" / "DX_step09_formal_unlock_audit_v1"
STATE = RUN / "STATE.json"

for p in [OUT_AUDIT.parent, OUT_PERMISSIONS.parent, OUT_REVIEW.parent, OUT_HASHES.parent, RUN]:
    p.mkdir(parents=True, exist_ok=True)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, df: pd.DataFrame):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def safe_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() == "true"


def ccc(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    vx, vy = x.var(), y.var()
    cov = np.mean((x - x.mean()) * (y - y.mean()))
    return float(2.0 * cov / (vx + vy + (x.mean() - y.mean()) ** 2 + 1e-12))


def sync_git():
    helper = DX / "scripts" / "git_sync_dx.py"
    if not helper.exists():
        print("⚠ git sync helper missing; outputs remain local.")
        return False
    r = subprocess.run(
        [sys.executable, str(helper), "Complete EViCT-Dx Step09 formal unlock audit"],
        cwd=str(ROOT),
        check=False,
    )
    if r.returncode != 0:
        print("⚠ Git sync did not complete. Preserve the Kaggle session and send the output.")
        return False
    return True


def main():
    print("=" * 108)
    print("EViCT-Dx STEP 09 — FORMAL SEGMENTATION / QUANTIFICATION UNLOCK AUDIT")
    print("AUDIT ONLY: NO TRAINING, NO THRESHOLD SEARCH, NO TEST RE-INFERENCE")
    print("=" * 108)

    # Core lock is checked again before producing derived permissions.
    subprocess.run(
        [sys.executable, str(DX / "scripts" / "verify_core_lock.py")],
        cwd=str(ROOT),
        check=True,
    )

    required = [STEP08_JSON, CASE_CSV, QUANT_CSV, METRICS_JSON, REPORT_CONTRACT_JSON, STEP05_JSON]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Required frozen source files missing: {missing}")

    step08 = json.loads(STEP08_JSON.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    contract = json.loads(REPORT_CONTRACT_JSON.read_text(encoding="utf-8"))
    step05 = json.loads(STEP05_JSON.read_text(encoding="utf-8"))
    cases = pd.read_csv(CASE_CSV)
    quant = pd.read_csv(QUANT_CSV)

    # Strict OOF integrity checks.
    if int(step08.get("strict_oof_cases", -1)) != 20:
        raise RuntimeError("Step08 summary does not declare exactly 20 strict OOF cases.")
    if len(cases) != 20 or cases.case_id.nunique() != 20:
        raise RuntimeError("Step08 case metrics are not exactly 20 unique OOF cases.")
    if len(quant) != 20 or quant.case_id.nunique() != 20:
        raise RuntimeError("Step08 quantification is not exactly 20 unique OOF cases.")
    if set(cases.case_id) != set(quant.case_id):
        raise RuntimeError("Step08 segmentation and quantification case IDs do not match.")

    # Verify that the published summary agrees with the CSVs.
    recomputed = {
        "mean_Dice_left": float(cases.Dice_left.mean()),
        "mean_Dice_right": float(cases.Dice_right.mean()),
        "mean_Dice_infection": float(cases.Dice_infection.mean()),
        "mean_IoU_infection": float(cases.IoU_infection.mean()),
        "mean_infection_sensitivity": float(cases.infection_sensitivity.mean()),
        "mean_infection_specificity": float(cases.infection_specificity.mean()),
    }
    for k, v in recomputed.items():
        if not np.isclose(v, float(step08[k]), atol=1e-10, rtol=0):
            raise RuntimeError(f"Step08 summary mismatch for {k}: CSV={v}, JSON={step08[k]}")

    quant_recomputed = {}
    for side in ["left", "right", "total"]:
        ref = quant[f"reference_{side}_percent"].to_numpy(dtype=float)
        pred = quant[f"predicted_{side}_percent"].to_numpy(dtype=float)
        err = pred - ref
        quant_recomputed[side] = {
            "MAE_percentage_points": float(np.mean(np.abs(err))),
            "CCC": ccc(ref, pred),
        }
        frozen = step08["quantification"][side]
        for k in ["MAE_percentage_points", "CCC"]:
            if not np.isclose(quant_recomputed[side][k], float(frozen[k]), atol=1e-10, rtol=0):
                raise RuntimeError(
                    f"Step08 quantification mismatch for {side}/{k}: "
                    f"CSV={quant_recomputed[side][k]}, JSON={frozen[k]}"
                )

    qcrit = metrics["lung_involvement_quantification"]["report_unlock_criteria"]
    q_mae_max = float(qcrit["MAE_percentage_points_max"])
    q_ccc_min = float(qcrit["CCC_min"])

    quant_gates = {}
    for side in ["left", "right", "total"]:
        q = step08["quantification"][side]
        mae_ok = float(q["MAE_percentage_points"]) <= q_mae_max
        ccc_ok = float(q["CCC"]) >= q_ccc_min
        quant_gates[side] = {
            "MAE_percentage_points": float(q["MAE_percentage_points"]),
            "MAE_required_max": q_mae_max,
            "MAE_pass": bool(mae_ok),
            "CCC": float(q["CCC"]),
            "CCC_required_min": q_ccc_min,
            "CCC_pass": bool(ccc_ok),
            "field_gate_pass": bool(mae_ok and ccc_ok),
        }

    bcrit = metrics["bilateral_involvement"]["unlock_criteria"]
    b = step08["bilateral"]
    bilateral_gate = {
        "balanced_accuracy": float(b["balanced_accuracy"]),
        "balanced_accuracy_required_min": float(bcrit["balanced_accuracy_min"]),
        "balanced_accuracy_pass": float(b["balanced_accuracy"]) >= float(bcrit["balanced_accuracy_min"]),
        "sensitivity": float(b["sensitivity"]),
        "sensitivity_required_min": float(bcrit["sensitivity_min"]),
        "sensitivity_pass": float(b["sensitivity"]) >= float(bcrit["sensitivity_min"]),
        "specificity": float(b["specificity"]),
        "specificity_required_min": float(bcrit["specificity_min"]),
        "specificity_pass": float(b["specificity"]) >= float(bcrit["specificity_min"]),
    }
    bilateral_gate["field_gate_pass"] = bool(
        bilateral_gate["balanced_accuracy_pass"]
        and bilateral_gate["sensitivity_pass"]
        and bilateral_gate["specificity_pass"]
    )

    diagnosis_unlocked = bool(step05.get("diagnostic_disease_names_unlocked", False))

    # Failure review is descriptive only; it cannot be used to tune Step08.
    merged = cases.merge(quant, on=["outer_fold", "case_id"], how="inner")
    merged["total_involvement_absolute_error_pp"] = (
        merged["predicted_total_percent"] - merged["reference_total_percent"]
    ).abs()
    merged["bilateral_mismatch"] = (
        merged["reference_bilateral"].map(safe_bool) != merged["predicted_bilateral"].map(safe_bool)
    )
    merged["infection_dice_rank_low_is_worse"] = merged["Dice_infection"].rank(method="min", ascending=True)
    merged["total_error_rank_high_is_worse"] = merged[
        "total_involvement_absolute_error_pp"
    ].rank(method="min", ascending=False)
    merged["review_priority"] = (
        merged["bilateral_mismatch"].astype(int) * 100
        + (1.0 - merged["Dice_infection"]) * 10
        + merged["total_involvement_absolute_error_pp"] / 10.0
    )
    review_cols = [
        "outer_fold", "case_id", "source",
        "Dice_left", "Dice_right", "Dice_infection", "IoU_infection",
        "infection_sensitivity", "infection_specificity",
        "reference_total_percent", "predicted_total_percent",
        "total_involvement_absolute_error_pp",
        "reference_bilateral", "predicted_bilateral", "bilateral_mismatch",
        "infection_dice_rank_low_is_worse", "total_error_rank_high_is_worse",
        "review_priority",
    ]
    review = merged[review_cols].sort_values("review_priority", ascending=False)
    atomic_csv(OUT_REVIEW, review)

    hashes = pd.DataFrame([
        {
            "relative_path": str(p.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(p),
            "bytes": p.stat().st_size,
        }
        for p in required
    ])
    atomic_csv(OUT_HASHES, hashes)

    permission_fields = dict(contract["fields"])
    for disease in ["COVID-19", "CAP", "Normal"]:
        permission_fields[disease] = (
            "UNLOCKED" if diagnosis_unlocked
            else "LOCKED_FAILED_DIAGNOSIS_VALIDATION"
        )

    mapping = {
        "left": "Left_lung_involvement_percent",
        "right": "Right_lung_involvement_percent",
        "total": "Total_lung_involvement_percent",
    }
    for side, field in mapping.items():
        permission_fields[field] = (
            "UNLOCKED_FOR_PROVISIONAL_AI_REPORT_WITH_MANDATORY_RADIOLOGIST_REVIEW"
            if quant_gates[side]["field_gate_pass"]
            else "LOCKED_FAILED_OOF_QUANTIFICATION_VALIDATION"
        )

    permission_fields["Bilateral_involvement"] = (
        "UNLOCKED_FOR_PROVISIONAL_AI_REPORT_WITH_MANDATORY_RADIOLOGIST_REVIEW"
        if bilateral_gate["field_gate_pass"]
        else "LOCKED_FAILED_OOF_VALIDATION"
    )

    permissions = {
        "project": "EViCT-Dx",
        "stage": "STEP_09_DERIVED_REPORT_FIELD_PERMISSIONS",
        "created_utc": now(),
        "derived_from_frozen_contract": "vlmDiagnosis/config/report_field_contract_step02.json",
        "mandatory_title": contract["mandatory_title"],
        "mandatory_footer": contract["mandatory_footer"],
        "fields": permission_fields,
        "important_scope": (
            "Unlock means the field passed the pre-specified research protocol for use in the "
            "Provisional AI Diagnostic Report. It is not autonomous clinical clearance and the "
            "mandatory radiologist-review statement remains required."
        ),
        "generic_infection_mask": (
            "EVALUATED_FOR_QUANTIFICATION_ONLY_NO_INDEPENDENT_REPORT_FIELD_UNLOCK_RULE"
        ),
        "left_right_lung_masks": (
            "EVALUATED_NO_FROZEN_REPORT_FIELD_UNLOCK_THRESHOLD"
        ),
        "oof_results_may_not_be_used_for_further_threshold_tuning": True,
    }
    atomic_json(OUT_PERMISSIONS, permissions)

    audit = {
        "project": "EViCT-Dx",
        "stage": "STEP_09_FORMAL_SEGMENTATION_QUANTIFICATION_UNLOCK_AUDIT",
        "run_id": "DX_step09_formal_unlock_audit_v1",
        "completed_utc": now(),
        "status": "PASS_AUDIT_COMPLETED",
        "training_performed": False,
        "threshold_search_performed": False,
        "test_reinference_performed": False,
        "strict_oof_cases_verified": 20,
        "source_hash_manifest": str(OUT_HASHES.relative_to(ROOT)).replace("\\", "/"),
        "step08_segmentation_summary": recomputed,
        "quantification_gates": quant_gates,
        "bilateral_gate": bilateral_gate,
        "diagnosis_disease_names_unlocked": diagnosis_unlocked,
        "infection_segmentation_interpretation": {
            "mean_Dice": float(step08["mean_Dice_infection"]),
            "mean_IoU": float(step08["mean_IoU_infection"]),
            "mean_sensitivity": float(step08["mean_infection_sensitivity"]),
            "mean_specificity": float(step08["mean_infection_specificity"]),
            "report_unlock_decision": "NO_INDEPENDENT_GENERIC_INFECTION_REPORT_FIELD_EXISTS",
        },
        "lung_segmentation_interpretation": {
            "mean_Dice_left": float(step08["mean_Dice_left"]),
            "mean_Dice_right": float(step08["mean_Dice_right"]),
            "report_unlock_decision": "NO_FROZEN_REPORT_FIELD_THRESHOLD_EXISTS",
        },
        "report_permissions_file": str(OUT_PERMISSIONS.relative_to(ROOT)).replace("\\", "/"),
        "failure_review_file": str(OUT_REVIEW.relative_to(ROOT)).replace("\\", "/"),
        "failure_review_is_descriptive_only_not_for_tuning": True,
        "next_action": (
            "FREEZE_STEP09_PERMISSIONS_THEN_DEVELOP_NEXT_COMPONENT_WITHOUT_TUNING_ON_STEP08_OOF_LABELS"
        ),
    }
    atomic_json(OUT_AUDIT, audit)

    state = {
        "run_id": "DX_step09_formal_unlock_audit_v1",
        "status": "COMPLETE",
        "updated_utc": now(),
        "strict_oof_cases_verified": 20,
        "left_involvement_unlocked": quant_gates["left"]["field_gate_pass"],
        "right_involvement_unlocked": quant_gates["right"]["field_gate_pass"],
        "total_involvement_unlocked": quant_gates["total"]["field_gate_pass"],
        "bilateral_involvement_unlocked": bilateral_gate["field_gate_pass"],
        "diagnosis_disease_names_unlocked": diagnosis_unlocked,
        "next_action": audit["next_action"],
    }
    atomic_json(STATE, state)

    synced = sync_git()

    print("\n" + "=" * 108)
    print("✅ STEP 09 COMPLETE — FORMAL UNLOCK AUDIT")
    print(f"Left involvement %        : {'UNLOCKED' if quant_gates['left']['field_gate_pass'] else 'LOCKED'}")
    print(f"Right involvement %       : {'UNLOCKED' if quant_gates['right']['field_gate_pass'] else 'LOCKED'}")
    print(f"Total involvement %       : {'UNLOCKED' if quant_gates['total']['field_gate_pass'] else 'LOCKED'}")
    print(f"Bilateral involvement     : {'UNLOCKED' if bilateral_gate['field_gate_pass'] else 'LOCKED'}")
    print(f"COVID/CAP/Normal names    : {'UNLOCKED' if diagnosis_unlocked else 'LOCKED'}")
    print(f"Generic infection Dice    : {step08['mean_Dice_infection']:.4f} (descriptive; no report-field gate)")
    print("OOF threshold retuning    : PROHIBITED")
    print(f"GitHub metadata sync      : {'SUCCESS' if synced else 'CHECK REQUIRED'}")
    print("NEXT                      : freeze permissions and continue to the next independent component")
    print("=" * 108)


if __name__ == "__main__":
    main()
