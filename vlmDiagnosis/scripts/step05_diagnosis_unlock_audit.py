from __future__ import annotations

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
TABLES = DX / "tables"
AUDIT = DX / "artifacts" / "audit"
RUN_ID = "DX_step05_diagnosis_unlock_audit_v1"
RUN = DX / "runs" / RUN_ID
RUN.mkdir(parents=True, exist_ok=True)

STEP04_RESULT = AUDIT / "step04_diagnosis_baseline_results.json"
VAL_PRED = TABLES / "step04_diagnosis_validation_predictions.csv"
METRICS_CFG = DX / "config" / "metrics_and_unlock_step02.json"

CLASSES = ["COVID-19", "CAP", "Normal"]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sync_git(message):
    script = DX / "scripts" / "git_sync_dx.py"
    if not script.exists():
        print("⚠ git_sync_dx.py missing; local artifacts retained.")
        return
    subprocess.run([sys.executable, str(script), message], cwd=str(ROOT), check=False)


def threshold_sweep(df: pd.DataFrame):
    probs = df[[f"prob_{c}" for c in CLASSES]].to_numpy(float)
    pred_idx = probs.argmax(1)
    conf = probs.max(1)
    y = df["target"].to_numpy(int)

    rows = []
    for thr in np.round(np.arange(0.30, 0.951, 0.01), 2):
        accepted = conf >= thr
        n = int(accepted.sum())
        coverage = float(accepted.mean())

        class_ppv = {}
        feasible_ppv = True
        for ci, cls in enumerate(CLASSES):
            m = accepted & (pred_idx == ci)
            if int(m.sum()) == 0:
                ppv = np.nan
                feasible_ppv = False
            else:
                ppv = float((y[m] == ci).mean())
                if ppv < 0.90:
                    feasible_ppv = False
            class_ppv[cls] = ppv

        rows.append({
            "threshold": float(thr),
            "accepted_n": n,
            "coverage": coverage,
            "COVID-19_PPV": class_ppv["COVID-19"],
            "CAP_PPV": class_ppv["CAP"],
            "Normal_PPV": class_ppv["Normal"],
            "all_class_PPV_ge_0.90": bool(feasible_ppv),
            "meets_min_coverage_0.25": bool(coverage >= 0.25),
            "frozen_rule_feasible": bool(feasible_ppv and coverage >= 0.25),
        })

    return pd.DataFrame(rows)


def per_class_thresholds(df: pd.DataFrame):
    probs = df[[f"prob_{c}" for c in CLASSES]].to_numpy(float)
    pred_idx = probs.argmax(1)
    y = df["target"].to_numpy(int)

    out = []
    for ci, cls in enumerate(CLASSES):
        best = None
        for thr in np.round(np.arange(0.30, 0.951, 0.01), 2):
            m = (pred_idx == ci) & (probs[:, ci] >= thr)
            n = int(m.sum())
            if n == 0:
                continue
            ppv = float((y[m] == ci).mean())
            overall_coverage = float(m.mean())
            if ppv >= 0.90:
                candidate = {
                    "class": cls,
                    "threshold": float(thr),
                    "accepted_n": n,
                    "PPV": ppv,
                    "overall_validation_coverage": overall_coverage,
                }
                if best is None or n > best["accepted_n"] or (
                    n == best["accepted_n"] and thr < best["threshold"]
                ):
                    best = candidate

        if best is None:
            best = {
                "class": cls,
                "threshold": np.nan,
                "accepted_n": 0,
                "PPV": np.nan,
                "overall_validation_coverage": 0.0,
            }
        out.append(best)

    return pd.DataFrame(out)


def main():
    print("=" * 100)
    print("EViCT-Dx STEP 05 — DIAGNOSIS UNLOCK + ABSTENTION AUDIT")
    print("=" * 100)

    verify = DX / "scripts" / "verify_core_lock.py"
    if verify.exists():
        subprocess.run([sys.executable, str(verify)], cwd=str(ROOT), check=True)

    for p in [STEP04_RESULT, VAL_PRED, METRICS_CFG]:
        if not p.exists():
            raise FileNotFoundError(p)

    result = json.loads(STEP04_RESULT.read_text(encoding="utf-8"))
    cfg = json.loads(METRICS_CFG.read_text(encoding="utf-8"))
    val = pd.read_csv(VAL_PRED)

    frozen = cfg["diagnosis"]["global_report_unlock_criteria"]
    tm = result["test_metrics"]

    criteria = {
        "held_out_macro_F1": {
            "value": float(tm["macro_F1"]),
            "required": float(frozen["held_out_macro_F1_min"]),
            "pass": bool(tm["macro_F1"] >= frozen["held_out_macro_F1_min"]),
        },
        "held_out_ECE": {
            "value": float(tm["ECE"]),
            "required_max": float(frozen["held_out_ECE_max"]),
            "pass": bool(tm["ECE"] <= frozen["held_out_ECE_max"]),
        },
    }

    for cls in CLASSES:
        pc = tm["per_class"][cls]
        criteria[f"{cls}_AUROC"] = {
            "value": float(pc["AUROC"]),
            "required": float(frozen["held_out_per_class_AUROC_min"]),
            "pass": bool(pc["AUROC"] >= frozen["held_out_per_class_AUROC_min"]),
        }
        criteria[f"{cls}_sensitivity"] = {
            "value": float(pc["sensitivity"]),
            "required": float(frozen["held_out_per_class_sensitivity_min"]),
            "pass": bool(pc["sensitivity"] >= frozen["held_out_per_class_sensitivity_min"]),
        }
        criteria[f"{cls}_specificity"] = {
            "value": float(pc["specificity"]),
            "required": float(frozen["held_out_per_class_specificity_min"]),
            "pass": bool(pc["specificity"] >= frozen["held_out_per_class_specificity_min"]),
        }

    sweep = threshold_sweep(val)
    atomic_csv(TABLES / "step05_diagnosis_abstention_threshold_sweep.csv", sweep)

    pct = per_class_thresholds(val)
    atomic_csv(TABLES / "step05_diagnosis_class_specific_thresholds.csv", pct)

    feasible = sweep[sweep["frozen_rule_feasible"] == True]  # noqa: E712
    selected = None
    if len(feasible):
        # Maximum validation coverage; ties choose the lower threshold.
        selected_row = feasible.sort_values(["coverage", "threshold"], ascending=[False, True]).iloc[0]
        selected = selected_row.to_dict()

    all_unlock = all(v["pass"] for v in criteria.values())

    audit = {
        "project": "EViCT-Dx",
        "stage": "STEP_05_DIAGNOSIS_UNLOCK_AUDIT",
        "run_id": RUN_ID,
        "completed_utc": now(),
        "source_result": "vlmDiagnosis/artifacts/audit/step04_diagnosis_baseline_results.json",
        "test_inference_repeated": False,
        "held_out_test_predictions_reused_for_tuning": False,
        "unlock_criteria": criteria,
        "all_global_unlock_criteria_pass": bool(all_unlock),
        "diagnostic_disease_names_unlocked": bool(all_unlock),
        "validation_only_abstention": {
            "target": "all predicted classes PPV >= 0.90 if feasible",
            "minimum_validation_coverage": 0.25,
            "global_threshold_feasible": bool(selected is not None),
            "selected_global_threshold": selected,
            "note": "Threshold search uses validation predictions only. It cannot override failed held-out global report-unlock criteria.",
        },
        "next_action": (
            "DIAGNOSIS_REPORT_FIELDS_MAY_BE_REVIEWED_FOR_UNLOCK"
            if all_unlock
            else "KEEP_DIAGNOSIS_FIELDS_LOCKED_AND_DEVELOP_STRONGER_MODEL_WITHOUT_REUSING_HELD_OUT_TEST"
        ),
    }

    atomic_json(AUDIT / "step05_diagnosis_unlock_audit.json", audit)
    atomic_json(RUN / "STATE.json", {
        "run_id": RUN_ID,
        "status": "COMPLETE",
        "updated_utc": now(),
        "test_inference_repeated": False,
        "diagnostic_disease_names_unlocked": bool(all_unlock),
        "next_action": audit["next_action"],
    })

    sync_git("Complete EViCT-Dx Step05 diagnosis unlock and abstention audit")

    print("\nFrozen report-unlock criteria:")
    for name, item in criteria.items():
        symbol = "PASS" if item["pass"] else "FAIL"
        req = item.get("required", item.get("required_max"))
        print(f"  {symbol:4s}  {name:28s} value={item['value']:.4f}  criterion={req}")

    print("\nValidation-only abstention:")
    if selected is None:
        print("  No global threshold achieved PPV >= 0.90 for all three predicted classes")
        print("  while retaining >=25% validation coverage.")
    else:
        print(f"  Selected threshold: {selected['threshold']:.2f}")
        print(f"  Coverage          : {selected['coverage']:.3f}")
        print(f"  COVID-19 PPV      : {selected['COVID-19_PPV']:.3f}")
        print(f"  CAP PPV           : {selected['CAP_PPV']:.3f}")
        print(f"  Normal PPV        : {selected['Normal_PPV']:.3f}")

    print("\n" + "=" * 100)
    print("✅ STEP 05 COMPLETE")
    print("Diagnostic disease-name fields:", "UNLOCKED" if all_unlock else "STILL LOCKED")
    print("Held-out test inference was NOT repeated.")
    if not all_unlock:
        print("NEXT: stronger diagnosis development using train/validation only; do not tune on the 61-patient test.")
    print("=" * 100)


if __name__ == "__main__":
    main()
