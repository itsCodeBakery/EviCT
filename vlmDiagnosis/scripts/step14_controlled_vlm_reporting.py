from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
DX = ROOT / "vlmDiagnosis"

CFG_PATH = DX / "config" / "step14_controlled_vlm_reporting.json"
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))

RUN_ID = CFG["run_id"]
RUN = DX / "runs" / RUN_ID
RUN.mkdir(parents=True, exist_ok=True)

EVIDENCE_DIR = ROOT / CFG["inputs"]["evidence_dir"]
DET_DIR = ROOT / CFG["inputs"]["deterministic_reports_dir"]
STEP13_AUDIT = ROOT / CFG["inputs"]["step13_audit"]
PERMISSIONS_PATH = ROOT / CFG["inputs"]["permissions"]
REPORT_CONTRACT_PATH = ROOT / CFG["inputs"]["report_contract"]

RAW_DIR = ROOT / CFG["outputs"]["raw_dir"]
FINAL_DIR = ROOT / CFG["outputs"]["final_dir"]
RAW_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_CSV = ROOT / CFG["outputs"]["validation_table"]
INDEX_CSV = ROOT / CFG["outputs"]["report_index"]
AUDIT_JSON = ROOT / CFG["outputs"]["audit"]
STATE_JSON = ROOT / CFG["outputs"]["state"]
PROGRESS_JSON = ROOT / CFG["outputs"]["progress"]

MODEL_ID = CFG["model"]["repository"]


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
    helper = DX / "scripts" / "git_sync_dx.py"
    r = subprocess.run([sys.executable, str(helper), message], cwd=str(ROOT), check=False)
    if r.returncode != 0:
        print(f"⚠ metadata Git sync returned {r.returncode}")
    return r.returncode == 0


def verify_upstream():
    subprocess.run(
        [sys.executable, str(DX / "scripts" / "verify_core_lock.py")],
        cwd=str(ROOT),
        check=True,
    )

    for p in [
        EVIDENCE_DIR, DET_DIR, STEP13_AUDIT,
        PERMISSIONS_PATH, REPORT_CONTRACT_PATH
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    step13 = json.loads(STEP13_AUDIT.read_text())
    if int(step13.get("cases", -1)) != 20:
        raise RuntimeError("Step13 does not contain exactly 20 cases.")
    if int(step13.get("reports_valid", -1)) != 20:
        raise RuntimeError("Step13 deterministic baseline is not 20/20 valid.")
    if bool(step13.get("ground_truth_used_for_evidence_construction")):
        raise RuntimeError("Step13 unexpectedly reports ground-truth use.")
    if bool(step13.get("llm_used")):
        raise RuntimeError("Step13 baseline provenance drift: llm_used is true.")

    permissions = json.loads(PERMISSIONS_PATH.read_text())
    allowed = {
        "Left_lung_involvement_percent",
        "Right_lung_involvement_percent",
        "Total_lung_involvement_percent",
    }
    unlocked = {
        k for k, v in permissions["fields"].items()
        if str(v).startswith("UNLOCKED")
    }
    if not allowed.issubset(unlocked):
        raise RuntimeError("Required involvement fields are not all unlocked.")

    forbidden = [
        "COVID-19", "CAP", "Normal", "GGO", "Consolidation",
        "Bilateral_involvement", "Pleural_effusion_slice_level",
    ]
    if any(str(permissions["fields"].get(k, "")).startswith("UNLOCKED") for k in forbidden):
        raise RuntimeError("Unexpected permission drift before Step14.")

    evidence_files = sorted(EVIDENCE_DIR.glob("*.json"))
    det_files = sorted(DET_DIR.glob("*.txt"))
    if len(evidence_files) != 20 or len(det_files) != 20:
        raise RuntimeError(
            f"Expected 20 evidence JSONs and 20 deterministic reports; "
            f"got {len(evidence_files)} and {len(det_files)}."
        )

    return permissions, json.loads(REPORT_CONTRACT_PATH.read_text())


def compact_evidence(e):
    m = e["validated_measurements"]
    return {
        "validated_measurements": {
            "left_lung_involvement_percent": m["Left_lung_involvement_percent"]["value"],
            "right_lung_involvement_percent": m["Right_lung_involvement_percent"]["value"],
            "total_lung_involvement_percent": m["Total_lung_involvement_percent"]["value"],
        },
        "validation_status": {
            "etiologic_diagnosis": "WITHHELD",
            "pathology_subtype": "WITHHELD",
            "bilateral_involvement": "WITHHELD",
            "pleural_effusion": "WITHHELD",
            "lobar_localization": "WITHHELD",
            "physical_area_or_volume": "WITHHELD",
            "treatment_or_management_recommendation": "PROHIBITED",
        },
    }


def build_prompt(e, title, footer):
    compact = compact_evidence(e)
    evidence_text = json.dumps(compact, indent=2)

    system = (
        "You are a controlled medical-report wording module in a research pipeline. "
        "You are NOT allowed to diagnose, infer, add, or recommend anything. "
        "Your only task is to convert validated structured evidence into a concise provisional report. "
        "If a field is withheld or prohibited, do not guess its value."
    )

    user = f"""Create one provisional report from the structured evidence below.

STRICT OUTPUT RULES:
1. The first line must be exactly:
{title}

2. Include a Findings section.
3. State exactly these three validated percentages, each exactly once and with one decimal place:
   - left lung involvement
   - right lung involvement
   - total lung involvement
4. Do not include ANY other numeric value.
5. Do not name or assert any etiologic disease.
6. Do not name or assert any pathology subtype.
7. Do not assert bilateral involvement.
8. Do not assert pleural effusion or lobar location.
9. Do not state physical area or volume.
10. Do not provide treatment, antibiotic, therapy, or management recommendations.
11. Do not include a patient/case identifier.
12. Include a short Validation-aware limitations section saying that unavailable diagnostic/subtype/anatomical fields are withheld because validation prerequisites were not met.
13. The final paragraph must be copied verbatim, with no characters added after it:
{footer}

Do not use markdown bullets, tables, code fences, headings with # symbols, or commentary outside the report.

VALIDATED STRUCTURED EVIDENCE:
{evidence_text}
"""
    return system, user


def load_model_and_processor():
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except Exception as exc:
        raise RuntimeError(
            "Step14 requires transformers with Qwen2.5-VL support. "
            "Install transformers>=4.49,<4.58 and accelerate, then rerun."
        ) from exc

    dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16

    print(f"Loading frozen report realizer: {MODEL_ID}")
    print(f"torch dtype: {dtype}")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


@torch.inference_mode()
def generate_one(model, processor, system_text, user_text):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        },
    ]

    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[chat_text],
        padding=True,
        return_tensors="pt",
    )

    # Move tensor inputs to the model's first parameter device.
    device = next(model.parameters()).device
    inputs = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }

    generated = model.generate(
        **inputs,
        max_new_tokens=int(CFG["model"]["max_new_tokens"]),
        do_sample=False,
        use_cache=True,
    )

    input_len = inputs["input_ids"].shape[1]
    trimmed = generated[:, input_len:]

    text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return text.strip()


def exact_expected_percent_strings(e):
    m = e["validated_measurements"]
    vals = [
        float(m["Left_lung_involvement_percent"]["value"]),
        float(m["Right_lung_involvement_percent"]["value"]),
        float(m["Total_lung_involvement_percent"]["value"]),
    ]
    return [f"{v:.1f}%" for v in vals]


def validate_report(report, e, title, footer):
    errors = []
    raw = report.strip()

    title_ok = raw.startswith(title + "\n")
    footer_ok = raw.endswith(footer)
    if not title_ok:
        errors.append("TITLE_NOT_EXACT")
    if not footer_ok:
        errors.append("FOOTER_NOT_EXACT")

    expected = exact_expected_percent_strings(e)

    # The three allowed percentage literals must appear exactly once each and in order.
    seen_percent = re.findall(r"(?<![\d.])\d{1,3}(?:\.\d+)?%", raw)
    numeric_copy_ok = seen_percent == expected
    if not numeric_copy_ok:
        errors.append(
            f"PROTECTED_PERCENT_MISMATCH expected={expected} seen={seen_percent}"
        )

    # Strip the exact protected percentage strings, then reject every remaining digit.
    scrubbed = raw
    for s in expected:
        scrubbed = scrubbed.replace(s, "", 1)
    extra_digits = re.findall(r"\d", scrubbed)
    no_extra_numbers = len(extra_digits) == 0
    if not no_extra_numbers:
        errors.append("UNSUPPORTED_NUMERIC_LITERAL")

    lower = raw.lower()

    required_terms_ok = (
        "left lung" in lower
        and "right lung" in lower
        and "total lung" in lower
        and "validation-aware limitations" in lower
    )
    if not required_terms_ok:
        errors.append("REQUIRED_STRUCTURE_OR_MEASUREMENT_LABEL_MISSING")

    # Remove the mandatory footer before recommendation scanning because the footer
    # legitimately contains the phrase "treatment planning".
    body_for_safety = raw
    if footer in body_for_safety:
        body_for_safety = body_for_safety.replace(footer, "")
    body_lower = body_for_safety.lower()

    locked_patterns = {
        "DISEASE_NAME_COVID": r"\bcovid\b|\bcoronavirus\b",
        "DISEASE_NAME_CAP": r"\bcommunity[- ]acquired pneumonia\b|\bcap\b",
        "NORMAL_DIAGNOSIS": r"\bnormal (scan|ct|lungs?)\b|\bdiagnosis\s*:\s*normal\b",
        "SUBTYPE_GGO": r"\bggo\b|\bground[- ]glass",
        "SUBTYPE_CONSOLIDATION": r"\bconsolidation\b",
        "PLEURAL_EFFUSION": r"\bpleural effusion\b",
        "BILATERAL_ASSERTION": r"\bbilateral\b",
        "LOBAR_ASSERTION": r"\b(lobar|lobe)\b",
        "PHYSICAL_AREA_VOLUME": r"\b(mm2|mm²|cm3|cm³|millimeter|centimeter|volume)\b",
    }

    locked_hits = []
    for name, pattern in locked_patterns.items():
        if re.search(pattern, body_lower, flags=re.I):
            locked_hits.append(name)
            errors.append(f"LOCKED_FIELD_MENTION:{name}")

    recommendation_patterns = [
        r"\brecommend\b",
        r"\bantibiotic",
        r"\btherapy\b",
        r"\btreatment\b",
        r"\bmanagement\b",
        r"\bfollow[- ]?up\b",
        r"\bmedication\b",
    ]
    recommendation_hits = []
    for pattern in recommendation_patterns:
        if re.search(pattern, body_lower, flags=re.I):
            recommendation_hits.append(pattern)
            errors.append(f"PROHIBITED_RECOMMENDATION_OR_MANAGEMENT:{pattern}")

    case_id = str(e.get("case_id", ""))
    case_id_ok = bool(case_id) and case_id not in raw
    if not case_id_ok:
        errors.append("CASE_ID_EMITTED")

    radiologist_ok = "radiologist" in lower
    if not radiologist_ok:
        errors.append("RADIOLOGIST_REVIEW_REQUIREMENT_MISSING")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "title_ok": title_ok,
        "footer_ok": footer_ok,
        "numeric_copy_ok": numeric_copy_ok,
        "no_extra_numbers": no_extra_numbers,
        "required_terms_ok": required_terms_ok,
        "locked_field_violation": len(locked_hits) > 0,
        "locked_field_hits": locked_hits,
        "prohibited_recommendation_violation": len(recommendation_hits) > 0,
        "recommendation_hits": recommendation_hits,
        "case_id_not_emitted": case_id_ok,
        "radiologist_ok": radiologist_ok,
        "expected_percent_strings": expected,
        "seen_percent_strings": seen_percent,
    }


def main():
    print("=" * 114)
    print("EViCT-Dx STEP 14 — CONTROLLED QWEN2.5-VL EVIDENCE REALIZATION")
    print("TEXT-ONLY STRUCTURED EVIDENCE INPUT — NO CT IMAGE — DETERMINISTIC FALLBACK")
    print("=" * 114)

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU before Step14.")

    permissions, contract = verify_upstream()
    title = contract["mandatory_title"]
    footer = contract["mandatory_footer"]

    if STATE_JSON.exists():
        prior = json.loads(STATE_JSON.read_text())
        if prior.get("status") == "COMPLETE":
            print("✓ Step14 is already COMPLETE; refusing to regenerate outputs.")
            print(json.dumps(prior, indent=2))
            return

    evidence_files = sorted(EVIDENCE_DIR.glob("*.json"))
    cases = [p.stem for p in evidence_files]

    progress = {
        "run_id": RUN_ID,
        "model": MODEL_ID,
        "model_role": "text_only_evidence_realizer",
        "visual_inputs_provided": False,
        "config_sha256": sha256(CFG_PATH),
        "completed_cases": {},
    }
    if PROGRESS_JSON.exists():
        prior_progress = json.loads(PROGRESS_JSON.read_text())
        if prior_progress.get("config_sha256") != sha256(CFG_PATH):
            raise RuntimeError("Step14 progress/config hash mismatch; refusing unsafe resume.")
        progress = prior_progress

    state = {
        "run_id": RUN_ID,
        "status": "RUNNING",
        "updated_utc": now(),
        "model": MODEL_ID,
        "visual_inputs_provided": False,
        "training_performed": False,
        "fine_tuning_performed": False,
        "completed_cases": len(progress["completed_cases"]),
        "total_cases": len(cases),
        "next_action": "CONTINUE_CONTROLLED_VLM_REALIZATION",
    }
    atomic_json(STATE_JSON, state)
    sync_git("Start EViCT-Dx Step14 controlled VLM realization")

    pending = [
        p for p in evidence_files
        if p.stem not in progress["completed_cases"]
    ]

    model = processor = None
    if pending:
        model, processor = load_model_and_processor()

    print(f"Cases total={len(cases)} already_complete={len(cases)-len(pending)} pending={len(pending)}")

    for idx, evidence_path in enumerate(pending, start=1):
        case_id = evidence_path.stem
        evidence = json.loads(evidence_path.read_text())

        if evidence.get("case_id") != case_id:
            raise RuntimeError(f"Evidence filename/case_id mismatch: {case_id}")

        det_path = DET_DIR / f"{case_id}.txt"
        if not det_path.exists():
            raise FileNotFoundError(det_path)
        deterministic = det_path.read_text().strip()

        # Safety-check deterministic fallback before calling the model.
        det_check = validate_report(deterministic, evidence, title, footer)
        if not det_check["valid"]:
            raise RuntimeError(
                f"Frozen Step13 fallback unexpectedly fails Step14 validator for {case_id}: "
                f"{det_check['errors']}"
            )

        system_text, user_text = build_prompt(evidence, title, footer)

        t0 = time.time()
        raw = generate_one(model, processor, system_text, user_text)
        generation_seconds = time.time() - t0

        raw_path = RAW_DIR / f"{case_id}.txt"
        atomic_text(raw_path, raw + "\n")

        raw_check = validate_report(raw, evidence, title, footer)

        if raw_check["valid"]:
            final = raw
            fallback_used = False
            final_source = "VLM_ACCEPTED"
        else:
            final = deterministic
            fallback_used = True
            final_source = "STEP13_DETERMINISTIC_FALLBACK"

        final_check = validate_report(final, evidence, title, footer)
        if not final_check["valid"]:
            raise RuntimeError(
                f"Final report invalid after fallback for {case_id}: {final_check['errors']}"
            )

        final_path = FINAL_DIR / f"{case_id}.txt"
        atomic_text(final_path, final.strip() + "\n")

        progress["completed_cases"][case_id] = {
            "completed_utc": now(),
            "generation_seconds": generation_seconds,
            "raw_report_sha256": sha256(raw_path),
            "raw_valid": raw_check["valid"],
            "raw_errors": raw_check["errors"],
            "raw_title_ok": raw_check["title_ok"],
            "raw_footer_ok": raw_check["footer_ok"],
            "raw_numeric_copy_ok": raw_check["numeric_copy_ok"],
            "raw_no_extra_numbers": raw_check["no_extra_numbers"],
            "raw_locked_field_violation": raw_check["locked_field_violation"],
            "raw_prohibited_recommendation_violation": raw_check["prohibited_recommendation_violation"],
            "fallback_used": fallback_used,
            "final_source": final_source,
            "final_report_sha256": sha256(final_path),
            "final_valid": final_check["valid"],
        }
        atomic_json(PROGRESS_JSON, progress)

        state.update({
            "updated_utc": now(),
            "completed_cases": len(progress["completed_cases"]),
        })
        atomic_json(STATE_JSON, state)

        print(
            f"[{len(progress['completed_cases']):02d}/{len(cases):02d}] {case_id} "
            f"raw={'PASS' if raw_check['valid'] else 'FAIL'} "
            f"fallback={'YES' if fallback_used else 'NO'} "
            f"time={generation_seconds:.1f}s"
        )

        if len(progress["completed_cases"]) % 5 == 0:
            sync_git(
                f"EViCT-Dx Step14 progress {len(progress['completed_cases'])}/{len(cases)}"
            )

    # Aggregate from progress and files.
    rows = []
    index_rows = []
    for case_id in cases:
        rec = progress["completed_cases"].get(case_id)
        if rec is None:
            raise RuntimeError(f"Missing Step14 progress record for {case_id}")

        evidence_path = EVIDENCE_DIR / f"{case_id}.json"
        raw_path = RAW_DIR / f"{case_id}.txt"
        final_path = FINAL_DIR / f"{case_id}.txt"
        det_path = DET_DIR / f"{case_id}.txt"

        evidence = json.loads(evidence_path.read_text())
        raw = raw_path.read_text().strip()
        final = final_path.read_text().strip()

        raw_check = validate_report(raw, evidence, title, footer)
        final_check = validate_report(final, evidence, title, footer)

        if not final_check["valid"]:
            raise RuntimeError(f"Final report revalidation failed for {case_id}")

        rows.append({
            "case_id": case_id,
            "raw_valid": raw_check["valid"],
            "fallback_used": bool(rec["fallback_used"]),
            "final_valid": final_check["valid"],
            "raw_title_ok": raw_check["title_ok"],
            "raw_footer_ok": raw_check["footer_ok"],
            "raw_numeric_copy_ok": raw_check["numeric_copy_ok"],
            "raw_no_extra_numbers": raw_check["no_extra_numbers"],
            "raw_locked_field_violation": raw_check["locked_field_violation"],
            "raw_prohibited_recommendation_violation": raw_check["prohibited_recommendation_violation"],
            "raw_errors": "|".join(raw_check["errors"]),
            "generation_seconds": float(rec["generation_seconds"]),
            "final_source": rec["final_source"],
        })

        index_rows.append({
            "case_id": case_id,
            "evidence_file": str(evidence_path.relative_to(ROOT)).replace("\\", "/"),
            "evidence_sha256": sha256(evidence_path),
            "deterministic_report_file": str(det_path.relative_to(ROOT)).replace("\\", "/"),
            "deterministic_report_sha256": sha256(det_path),
            "raw_vlm_report_file": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
            "raw_vlm_report_sha256": sha256(raw_path),
            "final_report_file": str(final_path.relative_to(ROOT)).replace("\\", "/"),
            "final_report_sha256": sha256(final_path),
            "final_source": rec["final_source"],
        })

    df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    idx_df = pd.DataFrame(index_rows).sort_values("case_id").reset_index(drop=True)
    atomic_csv(VALIDATION_CSV, df)
    atomic_csv(INDEX_CSV, idx_df)

    n = len(df)
    raw_accept = float(df.raw_valid.mean())
    fallback_rate = float(df.fallback_used.mean())
    final_safe = float(df.final_valid.mean())
    numeric_rate = float(df.raw_numeric_copy_ok.mean())
    title_rate = float(df.raw_title_ok.mean())
    footer_rate = float(df.raw_footer_ok.mean())
    locked_violation_rate = float(df.raw_locked_field_violation.mean())
    recommendation_violation_rate = float(df.raw_prohibited_recommendation_violation.mean())

    audit = {
        "project": "EViCT-Dx",
        "stage": "STEP_14_CONTROLLED_VLM_REALIZATION_AND_VALIDATION",
        "run_id": RUN_ID,
        "completed_utc": now(),
        "model": MODEL_ID,
        "model_role": "text_only_evidence_realizer",
        "visual_inputs_provided": False,
        "training_performed": False,
        "fine_tuning_performed": False,
        "cases": n,
        "raw_accepted": int(df.raw_valid.sum()),
        "raw_acceptance_rate": raw_accept,
        "fallback_cases": int(df.fallback_used.sum()),
        "fallback_rate": fallback_rate,
        "final_safe_reports": int(df.final_valid.sum()),
        "final_safe_rate": final_safe,
        "raw_exact_numeric_copy_rate": numeric_rate,
        "raw_title_compliance_rate": title_rate,
        "raw_footer_compliance_rate": footer_rate,
        "raw_locked_field_violation_rate": locked_violation_rate,
        "raw_prohibited_recommendation_rate": recommendation_violation_rate,
        "mean_generation_seconds": float(df.generation_seconds.mean()),
        "ground_truth_given_to_model": False,
        "CT_image_given_to_model": False,
        "locked_diagnostic_fields_given_as_positive_findings": False,
        "fallback_source": "Step13 deterministic validated report",
        "all_raw_outputs_preserved": True,
        "all_final_outputs_validated": True,
        "config_sha256": sha256(CFG_PATH),
        "permissions_sha256": sha256(PERMISSIONS_PATH),
        "report_contract_sha256": sha256(REPORT_CONTRACT_PATH),
        "next_action": CFG["next_stage"],
    }
    atomic_json(AUDIT_JSON, audit)

    final_state = {
        "run_id": RUN_ID,
        "status": "COMPLETE",
        "updated_utc": now(),
        "model": MODEL_ID,
        "visual_inputs_provided": False,
        "training_performed": False,
        "fine_tuning_performed": False,
        "cases": n,
        "raw_accepted": int(df.raw_valid.sum()),
        "fallback_cases": int(df.fallback_used.sum()),
        "final_safe_reports": int(df.final_valid.sum()),
        "next_action": CFG["next_stage"],
    }
    atomic_json(STATE_JSON, final_state)

    synced = sync_git("Complete EViCT-Dx Step14 controlled VLM reporting")

    print("\n" + "=" * 114)
    print("✅ STEP 14 COMPLETE — CONTROLLED VLM REALIZATION")
    print(f"Model                           : {MODEL_ID}")
    print("Visual inputs provided          : NO")
    print(f"Cases                           : {n}")
    print(f"Raw VLM accepted                : {int(df.raw_valid.sum())}/{n} ({100*raw_accept:.1f}%)")
    print(f"Deterministic fallbacks         : {int(df.fallback_used.sum())}/{n} ({100*fallback_rate:.1f}%)")
    print(f"Final safe reports              : {int(df.final_valid.sum())}/{n} ({100*final_safe:.1f}%)")
    print(f"Raw exact numeric copy          : {100*numeric_rate:.1f}%")
    print(f"Raw title compliance            : {100*title_rate:.1f}%")
    print(f"Raw footer compliance           : {100*footer_rate:.1f}%")
    print(f"Raw locked-field violation      : {100*locked_violation_rate:.1f}%")
    print(f"Raw prohibited recommendation   : {100*recommendation_violation_rate:.1f}%")
    print("Ground truth given to model     : NO")
    print("CT image given to model         : NO")
    print(f"GitHub metadata sync            : {'SUCCESS' if synced else 'CHECK REQUIRED'}")
    print(f"NEXT                            : {CFG['next_stage']}")
    print("=" * 114)


if __name__ == "__main__":
    main()
