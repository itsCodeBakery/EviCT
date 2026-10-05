from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

ROOT = Path(__file__).resolve().parents[2]
DX = ROOT / "vlmDiagnosis"
RUN_ID = "DX_step03d_covid_ct_md_audit_v1"
RUN = DX / "runs" / RUN_ID
SHARDS = RUN / "patient_shards"
TABLES = DX / "tables"
AUDIT = DX / "artifacts" / "audit"
MANIFESTS = DX / "artifacts" / "manifests"
CONFIG = DX / "config"

EXPECTED = {"COVID-19": 169, "CAP": 60, "Normal": 76}
SEED = 1705
SYNC_EVERY_PATIENTS = 20

CLASS_ALIASES = {
    "COVID-19 Cases": "COVID-19",
    "COVID-19": "COVID-19",
    "Cap Cases": "CAP",
    "CAP Cases": "CAP",
    "CAP": "CAP",
    "Normal Cases": "Normal",
    "Normal": "Normal",
}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, obj):
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def atomic_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return np.nan


def locate_archive():
    preferred = Path("/kaggle/working/evict_dx_figshare_temp/COVID-CT-MD.zip")
    if preferred.exists():
        return preferred

    roots = [Path("/kaggle/working"), Path("/kaggle/input")]
    hits = []
    for root in roots:
        if root.exists():
            hits.extend(root.rglob("COVID-CT-MD.zip"))
    if not hits:
        raise FileNotFoundError(
            "COVID-CT-MD.zip not found. Expected /kaggle/working/evict_dx_figshare_temp/COVID-CT-MD.zip"
        )
    hits = sorted(hits, key=lambda p: len(str(p)))
    return hits[0]


def parse_member(name: str):
    parts = [p for p in Path(name).parts if p not in (".", "")]
    if len(parts) < 3 or not name.lower().endswith(".dcm"):
        return None
    # Find the class folder anywhere in the archive path.
    for i, part in enumerate(parts[:-2]):
        if part in CLASS_ALIASES and i + 1 < len(parts):
            return CLASS_ALIASES[part], parts[i + 1]
    return None


def header_row(zf: zipfile.ZipFile, member: str, diagnosis: str, patient_id: str):
    with zf.open(member, "r") as f:
        ds = pydicom.dcmread(f, stop_before_pixels=True, force=False)

    ipp = getattr(ds, "ImagePositionPatient", None)
    iop = getattr(ds, "ImageOrientationPatient", None)
    px = getattr(ds, "PixelSpacing", None)

    return {
        "diagnosis": diagnosis,
        "patient_id": patient_id,
        "member": member,
        "study_uid": str(getattr(ds, "StudyInstanceUID", "")),
        "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
        "sop_uid": str(getattr(ds, "SOPInstanceUID", "")),
        "transfer_syntax_uid": str(getattr(ds.file_meta, "TransferSyntaxUID", "")),
        "rows": safe_int(getattr(ds, "Rows", np.nan)),
        "columns": safe_int(getattr(ds, "Columns", np.nan)),
        "pixel_spacing_row": safe_float(px[0]) if px is not None and len(px) >= 2 else np.nan,
        "pixel_spacing_col": safe_float(px[1]) if px is not None and len(px) >= 2 else np.nan,
        "slice_thickness": safe_float(getattr(ds, "SliceThickness", np.nan)),
        "spacing_between_slices": safe_float(getattr(ds, "SpacingBetweenSlices", np.nan)),
        "slice_location": safe_float(getattr(ds, "SliceLocation", np.nan)),
        "ipp_x": safe_float(ipp[0]) if ipp is not None and len(ipp) >= 3 else np.nan,
        "ipp_y": safe_float(ipp[1]) if ipp is not None and len(ipp) >= 3 else np.nan,
        "ipp_z": safe_float(ipp[2]) if ipp is not None and len(ipp) >= 3 else np.nan,
        "iop_0": safe_float(iop[0]) if iop is not None and len(iop) >= 6 else np.nan,
        "iop_1": safe_float(iop[1]) if iop is not None and len(iop) >= 6 else np.nan,
        "iop_2": safe_float(iop[2]) if iop is not None and len(iop) >= 6 else np.nan,
        "iop_3": safe_float(iop[3]) if iop is not None and len(iop) >= 6 else np.nan,
        "iop_4": safe_float(iop[4]) if iop is not None and len(iop) >= 6 else np.nan,
        "iop_5": safe_float(iop[5]) if iop is not None and len(iop) >= 6 else np.nan,
        "instance_number": safe_int(getattr(ds, "InstanceNumber", np.nan)),
        "rescale_slope": safe_float(getattr(ds, "RescaleSlope", 1.0)),
        "rescale_intercept": safe_float(getattr(ds, "RescaleIntercept", 0.0)),
        "bits_stored": safe_int(getattr(ds, "BitsStored", np.nan)),
        "pixel_representation": safe_int(getattr(ds, "PixelRepresentation", np.nan)),
        "photometric_interpretation": str(getattr(ds, "PhotometricInterpretation", "")),
    }


def sort_key_column(df: pd.DataFrame):
    if df["slice_location"].notna().all():
        return "slice_location"
    if df["ipp_z"].notna().all():
        return "ipp_z"
    if df["instance_number"].notna().all():
        return "instance_number"
    return None


def decode_sample(zf: zipfile.ZipFile, member: str):
    with zf.open(member, "r") as f:
        raw = f.read()
    ds = pydicom.dcmread(io.BytesIO(raw), force=False)
    arr = ds.pixel_array
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr.astype(np.float32) * slope + intercept
    return {
        "decode_ok": True,
        "decoded_shape": "x".join(map(str, arr.shape)),
        "decoded_dtype": str(arr.dtype),
        "raw_min": float(arr.min()),
        "raw_max": float(arr.max()),
        "hu_min": float(hu.min()),
        "hu_max": float(hu.max()),
    }


def process_patient(zf, diagnosis, patient_id, members):
    rows = []
    for member in sorted(members):
        rows.append(header_row(zf, member, diagnosis, patient_id))
    df = pd.DataFrame(rows)

    sort_col = sort_key_column(df)
    if sort_col:
        df = df.sort_values(sort_col, kind="mergesort").reset_index(drop=True)

    middle_member = df.iloc[len(df) // 2]["member"]
    try:
        sample = decode_sample(zf, middle_member)
        decode_error = ""
    except Exception as exc:
        sample = {
            "decode_ok": False,
            "decoded_shape": "",
            "decoded_dtype": "",
            "raw_min": np.nan,
            "raw_max": np.nan,
            "hu_min": np.nan,
            "hu_max": np.nan,
        }
        decode_error = repr(exc)

    def nunique_non_na(col):
        return int(df[col].dropna().nunique())

    summary = {
        "diagnosis": diagnosis,
        "patient_id": patient_id,
        "n_slices": int(len(df)),
        "n_study_uids": int(df["study_uid"].replace("", np.nan).dropna().nunique()),
        "n_series_uids": int(df["series_uid"].replace("", np.nan).dropna().nunique()),
        "n_sop_uids": int(df["sop_uid"].replace("", np.nan).dropna().nunique()),
        "n_transfer_syntaxes": int(df["transfer_syntax_uid"].replace("", np.nan).dropna().nunique()),
        "n_shapes": int(df[["rows", "columns"]].drop_duplicates().shape[0]),
        "n_pixel_spacings": int(df[["pixel_spacing_row", "pixel_spacing_col"]].drop_duplicates().shape[0]),
        "n_slice_thicknesses": nunique_non_na("slice_thickness"),
        "n_rescale_slopes": nunique_non_na("rescale_slope"),
        "n_rescale_intercepts": nunique_non_na("rescale_intercept"),
        "sort_key": sort_col or "NONE",
        "missing_slice_location": int(df["slice_location"].isna().sum()),
        "missing_ipp_z": int(df["ipp_z"].isna().sum()),
        "missing_instance_number": int(df["instance_number"].isna().sum()),
        "duplicate_sop_uid_count": int(len(df) - df["sop_uid"].replace("", np.nan).dropna().nunique())
        if (df["sop_uid"] != "").any() else np.nan,
        "sample_member": middle_member,
        "sample_decode_error": decode_error,
        **sample,
    }

    patient_key = f"{diagnosis.replace('-', '').replace(' ', '_')}__{patient_id}"
    atomic_csv(SHARDS / f"{patient_key}.csv", df)
    atomic_json(SHARDS / f"{patient_key}.summary.json", summary)
    return summary


def deterministic_stratified_split(case_df: pd.DataFrame):
    out = []
    rng = np.random.default_rng(SEED)
    for diagnosis in ["COVID-19", "CAP", "Normal"]:
        ids = sorted(case_df.loc[case_df.diagnosis == diagnosis, "patient_id"].tolist())
        ids = list(np.array(ids)[rng.permutation(len(ids))])

        n = len(ids)
        n_train = int(round(0.60 * n))
        n_val = int(round(0.20 * n))
        n_test = n - n_train - n_val

        assignments = (
            [("train", x) for x in ids[:n_train]]
            + [("validation", x) for x in ids[n_train:n_train + n_val]]
            + [("test", x) for x in ids[n_train + n_val:]]
        )
        for split, pid in assignments:
            out.append({"diagnosis": diagnosis, "patient_id": pid, "split": split})

        print(f"{diagnosis:9s}: train={n_train} val={n_val} test={n_test}")

    return pd.DataFrame(out).sort_values(["split", "diagnosis", "patient_id"]).reset_index(drop=True)


def sync_git(message):
    script = DX / "scripts" / "git_sync_dx.py"
    if not script.exists():
        print("⚠ git_sync_dx.py not found; outputs are local only.")
        return
    r = subprocess.run([sys.executable, str(script), message], cwd=str(ROOT), check=False)
    if r.returncode != 0:
        print(f"⚠ Git sync returned {r.returncode}; local artifacts are preserved.")


def main():
    print("=" * 108)
    print("EViCT-Dx STEP 03D — COVID-CT-MD FULL DICOM INVENTORY + GEOMETRY/INTENSITY AUDIT")
    print("=" * 108)

    for d in [RUN, SHARDS, TABLES, AUDIT, MANIFESTS, CONFIG]:
        d.mkdir(parents=True, exist_ok=True)

    # Frozen-core guard.
    verify = DX / "scripts" / "verify_core_lock.py"
    if verify.exists():
        subprocess.run([sys.executable, str(verify)], cwd=str(ROOT), check=True)

    archive = locate_archive()
    print("Archive :", archive)
    print("Size GiB:", f"{archive.stat().st_size / (1024**3):.3f}")
    print("pydicom :", pydicom.__version__)

    state_path = RUN / "STATE.json"
    state = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    completed = set(state.get("completed_patients", []))

    with zipfile.ZipFile(archive, "r") as zf:
        groups = defaultdict(list)
        for info in zf.infolist():
            parsed = parse_member(info.filename)
            if parsed is not None:
                groups[parsed].append(info.filename)

        counts = defaultdict(int)
        for diagnosis, patient_id in groups:
            counts[diagnosis] += 1

        print("\nPatient counts discovered:", dict(counts))
        if dict(counts) != EXPECTED:
            raise RuntimeError(f"Patient count mismatch. expected={EXPECTED}, discovered={dict(counts)}")

        ordered = sorted(groups.keys(), key=lambda x: (["COVID-19", "CAP", "Normal"].index(x[0]), x[1]))
        new_since_sync = 0

        for idx, (diagnosis, patient_id) in enumerate(ordered, 1):
            token = f"{diagnosis}|{patient_id}"
            shard = SHARDS / f"{diagnosis.replace('-', '').replace(' ', '_')}__{patient_id}.summary.json"
            if token in completed and shard.exists():
                print(f"[{idx:03d}/305] SKIP {diagnosis:9s} {patient_id}")
                continue

            print(f"[{idx:03d}/305] AUDIT {diagnosis:9s} {patient_id}  slices={len(groups[(diagnosis, patient_id)])}")
            summary = process_patient(zf, diagnosis, patient_id, groups[(diagnosis, patient_id)])

            completed.add(token)
            state = {
                "project": "EViCT-Dx",
                "run_id": RUN_ID,
                "stage": "STEP_03D_COVID_CT_MD_AUDIT",
                "status": "IN_PROGRESS",
                "archive_path": str(archive),
                "archive_size_bytes": archive.stat().st_size,
                "completed_patients": sorted(completed),
                "completed_count": len(completed),
                "total_patients": 305,
                "last_patient": token,
                "updated_utc": now(),
                "next_action": "resume_remaining_patients",
            }
            atomic_json(state_path, state)

            if not summary["decode_ok"]:
                raise RuntimeError(
                    f"Pixel decode failed for {token}: {summary['sample_decode_error']}"
                )

            new_since_sync += 1
            if new_since_sync >= SYNC_EVERY_PATIENTS:
                sync_git(f"EViCT-Dx Step03D progress: {len(completed)}/305 patients")
                new_since_sync = 0

    # Combine patient shards only after all 305 are complete.
    if len(completed) != 305:
        raise RuntimeError(f"Incomplete audit: {len(completed)}/305 patients")

    all_slice_frames = []
    summaries = []
    for p in sorted(SHARDS.glob("*.csv")):
        all_slice_frames.append(pd.read_csv(p))
    for p in sorted(SHARDS.glob("*.summary.json")):
        summaries.append(json.loads(p.read_text(encoding="utf-8")))

    inventory = pd.concat(all_slice_frames, ignore_index=True)
    patient_summary = pd.DataFrame(summaries).sort_values(["diagnosis", "patient_id"]).reset_index(drop=True)

    case_manifest = (
        inventory.groupby(["diagnosis", "patient_id"], as_index=False)
        .agg(
            n_slices=("member", "size"),
            study_uid=("study_uid", "first"),
            series_uid=("series_uid", "first"),
        )
        .sort_values(["diagnosis", "patient_id"])
        .reset_index(drop=True)
    )

    # Hard integrity checks before freezing a split.
    hard_failures = []
    if len(case_manifest) != 305:
        hard_failures.append(f"case_manifest_count={len(case_manifest)}")
    if patient_summary["decode_ok"].astype(bool).sum() != 305:
        hard_failures.append("sample_decode_failures_present")
    if (patient_summary["n_study_uids"] != 1).any():
        hard_failures.append("patients_with_non_single_study_uid")
    if (patient_summary["n_series_uids"] != 1).any():
        hard_failures.append("patients_with_non_single_series_uid")
    if (patient_summary["sort_key"] == "NONE").any():
        hard_failures.append("patients_without_reliable_slice_order_key")
    if patient_summary["duplicate_sop_uid_count"].fillna(0).gt(0).any():
        hard_failures.append("duplicate_sop_uids_within_patient")

    atomic_csv(TABLES / "step03d_covid_ct_md_dicom_inventory.csv", inventory)
    atomic_csv(TABLES / "step03d_covid_ct_md_geometry_intensity_summary.csv", patient_summary)
    atomic_csv(TABLES / "step03d_covid_ct_md_case_manifest.csv", case_manifest)

    split_df = deterministic_stratified_split(case_manifest)
    if split_df.duplicated(["diagnosis", "patient_id"]).any():
        hard_failures.append("duplicate_patient_split_assignment")
    if len(split_df) != 305:
        hard_failures.append("split_manifest_count_mismatch")

    expected_split_counts = {
        ("COVID-19", "train"): 101,
        ("COVID-19", "validation"): 34,
        ("COVID-19", "test"): 34,
        ("CAP", "train"): 36,
        ("CAP", "validation"): 12,
        ("CAP", "test"): 12,
        ("Normal", "train"): 46,
        ("Normal", "validation"): 15,
        ("Normal", "test"): 15,
    }
    actual_counts = split_df.groupby(["diagnosis", "split"]).size().to_dict()
    if actual_counts != expected_split_counts:
        hard_failures.append(f"unexpected_split_counts={actual_counts}")

    atomic_csv(TABLES / "step03d_covid_ct_md_split_manifest.csv", split_df)

    split_cfg = {
        "project": "EViCT-Dx",
        "dataset": "COVID-CT-MD",
        "stage": "STEP_03D",
        "status": "SPLIT_FROZEN_BEFORE_TRAINING" if not hard_failures else "AUDIT_FAILED_DO_NOT_TRAIN",
        "seed": SEED,
        "split_unit": "patient_CT_study",
        "stratification": "diagnostic_class",
        "counts": {
            "COVID-19": {"train": 101, "validation": 34, "test": 34},
            "CAP": {"train": 36, "validation": 12, "test": 12},
            "Normal": {"train": 46, "validation": 15, "test": 15},
            "total": {"train": 183, "validation": 61, "test": 61},
        },
        "rules": {
            "test_for_checkpoint_selection": False,
            "test_for_temperature_scaling": False,
            "test_for_abstention_threshold": False,
            "validation_for_checkpoint_selection": True,
            "validation_for_temperature_scaling": True,
            "validation_for_abstention_threshold": True,
        },
        "hard_failures": hard_failures,
    }
    atomic_json(CONFIG / "covid_ct_md_split_step03d.json", split_cfg)

    outputs = [
        TABLES / "step03d_covid_ct_md_dicom_inventory.csv",
        TABLES / "step03d_covid_ct_md_geometry_intensity_summary.csv",
        TABLES / "step03d_covid_ct_md_case_manifest.csv",
        TABLES / "step03d_covid_ct_md_split_manifest.csv",
        CONFIG / "covid_ct_md_split_step03d.json",
    ]
    hash_rows = [{"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p), "bytes": p.stat().st_size} for p in outputs]
    hash_df = pd.DataFrame(hash_rows)
    atomic_csv(MANIFESTS / "step03d_covid_ct_md_sha256.csv", hash_df)

    audit = {
        "project": "EViCT-Dx",
        "stage": "STEP_03D",
        "dataset": "COVID-CT-MD",
        "completed_utc": now(),
        "archive_path": str(archive),
        "archive_size_bytes": archive.stat().st_size,
        "patients": 305,
        "class_counts": EXPECTED,
        "dicom_files": int(len(inventory)),
        "sample_pixel_decode_successes": int(patient_summary["decode_ok"].astype(bool).sum()),
        "patients_single_study_uid": int((patient_summary["n_study_uids"] == 1).sum()),
        "patients_single_series_uid": int((patient_summary["n_series_uids"] == 1).sum()),
        "patients_with_slice_location_sort": int((patient_summary["sort_key"] == "slice_location").sum()),
        "patients_with_ipp_z_sort": int((patient_summary["sort_key"] == "ipp_z").sum()),
        "patients_with_instance_sort": int((patient_summary["sort_key"] == "instance_number").sum()),
        "hard_failures": hard_failures,
        "training_allowed": not bool(hard_failures),
        "split_manifest": "vlmDiagnosis/tables/step03d_covid_ct_md_split_manifest.csv",
        "hash_manifest": "vlmDiagnosis/artifacts/manifests/step03d_covid_ct_md_sha256.csv",
    }
    atomic_json(AUDIT / "step03d_covid_ct_md_full_audit.json", audit)

    state.update({
        "status": "COMPLETE" if not hard_failures else "FAILED_DO_NOT_TRAIN",
        "completed_count": 305,
        "updated_utc": now(),
        "next_action": "STEP_04_DIAGNOSIS_PREPROCESSING_BASELINE" if not hard_failures else "RESOLVE_STEP03D_AUDIT_FAILURES",
        "hard_failures": hard_failures,
    })
    atomic_json(state_path, state)

    sync_git("Freeze EViCT-Dx COVID-CT-MD Step03D audit and patient split")

    print("\n" + "=" * 108)
    if hard_failures:
        print("❌ STEP 03D FAILED — DO NOT START TRAINING")
        for x in hard_failures:
            print(" -", x)
    else:
        print("✅ STEP 03D COMPLETE")
        print("✅ 305/305 patient studies inventoried")
        print("✅ sample JPEG-lossless pixel decode succeeded for every patient")
        print("✅ patient-level 60/20/20 stratified split frozen")
        print("✅ train/validation/test = 183 / 61 / 61")
        print("✅ all artifacts are under vlmDiagnosis/")
        print("NEXT: STEP 04 — diagnosis preprocessing + baseline training")
    print("=" * 108)


if __name__ == "__main__":
    main()
