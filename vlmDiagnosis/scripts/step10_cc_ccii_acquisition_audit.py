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

ROOT = Path(__file__).resolve().parents[2]
DX = ROOT / "vlmDiagnosis"
CFG_PATH = DX / "config" / "step10_cc_ccii_acquisition_audit.json"
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))

TABLES = DX / "tables"
AUDIT_DIR = DX / "artifacts" / "audit"
RUN = DX / "runs" / "DX_step10_cc_ccii_acquisition_audit_v1"
TABLES.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
RUN.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = TABLES / "step10_cc_ccii_candidate_summary.csv"
FILES_CSV = TABLES / "step10_cc_ccii_candidate_file_inventory.csv"
AUDIT_JSON = AUDIT_DIR / "step10_cc_ccii_acquisition_audit.json"
STATE_JSON = RUN / "STATE.json"

IMAGE_EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".npy",".nii",".gz"}
META_EXTS = {".csv",".json",".txt",".xlsx",".xls",".yaml",".yml"}
MASK_TOKENS = ("mask","seg","label","annotation","annot","gt","groundtruth")
CLASS_TOKENS = {
    "ground_glass_opacity": ("ggo","groundglass","ground_glass","ground glass"),
    "consolidation": ("consolid","consolidation"),
    "pleural_effusion": ("effusion","pleural_effusion","pleural effusion"),
    "pulmonary_fibrosis": ("fibrosis","fibrotic"),
    "interstitial_thickening": ("interstitial","thickening"),
}
CANDIDATE_TOKENS = ("cc-ccii","cc_ccii","ccii","cc ccii","lesion")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, df: pd.DataFrame):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def normalize(s: str):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def candidate_roots():
    base = Path("/kaggle/input")
    if not base.exists():
        return []
    roots = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        n = normalize(child.name)
        if any(normalize(tok) in n for tok in CANDIDATE_TOKENS):
            roots.append(child)
            continue
        # shallow directory-name search only; avoids traversing unrelated mounted datasets
        try:
            for sub in child.iterdir():
                if sub.is_dir():
                    sn = normalize(sub.name)
                    if any(normalize(tok) in sn for tok in CANDIDATE_TOKENS):
                        roots.append(child)
                        break
        except PermissionError:
            pass
    # preserve unique roots
    out, seen = [], set()
    for p in roots:
        if p not in seen:
            out.append(p); seen.add(p)
    return out


def file_role(path: Path):
    text = normalize(str(path))
    if path.suffix.lower() in META_EXTS:
        return "metadata"
    if any(normalize(t) in text for t in MASK_TOKENS):
        return "mask_candidate"
    return "image_or_other"


def class_flags(path: Path):
    raw = str(path).lower()
    flags = {}
    for cls, toks in CLASS_TOKENS.items():
        flags[cls] = any(tok in raw for tok in toks)
    return flags


def infer_id(path: Path):
    stem = path.name
    if stem.lower().endswith(".nii.gz"):
        stem = stem[:-7]
    else:
        stem = Path(stem).stem
    # conservative: strip common mask/class suffixes but do not invent patient IDs
    s = normalize(stem)
    for token in [
        "mask","segmentation","seg","label","annotation","annot","groundtruth",
        "ggo","ground_glass_opacity","consolidation","pleural_effusion",
        "fibrosis","interstitial_thickening"
    ]:
        s = re.sub(rf"(^|_){re.escape(token)}($|_)", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or normalize(stem)


def sha256_small(path: Path, max_bytes=10*1024*1024):
    if path.stat().st_size > max_bytes:
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def sync_git():
    helper = DX / "scripts" / "git_sync_dx.py"
    r = subprocess.run(
        [sys.executable, str(helper), "Complete EViCT-Dx Step10 CC-CCII acquisition audit"],
        cwd=str(ROOT),
        check=False,
    )
    return r.returncode == 0


def main():
    print("="*108)
    print("EViCT-Dx STEP 10 — CC-CCII SUBTYPE DATA ACQUISITION / REPRODUCIBILITY AUDIT")
    print("AUDIT ONLY — NO TRAINING")
    print("="*108)

    subprocess.run(
        [sys.executable, str(DX / "scripts" / "verify_core_lock.py")],
        cwd=str(ROOT),
        check=True,
    )

    roots = candidate_roots()
    rows = []
    summary = []

    for root in roots:
        print(f"\nScanning candidate: {root}")
        files = [p for p in root.rglob("*") if p.is_file()]
        for p in files:
            rel = str(p.relative_to(root)).replace("\\","/")
            flags = class_flags(p)
            role = file_role(p)
            rows.append({
                "candidate_root": str(root),
                "relative_path": rel,
                "bytes": p.stat().st_size,
                "suffix": "".join(p.suffixes[-2:]) if p.name.lower().endswith(".nii.gz") else p.suffix.lower(),
                "role": role,
                "possible_study_id": infer_id(p),
                **{f"class_token_{k}": bool(v) for k,v in flags.items()},
                "sha256_if_le_10MB": sha256_small(p) if role == "metadata" else "",
            })

        rdf = pd.DataFrame([r for r in rows if r["candidate_root"] == str(root)])
        if rdf.empty:
            continue

        masks = rdf[rdf.role == "mask_candidate"]
        metas = rdf[rdf.role == "metadata"]
        class_counts = {
            cls: int(rdf[f"class_token_{cls}"].sum())
            for cls in CLASS_TOKENS
        }

        split_token_counts = {}
        joined = "\n".join(rdf.relative_path.astype(str).str.lower())
        for token in ["train","training","val","valid","validation","test"]:
            split_token_counts[token] = joined.count(token)

        summary.append({
            "candidate_root": str(root),
            "total_files": int(len(rdf)),
            "mask_candidate_files": int(len(masks)),
            "metadata_files": int(len(metas)),
            "possible_unique_ids_all_files": int(rdf.possible_study_id.nunique()),
            "possible_unique_ids_mask_files": int(masks.possible_study_id.nunique()) if len(masks) else 0,
            "documented_target_count_1302_match_by_mask_ids": bool(
                len(masks) > 0 and masks.possible_study_id.nunique() == 1302
            ),
            **{f"token_files_{k}": v for k,v in class_counts.items()},
            "has_any_train_token": split_token_counts["train"] + split_token_counts["training"] > 0,
            "has_any_validation_token": (
                split_token_counts["val"] + split_token_counts["valid"] + split_token_counts["validation"] > 0
            ),
            "has_any_test_token": split_token_counts["test"] > 0,
        })

    files_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary)

    if files_df.empty:
        files_df = pd.DataFrame(columns=[
            "candidate_root","relative_path","bytes","suffix","role","possible_study_id",
            *[f"class_token_{k}" for k in CLASS_TOKENS],
            "sha256_if_le_10MB",
        ])
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=[
            "candidate_root","total_files","mask_candidate_files","metadata_files",
            "possible_unique_ids_all_files","possible_unique_ids_mask_files",
            "documented_target_count_1302_match_by_mask_ids",
            *[f"token_files_{k}" for k in CLASS_TOKENS],
            "has_any_train_token","has_any_validation_token","has_any_test_token",
        ])

    atomic_csv(FILES_CSV, files_df)
    atomic_csv(SUMMARY_CSV, summary_df)

    if not roots:
        status = "NO_CC_CCII_CANDIDATE_MOUNTED"
        next_action = (
            "Attach one or more candidate CC-CCII lesion-segmentation datasets to the Kaggle notebook, "
            "then rerun Step10. Do not begin subtype training yet."
        )
        exact_verified = False
        semantic_verified = False
        study_disjoint_verified = False
    else:
        # Step10 is deliberately conservative: filenames/counts alone cannot prove annotation semantics.
        status = "CANDIDATE_FOUND_SEMANTICS_UNVERIFIED"
        next_action = (
            "Review candidate summary/file inventory and verify annotation semantics, exact source provenance, "
            "and study IDs before freezing a subtype split."
        )
        exact_verified = False
        semantic_verified = False
        study_disjoint_verified = False

    audit = {
        "project":"EViCT-Dx",
        "stage":"STEP_10_CC_CCII_SUBTYPE_ACQUISITION_AUDIT",
        "completed_utc":now(),
        "status":status,
        "training_performed":False,
        "candidate_roots":[str(p) for p in roots],
        "candidate_count":len(roots),
        "exact_1302_slice_resource_verified":exact_verified,
        "annotation_semantics_verified":semantic_verified,
        "study_disjoint_identifiers_verified":study_disjoint_verified,
        "required_classes":CFG["target_resource"]["required_classes"],
        "important_note":(
            "Counts, directory names, and filename tokens are discovery evidence only. "
            "They do not prove that a mounted dataset is the exact published 1,302-slice resource."
        ),
        "candidate_summary_csv":str(SUMMARY_CSV.relative_to(ROOT)).replace("\\","/"),
        "candidate_file_inventory_csv":str(FILES_CSV.relative_to(ROOT)).replace("\\","/"),
        "next_action":next_action,
    }
    atomic_json(AUDIT_JSON, audit)

    state = {
        "run_id":"DX_step10_cc_ccii_acquisition_audit_v1",
        "status":"COMPLETE",
        "audit_status":status,
        "updated_utc":now(),
        "candidate_count":len(roots),
        "exact_resource_verified":exact_verified,
        "next_action":next_action,
    }
    atomic_json(STATE_JSON, state)

    synced = sync_git()

    print("\n" + "="*108)
    print("✅ STEP 10 AUDIT COMPLETE")
    print(f"Candidate dataset roots      : {len(roots)}")
    print(f"Audit status                 : {status}")
    print(f"Exact 1,302-slice resource   : {'VERIFIED' if exact_verified else 'NOT VERIFIED'}")
    print(f"Annotation semantics         : {'VERIFIED' if semantic_verified else 'NOT VERIFIED'}")
    print(f"Study-disjoint IDs           : {'VERIFIED' if study_disjoint_verified else 'NOT VERIFIED'}")
    print(f"GitHub metadata sync         : {'SUCCESS' if synced else 'CHECK REQUIRED'}")
    print("Training                     : NOT STARTED")
    print("NEXT                         :", next_action)
    print("="*108)


if __name__ == "__main__":
    main()
