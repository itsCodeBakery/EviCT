from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DX = ROOT / "vlmDiagnosis"
CFG = json.loads((DX / "config" / "step10d_ncp_exact_freeze.json").read_text(encoding="utf-8"))

SRC = CFG["source"]
DATA_ROOT = Path(SRC["extracted_root"])
IMAGE_ROOT = DATA_ROOT / "image"
MASK_ROOT = DATA_ROOT / "mask"
README = DATA_ROOT / "README.txt"
ARCHIVE = Path(SRC["archive_path"])

TABLES = DX / "tables"
AUDIT_DIR = DX / "artifacts" / "audit"
MANIFEST_DIR = DX / "artifacts" / "manifests"
RUN = DX / "runs" / "DX_step10d_ncp_exact_freeze_v1"
for p in [TABLES, AUDIT_DIR, MANIFEST_DIR, RUN]:
    p.mkdir(parents=True, exist_ok=True)

PAIR_CSV = TABLES / "step10d_ncp_750_pair_manifest.csv"
PATIENT_SPLIT_CSV = TABLES / "step10d_ncp_patient_split.csv"
SLICE_SPLIT_CSV = TABLES / "step10d_ncp_slice_split_manifest.csv"
README_COPY = AUDIT_DIR / "step10d_ncp_official_README.txt"
AUDIT_JSON = AUDIT_DIR / "step10d_ncp_exact_freeze.json"
HASH_CSV = MANIFEST_DIR / "step10d_ncp_freeze_sha256.csv"
STATE_JSON = RUN / "STATE.json"

SEED = int(CFG["split"]["seed"])


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


def atomic_csv(path: Path, df: pd.DataFrame):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def normalize_text(s: str):
    return re.sub(r"\s+", " ", s.lower().replace("_", " ").replace("-", " ")).strip()


def near_number(text: str, aliases, n: int, window: int = 80):
    """True when a class alias and numeric label occur close together."""
    t = normalize_text(text)
    for alias in aliases:
        a = normalize_text(alias)
        for m in re.finditer(re.escape(a), t):
            lo = max(0, m.start() - window)
            hi = min(len(t), m.end() + window)
            chunk = t[lo:hi]
            if re.search(rf"(?<!\d){n}(?!\d)", chunk):
                return True
    return False


def infer_readme_mapping(text: str):
    aliases = {
        "background": ["background", "bg"],
        "lung_field": ["lung field", "lung", "lf"],
        "ground_glass_opacity": ["ground glass opacity", "ggo"],
        "consolidation": ["consolidation", "cl"],
    }
    candidates = [
        {
            "name": "BG0_LF1_CL2_GGO3",
            "mapping": {0:"background",1:"lung_field",2:"consolidation",3:"ground_glass_opacity"},
        },
        {
            "name": "BG0_LF1_GGO2_CL3",
            "mapping": {0:"background",1:"lung_field",2:"ground_glass_opacity",3:"consolidation"},
        },
    ]
    scored = []
    for c in candidates:
        details = {}
        score = 0
        for num, cls in c["mapping"].items():
            hit = near_number(text, aliases[cls], num)
            details[f"{num}_{cls}"] = hit
            score += int(hit)
        scored.append({"name":c["name"], "mapping":c["mapping"], "score":score, "details":details})

    scored = sorted(scored, key=lambda x: x["score"], reverse=True)
    if scored[0]["score"] >= 4 and (len(scored) == 1 or scored[0]["score"] > scored[1]["score"]):
        return True, scored[0]["mapping"], scored
    return False, None, scored


def sync_git():
    helper = DX / "scripts" / "git_sync_dx.py"
    r = subprocess.run(
        [sys.executable, str(helper), "Complete EViCT-Dx Step10D exact NCP label and split freeze"],
        cwd=str(ROOT),
        check=False,
    )
    return r.returncode == 0


def main():
    print("=" * 110)
    print("EViCT-Dx STEP 10D — EXACT PUBLIC NCP STRUCTURE / LABEL / PATIENT-SPLIT FREEZE")
    print("NO TRAINING")
    print("=" * 110)

    subprocess.run(
        [sys.executable, str(DX / "scripts" / "verify_core_lock.py")],
        cwd=str(ROOT),
        check=True,
    )

    if not ARCHIVE.exists():
        raise FileNotFoundError(f"Official archive missing: {ARCHIVE}")
    if ARCHIVE.stat().st_size != int(SRC["expected_archive_bytes"]):
        raise RuntimeError("Official archive byte size no longer matches the frozen source contract.")
    archive_sha = sha256(ARCHIVE)
    if archive_sha != SRC["archive_sha256"]:
        raise RuntimeError(f"Official archive SHA256 changed: {archive_sha}")

    for p in [DATA_ROOT, IMAGE_ROOT, MASK_ROOT, README]:
        if not p.exists():
            raise FileNotFoundError(f"Required official archive component missing: {p}")

    readme_text = README.read_text(encoding="utf-8", errors="replace")
    README_COPY.write_text(readme_text, encoding="utf-8")

    print("\n---------------- OFFICIAL ARCHIVE README ----------------")
    print(readme_text.strip())
    print("---------------------------------------------------------\n")

    mapping_verified, label_mapping, mapping_scores = infer_readme_mapping(readme_text)

    image_patients = sorted([p.name for p in IMAGE_ROOT.iterdir() if p.is_dir()], key=lambda x: int(x))
    mask_patients = sorted([p.name for p in MASK_ROOT.iterdir() if p.is_dir()], key=lambda x: int(x))

    image_set = set(image_patients)
    mask_set = set(mask_patients)
    if image_set != mask_set:
        raise RuntimeError(
            f"Image/mask patient directory mismatch. "
            f"image-only={sorted(image_set-mask_set)[:10]}, mask-only={sorted(mask_set-image_set)[:10]}"
        )

    pair_rows = []
    patient_rows = []
    global_labels = set()
    missing_pairs = []
    mask_counts_per_patient = Counter()

    total_images = 0
    total_masks = 0

    for pid in image_patients:
        image_dir = IMAGE_ROOT / pid
        mask_dir = MASK_ROOT / pid

        image_files = {
            p.stem: p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png"}
        }
        mask_files = {
            p.stem: p for p in mask_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png",".jpg",".jpeg"}
        }

        total_images += len(image_files)
        total_masks += len(mask_files)
        mask_counts_per_patient[len(mask_files)] += 1

        present_patient = set()
        for stem, mp in sorted(mask_files.items(), key=lambda kv: int(kv[0])):
            ip = image_files.get(stem)
            if ip is None:
                missing_pairs.append(f"{pid}/{stem}")
                continue

            with Image.open(mp) as im:
                arr = np.asarray(im)
                w, h = im.size

            if arr.ndim == 3:
                # Masks should be palette/grayscale. RGB is accepted only if all channels are identical.
                if not np.array_equal(arr[...,0], arr[...,1]) or not np.array_equal(arr[...,0], arr[...,2]):
                    raise RuntimeError(f"Mask is RGB with non-identical channels: {mp}")
                arr = arr[...,0]

            labels, counts = np.unique(arr, return_counts=True)
            labels = [int(x) for x in labels.tolist()]
            counts = [int(x) for x in counts.tolist()]
            if not set(labels).issubset({0,1,2,3}):
                raise RuntimeError(f"Unexpected mask values in {mp}: {labels}")

            global_labels.update(labels)
            present_patient.update(labels)
            count_map = dict(zip(labels, counts))

            pair_rows.append({
                "patient_id": pid,
                "slice_id": stem,
                "image_rel": str(ip.relative_to(DATA_ROOT)).replace("\\","/"),
                "mask_rel": str(mp.relative_to(DATA_ROOT)).replace("\\","/"),
                "width": w,
                "height": h,
                "labels_present": "|".join(map(str, labels)),
                "pixels_label_0": count_map.get(0,0),
                "pixels_label_1": count_map.get(1,0),
                "pixels_label_2": count_map.get(2,0),
                "pixels_label_3": count_map.get(3,0),
            })

        patient_rows.append({
            "patient_id": pid,
            "full_ct_slice_count": len(image_files),
            "annotated_mask_count": len(mask_files),
            "labels_present_across_annotated_slices": "|".join(map(str, sorted(present_patient))),
            "has_label_2": 2 in present_patient,
            "has_label_3": 3 in present_patient,
        })

    pairs = pd.DataFrame(pair_rows)
    patients = pd.DataFrame(patient_rows)

    contract = CFG["exact_structure_contract"]
    structure_checks = {
        "patient_groups_150": len(image_patients) == int(contract["patient_or_scan_groups"]),
        "masks_750": total_masks == int(contract["masks_total"]),
        "exactly_5_masks_each_group": (
            len(mask_counts_per_patient) == 1
            and mask_counts_per_patient.get(int(contract["masks_per_group"]),0) == int(contract["patient_or_scan_groups"])
        ),
        "all_masks_paired_to_same_patient_same_stem_image": len(missing_pairs) == 0 and len(pairs) == total_masks,
        "all_mask_dimensions_512x512": bool(((pairs.width == 512) & (pairs.height == 512)).all()),
        "global_mask_labels_exactly_0_1_2_3": global_labels == {0,1,2,3},
    }
    structure_verified = all(structure_checks.values())

    # Freeze deterministic patient-level 80/10/10 split. No slice-level randomization.
    pids = np.array(sorted(image_patients, key=lambda x: int(x)), dtype=object)
    rng = np.random.default_rng(SEED)
    perm = pids[rng.permutation(len(pids))]

    n_train = int(CFG["split"]["train_groups"])
    n_val = int(CFG["split"]["validation_groups"])
    train_ids = set(perm[:n_train].tolist())
    val_ids = set(perm[n_train:n_train+n_val].tolist())
    test_ids = set(perm[n_train+n_val:].tolist())

    if not (len(train_ids)==120 and len(val_ids)==15 and len(test_ids)==15):
        raise RuntimeError("Frozen patient split counts are not 120/15/15.")
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Patient split overlap detected.")

    split_rows = []
    for pid in sorted(image_patients, key=lambda x:int(x)):
        split = "train" if pid in train_ids else "validation" if pid in val_ids else "test"
        prow = patients.loc[patients.patient_id == pid].iloc[0]
        split_rows.append({
            "patient_id":pid,
            "split":split,
            "full_ct_slice_count":int(prow.full_ct_slice_count),
            "annotated_mask_count":int(prow.annotated_mask_count),
            "has_label_2":bool(prow.has_label_2),
            "has_label_3":bool(prow.has_label_3),
        })
    split_df = pd.DataFrame(split_rows)

    slice_split = pairs.merge(split_df[["patient_id","split"]], on="patient_id", how="left")
    if slice_split["split"].isna().any():
        raise RuntimeError("At least one annotated slice has no patient-level split assignment.")

    # Leakage check is exact because patient_id is the directory-level grouping unit.
    leakage_free = (
        set(split_df.loc[split_df.split=="train","patient_id"]).isdisjoint(
            set(split_df.loc[split_df.split=="validation","patient_id"])
        )
        and set(split_df.loc[split_df.split=="train","patient_id"]).isdisjoint(
            set(split_df.loc[split_df.split=="test","patient_id"])
        )
        and set(split_df.loc[split_df.split=="validation","patient_id"]).isdisjoint(
            set(split_df.loc[split_df.split=="test","patient_id"])
        )
    )

    atomic_csv(PAIR_CSV, pairs)
    atomic_csv(PATIENT_SPLIT_CSV, split_df)
    atomic_csv(SLICE_SPLIT_CSV, slice_split)

    split_summary = {}
    for split in ["train","validation","test"]:
        s = split_df[split_df.split==split]
        ps = slice_split[slice_split.split==split]
        split_summary[split] = {
            "patients":int(len(s)),
            "annotated_slices":int(len(ps)),
            "patients_with_label_2":int(s.has_label_2.sum()),
            "patients_with_label_3":int(s.has_label_3.sum()),
            "slices_with_label_2":int(ps.labels_present.str.split("|").apply(lambda z: "2" in z).sum()),
            "slices_with_label_3":int(ps.labels_present.str.split("|").apply(lambda z: "3" in z).sum()),
        }

    # Training can only start if both structure and numeric class mapping are unambiguous.
    training_allowed = bool(structure_verified and mapping_verified and leakage_free)

    audit = {
        "project":"EViCT-Dx",
        "stage":"STEP_10D_PUBLIC_NCP_EXACT_LABEL_AND_SPLIT_FREEZE",
        "completed_utc":now(),
        "training_performed":False,
        "archive_sha256":archive_sha,
        "step10c_heuristic_correction":{
            "issue":"The common path prefix ct_lesion_seg contains the token seg, so the Step10C substring heuristic marked every image path as a probable mask.",
            "impact":"Discovery-summary classification only; archive bytes, SHA256, ZIP integrity, dimensions and extracted files were unaffected.",
            "correct_rule":"Files under ct_lesion_seg/image are CT images; files under ct_lesion_seg/mask are segmentation masks."
        },
        "exact_structure":{
            "full_ct_image_files":int(total_images),
            "annotated_mask_files":int(total_masks),
            "patient_or_scan_groups":int(len(image_patients)),
            "mask_count_per_group_distribution":{str(k):int(v) for k,v in sorted(mask_counts_per_patient.items())},
            "paired_annotated_slices":int(len(pairs)),
            "missing_image_pairs":missing_pairs,
            "global_mask_labels":sorted(global_labels),
            "checks":structure_checks,
            "verified":structure_verified,
        },
        "official_readme_file":str(README_COPY.relative_to(ROOT)).replace("\\","/"),
        "readme_mapping_detection":{
            "verified":mapping_verified,
            "selected_mapping":label_mapping,
            "candidate_scores":mapping_scores,
        },
        "patient_split":{
            "seed":SEED,
            "method":CFG["split"]["method"],
            "leakage_free":bool(leakage_free),
            "summary":split_summary,
        },
        "training_allowed":training_allowed,
        "targets_if_allowed":["ground_glass_opacity","consolidation"],
        "pleural_effusion":"LOCKED_NOT_PRESENT_IN_PUBLIC_750_SLICE_RESOURCE",
        "next_action":(
            "STEP_11_TRAIN_GGO_CONSOLIDATION_SEGMENTATION"
            if training_allowed else
            "REVIEW_OFFICIAL_README_NUMERIC_LABEL_MAPPING_BEFORE_TRAINING"
        ),
    }
    atomic_json(AUDIT_JSON, audit)

    hash_sources = [
        DX / "config" / "step10d_ncp_exact_freeze.json",
        README_COPY, PAIR_CSV, PATIENT_SPLIT_CSV, SLICE_SPLIT_CSV, AUDIT_JSON
    ]
    hash_df = pd.DataFrame([
        {
            "relative_path":str(p.relative_to(ROOT)).replace("\\","/"),
            "sha256":sha256(p),
            "bytes":p.stat().st_size,
        }
        for p in hash_sources
    ])
    atomic_csv(HASH_CSV, hash_df)

    atomic_json(STATE_JSON, {
        "run_id":"DX_step10d_ncp_exact_freeze_v1",
        "status":"COMPLETE",
        "updated_utc":now(),
        "structure_verified":structure_verified,
        "label_mapping_verified":mapping_verified,
        "patient_split_leakage_free":bool(leakage_free),
        "training_allowed":training_allowed,
        "next_action":audit["next_action"],
    })

    synced = sync_git()

    print("\n" + "=" * 110)
    print("✅ STEP 10D COMPLETE — EXACT NCP FREEZE")
    print(f"Full CT image files             : {total_images}")
    print(f"Annotated masks                 : {total_masks}")
    print(f"Patient/scan groups             : {len(image_patients)}")
    print(f"Exactly 5 masks per group       : {structure_checks['exactly_5_masks_each_group']}")
    print(f"Mask→image exact pairs          : {len(pairs)}/{total_masks}")
    print(f"Global mask labels              : {sorted(global_labels)}")
    print(f"Structure verified              : {structure_verified}")
    print(f"README numeric mapping verified : {mapping_verified}")
    if label_mapping is not None:
        print(f"Frozen label mapping            : {label_mapping}")
    print(f"Patient split                   : 120 train / 15 validation / 15 test")
    print(f"Patient leakage                 : {'NONE' if leakage_free else 'DETECTED'}")
    print(f"Training allowed                : {training_allowed}")
    print(f"GitHub metadata sync            : {'SUCCESS' if synced else 'CHECK REQUIRED'}")
    print(f"NEXT                            : {audit['next_action']}")
    print("=" * 110)


if __name__ == "__main__":
    main()
