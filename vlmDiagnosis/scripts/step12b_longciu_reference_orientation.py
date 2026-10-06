from __future__ import annotations

import hashlib
import html
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import nibabel as nib
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torchvision.models import resnet34
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
DX = ROOT / "vlmDiagnosis"

CFG_PATH = DX / "config" / "step12_longciu_external_eval.json"
CORR_CFG_PATH = DX / "config" / "step12b_longciu_reference_orientation.json"
CORR_CFG = json.loads(CORR_CFG_PATH.read_text(encoding="utf-8"))
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))

RUN_ID = CORR_CFG["run_id"]
RUN = DX / "runs" / RUN_ID
TABLES = DX / "tables"
AUDIT_DIR = DX / "artifacts" / "audit"
MANIFEST_DIR = DX / "artifacts" / "manifests"
for p in [RUN, TABLES, AUDIT_DIR, MANIFEST_DIR]:
    p.mkdir(parents=True, exist_ok=True)

STEP11_RUN = DX / "runs" / CFG["frozen_model"]["step11_run_id"]
STEP11_STATE = STEP11_RUN / "STATE.json"
STEP11_BEST = STEP11_RUN / CFG["frozen_model"]["checkpoint_asset"]
STEP11_THRESH = STEP11_RUN / "validation_selected_thresholds.json"
STEP11_CONFIG = DX / "config" / "step11_ncp_ggo_consolidation.json"
STEP11_PAIR = DX / "tables" / "step10d_ncp_750_pair_manifest.csv"
STEP11_SPLIT = DX / "tables" / "step10d_ncp_patient_split.csv"

SOURCE_AUDIT = AUDIT_DIR / "step12b_longciu_source_audit.json"
SLICE_METRICS = TABLES / "step12b_longciu_slice_metrics.csv"
EXTERNAL_METRICS = TABLES / "step12b_longciu_external_metrics.csv"
RESULTS_JSON = AUDIT_DIR / "step12b_longciu_external_results.json"
PERMISSIONS_OUT = DX / "config" / "report_field_permissions_step12b.json"
STATE_JSON = RUN / "STATE.json"
CORRECTION_AUDIT = AUDIT_DIR / "step12b_orientation_correction_audit.json"
HASH_MANIFEST = MANIFEST_DIR / "step12_longciu_source_sha256.csv"

WORK = Path("/kaggle/working/evict_dx_longciu")
WORK.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    raise RuntimeError("Enable a Kaggle GPU before Step12 external inference.")

sys.path.insert(0, str(DX))
from runtime.recovery import GitHubReleaseStore, RecoveryPolicy, kaggle_secret, sha256_file  # noqa: E402


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


def sync_git(message):
    helper = DX / "scripts" / "git_sync_dx.py"
    r = subprocess.run([sys.executable, str(helper), message], cwd=str(ROOT), check=False)
    if r.returncode != 0:
        print(f"⚠ metadata Git sync returned {r.returncode}")


def load_policy():
    raw = json.loads((DX / "config" / "recovery_policy.json").read_text())
    allowed = set(RecoveryPolicy.__dataclass_fields__)
    return RecoveryPolicy(**{k: v for k, v in raw.items() if k in allowed})


POLICY = load_policy()


def release_store():
    return GitHubReleaseStore(
        repository=POLICY.github_repository,
        token=kaggle_secret(POLICY.kaggle_secret_name),
        release_prefix=POLICY.rolling_release_prefix,
    )


def verify_step11():
    if not STEP11_STATE.exists():
        raise RuntimeError("Step11 state is missing.")
    state = json.loads(STEP11_STATE.read_text())
    if state.get("status") != "COMPLETE":
        raise RuntimeError("Step11 is not COMPLETE.")
    if not bool(state.get("GGO_internal_gate_pass")):
        raise RuntimeError("Step11 GGO internal gate did not pass.")
    if not bool(state.get("consolidation_internal_gate_pass")):
        raise RuntimeError("Step11 consolidation internal gate did not pass.")
    if bool(state.get("external_test_accessed")):
        raise RuntimeError("Step11 state unexpectedly says external test already accessed.")
    if not STEP11_THRESH.exists():
        raise RuntimeError("Frozen Step11 validation thresholds are missing.")
    return state


def restore_step11_checkpoint_if_needed():
    if STEP11_BEST.exists():
        return
    print("Step11 best checkpoint is not local; attempting GitHub Release restore...")
    release_store().download(
        CFG["frozen_model"]["step11_run_id"],
        CFG["frozen_model"]["checkpoint_asset"],
        STEP11_BEST,
    )
    print("✓ Step11 checkpoint restored from GitHub Release.")


def verify_checkpoint_hashes(ck):
    expected = {
        "config_sha": sha256_file(STEP11_CONFIG),
        "pair_manifest_sha": sha256_file(STEP11_PAIR),
        "patient_split_sha": sha256_file(STEP11_SPLIT),
    }
    for k, v in expected.items():
        if ck.get(k) != v:
            raise RuntimeError(f"Frozen Step11 checkpoint hash mismatch for {k}.")


# --------------------------------------------------------------------------------------
# LONGCIU ACQUISITION
# --------------------------------------------------------------------------------------

REQUIRED = [
    "longciu_img.nii.gz",
    "longciu_STAPLE_tgt.nii.gz",
    "longciu_1_tgt.nii.gz",
    "longciu_2_tgt.nii.gz",
    "longciu_3_tgt.nii.gz",
]


def is_longciu_dir(p: Path):
    return p.is_dir() and all((p / name).exists() for name in REQUIRED)


def locate_existing_longciu():
    candidates = [
        WORK,
        WORK / "longciu",
        Path("/kaggle/working/longciu"),
    ]
    for c in candidates:
        if is_longciu_dir(c):
            return c

    for base in [Path("/kaggle/input"), Path("/kaggle/working")]:
        if not base.exists():
            continue
        for hit in base.rglob("longciu_img.nii.gz"):
            parent = hit.parent
            if is_longciu_dir(parent):
                return parent
    return None


def extract_zip_if_valid(zip_path: Path):
    if not zip_path.exists() or not zipfile.is_zipfile(zip_path):
        return None
    target = WORK / "extracted"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target)

    if is_longciu_dir(target):
        return target
    for p in target.rglob("longciu_img.nii.gz"):
        if is_longciu_dir(p.parent):
            return p.parent
    return None


def find_local_longciu_zip():
    for base in [Path("/kaggle/input"), Path("/kaggle/working")]:
        if not base.exists():
            continue
        for p in base.rglob("*.zip"):
            if "longciu" in p.name.lower():
                return p
    return None


def discover_longciu_zip_urls():
    doi_url = "https://doi.org/" + CFG["source"]["doi"]
    headers = {"User-Agent": "Mozilla/5.0 EViCT-Dx research reproducibility audit"}
    session = requests.Session()
    r = session.get(doi_url, headers=headers, timeout=45, allow_redirects=True)
    r.raise_for_status()

    landing = r.url
    text = r.text
    decoded = html.unescape(text).replace("\\/", "/")

    candidates = []

    # Exact zip URLs embedded in HTML / JSON.
    for m in re.findall(r'https?://[^"\'<>\s]+\.zip(?:\?[^"\'<>\s]*)?', decoded, flags=re.I):
        candidates.append(m)

    # Relative href/src-style links ending in .zip.
    for m in re.findall(r'(?:href|src)\s*=\s*["\']([^"\']+\.zip(?:\?[^"\']*)?)["\']', decoded, flags=re.I):
        candidates.append(urljoin(landing, m))

    # Be conservative: prefer links whose URL or decoded surrounding page identifies longciu.
    ordered = []
    seen = set()
    for u in candidates:
        u = u.replace("&amp;", "&")
        if u not in seen:
            seen.add(u)
            ordered.append(u)

    ordered.sort(key=lambda u: (0 if "longciu" in u.lower() else 1, len(u)))
    return landing, ordered


def wget_resume(url: str, out: Path):
    print(f"Trying dataset URL: {url}")
    result = subprocess.run(
        [
            "wget", "-c",
            "--tries=8",
            "--retry-connrefused",
            "--waitretry=10",
            "--timeout=60",
            "--read-timeout=60",
            "--progress=bar:force:noscroll",
            "-O", str(out), url,
        ],
        check=False,
    )
    return result.returncode == 0 and out.exists() and out.stat().st_size > 0


def acquire_longciu():
    existing = locate_existing_longciu()
    if existing:
        print(f"✓ Existing LongCIU dataset located: {existing}")
        return existing, {"mode": "existing_directory", "path": str(existing)}

    local_zip = find_local_longciu_zip()
    if local_zip:
        print(f"Found local LongCIU ZIP candidate: {local_zip}")
        extracted = extract_zip_if_valid(local_zip)
        if extracted:
            print(f"✓ LongCIU extracted from local ZIP: {extracted}")
            return extracted, {
                "mode": "local_zip",
                "zip_path": str(local_zip),
                "zip_sha256": sha256(local_zip),
                "path": str(extracted),
            }

    print("No local LongCIU copy found. Resolving the official DOI...")
    try:
        landing, urls = discover_longciu_zip_urls()
    except Exception as exc:
        landing, urls = None, []
        print(f"⚠ DOI auto-discovery failed: {exc}")

    print("Landing page:", landing)
    print("Discovered ZIP candidates:", len(urls))
    for u in urls[:10]:
        print("  •", u)

    archive = WORK / "longciu.zip"
    for u in urls:
        if archive.exists() and not zipfile.is_zipfile(archive):
            # preserve failed HTML/broken payload for debugging under a distinct name
            bad = WORK / f"failed_download_{int(time.time())}.bin"
            archive.replace(bad)
        ok = wget_resume(u, archive)
        if ok and zipfile.is_zipfile(archive):
            extracted = extract_zip_if_valid(archive)
            if extracted:
                print(f"✓ Official LongCIU dataset acquired: {extracted}")
                return extracted, {
                    "mode": "doi_auto_download",
                    "landing_page": landing,
                    "download_url": u,
                    "zip_path": str(archive),
                    "zip_sha256": sha256(archive),
                    "path": str(extracted),
                }

    wait_state = {
        "run_id": RUN_ID,
        "status": "WAITING_FOR_LONGCIU_DATASET",
        "updated_utc": now(),
        "external_test_accessed": False,
        "training_performed": False,
        "landing_page": landing,
        "candidate_zip_urls": urls[:50],
        "next_action": (
            "Attach/download the official longciu.zip from DOI 10.25820/data.007301 "
            "into Kaggle, then rerun Step12 unchanged."
        ),
    }
    atomic_json(STATE_JSON, wait_state)
    sync_git("EViCT-Dx Step12B waiting for LongCIU dataset")

    raise RuntimeError(
        "Could not automatically obtain LongCIU. No external predictions were made. "
        "Attach the official longciu.zip (or extracted folder) to Kaggle and rerun this same Step12."
    )


# --------------------------------------------------------------------------------------
# MODEL
# --------------------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        enc = resnet34(weights=None)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.pool = enc.maxpool
        self.e1 = enc.layer1
        self.e2 = enc.layer2
        self.e3 = enc.layer3
        self.e4 = enc.layer4
        self.d4 = ConvBlock(512 + 256, 256)
        self.d3 = ConvBlock(256 + 128, 128)
        self.d2 = ConvBlock(128 + 64, 64)
        self.d1 = ConvBlock(64 + 64, 64)
        self.head = nn.Conv2d(64, 4, 1)

    def forward(self, x):
        s0 = self.stem(x)
        e1 = self.e1(self.pool(s0))
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        x = F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.d4(torch.cat([x, e3], 1))
        x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.d3(torch.cat([x, e2], 1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.d2(torch.cat([x, e1], 1))
        x = F.interpolate(x, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        x = self.d1(torch.cat([x, s0], 1))
        x = F.interpolate(
            x,
            size=tuple(CFG["frozen_model"]["input_size"]),
            mode="bilinear",
            align_corners=False,
        )
        return self.head(x)


def load_frozen_model():
    restore_step11_checkpoint_if_needed()
    ck = torch.load(STEP11_BEST, map_location=DEVICE, weights_only=False)
    verify_checkpoint_hashes(ck)
    model = UNetResNet34().to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


# --------------------------------------------------------------------------------------
# DATA / METRICS
# --------------------------------------------------------------------------------------

def as_90_slices(path: Path):
    """Load LongCIU in the official SimpleITK-compatible [z, y, x] convention.

    Step12 v1 used nibabel and moved only the 90-slice axis to the front,
    producing [z, x, y]. The official LongCIU reference code uses
    SimpleITK.GetArrayFromImage, which returns [z, y, x].

    This is a single frozen technical correction, not an orientation search.
    """
    arr = np.asanyarray(nib.load(path).dataobj)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise RuntimeError(f"Expected 3D NIfTI for {path.name}; got shape={arr.shape}")

    expected_slices = int(CFG["source"]["expected_selected_slices"])

    # The distributed LongCIU NIfTI is stored in nibabel-native [x, y, z]
    # index order. Reverse axes to exactly emulate SimpleITK numpy order:
    # [x,y,z] -> [z,y,x].
    arr = np.transpose(arr, (2, 1, 0))

    if arr.shape[0] != expected_slices:
        raise RuntimeError(
            f"SimpleITK-compatible conversion did not yield {expected_slices} slices "
            f"for {path.name}; got shape={arr.shape}"
        )
    return arr


def preprocess_longciu_slice(arr):
    lo, hi = [float(x) for x in CFG["preprocessing"]["HU_clip"]]
    x = arr.astype(np.float32)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    x = torch.from_numpy(x)[None, None]
    x = F.interpolate(
        x,
        size=tuple(CFG["frozen_model"]["input_size"]),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    x = torch.stack([x, x, x], 0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (x - mean) / std


def counts_metrics(pred, gt):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    dice = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) else 1.0
    sens = float(tp / (tp + fn)) if (tp + fn) else np.nan
    spec = float(tn / (tn + fp)) if (tn + fp) else np.nan
    return {
        "Dice": dice,
        "IoU": iou,
        "sensitivity": sens,
        "specificity": spec,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def pooled_from_counts(rows):
    tp = sum(int(r["tp"]) for r in rows)
    fp = sum(int(r["fp"]) for r in rows)
    fn = sum(int(r["fn"]) for r in rows)
    tn = sum(int(r["tn"]) for r in rows)
    return counts_metrics_from_numbers(tp, fp, fn, tn)


def counts_metrics_from_numbers(tp, fp, fn, tn):
    dice = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) else 1.0
    sens = float(tp / (tp + fn)) if (tp + fn) else np.nan
    spec = float(tn / (tn + fp)) if (tn + fp) else np.nan
    return {
        "Dice": dice,
        "IoU": iou,
        "sensitivity": sens,
        "specificity": spec,
        "tp_pixels": int(tp),
        "fp_pixels": int(fp),
        "fn_pixels": int(fn),
        "tn_pixels": int(tn),
    }


def presence_metrics(y, score, threshold):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=float)
    pred = score >= float(threshold)
    tp = int((pred & y).sum())
    tn = int((~pred & ~y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())

    sens = float(tp / (tp + fn)) if tp + fn else np.nan
    spec = float(tn / (tn + fp)) if tn + fp else np.nan
    prec = float(tp / (tp + fp)) if tp + fp else np.nan
    bal = float(np.nanmean([sens, spec])) if np.isfinite(sens) or np.isfinite(spec) else np.nan
    auc = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan

    return {
        "threshold": float(threshold),
        "AUROC": auc,
        "balanced_accuracy": bal,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n": int(len(y)),
        "positives": int(y.sum()),
        "negatives": int((~y).sum()),
        "role": "DESCRIPTIVE_NON_GATING_ON_LONGCIU",
    }


def source_hash_rows(data_dir):
    rows = []
    for name in REQUIRED + ["longciu_splits.json", "README.md", "staple_pub_stats.json"]:
        p = data_dir / name
        if p.exists():
            rows.append({
                "filename": name,
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
            })
    return rows


@torch.inference_mode()
def run_external(model, data_dir, thresholds):
    img = as_90_slices(data_dir / "longciu_img.nii.gz").astype(np.float32)
    staple = as_90_slices(data_dir / "longciu_STAPLE_tgt.nii.gz").astype(np.int16)
    raters = {
        "rater1": as_90_slices(data_dir / "longciu_1_tgt.nii.gz").astype(np.int16),
        "rater2": as_90_slices(data_dir / "longciu_2_tgt.nii.gz").astype(np.int16),
        "rater3": as_90_slices(data_dir / "longciu_3_tgt.nii.gz").astype(np.int16),
    }

    if img.shape != staple.shape:
        raise RuntimeError(f"LongCIU image/STAPLE shape mismatch: {img.shape} vs {staple.shape}")
    for name, arr in raters.items():
        if arr.shape != staple.shape:
            raise RuntimeError(f"LongCIU {name}/STAPLE shape mismatch: {arr.shape} vs {staple.shape}")

    unique_staple = sorted(int(x) for x in np.unique(staple))
    if not set(unique_staple).issubset({0, 1, 2}):
        raise RuntimeError(f"Unexpected LongCIU STAPLE labels: {unique_staple}")

    slice_rows = []
    class_counts = {"GGO": [], "consolidation": []}
    rater_counts = {
        "rater1": {"GGO": [], "consolidation": []},
        "rater2": {"GGO": [], "consolidation": []},
        "rater3": {"GGO": [], "consolidation": []},
    }
    presence_truth = {"GGO": [], "consolidation": []}
    presence_scores = {"GGO": [], "consolidation": []}

    batch_x, batch_idx = [], []
    all_probs = {}

    for z in range(90):
        batch_x.append(preprocess_longciu_slice(img[z]))
        batch_idx.append(z)

        if len(batch_x) == 8 or z == 89:
            xx = torch.stack(batch_x).to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                probs = torch.softmax(model(xx), dim=1).float()
            probs = F.interpolate(
                probs,
                size=tuple(staple.shape[1:]),
                mode="bilinear",
                align_corners=False,
            ).cpu().numpy()
            for i, zz in enumerate(batch_idx):
                all_probs[zz] = probs[i]
            batch_x, batch_idx = [], []

    for z in tqdm(range(90), desc="LongCIU frozen external metrics"):
        prob = all_probs[z]
        pred = prob.argmax(0)

        for model_ch, gt_label, short in [
            (2, 1, "GGO"),
            (3, 2, "consolidation"),
        ]:
            pred_mask = pred == model_ch
            gt = staple[z] == gt_label
            m = counts_metrics(pred_mask, gt)
            class_counts[short].append(m)

            k = max(1, int(round(0.005 * prob.shape[-1] * prob.shape[-2])))
            score = float(np.partition(prob[model_ch].reshape(-1), -k)[-k:].mean())
            presence_truth[short].append(bool(gt.any()))
            presence_scores[short].append(score)

            row = {
                "slice_index": z,
                "class": short,
                "reference": "STAPLE",
                "Dice": m["Dice"],
                "IoU": m["IoU"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "true_presence": bool(gt.any()),
                "presence_score": score,
                "frozen_presence_threshold": float(thresholds[short]),
            }
            slice_rows.append(row)

            for rname, rarr in raters.items():
                rm = counts_metrics(pred_mask, rarr[z] == gt_label)
                rater_counts[rname][short].append(rm)

    return {
        "slice_rows": slice_rows,
        "class_counts": class_counts,
        "rater_counts": rater_counts,
        "presence_truth": presence_truth,
        "presence_scores": presence_scores,
        "staple_labels": unique_staple,
        "shape": list(staple.shape),
    }


def main():
    print("=" * 114)
    print("EViCT-Dx STEP 12B — LONGCIU REFERENCE-ORIENTATION CORRECTED EXTERNAL EVALUATION")
    print("FROZEN STEP11 MODEL — OFFICIAL [z,y,x] ARRAY CONVENTION — NO TUNING")
    print("=" * 114)

    subprocess.run(
        [sys.executable, str(DX / "scripts" / "verify_core_lock.py")],
        cwd=str(ROOT),
        check=True,
    )

    prior_results = AUDIT_DIR / "step12_longciu_external_results.json"
    prior_state = DX / "runs" / "DX_step12_longciu_external_ggo_consolidation_v1" / "STATE.json"
    if not prior_results.exists() or not prior_state.exists():
        raise RuntimeError("Step12 v1 artifacts are required for transparent correction traceability.")

    prior_state_payload = json.loads(prior_state.read_text())
    if prior_state_payload.get("status") != "COMPLETE":
        raise RuntimeError("Step12 v1 is not COMPLETE; correction provenance is incomplete.")

    atomic_json(CORRECTION_AUDIT, {
        "project": "EViCT-Dx",
        "stage": "STEP_12B_TECHNICAL_CORRECTION_AUDIT",
        "created_utc": now(),
        "prior_run_id": "DX_step12_longciu_external_ggo_consolidation_v1",
        "prior_results_retained": True,
        "prior_results_interpretation": "INVALIDATED_TECHNICAL_PREPROCESSING_ORIENTATION_MISMATCH",
        "issue": (
            "Step12 v1 used nibabel native [x,y,z] arrays and moved only the 90-slice axis "
            "to the front, yielding [z,x,y]. The official LongCIU reference code uses "
            "SimpleITK.GetArrayFromImage, yielding [z,y,x]."
        ),
        "correction": (
            "Step12B uses exactly one frozen conversion: nibabel [x,y,z] -> [z,y,x] "
            "via transpose(2,1,0), matching the official LongCIU array convention."
        ),
        "orientation_search": False,
        "performance_based_variant_selection": False,
        "model_changed": False,
        "thresholds_changed": False,
        "training_performed": False,
        "fine_tuning_performed": False
    })

    if STATE_JSON.exists():
        prior = json.loads(STATE_JSON.read_text())
        if prior.get("status") == "COMPLETE":
            print("✓ Step12 is already COMPLETE. Refusing to re-access the external test.")
            print(json.dumps(prior, indent=2))
            return

    step11_state = verify_step11()
    thresholds_payload = json.loads(STEP11_THRESH.read_text())
    thresholds = thresholds_payload["thresholds"]
    if thresholds_payload.get("selection_data") != "validation_only" or thresholds_payload.get("test_used"):
        raise RuntimeError("Step11 thresholds are not provenance-safe.")

    data_dir, acquisition = acquire_longciu()

    source_rows = source_hash_rows(data_dir)
    atomic_csv(HASH_MANIFEST, pd.DataFrame(source_rows))

    source_audit = {
        "project": "EViCT-Dx",
        "stage": "STEP_12_LONGCIU_SOURCE_AUDIT",
        "completed_utc": now(),
        "data_dir": str(data_dir),
        "acquisition": acquisition,
        "required_files_present": all((data_dir / n).exists() for n in REQUIRED),
        "source_hash_manifest": str(HASH_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "training_performed": False,
        "threshold_search_performed": False,
    }
    atomic_json(SOURCE_AUDIT, source_audit)

    model, ck = load_frozen_model()
    print(f"✓ Frozen Step11 best checkpoint loaded from epoch {ck.get('epoch')}.")
    print("✓ Frozen validation thresholds:")
    print(f"  GGO           : {thresholds['GGO']:.10f}")
    print(f"  consolidation : {thresholds['consolidation']:.10f}")

    # Once this state is written, all subsequent reruns are technical repeats only.
    atomic_json(STATE_JSON, {
        "run_id": RUN_ID,
        "status": "EXTERNAL_EVALUATION_RUNNING",
        "updated_utc": now(),
        "external_test_accessed": True,
        "training_performed": False,
        "fine_tuning_performed": False,
        "threshold_search_performed": False,
        "next_action": "COMPLETE_FROZEN_LONGCIU_INFERENCE",
    })
    sync_git("EViCT-Dx Step12B begin frozen LongCIU external evaluation")

    out = run_external(model, data_dir, thresholds)
    atomic_csv(SLICE_METRICS, pd.DataFrame(out["slice_rows"]))

    primary = {}
    descriptive_raters = {}
    presence = {}
    gate = {}
    metric_rows = []

    dice_min = float(CFG["external_confirmatory_gate"]["segmentation_Dice_min"])

    for short in ["GGO", "consolidation"]:
        pooled = pooled_from_counts(out["class_counts"][short])
        pooled["mean_slice_Dice"] = float(np.mean([r["Dice"] for r in out["class_counts"][short]]))
        primary[short] = pooled

        descriptive_raters[short] = {}
        for rname in ["rater1", "rater2", "rater3"]:
            rm = pooled_from_counts(out["rater_counts"][rname][short])
            rm["mean_slice_Dice"] = float(np.mean([r["Dice"] for r in out["rater_counts"][rname][short]]))
            descriptive_raters[short][rname] = rm

        presence[short] = presence_metrics(
            out["presence_truth"][short],
            out["presence_scores"][short],
            thresholds[short],
        )

        external_pass = bool(pooled["Dice"] >= dice_min)
        internal_pass = bool(
            step11_state["GGO_internal_gate_pass"]
            if short == "GGO"
            else step11_state["consolidation_internal_gate_pass"]
        )

        gate[short] = {
            "internal_step11_gate_pass": internal_pass,
            "external_STAPLE_Dice": pooled["Dice"],
            "external_Dice_required_min": dice_min,
            "external_confirmatory_gate_pass": external_pass,
            "final_subtype_validation_pass": bool(internal_pass and external_pass),
        }

        metric_rows.append({
            "class": short,
            "STAPLE_Dice": pooled["Dice"],
            "STAPLE_IoU": pooled["IoU"],
            "STAPLE_sensitivity": pooled["sensitivity"],
            "STAPLE_specificity": pooled["specificity"],
            "mean_slice_Dice": pooled["mean_slice_Dice"],
            "presence_AUROC_descriptive": presence[short]["AUROC"],
            "presence_sensitivity_descriptive": presence[short]["sensitivity"],
            "presence_specificity_descriptive": presence[short]["specificity"],
            "presence_precision_descriptive": presence[short]["precision"],
            "external_confirmatory_gate_pass": external_pass,
            "final_subtype_validation_pass": bool(internal_pass and external_pass),
        })

    atomic_csv(EXTERNAL_METRICS, pd.DataFrame(metric_rows))

    step09_permissions = json.loads(
        (DX / "config" / "report_field_permissions_step09.json").read_text()
    )
    permissions = json.loads(json.dumps(step09_permissions))
    permissions["stage"] = "STEP_12B_DERIVED_REPORT_FIELD_PERMISSIONS"
    permissions["created_utc"] = now()
    permissions["derived_from"] = [
        "vlmDiagnosis/config/report_field_permissions_step09.json",
        "vlmDiagnosis/artifacts/audit/step11_ncp_ggo_consolidation_results.json",
        "vlmDiagnosis/artifacts/audit/step12b_longciu_external_results.json",
    ]

    for short, field in [("GGO", "GGO"), ("consolidation", "Consolidation")]:
        if gate[short]["final_subtype_validation_pass"]:
            permissions["fields"][field] = (
                "UNLOCKED_FOR_PROVISIONAL_AI_REPORT_WITH_MANDATORY_RADIOLOGIST_REVIEW"
            )
        else:
            permissions["fields"][field] = "LOCKED_FAILED_EXTERNAL_SUBTYPE_VALIDATION"

    atomic_json(PERMISSIONS_OUT, permissions)

    results = {
        "project": "EViCT-Dx",
        "stage": "STEP_12B_LONGCIU_REFERENCE_ORIENTATION_CORRECTED_EXTERNAL_EVALUATION",
        "run_id": RUN_ID,
        "completed_utc": now(),
        "source": {
            "name": CFG["source"]["name"],
            "doi": CFG["source"]["doi"],
            "reference": "STAPLE consensus",
            "shape_after_slice_axis_normalization": out["shape"],
            "STAPLE_labels_observed": out["staple_labels"],
            "label_mapping": CFG["source"]["label_mapping"],
        },
        "frozen_step11_presence_thresholds": thresholds,
        "training_performed": False,
        "fine_tuning_performed": False,
        "checkpoint_selection_performed": False,
        "threshold_search_performed": False,
        "orientation_search_performed": False,
        "technical_correction": {
            "prior_step12_v1_retained": True,
            "prior_step12_v1_interpretation": "INVALIDATED_TECHNICAL_PREPROCESSING_ORIENTATION_MISMATCH",
            "array_convention": "official LongCIU SimpleITK-compatible [z,y,x]",
            "orientation_search": False,
            "performance_based_variant_selection": False
        },
        "primary_STAPLE_metrics": primary,
        "frozen_threshold_presence_metrics_descriptive": presence,
        "individual_rater_metrics_descriptive": descriptive_raters,
        "external_confirmatory_gate": gate,
        "report_field_permissions_file": str(PERMISSIONS_OUT.relative_to(ROOT)).replace("\\", "/"),
        "next_action": "STEP_13_STRUCTURED_EVIDENCE_AND_REPORT_GENERATION",
    }
    atomic_json(RESULTS_JSON, results)

    final_state = {
        "run_id": RUN_ID,
        "status": "COMPLETE",
        "updated_utc": now(),
        "external_test_accessed": True,
        "training_performed": False,
        "fine_tuning_performed": False,
        "threshold_search_performed": False,
        "GGO_external_gate_pass": bool(gate["GGO"]["external_confirmatory_gate_pass"]),
        "consolidation_external_gate_pass": bool(gate["consolidation"]["external_confirmatory_gate_pass"]),
        "GGO_report_field_unlocked": permissions["fields"]["GGO"].startswith("UNLOCKED"),
        "consolidation_report_field_unlocked": permissions["fields"]["Consolidation"].startswith("UNLOCKED"),
        "next_action": "STEP_13_STRUCTURED_EVIDENCE_AND_REPORT_GENERATION",
    }
    atomic_json(STATE_JSON, final_state)

    sync_git("Complete EViCT-Dx Step12B LongCIU external validation")

    print("\n" + "=" * 114)
    print("✅ STEP 12B COMPLETE — CORRECTED LONGCIU EXTERNAL VALIDATION")
    for short in ["GGO", "consolidation"]:
        m = primary[short]
        p = presence[short]
        g = gate[short]
        print(
            f"{short:14s} STAPLE Dice={m['Dice']:.4f} IoU={m['IoU']:.4f} "
            f"sens={m['sensitivity']:.4f} spec={m['specificity']:.4f} "
            f"| frozen-threshold presence AUROC={p['AUROC'] if np.isfinite(p['AUROC']) else float('nan'):.4f} "
            f"| EXTERNAL_GATE={'PASS' if g['external_confirmatory_gate_pass'] else 'FAIL'}"
        )
    print(f"GGO report field           : {permissions['fields']['GGO']}")
    print(f"Consolidation report field : {permissions['fields']['Consolidation']}")
    print("Training / fine-tuning     : NONE")
    print("Threshold search           : NONE")
    print("NEXT                       : STEP_13_STRUCTURED_EVIDENCE_AND_REPORT_GENERATION")
    print("=" * 114)


if __name__ == "__main__":
    main()
