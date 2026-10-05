from __future__ import annotations

import io
import json
import os
import random
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    convnext_tiny,
    efficientnet_b0,
)
from torchvision.transforms.functional import resize
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
DX = ROOT / "vlmDiagnosis"
RUN_ID = "DX_step07_frozen_dual_encoder_seed1705_v1"
RUN = DX / "runs" / RUN_ID
DESC_CACHE = RUN / "descriptor_cache"
MODEL_DIR = RUN / "final_models"
TABLES = DX / "tables"
AUDIT = DX / "artifacts" / "audit"
CFG_PATH = DX / "config" / "diagnosis_frozen_dual_encoder_step07.json"
INV_PATH = TABLES / "step03d_covid_ct_md_dicom_inventory.csv"
SPLIT_PATH = TABLES / "step03d_covid_ct_md_split_manifest.csv"
SNAPSHOT_DIR = Path("/kaggle/working/evict_dx_step07_snapshots")

for d in [RUN, DESC_CACHE, MODEL_DIR, TABLES, AUDIT, SNAPSHOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DX))
from runtime.recovery import GitHubReleaseStore, RecoveryPolicy, kaggle_secret, sha256_file  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    raise RuntimeError("Enable a Kaggle GPU for Step 07 descriptor extraction.")

CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
SEED = int(CFG["seed"])
CLASSES = list(CFG["classes"])
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
N_SLICES = int(CFG["input"]["slices_per_patient"])
BATCH_SLICES = 16

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


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


def locate_archive():
    preferred = Path("/kaggle/working/evict_dx_figshare_temp/COVID-CT-MD.zip")
    if preferred.exists():
        return preferred
    hits = []
    for base in [Path("/kaggle/working"), Path("/kaggle/input")]:
        if base.exists():
            hits.extend(base.rglob("COVID-CT-MD.zip"))
    if not hits:
        raise FileNotFoundError("COVID-CT-MD.zip not found.")
    return sorted(hits, key=lambda p: len(str(p)))[0]


def load_policy():
    raw = json.loads((DX / "config" / "recovery_policy.json").read_text(encoding="utf-8"))
    allowed = set(RecoveryPolicy.__dataclass_fields__)
    return RecoveryPolicy(**{k: v for k, v in raw.items() if k in allowed})


POLICY = load_policy()


def release_store():
    return GitHubReleaseStore(
        repository=POLICY.github_repository,
        token=kaggle_secret(POLICY.kaggle_secret_name),
        release_prefix=POLICY.rolling_release_prefix,
    )


def sync_git(message):
    script = DX / "scripts" / "git_sync_dx.py"
    if not script.exists():
        print("⚠ git_sync_dx.py missing; local metadata retained.")
        return
    r = subprocess.run([sys.executable, str(script), message], cwd=str(ROOT), check=False)
    if r.returncode != 0:
        print(f"⚠ Git sync returned {r.returncode}; local metadata retained.")


def window_channel(hu: torch.Tensor, lo: float, hi: float):
    return ((hu.clamp(lo, hi) - lo) / (hi - lo)).float()


def dicom_bytes_to_tensor(raw: bytes):
    ds = pydicom.dcmread(io.BytesIO(raw), force=False)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = torch.from_numpy(arr * slope + intercept)

    channels = []
    for w in CFG["input"]["channels"]:
        channels.append(window_channel(hu, float(w["hu_min"]), float(w["hu_max"])))
    x = torch.stack(channels, 0)
    x = resize(x, list(CFG["input"]["resize"]), antialias=True)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def sample_members(df: pd.DataFrame):
    df = df.reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("Empty patient DICOM inventory.")
    ids = np.linspace(0, len(df) - 1, N_SLICES).round().astype(int)
    return df.iloc[ids]["member"].tolist()


class DualFrozenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.eff = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.cnext = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        for p in self.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, x):
        e = self.eff.features(x)
        e = self.eff.avgpool(e)
        e = torch.flatten(e, 1)

        c = self.cnext.features(x)
        c = self.cnext.avgpool(c)
        # ConvNeXt classifier is norm -> flatten -> linear. Keep pre-linear representation.
        for layer in list(self.cnext.classifier.children())[:-1]:
            c = layer(c)
        return e, c


def robust_stats(features: np.ndarray):
    # [slices, dim] -> [6*dim]
    stats = [
        features.mean(axis=0),
        features.std(axis=0),
        features.max(axis=0),
        np.quantile(features, 0.25, axis=0),
        np.quantile(features, 0.50, axis=0),
        np.quantile(features, 0.75, axis=0),
    ]
    return np.concatenate(stats, axis=0).astype(np.float32)


@torch.inference_mode()
def patient_descriptor(zf, members, encoder):
    eff_all, cnext_all = [], []
    for i in range(0, len(members), BATCH_SLICES):
        xs = []
        for member in members[i:i + BATCH_SLICES]:
            with zf.open(member, "r") as f:
                xs.append(dicom_bytes_to_tensor(f.read()))
        x = torch.stack(xs).to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            e, c = encoder(x)
        eff_all.append(e.float().cpu().numpy())
        cnext_all.append(c.float().cpu().numpy())

    eff = np.concatenate(eff_all, 0)
    cnext = np.concatenate(cnext_all, 0)
    desc = np.concatenate([robust_stats(eff), robust_stats(cnext)]).astype(np.float32)

    expected = int(CFG["patient_descriptor"]["dimension"])
    if desc.shape != (expected,):
        raise RuntimeError(f"Descriptor shape mismatch: {desc.shape} != {(expected,)}")
    return desc


def latest_snapshot_name(store):
    assets = store.list_assets(RUN_ID)
    names = sorted(
        n for n in assets
        if n.startswith("descriptor_cache_") and n.endswith(".tar")
    )
    return names[-1] if names else None


def restore_descriptor_snapshot():
    state_path = RUN / "descriptor_state.json"
    if state_path.exists() and any(DESC_CACHE.glob("*.npz")):
        return
    try:
        store = release_store()
        name = latest_snapshot_name(store)
        if not name:
            return
        local = SNAPSHOT_DIR / name
        store.download(RUN_ID, name, local)
        with tarfile.open(local, "r") as tf:
            tf.extractall(RUN)
        print(f"✓ Restored durable Step07 descriptor snapshot: {name}")
    except Exception as exc:
        print("ℹ No remote Step07 descriptor snapshot restored:", exc)


def save_descriptor_snapshot(completed_count):
    state_path = RUN / "descriptor_state.json"
    if not state_path.exists():
        return

    name = f"descriptor_cache_{int(completed_count):03d}.tar"
    local = SNAPSHOT_DIR / name
    tmp = local.with_suffix(".tar.tmp")
    with tarfile.open(tmp, "w") as tf:
        tf.add(state_path, arcname="descriptor_state.json")
        for p in sorted(DESC_CACHE.glob("*.npz")):
            tf.add(p, arcname=f"descriptor_cache/{p.name}")
    os.replace(tmp, local)

    store = release_store()
    store.upload_or_replace(RUN_ID, local, asset_name=name)
    store.prune_assets(RUN_ID, prefix="descriptor_cache_", keep=2)
    print(f"✓ Durable descriptor snapshot: {name} ({local.stat().st_size/(1024**2):.1f} MB)")


def build_descriptors(inventory, split_df):
    restore_descriptor_snapshot()

    dev = split_df[split_df["split"].isin(["train", "validation"])].copy()
    if len(dev) != 244:
        raise RuntimeError(f"Expected 244 train+validation patients, found {len(dev)}.")
    if (dev["split"] == "test").any():
        raise RuntimeError("TEST LEAKAGE GUARD: test patient entered Step07 development set.")

    allowed = set(map(tuple, dev[["diagnosis", "patient_id"]].to_records(index=False)))
    groups = [
        (key, df)
        for key, df in inventory.groupby(["diagnosis", "patient_id"], sort=True)
        if key in allowed
    ]
    if len(groups) != 244:
        raise RuntimeError(f"Expected 244 inventory groups, found {len(groups)}.")

    state_path = RUN / "descriptor_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    completed = set(state.get("completed", []))

    encoder = DualFrozenEncoder().to(DEVICE).eval()
    archive = locate_archive()
    newly_completed = 0

    with zipfile.ZipFile(archive, "r") as zf:
        bar = tqdm(groups, total=len(groups), desc="Step07 dual-encoder descriptors")
        for (diagnosis, pid), df in bar:
            key = f"{diagnosis}|{pid}"
            out = DESC_CACHE / f"{diagnosis.replace('-', '').replace(' ', '_')}__{pid}.npz"

            if key in completed and out.exists():
                bar.set_postfix(done=len(completed), status="resume-skip")
                continue

            members = sample_members(df)
            desc = patient_descriptor(zf, members, encoder)

            tmp = out.with_suffix(".npz.tmp")
            with tmp.open("wb") as f:
                np.savez_compressed(
                    f,
                    descriptor=desc,
                    diagnosis=diagnosis,
                    patient_id=pid,
                )
            os.replace(tmp, out)

            completed.add(key)
            newly_completed += 1
            atomic_json(state_path, {
                "run_id": RUN_ID,
                "stage": "STEP_07_DESCRIPTOR_EXTRACTION",
                "completed": sorted(completed),
                "completed_count": len(completed),
                "total": 244,
                "updated_utc": now(),
            })
            bar.set_postfix(done=len(completed), patient=pid)

            every = int(CFG["recovery"]["descriptor_snapshot_every_patients"])
            if newly_completed > 0 and len(completed) % every == 0:
                save_descriptor_snapshot(len(completed))
                sync_git(f"EViCT-Dx Step07 descriptors {len(completed)}/244")

    if len(completed) != 244:
        raise RuntimeError(f"Step07 descriptor extraction incomplete: {len(completed)}/244")

    save_descriptor_snapshot(244)

    rows = []
    X = []
    for _, row in dev.iterrows():
        diagnosis, pid = row["diagnosis"], row["patient_id"]
        p = DESC_CACHE / f"{diagnosis.replace('-', '').replace(' ', '_')}__{pid}.npz"
        if not p.exists():
            raise FileNotFoundError(p)
        z = np.load(p, allow_pickle=False)
        X.append(z["descriptor"].astype(np.float32))
        rows.append({
            "diagnosis": diagnosis,
            "patient_id": pid,
            "split": row["split"],
            "descriptor_path": str(p.relative_to(ROOT)),
            "descriptor_sha256": sha256_file(p),
        })

    manifest = pd.DataFrame(rows)
    if set(manifest["split"]) != {"train", "validation"}:
        raise RuntimeError("TEST LEAKAGE GUARD: descriptor manifest contains unexpected split.")

    atomic_csv(TABLES / "step07_descriptor_manifest.csv", manifest)
    X = np.stack(X)
    return manifest, X


def align_probs(model, probs):
    aligned = np.zeros((len(probs), len(CLASSES)), dtype=np.float64)
    for j, cls_idx in enumerate(model.classes_):
        aligned[:, int(cls_idx)] = probs[:, j]
    return aligned


def metrics(y, probs):
    pred = probs.argmax(1)
    onehot = np.eye(len(CLASSES))[y]
    return {
        "macro_F1": float(f1_score(y, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_AUROC": float(roc_auc_score(onehot, probs, multi_class="ovr", average="macro")),
    }


def detailed_metrics(y, probs):
    pred = probs.argmax(1)
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    onehot = np.eye(len(CLASSES))[y]
    per_class = {}
    for i, cls in enumerate(CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        per_class[cls] = {
            "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
            "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
            "AUROC": float(roc_auc_score((y == i).astype(int), probs[:, i])),
        }
    return {
        **metrics(y, probs),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def candidate_specs():
    specs = []
    for pca in CFG["train_only_model_selection"]["logistic"]["pca_components"]:
        for C in CFG["train_only_model_selection"]["logistic"]["C"]:
            specs.append(("logistic", int(pca), float(C)))

    for pca in CFG["train_only_model_selection"]["rbf_svc"]["pca_components"]:
        for C in CFG["train_only_model_selection"]["rbf_svc"]["C"]:
            specs.append(("rbf_svc", int(pca), float(C)))
    return specs


def fit_classifier(kind, C):
    if kind == "logistic":
        return LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=3000,
            solver="lbfgs",
            random_state=SEED,
        )
    if kind == "rbf_svc":
        return SVC(
            C=C,
            gamma="scale",
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=SEED,
        )
    raise ValueError(kind)


def train_only_cv(X_train, y_train):
    specs = candidate_specs()
    oof = {
        (kind, pca, C): np.zeros((len(y_train), len(CLASSES)), dtype=np.float64)
        for kind, pca, C in specs
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(X_train, y_train))

    outer = tqdm(enumerate(folds, 1), total=len(folds), desc="Step07 train-only 5-fold CV")
    for fold, (tr, va) in outer:
        scaler = StandardScaler()
        Xtr_scaled = scaler.fit_transform(X_train[tr])
        Xva_scaled = scaler.transform(X_train[va])

        pca_cache = {}
        for ncomp in sorted(set(pca for _, pca, _ in specs)):
            pca = PCA(n_components=ncomp, svd_solver="randomized", random_state=SEED + fold + ncomp)
            pca_cache[ncomp] = (
                pca.fit_transform(Xtr_scaled),
                pca.transform(Xva_scaled),
            )

        for kind, ncomp, C in specs:
            Xtr_pca, Xva_pca = pca_cache[ncomp]
            clf = fit_classifier(kind, C)
            clf.fit(Xtr_pca, y_train[tr])
            probs = align_probs(clf, clf.predict_proba(Xva_pca))
            oof[(kind, ncomp, C)][va] = probs

        outer.set_postfix(fold=fold, candidates=len(specs))

    rows = []
    for key, probs in oof.items():
        kind, ncomp, C = key
        m = metrics(y_train, probs)
        rows.append({
            "kind": kind,
            "pca_components": ncomp,
            "C": C,
            **m,
        })

    results = pd.DataFrame(rows).sort_values(
        ["macro_F1", "macro_AUROC", "balanced_accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    atomic_csv(TABLES / "step07_train_oof_candidate_results.csv", results)

    best_log_row = (
        results[results.kind == "logistic"]
        .sort_values(["macro_F1", "macro_AUROC"], ascending=False)
        .iloc[0]
    )
    best_svc_row = (
        results[results.kind == "rbf_svc"]
        .sort_values(["macro_F1", "macro_AUROC"], ascending=False)
        .iloc[0]
    )

    log_key = ("logistic", int(best_log_row.pca_components), float(best_log_row.C))
    svc_key = ("rbf_svc", int(best_svc_row.pca_components), float(best_svc_row.C))
    p_log = oof[log_key]
    p_svc = oof[svc_key]

    blend_rows = []
    for w in CFG["train_only_model_selection"]["blend_weights"]:
        w = float(w)  # weight on logistic model
        p = w * p_log + (1.0 - w) * p_svc
        blend_rows.append({
            "logistic_weight": w,
            **metrics(y_train, p),
        })

    blend = pd.DataFrame(blend_rows).sort_values(
        ["macro_F1", "macro_AUROC", "balanced_accuracy", "logistic_weight"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    atomic_csv(TABLES / "step07_train_oof_blend_results.csv", blend)

    best_blend = blend.iloc[0]

    options = [
        {
            "type": "logistic",
            "macro_F1": float(best_log_row.macro_F1),
            "macro_AUROC": float(best_log_row.macro_AUROC),
            "balanced_accuracy": float(best_log_row.balanced_accuracy),
        },
        {
            "type": "rbf_svc",
            "macro_F1": float(best_svc_row.macro_F1),
            "macro_AUROC": float(best_svc_row.macro_AUROC),
            "balanced_accuracy": float(best_svc_row.balanced_accuracy),
        },
        {
            "type": "blend",
            "macro_F1": float(best_blend.macro_F1),
            "macro_AUROC": float(best_blend.macro_AUROC),
            "balanced_accuracy": float(best_blend.balanced_accuracy),
        },
    ]
    selected_type = sorted(
        options,
        key=lambda x: (x["macro_F1"], x["macro_AUROC"], x["balanced_accuracy"]),
        reverse=True,
    )[0]["type"]

    selection = {
        "best_logistic": {
            "kind": "logistic",
            "pca_components": int(best_log_row.pca_components),
            "C": float(best_log_row.C),
            "OOF_macro_F1": float(best_log_row.macro_F1),
            "OOF_macro_AUROC": float(best_log_row.macro_AUROC),
        },
        "best_rbf_svc": {
            "kind": "rbf_svc",
            "pca_components": int(best_svc_row.pca_components),
            "C": float(best_svc_row.C),
            "OOF_macro_F1": float(best_svc_row.macro_F1),
            "OOF_macro_AUROC": float(best_svc_row.macro_AUROC),
        },
        "best_blend": {
            "logistic_weight": float(best_blend.logistic_weight),
            "OOF_macro_F1": float(best_blend.macro_F1),
            "OOF_macro_AUROC": float(best_blend.macro_AUROC),
        },
        "selected_type": selected_type,
    }
    atomic_json(RUN / "train_only_selection.json", selection)
    return selection


def make_pipeline(kind, ncomp, C):
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=int(ncomp), svd_solver="randomized", random_state=SEED)),
        ("clf", fit_classifier(kind, float(C))),
    ])


def fit_selected_and_validate(manifest, X, selection):
    train_mask = (manifest["split"].to_numpy() == "train")
    val_mask = (manifest["split"].to_numpy() == "validation")
    if int(train_mask.sum()) != 183 or int(val_mask.sum()) != 61:
        raise RuntimeError("Unexpected Step07 train/validation patient counts.")

    y_all = manifest["diagnosis"].map(CLASS_TO_IDX).to_numpy(int)
    X_train, y_train = X[train_mask], y_all[train_mask]
    X_val, y_val = X[val_mask], y_all[val_mask]

    log_cfg = selection["best_logistic"]
    svc_cfg = selection["best_rbf_svc"]

    log_model = make_pipeline("logistic", log_cfg["pca_components"], log_cfg["C"])
    svc_model = make_pipeline("rbf_svc", svc_cfg["pca_components"], svc_cfg["C"])

    print("Fitting selected train-only models on all 183 training patients...")
    log_model.fit(X_train, y_train)
    svc_model.fit(X_train, y_train)

    log_probs = align_probs(log_model.named_steps["clf"], log_model.predict_proba(X_val))
    svc_probs = align_probs(svc_model.named_steps["clf"], svc_model.predict_proba(X_val))

    selected_type = selection["selected_type"]
    if selected_type == "logistic":
        probs = log_probs
    elif selected_type == "rbf_svc":
        probs = svc_probs
    elif selected_type == "blend":
        w = float(selection["best_blend"]["logistic_weight"])
        probs = w * log_probs + (1.0 - w) * svc_probs
    else:
        raise RuntimeError(selected_type)

    val_metrics = detailed_metrics(y_val, probs)
    val_rows = []
    val_meta = manifest[val_mask].reset_index(drop=True)
    pred = probs.argmax(1)
    for i in range(len(val_meta)):
        row = {
            "patient_id": val_meta.loc[i, "patient_id"],
            "diagnosis": val_meta.loc[i, "diagnosis"],
            "target": int(y_val[i]),
            "prediction": int(pred[i]),
        }
        for j, cls in enumerate(CLASSES):
            row[f"prob_{cls}"] = float(probs[i, j])
        val_rows.append(row)
    atomic_csv(TABLES / "step07_validation_predictions.csv", pd.DataFrame(val_rows))

    log_path = MODEL_DIR / "step07_logistic.joblib"
    svc_path = MODEL_DIR / "step07_rbf_svc.joblib"
    joblib.dump(log_model, log_path)
    joblib.dump(svc_model, svc_path)

    try:
        store = release_store()
        store.upload_or_replace(RUN_ID, log_path, asset_name="step07_logistic.joblib")
        store.upload_or_replace(RUN_ID, svc_path, asset_name="step07_rbf_svc.joblib")
        print("✓ Step07 fitted models uploaded as durable release assets.")
    except Exception as exc:
        print("⚠ Model release upload failed:", exc)

    baseline = float(CFG["validation"]["reference_step04_macro_F1"])
    audit = {
        "project": "EViCT-Dx",
        "stage": "STEP_07_FROZEN_DUAL_ENCODER_DEVELOPMENT",
        "run_id": RUN_ID,
        "completed_utc": now(),
        "held_out_test_accessed": False,
        "held_out_test_inference_run": False,
        "development_patients": {"train": 183, "validation": 61, "test": 0},
        "descriptor": "48 ordered slices; frozen EfficientNet-B0 + frozen ConvNeXt-Tiny; mean/std/max/Q25/median/Q75",
        "train_only_selection": selection,
        "validation_metrics": val_metrics,
        "step04_validation_macro_F1": baseline,
        "validation_macro_F1_delta_vs_step04": float(val_metrics["macro_F1"] - baseline),
        "report_fields_unlocked": False,
        "decision": (
            "CANDIDATE_IMPROVED_ON_VALIDATION"
            if val_metrics["macro_F1"] > baseline
            else "CANDIDATE_DID_NOT_IMPROVE_STEP04_BASELINE"
        ),
        "next_action": (
            "FREEZE_NEXT_EVALUATION_PROTOCOL"
            if val_metrics["macro_F1"] > baseline
            else "RETAIN_STEP04_AS_DIAGNOSIS_BASELINE_AND_DO_NOT_USE_STEP07_ON_TEST"
        ),
    }
    atomic_json(AUDIT / "step07_frozen_dual_encoder_development.json", audit)
    atomic_json(RUN / "STATE.json", {
        "run_id": RUN_ID,
        "status": "COMPLETE",
        "updated_utc": now(),
        "held_out_test_accessed": False,
        "selected_type": selection["selected_type"],
        "validation_macro_F1": val_metrics["macro_F1"],
        "next_action": audit["next_action"],
    })

    sync_git("Complete EViCT-Dx Step07 frozen dual-encoder development without test access")

    print("\n" + "=" * 108)
    print("✅ STEP 07 COMPLETE — DEVELOPMENT ONLY")
    print(f"Selected train-only model        : {selection['selected_type']}")
    print(f"Train OOF selected Macro-F1      : {max(selection['best_logistic']['OOF_macro_F1'], selection['best_rbf_svc']['OOF_macro_F1'], selection['best_blend']['OOF_macro_F1']):.4f}")
    print(f"Validation Macro-F1              : {val_metrics['macro_F1']:.4f}")
    print(f"Validation Balanced Accuracy     : {val_metrics['balanced_accuracy']:.4f}")
    print(f"Validation Macro-AUROC           : {val_metrics['macro_AUROC']:.4f}")
    print(f"Δ Macro-F1 vs Step04 validation  : {val_metrics['macro_F1'] - baseline:+.4f}")
    print("Held-out 61-patient test         : NOT ACCESSED")
    print("Diagnostic report fields         : STILL LOCKED")
    print("Decision                         :", audit["decision"])
    print("NEXT                             :", audit["next_action"])
    print("=" * 108)


def main():
    print("=" * 108)
    print("EViCT-Dx STEP 07 — FROZEN DUAL-ENCODER PATIENT CLASSIFICATION")
    print("EfficientNet-B0 + ConvNeXt-Tiny descriptors; train-only CV; validation once")
    print("=" * 108)
    print("GPU:", torch.cuda.get_device_name(0))

    verify = DX / "scripts" / "verify_core_lock.py"
    if verify.exists():
        subprocess.run([sys.executable, str(verify)], cwd=str(ROOT), check=True)

    for p in [CFG_PATH, INV_PATH, SPLIT_PATH]:
        if not p.exists():
            raise FileNotFoundError(p)

    completed = AUDIT / "step07_frozen_dual_encoder_development.json"
    if completed.exists():
        print("✓ Step07 already complete. Refusing accidental rerun.")
        print(completed)
        return

    inventory = pd.read_csv(INV_PATH)
    split_df = pd.read_csv(SPLIT_PATH)

    if len(split_df) != 305:
        raise RuntimeError(f"Expected 305 frozen patients, found {len(split_df)}.")
    if split_df.duplicated(["diagnosis", "patient_id"]).any():
        raise RuntimeError("Duplicate patient split assignment detected.")
    if int((split_df["split"] == "test").sum()) != 61:
        raise RuntimeError("Frozen held-out test count is not 61.")

    # Explicit guard: Step07 never loads any Step04 test-prediction artifact.
    forbidden = TABLES / "step04_diagnosis_test_predictions.csv"
    print("Leakage guard: Step04 test predictions exist but are intentionally NOT read:", forbidden.exists())

    manifest, X = build_descriptors(inventory, split_df)

    train_mask = manifest["split"].to_numpy() == "train"
    y = manifest.loc[train_mask, "diagnosis"].map(CLASS_TO_IDX).to_numpy(int)
    selection = train_only_cv(X[train_mask], y)

    atomic_csv(
        TABLES / "step07_descriptor_manifest.csv",
        manifest,
    )
    sync_git("Freeze EViCT-Dx Step07 train-validation descriptor manifest and train-only selection")

    fit_selected_and_validate(manifest, X, selection)


if __name__ == "__main__":
    main()
