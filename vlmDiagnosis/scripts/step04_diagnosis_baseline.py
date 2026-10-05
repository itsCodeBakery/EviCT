from __future__ import annotations

import io
import json
import math
import os
import random
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.transforms.functional import resize

ROOT = Path("/kaggle/working/EViCT")
if not ROOT.exists():
    ROOT = Path("/kaggle/working/EviCT")
DX = ROOT / "vlmDiagnosis"
RUN_ID = "DX_step04_covid_ct_md_effb0_attention_seed1705_v1"
RUN = DX / "runs" / RUN_ID
CACHE = RUN / "feature_cache"
TABLES = DX / "tables"
AUDIT = DX / "artifacts" / "audit"
CFG_PATH = DX / "config" / "diagnosis_baseline_step04.json"
INV_PATH = TABLES / "step03d_covid_ct_md_dicom_inventory.csv"
SPLIT_PATH = TABLES / "step03d_covid_ct_md_split_manifest.csv"

for d in [RUN, CACHE, TABLES, AUDIT]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DX))
from runtime.recovery import (  # noqa: E402
    GitHubReleaseStore,
    RecoveryPolicy,
    RunRecoveryManager,
    kaggle_secret,
    sha256_file,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    raise RuntimeError("Enable a Kaggle GPU for Step 04.")

CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
SEED = int(CFG["seed"])
CLASSES = list(CFG["classes"])
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
N_SLICES = int(CFG["preprocessing"]["slices_per_volume"])
BATCH_SLICES = 16

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


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


def sync_git(message):
    script = DX / "scripts" / "git_sync_dx.py"
    if not script.exists():
        print("⚠ git_sync_dx.py missing; local artifacts retained.")
        return
    r = subprocess.run([sys.executable, str(script), message], cwd=str(ROOT), check=False)
    if r.returncode != 0:
        print(f"⚠ Git sync returned {r.returncode}; local artifacts retained.")


def window_channel(hu: torch.Tensor, lo: float, hi: float):
    return ((hu.clamp(lo, hi) - lo) / (hi - lo)).float()


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def dicom_to_tensor(raw: bytes):
    ds = pydicom.dcmread(io.BytesIO(raw), force=False)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = torch.from_numpy(arr * slope + intercept)

    channels = []
    for w in CFG["preprocessing"]["channels"]:
        channels.append(window_channel(hu, float(w["hu_min"]), float(w["hu_max"])))
    x = torch.stack(channels, dim=0)
    x = resize(x, list(CFG["preprocessing"]["resize"]), antialias=True)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x


class FrozenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
        self.features = model.features
        self.pool = model.avgpool
        for p in self.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


def selected_members(patient_df: pd.DataFrame):
    patient_df = patient_df.reset_index(drop=True)
    n = len(patient_df)
    ids = np.linspace(0, n - 1, N_SLICES).round().astype(int)
    return patient_df.iloc[ids]["member"].tolist()


@torch.inference_mode()
def extract_patient_features(zf, members, encoder):
    feats = []
    for i in range(0, len(members), BATCH_SLICES):
        batch_members = members[i:i + BATCH_SLICES]
        xs = []
        for member in batch_members:
            with zf.open(member, "r") as f:
                xs.append(dicom_to_tensor(f.read()))
        x = torch.stack(xs).to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            f = encoder(x)
        feats.append(f.float().cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float16)


def build_feature_cache(inventory: pd.DataFrame, split_df: pd.DataFrame):
    archive = locate_archive()
    encoder = FrozenEncoder().to(DEVICE).eval()

    state_path = RUN / "feature_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    completed = set(state.get("completed", []))

    grouped = inventory.groupby(["diagnosis", "patient_id"], sort=True)
    with zipfile.ZipFile(archive, "r") as zf:
        for k, ((diagnosis, pid), df) in enumerate(grouped, 1):
            key = f"{diagnosis}|{pid}"
            out = CACHE / f"{diagnosis.replace('-', '').replace(' ', '_')}__{pid}.npz"
            if out.exists() and key in completed:
                print(f"[FEATURE {k:03d}/305] SKIP {diagnosis:9s} {pid}")
                continue

            members = selected_members(df)
            print(f"[FEATURE {k:03d}/305] {diagnosis:9s} {pid}  slices={len(members)}")
            features = extract_patient_features(zf, members, encoder)
            tmp = out.with_suffix(".npz.tmp")
            with tmp.open("wb") as f:
                np.savez_compressed(
                    f,
                    features=features,
                    diagnosis=diagnosis,
                    patient_id=pid,
                    members=np.array(members, dtype=object),
                )
            os.replace(tmp, out)

            completed.add(key)
            atomic_json(state_path, {
                "run_id": RUN_ID,
                "stage": "STEP_04_FEATURE_CACHE",
                "completed": sorted(completed),
                "completed_count": len(completed),
                "total": 305,
                "updated_utc": now(),
            })

            if len(completed) % 25 == 0:
                sync_git(f"EViCT-Dx Step04 feature cache progress {len(completed)}/305")

    if len(completed) != 305:
        raise RuntimeError(f"Feature cache incomplete: {len(completed)}/305")

    manifest_rows = []
    for _, row in split_df.iterrows():
        diagnosis, pid = row["diagnosis"], row["patient_id"]
        p = CACHE / f"{diagnosis.replace('-', '').replace(' ', '_')}__{pid}.npz"
        if not p.exists():
            raise FileNotFoundError(p)
        manifest_rows.append({
            "diagnosis": diagnosis,
            "patient_id": pid,
            "split": row["split"],
            "feature_path": str(p.relative_to(ROOT)),
            "feature_sha256": sha256_file(p),
        })
    manifest = pd.DataFrame(manifest_rows)
    atomic_csv(TABLES / "step04_diagnosis_feature_manifest.csv", manifest)
    return manifest


class FeatureDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        z = np.load(ROOT / row["feature_path"], allow_pickle=True)
        x = torch.from_numpy(z["features"].astype(np.float32))
        y = CLASS_TO_IDX[row["diagnosis"]]
        return x, y, row["patient_id"], row["diagnosis"]


class GatedAttentionClassifier(nn.Module):
    def __init__(self, d=1280, a=256, n_classes=3, dropout=0.25):
        super().__init__()
        self.v = nn.Linear(d, a)
        self.u = nn.Linear(d, a)
        self.w = nn.Linear(a, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d, n_classes)

    def forward(self, x):
        # x: [B, S, D]
        a = self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))).squeeze(-1)
        a = torch.softmax(a, dim=1)
        pooled = torch.sum(x * a.unsqueeze(-1), dim=1)
        logits = self.classifier(self.dropout(pooled))
        return logits, a


def make_loader(df, shuffle):
    return DataLoader(
        FeatureDataset(df),
        batch_size=int(CFG["training"]["patient_batch_size"]),
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )


def collect_logits(model, loader):
    model.eval()
    ys, logits, pids, diags = [], [], [], []
    with torch.inference_mode():
        for x, y, pid, diag in loader:
            x = x.to(DEVICE, non_blocking=True)
            z, _ = model(x)
            ys.append(y.numpy())
            logits.append(z.float().cpu().numpy())
            pids.extend(list(pid))
            diags.extend(list(diag))
    return np.concatenate(ys), np.concatenate(logits), pids, diags


def ece_score(probs, y, bins=10):
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(float)
    ece = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        hi = lo + 1.0 / bins
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)


def full_metrics(y, logits, temperature=1.0):
    z = logits / float(temperature)
    probs = torch.softmax(torch.tensor(z), dim=1).numpy()
    pred = probs.argmax(1)

    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    rows = {}
    for i, cls in enumerate(CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        rows[cls] = {
            "precision": float(tp / (tp + fp)) if tp + fp else np.nan,
            "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
            "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
            "AUROC": float(roc_auc_score((y == i).astype(int), probs[:, i])),
        }

    onehot = np.eye(len(CLASSES))[y]
    return {
        "macro_F1": float(f1_score(y, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_AUROC": float(roc_auc_score(onehot, probs, multi_class="ovr", average="macro")),
        "NLL": float(log_loss(y, probs, labels=range(len(CLASSES)))),
        "Brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        "ECE": ece_score(probs, y),
        "per_class": rows,
        "confusion_matrix": cm.tolist(),
    }, probs, pred


def fit_temperature(logits, y):
    t_raw = torch.tensor([0.0], device=DEVICE, requires_grad=True)
    z = torch.tensor(logits, dtype=torch.float32, device=DEVICE)
    target = torch.tensor(y, dtype=torch.long, device=DEVICE)
    opt = torch.optim.LBFGS([t_raw], lr=0.1, max_iter=80)

    def closure():
        opt.zero_grad()
        t = F.softplus(t_raw) + 1e-3
        loss = F.cross_entropy(z / t, target)
        loss.backward()
        return loss

    opt.step(closure)
    return float((F.softplus(t_raw) + 1e-3).detach().cpu())


def bootstrap_ci(y, logits, temperature, n=2000):
    rng = np.random.default_rng(SEED)
    f1s, bals = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        # Require all classes in bootstrap sample.
        if len(np.unique(y[idx])) < len(CLASSES):
            continue
        m, _, _ = full_metrics(y[idx], logits[idx], temperature)
        f1s.append(m["macro_F1"])
        bals.append(m["balanced_accuracy"])
    return {
        "macro_F1_95CI": [float(np.quantile(f1s, .025)), float(np.quantile(f1s, .975))],
        "balanced_accuracy_95CI": [float(np.quantile(bals, .025)), float(np.quantile(bals, .975))],
        "valid_bootstrap_replicates": len(f1s),
    }


def train_classifier(manifest):
    train_df = manifest[manifest.split == "train"].copy()
    val_df = manifest[manifest.split == "validation"].copy()
    test_df = manifest[manifest.split == "test"].copy()

    train_loader = make_loader(train_df, True)
    val_loader = make_loader(val_df, False)
    test_loader = make_loader(test_df, False)

    model = GatedAttentionClassifier(
        d=int(CFG["encoder"]["feature_dim"]),
        a=int(CFG["aggregator"]["attention_dim"]),
        n_classes=len(CLASSES),
        dropout=float(CFG["aggregator"]["dropout"]),
    ).to(DEVICE)

    counts = train_df["diagnosis"].value_counts()
    weights = np.array([len(train_df) / (len(CLASSES) * counts[c]) for c in CLASSES], dtype=np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=DEVICE))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(CFG["training"]["learning_rate"]),
        weight_decay=float(CFG["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(CFG["training"]["scheduler_factor"]),
        patience=int(CFG["training"]["scheduler_patience"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(CFG["training"]["mixed_precision"]))

    policy_cfg = json.loads((DX / "config" / "recovery_policy.json").read_text())
    allowed = set(RecoveryPolicy.__dataclass_fields__)
    policy = RecoveryPolicy(**{k: v for k, v in policy_cfg.items() if k in allowed})
    recovery = RunRecoveryManager(ROOT, RUN_ID, policy=policy, run_dir=RUN)

    config_hash = sha256_file(CFG_PATH)
    split_hash = sha256_file(SPLIT_PATH)
    feature_manifest_path = TABLES / "step04_diagnosis_feature_manifest.csv"
    manifest_hash = sha256_file(feature_manifest_path)

    start_epoch = 1
    best_f1 = -1.0
    patience = 0
    log_rows = []

    if recovery.checkpoint_path().exists():
        restored = recovery.restore(
            student=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_config_hash=config_hash,
            expected_split_hash=split_hash,
            expected_manifest_hash=manifest_hash,
        )
        start_epoch = int(restored["global_step"]) + 1
        best_f1 = float(restored.get("best_score") or -1.0)
        patience = int(restored.get("patience_counter") or 0)
        print(f"✓ Resuming classifier from epoch {start_epoch}")

    best_path = RUN / "best_classifier.pt"
    max_epochs = int(CFG["training"]["max_epochs"])
    patience_limit = int(CFG["training"]["early_stopping_patience"])

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        losses = []
        for x, y, _, _ in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                logits, _ = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        vy, vlogits, _, _ = collect_logits(model, val_loader)
        val_metrics, _, _ = full_metrics(vy, vlogits, temperature=1.0)
        val_f1 = val_metrics["macro_F1"]
        scheduler.step(val_f1)

        improved = val_f1 > best_f1 + 1e-6
        if improved:
            best_f1 = val_f1
            patience = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_macro_F1": best_f1,
                "config_hash": config_hash,
                "split_hash": split_hash,
                "manifest_hash": manifest_hash,
            }, best_path)
        else:
            patience += 1

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_macro_F1": val_f1,
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
            "best_val_macro_F1": best_f1,
            "patience": patience,
        }
        log_rows.append(row)
        atomic_csv(RUN / "train_log.csv", pd.DataFrame(log_rows))
        print(
            f"EPOCH {epoch:02d}  loss={row['train_loss']:.4f}  "
            f"valF1={val_f1:.4f}  best={best_f1:.4f}  patience={patience}/{patience_limit}"
        )

        remote = (epoch % 10 == 0) or improved
        recovery.save_training_checkpoint(
            student=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            global_step=epoch,
            epoch=epoch,
            best_score=best_f1,
            patience_counter=patience,
            config_hash=config_hash,
            split_hash=split_hash,
            manifest_hash=manifest_hash,
            remote=remote,
            status="training",
        )
        if epoch % 5 == 0 or improved:
            recovery.sync_metadata(f"EViCT-Dx Step04 epoch {epoch} valF1 {val_f1:.4f}")

        if patience >= patience_limit:
            print("✓ Early stopping.")
            break

    if not best_path.exists():
        raise RuntimeError("No best classifier checkpoint produced.")

    best = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best["model"])

    # Calibration uses validation only.
    vy, vlogits, vpids, vdiag = collect_logits(model, val_loader)
    temperature = fit_temperature(vlogits, vy)
    val_uncal, _, _ = full_metrics(vy, vlogits, 1.0)
    val_cal, vprobs, vpred = full_metrics(vy, vlogits, temperature)

    atomic_json(RUN / "temperature.json", {
        "temperature": temperature,
        "fit_split": "validation_only",
        "best_epoch": int(best["epoch"]),
        "best_val_macro_F1_uncalibrated": float(best["val_macro_F1"]),
        "validation_metrics_uncalibrated": val_uncal,
        "validation_metrics_calibrated": val_cal,
    })

    val_rows = []
    for i in range(len(vy)):
        r = {"patient_id": vpids[i], "diagnosis": vdiag[i], "target": int(vy[i]), "prediction": int(vpred[i])}
        for j, cls in enumerate(CLASSES):
            r[f"prob_{cls}"] = float(vprobs[i, j])
        val_rows.append(r)
    atomic_csv(TABLES / "step04_diagnosis_validation_predictions.csv", pd.DataFrame(val_rows))

    # Held-out test is touched once, after checkpoint selection and calibration are frozen.
    ty, tlogits, tpids, tdiag = collect_logits(model, test_loader)
    test_metrics, tprobs, tpred = full_metrics(ty, tlogits, temperature)
    ci = bootstrap_ci(ty, tlogits, temperature, n=int(CFG["evaluation"]["bootstrap_replicates"]))

    test_rows = []
    for i in range(len(ty)):
        r = {"patient_id": tpids[i], "diagnosis": tdiag[i], "target": int(ty[i]), "prediction": int(tpred[i])}
        for j, cls in enumerate(CLASSES):
            r[f"prob_{cls}"] = float(tprobs[i, j])
        test_rows.append(r)
    atomic_csv(TABLES / "step04_diagnosis_test_predictions.csv", pd.DataFrame(test_rows))

    # Durable copy of the selected best classifier. It is intentionally
    # stored as a GitHub Release asset rather than ordinary Git.
    try:
        token = kaggle_secret(policy.kaggle_secret_name)
        store = GitHubReleaseStore(
            repository=policy.github_repository,
            token=token,
            release_prefix=policy.rolling_release_prefix,
        )
        store.upload_or_replace(RUN_ID, best_path, asset_name="best_classifier.pt")
        print("✓ best_classifier.pt uploaded to rolling recovery release")
    except Exception as exc:
        print("⚠ Could not upload best classifier release asset:", exc)
        print("  Local best checkpoint remains available in the current Kaggle session.")

    result = {
        "project": "EViCT-Dx",
        "stage": "STEP_04_DIAGNOSIS_BASELINE",
        "run_id": RUN_ID,
        "completed_utc": now(),
        "encoder": "EfficientNet-B0 ImageNet1K V1 frozen",
        "aggregator": "gated attention pooling",
        "slices_per_volume": N_SLICES,
        "best_epoch": int(best["epoch"]),
        "temperature": temperature,
        "held_out_test_n": int(len(ty)),
        "test_metrics": test_metrics,
        "bootstrap_95CI": ci,
        "report_fields_unlocked": False,
        "note": "Baseline experiment only. Diagnostic report fields remain locked until the frozen unlock criteria are assessed explicitly.",
    }
    atomic_json(AUDIT / "step04_diagnosis_baseline_results.json", result)

    recovery.write_state(
        status="COMPLETE",
        global_step=int(best["epoch"]),
        best_score=float(best["val_macro_F1"]),
        config_hash=config_hash,
        split_hash=split_hash,
        manifest_hash=manifest_hash,
        next_action="STEP_05_DIAGNOSIS_CALIBRATION_ABSTENTION_AND_STRONGER_MODEL",
    )
    recovery.sync_metadata("Complete EViCT-Dx Step04 diagnosis baseline")

    print("\n" + "=" * 108)
    print("✅ STEP 04 COMPLETE")
    print(f"Best validation Macro-F1 : {best['val_macro_F1']:.4f}")
    print(f"Temperature              : {temperature:.4f}")
    print(f"TEST Macro-F1            : {test_metrics['macro_F1']:.4f}")
    print(f"TEST Balanced Accuracy   : {test_metrics['balanced_accuracy']:.4f}")
    print(f"TEST Macro-AUROC         : {test_metrics['macro_AUROC']:.4f}")
    print(f"TEST ECE                 : {test_metrics['ECE']:.4f}")
    print(f"Macro-F1 95% CI          : {ci['macro_F1_95CI']}")
    print("Diagnostic report fields : STILL LOCKED")
    print("NEXT: Step 05 — calibration/abstention audit + stronger diagnosis model")
    print("=" * 108)


def main():
    print("=" * 108)
    print("EViCT-Dx STEP 04 — COVID-CT-MD VOLUME-LEVEL DIAGNOSIS BASELINE")
    print("=" * 108)
    print("GPU:", torch.cuda.get_device_name(0))

    verify = DX / "scripts" / "verify_core_lock.py"
    if verify.exists():
        subprocess.run([sys.executable, str(verify)], cwd=str(ROOT), check=True)

    if not INV_PATH.exists() or not SPLIT_PATH.exists():
        raise FileNotFoundError(
            "Step03D outputs are missing. Expected:\n"
            f"  {INV_PATH}\n  {SPLIT_PATH}"
        )

    inventory = pd.read_csv(INV_PATH)
    split_df = pd.read_csv(SPLIT_PATH)

    if len(split_df) != 305:
        raise RuntimeError(f"Unexpected split manifest size: {len(split_df)}")
    if split_df.duplicated(["diagnosis", "patient_id"]).any():
        raise RuntimeError("Duplicate patient split assignments detected.")

    manifest = build_feature_cache(inventory, split_df)
    sync_git("Freeze EViCT-Dx Step04 patient feature manifest")
    train_classifier(manifest)


if __name__ == "__main__":
    main()
