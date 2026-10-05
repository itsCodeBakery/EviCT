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

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.transforms.functional import resize
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
DX = ROOT / "vlmDiagnosis"
RUN_ID = "DX_step06_convnext_transformer_mil_seed1705_v1"
RUN = DX / "runs" / RUN_ID
CACHE = RUN / "feature_cache_strong"
TABLES = DX / "tables"
AUDIT = DX / "artifacts" / "audit"
CFG_PATH = DX / "config" / "diagnosis_strong_step06.json"
INV_PATH = TABLES / "step03d_covid_ct_md_dicom_inventory.csv"
SPLIT_PATH = TABLES / "step03d_covid_ct_md_split_manifest.csv"
BASELINE_VAL_PATH = TABLES / "step04_diagnosis_validation_predictions.csv"
SNAPSHOT_DIR = Path("/kaggle/working/evict_dx_step06_snapshots")

for d in [RUN, CACHE, TABLES, AUDIT, SNAPSHOT_DIR]:
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
    raise RuntimeError("Enable a Kaggle GPU for Step 06.")

CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
SEED = int(CFG["seed"])
CLASSES = list(CFG["classes"])
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
ADAPT_SLICES = int(CFG["input"]["weak_slice_adaptation_slices_per_train_patient"])
MIL_SLICES = int(CFG["input"]["mil_slices_per_patient"])

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


def sync_git(message):
    script = DX / "scripts" / "git_sync_dx.py"
    if not script.exists():
        print("⚠ git_sync_dx.py missing; local metadata retained.")
        return
    r = subprocess.run([sys.executable, str(script), message], cwd=str(ROOT), check=False)
    if r.returncode != 0:
        print(f"⚠ Git sync returned {r.returncode}; local metadata retained.")


def load_policy():
    cfg = json.loads((DX / "config" / "recovery_policy.json").read_text(encoding="utf-8"))
    allowed = set(RecoveryPolicy.__dataclass_fields__)
    return RecoveryPolicy(**{k: v for k, v in cfg.items() if k in allowed})


POLICY = load_policy()


def release_store():
    return GitHubReleaseStore(
        repository=POLICY.github_repository,
        token=kaggle_secret(POLICY.kaggle_secret_name),
        release_prefix=POLICY.rolling_release_prefix,
    )


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


def sample_members(df: pd.DataFrame, n: int):
    df = df.reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("Empty patient DICOM inventory.")
    ids = np.linspace(0, len(df) - 1, n).round().astype(int)
    return df.iloc[ids]["member"].tolist()


class WeakSliceDataset(Dataset):
    def __init__(self, archive: Path, records):
        self.archive = Path(archive)
        self.records = records
        self._zf = None

    def __len__(self):
        return len(self.records)

    def _zip(self):
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.archive, "r")
        return self._zf

    def __getitem__(self, idx):
        member, label = self.records[idx]
        with self._zip().open(member, "r") as f:
            x = dicom_bytes_to_tensor(f.read())
        return x, int(label)


def make_convnext():
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = convnext_tiny(weights=weights)

    last_linear = None
    last_index = None
    for i, layer in enumerate(model.classifier):
        if isinstance(layer, nn.Linear):
            last_linear = layer
            last_index = i
    if last_linear is None:
        raise RuntimeError("Unable to locate ConvNeXt classifier linear layer.")
    model.classifier[last_index] = nn.Linear(last_linear.in_features, len(CLASSES))

    for p in model.parameters():
        p.requires_grad = False
    for p in model.features[5:].parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
    return model


def convnext_features(model, x):
    x = model.features(x)
    x = model.avgpool(x)
    for layer in list(model.classifier.children())[:-1]:
        x = layer(x)
    return x


def build_weak_records(inventory, split_df):
    train_ids = split_df.loc[split_df["split"] == "train", ["diagnosis", "patient_id"]]
    allowed = set(map(tuple, train_ids.to_records(index=False)))
    records = []
    for key, df in inventory.groupby(["diagnosis", "patient_id"], sort=True):
        if key not in allowed:
            continue
        diagnosis, _pid = key
        for member in sample_members(df, ADAPT_SLICES):
            records.append((member, CLASS_TO_IDX[diagnosis]))
    expected = len(train_ids) * ADAPT_SLICES
    if len(records) != expected:
        raise RuntimeError(f"Weak adaptation record mismatch: {len(records)} vs {expected}")
    return records


def train_encoder(inventory, split_df):
    encoder_path = RUN / "encoder_adapted.pt"
    recovery = RunRecoveryManager(
        ROOT,
        RUN_ID + "_encoder",
        policy=POLICY,
        run_dir=RUN / "encoder_recovery",
    )

    model = make_convnext().to(DEVICE)
    train_counts = split_df.loc[split_df.split == "train", "diagnosis"].value_counts()
    class_weights = np.array(
        [len(split_df[split_df.split == "train"]) / (len(CLASSES) * train_counts[c]) for c in CLASSES],
        dtype=np.float32,
    )
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=DEVICE))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(CFG["encoder"]["learning_rate"]),
        weight_decay=float(CFG["encoder"]["weight_decay"]),
    )
    epochs = int(CFG["encoder"]["adaptation_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    config_hash = sha256_file(CFG_PATH)
    split_hash = sha256_file(SPLIT_PATH)
    manifest_hash = sha256_file(INV_PATH)

    start_epoch = 1
    if recovery.state_path.exists():
        try:
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
            print(f"✓ Restored encoder adaptation at epoch {start_epoch}")
        except FileNotFoundError:
            print("ℹ Encoder recovery metadata exists but local checkpoint is absent; attempting remote pull.")
            subprocess.run(
                [
                    sys.executable,
                    str(DX / "scripts" / "kaggle_recovery.py"),
                    "pull-remote",
                    "--run-id",
                    RUN_ID + "_encoder",
                ],
                cwd=str(ROOT),
                check=False,
            )
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

    if start_epoch > epochs and encoder_path.exists():
        print("✓ Encoder adaptation already complete.")
        saved = torch.load(encoder_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(saved["model"])
        return model

    records = build_weak_records(inventory, split_df)
    dataset = WeakSliceDataset(locate_archive(), records)
    history_path = RUN / "encoder_train_log.csv"
    history = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []

    for epoch in range(start_epoch, epochs + 1):
        generator = torch.Generator()
        generator.manual_seed(SEED + epoch)
        loader = DataLoader(
            dataset,
            batch_size=int(CFG["encoder"]["batch_size"]),
            shuffle=True,
            generator=generator,
            num_workers=2,
            pin_memory=True,
            persistent_workers=False,
        )

        model.train()
        losses = []
        correct = 0
        seen = 0
        bar = tqdm(loader, desc=f"Encoder adaptation {epoch}/{epochs}", leave=True)
        for x, y in bar:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.detach().cpu()))
            pred = logits.detach().argmax(1)
            correct += int((pred == y).sum())
            seen += int(y.numel())
            bar.set_postfix(loss=f"{np.mean(losses):.4f}", acc=f"{correct/max(seen,1):.3f}")

        scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "weak_slice_accuracy": float(correct / max(seen, 1)),
            "lr": float(optimizer.param_groups[0]["lr"]),
        })
        atomic_csv(history_path, pd.DataFrame(history))

        recovery.save_training_checkpoint(
            student=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            global_step=epoch,
            epoch=epoch,
            config_hash=config_hash,
            split_hash=split_hash,
            manifest_hash=manifest_hash,
            remote=True,
            status="encoder_adaptation",
        )
        recovery.sync_metadata(f"EViCT-Dx Step06 encoder epoch {epoch}/{epochs}")

    torch.save({
        "model": model.state_dict(),
        "completed_epoch": epochs,
        "config_hash": config_hash,
        "split_hash": split_hash,
        "manifest_hash": manifest_hash,
    }, encoder_path)

    try:
        release_store().upload_or_replace(RUN_ID, encoder_path, asset_name="encoder_adapted.pt")
        print("✓ Adapted ConvNeXt encoder uploaded as durable release asset.")
    except Exception as exc:
        print("⚠ Encoder release upload failed:", exc)

    return model


def latest_feature_snapshot(store):
    assets = store.list_assets(RUN_ID)
    names = sorted([n for n in assets if n.startswith("feature_cache_") and n.endswith(".tar")])
    return names[-1] if names else None


def restore_feature_snapshot_if_needed():
    state_path = RUN / "feature_state.json"
    if state_path.exists() and any(CACHE.glob("*.npz")):
        return
    try:
        store = release_store()
        name = latest_feature_snapshot(store)
        if name is None:
            return
        local = SNAPSHOT_DIR / name
        store.download(RUN_ID, name, local)
        with tarfile.open(local, "r") as tf:
            tf.extractall(RUN)
        print(f"✓ Restored durable feature snapshot: {name}")
    except Exception as exc:
        print("ℹ No remote Step06 feature snapshot restored:", exc)


def save_feature_snapshot(completed_count):
    state_path = RUN / "feature_state.json"
    if not state_path.exists():
        return
    name = f"feature_cache_{int(completed_count):03d}.tar"
    local = SNAPSHOT_DIR / name
    tmp = local.with_suffix(".tar.tmp")
    with tarfile.open(tmp, "w") as tf:
        tf.add(state_path, arcname="feature_state.json")
        for p in sorted(CACHE.glob("*.npz")):
            tf.add(p, arcname=f"feature_cache_strong/{p.name}")
    os.replace(tmp, local)
    store = release_store()
    store.upload_or_replace(RUN_ID, local, asset_name=name)
    store.prune_assets(RUN_ID, prefix="feature_cache_", keep=2)
    print(f"✓ Durable feature snapshot: {name} ({local.stat().st_size/(1024**2):.1f} MB)")


@torch.inference_mode()
def extract_patient_features(zf, members, model):
    model.eval()
    outputs = []
    chunk = 16
    for i in range(0, len(members), chunk):
        xs = []
        for member in members[i:i + chunk]:
            with zf.open(member, "r") as f:
                xs.append(dicom_bytes_to_tensor(f.read()))
        x = torch.stack(xs).to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            feat = convnext_features(model, x)
        outputs.append(feat.float().cpu().numpy())
    return np.concatenate(outputs, 0).astype(np.float16)


def build_feature_cache(inventory, split_df, model):
    restore_feature_snapshot_if_needed()
    state_path = RUN / "feature_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    completed = set(state.get("completed", []))

    dev_split = split_df[split_df["split"].isin(["train", "validation"])].copy()
    if (dev_split["split"] == "test").any():
        raise RuntimeError("TEST LEAKAGE GUARD: test patient entered Step06 dev split.")
    if len(dev_split) != 244:
        raise RuntimeError(f"Expected 244 train+validation patients, found {len(dev_split)}.")

    allowed = set(map(tuple, dev_split[["diagnosis", "patient_id"]].to_records(index=False)))
    groups = [(key, df) for key, df in inventory.groupby(["diagnosis", "patient_id"], sort=True) if key in allowed]
    if len(groups) != 244:
        raise RuntimeError(f"Expected 244 inventory groups, found {len(groups)}.")

    archive = locate_archive()
    newly_completed = 0
    with zipfile.ZipFile(archive, "r") as zf:
        bar = tqdm(groups, desc="Step06 ConvNeXt feature extraction", total=len(groups))
        for (diagnosis, pid), df in bar:
            key = f"{diagnosis}|{pid}"
            out = CACHE / f"{diagnosis.replace('-', '').replace(' ', '_')}__{pid}.npz"
            if key in completed and out.exists():
                bar.set_postfix(done=len(completed), status="resume-skip")
                continue

            members = sample_members(df, MIL_SLICES)
            features = extract_patient_features(zf, members, model)
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
            newly_completed += 1

            atomic_json(state_path, {
                "run_id": RUN_ID,
                "stage": "STEP_06_STRONG_FEATURE_CACHE",
                "completed": sorted(completed),
                "completed_count": len(completed),
                "total": 244,
                "updated_utc": now(),
            })
            bar.set_postfix(done=len(completed), patient=pid)

            if newly_completed > 0 and len(completed) % int(CFG["feature_cache"]["remote_snapshot_every_patients"]) == 0:
                save_feature_snapshot(len(completed))
                sync_git(f"EViCT-Dx Step06 strong feature cache {len(completed)}/244")

    if len(completed) != 244:
        raise RuntimeError(f"Step06 feature cache incomplete: {len(completed)}/244")

    save_feature_snapshot(244)

    rows = []
    for _, row in dev_split.iterrows():
        diagnosis = row["diagnosis"]
        pid = row["patient_id"]
        p = CACHE / f"{diagnosis.replace('-', '').replace(' ', '_')}__{pid}.npz"
        if not p.exists():
            raise FileNotFoundError(p)
        rows.append({
            "diagnosis": diagnosis,
            "patient_id": pid,
            "split": row["split"],
            "feature_path": str(p.relative_to(ROOT)),
            "feature_sha256": sha256_file(p),
        })
    manifest = pd.DataFrame(rows)
    if set(manifest["split"]) != {"train", "validation"}:
        raise RuntimeError("TEST LEAKAGE GUARD: Step06 feature manifest has unexpected split.")
    atomic_csv(TABLES / "step06_strong_feature_manifest.csv", manifest)
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
        return x, CLASS_TO_IDX[row["diagnosis"]], row["patient_id"], row["diagnosis"]


class SliceTransformerMIL(nn.Module):
    def __init__(self):
        super().__init__()
        d_in = int(CFG["feature_cache"]["feature_dim"])
        d = int(CFG["aggregator"]["projection_dim"])
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU())
        self.pos = nn.Parameter(torch.zeros(1, MIL_SLICES, d))
        nn.init.trunc_normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=int(CFG["aggregator"]["attention_heads"]),
            dim_feedforward=int(CFG["aggregator"]["feedforward_dim"]),
            dropout=float(CFG["aggregator"]["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(CFG["aggregator"]["transformer_layers"]))
        self.attn = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.Tanh(),
            nn.Linear(d // 2, 1),
        )
        pooled = d * 3
        hidden = int(CFG["aggregator"]["classifier_hidden"])
        self.head = nn.Sequential(
            nn.LayerNorm(pooled),
            nn.Linear(pooled, hidden),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(hidden, len(CLASSES)),
        )

    def forward(self, x):
        h = self.proj(x) + self.pos[:, :x.shape[1]]
        h = self.transformer(h)
        a = torch.softmax(self.attn(h).squeeze(-1), dim=1)
        attn_pool = torch.sum(h * a.unsqueeze(-1), dim=1)
        mean_pool = h.mean(dim=1)
        max_pool = h.max(dim=1).values
        return self.head(torch.cat([attn_pool, mean_pool, max_pool], dim=1)), a


def make_loader(df, shuffle, epoch_seed=None):
    g = None
    if shuffle:
        g = torch.Generator()
        g.manual_seed(int(epoch_seed if epoch_seed is not None else SEED))
    return DataLoader(
        FeatureDataset(df),
        batch_size=int(CFG["mil_training"]["batch_size"]),
        shuffle=shuffle,
        generator=g,
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
    )


@torch.inference_mode()
def collect_logits(model, loader):
    model.eval()
    ys, zs, pids, diags = [], [], [], []
    for x, y, pid, diag in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            z, _ = model(x)
        ys.append(y.numpy())
        zs.append(z.float().cpu().numpy())
        pids.extend(list(pid))
        diags.extend(list(diag))
    return np.concatenate(ys), np.concatenate(zs), pids, diags


def probabilities(logits, temperature=1.0):
    return torch.softmax(torch.tensor(logits / float(temperature), dtype=torch.float32), dim=1).numpy()


def metric_bundle(y, probs):
    pred = probs.argmax(1)
    onehot = np.eye(len(CLASSES))[y]
    return {
        "macro_F1": float(f1_score(y, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_AUROC": float(roc_auc_score(onehot, probs, multi_class="ovr", average="macro")),
    }


def fit_temperature(logits, y):
    raw = torch.tensor([0.0], device=DEVICE, requires_grad=True)
    z = torch.tensor(logits, dtype=torch.float32, device=DEVICE)
    target = torch.tensor(y, dtype=torch.long, device=DEVICE)
    opt = torch.optim.LBFGS([raw], lr=0.1, max_iter=80)

    def closure():
        opt.zero_grad()
        t = F.softplus(raw) + 1e-3
        loss = F.cross_entropy(z / t, target)
        loss.backward()
        return loss

    opt.step(closure)
    return float((F.softplus(raw) + 1e-3).detach().cpu())


def train_mil(manifest):
    train_df = manifest[manifest.split == "train"].copy()
    val_df = manifest[manifest.split == "validation"].copy()
    if len(train_df) != 183 or len(val_df) != 61:
        raise RuntimeError(f"Unexpected Step06 train/validation counts: {len(train_df)}/{len(val_df)}")

    model = SliceTransformerMIL().to(DEVICE)
    counts = train_df["diagnosis"].value_counts()
    weights = np.array([len(train_df) / (len(CLASSES) * counts[c]) for c in CLASSES], dtype=np.float32)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, device=DEVICE),
        label_smoothing=0.05,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(CFG["mil_training"]["learning_rate"]),
        weight_decay=float(CFG["mil_training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(CFG["mil_training"]["scheduler_factor"]),
        patience=int(CFG["mil_training"]["scheduler_patience"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    manifest_path = TABLES / "step06_strong_feature_manifest.csv"
    config_hash = sha256_file(CFG_PATH)
    split_hash = sha256_file(SPLIT_PATH)
    manifest_hash = sha256_file(manifest_path)

    recovery = RunRecoveryManager(
        ROOT,
        RUN_ID + "_mil",
        policy=POLICY,
        run_dir=RUN / "mil_recovery",
    )

    best_path = RUN / "best_mil.pt"
    log_path = RUN / "mil_train_log.csv"
    history = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []
    start_epoch = 1
    best_f1 = -1.0
    patience = 0

    if recovery.state_path.exists():
        try:
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
            print(f"✓ Restored MIL training at epoch {start_epoch}")
        except Exception as exc:
            print("ℹ MIL local restore unavailable:", exc)

    max_epochs = int(CFG["mil_training"]["max_epochs"])
    patience_limit = int(CFG["mil_training"]["early_stopping_patience"])
    val_loader = make_loader(val_df, False)

    epoch_bar = tqdm(range(start_epoch, max_epochs + 1), desc="Step06 Transformer-MIL training")
    for epoch in epoch_bar:
        train_loader = make_loader(train_df, True, SEED + 1000 + epoch)
        model.train()
        losses = []
        for x, y, _, _ in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        vy, vlogits, _, _ = collect_logits(model, val_loader)
        vmetrics = metric_bundle(vy, probabilities(vlogits, 1.0))
        val_f1 = vmetrics["macro_F1"]
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
            "val_balanced_accuracy": vmetrics["balanced_accuracy"],
            "val_macro_AUROC": vmetrics["macro_AUROC"],
            "best_val_macro_F1": best_f1,
            "patience": patience,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        atomic_csv(log_path, pd.DataFrame(history))
        epoch_bar.set_postfix(
            loss=f"{row['train_loss']:.4f}",
            valF1=f"{val_f1:.3f}",
            best=f"{best_f1:.3f}",
            patience=f"{patience}/{patience_limit}",
        )

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
            remote=(improved or epoch % 5 == 0),
            status="mil_training",
        )
        if improved or epoch % 5 == 0:
            recovery.sync_metadata(f"EViCT-Dx Step06 MIL epoch {epoch} valF1 {val_f1:.4f}")

        if patience >= patience_limit:
            break

    if not best_path.exists():
        raise RuntimeError("Step06 did not produce best_mil.pt")

    best = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best["model"])
    vy, vlogits, vpids, vdiags = collect_logits(model, val_loader)
    temp = fit_temperature(vlogits, vy)
    strong_probs = probabilities(vlogits, temp)
    strong_metrics = metric_bundle(vy, strong_probs)

    rows = []
    for i in range(len(vy)):
        row = {
            "patient_id": vpids[i],
            "diagnosis": vdiags[i],
            "target": int(vy[i]),
            "prediction": int(strong_probs[i].argmax()),
        }
        for j, cls in enumerate(CLASSES):
            row[f"prob_{cls}"] = float(strong_probs[i, j])
        rows.append(row)
    strong_df = pd.DataFrame(rows)
    atomic_csv(TABLES / "step06_strong_validation_predictions.csv", strong_df)

    baseline = pd.read_csv(BASELINE_VAL_PATH)
    merged = strong_df.merge(
        baseline,
        on=["patient_id", "diagnosis", "target"],
        suffixes=("_strong", "_baseline"),
        validate="one_to_one",
    )
    if len(merged) != 61:
        raise RuntimeError(f"Baseline/strong validation alignment failed: {len(merged)}/61")

    y = merged["target"].to_numpy(int)
    strong = merged[[f"prob_{c}_strong" for c in CLASSES]].to_numpy(float)
    base = merged[[f"prob_{c}_baseline" for c in CLASSES]].to_numpy(float)

    ensemble_rows = []
    for w in CFG["validation"]["optional_ensemble"]["weights"]:
        w = float(w)
        p = w * strong + (1.0 - w) * base
        m = metric_bundle(y, p)
        ensemble_rows.append({"strong_weight": w, **m})
    ensemble_df = pd.DataFrame(ensemble_rows)
    ensemble_df = ensemble_df.sort_values(
        ["macro_F1", "macro_AUROC", "strong_weight"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    atomic_csv(TABLES / "step06_validation_ensemble_sweep.csv", ensemble_df)
    selected = ensemble_df.iloc[0].to_dict()

    baseline_val_f1 = float(CFG["validation"]["baseline_reference"].split()[-1])
    audit = {
        "project": "EViCT-Dx",
        "stage": "STEP_06_STRONG_DIAGNOSIS_DEVELOPMENT",
        "run_id": RUN_ID,
        "completed_utc": now(),
        "held_out_test_accessed": False,
        "held_out_test_inference_run": False,
        "development_patients": {"train": 183, "validation": 61, "test": 0},
        "encoder": "ConvNeXt-Tiny ImageNet weakly adapted on train-patient slice labels only",
        "mil_model": "ordered 2-layer slice Transformer with attention+mean+max pooling",
        "mil_slices_per_patient": MIL_SLICES,
        "best_epoch": int(best["epoch"]),
        "temperature_validation_only": temp,
        "strong_validation_metrics": strong_metrics,
        "step04_baseline_validation_macro_F1": baseline_val_f1,
        "validation_macro_F1_delta_vs_step04": float(strong_metrics["macro_F1"] - baseline_val_f1),
        "selected_validation_ensemble": selected,
        "report_fields_unlocked": False,
        "next_action": "FREEZE_STEP07_FINAL_EVALUATION_PROTOCOL_BEFORE_ANY_STRONG_MODEL_TEST_INFERENCE",
    }
    atomic_json(AUDIT / "step06_strong_diagnosis_development.json", audit)
    atomic_json(RUN / "STATE.json", {
        "run_id": RUN_ID,
        "status": "COMPLETE",
        "updated_utc": now(),
        "held_out_test_accessed": False,
        "best_validation_macro_F1": strong_metrics["macro_F1"],
        "selected_ensemble_strong_weight": selected["strong_weight"],
        "next_action": audit["next_action"],
    })

    try:
        release_store().upload_or_replace(RUN_ID, best_path, asset_name="best_mil.pt")
        print("✓ best_mil.pt uploaded as durable release asset.")
    except Exception as exc:
        print("⚠ best_mil.pt release upload failed:", exc)

    sync_git("Complete EViCT-Dx Step06 stronger diagnosis development without test access")

    print("\n" + "=" * 108)
    print("✅ STEP 06 COMPLETE — DEVELOPMENT ONLY")
    print(f"Best strong validation Macro-F1 : {strong_metrics['macro_F1']:.4f}")
    print(f"Validation Balanced Accuracy    : {strong_metrics['balanced_accuracy']:.4f}")
    print(f"Validation Macro-AUROC          : {strong_metrics['macro_AUROC']:.4f}")
    print(f"Δ Macro-F1 vs Step04 validation : {strong_metrics['macro_F1'] - baseline_val_f1:+.4f}")
    print(f"Selected ensemble strong weight : {selected['strong_weight']:.2f}")
    print(f"Selected ensemble validation F1 : {selected['macro_F1']:.4f}")
    print("Held-out 61-patient test        : NOT ACCESSED")
    print("Diagnostic report fields        : STILL LOCKED")
    print("NEXT: freeze Step07 before any strong-model held-out test inference.")
    print("=" * 108)


def main():
    print("=" * 108)
    print("EViCT-Dx STEP 06 — STRONGER DIAGNOSIS DEVELOPMENT")
    print("ConvNeXt weak adaptation + ordered slice Transformer MIL")
    print("=" * 108)
    print("GPU:", torch.cuda.get_device_name(0))

    verify = DX / "scripts" / "verify_core_lock.py"
    if verify.exists():
        subprocess.run([sys.executable, str(verify)], cwd=str(ROOT), check=True)

    for p in [INV_PATH, SPLIT_PATH, BASELINE_VAL_PATH, CFG_PATH]:
        if not p.exists():
            raise FileNotFoundError(p)

    inventory = pd.read_csv(INV_PATH)
    split_df = pd.read_csv(SPLIT_PATH)

    if len(split_df) != 305:
        raise RuntimeError(f"Expected 305 frozen split rows, found {len(split_df)}")
    if split_df.duplicated(["diagnosis", "patient_id"]).any():
        raise RuntimeError("Duplicate patient split assignments.")
    if (split_df["split"] == "test").sum() != 61:
        raise RuntimeError("Frozen held-out test count is not 61.")

    completed = AUDIT / "step06_strong_diagnosis_development.json"
    if completed.exists():
        print("✓ Step06 already complete. Refusing accidental rerun.")
        print(completed)
        return

    model = train_encoder(inventory, split_df)
    manifest = build_feature_cache(inventory, split_df, model)
    sync_git("Freeze EViCT-Dx Step06 train-validation feature manifest")
    train_mil(manifest)


if __name__ == "__main__":
    main()
