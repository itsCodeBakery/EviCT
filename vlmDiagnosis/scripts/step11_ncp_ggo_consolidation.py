from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet34_Weights, resnet34
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
DX = ROOT / "vlmDiagnosis"
CFG_PATH = DX / "config" / "step11_ncp_ggo_consolidation.json"
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))

RUN_ID = CFG["run_id"]
RUN = DX / "runs" / RUN_ID
TABLES = DX / "tables"
AUDIT_DIR = DX / "artifacts" / "audit"
MANIFEST_DIR = DX / "artifacts" / "manifests"
PAIR_MANIFEST = ROOT / CFG["source_contract"]["pair_manifest"]
PATIENT_SPLIT = ROOT / CFG["source_contract"]["patient_split"]
STEP10D_STATE = DX / "runs" / "DX_step10d_ncp_exact_freeze_v1" / "STATE.json"
DATA_ROOT = Path(CFG["source_contract"]["data_root"])

for p in [RUN, TABLES, AUDIT_DIR, MANIFEST_DIR]:
    p.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DX))
from runtime.recovery import GitHubReleaseStore, RecoveryPolicy, kaggle_secret, sha256_file  # noqa: E402

SEED = int(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    raise RuntimeError("Enable a Kaggle GPU before Step11.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

CLASS_NAMES = ["background", "lung_field", "ground_glass_opacity", "consolidation"]
TARGET_CLASSES = {2: "GGO", 3: "consolidation"}
INPUT_SIZE = tuple(CFG["model"]["input_size"])


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


def upload_checkpoint(path: Path, asset_name: str):
    try:
        release_store().upload_or_replace(RUN_ID, path, asset_name=asset_name)
        print(f"✓ Remote recovery upload: {asset_name}")
        return True
    except Exception as exc:
        print(f"⚠ Remote recovery upload failed for {asset_name}: {exc}")
        return False


def try_restore_remote(path: Path, asset_name: str):
    if path.exists():
        return True
    try:
        release_store().download(RUN_ID, asset_name, path)
        print(f"✓ Restored {asset_name} from GitHub Release.")
        return True
    except Exception:
        return False


def verify_inputs():
    if not STEP10D_STATE.exists():
        raise RuntimeError("Step10D state is missing. Run the corrected Step10D first.")
    state = json.loads(STEP10D_STATE.read_text())
    if not bool(state.get("structure_verified")):
        raise RuntimeError("Step10D structure_verified is not true.")
    if not bool(state.get("label_mapping_verified")):
        raise RuntimeError("Step10D label_mapping_verified is not true.")
    if not bool(state.get("patient_split_leakage_free")):
        raise RuntimeError("Step10D patient split is not leakage-free.")
    if not bool(state.get("training_allowed")):
        raise RuntimeError("Step10D has not permitted training.")
    if not PAIR_MANIFEST.exists() or not PATIENT_SPLIT.exists():
        raise RuntimeError("Step10D manifests are missing.")
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Extracted NCP dataset is missing: {DATA_ROOT}. "
            "Stay in the same Kaggle session or restore the source dataset."
        )
    return state


def resolve_pair_paths(pair_df):
    rows = []
    for r in pair_df.itertuples(index=False):
        ip = DATA_ROOT / r.image_rel
        mp = DATA_ROOT / r.mask_rel
        if not ip.exists():
            raise FileNotFoundError(ip)
        if not mp.exists():
            raise FileNotFoundError(mp)
        rows.append({
            "patient_id": str(r.patient_id),
            "slice_id": str(r.slice_id),
            "image_path": str(ip),
            "mask_path": str(mp),
        })
    return pd.DataFrame(rows)


class NCPDataset(Dataset):
    def __init__(self, df: pd.DataFrame, training: bool):
        self.df = df.reset_index(drop=True)
        self.training = training
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(r.image_path).convert("RGB")
        mask = Image.open(r.mask_path)

        img = TF.resize(img, INPUT_SIZE, interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask, INPUT_SIZE, interpolation=InterpolationMode.NEAREST)

        if self.training:
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() < 0.25:
                angle = random.uniform(-7.0, 7.0)
                img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)

        x = TF.pil_to_tensor(img).float() / 255.0
        x = (x - self.mean) / self.std

        y = torch.from_numpy(np.array(mask, dtype=np.int64))
        if int(y.min()) < 0 or int(y.max()) > 3:
            raise RuntimeError(f"Unexpected mask value in {r.mask_path}: {int(y.min())}..{int(y.max())}")

        return x, y.long(), str(r.patient_id), str(r.slice_id)


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
        enc = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
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
        x = F.interpolate(x, size=INPUT_SIZE, mode="bilinear", align_corners=False)
        return self.head(x)


def compute_class_weights(train_df):
    counts = np.zeros(4, dtype=np.float64)
    print("Computing training-only pixel frequencies...")
    for p in tqdm(train_df.mask_path.tolist(), desc="Training mask frequency audit"):
        arr = np.array(Image.open(p), dtype=np.uint8)
        binc = np.bincount(arr.reshape(-1), minlength=4)[:4]
        counts += binc
    freq = counts / counts.sum()
    weights = 1.0 / np.sqrt(np.maximum(freq, 1e-12))
    weights /= weights.mean()
    return counts, freq, weights


def segmentation_loss(logits, y, class_weights):
    ce = F.cross_entropy(logits, y, weight=class_weights)
    p = torch.softmax(logits, dim=1)
    onehot = F.one_hot(y, num_classes=4).permute(0, 3, 1, 2).float()

    dices = []
    for ch in [1, 2, 3]:
        inter = (p[:, ch] * onehot[:, ch]).sum((0, 1, 2))
        den = p[:, ch].sum((0, 1, 2)) + onehot[:, ch].sum((0, 1, 2))
        dices.append((2 * inter + 1e-6) / (den + 1e-6))
    dice_loss = 1.0 - torch.stack(dices).mean()
    return 0.5 * ce + 0.5 * dice_loss


def dice_from_counts(tp, fp, fn):
    den = 2 * tp + fp + fn
    if den == 0:
        return 1.0
    return float((2 * tp) / den)


@torch.inference_mode()
def evaluate_segmentation(model, loader, collect_presence=False):
    model.eval()

    counts = {2: np.zeros(4, dtype=np.float64), 3: np.zeros(4, dtype=np.float64)}
    presence_rows = []

    for x, y, pids, sids in tqdm(loader, desc="Evaluation", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

        pred = probs.argmax(1)

        for ch in [2, 3]:
            pp = pred == ch
            yy = y == ch
            tp = int((pp & yy).sum())
            fp = int((pp & ~yy).sum())
            fn = int((~pp & yy).sum())
            tn = int((~pp & ~yy).sum())
            counts[ch] += np.array([tp, fp, fn, tn], dtype=np.float64)

        if collect_presence:
            # Fixed pre-specified score: mean of the top 0.5% pixel probabilities.
            k = max(1, int(round(0.005 * probs.shape[-1] * probs.shape[-2])))
            for i in range(len(pids)):
                row = {
                    "patient_id": str(pids[i]),
                    "slice_id": str(sids[i]),
                }
                for ch, short in TARGET_CLASSES.items():
                    score = float(torch.topk(probs[i, ch].flatten(), k=k).values.mean().cpu())
                    true_presence = bool((y[i] == ch).any().item())
                    row[f"{short}_score"] = score
                    row[f"{short}_true"] = true_presence
                presence_rows.append(row)

    metrics = {}
    for ch, short in TARGET_CLASSES.items():
        tp, fp, fn, tn = counts[ch]
        metrics[short] = {
            "Dice": dice_from_counts(tp, fp, fn),
            "IoU": float(tp / (tp + fp + fn)) if tp + fp + fn else 1.0,
            "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
            "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
            "tp_pixels": int(tp),
            "fp_pixels": int(fp),
            "fn_pixels": int(fn),
            "tn_pixels": int(tn),
        }

    return metrics, pd.DataFrame(presence_rows)


def val_selection_score(model, loader):
    m, _ = evaluate_segmentation(model, loader, collect_presence=False)
    return float(np.mean([m["GGO"]["Dice"], m["consolidation"]["Dice"]]))


def threshold_metrics(y, score, threshold):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=float)
    pred = score >= threshold
    tp = int((pred & y).sum())
    tn = int((~pred & ~y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    sens = float(tp / (tp + fn)) if tp + fn else np.nan
    spec = float(tn / (tn + fp)) if tn + fp else np.nan
    prec = float(tp / (tp + fp)) if tp + fp else np.nan
    bal = float(np.nanmean([sens, spec]))
    return {
        "threshold": float(threshold),
        "balanced_accuracy": bal,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def select_presence_threshold(y, score):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=float)
    ordered = np.sort(np.unique(score))
    midpoints = (
        (ordered[:-1] + ordered[1:]) / 2.0
        if len(ordered) > 1 else np.array([], dtype=float)
    )
    candidates = np.unique(np.concatenate([
        [0.0],
        ordered,
        midpoints,
        [1.0],
    ]))
    candidates = np.unique(np.clip(candidates, 0, 1))

    rows = [threshold_metrics(y, score, t) for t in candidates]
    rows = sorted(
        rows,
        key=lambda r: (
            -r["balanced_accuracy"],
            -min(r["sensitivity"], r["specificity"]) if np.isfinite(r["sensitivity"]) and np.isfinite(r["specificity"]) else 1e9,
            -r["specificity"] if np.isfinite(r["specificity"]) else 1e9,
            r["threshold"],
        ),
    )
    return rows[0], pd.DataFrame(rows)


def evaluate_presence_with_fixed_threshold(df, short, threshold):
    y = df[f"{short}_true"].astype(bool).to_numpy()
    score = df[f"{short}_score"].astype(float).to_numpy()
    if len(np.unique(y)) < 2:
        auroc = np.nan
    else:
        auroc = float(roc_auc_score(y, score))
    out = threshold_metrics(y, score, threshold)
    out["AUROC"] = auroc
    out["n"] = int(len(y))
    out["positives"] = int(y.sum())
    out["negatives"] = int((~y).sum())
    return out


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(s):
    if not s:
        return
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state_all(s["cuda"])


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best, patience, class_weights):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "best": float(best),
        "patience": int(patience),
        "class_weights": class_weights.detach().cpu(),
        "rng": capture_rng(),
        "config_sha": sha256_file(CFG_PATH),
        "pair_manifest_sha": sha256_file(PAIR_MANIFEST),
        "patient_split_sha": sha256_file(PATIENT_SPLIT),
    }, path)


def restore_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    expected = {
        "config_sha": sha256_file(CFG_PATH),
        "pair_manifest_sha": sha256_file(PAIR_MANIFEST),
        "patient_split_sha": sha256_file(PATIENT_SPLIT),
    }
    for k, v in expected.items():
        if ck.get(k) != v:
            raise RuntimeError(f"Recovery hash mismatch for {k}; refusing resume.")

    model.load_state_dict(ck["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        restore_rng(ck.get("rng"))
    return ck


def main():
    print("=" * 112)
    print("EViCT-Dx STEP 11 — NCP GGO + CONSOLIDATION SEGMENTATION")
    print("PATIENT-DISJOINT 120/15/15 — VALIDATION-ONLY SELECTION — TEST ONCE")
    print("=" * 112)

    existing_state_path = RUN / "STATE.json"
    if existing_state_path.exists():
        existing_state = json.loads(existing_state_path.read_text())
        if existing_state.get("status") == "COMPLETE":
            print("✓ Step11 is already COMPLETE. Refusing to retrain or re-access the internal test.")
            print(json.dumps(existing_state, indent=2))
            return

    step10d = verify_inputs()
    print("✓ Step10D permits training.")
    print("✓ Exact official mapping: 0 BG, 1 lung field, 2 GGO, 3 consolidation.")

    pair_df = pd.read_csv(PAIR_MANIFEST, dtype={"patient_id": str, "slice_id": str})
    split_df = pd.read_csv(PATIENT_SPLIT, dtype={"patient_id": str})
    if len(pair_df) != 750 or pair_df.patient_id.nunique() != 150:
        raise RuntimeError("Pair manifest is not the frozen 750-slice / 150-group resource.")

    pairs = resolve_pair_paths(pair_df)
    pairs = pairs.merge(split_df[["patient_id", "split"]], on="patient_id", how="left")
    if pairs["split"].isna().any():
        raise RuntimeError("Split assignment missing for one or more annotated slices.")

    train_df = pairs[pairs.split == "train"].copy()
    val_df = pairs[pairs.split == "validation"].copy()
    test_df = pairs[pairs.split == "test"].copy()

    print(f"Annotated slices — train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(
        "Patient groups    — "
        f"train={train_df.patient_id.nunique()}, "
        f"val={val_df.patient_id.nunique()}, "
        f"test={test_df.patient_id.nunique()}"
    )

    if (len(train_df), len(val_df), len(test_df)) != (600, 75, 75):
        raise RuntimeError("Expected exactly 600/75/75 annotated slices.")
    if (train_df.patient_id.nunique(), val_df.patient_id.nunique(), test_df.patient_id.nunique()) != (120, 15, 15):
        raise RuntimeError("Expected exactly 120/15/15 patient groups.")

    counts, freq, weights_np = compute_class_weights(train_df)
    class_weights = torch.tensor(weights_np, dtype=torch.float32, device=DEVICE)
    print("\nTraining-only class frequencies / weights:")
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"  {i} {name:22s} pixels={int(counts[i]):12d} "
            f"freq={freq[i]:.6f} weight={weights_np[i]:.4f}"
        )

    train_ds = NCPDataset(train_df, training=True)
    val_ds = NCPDataset(val_df, training=False)
    test_ds = NCPDataset(test_df, training=False)

    batch_size = int(CFG["training"]["batch_size"])
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=True
    )

    model = UNetResNet34().to(DEVICE)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(CFG["training"]["learning_rate"]),
        weight_decay=float(CFG["training"]["weight_decay"]),
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max",
        factor=float(CFG["training"]["scheduler_factor"]),
        patience=int(CFG["training"]["scheduler_patience"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    last_path = RUN / "last.pt"
    best_path = RUN / "best.pt"
    train_log = RUN / "train_log.csv"

    try_restore_remote(last_path, "last.pt")
    start_epoch = 1
    best = -1.0
    patience = 0

    if last_path.exists():
        ck = restore_checkpoint(last_path, model, opt, sched, scaler)
        start_epoch = int(ck["epoch"]) + 1
        best = float(ck["best"])
        patience = int(ck["patience"])
        print(f"✓ Resuming Step11 at epoch {start_epoch}; best={best:.4f}; patience={patience}")

    history = pd.read_csv(train_log).to_dict("records") if train_log.exists() else []

    max_epochs = int(CFG["training"]["max_epochs"])
    patience_limit = int(CFG["training"]["early_stopping_patience"])

    state = {
        "run_id": RUN_ID,
        "status": "RUNNING",
        "updated_utc": now(),
        "current_epoch": start_epoch - 1,
        "best_validation_target_mean_dice": best,
        "test_accessed": False,
        "next_action": "CONTINUE_STEP11_TRAINING",
    }
    atomic_json(RUN / "STATE.json", state)
    sync_git("Start EViCT-Dx Step11 NCP subtype segmentation")

    for epoch in ([] if patience >= patience_limit else range(start_epoch, max_epochs + 1)):
        model.train()
        losses = []
        bar = tqdm(train_loader, desc=f"Step11 epoch {epoch}/{max_epochs}", leave=True)

        for x, y, _, _ in bar:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = segmentation_loss(logits, y, class_weights)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.detach().cpu()))
            bar.set_postfix(loss=f"{np.mean(losses):.4f}")

        val_score = val_selection_score(model, val_loader)
        sched.step(val_score)

        improved = val_score > best + 1e-5
        if improved:
            best = val_score
            patience = 0
            save_checkpoint(best_path, model, opt, sched, scaler, epoch, best, patience, class_weights)
            upload_checkpoint(best_path, "best.pt")
        else:
            patience += 1

        save_checkpoint(last_path, model, opt, sched, scaler, epoch, best, patience, class_weights)
        if epoch % int(CFG["recovery"]["remote_checkpoint_every_n_epochs"]) == 0:
            upload_checkpoint(last_path, "last.pt")

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_target_mean_dice": float(val_score),
            "best_validation_target_mean_dice": float(best),
            "patience": int(patience),
            "lr": float(opt.param_groups[0]["lr"]),
        })
        atomic_csv(train_log, pd.DataFrame(history))

        state.update({
            "updated_utc": now(),
            "current_epoch": epoch,
            "best_validation_target_mean_dice": best,
            "patience": patience,
        })
        atomic_json(RUN / "STATE.json", state)

        if epoch % 5 == 0 or improved:
            sync_git(f"EViCT-Dx Step11 epoch {epoch} metadata")

        print(
            f"Epoch {epoch}: train_loss={np.mean(losses):.4f} "
            f"val_target_mean_dice={val_score:.4f} best={best:.4f} patience={patience}"
        )

        if patience >= patience_limit:
            print("✓ Early stopping reached.")
            break

    if not best_path.exists():
        if not try_restore_remote(best_path, "best.pt"):
            raise RuntimeError("Best Step11 checkpoint is unavailable.")

    restore_checkpoint(best_path, model)

    # ------------------------------------------------------------------
    # VALIDATION-ONLY threshold selection
    # ------------------------------------------------------------------
    print("\nSelecting subtype-presence thresholds on validation ONLY...")
    val_seg, val_presence = evaluate_segmentation(model, val_loader, collect_presence=True)
    atomic_csv(TABLES / "step11_validation_presence_predictions.csv", val_presence)

    thresholds = {}
    threshold_sweeps = []
    for short in ["GGO", "consolidation"]:
        y = val_presence[f"{short}_true"].astype(bool).to_numpy()
        score = val_presence[f"{short}_score"].astype(float).to_numpy()
        selected, sweep = select_presence_threshold(y, score)
        thresholds[short] = float(selected["threshold"])
        sweep.insert(0, "class", short)
        threshold_sweeps.append(sweep)
        print(
            f"Validation {short}: threshold={selected['threshold']:.6f} "
            f"balacc={selected['balanced_accuracy']:.4f} "
            f"sens={selected['sensitivity']:.4f} spec={selected['specificity']:.4f}"
        )

    atomic_csv(
        TABLES / "step11_validation_presence_threshold_sweep.csv",
        pd.concat(threshold_sweeps, ignore_index=True)
    )
    atomic_json(
        RUN / "validation_selected_thresholds.json",
        {
            "selected_utc": now(),
            "selection_data": "validation_only",
            "presence_score": CFG["presence"]["score"],
            "thresholds": thresholds,
            "test_used": False,
        }
    )

    # ------------------------------------------------------------------
    # ONE-TIME INTERNAL TEST
    # ------------------------------------------------------------------
    state.update({
        "updated_utc": now(),
        "status": "TEST_EVALUATION_RUNNING",
        "test_accessed": True,
        "next_action": "COMPLETE_ONE_TIME_INTERNAL_TEST",
    })
    atomic_json(RUN / "STATE.json", state)
    sync_git("EViCT-Dx Step11 begin one-time internal test")

    print("\nRunning the frozen one-time internal test...")
    test_seg, test_presence = evaluate_segmentation(model, test_loader, collect_presence=True)
    atomic_csv(TABLES / "step11_internal_test_presence_predictions.csv", test_presence)

    test_presence_metrics = {}
    unlock = {}
    crit = CFG["unlock_criteria"]

    for short in ["GGO", "consolidation"]:
        pm = evaluate_presence_with_fixed_threshold(
            test_presence, short, thresholds[short]
        )
        test_presence_metrics[short] = pm

        seg_pass = test_seg[short]["Dice"] >= float(crit["segmentation_Dice_min"])
        auc_pass = np.isfinite(pm["AUROC"]) and pm["AUROC"] >= float(crit["internal_test_presence_AUROC_min"])
        sens_pass = np.isfinite(pm["sensitivity"]) and pm["sensitivity"] >= float(crit["internal_test_presence_sensitivity_min"])
        spec_pass = np.isfinite(pm["specificity"]) and pm["specificity"] >= float(crit["internal_test_presence_specificity_min"])

        unlock[short] = {
            "segmentation_Dice_pass": bool(seg_pass),
            "presence_AUROC_pass": bool(auc_pass),
            "presence_sensitivity_pass": bool(sens_pass),
            "presence_specificity_pass": bool(spec_pass),
            "internal_gate_pass": bool(seg_pass and auc_pass and sens_pass and spec_pass),
            "external_validation_still_required": True,
            "report_field_unlocked": False,
        }

    result = {
        "project": "EViCT-Dx",
        "stage": "STEP_11_NCP_GGO_CONSOLIDATION_SEGMENTATION",
        "run_id": RUN_ID,
        "completed_utc": now(),
        "dataset": {
            "annotated_slices": 750,
            "patient_or_scan_groups": 150,
            "split": {"train":120, "validation":15, "test":15},
            "archive_sha256": CFG["source_contract"]["archive_sha256"],
            "label_mapping": CFG["source_contract"]["label_mapping"],
        },
        "best_validation_target_mean_dice": float(best),
        "validation_segmentation": val_seg,
        "validation_selected_presence_thresholds": thresholds,
        "internal_test_segmentation": test_seg,
        "internal_test_presence": test_presence_metrics,
        "internal_unlock_gate": unlock,
        "important_scope": (
            "Passing the internal gate does not unlock report fields yet because the frozen Step02 protocol "
            "also specifies LongCIU as the external test for GGO and consolidation."
        ),
        "external_test_accessed": False,
        "next_action": "STEP_12_EXTERNAL_LONGCIU_GGO_CONSOLIDATION_EVALUATION",
    }
    atomic_json(AUDIT_DIR / "step11_ncp_ggo_consolidation_results.json", result)

    # Compact metrics table for manuscript traceability.
    rows = []
    for short in ["GGO", "consolidation"]:
        rows.append({
            "class": short,
            "segmentation_Dice": test_seg[short]["Dice"],
            "segmentation_IoU": test_seg[short]["IoU"],
            "segmentation_sensitivity": test_seg[short]["sensitivity"],
            "segmentation_specificity": test_seg[short]["specificity"],
            "presence_threshold_validation_selected": thresholds[short],
            "presence_AUROC": test_presence_metrics[short]["AUROC"],
            "presence_sensitivity": test_presence_metrics[short]["sensitivity"],
            "presence_specificity": test_presence_metrics[short]["specificity"],
            "presence_precision": test_presence_metrics[short]["precision"],
            "internal_gate_pass": unlock[short]["internal_gate_pass"],
            "report_field_unlocked": False,
        })
    atomic_csv(TABLES / "step11_internal_test_metrics.csv", pd.DataFrame(rows))

    state = {
        "run_id": RUN_ID,
        "status": "COMPLETE",
        "updated_utc": now(),
        "best_validation_target_mean_dice": float(best),
        "test_accessed": True,
        "external_test_accessed": False,
        "GGO_internal_gate_pass": bool(unlock["GGO"]["internal_gate_pass"]),
        "consolidation_internal_gate_pass": bool(unlock["consolidation"]["internal_gate_pass"]),
        "report_fields_unlocked": False,
        "next_action": "STEP_12_EXTERNAL_LONGCIU_GGO_CONSOLIDATION_EVALUATION",
    }
    atomic_json(RUN / "STATE.json", state)
    sync_git("Complete EViCT-Dx Step11 NCP GGO consolidation segmentation")

    print("\n" + "=" * 112)
    print("✅ STEP 11 COMPLETE — INTERNAL NCP SUBTYPE EVALUATION")
    print(f"Best validation target mean Dice : {best:.4f}")
    for short in ["GGO", "consolidation"]:
        s = test_seg[short]
        p = test_presence_metrics[short]
        print(
            f"{short:14s} Dice={s['Dice']:.4f} IoU={s['IoU']:.4f} "
            f"| presence AUROC={p['AUROC']:.4f} sens={p['sensitivity']:.4f} "
            f"spec={p['specificity']:.4f} precision={p['precision']:.4f} "
            f"| INTERNAL_GATE={'PASS' if unlock[short]['internal_gate_pass'] else 'FAIL'}"
        )
    print("Report fields                       : STILL LOCKED PENDING LONGCIU EXTERNAL TEST")
    print("NEXT                                : STEP_12_EXTERNAL_LONGCIU_GGO_CONSOLIDATION_EVALUATION")
    print("=" * 112)


if __name__ == "__main__":
    main()
