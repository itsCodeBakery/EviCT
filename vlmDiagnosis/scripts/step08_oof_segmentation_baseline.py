from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet34_Weights, resnet34
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
DX = ROOT / "vlmDiagnosis"
RUN_ID = "DX_step08_oof_segmentation_resnet34_unet_seed1705_v1"
RUN = DX / "runs" / RUN_ID
TABLES = DX / "tables"
AUDIT = DX / "artifacts" / "audit"
CFG_PATH = DX / "config" / "segmentation_baseline_step08.json"
CASE_MANIFEST = TABLES / "step03c_20case_case_manifest.csv"
SPLIT_MANIFEST = TABLES / "step03c_20case_split_manifest.csv"

for p in [RUN, TABLES, AUDIT]:
    p.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DX))
from runtime.recovery import GitHubReleaseStore, RecoveryPolicy, kaggle_secret, sha256_file  # noqa: E402

CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
SEED = int(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    raise RuntimeError("Enable a Kaggle GPU before Step08.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


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
    script = DX / "scripts" / "git_sync_dx.py"
    r = subprocess.run([sys.executable, str(script), message], cwd=str(ROOT), check=False)
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


def locate_dataset_root():
    candidates = [
        Path("/kaggle/input/datasets/ipythonx/covid19-ct"),
        Path("/kaggle/input/covid19-ct"),
    ]
    for c in candidates:
        if (c / "COVID-19-CT-Seg_20cases").exists():
            return c
    for base in [Path("/kaggle/input"), Path("/kaggle/working")]:
        if not base.exists():
            continue
        for hit in base.rglob("COVID-19-CT-Seg_20cases"):
            if hit.is_dir():
                return hit.parent
    raise FileNotFoundError(
        "COVID-19 CT 20-case dataset not found. Attach Kaggle dataset ipythonx/covid19-ct."
    )


DATA_ROOT = locate_dataset_root()


def abs_data(rel):
    p = DATA_ROOT / rel
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def source_image(arr, source):
    arr = arr.astype(np.float32)
    if source == "CoronaCases":
        chans = []
        for lo, hi in [(-1000, 400), (-600, 150), (-1200, 600)]:
            x = np.clip(arr, lo, hi)
            x = (x - lo) / (hi - lo)
            chans.append(x)
        return np.stack(chans, 0).astype(np.float32)
    lo, hi = np.percentile(arr[np.isfinite(arr)], [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr) + 1e-6)
    x = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
    return np.stack([x, x, x], 0).astype(np.float32)


class SliceDataset(Dataset):
    def __init__(self, cases_df, case_ids, training):
        self.cases = {r.case_id: r for r in cases_df.itertuples(index=False)}
        self.training = training
        self.items = []
        self._handles = {}
        for cid in case_ids:
            r = self.cases[cid]
            lung = np.asanyarray(nib.load(abs_data(r.lung_mask_path)).dataobj)
            inf = np.asanyarray(nib.load(abs_data(r.infection_mask_path)).dataobj)
            zdim = lung.shape[2]
            for z in range(zdim):
                pos = bool(np.any(lung[:, :, z] > 0) or np.any(inf[:, :, z] > 0))
                if (not training) or pos or (z % 4 == 0):
                    self.items.append((cid, z))

    def __len__(self):
        return len(self.items)

    def _case_handles(self, cid):
        if cid not in self._handles:
            r = self.cases[cid]
            self._handles[cid] = (
                nib.load(abs_data(r.ct_path)),
                nib.load(abs_data(r.lung_mask_path)),
                nib.load(abs_data(r.infection_mask_path)),
                r,
            )
        return self._handles[cid]

    def __getitem__(self, idx):
        cid, z = self.items[idx]
        ct, lung, inf, r = self._case_handles(cid)
        img = np.asanyarray(ct.dataobj[:, :, z], dtype=np.float32)
        lung_s = np.asanyarray(lung.dataobj[:, :, z])
        inf_s = np.asanyarray(inf.dataobj[:, :, z])

        x = torch.from_numpy(source_image(img, r.source))
        left = torch.from_numpy((lung_s == float(r.left_raw_label)).astype(np.float32))
        right = torch.from_numpy((lung_s == float(r.right_raw_label)).astype(np.float32))
        infection = torch.from_numpy((inf_s > 0).astype(np.float32))
        y = torch.stack([left, right, infection], 0)

        size = tuple(CFG["preprocessing"]["resize"])
        x = F.interpolate(x[None], size=size, mode="bilinear", align_corners=False)[0]
        y = F.interpolate(y[None], size=size, mode="nearest")[0]
        return x, y, cid


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
        self.e1, self.e2, self.e3, self.e4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
        self.d4 = ConvBlock(512 + 256, 256)
        self.d3 = ConvBlock(256 + 128, 128)
        self.d2 = ConvBlock(128 + 64, 64)
        self.d1 = ConvBlock(64 + 64, 64)
        self.head = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        s0 = self.stem(x)          # 1/2
        e1 = self.e1(self.pool(s0))# 1/4
        e2 = self.e2(e1)           # 1/8
        e3 = self.e3(e2)           # 1/16
        e4 = self.e4(e3)           # 1/32
        x = F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.d4(torch.cat([x, e3], 1))
        x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.d3(torch.cat([x, e2], 1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.d2(torch.cat([x, e1], 1))
        x = F.interpolate(x, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        x = self.d1(torch.cat([x, s0], 1))
        x = F.interpolate(x, size=x.shape[-2] * 2, mode="bilinear", align_corners=False)
        return self.head(x)


POS_W = torch.tensor([1.0, 1.0, 3.0], device=DEVICE).view(1, 3, 1, 1)


def loss_fn(logits, y):
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    weights = 1.0 + y * (POS_W - 1.0)
    bce = (bce * weights).mean()
    p = torch.sigmoid(logits)
    inter = (p * y).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + y.sum((0, 2, 3))
    dice_loss = 1.0 - ((2 * inter + 1e-6) / (den + 1e-6)).mean()
    return 0.5 * bce + 0.5 * dice_loss


def dice_from_counts(tp, fp, fn):
    return (2 * tp + 1e-8) / (2 * tp + fp + fn + 1e-8)


@torch.inference_mode()
def validation_score(model, loader):
    model.eval()
    counts = {}
    for x, y, cids in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(model(x)) >= 0.5
        yt = y >= 0.5
        for i, cid in enumerate(cids):
            if cid not in counts:
                counts[cid] = np.zeros((3, 3), dtype=np.float64) # tp fp fn
            for ch in range(3):
                pp, yy = p[i, ch], yt[i, ch]
                counts[cid][ch, 0] += int((pp & yy).sum())
                counts[cid][ch, 1] += int((pp & ~yy).sum())
                counts[cid][ch, 2] += int((~pp & yy).sum())
    vals = []
    for arr in counts.values():
        vals.append(np.mean([dice_from_counts(*arr[ch]) for ch in range(3)]))
    return float(np.mean(vals))


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


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best, patience):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best": best,
        "patience": patience,
        "rng": capture_rng(),
        "config_sha": sha256_file(CFG_PATH),
        "split_sha": sha256_file(SPLIT_MANIFEST),
    }, path)


def restore_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    if ck["config_sha"] != sha256_file(CFG_PATH) or ck["split_sha"] != sha256_file(SPLIT_MANIFEST):
        raise RuntimeError("Recovery hash mismatch; refusing resume.")
    model.load_state_dict(ck["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        restore_rng(ck.get("rng"))
    return ck


def try_restore_remote(fold_dir, fold):
    last = fold_dir / "last.pt"
    if last.exists():
        return
    try:
        release_store().download(RUN_ID, f"fold{fold}_last.pt", last)
        print(f"✓ restored fold {fold} checkpoint from release")
    except Exception:
        pass


def upload_ckpt(path, fold, kind):
    release_store().upload_or_replace(RUN_ID, path, asset_name=f"fold{fold}_{kind}.pt")


def train_fold(fold, cases, splits):
    fold_dir = RUN / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    completed = fold_dir / "COMPLETE.json"
    if completed.exists():
        print(f"✓ Fold {fold} already complete; skipping.")
        return json.loads(completed.read_text())

    f = splits[splits.outer_fold == fold]
    train_ids = f[f.role == "train"].case_id.tolist()
    val_ids = f[f.role == "validation"].case_id.tolist()
    test_ids = f[f.role == "test"].case_id.tolist()
    assert len(train_ids) == 12 and len(val_ids) == 4 and len(test_ids) == 4

    train_ds = SliceDataset(cases, train_ids, True)
    val_ds = SliceDataset(cases, val_ids, False)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    model = UNetResNet34().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    last = fold_dir / "last.pt"
    best_path = fold_dir / "best.pt"
    try_restore_remote(fold_dir, fold)

    start, best, patience = 1, -1.0, 0
    if last.exists():
        ck = restore_checkpoint(last, model, opt, sched, scaler)
        start = int(ck["epoch"]) + 1
        best = float(ck["best"])
        patience = int(ck["patience"])
        print(f"✓ Fold {fold} resume at epoch {start}")

    log_path = fold_dir / "train_log.csv"
    history = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []

    for epoch in range(start, 21):
        model.train()
        losses = []
        bar = tqdm(train_loader, desc=f"Fold {fold}/4 epoch {epoch}/20", leave=True)
        for x, y, _ in bar:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            bar.set_postfix(loss=f"{np.mean(losses):.4f}")

        val_score = validation_score(model, val_loader)
        sched.step(val_score)
        improved = val_score > best + 1e-5
        if improved:
            best, patience = val_score, 0
            save_checkpoint(best_path, model, opt, sched, scaler, epoch, best, patience)
            upload_ckpt(best_path, fold, "best")
        else:
            patience += 1

        save_checkpoint(last, model, opt, sched, scaler, epoch, best, patience)
        if epoch % 2 == 0 or improved:
            upload_ckpt(last, fold, "last")

        history.append({
            "fold": fold,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_mean_dice": val_score,
            "best_validation_mean_dice": best,
            "patience": patience,
            "lr": opt.param_groups[0]["lr"],
        })
        atomic_csv(log_path, pd.DataFrame(history))
        print(f"Fold {fold} epoch {epoch}: val_mean_dice={val_score:.4f}, best={best:.4f}")

        if patience >= 5:
            print(f"✓ Fold {fold} early stopping.")
            break

    if not best_path.exists():
        try:
            release_store().download(RUN_ID, f"fold{fold}_best.pt", best_path)
        except Exception as exc:
            raise RuntimeError(f"Fold {fold} best checkpoint unavailable: {exc}")

    restore_checkpoint(best_path, model)
    fold_results = evaluate_oof_fold(model, fold, test_ids, cases)
    atomic_json(completed, {
        "fold": fold,
        "status": "COMPLETE",
        "best_validation_mean_dice": best,
        "test_case_ids": test_ids,
        "completed_utc": now(),
    })
    sync_git(f"Complete EViCT-Dx Step08 fold {fold} OOF segmentation")
    return fold_results


@torch.inference_mode()
def predict_volume(model, row):
    ct_img = nib.load(abs_data(row.ct_path))
    lung_img = nib.load(abs_data(row.lung_mask_path))
    inf_img = nib.load(abs_data(row.infection_mask_path))
    shape = ct_img.shape
    H, W, Z = shape

    pred_left = np.zeros(shape, dtype=np.uint8)
    pred_right = np.zeros(shape, dtype=np.uint8)
    pred_inf = np.zeros(shape, dtype=np.uint8)

    batch_x, batch_z = [], []
    for z in range(Z):
        arr = np.asanyarray(ct_img.dataobj[:, :, z], dtype=np.float32)
        x = torch.from_numpy(source_image(arr, row.source))
        x = F.interpolate(x[None], size=(256, 256), mode="bilinear", align_corners=False)[0]
        batch_x.append(x)
        batch_z.append(z)
        if len(batch_x) == 8 or z == Z - 1:
            xx = torch.stack(batch_x).to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16):
                prob = torch.sigmoid(model(xx)).float()
            prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False).cpu().numpy()
            for i, zz in enumerate(batch_z):
                L, R, I = prob[i]
                overlap = (L >= .5) & (R >= .5)
                left = L >= .5
                right = R >= .5
                left[overlap] = L[overlap] >= R[overlap]
                right[overlap] = R[overlap] > L[overlap]
                pred_left[:, :, zz] = left.astype(np.uint8)
                pred_right[:, :, zz] = right.astype(np.uint8)
                pred_inf[:, :, zz] = (I >= .5).astype(np.uint8)
            batch_x, batch_z = [], []

    gt_lung = np.asanyarray(lung_img.dataobj)
    gt_inf = np.asanyarray(inf_img.dataobj) > 0
    gt_left = gt_lung == float(row.left_raw_label)
    gt_right = gt_lung == float(row.right_raw_label)
    return pred_left, pred_right, pred_inf, gt_left, gt_right, gt_inf


def dice_np(p, g):
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    return float(dice_from_counts(tp, fp, fn))


def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx, vy = x.var(), y.var()
    return float(2 * np.cov(x, y, ddof=0)[0,1] / (vx + vy + (x.mean()-y.mean())**2 + 1e-12))


def evaluate_oof_fold(model, fold, test_ids, cases):
    rows, qrows = [], []
    pred_dir = RUN / f"fold_{fold}" / "oof_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    case_map = {r.case_id: r for r in cases.itertuples(index=False)}

    for cid in tqdm(test_ids, desc=f"Fold {fold} OOF test volumes"):
        r = case_map[cid]
        pl, pr, pi, gl, gr, gi = predict_volume(model, r)

        lung_pred = (pl > 0) | (pr > 0)
        pi_inside = (pi > 0) & lung_pred
        gt_lung = gl | gr

        dL, dR, dI = dice_np(pl > 0, gl), dice_np(pr > 0, gr), dice_np(pi > 0, gi)
        inter = np.logical_and(pi > 0, gi).sum()
        union = np.logical_or(pi > 0, gi).sum()
        iouI = float(inter / union) if union else 1.0
        tp = np.logical_and(pi > 0, gi).sum()
        fn = np.logical_and(pi == 0, gi).sum()
        tn = np.logical_and(pi == 0, ~gi).sum()
        fp = np.logical_and(pi > 0, ~gi).sum()

        pL = 100.0 * np.logical_and(pi_inside, pl > 0).sum() / max((pl > 0).sum(), 1)
        pR = 100.0 * np.logical_and(pi_inside, pr > 0).sum() / max((pr > 0).sum(), 1)
        pT = 100.0 * pi_inside.sum() / max(lung_pred.sum(), 1)
        bilateral = bool(np.logical_and(pi_inside, pl > 0).any() and np.logical_and(pi_inside, pr > 0).any())

        rows.append({
            "outer_fold": fold, "case_id": cid, "source": r.source,
            "Dice_left": dL, "Dice_right": dR, "Dice_infection": dI,
            "IoU_infection": iouI,
            "infection_sensitivity": float(tp/(tp+fn)) if tp+fn else np.nan,
            "infection_specificity": float(tn/(tn+fp)) if tn+fp else np.nan,
        })
        qrows.append({
            "outer_fold": fold, "case_id": cid,
            "reference_left_percent": float(r.reference_left_involvement_percent),
            "predicted_left_percent": pL,
            "reference_right_percent": float(r.reference_right_involvement_percent),
            "predicted_right_percent": pR,
            "reference_total_percent": float(r.reference_total_involvement_percent),
            "predicted_total_percent": pT,
            "reference_bilateral": bool(r.reference_bilateral_involvement),
            "predicted_bilateral": bilateral,
        })

        np.savez_compressed(
            pred_dir / f"{cid}.npz",
            left=pl, right=pr, infection=pi,
        )

    fold_metrics = pd.DataFrame(rows)
    fold_quant = pd.DataFrame(qrows)
    atomic_csv(RUN / f"fold_{fold}" / "oof_case_metrics.csv", fold_metrics)
    atomic_csv(RUN / f"fold_{fold}" / "oof_quantification.csv", fold_quant)

    tar_path = RUN / f"fold_{fold}" / f"fold{fold}_oof_predictions.tar"
    with tarfile.open(tar_path, "w") as tf:
        for p in sorted(pred_dir.glob("*.npz")):
            tf.add(p, arcname=p.name)
    release_store().upload_or_replace(RUN_ID, tar_path, asset_name=f"fold{fold}_oof_predictions.tar")
    return {"metrics": rows, "quant": qrows}


def summarize_all():
    metric_parts, quant_parts = [], []
    for fold in range(5):
        metric_parts.append(pd.read_csv(RUN / f"fold_{fold}" / "oof_case_metrics.csv"))
        quant_parts.append(pd.read_csv(RUN / f"fold_{fold}" / "oof_quantification.csv"))
    m = pd.concat(metric_parts, ignore_index=True).sort_values("case_id")
    q = pd.concat(quant_parts, ignore_index=True).sort_values("case_id")
    if len(m) != 20 or m.case_id.nunique() != 20:
        raise RuntimeError("Strict OOF coverage is not exactly 20 unique cases.")
    atomic_csv(TABLES / "step08_oof_case_metrics.csv", m)
    atomic_csv(TABLES / "step08_oof_involvement_predictions.csv", q)

    quant_summary = {}
    for side in ["left", "right", "total"]:
        ref = q[f"reference_{side}_percent"].to_numpy(float)
        pred = q[f"predicted_{side}_percent"].to_numpy(float)
        err = pred - ref
        quant_summary[side] = {
            "MAE_percentage_points": float(np.mean(np.abs(err))),
            "median_absolute_error": float(np.median(np.abs(err))),
            "RMSE_percentage_points": float(np.sqrt(np.mean(err**2))),
            "Pearson_r": float(np.corrcoef(ref, pred)[0,1]),
            "CCC": ccc(ref, pred),
            "Bland_Altman_bias": float(np.mean(err)),
            "Bland_Altman_LOA95": [float(np.mean(err)-1.96*np.std(err)), float(np.mean(err)+1.96*np.std(err))],
        }

    yr = q.reference_bilateral.astype(bool).to_numpy()
    yp = q.predicted_bilateral.astype(bool).to_numpy()
    tn = int((~yr & ~yp).sum()); tp = int((yr & yp).sum())
    fp = int((~yr & yp).sum()); fn = int((yr & ~yp).sum())
    bilateral = {
        "balanced_accuracy": float(balanced_accuracy_score(yr, yp)),
        "sensitivity": float(tp/(tp+fn)) if tp+fn else np.nan,
        "specificity": float(tn/(tn+fp)) if tn+fp else np.nan,
        "F1": float(f1_score(yr, yp)),
    }

    result = {
        "project":"EViCT-Dx",
        "stage":"STEP_08_OOF_LUNG_INFECTION_SEGMENTATION_BASELINE",
        "run_id":RUN_ID,
        "completed_utc":now(),
        "strict_oof_cases":20,
        "mean_Dice_left":float(m.Dice_left.mean()),
        "mean_Dice_right":float(m.Dice_right.mean()),
        "mean_Dice_infection":float(m.Dice_infection.mean()),
        "mean_IoU_infection":float(m.IoU_infection.mean()),
        "mean_infection_sensitivity":float(m.infection_sensitivity.mean()),
        "mean_infection_specificity":float(m.infection_specificity.mean()),
        "quantification":quant_summary,
        "bilateral":bilateral,
        "involvement_unlock":{
            "total_MAE_le_5pp": bool(quant_summary["total"]["MAE_percentage_points"] <= 5.0),
            "total_CCC_ge_0.85": bool(quant_summary["total"]["CCC"] >= 0.85),
        },
        "report_fields_unlocked":False,
        "next_action":"STEP_09_REVIEW_OOF_RESULTS_AND_DECIDE_SEGMENTATION_IMPROVEMENT",
    }
    atomic_json(AUDIT / "step08_oof_segmentation_baseline_results.json", result)
    atomic_json(RUN / "STATE.json", {
        "run_id":RUN_ID, "status":"COMPLETE", "updated_utc":now(),
        "strict_oof_cases":20, "next_action":result["next_action"],
    })
    sync_git("Complete EViCT-Dx Step08 strict OOF segmentation baseline")

    print("\n" + "="*108)
    print("✅ STEP 08 COMPLETE — STRICT 5-FOLD OOF")
    print(f"Mean Left Lung Dice          : {result['mean_Dice_left']:.4f}")
    print(f"Mean Right Lung Dice         : {result['mean_Dice_right']:.4f}")
    print(f"Mean Infection Dice          : {result['mean_Dice_infection']:.4f}")
    print(f"Total involvement MAE (pp)   : {quant_summary['total']['MAE_percentage_points']:.3f}")
    print(f"Total involvement CCC        : {quant_summary['total']['CCC']:.4f}")
    print(f"Bilateral balanced accuracy  : {bilateral['balanced_accuracy']:.4f}")
    print("Report fields                : STILL LOCKED pending explicit Step09 unlock audit")
    print("="*108)


def main():
    print("="*108)
    print("EViCT-Dx STEP 08 — LUNG + INFECTION SEGMENTATION BASELINE")
    print("5-fold strict out-of-fold evaluation on the frozen 20-volume protocol")
    print("="*108)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Dataset root:", DATA_ROOT)

    verify = DX / "scripts" / "verify_core_lock.py"
    subprocess.run([sys.executable, str(verify)], cwd=str(ROOT), check=True)

    cases = pd.read_csv(CASE_MANIFEST)
    splits = pd.read_csv(SPLIT_MANIFEST)
    if len(cases) != 20 or cases.case_id.nunique() != 20:
        raise RuntimeError("Step03c case manifest must contain exactly 20 unique volumes.")
    if len(splits) != 100:
        raise RuntimeError("Step03c 5-fold split manifest must contain 100 rows.")

    for fold in range(5):
        train_fold(fold, cases, splits)

    summarize_all()


if __name__ == "__main__":
    main()
