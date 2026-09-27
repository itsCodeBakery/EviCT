from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torchvision.transforms.functional as TVF
from torchvision.transforms import InterpolationMode

from kaggle_secrets import UserSecretsClient


ROOT = Path("/kaggle/working/EviCT")
WORK = Path("/kaggle/working")

CONFIG_PATH = ROOT / "config/notebook05_unet_baseline.json"
MODEL_CODE = ROOT / "src/evict/models/unet2d.py"
METRICS_CODE = ROOT / "src/evict/metrics.py"
SPLITS_PATH = ROOT / "manifests/splits.csv"
TRAIN_MANIFEST = ROOT / "manifests/slice_loaders/seed_17/b100/labeled_slices.csv"
SELECTION_MANIFEST = ROOT / "manifests/source_selection_slices.csv"
STATE_PATH = ROOT / "STATE.md"
HANDOFF_STATE = ROOT / "handoff/STATE.md"
GIT_SYNC = ROOT / "scripts/git_sync.py"

RUN_ROOT = ROOT / "artifacts/large/notebook05"
PRED_ROOT = ROOT / "predictions/notebook05"
AUDIT_ROOT = ROOT / "artifacts/audit"

CACHE_PROBE = ROOT / "cache/segdb2/images/coronacases_003.npy"
CACHE_TAR = WORK / "EviCT_Notebook03_Cache.tar"
CACHE_URL = (
    "https://github.com/itsCodeBakery/EviCT/releases/download/"
    "evict-project-durable-backup-20260927/EviCT_Notebook03_Cache.tar"
)
CACHE_SHA256 = "d88e786cd467816d6cb446333385918016946a83cdd3c258b3011f458fbd3fc4"

SEED = 17
MAX_UPDATES = 5000
SEGMENT_STOP = 1000
WARMUP_UPDATES = 200
VALIDATE_EVERY = 250
RECOVERY_EVERY = 50
PATIENCE_LIMIT = 8
MICRO_BATCH = 4
GRAD_ACCUM = 4
EFFECTIVE_BATCH = 16
POSITIVE_PROB = 0.5
WEIGHT_DECAY = 0.01
THRESHOLD = 0.5

LR_CANDIDATES = [
    ("lr1e4", 1.0e-4),
    ("lr3e4", 3.0e-4),
]

REPO_OWNER = "itsCodeBakery"
REPO_NAME = "EviCT"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def stable_json_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def model_state_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def atomic_torch_save(payload, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def download_verified(url: str, destination: Path, expected_sha: str) -> Path:
    destination = Path(destination)
    if destination.exists() and sha256_file(destination) == expected_sha:
        print(f"✓ Existing verified file  : {destination.name}")
        return destination

    destination.unlink(missing_ok=True)
    part = Path(str(destination) + ".part")
    part.unlink(missing_ok=True)

    print(f"Downloading               : {destination.name}")

    with requests.get(
        url,
        stream=True,
        timeout=(60, 900),
        allow_redirects=True,
    ) as response:
        response.raise_for_status()
        downloaded = 0
        last_report = 0
        with part.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded - last_report >= 256 * 1024 * 1024:
                    print(f"  downloaded              : {downloaded / 1024**2:.0f} MiB")
                    last_report = downloaded

    actual = sha256_file(part)
    assert actual == expected_sha, (
        f"SHA mismatch for {destination.name}\n"
        f"Expected: {expected_sha}\nActual:   {actual}"
    )
    os.replace(part, destination)
    print(f"✓ Download SHA verified   : {destination.name}")
    return destination


def safe_extract_tar(tar_path: Path, destination: Path) -> None:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    with tarfile.open(tar_path, "r") as archive:
        members = archive.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            assert str(target).startswith(str(root)), f"Unsafe TAR member: {member.name}"
        for member in members:
            archive.extract(member, path=destination)


def ensure_cache() -> None:
    if CACHE_PROBE.exists():
        print("✓ Notebook-03 source cache : READY")
        return

    print("Source cache absent. Restoring verified Notebook-03 cache...")
    download_verified(CACHE_URL, CACHE_TAR, CACHE_SHA256)
    safe_extract_tar(CACHE_TAR, WORK)
    assert CACHE_PROBE.exists()
    CACHE_TAR.unlink(missing_ok=True)
    print("✓ Notebook-03 source cache : RESTORED")


def reset_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": [x.cpu() for x in torch.cuda.get_rng_state_all()],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = [x.cpu() for x in state["torch_cuda"]]
    torch.cuda.set_rng_state_all(cuda_states)


def git_sync(message: str) -> None:
    last = None
    for attempt in range(1, 4):
        result = subprocess.run(
            [sys.executable, str(GIT_SYNC), message],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(result.stdout)
        if result.returncode == 0:
            return
        last = result
        print(result.stderr)
        print(f"⚠ Git sync attempt {attempt}/3 failed")
        time.sleep(5 * attempt)

    raise RuntimeError(
        "Git synchronization failed.\n"
        + (last.stderr if last is not None else "")
    )


def normalize_batch(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return (x.repeat(1, 3, 1, 1) - mean) / std


class FittingCaseStore:
    def __init__(self, df: pd.DataFrame):
        self.case_ids = sorted(df["case_id"].astype(str).unique().tolist())
        self.cases = {}

        for case_id in self.case_ids:
            rows = df[df["case_id"].astype(str) == case_id].sort_values("image_array_index")
            first = rows.iloc[0]

            image = np.load(str(first["image_path"]), mmap_mode="r")
            target = np.load(str(first["infection_cache_path"]), mmap_mode="r")
            valid = np.load(str(first["valid_mask_path"]), mmap_mode="r")

            all_z = rows["image_array_index"].astype(int).to_numpy()
            lesion_area = target[all_z].reshape(len(all_z), -1).sum(axis=1)
            positive_z = all_z[lesion_area > 0]

            self.cases[case_id] = {
                "image": image,
                "target": target,
                "valid": valid,
                "all_z": all_z,
                "positive_z": positive_z,
            }


class PatientUniformSampler:
    def __init__(self, store: FittingCaseStore, seed: int):
        self.store = store
        self.case_ids = list(store.case_ids)
        self.rng = np.random.default_rng(seed)
        self.samples_seen = 0

    def sample(self, batch_size: int):
        images, targets, valids = [], [], []

        for _ in range(batch_size):
            case_id = str(self.rng.choice(self.case_ids))
            case = self.store.cases[case_id]

            choose_positive = (
                len(case["positive_z"]) > 0
                and self.rng.random() < POSITIVE_PROB
            )

            z = int(
                self.rng.choice(
                    case["positive_z"] if choose_positive else case["all_z"]
                )
            )

            images.append(
                np.asarray(case["image"][z], dtype=np.float32)[None, ...]
            )
            targets.append(
                np.asarray(case["target"][z], dtype=np.float32)[None, ...]
            )
            valids.append(
                np.asarray(case["valid"], dtype=np.float32)[None, ...]
            )

        self.samples_seen += batch_size
        return np.stack(images), np.stack(targets), np.stack(valids)

    def state_dict(self) -> dict:
        return {
            "bit_generator_state": self.rng.bit_generator.state,
            "samples_seen": int(self.samples_seen),
        }

    def load_state_dict(self, state: dict) -> None:
        self.rng.bit_generator.state = state["bit_generator_state"]
        self.samples_seen = int(state["samples_seen"])


def augment(images: torch.Tensor, targets: torch.Tensor, valids: torch.Tensor):
    out_i, out_t, out_v = [], [], []

    for idx in range(images.shape[0]):
        image = images[idx]
        target = targets[idx]
        valid = valids[idx]

        angle = random.uniform(-10.0, 10.0)
        scale = random.uniform(0.9, 1.1)
        tx = int(round(random.uniform(-0.05, 0.05) * 336))
        ty = int(round(random.uniform(-0.05, 0.05) * 336))

        image = TVF.affine(
            image,
            angle=angle,
            translate=[tx, ty],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        target = TVF.affine(
            target,
            angle=angle,
            translate=[tx, ty],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0.0,
        )
        valid = TVF.affine(
            valid,
            angle=angle,
            translate=[tx, ty],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0.0,
        )

        target = (target > 0.5).float()
        valid = (valid > 0.5).float()

        image = TVF.adjust_brightness(image, random.uniform(0.9, 1.1))
        image = TVF.adjust_contrast(image, random.uniform(0.9, 1.1))
        image = torch.clamp(image, 0.0, 1.0)
        image = TVF.adjust_gamma(image, gamma=random.uniform(0.8, 1.2))

        if random.random() < 0.30:
            image = image + torch.randn_like(image) * random.uniform(0.0, 0.03)

        if random.random() < 0.20:
            image = TVF.gaussian_blur(
                image,
                kernel_size=[3, 3],
                sigma=[0.1, 1.0],
            )

        image = torch.clamp(image, 0.0, 1.0) * valid

        out_i.append(image)
        out_t.append(target)
        out_v.append(valid)

    return torch.stack(out_i), torch.stack(out_t), torch.stack(out_v)


def scheduler_multiplier(step: int) -> float:
    if step < WARMUP_UPDATES:
        return float(step + 1) / float(WARMUP_UPDATES)

    progress = (step - WARMUP_UPDATES) / float(MAX_UPDATES - WARMUP_UPDATES)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def confusion_counts(reference, prediction, valid):
    ref = np.asarray(reference) > 0
    pred = np.asarray(prediction) > 0
    v = np.asarray(valid) > 0

    tp = int(np.logical_and.reduce((v, ref, pred)).sum())
    tn = int(np.logical_and.reduce((v, ~ref, ~pred)).sum())
    fp = int(np.logical_and.reduce((v, ~ref, pred)).sum())
    fn = int(np.logical_and.reduce((v, ref, ~pred)).sum())
    return tp, tn, fp, fn


def make_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    sampler,
    candidate_name,
    learning_rate,
    global_step,
    best_score,
    best_step,
    patience_count,
    last_validation_step,
    images_seen,
    config_hash,
    split_hash,
    model_code_hash,
    metrics_code_hash,
    init_hash,
):
    return {
        "format_version": 1,
        "project": "EviCT",
        "stage": "NOTEBOOK_05B",
        "run_id": f"unet2d_seed17_{candidate_name}_fp32",
        "candidate_name": candidate_name,
        "learning_rate": float(learning_rate),
        "seed": SEED,
        "split_seed": 17,
        "precision": "FP32",
        "amp_enabled": False,
        "global_step": int(global_step),
        "best_score": float(best_score),
        "best_step": int(best_step),
        "patience_count": int(patience_count),
        "last_validation_step": int(last_validation_step),
        "images_seen": int(images_seen),
        "student_state_dict": {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "sampler_state": sampler.state_dict(),
        "config_hash": config_hash,
        "split_hash": split_hash,
        "model_code_hash": model_code_hash,
        "metrics_code_hash": metrics_code_hash,
        "initialization_hash": init_hash,
        "target_accessed": False,
        "calibration_accessed": False,
    }


def validate(
    *,
    model,
    device,
    selection_df,
    masked_supervised_loss,
    CaseMetricAccumulator,
    binary_segmentation_metrics,
    metrics_from_confusion,
    macro_case_summary,
):
    model.eval()

    case_rows = []
    all_slice_dice = []
    per_image_losses = []
    pooled_tp = pooled_tn = pooled_fp = pooled_fn = 0
    raw_logits = {}

    with torch.no_grad():
        for case_id in sorted(selection_df["case_id"].astype(str).unique()):
            rows = selection_df[
                selection_df["case_id"].astype(str) == case_id
            ].sort_values("image_array_index")

            first = rows.iloc[0]
            image_vol = np.load(str(first["image_path"]), mmap_mode="r")
            target_vol = np.load(str(first["infection_cache_path"]), mmap_mode="r")
            valid = np.load(str(first["valid_mask_path"]), mmap_mode="r").astype(np.float32)

            all_z = rows["image_array_index"].astype(int).to_numpy()
            assert len(all_z) == image_vol.shape[0], (
                f"Selection manifest is not complete for {case_id}"
            )

            accumulator = CaseMetricAccumulator(case_id)
            case_logits = np.empty(
                (len(all_z), valid.shape[0], valid.shape[1]),
                dtype=np.float32,
            )

            for start in range(0, len(all_z), 8):
                end = min(start + 8, len(all_z))
                z = all_z[start:end]

                images = torch.from_numpy(
                    np.asarray(image_vol[z], dtype=np.float32)[:, None, :, :]
                ).to(device=device, dtype=torch.float32)

                targets = torch.from_numpy(
                    np.asarray(target_vol[z], dtype=np.float32)[:, None, :, :]
                ).to(device=device, dtype=torch.float32)

                valids = torch.from_numpy(
                    np.broadcast_to(
                        valid[None, None, :, :],
                        (len(z), 1, valid.shape[0], valid.shape[1]),
                    ).copy()
                ).to(device=device, dtype=torch.float32)

                inputs = normalize_batch(images, device)
                logits = model(inputs)["logits"].float()

                loss_dict = masked_supervised_loss(logits, targets, valids)
                per_image_losses.extend(
                    loss_dict["per_image_loss"].detach().cpu().numpy().astype(float).tolist()
                )

                logits_np = logits[:, 0].detach().cpu().numpy().astype(np.float32)
                case_logits[start:end] = logits_np

                pred_np = (logits_np >= 0.0).astype(np.uint8)
                target_np = targets[:, 0].detach().cpu().numpy().astype(np.uint8)
                valid_np = valids[:, 0].detach().cpu().numpy().astype(np.uint8)

                for j in range(len(z)):
                    accumulator.update(
                        target_np[j],
                        pred_np[j],
                        valid_np[j],
                    )
                    sm = binary_segmentation_metrics(
                        target_np[j],
                        pred_np[j],
                        valid_np[j],
                    )
                    all_slice_dice.append(float(sm["dice"]))

                    tp, tn, fp, fn = confusion_counts(
                        target_np[j],
                        pred_np[j],
                        valid_np[j],
                    )
                    pooled_tp += tp
                    pooled_tn += tn
                    pooled_fp += fp
                    pooled_fn += fn

                del images, targets, valids, inputs, logits, loss_dict

            case_record = accumulator.compute()
            case_rows.append(case_record)
            raw_logits[case_id] = case_logits

    summary = macro_case_summary(case_rows)
    pooled = metrics_from_confusion(
        pooled_tp,
        pooled_tn,
        pooled_fp,
        pooled_fn,
    )

    metrics = {
        "selection_loss": float(np.mean(per_image_losses)),
        "macro_case_dice": float(summary["macro_dice"]),
        "macro_case_iou": float(summary["macro_iou"]),
        "macro_case_sensitivity": float(summary["macro_sensitivity"]),
        "macro_case_specificity": float(summary["macro_specificity"]),
        "macro_slice_dice": float(np.mean(all_slice_dice)),
        "pooled_dice": float(pooled["dice"]),
        "pooled_iou": float(pooled["iou"]),
        "threshold": THRESHOLD,
        "n_cases": int(summary["n_cases"]),
    }

    model.train()
    return metrics, case_rows, raw_logits


def install_best_logits(
    *,
    prediction_dir: Path,
    raw_logits: dict,
    step: int,
    candidate_name: str,
):
    best_dir = prediction_dir / "best_selection_logits"
    tmp_dir = prediction_dir / "best_selection_logits.tmp"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cases = []

    for case_id, arr in raw_logits.items():
        path = tmp_dir / f"{case_id}.npy"
        np.save(path, np.asarray(arr, dtype=np.float32))
        cases.append({
            "case_id": case_id,
            "shape": list(arr.shape),
            "dtype": "float32",
            "sha256": sha256_file(path),
        })

    if best_dir.exists():
        shutil.rmtree(best_dir)
    os.replace(tmp_dir, best_dir)

    manifest = {
        "timestamp_utc": utc_now(),
        "stage": "NOTEBOOK_05B",
        "candidate_name": candidate_name,
        "seed": SEED,
        "step": int(step),
        "threshold": THRESHOLD,
        "calibration_accessed": False,
        "target_accessed": False,
        "cases": cases,
    }

    manifest_path = AUDIT_ROOT / f"notebook05b_unet_seed17_{candidate_name}_best_logits_manifest.json"
    atomic_text(manifest_path, json.dumps(manifest, indent=2))


def write_logs(
    candidate_name: str,
    train_rows: list,
    selection_rows: list,
    case_metric_rows: list,
):
    pd.DataFrame(train_rows).to_csv(
        AUDIT_ROOT / f"notebook05b_unet_seed17_{candidate_name}_train_log.csv",
        index=False,
    )
    pd.DataFrame(selection_rows).to_csv(
        AUDIT_ROOT / f"notebook05b_unet_seed17_{candidate_name}_selection_metrics.csv",
        index=False,
    )
    pd.DataFrame(case_metric_rows).to_csv(
        AUDIT_ROOT / f"notebook05b_unet_seed17_{candidate_name}_selection_case_metrics.csv",
        index=False,
    )


def update_state(
    *,
    candidate_name: str,
    learning_rate: float,
    global_step: int,
    best_score: float,
    best_step: int,
    patience_count: int,
    images_seen: int,
    status: str,
):
    text = f"""# EviCT Execution State

## Current stage

{status}

## Timestamp

{utc_now()}

## Notebook 04

Status:

COMPLETE_SOURCE_BASELINE_FROZEN

Notebook 04 retraining:

NO

## Notebook 05B

Baseline:

Competitive residual 2D U-Net

Pilot seed:

17

Candidate:

{candidate_name}

Learning rate:

{learning_rate:.8f}

Precision:

FP32

Optimizer step:

{global_step}

Best source-selection macro case Dice:

{best_score:.8f}

Best checkpoint step:

{best_step}

Patience:

{patience_count} / {PATIENCE_LIMIT}

Image exposures:

{images_seen}

Selection threshold:

0.5 fixed

Architecture tuning:

NO

## Isolation

Fitting cases:

12 frozen source fitting cases

Selection cases:

4 frozen complete source-selection cases

Calibration accessed:

NO

Target / MedSeg accessed:

NO

Target lock:

ACTIVE

## Pilot contract

Learning-rate candidates:

0.0001, 0.0003

Candidate comparison:

PENDING until both candidates complete the full declared budget or early stopping.

## Next

Continue the declared U-Net learning-rate pilot without changing architecture,
data, loss, selection metric, threshold, or target lock.
"""
    atomic_text(STATE_PATH, text)
    HANDOFF_STATE.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(HANDOFF_STATE, text)


def create_recovery_archive(
    *,
    candidate_name: str,
    run_dir: Path,
    step: int,
):
    archive = WORK / (
        f"EviCT_Notebook05B_unet_seed17_{candidate_name}_"
        f"step{step}_Recovery.tar"
    )

    if archive.exists():
        archive.unlink()

    members = [
        run_dir / "recovery.pt",
        run_dir / "best.pt",
    ]

    with tarfile.open(archive, "w") as tar:
        for path in members:
            assert path.exists()
            tar.add(
                path,
                arcname=str(
                    Path("EviCT/artifacts/large/notebook05")
                    / run_dir.name
                    / path.name
                ),
            )

        for suffix in [
            "train_log.csv",
            "selection_metrics.csv",
            "selection_case_metrics.csv",
            "best_logits_manifest.json",
        ]:
            if suffix == "best_logits_manifest.json":
                source = AUDIT_ROOT / f"notebook05b_unet_seed17_{candidate_name}_best_logits_manifest.json"
            else:
                source = AUDIT_ROOT / f"notebook05b_unet_seed17_{candidate_name}_{suffix}"
            if source.exists():
                tar.add(
                    source,
                    arcname=str(
                        Path("EviCT/artifacts/audit") / source.name
                    ),
                )

    digest = sha256_file(archive)
    sha_path = Path(str(archive) + ".sha256")
    atomic_text(sha_path, digest)
    return archive, sha_path, digest


def create_or_get_release(token: str, tag: str, name: str, body: str):
    api = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    lookup = requests.get(
        f"{api}/releases/tags/{tag}",
        headers=headers,
        timeout=60,
    )

    if lookup.status_code == 200:
        return lookup.json(), headers, api

    assert lookup.status_code == 404, (
        f"Release lookup failed: HTTP {lookup.status_code}\n{lookup.text[:2000]}"
    )

    create = requests.post(
        f"{api}/releases",
        headers=headers,
        json={
            "tag_name": tag,
            "target_commitish": "main",
            "name": name,
            "body": body,
            "draft": False,
            "prerelease": False,
        },
        timeout=60,
    )
    assert create.status_code in {200, 201}, (
        f"Release creation failed: HTTP {create.status_code}\n{create.text[:2000]}"
    )
    return create.json(), headers, api


def upload_asset(
    *,
    release,
    headers,
    api,
    file_path: Path,
    content_type: str,
):
    release_id = int(release["id"])
    current = requests.get(
        f"{api}/releases/{release_id}",
        headers=headers,
        timeout=60,
    )
    current.raise_for_status()

    assets = {
        asset["name"]: asset
        for asset in current.json().get("assets", [])
    }

    local_sha = sha256_file(file_path)

    if file_path.name in assets:
        asset = assets[file_path.name]
        remote_digest = asset.get("digest")
        if (
            int(asset["size"]) == file_path.stat().st_size
            and (
                remote_digest is None
                or remote_digest == f"sha256:{local_sha}"
            )
        ):
            print(f"✓ Already durable          : {file_path.name}")
            return asset

        delete = requests.delete(
            f"{api}/releases/assets/{asset['id']}",
            headers=headers,
            timeout=60,
        )
        assert delete.status_code in {204, 404}

    upload_headers = {
        **headers,
        "Content-Type": content_type,
        "Content-Length": str(file_path.stat().st_size),
    }

    upload_url = (
        f"https://uploads.github.com/repos/{REPO_OWNER}/{REPO_NAME}/"
        f"releases/{release_id}/assets"
    )

    print(
        f"Uploading                 : {file_path.name} "
        f"({file_path.stat().st_size / 1024**2:.2f} MiB)"
    )

    with file_path.open("rb") as handle:
        response = requests.post(
            upload_url,
            headers=upload_headers,
            params={"name": file_path.name},
            data=handle,
            timeout=7200,
        )

    assert response.status_code in {200, 201}, (
        f"Upload failed for {file_path.name}: "
        f"HTTP {response.status_code}\n{response.text[:2000]}"
    )

    asset = response.json()
    assert int(asset["size"]) == file_path.stat().st_size

    remote_digest = asset.get("digest")
    if remote_digest is not None:
        assert remote_digest == f"sha256:{local_sha}"

    print(f"✓ Uploaded and verified    : {file_path.name}")
    return asset


def make_durable_step1000(
    *,
    token,
    candidate_name,
    learning_rate,
    run_dir,
    best_score,
    best_step,
    init_hash,
):
    archive, sha_path, digest = create_recovery_archive(
        candidate_name=candidate_name,
        run_dir=run_dir,
        step=SEGMENT_STOP,
    )

    tag = f"evict-nb05b-unet-seed17-{candidate_name}-step1000"
    release_name = (
        f"EviCT Notebook 05B — U-Net seed17 {candidate_name} — Step 1000"
    )
    body = (
        "Durable step-1000 recovery for the Notebook 05B competitive residual "
        f"2D U-Net learning-rate pilot. Candidate={candidate_name}, "
        f"lr={learning_rate}, seed=17, best source-selection macro case Dice "
        f"so far={best_score:.8f} at step {best_step}. "
        "Calibration and target/MedSeg were not accessed."
    )

    release, headers, api = create_or_get_release(
        token,
        tag,
        release_name,
        body,
    )

    tar_asset = upload_asset(
        release=release,
        headers=headers,
        api=api,
        file_path=archive,
        content_type="application/x-tar",
    )
    sha_asset = upload_asset(
        release=release,
        headers=headers,
        api=api,
        file_path=sha_path,
        content_type="text/plain",
    )

    audit = {
        "timestamp_utc": utc_now(),
        "stage": "NOTEBOOK_05B",
        "candidate_name": candidate_name,
        "learning_rate": learning_rate,
        "seed": SEED,
        "step": SEGMENT_STOP,
        "best_macro_case_dice": best_score,
        "best_step": best_step,
        "initialization_hash": init_hash,
        "recovery_archive": archive.name,
        "recovery_archive_sha256": digest,
        "release_tag": tag,
        "release_url": release["html_url"],
        "tar_asset_id": int(tar_asset["id"]),
        "sha_asset_id": int(sha_asset["id"]),
        "calibration_accessed": False,
        "target_accessed": False,
        "status": "DURABLE_STEP1000",
    }

    audit_path = AUDIT_ROOT / (
        f"notebook05b_unet_seed17_{candidate_name}_step1000_durable.json"
    )
    atomic_text(audit_path, json.dumps(audit, indent=2))

    print(f"✓ Step-1000 durable release: {release['html_url']}")
    return audit


def run_candidate(
    *,
    candidate_name: str,
    learning_rate: float,
    device: torch.device,
    train_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    config_hash: str,
    split_hash: str,
    model_code_hash: str,
    metrics_code_hash: str,
    token: str,
    expected_init_hash: str | None,
):
    from evict.models.unet2d import EviCTResidualUNet2D, masked_supervised_loss
    from evict.metrics import (
        CaseMetricAccumulator,
        binary_segmentation_metrics,
        macro_case_summary,
        metrics_from_confusion,
    )

    print()
    print("=" * 110)
    print(
        f"NOTEBOOK 05B — U-NET SEED 17 — {candidate_name} "
        f"(LR={learning_rate:.1e}) — 0 -> 1000"
    )
    print("=" * 110)

    run_name = f"unet_seed17_{candidate_name}"
    run_dir = RUN_ROOT / run_name
    prediction_dir = PRED_ROOT / run_name

    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    best_pt = run_dir / "best.pt"
    last_pt = run_dir / "last.pt"
    recovery_pt = run_dir / "recovery.pt"

    # For a fair LR pilot, each candidate begins from the exact same random
    # initialization and exact same RNG/sampling trajectory.
    reset_rng(SEED)

    store = FittingCaseStore(train_df)
    sampler = PatientUniformSampler(store, SEED)

    model = EviCTResidualUNet2D(
        in_channels=3,
        base_channels=32,
    )

    init_hash = model_state_hash(model)
    print(f"Initial model SHA-256      : {init_hash}")

    if expected_init_hash is not None:
        assert init_hash == expected_init_hash, (
            "Learning-rate candidates did not start from identical initialization."
        )
        print("✓ Identical initialization : PASS")

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=scheduler_multiplier,
    )

    # Segment-1 starts fresh. If local candidate checkpoints exist, accept only
    # a valid <=1000 resume. This protects reruns inside the same live session.
    global_step = 0
    best_score = float("-inf")
    best_step = 0
    patience_count = 0
    last_validation_step = 0
    images_seen = 0
    train_rows = []
    selection_rows = []
    case_metric_rows = []

    local_candidates = [p for p in [last_pt, recovery_pt] if p.exists()]

    if local_candidates:
        metas = []
        for path in local_candidates:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            try:
                assert payload["candidate_name"] == candidate_name
                assert abs(float(payload["learning_rate"]) - learning_rate) < 1e-15
                assert int(payload["seed"]) == SEED
                assert payload["precision"] == "FP32"
                assert payload["config_hash"] == config_hash
                assert payload["split_hash"] == split_hash
                assert payload["model_code_hash"] == model_code_hash
                assert payload["initialization_hash"] == init_hash
                metas.append((int(payload["global_step"]), path))
            finally:
                del payload

        if metas:
            _, resume_path = max(metas)
            payload = torch.load(resume_path, map_location="cpu", weights_only=False)

            model.load_state_dict(payload["student_state_dict"])
            model.to(device)

            optimizer.load_state_dict(payload["optimizer_state_dict"])
            for state in optimizer.state.values():
                for key, value in list(state.items()):
                    if torch.is_tensor(value):
                        state[key] = value.to(device)

            scheduler.load_state_dict(payload["scheduler_state_dict"])
            sampler.load_state_dict(payload["sampler_state"])
            restore_rng_state(payload["rng_state"])

            global_step = int(payload["global_step"])
            best_score = float(payload["best_score"])
            best_step = int(payload["best_step"])
            patience_count = int(payload["patience_count"])
            last_validation_step = int(payload["last_validation_step"])
            images_seen = int(payload["images_seen"])
            del payload

            for kind, target in [
                ("train_log", train_rows),
                ("selection_metrics", selection_rows),
                ("selection_case_metrics", case_metric_rows),
            ]:
                csv_path = AUDIT_ROOT / f"notebook05b_unet_seed17_{candidate_name}_{kind}.csv"
                if csv_path.exists():
                    target.extend(pd.read_csv(csv_path).to_dict("records"))

            print(f"✓ Local resume             : step {global_step}")

    assert global_step <= SEGMENT_STOP

    model.train()

    start_wall = time.perf_counter()

    while global_step < SEGMENT_STOP:
        optimizer.zero_grad(set_to_none=True)

        accum_loss = 0.0
        accum_dice = 0.0
        accum_bce = 0.0

        for _ in range(GRAD_ACCUM):
            images_np, targets_np, valids_np = sampler.sample(MICRO_BATCH)

            images = torch.from_numpy(images_np)
            targets = torch.from_numpy(targets_np)
            valids = torch.from_numpy(valids_np)

            images, targets, valids = augment(images, targets, valids)

            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device=device, dtype=torch.float32, non_blocking=True)
            valids = valids.to(device=device, dtype=torch.float32, non_blocking=True)

            inputs = normalize_batch(images, device)
            logits = model(inputs)["logits"]

            loss_dict = masked_supervised_loss(
                logits,
                targets,
                valids,
            )

            loss = loss_dict["loss"] / GRAD_ACCUM
            assert torch.isfinite(loss)

            loss.backward()

            accum_loss += float(loss_dict["loss"].detach().cpu()) / GRAD_ACCUM
            accum_dice += float(loss_dict["dice_loss"].detach().cpu()) / GRAD_ACCUM
            accum_bce += float(loss_dict["bce_loss"].detach().cpu()) / GRAD_ACCUM

            del images, targets, valids, inputs, logits, loss_dict, loss

        for parameter in model.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), (
                    f"Non-finite gradient at update {global_step + 1}"
                )

        optimizer.step()
        scheduler.step()

        global_step += 1
        images_seen += EFFECTIVE_BATCH

        train_rows.append({
            "step": global_step,
            "loss": accum_loss,
            "dice_loss": accum_dice,
            "bce_loss": accum_bce,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "images_seen": images_seen,
            "precision": "FP32",
        })

        if global_step % 25 == 0:
            elapsed = time.perf_counter() - start_wall
            rate = global_step / max(elapsed, 1e-9)
            eta_min = (SEGMENT_STOP - global_step) / max(rate, 1e-9) / 60.0
            print(
                f"[{candidate_name}] step {global_step:4d}/{SEGMENT_STOP} | "
                f"loss={accum_loss:.6f} | "
                f"lr={optimizer.param_groups[0]['lr']:.3e} | "
                f"best={best_score if np.isfinite(best_score) else float('nan'):.6f} | "
                f"ETA~{eta_min:.1f} min"
            )

        if global_step % RECOVERY_EVERY == 0 and global_step % VALIDATE_EVERY != 0:
            checkpoint = make_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                candidate_name=candidate_name,
                learning_rate=learning_rate,
                global_step=global_step,
                best_score=best_score,
                best_step=best_step,
                patience_count=patience_count,
                last_validation_step=last_validation_step,
                images_seen=images_seen,
                config_hash=config_hash,
                split_hash=split_hash,
                model_code_hash=model_code_hash,
                metrics_code_hash=metrics_code_hash,
                init_hash=init_hash,
            )
            atomic_torch_save(checkpoint, recovery_pt)
            del checkpoint

        if global_step % VALIDATE_EVERY == 0:
            print()
            print(
                f"[{candidate_name}] SOURCE-SELECTION VALIDATION @ step {global_step}"
            )

            validation_metrics, case_rows, raw_logits = validate(
                model=model,
                device=device,
                selection_df=selection_df,
                masked_supervised_loss=masked_supervised_loss,
                CaseMetricAccumulator=CaseMetricAccumulator,
                binary_segmentation_metrics=binary_segmentation_metrics,
                metrics_from_confusion=metrics_from_confusion,
                macro_case_summary=macro_case_summary,
            )

            score = validation_metrics["macro_case_dice"]
            improved = score > best_score

            if improved:
                best_score = score
                best_step = global_step
                patience_count = 0

                best_checkpoint = make_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    candidate_name=candidate_name,
                    learning_rate=learning_rate,
                    global_step=global_step,
                    best_score=best_score,
                    best_step=best_step,
                    patience_count=patience_count,
                    last_validation_step=global_step,
                    images_seen=images_seen,
                    config_hash=config_hash,
                    split_hash=split_hash,
                    model_code_hash=model_code_hash,
                    metrics_code_hash=metrics_code_hash,
                    init_hash=init_hash,
                )
                atomic_torch_save(best_checkpoint, best_pt)
                del best_checkpoint

                install_best_logits(
                    prediction_dir=prediction_dir,
                    raw_logits=raw_logits,
                    step=global_step,
                    candidate_name=candidate_name,
                )
            else:
                patience_count += 1

            last_validation_step = global_step

            selection_rows.append({
                "candidate_name": candidate_name,
                "learning_rate": learning_rate,
                "seed": SEED,
                "step": global_step,
                **validation_metrics,
                "best_so_far": best_score,
                "best_step": best_step,
                "patience_count": patience_count,
                "precision": "FP32",
            })

            for row in case_rows:
                case_metric_rows.append({
                    "candidate_name": candidate_name,
                    "learning_rate": learning_rate,
                    "seed": SEED,
                    "step": global_step,
                    **row,
                })

            current_checkpoint = make_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                candidate_name=candidate_name,
                learning_rate=learning_rate,
                global_step=global_step,
                best_score=best_score,
                best_step=best_step,
                patience_count=patience_count,
                last_validation_step=last_validation_step,
                images_seen=images_seen,
                config_hash=config_hash,
                split_hash=split_hash,
                model_code_hash=model_code_hash,
                metrics_code_hash=metrics_code_hash,
                init_hash=init_hash,
            )

            atomic_torch_save(current_checkpoint, last_pt)
            atomic_torch_save(current_checkpoint, recovery_pt)
            del current_checkpoint

            write_logs(
                candidate_name,
                train_rows,
                selection_rows,
                case_metric_rows,
            )

            update_state(
                candidate_name=candidate_name,
                learning_rate=learning_rate,
                global_step=global_step,
                best_score=best_score,
                best_step=best_step,
                patience_count=patience_count,
                images_seen=images_seen,
                status=f"NOTEBOOK_05B_UNET_{candidate_name.upper()}_RUNNING_STEP_{global_step}",
            )

            print(
                f"  macro case Dice          : {score:.8f}\n"
                f"  macro case IoU           : {validation_metrics['macro_case_iou']:.8f}\n"
                f"  macro slice Dice         : {validation_metrics['macro_slice_dice']:.8f}\n"
                f"  pooled Dice              : {validation_metrics['pooled_dice']:.8f}\n"
                f"  best                     : {best_score:.8f} @ {best_step}\n"
                f"  patience                 : {patience_count}/{PATIENCE_LIMIT}"
            )

            git_sync(
                f"Notebook 05B U-Net seed17 {candidate_name} validation step {global_step}"
            )

            del raw_logits
            gc.collect()
            torch.cuda.empty_cache()

    assert global_step == SEGMENT_STOP
    assert last_validation_step == SEGMENT_STOP
    assert best_pt.exists()
    assert recovery_pt.exists()

    durable = make_durable_step1000(
        token=token,
        candidate_name=candidate_name,
        learning_rate=learning_rate,
        run_dir=run_dir,
        best_score=best_score,
        best_step=best_step,
        init_hash=init_hash,
    )

    update_state(
        candidate_name=candidate_name,
        learning_rate=learning_rate,
        global_step=global_step,
        best_score=best_score,
        best_step=best_step,
        patience_count=patience_count,
        images_seen=images_seen,
        status=f"NOTEBOOK_05B_UNET_{candidate_name.upper()}_STEP1000_DURABLE",
    )

    git_sync(
        f"Record durable Notebook 05B U-Net seed17 {candidate_name} step1000"
    )

    summary = {
        "candidate_name": candidate_name,
        "learning_rate": learning_rate,
        "seed": SEED,
        "step": global_step,
        "best_macro_case_dice": best_score,
        "best_step": best_step,
        "patience_count": patience_count,
        "images_seen": images_seen,
        "initialization_hash": init_hash,
        "release_url": durable["release_url"],
        "recovery_archive_sha256": durable["recovery_archive_sha256"],
    }

    del model, optimizer, scheduler, sampler, store
    gc.collect()
    torch.cuda.empty_cache()

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["segment1"],
        default="segment1",
    )
    args = parser.parse_args()

    print("=" * 110)
    print("EVICT NOTEBOOK 05B — U-NET LEARNING-RATE PILOT — STEP 0 -> 1000")
    print("=" * 110)

    assert args.mode == "segment1"
    assert ROOT.exists() and (ROOT / ".git").exists()
    assert CONFIG_PATH.exists()
    assert MODEL_CODE.exists()
    assert METRICS_CODE.exists()
    assert SPLITS_PATH.exists()
    assert TRAIN_MANIFEST.exists()
    assert SELECTION_MANIFEST.exists()
    assert GIT_SYNC.exists()

    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))

    token = UserSecretsClient().get_secret("pushEviCT")
    assert token
    print("✓ GitHub secret            : PASS")

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    print(f"✓ GPU                      : {torch.cuda.get_device_name(device)}")

    ensure_cache()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["stage"] == "NOTEBOOK_05"
    assert config["baseline_name"] == "competitive_residual_unet2d"
    assert config["architecture"]["pretraining"] == "none"
    assert config["optimizer_plan"]["learning_rate_candidates"] == [0.0001, 0.0003]
    assert config["data_contract"]["target_access_allowed"] is False
    assert config["validation"]["threshold_tuning"] is False
    assert config["batch"]["micro_batch"] == MICRO_BATCH
    assert config["batch"]["gradient_accumulation"] == GRAD_ACCUM
    assert config["batch"]["images_per_optimizer_update"] == EFFECTIVE_BATCH

    split_df = pd.read_csv(SPLITS_PATH)
    train_df = pd.read_csv(TRAIN_MANIFEST)
    selection_df = pd.read_csv(SELECTION_MANIFEST)

    fitting_cases = set(
        split_df.loc[split_df["source_split"] == "fitting", "case_id"].astype(str)
    )
    selection_cases = set(
        split_df.loc[split_df["source_split"] == "selection", "case_id"].astype(str)
    )
    calibration_cases = set(
        split_df.loc[split_df["source_split"] == "calibration", "case_id"].astype(str)
    )

    assert len(fitting_cases) == 12
    assert len(selection_cases) == 4
    assert len(calibration_cases) == 4
    assert fitting_cases.isdisjoint(selection_cases)
    assert fitting_cases.isdisjoint(calibration_cases)
    assert selection_cases.isdisjoint(calibration_cases)

    assert set(train_df["case_id"].astype(str)) == fitting_cases
    assert set(selection_df["case_id"].astype(str)) == selection_cases
    assert set(train_df["seed"].astype(int)) == {17}

    manifest_text = (
        " ".join(train_df.fillna("").astype(str).values.ravel())
        + " "
        + " ".join(selection_df.fillna("").astype(str).values.ravel())
    ).lower()

    for forbidden in ["medseg", "segdb1", "images_medseg", "masks_medseg"]:
        assert forbidden not in manifest_text

    print("✓ Frozen fitting cases     : 12")
    print("✓ Frozen selection cases   : 4")
    print("✓ Calibration accessed     : NO")
    print("✓ Target / MedSeg accessed : NO")

    config_hash = stable_json_hash(config)
    split_hash = stable_json_hash({
        "splits_csv": sha256_file(SPLITS_PATH),
        "train_manifest": sha256_file(TRAIN_MANIFEST),
        "selection_manifest": sha256_file(SELECTION_MANIFEST),
    })
    model_code_hash = sha256_file(MODEL_CODE)
    metrics_code_hash = sha256_file(METRICS_CODE)

    results = []
    shared_init_hash = None

    for candidate_name, learning_rate in LR_CANDIDATES:
        result = run_candidate(
            candidate_name=candidate_name,
            learning_rate=learning_rate,
            device=device,
            train_df=train_df,
            selection_df=selection_df,
            config_hash=config_hash,
            split_hash=split_hash,
            model_code_hash=model_code_hash,
            metrics_code_hash=metrics_code_hash,
            token=token,
            expected_init_hash=shared_init_hash,
        )

        if shared_init_hash is None:
            shared_init_hash = result["initialization_hash"]
        else:
            assert result["initialization_hash"] == shared_init_hash

        results.append(result)

    audit = {
        "timestamp_utc": utc_now(),
        "stage": "NOTEBOOK_05B",
        "pilot_seed": SEED,
        "segment_stop": SEGMENT_STOP,
        "candidates": results,
        "identical_initialization": True,
        "initialization_hash": shared_init_hash,
        "architecture_tuning": False,
        "candidate_winner_selected": False,
        "winner_selection_deferred_until_full_declared_budget": True,
        "threshold": THRESHOLD,
        "threshold_tuned": False,
        "calibration_accessed": False,
        "target_accessed": False,
        "status": "BOTH_CANDIDATES_STEP1000_DURABLE",
    }

    audit_path = AUDIT_ROOT / "notebook05b_unet_lr_pilot_step1000.json"
    atomic_text(audit_path, json.dumps(audit, indent=2))

    text = f"""# EviCT Execution State

## Current stage

NOTEBOOK_05B_UNET_LR_PILOT_BOTH_STEP1000_DURABLE

## Timestamp

{utc_now()}

## Notebook 05B

Baseline:

Competitive residual 2D U-Net

Pilot seed:

17

Architecture tuning:

NO

Candidate initialization:

IDENTICAL

Initialization SHA-256:

{shared_init_hash}

## Candidate 1

Learning rate:

0.0001

Step:

1000

Best source-selection macro case Dice so far:

{results[0]['best_macro_case_dice']:.8f}

Best step so far:

{results[0]['best_step']}

Durable release:

{results[0]['release_url']}

## Candidate 2

Learning rate:

0.0003

Step:

1000

Best source-selection macro case Dice so far:

{results[1]['best_macro_case_dice']:.8f}

Best step so far:

{results[1]['best_step']}

Durable release:

{results[1]['release_url']}

## Selection rule

No winner is selected at step 1000.

Both predeclared candidates must complete the same full declared training
budget or early-stopping rule before the learning rate is frozen.

## Isolation

Calibration accessed:

NO

Target / MedSeg accessed:

NO

Target lock:

ACTIVE

## Next

Run 20-update recovery audits for both candidates, then continue both under
the identical frozen protocol to early stopping or step 5000.
"""

    atomic_text(STATE_PATH, text)
    atomic_text(HANDOFF_STATE, text)

    git_sync("Complete Notebook 05B U-Net LR pilot step1000 for both candidates")

    subprocess.run(["git", "fetch", "origin", "main"], cwd=str(ROOT), check=True)
    local_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(ROOT),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    remote_head = subprocess.run(
        ["git", "rev-parse", "origin/main"],
        cwd=str(ROOT),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    assert local_head == remote_head

    print()
    print("=" * 110)
    print("EVICT NOTEBOOK 05B — BOTH U-NET LR CANDIDATES — STEP1000 DURABLE PASS")
    print("=" * 110)
    print(f"Identical initialization : PASS")
    print(f"Initialization SHA-256   : {shared_init_hash}")
    print()
    for result in results:
        print(
            f"{result['candidate_name']:6s} | "
            f"LR={result['learning_rate']:.1e} | "
            f"best Dice={result['best_macro_case_dice']:.8f} | "
            f"best step={result['best_step']} | "
            f"release={result['release_url']}"
        )
    print()
    print("Winner selected           : NO")
    print("Reason                    : both candidates must finish the declared budget")
    print("Calibration accessed      : NO")
    print("Target / MedSeg accessed  : NO")
    print("Target lock               : ACTIVE")
    print(f"GitHub HEAD               : {local_head}")
    print()
    print("NEXT: recovery audit 1000 -> 1020 for both candidates, then full continuation.")


if __name__ == "__main__":
    main()
