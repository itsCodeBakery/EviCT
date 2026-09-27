from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torchvision.transforms.functional as TVF
from torchvision.transforms import InterpolationMode


# ============================================================
# EviCT Notebook 04D — supervised SegFormer-B1, seed 2026
#
# Stage:
#   Fresh deterministic training from ImageNet MiT-B1
#   through the first durable boundary at optimizer step 1000.
#
# Important:
#   - This is NOT initialized from seed 17 or seed 42.
#   - Frozen patient split remains split_seed=17.
#   - Only RNG/sampling/augmentation training seed changes.
#   - Calibration and target/MedSeg remain locked.
# ============================================================


ROOT = Path("/kaggle/working/EviCT")
WORK = Path("/kaggle/working")

SEED = 2026
RUN_ID = "segformer_b1_supervised_seed2026_fp32"
MAX_UPDATES = 5000
SEGMENT_STOP = 1000
WARMUP_UPDATES = 200
VALIDATE_EVERY = 250
RECOVERY_EVERY = 50
PATIENCE_LIMIT = 8
MICRO_BATCH = 4
ACCUM = 4
IMAGES_PER_UPDATE = 16
VAL_BATCH = 8
ENCODER_LR = 1e-4
DECODER_LR = 3e-4
WEIGHT_DECAY = 0.01
POSITIVE_PROB = 0.50
THRESHOLD = 0.50

EXPECTED_CACHE_SHA = "d88e786cd467816d6cb446333385918016946a83cdd3c258b3011f458fbd3fc4"
CACHE_URL = (
    "https://github.com/itsCodeBakery/EviCT/releases/download/"
    "evict-project-durable-backup-20260927/EviCT_Notebook03_Cache.tar"
)

CONFIG_PATH = ROOT / "config/notebook04d_seed2026_config.json"
BASELINE_CONFIG_PATH = ROOT / "config/segformer_b1_baseline.json"
SPLITS_PATH = ROOT / "manifests/splits.csv"
TRAIN_MANIFEST = ROOT / "manifests/slice_loaders/seed_2026/b100/labeled_slices.csv"
SELECTION_MANIFEST = ROOT / "manifests/source_selection_slices.csv"

MODEL_CODE = ROOT / "src/evict/models/segformer_b1.py"
METRICS_CODE = ROOT / "src/evict/metrics.py"

AUDIT_DIR = ROOT / "artifacts/audit"
LARGE_DIR = ROOT / "artifacts/large/notebook04d/seed_2026"
PRED_DIR = ROOT / "predictions/notebook04d/seed_2026"
BEST_LOGITS_DIR = PRED_DIR / "best_selection_logits"
BEST_LOGITS_BACKUP = PRED_DIR / "_best_selection_logits_previous"

TRAIN_LOG = AUDIT_DIR / "notebook04d_seed2026_train_log.csv"
SELECTION_LOG = AUDIT_DIR / "notebook04d_seed2026_selection_metrics.csv"
CASE_LOG = AUDIT_DIR / "notebook04d_seed2026_selection_case_metrics.csv"
BEST_LOGITS_MANIFEST = AUDIT_DIR / "notebook04d_seed2026_best_logits_manifest.json"
RUNTIME_STATE = AUDIT_DIR / "notebook04d_seed2026_runtime_state.json"
START_AUDIT = AUDIT_DIR / "notebook04d_seed2026_start_audit.json"

LAST_PT = LARGE_DIR / "last.pt"
LAST_GOOD_PT = LARGE_DIR / "last_known_good.pt"
RECOVERY_PT = LARGE_DIR / "recovery.pt"
BEST_PT = LARGE_DIR / "best.pt"
BEST_PREVIOUS_PT = LARGE_DIR / "best_previous.pt"
EMERGENCY_PT = LARGE_DIR / "emergency_preupdate.pt"

STATE_PATH = ROOT / "STATE.md"
HANDOFF_STATE = ROOT / "handoff/STATE.md"
GIT_SYNC = ROOT / "scripts/git_sync.py"
HF_CACHE = WORK / "hf_cache"

for d in [AUDIT_DIR, LARGE_DIR, PRED_DIR, ROOT / "handoff", HF_CACHE]:
    d.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def download_large(url: str, dest: Path, expected_sha: str) -> None:
    dest = Path(dest)
    if dest.exists() and sha256_file(dest) == expected_sha:
        print(f"✓ Existing verified download: {dest.name}")
        return

    partial = dest.with_suffix(dest.suffix + ".part")
    if partial.exists():
        partial.unlink()

    print(f"Downloading              : {dest.name}")
    print(f"Source                   : {url}")

    with requests.get(url, stream=True, timeout=(60, 7200), allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        last_report = 0
        with partial.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if done - last_report >= 256 * 1024 * 1024:
                    if total:
                        print(f"  {done / 1024**2:.0f}/{total / 1024**2:.0f} MiB")
                    else:
                        print(f"  {done / 1024**2:.0f} MiB")
                    last_report = done

    actual = sha256_file(partial)
    assert actual == expected_sha, (
        f"SHA mismatch for {dest.name}\nExpected: {expected_sha}\nActual:   {actual}"
    )
    os.replace(partial, dest)
    print(f"✓ Download SHA verified  : {dest.name}")


def safe_extract_tar(tar_path: Path, destination: Path) -> None:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            assert str(target).startswith(str(root)), f"Unsafe TAR member: {member.name}"
        tar.extractall(destination)


def ensure_cache() -> None:
    expected_probe = ROOT / "cache/segdb2/images/coronacases_001.npy"
    if expected_probe.exists():
        print("✓ Notebook-03 cache already present")
        return

    tar_path = WORK / "EviCT_Notebook03_Cache.tar"
    download_large(CACHE_URL, tar_path, EXPECTED_CACHE_SHA)

    print("Extracting Notebook-03 cache...")
    with tarfile.open(tar_path, "r") as tar:
        names = tar.getnames()

    if any(name.startswith("EviCT/cache/") for name in names):
        safe_extract_tar(tar_path, WORK)
    else:
        safe_extract_tar(tar_path, ROOT)

    if not expected_probe.exists():
        candidates = list(WORK.rglob("cache/segdb2/images/coronacases_001.npy"))
        assert candidates, "Cache extraction finished but expected cache layout was not found."
        discovered_cache = candidates[0].parents[2]
        target_cache = ROOT / "cache"
        if discovered_cache != target_cache:
            if target_cache.exists():
                shutil.rmtree(target_cache)
            shutil.move(str(discovered_cache), str(target_cache))

    assert expected_probe.exists()
    print("✓ Notebook-03 cache restored")


def git_sync(message: str) -> None:
    subprocess.run(
        ["git", "config", "user.name", "itsCodeBakery"],
        cwd=str(ROOT),
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "itsCodeBakery@users.noreply.github.com"],
        cwd=str(ROOT),
        check=True,
    )
    result = subprocess.run(
        [sys.executable, str(GIT_SYNC), message],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Git synchronization failed.")


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state_safe(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])

    cpu = state["torch_cpu"]
    if not torch.is_tensor(cpu):
        cpu = torch.tensor(cpu, dtype=torch.uint8)
    cpu = cpu.detach().cpu().to(torch.uint8).contiguous()
    torch.set_rng_state(cpu)

    cuda_states = state["torch_cuda"]
    assert len(cuda_states) >= 1
    for device_idx in range(min(torch.cuda.device_count(), len(cuda_states))):
        s = cuda_states[device_idx]
        if not torch.is_tensor(s):
            s = torch.tensor(s, dtype=torch.uint8)
        s = s.detach().cpu().to(torch.uint8).contiguous()
        torch.cuda.set_rng_state(s, device=device_idx)


TRAIN_FIELDS = [
    "step", "precision", "loss", "dice_loss", "bce_loss", "gradient_norm",
    "encoder_lr_used", "decoder_lr_used", "encoder_lr_next",
    "decoder_lr_next", "images_seen",
]
SELECTION_FIELDS = [
    "step", "selection_loss", "macro_case_dice", "macro_case_iou",
    "macro_case_sensitivity", "macro_case_specificity", "macro_slice_dice",
    "pooled_dice", "pooled_iou", "n_cases", "n_slices", "threshold",
    "precision", "encoder_lr_next", "decoder_lr_next", "is_best",
    "best_score", "best_step", "patience_count",
]
CASE_FIELDS = [
    "step", "case_id", "n_slices", "tp", "tn", "fp", "fn",
    "valid_pixels", "dice", "iou", "sensitivity", "specificity",
    "empty_reference", "empty_prediction",
]


def assert_csv_schema(path: Path, fields: list[str]) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f), None)
    assert header == fields, f"CSV schema mismatch: {path}"


def append_csv(path: Path, row: dict, fields: list[str]) -> None:
    assert list(row.keys()) == fields
    exists = path.exists()
    if exists:
        assert_csv_schema(path, fields)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def truncate_unique_log(path: Path, max_step: int) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    if len(df) == 0:
        return
    df = (
        df[df["step"] <= max_step]
        .drop_duplicates(subset=["step"], keep="last")
        .sort_values("step")
    )
    df.to_csv(path, index=False)


def main() -> None:
    assert ROOT.exists(), f"Repository must be cloned first: {ROOT}"

    for path in [
        CONFIG_PATH, BASELINE_CONFIG_PATH, SPLITS_PATH, TRAIN_MANIFEST,
        SELECTION_MANIFEST, MODEL_CODE, METRICS_CODE, GIT_SYNC,
    ]:
        assert path.exists(), f"Missing required project file: {path}"

    print("=" * 100)
    print("EVICT NOTEBOOK 04D — SEED 2026 — FIRST DURABLE SEGMENT")
    print("=" * 100)

    ensure_cache()

    run_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    baseline_config = json.loads(BASELINE_CONFIG_PATH.read_text(encoding="utf-8"))

    assert run_config["seed"] == 2026
    assert run_config["split_seed"] == 17
    assert run_config["precision"] == "FP32"
    assert run_config["target_access_allowed"] is False

    run_config_hash = stable_json_hash(run_config)
    split_hash = stable_json_hash({
        "splits_csv": sha256_file(SPLITS_PATH),
        "train_manifest": sha256_file(TRAIN_MANIFEST),
        "selection_manifest": sha256_file(SELECTION_MANIFEST),
    })
    model_code_hash = sha256_file(MODEL_CODE)
    metrics_code_hash = sha256_file(METRICS_CODE)

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
    assert set(train_df["seed"].astype(int)) == {2026}

    manifest_text = (
        " ".join(train_df.fillna("").astype(str).values.ravel())
        + " "
        + " ".join(selection_df.fillna("").astype(str).values.ravel())
    ).lower()
    for forbidden in ["medseg", "segdb1", "images_medseg", "masks_medseg"]:
        assert forbidden not in manifest_text

    assert torch.cuda.is_available(), "GPU required."
    device = torch.device("cuda:0")

    print(f"GPU                      : {torch.cuda.get_device_name(device)}")
    print("Training seed            : 2026")
    print("Frozen split seed        : 17")
    print("✓ Fitting cases           : 12")
    print("✓ Selection cases         : 4")
    print("✓ Calibration cases       : 4 — NOT ACCESSED")
    print("✓ Target / MedSeg accessed: NO")

    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    importlib.invalidate_caches()

    from evict.models.segformer_b1 import EviCTSegFormerB1, masked_supervised_loss
    from evict.metrics import (
        CaseMetricAccumulator,
        binary_segmentation_metrics,
        macro_case_summary,
        metrics_from_confusion,
    )
    from huggingface_hub import snapshot_download

    repo_id = baseline_config["checkpoint"]["repository"]
    revision = baseline_config["checkpoint"]["revision"]
    expected_weight_sha = baseline_config["checkpoint"]["weight_sha256"]

    try:
        snapshot = Path(snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(HF_CACHE),
            allow_patterns=[
                "config.json", "preprocessor_config.json",
                "model.safetensors", "pytorch_model.bin",
            ],
            local_files_only=True,
        ))
    except Exception:
        snapshot = Path(snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(HF_CACHE),
            allow_patterns=[
                "config.json", "preprocessor_config.json",
                "model.safetensors", "pytorch_model.bin",
            ],
        ))

    weights = sorted(snapshot.glob("*.safetensors")) + sorted(snapshot.glob("*.bin"))
    assert weights and sha256_file(weights[0]) == expected_weight_sha
    print("✓ Pinned ImageNet MiT-B1 weights verified")

    # --------------------------------------------------------
    # Set the seed BEFORE fresh decoder/head initialization.
    # If a checkpoint exists, its exact RNG state is restored
    # after loading, so these values do not alter continuation.
    # --------------------------------------------------------
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    mean = torch.tensor(
        baseline_config["input"]["normalization_mean"],
        dtype=torch.float32,
        device=device,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        baseline_config["input"]["normalization_std"],
        dtype=torch.float32,
        device=device,
    ).view(1, 3, 1, 1)

    def normalize(x):
        return (x.repeat(1, 3, 1, 1) - mean) / std

    class FittingCaseStore:
        def __init__(self, df):
            self.case_ids = sorted(df["case_id"].astype(str).unique().tolist())
            self.cases = {}
            for case_id in self.case_ids:
                rows = df[df["case_id"].astype(str) == case_id].sort_values(
                    "image_array_index"
                )
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
        def __init__(self, store, seed):
            self.store = store
            self.case_ids = list(store.case_ids)
            self.rng = np.random.default_rng(seed)
            self.samples_seen = 0

        def sample(self, batch_size):
            images, targets, valids = [], [], []
            for _ in range(batch_size):
                case_id = str(self.rng.choice(self.case_ids))
                case = self.store.cases[case_id]
                positive = (
                    len(case["positive_z"]) > 0
                    and self.rng.random() < POSITIVE_PROB
                )
                z = int(self.rng.choice(
                    case["positive_z"] if positive else case["all_z"]
                ))
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

        def state_dict(self):
            return {
                "bit_generator_state": self.rng.bit_generator.state,
                "samples_seen": int(self.samples_seen),
            }

        def load_state_dict(self, state):
            self.rng.bit_generator.state = state["bit_generator_state"]
            self.samples_seen = int(state["samples_seen"])

    store = FittingCaseStore(train_df)
    sampler = PatientUniformSampler(store, SEED)

    def augment(images, targets, valids):
        out_i, out_t, out_v = [], [], []
        for idx in range(images.shape[0]):
            image, target, valid = images[idx], targets[idx], valids[idx]

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

    model = EviCTSegFormerB1(
        checkpoint_path=str(snapshot),
        decoder_dim=256,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": ENCODER_LR,
            },
            {
                "params": (
                    list(model.decoder.parameters())
                    + list(model.visual_head.parameters())
                ),
                "lr": DECODER_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    def lr_multiplier(step):
        if step < WARMUP_UPDATES:
            return float(step + 1) / float(WARMUP_UPDATES)

        progress = (
            (step - WARMUP_UPDATES)
            / float(MAX_UPDATES - WARMUP_UPDATES)
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lr_multiplier, lr_multiplier],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    def checkpoint_meta(path):
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        try:
            assert payload["seed"] == 2026
            assert payload["run_config_hash"] == run_config_hash
            assert payload["split_hash"] == split_hash
            assert payload["model_code_hash"] == model_code_hash
            assert payload["pretrained_weight_sha256"] == expected_weight_sha
            assert payload["precision"] == "FP32"
            return {
                "path": path,
                "step": int(payload["global_step"]),
                "last_validation_step": int(payload["last_validation_step"]),
            }
        finally:
            del payload

    candidates = [
        checkpoint_meta(p)
        for p in [LAST_PT, RECOVERY_PT]
        if p.exists()
    ]

    if candidates:
        resume_info = max(candidates, key=lambda x: x["step"])
        resume_path = resume_info["path"]

        checkpoint = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )

        model.load_state_dict(checkpoint["student_state_dict"])
        model.to(device)

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for opt_state in optimizer.state.values():
            for k, v in list(opt_state.items()):
                if torch.is_tensor(v):
                    opt_state[k] = v.to(device)

        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["amp_scaler_state_dict"])

        global_step = int(checkpoint["global_step"])
        best_score = float(checkpoint["best_score"])
        best_step = int(checkpoint["best_step"])
        patience_count = int(checkpoint["patience_count"])
        last_validation_step = int(checkpoint["last_validation_step"])
        images_seen = int(checkpoint["images_seen"])

        sampler.load_state_dict(checkpoint["sampler_state"])
        saved_rng = checkpoint["rng_state"]
        del checkpoint

        truncate_unique_log(TRAIN_LOG, global_step)
        truncate_unique_log(SELECTION_LOG, last_validation_step)

        if CASE_LOG.exists():
            df = pd.read_csv(CASE_LOG)
            df = df[df["step"] <= last_validation_step]
            df = df.drop_duplicates(
                subset=["step", "case_id"],
                keep="last",
            )
            df = df.sort_values(["step", "case_id"])
            df.to_csv(CASE_LOG, index=False)

        restore_rng_state_safe(saved_rng)

        print()
        print("=" * 100)
        print("LOCAL SEED-2026 CHECKPOINT RESTORE — PASS")
        print("=" * 100)
        print(f"Checkpoint               : {resume_path.name}")
        print(f"Global step              : {global_step}")
        print(f"Best macro case Dice     : {best_score:.8f}")
        print(f"Best step                : {best_step}")
        print(f"Patience                 : {patience_count}/8")
        print(f"Images seen              : {images_seen}")
        print("Optimizer                : RESTORED")
        print("Scheduler                : RESTORED")
        print("CPU RNG                  : RESTORED")
        print("CUDA RNG                 : RESTORED")
        print("Sampler RNG              : RESTORED")

    else:
        global_step = 0
        best_score = float("-inf")
        best_step = 0
        patience_count = 0
        last_validation_step = 0
        images_seen = 0

        START_AUDIT.write_text(
            json.dumps(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "seed": 2026,
                    "split_seed": 17,
                    "initialization": "fresh_pinned_imagenet_mit_b1",
                    "initialized_from_seed17": False,
                    "initialized_from_seed42": False,
                    "precision": "FP32",
                    "run_config_hash": run_config_hash,
                    "split_hash": split_hash,
                    "model_code_hash": model_code_hash,
                    "metrics_code_hash": metrics_code_hash,
                    "pretrained_weight_sha256": expected_weight_sha,
                    "calibration_accessed": False,
                    "target_accessed": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print()
        print("=" * 100)
        print("FRESH SEED-2026 INITIALIZATION — PASS")
        print("=" * 100)
        print("Initialization            : PINNED IMAGENET MiT-B1")
        print("Seed-17 checkpoint        : NOT USED")
        print("Seed-42 checkpoint        : NOT USED")
        print("Optimizer step            : 0")
        print("Training RNG seed         : 2026")
        print("Sampler RNG seed          : 2026")

    assert 0 <= global_step <= SEGMENT_STOP

    def build_checkpoint():
        return {
            "format_version": 2,
            "run_id": RUN_ID,
            "seed": SEED,
            "precision": "FP32",
            "amp_enabled": False,
            "run_config_hash": run_config_hash,
            "split_hash": split_hash,
            "model_code_hash": model_code_hash,
            "metrics_code_hash": metrics_code_hash,
            "pretrained_weight_sha256": expected_weight_sha,
            "student_state_dict": model.state_dict(),
            "ema_state_dict": None,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "amp_scaler_state_dict": scaler.state_dict(),
            "global_step": int(global_step),
            "best_score": float(best_score),
            "best_step": int(best_step),
            "patience_count": int(patience_count),
            "last_validation_step": int(last_validation_step),
            "images_seen": int(images_seen),
            "rng_state": capture_rng_state(),
            "sampler_state": sampler.state_dict(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

    def atomic_save(dest, preserve_previous=None):
        temp = dest.with_suffix(dest.suffix + ".tmp")

        if temp.exists():
            temp.unlink()

        torch.save(build_checkpoint(), temp)

        test = torch.load(
            temp,
            map_location="cpu",
            weights_only=False,
        )

        assert int(test["global_step"]) == global_step
        assert test["seed"] == 2026
        assert test["rng_state"]["torch_cpu"].device.type == "cpu"
        assert test["rng_state"]["torch_cpu"].dtype == torch.uint8

        del test

        if preserve_previous is not None and dest.exists():
            shutil.copy2(dest, preserve_previous)

        os.replace(temp, dest)

    def gradient_audit():
        finite = True
        squared = 0.0
        count = 0

        for p in model.parameters():
            if p.grad is None:
                continue

            count += 1
            g = p.grad.detach().float()

            if not torch.isfinite(g).all():
                finite = False
                continue

            n = float(g.norm().item())
            squared += n * n

        return (
            finite,
            math.sqrt(squared) if finite else float("inf"),
            count,
        )

    def train_one_update():
        nonlocal global_step, images_seen

        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_dice = 0.0
        total_bce = 0.0

        for _ in range(ACCUM):
            x_np, y_np, v_np = sampler.sample(MICRO_BATCH)

            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            v = torch.from_numpy(v_np).to(device)

            x, y, v = augment(x, y, v)

            output = model(normalize(x))

            losses = masked_supervised_loss(
                logits=output["logits"].float(),
                targets=y,
                valid_mask=v,
            )

            loss = losses["loss"]

            if not torch.isfinite(loss).item():
                atomic_save(EMERGENCY_PT)
                raise RuntimeError(
                    f"Non-finite loss before step {global_step + 1}"
                )

            (loss / ACCUM).backward()

            total_loss += float(loss.detach().cpu()) / ACCUM
            total_dice += (
                float(losses["dice_loss"].detach().cpu()) / ACCUM
            )
            total_bce += (
                float(losses["bce_loss"].detach().cpu()) / ACCUM
            )

        finite, grad_norm, grad_count = gradient_audit()

        if not finite or grad_count == 0:
            atomic_save(EMERGENCY_PT)
            raise RuntimeError(
                f"Non-finite gradient before step {global_step + 1}"
            )

        encoder_lr_used = float(
            optimizer.param_groups[0]["lr"]
        )
        decoder_lr_used = float(
            optimizer.param_groups[1]["lr"]
        )

        optimizer.step()
        scheduler.step()

        global_step += 1
        images_seen += IMAGES_PER_UPDATE

        row = {
            "step": global_step,
            "precision": "FP32",
            "loss": total_loss,
            "dice_loss": total_dice,
            "bce_loss": total_bce,
            "gradient_norm": grad_norm,
            "encoder_lr_used": encoder_lr_used,
            "decoder_lr_used": decoder_lr_used,
            "encoder_lr_next": float(
                optimizer.param_groups[0]["lr"]
            ),
            "decoder_lr_next": float(
                optimizer.param_groups[1]["lr"]
            ),
            "images_seen": images_seen,
        }

        append_csv(
            TRAIN_LOG,
            row,
            TRAIN_FIELDS,
        )

        return row

    def install_best_logits(tmp_dir: Path, step: int):
        if BEST_LOGITS_BACKUP.exists():
            shutil.rmtree(BEST_LOGITS_BACKUP)

        if BEST_LOGITS_DIR.exists():
            os.replace(
                BEST_LOGITS_DIR,
                BEST_LOGITS_BACKUP,
            )

        os.replace(
            tmp_dir,
            BEST_LOGITS_DIR,
        )

        files = sorted(
            BEST_LOGITS_DIR.glob("*.npy")
        )

        assert len(files) == 4

        manifest = {
            "seed": 2026,
            "step": int(step),
            "dtype": "float32",
            "space": "padded_336x336_raw_logits",
            "sigmoid_applied": False,
            "threshold_applied": False,
            "cases": [],
        }

        for file in files:
            arr = np.load(
                file,
                mmap_mode="r",
            )

            manifest["cases"].append(
                {
                    "case_id": file.stem,
                    "shape": list(arr.shape),
                    "sha256": sha256_file(file),
                }
            )

        BEST_LOGITS_MANIFEST.write_text(
            json.dumps(
                manifest,
                indent=2,
            ),
            encoding="utf-8",
        )

        if BEST_LOGITS_BACKUP.exists():
            shutil.rmtree(BEST_LOGITS_BACKUP)

    def run_validation(step: int):
        model.eval()

        for abandoned in PRED_DIR.glob(
            "_selection_step_*_tmp"
        ):
            if abandoned.is_dir():
                shutil.rmtree(abandoned)

        tmp_dir = (
            PRED_DIR
            / f"_selection_step_{step:05d}_tmp"
        )

        tmp_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        case_records = []
        slice_dices = []

        tp = tn = fp = fn = 0
        loss_sum = 0.0
        n_slices = 0

        with torch.no_grad():
            for case_id in sorted(selection_cases):
                rows = selection_df[
                    selection_df["case_id"].astype(str)
                    == case_id
                ].sort_values("image_array_index")

                first = rows.iloc[0]

                images_volume = np.load(
                    str(first["image_path"]),
                    mmap_mode="r",
                )

                target_volume = np.load(
                    str(first["infection_cache_path"]),
                    mmap_mode="r",
                )

                valid_case = np.load(
                    str(first["valid_mask_path"]),
                    mmap_mode="r",
                )

                z_indices = (
                    rows["image_array_index"]
                    .astype(int)
                    .tolist()
                )

                assert z_indices == list(
                    range(images_volume.shape[0])
                )

                acc = CaseMetricAccumulator(case_id)

                logit_path = (
                    tmp_dir
                    / f"{case_id}.npy"
                )

                logit_mm = np.lib.format.open_memmap(
                    logit_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(
                        images_volume.shape[0],
                        336,
                        336,
                    ),
                )

                for start in range(
                    0,
                    len(z_indices),
                    VAL_BATCH,
                ):
                    batch_z = z_indices[
                        start:
                        start + VAL_BATCH
                    ]

                    x_np = np.asarray(
                        images_volume[batch_z],
                        dtype=np.float32,
                    )[:, None, :, :]

                    y_np = np.asarray(
                        target_volume[batch_z],
                        dtype=np.float32,
                    )[:, None, :, :]

                    v_np = np.repeat(
                        np.asarray(
                            valid_case,
                            dtype=np.float32,
                        )[None, None, :, :],
                        repeats=len(batch_z),
                        axis=0,
                    )

                    x = torch.from_numpy(
                        x_np
                    ).to(device)

                    y = torch.from_numpy(
                        y_np
                    ).to(device)

                    v = torch.from_numpy(
                        v_np
                    ).to(device)

                    logits = model(
                        normalize(x)
                    )["logits"].float()

                    losses = masked_supervised_loss(
                        logits=logits,
                        targets=y,
                        valid_mask=v,
                    )

                    assert torch.isfinite(
                        losses["loss"]
                    ).item()

                    batch_size = len(batch_z)

                    loss_sum += (
                        float(
                            losses["loss"].cpu()
                        )
                        * batch_size
                    )

                    n_slices += batch_size

                    pred = (
                        torch.sigmoid(logits)
                        >= THRESHOLD
                    )

                    logits_cpu = (
                        logits[:, 0]
                        .cpu()
                        .numpy()
                    )

                    target_cpu = (
                        y[:, 0]
                        .cpu()
                        .numpy()
                    )

                    pred_cpu = (
                        pred[:, 0]
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )

                    valid_cpu = (
                        v[:, 0]
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )

                    for local_idx, z in enumerate(batch_z):
                        logit_mm[z] = logits_cpu[local_idx]

                        acc.update(
                            target_cpu[local_idx],
                            pred_cpu[local_idx],
                            valid_mask=valid_cpu[local_idx],
                        )

                        m = binary_segmentation_metrics(
                            target_cpu[local_idx],
                            pred_cpu[local_idx],
                            valid_mask=valid_cpu[local_idx],
                        )

                        slice_dices.append(
                            float(m["dice"])
                        )

                logit_mm.flush()
                del logit_mm

                record = acc.compute()
                case_records.append(record)

                tp += int(record["tp"])
                tn += int(record["tn"])
                fp += int(record["fp"])
                fn += int(record["fn"])

        macro = macro_case_summary(
            case_records
        )

        pooled = metrics_from_confusion(
            tp,
            tn,
            fp,
            fn,
        )

        summary = {
            "step": step,
            "selection_loss": float(
                loss_sum / n_slices
            ),
            "macro_case_dice": float(
                macro["macro_dice"]
            ),
            "macro_case_iou": float(
                macro["macro_iou"]
            ),
            "macro_case_sensitivity": float(
                macro["macro_sensitivity"]
            ),
            "macro_case_specificity": float(
                macro["macro_specificity"]
            ),
            "macro_slice_dice": float(
                np.mean(
                    np.asarray(
                        slice_dices,
                        dtype=np.float64,
                    )
                )
            ),
            "pooled_dice": float(
                pooled["dice"]
            ),
            "pooled_iou": float(
                pooled["iou"]
            ),
            "n_cases": 4,
            "n_slices": int(n_slices),
            "threshold": THRESHOLD,
        }

        model.train()

        return (
            summary,
            case_records,
            tmp_dir,
        )

    def write_state(status: str):
        display_best = (
            best_score
            if math.isfinite(best_score)
            else 0.0
        )

        text = f"""# EviCT Execution State

## Current stage

{status}

## Timestamp

{datetime.now(timezone.utc).isoformat()}

## Notebook 04D

Model:

Supervised SegFormer MiT-B1

Training seed:

2026

Frozen split seed:

17

Precision:

FP32

Current optimizer step:

{global_step} / 5000

Current durable segment:

0 -> 1000

Images seen:

{images_seen}

Last source-selection validation:

{last_validation_step}

Best source-selection macro case Dice:

{display_best:.8f}

Best step:

{best_step}

Patience:

{patience_count} / 8

## Initialization

Pinned ImageNet MiT-B1:

YES

Seed-17 checkpoint used:

NO

Seed-42 checkpoint used:

NO

## Isolation

Training:

12 frozen fitting cases only

Selection:

4 frozen complete source-selection cases only

Calibration accessed:

NO

Target / MedSeg accessed:

NO

## Target lock

ACTIVE
"""

        STATE_PATH.write_text(
            text,
            encoding="utf-8",
        )

        HANDOFF_STATE.write_text(
            text,
            encoding="utf-8",
        )

    def do_validation():
        nonlocal (
            best_score,
            best_step,
            patience_count,
            last_validation_step,
        )

        print()
        print("=" * 100)
        print(
            f"SOURCE-SELECTION VALIDATION — STEP {global_step}"
        )
        print("=" * 100)

        summary, records, tmp_logits = (
            run_validation(global_step)
        )

        score = float(
            summary["macro_case_dice"]
        )

        improved = (
            not math.isfinite(best_score)
            or score > best_score + 1e-8
        )

        if improved:
            best_score = score
            best_step = global_step
            patience_count = 0
            last_validation_step = global_step

            install_best_logits(
                tmp_logits,
                global_step,
            )

            atomic_save(
                BEST_PT,
                preserve_previous=(
                    BEST_PREVIOUS_PT
                    if BEST_PT.exists()
                    else None
                ),
            )

            print(
                "✓ NEW BEST SOURCE-SELECTION CHECKPOINT"
            )

        else:
            patience_count += 1
            shutil.rmtree(tmp_logits)

        last_validation_step = global_step

        selection_row = {
            "step": global_step,
            "selection_loss": summary["selection_loss"],
            "macro_case_dice": summary["macro_case_dice"],
            "macro_case_iou": summary["macro_case_iou"],
            "macro_case_sensitivity": summary["macro_case_sensitivity"],
            "macro_case_specificity": summary["macro_case_specificity"],
            "macro_slice_dice": summary["macro_slice_dice"],
            "pooled_dice": summary["pooled_dice"],
            "pooled_iou": summary["pooled_iou"],
            "n_cases": summary["n_cases"],
            "n_slices": summary["n_slices"],
            "threshold": summary["threshold"],
            "precision": "FP32",
            "encoder_lr_next": float(
                optimizer.param_groups[0]["lr"]
            ),
            "decoder_lr_next": float(
                optimizer.param_groups[1]["lr"]
            ),
            "is_best": bool(improved),
            "best_score": float(best_score),
            "best_step": int(best_step),
            "patience_count": int(patience_count),
        }

        append_csv(
            SELECTION_LOG,
            selection_row,
            SELECTION_FIELDS,
        )

        for record in records:
            case_row = {
                "step": global_step,
                "case_id": record["case_id"],
                "n_slices": record["n_slices"],
                "tp": record["tp"],
                "tn": record["tn"],
                "fp": record["fp"],
                "fn": record["fn"],
                "valid_pixels": record["valid_pixels"],
                "dice": record["dice"],
                "iou": record["iou"],
                "sensitivity": record["sensitivity"],
                "specificity": record["specificity"],
                "empty_reference": record["empty_reference"],
                "empty_prediction": record["empty_prediction"],
            }

            append_csv(
                CASE_LOG,
                case_row,
                CASE_FIELDS,
            )

        atomic_save(
            LAST_PT,
            preserve_previous=(
                LAST_GOOD_PT
                if LAST_PT.exists()
                else None
            ),
        )

        RUNTIME_STATE.write_text(
            json.dumps(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "seed": 2026,
                    "global_step": global_step,
                    "last_validation_step": last_validation_step,
                    "best_score": best_score,
                    "best_step": best_step,
                    "patience_count": patience_count,
                    "images_seen": images_seen,
                    "precision": "FP32",
                    "target_accessed": False,
                    "calibration_accessed": False,
                    "last_checkpoint_sha256": sha256_file(LAST_PT),
                    "best_checkpoint_sha256": sha256_file(BEST_PT),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"Selection loss           : "
            f"{summary['selection_loss']:.6f}"
        )
        print(
            f"Macro case Dice          : "
            f"{summary['macro_case_dice']:.6f}"
        )
        print(
            f"Macro case IoU           : "
            f"{summary['macro_case_iou']:.6f}"
        )
        print(
            f"Macro slice Dice         : "
            f"{summary['macro_slice_dice']:.6f}"
        )
        print(
            f"Pooled Dice              : "
            f"{summary['pooled_dice']:.6f}"
        )
        print(
            f"Patience                 : "
            f"{patience_count}/8"
        )

        write_state(
            f"NOTEBOOK_04D_SEED2026_RUNNING_STEP_{global_step}"
        )

        git_sync(
            f"Notebook 04D seed2026 FP32 validation step {global_step}"
        )

    if (
        global_step > 0
        and global_step % VALIDATE_EVERY == 0
        and last_validation_step < global_step
    ):
        do_validation()

    print()
    print("=" * 100)
    print("SEED-2026 TRAINING — FIRST DURABLE SEGMENT")
    print("=" * 100)
    print(f"Starting optimizer step  : {global_step}")
    print(f"Segment stop             : {SEGMENT_STOP}")
    print("Fresh from seed17/42     : YES — neither checkpoint used")

    while global_step < SEGMENT_STOP:
        row = train_one_update()

        if global_step % 25 == 0:
            print(
                f"step={global_step:4d} | "
                f"loss={row['loss']:.5f} | "
                f"dice={row['dice_loss']:.5f} | "
                f"bce={row['bce_loss']:.5f} | "
                f"grad={row['gradient_norm']:.4f} | "
                f"patience={patience_count}/8"
            )

        if (
            global_step % RECOVERY_EVERY == 0
            and global_step % VALIDATE_EVERY != 0
        ):
            atomic_save(RECOVERY_PT)

            print(
                f"  ✓ recovery.pt saved at step {global_step}"
            )

        if global_step % VALIDATE_EVERY == 0:
            atomic_save(
                LAST_PT,
                preserve_previous=(
                    LAST_GOOD_PT
                    if LAST_PT.exists()
                    else None
                ),
            )

            do_validation()

    assert global_step == SEGMENT_STOP
    assert last_validation_step == SEGMENT_STOP
    assert images_seen == SEGMENT_STOP * IMAGES_PER_UPDATE
    assert BEST_PT.exists()
    assert LAST_PT.exists()
    assert BEST_LOGITS_DIR.exists()

    atomic_save(
        LAST_PT,
        preserve_previous=(
            LAST_GOOD_PT
            if LAST_PT.exists()
            else None
        ),
    )

    archive = (
        WORK
        / "EviCT_Notebook04D_seed2026_step1000_Recovery.tar"
    )

    if archive.exists():
        archive.unlink()

    with tarfile.open(
        archive,
        "w",
    ) as tar:
        for path in [
            LAST_PT,
            LAST_GOOD_PT,
            RECOVERY_PT,
            BEST_PT,
            CONFIG_PATH,
            TRAIN_LOG,
            SELECTION_LOG,
            CASE_LOG,
            BEST_LOGITS_MANIFEST,
            RUNTIME_STATE,
            START_AUDIT,
            STATE_PATH,
            MODEL_CODE,
            METRICS_CODE,
            SPLITS_PATH,
            TRAIN_MANIFEST,
            SELECTION_MANIFEST,
        ]:
            if path.exists():
                tar.add(
                    path,
                    arcname=str(
                        path.relative_to(ROOT)
                    ),
                )

        if BEST_LOGITS_DIR.exists():
            tar.add(
                BEST_LOGITS_DIR,
                arcname=str(
                    BEST_LOGITS_DIR.relative_to(ROOT)
                ),
            )

    archive_sha = sha256_file(
        archive
    )

    sha_path = Path(
        str(archive)
        + ".sha256"
    )

    sha_path.write_text(
        archive_sha + "\n",
        encoding="utf-8",
    )

    write_state(
        "NOTEBOOK_04D_SEED2026_STEP1000_AWAITING_DURABLE_RELEASE_BACKUP"
    )

    git_sync(
        "Complete Notebook 04D seed2026 durable segment through step 1000"
    )

    print()
    print("=" * 100)
    print("EVICT NOTEBOOK 04D — SEED 2026 — STEP 1000 PASS")
    print("=" * 100)

    print(f"Optimizer step           : {global_step}")
    print(f"Last validation          : {last_validation_step}")
    print(f"Best macro case Dice     : {best_score:.8f}")
    print(f"Best checkpoint step     : {best_step}")
    print(f"Patience                 : {patience_count}/8")
    print(f"Images seen              : {images_seen}")
    print(f"Recovery archive         : {archive}")
    print(f"Archive SHA-256          : {archive_sha}")
    print("Initialization           : FRESH PINNED IMAGENET MiT-B1")
    print("Seed-17 checkpoint used  : NO")
    print("Seed-42 checkpoint used  : NO")
    print("Calibration accessed     : NO")
    print("Target / MedSeg accessed : NO")
    print("Git metadata             : SYNCHRONIZED")
    print()
    print("NEXT: upload TAR + SHA to GitHub Release before continuation.")


if __name__ == "__main__":
    main()
