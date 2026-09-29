from __future__ import annotations

import gc
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from kaggle_secrets import UserSecretsClient


ROOT = Path("/kaggle/working/EviCT")
WORK = Path("/kaggle/working")

BASE_RUNNER_PATH = ROOT / "scripts/nb05b_unet_lr_pilot_segment1.py"
CONFIG_PATH = ROOT / "config/notebook05_segct_clip_visual_adaptation.json"
PIN_PATH = ROOT / "config/notebook05_segct_clip_visual_backbone_pin.json"
MODEL_CODE = ROOT / "src/evict/models/segct_clip_visual_adaptation.py"
METRICS_CODE = ROOT / "src/evict/metrics.py"
SPLITS_PATH = ROOT / "manifests/splits.csv"
TRAIN_MANIFEST_17 = ROOT / "manifests/slice_loaders/seed_17/b100/labeled_slices.csv"
TRAIN_MANIFEST_42 = ROOT / "manifests/slice_loaders/seed_42/b100/labeled_slices.csv"
TRAIN_MANIFEST_2026 = ROOT / "manifests/slice_loaders/seed_2026/b100/labeled_slices.csv"
SELECTION_MANIFEST = ROOT / "manifests/source_selection_slices.csv"
STATE_PATH = ROOT / "STATE.md"
HANDOFF_STATE = ROOT / "handoff/STATE.md"
GIT_SYNC = ROOT / "scripts/git_sync.py"

AUDIT_DIR = ROOT / "artifacts/audit"
TABLE_DIR = ROOT / "tables"
REPORT_DIR = ROOT / "reports"
RUN_ROOT = ROOT / "artifacts/large/notebook05e"
PRED_ROOT = ROOT / "predictions/notebook05e"
FEATURE_ROOT = ROOT / "cache/segct_clip_visual_adaptation"
FEATURE_MANIFEST = AUDIT_DIR / "notebook05e_segct_clip_visual_feature_cache.json"
SMOKE_AUDIT = AUDIT_DIR / "notebook05e_segct_clip_visual_gpu_smoke.json"
FINAL_AGG_AUDIT = AUDIT_DIR / "notebook05e_segct_clip_visual_three_seed.json"

SEEDS = [17, 42, 2026]

MAX_UPDATES = 5000
WARMUP_UPDATES = 200
VALIDATE_EVERY = 250
RECOVERY_EVERY = 50
PATIENCE_LIMIT = 8
EFFECTIVE_BATCH = 16
THRESHOLD = 0.5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
POSITIVE_PROB = 0.5
REMOTE_MILESTONES = {1000, 2000, 3000, 4000}

LOW_HIDDEN_INDEX = 12
HIGH_HIDDEN_INDEX = 24
HIDDEN_SIZE = 1024
PATCH_GRID = 24

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

REPO_OWNER = "itsCodeBakery"
REPO_NAME = "EviCT"

# Notebook-03 durable source-cache restore contract.
# The release snapshot explicitly states that the TAR contains "cache/segdb2"
# and MUST be extracted into /kaggle/working/EviCT, not /kaggle/working.
CACHE_PROBE = ROOT / "cache/segdb2/images/coronacases_003.npy"
CACHE_ROOT = ROOT / "cache/segdb2"
MISPLACED_CACHE_ROOT = WORK / "cache/segdb2"
CACHE_TAR = WORK / "EviCT_Notebook03_Cache.tar"
CACHE_URL = (
    "https://github.com/itsCodeBakery/EviCT/releases/download/"
    "evict-project-durable-backup-20260927/EviCT_Notebook03_Cache.tar"
)
CACHE_SHA256 = "d88e786cd467816d6cb446333385918016946a83cdd3c258b3011f458fbd3fc4"
CACHE_HASH_MANIFEST = ROOT / "config/cache_manifest_hashes.json"


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "evict_nb05e_base",
        str(BASE_RUNNER_PATH),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import base runner: {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def verify_source_cache() -> None:
    """Verify all 20 cached source cases against the frozen per-case SHA-256 manifest."""
    assert CACHE_HASH_MANIFEST.exists(), f"Missing cache hash manifest: {CACHE_HASH_MANIFEST}"
    manifest = json.loads(CACHE_HASH_MANIFEST.read_text(encoding="utf-8"))
    rows = manifest.get("per_case_cache_sha256", [])
    assert len(rows) == 20, f"Expected 20 cache-hash rows, found {len(rows)}"

    for row in tqdm(
        rows,
        desc="verify Notebook-03 cache",
        unit="case",
        file=sys.stdout,
        leave=False,
    ):
        case_id = row["case_id"]
        expected = {
            ROOT / f"cache/segdb2/images/{case_id}.npy": row["image_sha256"],
            ROOT / f"cache/segdb2/gt_vault/infection/{case_id}.npy": row["infection_sha256"],
            ROOT / f"cache/segdb2/gt_vault/lung/{case_id}.npy": row["lung_sha256"],
            ROOT / f"cache/segdb2/valid/{case_id}.npy": row["valid_sha256"],
        }
        for path, sha in expected.items():
            assert path.exists(), f"Missing cache artifact: {path}"
            actual = base.sha256_file(path)
            assert actual == sha, (
                f"Cache SHA mismatch for {path}\n"
                f"Expected: {sha}\n"
                f"Actual:   {actual}"
            )

    assert CACHE_PROBE.exists()
    print("✓ Notebook-03 source cache : 20/20 CASES SHA-VERIFIED")


def ensure_source_cache() -> None:
    """
    Restore the frozen Notebook-03 cache using the release's documented restore rule.

    Recovery order:
      1. verified cache already at ROOT/cache/segdb2;
      2. repair the known misplaced extraction at WORK/cache/segdb2;
      3. reuse a verified local TAR if present;
      4. download the SHA-pinned release TAR.

    The archive contains cache/segdb2, so it is extracted into ROOT.
    """
    if CACHE_PROBE.exists():
        print("✓ Notebook-03 source cache : PRESENT")
        verify_source_cache()
        CACHE_TAR.unlink(missing_ok=True)
        return

    # Repair the earlier wrong extraction destination without redownloading 1.59 GB.
    if MISPLACED_CACHE_ROOT.exists():
        print("⚠ Found cache extracted at wrong location:")
        print(f"  {MISPLACED_CACHE_ROOT}")
        print("  Repairing to /kaggle/working/EviCT/cache/segdb2 ...")

        CACHE_ROOT.parent.mkdir(parents=True, exist_ok=True)
        if CACHE_ROOT.exists():
            shutil.rmtree(CACHE_ROOT)

        shutil.move(
            str(MISPLACED_CACHE_ROOT),
            str(CACHE_ROOT),
        )

        # Remove now-empty /kaggle/working/cache if possible.
        misplaced_parent = MISPLACED_CACHE_ROOT.parent
        try:
            misplaced_parent.rmdir()
        except OSError:
            pass

        assert CACHE_PROBE.exists(), (
            "Misplaced cache was moved, but the expected cache probe is still absent."
        )
        verify_source_cache()
        CACHE_TAR.unlink(missing_ok=True)
        print("✓ Misplaced Notebook-03 cache repaired without redownload.")
        return

    # If the previous wrapper downloaded the TAR but failed after extraction,
    # preserve bandwidth by verifying and reusing it.
    if CACHE_TAR.exists():
        print("Found local Notebook-03 cache TAR; verifying before reuse...")
        actual = base.sha256_file(CACHE_TAR)
        if actual != CACHE_SHA256:
            print("⚠ Local TAR SHA mismatch; deleting and redownloading.")
            CACHE_TAR.unlink(missing_ok=True)
        else:
            print("✓ Local cache TAR SHA      : VERIFIED")

    if not CACHE_TAR.exists():
        print("Notebook-03 source cache absent. Downloading verified release archive...")
        base.download_verified(
            CACHE_URL,
            CACHE_TAR,
            CACHE_SHA256,
        )

    print("Extracting Notebook-03 cache into the EviCT repository root...")
    # IMPORTANT: archive contains cache/segdb2; destination MUST be ROOT.
    base.safe_extract_tar(
        CACHE_TAR,
        ROOT,
    )

    assert CACHE_PROBE.exists(), (
        "Verified cache TAR extracted into EviCT, but expected cache probe is missing."
    )

    verify_source_cache()

    # Free ~1.59 GB once the extracted cache has been verified.
    CACHE_TAR.unlink(missing_ok=True)
    print("✓ Notebook-03 cache TAR    : REMOVED AFTER VERIFIED RESTORE")


def train_manifest(seed: int) -> Path:
    return ROOT / f"manifests/slice_loaders/seed_{seed}/b100/labeled_slices.csv"


def paths_for(seed: int) -> dict:
    run_dir = RUN_ROOT / f"seed_{seed}"
    pred_dir = PRED_ROOT / f"seed_{seed}"
    prefix = f"notebook05e_segct_clip_visual_seed{seed}"
    return {
        "run_dir": run_dir,
        "pred_dir": pred_dir,
        "best_pt": run_dir / "best.pt",
        "last_pt": run_dir / "last.pt",
        "recovery_pt": run_dir / "recovery.pt",
        "train_log": AUDIT_DIR / f"{prefix}_train_log.csv",
        "selection_log": AUDIT_DIR / f"{prefix}_selection_metrics.csv",
        "case_log": AUDIT_DIR / f"{prefix}_selection_case_metrics.csv",
        "best_logits_manifest": AUDIT_DIR / f"{prefix}_best_logits_manifest.json",
        "rolling_audit": AUDIT_DIR / f"{prefix}_rolling_durable.json",
        "final_audit": AUDIT_DIR / f"{prefix}_final_durable.json",
    }


def direct_release_url(tag: str, asset_name: str) -> str:
    return (
        f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/"
        f"{tag}/{asset_name}"
    )


def scheduler_multiplier(step: int) -> float:
    if step < WARMUP_UPDATES:
        return float(step + 1) / float(WARMUP_UPDATES)
    progress = (step - WARMUP_UPDATES) / float(MAX_UPDATES - WARMUP_UPDATES)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def ensure_backbone_pin(token: str | None) -> dict:
    if PIN_PATH.exists():
        pin = json.loads(PIN_PATH.read_text(encoding="utf-8"))
        assert pin["repository"] == "openai/clip-vit-large-patch14-336"
        assert pin["revision"]
        return pin

    from huggingface_hub import HfApi

    info = HfApi().model_info("openai/clip-vit-large-patch14-336")
    revision = str(info.sha)

    pin = {
        "timestamp_utc": base.utc_now(),
        "repository": "openai/clip-vit-large-patch14-336",
        "revision": revision,
        "resolved_once": True,
        "purpose": "Notebook05E explicit SegCT-CLIP visual-pathway adaptation",
    }

    base.atomic_text(
        PIN_PATH,
        json.dumps(pin, indent=2),
    )

    base.git_sync(
        "Pin CLIP ViT-L/14-336 revision for Notebook 05E adaptation"
    )

    return pin


def load_clip_vision(pin: dict, device: torch.device):
    from huggingface_hub import snapshot_download
    from transformers import CLIPVisionModel

    hf_cache = WORK / "hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)

    snapshot = Path(
        snapshot_download(
            repo_id=pin["repository"],
            revision=pin["revision"],
            cache_dir=str(hf_cache),
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "model.safetensors",
                "pytorch_model.bin",
            ],
        )
    )

    weight_files = (
        sorted(snapshot.glob("*.safetensors"))
        + sorted(snapshot.glob("*.bin"))
    )
    assert weight_files, "No CLIP weight file found in pinned snapshot."

    hashes = {
        path.name: base.sha256_file(path)
        for path in weight_files
    }

    vision = CLIPVisionModel.from_pretrained(
        str(snapshot),
        torch_dtype=torch.float16,
    )
    vision.eval()
    vision.requires_grad_(False)
    vision.to(device)

    return vision, snapshot, hashes


def source_manifests_equivalent():
    frames = {}
    ignore = {"seed", "budget_id"}

    for seed in SEEDS:
        df = pd.read_csv(train_manifest(seed))
        compare_cols = [
            c for c in df.columns
            if c not in ignore
        ]
        frames[seed] = (
            df[compare_cols]
            .sort_values(["case_id", "image_array_index"])
            .reset_index(drop=True)
        )

    assert frames[17].equals(frames[42])
    assert frames[17].equals(frames[2026])


def validate_source_contract():
    split_df = pd.read_csv(SPLITS_PATH)
    train_df = pd.read_csv(TRAIN_MANIFEST_17)
    selection_df = pd.read_csv(SELECTION_MANIFEST)

    fitting_cases = set(
        split_df.loc[
            split_df["source_split"] == "fitting",
            "case_id",
        ].astype(str)
    )
    selection_cases = set(
        split_df.loc[
            split_df["source_split"] == "selection",
            "case_id",
        ].astype(str)
    )
    calibration_cases = set(
        split_df.loc[
            split_df["source_split"] == "calibration",
            "case_id",
        ].astype(str)
    )

    assert len(fitting_cases) == 12
    assert len(selection_cases) == 4
    assert len(calibration_cases) == 4
    assert fitting_cases.isdisjoint(selection_cases)
    assert fitting_cases.isdisjoint(calibration_cases)
    assert selection_cases.isdisjoint(calibration_cases)
    assert set(train_df["case_id"].astype(str)) == fitting_cases
    assert set(selection_df["case_id"].astype(str)) == selection_cases

    text = (
        " ".join(train_df.fillna("").astype(str).values.ravel())
        + " "
        + " ".join(selection_df.fillna("").astype(str).values.ravel())
    ).lower()

    for forbidden in [
        "medseg",
        "segdb1",
        "images_medseg",
        "masks_medseg",
    ]:
        assert forbidden not in text

    return split_df, train_df, selection_df


def clip_normalize(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.repeat(1, 3, 1, 1)
    mean = torch.tensor(
        CLIP_MEAN,
        dtype=torch.float16,
        device=device,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        CLIP_STD,
        dtype=torch.float16,
        device=device,
    ).view(1, 3, 1, 1)
    return (
        images.to(
            device=device,
            dtype=torch.float16,
            non_blocking=True,
        )
        - mean
    ) / std


def hidden_to_grid(hidden: torch.Tensor) -> torch.Tensor:
    assert hidden.ndim == 3
    assert hidden.shape[1] == 1 + PATCH_GRID * PATCH_GRID
    assert hidden.shape[2] == HIDDEN_SIZE

    patch = hidden[:, 1:, :]
    patch = patch.transpose(1, 2).contiguous()
    patch = patch.view(
        hidden.shape[0],
        HIDDEN_SIZE,
        PATCH_GRID,
        PATCH_GRID,
    )
    return patch


def probe_encoder_batch(
    vision,
    train_df: pd.DataFrame,
    device: torch.device,
) -> tuple[int, dict]:
    first_case = str(train_df.iloc[0]["case_id"])
    rows = (
        train_df[
            train_df["case_id"].astype(str) == first_case
        ]
        .sort_values("image_array_index")
        .head(16)
    )

    first = rows.iloc[0]
    images = np.load(
        str(first["image_path"]),
        mmap_mode="r",
    )
    z = rows["image_array_index"].astype(int).to_numpy()

    attempts = []

    for batch_size in [16, 8, 4, 2, 1]:
        if len(z) < batch_size:
            continue

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        try:
            batch_np = np.asarray(
                images[z[:batch_size]],
                dtype=np.float32,
            )[:, None, :, :]
            batch = torch.from_numpy(batch_np)
            inputs = clip_normalize(batch, device)

            with torch.no_grad():
                outputs = vision(
                    pixel_values=inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
                low = hidden_to_grid(
                    outputs.hidden_states[LOW_HIDDEN_INDEX]
                )
                high = hidden_to_grid(
                    outputs.hidden_states[HIGH_HIDDEN_INDEX]
                )

            assert low.shape == (
                batch_size,
                HIDDEN_SIZE,
                PATCH_GRID,
                PATCH_GRID,
            )
            assert high.shape == low.shape
            assert torch.isfinite(low).all()
            assert torch.isfinite(high).all()

            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            attempts.append({
                "batch_size": batch_size,
                "status": "PASS",
                "peak_allocated_gib": peak,
            })

            del batch, inputs, outputs, low, high
            gc.collect()
            torch.cuda.empty_cache()

            return batch_size, {
                "attempts": attempts,
                "selected_encoder_batch": batch_size,
                "peak_allocated_gib": peak,
            }

        except torch.cuda.OutOfMemoryError:
            attempts.append({
                "batch_size": batch_size,
                "status": "OOM",
            })
            gc.collect()
            torch.cuda.empty_cache()

    raise RuntimeError(
        "CLIP ViT-L/14-336 frozen feature extraction did not fit even at batch 1."
    )


def feature_paths(split_name: str, case_id: str):
    case_dir = FEATURE_ROOT / split_name
    case_dir.mkdir(parents=True, exist_ok=True)
    return (
        case_dir / f"{case_id}_low.npy",
        case_dir / f"{case_id}_high.npy",
    )


def build_feature_cache(
    *,
    vision,
    encoder_batch: int,
    train_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    device: torch.device,
    pin: dict,
    weight_hashes: dict,
    config_hash: str,
):
    train_sha = base.sha256_file(TRAIN_MANIFEST_17)
    selection_sha = base.sha256_file(SELECTION_MANIFEST)

    prior_manifest = None
    prior_contract_matches = False

    if FEATURE_MANIFEST.exists():
        prior_manifest = json.loads(
            FEATURE_MANIFEST.read_text(encoding="utf-8")
        )

        prior_contract_matches = bool(
            prior_manifest.get("status") == "COMPLETE"
            and prior_manifest.get("config_hash") == config_hash
            and prior_manifest.get("backbone_revision") == pin["revision"]
            and prior_manifest.get("train_manifest_sha256") == train_sha
            and prior_manifest.get("selection_manifest_sha256") == selection_sha
        )

        if prior_contract_matches:
            all_ok = True

            for item in prior_manifest.get("files", []):
                path = ROOT / item["relative_path"]

                if (
                    not path.exists()
                    or path.stat().st_size != int(item["size_bytes"])
                ):
                    all_ok = False
                    break

            if all_ok:
                print("✓ Frozen CLIP feature cache : REUSED")
                return prior_manifest

    if FEATURE_ROOT.exists():
        shutil.rmtree(FEATURE_ROOT)
    FEATURE_ROOT.mkdir(parents=True, exist_ok=True)

    files = []

    for split_name, df in [
        ("fitting", train_df),
        ("selection", selection_df),
    ]:
        case_ids = sorted(
            df["case_id"].astype(str).unique().tolist()
        )

        for case_id in case_ids:
            rows = (
                df[df["case_id"].astype(str) == case_id]
                .sort_values("image_array_index")
                .reset_index(drop=True)
            )
            first = rows.iloc[0]

            image = np.load(
                str(first["image_path"]),
                mmap_mode="r",
            )
            z = rows["image_array_index"].astype(int).to_numpy()

            low_path, high_path = feature_paths(
                split_name,
                case_id,
            )

            low_mm = np.lib.format.open_memmap(
                low_path,
                mode="w+",
                dtype=np.float16,
                shape=(
                    len(rows),
                    HIDDEN_SIZE,
                    PATCH_GRID,
                    PATCH_GRID,
                ),
            )
            high_mm = np.lib.format.open_memmap(
                high_path,
                mode="w+",
                dtype=np.float16,
                shape=(
                    len(rows),
                    HIDDEN_SIZE,
                    PATCH_GRID,
                    PATCH_GRID,
                ),
            )

            progress = tqdm(
                range(0, len(rows), encoder_batch),
                desc=f"cache {split_name}/{case_id}",
                unit="batch",
                file=sys.stdout,
                leave=False,
            )

            for start in progress:
                end = min(
                    len(rows),
                    start + encoder_batch,
                )

                batch_np = np.asarray(
                    image[z[start:end]],
                    dtype=np.float32,
                )[:, None, :, :]

                batch = torch.from_numpy(batch_np)
                inputs = clip_normalize(batch, device)

                with torch.no_grad():
                    outputs = vision(
                        pixel_values=inputs,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                    low = hidden_to_grid(
                        outputs.hidden_states[LOW_HIDDEN_INDEX]
                    )
                    high = hidden_to_grid(
                        outputs.hidden_states[HIGH_HIDDEN_INDEX]
                    )

                low_mm[start:end] = (
                    low.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
                high_mm[start:end] = (
                    high.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )

                del batch, inputs, outputs, low, high

            low_mm.flush()
            high_mm.flush()
            del low_mm, high_mm
            gc.collect()

            for path in [low_path, high_path]:
                files.append({
                    "relative_path": str(path.relative_to(ROOT)),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": base.sha256_file(path),
                })

    manifest = {
        "timestamp_utc": base.utc_now(),
        "stage": "NOTEBOOK_05E",
        "status": "COMPLETE",
        "claim_status": "EXPLICIT_ADAPTATION_NOT_EXACT_SEGCT_CLIP_REPRODUCTION",
        "backbone_repository": pin["repository"],
        "backbone_revision": pin["revision"],
        "backbone_weight_hashes": weight_hashes,
        "config_hash": config_hash,
        "train_manifest_sha256": train_sha,
        "selection_manifest_sha256": selection_sha,
        "low_hidden_state_index": LOW_HIDDEN_INDEX,
        "high_hidden_state_index": HIGH_HIDDEN_INDEX,
        "feature_dtype": "float16",
        "patch_grid": [PATCH_GRID, PATCH_GRID],
        "hidden_size": HIDDEN_SIZE,
        "encoder_batch": encoder_batch,
        "n_fitting_slices": int(len(train_df)),
        "n_selection_slices": int(len(selection_df)),
        "files": files,
        "calibration_accessed": False,
        "target_accessed": False,
    }

    # Fresh Kaggle runtimes regenerate the ignored feature cache. If a
    # previously committed manifest exists, require byte-identical cached
    # features before preserving its exact scientific contract. This lets
    # rolling checkpoints resume without silently changing frozen features.
    if prior_contract_matches and prior_manifest is not None:
        prior_files = {
            item["relative_path"]: (
                int(item["size_bytes"]),
                item["sha256"],
            )
            for item in prior_manifest.get("files", [])
        }

        new_files = {
            item["relative_path"]: (
                int(item["size_bytes"]),
                item["sha256"],
            )
            for item in files
        }

        if prior_files == new_files:
            print(
                "✓ Regenerated frozen features are byte-identical "
                "to the committed cache contract."
            )
            return prior_manifest

        raise RuntimeError(
            "Regenerated frozen CLIP features differ from the committed "
            "feature-cache manifest. Refusing to resume prior checkpoints "
            "under a changed feature basis."
        )

    base.atomic_text(
        FEATURE_MANIFEST,
        json.dumps(manifest, indent=2),
    )

    base.git_sync(
        "Record Notebook 05E frozen CLIP feature cache manifest"
    )

    print("✓ Frozen CLIP feature cache : COMPLETE")
    return manifest


class FeatureCaseStore:
    def __init__(
        self,
        df: pd.DataFrame,
        split_name: str,
    ):
        self.case_ids = sorted(
            df["case_id"].astype(str).unique().tolist()
        )
        self.cases = {}

        for case_id in self.case_ids:
            rows = (
                df[df["case_id"].astype(str) == case_id]
                .sort_values("image_array_index")
                .reset_index(drop=True)
            )
            first = rows.iloc[0]

            low_path, high_path = feature_paths(
                split_name,
                case_id,
            )

            low = np.load(
                low_path,
                mmap_mode="r",
            )
            high = np.load(
                high_path,
                mmap_mode="r",
            )

            target = np.load(
                str(first["infection_cache_path"]),
                mmap_mode="r",
            )
            valid = np.load(
                str(first["valid_mask_path"]),
                mmap_mode="r",
            )

            z = rows["image_array_index"].astype(int).to_numpy()
            lesion_area = (
                target[z]
                .reshape(len(z), -1)
                .sum(axis=1)
            )
            positive_row = np.flatnonzero(lesion_area > 0)

            self.cases[case_id] = {
                "rows": rows,
                "low": low,
                "high": high,
                "target": target,
                "valid": valid,
                "z": z,
                "all_row": np.arange(len(z), dtype=np.int64),
                "positive_row": positive_row.astype(np.int64),
            }


class PatientUniformSampler:
    def __init__(
        self,
        store: FeatureCaseStore,
        seed: int,
    ):
        self.store = store
        self.case_ids = list(store.case_ids)
        self.rng = np.random.default_rng(seed)
        self.samples_seen = 0

    def sample(self, batch_size: int):
        lows = []
        highs = []
        targets = []
        valids = []

        for _ in range(batch_size):
            case_id = str(
                self.rng.choice(self.case_ids)
            )
            case = self.store.cases[case_id]

            use_positive = (
                len(case["positive_row"]) > 0
                and self.rng.random() < POSITIVE_PROB
            )

            row_idx = int(
                self.rng.choice(
                    case["positive_row"]
                    if use_positive
                    else case["all_row"]
                )
            )
            z = int(case["z"][row_idx])

            lows.append(
                np.asarray(
                    case["low"][row_idx],
                    dtype=np.float32,
                )
            )
            highs.append(
                np.asarray(
                    case["high"][row_idx],
                    dtype=np.float32,
                )
            )
            targets.append(
                np.asarray(
                    case["target"][z],
                    dtype=np.float32,
                )[None, ...]
            )
            valids.append(
                np.asarray(
                    case["valid"],
                    dtype=np.float32,
                )[None, ...]
            )

        self.samples_seen += batch_size

        return (
            np.stack(lows),
            np.stack(highs),
            np.stack(targets),
            np.stack(valids),
        )

    def state_dict(self):
        return {
            "bit_generator_state": self.rng.bit_generator.state,
            "samples_seen": int(self.samples_seen),
        }

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["bit_generator_state"]
        self.samples_seen = int(state["samples_seen"])


def build_decoder(seed: int, device: torch.device):
    from evict.models.segct_clip_visual_adaptation import (
        SegCTClipVisualDecoder,
    )

    base.reset_rng(seed)

    model = SegCTClipVisualDecoder(
        in_channels=HIDDEN_SIZE,
        projection_channels=128,
    )

    init_hash = base.model_state_hash(model)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=scheduler_multiplier,
    )

    return model, optimizer, scheduler, init_hash


def probe_decoder_microbatch(
    *,
    store: FeatureCaseStore,
    device: torch.device,
) -> dict:
    from evict.models.segct_clip_visual_adaptation import (
        SegCTClipVisualDecoder,
        masked_soft_dice_loss,
    )

    candidate_order = [16, 8, 4, 2, 1]
    attempts = []

    case = store.cases[store.case_ids[0]]
    available = len(case["all_row"])

    for batch_size in candidate_order:
        if EFFECTIVE_BATCH % batch_size != 0:
            continue
        if available < batch_size:
            continue

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        base.reset_rng(17)

        try:
            model = SegCTClipVisualDecoder(
                in_channels=HIDDEN_SIZE,
                projection_channels=128,
            ).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            )

            idx = np.arange(batch_size)
            z = case["z"][idx]

            low = torch.from_numpy(
                np.asarray(
                    case["low"][idx],
                    dtype=np.float32,
                )
            ).to(device)

            high = torch.from_numpy(
                np.asarray(
                    case["high"][idx],
                    dtype=np.float32,
                )
            ).to(device)

            target = torch.from_numpy(
                np.asarray(
                    case["target"][z],
                    dtype=np.float32,
                )[:, None, :, :]
            ).to(device)

            valid_single = np.asarray(
                case["valid"],
                dtype=np.float32,
            )[None, None, :, :]

            valid = torch.from_numpy(
                np.repeat(
                    valid_single,
                    batch_size,
                    axis=0,
                )
            ).to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(low, high)["logits"]
            loss_dict = masked_soft_dice_loss(
                logits,
                target,
                valid,
            )

            assert torch.isfinite(loss_dict["loss"])
            loss_dict["loss"].backward()

            for parameter in model.parameters():
                if parameter.grad is not None:
                    assert torch.isfinite(parameter.grad).all()

            peak = torch.cuda.max_memory_allocated(device) / 1024**3

            attempts.append({
                "micro_batch": batch_size,
                "gradient_accumulation": EFFECTIVE_BATCH // batch_size,
                "status": "PASS",
                "peak_allocated_gib": peak,
                "smoke_loss": float(
                    loss_dict["loss"].detach().cpu()
                ),
            })

            del (
                model,
                optimizer,
                low,
                high,
                target,
                valid,
                logits,
                loss_dict,
            )
            gc.collect()
            torch.cuda.empty_cache()

            result = {
                "timestamp_utc": base.utc_now(),
                "stage": "NOTEBOOK_05E",
                "attempts": attempts,
                "selected_micro_batch": batch_size,
                "selected_gradient_accumulation": EFFECTIVE_BATCH // batch_size,
                "effective_batch": EFFECTIVE_BATCH,
                "calibration_accessed": False,
                "target_accessed": False,
                "status": "PASS",
            }

            base.atomic_text(
                SMOKE_AUDIT,
                json.dumps(result, indent=2),
            )

            base.git_sync(
                "Record Notebook 05E SegCT-CLIP visual adaptation GPU smoke audit"
            )

            return result

        except torch.cuda.OutOfMemoryError:
            attempts.append({
                "micro_batch": batch_size,
                "status": "OOM",
            })
            gc.collect()
            torch.cuda.empty_cache()

    raise RuntimeError("Decoder did not fit even at micro-batch 1.")


def validate(
    *,
    model,
    store: FeatureCaseStore,
    device: torch.device,
    batch_size: int,
    save_logits: bool,
):
    from evict.models.segct_clip_visual_adaptation import (
        masked_soft_dice_loss,
    )
    from evict.metrics import (
        CaseMetricAccumulator,
        binary_segmentation_metrics,
        macro_case_summary,
        metrics_from_confusion,
    )

    model.eval()

    case_rows = []
    raw_logits = {}
    slice_dice = []
    total_loss_weighted = 0.0
    total_slices = 0

    pooled_tp = 0
    pooled_tn = 0
    pooled_fp = 0
    pooled_fn = 0

    with torch.no_grad():
        for case_id in store.case_ids:
            case = store.cases[case_id]
            n = len(case["z"])
            accumulator = CaseMetricAccumulator(case_id)
            case_logits = []

            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                row_idx = np.arange(start, end)
                z = case["z"][row_idx]

                low = torch.from_numpy(
                    np.asarray(
                        case["low"][row_idx],
                        dtype=np.float32,
                    )
                ).to(device)

                high = torch.from_numpy(
                    np.asarray(
                        case["high"][row_idx],
                        dtype=np.float32,
                    )
                ).to(device)

                target_np = np.asarray(
                    case["target"][z],
                    dtype=np.float32,
                )[:, None, :, :]

                valid_np = np.repeat(
                    np.asarray(
                        case["valid"],
                        dtype=np.float32,
                    )[None, None, :, :],
                    end - start,
                    axis=0,
                )

                target = torch.from_numpy(target_np).to(device)
                valid = torch.from_numpy(valid_np).to(device)

                logits = model(low, high)["logits"]
                loss_dict = masked_soft_dice_loss(
                    logits,
                    target,
                    valid,
                )
                probabilities = torch.sigmoid(logits)
                predictions = (
                    probabilities >= THRESHOLD
                ).to(torch.uint8)

                total_loss_weighted += (
                    float(loss_dict["loss"].detach().cpu())
                    * (end - start)
                )
                total_slices += end - start

                logits_np = (
                    logits[:, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                pred_np = (
                    predictions[:, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                for local_idx in range(end - start):
                    ref = target_np[local_idx, 0]
                    valid2d = valid_np[local_idx, 0]
                    pred = pred_np[local_idx]

                    accumulator.update(
                        ref,
                        pred,
                        valid2d,
                    )

                    sm = binary_segmentation_metrics(
                        ref,
                        pred,
                        valid2d,
                    )
                    slice_dice.append(float(sm["dice"]))

                if save_logits:
                    case_logits.append(logits_np)

                del (
                    low,
                    high,
                    target,
                    valid,
                    logits,
                    loss_dict,
                    probabilities,
                    predictions,
                )

            record = accumulator.compute()
            case_rows.append(record)

            pooled_tp += int(record["tp"])
            pooled_tn += int(record["tn"])
            pooled_fp += int(record["fp"])
            pooled_fn += int(record["fn"])

            if save_logits:
                raw_logits[case_id] = np.concatenate(
                    case_logits,
                    axis=0,
                )

    macro = macro_case_summary(case_rows)
    pooled = metrics_from_confusion(
        pooled_tp,
        pooled_tn,
        pooled_fp,
        pooled_fn,
    )

    metrics = {
        "selection_loss": total_loss_weighted / max(total_slices, 1),
        "macro_case_dice": float(macro["macro_dice"]),
        "macro_case_iou": float(macro["macro_iou"]),
        "macro_case_sensitivity": float(macro["macro_sensitivity"]),
        "macro_case_specificity": float(macro["macro_specificity"]),
        "macro_slice_dice": float(np.mean(slice_dice)),
        "pooled_dice": float(pooled["dice"]),
        "pooled_iou": float(pooled["iou"]),
        "n_cases": len(case_rows),
        "n_slices": total_slices,
    }

    model.train()
    return metrics, case_rows, raw_logits


def install_best_logits(
    *,
    seed: int,
    p: dict,
    raw_logits: dict,
    step: int,
):
    best_dir = p["pred_dir"] / "best_selection_logits"
    tmp_dir = p["pred_dir"] / "best_selection_logits.tmp"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cases = []

    for case_id, arr in raw_logits.items():
        path = tmp_dir / f"{case_id}.npy"
        np.save(
            path,
            np.asarray(arr, dtype=np.float32),
        )
        cases.append({
            "case_id": case_id,
            "shape": list(arr.shape),
            "dtype": "float32",
            "sha256": base.sha256_file(path),
        })

    if best_dir.exists():
        shutil.rmtree(best_dir)

    os.replace(
        tmp_dir,
        best_dir,
    )

    manifest = {
        "timestamp_utc": base.utc_now(),
        "stage": "NOTEBOOK_05E",
        "seed": seed,
        "step": step,
        "threshold": THRESHOLD,
        "claim_status": "EXPLICIT_ADAPTATION_NOT_EXACT_SEGCT_CLIP_REPRODUCTION",
        "cases": cases,
        "calibration_accessed": False,
        "target_accessed": False,
    }

    base.atomic_text(
        p["best_logits_manifest"],
        json.dumps(manifest, indent=2),
    )


def checkpoint_payload(
    *,
    seed: int,
    model,
    optimizer,
    scheduler,
    sampler,
    global_step: int,
    best_score: float,
    best_step: int,
    patience_count: int,
    last_validation_step: int,
    images_seen: int,
    config_hash: str,
    pin_hash: str,
    feature_manifest_hash: str,
    split_hash: str,
    model_code_hash: str,
    metrics_code_hash: str,
    init_hash: str,
    micro_batch: int,
    grad_accum: int,
):
    return {
        "format_version": 1,
        "project": "EviCT",
        "stage": "NOTEBOOK_05E",
        "run_id": f"segct_clip_visual_adaptation_seed{seed}",
        "claim_status": "EXPLICIT_ADAPTATION_NOT_EXACT_SEGCT_CLIP_REPRODUCTION",
        "seed": seed,
        "split_seed": 17,
        "global_step": int(global_step),
        "best_score": float(best_score),
        "best_step": int(best_step),
        "patience_count": int(patience_count),
        "last_validation_step": int(last_validation_step),
        "images_seen": int(images_seen),
        "decoder_state_dict": {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "sampler_state": sampler.state_dict(),
        "rng_state": base.capture_rng_state(),
        "config_hash": config_hash,
        "pin_hash": pin_hash,
        "feature_manifest_hash": feature_manifest_hash,
        "split_hash": split_hash,
        "model_code_hash": model_code_hash,
        "metrics_code_hash": metrics_code_hash,
        "initialization_hash": init_hash,
        "micro_batch": micro_batch,
        "gradient_accumulation": grad_accum,
        "effective_batch": EFFECTIVE_BATCH,
        "learning_rate": LEARNING_RATE,
        "threshold": THRESHOLD,
        "calibration_accessed": False,
        "target_accessed": False,
    }


def save_checkpoint(
    *,
    seed: int,
    p: dict,
    model,
    optimizer,
    scheduler,
    sampler,
    global_step: int,
    best_score: float,
    best_step: int,
    patience_count: int,
    last_validation_step: int,
    images_seen: int,
    config_hash: str,
    pin_hash: str,
    feature_manifest_hash: str,
    split_hash: str,
    model_code_hash: str,
    metrics_code_hash: str,
    init_hash: str,
    micro_batch: int,
    grad_accum: int,
    write_last: bool,
):
    payload = checkpoint_payload(
        seed=seed,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        global_step=global_step,
        best_score=best_score,
        best_step=best_step,
        patience_count=patience_count,
        last_validation_step=last_validation_step,
        images_seen=images_seen,
        config_hash=config_hash,
        pin_hash=pin_hash,
        feature_manifest_hash=feature_manifest_hash,
        split_hash=split_hash,
        model_code_hash=model_code_hash,
        metrics_code_hash=metrics_code_hash,
        init_hash=init_hash,
        micro_batch=micro_batch,
        grad_accum=grad_accum,
    )

    base.atomic_torch_save(
        payload,
        p["recovery_pt"],
    )

    if write_last:
        base.atomic_torch_save(
            payload,
            p["last_pt"],
        )

    del payload


def validate_checkpoint(
    payload: dict,
    *,
    seed: int,
    config_hash: str,
    pin_hash: str,
    feature_manifest_hash: str,
    split_hash: str,
    model_code_hash: str,
    metrics_code_hash: str,
):
    assert payload["project"] == "EviCT"
    assert payload["stage"] == "NOTEBOOK_05E"
    assert payload["seed"] == seed
    assert payload["claim_status"] == "EXPLICIT_ADAPTATION_NOT_EXACT_SEGCT_CLIP_REPRODUCTION"
    assert payload["config_hash"] == config_hash
    assert payload["pin_hash"] == pin_hash
    assert payload["feature_manifest_hash"] == feature_manifest_hash
    assert payload["split_hash"] == split_hash
    assert payload["model_code_hash"] == model_code_hash
    assert payload["metrics_code_hash"] == metrics_code_hash
    assert payload["calibration_accessed"] is False
    assert payload["target_accessed"] is False


def restore_checkpoint(
    *,
    seed: int,
    path: Path,
    train_store: FeatureCaseStore,
    device: torch.device,
    config_hash: str,
    pin_hash: str,
    feature_manifest_hash: str,
    split_hash: str,
    model_code_hash: str,
    metrics_code_hash: str,
):
    model, optimizer, scheduler, init_hash = build_decoder(
        seed,
        device,
    )
    sampler = PatientUniformSampler(
        train_store,
        seed,
    )

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    validate_checkpoint(
        payload,
        seed=seed,
        config_hash=config_hash,
        pin_hash=pin_hash,
        feature_manifest_hash=feature_manifest_hash,
        split_hash=split_hash,
        model_code_hash=model_code_hash,
        metrics_code_hash=metrics_code_hash,
    )

    assert payload["initialization_hash"] == init_hash

    model.load_state_dict(
        payload["decoder_state_dict"]
    )
    model.to(device)

    optimizer.load_state_dict(
        payload["optimizer_state_dict"]
    )
    for opt_state in optimizer.state.values():
        for key, value in list(opt_state.items()):
            if torch.is_tensor(value):
                opt_state[key] = value.to(device)

    scheduler.load_state_dict(
        payload["scheduler_state_dict"]
    )
    sampler.load_state_dict(
        payload["sampler_state"]
    )
    base.restore_rng_state(
        payload["rng_state"]
    )

    state = {
        "global_step": int(payload["global_step"]),
        "best_score": float(payload["best_score"]),
        "best_step": int(payload["best_step"]),
        "patience_count": int(payload["patience_count"]),
        "last_validation_step": int(payload["last_validation_step"]),
        "images_seen": int(payload["images_seen"]),
        "micro_batch": int(payload["micro_batch"]),
        "grad_accum": int(payload["gradient_accumulation"]),
    }

    del payload
    return model, optimizer, scheduler, sampler, init_hash, state


def load_logs(p: dict):
    train_rows = (
        pd.read_csv(p["train_log"]).to_dict("records")
        if p["train_log"].exists()
        else []
    )
    selection_rows = (
        pd.read_csv(p["selection_log"]).to_dict("records")
        if p["selection_log"].exists()
        else []
    )
    case_rows = (
        pd.read_csv(p["case_log"]).to_dict("records")
        if p["case_log"].exists()
        else []
    )
    return train_rows, selection_rows, case_rows


def dedupe_logs(train_rows, selection_rows, case_rows):
    if train_rows:
        df = pd.DataFrame(train_rows)
        df = (
            df.sort_values("step")
            .drop_duplicates("step", keep="last")
        )
        train_rows[:] = df.to_dict("records")

    if selection_rows:
        df = pd.DataFrame(selection_rows)
        df = (
            df.sort_values("step")
            .drop_duplicates("step", keep="last")
        )
        selection_rows[:] = df.to_dict("records")

    if case_rows:
        df = pd.DataFrame(case_rows)
        df = (
            df.sort_values(["step", "case_id"])
            .drop_duplicates(
                ["step", "case_id"],
                keep="last",
            )
        )
        case_rows[:] = df.to_dict("records")


def write_logs(p: dict, train_rows, selection_rows, case_rows):
    pd.DataFrame(train_rows).to_csv(
        p["train_log"],
        index=False,
    )
    pd.DataFrame(selection_rows).to_csv(
        p["selection_log"],
        index=False,
    )
    pd.DataFrame(case_rows).to_csv(
        p["case_log"],
        index=False,
    )


def update_state(
    *,
    seed: int,
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

{base.utc_now()}

## Baseline

SegCT-CLIP visual-pathway adaptation

Claim status:

EXPLICIT ADAPTATION — NOT EXACT SEGCT-CLIP REPRODUCTION

Backbone:

Frozen CLIP ViT-L/14-336

Caption bank:

NOT USED — exact author caption bank unavailable

Contrastive loss:

NOT USED — exact caption supervision unavailable

Visual design:

Dual-level frozen CLIP patch features + explicit EviCT lightweight decoder adaptation

Training seed:

{seed}

Split seed:

17

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

## Isolation

Calibration accessed:

NO

Target / MedSeg accessed:

NO

Target lock:

ACTIVE

## Next

Complete seeds 17, 42 and 2026 under the same frozen adaptation protocol,
aggregate the source-selection results, then proceed to Notebook 06.
"""

    base.atomic_text(
        STATE_PATH,
        text,
    )
    base.atomic_text(
        HANDOFF_STATE,
        text,
    )


def one_optimizer_update(
    *,
    model,
    optimizer,
    scheduler,
    sampler,
    device,
    micro_batch: int,
    grad_accum: int,
):
    from evict.models.segct_clip_visual_adaptation import (
        masked_soft_dice_loss,
    )

    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0

    for _ in range(grad_accum):
        low_np, high_np, target_np, valid_np = sampler.sample(
            micro_batch
        )

        low = torch.from_numpy(low_np).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        high = torch.from_numpy(high_np).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        target = torch.from_numpy(target_np).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        valid = torch.from_numpy(valid_np).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        logits = model(low, high)["logits"]
        loss_dict = masked_soft_dice_loss(
            logits,
            target,
            valid,
        )
        loss = loss_dict["loss"] / grad_accum

        assert torch.isfinite(loss)
        loss.backward()

        total_loss += (
            float(loss_dict["loss"].detach().cpu())
            / grad_accum
        )

        del low, high, target, valid, logits, loss_dict, loss

    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()

    optimizer.step()
    scheduler.step()

    return {
        "loss": total_loss,
        "learning_rate": float(
            optimizer.param_groups[0]["lr"]
        ),
    }


def create_archive(
    *,
    seed: int,
    p: dict,
    step: int,
    final: bool,
):
    if final:
        name = (
            f"EviCT_Notebook05E_SegCTClipVisual_seed{seed}_"
            f"final_step{step}_Recovery.tar"
        )
    else:
        name = (
            f"EviCT_Notebook05E_SegCTClipVisual_seed{seed}_"
            "Rolling_Recovery.tar"
        )

    archive_path = WORK / name
    archive_path.unlink(missing_ok=True)

    with tarfile.open(
        archive_path,
        "w",
    ) as archive:
        checkpoint_paths = [
            p["recovery_pt"],
            p["best_pt"],
        ]
        if final:
            checkpoint_paths.append(
                p["last_pt"]
            )

        for checkpoint in checkpoint_paths:
            assert checkpoint.exists()
            archive.add(
                checkpoint,
                arcname=str(
                    Path("EviCT/artifacts/large/notebook05e")
                    / f"seed_{seed}"
                    / checkpoint.name
                ),
            )

        for source in [
            p["train_log"],
            p["selection_log"],
            p["case_log"],
            p["best_logits_manifest"],
            CONFIG_PATH,
            PIN_PATH,
            FEATURE_MANIFEST,
            MODEL_CODE,
            METRICS_CODE,
            SPLITS_PATH,
            train_manifest(seed),
            SELECTION_MANIFEST,
        ]:
            if not source.exists():
                continue
            archive.add(
                source,
                arcname=str(
                    Path("EviCT")
                    / source.relative_to(ROOT)
                ),
            )

        if final:
            best_logits_dir = (
                p["pred_dir"]
                / "best_selection_logits"
            )
            assert best_logits_dir.exists()
            archive.add(
                best_logits_dir,
                arcname=str(
                    Path("EviCT/predictions/notebook05e")
                    / f"seed_{seed}"
                    / "best_selection_logits"
                ),
            )

    digest = base.sha256_file(
        archive_path
    )
    sha_path = Path(
        str(archive_path) + ".sha256"
    )
    base.atomic_text(
        sha_path,
        digest,
    )

    return archive_path, sha_path, digest


def upload_rolling(
    *,
    token: str,
    seed: int,
    p: dict,
    step: int,
    best_score: float,
    best_step: int,
):
    archive_path, sha_path, digest = create_archive(
        seed=seed,
        p=p,
        step=step,
        final=False,
    )

    tag = (
        f"evict-nb05e-segctclip-visual-seed{seed}-rolling"
    )

    release, headers, api = base.create_or_get_release(
        token,
        tag,
        f"EviCT Notebook 05E — SegCT-CLIP visual adaptation seed {seed} — rolling",
        (
            "Rolling recovery for the explicitly labeled SegCT-CLIP visual-pathway adaptation. "
            "This is not an exact reproduction. The CLIP ViT-L/14-336 vision encoder is frozen; "
            "the author caption bank and contrastive supervision are unavailable and are not invented. "
            f"step={step}, best source-selection macro case Dice={best_score:.8f} "
            f"at step {best_step}. Calibration and target/MedSeg remain locked."
        ),
    )

    tar_asset = base.upload_asset(
        release=release,
        headers=headers,
        api=api,
        file_path=archive_path,
        content_type="application/x-tar",
    )
    sha_asset = base.upload_asset(
        release=release,
        headers=headers,
        api=api,
        file_path=sha_path,
        content_type="text/plain",
    )

    audit = {
        "timestamp_utc": base.utc_now(),
        "stage": "NOTEBOOK_05E",
        "seed": seed,
        "step": step,
        "best_macro_case_dice": best_score,
        "best_step": best_step,
        "archive_name": archive_path.name,
        "archive_sha256": digest,
        "release_tag": tag,
        "release_url": release["html_url"],
        "tar_asset_id": int(tar_asset["id"]),
        "sha_asset_id": int(sha_asset["id"]),
        "claim_status": "EXPLICIT_ADAPTATION_NOT_EXACT_SEGCT_CLIP_REPRODUCTION",
        "calibration_accessed": False,
        "target_accessed": False,
        "status": "ROLLING_DURABLE",
    }

    base.atomic_text(
        p["rolling_audit"],
        json.dumps(audit, indent=2),
    )

    base.git_sync(
        f"Record Notebook 05E SegCT-CLIP visual seed{seed} rolling step {step}"
    )

    archive_path.unlink(missing_ok=True)
    sha_path.unlink(missing_ok=True)


def restore_remote_if_needed(
    *,
    seed: int,
    p: dict,
):
    p["run_dir"].mkdir(
        parents=True,
        exist_ok=True,
    )
    p["pred_dir"].mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        p["recovery_pt"].exists()
        or p["last_pt"].exists()
    ):
        return

    if not p["rolling_audit"].exists():
        return

    audit = json.loads(
        p["rolling_audit"].read_text(
            encoding="utf-8"
        )
    )

    if audit.get("status") != "ROLLING_DURABLE":
        return

    archive_path = WORK / audit["archive_name"]

    print(
        f"Restoring seed {seed} from remote rolling recovery "
        f"at step {audit['step']}..."
    )

    base.download_verified(
        direct_release_url(
            audit["release_tag"],
            audit["archive_name"],
        ),
        archive_path,
        audit["archive_sha256"],
    )

    base.safe_extract_tar(
        archive_path,
        WORK,
    )

    assert p["recovery_pt"].exists()
    archive_path.unlink(missing_ok=True)


def upload_final(
    *,
    token: str,
    seed: int,
    p: dict,
    final_step: int,
    stop_reason: str,
    best_score: float,
    best_step: int,
    init_hash: str,
):
    archive_path, sha_path, digest = create_archive(
        seed=seed,
        p=p,
        step=final_step,
        final=True,
    )

    tag = (
        f"evict-nb05e-segctclip-visual-seed{seed}-"
        f"final-step{final_step}"
    )

    release, headers, api = base.create_or_get_release(
        token,
        tag,
        f"EviCT Notebook 05E — SegCT-CLIP visual adaptation seed {seed} — final",
        (
            "Final durable recovery for the explicitly labeled SegCT-CLIP visual-pathway adaptation. "
            "This is NOT an exact SegCT-CLIP reproduction. "
            f"seed={seed}, final step={final_step}, stop={stop_reason}, "
            f"best source-selection macro case Dice={best_score:.8f} at step {best_step}. "
            "Calibration and target/MedSeg were not accessed."
        ),
    )

    tar_asset = base.upload_asset(
        release=release,
        headers=headers,
        api=api,
        file_path=archive_path,
        content_type="application/x-tar",
    )
    sha_asset = base.upload_asset(
        release=release,
        headers=headers,
        api=api,
        file_path=sha_path,
        content_type="text/plain",
    )

    audit = {
        "timestamp_utc": base.utc_now(),
        "stage": "NOTEBOOK_05E",
        "seed": seed,
        "final_step": final_step,
        "stop_reason": stop_reason,
        "best_macro_case_dice": best_score,
        "best_step": best_step,
        "initialization_hash": init_hash,
        "final_archive": archive_path.name,
        "final_archive_sha256": digest,
        "release_tag": tag,
        "release_url": release["html_url"],
        "tar_asset_id": int(tar_asset["id"]),
        "sha_asset_id": int(sha_asset["id"]),
        "claim_status": "EXPLICIT_ADAPTATION_NOT_EXACT_SEGCT_CLIP_REPRODUCTION",
        "calibration_accessed": False,
        "target_accessed": False,
        "status": "DURABLE_COMPLETE",
    }

    base.atomic_text(
        p["final_audit"],
        json.dumps(audit, indent=2),
    )

    archive_path.unlink(missing_ok=True)
    sha_path.unlink(missing_ok=True)

    return audit


def run_seed(
    *,
    token: str,
    seed: int,
    train_store: FeatureCaseStore,
    selection_store: FeatureCaseStore,
    train_df: pd.DataFrame,
    device: torch.device,
    config_hash: str,
    pin_hash: str,
    feature_manifest_hash: str,
    model_code_hash: str,
    metrics_code_hash: str,
    micro_batch: int,
    grad_accum: int,
):
    p = paths_for(seed)
    p["run_dir"].mkdir(parents=True, exist_ok=True)
    p["pred_dir"].mkdir(parents=True, exist_ok=True)

    if p["final_audit"].exists():
        existing = json.loads(
            p["final_audit"].read_text(
                encoding="utf-8"
            )
        )
        if existing.get("status") == "DURABLE_COMPLETE":
            print(
                f"✓ seed {seed} already DURABLE_COMPLETE; skipping."
            )
            return existing

    split_hash = base.stable_json_hash({
        "splits_csv": base.sha256_file(SPLITS_PATH),
        "train_manifest": base.sha256_file(
            train_manifest(seed)
        ),
        "selection_manifest": base.sha256_file(
            SELECTION_MANIFEST
        ),
    })

    restore_remote_if_needed(
        seed=seed,
        p=p,
    )

    train_rows, selection_rows, case_rows = load_logs(p)
    dedupe_logs(
        train_rows,
        selection_rows,
        case_rows,
    )

    checkpoint_candidates = []

    for path in [
        p["last_pt"],
        p["recovery_pt"],
    ]:
        if not path.exists():
            continue

        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        try:
            validate_checkpoint(
                payload,
                seed=seed,
                config_hash=config_hash,
                pin_hash=pin_hash,
                feature_manifest_hash=feature_manifest_hash,
                split_hash=split_hash,
                model_code_hash=model_code_hash,
                metrics_code_hash=metrics_code_hash,
            )
            checkpoint_candidates.append({
                "path": path,
                "step": int(payload["global_step"]),
            })
        finally:
            del payload

    print()
    print("=" * 110)
    print(
        f"NOTEBOOK 05E — SEGCT-CLIP VISUAL ADAPTATION — SEED {seed}"
    )
    print("=" * 110)

    if checkpoint_candidates:
        newest = max(
            checkpoint_candidates,
            key=lambda x: x["step"],
        )

        (
            model,
            optimizer,
            scheduler,
            sampler,
            init_hash,
            state,
        ) = restore_checkpoint(
            seed=seed,
            path=newest["path"],
            train_store=train_store,
            device=device,
            config_hash=config_hash,
            pin_hash=pin_hash,
            feature_manifest_hash=feature_manifest_hash,
            split_hash=split_hash,
            model_code_hash=model_code_hash,
            metrics_code_hash=metrics_code_hash,
        )

        assert state["micro_batch"] == micro_batch
        assert state["grad_accum"] == grad_accum

        print(
            f"✓ resumed seed {seed}       : step {state['global_step']}"
        )

    else:
        model, optimizer, scheduler, init_hash = build_decoder(
            seed,
            device,
        )
        sampler = PatientUniformSampler(
            train_store,
            seed,
        )
        state = {
            "global_step": 0,
            "best_score": float("-inf"),
            "best_step": 0,
            "patience_count": 0,
            "last_validation_step": 0,
            "images_seen": 0,
            "micro_batch": micro_batch,
            "grad_accum": grad_accum,
        }

        print(
            f"✓ new seed {seed} run       : step 0"
        )

    global_step = state["global_step"]
    best_score = state["best_score"]
    best_step = state["best_step"]
    patience_count = state["patience_count"]
    last_validation_step = state["last_validation_step"]
    images_seen = state["images_seen"]

    model.train()

    progress = tqdm(
        total=MAX_UPDATES - global_step,
        desc=f"segctvis seed{seed}",
        unit="update",
        ncols=138,
        file=sys.stdout,
        leave=True,
    )

    stop_reason = None

    while global_step < MAX_UPDATES:
        update = one_optimizer_update(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            device=device,
            micro_batch=micro_batch,
            grad_accum=grad_accum,
        )

        global_step += 1
        images_seen += EFFECTIVE_BATCH

        train_rows.append({
            "seed": seed,
            "step": global_step,
            "loss": update["loss"],
            "learning_rate": update["learning_rate"],
            "images_seen": images_seen,
            "micro_batch": micro_batch,
            "gradient_accumulation": grad_accum,
            "effective_batch": EFFECTIVE_BATCH,
        })

        progress.set_postfix(
            step=global_step,
            loss=f"{update['loss']:.4f}",
            lr=f"{update['learning_rate']:.2e}",
            best=(
                f"{best_score:.4f}"
                if np.isfinite(best_score)
                else "nan"
            ),
            patience=f"{patience_count}/{PATIENCE_LIMIT}",
        )
        progress.update(1)

        if (
            global_step % RECOVERY_EVERY == 0
            and global_step % VALIDATE_EVERY != 0
        ):
            save_checkpoint(
                seed=seed,
                p=p,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                global_step=global_step,
                best_score=best_score,
                best_step=best_step,
                patience_count=patience_count,
                last_validation_step=last_validation_step,
                images_seen=images_seen,
                config_hash=config_hash,
                pin_hash=pin_hash,
                feature_manifest_hash=feature_manifest_hash,
                split_hash=split_hash,
                model_code_hash=model_code_hash,
                metrics_code_hash=metrics_code_hash,
                init_hash=init_hash,
                micro_batch=micro_batch,
                grad_accum=grad_accum,
                write_last=False,
            )

        if global_step % VALIDATE_EVERY == 0:
            progress.write(
                f"\n[seed {seed}] SOURCE-SELECTION VALIDATION @ step {global_step}"
            )

            metrics, current_case_rows, raw_logits = validate(
                model=model,
                store=selection_store,
                device=device,
                batch_size=micro_batch,
                save_logits=True,
            )

            score = metrics["macro_case_dice"]
            improved = score > best_score

            if improved:
                best_score = score
                best_step = global_step
                patience_count = 0

                payload = checkpoint_payload(
                    seed=seed,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    global_step=global_step,
                    best_score=best_score,
                    best_step=best_step,
                    patience_count=patience_count,
                    last_validation_step=global_step,
                    images_seen=images_seen,
                    config_hash=config_hash,
                    pin_hash=pin_hash,
                    feature_manifest_hash=feature_manifest_hash,
                    split_hash=split_hash,
                    model_code_hash=model_code_hash,
                    metrics_code_hash=metrics_code_hash,
                    init_hash=init_hash,
                    micro_batch=micro_batch,
                    grad_accum=grad_accum,
                )

                base.atomic_torch_save(
                    payload,
                    p["best_pt"],
                )
                del payload

                install_best_logits(
                    seed=seed,
                    p=p,
                    raw_logits=raw_logits,
                    step=global_step,
                )

            else:
                patience_count += 1

            last_validation_step = global_step

            selection_rows.append({
                "seed": seed,
                "step": global_step,
                **metrics,
                "best_so_far": best_score,
                "best_step": best_step,
                "patience_count": patience_count,
                "threshold": THRESHOLD,
            })

            for row in current_case_rows:
                case_rows.append({
                    "seed": seed,
                    "step": global_step,
                    **row,
                })

            dedupe_logs(
                train_rows,
                selection_rows,
                case_rows,
            )

            save_checkpoint(
                seed=seed,
                p=p,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                global_step=global_step,
                best_score=best_score,
                best_step=best_step,
                patience_count=patience_count,
                last_validation_step=last_validation_step,
                images_seen=images_seen,
                config_hash=config_hash,
                pin_hash=pin_hash,
                feature_manifest_hash=feature_manifest_hash,
                split_hash=split_hash,
                model_code_hash=model_code_hash,
                metrics_code_hash=metrics_code_hash,
                init_hash=init_hash,
                micro_batch=micro_batch,
                grad_accum=grad_accum,
                write_last=True,
            )

            write_logs(
                p,
                train_rows,
                selection_rows,
                case_rows,
            )

            update_state(
                seed=seed,
                global_step=global_step,
                best_score=best_score,
                best_step=best_step,
                patience_count=patience_count,
                images_seen=images_seen,
                status=(
                    f"NOTEBOOK_05E_SEGCT_CLIP_VISUAL_SEED_{seed}_"
                    f"RUNNING_STEP_{global_step}"
                ),
            )

            progress.write(
                f"  macro case Dice          : {score:.8f}\n"
                f"  macro case IoU           : {metrics['macro_case_iou']:.8f}\n"
                f"  macro slice Dice         : {metrics['macro_slice_dice']:.8f}\n"
                f"  pooled Dice              : {metrics['pooled_dice']:.8f}\n"
                f"  best                     : {best_score:.8f} @ {best_step}\n"
                f"  patience                 : {patience_count}/{PATIENCE_LIMIT}"
            )

            base.git_sync(
                f"Notebook 05E SegCT-CLIP visual seed{seed} validation step {global_step}"
            )

            del raw_logits
            gc.collect()
            torch.cuda.empty_cache()

            if (
                global_step in REMOTE_MILESTONES
                and patience_count < PATIENCE_LIMIT
            ):
                upload_rolling(
                    token=token,
                    seed=seed,
                    p=p,
                    step=global_step,
                    best_score=best_score,
                    best_step=best_step,
                )

            if patience_count >= PATIENCE_LIMIT:
                stop_reason = "EARLY_STOPPING_PATIENCE_8"
                break

    progress.close()

    if stop_reason is None:
        assert global_step == MAX_UPDATES
        stop_reason = "MAXIMUM_5000_UPDATES"

    assert global_step == last_validation_step
    assert p["best_pt"].exists()
    assert p["last_pt"].exists()
    assert p["recovery_pt"].exists()

    final = upload_final(
        token=token,
        seed=seed,
        p=p,
        final_step=global_step,
        stop_reason=stop_reason,
        best_score=best_score,
        best_step=best_step,
        init_hash=init_hash,
    )

    update_state(
        seed=seed,
        global_step=global_step,
        best_score=best_score,
        best_step=best_step,
        patience_count=patience_count,
        images_seen=images_seen,
        status=f"NOTEBOOK_05E_SEGCT_CLIP_VISUAL_SEED_{seed}_COMPLETE_DURABLE",
    )

    base.git_sync(
        f"Complete durable Notebook 05E SegCT-CLIP visual seed{seed}"
    )

    del model, optimizer, scheduler, sampler
    gc.collect()
    torch.cuda.empty_cache()

    return final


def aggregate_three_seeds():
    rows = []
    case_rows = []

    for seed in SEEDS:
        p = paths_for(seed)

        final = json.loads(
            p["final_audit"].read_text(
                encoding="utf-8"
            )
        )
        assert final["status"] == "DURABLE_COMPLETE"

        sel = pd.read_csv(
            p["selection_log"]
        )
        best_step = int(final["best_step"])
        row = sel[
            sel["step"].astype(int) == best_step
        ]
        assert len(row) == 1
        rec = row.iloc[0].to_dict()

        rows.append({
            "seed": seed,
            "best_step": best_step,
            "final_step": int(final["final_step"]),
            "stop_reason": final["stop_reason"],
            "macro_case_dice": float(rec["macro_case_dice"]),
            "macro_case_iou": float(rec["macro_case_iou"]),
            "macro_case_sensitivity": float(rec["macro_case_sensitivity"]),
            "macro_case_specificity": float(rec["macro_case_specificity"]),
            "macro_slice_dice": float(rec["macro_slice_dice"]),
            "pooled_dice": float(rec["pooled_dice"]),
            "pooled_iou": float(rec["pooled_iou"]),
            "release_url": final["release_url"],
        })

        cdf = pd.read_csv(
            p["case_log"]
        )
        cdf = cdf[
            cdf["step"].astype(int) == best_step
        ].copy()
        assert len(cdf) == 4
        cdf["best_step"] = best_step
        case_rows.extend(
            cdf.to_dict("records")
        )

    df = pd.DataFrame(rows)
    case_df = pd.DataFrame(case_rows)

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        TABLE_DIR
        / "notebook05e_segct_clip_visual_three_seed.csv"
    )
    case_path = (
        TABLE_DIR
        / "notebook05e_segct_clip_visual_case_metrics.csv"
    )

    df.to_csv(
        summary_path,
        index=False,
    )
    case_df.to_csv(
        case_path,
        index=False,
    )

    metrics = {}

    for col in [
        "macro_case_dice",
        "macro_case_iou",
        "macro_case_sensitivity",
        "macro_case_specificity",
        "macro_slice_dice",
        "pooled_dice",
        "pooled_iou",
    ]:
        values = df[col].astype(float).to_numpy()
        metrics[col] = {
            "mean": float(
                np.mean(values)
            ),
            "sample_sd": float(
                np.std(values, ddof=1)
            ),
        }

    audit = {
        "timestamp_utc": base.utc_now(),
        "stage": "NOTEBOOK_05E",
        "status": "THREE_SEED_VISUAL_ADAPTATION_FROZEN",
        "method": "SegCT-CLIP visual-pathway adaptation",
        "claim_status": "EXPLICIT_ADAPTATION_NOT_EXACT_SEGCT_CLIP_REPRODUCTION",
        "primary_seeds": SEEDS,
        "backbone": "frozen CLIP ViT-L/14-336",
        "caption_bank_used": False,
        "contrastive_loss_used": False,
        "segmentation_loss": "masked soft Dice",
        "selection_threshold": THRESHOLD,
        "metrics": metrics,
        "seed_results": rows,
        "historical_paper_metrics_entered_as_rerun_results": False,
        "calibration_accessed": False,
        "target_accessed": False,
    }

    base.atomic_text(
        FINAL_AGG_AUDIT,
        json.dumps(audit, indent=2),
    )

    mean_dice = metrics["macro_case_dice"]["mean"]
    sd_dice = metrics["macro_case_dice"]["sample_sd"]

    report = [
        "# Notebook 05E — SegCT-CLIP Visual-Pathway Adaptation",
        "",
        "**Status: explicit adaptation, not an exact SegCT-CLIP reproduction.**",
        "",
        "The Notebook-05D reproduction audit found that the author repository, exact checkpoint, caption bank, caption-to-slice mapping, exact low/high layer indices, fusion operator, decoder topology and training hyperparameters were not sufficiently specified for an exact rerun.",
        "",
        "This adaptation therefore uses the documented CLIP ViT-L/14-336 visual backbone and dual-level visual-feature idea, freezes the CLIP encoder, and trains an explicitly defined EviCT decoder on frozen source features. The unavailable caption bank, text encoder supervision, contrastive loss and caption-refinement module are not fabricated.",
        "",
        f"Three-seed source-selection macro-case Dice: **{mean_dice:.6f} ± {sd_dice:.6f}** (sample SD).",
        "",
        "| Seed | Best step | Final step | Macro case Dice | Macro case IoU | Macro slice Dice | Pooled Dice |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in df.iterrows():
        report.append(
            f"| {int(row['seed'])} | {int(row['best_step'])} | "
            f"{int(row['final_step'])} | "
            f"{row['macro_case_dice']:.6f} | "
            f"{row['macro_case_iou']:.6f} | "
            f"{row['macro_slice_dice']:.6f} | "
            f"{row['pooled_dice']:.6f} |"
        )

    report.extend([
        "",
        "Selection threshold: 0.5 fixed.",
        "Calibration accessed: NO.",
        "Target / MedSeg accessed: NO.",
        "Historical SegCT-CLIP table values are not rerun results.",
        "",
        "Next: Notebook 06 — fixed biomedical text prototypes and the proposed EviCT semantic branch.",
    ])

    report_path = (
        REPORT_DIR
        / "notebook05e_segct_clip_visual_adaptation_report.md"
    )
    base.atomic_text(
        report_path,
        "\n".join(report) + "\n",
    )

    state = f"""# EviCT Execution State

## Current stage

NOTEBOOK_05E_SEGCT_CLIP_VISUAL_ADAPTATION_THREE_SEED_FROZEN

## Timestamp

{base.utc_now()}

## Method

SegCT-CLIP visual-pathway adaptation

Claim status:

EXPLICIT ADAPTATION — NOT EXACT SEGCT-CLIP REPRODUCTION

Backbone:

Frozen CLIP ViT-L/14-336

Caption bank:

NOT USED

Contrastive loss:

NOT USED

Reason:

The exact author caption repository and caption-to-slice supervision were not available.
They were not invented.

Primary seeds:

17, 42, 2026

Three-seed source-selection macro case Dice:

{mean_dice:.8f}

Sample standard deviation:

{sd_dice:.8f}

Selection threshold:

0.5 fixed

Historical paper values treated as rerun results:

NO

## Isolation

Calibration accessed:

NO

Target / MedSeg accessed:

NO

Target lock:

ACTIVE

## Next

Notebook 05 supervised/reference-model stage is complete with:
1. frozen SegFormer-B1,
2. frozen competitive residual 2D U-Net,
3. explicitly labeled SegCT-CLIP visual-pathway adaptation.

Proceed to Notebook 06: fixed biomedical text prototypes and proposed EviCT semantic branch.
"""

    base.atomic_text(
        STATE_PATH,
        state,
    )
    base.atomic_text(
        HANDOFF_STATE,
        state,
    )

    base.git_sync(
        "Freeze Notebook 05E three-seed SegCT-CLIP visual adaptation"
    )

    return audit


def main():
    print("=" * 110)
    print("EVICT NOTEBOOK 05E — SEGCT-CLIP VISUAL-PATHWAY ADAPTATION")
    print("=" * 110)
    print("CLAIM: EXPLICIT ADAPTATION — NOT EXACT SEGCT-CLIP REPRODUCTION")

    for path in [
        CONFIG_PATH,
        MODEL_CODE,
        METRICS_CODE,
        SPLITS_PATH,
        TRAIN_MANIFEST_17,
        TRAIN_MANIFEST_42,
        TRAIN_MANIFEST_2026,
        SELECTION_MANIFEST,
        STATE_PATH,
        GIT_SYNC,
    ]:
        assert path.exists(), f"Missing required file: {path}"

    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))

    token = UserSecretsClient().get_secret("pushEviCT")
    assert token
    print("✓ GitHub secret            : PASS")

    assert torch.cuda.is_available(), "GPU required."
    device = torch.device("cuda:0")
    print(f"✓ GPU                      : {torch.cuda.get_device_name(device)}")

    state = STATE_PATH.read_text(encoding="utf-8")
    assert (
        "NOTEBOOK_05D_SEGCT_CLIP_REPRODUCTION_AUDIT_COMPLETE" in state
        or "NOTEBOOK_05E_SEGCT_CLIP_VISUAL" in state
    )
    assert "Calibration accessed:\n\nNO" in state
    assert "Target / MedSeg accessed:\n\nNO" in state
    assert "Target lock:\n\nACTIVE" in state

    config = json.loads(
        CONFIG_PATH.read_text(encoding="utf-8")
    )
    assert config["claim_status"] == "EXPLICIT_ADAPTATION_NOT_EXACT_SEGCT_CLIP_REPRODUCTION"
    assert config["data_contract"]["calibration_access_allowed"] is False
    assert config["data_contract"]["target_access_allowed"] is False
    assert config["adaptation"]["caption_bank_used"] is False
    assert config["adaptation"]["contrastive_loss_used"] is False

    source_manifests_equivalent()
    split_df, train_df, selection_df = validate_source_contract()

    print("✓ Frozen fitting cases     : 12")
    print("✓ Frozen selection cases   : 4")
    print("✓ Primary seeds            : 17, 42, 2026")
    print("✓ Calibration accessed     : NO")
    print("✓ Target / MedSeg accessed : NO")

    ensure_source_cache()

    pin = ensure_backbone_pin(token)
    config_hash = base.stable_json_hash(config)
    pin_hash = base.stable_json_hash(pin)
    model_code_hash = base.sha256_file(MODEL_CODE)
    metrics_code_hash = base.sha256_file(METRICS_CODE)

    vision, snapshot, weight_hashes = load_clip_vision(
        pin,
        device,
    )

    encoder_batch, encoder_smoke = probe_encoder_batch(
        vision,
        train_df,
        device,
    )

    print(f"✓ Frozen CLIP encoder batch: {encoder_batch}")
    print(
        f"✓ Encoder peak VRAM        : "
        f"{encoder_smoke['peak_allocated_gib']:.2f} GiB"
    )

    feature_manifest = build_feature_cache(
        vision=vision,
        encoder_batch=encoder_batch,
        train_df=train_df,
        selection_df=selection_df,
        device=device,
        pin=pin,
        weight_hashes=weight_hashes,
        config_hash=config_hash,
    )

    del vision
    gc.collect()
    torch.cuda.empty_cache()

    train_store = FeatureCaseStore(
        train_df,
        "fitting",
    )
    selection_store = FeatureCaseStore(
        selection_df,
        "selection",
    )

    if SMOKE_AUDIT.exists():
        smoke = json.loads(
            SMOKE_AUDIT.read_text(encoding="utf-8")
        )
        assert smoke["status"] == "PASS"
    else:
        smoke = probe_decoder_microbatch(
            store=train_store,
            device=device,
        )

    micro_batch = int(
        smoke["selected_micro_batch"]
    )
    grad_accum = int(
        smoke["selected_gradient_accumulation"]
    )

    assert micro_batch * grad_accum == EFFECTIVE_BATCH

    print(f"✓ Decoder micro-batch      : {micro_batch}")
    print(f"✓ Gradient accumulation    : {grad_accum}")
    print("✓ Effective batch          : 16")

    feature_manifest_hash = base.stable_json_hash(
        feature_manifest
    )

    results = []

    for seed in SEEDS:
        result = run_seed(
            token=token,
            seed=seed,
            train_store=train_store,
            selection_store=selection_store,
            train_df=train_df,
            device=device,
            config_hash=config_hash,
            pin_hash=pin_hash,
            feature_manifest_hash=feature_manifest_hash,
            model_code_hash=model_code_hash,
            metrics_code_hash=metrics_code_hash,
            micro_batch=micro_batch,
            grad_accum=grad_accum,
        )
        results.append(result)

    aggregate = aggregate_three_seeds()

    subprocess.run(
        ["git", "fetch", "origin", "main"],
        cwd=str(ROOT),
        check=True,
    )

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

    mean_dice = aggregate["metrics"]["macro_case_dice"]["mean"]
    sd_dice = aggregate["metrics"]["macro_case_dice"]["sample_sd"]

    print()
    print("=" * 110)
    print("EVICT NOTEBOOK 05E — THREE-SEED SEGCT-CLIP VISUAL ADAPTATION — DURABLE PASS")
    print("=" * 110)

    for result in results:
        print(
            f"seed {result['seed']:4d} | "
            f"final={result['final_step']} | "
            f"stop={result['stop_reason']} | "
            f"best Dice={result['best_macro_case_dice']:.8f} "
            f"@ {result['best_step']}"
        )

    print()
    print(f"Three-seed macro case Dice: {mean_dice:.8f}")
    print(f"Sample SD                 : {sd_dice:.8f}")
    print("Claim status              : ADAPTATION, NOT EXACT REPRODUCTION")
    print("Caption bank used         : NO")
    print("Contrastive loss used     : NO")
    print("Calibration accessed      : NO")
    print("Target / MedSeg accessed  : NO")
    print("Target lock               : ACTIVE")
    print(f"GitHub HEAD               : {local_head}")
    print()
    print("NEXT: Notebook 06 — fixed biomedical text prototypes and proposed EviCT semantic branch.")


if __name__ == "__main__":
    main()
