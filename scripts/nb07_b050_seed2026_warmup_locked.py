from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from tqdm.auto import tqdm

ROOT = Path("/kaggle/working/EviCT")
WORK = Path("/kaggle/working")

CFG = ROOT / "config/notebook07_teacher_student.json"
STATE = ROOT / "artifacts/audit/notebook07_b050_seed2026_training_state.md"

MODEL_CODE = ROOT / "src/evict/models/semantic_segformer.py"
METRICS_CODE = ROOT / "src/evict/metrics.py"
PROTO = ROOT / "artifacts/audit/notebook06_text_prototypes.pt"

LABELED = (
    ROOT
    / "manifests/slice_loaders/seed_2026/b050/labeled_slices.csv"
)

UNLABELED = (
    ROOT
    / "manifests/slice_loaders/seed_2026/b050/unlabeled_slices.csv"
)

SELECTION = ROOT / "manifests/source_selection_slices.csv"
SPLITS = ROOT / "manifests/splits.csv"

AUD = ROOT / "artifacts/audit"
LARGE = ROOT / "artifacts/large/notebook07/b050/seed_2026/warmup"

AUD.mkdir(parents=True, exist_ok=True)
LARGE.mkdir(parents=True, exist_ok=True)

TRAIN_LOG = AUD / "notebook07_b050_seed2026_warmup_train_log.csv"
SEL_LOG = AUD / "notebook07_b050_seed2026_warmup_selection_metrics.csv"
CASE_LOG = AUD / "notebook07_b050_seed2026_warmup_selection_case_metrics.csv"
FINAL_AUDIT = AUD / "notebook07_b050_seed2026_warmup_final_durable.json"

LAST = LARGE / "last.pt"
RECOVERY = LARGE / "recovery.pt"
BEST = LARGE / "best.pt"
BRANCH = LARGE / "branch_step1000.pt"

SEED = 2026
BUDGET = "b050"
METHOD = "supervised_warmup"

MAX_STEP = 1000
VALIDATE_EVERY = 250
RECOVERY_EVERY = 50

MICRO = 4
LABELED_MICROBATCHES_PER_UPDATE = 2
LABELED_IMAGES_PER_UPDATE = MICRO * LABELED_MICROBATCHES_PER_UPDATE

ENC_LR = 1e-4
HEAD_LR = 3e-4
WD = 0.01
AUX_WEIGHT = 0.1

ROLLING_STEP = 500

ROLLING_TAG = "evict-nb07-b050-seed2026-warmup-rolling"
FINAL_TAG = "evict-nb07-b050-seed2026-warmup-final-step1000"


# ==========================================================================================
# Helpers
# ==========================================================================================

def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_hash(obj):
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ==========================================================================================
# Import frozen Notebook-06 / shared utilities
# ==========================================================================================

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

nb06 = load_module(
    ROOT / "scripts/nb06_semantic_branch.py",
    "evict_nb07_parent_nb06",
)

base = nb06.base

# Outer confirmatory launcher controls metadata.
base.git_sync = lambda *args, **kwargs: None


# ==========================================================================================
# Scientific-state verification
# ==========================================================================================

def verify_protocol():

    assert CFG.exists(), CFG
    assert STATE.exists(), STATE
    assert LABELED.exists(), LABELED
    assert UNLABELED.exists(), UNLABELED
    assert SELECTION.exists(), SELECTION
    assert PROTO.exists(), PROTO

    state = STATE.read_text(errors="replace")

    allowed_states = [
        "NOTEBOOK_07_READY_FOR_PILOT",
        "NOTEBOOK_07_B050_SEED2026_WARMUP_RUNNING",
        "NOTEBOOK_07_B050_SEED2026_WARMUP_FROZEN",
    ]

    assert any(x in state for x in allowed_states), (
        "Unexpected execution state.\n\n" + state[:2000]
    )

    assert "Calibration accessed:\n\nNO" in state
    assert "Target / MedSeg accessed:\n\nNO" in state
    assert "Target lock:\n\nACTIVE" in state

    cfg = json.loads(CFG.read_text())

    recorded_hash = cfg["config_hash"]

    check_cfg = dict(cfg)
    check_cfg.pop("config_hash", None)

    actual_hash = stable_hash(check_cfg)

    assert actual_hash == recorded_hash, (
        "Notebook-07 config changed after freeze.\n"
        f"Expected: {recorded_hash}\n"
        f"Actual:   {actual_hash}"
    )

    assert cfg["training"]["supervised_warmup_updates"] == 1000
    assert cfg["training"]["validation_every_updates"] == 250
    assert abs(cfg["training"]["ema_decay"] - 0.99) < 1e-12

    # ----------------------------------------------------------------------
    # Labeled side: exactly six visible fitting patients.
    # ----------------------------------------------------------------------

    lab = pd.read_csv(LABELED)

    assert "infection_cache_path" in lab.columns
    assert "label_visible" in lab.columns
    assert lab["label_visible"].astype(bool).all()

    n_lab_cases = lab["case_id"].astype(str).nunique()
    assert n_lab_cases == 6, n_lab_cases

    # ----------------------------------------------------------------------
    # Hidden side: NO infection/GT path may be exposed.
    # We inspect columns only; this warm-up never loads the hidden dataframe
    # into a training sampler.
    # ----------------------------------------------------------------------

    unlab = pd.read_csv(UNLABELED)

    assert "label_visible" in unlab.columns
    assert not unlab["label_visible"].astype(bool).any()

    forbidden = [
        "infection_cache",
        "lung_cache",
        "gt_vault",
        "ground_truth",
        "lesion_area",
        "lesion_presence",
    ]

    for col in unlab.columns:
        low = str(col).lower()
        assert not any(x in low for x in forbidden), (
            f"Hidden-label column exposed: {col}"
        )

    for col in [
        c for c in unlab.columns
        if unlab[c].dtype == object
    ]:
        strings = unlab[col].fillna("").astype(str).str.lower()

        assert not strings.str.contains(
            "gt_vault",
            regex=False,
        ).any(), f"Hidden GT path exposed in {col}"

    lab_cases = set(lab["case_id"].astype(str))
    unlab_cases = set(unlab["case_id"].astype(str))

    assert lab_cases.isdisjoint(unlab_cases)
    assert len(unlab_cases) == 6

    print("✓ Notebook-07 config        : HASH VERIFIED")
    print("✓ Visible fitting patients  : 6")
    print("✓ Hidden fitting patients   : 6")
    print("✓ Hidden mask paths exposed : NO")
    print("✓ Calibration accessed      : NO")
    print("✓ Target / MedSeg accessed  : NO")

    return cfg, lab


# ==========================================================================================
# Run hashes
# ==========================================================================================

def run_hashes(cfg):

    return {
        "config_hash": cfg["config_hash"],
        "model_code_hash": sha(MODEL_CODE),
        "metrics_code_hash": sha(METRICS_CODE),
        "prototype_file_sha256": sha(PROTO),
        "labeled_manifest_sha256": sha(LABELED),
        "unlabeled_manifest_sha256": sha(UNLABELED),
        "selection_manifest_sha256": sha(SELECTION),
        "splits_sha256": sha(SPLITS),
        "runner_sha256": sha(Path(__file__)),
    }


# ==========================================================================================
# Scheduler: same 5,000-update schedule that all branches will share.
#
# We stop this script at step 1000, but scheduler state is already on the
# declared 0 -> 5000 trajectory so all branches continue identically.
# ==========================================================================================

def scheduler_multiplier(step: int):

    warm = 200
    total = 5000

    if step < warm:
        return float(step + 1) / float(warm)

    q = (step - warm) / float(total - warm)
    q = min(max(q, 0.0), 1.0)

    return 0.5 * (1.0 + math.cos(math.pi * q))


# ==========================================================================================
# Checkpoint
# ==========================================================================================

def checkpoint_payload(
    *,
    model,
    optimizer,
    scheduler,
    sampler,
    step,
    best_score,
    best_step,
    last_validation_step,
    images_seen,
    hashes,
):

    return {
        "format_version": 1,
        "project": "EviCT",
        "stage": "NOTEBOOK_07_WARMUP",
        "method": METHOD,
        "budget_id": BUDGET,
        "seed": SEED,

        "global_step": int(step),
        "warmup_terminal_step": MAX_STEP,

        "best_score": float(best_score),
        "best_step": int(best_step),
        "last_validation_step": int(last_validation_step),

        # Patience begins AFTER the common 1000-update warm-up.
        "post_warmup_patience_count": 0,

        "labeled_images_seen": int(images_seen),
        "unlabeled_images_seen": 0,

        "student_state_dict": {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
        },

        # No EMA teacher exists during common supervised warm-up.
        "teacher_state_dict": None,

        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),

        # FP32 pilot: no AMP scaler state exists.
        "amp_enabled": False,
        "scaler_state_dict": None,

        "rng_state": base.capture_rng_state(),
        "labeled_sampler_state": sampler.state_dict(),

        **hashes,

        "calibration_accessed": False,
        "target_accessed": False,

        "timestamp_utc": now(),
    }


def save_checkpoint(path, **kwargs):
    base.atomic_torch_save(
        checkpoint_payload(**kwargs),
        path,
    )


def validate_checkpoint(q, hashes):

    assert q["stage"] == "NOTEBOOK_07_WARMUP"
    assert q["method"] == METHOD
    assert q["budget_id"] == BUDGET
    assert int(q["seed"]) == SEED

    for key, value in hashes.items():
        assert q[key] == value, (
            f"Checkpoint hash mismatch: {key}\n"
            f"checkpoint={q[key]}\n"
            f"current={value}"
        )

    assert q["calibration_accessed"] is False
    assert q["target_accessed"] is False


# ==========================================================================================
# State
# ==========================================================================================

def write_state(
    *,
    status,
    step,
    best_score,
    best_step,
    images_seen,
):

    text = f"""# EviCT Execution State

## Current stage

{status}

## Notebook 07 warm-up

Budget:

50 percent visible fitting masks (b050)

Seed:

17

Method:

Common supervised real-text warm-up

Optimizer step:

{step}

Warm-up terminal step:

1000

Best source-selection macro-case Dice:

{best_score:.8f}

Best validation step:

{best_step}

Labeled image exposures:

{images_seen}

Hidden fitting masks used:

NO

## Branching contract

The terminal step-1000 student checkpoint is the common initialization for:

1. supervised continuation
2. confidence-only EMA
3. agreement-filtered EMA

No branch may replace this warm-up with the 100%-label Notebook-06 trained checkpoint.

## Isolation

Calibration accessed:

NO

Target / MedSeg accessed:

NO

Target lock:

ACTIVE

## Next

Complete the common step-1000 warm-up, then branch the three Notebook-07 methods
without changing the patient split, optimizer schedule, text prototypes, or warm-up state.
"""

    base.atomic_text(
        STATE,
        text,
    )


# ==========================================================================================
# Release restore helpers
# ==========================================================================================

def release_headers(token):

    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def get_release(token, tag):

    headers = release_headers(token)

    url = (
        "https://api.github.com/repos/"
        "itsCodeBakery/EviCT/releases/tags/"
        + tag
    )

    r = requests.get(
        url,
        headers=headers,
        timeout=60,
    )

    if r.status_code == 404:
        return None

    r.raise_for_status()
    return r.json()


def restore_release(token, tag):

    rel = get_release(token, tag)

    if rel is None:
        return None

    headers = release_headers(token)

    assets = {
        x["name"]: x
        for x in rel.get("assets", [])
    }

    tar_name = next(
        (
            n for n in assets
            if n.endswith("_Recovery.tar")
        ),
        None,
    )

    sha_name = next(
        (
            n for n in assets
            if n.endswith(".tar.sha256")
        ),
        None,
    )

    if tar_name is None or sha_name is None:
        raise RuntimeError(
            f"Incomplete durable release: {tag}"
        )

    s = requests.get(
        assets[sha_name]["browser_download_url"],
        headers=headers,
        timeout=60,
    )
    s.raise_for_status()

    expected = s.text.strip()

    assert re.fullmatch(
        r"[0-9a-fA-F]{64}",
        expected,
    )

    dest = WORK / tar_name

    if dest.exists() and sha(dest) != expected:
        dest.unlink()

    if not dest.exists():
        base.download_verified(
            assets[tar_name]["browser_download_url"],
            dest,
            expected,
        )

    base.safe_extract_tar(
        dest,
        WORK,
    )

    dest.unlink(missing_ok=True)

    print(
        f"✓ Restored GitHub release  : {tag}"
    )

    return rel, expected


# ==========================================================================================
# Recovery archive
# ==========================================================================================

def create_archive(
    *,
    step,
    final,
):

    if final:
        source = BRANCH
        stem = (
            "EViCT_Notebook07_b050_seed2026_"
            "warmup_final_step1000_Recovery"
        )
    else:
        source = RECOVERY
        stem = (
            "EViCT_Notebook07_b050_seed2026_"
            "warmup_Rolling_Recovery"
        )

    assert source.exists()
    assert BEST.exists()

    tar_path = WORK / f"{stem}.tar"
    sha_path = Path(str(tar_path) + ".sha256")

    tar_path.unlink(missing_ok=True)
    sha_path.unlink(missing_ok=True)

    # Keep best checkpoint slim inside remote archive.
    q = torch.load(
        BEST,
        map_location="cpu",
        weights_only=False,
    )

    slim_best = WORK / "nb07_b050_seed2026_warmup_best_tmp.pt"

    base.atomic_torch_save(
        {
            "stage": "NOTEBOOK_07_WARMUP",
            "method": METHOD,
            "budget_id": BUDGET,
            "seed": SEED,
            "global_step": int(q["global_step"]),
            "best_score": float(q["best_score"]),
            "best_step": int(q["best_step"]),
            "student_state_dict": q["student_state_dict"],
            "calibration_accessed": False,
            "target_accessed": False,
        },
        slim_best,
    )

    del q

    with tarfile.open(tar_path, "w") as tar:

        tar.add(
            source,
            arcname=str(
                Path("EviCT")
                / source.relative_to(ROOT)
            ),
        )

        tar.add(
            slim_best,
            arcname=str(
                Path("EviCT")
                / BEST.relative_to(ROOT)
            ),
        )

        for p in [
            TRAIN_LOG,
            SEL_LOG,
            CASE_LOG,
        ]:
            if p.exists():
                tar.add(
                    p,
                    arcname=str(
                        Path("EviCT")
                        / p.relative_to(ROOT)
                    ),
                )

    slim_best.unlink(missing_ok=True)

    digest = sha(tar_path)

    base.atomic_text(
        sha_path,
        digest + "\n",
    )

    return tar_path, sha_path, digest


def publish_release(
    *,
    token,
    step,
    final,
    best_score,
    best_step,
):

    tar_path, sha_path, digest = create_archive(
        step=step,
        final=final,
    )

    if final:

        tag = FINAL_TAG

        name = (
            "EviCT Notebook 07 — b050 seed2026 "
            "common supervised warm-up — final step 1000"
        )

    else:

        tag = ROLLING_TAG

        name = (
            "EviCT Notebook 07 — b050 seed2026 "
            "common supervised warm-up — rolling"
        )

    body = (
        f"Notebook 07 common supervised warm-up. "
        f"Budget=b050, seed=17, step={step}, "
        f"best source-selection macro-case Dice="
        f"{best_score:.8f}@{best_step}. "
        f"Only 6 visible fitting patients were used. "
        f"Hidden fitting masks, source calibration, "
        f"and target/MedSeg were not accessed."
    )

    release, headers, api = (
        base.create_or_get_release(
            token,
            tag,
            name,
            body,
        )
    )

    base.upload_asset(
        release=release,
        headers=headers,
        api=api,
        file_path=tar_path,
        content_type="application/x-tar",
    )

    base.upload_asset(
        release=release,
        headers=headers,
        api=api,
        file_path=sha_path,
        content_type="text/plain",
    )

    tar_path.unlink(missing_ok=True)
    sha_path.unlink(missing_ok=True)

    print(
        f"✓ Durable release          : {release['html_url']}"
    )

    return release["html_url"], digest


# ==========================================================================================
# Finalize an already-restored terminal release
# ==========================================================================================

def finalize_restored_final(
    *,
    rel,
    digest,
    hashes,
):

    assert BRANCH.exists()

    q = torch.load(
        BRANCH,
        map_location="cpu",
        weights_only=False,
    )

    validate_checkpoint(q, hashes)

    assert int(q["global_step"]) == 1000

    best_score = float(q["best_score"])
    best_step = int(q["best_step"])
    images_seen = int(q["labeled_images_seen"])

    result = {
        "timestamp_utc": now(),
        "stage": "NOTEBOOK_07_WARMUP",
        "status": "DURABLE_COMPLETE",
        "method": METHOD,
        "budget_id": BUDGET,
        "training_mask_fraction": 0.50,
        "seed": SEED,

        "visible_fitting_cases": 6,
        "hidden_fitting_cases": 6,

        "final_step": 1000,
        "best_step": best_step,
        "best_source_selection_macro_case_dice": best_score,

        "labeled_images_seen": images_seen,
        "unlabeled_images_seen": 0,

        "branch_checkpoint": str(
            BRANCH.relative_to(ROOT)
        ),
        "branch_checkpoint_sha256": sha(BRANCH),

        "release_url": rel["html_url"],
        "archive_sha256": digest,

        **hashes,

        "calibration_accessed": False,
        "target_accessed": False,
    }

    base.atomic_text(
        FINAL_AUDIT,
        json.dumps(result, indent=2) + "\n",
    )

    write_state(
        status="NOTEBOOK_07_B050_SEED2026_WARMUP_FROZEN",
        step=1000,
        best_score=best_score,
        best_step=best_step,
        images_seen=images_seen,
    )

    base.git_sync(
        "Freeze Notebook 07 b050 seed2026 common warm-up"
    )

    print(
        "✓ Restored terminal warm-up : STEP 1000 FROZEN"
    )

    return result


# ==========================================================================================
# Main
# ==========================================================================================

def main():

    print("=" * 114)
    print(
        "EVICT NOTEBOOK 07B — b050 SEED2026 "
        "COMMON SUPERVISED WARM-UP"
    )
    print("=" * 114)

    from kaggle_secrets import UserSecretsClient

    token = UserSecretsClient().get_secret("pushEviCT")

    assert token, "Kaggle secret pushEviCT unavailable."
    assert torch.cuda.is_available(), "CUDA unavailable."

    device = torch.device("cuda:0")

    print(
        "✓ GPU                       :",
        torch.cuda.get_device_name(0),
    )

    cfg, train_df = verify_protocol()
    hashes = run_hashes(cfg)

    # ----------------------------------------------------------------------
    # Ensure source cache.
    # ----------------------------------------------------------------------

    nb06.ensure_cache()

    # ----------------------------------------------------------------------
    # Selection only. No calibration dataframe is loaded.
    # ----------------------------------------------------------------------

    selection_df = pd.read_csv(SELECTION)

    assert (
        selection_df["case_id"]
        .astype(str)
        .nunique()
        == 4
    )

    # ----------------------------------------------------------------------
    # Frozen semantic prototypes + MiT-B1 ImageNet encoder.
    #
    # CRITICAL:
    # No Notebook-06 trained 100%-label checkpoint is loaded here.
    # ----------------------------------------------------------------------

    proto = nb06.text_prototypes()

    snap = nb06.mit_snapshot()

    nb06.smoke(
        snap,
        proto,
        device,
    )

    # ----------------------------------------------------------------------
    # If final warm-up release already exists, restore it and stop.
    # ----------------------------------------------------------------------

    final_restored = restore_release(
        token,
        FINAL_TAG,
    )

    if final_restored is not None:

        rel, digest = final_restored

        result = finalize_restored_final(
            rel=rel,
            digest=digest,
            hashes=hashes,
        )

        print()
        print(json.dumps(result, indent=2))
        return

    # ----------------------------------------------------------------------
    # Otherwise recover rolling step if local state vanished.
    # ----------------------------------------------------------------------

    if not RECOVERY.exists():

        rolling = restore_release(
            token,
            ROLLING_TAG,
        )

        if rolling is not None:
            print(
                "✓ Rolling warm-up state     : RECOVERED"
            )

    # ----------------------------------------------------------------------
    # Imports used for training / selection.
    # ----------------------------------------------------------------------

    from evict.models.semantic_segformer import (
        masked_supervised_loss,
        semantic_patch_auxiliary_loss,
    )

    from evict.metrics import (
        CaseMetricAccumulator,
        binary_segmentation_metrics,
        macro_case_summary,
        metrics_from_confusion,
    )

    # ----------------------------------------------------------------------
    # Exact low-label sampling trajectory.
    # ----------------------------------------------------------------------

    base.reset_rng(SEED)

    store = base.FittingCaseStore(
        train_df
    )

    sampler = base.PatientUniformSampler(
        store,
        SEED,
    )

    assert len(store.case_ids) == 6

    # ----------------------------------------------------------------------
    # Fresh ImageNet + frozen-real-text initialization.
    # ----------------------------------------------------------------------

    model = nb06.build(
        snap,
        proto,
        "real_text",
        device,
    )

    encoder_params = list(
        model.encoder.parameters()
    )

    head_params = [
        p
        for name, p in model.named_parameters()
        if not name.startswith("encoder.")
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_params,
                "lr": ENC_LR,
            },
            {
                "params": head_params,
                "lr": HEAD_LR,
            },
        ],
        weight_decay=WD,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=scheduler_multiplier,
    )

    # ----------------------------------------------------------------------
    # State defaults.
    # ----------------------------------------------------------------------

    step = 0
    best_score = -1.0
    best_step = 0
    last_validation = 0
    images_seen = 0

    train_rows = (
        []
        if not TRAIN_LOG.exists()
        else pd.read_csv(TRAIN_LOG).to_dict("records")
    )

    sel_rows = (
        []
        if not SEL_LOG.exists()
        else pd.read_csv(SEL_LOG).to_dict("records")
    )

    case_rows = (
        []
        if not CASE_LOG.exists()
        else pd.read_csv(CASE_LOG).to_dict("records")
    )

    # ----------------------------------------------------------------------
    # Local or restored recovery.
    # ----------------------------------------------------------------------

    if RECOVERY.exists():

        q = torch.load(
            RECOVERY,
            map_location="cpu",
            weights_only=False,
        )

        validate_checkpoint(
            q,
            hashes,
        )

        model.load_state_dict(
            q["student_state_dict"]
        )

        optimizer.load_state_dict(
            q["optimizer_state_dict"]
        )

        # Move optimizer tensors back to active GPU.
        for state_item in optimizer.state.values():
            for key, value in list(
                state_item.items()
            ):
                if torch.is_tensor(value):
                    state_item[key] = value.to(
                        device
                    )

        scheduler.load_state_dict(
            q["scheduler_state_dict"]
        )

        sampler.load_state_dict(
            q["labeled_sampler_state"]
        )

        base.restore_rng_state(
            q["rng_state"]
        )

        step = int(q["global_step"])
        best_score = float(q["best_score"])
        best_step = int(q["best_step"])
        last_validation = int(
            q["last_validation_step"]
        )
        images_seen = int(
            q["labeled_images_seen"]
        )

        train_rows = [
            r
            for r in train_rows
            if int(r["step"]) <= step
        ]

        sel_rows = [
            r
            for r in sel_rows
            if int(r["step"]) <= last_validation
        ]

        print(
            f"✓ Warm-up resumed           : step {step}"
        )

        del q

    else:

        print(
            "✓ Warm-up initialization    : NEW STEP 0"
        )
        print(
            "✓ Full-label NB06 weights   : NOT LOADED"
        )

    if step > MAX_STEP:
        raise RuntimeError(
            f"Invalid warm-up step: {step}"
        )

    # ----------------------------------------------------------------------
    # Save helper.
    # ----------------------------------------------------------------------

    def save(path):

        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            step=step,
            best_score=best_score,
            best_step=best_step,
            last_validation_step=last_validation,
            images_seen=images_seen,
            hashes=hashes,
        )

    # ----------------------------------------------------------------------
    # Training.
    #
    # One low-label update = approximately 8 LABELED images during common
    # supervised warm-up: two micro-batches × four images.
    # ----------------------------------------------------------------------

    bar = tqdm(
        total=MAX_STEP - step,
        desc="NB07 b050 seed2026 warm-up",
        unit="update",
    )

    while step < MAX_STEP:

        model.train()
        optimizer.zero_grad(
            set_to_none=True
        )

        total_loss = 0.0
        sup_loss_total = 0.0
        text_loss_total = 0.0
        selected_patches = 0

        for _ in range(
            LABELED_MICROBATCHES_PER_UPDATE
        ):

            xn, yn, vn = sampler.sample(
                MICRO
            )

            x = torch.from_numpy(
                xn
            ).to(
                device=device,
                dtype=torch.float32,
            )

            y = torch.from_numpy(
                yn
            ).to(
                device=device,
                dtype=torch.float32,
            )

            vm = torch.from_numpy(
                vn
            ).to(
                device=device,
                dtype=torch.float32,
            )

            x, y, vm = base.augment(
                x,
                y,
                vm,
            )

            out = model(
                base.normalize_batch(
                    x,
                    device,
                )
            )

            sup = masked_supervised_loss(
                out["logits"],
                y,
                vm,
            )

            aux = semantic_patch_auxiliary_loss(
                out["semantic_quarter_logits"],
                y,
                vm,
            )

            loss = (
                sup["loss"]
                + AUX_WEIGHT * aux["loss"]
            )

            assert torch.isfinite(
                loss
            ).item()

            (
                loss
                / LABELED_MICROBATCHES_PER_UPDATE
            ).backward()

            total_loss += (
                float(
                    loss.detach().cpu()
                )
                / LABELED_MICROBATCHES_PER_UPDATE
            )

            sup_loss_total += (
                float(
                    sup["loss"]
                    .detach()
                    .cpu()
                )
                / LABELED_MICROBATCHES_PER_UPDATE
            )

            text_loss_total += (
                float(
                    aux["loss"]
                    .detach()
                    .cpu()
                )
                / LABELED_MICROBATCHES_PER_UPDATE
            )

            selected_patches += int(
                aux["selected_patches"]
            )

            del x, y, vm, out, sup, aux, loss

        # ------------------------------------------------------------------
        # Gradient integrity.
        # ------------------------------------------------------------------

        assert all(
            torch.isfinite(p.grad).all().item()
            for p in model.parameters()
            if p.grad is not None
        )

        enc_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        head_lr = float(
            optimizer.param_groups[1]["lr"]
        )

        optimizer.step()
        scheduler.step()

        step += 1
        images_seen += LABELED_IMAGES_PER_UPDATE

        alpha = float(
            torch.sigmoid(
                model.alpha_logit.detach()
            )
            .cpu()
        )

        train_rows.append(
            {
                "step": step,
                "loss": total_loss,
                "supervised_loss": sup_loss_total,
                "text_aux_loss": text_loss_total,
                "unsupervised_loss": 0.0,
                "lambda_u": 0.0,

                "accepted_foreground_pixels": 0,
                "accepted_background_pixels": 0,
                "accepted_foreground_fraction": 0.0,
                "accepted_background_fraction": 0.0,
                "zero_acceptance_event": False,

                "alpha": alpha,
                "selected_aux_patches": selected_patches,

                "encoder_lr": enc_lr,
                "head_lr": head_lr,

                "labeled_images_seen": images_seen,
                "unlabeled_images_seen": 0,
            }
        )

        # ------------------------------------------------------------------
        # Local recovery every 50.
        # ------------------------------------------------------------------

        if step % RECOVERY_EVERY == 0:

            save(LAST)
            save(RECOVERY)

            pd.DataFrame(
                train_rows
            ).to_csv(
                TRAIN_LOG,
                index=False,
            )

        bar.update(1)

        bar.set_postfix(
            step=step,
            loss=f"{total_loss:.4f}",
            best=f"{best_score:.4f}",
            alpha=f"{alpha:.3f}",
        )

        # ------------------------------------------------------------------
        # Source-selection validation every 250.
        # ------------------------------------------------------------------

        if step % VALIDATE_EVERY == 0:

            metrics, per_case, _ = base.validate(
                model=model,
                device=device,
                selection_df=selection_df,
                masked_supervised_loss=masked_supervised_loss,
                CaseMetricAccumulator=CaseMetricAccumulator,
                binary_segmentation_metrics=binary_segmentation_metrics,
                metrics_from_confusion=metrics_from_confusion,
                macro_case_summary=macro_case_summary,
            )

            last_validation = step

            score = float(
                metrics["macro_case_dice"]
            )

            improved = score > best_score

            if improved:

                best_score = score
                best_step = step

                save(BEST)

            sel_rows.append(
                {
                    "step": step,
                    "macro_case_dice":
                        score,
                    "macro_case_iou":
                        metrics["macro_case_iou"],
                    "macro_case_sensitivity":
                        metrics["macro_case_sensitivity"],
                    "macro_case_specificity":
                        metrics["macro_case_specificity"],
                    "macro_slice_dice":
                        metrics["macro_slice_dice"],
                    "pooled_dice":
                        metrics["pooled_dice"],
                    "pooled_iou":
                        metrics["pooled_iou"],
                    "selection_loss":
                        metrics["selection_loss"],
                    "best_score":
                        best_score,
                    "best_step":
                        best_step,
                    "alpha":
                        alpha,
                }
            )

            for row in per_case:
                case_rows.append(
                    {
                        "step": step,
                        **row,
                    }
                )

            pd.DataFrame(
                train_rows
            ).to_csv(
                TRAIN_LOG,
                index=False,
            )

            pd.DataFrame(
                sel_rows
            ).to_csv(
                SEL_LOG,
                index=False,
            )

            pd.DataFrame(
                case_rows
            ).to_csv(
                CASE_LOG,
                index=False,
            )

            save(LAST)
            save(RECOVERY)

            write_state(
                status=(
                    "NOTEBOOK_07_B050_SEED2026_"
                    "WARMUP_RUNNING"
                ),
                step=step,
                best_score=best_score,
                best_step=best_step,
                images_seen=images_seen,
            )

            print()
            print(
                f"[NB07 b050 s17] VAL {step}: "
                f"Dice={score:.8f}, "
                f"best={best_score:.8f}@{best_step}, "
                f"alpha={alpha:.4f}"
            )

            # Small audit/log files go to ordinary Git.
            base.git_sync(
                f"Notebook 07 b050 seed2026 warm-up step {step}"
            )

            # --------------------------------------------------------------
            # Remote rolling recovery exactly halfway.
            # --------------------------------------------------------------

            if step == ROLLING_STEP:

                url, digest = publish_release(
                    token=token,
                    step=step,
                    final=False,
                    best_score=best_score,
                    best_step=best_step,
                )

                rolling_audit = {
                    "timestamp_utc": now(),
                    "stage": "NOTEBOOK_07_WARMUP",
                    "status": "DURABLE_ROLLING",
                    "method": METHOD,
                    "budget_id": BUDGET,
                    "seed": SEED,
                    "step": step,
                    "best_score": best_score,
                    "best_step": best_step,
                    "release_url": url,
                    "archive_sha256": digest,
                    "calibration_accessed": False,
                    "target_accessed": False,
                }

                base.atomic_text(
                    AUD
                    / "notebook07_b050_seed2026_"
                      "warmup_rolling_durable.json",
                    json.dumps(
                        rolling_audit,
                        indent=2,
                    )
                    + "\n",
                )

                base.git_sync(
                    "Record Notebook 07 b050 seed2026 "
                    "warm-up rolling recovery"
                )

    bar.close()

    # ======================================================================================
    # Fixed terminal warm-up point
    #
    # This is NOT the best-selection checkpoint.
    # This exact terminal step-1000 state becomes the common branch point.
    # ======================================================================================

    save(LAST)
    save(RECOVERY)
    save(BRANCH)

    assert int(step) == 1000
    assert BRANCH.exists()
    assert BEST.exists()

    # ======================================================================================
    # Final durable release
    # ======================================================================================

    release_url, archive_digest = publish_release(
        token=token,
        step=step,
        final=True,
        best_score=best_score,
        best_step=best_step,
    )

    result = {
        "timestamp_utc": now(),
        "stage": "NOTEBOOK_07_WARMUP",
        "status": "DURABLE_COMPLETE",

        "method": METHOD,
        "budget_id": BUDGET,
        "training_mask_fraction": 0.50,
        "seed": SEED,

        "visible_fitting_cases": 6,
        "hidden_fitting_cases": 6,

        "final_step": step,

        "best_step": best_step,
        "best_source_selection_macro_case_dice":
            best_score,

        "labeled_images_seen":
            images_seen,

        "unlabeled_images_seen":
            0,

        "branch_checkpoint":
            str(
                BRANCH.relative_to(ROOT)
            ),

        "branch_checkpoint_sha256":
            sha(BRANCH),

        "release_url":
            release_url,

        "archive_sha256":
            archive_digest,

        **hashes,

        "calibration_accessed": False,
        "target_accessed": False,
    }

    base.atomic_text(
        FINAL_AUDIT,
        json.dumps(
            result,
            indent=2,
        )
        + "\n",
    )

    write_state(
        status="NOTEBOOK_07_B050_SEED2026_WARMUP_FROZEN",
        step=step,
        best_score=best_score,
        best_step=best_step,
        images_seen=images_seen,
    )

    base.git_sync(
        "Freeze Notebook 07 b050 seed2026 common warm-up"
    )

    print()
    print("=" * 114)
    print("✅ NOTEBOOK 07 COMMON WARM-UP FROZEN")
    print("=" * 114)
    print(
        f"Best source-selection Dice : "
        f"{best_score:.8f} @ {best_step}"
    )
    print(
        f"Terminal branch step       : {step}"
    )
    print(
        f"Labeled image exposures    : {images_seen}"
    )
    print(
        "Hidden masks used          : NO"
    )
    print(
        "Calibration accessed       : NO"
    )
    print(
        "Target / MedSeg accessed   : NO"
    )
    print(
        f"Durable release            : {release_url}"
    )
    print()
    print(
        "NEXT: branch this exact step-1000 checkpoint into:"
    )
    print(
        "  supervised continuation"
    )
    print(
        "  confidence-only EMA"
    )
    print(
        "  agreement-filtered EMA"
    )
    print("=" * 114)

    print()
    print(json.dumps(result, indent=2))

    del model
    del optimizer
    del scheduler
    del sampler
    del store

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
