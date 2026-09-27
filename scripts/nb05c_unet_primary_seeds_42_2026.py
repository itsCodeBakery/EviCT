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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from kaggle_secrets import UserSecretsClient


ROOT = Path("/kaggle/working/EviCT")
WORK = Path("/kaggle/working")

BASE_RUNNER_PATH = ROOT / "scripts/nb05b_unet_lr_pilot_segment1.py"
CONFIG_PATH = ROOT / "config/notebook05_unet_baseline.json"
SELECTED_LR_PATH = ROOT / "config/notebook05_unet_selected_lr.json"
MODEL_CODE = ROOT / "src/evict/models/unet2d.py"
METRICS_CODE = ROOT / "src/evict/metrics.py"
SPLITS_PATH = ROOT / "manifests/splits.csv"
SELECTION_MANIFEST = ROOT / "manifests/source_selection_slices.csv"
STATE_PATH = ROOT / "STATE.md"
HANDOFF_STATE = ROOT / "handoff/STATE.md"
GIT_SYNC = ROOT / "scripts/git_sync.py"

RUN_ROOT = ROOT / "artifacts/large/notebook05"
PRED_ROOT = ROOT / "predictions/notebook05"
AUDIT_ROOT = ROOT / "artifacts/audit"
TABLE_ROOT = ROOT / "tables"
REPORT_ROOT = ROOT / "reports"

SEEDS = [42, 2026]
PRIMARY_SEEDS = [17, 42, 2026]
FROZEN_LR = 3.0e-4

MAX_UPDATES = 5000
WARMUP_UPDATES = 200
VALIDATE_EVERY = 250
RECOVERY_EVERY = 50
PATIENCE_LIMIT = 8
MICRO_BATCH = 4
GRAD_ACCUM = 4
EFFECTIVE_BATCH = 16
THRESHOLD = 0.5

REMOTE_MILESTONES = {1000, 2000, 3000, 4000}

REPO_OWNER = "itsCodeBakery"
REPO_NAME = "EviCT"


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "evict_nb05b_base",
        str(BASE_RUNNER_PATH),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import base runner: {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def train_manifest(seed: int) -> Path:
    return ROOT / f"manifests/slice_loaders/seed_{seed}/b100/labeled_slices.csv"


def paths_for(seed: int) -> dict:
    run_dir = RUN_ROOT / f"unet_seed{seed}_lr3e4"
    pred_dir = PRED_ROOT / f"unet_seed{seed}_lr3e4"
    prefix = f"notebook05c_unet_seed{seed}_lr3e4"
    return {
        "run_dir": run_dir,
        "pred_dir": pred_dir,
        "best_pt": run_dir / "best.pt",
        "last_pt": run_dir / "last.pt",
        "recovery_pt": run_dir / "recovery.pt",
        "train_log": AUDIT_ROOT / f"{prefix}_train_log.csv",
        "selection_log": AUDIT_ROOT / f"{prefix}_selection_metrics.csv",
        "case_log": AUDIT_ROOT / f"{prefix}_selection_case_metrics.csv",
        "best_logits_manifest": AUDIT_ROOT / f"{prefix}_best_logits_manifest.json",
        "rolling_audit": AUDIT_ROOT / f"{prefix}_rolling_durable.json",
        "final_audit": AUDIT_ROOT / f"{prefix}_final_durable.json",
    }


def direct_release_url(tag: str, asset_name: str) -> str:
    return (
        f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/"
        f"{tag}/{asset_name}"
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
    frozen_lr_hash: str,
    split_hash: str,
    model_code_hash: str,
    metrics_code_hash: str,
    init_hash: str,
):
    return {
        "format_version": 1,
        "project": "EviCT",
        "stage": "NOTEBOOK_05C",
        "run_id": f"unet2d_seed{seed}_lr3e4_fp32",
        "candidate_name": "lr3e4",
        "learning_rate": float(FROZEN_LR),
        "seed": int(seed),
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
        "rng_state": base.capture_rng_state(),
        "sampler_state": sampler.state_dict(),
        "config_hash": config_hash,
        "frozen_lr_hash": frozen_lr_hash,
        "split_hash": split_hash,
        "model_code_hash": model_code_hash,
        "metrics_code_hash": metrics_code_hash,
        "initialization_hash": init_hash,
        "threshold": THRESHOLD,
        "architecture_tuning": False,
        "threshold_tuned": False,
        "target_accessed": False,
        "calibration_accessed": False,
    }


def validate_payload(
    payload: dict,
    *,
    seed: int,
    config_hash: str,
    frozen_lr_hash: str,
    split_hash: str,
    model_code_hash: str,
    metrics_code_hash: str,
):
    assert payload["project"] == "EviCT"
    assert payload["stage"] == "NOTEBOOK_05C"
    assert payload["candidate_name"] == "lr3e4"
    assert abs(float(payload["learning_rate"]) - FROZEN_LR) < 1e-15
    assert int(payload["seed"]) == int(seed)
    assert int(payload["split_seed"]) == 17
    assert payload["precision"] == "FP32"
    assert payload["amp_enabled"] is False
    assert payload["config_hash"] == config_hash
    assert payload["frozen_lr_hash"] == frozen_lr_hash
    assert payload["split_hash"] == split_hash
    assert payload["model_code_hash"] == model_code_hash
    assert payload["metrics_code_hash"] == metrics_code_hash
    assert float(payload["threshold"]) == THRESHOLD
    assert payload["architecture_tuning"] is False
    assert payload["threshold_tuned"] is False
    assert payload["target_accessed"] is False
    assert payload["calibration_accessed"] is False


def build_components(seed: int, train_df: pd.DataFrame, device: torch.device):
    from evict.models.unet2d import EviCTResidualUNet2D

    base.reset_rng(seed)

    store = base.FittingCaseStore(train_df)
    sampler = base.PatientUniformSampler(store, seed)

    model = EviCTResidualUNet2D(
        in_channels=3,
        base_channels=32,
    )

    init_hash = base.model_state_hash(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FROZEN_LR,
        weight_decay=base.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=base.scheduler_multiplier,
    )

    return store, sampler, model, optimizer, scheduler, init_hash


def restore_checkpoint(
    *,
    seed: int,
    path: Path,
    train_df: pd.DataFrame,
    device: torch.device,
    config_hash: str,
    frozen_lr_hash: str,
    split_hash: str,
    model_code_hash: str,
    metrics_code_hash: str,
):
    store, sampler, model, optimizer, scheduler, init_hash = build_components(
        seed,
        train_df,
        device,
    )

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    validate_payload(
        payload,
        seed=seed,
        config_hash=config_hash,
        frozen_lr_hash=frozen_lr_hash,
        split_hash=split_hash,
        model_code_hash=model_code_hash,
        metrics_code_hash=metrics_code_hash,
    )

    assert payload["initialization_hash"] == init_hash

    model.load_state_dict(payload["student_state_dict"])
    model.to(device)

    optimizer.load_state_dict(payload["optimizer_state_dict"])
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)

    scheduler.load_state_dict(payload["scheduler_state_dict"])
    sampler.load_state_dict(payload["sampler_state"])
    base.restore_rng_state(payload["rng_state"])

    state = {
        "global_step": int(payload["global_step"]),
        "best_score": float(payload["best_score"]),
        "best_step": int(payload["best_step"]),
        "patience_count": int(payload["patience_count"]),
        "last_validation_step": int(payload["last_validation_step"]),
        "images_seen": int(payload["images_seen"]),
    }

    del payload

    return store, sampler, model, optimizer, scheduler, init_hash, state


def checkpoint_metadata(
    *,
    seed: int,
    path: Path,
    config_hash: str,
    frozen_lr_hash: str,
    split_hash: str,
    model_code_hash: str,
    metrics_code_hash: str,
):
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    try:
        validate_payload(
            payload,
            seed=seed,
            config_hash=config_hash,
            frozen_lr_hash=frozen_lr_hash,
            split_hash=split_hash,
            model_code_hash=model_code_hash,
            metrics_code_hash=metrics_code_hash,
        )
        return {
            "path": path,
            "step": int(payload["global_step"]),
            "best_score": float(payload["best_score"]),
            "best_step": int(payload["best_step"]),
        }
    finally:
        del payload


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
        df = df.sort_values("step").drop_duplicates("step", keep="last")
        train_rows[:] = df.to_dict("records")
    if selection_rows:
        df = pd.DataFrame(selection_rows)
        df = df.sort_values("step").drop_duplicates("step", keep="last")
        selection_rows[:] = df.to_dict("records")
    if case_rows:
        df = pd.DataFrame(case_rows)
        df = (
            df.sort_values(["step", "case_id"])
            .drop_duplicates(["step", "case_id"], keep="last")
        )
        case_rows[:] = df.to_dict("records")


def write_logs(p: dict, train_rows, selection_rows, case_rows):
    pd.DataFrame(train_rows).to_csv(p["train_log"], index=False)
    pd.DataFrame(selection_rows).to_csv(p["selection_log"], index=False)
    pd.DataFrame(case_rows).to_csv(p["case_log"], index=False)


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

Competitive residual 2D U-Net

## Frozen protocol

Architecture:

Residual 2D U-Net with GroupNorm

Learning rate:

0.00030000

Learning-rate status:

FROZEN FROM SEED-17 PILOT

Training seed:

{seed}

Split seed:

17

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

Threshold tuning:

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

## Next

Complete the frozen U-Net primary-seed runs for seeds 42 and 2026,
then aggregate seeds 17, 42, and 2026 without changing the protocol.
"""
    base.atomic_text(STATE_PATH, text)
    base.atomic_text(HANDOFF_STATE, text)


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
        np.save(path, np.asarray(arr, dtype=np.float32))
        cases.append({
            "case_id": case_id,
            "shape": list(arr.shape),
            "dtype": "float32",
            "sha256": base.sha256_file(path),
        })

    if best_dir.exists():
        shutil.rmtree(best_dir)
    os.replace(tmp_dir, best_dir)

    manifest = {
        "timestamp_utc": base.utc_now(),
        "stage": "NOTEBOOK_05C",
        "seed": int(seed),
        "learning_rate": FROZEN_LR,
        "step": int(step),
        "threshold": THRESHOLD,
        "calibration_accessed": False,
        "target_accessed": False,
        "cases": cases,
    }
    base.atomic_text(
        p["best_logits_manifest"],
        json.dumps(manifest, indent=2),
    )


def one_update(
    *,
    model,
    optimizer,
    scheduler,
    sampler,
    device,
    masked_supervised_loss,
):
    optimizer.zero_grad(set_to_none=True)

    accum_loss = 0.0
    accum_dice = 0.0
    accum_bce = 0.0

    for _ in range(GRAD_ACCUM):
        images_np, targets_np, valids_np = sampler.sample(MICRO_BATCH)

        images = torch.from_numpy(images_np)
        targets = torch.from_numpy(targets_np)
        valids = torch.from_numpy(valids_np)

        images, targets, valids = base.augment(
            images,
            targets,
            valids,
        )

        images = images.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        targets = targets.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        valids = valids.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        inputs = base.normalize_batch(images, device)
        logits = model(inputs)["logits"]

        loss_dict = masked_supervised_loss(
            logits,
            targets,
            valids,
        )

        loss = loss_dict["loss"] / GRAD_ACCUM
        assert torch.isfinite(loss)
        loss.backward()

        accum_loss += (
            float(loss_dict["loss"].detach().cpu())
            / GRAD_ACCUM
        )
        accum_dice += (
            float(loss_dict["dice_loss"].detach().cpu())
            / GRAD_ACCUM
        )
        accum_bce += (
            float(loss_dict["bce_loss"].detach().cpu())
            / GRAD_ACCUM
        )

        del images, targets, valids, inputs, logits, loss_dict, loss

    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()

    optimizer.step()
    scheduler.step()

    return {
        "loss": accum_loss,
        "dice_loss": accum_dice,
        "bce_loss": accum_bce,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
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
    frozen_lr_hash: str,
    split_hash: str,
    model_code_hash: str,
    metrics_code_hash: str,
    init_hash: str,
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
        frozen_lr_hash=frozen_lr_hash,
        split_hash=split_hash,
        model_code_hash=model_code_hash,
        metrics_code_hash=metrics_code_hash,
        init_hash=init_hash,
    )
    base.atomic_torch_save(payload, p["recovery_pt"])
    if write_last:
        base.atomic_torch_save(payload, p["last_pt"])
    del payload


def create_recovery_archive(
    *,
    seed: int,
    p: dict,
    step: int,
    final: bool,
):
    if final:
        name = (
            f"EviCT_Notebook05C_unet_seed{seed}_lr3e4_"
            f"final_step{step}_Recovery.tar"
        )
    else:
        name = (
            f"EviCT_Notebook05C_unet_seed{seed}_lr3e4_"
            "Rolling_Recovery.tar"
        )

    archive_path = WORK / name
    archive_path.unlink(missing_ok=True)

    with tarfile.open(archive_path, "w") as archive:
        checkpoint_paths = [
            p["recovery_pt"],
            p["best_pt"],
        ]
        if final:
            checkpoint_paths.append(p["last_pt"])

        for checkpoint in checkpoint_paths:
            assert checkpoint.exists()
            archive.add(
                checkpoint,
                arcname=str(
                    Path("EviCT/artifacts/large/notebook05")
                    / p["run_dir"].name
                    / checkpoint.name
                ),
            )

        for source in [
            p["train_log"],
            p["selection_log"],
            p["case_log"],
            p["best_logits_manifest"],
        ]:
            if source.exists():
                archive.add(
                    source,
                    arcname=str(
                        Path("EviCT/artifacts/audit")
                        / source.name
                    ),
                )

        if final:
            best_logits_dir = p["pred_dir"] / "best_selection_logits"
            assert best_logits_dir.exists()
            archive.add(
                best_logits_dir,
                arcname=str(
                    Path("EviCT/predictions/notebook05")
                    / p["pred_dir"].name
                    / "best_selection_logits"
                ),
            )

        for source in [
            CONFIG_PATH,
            SELECTED_LR_PATH,
            MODEL_CODE,
            METRICS_CODE,
            SPLITS_PATH,
            train_manifest(seed),
            SELECTION_MANIFEST,
        ]:
            if source.exists():
                archive.add(
                    source,
                    arcname=str(
                        Path("EViCT") / source.relative_to(ROOT)
                    ),
                )

    digest = base.sha256_file(archive_path)
    sha_path = Path(str(archive_path) + ".sha256")
    base.atomic_text(sha_path, digest)
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
    archive_path, sha_path, digest = create_recovery_archive(
        seed=seed,
        p=p,
        step=step,
        final=False,
    )

    tag = f"evict-nb05c-unet-seed{seed}-lr3e4-rolling"

    release, headers, api = base.create_or_get_release(
        token,
        tag,
        f"EviCT Notebook 05C — U-Net seed {seed} — Rolling Recovery",
        (
            "Rolling durable recovery for the frozen U-Net primary-seed run. "
            f"seed={seed}, lr=0.0003, optimizer step={step}, "
            f"best source-selection macro case Dice={best_score:.8f} "
            f"at step {best_step}. Calibration and target/MedSeg were not accessed."
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
        "stage": "NOTEBOOK_05C",
        "seed": int(seed),
        "learning_rate": FROZEN_LR,
        "step": int(step),
        "best_macro_case_dice": float(best_score),
        "best_step": int(best_step),
        "archive_name": archive_path.name,
        "archive_sha256": digest,
        "release_tag": tag,
        "release_url": release["html_url"],
        "tar_asset_id": int(tar_asset["id"]),
        "sha_asset_id": int(sha_asset["id"]),
        "calibration_accessed": False,
        "target_accessed": False,
        "status": "ROLLING_DURABLE",
    }

    base.atomic_text(
        p["rolling_audit"],
        json.dumps(audit, indent=2),
    )
    base.git_sync(
        f"Record Notebook 05C U-Net seed{seed} rolling recovery step {step}"
    )

    archive_path.unlink(missing_ok=True)
    sha_path.unlink(missing_ok=True)

    print(f"✓ seed {seed} rolling recovery durable @ step {step}")


def restore_remote_if_needed(
    *,
    seed: int,
    p: dict,
):
    p["run_dir"].mkdir(parents=True, exist_ok=True)
    p["pred_dir"].mkdir(parents=True, exist_ok=True)

    if p["recovery_pt"].exists() or p["last_pt"].exists():
        return

    if not p["rolling_audit"].exists():
        return

    audit = json.loads(
        p["rolling_audit"].read_text(encoding="utf-8")
    )
    if audit.get("status") != "ROLLING_DURABLE":
        return

    print(
        f"Restoring seed {seed} from remote rolling recovery "
        f"at step {audit['step']}..."
    )

    archive_path = WORK / audit["archive_name"]

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
    archive_path, sha_path, digest = create_recovery_archive(
        seed=seed,
        p=p,
        step=final_step,
        final=True,
    )

    tag = (
        f"evict-nb05c-unet-seed{seed}-lr3e4-"
        f"final-step{final_step}"
    )

    release, headers, api = base.create_or_get_release(
        token,
        tag,
        f"EviCT Notebook 05C — U-Net seed {seed} — Final",
        (
            "Final durable recovery for the frozen competitive residual 2D U-Net. "
            f"seed={seed}, lr=0.0003, final step={final_step}, "
            f"stop={stop_reason}, best source-selection macro case Dice="
            f"{best_score:.8f} at step {best_step}. "
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
        "stage": "NOTEBOOK_05C",
        "seed": int(seed),
        "learning_rate": FROZEN_LR,
        "final_step": int(final_step),
        "stop_reason": stop_reason,
        "best_macro_case_dice": float(best_score),
        "best_step": int(best_step),
        "initialization_hash": init_hash,
        "final_archive": archive_path.name,
        "final_archive_sha256": digest,
        "release_tag": tag,
        "release_url": release["html_url"],
        "tar_asset_id": int(tar_asset["id"]),
        "sha_asset_id": int(sha_asset["id"]),
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


def rebuild_best_logits(
    *,
    seed: int,
    p: dict,
    train_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    device: torch.device,
    config_hash: str,
    frozen_lr_hash: str,
    split_hash: str,
    model_code_hash: str,
    metrics_code_hash: str,
    best_score: float,
    best_step: int,
):
    from evict.models.unet2d import masked_supervised_loss
    from evict.metrics import (
        CaseMetricAccumulator,
        binary_segmentation_metrics,
        macro_case_summary,
        metrics_from_confusion,
    )

    (
        store,
        sampler,
        model,
        optimizer,
        scheduler,
        init_hash,
        state,
    ) = restore_checkpoint(
        seed=seed,
        path=p["best_pt"],
        train_df=train_df,
        device=device,
        config_hash=config_hash,
        frozen_lr_hash=frozen_lr_hash,
        split_hash=split_hash,
        model_code_hash=model_code_hash,
        metrics_code_hash=metrics_code_hash,
    )

    assert state["global_step"] == best_step
    assert state["best_step"] == best_step
    assert abs(state["best_score"] - best_score) < 1e-10

    metrics, _, raw_logits = base.validate(
        model=model,
        device=device,
        selection_df=selection_df,
        masked_supervised_loss=masked_supervised_loss,
        CaseMetricAccumulator=CaseMetricAccumulator,
        binary_segmentation_metrics=binary_segmentation_metrics,
        metrics_from_confusion=metrics_from_confusion,
        macro_case_summary=macro_case_summary,
    )

    assert abs(metrics["macro_case_dice"] - best_score) < 1e-8

    install_best_logits(
        seed=seed,
        p=p,
        raw_logits=raw_logits,
        step=best_step,
    )

    del (
        raw_logits,
        model,
        optimizer,
        scheduler,
        sampler,
        store,
    )
    gc.collect()
    torch.cuda.empty_cache()


def run_seed(
    *,
    token: str,
    seed: int,
    split_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    device: torch.device,
    config_hash: str,
    frozen_lr_hash: str,
    model_code_hash: str,
    metrics_code_hash: str,
):
    from evict.models.unet2d import masked_supervised_loss
    from evict.metrics import (
        CaseMetricAccumulator,
        binary_segmentation_metrics,
        macro_case_summary,
        metrics_from_confusion,
    )

    p = paths_for(seed)
    p["run_dir"].mkdir(parents=True, exist_ok=True)
    p["pred_dir"].mkdir(parents=True, exist_ok=True)
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    if p["final_audit"].exists():
        existing = json.loads(
            p["final_audit"].read_text(encoding="utf-8")
        )
        if existing.get("status") == "DURABLE_COMPLETE":
            print(
                f"✓ seed {seed} already DURABLE_COMPLETE "
                f"at step {existing['final_step']}; skipping."
            )
            return existing

    manifest_path = train_manifest(seed)
    assert manifest_path.exists()

    train_df = pd.read_csv(manifest_path)

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

    assert set(train_df["case_id"].astype(str)) == fitting_cases
    assert set(train_df["seed"].astype(int)) == {seed}
    assert set(selection_df["case_id"].astype(str)) == selection_cases

    manifest_text = (
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
        assert forbidden not in manifest_text

    split_hash = base.stable_json_hash({
        "splits_csv": base.sha256_file(SPLITS_PATH),
        "train_manifest": base.sha256_file(manifest_path),
        "selection_manifest": base.sha256_file(SELECTION_MANIFEST),
    })

    restore_remote_if_needed(
        seed=seed,
        p=p,
    )

    train_rows, selection_rows, case_rows = load_logs(p)
    dedupe_logs(train_rows, selection_rows, case_rows)

    local = []
    for checkpoint in [
        p["last_pt"],
        p["recovery_pt"],
    ]:
        if checkpoint.exists():
            local.append(
                checkpoint_metadata(
                    seed=seed,
                    path=checkpoint,
                    config_hash=config_hash,
                    frozen_lr_hash=frozen_lr_hash,
                    split_hash=split_hash,
                    model_code_hash=model_code_hash,
                    metrics_code_hash=metrics_code_hash,
                )
            )

    print()
    print("=" * 110)
    print(
        f"NOTEBOOK 05C — FROZEN U-NET — SEED {seed} "
        f"(LR={FROZEN_LR:.1e})"
    )
    print("=" * 110)

    if local:
        newest = max(local, key=lambda x: x["step"])
        (
            store,
            sampler,
            model,
            optimizer,
            scheduler,
            init_hash,
            state,
        ) = restore_checkpoint(
            seed=seed,
            path=newest["path"],
            train_df=train_df,
            device=device,
            config_hash=config_hash,
            frozen_lr_hash=frozen_lr_hash,
            split_hash=split_hash,
            model_code_hash=model_code_hash,
            metrics_code_hash=metrics_code_hash,
        )
        print(
            f"✓ Resumed seed {seed}       : "
            f"step {state['global_step']}"
        )
    else:
        (
            store,
            sampler,
            model,
            optimizer,
            scheduler,
            init_hash,
        ) = build_components(
            seed,
            train_df,
            device,
        )
        state = {
            "global_step": 0,
            "best_score": float("-inf"),
            "best_step": 0,
            "patience_count": 0,
            "last_validation_step": 0,
            "images_seen": 0,
        }
        print(f"✓ New seed {seed} run       : step 0")

    global_step = state["global_step"]
    best_score = state["best_score"]
    best_step = state["best_step"]
    patience_count = state["patience_count"]
    last_validation_step = state["last_validation_step"]
    images_seen = state["images_seen"]

    assert 0 <= global_step <= MAX_UPDATES
    model.train()

    progress = tqdm(
        total=MAX_UPDATES - global_step,
        desc=f"seed{seed} train",
        unit="update",
        ncols=138,
        file=sys.stdout,
        leave=True,
    )

    stop_reason = None

    while global_step < MAX_UPDATES:
        update = one_update(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            device=device,
            masked_supervised_loss=masked_supervised_loss,
        )

        global_step += 1
        images_seen += EFFECTIVE_BATCH

        train_rows.append({
            "seed": seed,
            "step": global_step,
            "loss": update["loss"],
            "dice_loss": update["dice_loss"],
            "bce_loss": update["bce_loss"],
            "learning_rate": update["learning_rate"],
            "images_seen": images_seen,
            "precision": "FP32",
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
                frozen_lr_hash=frozen_lr_hash,
                split_hash=split_hash,
                model_code_hash=model_code_hash,
                metrics_code_hash=metrics_code_hash,
                init_hash=init_hash,
                write_last=False,
            )

        if global_step % VALIDATE_EVERY == 0:
            progress.write(
                f"\n[seed {seed}] SOURCE-SELECTION "
                f"VALIDATION @ step {global_step}"
            )

            metrics, current_case_rows, raw_logits = base.validate(
                model=model,
                device=device,
                selection_df=selection_df,
                masked_supervised_loss=masked_supervised_loss,
                CaseMetricAccumulator=CaseMetricAccumulator,
                binary_segmentation_metrics=binary_segmentation_metrics,
                metrics_from_confusion=metrics_from_confusion,
                macro_case_summary=macro_case_summary,
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
                    frozen_lr_hash=frozen_lr_hash,
                    split_hash=split_hash,
                    model_code_hash=model_code_hash,
                    metrics_code_hash=metrics_code_hash,
                    init_hash=init_hash,
                )
                base.atomic_torch_save(payload, p["best_pt"])
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
                "learning_rate": FROZEN_LR,
                "step": global_step,
                **metrics,
                "best_so_far": best_score,
                "best_step": best_step,
                "patience_count": patience_count,
                "precision": "FP32",
            })

            for row in current_case_rows:
                case_rows.append({
                    "seed": seed,
                    "learning_rate": FROZEN_LR,
                    "step": global_step,
                    **row,
                })

            dedupe_logs(train_rows, selection_rows, case_rows)

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
                frozen_lr_hash=frozen_lr_hash,
                split_hash=split_hash,
                model_code_hash=model_code_hash,
                metrics_code_hash=metrics_code_hash,
                init_hash=init_hash,
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
                    f"NOTEBOOK_05C_UNET_SEED_{seed}_"
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
                f"Notebook 05C U-Net seed{seed} validation step {global_step}"
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

    rebuild_best_logits(
        seed=seed,
        p=p,
        train_df=train_df,
        selection_df=selection_df,
        device=device,
        config_hash=config_hash,
        frozen_lr_hash=frozen_lr_hash,
        split_hash=split_hash,
        model_code_hash=model_code_hash,
        metrics_code_hash=metrics_code_hash,
        best_score=best_score,
        best_step=best_step,
    )

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
        status=f"NOTEBOOK_05C_UNET_SEED_{seed}_COMPLETE_DURABLE",
    )

    base.git_sync(
        f"Complete durable Notebook 05C U-Net seed{seed} at step {global_step}"
    )

    del model, optimizer, scheduler, sampler, store
    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"✓ seed {seed} DURABLE_COMPLETE | "
        f"final={global_step} | "
        f"best Dice={best_score:.8f} @ {best_step}"
    )

    return final


def aggregate_three_seeds():
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    seed17_final_path = (
        AUDIT_ROOT
        / "notebook05b_unet_seed17_lr3e4_final_durable.json"
    )
    seed17_selection_path = (
        AUDIT_ROOT
        / "notebook05b_unet_seed17_lr3e4_selection_metrics.csv"
    )
    seed17_case_path = (
        AUDIT_ROOT
        / "notebook05b_unet_seed17_lr3e4_selection_case_metrics.csv"
    )

    assert seed17_final_path.exists()
    assert seed17_selection_path.exists()
    assert seed17_case_path.exists()

    final_by_seed = {
        17: json.loads(
            seed17_final_path.read_text(encoding="utf-8")
        ),
        42: json.loads(
            paths_for(42)["final_audit"].read_text(encoding="utf-8")
        ),
        2026: json.loads(
            paths_for(2026)["final_audit"].read_text(encoding="utf-8")
        ),
    }

    selection_paths = {
        17: seed17_selection_path,
        42: paths_for(42)["selection_log"],
        2026: paths_for(2026)["selection_log"],
    }

    case_paths = {
        17: seed17_case_path,
        42: paths_for(42)["case_log"],
        2026: paths_for(2026)["case_log"],
    }

    rows = []
    case_rows = []

    for seed in PRIMARY_SEEDS:
        final = final_by_seed[seed]
        assert final["status"] == "DURABLE_COMPLETE"
        assert final["calibration_accessed"] is False
        assert final["target_accessed"] is False

        best_step = int(final["best_step"])
        selection_df = pd.read_csv(selection_paths[seed])
        row = selection_df[
            selection_df["step"].astype(int) == best_step
        ]
        assert len(row) == 1
        record = row.iloc[0].to_dict()

        rows.append({
            "seed": seed,
            "learning_rate": FROZEN_LR,
            "best_step": best_step,
            "final_step": int(final["final_step"]),
            "stop_reason": final["stop_reason"],
            "macro_case_dice": float(record["macro_case_dice"]),
            "macro_case_iou": float(record["macro_case_iou"]),
            "macro_case_sensitivity": float(record["macro_case_sensitivity"]),
            "macro_case_specificity": float(record["macro_case_specificity"]),
            "macro_slice_dice": float(record["macro_slice_dice"]),
            "pooled_dice": float(record["pooled_dice"]),
            "pooled_iou": float(record["pooled_iou"]),
            "threshold": THRESHOLD,
            "release_url": final["release_url"],
        })

        cdf = pd.read_csv(case_paths[seed])
        cdf = cdf[cdf["step"].astype(int) == best_step].copy()
        assert len(cdf) == 4
        cdf["seed"] = seed
        cdf["best_step"] = best_step
        case_rows.extend(cdf.to_dict("records"))

    df = pd.DataFrame(rows)
    case_df = pd.DataFrame(case_rows)

    summary_path = TABLE_ROOT / "notebook05_unet_three_seed_source_baseline.csv"
    case_path = TABLE_ROOT / "notebook05_unet_three_seed_case_metrics.csv"

    df.to_csv(summary_path, index=False)
    case_df.to_csv(case_path, index=False)

    metric_cols = [
        "macro_case_dice",
        "macro_case_iou",
        "macro_case_sensitivity",
        "macro_case_specificity",
        "macro_slice_dice",
        "pooled_dice",
        "pooled_iou",
    ]

    aggregate = {}
    for col in metric_cols:
        values = df[col].astype(float).to_numpy()
        aggregate[col] = {
            "mean": float(np.mean(values)),
            "sample_sd": float(np.std(values, ddof=1)),
        }

    audit = {
        "timestamp_utc": base.utc_now(),
        "stage": "NOTEBOOK_05",
        "baseline": "competitive_residual_unet2d",
        "status": "THREE_SEED_SOURCE_BASELINE_FROZEN",
        "primary_seeds": PRIMARY_SEEDS,
        "selected_learning_rate": FROZEN_LR,
        "learning_rate_frozen_before_seeds_42_2026": True,
        "architecture_tuning": False,
        "threshold": THRESHOLD,
        "threshold_tuned": False,
        "metrics": aggregate,
        "seed_results": rows,
        "calibration_accessed": False,
        "target_accessed": False,
    }

    audit_path = (
        AUDIT_ROOT
        / "notebook05_unet_three_seed_source_baseline.json"
    )
    base.atomic_text(
        audit_path,
        json.dumps(audit, indent=2),
    )

    mean_dice = aggregate["macro_case_dice"]["mean"]
    sd_dice = aggregate["macro_case_dice"]["sample_sd"]

    report_lines = [
        "# Notebook 05 — Frozen Competitive Residual 2D U-Net Source Baseline",
        "",
        "Learning rate: 0.0003, frozen from the seed-17 pilot before running seeds 42 and 2026.",
        "",
        f"Three-seed source-selection macro case Dice: {mean_dice:.6f} ± {sd_dice:.6f} (sample SD).",
        "",
        "| Seed | Best step | Final step | Macro case Dice | Macro case IoU | Macro slice Dice | Pooled Dice |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in df.iterrows():
        report_lines.append(
            f"| {int(row['seed'])} | {int(row['best_step'])} | "
            f"{int(row['final_step'])} | "
            f"{row['macro_case_dice']:.6f} | "
            f"{row['macro_case_iou']:.6f} | "
            f"{row['macro_slice_dice']:.6f} | "
            f"{row['pooled_dice']:.6f} |"
        )

    report_lines.extend([
        "",
        "Selection threshold: 0.5 fixed; no threshold search.",
        "Calibration accessed: NO.",
        "Target / MedSeg accessed: NO.",
        "Target lock: ACTIVE.",
        "",
        "Figures can be regenerated later from the frozen best checkpoints, raw logits, and metric tables.",
    ])

    report_path = REPORT_ROOT / "notebook05_unet_source_baseline_report.md"
    base.atomic_text(
        report_path,
        "\n".join(report_lines) + "\n",
    )

    state = f"""# EviCT Execution State

## Current stage

NOTEBOOK_05_UNET_THREE_SEED_SOURCE_BASELINE_FROZEN

## Timestamp

{base.utc_now()}

## Baseline

Competitive residual 2D U-Net

Architecture:

Residual 2D U-Net with GroupNorm

Frozen learning rate:

0.00030000

Primary seeds:

17, 42, 2026

Three-seed source-selection macro case Dice:

{mean_dice:.8f}

Sample standard deviation:

{sd_dice:.8f}

Selection threshold:

0.5 fixed

Architecture tuning:

NO

Threshold tuning:

NO

## Isolation

Calibration accessed:

NO

Target / MedSeg accessed:

NO

Target lock:

ACTIVE

## Durability

Seed 17 final recovery:

{final_by_seed[17]['release_url']}

Seed 42 final recovery:

{final_by_seed[42]['release_url']}

Seed 2026 final recovery:

{final_by_seed[2026]['release_url']}

## Next

The competitive U-Net reference baseline is frozen.
Proceed to the remaining E03 comparison component (SegCT-CLIP reproduction/reimplementation)
before starting the proposed semantic branch.
"""

    base.atomic_text(STATE_PATH, state)
    base.atomic_text(HANDOFF_STATE, state)

    base.git_sync(
        "Freeze Notebook 05 three-seed competitive U-Net source baseline"
    )

    return audit


def main():
    print("=" * 110)
    print("EVICT NOTEBOOK 05C — FROZEN U-NET PRIMARY SEEDS 42 + 2026")
    print("=" * 110)

    assert ROOT.exists() and (ROOT / ".git").exists()
    assert BASE_RUNNER_PATH.exists()
    assert CONFIG_PATH.exists()
    assert SELECTED_LR_PATH.exists()
    assert MODEL_CODE.exists()
    assert METRICS_CODE.exists()
    assert SPLITS_PATH.exists()
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

    base.ensure_cache()

    config = json.loads(
        CONFIG_PATH.read_text(encoding="utf-8")
    )
    selected_lr = json.loads(
        SELECTED_LR_PATH.read_text(encoding="utf-8")
    )

    assert config["stage"] == "NOTEBOOK_05"
    assert config["baseline_name"] == "competitive_residual_unet2d"
    assert config["architecture"]["pretraining"] == "none"
    assert config["primary_training_seeds"] == [17, 42, 2026]
    assert config["data_contract"]["target_access_allowed"] is False
    assert config["validation"]["threshold_tuning"] is False
    assert config["batch"]["micro_batch"] == MICRO_BATCH
    assert config["batch"]["gradient_accumulation"] == GRAD_ACCUM
    assert config["batch"]["images_per_optimizer_update"] == EFFECTIVE_BATCH
    assert config["scheduler"]["maximum_updates"] == MAX_UPDATES
    assert config["scheduler"]["warmup_updates"] == WARMUP_UPDATES
    assert config["early_stopping"]["patience_validations"] == PATIENCE_LIMIT

    assert selected_lr["status"] == "LEARNING_RATE_FROZEN"
    assert selected_lr["selected_candidate"] == "lr3e4"
    assert abs(
        float(selected_lr["selected_learning_rate"]) - FROZEN_LR
    ) < 1e-15
    assert selected_lr["calibration_accessed"] is False
    assert selected_lr["target_accessed"] is False

    split_df = pd.read_csv(SPLITS_PATH)
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
    assert set(selection_df["case_id"].astype(str)) == selection_cases

    for seed in SEEDS:
        assert train_manifest(seed).exists()

    print("✓ Frozen learning rate     : 0.00030000")
    print("✓ Frozen fitting cases     : 12")
    print("✓ Frozen selection cases   : 4")
    print("✓ Seeds to execute         : 42, 2026")
    print("✓ Calibration accessed     : NO")
    print("✓ Target / MedSeg accessed : NO")
    print("✓ Target lock              : ACTIVE")

    config_hash = base.stable_json_hash(config)
    frozen_lr_hash = base.stable_json_hash(selected_lr)
    model_code_hash = base.sha256_file(MODEL_CODE)
    metrics_code_hash = base.sha256_file(METRICS_CODE)

    results = []

    for seed in SEEDS:
        result = run_seed(
            token=token,
            seed=seed,
            split_df=split_df,
            selection_df=selection_df,
            device=device,
            config_hash=config_hash,
            frozen_lr_hash=frozen_lr_hash,
            model_code_hash=model_code_hash,
            metrics_code_hash=metrics_code_hash,
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
    print("EVICT NOTEBOOK 05 — THREE-SEED U-NET SOURCE BASELINE — DURABLE PASS")
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
    print("Learning rate             : 0.00030000 FROZEN")
    print("Calibration accessed      : NO")
    print("Target / MedSeg accessed  : NO")
    print("Target lock               : ACTIVE")
    print(f"GitHub HEAD               : {local_head}")
    print()
    print(
        "NEXT: SegCT-CLIP reproduction/reimplementation, "
        "then Notebook 06 proposed semantic branch."
    )


if __name__ == "__main__":
    main()
