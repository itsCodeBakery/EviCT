from __future__ import annotations

import argparse
import gc
import importlib.util
import json
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
import requests
import torch
from tqdm import tqdm

from kaggle_secrets import UserSecretsClient


ROOT = Path("/kaggle/working/EviCT")
WORK = Path("/kaggle/working")

BASE_RUNNER_PATH = ROOT / "scripts/nb05b_unet_lr_pilot_segment1.py"
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

PILOT_AUDIT_PATH = AUDIT_ROOT / "notebook05b_unet_lr_pilot_step1000.json"
SELECTED_LR_PATH = ROOT / "config/notebook05_unet_selected_lr.json"

SEED = 17
MAX_UPDATES = 5000
VALIDATE_EVERY = 250
RECOVERY_EVERY = 50
PATIENCE_LIMIT = 8
MICRO_BATCH = 4
GRAD_ACCUM = 4
EFFECTIVE_BATCH = 16
THRESHOLD = 0.5

ROLLING_MILESTONES = {2000, 3000, 4000}

REPO_OWNER = "itsCodeBakery"
REPO_NAME = "EviCT"


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "evict_nb05b_segment1",
        str(BASE_RUNNER_PATH),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import base runner: {BASE_RUNNER_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def candidate_paths(candidate_name: str):
    run_dir = RUN_ROOT / f"unet_seed17_{candidate_name}"
    pred_dir = PRED_ROOT / f"unet_seed17_{candidate_name}"
    return {
        "run_dir": run_dir,
        "pred_dir": pred_dir,
        "best_pt": run_dir / "best.pt",
        "last_pt": run_dir / "last.pt",
        "recovery_pt": run_dir / "recovery.pt",
        "resume_audit": AUDIT_ROOT / (
            f"notebook05b_unet_seed17_{candidate_name}_resume_1020.json"
        ),
        "rolling_audit": AUDIT_ROOT / (
            f"notebook05b_unet_seed17_{candidate_name}_rolling_durable.json"
        ),
        "final_audit": AUDIT_ROOT / (
            f"notebook05b_unet_seed17_{candidate_name}_final_durable.json"
        ),
        "train_log": AUDIT_ROOT / (
            f"notebook05b_unet_seed17_{candidate_name}_train_log.csv"
        ),
        "selection_log": AUDIT_ROOT / (
            f"notebook05b_unet_seed17_{candidate_name}_selection_metrics.csv"
        ),
        "case_log": AUDIT_ROOT / (
            f"notebook05b_unet_seed17_{candidate_name}_selection_case_metrics.csv"
        ),
    }


def direct_release_url(tag: str, asset_name: str) -> str:
    return (
        f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/"
        f"{tag}/{asset_name}"
    )


def expected_step1000_archive(candidate_name: str):
    tag = f"evict-nb05b-unet-seed17-{candidate_name}-step1000"
    name = (
        f"EviCT_Notebook05B_unet_seed17_{candidate_name}_"
        "step1000_Recovery.tar"
    )
    return tag, name


def validate_checkpoint_payload(
    payload: dict,
    *,
    candidate_name: str,
    learning_rate: float,
    config_hash: str,
    split_hash: str,
    model_code_hash: str,
    metrics_code_hash: str,
    init_hash: str,
):
    assert payload["project"] == "EviCT"
    assert payload["stage"] == "NOTEBOOK_05B"
    assert payload["candidate_name"] == candidate_name
    assert abs(float(payload["learning_rate"]) - learning_rate) < 1e-15
    assert int(payload["seed"]) == SEED
    assert int(payload["split_seed"]) == 17
    assert payload["precision"] == "FP32"
    assert payload["amp_enabled"] is False
    assert payload["config_hash"] == config_hash
    assert payload["split_hash"] == split_hash
    assert payload["model_code_hash"] == model_code_hash
    assert payload["metrics_code_hash"] == metrics_code_hash
    assert payload["initialization_hash"] == init_hash
    assert payload["target_accessed"] is False
    assert payload["calibration_accessed"] is False


def load_logs(paths: dict):
    train_rows = (
        pd.read_csv(paths["train_log"]).to_dict("records")
        if paths["train_log"].exists()
        else []
    )
    selection_rows = (
        pd.read_csv(paths["selection_log"]).to_dict("records")
        if paths["selection_log"].exists()
        else []
    )
    case_rows = (
        pd.read_csv(paths["case_log"]).to_dict("records")
        if paths["case_log"].exists()
        else []
    )

    return train_rows, selection_rows, case_rows


def deduplicate_logs(train_rows, selection_rows, case_rows):
    if train_rows:
        df = pd.DataFrame(train_rows)
        df = df.sort_values("step").drop_duplicates(subset=["step"], keep="last")
        train_rows[:] = df.to_dict("records")

    if selection_rows:
        df = pd.DataFrame(selection_rows)
        df = df.sort_values("step").drop_duplicates(subset=["step"], keep="last")
        selection_rows[:] = df.to_dict("records")

    if case_rows:
        df = pd.DataFrame(case_rows)
        df = (
            df.sort_values(["step", "case_id"])
            .drop_duplicates(subset=["step", "case_id"], keep="last")
        )
        case_rows[:] = df.to_dict("records")


def build_components(
    *,
    candidate_name: str,
    learning_rate: float,
    train_df: pd.DataFrame,
    device: torch.device,
    expected_init_hash: str,
):
    from evict.models.unet2d import EviCTResidualUNet2D

    base.reset_rng(SEED)

    store = base.FittingCaseStore(train_df)
    sampler = base.PatientUniformSampler(store, SEED)

    model = EviCTResidualUNet2D(
        in_channels=3,
        base_channels=32,
    )

    init_hash = base.model_state_hash(model)
    assert init_hash == expected_init_hash, (
        f"Initialization drift for {candidate_name}\n"
        f"Expected: {expected_init_hash}\nActual:   {init_hash}"
    )

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=base.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=base.scheduler_multiplier,
    )

    return store, sampler, model, optimizer, scheduler, init_hash


def restore_components_from_checkpoint(
    *,
    checkpoint_path: Path,
    candidate_name: str,
    learning_rate: float,
    train_df: pd.DataFrame,
    device: torch.device,
    expected_init_hash: str,
    config_hash: str,
    split_hash: str,
    model_code_hash: str,
    metrics_code_hash: str,
):
    store, sampler, model, optimizer, scheduler, init_hash = build_components(
        candidate_name=candidate_name,
        learning_rate=learning_rate,
        train_df=train_df,
        device=device,
        expected_init_hash=expected_init_hash,
    )

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    validate_checkpoint_payload(
        payload,
        candidate_name=candidate_name,
        learning_rate=learning_rate,
        config_hash=config_hash,
        split_hash=split_hash,
        model_code_hash=model_code_hash,
        metrics_code_hash=metrics_code_hash,
        init_hash=init_hash,
    )

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
    path: Path,
    *,
    candidate_name: str,
    learning_rate: float,
    config_hash: str,
    split_hash: str,
    model_code_hash: str,
    metrics_code_hash: str,
    init_hash: str,
):
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    try:
        validate_checkpoint_payload(
            payload,
            candidate_name=candidate_name,
            learning_rate=learning_rate,
            config_hash=config_hash,
            split_hash=split_hash,
            model_code_hash=model_code_hash,
            metrics_code_hash=metrics_code_hash,
            init_hash=init_hash,
        )

        return {
            "path": path,
            "global_step": int(payload["global_step"]),
            "best_score": float(payload["best_score"]),
            "best_step": int(payload["best_step"]),
            "patience_count": int(payload["patience_count"]),
            "last_validation_step": int(payload["last_validation_step"]),
            "images_seen": int(payload["images_seen"]),
        }
    finally:
        del payload


def ensure_remote_resume_available(
    *,
    candidate_name: str,
    pilot_info: dict,
    paths: dict,
):
    paths["run_dir"].mkdir(parents=True, exist_ok=True)

    # If a valid local checkpoint exists, it is preferred.
    if paths["recovery_pt"].exists() or paths["last_pt"].exists():
        return

    rolling = None

    if paths["rolling_audit"].exists():
        candidate = json.loads(
            paths["rolling_audit"].read_text(encoding="utf-8")
        )
        if candidate.get("status") == "ROLLING_DURABLE":
            rolling = candidate

    if rolling is not None and int(rolling["step"]) > 1000:
        print(
            f"Restoring {candidate_name} from rolling remote recovery "
            f"at step {rolling['step']}..."
        )

        archive_name = rolling["archive_name"]
        archive_sha = rolling["archive_sha256"]
        tag = rolling["release_tag"]
        url = direct_release_url(tag, archive_name)

    else:
        tag, archive_name = expected_step1000_archive(candidate_name)
        archive_sha = pilot_info["recovery_archive_sha256"]
        url = direct_release_url(tag, archive_name)

        print(
            f"Restoring {candidate_name} from durable step-1000 release..."
        )

    archive_path = WORK / archive_name

    base.download_verified(
        url,
        archive_path,
        archive_sha,
    )

    base.safe_extract_tar(
        archive_path,
        WORK,
    )

    assert paths["recovery_pt"].exists(), (
        f"Recovery checkpoint was not restored for {candidate_name}"
    )

    archive_path.unlink(missing_ok=True)


def one_optimizer_update(
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

        inputs = base.normalize_batch(
            images,
            device,
        )

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

        del (
            images,
            targets,
            valids,
            inputs,
            logits,
            loss_dict,
            loss,
        )

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
    paths: dict,
    model,
    optimizer,
    scheduler,
    sampler,
    candidate_name: str,
    learning_rate: float,
    global_step: int,
    best_score: float,
    best_step: int,
    patience_count: int,
    last_validation_step: int,
    images_seen: int,
    config_hash: str,
    split_hash: str,
    model_code_hash: str,
    metrics_code_hash: str,
    init_hash: str,
    write_last: bool,
):
    payload = base.make_checkpoint(
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

    base.atomic_torch_save(
        payload,
        paths["recovery_pt"],
    )

    if write_last:
        base.atomic_torch_save(
            payload,
            paths["last_pt"],
        )

    del payload


def recursive_equal(a, b) -> bool:
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.equal(a.cpu(), b.cpu())

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)

    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(recursive_equal(a[k], b[k]) for k in a)

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(recursive_equal(x, y) for x, y in zip(a, b))

    return a == b


def verify_current_rng_matches(saved_rng: dict) -> bool:
    if random.getstate() != saved_rng["python"]:
        return False

    current_np = np.random.get_state()
    saved_np = saved_rng["numpy"]

    if current_np[0] != saved_np[0]:
        return False
    if not np.array_equal(current_np[1], saved_np[1]):
        return False
    if current_np[2:] != saved_np[2:]:
        return False

    if not torch.equal(
        torch.get_rng_state().cpu(),
        saved_rng["torch_cpu"].cpu(),
    ):
        return False

    current_cuda = [
        state.cpu()
        for state in torch.cuda.get_rng_state_all()
    ]
    saved_cuda = [
        state.cpu()
        for state in saved_rng["torch_cuda"]
    ]

    if len(current_cuda) != len(saved_cuda):
        return False

    return all(
        torch.equal(x, y)
        for x, y in zip(current_cuda, saved_cuda)
    )


def run_recovery_audit_1000_to_1020(
    *,
    candidate_name: str,
    learning_rate: float,
    paths: dict,
    pilot_info: dict,
    train_df: pd.DataFrame,
    device: torch.device,
    expected_init_hash: str,
    config_hash: str,
    split_hash: str,
    model_code_hash: str,
    metrics_code_hash: str,
    train_rows: list,
    selection_rows: list,
    case_metric_rows: list,
):
    from evict.models.unet2d import masked_supervised_loss

    if (
        paths["resume_audit"].exists()
        and paths["recovery_pt"].exists()
    ):
        existing = json.loads(
            paths["resume_audit"].read_text(encoding="utf-8")
        )

        if (
            existing.get("status") == "PASS"
            and int(existing.get("ending_step", 0)) >= 1020
        ):
            print(
                f"✓ {candidate_name} recovery audit already PASS "
                f"through step {existing['ending_step']}"
            )
            return restore_components_from_checkpoint(
                checkpoint_path=paths["recovery_pt"],
                candidate_name=candidate_name,
                learning_rate=learning_rate,
                train_df=train_df,
                device=device,
                expected_init_hash=expected_init_hash,
                config_hash=config_hash,
                split_hash=split_hash,
                model_code_hash=model_code_hash,
                metrics_code_hash=metrics_code_hash,
            )

    (
        store,
        sampler,
        model,
        optimizer,
        scheduler,
        init_hash,
        state,
    ) = restore_components_from_checkpoint(
        checkpoint_path=paths["recovery_pt"],
        candidate_name=candidate_name,
        learning_rate=learning_rate,
        train_df=train_df,
        device=device,
        expected_init_hash=expected_init_hash,
        config_hash=config_hash,
        split_hash=split_hash,
        model_code_hash=model_code_hash,
        metrics_code_hash=metrics_code_hash,
    )

    assert state["global_step"] == 1000
    assert state["last_validation_step"] == 1000
    assert abs(
        state["best_score"]
        - float(pilot_info["best_macro_case_dice"])
    ) < 1e-10
    assert state["best_step"] == int(pilot_info["best_step"])
    assert state["patience_count"] == int(pilot_info["patience_count"])
    assert state["images_seen"] == 16000

    print()
    print("=" * 110)
    print(
        f"RECOVERY AUDIT — {candidate_name} — "
        "STEP 1000 -> 1020"
    )
    print("=" * 110)

    global_step = state["global_step"]
    images_seen = state["images_seen"]

    progress = tqdm(
        total=20,
        desc=f"{candidate_name} recovery",
        unit="update",
        ncols=120,
        file=sys.stdout,
        leave=True,
    )

    for _ in range(20):
        update = one_optimizer_update(
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
        )
        progress.update(1)

    progress.close()

    save_checkpoint(
        paths=paths,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        candidate_name=candidate_name,
        learning_rate=learning_rate,
        global_step=global_step,
        best_score=state["best_score"],
        best_step=state["best_step"],
        patience_count=state["patience_count"],
        last_validation_step=state["last_validation_step"],
        images_seen=images_seen,
        config_hash=config_hash,
        split_hash=split_hash,
        model_code_hash=model_code_hash,
        metrics_code_hash=metrics_code_hash,
        init_hash=init_hash,
        write_last=False,
    )

    base.write_logs(
        candidate_name,
        train_rows,
        selection_rows,
        case_metric_rows,
    )

    model_hash_before_reload = base.model_state_hash(model)
    optimizer_before_reload = optimizer.state_dict()
    scheduler_before_reload = scheduler.state_dict()
    sampler_before_reload = sampler.state_dict()

    saved_payload = torch.load(
        paths["recovery_pt"],
        map_location="cpu",
        weights_only=False,
    )
    saved_rng = saved_payload["rng_state"]
    del saved_payload

    del model, optimizer, scheduler, sampler, store
    gc.collect()
    torch.cuda.empty_cache()

    (
        store,
        sampler,
        model,
        optimizer,
        scheduler,
        init_hash,
        reloaded_state,
    ) = restore_components_from_checkpoint(
        checkpoint_path=paths["recovery_pt"],
        candidate_name=candidate_name,
        learning_rate=learning_rate,
        train_df=train_df,
        device=device,
        expected_init_hash=expected_init_hash,
        config_hash=config_hash,
        split_hash=split_hash,
        model_code_hash=model_code_hash,
        metrics_code_hash=metrics_code_hash,
    )

    assert reloaded_state["global_step"] == 1020
    assert reloaded_state["images_seen"] == 16320
    assert reloaded_state["last_validation_step"] == 1000
    assert reloaded_state["best_step"] == state["best_step"]
    assert abs(
        reloaded_state["best_score"]
        - state["best_score"]
    ) < 1e-12
    assert reloaded_state["patience_count"] == state["patience_count"]

    assert base.model_state_hash(model) == model_hash_before_reload
    assert recursive_equal(
        optimizer.state_dict(),
        optimizer_before_reload,
    )
    assert recursive_equal(
        scheduler.state_dict(),
        scheduler_before_reload,
    )
    assert recursive_equal(
        sampler.state_dict(),
        sampler_before_reload,
    )
    assert verify_current_rng_matches(saved_rng)

    audit = {
        "timestamp_utc": base.utc_now(),
        "stage": "NOTEBOOK_05B",
        "candidate_name": candidate_name,
        "learning_rate": learning_rate,
        "seed": SEED,
        "starting_step": 1000,
        "ending_step": 1020,
        "updates_executed": 20,
        "model_state_restored": True,
        "optimizer_state_restored": True,
        "scheduler_state_restored": True,
        "sampler_state_restored": True,
        "python_rng_restored": True,
        "numpy_rng_restored": True,
        "torch_cpu_rng_restored": True,
        "torch_cuda_rng_restored": True,
        "best_score_unchanged": True,
        "best_step_unchanged": True,
        "last_validation_step": 1000,
        "images_seen": 16320,
        "recovery_checkpoint_sha256": base.sha256_file(
            paths["recovery_pt"]
        ),
        "calibration_accessed": False,
        "target_accessed": False,
        "status": "PASS",
    }

    base.atomic_text(
        paths["resume_audit"],
        json.dumps(audit, indent=2),
    )

    base.update_state(
        candidate_name=candidate_name,
        learning_rate=learning_rate,
        global_step=1020,
        best_score=reloaded_state["best_score"],
        best_step=reloaded_state["best_step"],
        patience_count=reloaded_state["patience_count"],
        images_seen=reloaded_state["images_seen"],
        status=(
            f"NOTEBOOK_05B_UNET_"
            f"{candidate_name.upper()}_RECOVERY_AUDIT_1020_PASS"
        ),
    )

    base.git_sync(
        f"Verify Notebook 05B U-Net seed17 {candidate_name} "
        "recovery through step 1020"
    )

    print(
        f"✓ {candidate_name} recovery audit: PASS "
        "(1000 -> 1020)"
    )

    return (
        store,
        sampler,
        model,
        optimizer,
        scheduler,
        init_hash,
        reloaded_state,
    )


def create_compact_recovery_archive(
    *,
    candidate_name: str,
    paths: dict,
    output_path: Path,
):
    output_path.unlink(missing_ok=True)

    with tarfile.open(output_path, "w") as archive:
        for checkpoint in [
            paths["recovery_pt"],
            paths["best_pt"],
        ]:
            assert checkpoint.exists()
            archive.add(
                checkpoint,
                arcname=str(
                    Path("EviCT/artifacts/large/notebook05")
                    / paths["run_dir"].name
                    / checkpoint.name
                ),
            )

    return base.sha256_file(output_path)


def upload_rolling_recovery(
    *,
    token: str,
    candidate_name: str,
    learning_rate: float,
    step: int,
    best_score: float,
    best_step: int,
    paths: dict,
):
    archive_name = (
        f"EviCT_Notebook05B_unet_seed17_"
        f"{candidate_name}_Rolling_Recovery.tar"
    )
    archive_path = WORK / archive_name

    digest = create_compact_recovery_archive(
        candidate_name=candidate_name,
        paths=paths,
        output_path=archive_path,
    )

    sha_path = Path(str(archive_path) + ".sha256")
    base.atomic_text(
        sha_path,
        digest,
    )

    tag = f"evict-nb05b-unet-seed17-{candidate_name}-rolling"

    release, headers, api = base.create_or_get_release(
        token,
        tag,
        (
            "EviCT Notebook 05B — U-Net seed17 "
            f"{candidate_name} — Rolling Recovery"
        ),
        (
            "Latest rolling remote recovery for the Notebook 05B "
            "U-Net learning-rate pilot. This release asset is replaced "
            "at declared milestones to avoid accumulating redundant "
            "recovery archives. Calibration and target/MedSeg remain locked."
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
        "stage": "NOTEBOOK_05B",
        "candidate_name": candidate_name,
        "learning_rate": learning_rate,
        "seed": SEED,
        "step": step,
        "best_macro_case_dice": best_score,
        "best_step": best_step,
        "archive_name": archive_name,
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
        paths["rolling_audit"],
        json.dumps(audit, indent=2),
    )

    base.git_sync(
        f"Record rolling Notebook 05B U-Net seed17 "
        f"{candidate_name} recovery step {step}"
    )

    archive_path.unlink(missing_ok=True)
    sha_path.unlink(missing_ok=True)

    print(
        f"✓ {candidate_name} remote rolling recovery "
        f"durable @ step {step}"
    )


def rebuild_best_logits(
    *,
    candidate_name: str,
    learning_rate: float,
    paths: dict,
    train_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    device: torch.device,
    expected_init_hash: str,
    config_hash: str,
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
    ) = restore_components_from_checkpoint(
        checkpoint_path=paths["best_pt"],
        candidate_name=candidate_name,
        learning_rate=learning_rate,
        train_df=train_df,
        device=device,
        expected_init_hash=expected_init_hash,
        config_hash=config_hash,
        split_hash=split_hash,
        model_code_hash=model_code_hash,
        metrics_code_hash=metrics_code_hash,
    )

    assert state["global_step"] == best_step
    assert state["best_step"] == best_step
    assert abs(state["best_score"] - best_score) < 1e-10

    validation_metrics, case_rows, raw_logits = base.validate(
        model=model,
        device=device,
        selection_df=selection_df,
        masked_supervised_loss=masked_supervised_loss,
        CaseMetricAccumulator=CaseMetricAccumulator,
        binary_segmentation_metrics=binary_segmentation_metrics,
        metrics_from_confusion=metrics_from_confusion,
        macro_case_summary=macro_case_summary,
    )

    assert abs(
        validation_metrics["macro_case_dice"]
        - best_score
    ) < 1e-8

    base.install_best_logits(
        prediction_dir=paths["pred_dir"],
        raw_logits=raw_logits,
        step=best_step,
        candidate_name=candidate_name,
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


def create_final_archive(
    *,
    candidate_name: str,
    final_step: int,
    paths: dict,
):
    archive_path = WORK / (
        f"EviCT_Notebook05B_unet_seed17_{candidate_name}_"
        f"final_step{final_step}_Recovery.tar"
    )
    archive_path.unlink(missing_ok=True)

    with tarfile.open(archive_path, "w") as archive:
        for checkpoint in [
            paths["best_pt"],
            paths["last_pt"],
            paths["recovery_pt"],
        ]:
            assert checkpoint.exists()
            archive.add(
                checkpoint,
                arcname=str(
                    Path("EviCT/artifacts/large/notebook05")
                    / paths["run_dir"].name
                    / checkpoint.name
                ),
            )

        best_logits_dir = (
            paths["pred_dir"]
            / "best_selection_logits"
        )

        assert best_logits_dir.exists()

        archive.add(
            best_logits_dir,
            arcname=str(
                Path("EviCT/predictions/notebook05")
                / paths["pred_dir"].name
                / "best_selection_logits"
            ),
        )

        small_paths = [
            paths["train_log"],
            paths["selection_log"],
            paths["case_log"],
            AUDIT_ROOT / (
                f"notebook05b_unet_seed17_{candidate_name}_"
                "best_logits_manifest.json"
            ),
            paths["resume_audit"],
            CONFIG_PATH,
            MODEL_CODE,
            METRICS_CODE,
            SPLITS_PATH,
            TRAIN_MANIFEST,
            SELECTION_MANIFEST,
        ]

        for source in small_paths:
            if source.exists():
                try:
                    relative = source.relative_to(ROOT)
                except ValueError:
                    relative = Path(source.name)

                archive.add(
                    source,
                    arcname=str(
                        Path("EviCT") / relative
                    ),
                )

    digest = base.sha256_file(archive_path)
    sha_path = Path(str(archive_path) + ".sha256")
    base.atomic_text(sha_path, digest)

    return archive_path, sha_path, digest


def upload_final_archive(
    *,
    token: str,
    candidate_name: str,
    learning_rate: float,
    final_step: int,
    stop_reason: str,
    best_score: float,
    best_step: int,
    paths: dict,
):
    archive_path, sha_path, digest = create_final_archive(
        candidate_name=candidate_name,
        final_step=final_step,
        paths=paths,
    )

    tag = (
        f"evict-nb05b-unet-seed17-{candidate_name}-"
        f"final-step{final_step}"
    )

    release, headers, api = base.create_or_get_release(
        token,
        tag,
        (
            "EviCT Notebook 05B — U-Net seed17 "
            f"{candidate_name} — Final step {final_step}"
        ),
        (
            "Final durable recovery for the Notebook 05B competitive "
            "residual 2D U-Net learning-rate pilot. "
            f"Candidate={candidate_name}, lr={learning_rate}, "
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
        "stage": "NOTEBOOK_05B",
        "candidate_name": candidate_name,
        "learning_rate": learning_rate,
        "seed": SEED,
        "final_step": final_step,
        "stop_reason": stop_reason,
        "best_macro_case_dice": best_score,
        "best_step": best_step,
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
        paths["final_audit"],
        json.dumps(audit, indent=2),
    )

    return audit


def run_full_candidate(
    *,
    token: str,
    candidate_name: str,
    learning_rate: float,
    pilot_info: dict,
    train_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    device: torch.device,
    expected_init_hash: str,
    config_hash: str,
    split_hash: str,
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

    paths = candidate_paths(candidate_name)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    paths["pred_dir"].mkdir(parents=True, exist_ok=True)
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    if paths["final_audit"].exists():
        existing = json.loads(
            paths["final_audit"].read_text(encoding="utf-8")
        )

        if existing.get("status") == "DURABLE_COMPLETE":
            print(
                f"✓ {candidate_name} already DURABLE_COMPLETE "
                f"at step {existing['final_step']}; skipping training."
            )
            return existing

    ensure_remote_resume_available(
        candidate_name=candidate_name,
        pilot_info=pilot_info,
        paths=paths,
    )

    # Build an initialization-only model once so every checkpoint can be
    # contract-checked before deciding the newest resume state.
    (
        tmp_store,
        tmp_sampler,
        tmp_model,
        tmp_optimizer,
        tmp_scheduler,
        init_hash,
    ) = build_components(
        candidate_name=candidate_name,
        learning_rate=learning_rate,
        train_df=train_df,
        device=device,
        expected_init_hash=expected_init_hash,
    )

    del (
        tmp_store,
        tmp_sampler,
        tmp_model,
        tmp_optimizer,
        tmp_scheduler,
    )
    gc.collect()
    torch.cuda.empty_cache()

    local_checkpoint_candidates = []

    for path in [
        paths["last_pt"],
        paths["recovery_pt"],
    ]:
        if path.exists():
            local_checkpoint_candidates.append(
                checkpoint_metadata(
                    path,
                    candidate_name=candidate_name,
                    learning_rate=learning_rate,
                    config_hash=config_hash,
                    split_hash=split_hash,
                    model_code_hash=model_code_hash,
                    metrics_code_hash=metrics_code_hash,
                    init_hash=init_hash,
                )
            )

    assert local_checkpoint_candidates

    # Self-heal one specific interrupted-audit state:
    # the previous runner could successfully execute 1000->1020 and save
    # recovery.pt, then fail while verifying RNG state before writing the
    # resume-audit JSON. In that case last.pt is still the validated step-1000
    # anchor and recovery.pt is step 1020. We deliberately replay those 20
    # updates from step 1000 rather than accepting an unaudited step-1020 file.
    if not paths["resume_audit"].exists():
        last_meta = next(
            (
                item
                for item in local_checkpoint_candidates
                if item["path"] == paths["last_pt"]
            ),
            None,
        )
        recovery_meta = next(
            (
                item
                for item in local_checkpoint_candidates
                if item["path"] == paths["recovery_pt"]
            ),
            None,
        )

        if (
            last_meta is not None
            and recovery_meta is not None
            and int(last_meta["global_step"]) == 1000
            and int(recovery_meta["global_step"]) == 1020
        ):
            interrupted_record = {
                "timestamp_utc": base.utc_now(),
                "stage": "NOTEBOOK_05B",
                "candidate_name": candidate_name,
                "learning_rate": learning_rate,
                "detected_state": "INTERRUPTED_1000_TO_1020_AUDIT",
                "last_checkpoint_step": 1000,
                "unaudited_recovery_step": 1020,
                "action": (
                    "Discard unaudited local step-1020 recovery and replay "
                    "the 20-update recovery audit from the validated step-1000 last.pt anchor."
                ),
                "calibration_accessed": False,
                "target_accessed": False,
                "status": "REPAIR_APPLIED",
            }

            repair_path = AUDIT_ROOT / (
                f"notebook05b_unet_seed17_{candidate_name}_"
                "interrupted_resume_repair.json"
            )
            base.atomic_text(
                repair_path,
                json.dumps(interrupted_record, indent=2),
            )

            shutil.copy2(
                paths["last_pt"],
                paths["recovery_pt"],
            )

            print(
                f"✓ {candidate_name}: detected interrupted step-1020 "
                "audit; restored validated step-1000 anchor for replay."
            )

            local_checkpoint_candidates = []

            for path in [
                paths["last_pt"],
                paths["recovery_pt"],
            ]:
                if path.exists():
                    local_checkpoint_candidates.append(
                        checkpoint_metadata(
                            path,
                            candidate_name=candidate_name,
                            learning_rate=learning_rate,
                            config_hash=config_hash,
                            split_hash=split_hash,
                            model_code_hash=model_code_hash,
                            metrics_code_hash=metrics_code_hash,
                            init_hash=init_hash,
                        )
                    )

    newest = max(
        local_checkpoint_candidates,
        key=lambda x: x["global_step"],
    )

    print()
    print("=" * 110)
    print(
        f"NOTEBOOK 05B CONTINUATION — {candidate_name} "
        f"(LR={learning_rate:.1e})"
    )
    print("=" * 110)

    print(
        f"Resume checkpoint         : "
        f"{newest['path'].name}"
    )
    print(
        f"Resume optimizer step     : "
        f"{newest['global_step']}"
    )

    train_rows, selection_rows, case_metric_rows = load_logs(
        paths
    )
    deduplicate_logs(
        train_rows,
        selection_rows,
        case_metric_rows,
    )

    # Mandatory first recovery audit when resuming from the durable step-1000 anchor.
    if newest["global_step"] == 1000:
        (
            store,
            sampler,
            model,
            optimizer,
            scheduler,
            init_hash,
            state,
        ) = run_recovery_audit_1000_to_1020(
            candidate_name=candidate_name,
            learning_rate=learning_rate,
            paths=paths,
            pilot_info=pilot_info,
            train_df=train_df,
            device=device,
            expected_init_hash=expected_init_hash,
            config_hash=config_hash,
            split_hash=split_hash,
            model_code_hash=model_code_hash,
            metrics_code_hash=metrics_code_hash,
            train_rows=train_rows,
            selection_rows=selection_rows,
            case_metric_rows=case_metric_rows,
        )
    else:
        assert newest["global_step"] >= 1020

        if not paths["resume_audit"].exists():
            raise RuntimeError(
                f"{candidate_name} is beyond step 1000 but its "
                "1000->1020 recovery audit is missing."
            )

        resume_audit = json.loads(
            paths["resume_audit"].read_text(encoding="utf-8")
        )
        assert resume_audit["status"] == "PASS"

        (
            store,
            sampler,
            model,
            optimizer,
            scheduler,
            init_hash,
            state,
        ) = restore_components_from_checkpoint(
            checkpoint_path=newest["path"],
            candidate_name=candidate_name,
            learning_rate=learning_rate,
            train_df=train_df,
            device=device,
            expected_init_hash=expected_init_hash,
            config_hash=config_hash,
            split_hash=split_hash,
            model_code_hash=model_code_hash,
            metrics_code_hash=metrics_code_hash,
        )

    global_step = state["global_step"]
    best_score = state["best_score"]
    best_step = state["best_step"]
    patience_count = state["patience_count"]
    last_validation_step = state["last_validation_step"]
    images_seen = state["images_seen"]

    assert global_step >= 1020
    assert global_step <= MAX_UPDATES

    model.train()

    progress = tqdm(
        total=MAX_UPDATES - global_step,
        desc=f"{candidate_name} train",
        unit="update",
        ncols=135,
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
            masked_supervised_loss=masked_supervised_loss,
        )

        global_step += 1
        images_seen += EFFECTIVE_BATCH

        train_rows.append({
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
            best=f"{best_score:.4f}",
            patience=f"{patience_count}/{PATIENCE_LIMIT}",
        )
        progress.update(1)

        if (
            global_step % RECOVERY_EVERY == 0
            and global_step % VALIDATE_EVERY != 0
        ):
            save_checkpoint(
                paths=paths,
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
                write_last=False,
            )

        if global_step % VALIDATE_EVERY == 0:
            progress.write(
                f"\n[{candidate_name}] SOURCE-SELECTION "
                f"VALIDATION @ step {global_step}"
            )

            validation_metrics, current_case_rows, raw_logits = base.validate(
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

                best_payload = base.make_checkpoint(
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

                base.atomic_torch_save(
                    best_payload,
                    paths["best_pt"],
                )
                del best_payload

                base.install_best_logits(
                    prediction_dir=paths["pred_dir"],
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

            for row in current_case_rows:
                case_metric_rows.append({
                    "candidate_name": candidate_name,
                    "learning_rate": learning_rate,
                    "seed": SEED,
                    "step": global_step,
                    **row,
                })

            deduplicate_logs(
                train_rows,
                selection_rows,
                case_metric_rows,
            )

            save_checkpoint(
                paths=paths,
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
                write_last=True,
            )

            base.write_logs(
                candidate_name,
                train_rows,
                selection_rows,
                case_metric_rows,
            )

            base.update_state(
                candidate_name=candidate_name,
                learning_rate=learning_rate,
                global_step=global_step,
                best_score=best_score,
                best_step=best_step,
                patience_count=patience_count,
                images_seen=images_seen,
                status=(
                    f"NOTEBOOK_05B_UNET_"
                    f"{candidate_name.upper()}_RUNNING_STEP_{global_step}"
                ),
            )

            progress.write(
                f"  macro case Dice          : "
                f"{score:.8f}\n"
                f"  macro case IoU           : "
                f"{validation_metrics['macro_case_iou']:.8f}\n"
                f"  macro slice Dice         : "
                f"{validation_metrics['macro_slice_dice']:.8f}\n"
                f"  pooled Dice              : "
                f"{validation_metrics['pooled_dice']:.8f}\n"
                f"  best                     : "
                f"{best_score:.8f} @ {best_step}\n"
                f"  patience                 : "
                f"{patience_count}/{PATIENCE_LIMIT}"
            )

            base.git_sync(
                f"Notebook 05B U-Net seed17 "
                f"{candidate_name} validation step {global_step}"
            )

            del raw_logits
            gc.collect()
            torch.cuda.empty_cache()

            if (
                global_step in ROLLING_MILESTONES
                and patience_count < PATIENCE_LIMIT
            ):
                upload_rolling_recovery(
                    token=token,
                    candidate_name=candidate_name,
                    learning_rate=learning_rate,
                    step=global_step,
                    best_score=best_score,
                    best_step=best_step,
                    paths=paths,
                )

            if patience_count >= PATIENCE_LIMIT:
                stop_reason = "EARLY_STOPPING_PATIENCE_8"
                break

    progress.close()

    if stop_reason is None:
        assert global_step == MAX_UPDATES
        stop_reason = "MAXIMUM_5000_UPDATES"

    assert global_step == last_validation_step
    assert paths["best_pt"].exists()
    assert paths["last_pt"].exists()
    assert paths["recovery_pt"].exists()

    # Rebuild/verify best raw logits from the actual selected best checkpoint.
    rebuild_best_logits(
        candidate_name=candidate_name,
        learning_rate=learning_rate,
        paths=paths,
        train_df=train_df,
        selection_df=selection_df,
        device=device,
        expected_init_hash=expected_init_hash,
        config_hash=config_hash,
        split_hash=split_hash,
        model_code_hash=model_code_hash,
        metrics_code_hash=metrics_code_hash,
        best_score=best_score,
        best_step=best_step,
    )

    final_audit = upload_final_archive(
        token=token,
        candidate_name=candidate_name,
        learning_rate=learning_rate,
        final_step=global_step,
        stop_reason=stop_reason,
        best_score=best_score,
        best_step=best_step,
        paths=paths,
    )

    base.update_state(
        candidate_name=candidate_name,
        learning_rate=learning_rate,
        global_step=global_step,
        best_score=best_score,
        best_step=best_step,
        patience_count=patience_count,
        images_seen=images_seen,
        status=(
            f"NOTEBOOK_05B_UNET_"
            f"{candidate_name.upper()}_COMPLETE_DURABLE"
        ),
    )

    base.git_sync(
        f"Complete durable Notebook 05B U-Net seed17 "
        f"{candidate_name} at step {global_step}"
    )

    del model, optimizer, scheduler, sampler, store
    gc.collect()
    torch.cuda.empty_cache()

    print()
    print(
        f"✓ {candidate_name} DURABLE_COMPLETE | "
        f"final step={global_step} | "
        f"best Dice={best_score:.8f} @ {best_step}"
    )

    return final_audit


def freeze_learning_rate(
    *,
    results: list[dict],
    expected_init_hash: str,
):
    assert len(results) == 2

    sorted_results = sorted(
        results,
        key=lambda x: float(x["best_macro_case_dice"]),
        reverse=True,
    )

    top = sorted_results[0]
    second = sorted_results[1]

    difference = (
        float(top["best_macro_case_dice"])
        - float(second["best_macro_case_dice"])
    )

    if abs(difference) <= 1e-12:
        raise RuntimeError(
            "The two LR candidates tied exactly on the frozen primary "
            "selection metric. No tie-break rule was predeclared, so the "
            "learning rate is NOT frozen automatically."
        )

    selected = {
        "timestamp_utc": base.utc_now(),
        "stage": "NOTEBOOK_05B",
        "baseline": "competitive_residual_unet2d",
        "pilot_seed": SEED,
        "architecture_tuning": False,
        "selection_metric": "source-selection macro case Dice",
        "threshold": THRESHOLD,
        "threshold_tuned": False,
        "candidate_initialization_identical": True,
        "initialization_hash": expected_init_hash,
        "selected_candidate": top["candidate_name"],
        "selected_learning_rate": float(top["learning_rate"]),
        "selected_best_macro_case_dice": float(
            top["best_macro_case_dice"]
        ),
        "selected_best_step": int(top["best_step"]),
        "selected_final_step": int(top["final_step"]),
        "selected_stop_reason": top["stop_reason"],
        "other_candidate": second["candidate_name"],
        "other_learning_rate": float(second["learning_rate"]),
        "other_best_macro_case_dice": float(
            second["best_macro_case_dice"]
        ),
        "selection_margin": float(difference),
        "calibration_accessed": False,
        "target_accessed": False,
        "status": "LEARNING_RATE_FROZEN",
    }

    base.atomic_text(
        SELECTED_LR_PATH,
        json.dumps(selected, indent=2),
    )

    text = f"""# EviCT Execution State

## Current stage

NOTEBOOK_05B_UNET_LEARNING_RATE_FROZEN

## Timestamp

{base.utc_now()}

## Baseline

Competitive residual 2D U-Net

Pilot seed:

17

Architecture tuning:

NO

Candidate initialization:

IDENTICAL

Initialization SHA-256:

{expected_init_hash}

## Candidate 1

Learning rate:

{results[0]['learning_rate']:.8f}

Final optimizer step:

{results[0]['final_step']}

Stop reason:

{results[0]['stop_reason']}

Best source-selection macro case Dice:

{results[0]['best_macro_case_dice']:.8f}

Best checkpoint step:

{results[0]['best_step']}

Durable release:

{results[0]['release_url']}

## Candidate 2

Learning rate:

{results[1]['learning_rate']:.8f}

Final optimizer step:

{results[1]['final_step']}

Stop reason:

{results[1]['stop_reason']}

Best source-selection macro case Dice:

{results[1]['best_macro_case_dice']:.8f}

Best checkpoint step:

{results[1]['best_step']}

Durable release:

{results[1]['release_url']}

## Frozen learning rate

Selected candidate:

{top['candidate_name']}

Selected learning rate:

{float(top['learning_rate']):.8f}

Selection metric:

Frozen source-selection macro case Dice

Selection margin:

{difference:.8f}

Threshold:

0.5 fixed

## Isolation

Calibration accessed:

NO

Target / MedSeg accessed:

NO

Target lock:

ACTIVE

## Next

Keep the selected U-Net architecture and learning rate fixed.
Run the remaining primary training seeds 42 and 2026 under the
same source-only protocol before final U-Net aggregation.
"""

    base.atomic_text(
        STATE_PATH,
        text,
    )
    base.atomic_text(
        HANDOFF_STATE,
        text,
    )

    base.git_sync(
        "Freeze Notebook 05B U-Net learning rate after full seed17 pilot"
    )

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["audit-and-continue"],
        default="audit-and-continue",
    )
    args = parser.parse_args()

    assert args.mode == "audit-and-continue"

    print("=" * 110)
    print(
        "EVICT NOTEBOOK 05B — RECOVERY AUDIT + "
        "FULL LR PILOT CONTINUATION"
    )
    print("=" * 110)

    assert ROOT.exists() and (ROOT / ".git").exists()
    assert BASE_RUNNER_PATH.exists()
    assert CONFIG_PATH.exists()
    assert MODEL_CODE.exists()
    assert METRICS_CODE.exists()
    assert SPLITS_PATH.exists()
    assert TRAIN_MANIFEST.exists()
    assert SELECTION_MANIFEST.exists()
    assert PILOT_AUDIT_PATH.exists()
    assert GIT_SYNC.exists()

    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))

    token = UserSecretsClient().get_secret("pushEviCT")
    assert token
    print("✓ GitHub secret            : PASS")

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    print(
        f"✓ GPU                      : "
        f"{torch.cuda.get_device_name(device)}"
    )

    base.ensure_cache()

    config = json.loads(
        CONFIG_PATH.read_text(encoding="utf-8")
    )

    assert config["stage"] == "NOTEBOOK_05"
    assert config["baseline_name"] == "competitive_residual_unet2d"
    assert config["architecture"]["pretraining"] == "none"
    assert config["optimizer_plan"]["learning_rate_candidates"] == [
        0.0001,
        0.0003,
    ]
    assert config["data_contract"]["target_access_allowed"] is False
    assert config["validation"]["threshold_tuning"] is False
    assert config["batch"]["micro_batch"] == MICRO_BATCH
    assert config["batch"]["gradient_accumulation"] == GRAD_ACCUM
    assert config["batch"]["images_per_optimizer_update"] == EFFECTIVE_BATCH

    pilot = json.loads(
        PILOT_AUDIT_PATH.read_text(encoding="utf-8")
    )

    assert pilot["status"] == "BOTH_CANDIDATES_STEP1000_DURABLE"
    assert pilot["identical_initialization"] is True
    assert pilot["candidate_winner_selected"] is False
    assert pilot["calibration_accessed"] is False
    assert pilot["target_accessed"] is False

    expected_init_hash = pilot["initialization_hash"]

    pilot_by_name = {
        item["candidate_name"]: item
        for item in pilot["candidates"]
    }

    assert set(pilot_by_name) == {"lr1e4", "lr3e4"}

    split_df = pd.read_csv(SPLITS_PATH)
    train_df = pd.read_csv(TRAIN_MANIFEST)
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
    assert set(train_df["seed"].astype(int)) == {17}

    manifest_text = (
        " ".join(
            train_df.fillna("").astype(str).values.ravel()
        )
        + " "
        + " ".join(
            selection_df.fillna("").astype(str).values.ravel()
        )
    ).lower()

    for forbidden in [
        "medseg",
        "segdb1",
        "images_medseg",
        "masks_medseg",
    ]:
        assert forbidden not in manifest_text

    print("✓ Frozen fitting cases     : 12")
    print("✓ Frozen selection cases   : 4")
    print("✓ Calibration accessed     : NO")
    print("✓ Target / MedSeg accessed : NO")
    print("✓ Target lock              : ACTIVE")

    config_hash = base.stable_json_hash(config)
    split_hash = base.stable_json_hash({
        "splits_csv": base.sha256_file(SPLITS_PATH),
        "train_manifest": base.sha256_file(TRAIN_MANIFEST),
        "selection_manifest": base.sha256_file(
            SELECTION_MANIFEST
        ),
    })
    model_code_hash = base.sha256_file(MODEL_CODE)
    metrics_code_hash = base.sha256_file(METRICS_CODE)

    results = []

    for candidate_name, learning_rate in [
        ("lr1e4", 1.0e-4),
        ("lr3e4", 3.0e-4),
    ]:
        result = run_full_candidate(
            token=token,
            candidate_name=candidate_name,
            learning_rate=learning_rate,
            pilot_info=pilot_by_name[candidate_name],
            train_df=train_df,
            selection_df=selection_df,
            device=device,
            expected_init_hash=expected_init_hash,
            config_hash=config_hash,
            split_hash=split_hash,
            model_code_hash=model_code_hash,
            metrics_code_hash=metrics_code_hash,
        )
        results.append(result)

    selected = freeze_learning_rate(
        results=results,
        expected_init_hash=expected_init_hash,
    )

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

    print()
    print("=" * 110)
    print(
        "EVICT NOTEBOOK 05B — FULL LR PILOT — "
        "DURABLE COMPLETION PASS"
    )
    print("=" * 110)

    for result in results:
        print(
            f"{result['candidate_name']:6s} | "
            f"LR={float(result['learning_rate']):.1e} | "
            f"final={result['final_step']} | "
            f"stop={result['stop_reason']} | "
            f"best Dice={result['best_macro_case_dice']:.8f} "
            f"@ {result['best_step']}"
        )

    print()
    print(
        f"SELECTED CANDIDATE        : "
        f"{selected['selected_candidate']}"
    )
    print(
        f"FROZEN LEARNING RATE      : "
        f"{selected['selected_learning_rate']:.8f}"
    )
    print(
        f"Selection margin          : "
        f"{selected['selection_margin']:.8f}"
    )
    print("Recovery audits           : PASS")
    print("Final archives            : REMOTELY DURABLE")
    print("Calibration accessed      : NO")
    print("Target / MedSeg accessed  : NO")
    print("Target lock               : ACTIVE")
    print(f"GitHub HEAD               : {local_head}")
    print()
    print(
        "NEXT: train the frozen U-Net configuration with "
        "primary seeds 42 and 2026."
    )


if __name__ == "__main__":
    main()
