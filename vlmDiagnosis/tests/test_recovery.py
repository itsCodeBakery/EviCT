from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
DX = ROOT / "vlmDiagnosis"
sys.path.insert(0, str(DX))

from runtime.recovery import (  # noqa: E402
    RecoveryPolicy,
    RunRecoveryManager,
    atomic_torch_save,
    build_training_checkpoint,
    load_checkpoint,
    restore_training_checkpoint,
    sha256_file,
    sha256_json,
)


def test_sha256_json_is_order_independent():
    a = {"b": 2, "a": 1}
    b = {"a": 1, "b": 2}
    assert sha256_json(a) == sha256_json(b)


def test_atomic_checkpoint_roundtrip_restores_training_state(tmp_path: Path):
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)

    model = torch.nn.Linear(4, 2)
    teacher = torch.nn.Linear(4, 2)
    teacher.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    x = torch.randn(3, 4)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    expected_weight = model.weight.detach().clone()
    payload = build_training_checkpoint(
        student=model,
        ema_teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=250,
        epoch=3,
        best_score=0.71,
        patience_counter=2,
        sampler_state={"epoch": 3, "offset": 48},
        config_hash="cfg",
        split_hash="split",
        manifest_hash="manifest",
        run_id="test-run",
    )

    path = tmp_path / "last.pt"
    digest = atomic_torch_save(payload, path)
    assert digest == sha256_file(path)

    with torch.no_grad():
        model.weight.zero_()

    loaded = load_checkpoint(path, expected_sha256=digest)
    meta = restore_training_checkpoint(
        loaded,
        student=model,
        ema_teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    assert torch.allclose(model.weight, expected_weight)
    assert meta["global_step"] == 250
    assert meta["epoch"] == 3
    assert meta["sampler_state"]["offset"] == 48
    assert meta["config_hash"] == "cfg"
    assert meta["split_hash"] == "split"
    assert meta["manifest_hash"] == "manifest"


def test_checkpoint_hash_mismatch_is_rejected(tmp_path: Path):
    model = torch.nn.Linear(2, 1)
    payload = build_training_checkpoint(student=model, global_step=1)
    path = tmp_path / "last.pt"
    atomic_torch_save(payload, path)

    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_checkpoint(path, expected_sha256="0" * 64)


def test_run_manager_writes_machine_readable_and_markdown_state(tmp_path: Path):
    policy = RecoveryPolicy(
        checkpoint_every_steps=10,
        remote_checkpoint_every_steps=20,
        metadata_sync_every_steps=10,
    )
    manager = RunRecoveryManager(tmp_path, "run-1", policy=policy, run_dir=tmp_path / "vlmDiagnosis" / "runs" / "run-1")
    state = manager.write_state(
        status="training",
        global_step=10,
        config_hash="cfg",
        split_hash="split",
        manifest_hash="manifest",
        next_action="resume_from_step_11",
    )

    assert manager.state_path.exists()
    assert manager.state_md_path.exists()
    on_disk = json.loads(manager.state_path.read_text(encoding="utf-8"))
    assert on_disk["run_id"] == "run-1"
    assert on_disk["global_step"] == 10
    assert state["recovery_policy"]["checkpoint_every_steps"] == 10
    assert "resume_from_step_11" in manager.state_md_path.read_text(encoding="utf-8")


def test_checkpoint_intervals():
    policy = RecoveryPolicy(
        checkpoint_every_steps=250,
        remote_checkpoint_every_steps=500,
        metadata_sync_every_steps=250,
    )
    manager = RunRecoveryManager(Path("."), "interval-test", policy=policy, run_dir=Path("/tmp/evict_dx_interval_test"))
    assert manager.should_checkpoint(250)
    assert manager.should_checkpoint(500)
    assert not manager.should_checkpoint(251)
    assert manager.should_remote_checkpoint(500)
    assert not manager.should_remote_checkpoint(250)
    assert manager.should_sync_metadata(250)
