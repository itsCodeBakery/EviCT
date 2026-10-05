from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import requests
import torch


DEFAULT_REPO = "itsCodeBakery/EviCT"
DEFAULT_SECRET_NAME = "pushEviCT"


@dataclass(frozen=True)
class RecoveryPolicy:
    checkpoint_every_steps: int = 250
    remote_checkpoint_every_steps: int = 500
    metadata_sync_every_steps: int = 250
    keep_local_generations: int = 2
    github_repository: str = DEFAULT_REPO
    github_branch: str = "main"
    kaggle_secret_name: str = DEFAULT_SECRET_NAME
    rolling_release_prefix: str = "evict-recovery"
    max_release_asset_gb: float = 1.90


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256_bytes(payload)


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def git_commit_sha(repo_root: Path) -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def _rotate_checkpoint_generations(path: Path, keep: int) -> None:
    path = Path(path)
    if keep <= 1 or not path.exists():
        return
    for idx in range(keep - 1, 0, -1):
        src = path.with_name(f"{path.stem}.previous{idx - 1 if idx > 1 else ''}{path.suffix}")
        dst = path.with_name(f"{path.stem}.previous{idx}{path.suffix}")
        if src.exists():
            os.replace(src, dst)
    previous = path.with_name(f"{path.stem}.previous{path.suffix}")
    shutil.copy2(path, previous)


def atomic_torch_save(payload: Dict[str, Any], path: Path, keep_generations: int = 2) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)

    # Verify that the file is structurally loadable before replacing last-known-good.
    verified = torch.load(tmp, map_location="cpu", weights_only=False)
    if not isinstance(verified, dict):
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Checkpoint verification failed: payload is not a dictionary.")

    _rotate_checkpoint_generations(path, keep_generations)
    os.replace(tmp, path)
    return sha256_file(path)


def load_checkpoint(path: Path, expected_sha256: Optional[str] = None) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if expected_sha256 and actual != expected_sha256:
        raise RuntimeError(
            f"Checkpoint hash mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint payload: {path}")
    return payload


def build_training_checkpoint(
    *,
    student: torch.nn.Module,
    global_step: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    ema_teacher: Optional[torch.nn.Module] = None,
    epoch: Optional[int] = None,
    best_score: Optional[float] = None,
    patience_counter: Optional[int] = None,
    sampler_state: Optional[Any] = None,
    config_hash: Optional[str] = None,
    split_hash: Optional[str] = None,
    manifest_hash: Optional[str] = None,
    run_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "created_utc": utc_now(),
        "run_id": run_id,
        "global_step": int(global_step),
        "epoch": epoch,
        "best_score": best_score,
        "patience_counter": patience_counter,
        "student_state_dict": student.state_dict(),
        "ema_teacher_state_dict": ema_teacher.state_dict() if ema_teacher is not None else None,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "sampler_state": sampler_state,
        "rng_state": capture_rng_state(),
        "config_hash": config_hash,
        "split_hash": split_hash,
        "manifest_hash": manifest_hash,
        "extra": extra or {},
    }


def restore_training_checkpoint(
    payload: Dict[str, Any],
    *,
    student: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    ema_teacher: Optional[torch.nn.Module] = None,
    strict: bool = True,
    restore_rng: bool = True,
) -> Dict[str, Any]:
    student.load_state_dict(payload["student_state_dict"], strict=strict)

    teacher_state = payload.get("ema_teacher_state_dict")
    if ema_teacher is not None and teacher_state is not None:
        ema_teacher.load_state_dict(teacher_state, strict=strict)

    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])

    if restore_rng and payload.get("rng_state") is not None:
        restore_rng_state(payload["rng_state"])

    return {
        "global_step": int(payload.get("global_step", 0)),
        "epoch": payload.get("epoch"),
        "best_score": payload.get("best_score"),
        "patience_counter": payload.get("patience_counter"),
        "sampler_state": payload.get("sampler_state"),
        "config_hash": payload.get("config_hash"),
        "split_hash": payload.get("split_hash"),
        "manifest_hash": payload.get("manifest_hash"),
        "extra": payload.get("extra", {}),
    }


def kaggle_secret(name: str = DEFAULT_SECRET_NAME) -> str:
    from kaggle_secrets import UserSecretsClient

    token = UserSecretsClient().get_secret(name)
    if not token:
        raise RuntimeError(f"Kaggle Secret {name!r} is unavailable.")
    return token


class GitHubReleaseStore:
    """Rolling large-artifact storage using a GitHub Release asset.

    Ordinary git remains the control plane for code, JSON/CSV state, hashes, and logs.
    Large checkpoints are uploaded as release assets rather than committed to the git history.
    """

    def __init__(
        self,
        *,
        repository: str = DEFAULT_REPO,
        token: str,
        release_prefix: str = "evict-recovery",
        timeout: int = 120,
    ) -> None:
        self.repository = repository
        self.token = token
        self.release_prefix = release_prefix
        self.timeout = timeout
        self.api = f"https://api.github.com/repos/{repository}"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def rolling_tag(self, run_id: str) -> str:
        safe = "".join(c if c.isalnum() or c in ".-_" else "-" for c in run_id)
        return f"{self.release_prefix}-{safe}-rolling"

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        last: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                r = requests.request(
                    method,
                    url,
                    headers=kwargs.pop("headers", self.headers),
                    timeout=kwargs.pop("timeout", self.timeout),
                    **kwargs,
                )
                if r.status_code >= 400:
                    raise RuntimeError(f"GitHub HTTP {r.status_code}: {r.text[:1000]}")
                return r
            except Exception as exc:
                last = exc
                if attempt < 3:
                    time.sleep(attempt * 3)
        raise RuntimeError(f"GitHub request failed after retries: {last}")

    def ensure_release(self, run_id: str) -> Dict[str, Any]:
        tag = self.rolling_tag(run_id)
        r = requests.get(
            f"{self.api}/releases/tags/{tag}",
            headers=self.headers,
            timeout=self.timeout,
        )
        if r.status_code == 200:
            return r.json()
        if r.status_code != 404:
            raise RuntimeError(f"Unable to query release {tag}: HTTP {r.status_code} {r.text[:800]}")

        body = {
            "tag_name": tag,
            "target_commitish": "main",
            "name": f"EviCT recovery: {run_id}",
            "body": "Rolling recovery checkpoint. Machine-managed; do not use as a scientific result by itself.",
            "draft": False,
            "prerelease": True,
        }
        return self._request("POST", f"{self.api}/releases", json=body).json()

    def upload_or_replace(self, run_id: str, path: Path, asset_name: Optional[str] = None) -> Dict[str, Any]:
        path = Path(path)
        release = self.ensure_release(run_id)
        asset_name = asset_name or path.name

        assets = self._request("GET", f"{self.api}/releases/{release['id']}/assets").json()
        for asset in assets:
            if asset.get("name") == asset_name:
                self._request("DELETE", f"{self.api}/releases/assets/{asset['id']}")

        upload_url = release["upload_url"].split("{", 1)[0]
        headers = dict(self.headers)
        headers["Content-Type"] = "application/octet-stream"
        with path.open("rb") as f:
            uploaded = self._request(
                "POST",
                upload_url,
                headers=headers,
                params={"name": asset_name},
                data=f,
                timeout=max(self.timeout, 600),
            ).json()
        return uploaded

    def download(self, run_id: str, asset_name: str, destination: Path) -> Dict[str, Any]:
        release = self.ensure_release(run_id)
        assets = self._request("GET", f"{self.api}/releases/{release['id']}/assets").json()
        asset = next((x for x in assets if x.get("name") == asset_name), None)
        if asset is None:
            raise FileNotFoundError(f"Release asset {asset_name!r} not found for run {run_id}")

        headers = dict(self.headers)
        headers["Accept"] = "application/octet-stream"
        r = self._request(
            "GET",
            f"{self.api}/releases/assets/{asset['id']}",
            headers=headers,
            timeout=max(self.timeout, 600),
            stream=True,
        )
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(destination.suffix + ".download")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, destination)
        return asset


class RunRecoveryManager:
    def __init__(
        self,
        repo_root: Path,
        run_id: str,
        *,
        policy: Optional[RecoveryPolicy] = None,
        run_dir: Optional[Path] = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.run_id = run_id
        self.policy = policy or RecoveryPolicy()
        self.run_dir = Path(run_dir) if run_dir else self.repo_root / "runs" / run_id
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.state_path = self.run_dir / "STATE.json"
        self.state_md_path = self.run_dir / "STATE.md"
        self.manifest_path = self.run_dir / "checkpoint_manifest.json"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def read_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def write_state(self, **updates: Any) -> Dict[str, Any]:
        state = self.read_state()
        state.update(updates)
        state.setdefault("run_id", self.run_id)
        state["updated_utc"] = utc_now()
        state["git_commit"] = git_commit_sha(self.repo_root)
        state["recovery_policy"] = asdict(self.policy)
        atomic_write_json(self.state_path, state)
        atomic_write_text(self.state_md_path, self._state_markdown(state))
        return state

    @staticmethod
    def _state_markdown(state: Dict[str, Any]) -> str:
        keys = [
            "run_id",
            "status",
            "global_step",
            "epoch",
            "best_score",
            "latest_checkpoint",
            "latest_checkpoint_sha256",
            "remote_asset",
            "remote_checkpoint_sha256",
            "config_hash",
            "split_hash",
            "manifest_hash",
            "git_commit",
            "updated_utc",
            "next_action",
        ]
        rows = ["# EviCT Run Recovery State", ""]
        for key in keys:
            if key in state and state[key] is not None:
                rows.append(f"- **{key}**: {state[key]}")
        rows.append("")
        rows.append("This file is machine-managed. Scientific settings must not be changed from target results.")
        rows.append("")
        return "\n".join(rows)

    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "last.pt"

    def save_training_checkpoint(
        self,
        *,
        student: torch.nn.Module,
        global_step: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        ema_teacher: Optional[torch.nn.Module] = None,
        epoch: Optional[int] = None,
        best_score: Optional[float] = None,
        patience_counter: Optional[int] = None,
        sampler_state: Optional[Any] = None,
        config_hash: Optional[str] = None,
        split_hash: Optional[str] = None,
        manifest_hash: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        status: str = "training",
        remote: bool = False,
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = build_training_checkpoint(
            student=student,
            global_step=global_step,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema_teacher=ema_teacher,
            epoch=epoch,
            best_score=best_score,
            patience_counter=patience_counter,
            sampler_state=sampler_state,
            config_hash=config_hash,
            split_hash=split_hash,
            manifest_hash=manifest_hash,
            run_id=self.run_id,
            extra=extra,
        )
        path = self.checkpoint_path()
        digest = atomic_torch_save(
            payload,
            path,
            keep_generations=self.policy.keep_local_generations,
        )

        state = self.write_state(
            status=status,
            global_step=int(global_step),
            epoch=epoch,
            best_score=best_score,
            latest_checkpoint=str(path.relative_to(self.repo_root)),
            latest_checkpoint_sha256=digest,
            config_hash=config_hash,
            split_hash=split_hash,
            manifest_hash=manifest_hash,
            next_action=f"resume_from_step_{int(global_step) + 1}",
        )

        manifest = {
            "run_id": self.run_id,
            "checkpoint": state["latest_checkpoint"],
            "sha256": digest,
            "global_step": int(global_step),
            "created_utc": utc_now(),
            "git_commit": state.get("git_commit"),
            "config_hash": config_hash,
            "split_hash": split_hash,
            "manifest_hash": manifest_hash,
        }

        if remote:
            token = token or kaggle_secret(self.policy.kaggle_secret_name)
            max_bytes = int(self.policy.max_release_asset_gb * (1024 ** 3))
            if path.stat().st_size > max_bytes:
                raise RuntimeError(
                    f"Checkpoint is {path.stat().st_size / (1024 ** 3):.2f} GiB; "
                    f"larger than configured release-asset safety limit."
                )
            store = GitHubReleaseStore(
                repository=self.policy.github_repository,
                token=token,
                release_prefix=self.policy.rolling_release_prefix,
            )
            asset_name = "last.pt"
            uploaded = store.upload_or_replace(self.run_id, path, asset_name=asset_name)
            manifest["remote_asset"] = asset_name
            manifest["remote_asset_id"] = uploaded.get("id")
            manifest["remote_release_tag"] = store.rolling_tag(self.run_id)
            manifest["remote_checkpoint_sha256"] = digest
            state = self.write_state(
                remote_asset=asset_name,
                remote_release_tag=manifest["remote_release_tag"],
                remote_checkpoint_sha256=digest,
            )

        atomic_write_json(self.manifest_path, manifest)
        return state

    def restore(
        self,
        *,
        student: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        ema_teacher: Optional[torch.nn.Module] = None,
        strict: bool = True,
        expected_config_hash: Optional[str] = None,
        expected_split_hash: Optional[str] = None,
        expected_manifest_hash: Optional[str] = None,
        allow_remote: bool = True,
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        state = self.read_state()
        path = self.checkpoint_path()

        if not path.exists() and allow_remote and state.get("remote_asset"):
            token = token or kaggle_secret(self.policy.kaggle_secret_name)
            store = GitHubReleaseStore(
                repository=self.policy.github_repository,
                token=token,
                release_prefix=self.policy.rolling_release_prefix,
            )
            store.download(self.run_id, state["remote_asset"], path)

        expected_sha = state.get("latest_checkpoint_sha256")
        if state.get("remote_checkpoint_sha256") and not path.exists():
            expected_sha = state["remote_checkpoint_sha256"]

        payload = load_checkpoint(path, expected_sha256=expected_sha)

        checks = {
            "config_hash": expected_config_hash,
            "split_hash": expected_split_hash,
            "manifest_hash": expected_manifest_hash,
        }
        for field, expected in checks.items():
            actual = payload.get(field)
            if expected is not None and actual != expected:
                raise RuntimeError(
                    f"Unsafe resume: {field} mismatch. expected={expected} checkpoint={actual}"
                )

        restored = restore_training_checkpoint(
            payload,
            student=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema_teacher=ema_teacher,
            strict=strict,
            restore_rng=True,
        )
        self.write_state(
            status="resumed",
            global_step=restored["global_step"],
            epoch=restored.get("epoch"),
            best_score=restored.get("best_score"),
            next_action=f"resume_from_step_{restored['global_step'] + 1}",
        )
        return restored

    def should_checkpoint(self, global_step: int) -> bool:
        return global_step > 0 and global_step % self.policy.checkpoint_every_steps == 0

    def should_remote_checkpoint(self, global_step: int) -> bool:
        return global_step > 0 and global_step % self.policy.remote_checkpoint_every_steps == 0

    def should_sync_metadata(self, global_step: int) -> bool:
        return global_step > 0 and global_step % self.policy.metadata_sync_every_steps == 0

    def sync_metadata(self, message: Optional[str] = None) -> int:
        script = self.repo_root / "scripts" / "git_sync.py"
        if not script.exists():
            raise FileNotFoundError(script)
        r = subprocess.run(
            [
                sys.executable,
                str(script),
                message or f"EviCT recovery state: {self.run_id}",
            ],
            cwd=str(self.repo_root),
            check=False,
        )
        return int(r.returncode)

    def register_signal_checkpoint(self, callback: Callable[[str], None]) -> None:
        def handler(signum: int, _frame: Any) -> None:
            name = signal.Signals(signum).name
            callback(name)

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, handler)


def print_resume_banner(run_id: str, state: Dict[str, Any]) -> None:
    print("=" * 88)
    print("EVICT RESUME MANAGER")
    print("=" * 88)
    print(f"Run ID                 : {run_id}")
    print(f"Status                 : {state.get('status', 'UNKNOWN')}")
    print(f"Global step            : {state.get('global_step', 0)}")
    print(f"Checkpoint             : {state.get('latest_checkpoint', 'NONE')}")
    print(f"Checkpoint SHA256      : {state.get('latest_checkpoint_sha256', 'NONE')}")
    print(f"Remote asset           : {state.get('remote_asset', 'NONE')}")
    print(f"Git revision           : {state.get('git_commit', 'UNKNOWN')}")
    print(f"Config hash            : {state.get('config_hash', 'UNKNOWN')}")
    print(f"Split hash             : {state.get('split_hash', 'UNKNOWN')}")
    print(f"Manifest hash          : {state.get('manifest_hash', 'UNKNOWN')}")
    print(f"Next action            : {state.get('next_action', 'initialize_or_resume')}")
    print("=" * 88)
