from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/kaggle/working/EviCT")
DX = ROOT / "vlmDiagnosis"
if str(DX) not in sys.path:
    sys.path.insert(0, str(DX))

from runtime.recovery import (  # noqa: E402
    GitHubReleaseStore,
    RecoveryPolicy,
    RunRecoveryManager,
    kaggle_secret,
    load_checkpoint,
    print_resume_banner,
    sha256_file,
)


def load_policy(root: Path) -> RecoveryPolicy:
    path = root / "vlmDiagnosis" / "config" / "recovery_policy.json"
    if not path.exists():
        return RecoveryPolicy()
    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = set(RecoveryPolicy.__dataclass_fields__)
    return RecoveryPolicy(**{k: v for k, v in data.items() if k in allowed})


def cmd_status(manager: RunRecoveryManager) -> None:
    state = manager.read_state()
    print_resume_banner(manager.run_id, state)
    cp = manager.checkpoint_path()
    print(f"Local checkpoint exists : {cp.exists()}")
    if cp.exists():
        print(f"Local checkpoint SHA256 : {sha256_file(cp)}")


def cmd_verify(manager: RunRecoveryManager) -> None:
    state = manager.read_state()
    cp = manager.checkpoint_path()
    if not cp.exists():
        raise FileNotFoundError(f"No local checkpoint: {cp}")
    expected = state.get("latest_checkpoint_sha256")
    load_checkpoint(cp, expected_sha256=expected)
    print("✓ checkpoint structure loadable")
    print("✓ checkpoint SHA256 verified")
    print(f"✓ {cp}")


def cmd_pull_remote(manager: RunRecoveryManager) -> None:
    token = kaggle_secret(manager.policy.kaggle_secret_name)
    store = GitHubReleaseStore(
        repository=manager.policy.github_repository,
        token=token,
        release_prefix=manager.policy.rolling_release_prefix,
    )

    state = manager.read_state()
    if not state:
        # A previous metadata Git push may have been deferred by a concurrent
        # main-branch update. Recovery metadata is therefore also mirrored as
        # a release asset beside the step-versioned checkpoint.
        try:
            store.download(manager.run_id, "STATE.json", manager.state_path)
            state = manager.read_state()
            print("✓ recovered STATE.json from rolling release")
        except Exception as exc:
            raise RuntimeError(
                "No local run state and remote STATE.json recovery failed. "
                f"Run ID: {manager.run_id}. Error: {exc}"
            ) from exc

    asset = state.get("remote_asset", "last.pt")
    expected = state.get("remote_checkpoint_sha256")
    cp = manager.checkpoint_path()
    store.download(manager.run_id, asset, cp)
    actual = sha256_file(cp)
    if expected and actual != expected:
        cp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Remote checkpoint hash mismatch: expected {expected}, got {actual}"
        )

    try:
        store.download(
            manager.run_id,
            "checkpoint_manifest.json",
            manager.manifest_path,
        )
    except Exception:
        pass

    print(f"✓ restored remote checkpoint: {cp}")
    print(f"✓ SHA256: {actual}")


def cmd_sync(manager: RunRecoveryManager, message: str | None) -> None:
    rc = manager.sync_metadata(message)
    if rc != 0:
        raise RuntimeError(f"Metadata Git sync failed with return code {rc}.")
    print("✓ recovery metadata synchronized to GitHub")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and recover EviCT long-running Kaggle jobs."
    )
    parser.add_argument("command", choices=["status", "verify", "pull-remote", "sync"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--message", default=None)
    args = parser.parse_args()

    root = Path(args.repo_root)
    policy = load_policy(root)
    manager = RunRecoveryManager(
        root,
        args.run_id,
        policy=policy,
        run_dir=root / "vlmDiagnosis" / "runs" / args.run_id,
    )

    if args.command == "status":
        cmd_status(manager)
    elif args.command == "verify":
        cmd_verify(manager)
    elif args.command == "pull-remote":
        cmd_pull_remote(manager)
    elif args.command == "sync":
        cmd_sync(manager, args.message)


if __name__ == "__main__":
    main()
