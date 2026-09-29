from __future__ import annotations

import base64
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path("/kaggle/working/EviCT")
REPO_API = "https://api.github.com/repos/itsCodeBakery/EviCT"
REPO_URL = "https://github.com/itsCodeBakery/EviCT.git"
SECRET_NAME = "pushEviCT"
OPENCLIP_VERSION = "3.3.0"

LATEST_REQUIRED_FILES = [
    "scripts/nb06_semantic_branch.py",
    "scripts/git_sync.py",
    "scripts/nb06_isolated_orchestrator.py",
]


def run(cmd, *, cwd=None, timeout=120, check=True, capture=False, env=None):
    r = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        env=env,
    )
    if check and r.returncode != 0:
        if capture and r.stdout:
            print(r.stdout)
        raise RuntimeError("Command failed: " + " ".join(map(str, cmd)))
    return r


def active_nb06_processes():
    ps = subprocess.check_output(["ps", "-eo", "pid,ppid,etime,args"], text=True)
    out = []
    me = os.getpid()
    parent = os.getppid()
    for line in ps.splitlines()[1:]:
        low = line.lower()
        if (
            "nb06_semantic_branch.py" in low
            or "nb06_isolated_orchestrator.py" in low
        ):
            parts = line.strip().split(None, 3)
            if not parts:
                continue
            try:
                pid = int(parts[0])
            except Exception:
                continue
            if pid not in {me, parent}:
                out.append(line.strip())
    return out


def github_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }


def get_latest_file(token, path):
    url = f"{REPO_API}/contents/{path}?ref=main"
    last = None
    for attempt in range(1, 4):
        try:
            r = requests.get(url, headers=github_headers(token), timeout=30)
            r.raise_for_status()
            j = r.json()
            content = base64.b64decode(j["content"])
            print(f"✓ latest code fetched       : {path}")
            return content
        except Exception as exc:
            last = exc
            print(f"⚠ code fetch attempt {attempt}/3 failed for {path}: {exc}")
            if attempt < 3:
                time.sleep(5 * attempt)
    raise RuntimeError(f"Could not fetch latest {path}: {last}")


def clone_repo(token):
    auth = base64.b64encode(f"itsCodeBakery:{token}".encode()).decode()
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    run(
        [
            "git",
            "-c",
            f"http.extraHeader=AUTHORIZATION: basic {auth}",
            "clone",
            "--branch",
            "main",
            "--single-branch",
            REPO_URL,
            str(ROOT),
        ],
        timeout=180,
        env=env,
    )


def overwrite_execution_files(token):
    for rel in LATEST_REQUIRED_FILES:
        content = get_latest_file(token, rel)
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(content)
        os.replace(tmp, path)


def ensure_openclip():
    try:
        v = importlib.metadata.version("open_clip_torch")
    except importlib.metadata.PackageNotFoundError:
        v = None
    if v != OPENCLIP_VERSION:
        print(f"Installing open_clip_torch=={OPENCLIP_VERSION} ...")
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--disable-pip-version-check",
                f"open_clip_torch=={OPENCLIP_VERSION}",
            ],
            timeout=600,
        )
    assert importlib.metadata.version("open_clip_torch") == OPENCLIP_VERSION
    print(f"✓ OpenCLIP                 : {OPENCLIP_VERSION}")


def release_summary(token):
    r = requests.get(
        f"{REPO_API}/releases?per_page=100",
        headers=github_headers(token),
        timeout=30,
    )
    r.raise_for_status()
    rel = r.json()
    interesting = []
    for x in rel:
        tag = str(x.get("tag_name", ""))
        if tag.startswith("evict-nb06-"):
            interesting.append((tag, x.get("body", "")))
    return interesting


def main():
    print("=" * 112)
    print("EVICT NOTEBOOK 06 — RESUME AFTER PC / BROWSER SHUTDOWN")
    print("=" * 112)

    from kaggle_secrets import UserSecretsClient
    import torch

    assert torch.cuda.is_available(), "Enable a Kaggle GPU before resuming."
    print("GPU                       :", torch.cuda.get_device_name(0))

    token = UserSecretsClient().get_secret(SECRET_NAME)
    assert token, f"Kaggle Secret {SECRET_NAME!r} is unavailable."
    print("GitHub secret             : PASS")

    # If the cloud kernel survived the PC/browser shutdown, never start a
    # duplicate training process.
    alive = active_nb06_processes()
    if alive:
        print()
        print("=" * 112)
        print("AN EXISTING NOTEBOOK-06 PROCESS IS STILL RUNNING")
        print("=" * 112)
        for line in alive:
            print(line)
        print()
        print(
            "Do not start another copy. The Kaggle cloud session survived the PC/browser shutdown. "
            "Leave the current process running and inspect its notebook output."
        )
        return

    if not ROOT.exists():
        print("Local repository          : MISSING -> cloning latest main")
        clone_repo(token)
    else:
        assert (ROOT / ".git").exists(), f"{ROOT} exists but is not the EviCT Git repository."
        print("Local repository          : PRESENT")

    # Preserve all local artifacts/checkpoints. Do not fetch, pull, reset or
    # clean. Only replace the three execution-control source files from main.
    overwrite_execution_files(token)

    run(["git", "config", "user.name", "itsCodeBakery"], cwd=ROOT, timeout=15)
    run(
        ["git", "config", "user.email", "itsCodeBakery@users.noreply.github.com"],
        cwd=ROOT,
        timeout=15,
    )
    print("Git identity              : PASS")

    for rel in LATEST_REQUIRED_FILES:
        run([sys.executable, "-m", "py_compile", str(ROOT / rel)], cwd=ROOT, timeout=60)
    print("Execution-code syntax     : PASS")

    ensure_openclip()

    print()
    print("=" * 112)
    print("REMOTE DURABILITY DISCOVERY")
    print("=" * 112)
    try:
        items = release_summary(token)
        for tag, body in sorted(items):
            if (
                "real_text-seed17-final" in tag
                or "real_text-seed42-final" in tag
                or "real_text-seed2026-final" in tag
                or "swapped_text-seed17-rolling" in tag
            ):
                print(tag)
                if body:
                    print("  " + body)
    except Exception as exc:
        print("⚠ Could not print release summary:", exc)
        print("  Training can still restore releases inside the runner if GitHub is reachable.")

    print()
    print("=" * 112)
    print("SAFE RESUME POLICY")
    print("=" * 112)
    print(
        """
1. Existing local recovery checkpoints are kept and take priority.
2. Completed Notebook-06 units are restored from FINAL GitHub Releases and revalidated; they are NOT retrained.
3. If no local checkpoint exists for an unfinished unit, its latest rolling GitHub Release is restored.
4. Every remaining variant/seed runs in a fresh Python process.
5. Metadata-only non-fast-forward pushes no longer stop GPU training.
6. Calibration remains untouched.
7. MedSeg/target remains untouched.
"""
    )

    orch = ROOT / "scripts" / "nb06_isolated_orchestrator.py"
    assert orch.exists()

    print("=" * 112)
    print("STARTING / RESUMING NOTEBOOK 06")
    print("=" * 112)

    result = subprocess.run(
        [sys.executable, "-u", str(orch)],
        cwd=str(ROOT),
        text=True,
    )

    if result.returncode != 0:
        print()
        print("=" * 112)
        print("NOTEBOOK 06 STOPPED SAFELY")
        print("=" * 112)
        raise RuntimeError(
            "Do not restart the Kaggle session. Send the final 40-60 output lines. "
            "The latest local or GitHub Release recovery remains the restart point."
        )


if __name__ == "__main__":
    main()
