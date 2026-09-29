from __future__ import annotations

import base64
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import requests
import torch

ROOT = Path("/kaggle/working/EviCT")
REPO_API = "https://api.github.com/repos/itsCodeBakery/EviCT"
REPO_URL = "https://github.com/itsCodeBakery/EviCT.git"
SECRET_NAME = "pushEviCT"
OPENCLIP_VERSION = "3.3.0"
STALL_SECONDS = 480
MAX_RETRIES = 2

EXECUTION_FILES = [
    "scripts/nb06_semantic_branch.py",
    "scripts/git_sync.py",
    "scripts/nb06_three_hour_turbo.py",
]

REAL_TEXT = [("real_text", 17), ("real_text", 42), ("real_text", 2026)]
CONTROL_TASKS = [
    ("swapped_text", 17),
    ("swapped_text", 42),
    ("swapped_text", 2026),
    ("random_prototypes", 17),
    ("random_prototypes", 42),
    ("random_prototypes", 2026),
]

LOG_DIR = Path("/kaggle/working/nb06_turbo_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }


def run(cmd, *, cwd=None, timeout=180, capture=False, env=None, check=True):
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


def clone_if_missing(token):
    if ROOT.exists():
        if not (ROOT / ".git").exists():
            raise RuntimeError(f"{ROOT} exists but is not a Git repository.")
        return
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


def fetch_latest_file(token, rel):
    url = f"{REPO_API}/contents/{rel}?ref=main"
    last = None
    for attempt in range(1, 4):
        try:
            r = requests.get(url, headers=headers(token), timeout=30)
            r.raise_for_status()
            return base64.b64decode(r.json()["content"])
        except Exception as exc:
            last = exc
            print(f"⚠ fetch {rel}, attempt {attempt}/3: {exc}")
            if attempt < 3:
                time.sleep(5 * attempt)
    raise RuntimeError(f"Unable to fetch latest {rel}: {last}")


def refresh_execution_files(token):
    for rel in EXECUTION_FILES:
        data = fetch_latest_file(token, rel)
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".api_tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        print("✓ execution file refreshed :", rel)


def kill_old_nb06_processes():
    ps = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    pids = []
    for line in ps.splitlines()[1:]:
        low = line.lower()
        if (
            "nb06_semantic_branch.py" in low
            or "nb06_isolated_orchestrator.py" in low
            or "nb06_resume_after_shutdown.py" in low
        ):
            parts = line.strip().split(None, 1)
            if not parts:
                continue
            try:
                pid = int(parts[0])
            except Exception:
                continue
            if pid not in {os.getpid(), os.getppid()}:
                pids.append(pid)
    if not pids:
        print("Old Notebook06 process    : NONE")
        return
    print("Old Notebook06 processes  :", pids)
    print("Stopping them so the dual-GPU scheduler can resume from their last checkpoint...")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(4)
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    print("Old Notebook06 process    : CLEARED")


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
    print("OpenCLIP                  :", OPENCLIP_VERSION, "VERIFIED")


def final_path(variant, seed):
    return ROOT / "artifacts/audit" / f"notebook06_{variant}_seed{seed}_final_durable.json"


def final_ok(variant, seed):
    p = final_path(variant, seed)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if obj.get("status") == "DURABLE_COMPLETE" else None


def release_summary(token):
    r = requests.get(f"{REPO_API}/releases?per_page=100", headers=headers(token), timeout=30)
    r.raise_for_status()
    out = {}
    for rel in r.json():
        tag = str(rel.get("tag_name", ""))
        if tag.startswith("evict-nb06-"):
            out[tag] = rel.get("body", "")
    return out


def warm_and_restore_real_text():
    runner = ROOT / "scripts/nb06_semantic_branch.py"
    env = os.environ.copy()
    env["EVICT_NB06_PARALLEL_SAFE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"

    print()
    print("=" * 112)
    print("PHASE 1 — PREPARE CACHE/MODEL AND RESTORE COMPLETED REAL-TEXT RUNS")
    print("=" * 112)

    for variant, seed in REAL_TEXT:
        print(f"Preparing/restoring {variant} seed{seed} ...")
        r = subprocess.run(
            [sys.executable, "-u", str(runner), "--variant", variant, "--seed", str(seed)],
            cwd=str(ROOT),
            env=env,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"Preparation/restore failed for {variant} seed{seed}.")
        obj = final_ok(variant, seed)
        if obj is None:
            raise RuntimeError(f"Durable final audit missing after restore: {variant} seed{seed}")
        print(
            f"✓ {variant} seed{seed}: Dice={float(obj['macro_case_dice']):.8f}, "
            f"best={int(obj['best_step'])}, final={int(obj['final_step'])}"
        )


def log_tail(path, chars=1600):
    if not path.exists():
        return ""
    with path.open("rb") as f:
        try:
            f.seek(-chars, os.SEEK_END)
        except OSError:
            f.seek(0)
        text = f.read().decode("utf-8", errors="replace").replace("\r", "\n")
    lines = [x for x in text.splitlines() if x.strip()]
    return "\n".join(lines[-8:])


def checkpoint_hint(variant, seed):
    # Heartbeats must be cheap: do not deserialize the ~158 MiB recovery
    # checkpoint every 30 seconds while two GPUs are training. The child log
    # already reports exact steps and checkpoint advances.
    p = ROOT / "artifacts/large/notebook06" / variant / f"seed_{seed}" / "recovery.pt"
    if not p.exists():
        return "none"
    size = p.stat().st_size / (1024 ** 2)
    age = max(0, int(time.time() - p.stat().st_mtime))
    return f"checkpoint={size:.1f} MiB, age={age}s"


def launch(task, gpu, attempt):
    variant, seed = task
    runner = ROOT / "scripts/nb06_semantic_branch.py"
    log = LOG_DIR / f"{variant}_seed{seed}_attempt{attempt}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["EVICT_NB06_PARALLEL_SAFE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    out = log.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(runner), "--variant", variant, "--seed", str(seed)],
        cwd=str(ROOT),
        env=env,
        stdout=out,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "task": task,
        "gpu": gpu,
        "attempt": attempt,
        "proc": proc,
        "handle": out,
        "log": log,
        "last_mtime": log.stat().st_mtime,
        "last_activity": time.monotonic(),
        "last_report": 0.0,
    }


def run_controls_dual_gpu():
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        raise RuntimeError(
            f"Only {gpu_count} GPU detected. Six remaining control runs are unlikely to finish "
            "inside the requested three-hour window on one GPU. Enable Kaggle's dual-T4 accelerator."
        )

    pending = deque([t for t in CONTROL_TASKS if final_ok(*t) is None])
    already = [t for t in CONTROL_TASKS if final_ok(*t) is not None]

    print()
    print("=" * 112)
    print("PHASE 2 — DUAL-GPU CONTROL ABLATIONS")
    print("=" * 112)
    print("Detected GPUs             :", gpu_count)
    print("Using physical GPUs       : 0 and 1")
    print("Already complete          :", already if already else "none")
    print("Pending                   :", list(pending))

    running = {}
    attempts = {t: 0 for t in CONTROL_TASKS}

    while pending or running:
        for gpu in [0, 1]:
            if gpu in running or not pending:
                continue
            task = pending.popleft()
            if final_ok(*task) is not None:
                print("✓ SKIP COMPLETE", task)
                continue
            attempts[task] += 1
            info = launch(task, gpu, attempts[task])
            running[gpu] = info
            print(
                f"▶ GPU{gpu} START {task[0]} seed{task[1]} attempt {attempts[task]} | "
                f"recovery {checkpoint_hint(*task)}",
                flush=True,
            )

        time.sleep(5)
        now = time.monotonic()

        for gpu in list(running):
            info = running[gpu]
            proc = info["proc"]
            log = info["log"]

            if log.exists():
                mt = log.stat().st_mtime
                if mt != info["last_mtime"]:
                    info["last_mtime"] = mt
                    info["last_activity"] = now

            code = proc.poll()

            if now - info["last_report"] >= 30:
                task = info["task"]
                print(
                    f"[GPU{gpu}] {task[0]} seed{task[1]} | "
                    f"{checkpoint_hint(*task)} | silent={int(now-info['last_activity'])}s",
                    flush=True,
                )
                tail = log_tail(log)
                if tail:
                    print(tail, flush=True)
                info["last_report"] = now

            if code is not None:
                info["handle"].close()
                task = info["task"]
                obj = final_ok(*task)

                if code == 0 and obj is not None:
                    print(
                        f"✓ GPU{gpu} COMPLETE {task[0]} seed{task[1]} | "
                        f"Dice={float(obj['macro_case_dice']):.8f} "
                        f"best={int(obj['best_step'])} final={int(obj['final_step'])}",
                        flush=True,
                    )
                    del running[gpu]
                    continue

                print(f"⚠ GPU{gpu} child ended with code={code}: {task}")
                print(log_tail(log, 5000))

                if attempts[task] < MAX_RETRIES:
                    print("Retrying from latest durable/local checkpoint...")
                    pending.appendleft(task)
                    del running[gpu]
                    continue

                raise RuntimeError(
                    f"{task[0]} seed{task[1]} failed after {MAX_RETRIES} attempts."
                )

            if now - info["last_activity"] > STALL_SECONDS:
                task = info["task"]
                print(f"⚠ WATCHDOG: GPU{gpu} stalled on {task}; terminating child.")
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
                info["handle"].close()
                print(log_tail(log, 5000))
                if attempts[task] < MAX_RETRIES:
                    pending.appendleft(task)
                    del running[gpu]
                else:
                    raise RuntimeError(
                        f"{task[0]} seed{task[1]} stalled twice; latest checkpoint is preserved."
                    )


def publish_file_via_api(token, rel):
    path = ROOT / rel
    if not path.exists():
        return
    url = f"{REPO_API}/contents/{rel}"
    remote = requests.get(url + "?ref=main", headers=headers(token), timeout=30)
    payload = {
        "message": f"Freeze Notebook 06 result: {rel}",
        "content": base64.b64encode(path.read_bytes()).decode(),
        "branch": "main",
    }
    if remote.status_code == 200:
        payload["sha"] = remote.json()["sha"]
    elif remote.status_code != 404:
        remote.raise_for_status()
    r = requests.put(url, headers=headers(token), json=payload, timeout=60)
    r.raise_for_status()
    print("✓ published final metadata :", rel)


def aggregate_and_publish(token):
    print()
    print("=" * 112)
    print("PHASE 3 — FINAL NOTEBOOK-06 AGGREGATION")
    print("=" * 112)

    runner = ROOT / "scripts/nb06_semantic_branch.py"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["EVICT_NB06_PARALLEL_SAFE"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"

    r = subprocess.run(
        [sys.executable, "-u", str(runner), "--aggregate-only"],
        cwd=str(ROOT),
        env=env,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError("Notebook06 aggregation failed.")

    final = ROOT / "artifacts/audit/notebook06_semantic_ablation.json"
    assert final.exists()
    audit = json.loads(final.read_text(encoding="utf-8"))
    assert audit["status"] == "SEMANTIC_BRANCH_FROZEN"
    assert audit["calibration_accessed"] is False
    assert audit["target_accessed"] is False

    # Mirror the key final metadata through the GitHub Contents API as a
    # branch-divergence-independent final safeguard.
    important = [
        "STATE.md",
        "artifacts/audit/notebook06_semantic_ablation.json",
        "tables/notebook06_semantic_ablation.csv",
        "reports/notebook06_semantic_branch_report.md",
    ]
    for variant, seed in REAL_TEXT + CONTROL_TASKS:
        important.append(f"artifacts/audit/notebook06_{variant}_seed{seed}_final_durable.json")
        important.append(f"artifacts/audit/notebook06_{variant}_seed{seed}_final_case_metrics.csv")

    for rel in important:
        try:
            publish_file_via_api(token, rel)
        except Exception as exc:
            print(f"⚠ final metadata API mirror failed for {rel}: {exc}")

    print()
    print("=" * 112)
    print("NOTEBOOK 06 — FULLY COMPLETE")
    print("=" * 112)
    print(
        "Vision-only reference     :",
        f"{float(audit['reference_vision_only']['macro_case_dice_mean']):.8f}",
    )
    for row in audit["variant_summaries"]:
        print(
            f"{row['variant']:<20} "
            f"{float(row['macro_case_dice_mean']):.8f} ± "
            f"{float(row['macro_case_dice_sample_sd']):.8f}"
        )
    print(
        "Semantic-benefit rule     :",
        "SUPPORTED" if audit["semantic_benefit_supported_by_predeclared_rule"] else "NOT SUPPORTED",
    )
    print("Calibration accessed      : NO")
    print("Target / MedSeg           : NO")
    print("NEXT                      : Notebook 07")


def main():
    started = time.time()
    print("=" * 112)
    print("EVICT NOTEBOOK 06 — THREE-HOUR DUAL-GPU FINISH MODE")
    print("=" * 112)
    print("Goal: finish the six remaining controls as quickly as possible without changing the scientific protocol.")

    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret(SECRET_NAME)
    assert token, f"Kaggle Secret {SECRET_NAME!r} is unavailable."

    print("CUDA devices              :", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  GPU{i}                   : {torch.cuda.get_device_name(i)}")

    clone_if_missing(token)
    refresh_execution_files(token)
    kill_old_nb06_processes()

    run(["git", "config", "user.name", "itsCodeBakery"], cwd=ROOT, timeout=15)
    run(
        ["git", "config", "user.email", "itsCodeBakery@users.noreply.github.com"],
        cwd=ROOT,
        timeout=15,
    )

    for rel in EXECUTION_FILES:
        run([sys.executable, "-m", "py_compile", str(ROOT / rel)], cwd=ROOT, timeout=60)
    print("Execution syntax          : PASS")

    ensure_openclip()

    try:
        rs = release_summary(token)
        for key in sorted(rs):
            if (
                "real_text" in key
                or "swapped_text-seed17-rolling" in key
            ):
                print(key)
                if rs[key]:
                    print("  " + rs[key])
    except Exception as exc:
        print("⚠ release summary unavailable:", exc)

    warm_and_restore_real_text()
    run_controls_dual_gpu()
    aggregate_and_publish(token)

    elapsed = (time.time() - started) / 60
    print(f"Total wall time           : {elapsed:.1f} minutes")


if __name__ == "__main__":
    main()
