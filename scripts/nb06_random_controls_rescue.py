from __future__ import annotations

import base64
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
import torch

ROOT = Path("/kaggle/working/EviCT")
REPO_API = "https://api.github.com/repos/itsCodeBakery/EviCT"
SECRET_NAME = "pushEviCT"
RUNNER = ROOT / "scripts" / "nb06_semantic_branch.py"
LOG_DIR = Path("/kaggle/working/nb06_random_rescue_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEEDS = [17, 42, 2026]
STALL_SECONDS = 480
MAX_RELAUNCHES = 2

LATEST_FILES = [
    "scripts/nb05b_unet_lr_pilot_segment1.py",
    "scripts/nb06_semantic_branch.py",
    "scripts/git_sync.py",
]


def headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }


def fetch_latest_file(token, rel):
    r = requests.get(
        f"{REPO_API}/contents/{rel}?ref=main",
        headers=headers(token),
        timeout=30,
    )
    r.raise_for_status()
    return base64.b64decode(r.json()["content"])


def refresh_runtime_files(token):
    for rel in LATEST_FILES:
        data = fetch_latest_file(token, rel)
        p = ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(p) + ".rescue_tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)
        print("✓ refreshed runtime file   :", rel)


def final_path(seed):
    return ROOT / "artifacts/audit" / f"notebook06_random_prototypes_seed{seed}_final_durable.json"


def final_obj(seed):
    p = final_path(seed)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if obj.get("status") == "DURABLE_COMPLETE" else None


def recovery_path(seed):
    return (
        ROOT
        / "artifacts/large/notebook06/random_prototypes"
        / f"seed_{seed}/recovery.pt"
    )


def checkpoint_status(seed):
    p = recovery_path(seed)
    if not p.exists():
        return "none"
    try:
        q = torch.load(p, map_location="cpu", weights_only=False)
        step = int(q.get("global_step", -1))
        best = float(q.get("best_score", -1))
        pat = int(q.get("patience_count", -1))
        del q
        return f"step={step}, best={best:.6f}, pat={pat}/8"
    except Exception as exc:
        return f"unreadable: {exc}"


def parse_seed(args):
    m = re.search(r"--variant\s+random_prototypes\s+--seed\s+(17|42|2026)", args)
    return int(m.group(1)) if m else None


def proc_cuda_visible(pid):
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
        env = {}
        for part in raw.split(b"\0"):
            if b"=" in part:
                k, v = part.split(b"=", 1)
                env[k.decode(errors="ignore")] = v.decode(errors="ignore")
        val = env.get("CUDA_VISIBLE_DEVICES")
        if val is not None and val.strip() in {"0", "1"}:
            return int(val.strip())
    except Exception:
        pass
    return None


def find_active_children():
    ps = subprocess.check_output(
        ["ps", "-eo", "pid,ppid,etime,args"],
        text=True,
    )
    out = {}
    for line in ps.splitlines()[1:]:
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
        except Exception:
            continue
        seed = parse_seed(parts[3])
        if seed is None:
            continue
        out[seed] = {
            "pid": pid,
            "gpu": proc_cuda_visible(pid),
            "args": parts[3],
            "external": True,
        }
    return out


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def log_tail(path, chars=4000):
    if not path or not Path(path).exists():
        return ""
    p = Path(path)
    with p.open("rb") as f:
        try:
            f.seek(-chars, os.SEEK_END)
        except OSError:
            f.seek(0)
        txt = f.read().decode("utf-8", errors="replace").replace("\r", "\n")
    lines = [x for x in txt.splitlines() if x.strip()]
    return "\n".join(lines[-20:])


def launch(seed, gpu, attempt):
    log = LOG_DIR / f"random_seed{seed}_attempt{attempt}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["EVICT_NB06_PARALLEL_SAFE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    handle = log.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(RUNNER),
            "--variant",
            "random_prototypes",
            "--seed",
            str(seed),
        ],
        cwd=str(ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print(
        f"▶ GPU{gpu} START random seed{seed} attempt {attempt} | "
        f"{checkpoint_status(seed)}",
        flush=True,
    )

    return {
        "pid": proc.pid,
        "gpu": gpu,
        "proc": proc,
        "handle": handle,
        "log": log,
        "external": False,
        "last_cp_mtime": recovery_path(seed).stat().st_mtime if recovery_path(seed).exists() else None,
        "last_activity": time.monotonic(),
        "last_report": 0.0,
    }


def external_log(seed):
    candidates = sorted(
        Path("/kaggle/working/nb06_turbo_logs").glob(
            f"random_prototypes_seed{seed}_attempt*.log"
        )
    )
    return candidates[-1] if candidates else None


def finish_metadata_restore_if_needed():
    # The current runtime should already hold the six completed real/swapped
    # final JSONs. If one is absent, restore it from the final GitHub Release
    # after all GPU work is done, so it cannot steal GPU time from the controls.
    completed = [
        ("real_text", 17),
        ("real_text", 42),
        ("real_text", 2026),
        ("swapped_text", 17),
        ("swapped_text", 42),
        ("swapped_text", 2026),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["EVICT_NB06_PARALLEL_SAFE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    for variant, seed in completed:
        p = ROOT / "artifacts/audit" / f"notebook06_{variant}_seed{seed}_final_durable.json"
        good = False
        if p.exists():
            try:
                good = json.loads(p.read_text(encoding="utf-8")).get("status") == "DURABLE_COMPLETE"
            except Exception:
                pass
        if good:
            continue
        print(f"Restoring missing metadata: {variant} seed{seed}")
        r = subprocess.run(
            [
                sys.executable,
                "-u",
                str(RUNNER),
                "--variant",
                variant,
                "--seed",
                str(seed),
            ],
            cwd=str(ROOT),
            env=env,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"Could not restore metadata for {variant} seed{seed}")


def publish_file(token, rel):
    p = ROOT / rel
    if not p.exists():
        return
    url = f"{REPO_API}/contents/{rel}"
    remote = requests.get(url + "?ref=main", headers=headers(token), timeout=30)
    payload = {
        "message": f"Freeze Notebook 06 result: {rel}",
        "content": base64.b64encode(p.read_bytes()).decode(),
        "branch": "main",
    }
    if remote.status_code == 200:
        payload["sha"] = remote.json()["sha"]
    elif remote.status_code != 404:
        remote.raise_for_status()
    r = requests.put(url, headers=headers(token), json=payload, timeout=60)
    r.raise_for_status()
    print("✓ published                :", rel)


def aggregate(token):
    finish_metadata_restore_if_needed()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["EVICT_NB06_PARALLEL_SAFE"] = "0"
    env["PYTHONUNBUFFERED"] = "1"

    print()
    print("=" * 112)
    print("FINAL NOTEBOOK-06 AGGREGATION")
    print("=" * 112)

    r = subprocess.run(
        [sys.executable, "-u", str(RUNNER), "--aggregate-only"],
        cwd=str(ROOT),
        env=env,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError("Notebook06 aggregation failed.")

    final = ROOT / "artifacts/audit/notebook06_semantic_ablation.json"
    audit = json.loads(final.read_text(encoding="utf-8"))
    assert audit["status"] == "SEMANTIC_BRANCH_FROZEN"
    assert audit["calibration_accessed"] is False
    assert audit["target_accessed"] is False

    for rel in [
        "STATE.md",
        "artifacts/audit/notebook06_semantic_ablation.json",
        "tables/notebook06_semantic_ablation.csv",
        "reports/notebook06_semantic_branch_report.md",
    ]:
        try:
            publish_file(token, rel)
        except Exception as exc:
            print(f"⚠ metadata publish failed for {rel}: {exc}")

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


def main():
    from kaggle_secrets import UserSecretsClient

    token = UserSecretsClient().get_secret(SECRET_NAME)
    assert token

    print("=" * 112)
    print("EVICT NB06 — LAST THREE RANDOM CONTROLS RESCUE")
    print("=" * 112)
    print("CUDA devices              :", torch.cuda.device_count())
    assert torch.cuda.device_count() >= 2, "Two T4s are required for the fast rescue."

    refresh_runtime_files(token)

    for rel in LATEST_FILES:
        r = subprocess.run(
            [sys.executable, "-m", "py_compile", str(ROOT / rel)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if r.returncode != 0:
            print(r.stdout)
            raise RuntimeError(f"Syntax error: {rel}")

    active = find_active_children()
    print("Existing random children  :", {s:(x['pid'],x['gpu']) for s,x in active.items()})

    # The diagnostic showed seed42 is already alive on physical GPU1. Preserve
    # it. Any future orphan is treated the same way.
    workers = {}
    used_gpus = set()

    for seed, info in active.items():
        if final_obj(seed) is not None:
            continue
        gpu = info["gpu"]
        if gpu is None:
            # If environment inspection is unavailable, infer only when one GPU
            # is clearly occupied. The current known orphan is seed42 on GPU1.
            gpu = 1 if seed == 42 else 0
        info["gpu"] = gpu
        info["log"] = external_log(seed)
        info["last_cp_mtime"] = recovery_path(seed).stat().st_mtime if recovery_path(seed).exists() else None
        info["last_activity"] = time.monotonic()
        info["last_report"] = 0.0
        workers[seed] = info
        used_gpus.add(gpu)
        print(
            f"✓ KEEP RUNNING GPU{gpu} random seed{seed} PID={info['pid']} | "
            f"{checkpoint_status(seed)}"
        )

    attempts = {s:0 for s in RANDOM_SEEDS}

    pending = [
        s for s in RANDOM_SEEDS
        if final_obj(s) is None and s not in workers
    ]

    while pending or workers:
        # Fill every free GPU.
        for gpu in [0, 1]:
            if gpu in {w["gpu"] for w in workers.values()}:
                continue
            if not pending:
                continue
            seed = pending.pop(0)
            if final_obj(seed) is not None:
                continue
            attempts[seed] += 1
            workers[seed] = launch(seed, gpu, attempts[seed])

        time.sleep(5)
        now = time.monotonic()

        for seed in list(workers):
            w = workers[seed]
            cp = recovery_path(seed)

            if cp.exists():
                mt = cp.stat().st_mtime
                if mt != w["last_cp_mtime"]:
                    w["last_cp_mtime"] = mt
                    w["last_activity"] = now

            if now - w["last_report"] >= 30:
                print(
                    f"[GPU{w['gpu']}] random seed{seed} | {checkpoint_status(seed)} | "
                    f"silent={int(now-w['last_activity'])}s",
                    flush=True,
                )
                tail = log_tail(w.get("log"), 1400)
                if tail:
                    print(tail, flush=True)
                w["last_report"] = now

            alive = pid_alive(w["pid"])
            obj = final_obj(seed)

            if obj is not None:
                if alive:
                    # The final JSON is written shortly before process cleanup.
                    # Keep the GPU reserved until the child truly exits so a
                    # successor cannot overlap and cause an avoidable OOM.
                    continue
                if not w["external"]:
                    try:
                        w["handle"].close()
                    except Exception:
                        pass
                print(
                    f"✓ COMPLETE random seed{seed}: "
                    f"Dice={float(obj['macro_case_dice']):.8f}, "
                    f"best={int(obj['best_step'])}, final={int(obj['final_step'])}",
                    flush=True,
                )
                del workers[seed]
                continue

            if not alive:
                if not w["external"]:
                    try:
                        w["handle"].close()
                    except Exception:
                        pass
                if obj is not None:
                    print(
                        f"✓ COMPLETE random seed{seed}: "
                        f"Dice={float(obj['macro_case_dice']):.8f}"
                    )
                    del workers[seed]
                    continue

                attempts[seed] += 1
                if attempts[seed] > MAX_RELAUNCHES:
                    print(log_tail(w.get("log"), 5000))
                    raise RuntimeError(f"random seed{seed} failed too many times.")
                print(
                    f"⚠ random seed{seed} exited before completion; "
                    f"relaunching from {checkpoint_status(seed)}"
                )
                gpu = w["gpu"]
                del workers[seed]
                workers[seed] = launch(seed, gpu, attempts[seed])
                continue

            if now - w["last_activity"] > STALL_SECONDS:
                print(
                    f"⚠ WATCHDOG random seed{seed} stalled on GPU{w['gpu']} | "
                    f"{checkpoint_status(seed)}"
                )
                try:
                    os.kill(w["pid"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
                time.sleep(5)
                if pid_alive(w["pid"]):
                    try:
                        os.kill(w["pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass

                if not w["external"]:
                    try:
                        w["handle"].close()
                    except Exception:
                        pass

                attempts[seed] += 1
                if attempts[seed] > MAX_RELAUNCHES:
                    raise RuntimeError(f"random seed{seed} stalled too many times.")

                gpu = w["gpu"]
                del workers[seed]
                workers[seed] = launch(seed, gpu, attempts[seed])

    aggregate(token)


if __name__ == "__main__":
    main()
