from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/kaggle/working/EViCT")
RUNNER = ROOT / "scripts/nb06_semantic_branch.py"
AUD = ROOT / "artifacts/audit"
LARGE = ROOT / "artifacts/large/notebook06"
LOG_ROOT = Path("/kaggle/working/evict_nb06_isolated_logs")
LOG_ROOT.mkdir(parents=True, exist_ok=True)

ORDER = [
    ("real_text", 17),
    ("real_text", 42),
    ("real_text", 2026),
    ("swapped_text", 17),
    ("swapped_text", 42),
    ("swapped_text", 2026),
    ("random_prototypes", 17),
    ("random_prototypes", 42),
    ("random_prototypes", 2026),
]

HEARTBEAT_SECONDS = 30
STALL_SECONDS = 360


def final_path(variant: str, seed: int) -> Path:
    return AUD / f"notebook06_{variant}_seed{seed}_final_durable.json"


def recovery_path(variant: str, seed: int) -> Path:
    return LARGE / variant / f"seed_{seed}" / "recovery.pt"


def read_final(variant: str, seed: int):
    p = final_path(variant, seed)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if obj.get("status") == "DURABLE_COMPLETE":
        return obj
    return None


def read_checkpoint_step(path: Path):
    if not path.exists():
        return None
    try:
        q = torch.load(path, map_location="cpu", weights_only=False)
        step = int(q.get("global_step", -1))
        del q
        return step
    except Exception:
        return None


def gpu_status():
    r = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return " | ".join(x.strip() for x in r.stdout.splitlines() if x.strip())


def log_tail(path: Path, max_chars: int = 3500):
    if not path.exists():
        return ""
    with path.open("rb") as f:
        try:
            f.seek(-max_chars, os.SEEK_END)
        except OSError:
            f.seek(0)
        text = f.read().decode("utf-8", errors="replace")
    text = text.replace("\r", "\n")
    lines = [x for x in text.splitlines() if x.strip()]
    return "\n".join(lines[-12:])


def run_unit(variant: str, seed: int):
    done = read_final(variant, seed)
    if done is not None:
        print(
            f"✓ SKIP COMPLETE {variant} seed{seed}: "
            f"Dice={float(done['macro_case_dice']):.8f} "
            f"best@{int(done['best_step'])}",
            flush=True,
        )
        return

    rec = recovery_path(variant, seed)
    initial_step = read_checkpoint_step(rec)
    print("\n" + "=" * 112, flush=True)
    print(f"START ISOLATED UNIT — {variant} seed{seed}", flush=True)
    print(f"Existing recovery step    : {initial_step}", flush=True)
    print("=" * 112, flush=True)

    log = LOG_ROOT / f"{variant}_seed{seed}.log"
    with log.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen(
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
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

    last_step = initial_step
    last_mtime = rec.stat().st_mtime if rec.exists() else None
    last_advance = time.monotonic()
    last_heartbeat = 0.0

    while True:
        code = proc.poll()
        now = time.monotonic()

        current_mtime = rec.stat().st_mtime if rec.exists() else None
        if current_mtime is not None and current_mtime != last_mtime:
            current_step = read_checkpoint_step(rec)
            last_mtime = current_mtime
            if current_step is not None and current_step != last_step:
                print(
                    f">>> REAL CHECKPOINT ADVANCE — {variant} seed{seed}: "
                    f"{last_step} -> {current_step}",
                    flush=True,
                )
                last_step = current_step
                last_advance = now

        if now - last_heartbeat >= HEARTBEAT_SECONDS:
            print(
                f"[heartbeat] {variant} seed{seed} | "
                f"pid={proc.pid} | return={code} | recovery={last_step} | "
                f"idle={int(now-last_advance)}s | GPU {gpu_status()}",
                flush=True,
            )
            last_heartbeat = now

        if code is not None:
            tail = log_tail(log)
            if tail:
                print("\n--- child log tail ---", flush=True)
                print(tail, flush=True)
                print("--- end child log tail ---\n", flush=True)

            if code != 0:
                raise RuntimeError(
                    f"Notebook06 isolated child failed: {variant} seed{seed}, "
                    f"return code {code}. Log: {log}"
                )

            done = read_final(variant, seed)
            if done is None:
                raise RuntimeError(
                    f"{variant} seed{seed} child exited 0 but durable final audit is missing."
                )

            print(
                f"✓ ISOLATED COMPLETE {variant} seed{seed}: "
                f"Dice={float(done['macro_case_dice']):.8f}, "
                f"best={int(done['best_step'])}, final={int(done['final_step'])}",
                flush=True,
            )
            return

        if now - last_advance > STALL_SECONDS:
            print("\n" + "=" * 112, flush=True)
            print(f"WATCHDOG STALL — {variant} seed{seed}", flush=True)
            print(f"No recovery-step advance for > {STALL_SECONDS} seconds.", flush=True)
            print(f"Last recovery step: {last_step}", flush=True)
            print("GPU:", gpu_status(), flush=True)
            print("\n--- child log tail ---", flush=True)
            print(log_tail(log), flush=True)
            print("--- end child log tail ---", flush=True)

            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

            raise RuntimeError(
                f"Watchdog stopped stalled child {variant} seed{seed}. "
                f"Latest local recovery remains at step {last_step}."
            )

        time.sleep(5)


def main():
    print("=" * 112, flush=True)
    print("EVICT NOTEBOOK 06 — PROCESS-ISOLATED ORCHESTRATOR", flush=True)
    print("=" * 112, flush=True)
    print("Each variant/seed runs in a fresh Python process.", flush=True)
    print("Calibration accessed     : NO", flush=True)
    print("Target / MedSeg accessed : NO", flush=True)
    print(f"Stall watchdog           : {STALL_SECONDS}s without recovery-step advance", flush=True)

    for variant, seed in ORDER:
        run_unit(variant, seed)

    print("\n" + "=" * 112, flush=True)
    print("ALL NINE UNITS DURABLE — AGGREGATING", flush=True)
    print("=" * 112, flush=True)

    result = subprocess.run(
        [sys.executable, "-u", str(RUNNER), "--aggregate-only"],
        cwd=str(ROOT),
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Notebook06 aggregation failed.")

    final = AUD / "notebook06_semantic_ablation.json"
    if not final.exists():
        raise RuntimeError("Final Notebook06 semantic ablation audit missing after aggregation.")

    audit = json.loads(final.read_text(encoding="utf-8"))
    if audit.get("status") != "SEMANTIC_BRANCH_FROZEN":
        raise RuntimeError("Final Notebook06 audit is not frozen.")

    print("\n" + "=" * 112, flush=True)
    print("NOTEBOOK 06 — DURABLE COMPLETE", flush=True)
    print("=" * 112, flush=True)
    for row in audit["variant_summaries"]:
        print(
            f"{row['variant']:<20} "
            f"{float(row['macro_case_dice_mean']):.8f} ± "
            f"{float(row['macro_case_dice_sample_sd']):.8f}",
            flush=True,
        )
    print(
        "Semantic-benefit rule   :",
        "SUPPORTED" if audit["semantic_benefit_supported_by_predeclared_rule"] else "NOT SUPPORTED",
        flush=True,
    )
    print("Calibration accessed     : NO", flush=True)
    print("Target / MedSeg accessed : NO", flush=True)
    print("NEXT                     : Notebook 07", flush=True)


if __name__ == "__main__":
    main()
