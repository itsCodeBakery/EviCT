# EViCT-Dx Kaggle Recovery and GitHub Synchronization

**Isolation rule:** all new diagnostic-extension code, state, manifests, and run metadata remain under `vlmDiagnosis/`. Frozen EViCT-Core files are read-only.

Long-running EViCT-Dx jobs must be resumable. Kaggle is the compute environment; GitHub is the control plane. Ordinary Git stores code, configurations, run state, hashes, logs, metrics, and manifests. Large model checkpoints are stored as rolling GitHub Release assets and are never committed to ordinary Git.

## Security

- Store the GitHub PAT only as the Kaggle Secret `pushEviCT`.
- Never paste the PAT into a notebook cell, remote URL, log, STATE file, or commit.
- `scripts/git_sync.py` authenticates with a temporary HTTP header and restores the secret-free repository URL.

## Required training integration

Add the repository source directory to Python and create one recovery manager per run:

    from pathlib import Path
    import json, sys

    ROOT = Path('/kaggle/working/EviCT')
    sys.path.insert(0, str(ROOT / 'vlmDiagnosis'))

    from runtime.recovery import RecoveryPolicy, RunRecoveryManager

    policy_cfg = json.loads((ROOT / 'vlmDiagnosis/config/recovery_policy.json').read_text())
    allowed = set(RecoveryPolicy.__dataclass_fields__)
    policy = RecoveryPolicy(**{k: v for k, v in policy_cfg.items() if k in allowed})

    RUN_ID = 'DX_covid_ct_md_diagnosis_seed1705_v1'
    recovery = RunRecoveryManager(ROOT, RUN_ID, policy=policy, run_dir=ROOT / 'vlmDiagnosis/runs' / RUN_ID)

Before training, build the model, optimizer, scheduler, scaler and optional EMA teacher. If `vlmDiagnosis/runs/<RUN_ID>/STATE.json` exists, call `recovery.restore(...)` and continue from `global_step + 1`. Pass the expected configuration, split, and manifest hashes. A mismatch aborts the resume rather than silently changing the experiment.

Example:

    restored = recovery.restore(
        student=model,
        ema_teacher=ema_model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_config_hash=CONFIG_HASH,
        expected_split_hash=SPLIT_HASH,
        expected_manifest_hash=MANIFEST_HASH,
    )
    start_step = restored['global_step'] + 1

If no state exists, start at step 1 and write an initial state record.

## Checkpoint loop

Every optimizer step, keep the scientific step counter explicit. At the configured interval (default 250 steps), write an atomic checkpoint containing the student, EMA teacher, optimizer, scheduler, AMP scaler, RNG states, sampler state, counters, and scientific hashes.

    if recovery.should_checkpoint(global_step):
        remote = recovery.should_remote_checkpoint(global_step)
        recovery.save_training_checkpoint(
            student=model,
            ema_teacher=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            global_step=global_step,
            epoch=epoch,
            best_score=best_score,
            patience_counter=patience_counter,
            sampler_state=sampler_state,
            config_hash=CONFIG_HASH,
            split_hash=SPLIT_HASH,
            manifest_hash=MANIFEST_HASH,
            remote=remote,
        )

        if recovery.should_sync_metadata(global_step):
            recovery.sync_metadata(f'{RUN_ID}: checkpoint step {global_step}')

The local file is written to a temporary path, loaded back for structural verification, and only then replaces `last.pt`. Previous generations are retained. A step-versioned remote checkpoint is uploaded at the configured remote interval (default 500 steps) to a prerelease named `evict-dx-recovery-<RUN_ID>-rolling`. The newest two checkpoint assets are retained, so a failed upload cannot destroy the previous durable recovery point.

## Graceful interruption

For SIGTERM/SIGINT, register a callback that immediately writes a remote checkpoint using the current training objects. This helps on graceful termination, although a hard GPU/runtime kill may not deliver a signal.

    def emergency(reason):
        recovery.save_training_checkpoint(
            student=model, ema_teacher=ema_model, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, global_step=global_step,
            config_hash=CONFIG_HASH, split_hash=SPLIT_HASH,
            manifest_hash=MANIFEST_HASH, remote=True,
            status=f'interrupted:{reason}',
        )
        recovery.sync_metadata(f'{RUN_ID}: emergency checkpoint at {global_step}')
        raise SystemExit(128)

    recovery.register_signal_checkpoint(emergency)

## New Kaggle session

Clone or attach the repository, then inspect the run:

    !python /kaggle/working/EviCT/vlmDiagnosis/scripts/kaggle_recovery.py status --run-id DX_covid_ct_md_diagnosis_seed1705_v1

If the local checkpoint is absent but `STATE.json` records a rolling asset:

    !python /kaggle/working/EviCT/vlmDiagnosis/scripts/kaggle_recovery.py pull-remote --run-id DX_covid_ct_md_diagnosis_seed1705_v1
    !python /kaggle/working/EviCT/vlmDiagnosis/scripts/kaggle_recovery.py verify --run-id DX_covid_ct_md_diagnosis_seed1705_v1

Then rerun the model-construction cell and call `recovery.restore(...)`. The checkpoint restores optimizer/scheduler/scaler/EMA/RNG state; loading weights alone is not considered a valid resume.

## Long preprocessing or inference jobs

For jobs without an optimizer, use `STATE.json` as an atomic progress ledger. Process deterministic units (case IDs, shards, or volumes), write each output atomically, append the completed IDs, hash the output, and sync metadata periodically. On restart, skip only units whose output exists and whose stored hash verifies. Never mark a case complete before its output is durable.

## What must remain immutable across a resume

- run ID
- source/target role
- patient/group split
- label budget
- model configuration
- preprocessing contract
- configuration hash
- split hash
- manifest hash
- target-evaluation gate/frozen protocol

If any of these change, create a new run ID instead of resuming the old run.

## Useful commands

    python vlmDiagnosis/scripts/kaggle_recovery.py status --run-id <RUN_ID>
    python vlmDiagnosis/scripts/kaggle_recovery.py verify --run-id <RUN_ID>
    python vlmDiagnosis/scripts/kaggle_recovery.py pull-remote --run-id <RUN_ID>
    python vlmDiagnosis/scripts/kaggle_recovery.py sync --run-id <RUN_ID>

## Recovery objective

A Kaggle interruption should cost at most the interval since the most recent durable rolling checkpoint. The restart must be traceable through the run ID, checkpoint SHA-256, configuration/split/manifest hashes, and Git commit.
