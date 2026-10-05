# EviCT

Uncertainty-aware vision-language segmentation and evidence-grounded lung CT reporting.

## Kaggle execution

Long-running EviCT jobs use resumable checkpointing and GitHub synchronization. See [docs/KAGGLE_RECOVERY.md](docs/KAGGLE_RECOVERY.md).

Core recovery rules:

- atomic local checkpoints;
- model, EMA teacher, optimizer, scheduler, AMP scaler, RNG and sampler state restoration;
- configuration/split/manifest hash verification before resume;
- GitHub synchronization for code, logs and run state;
- rolling GitHub Release assets for large checkpoints rather than ordinary Git;
- safe continuation from the latest verified step after a Kaggle interruption.

The current scientific stage remains governed by `STATE.md` and the frozen protocol files under `config/`.
