# Notebook 08A — Source-Only Calibration and Protocol Pilot

Scope: b050 / seed17 only.

This is not the final target-unlock protocol.

Training lock GitHub commit: `8fc879f610c4deb44b291174562c3a67d163850f`

## Method summary

| Method | Threshold | Temperature | Calibrated threshold | ECE before | ECE after | Single-view Dice | Four-view Dice |
|---|---:|---:|---:|---:|---:|---:|---:|
| Supervised | 0.30 | 1.48014 | 0.36067 | 0.198% | 0.124% | 0.7349 | 0.7542 |
| Confidence-only EMA | 0.30 | 1.17200 | 0.32674 | 0.152% | 0.216% | 0.6821 | 0.6979 |
| Agreement-filtered EMA | 0.30 | 1.17162 | 0.32669 | 0.152% | 0.216% | 0.6823 | 0.6982 |

## Protocol status

- Source selection accessed: YES.
- Source calibration accessed: YES, for Notebook 08 only.
- Target / MedSeg accessed: NO.
- Target lock: ACTIVE.
- allow_target_evaluation: FALSE.

The remaining b050 and b025 confirmatory Notebook-07 runs are still required.
Their training configuration was frozen before calibration access and may not be changed based on these calibration results.
