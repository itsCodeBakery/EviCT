# Notebook 05 — Frozen Competitive Residual 2D U-Net Source Baseline

Learning rate: 0.0003, frozen from the seed-17 pilot before running seeds 42 and 2026.

Three-seed source-selection macro case Dice: 0.722207 ± 0.018059 (sample SD).

| Seed | Best step | Final step | Macro case Dice | Macro case IoU | Macro slice Dice | Pooled Dice |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 1500 | 3500 | 0.742388 | 0.598008 | 0.648168 | 0.804625 |
| 42 | 2000 | 4000 | 0.707572 | 0.559979 | 0.578396 | 0.799416 |
| 2026 | 2500 | 4500 | 0.716661 | 0.570439 | 0.680748 | 0.805893 |

Selection threshold: 0.5 fixed; no threshold search.
Calibration accessed: NO.
Target / MedSeg accessed: NO.
Target lock: ACTIVE.

Figures can be regenerated later from the frozen best checkpoints, raw logits, and metric tables.
