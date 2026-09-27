# Notebook 04 — Frozen Supervised Source Baseline

Model: supervised SegFormer MiT-B1.

Three-seed source-selection macro case Dice: 0.754051 ± 0.012365 (sample SD).

| Seed | Best step | Macro case Dice | Macro case IoU | Macro slice Dice | Pooled Dice |
|---:|---:|---:|---:|---:|---:|
| 17 | 3500 | 0.752581 | 0.611749 | 0.759814 | 0.817431 |
| 42 | 2000 | 0.767085 | 0.628550 | 0.757410 | 0.827764 |
| 2026 | 4250 | 0.742487 | 0.600968 | 0.758276 | 0.824285 |

Selection threshold: 0.5 fixed; no threshold search in Notebook 04.
Calibration accessed: NO.
Target / MedSeg accessed: NO.
Validation overlay slices are selected deterministically as the largest valid ground-truth lesion slice in each source-selection case.