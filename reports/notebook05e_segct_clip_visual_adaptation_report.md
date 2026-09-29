# Notebook 05E — SegCT-CLIP Visual-Pathway Adaptation

**Status: explicit adaptation, not an exact SegCT-CLIP reproduction.**

The Notebook-05D reproduction audit found that the author repository, exact checkpoint, caption bank, caption-to-slice mapping, exact low/high layer indices, fusion operator, decoder topology and training hyperparameters were not sufficiently specified for an exact rerun.

This adaptation therefore uses the documented CLIP ViT-L/14-336 visual backbone and dual-level visual-feature idea, freezes the CLIP encoder, and trains an explicitly defined EviCT decoder on frozen source features. The unavailable caption bank, text encoder supervision, contrastive loss and caption-refinement module are not fabricated.

Three-seed source-selection macro-case Dice: **0.727279 ± 0.009356** (sample SD).

| Seed | Best step | Final step | Macro case Dice | Macro case IoU | Macro slice Dice | Pooled Dice |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 2000 | 4000 | 0.730943 | 0.582892 | 0.475763 | 0.766081 |
| 42 | 1000 | 3000 | 0.734250 | 0.585821 | 0.452855 | 0.763905 |
| 2026 | 1750 | 3750 | 0.716646 | 0.565321 | 0.641060 | 0.755931 |

Selection threshold: 0.5 fixed.
Calibration accessed: NO.
Target / MedSeg accessed: NO.
Historical SegCT-CLIP table values are not rerun results.

Next: Notebook 06 — fixed biomedical text prototypes and the proposed EviCT semantic branch.
