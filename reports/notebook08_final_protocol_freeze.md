# Notebook 08 — Final Multi-Seed Protocol Freeze

Timestamp UTC: 2026-10-01T14:28:42.243884+00:00

## Scope

Primary reduced-compute experiment:

- budget: b050
- seeds: 17, 42, 2026
- methods:
  - supervised
  - confidence-only EMA
  - agreement-filtered EMA

b025 is deferred because of the declared compute-budget limitation.

## Source-only development

Threshold selection:

SOURCE SELECTION ONLY.

Temperature fitting:

SOURCE CALIBRATION ONLY.

Target / MedSeg accessed before freeze:

NO.

## Final target gate

`allow_target_evaluation = true`

This flag exists only in the frozen evaluation protocol after completion
of the source-side protocol freeze.

## Qualitative reporting

The target qualitative selection rules were frozen before target access.

The final study will include extensive deterministic qualitative evidence:

- fixed-rank best / quartile / median / worst cases
- fixed lesion-burden examples
- uncertainty-ranked examples
- FP-heavy and FN-heavy failures
- seed-consistency panels
- calibrated probability heatmaps
- TP / FP / FN overlays
- four-view uncertainty heatmaps
- view-variance maps
- small-lesion analyses
- failure taxonomy
- empty-reference examples when eligible

All fixed failures will be retained.

No target score may be used to alter training, thresholding, calibration,
prompts, post-processing, or checkpoint selection.
