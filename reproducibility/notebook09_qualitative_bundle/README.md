# EViCT Notebook-09 Qualitative Reproducibility Bundle

Generated: 2026-10-01T17:19:26.660957+00:00

## Purpose

This bundle preserves the corrected locked MedSeg external-target qualitative analysis.

It is intended to allow later regeneration or presentation-only modification of figures
without retraining the models or changing the scientific protocol.

## Scientific status

- Original Notebook-09 target evaluation using `x/255 -> clip [0,1]` was invalid due to a documented preprocessing implementation error.
- Corrected MedSeg preprocessing uses the frozen source CT pathway:
  - clip signed CT values to `[-1250, 250]`
  - scale linearly to `[0,1]`
- No checkpoint was changed.
- No threshold was changed.
- No temperature was changed.
- No model architecture was changed.
- No training was rerun.
- No post-processing was changed.
- The same correction was applied to all seeds and all methods.

## Corrected primary results

Agreement-filtered EMA:
- Dice: 0.676162 ± 0.025522
- IoU: 0.529466
- Sensitivity: 0.797594
- Specificity: 0.968758

Confidence-only EMA:
- Dice: 0.671247 ± 0.031176
- IoU: 0.525264
- Sensitivity: 0.795928
- Specificity: 0.968387

Supervised:
- Dice: 0.645673 ± 0.051701
- IoU: 0.500116
- Sensitivity: 0.679655
- Specificity: 0.979064

## Qualitative policy

Original qualitative case-selection rules were frozen before target access.

Additional high-Dice figures are post-evaluation descriptive visualizations only.
They must not be described as predeclared qualitative selections.

## What may be changed later without rerunning inference

Presentation-only changes:
- title wording
- font size
- layout
- panel ordering
- legend placement
- DPI
- PNG/PDF/SVG export
- figure dimensions

These do not change the scientific analysis.

## What requires explicit documentation as post-hoc

Any new:
- case-selection rule
- performance ranking
- subset definition
- method-specific example selection

must be described transparently as post-hoc descriptive visualization.

## Prediction files

The full corrected prediction arrays remain under:

`artifacts/large/notebook09_corrected/`

Those large arrays should be preserved separately or archived in a durable GitHub Release.

This bundle preserves the manifests needed to verify them.
