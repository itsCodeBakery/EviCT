# EViCT Publication Qualitative Figures

This directory contains manuscript-oriented qualitative figures from the corrected
Notebook-09 external-target MedSeg evaluation.

## Design

All figures use:

- exactly three case rows;
- large typography for readability after LaTeX scaling;
- vector PDF and SVG exports;
- 600-DPI PNG exports;
- no Dice values or other quantitative performance scores displayed inside the panels.

Quantitative results should be reported in the manuscript tables and captions rather
than embedded within the images.

## Figures

### 01 — Comparative Segmentation Morphology Across Training Strategies on Representative External CT Slices

Shows representative CT slices, reference masks, and predictions from supervised,
confidence-only EMA, and agreement-filtered EMA training.

### 02 — Characteristic Error Phenotypes in External-Target Lung Lesion Segmentation

Shows qualitatively distinct under-segmentation, over-segmentation, and severe
boundary-mismatch patterns.

### 03 — Morphologically Concordant Lung Lesion Segmentations Under Agreement-Filtered Semi-Supervision

Shows cases with strong reference-prediction morphological concordance.
These cases were selected post hoc using an objective ranking rule for descriptive
visualization only.

### 04 — Consistent Lesion Delineation Across Training Strategies and Random Seeds

Shows cases exhibiting strong aggregate segmentation behavior across the complete
method/seed evaluation set.

### 05 — Qualitative Patterns Associated with Improved Lesion Delineation Following Agreement-Filtered Semi-Supervision

Shows cases with the largest objectively ranked improvement relative to supervised
training. This is a post-hoc descriptive visualization and should be identified as such.

### 06 — Cross-Seed Stability of Agreement-Filtered Lung Lesion Segmentation

Shows cases with low pairwise prediction-mask disagreement across random seeds 17,
42, and 2026. Stability is computed inside the union of the reference and seed-specific
prediction masks.

## Scientific status

These figures are generated entirely from the already-preserved corrected Notebook-09
prediction tensors.

No model training, model inference, threshold selection, temperature calibration,
checkpoint selection, or post-processing modification is performed here.

Case selection rules are preserved in:

`tables/notebook09_publication_figures/publication_qualitative_case_selection.csv`
