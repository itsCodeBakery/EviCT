# VlmReporting — 20 curated visual CT reports

> **Research-only qualitative gallery.** Cases selected retrospectively by high reference-mask Dice;
> not a representative sample, not an independent clinical reader study.

## Selection protocol
- Frozen selection model: seed **42**, method **agreement_filtered_ema**, **single** prediction.
- Top **20 distinct** MedSeg slices (out of 100 candidates), ranked by Dice.
- Non-empty foreground prediction and at least 128 reference lesion pixels required.
- Reference mask: verified MedSeg channels **0 (GGO) OR 1 (consolidation)**.
- Ground truth influences **only showcase selection and the selection manifest**, never the report/VLM.
- Dice scores are external image-level segmentation metrics, not diagnosis or report accuracy.
- Model pixel extent in PDFs is descriptive (not guaranteed source-p80 eligible).
- The original frozen Notebook-11 validation does NOT automatically apply to these new VLM PDFs.

## Reports

| Rank | MedSeg slice | Dice (research selection) | Premium PDF |
|---:|---|---:|---|
| 1 | medseg_slice_000 | 0.8987 | [View PDF](pdf_reports/medseg_slice_000_seed42_agreement_filtered_ema.pdf) |
| 2 | medseg_slice_032 | 0.8807 | [View PDF](pdf_reports/medseg_slice_032_seed42_agreement_filtered_ema.pdf) |
| 3 | medseg_slice_009 | 0.8645 | [View PDF](pdf_reports/medseg_slice_009_seed42_agreement_filtered_ema.pdf) |
| 4 | medseg_slice_008 | 0.8606 | [View PDF](pdf_reports/medseg_slice_008_seed42_agreement_filtered_ema.pdf) |
| 5 | medseg_slice_067 | 0.8381 | [View PDF](pdf_reports/medseg_slice_067_seed42_agreement_filtered_ema.pdf) |
| 6 | medseg_slice_015 | 0.8362 | [View PDF](pdf_reports/medseg_slice_015_seed42_agreement_filtered_ema.pdf) |
| 7 | medseg_slice_096 | 0.8349 | [View PDF](pdf_reports/medseg_slice_096_seed42_agreement_filtered_ema.pdf) |
| 8 | medseg_slice_077 | 0.8238 | [View PDF](pdf_reports/medseg_slice_077_seed42_agreement_filtered_ema.pdf) |
| 9 | medseg_slice_073 | 0.8237 | [View PDF](pdf_reports/medseg_slice_073_seed42_agreement_filtered_ema.pdf) |
| 10 | medseg_slice_035 | 0.8222 | [View PDF](pdf_reports/medseg_slice_035_seed42_agreement_filtered_ema.pdf) |
| 11 | medseg_slice_025 | 0.8207 | [View PDF](pdf_reports/medseg_slice_025_seed42_agreement_filtered_ema.pdf) |
| 12 | medseg_slice_063 | 0.8094 | [View PDF](pdf_reports/medseg_slice_063_seed42_agreement_filtered_ema.pdf) |
| 13 | medseg_slice_007 | 0.8085 | [View PDF](pdf_reports/medseg_slice_007_seed42_agreement_filtered_ema.pdf) |
| 14 | medseg_slice_065 | 0.8069 | [View PDF](pdf_reports/medseg_slice_065_seed42_agreement_filtered_ema.pdf) |
| 15 | medseg_slice_042 | 0.8033 | [View PDF](pdf_reports/medseg_slice_042_seed42_agreement_filtered_ema.pdf) |
| 16 | medseg_slice_053 | 0.7991 | [View PDF](pdf_reports/medseg_slice_053_seed42_agreement_filtered_ema.pdf) |
| 17 | medseg_slice_058 | 0.7983 | [View PDF](pdf_reports/medseg_slice_058_seed42_agreement_filtered_ema.pdf) |
| 18 | medseg_slice_029 | 0.7947 | [View PDF](pdf_reports/medseg_slice_029_seed42_agreement_filtered_ema.pdf) |
| 19 | medseg_slice_098 | 0.7942 | [View PDF](pdf_reports/medseg_slice_098_seed42_agreement_filtered_ema.pdf) |
| 20 | medseg_slice_079 | 0.7897 | [View PDF](pdf_reports/medseg_slice_079_seed42_agreement_filtered_ema.pdf) |

## Files
- [selection_manifest.csv](selection_manifest.csv) — selected-case metrics and ranking.
- [full_eligible_ranking.csv](full_eligible_ranking.csv) — all positive eligible candidates for transparency.
- [report_index.csv](report_index.csv) — displayed evidence and VLM wording screens.
- `evidence/` — report-level structured evidence (includes raw generated draft for audit).
- `pdf_reports/` — premium original-CT / overlay / uncertainty PDFs.

## Limitations
The binary target cannot separate GGO from consolidation. No laterality, lobar location,
physical area, volume, whole-lung burden, diagnosis, clinical probability, or radiologist-level
validity is claimed. Qwen visual drafts are **not clinically verified**; forbidden claims are screened
and withheld where detected. No verified patient identifiers are present in these selected slices.
