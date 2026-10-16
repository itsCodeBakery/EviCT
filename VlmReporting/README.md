# VlmReporting — Curated EviCT visual CT reporting gallery

This root-level directory stores premium, research-only PDF reports generated from the *frozen EviCT* Notebook-09 predictions, with the original MedSeg CT slices and an optional Qwen2.5-VL visual narrative.

## Report generation
- Selection: **20 distinct MedSeg CT slices** with the highest non-empty-lesion Dice for the **preselected seed 42 / agreement_filtered_ema / single-view prediction**.
- Evaluation-only selection reference: MedSeg ground-glass (channel 0) OR consolidation (channel 1), according to the repository's frozen dataset audit. This reference is **never passed** to the report generator or VLM.
- The ranked, best-performing selected cases are an **illustrative gallery**, not an unbiased estimate of average model accuracy or clinical utility.
- Report format: original CT image, predicted-opacity overlay, uncertainty map, constrained research findings, an explicitly **unverified** VLM visual description, and research-only impression.
- Any pixel extent displayed in the report is **descriptive and not necessarily approved by the source-derived p80 uncertainty eligibility gate**.
- Missing independent radiologist review, anatomical localization, verified physical geometry and standalone etiological diagnosis must not be inferred or invented.
- The original Notebook-11 validated reporting arms are separate; these newly formatted PDFs have **not** been validated by the original reporting study.

## Status
Directory initialized. Run the accompanying Kaggle generation notebook to publish the curated `pdf_reports/`, `evidence/`, and `selection_manifest.csv` artifacts to this folder.

## Research protocol
See `protocol_audit_segdb1.md`, `config/frozen_protocol.json`, `config/notebook11_final_freeze.json` and `tables/final_result_source_map.csv` at the repository root.
