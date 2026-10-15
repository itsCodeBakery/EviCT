# EviCT evidence-grounded CT reporting

This subdirectory contains research-only evidence-grounded lung CT reporting artifacts. Use the companion one-cell Kaggle notebook (`EviCT_One_Cell_Reporting.ipynb`) to generate reports from saved original-grid EviCT probability outputs. The notebook writes `evidence/`, `reports/`, `tables/`, `figures/`, `audit/`, and `handoff/` and can synchronize those files here using a Kaggle `GITHUB_TOKEN` secret.

Reports must not infer COVID-19, pneumonia, GGO vs consolidation, laterality, or lobar involvement from binary lesion masks. Physical measurements require verified geometry. Low-confidence measurements are withheld. The output does not substitute for clinical radiology assessment.

**Important:** Kaggle must have genuine compatible EviCT prediction `.npz` records with explicit `case_id` and source-selected threshold. Raw checkpoints require an architecture-compatible inference adapter. Reports are not created from missing data. LLM and GitHub uploads are conditional on available resources and credentials.
