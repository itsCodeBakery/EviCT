# NOTEBOOK 11 HANDOFF

Status: COMPLETE AND FROZEN

Final freeze:
`config/notebook11_final_freeze.json`

Reporting-arm summary:
`tables/notebook11_reporting_arm_summary.csv`

Deterministic:
900/900 valid

Strict sentence-ID Qwen:
896/900 directly valid
4/900 fallback
900/900 safe final

Controlled free text:
896/900 accepted
4/900 fallback
100.00% numerical-copy accuracy
900/900 safe final

Remaining free-text failure:
empty_prediction_wording = 4 records

Raw free-text SHA256:
e8277e957085e0fd8a44dbf70a3c4d3ebe83081852ee1396804c6383a8360405

Important:
- Do not rerun Qwen for Notebook 11.
- Do not overwrite original Notebook-11C artifacts.
- Preserve validator-correction provenance.
- Do not call the remaining four failures clinical hallucinations.
- Optional retrieval ablation was not required for core completion.

Next:
Notebook 12 — Reporting / Robustness / Efficiency.
