# NOTEBOOK 11A HANDOFF

Status: PASS

Deterministic strict baseline:
`reports/notebook11_deterministic/reports.jsonl`

Sentence catalog:
`config/notebook11_sentence_catalog.json`

Sentence catalog SHA256:
`b7f8a5ed5ddc1961e56e46e8ad2f2f27f7a525ba2a25714d567310fcfa2cb188`

Strict ID protocol:
`config/notebook11_strict_id_protocol.json`

Reports generated:
900

Validation failures:
0

Numerical lesion extent emitted:
38

Numerical lesion extent withheld:
862

Important rules for Notebook 11B:

1. Do not provide target ground truth to the LLM.
2. Do not provide target Dice or IoU to the LLM.
3. The LLM may return approved sentence IDs only.
4. The LLM may not generate numerical values.
5. Python renders all final sentences.
6. Missing mandatory limitations trigger deterministic fallback.
7. Unknown IDs trigger deterministic fallback.
8. Keep the raw LLM response before validation.

Next:
Notebook 11B — strict sentence-ID LLM realization.
