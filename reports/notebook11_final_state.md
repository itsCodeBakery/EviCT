# EViCT Notebook 11 — Final State

## Status

**COMPLETE AND FROZEN**

## Reporting arms

### Deterministic baseline

- Reports: **900**
- Valid reports: **900 / 900**
- Fallback required: **0**
- Final safe delivery: **100%**

### Strict sentence-ID Qwen

- Reports represented: **900**
- Directly valid: **896 / 900**
- Direct validity: **99.56%**
- Fallback: **4 / 900**
- Fallback rate: **0.44%**
- Final safe delivery: **900 / 900**

The language model did not receive protected lesion-pixel values.
Python rendered the final numerical report.

### Controlled free-text Qwen

- Records: **900**
- Accepted: **896 / 900**
- Acceptance rate: **99.56%**
- Rejected: **4 / 900**
- Fallback rate: **0.44%**
- Numerical-copy accuracy: **100.00%**
- Final safe delivery: **900 / 900**

## Remaining free-text failure

The only remaining failure class is:

`empty_prediction_wording`

Affected records:

**4 / 900**

Qwen wrote a semantically similar empty-prediction phrase but did not reproduce
the explicitly required exact wording.

These are reported as instruction-following failures, not as clinical
hallucinations.

## Validator software corrections

The original controlled free-text evaluation produced an apparent 0% acceptance
rate because the validator was overly literal.

Two validator-only corrections were subsequently applied to the exact same
frozen raw Qwen outputs.

No Qwen generation was rerun.

No prompt, model, evidence, or target-derived information was changed.

Original raw generation SHA256:

`e8277e957085e0fd8a44dbf70a3c4d3ebe83081852ee1396804c6383a8360405`

## Safety behavior

Every rejected strict-ID or free-text output is preserved for analysis.

The deployed/safe output falls back to the previously validated deterministic
evidence-grounded report.

Therefore final safe delivery is:

**900 / 900 reports for all reporting pathways.**

## Optional retrieval

The retrieval ablation is not run in the core experiment because it was
optional and the principal reporting comparison is already complete.

## Next stage

**Notebook 12 — Reporting, Robustness, and Efficiency Evaluation**

Notebook 12 will include:

- formal reporting-arm comparison;
- matched-coverage uncertainty analysis;
- robustness / perturbation tests;
- completeness and fallback behavior;
- runtime and memory accounting;
- final publication figures and tables.
