# EViCT Notebook 11C — Validator-Only Correction

## Status

**PASS**

No language-model generation was rerun.

The exact original Notebook-11C raw output package was reused.

Raw package SHA256:

`e8277e957085e0fd8a44dbf70a3c4d3ebe83081852ee1396804c6383a8360405`

## Why correction was necessary

The original validator contained two implementation errors.

First, it required the literal phrase `not a treatment recommendation`.
Qwen frequently generated the semantically equivalent phrase
`not a diagnosis or treatment recommendation`. The shared negation applies
to both diagnosis and treatment recommendation, but the string validator
incorrectly rejected it.

Second, the numerical validator required the correct lesion-pixel value to
appear exactly once. Some reports repeated the same correct evidence value
in two sections. No different value was generated, but the report was
incorrectly classified as a numerical-copy failure.

## What was not changed

- Qwen model: unchanged
- Qwen revision: unchanged
- prompts: unchanged
- raw generations: unchanged
- evidence: unchanged
- target data: not accessed
- target performance: not accessed

The exact empty-mask wording requirement was retained.

## Corrected result

Accepted controlled free-text records:

**888 / 900**

Rejected controlled free-text records:

**12 / 900**

Acceptance rate:

**98.67%**

Fallback rate:

**1.33%**

Numerical-copy accuracy:

**100.00%**

Research limitation compliance:

**99.11%**

Target-evidence wording compliance:

**99.56%**

## Interpretation

The original 0% acceptance rate must not be reported as a model-level
free-text failure because it was caused by an over-literal validation rule.

The corrected evaluation preserves genuine instruction-following failures
while removing false validator failures.
