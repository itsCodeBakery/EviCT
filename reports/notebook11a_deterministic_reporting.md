# EViCT Notebook 11A — Deterministic Evidence-Grounded Reporting

## Status

**PASS**

## Reports

- Evidence records processed: **900**
- Deterministic reports generated: **900**
- Validation failures: **0**
- LLM used: **No**
- GPU required: **No**

## Numerical reporting

Numerical lesion extent was emitted only when the Notebook-10 source-derived
p80 eligibility rule allowed it.

- Extent emitted: **38**
- Extent withheld: **862**

The numerical values are rendered directly from the validated evidence JSON.
They are not generated or rewritten by a language model.

## Empty-mask wording

When the predicted lesion mask is empty, the approved wording is:

> no target opacity detected by the model

The system does not convert an empty model prediction into a claim that the scan
is normal or disease-free.

## Mandatory limitations

Every report states that:

- the input is a selected 2D image, not a complete CT volume;
- physical lesion area and volume are unavailable;
- whole-lung involvement is unavailable;
- laterality and lobar localization are unavailable;
- a genuine four-view lesion-extent stability range is unavailable;
- the output is a research summary, not a diagnosis or treatment recommendation.

## Validator tests

- Unknown sentence ID rejection: **PASS**
- Missing mandatory limitation rejection: **PASS**
- Altered protected numerical value detection: **PASS**

## Next stage

Notebook 11B may load the pinned language model.

The language model will be allowed to select only approved sentence IDs.
Python will continue to render all final sentences and protected numerical
values.
