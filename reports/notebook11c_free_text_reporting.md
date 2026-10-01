# EViCT Notebook 11C — Controlled Free-Text Reporting Arm

## Status

**PASS**

## Experimental design

The same validated prediction-derived evidence used by the deterministic and
strict sentence-ID arms was supplied to Qwen2.5-3B-Instruct.

Unlike the strict arm, Qwen was allowed to write the report prose.

The model did not receive target ground truth, Dice, IoU, case identity, or
patient identity.

For eligible fields, the predicted lesion-pixel value was supplied so that
numerical copying could be evaluated. For ineligible fields, the lesion extent
was withheld from the model.

## Raw preservation

Raw generations were written to disk before any validation.

Unique model prompts:

**33**

Records represented:

**900**

## Automated validation

Accepted free-text reports:

**0 / 900**

Rejected free-text reports:

**900 / 900**

Acceptance rate:

**0.00%**

Fallback rate:

**100.00%**

Exact numerical-copy check:

**99.56%**

Required structure:

**100.00%**

Scope limitation:

**100.00%**

Geometry limitation:

**100.00%**

Whole-lung limitation:

**100.00%**

Laterality/lobar limitation:

**100.00%**

Research-only / no-diagnosis limitation:

**0.00%**

## Safety handling

Raw rejected text is preserved for analysis.

Rejected text is not treated as an accepted report.

For a rejected output, the safe final output is the previously validated
Notebook-11A deterministic report.

## Interpretation limitation

The automated validator can establish exact numerical copying, structural
completeness, and several explicit unsupported-field constraints.

It cannot prove complete clinical factual correctness of unrestricted prose.
Any stronger clinical factuality claim would require independent expert review.

## Runtime

Model load time:

**5.68 s**

Generation time:

**66.65 s**

Peak allocated VRAM:

**3.232 GiB**

## Next

Notebook 11 can now be finalized, or the optional retrieval ablation can be run
as a separate experiment.

Notebook 12 will perform the formal reporting/reliability comparisons,
matched-coverage analysis, robustness tests, and efficiency accounting.
