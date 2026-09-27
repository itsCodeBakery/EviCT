# EviCT Protocol Audit

## Notebook 01A — SegDB-2

Timestamp: 2026-09-27T05:17:12.751475+00:00

## Pairing correction

The first Notebook 01A implementation assumed that the CT volume
and its mask files had identical filename stems.

That was incorrect.

The dataset's `metadata.csv` explicitly maps each CT volume to its
lung mask, infection mask, and combined mask.

The corrected audit therefore uses `metadata.csv` rather than
filename-stem inference.

The failed audit remains preserved in Git history.

## Corrected source audit

Metadata-defined cases: **20**

Geometry-valid cases: **20/20**

Excluded cases: **0**

### Provenance groups

{
  "coronacases_named": 10,
  "radiopaedia_named": 10
}

## Observed mask values

Infection:

`[0.0, 1.0]`

Lung:

`[0.0, 1.0, 2.0]`

Combined:

`[0.0, 1.0, 2.0, 3.0]`

Infection mask binary-compatible:

**True**

## Intensity policy

No preprocessing has been applied.

No stored value is being assumed to be raw Hounsfield units solely
because the file is NIfTI.

Radiopaedia-named and coronacases-named records remain identifiable
for provenance-specific preprocessing decisions.

## Grouping

One metadata row corresponds to one complete CT volume and its
associated masks.

`group_id` currently represents complete-volume identity.

It is not being claimed as an independently recovered patient ID.

## Track R

Matched CT-Insight VLM reproduction:

**NOT READY**

Critical unresolved items include the original 367-slice selection,
training partitions, caption bank, preprocessing details, optimizer
schedule, model-selection convention, and metric reduction.

## Track C

SegDB-2 source audit:

**PASS**

SegDB-1 / MedSeg still requires independent audit before the
bidirectional controlled protocol can be frozen.
