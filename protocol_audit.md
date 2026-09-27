# EviCT Protocol Audit

## Audit stage

Notebook 01A — SegDB-2 source audit

Timestamp: 2026-09-27T04:48:37.071697+00:00

---

## Dataset currently audited

Execution copy:

`/kaggle/input/datasets/andrewmvd/covid19-ct-scans`

Observed unique volume identifiers:

**40**

Cases with CT + infection mask + lung mask + combined mask:

**0/40**

Cases with header-level image/mask geometry consistency:

**0/40**

Exact duplicate file records detected within the same role:

**0**

---

## Observed mask labels

### Infection mask

[]

Binary-compatible:

False

### Lung mask

[]

No numeric left/right mapping is assumed at this stage.

### Lung-and-infection mask

[]

No combined-label semantic mapping is assumed at this stage.

---

## Intensity handling

CT intensity values were inspected only through deterministic sampled
statistics.

No HU conversion, clipping, normalization, windowing, resizing,
interpolation, or augmentation has been performed.

The file suffix `.nii` or `.nii.gz` is not treated as proof that stored
values are raw Hounsfield units.

The Radiopaedia-named and coronacases-named records are retained as
separate provenance strata for the subsequent preprocessing audit.

---

## Grouping

`case_id` is the volume-file identifier.

For this audit, `group_id` equals `case_id` only to preserve volume-level
independence.

This is NOT being presented as a recovered patient identifier.

No synthetic patient IDs were created.

---

## Kaggle copy versus authoritative source

The Kaggle dataset is being treated as the execution copy.

File-level SHA-256 hashes were calculated for the attached files.

The extracted Kaggle files have NOT yet been independently compared byte
for byte against the official source archives.

Therefore exact mirror equivalence is not yet claimed.

---

## CT-Insight VLM reproduction status

Linked repository tested:

`https://github.com/Owais-CodeHub/CT-Insight-VLM.git`

Repository reachable from this Kaggle session:

**False**

A matched reproduction of the base paper is NOT yet declared.

The following remain unresolved until recovered directly from code,
supplementary material, source records, or authors:

1. Exact SegDB-2 367-slice selection.
2. Exact source/target training partitions.
3. Exact binary lesion mapping used in the paper.
4. Caption repository / caption-bank construction.
5. Decoder implementation details.
6. Complete optimizer and epoch/update schedule.
7. Exact checkpoint selection rule.
8. Metric reduction convention.
9. Exact author code revision.
10. Exact preprocessing sequence.

Historical published numbers must therefore remain separate from future
EviCT rerun results.

---

## Track status

### Track R — matched reproduction

**NOT READY**

Reason: critical base-paper protocol details remain unresolved.

### Track C — controlled benchmark

**SEGDB-2 DATA AUDIT IN PROGRESS**

The dataset may proceed to later preprocessing only after this audit
passes and the second core dataset is independently audited.

---

## Scientific actions prohibited at this point

- No target-based model tuning.
- No threshold selection.
- No train/selection/calibration split yet.
- No hidden-label access policy yet.
- No preprocessing decisions from target performance.
- No model training.
- No claim of outperforming CT-Insight VLM.

