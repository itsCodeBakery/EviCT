# GDCM DICOM Decoder Verification

Date recorded: 2026-10-05

## Environment

- pydicom: 3.0.2
- python-gdcm: 3.2.6

## Preserved source archive

- Kaggle working path during verification: `/kaggle/working/evict_dx_figshare_temp/COVID-CT-MD.zip`
- Reported size: 10.50 GB
- Note: Kaggle working storage is session-scoped; this path is evidence of the verification run, not a durable dataset location.

## JPEG-lossless decode checks

All sampled DICOM files used transfer syntax:

`JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])`

Successful decodes:

| Cohort | File | Shape | dtype | Pixel range |
|---|---|---:|---|---:|
| CAP | `Cap Cases/cap014/IM0162.dcm` | 512×512 | uint16 | 0–2323 |
| COVID-19 | `COVID-19 Cases/P108/IM0036.dcm` | 512×512 | uint16 | 0–2510 |
| COVID-19 | `COVID-19 Cases/P103/IM0124.dcm` | 512×512 | uint16 | 0–2165 |
| Normal | `Normal Cases/normal046/IM0063.dcm` | 512×512 | uint16 | 0–4095 |
| Normal | `Normal Cases/normal002/IM0153.dcm` | 512×512 | uint16 | 0–2190 |

## Decision

GDCM JPEG-lossless pixel decoding is operational for sampled CAP, COVID-19, and Normal cases. Continue the full audit/preprocessing pipeline with decoder failures treated as hard data-integrity errors and logged by file/case ID.

This verification does not by itself validate RescaleSlope/RescaleIntercept handling, HU conversion, orientation, spacing, series ordering, or cohort labels. Those remain separate preprocessing/data-audit checks.
