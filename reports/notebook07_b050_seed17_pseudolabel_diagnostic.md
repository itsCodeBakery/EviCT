
# Notebook 07 — b050 seed17 pseudo-label diagnostic

This is a source-only post-training diagnostic.

Diagnostic cases: 4 source-selection cases.

Hidden fitting masks were not accessed.

Calibration data were not accessed.

Target / MedSeg data were not accessed.

## Matching policy results

Confidence-only EMA:
- Coverage: 99.162%
- Foreground precision: 83.586%
- Background correctness: 99.828%
- Accepted accuracy: 99.588%
- True-foreground coverage: 83.185%

Agreement-filtered EMA:
- Coverage: 99.162%
- Foreground precision: 83.599%
- Background correctness: 99.828%
- Accepted accuracy: 99.588%
- True-foreground coverage: 83.175%

Same-teacher agreement-filter diagnostic:

- Confident pixels removed: 0.0000%
- Foreground precision change: +0.0000 percentage points

The fixed confidence threshold 0.95 and disagreement threshold 0.10
were not changed using this diagnostic.
