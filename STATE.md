# EviCT Execution State

## Current stage

NOTEBOOK_08_B050_SEED17_SOURCE_PROTOCOL_PILOT_COMPLETE

## Notebook 07

b050 seed17 common warm-up:

COMPLETE

b050 seed17 three-way comparison:

COMPLETE

Pseudo-label diagnostic:

COMPLETE

## Notebook 08A

Source-selection threshold sweep:

COMPLETE

Source-calibration temperature scaling:

COMPLETE

Single-view inference implementation:

COMPLETE

Four-view inference implementation:

COMPLETE

Uncertainty scoring:

COMPLETE

Source protocol pilot:

COMPLETE

## Pre-calibration training lock

GitHub commit:

8fc879f610c4deb44b291174562c3a67d163850f

Training lock SHA256:

2909630c2b92cbc3df31244a79b1344c5d1ec95061babd59cee68f70b88818c8

## Deferred confirmatory Notebook-07 runs

1. b050 seed42
2. b050 seed2026
3. b025 seed17
4. b025 seed42
5. b025 seed2026

These remain required.

Their training configuration may NOT be changed based on Notebook-08 calibration results.

## Isolation

Hidden fitting masks used:

NO

Source calibration accessed:

YES — Notebook 08 only

Target / MedSeg accessed:

NO

Target lock:

ACTIVE

allow_target_evaluation:

FALSE

## Next

Complete the five deferred Notebook-07 confirmatory runs using the pre-calibration training lock.

Then perform the FINAL Notebook-08 multi-seed protocol freeze.

Notebook 09 target evaluation remains LOCKED.
