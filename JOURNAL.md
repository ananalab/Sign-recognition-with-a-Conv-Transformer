# Research log

Decisions, rationale and measured results, in chronological order.

## Scope and environment

**Decisions**
- Scope: Google ISLR (ASL, 250 signs) end to end, then LSFB-ISOL (French Belgian Sign Language)
  and cross-lingual transfer ASL -> LSFB.
- Python 3.12 with `uv`. `mediapipe` 1.0.1 ships a `py3-none` wheel, but 3.12 is the safest version
  for the torch / onnxruntime / lightgbm stack.
- The raw Kaggle data (> 40 GB) do not fit on the development laptop (≈ 16 GB free): the conversion
  to the compact unified format runs inside a Kaggle notebook and only its output is downloaded.
- LSFB poses are streamed from the UNamur server and reduced to the unified subset on the fly
  (≈ 10 KB per clip instead of ≈ 70 KB).
- Hips are excluded from the landmark subset: they are usually out of frame (y > 1) in webcam and
  LSFB recordings. Final subset: 2 × 21 hand points, 7 upper-body points, 40 lip points = 89.

## LSFB-ISOL inspection

- 120,739 instances, 4,657 glosses, 99 signers, 50 fps; long-tailed gloss distribution.
- LSFB v2 poses follow the MediaPipe *Tasks* numbering (33 pose / 478 face points), i.e. the same
  extractor as the browser demo, whereas the Kaggle landmarks come from the legacy Holistic solution.
- The official `train`/`test` splits are signer-independent (no signer in both).
- Vocabulary: 100 most frequent glosses performed by ≥ 10 signers -> 55,599 instances (≥ 217 per gloss).
- We use the *raw* poses (no interpolation or smoothing): missing hands appear as NaN, like in the
  Kaggle data and in live webcam use.
- Clips shorter than 4 frames (mostly empty raw pose files) are dropped: 0.56 % of the selected clips
  (an early estimate of 3.1 % came from a non-random 10k sample).
- Final LSFB data: 55,286 clips; signer split 26,114 train (50 signers) / 4,423 val (9) /
  24,749 test (40, official test signers). Training examples per gloss: 72 to 1,492 (median 185).

## Baselines (ASL, signer-independent)

