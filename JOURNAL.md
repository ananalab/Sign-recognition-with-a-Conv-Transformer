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

