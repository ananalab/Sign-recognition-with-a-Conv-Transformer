# Model card

Format inspired by Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019).
Models of the signrec project. All figures below come from the evaluation scripts (see `reports/`
and `report.pdf`).

## Model details

| | ASL model | LSFB model |
|---|---|---|
| Task | Isolated sign recognition (one sign per clip, closed vocabulary) | same |
| Vocabulary | 250 ASL signs (Google ISLR) | 100 LSFB glosses (most frequent in LSFB-ISOL, ≥ 10 signers) |
| Input | MediaPipe Holistic landmarks: 2 × 21 hand, 7 upper-body, 40 lip points per frame | same |
| Architecture | Conv-Transformer (depthwise temporal convolutions + self-attention), 2.2 M parameters | same, trained from scratch (ASL initialisation did not help) |
| Output | Probability over the vocabulary (top-5 shown in the demo) | same |
| Formats | PyTorch checkpoint, ONNX FP32 (shipped), ONNX dynamic INT8 | same |

## Intended use

- Demonstrations and research on pose-based isolated sign recognition.
- Educational use: practising the signs of a fixed vocabulary with immediate feedback.
- Runs locally in a web browser; video is never sent to a server.

