// Real-time sign segmentation from hand presence and motion energy.
//
// States: idle -> signing once hands are visible for `startFrames` frames; signing ends when the
// hands are gone for `endNoHandFrames`, still for `stillFrames` or after `maxFrames`. After a
// "still" end the segmenter waits in cooldown until the hands move again or go down.
//
// Frames are unified-format Float64Array(L*3). The segmenter only buffers them; the caller
// classifies the emitted segment with preprocess() and the ONNX model.

import { N_LANDMARKS, POSE_POS, SLICES } from "./landmarks.js";

