// Line-by-line port of src/signrec/preprocess.py (parity checked by tests/test_js_parity.py).
// A sequence is {T, data: Float64Array(T * L * 3)} with NaN for missing landmarks.

import { MIRROR, N_LANDMARKS, POSE_POS, SLICES } from "./landmarks.js";

