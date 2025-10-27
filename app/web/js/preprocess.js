// Line-by-line port of src/signrec/preprocess.py (parity checked by tests/test_js_parity.py).
// A sequence is {T, data: Float64Array(T * L * 3)} with NaN for missing landmarks.

import { MIRROR, N_LANDMARKS, POSE_POS, SLICES } from "./landmarks.js";

const L = N_LANDMARKS;
const EPS = 1e-6;
const [LH0, LH1] = SLICES.left_hand;
const [RH0, RH1] = SLICES.right_hand;

export function trimHandless(seq) {
  const { lh, rh } = handPresence(seq);
  let first = -1;
  let last = -1;
  for (let t = 0; t < seq.T; t++) if (lh[t] || rh[t]) { if (first < 0) first = t; last = t; }
  return first < 0 ? seq : slice(seq, first, last + 1);
}

export function mirror(seq) {
  const { T, data } = seq;
  const out = new Float64Array(data.length);
  for (let t = 0; t < T; t++) {
    for (let l = 0; l < L; l++) {
      const src = (t * L + MIRROR[l]) * 3;
      const dst = (t * L + l) * 3;
      out[dst] = 1.0 - data[src];
      out[dst + 1] = data[src + 1];
      out[dst + 2] = data[src + 2];
    }
  }
  return { T, data: out };
}

export function normalize(seq, minShoulderWidth) {
  const { T, data } = seq;
  const ls = POSE_POS.l_shoulder;
  const rs = POSE_POS.r_shoulder;
  let wSum = 0, cx = 0, cy = 0, nOk = 0;
  for (let t = 0; t < T; t++) {
    const a = [data[at(t, ls, 0)], data[at(t, ls, 1)]];
    const b = [data[at(t, rs, 0)], data[at(t, rs, 1)]];
    if ([...a, ...b].some(Number.isNaN)) continue;
    wSum += Math.hypot(a[0] - b[0], a[1] - b[1]);
    cx += (a[0] + b[0]) / 2;
    cy += (a[1] + b[1]) / 2;
    nOk++;
  }
  const width = nOk ? wSum / nOk : 0;
  let center, scale;
  if (width >= minShoulderWidth) {
    center = [cx / nOk, cy / nOk];
    scale = width;
  } else {
    let n = 0, sx = 0, sy = 0;
    for (let i = 0; i < T * L; i++) {
      const x = data[i * 3], y = data[i * 3 + 1];
      if (Number.isNaN(x) || Number.isNaN(y)) continue;
      sx += x; sy += y; n++;
    }
    if (n === 0) return { T, data: data.slice() };
    center = [sx / n, sy / n];
    let vx = 0, vy = 0;
    for (let i = 0; i < T * L; i++) {
      const x = data[i * 3], y = data[i * 3 + 1];
      if (Number.isNaN(x) || Number.isNaN(y)) continue;
      vx += (x - center[0]) ** 2; vy += (y - center[1]) ** 2;
    }
    const std = (Math.sqrt(vx / n) + Math.sqrt(vy / n)) / 2;
    scale = Math.max(std * 4.0, EPS);
  }
  const out = new Float64Array(data.length);
  for (let i = 0; i < T * L; i++) {
    out[i * 3] = (data[i * 3] - center[0]) / scale;
    out[i * 3 + 1] = (data[i * 3 + 1] - center[1]) / scale;
    out[i * 3 + 2] = data[i * 3 + 2] / scale;
  }
  return { T, data: out };
}

