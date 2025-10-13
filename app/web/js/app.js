// Browser demo: MediaPipe Holistic landmarks, then preprocess.js, then ONNX Runtime Web.
import { FilesetResolver, HolisticLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.min.mjs";
import { EDGES, GROUPS, N_LANDMARKS, SLICES } from "./landmarks.js";
import { featureCount, preprocess } from "./preprocess.js";
import { Segmenter, handsVisible, toSequence } from "./segmenter.js";

const MP_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const MP_MODEL =
  "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task";
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

const L = N_LANDMARKS;
const $ = (id) => document.getElementById(id);
const ui = {
  status: $("status"), statusText: $("status-text"), stage: $("stage"), video: $("video"), overlay: $("overlay"),
  empty: $("stage-empty"), fps: $("hud-fps"), lat: $("hud-lat"), badge: $("state-badge"), recFill: $("rec-fill"),
  bars: $("bars"), gloss: $("top1-gloss"), hint: $("top1-hint"), conf: $("top1-conf"),
  pred: $("prediction"), predMeta: $("pred-meta"), history: $("history"), easyGrid: $("easy-grid"), easyNote: $("easy-note"),
  vocabGrid: $("vocab-grid"), vocabSearch: $("vocab-search"), vocabCount: $("vocab-count"), vocabNote: $("vocab-note"),
  modelPicker: $("model-picker"), modePicker: $("mode-picker"), fileInput: $("file-input"),
  optSkeleton: $("opt-skeleton"), optMirror: $("opt-mirror"),
};

const app = {
  holistic: null, models: [], model: null, mode: "auto", running: false, stream: null,
  segmenter: new Segmenter(), holdFrames: null, fileFrames: null, lastFrame: null,
  frameTimes: [], mpMs: 0, modelMs: 0, history: [], lastVideoTime: -1, tsCounter: 0,
  examples: null, exampleOrder: null, exampleYScale: 1.0, fit: null, replayId: 0,
};

async function main() {
  bindEvents();
  try {
    await loadModelIndex();
    setStatus("Loading landmarker…");
    await loadHolistic();
    setStatus("Ready", "ready");
  } catch (e) {
    console.error(e);
    setStatus(e.message.slice(0, 80), "error");
  }
}

