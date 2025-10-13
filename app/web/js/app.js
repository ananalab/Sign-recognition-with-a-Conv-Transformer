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

function setStatus(text, state = "loading") {
  ui.statusText.textContent = text;
  ui.status.dataset.state = state;
}
function setBadge(text, state) {
  ui.badge.textContent = text;
  ui.badge.dataset.state = state;
}
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

async function loadModelIndex() {
  const res = await fetch("models/index.json");
  if (!res.ok) throw new Error("models/index.json not found, run scripts/export_web_assets.py");
  app.models = (await res.json()).models;
  ui.modelPicker.innerHTML = "";
  for (const m of app.models) {
    const b = document.createElement("button");
    b.setAttribute("role", "radio");
    b.dataset.id = m.id;
    b.innerHTML = `${m.short} <span class="sub">${m.n_classes} signs</span>`;
    b.onclick = () => selectModel(m.id);
    ui.modelPicker.appendChild(b);
  }
  $("footer-models").textContent = app.models.map((m) => `${m.name} (${m.citation})`).join("; ");
  const wanted = new URLSearchParams(location.search).get("model");
  await selectModel(app.models.some((m) => m.id === wanted) ? wanted : app.models[0].id);
}

async function selectModel(id) {
  const meta = app.models.find((m) => m.id === id);
  setStatus(`Loading ${meta.short} model…`);
  const base = `models/${id}/`;
  const [labels, cfg, session] = await Promise.all([
    fetch(base + "labels.json").then((r) => r.json()),
    fetch(base + "preprocess.json").then((r) => r.json()),
    ort.InferenceSession.create(base + meta.file, { executionProviders: ["wasm"], graphOptimizationLevel: "all" }),
  ]);
  app.model = { ...meta, labels, cfg, session };
  // warm-up run so that the first displayed latency is representative
  const F = featureCount(cfg);
  await session.run({
    x: new ort.Tensor("float32", new Float32Array(cfg.T * F), [1, cfg.T, F]),
    mask: new ort.Tensor("float32", new Float32Array(cfg.T).fill(1), [1, cfg.T]),
  });
  for (const b of ui.modelPicker.children) b.setAttribute("aria-checked", String(b.dataset.id === id));
  renderVocab();
  renderEasy();
  resetPrediction();
  await loadExamples(meta);
  const u = new URL(location);
  u.searchParams.set("model", id);
  history.replaceState(null, "", u);
  setStatus(app.holistic ? "Ready" : "Loading landmarker…", app.holistic ? "ready" : "loading");
}

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

