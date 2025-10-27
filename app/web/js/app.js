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

async function loadHolistic() {
  const vision = await FilesetResolver.forVisionTasks(MP_WASM);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MP_MODEL, delegate },
    runningMode: "VIDEO",
    minHandLandmarksConfidence: 0.5,
  });
  try {
    app.holistic = await HolisticLandmarker.createFromOptions(vision, opts("GPU"));
  } catch {
    app.holistic = await HolisticLandmarker.createFromOptions(vision, opts("CPU"));
  }
}

function toUnified(res) {
  const frame = new Float64Array(L * 3).fill(NaN);
  const src = {
    left_hand: res.leftHandLandmarks?.[0],
    right_hand: res.rightHandLandmarks?.[0],
    pose: res.poseLandmarks?.[0],
    face: res.faceLandmarks?.[0],
  };
  let k = 0;
  for (const g of GROUPS) {
    const pts = src[g.source];
    for (const i of g.indices) {
      const p = pts?.[i];
      if (p) frame.set([p.x, p.y, p.z], k * 3);
      k++;
    }
  }
  return frame;
}

function currentYScale() {
  // MediaPipe normalises x by the width and y by the height of the frame
  if (app.fit) return app.exampleYScale ?? 1.0;
  const v = ui.video;
  return v.videoWidth ? v.videoHeight / v.videoWidth : 0.75;
}

async function classify(frames, source) {
  if (!app.model || frames.length < 2) return;
  const m = app.model;
  const t0 = performance.now();
  const { features, mask } = preprocess(toSequence(frames), m.cfg, currentYScale());
  const F = features.length / m.cfg.T;
  const out = await m.session.run({
    x: new ort.Tensor("float32", features, [1, m.cfg.T, F]),
    mask: new ort.Tensor("float32", mask, [1, m.cfg.T]),
  });
  app.modelMs = performance.now() - t0;
  ui.lat.textContent = app.fit
    ? `recorded landmarks · model ${app.modelMs.toFixed(1)} ms`
    : `landmarks ${app.mpMs.toFixed(0)} ms · model ${app.modelMs.toFixed(1)} ms`;
  const probs = out.probs.data;
  const order = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]).slice(0, 5);
  showPrediction(order.map((i) => ({ label: m.labels[i], p: probs[i] })), frames.length, source);
}

function resetPrediction() {
  ui.gloss.textContent = "—";
  ui.hint.textContent = "Sign something to see the model's guess.";
  ui.conf.textContent = "";
  ui.bars.innerHTML = "";
  ui.predMeta.textContent = "";
}

function showPrediction(top, nFrames, source) {
  const best = top[0];
  const sure = best.p >= 0.35;
  ui.pred.classList.toggle("uncertain", !sure);
  ui.gloss.textContent = best.label;
  ui.hint.textContent = sure ? `${source} · ${nFrames} frames` : "Low confidence, try again a bit slower";
  ui.conf.textContent = `${Math.round(best.p * 100)}%`;
  ui.predMeta.textContent = `${app.modelMs.toFixed(1)} ms`;
  ui.bars.innerHTML = top
    .map((t) => `<li><span class="name">${t.label}</span><span class="val">${(t.p * 100).toFixed(1)}%</span>
      <div class="track"><div class="fill" style="width:${(t.p * 100).toFixed(1)}%"></div></div></li>`)
    .join("");
  app.history.push({ label: best.label, p: best.p });
  renderHistory();
  highlightVocab(best.label);
}

function renderHistory() {
  ui.history.innerHTML = app.history.length
    ? app.history.slice(-24).map((h) => `<span class="${h.p < 0.35 ? "low" : ""}">${h.label}<small>${Math.round(h.p * 100)}%</small></span>`).join("")
    : '<span class="muted">Nothing yet.</span>';
}

function signLink(m, l, extra = "") {
  return m.dictionary
    ? `<a href="${m.dictionary.replace("{gloss}", encodeURIComponent(l.toLowerCase()))}" target="_blank" rel="noopener" data-l="${l}">${l}${extra}</a>`
    : `<span data-l="${l}">${l}${extra}</span>`;
}
function renderEasy() {
  const m = app.model;
  const easy = m.easy || [];
  ui.easyGrid.innerHTML = easy.map((e) => signLink(m, e.label, `<small>${Math.round(e.acc * 100)}%</small>`)).join("");
  ui.easyNote.textContent = easy.length
    ? `Best-recognised signs on unseen test signers (per-class accuracy).${m.dictionary ? " Click one to see how it is performed." : ""}`
    : "";
}
function renderVocab(filter = "") {
  const m = app.model;
  const q = filter.trim().toLowerCase();
  const items = m.labels.filter((l) => l.toLowerCase().includes(q));
  ui.vocabCount.textContent = `· ${m.labels.length}`;
  ui.vocabGrid.innerHTML = items.map((l) => signLink(m, l)).join("");
  ui.vocabNote.textContent = m.dictionary ? "Click a sign to see how it is performed (external dictionary)." : m.note || "";
}
function highlightVocab(label) {
  for (const el of [...ui.vocabGrid.children, ...ui.easyGrid.children]) el.classList.toggle("hit", el.dataset.l === label);
}

async function analyseFile(file) {
  if (!file || !app.holistic) return;
  stopCamera();
  setMode("file", false);
  ui.empty.hidden = true;
  ui.stage.classList.remove("mirrored");
  app.fit = null;
  app.replayId++;
  ui.video.src = URL.createObjectURL(file);
  ui.video.playbackRate = 1;
  app.fileFrames = [];
  setBadge("Analysing video…", "signing");
  const step = (now, meta) => {
    if (!app.fileFrames) return;
    onFrame(detect(meta.mediaTime * 1000), now);
    if (!ui.video.ended) ui.video.requestVideoFrameCallback(step);
  };
  ui.video.onended = async () => {
    const frames = app.fileFrames || [];
    setBadge("Recognised", "done");
    await classify(frames, `video file`);
    app.fileFrames = null;
  };
  ui.video.requestVideoFrameCallback(step);
  await ui.video.play();
}

function holdStart() {
  if (app.mode !== "hold" || app.holdFrames || !app.running) return;
  app.holdFrames = [];
  setBadge("Recording…", "signing");
}
function holdEnd() {
  if (!app.holdFrames) return;
  const frames = app.holdFrames;
  app.holdFrames = null;
  ui.recFill.style.width = "0%";
  setBadge("Recognised", "done");
  classify(frames, "hold");
}

function bindEvents() {
  for (const b of ui.modePicker.children) b.onclick = () => setMode(b.dataset.mode);
  $("btn-camera").onclick = () => (app.running ? stopCamera() : startCamera());
  $("btn-camera-big").onclick = startCamera;
  $("btn-file-big").onclick = () => ui.fileInput.click();
  $("btn-example").onclick = playExample;
  $("btn-example-big").onclick = playExample;
  ui.fileInput.onchange = (e) => analyseFile(e.target.files[0]);
  $("btn-clear").onclick = () => { app.history = []; renderHistory(); };
  ui.optMirror.onchange = () => ui.stage.classList.toggle("mirrored", ui.optMirror.checked && app.running);
  ui.optSkeleton.onchange = () => drawOverlay(app.lastFrame);
  ui.vocabSearch.oninput = (e) => renderVocab(e.target.value);
  window.addEventListener("keydown", (e) => { if (e.code === "Space" && e.target === document.body) { e.preventDefault(); holdStart(); } });
  window.addEventListener("keyup", (e) => { if (e.code === "Space") holdEnd(); });
  ui.stage.addEventListener("pointerdown", holdStart);
  window.addEventListener("pointerup", holdEnd);
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

