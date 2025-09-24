"""CPU latency of the exported models, natively (onnxruntime) and in a browser (onnxruntime-web).

    uv run python scripts/bench_latency.py \
        --ckpt asl-transformer=models/asl-transformer-s0/best.pt \
        --ckpt asl-gru=models/asl-gru-s0/best.pt

Writes reports/latency.json. The browser benchmark runs headless Chromium (Playwright), WebAssembly
backend, one thread, i.e. the configuration of the static demo (no cross-origin isolation).
"""

