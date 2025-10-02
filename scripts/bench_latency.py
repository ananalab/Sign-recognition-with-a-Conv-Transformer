"""CPU latency of the exported models, natively (onnxruntime) and in a browser (onnxruntime-web).

    uv run python scripts/bench_latency.py \
        --ckpt asl-transformer=models/asl-transformer-s0/best.pt \
        --ckpt asl-gru=models/asl-gru-s0/best.pt

Writes reports/latency.json. The browser benchmark runs headless Chromium (Playwright), WebAssembly
backend, one thread, i.e. the configuration of the static demo (no cross-origin isolation).
"""

from __future__ import annotations

from playwright.async_api import async_playwright

async def browser_bench(folder: Path, models: list[tuple[str, int, int]]) -> dict:
    page_src = PAGE.replace("__ORT__", ORT_WEB).replace("__MODELS__", json.dumps(models))
    (folder / "index.html").write_text(page_src)
    with serve(folder) as port:
        async with async_playwright() as p:
            b = await p.chromium.launch()
            page = await b.new_page()
            await page.goto(f"http://127.0.0.1:{port}/index.html")
            await page.wait_for_function("window.BENCH !== undefined", timeout=600_000)
            res = await page.evaluate("window.BENCH")
            await b.close()
    return res


