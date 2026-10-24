"""Lab 08: FP8 weights. The official FP8 checkpoint on the same rate sweep as the BF16 one: a knee that moved.
Run: modal run labs/08-fp8-weights/modal/lab.py"""
import os, sys, time, subprocess, json, re
import modal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "modal"))
from common import *  # noqa: E402,F403

app = modal.App("llminference-lab08")
FP8 = "Qwen/Qwen3-4B-Instruct-2507-FP8"


@app.function(image=image, gpu="L4", timeout=2400, volumes=VOLUMES)
def run():
    for label, model in (("BF16", MODEL), ("FP8", FP8)):
        banner(f"{label}: {model}")
        proc, ready = serve(["--max-model-len", "4096"], model=model); print(f"ready after {ready} s")
        for k, v in log_lines("/tmp/serve.log", {"q": r"quantization=[a-z0-9_]+", "kv": r"GPU KV cache size: [^\n]*"}).items():
            print("  " + v[:120])
        print(bench_row("1 at a time, 1,024 in", bench_serve(rate="inf", num_prompts=16, input_len=1024, max_concurrency=1, model=model)))
        for rate in (2, 4, 8, 12):
            print(bench_row(f"{rate} req/s, 512 in", bench_serve(rate=rate, num_prompts=60, input_len=512, model=model)))
        stop(proc)


@app.local_entrypoint()
def main():
    run.remote()
