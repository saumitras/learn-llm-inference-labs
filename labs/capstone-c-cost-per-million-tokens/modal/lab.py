"""Capstone C: cost per million tokens. Five configurations, each driven from one request at a time up to 128 at
once, and the price of a million output tokens at each point from the GPU's hourly rate.
Run: modal run labs/capstone-c-cost-per-million-tokens/modal/lab.py"""
import os, sys, time, subprocess, json, re, threading
import modal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "modal"))
from common import *  # noqa: E402,F403

app = modal.App("llminference-capstone-c")
FP8 = "Qwen/Qwen3-4B-Instruct-2507-FP8"
PRICE_PER_GPU_HOUR = 0.80
INPUT, OUTPUT = 1024, 256
LADDER = (1, 8, 32, 64, 128)


def ladder(label, model, gpus):
    print(f"  {'concurrency':>12} {'out tok/s':>10} {'TPOT p50':>9} {'TTFT p99':>9}  {'$ per 1M output tokens':>22}  {'$ per 1M total tokens':>21}")
    for c in LADDER:
        r = bench_serve(rate="inf", num_prompts=max(16, 2 * c), input_len=INPUT, output_len=OUTPUT, max_concurrency=c, model=model)
        out = r.get("Output token throughput (tok/s)", 0.0); total = r.get("Total Token throughput (tok/s)", r.get("Total token throughput (tok/s)", out * (INPUT + OUTPUT) / OUTPUT))
        hourly = PRICE_PER_GPU_HOUR * gpus
        per_m_out = hourly / (out * 3600 / 1e6) if out else float("inf"); per_m_total = hourly / (total * 3600 / 1e6) if total else float("inf")
        print(f"  {c:>12} {out:10.1f} {r.get('P50 TPOT (ms)', 0):8.1f}ms {r.get('P99 TTFT (ms)', 0):8.0f}ms  {per_m_out:22.3f}  {per_m_total:21.3f}")


@app.function(image=image, gpu="L4", timeout=3600, volumes=VOLUMES)
def one_gpu():
    for label, model, extra in (("One L4, BF16", MODEL, []), ("One L4, FP8 weights", FP8, []), ("One L4, FP8 weights + FP8 KV cache", FP8, ["--kv-cache-dtype", "fp8"])):
        banner(label); proc, ready = serve(["--max-model-len", "4096", *extra], model=model); print(f"  ready after {ready} s"); ladder(label, model, 1); stop(proc)


@app.function(image=image, gpu="L4:2", timeout=3600, volumes=VOLUMES)
def two_gpus():
    for label, extra in (("Two L4s, FP8, tensor parallel", ["--tensor-parallel-size", "2"]), ("Two L4s, FP8, two copies", ["--data-parallel-size", "2"])):
        banner(label); proc, ready = serve(["--max-model-len", "4096", "--kv-cache-dtype", "fp8", *extra], model=FP8); print(f"  ready after {ready} s"); ladder(label, FP8, 2); stop(proc)


@app.local_entrypoint()
def main():
    one_gpu.remote()
    two_gpus.remote()
