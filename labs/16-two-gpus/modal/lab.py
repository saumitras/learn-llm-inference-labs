"""Lab 16: two GPUs. Tensor parallel across two L4s, then two data-parallel copies, against one GPU.
Run: modal run labs/16-two-gpus/modal/lab.py"""
import os, sys, time, subprocess, json, re
import modal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "modal"))
from common import *  # noqa: E402,F403

app = modal.App("llminference-lab16")


def measure(label):
    for k, v in log_lines("/tmp/serve.log", {"kv": r"GPU KV cache size: [^\n]*"}).items(): print("  " + v)
    print("  " + bench_row(f"{label}: 1 at a time", bench_serve(rate="inf", num_prompts=16, input_len=512, max_concurrency=1)))
    for rate in (4, 8, 16):
        print("  " + bench_row(f"{label}: {rate} req/s", bench_serve(rate=rate, num_prompts=96, input_len=512)))


@app.function(image=image, gpu="L4", timeout=2400, volumes=VOLUMES)
def one_gpu():
    banner("One L4"); proc, ready = serve(["--max-model-len", "4096"]); print(f"  ready after {ready} s"); measure("1 GPU"); stop(proc)


@app.function(image=image, gpu="L4:2", timeout=3000, volumes=VOLUMES)
def two_gpus():
    banner("Two L4s, tensor parallel"); proc, ready = serve(["--max-model-len", "4096", "--tensor-parallel-size", "2"]); print(f"  ready after {ready} s"); measure("TP=2"); stop(proc)
    banner("Two L4s, data parallel"); proc, ready = serve(["--max-model-len", "4096", "--data-parallel-size", "2"]); print(f"  ready after {ready} s"); measure("DP=2"); stop(proc)


@app.local_entrypoint()
def main():
    one_gpu.remote()
    two_gpus.remote()
