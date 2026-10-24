"""Lab 10: the FP8 KV cache. Half the bytes per column: what it does to the shelf and to the step at long context.
Run: modal run labs/10-fp8-kv-cache/modal/lab.py"""
import os, sys, time, subprocess, json, re
import modal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "modal"))
from common import *  # noqa: E402,F403

app = modal.App("llminference-lab10")


@app.function(image=image, gpu="L4", timeout=2400, volumes=VOLUMES)
def run():
    for label, extra in (("BF16 cache", []), ("FP8 cache", ["--kv-cache-dtype", "fp8"])):
        banner(label)
        proc, ready = serve(["--max-model-len", "8192", *extra]); print(f"ready after {ready} s")
        for k, v in log_lines("/tmp/serve.log", {"kv_memory": r"Available KV cache memory: [^\n]*", "kv_size": r"GPU KV cache size: [^\n]*"}).items():
            print("  " + v)
        print(bench_row("1 at a time, 512 in", bench_serve(rate="inf", num_prompts=16, input_len=512, max_concurrency=1)))
        print(bench_row("1 at a time, 6,000 in", bench_serve(rate="inf", num_prompts=8, input_len=6000, output_len=256, max_concurrency=1)))
        print(bench_row("32 concurrent, 2,048 in", bench_serve(rate="inf", num_prompts=64, input_len=2048, max_concurrency=32)))
        r = chat([{"role": "user", "content": "Explain in two sentences why the sky is blue."}], max_tokens=60, temperature=0)
        print("  sample:", r["choices"][0]["message"]["content"].strip()[:160])
        stop(proc)


@app.local_entrypoint()
def main():
    run.remote()
