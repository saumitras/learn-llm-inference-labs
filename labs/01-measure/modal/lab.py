"""Lab 01: measure time to first token, time per output token and throughput.

Uses the engine's own benchmark tool against the running server: one request at a time first, then a rate below
the knee, then one above it. Run: modal run labs/01-measure/modal/lab.py
"""
import os
import subprocess
import sys

import modal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "modal"))
from common import MODEL, PORT, VOLUMES, banner, image, serve, stop  # noqa: E402

app = modal.App("llminference-lab01")


def bench(rate, num_prompts, input_len=1024, output_len=128, max_concurrency=None):
    args = [
        "vllm", "bench", "serve", "--model", MODEL, "--port", str(PORT), "--dataset-name", "random",
        "--random-input-len", str(input_len), "--random-output-len", str(output_len), "--num-prompts", str(num_prompts),
        "--request-rate", str(rate), "--metric-percentiles", "50,95,99", "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--ignore-eos", "--seed", "0",
    ]
    if max_concurrency:
        args += ["--max-concurrency", str(max_concurrency)]
    out = subprocess.run(args, capture_output=True, text=True)
    text = out.stdout
    start = text.find("Serving Benchmark Result")
    print(text[start:] if start >= 0 else text[-3000:])
    if out.returncode != 0:
        print(out.stderr[-2000:])


@app.function(image=image, gpu="L4", timeout=1800, volumes=VOLUMES)
def run():
    proc, ready_s = serve(["--max-model-len", "4096"])
    print(f"server ready after {ready_s} s\n")

    banner("1. One request at a time: 16 prompts of 1,024 tokens, 128-token answers")
    bench(rate="inf", num_prompts=16, max_concurrency=1)

    banner("2. A short prompt, one at a time: the first-token time with almost no prefill")
    bench(rate="inf", num_prompts=16, input_len=64, max_concurrency=1)

    banner("3. Two requests a second: below the knee")
    bench(rate=2, num_prompts=48)

    banner("4. Eight requests a second: past the knee")
    bench(rate=8, num_prompts=96)

    stop(proc)


@app.local_entrypoint()
def main():
    run.remote()
