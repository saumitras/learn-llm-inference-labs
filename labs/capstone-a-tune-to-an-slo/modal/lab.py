"""Capstone A: tune to an SLO. The support bot's traffic shape and targets; a rate sweep on each configuration until
the p99s cross the targets; the GPUs the fleet needs; a quality gate on the chosen configuration.
Run: modal run labs/capstone-a-tune-to-an-slo/modal/lab.py"""
import os, sys, time, subprocess, json, re, threading
import modal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "modal"))
from common import *  # noqa: E402,F403

app = modal.App("llminference-capstone-a")
eval_image = base_image.pip_install("lm_eval[api]==0.4.13").add_local_python_source("common")
FP8 = "Qwen/Qwen3-4B-Instruct-2507-FP8"
# The support bot from the capacity module: a 600-token system prompt shared by everyone, about 700 more tokens of
# history and question, answers of about 300 tokens, 500 users at one message a minute = 8.3 requests a second.
PREFIX, INPUT, OUTPUT, NEEDED_RATE = 600, 1300, 300, 8.3
TTFT_P99_MS, TPOT_P99_MS = 1000, 75
RATES = (2, 4, 6, 8, 10, 12)
CONFIGS = [
    ("BF16, defaults", MODEL, []),
    ("FP8 weights", FP8, []),
    ("FP8 weights + FP8 KV cache", FP8, ["--kv-cache-dtype", "fp8"]),
    ("FP8 + FP8 KV, max-num-seqs 32", FP8, ["--kv-cache-dtype", "fp8", "--max-num-seqs", "32"]),
]


def lm_eval(model_name, limit=100):
    args = ["lm_eval", "--model", "local-completions", "--model_args", f"model={model_name},base_url={URL}/v1/completions,num_concurrent=8,max_retries=2,tokenized_requests=False",
            "--tasks", "hellaswag", "--limit", str(limit), "--batch_size", "8"]
    out = subprocess.run(args, capture_output=True, text=True)
    acc = re.findall(r"acc_norm\s*\|[^|]*\|\s*([0-9.]+)", out.stdout + out.stderr)
    return acc[0] if acc else "?"


@app.function(image=eval_image, gpu="L4", timeout=3600, volumes=VOLUMES)
def run():
    banner(f"Target: TTFT p99 <= {TTFT_P99_MS} ms and TPOT p99 <= {TPOT_P99_MS} ms, prompts {INPUT} tokens ({PREFIX} shared), answers {OUTPUT} tokens, {NEEDED_RATE} req/s needed")
    best = {}
    for label, model, extra in CONFIGS:
        banner(label)
        proc, ready = serve(["--max-model-len", "4096", *extra], model=model); print(f"  ready after {ready} s   " + log_lines("/tmp/serve.log", {"kv": r"GPU KV cache size: [0-9,]+ tokens"}).get("kv", ""))
        top = 0.0
        for rate in RATES:
            r = bench_serve(rate=rate, num_prompts=72, input_len=INPUT - PREFIX, output_len=OUTPUT, prefix_len=PREFIX, model=model)
            ok = r.get("P99 TTFT (ms)", 1e9) <= TTFT_P99_MS and r.get("P99 TPOT (ms)", 1e9) <= TPOT_P99_MS
            print(("  OK   " if ok else "  MISS ") + bench_row(f"{rate} req/s", r))
            if ok: top = r.get("Request throughput (req/s)", 0.0)
            elif top: break
        best[label] = top
        print(f"  highest rate inside the target: {top:.2f} req/s achieved  ->  GPUs for {NEEDED_RATE} req/s: {NEEDED_RATE / top if top else float('inf'):.1f}")
        stop(proc)
    banner("Summary")
    for label, top in best.items():
        import math
        print(f"  {label:<34} {top:5.2f} req/s per GPU inside the target   fleet for 500 users: {math.ceil(NEEDED_RATE / top) if top else '-'} GPUs")
    banner("Quality gate on the chosen configuration: lm_eval HellaSwag, 100 examples, BF16 then FP8")
    for label, model in (("BF16", MODEL), ("FP8", FP8)):
        proc, _ = serve(["--max-model-len", "4096"], model=model)
        print(f"  {label}: acc_norm {lm_eval(model)}")
        stop(proc)


@app.local_entrypoint()
def main():
    run.remote()
