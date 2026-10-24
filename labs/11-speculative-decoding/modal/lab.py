"""Lab 11: speculative decoding, three ways. n-gram, Qwen3-0.6B as a draft model, an EAGLE3 head; acceptance from the
counters; batch 1 against batch 32. Run: modal run labs/11-speculative-decoding/modal/lab.py"""
import os, sys, time, subprocess, json, re
import modal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "modal"))
from common import *  # noqa: E402,F403

app = modal.App("llminference-lab11")
CONFIGS = [
    ("none", []),
    ("n-gram", ["--speculative-config", json.dumps({"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 5, "prompt_lookup_min": 2})]),
    ("draft model 0.6B", ["--speculative-config", json.dumps({"model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 5})]),
    ("EAGLE3 head", ["--speculative-config", json.dumps({"method": "eagle3", "model": "taobao-mnn/Qwen3-4B-Instruct-2507-Eagle3", "num_speculative_tokens": 3})]),
]
CODE = "Write a Python function that parses a CSV file into a list of dictionaries, with a docstring and type hints, and a second function that writes it back."
PROSE = "Write a short, original poem about a city at night, without rhyme."


def spec_metrics():
    out = {}
    import urllib.request
    for line in urllib.request.urlopen(URL + "/metrics", timeout=5).read().decode().splitlines():
        if line.startswith("vllm:spec_decode") and not line.startswith("#"):
            name, val = line.rsplit(" ", 1); out[name.split("{")[0]] = float(val)
    return out


@app.function(image=image, gpu="L4", timeout=3000, volumes=VOLUMES)
def run():
    for label, extra in CONFIGS:
        banner(label)
        try:
            proc, ready = serve(["--max-model-len", "4096", *extra]); print(f"ready after {ready} s")
        except RuntimeError as e:
            print("could not start:", str(e)[-500:]); continue
        for name, prompt in (("code", CODE), ("prose", PROSE)):
            t0 = time.time(); first, gaps, text = stream_chat([{"role": "user", "content": prompt}], max_tokens=200, temperature=0, ignore_eos=True); took = time.time() - t0
            print(f"  batch 1, {name:<5}: {len(gaps) + 1} tokens in {took:4.1f} s = {(len(gaps) + 1) / took:5.1f} tok/s, time per token {1000 * sum(gaps) / max(1, len(gaps)):5.1f} ms")
        m = spec_metrics()
        if m:
            d = m.get("vllm:spec_decode_num_draft_tokens_total", 0); a = m.get("vllm:spec_decode_num_accepted_tokens_total", 0)
            print(f"  drafted {int(d)} tokens, accepted {int(a)}: acceptance {100 * a / max(1, d):.0f}%")
        print("  " + bench_row("32 concurrent, 512 in", bench_serve(rate="inf", num_prompts=64, input_len=512, output_len=128, max_concurrency=32)))
        stop(proc)


@app.local_entrypoint()
def main():
    run.remote()
