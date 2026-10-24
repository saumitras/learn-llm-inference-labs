"""Lab 07: prefix caching and cache hygiene. Cold and warm first token; twenty questions on one system prompt; a
timestamp at the top and at the bottom; a sweep over the shared-prefix fraction. Run: modal run labs/07-prefix-caching/modal/lab.py"""
import os, sys, time, subprocess, json, re
import modal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "modal"))
from common import *  # noqa: E402,F403

app = modal.App("llminference-lab07")
NAMES = ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total"]


def hits():
    m = metrics(NAMES); return m.get(NAMES[0], 0), m.get(NAMES[1], 0)


@app.function(image=image, gpu="L4", timeout=2400, volumes=VOLUMES)
def run():
    proc, ready = serve(["--max-model-len", "8192"]); print(f"server ready after {ready} s")
    doc = ("Paragraph %d of the policy: items may be returned within thirty days in their original packaging with proof of purchase. " * 1)
    document = "".join(doc % i for i in range(60))   # about 2,000 tokens
    banner("1. The same 2,000-token prompt, cold and then warm")
    for label in ("cold", "warm"):
        q0, h0 = hits()
        first, _, _ = stream_chat([{"role": "user", "content": document + "\n\nIn one sentence, how long do I have to return an item?"}], max_tokens=30, temperature=0)
        q1, h1 = hits()
        print(f"  {label}: first token {first * 1000:5.0f} ms   prefix blocks queried {int(q1 - q0):4d}, hit {int(h1 - h0):4d}")
    banner("2. Twenty questions on one 600-token system prompt")
    system = "You are the support assistant for a small online shop. " + document[:2400]
    q0, h0 = hits(); firsts = []
    for i in range(20):
        first, _, _ = stream_chat([{"role": "system", "content": system}, {"role": "user", "content": f"Question {i}: do you ship to country number {i}?"}], max_tokens=20, temperature=0)
        firsts.append(first)
    q1, h1 = hits()
    print(f"  queried {int(q1 - q0)} blocks, hit {int(h1 - h0)}: hit rate {100 * (h1 - h0) / max(1, q1 - q0):.0f}%   first token: first request {firsts[0] * 1000:.0f} ms, median of the rest {sorted(firsts[1:])[9] * 1000:.0f} ms")
    banner("3. The same twenty, with a timestamp at the top of the system prompt, then at the bottom")
    for where in ("top", "bottom"):
        q0, h0 = hits()
        for i in range(20):
            stamp = f"Current time: 2026-10-24T12:00:{i:02d}Z. "
            sysm = (stamp + system) if where == "top" else (system + " " + stamp)
            stream_chat([{"role": "system", "content": sysm}, {"role": "user", "content": f"Question {i}: do you ship to country number {i}?"}], max_tokens=20, temperature=0)
        q1, h1 = hits()
        print(f"  timestamp at the {where:<6}: hit rate {100 * (h1 - h0) / max(1, q1 - q0):3.0f}%")
    banner("4. Throughput against the shared-prefix fraction (1,024-token prompts, 4 req/s, 48 prompts)")
    for prefix in (0, 256, 512, 768):
        r = bench_serve(rate=4, num_prompts=48, input_len=1024 - prefix, output_len=128, prefix_len=prefix)
        print(bench_row(f"shared {prefix:>4} of 1,024 ({100 * prefix // 1024:2d}%)", r))
    stop(proc)


@app.local_entrypoint()
def main():
    run.remote()
