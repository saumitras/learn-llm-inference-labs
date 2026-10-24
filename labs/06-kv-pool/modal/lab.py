"""Lab 06: the KV pool, the length limit and preemption. Start without a limit and read the refusal; start with one and
read the shelf; then overfill it and watch the preemption counter. Run: modal run labs/06-kv-pool/modal/lab.py"""
import os, sys, time, subprocess, json, re
import modal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "modal"))
from common import *  # noqa: E402,F403
import threading

app = modal.App("llminference-lab06")
NAMES = ["vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc", "vllm:num_preemptions_total"]


@app.function(image=image, gpu="L4", timeout=2400, volumes=VOLUMES)
def run():
    banner("1. Start with no length limit: the model card's 262,144")
    try:
        proc, _ = serve([], timeout_s=600)
        stop(proc); print("started (unexpected)")
    except RuntimeError:
        text = open("/tmp/serve.log").read()
        m = re.search(r"ValueError: [^\n]*", text) or re.search(r"Error[^\n]*KV cache[^\n]*", text)
        print("refused:", (m.group(0) if m else text[-600:])[:400])
    banner("2. Start with --max-model-len 4096: the shelf")
    proc, ready = serve(["--max-model-len", "4096", "--max-num-seqs", "64", "--no-enable-prefix-caching"]); print(f"ready after {ready} s")   # prefix caching off, or 48 identical prompts share one copy
    for k, v in log_lines("/tmp/serve.log", {"kv_memory": r"Available KV cache memory: [^\n]*", "kv_size": r"GPU KV cache size: [^\n]*"}).items():
        print(f"  {v}")
    print("  arithmetic: 11.8 GiB ÷ 147,456 bytes per token ≈ 86,000 tokens; 4,096 per request → 21 at once")
    banner("3. Overfill it: 48 requests of ~3,000 prompt tokens plus 600 output, admitted 64 at a time")
    prompt = ("The quick brown fox jumps over the lazy dog. " * 340)
    done = []
    def one(i):
        first, gaps, _ = stream_chat([{"role": "user", "content": f"Request {i}. " + prompt + f" Question {i}: continue the story."}], max_tokens=600, temperature=0, ignore_eos=True)
        done.append((first, max(gaps) if gaps else 0))
    ths = [threading.Thread(target=one, args=(i,)) for i in range(48)]
    for t in ths: t.start()
    print("second  running  waiting  kv%   preemptions")
    for s in range(1, 41):
        time.sleep(1)
        m = metrics(NAMES)
        print(f"{s:>6}  {int(m.get(NAMES[0], 0)):>7}  {int(m.get(NAMES[1], 0)):>7}  {100 * m.get(NAMES[2], 0):4.0f}  {int(m.get(NAMES[3], 0)):>11}")
        if s % 5: continue
    for t in ths: t.join()
    worst = max(g for _, g in done)
    print(f"\nworst gap between two tokens in any stream: {worst * 1000:.0f} ms (a preempted request pauses, then resumes)")
    stop(proc)


@app.local_entrypoint()
def main():
    run.remote()
