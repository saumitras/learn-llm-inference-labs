"""Lab 14: cold start, the metrics endpoint, and the thinking switch. Run: modal run labs/14-cold-start-and-metrics/modal/lab.py"""
import os, sys, time, subprocess, json, re
import modal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "modal"))
from common import *  # noqa: E402,F403
import threading

app = modal.App("llminference-lab14")
HYBRID = "Qwen/Qwen3-4B"
NAMES = ["vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc", "vllm:num_preemptions_total", "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total", "vllm:generation_tokens_total"]


@app.function(image=image, gpu="L4", timeout=3000, volumes=VOLUMES)
def run():
    banner("1. Cold start: default, then with compilation off, then default again (the compile cache)")
    for label, extra in (("default", []), ("--enforce-eager", ["--enforce-eager"]), ("default, second start", [])):
        proc, ready = serve(["--max-model-len", "4096", *extra]); ll = log_lines("/tmp/serve.log", STARTUP_PATTERNS)
        print(f"  {label:<22} ready after {ready:6.1f} s   weights {ll.get('weights', '')[:30]}   {ll.get('graphs', 'no graph capture')[:45]}")
        if label != "default, second start": stop(proc)
    banner("2. The metrics endpoint, idle and during a burst of forty")
    print("  idle:", {k.split(':')[1]: v for k, v in metrics(NAMES).items()})
    sysmsg = "You are the support assistant for a small online shop. " * 20
    def one(i):
        chat([{"role": "system", "content": sysmsg}, {"role": "user", "content": f"Question {i}: tell me about your returns policy in detail."}], max_tokens=300, temperature=0)
    ts = [threading.Thread(target=one, args=(i,)) for i in range(40)]
    [t.start() for t in ts]
    print("  second running waiting kv%  preempt  prefix hit%  gen tokens")
    g0 = metrics(NAMES).get(NAMES[6], 0)
    for s in range(1, 11):
        time.sleep(1); m = metrics(NAMES)
        print(f"  {s:>6} {int(m.get(NAMES[0], 0)):>7} {int(m.get(NAMES[1], 0)):>7} {100 * m.get(NAMES[2], 0):4.0f} {int(m.get(NAMES[3], 0)):>8} {100 * m.get(NAMES[5], 0) / max(1, m.get(NAMES[4], 1)):10.0f}% {int(m.get(NAMES[6], 0) - g0):>10}")
    [t.join() for t in ts]
    stop(proc)
    banner("3. The thinking switch on the hybrid Qwen3-4B")
    proc, ready = serve(["--max-model-len", "8192"], model=HYBRID); print(f"  ready after {ready} s")
    q = "A shop sells pens at 3 for 2 dollars. How much do 12 pens cost? Answer briefly."
    for think in (False, True):
        t0 = time.time(); r = chat([{"role": "user", "content": q}], model=HYBRID, max_tokens=1500, temperature=0, chat_template_kwargs={"enable_thinking": think}); took = time.time() - t0
        u = r["usage"]; txt = r["choices"][0]["message"]["content"]
        print(f"  thinking {str(think):<5}: {u['completion_tokens']:5d} completion tokens in {took:5.1f} s   answer: {txt.strip().replace(chr(10), ' ')[-90:]}")
    stop(proc)


@app.local_entrypoint()
def main():
    run.remote()
