"""Lab 05: chunked prefill. Twenty streaming chats, an 8,000-token prompt dropped in, and the largest gap in anyone's
stream, with three token budgets per step. Run: modal run labs/05-chunked-prefill/modal/lab.py"""
import os, sys, time, subprocess, json, re
import modal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "modal"))
from common import *  # noqa: E402,F403
import threading

app = modal.App("llminference-lab05")


def trial(budget_label):
    sysmsg = "You are a patient assistant who answers at length. "
    long_prompt = ("The quick brown fox jumps over the lazy dog. " * 900)[:32000]   # about 8,000 tokens
    results = {}
    def chat_stream(i):
        first, gaps, _ = stream_chat([{"role": "system", "content": sysmsg}, {"role": "user", "content": f"Tell me a long story about the number {i}."}], max_tokens=200, temperature=0, ignore_eos=True)
        results[i] = (first, gaps)
    ths = [threading.Thread(target=chat_stream, args=(i,)) for i in range(20)]
    for t in ths: t.start()
    time.sleep(2.5)
    t0 = time.time()
    first_long, gaps_long, _ = stream_chat([{"role": "user", "content": long_prompt + "\n\nSummarise the above in one sentence."}], max_tokens=20, temperature=0, ignore_eos=True)
    for t in ths: t.join()
    worst = max(max(g) for _, g in results.values())
    median_gap = sorted(g for _, gs in results.values() for g in gs)[len([g for _, gs in results.values() for g in gs]) // 2]
    print(f"{budget_label:<28} chats: median gap {median_gap * 1000:5.0f} ms, worst gap {worst * 1000:5.0f} ms   long prompt: first token {first_long * 1000:5.0f} ms")


@app.function(image=image, gpu="L4", timeout=2400, volumes=VOLUMES)
def run():
    banner("Twenty streaming chats; at 2.5 s an 8,000-token prompt arrives")
    for budget in (16384, 2048, 512):
        proc, ready = serve(["--max-model-len", "12288", "--max-num-batched-tokens", str(budget), "--max-num-seqs", "64"])
        trial(f"budget {budget} tokens/step")
        stop(proc)


@app.local_entrypoint()
def main():
    run.remote()
