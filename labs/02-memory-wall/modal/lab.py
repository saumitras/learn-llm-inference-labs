"""Lab 02: the memory wall. Watch the GPU read its weights once per token, and see why the utilisation figure says nothing.
Run: modal run labs/02-memory-wall/modal/lab.py"""
import os, sys, time, subprocess, json, re
import modal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "modal"))
from common import *  # noqa: E402,F403

app = modal.App("llminference-lab02")
WEIGHT_BYTES = 8.04e9
SPEC_GBPS = 300


def dmon(seconds):
    f = open("/tmp/dmon.txt", "w")
    p = subprocess.Popen(["nvidia-smi", "dmon", "-s", "um", "-d", "1"], stdout=f, stderr=subprocess.STDOUT)
    time.sleep(seconds)
    p.terminate(); p.wait(); f.close()
    rows = []
    for line in open("/tmp/dmon.txt"):
        parts = line.split()
        if parts and parts[0] != "#":
            try: rows.append((float(parts[1]), float(parts[2])))
            except (ValueError, IndexError): pass
    rows = rows[1:] if len(rows) > 2 else rows
    return rows


@app.function(image=image, gpu="L4", timeout=1800, volumes=VOLUMES)
def run():
    from vllm import LLM, SamplingParams
    banner("1. Load the model once")
    llm = LLM(MODEL, max_model_len=4096, gpu_memory_utilization=0.9, enable_prefix_caching=False, seed=0)
    prompts = [f"Write a short paragraph about the number {i}." for i in range(32)]
    llm.generate(prompts[:1], SamplingParams(max_tokens=8, temperature=0, ignore_eos=True))
    for bs in (1, 32):
        banner(f"{2 if bs == 1 else 3}. Decode at batch {bs}: step time and what it implies")
        ps = prompts[:bs]
        t1 = time.time(); llm.generate(ps, SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)); t_one = time.time() - t1
        t1 = time.time(); llm.generate(ps, SamplingParams(max_tokens=257, temperature=0, ignore_eos=True)); t_many = time.time() - t1
        step = (t_many - t_one) / 256
        print(f"decode step: {step * 1000:.1f} ms   tokens/s: {bs / step:.0f}")
        print(f"weights read per step: {WEIGHT_BYTES / 1e9:.2f} GB  ->  implied read rate {WEIGHT_BYTES / step / 1e9:.0f} GB/s  (spec sheet {SPEC_GBPS} GB/s: {WEIGHT_BYTES / step / 1e9 / SPEC_GBPS * 100:.0f}%)")
        print(f"arithmetic per step: {2 * 4.02e9 * bs / 1e12:.3f} TFLOP  ->  {2 * 4.02e9 * bs / step / 1e12:.1f} TFLOP/s of 121 available ({2 * 4.02e9 * bs / step / 1e12 / 121 * 100:.1f}%)")
    banner("4. What nvidia-smi says during a long batch-1 decode")
    import threading
    rows = []
    t = threading.Thread(target=lambda: rows.extend(dmon(8)))
    t.start(); time.sleep(1.0)
    llm.generate(prompts[:1], SamplingParams(max_tokens=200, temperature=0, ignore_eos=True))
    t.join()
    print("second  sm%  mem%")
    for i, (sm, mem) in enumerate(rows[:8]):
        print(f"{i + 1:>6}  {sm:4.0f}  {mem:4.0f}")
    print("sm% is the share of time any kernel was running; mem% the share of time the memory controller was busy. Both say 'busy'; neither says which wall.")
    hf_cache.commit()


@app.local_entrypoint()
def main():
    run.remote()
