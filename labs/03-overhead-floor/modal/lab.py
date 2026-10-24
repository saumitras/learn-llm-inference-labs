"""Lab 03: the overhead floor. Batch-1 step time with kernels launched one by one and with the launch sequence replayed,
on the 4B model and on the 0.6B model, beside the bytes-over-bandwidth prediction. Run: modal run labs/03-overhead-floor/modal/lab.py"""
import os, sys, time, subprocess, json, re
import modal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "modal"))
from common import *  # noqa: E402,F403

app = modal.App("llminference-lab03")
SPEC_GBPS = 300


@app.function(image=image, gpu="L4", timeout=1800, volumes=VOLUMES)
def measure(model: str, eager: bool, weight_gb: float):
    from vllm import LLM, SamplingParams
    llm = LLM(model, max_model_len=4096, gpu_memory_utilization=0.9, enforce_eager=eager, enable_prefix_caching=False, seed=0)
    p = ["Write a short paragraph about the number 7."]
    llm.generate(p, SamplingParams(max_tokens=8, temperature=0, ignore_eos=True))
    t1 = time.time(); llm.generate(p, SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)); t_one = time.time() - t1
    t1 = time.time(); llm.generate(p, SamplingParams(max_tokens=257, temperature=0, ignore_eos=True)); t_many = time.time() - t1
    step_ms = (t_many - t_one) / 256 * 1000
    hf_cache.commit()
    return {"model": model, "eager": eager, "step_ms": round(step_ms, 1), "predicted_ms": round(weight_gb / SPEC_GBPS * 1000, 1)}


@app.local_entrypoint()
def main():
    jobs = [(MODEL, True, 8.04), (MODEL, False, 8.04), ("Qwen/Qwen3-0.6B", True, 1.50), ("Qwen/Qwen3-0.6B", False, 1.50)]
    results = list(measure.starmap(jobs))
    banner("Batch-1 decode step on an L4: predicted (weights ÷ 300 GB/s) beside measured")
    print(f"{'model':<26} {'launches':<14} {'predicted':>10} {'measured':>10} {'floor':>8}")
    for r in results:
        print(f"{r['model']:<26} {'one by one' if r['eager'] else 'replayed':<14} {r['predicted_ms']:>8.1f} ms {r['step_ms']:>8.1f} ms {r['step_ms'] - r['predicted_ms']:>6.1f} ms")
    print("\nfloor = measured minus predicted: the per-step cost that does not depend on how many bytes are read")
