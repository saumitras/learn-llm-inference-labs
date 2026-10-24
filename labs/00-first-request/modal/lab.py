"""Lab 00: install, first request, what you downloaded.

Serves the course model, sends one request with the OpenAI client, reads the startup lines that every later lab
refers to, and lists what landed on disk. Run: modal run labs/00-first-request/modal/lab.py
"""
import os
import sys

import modal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "modal"))
from common import MODEL, STARTUP_PATTERNS, VOLUMES, banner, chat, image, log_lines, serve, stop  # noqa: E402

app = modal.App("llminference-lab00")


@app.function(image=image, gpu="L4", timeout=1800, volumes=VOLUMES)
def run():
    import subprocess
    banner("1. The GPU this lab got")
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
    import vllm, torch
    print(f"vllm {vllm.__version__} · torch {torch.__version__} · compute capability {torch.cuda.get_device_capability()}")

    banner("2. Start the server")
    print(f"vllm serve {MODEL} --max-model-len 4096")
    proc, ready_s = serve(["--max-model-len", "4096"])
    print(f"ready after {ready_s} s")
    for name, line in log_lines("/tmp/serve.log", STARTUP_PATTERNS).items():
        print(f"  {name:<10} {line}")

    banner("3. One request")
    r = chat([{"role": "user", "content": "In one sentence, what does an inference server do?"}], max_tokens=60, temperature=0)
    print(r["choices"][0]["message"]["content"].strip())
    u = r["usage"]
    print(f"\nprompt tokens {u['prompt_tokens']} · completion tokens {u['completion_tokens']}")

    banner("4. The same request, with the model's confidence per token")
    r = chat([{"role": "user", "content": "The capital of France is"}], max_tokens=5, temperature=0, logprobs=True, top_logprobs=3)
    for tok in r["choices"][0]["logprobs"]["content"]:
        alts = ", ".join(f"{a['token']!r} {100 * 2.718281828 ** a['logprob']:.0f}%" for a in tok["top_logprobs"])
        print(f"  {tok['token']!r:<12} chosen with {100 * 2.718281828 ** tok['logprob']:.0f}%   alternatives: {alts}")

    banner("5. What landed on disk")
    base = "/root/.cache/huggingface/hub/models--" + MODEL.replace("/", "--") + "/snapshots"
    snap = os.path.join(base, os.listdir(base)[0])
    total = 0
    for f in sorted(os.listdir(snap)):
        size = os.path.getsize(os.path.join(snap, f))
        total += size
        print(f"  {f:<40} {size / 1e6:9.1f} MB")
    print(f"  {'total':<40} {total / 1e9:9.2f} GB")

    stop(proc)


@app.local_entrypoint()
def main():
    run.remote()
