"""Generates the Kaggle notebooks from the same steps as the Modal scripts. Run: python ops/kaggle_notebooks.py"""
import json

MODEL = "Qwen/Qwen3-4B-Instruct-2507"

CHECK = r'''# Cell 1: the GPU this notebook got. On Kaggle: Settings → Accelerator → GPU T4 x2, then Run all.
import subprocess, sys
out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True)
if out.returncode != 0 or not out.stdout.strip():
    sys.exit("No GPU. Open Settings → Accelerator and choose GPU T4 x2, then run this cell again.")
print(out.stdout.strip())
if "T4" in out.stdout:
    print("Path: Kaggle T4. Labs 08 and 10 are read-only on this GPU (no FP8 arithmetic).")'''

INSTALL = r'''# Cell 2: install the pinned engine. About nine minutes on Kaggle. Kaggle's preinstalled torchaudio is built
# for an older CUDA and breaks the import, so it is removed.
%pip install -q "vllm==0.32.0" 2>&1 | tail -2
%pip uninstall -y -q torchaudio
import vllm, torch
print("vllm", vllm.__version__, "· torch", torch.__version__, "· compute capability", torch.cuda.get_device_capability())'''

HELPERS = r'''# Cell 3: helpers used by every lab: start the server in the background, wait for it, talk to it.
import json, re, subprocess, time, urllib.request
MODEL = "%s"
URL = "http://127.0.0.1:8000"

def serve(extra_args=(), log_path="/kaggle/working/serve.log", timeout_s=900):
    args = ["vllm", "serve", MODEL, "--port", "8000", "--dtype", "half", *extra_args]   # the T4 has no bfloat16
    log = open(log_path, "w")
    t0 = time.time()
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError("vllm serve exited; see " + log_path)
        try:
            urllib.request.urlopen(URL + "/health", timeout=2); return proc, round(time.time() - t0, 1)
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("server not ready in time")

def stop(proc):
    proc.terminate(); proc.wait()

def log_lines(path, patterns):
    text = open(path).read(); out = {}
    for name, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            line = re.sub(r"^\(EngineCore[^)]*\) ", "", m.group(0)); line = re.sub(r"^INFO \d\d-\d\d \d\d:\d\d:\d\d \[[^\]]+\] ", "", line); out[name] = line.strip()
    return out

STARTUP = {"dtype": r"INFO[^\n]*dtype=torch\.[a-z0-9]+", "backend": r"Using [A-Z_]+ attention backend[^\n]*", "weights": r"Loading weights took [^\n]*",
           "model": r"Model loading took [^\n]*", "kv_memory": r"Available KV cache memory: [^\n]*", "kv_size": r"GPU KV cache size: [^\n]*", "graphs": r"Graph capturing finished[^\n]*"}

def chat(messages, **params):
    body = json.dumps({"model": MODEL, "messages": messages, **params}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())
print("helpers ready")''' % MODEL

LAB00 = [
    ("markdown", "# Lab 00: install, first request, what you downloaded\n\nServe the course model, send one request, read the startup lines every later lab refers to, and list what landed on disk. Run the cells in order; the install cell takes about nine minutes on Kaggle."),
    ("code", CHECK), ("code", INSTALL), ("code", HELPERS),
    ("code", r'''# Cell 4: start the server and read what it printed.
proc, ready_s = serve(["--max-model-len", "4096"])
print(f"ready after {ready_s} s")
for name, line in log_lines("/kaggle/working/serve.log", STARTUP).items():
    print(f"  {name:<10} {line}")'''),
    ("code", r'''# Cell 5: one request.
r = chat([{"role": "user", "content": "In one sentence, what does an inference server do?"}], max_tokens=60, temperature=0)
print(r["choices"][0]["message"]["content"].strip())
u = r["usage"]; print(f"\nprompt tokens {u['prompt_tokens']} · completion tokens {u['completion_tokens']}")'''),
    ("code", r'''# Cell 6: the same request, with the model's confidence per token.
import math
r = chat([{"role": "user", "content": "The capital of France is"}], max_tokens=5, temperature=0, logprobs=True, top_logprobs=3)
for tok in r["choices"][0]["logprobs"]["content"]:
    alts = ", ".join(f"{a['token']!r} {100 * math.exp(a['logprob']):.0f}%" for a in tok["top_logprobs"])
    print(f"  {tok['token']!r:<12} chosen with {100 * math.exp(tok['logprob']):.0f}%   alternatives: {alts}")'''),
    ("code", r'''# Cell 7: what landed on disk.
import os
base = os.path.expanduser("~/.cache/huggingface/hub/models--" + MODEL.replace("/", "--") + "/snapshots")
snap = os.path.join(base, os.listdir(base)[0]); total = 0
for f in sorted(os.listdir(snap)):
    size = os.path.getsize(os.path.join(snap, f)); total += size
    print(f"  {f:<40} {size / 1e6:9.1f} MB")
print(f"  {'total':<40} {total / 1e9:9.2f} GB")
stop(proc)'''),
]

BENCH = r'''def bench(rate, num_prompts, input_len=1024, output_len=128, max_concurrency=None):
    args = ["vllm", "bench", "serve", "--model", MODEL, "--port", "8000", "--dataset-name", "random", "--random-input-len", str(input_len),
            "--random-output-len", str(output_len), "--num-prompts", str(num_prompts), "--request-rate", str(rate),
            "--metric-percentiles", "50,95,99", "--percentile-metrics", "ttft,tpot,itl,e2el", "--ignore-eos", "--seed", "0"]
    if max_concurrency: args += ["--max-concurrency", str(max_concurrency)]
    out = subprocess.run(args, capture_output=True, text=True); text = out.stdout; i = text.find("Serving Benchmark Result")
    print(text[i:] if i >= 0 else text[-3000:])
    if out.returncode != 0: print(out.stderr[-2000:])'''

LAB01 = [
    ("markdown", "# Lab 01: measure time to first token, time per output token and throughput\n\nThe engine's own benchmark tool against the running server: one request at a time, then a rate below the knee, then one above it. Read the p50, p95 and p99 columns, not the mean."),
    ("code", CHECK), ("code", INSTALL), ("code", HELPERS),
    ("code", "# Cell 4: start the server and define the benchmark call.\nproc, ready_s = serve([\"--max-model-len\", \"4096\"])\nprint(f\"ready after {ready_s} s\")\n" + BENCH),
    ("code", '# Cell 5: one request at a time, 16 prompts of 1,024 tokens, 128-token answers.\nbench(rate="inf", num_prompts=16, max_concurrency=1)'),
    ("code", '# Cell 6: a short prompt, one at a time: the first-token time with almost no prefill.\nbench(rate="inf", num_prompts=16, input_len=64, max_concurrency=1)'),
    ("code", '# Cell 7: two requests a second: below the knee.\nbench(rate=2, num_prompts=48)'),
    ("code", '# Cell 8: eight requests a second: past the knee.\nbench(rate=8, num_prompts=96)\nstop(proc)'),
]


def write(path, cells):
    nb = {"cells": [{"cell_type": t, "metadata": {}, "source": src.splitlines(keepends=True), **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, src in cells],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}, "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)


if __name__ == "__main__":
    write("labs/00-first-request/kaggle/lab.ipynb", LAB00)
    write("labs/01-measure/kaggle/lab.ipynb", LAB01)
    print("wrote 2 notebooks")
