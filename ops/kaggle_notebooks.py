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

STREAM = r"""def stream_chat(messages, **params):
    body = json.dumps({"model": MODEL, "messages": messages, "stream": True, **params}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; last = None; gaps = []; text = []
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            delta = json.loads(line[5:])["choices"][0].get("delta", {}).get("content")
            if not delta: continue
            now = time.time()
            if first is None: first = now - t0
            else: gaps.append(now - last)
            last = now; text.append(delta)
    return first, gaps, "".join(text)

def metrics(names):
    out = {}
    for line in urllib.request.urlopen(URL + "/metrics", timeout=5).read().decode().splitlines():
        if line.startswith("#"): continue
        for n in names:
            if line.startswith(n + "{") or line.startswith(n + " "):
                try: out[n] = float(line.rsplit(" ", 1)[1])
                except ValueError: pass
    return out
print("streaming and metrics helpers ready")"""

LAB02 = [
    ("markdown", "# Lab 02: the memory wall\n\nWatch the GPU read its weights once per token, and see why the utilisation figure says nothing. Uses the engine in-process rather than as a server."),
    ("code", CHECK), ("code", INSTALL),
    ("code", r"""# Cell 3: load the model once and measure the decode step at batch 1 and batch 32. On the T4 the weights are fp16: the same 8.04 GB.
import time, subprocess, threading
from vllm import LLM, SamplingParams
MODEL = "%s"; WEIGHT_BYTES = 8.04e9; SPEC_GBPS = 320   # the T4's memory bandwidth
llm = LLM(MODEL, dtype="half", max_model_len=4096, gpu_memory_utilization=0.9, enable_prefix_caching=False, seed=0)
prompts = [f"Write a short paragraph about the number {i}." for i in range(32)]
llm.generate(prompts[:1], SamplingParams(max_tokens=8, temperature=0, ignore_eos=True))
for bs in (1, 32):
    ps = prompts[:bs]
    t1 = time.time(); llm.generate(ps, SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)); t_one = time.time() - t1
    t1 = time.time(); llm.generate(ps, SamplingParams(max_tokens=257, temperature=0, ignore_eos=True)); t_many = time.time() - t1
    step = (t_many - t_one) / 256
    print(f"batch {bs}: decode step {step * 1000:.1f} ms, {bs / step:.0f} tokens/s; implied read rate {WEIGHT_BYTES / step / 1e9:.0f} GB/s of {SPEC_GBPS}; arithmetic {2 * 4.02e9 * bs / step / 1e12:.1f} TFLOP/s of 65")""" % MODEL),
    ("code", r"""# Cell 4: what nvidia-smi says during a long batch-1 decode.
def dmon(seconds):
    f = open("/kaggle/working/dmon.txt", "w"); p = subprocess.Popen(["nvidia-smi", "dmon", "-s", "um", "-d", "1"], stdout=f, stderr=subprocess.STDOUT)
    time.sleep(seconds); p.terminate(); p.wait(); f.close()
    rows = []
    for line in open("/kaggle/working/dmon.txt"):
        parts = line.split()
        if parts and parts[0] != "#":
            try: rows.append((float(parts[1]), float(parts[2])))
            except (ValueError, IndexError): pass
    return rows[1:]
rows = []; t = threading.Thread(target=lambda: rows.extend(dmon(8))); t.start(); time.sleep(1.0)
llm.generate(prompts[:1], SamplingParams(max_tokens=200, temperature=0, ignore_eos=True)); t.join()
print("second  sm%  mem%")
for i, (sm, mem) in enumerate(rows[:8]): print(f"{i + 1:>6}  {sm:4.0f}  {mem:4.0f}")"""),
]

LAB03 = [
    ("markdown", "# Lab 03: the overhead floor\n\nBatch-1 step time with kernels launched one by one and with the launch sequence replayed, on the 4B model and the 0.6B model, beside the bytes-over-bandwidth prediction. Each configuration loads its own engine; run the cells one at a time and restart the kernel between them if memory is not freed."),
    ("code", CHECK), ("code", INSTALL),
    ("code", r"""# Cell 3: a measuring function. Restart the kernel (Run → Restart) between configurations if the GPU memory is not released.
import time, gc, torch
from vllm import LLM, SamplingParams
SPEC_GBPS = 320
def measure(model, eager, weight_gb):
    llm = LLM(model, dtype="half", max_model_len=4096, gpu_memory_utilization=0.9, enforce_eager=eager, enable_prefix_caching=False, seed=0)
    p = ["Write a short paragraph about the number 7."]
    llm.generate(p, SamplingParams(max_tokens=8, temperature=0, ignore_eos=True))
    t1 = time.time(); llm.generate(p, SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)); t_one = time.time() - t1
    t1 = time.time(); llm.generate(p, SamplingParams(max_tokens=257, temperature=0, ignore_eos=True)); t_many = time.time() - t1
    step = (t_many - t_one) / 256 * 1000; pred = weight_gb / SPEC_GBPS * 1000
    print(f"{model:<26} {'one by one' if eager else 'replayed':<12} predicted {pred:5.1f} ms  measured {step:5.1f} ms  floor {step - pred:5.1f} ms")
    del llm; gc.collect(); torch.cuda.empty_cache()"""),
    ("code", 'measure("%s", True, 8.04)' % MODEL),
    ("code", 'measure("%s", False, 8.04)' % MODEL),
    ("code", 'measure("Qwen/Qwen3-0.6B", True, 1.50)'),
    ("code", 'measure("Qwen/Qwen3-0.6B", False, 1.50)'),
]

LAB04 = [
    ("markdown", "# Lab 04: continuous batching and the knee\n\nA rate sweep with the running and waiting counts read from the server, then the same sweep with the batch capped."),
    ("code", CHECK), ("code", INSTALL), ("code", HELPERS), ("code", STREAM),
    ("code", "# Cell 5: the benchmark call and the sweep.\nimport threading\n" + BENCH.replace("def bench(", "def bench_quiet(").replace("    print(text[i:] if i >= 0 else text[-3000:])\n    if out.returncode != 0: print(out.stderr[-2000:])", "    res = {}\n    for line in text.splitlines():\n        m = re.match(r'^(.*?):\\s+([0-9.]+)\\s*$', line)\n        if m: res[m.group(1).strip()] = float(m.group(2))\n    return res") + r"""
NAMES = ["vllm:num_requests_running", "vllm:num_requests_waiting"]
def sweep(rates, num_prompts=60):
    for rate in rates:
        peak = {"running": 0, "waiting": 0}; stop_flag = threading.Event()
        def watch():
            while not stop_flag.is_set():
                try:
                    m = metrics(NAMES); peak["running"] = max(peak["running"], int(m.get(NAMES[0], 0))); peak["waiting"] = max(peak["waiting"], int(m.get(NAMES[1], 0)))
                except Exception: pass
                time.sleep(0.5)
        th = threading.Thread(target=watch); th.start()
        r = bench_quiet(rate=rate, num_prompts=num_prompts, input_len=512, output_len=128); stop_flag.set(); th.join()
        g = lambda k: r.get(k, float("nan"))
        print(f"{rate:>3} req/s offered  req/s {g('Request throughput (req/s)'):5.2f}  out tok/s {g('Output token throughput (tok/s)'):7.1f}  TTFT p50 {g('P50 TTFT (ms)'):6.0f} p99 {g('P99 TTFT (ms)'):6.0f} ms  TPOT p50 {g('P50 TPOT (ms)'):5.1f} ms  peak running {peak['running']:3d} waiting {peak['waiting']:3d}")"""),
    ("code", '# Cell 6: the sweep.\nproc, ready_s = serve(["--max-model-len", "4096"]); print(f"ready after {ready_s} s")\nsweep([1, 2, 4, 6, 8, 12])\nfor line in re.findall(r"Running: \\d+ reqs, Waiting: \\d+ reqs[^\\n]*", open("/kaggle/working/serve.log").read())[-4:]: print("  " + line[:150])\nstop(proc)'),
    ("code", '# Cell 7: the same sweep with the batch capped at 8.\nproc, ready_s = serve(["--max-model-len", "4096", "--max-num-seqs", "8"]); print(f"ready after {ready_s} s")\nsweep([1, 2, 4, 6])\nstop(proc)'),
]

LAB05 = [
    ("markdown", "# Lab 05: chunked prefill\n\nTwenty streaming chats, an 8,000-token prompt dropped in, and the largest gap in anyone's stream, with three token budgets per step."),
    ("code", CHECK), ("code", INSTALL), ("code", HELPERS), ("code", STREAM),
    ("code", r"""# Cell 5: one trial: twenty chats, then the long prompt at 2.5 s.
import threading
def trial(label):
    sysmsg = "You are a patient assistant who answers at length. "; long_prompt = ("The quick brown fox jumps over the lazy dog. " * 900)[:32000]
    results = {}
    def chat_stream(i):
        first, gaps, _ = stream_chat([{"role": "system", "content": sysmsg}, {"role": "user", "content": f"Tell me a long story about the number {i}."}], max_tokens=200, temperature=0, ignore_eos=True)
        results[i] = (first, gaps)
    ths = [threading.Thread(target=chat_stream, args=(i,)) for i in range(20)]
    for t in ths: t.start()
    time.sleep(2.5)
    first_long, _, _ = stream_chat([{"role": "user", "content": long_prompt + "\n\nSummarise the above in one sentence."}], max_tokens=20, temperature=0, ignore_eos=True)
    for t in ths: t.join()
    allg = sorted(g for _, gs in results.values() for g in gs)
    print(f"{label:<28} chats: median gap {allg[len(allg) // 2] * 1000:5.0f} ms, worst gap {allg[-1] * 1000:5.0f} ms   long prompt: first token {first_long * 1000:5.0f} ms")"""),
    ("code", '# Cell 6: three budgets. Each start takes a few minutes.\nfor budget in (16384, 2048, 512):\n    proc, _ = serve(["--max-model-len", "12288", "--max-num-batched-tokens", str(budget), "--max-num-seqs", "64"])\n    trial(f"budget {budget} tokens/step")\n    stop(proc)'),
]

LAB06 = [
    ("markdown", "# Lab 06: the KV pool, the length limit and preemption\n\nStart without a limit and read the refusal; start with one and read the shelf; then overfill it and watch the preemption counter."),
    ("code", CHECK), ("code", INSTALL), ("code", HELPERS), ("code", STREAM),
    ("code", r"""# Cell 5: start with no length limit: the model card's 262,144.
try:
    proc, _ = serve([], timeout_s=600); stop(proc); print("started (unexpected)")
except RuntimeError:
    text = open("/kaggle/working/serve.log").read(); m = re.search(r"ValueError: [^\n]*", text)
    print("refused:", (m.group(0) if m else text[-600:])[:400])"""),
    ("code", r"""# Cell 6: start with --max-model-len 4096 and read the shelf.
proc, ready_s = serve(["--max-model-len", "4096", "--max-num-seqs", "64", "--no-enable-prefix-caching"]); print(f"ready after {ready_s} s")   # prefix caching off, or 48 identical prompts share one copy
for k, v in log_lines("/kaggle/working/serve.log", {"kv_memory": r"Available KV cache memory: [^\n]*", "kv_size": r"GPU KV cache size: [^\n]*"}).items(): print("  " + v)"""),
    ("code", r"""# Cell 7: overfill it: 48 requests of ~3,000 prompt tokens plus 600 output.
import threading
NAMES = ["vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc", "vllm:num_preemptions_total"]
prompt = ("The quick brown fox jumps over the lazy dog. " * 340); done = []
def one(i):
    first, gaps, _ = stream_chat([{"role": "user", "content": f"Request {i}. " + prompt + f" Question {i}: continue the story."}], max_tokens=600, temperature=0, ignore_eos=True); done.append((first, max(gaps) if gaps else 0))
ths = [threading.Thread(target=one, args=(i,)) for i in range(48)]
for t in ths: t.start()
print("second  running  waiting  kv%   preemptions")
for s in range(1, 41):
    time.sleep(1); m = metrics(NAMES)
    print(f"{s:>6}  {int(m.get(NAMES[0], 0)):>7}  {int(m.get(NAMES[1], 0)):>7}  {100 * m.get(NAMES[2], 0):4.0f}  {int(m.get(NAMES[3], 0)):>11}")
for t in ths: t.join()
print(f"worst gap between two tokens in any stream: {max(g for _, g in done) * 1000:.0f} ms")
stop(proc)"""),
]

LAB07 = [
    ("markdown", "# Lab 07: prefix caching and cache hygiene\n\nCold and warm first token; twenty questions on one system prompt; a timestamp at the top and at the bottom; a sweep over the shared-prefix fraction."),
    ("code", CHECK), ("code", INSTALL), ("code", HELPERS), ("code", STREAM),
    ("code", r"""# Cell 5: cold and warm.
NAMES = ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total"]
def hits():
    m = metrics(NAMES); return m.get(NAMES[0], 0), m.get(NAMES[1], 0)
proc, ready_s = serve(["--max-model-len", "8192"]); print(f"ready after {ready_s} s")
document = "".join("Paragraph %d of the policy: items may be returned within thirty days in their original packaging with proof of purchase. " % i for i in range(60))
for label in ("cold", "warm"):
    q0, h0 = hits()
    first, _, _ = stream_chat([{"role": "user", "content": document + "\n\nIn one sentence, how long do I have to return an item?"}], max_tokens=30, temperature=0)
    q1, h1 = hits(); print(f"  {label}: first token {first * 1000:5.0f} ms   prefix blocks queried {int(q1 - q0):4d}, hit {int(h1 - h0):4d}")"""),
    ("code", r"""# Cell 6: twenty questions on one system prompt, then with a timestamp at the top and at the bottom.
system = "You are the support assistant for a small online shop. " + document[:2400]
q0, h0 = hits(); firsts = []
for i in range(20):
    first, _, _ = stream_chat([{"role": "system", "content": system}, {"role": "user", "content": f"Question {i}: do you ship to country number {i}?"}], max_tokens=20, temperature=0); firsts.append(first)
q1, h1 = hits(); print(f"  hit rate {100 * (h1 - h0) / max(1, q1 - q0):.0f}%   first token: first request {firsts[0] * 1000:.0f} ms, median of the rest {sorted(firsts[1:])[9] * 1000:.0f} ms")
for where in ("top", "bottom"):
    q0, h0 = hits()
    for i in range(20):
        stamp = f"Current time: 2026-10-24T12:00:{i:02d}Z. "; sysm = (stamp + system) if where == "top" else (system + " " + stamp)
        stream_chat([{"role": "system", "content": sysm}, {"role": "user", "content": f"Question {i}: do you ship to country number {i}?"}], max_tokens=20, temperature=0)
    q1, h1 = hits(); print(f"  timestamp at the {where:<6}: hit rate {100 * (h1 - h0) / max(1, q1 - q0):3.0f}%")"""),
    ("code", "# Cell 7: throughput against the shared-prefix fraction.\n" + BENCH.replace("def bench(", "def bench_quiet(").replace('"--ignore-eos", "--seed", "0"]', '"--ignore-eos", "--seed", "0", "--random-prefix-len", str(prefix_len)]').replace("def bench_quiet(rate, num_prompts, input_len=1024, output_len=128, max_concurrency=None):", "def bench_quiet(rate, num_prompts, input_len=1024, output_len=128, max_concurrency=None, prefix_len=0):").replace("    print(text[i:] if i >= 0 else text[-3000:])\n    if out.returncode != 0: print(out.stderr[-2000:])", "    res = {}\n    for line in text.splitlines():\n        m = re.match(r'^(.*?):\\s+([0-9.]+)\\s*$', line)\n        if m: res[m.group(1).strip()] = float(m.group(2))\n    return res") + r"""
for prefix in (0, 256, 512, 768):
    r = bench_quiet(rate=4, num_prompts=48, input_len=1024 - prefix, output_len=128, prefix_len=prefix); g = lambda k: r.get(k, float("nan"))
    print(f"shared {prefix:>4} of 1,024: out tok/s {g('Output token throughput (tok/s)'):7.1f}  TTFT p50 {g('P50 TTFT (ms)'):6.0f} p99 {g('P99 TTFT (ms)'):6.0f} ms")
stop(proc)"""),
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
    write("labs/02-memory-wall/kaggle/lab.ipynb", LAB02)
    write("labs/03-overhead-floor/kaggle/lab.ipynb", LAB03)
    write("labs/04-continuous-batching/kaggle/lab.ipynb", LAB04)
    write("labs/05-chunked-prefill/kaggle/lab.ipynb", LAB05)
    write("labs/06-kv-pool/kaggle/lab.ipynb", LAB06)
    write("labs/07-prefix-caching/kaggle/lab.ipynb", LAB07)
    print("wrote 8 notebooks")
