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
for s in range(1, 181):   # the T4 admits these prompts slowly, so watch for three minutes and print every fifth second
    time.sleep(1); m = metrics(NAMES)
    if s % 5 == 0: print(f"{s:>6}  {int(m.get(NAMES[0], 0)):>7}  {int(m.get(NAMES[1], 0)):>7}  {100 * m.get(NAMES[2], 0):4.0f}  {int(m.get(NAMES[3], 0)):>11}")
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

BENCH_QUIET = BENCH.replace("def bench(", "def bench_quiet(").replace("def bench_quiet(rate, num_prompts, input_len=1024, output_len=128, max_concurrency=None):", "def bench_quiet(rate, num_prompts, input_len=1024, output_len=128, max_concurrency=None, model=None):").replace('["vllm", "bench", "serve", "--model", MODEL,', '["vllm", "bench", "serve", "--model", model or MODEL, "--tokenizer", MODEL,').replace("    print(text[i:] if i >= 0 else text[-3000:])\n    if out.returncode != 0: print(out.stderr[-2000:])", "    res = {}\n    for line in text.splitlines():\n        m = re.match(r'^(.*?):\\s+([0-9.]+)\\s*$', line)\n        if m: res[m.group(1).strip()] = float(m.group(2))\n    return res") + r"""
def row(label, r):
    g = lambda k: r.get(k, float("nan"))
    return f"{label:<28} req/s {g('Request throughput (req/s)'):5.2f}  out tok/s {g('Output token throughput (tok/s)'):7.1f}  TTFT p50 {g('P50 TTFT (ms)'):6.0f} p99 {g('P99 TTFT (ms)'):6.0f} ms  TPOT p50 {g('P50 TPOT (ms)'):5.1f} p99 {g('P99 TPOT (ms)'):5.1f} ms  peak conc {g('Maximum request concurrency'):3.0f}"
"""

LOAD = r"""import random, threading
def random_prompt(n_tokens, seed):
    rng = random.Random(seed); words = ["river", "stone", "quiet", "engine", "orange", "ledger", "window", "copper", "signal", "harbor", "meadow", "lantern", "velvet", "summit", "cinder", "fable"]
    return " ".join(rng.choice(words) for _ in range(int(n_tokens * 0.8)))

def load_test(rate, num_prompts, prompt_len=512, out_len=128, prefix_len=0, model=None, concurrency=None):
    rng = random.Random(1); prefix = random_prompt(prefix_len, 99) + " " if prefix_len else ""
    prompts = [prefix + random_prompt(prompt_len - prefix_len, i) for i in range(num_prompts)]; results = []; lock = threading.Lock()
    def one(i):
        first, gaps, text = stream_chat([{"role": "user", "content": prompts[i]}], **({"model": model} if model else {}), max_tokens=out_len, temperature=0, ignore_eos=True)
        with lock: results.append((first, (sum(gaps) / len(gaps)) if gaps else 0.0, len(gaps) + 1))
    sem = threading.Semaphore(concurrency or num_prompts); t0 = time.time(); threads = []
    def gated(i):
        with sem: one(i)
    for i in range(num_prompts):
        th = threading.Thread(target=gated, args=(i,)); th.start(); threads.append(th)
        if not concurrency: time.sleep(rng.expovariate(rate))
    for th in threads: th.join()
    dur = time.time() - t0; ttft = sorted(r[0] for r in results); tpot = sorted(r[1] for r in results); q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"req_s": round(len(results) / dur, 2), "out_tok_s": round(sum(r[2] for r in results) / dur, 1), "ttft_p50_ms": round(1000 * q(ttft, 0.5)), "ttft_p99_ms": round(1000 * q(ttft, 0.99)), "tpot_p50_ms": round(1000 * q(tpot, 0.5), 1), "tpot_p99_ms": round(1000 * q(tpot, 0.99), 1)}

def load_row(label, r):
    return f"{label:<30} req/s {r['req_s']:5.2f}  out tok/s {r['out_tok_s']:7.1f}  TTFT p50 {r['ttft_p50_ms']:6d} p99 {r['ttft_p99_ms']:6d} ms  TPOT p50 {r['tpot_p50_ms']:5.1f} p99 {r['tpot_p99_ms']:5.1f} ms"
print("load generator ready")"""

STREAM_MODEL = STREAM.replace('body = json.dumps({"model": MODEL, "messages": messages, "stream": True, **params}).encode()', 'body = json.dumps({"model": params.pop("model", MODEL), "messages": messages, "stream": True, **params}).encode()')

LAB08 = [
    ("markdown", "# Lab 08: FP8 weights\n\nThe official FP8 checkpoint on the same rate sweep as the BF16 one. **On the Kaggle T4 this lab is read-only for the FP8 half**: the T4 has no FP8 arithmetic and the engine refuses the checkpoint. The notebook shows the refusal, then runs the fp16 half of the sweep so you have your own baseline against the L4 numbers on the lab page."),
    ("code", CHECK), ("code", INSTALL), ("code", HELPERS),
    ("code", "# Cell 4: the benchmark call.\n" + BENCH_QUIET),
    ("code", r"""# Cell 5: ask the T4 to serve the FP8 checkpoint. If the engine accepts it, run the sweep on it; if not, read why.
FP8 = "Qwen/Qwen3-4B-Instruct-2507-FP8"
MODEL_SAVED = MODEL; MODEL = FP8
try:
    proc, ready_s = serve(["--max-model-len", "4096"]); print(f"ready after {ready_s} s (this GPU accepted the FP8 checkpoint: the engine unpacks the weights to fp16 for the arithmetic)")
    print("  " + log_lines("/kaggle/working/serve.log", {"q": r"quantization=[a-z0-9_]+"}).get("q", ""))
    print(row("FP8: 1 at a time, 1,024 in", bench_quiet(rate="inf", num_prompts=16, input_len=1024, max_concurrency=1)))
    for rate in (1, 2, 4):
        print(row(f"FP8: {rate} req/s, 512 in", bench_quiet(rate=rate, num_prompts=40, input_len=512)))
    stop(proc)
except RuntimeError as e:
    text = open("/kaggle/working/serve.log").read()
    for line in text.splitlines():
        if "fp8" in line.lower() and ("error" in line.lower() or "support" in line.lower() or "capability" in line.lower()): print("  " + line[-200:])
    print("  refused:", e)
MODEL = MODEL_SAVED"""),
    ("code", r"""# Cell 6: the fp16 half of the sweep, for your own baseline.
proc, ready_s = serve(["--max-model-len", "4096"]); print(f"ready after {ready_s} s")
print(row("1 at a time, 1,024 in", bench_quiet(rate="inf", num_prompts=16, input_len=1024, max_concurrency=1)))
for rate in (1, 2, 4):
    print(row(f"{rate} req/s, 512 in", bench_quiet(rate=rate, num_prompts=40, input_len=512)))
stop(proc)"""),
]

LAB09 = [
    ("markdown", "# Lab 09: INT4, quantised by you\n\nQuantise Qwen3-0.6B to W4A16 with LLM Compressor, score the original and the result with lm_eval against the running server, then measure the prefill penalty. The quantisation step runs on the T4; expect ten to twenty minutes for it."),
    ("code", CHECK), ("code", INSTALL),
    ("code", '# Cell 3: the quantisation and evaluation tools.\n%pip install -q "llmcompressor==0.14.0.1" "lm_eval[api]==0.4.13" 2>&1 | tail -2\nimport llmcompressor, lm_eval; print("llmcompressor", llmcompressor.__version__, "· lm_eval", lm_eval.__version__)'),
    ("code", HELPERS.replace('MODEL = "%s"' % MODEL, 'MODEL = "Qwen/Qwen3-0.6B"')),
    ("code", "# Cell 5: the benchmark call.\n" + BENCH_QUIET),
    ("code", r"""# Cell 6: quantise. GPTQ, 4-bit weights with 16-bit activations, 128 calibration samples from wikitext.
import os, time, torch
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
OUT = "/kaggle/working/Qwen3-0.6B-W4A16"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cuda", dtype=torch.float16)
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train").filter(lambda x: len(x["text"]) > 400).shuffle(seed=42).select(range(128))
ds = ds.map(lambda x: tok(x["text"], max_length=2048, truncation=True, add_special_tokens=False), remove_columns=ds.column_names)
t0 = time.time()
oneshot(model=model, tokenizer=tok, dataset=ds, recipe=GPTQModifier(scheme="W4A16", targets="Linear", ignore=["lm_head"]), output_dir=OUT, max_seq_length=2048, num_calibration_samples=128, pipeline="basic")
size = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.endswith(".safetensors"))
print(f"took {time.time() - t0:.0f} s; checkpoint {size / 1e9:.2f} GB against 1.50 GB")
del model; torch.cuda.empty_cache()"""),
    ("code", r"""# Cell 7: serve each one, score it on 100 HellaSwag examples, measure it.
def lm_eval_run(model_name, limit=100):
    args = ["lm_eval", "--model", "local-completions", "--model_args", f"model={model_name},base_url={URL}/v1/completions,num_concurrent=8,max_retries=2,tokenized_requests=False", "--tasks", "hellaswag", "--limit", str(limit), "--batch_size", "8"]
    out = subprocess.run(args, capture_output=True, text=True); text = out.stdout + out.stderr
    acc = re.findall(r"acc_norm\s*\|[^|]*\|\s*([0-9.]+)", text); return acc[0] if acc else text[-600:]
for label, model in (("fp16 original", "Qwen/Qwen3-0.6B"), ("W4A16 quantised", OUT)):
    MODEL = model
    proc, ready_s = serve(["--max-model-len", "4096", "--gpu-memory-utilization", "0.8"]); print(f"{label}: ready after {ready_s} s")   # the notebook process keeps about 2 GiB after quantising, so the server asks for less
    print("  " + log_lines("/kaggle/working/serve.log", {"q": r"quantization=[a-z0-9_]+"}).get("q", "quantization=none"))
    print(f"  hellaswag acc_norm {lm_eval_run(model)}")
    print("  " + row("1 at a time, 1,024 in", bench_quiet(rate="inf", num_prompts=16, input_len=1024, max_concurrency=1, model=model)))
    print("  " + row("32 concurrent, 1,024 in", bench_quiet(rate="inf", num_prompts=64, input_len=1024, max_concurrency=32, model=model)))
    stop(proc)
MODEL = "Qwen/Qwen3-0.6B"
"""),
]

LAB10 = [
    ("markdown", "# Lab 10: the FP8 KV cache\n\nHalf the bytes per column. **On the Kaggle T4 this lab is read-only**: the FP8 KV cache needs compute capability 8.9 and the engine refuses it on the T4. The notebook shows the refusal, then reads the fp16 cache size and runs the long-context step so you can compare with the L4 numbers on the lab page."),
    ("code", CHECK), ("code", INSTALL), ("code", HELPERS),
    ("code", "# Cell 4: the benchmark call.\n" + BENCH_QUIET),
    ("code", r"""# Cell 5: ask for the FP8 cache and read the refusal.
try:
    proc, ready_s = serve(["--max-model-len", "8192", "--kv-cache-dtype", "fp8"]); print(f"ready after {ready_s} s (this GPU accepted the FP8 cache)"); stop(proc)
except RuntimeError as e:
    for line in open("/kaggle/working/serve.log").read().splitlines():
        if "fp8" in line.lower() and ("error" in line.lower() or "support" in line.lower() or "capability" in line.lower()): print("  " + line[-200:])
    print("  refused:", e)"""),
    ("code", r"""# Cell 6: the fp16 cache: its size, and the step at short and long context.
proc, ready_s = serve(["--max-model-len", "8192"]); print(f"ready after {ready_s} s")
for k, v in log_lines("/kaggle/working/serve.log", {"kv_memory": r"Available KV cache memory: [^\n]*", "kv_size": r"GPU KV cache size: [^\n]*"}).items(): print("  " + v)
print(row("1 at a time, 512 in", bench_quiet(rate="inf", num_prompts=16, input_len=512, max_concurrency=1)))
print(row("1 at a time, 6,000 in", bench_quiet(rate="inf", num_prompts=8, input_len=6000, output_len=256, max_concurrency=1)))
print(row("16 concurrent, 2,048 in", bench_quiet(rate="inf", num_prompts=32, input_len=2048, max_concurrency=16)))
stop(proc)"""),
]

LAB11 = [
    ("markdown", "# Lab 11: speculative decoding, three ways\n\nn-gram lookup, Qwen3-0.6B as a draft model, and an EAGLE3 head, each against the plain server: time per token at batch 1, the acceptance rate from the counters, and what happens at batch 32. Each configuration restarts the server, so the whole notebook takes a while. The EAGLE3 head may not load on the T4; the cell says so if it does not."),
    ("code", CHECK), ("code", INSTALL), ("code", HELPERS), ("code", STREAM),
    ("code", "# Cell 5: the benchmark call.\n" + BENCH_QUIET),
    ("code", r"""# Cell 6: four servers in turn.
CONFIGS = [("none", []),
    ("n-gram", ["--speculative-config", json.dumps({"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 5, "prompt_lookup_min": 2})]),
    ("draft model 0.6B", ["--speculative-config", json.dumps({"model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 5})]),
    ("EAGLE3 head", ["--speculative-config", json.dumps({"method": "eagle3", "model": "taobao-mnn/Qwen3-4B-Instruct-2507-Eagle3", "num_speculative_tokens": 3})])]
CODE = "Write a Python function that parses a CSV file into a list of dictionaries, with a docstring and type hints, and a second function that writes it back."
PROSE = "Write a short, original poem about a city at night, without rhyme."
SPEC = ["vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total"]
for label, extra in CONFIGS:
    print(f"\n== {label}")
    try:
        proc, ready_s = serve(["--max-model-len", "4096", *extra]); print(f"  ready after {ready_s} s")
    except RuntimeError as e:
        print("  could not start:", e); continue
    for name, prompt in (("code", CODE), ("prose", PROSE)):
        t0 = time.time(); first, gaps, text = stream_chat([{"role": "user", "content": prompt}], max_tokens=200, temperature=0, ignore_eos=True); took = time.time() - t0
        print(f"  batch 1, {name:<5}: {len(gaps) + 1} tokens in {took:4.1f} s = {(len(gaps) + 1) / took:5.1f} tok/s")
    m = metrics(SPEC)
    if m: print(f"  drafted {int(m.get(SPEC[0], 0))} tokens, accepted {int(m.get(SPEC[1], 0))}: acceptance {100 * m.get(SPEC[1], 0) / max(1, m.get(SPEC[0], 1)):.0f}%")
    print("  " + row("32 concurrent, 512 in", bench_quiet(rate="inf", num_prompts=64, input_len=512, max_concurrency=32)))
    stop(proc)"""),
]

LAB12 = [
    ("markdown", "# Lab 12: structured output\n\nA JSON schema on the request, the first-request grammar cost, what the grammar does to the step, and a failure mode you should see once: the grammar lets the model pad with whitespace."),
    ("code", CHECK), ("code", INSTALL), ("code", HELPERS), ("code", STREAM),
    ("code", r"""# Cell 5: two schemas and the request.
SIMPLE = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "city": {"type": "string"}}, "required": ["name", "age", "city"], "additionalProperties": False}
NESTED = {"type": "object", "properties": {"customer": {"type": "object", "properties": {"name": {"type": "string"}, "email": {"type": "string"}, "address": {"type": "object", "properties": {"street": {"type": "string"}, "city": {"type": "string"}, "postcode": {"type": "string"}}, "required": ["street", "city", "postcode"]}}, "required": ["name", "email", "address"]},
  "items": {"type": "array", "items": {"type": "object", "properties": {"sku": {"type": "string"}, "quantity": {"type": "integer"}, "unit_price": {"type": "number"}}, "required": ["sku", "quantity", "unit_price"]}, "minItems": 2}, "total": {"type": "number"}, "paid": {"type": "boolean"}}, "required": ["customer", "items", "total", "paid"], "additionalProperties": False}
PROMPT = "Invent an order for a small online shop with a customer in Pune and two items, as JSON."
def ask(schema=None, name="s"):
    params = {"max_tokens": 200, "temperature": 0}
    if schema: params["response_format"] = {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}
    first, gaps, text = stream_chat([{"role": "user", "content": PROMPT}], **params); ok = "n/a"
    if schema:
        try: json.loads(text); ok = "valid JSON"
        except Exception: ok = "INVALID: ends with " + repr(text[-30:])
    return first, gaps, text, ok
def rounds(title, schema, name):
    print(title)
    for i in range(2):
        first, gaps, text, ok = ask(schema, name); print(f"  request {i + 1}: first token {1000 * first:.0f} ms, time per token {1000 * sum(gaps) / len(gaps):.1f} ms, {len(gaps) + 1} tokens, {ok}")
    print("  " + text.replace("\n", " ")[:300])"""),
    ("code", r"""# Cell 6: no schema, then the two schemas, twice each.
proc, ready_s = serve(["--max-model-len", "4096"]); print(f"ready after {ready_s} s")
first, gaps, text, _ = ask(); print(f"no schema: first token {1000 * first:.0f} ms, time per token {1000 * sum(gaps) / len(gaps):.1f} ms, {len(gaps) + 1} tokens\n  " + text.replace("\n", " ")[:200])
rounds("simple schema", SIMPLE, "simple"); rounds("nested schema", NESTED, "nested")
stop(proc)"""),
    ("code", r"""# Cell 7: the same with any-whitespace disabled in the grammar.
proc, ready_s = serve(["--max-model-len", "4096", "--structured-outputs-config", json.dumps({"backend": "xgrammar", "disable_any_whitespace": True})]); print(f"ready after {ready_s} s")
rounds("simple schema", SIMPLE, "simple"); rounds("nested schema", NESTED, "nested")
stop(proc)"""),
]

LAB13 = [
    ("markdown", "# Lab 13: many fine-tunes on one base\n\nTwo rank-8 adapters made in the notebook (random weights: this lab measures cost, not quality) behind one server, and what each costs per step."),
    ("code", CHECK), ("code", INSTALL),
    ("code", '# Cell 3: peft, to write the adapters.\n%pip install -q "peft==0.21.2" 2>&1 | tail -1'),
    ("code", HELPERS), ("code", STREAM_MODEL), ("code", "# Cell 6: the load generator and the benchmark call.\n" + LOAD + "\n" + BENCH_QUIET),
    ("code", r"""# Cell 7: two adapters.
import os, torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
for name in ("legal", "support"):
    m = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.0))
    for p in m.parameters():
        if p.requires_grad: torch.nn.init.normal_(p, std=0.02)
    m.save_pretrained(f"/kaggle/working/lora-{name}"); del m
    print(f"  adapter {name}: {sum(os.path.getsize(os.path.join(f'/kaggle/working/lora-{name}', f)) for f in os.listdir(f'/kaggle/working/lora-{name}')) / 1e6:.1f} MB")
del base; torch.cuda.empty_cache()"""),
    ("code", r"""# Cell 8: one server, the base plus both adapters.
proc, ready_s = serve(["--max-model-len", "4096", "--enable-lora", "--max-lora-rank", "8", "--lora-modules", "legal=/kaggle/working/lora-legal", "support=/kaggle/working/lora-support"]); print(f"ready after {ready_s} s")
for label, model in (("base", MODEL), ("adapter legal", "legal"), ("adapter support", "support")):
    print(row(f"{label}, 1 at a time", bench_quiet(rate="inf", num_prompts=16, input_len=512, max_concurrency=1, model=model)))
print(row("base x32", bench_quiet(rate="inf", num_prompts=64, input_len=512, max_concurrency=32, model=MODEL)))
res = {}
def go(name): res[name] = load_test(rate=0, num_prompts=32, prompt_len=512, out_len=128, model=name, concurrency=16)
ts = [threading.Thread(target=go, args=(n,)) for n in ("legal", "support")]; [t.start() for t in ts]; [t.join() for t in ts]
print(f"mixed x32 (16 legal + 16 support)      out tok/s {sum(r['out_tok_s'] for r in res.values()):7.1f}  TPOT p50 {res['legal']['tpot_p50_ms']:5.1f} / {res['support']['tpot_p50_ms']:5.1f} ms")
stop(proc)"""),
]

LAB14 = [
    ("markdown", "# Lab 14: cold start, the metrics endpoint, and the thinking switch\n\nTime to ready with and without compilation; the metrics endpoint idle and under a burst of forty; the hybrid Qwen3-4B with thinking off and on."),
    ("code", CHECK), ("code", INSTALL), ("code", HELPERS), ("code", STREAM),
    ("code", r"""# Cell 5: cold start three ways.
for label, extra in (("default", []), ("--enforce-eager", ["--enforce-eager"]), ("default, second start", [])):
    proc, ready_s = serve(["--max-model-len", "4096", *extra]); ll = log_lines("/kaggle/working/serve.log", STARTUP)
    print(f"  {label:<22} ready after {ready_s:6.1f} s   {ll.get('weights', '')[:30]}   {ll.get('graphs', 'no graph capture')[:45]}")
    if label != "default, second start": stop(proc)"""),
    ("code", r"""# Cell 6: the metrics endpoint, idle and during a burst of forty.
import threading
NAMES = ["vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc", "vllm:num_preemptions_total", "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total", "vllm:generation_tokens_total"]
print("  idle:", {k.split(':')[1]: v for k, v in metrics(NAMES).items()})
sysmsg = "You are the support assistant for a small online shop. " * 20
def one(i): chat([{"role": "system", "content": sysmsg}, {"role": "user", "content": f"Question {i}: tell me about your returns policy in detail."}], max_tokens=300, temperature=0)
ts = [threading.Thread(target=one, args=(i,)) for i in range(40)]; [t.start() for t in ts]
print("  second running waiting kv%  preempt  prefix hit%  gen tokens"); g0 = metrics(NAMES).get(NAMES[6], 0)
for s in range(1, 16):
    time.sleep(1); m = metrics(NAMES)
    print(f"  {s:>6} {int(m.get(NAMES[0], 0)):>7} {int(m.get(NAMES[1], 0)):>7} {100 * m.get(NAMES[2], 0):4.0f} {int(m.get(NAMES[3], 0)):>8} {100 * m.get(NAMES[5], 0) / max(1, m.get(NAMES[4], 1)):10.0f}% {int(m.get(NAMES[6], 0) - g0):>10}")
[t.join() for t in ts]; stop(proc)"""),
    ("code", r"""# Cell 7: the thinking switch on the hybrid Qwen3-4B.
MODEL = "Qwen/Qwen3-4B"
proc, ready_s = serve(["--max-model-len", "8192"]); print(f"ready after {ready_s} s")
q = "A shop sells pens at 3 for 2 dollars. How much do 12 pens cost? Answer briefly."
for think in (False, True):
    t0 = time.time(); r = chat([{"role": "user", "content": q}], max_tokens=1500, temperature=0, chat_template_kwargs={"enable_thinking": think}); took = time.time() - t0
    print(f"  thinking {str(think):<5}: {r['usage']['completion_tokens']:5d} completion tokens in {took:5.1f} s   answer: {r['choices'][0]['message']['content'].strip().replace(chr(10), ' ')[-90:]}")
stop(proc)"""),
]

LAB15 = [
    ("markdown", "# Lab 15: two engines, same traffic\n\nvLLM and SGLang one after the other on random and shared-prefix traffic, with the same load generator so the client is identical. SGLang is installed into the same environment after the vLLM half; if its install conflicts with the engine already installed, the second half will say so."),
    ("code", CHECK), ("code", INSTALL), ("code", HELPERS), ("code", STREAM), ("code", "# Cell 5: the load generator.\n" + LOAD),
    ("code", r"""# Cell 6: vLLM.
def traffic(engine):
    for label, prefix in (("random prompts", 0), ("75% shared prefix", 768)):
        print("  " + load_row(f"{engine}: {label}", load_test(rate=2, num_prompts=32, prompt_len=1024, out_len=128, prefix_len=prefix)))
proc, ready_s = serve(["--max-model-len", "4096"]); print(f"vLLM ready after {ready_s} s"); traffic("vLLM"); stop(proc)"""),
    ("code", r"""# Cell 7: SGLang.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sglang[all]==0.5.22"], check=False)
log = open("/kaggle/working/sglang.log", "w"); t0 = time.time()
proc = subprocess.Popen([sys.executable, "-m", "sglang.launch_server", "--model-path", MODEL, "--port", "8000", "--context-length", "4096", "--dtype", "half"], stdout=log, stderr=subprocess.STDOUT)
while time.time() - t0 < 900 and proc.poll() is None:
    try: urllib.request.urlopen(URL + "/health", timeout=2); break
    except Exception: time.sleep(0.5)
if proc.poll() is not None: print("SGLang did not start; last lines:\n" + "\n".join(open("/kaggle/working/sglang.log").read().splitlines()[-15:]))
else: print(f"SGLang ready after {time.time() - t0:.0f} s"); traffic("SGLang"); proc.terminate(); proc.wait()"""),
]

LAB16 = [
    ("markdown", "# Lab 16: two GPUs\n\nKaggle gives you two T4s. One GPU, then tensor parallel across both, then two data-parallel copies: the step, the shelf and the throughput for each."),
    ("code", CHECK), ("code", INSTALL), ("code", HELPERS),
    ("code", "# Cell 4: the benchmark call and the measurement.\n" + BENCH_QUIET + r"""
def measure(label):
    for k, v in log_lines("/kaggle/working/serve.log", {"kv": r"GPU KV cache size: [^\n]*"}).items(): print("  " + v)
    print("  " + row(f"{label}: 1 at a time", bench_quiet(rate="inf", num_prompts=16, input_len=512, max_concurrency=1)))
    for rate in (2, 4, 8):
        print("  " + row(f"{label}: {rate} req/s", bench_quiet(rate=rate, num_prompts=64, input_len=512)))"""),
    ("code", '# Cell 5: one T4.\nproc, ready_s = serve(["--max-model-len", "4096"]); print(f"ready after {ready_s} s"); measure("1 GPU"); stop(proc)'),
    ("code", '# Cell 6: tensor parallel across both.\nproc, ready_s = serve(["--max-model-len", "4096", "--tensor-parallel-size", "2"]); print(f"ready after {ready_s} s"); measure("TP=2"); stop(proc)'),
    ("code", '# Cell 7: two data-parallel copies.\nproc, ready_s = serve(["--max-model-len", "4096", "--data-parallel-size", "2"]); print(f"ready after {ready_s} s"); measure("DP=2"); stop(proc)'),
]

CAP_A = [
    ("markdown", "# Capstone A: tune to an SLO\n\nThe support bot's traffic shape and targets, a rate sweep on each configuration until the p99s cross the targets, the GPUs the fleet needs, and a quality gate. On the Kaggle T4 the FP8 configurations are read-only (no FP8 arithmetic), so the notebook sweeps the fp16 server and one scheduler knob, then runs the quality gate on it. Budget about forty minutes."),
    ("code", CHECK), ("code", INSTALL),
    ("code", '# Cell 3: the evaluation harness for the quality gate.\n%pip install -q "lm_eval[api]==0.4.13" 2>&1 | tail -1'),
    ("code", HELPERS),
    ("code", "# Cell 5: the benchmark call, the targets and the traffic shape.\n" + BENCH_QUIET.replace("def bench_quiet(rate, num_prompts, input_len=1024, output_len=128, max_concurrency=None, model=None):", "def bench_quiet(rate, num_prompts, input_len=1024, output_len=128, max_concurrency=None, model=None, prefix_len=0):").replace('"--ignore-eos", "--seed", "0"]', '"--ignore-eos", "--seed", "0", "--random-prefix-len", str(prefix_len)]') + r"""
PREFIX, INPUT, OUTPUT, NEEDED_RATE = 600, 1300, 300, 8.3      # the support bot from the capacity module: 500 users, one message a minute
TTFT_P99_MS, TPOT_P99_MS = 1000, 75
def sweep(label, rates=(1, 2, 3, 4, 6, 8)):
    top = 0.0
    for rate in rates:
        r = bench_quiet(rate=rate, num_prompts=48, input_len=INPUT - PREFIX, output_len=OUTPUT, prefix_len=PREFIX)
        ok = r.get("P99 TTFT (ms)", 1e9) <= TTFT_P99_MS and r.get("P99 TPOT (ms)", 1e9) <= TPOT_P99_MS
        print(("  OK   " if ok else "  MISS ") + row(f"{rate} req/s", r))
        if ok: top = r.get("Request throughput (req/s)", 0.0)
        elif top: break
    print(f"  {label}: highest rate inside the target {top:.2f} req/s  ->  GPUs for {NEEDED_RATE} req/s: {NEEDED_RATE / top if top else float('inf'):.1f}")
    return top"""),
    ("code", r"""# Cell 6: fp16, defaults.
proc, ready_s = serve(["--max-model-len", "4096"]); print(f"ready after {ready_s} s"); best = {}
best["fp16, defaults"] = sweep("fp16, defaults"); stop(proc)"""),
    ("code", r"""# Cell 7: the same with max-num-seqs capped at 16: the step stays short, the queue takes the overflow.
proc, ready_s = serve(["--max-model-len", "4096", "--max-num-seqs", "16"]); print(f"ready after {ready_s} s")
best["fp16, max-num-seqs 16"] = sweep("fp16, max-num-seqs 16"); stop(proc)"""),
    ("code", r"""# Cell 8: the FP8 configurations are read-only on the T4: the engine refuses the checkpoint. Read the L4 rows on the capstone page.
import math
for label, top in best.items():
    print(f"  {label:<26} {top:5.2f} req/s per GPU inside the target   fleet for 500 users: {math.ceil(NEEDED_RATE / top) if top else '-'} GPUs")"""),
    ("code", r"""# Cell 9: the quality gate: lm_eval on 100 HellaSwag examples against the running server.
proc, ready_s = serve(["--max-model-len", "4096"]); print(f"ready after {ready_s} s")
args = ["lm_eval", "--model", "local-completions", "--model_args", f"model={MODEL},base_url={URL}/v1/completions,num_concurrent=8,max_retries=2,tokenized_requests=False", "--tasks", "hellaswag", "--limit", "100", "--batch_size", "8"]
out = subprocess.run(args, capture_output=True, text=True); acc = re.findall(r"acc_norm\s*\|[^|]*\|\s*([0-9.]+)", out.stdout + out.stderr)
print("  fp16 acc_norm", acc[0] if acc else (out.stdout + out.stderr)[-600:]); stop(proc)"""),
]

CAP_B = [
    ("markdown", "# Capstone B: engineer the cache\n\nA multi-turn support agent on one engine with a careless prompt layout, then a careful one; then two engines, one per T4, behind a router that is round-robin, then session-sticky. Kaggle's two T4s make the two-engine half run as written."),
    ("code", CHECK), ("code", INSTALL), ("code", HELPERS.replace('def serve(extra_args=(), log_path="/kaggle/working/serve.log", timeout_s=900):', 'def serve(extra_args=(), log_path="/kaggle/working/serve.log", timeout_s=900, port=8000, env=None):').replace('args = ["vllm", "serve", MODEL, "--port", "8000", "--dtype", "half", *extra_args]', 'import os\n    args = ["vllm", "serve", MODEL, "--port", str(port), "--dtype", "half", *extra_args]').replace('proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)', 'proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})})').replace('urllib.request.urlopen(URL + "/health", timeout=2); return proc', 'urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); return proc')),
    ("code", STREAM.replace('def stream_chat(messages, **params):', 'def stream_chat(messages, url=URL, **params):').replace('req = urllib.request.Request(URL + "/v1/chat/completions"', 'req = urllib.request.Request(url + "/v1/chat/completions"').replace('def metrics(names):', 'def metrics(names, url=URL):').replace('urllib.request.urlopen(URL + "/metrics", timeout=5)', 'urllib.request.urlopen(url + "/metrics", timeout=5)')),
    ("code", r"""# Cell 5: the agent: a policy, a tool list, tool results that grow turn by turn, and two layouts of the same prompt.
import hashlib, random, threading
NAMES = ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total"]
POLICY = "You are the support assistant for a small online shop. " + " ".join(f"Policy clause {i}: items may be returned within thirty days in their original packaging with proof of purchase, and refunds reach the original payment method within five working days." for i in range(22))
TOOLS = "Tools you may call: " + " ".join(f"tool_{i}(order_id) returns the {w} of an order." for i, w in enumerate(["status", "items", "shipping address", "carrier", "tracking events", "invoice", "refund state", "notes"] * 2))
TURNS = 6
def tool_result(session, turn):
    rng = random.Random(session * 100 + turn)
    return "Tool result: " + " ".join(f"order {rng.randint(1000, 9999)} line {i}: {rng.choice(['shipped', 'packed', 'delayed', 'delivered'])} via carrier {rng.randint(1, 9)};" for i in range(14))
def session_messages(session, turn, layout):
    stamp = f"Session {session}, current time 2026-10-24T12:{turn:02d}:00Z."
    tools_so_far = " ".join(tool_result(session, t) for t in range(turn))
    system = (stamp + " " + tools_so_far + " " + POLICY + " " + TOOLS) if layout == "careless" else (POLICY + " " + TOOLS + " " + tools_so_far + " " + stamp)
    history = []
    for t in range(turn):
        history += [{"role": "user", "content": f"Question {t} from session {session}: where is my order and when does it arrive?"}, {"role": "assistant", "content": f"Answer {t}: " + "your order is on its way and should arrive within two working days; " * 6}]
    return [{"role": "system", "content": system}, *history, {"role": "user", "content": f"Question {turn} from session {session}: where is my order and when does it arrive?"}]
def hits(url=URL):
    m = metrics(NAMES, url=url); return m.get(NAMES[0], 0), m.get(NAMES[1], 0)
def run_sessions(n_sessions, layout, route, urls):
    firsts = {}; lock = threading.Lock(); before = [hits(u) for u in urls]; t0 = time.time()
    def one(session):
        for turn in range(TURNS):
            first, _, _ = stream_chat(session_messages(session, turn, layout), url=urls[route(session, turn, len(urls))], max_tokens=24, temperature=0)
            with lock: firsts.setdefault(turn, []).append(first)
    ths = [threading.Thread(target=one, args=(s,)) for s in range(n_sessions)]; [t.start() for t in ths]; [t.join() for t in ths]
    took = time.time() - t0; after = [hits(u) for u in urls]
    q = sum(a[0] - b[0] for a, b in zip(after, before)); h = sum(a[1] - b[1] for a, b in zip(after, before)); allf = sorted(f for fs in firsts.values() for f in fs)
    per_turn = " ".join(f"t{t}:{1000 * sorted(firsts[t])[len(firsts[t]) // 2]:4.0f}" for t in range(TURNS))
    print(f"  hit rate {100 * h / max(1, q):3.0f}%   TTFT p50 {1000 * allf[len(allf) // 2]:5.0f} p99 {1000 * allf[int(0.99 * (len(allf) - 1))]:5.0f} ms   per-turn p50 ms {per_turn}   {took:.0f} s")
round_robin = lambda s, t, n: (s * TURNS + t) % n
sticky = lambda s, t, n: int(hashlib.md5(str(s).encode()).hexdigest(), 16) % n"""),
    ("code", r"""# Cell 6: one engine on GPU 0: careless, then careful, then 48 sessions.
proc, ready_s = serve(["--max-model-len", "8192"], env={"CUDA_VISIBLE_DEVICES": "0"}); print(f"ready after {ready_s} s")
for layout in ("careless", "careful"):
    print(f"{layout}:"); run_sessions(24, layout, round_robin, [URL])
print("careful, 48 sessions:"); run_sessions(48, "careful", round_robin, [URL])"""),
    ("code", r"""# Cell 7: a second engine on GPU 1; 48 sessions behind round-robin, then session-sticky.
proc2, ready2 = serve(["--max-model-len", "8192"], port=8001, log_path="/kaggle/working/serve2.log", env={"CUDA_VISIBLE_DEVICES": "1"}); print(f"second engine ready after {ready2} s")
urls = [URL, "http://127.0.0.1:8001"]
for name, policy in (("round-robin", round_robin), ("session-sticky", sticky)):
    print(f"{name}:"); run_sessions(48, "careful", policy, urls)
stop(proc); stop(proc2)"""),
]

CAP_C = [
    ("markdown", "# Capstone C: cost per million tokens\n\nEach configuration driven from one request at a time up to 128 at once, and the price of a million output tokens at every point from the GPU's hourly rate. On Kaggle the T4 is free, so the notebook prices it as if rented: about $0.35 an hour is a typical cloud T4 rate. The FP8 rows on the capstone page are L4-only."),
    ("code", CHECK), ("code", INSTALL), ("code", HELPERS),
    ("code", "# Cell 4: the benchmark call and the ladder.\n" + BENCH_QUIET + r"""
PRICE_PER_GPU_HOUR = 0.35
INPUT, OUTPUT = 1024, 256
def ladder(gpus):
    print(f"  {'concurrency':>12} {'out tok/s':>10} {'TPOT p50':>9} {'TTFT p99':>9}  {'$ per 1M output':>15}  {'$ per 1M total':>14}")
    for c in (1, 8, 32, 64):
        r = bench_quiet(rate="inf", num_prompts=max(16, 2 * c), input_len=INPUT, output_len=OUTPUT, max_concurrency=c)
        out = r.get("Output token throughput (tok/s)", 0.0); hourly = PRICE_PER_GPU_HOUR * gpus
        per_out = hourly / (out * 3600 / 1e6) if out else float("inf"); per_total = per_out * OUTPUT / (INPUT + OUTPUT)
        print(f"  {c:>12} {out:10.1f} {r.get('P50 TPOT (ms)', 0):8.1f}ms {r.get('P99 TTFT (ms)', 0):8.0f}ms  {per_out:15.3f}  {per_total:14.3f}")"""),
    ("code", '# Cell 5: one T4, fp16.\nproc, ready_s = serve(["--max-model-len", "4096"]); print(f"ready after {ready_s} s"); ladder(1); stop(proc)'),
    ("code", '# Cell 6: two T4s, tensor parallel.\nproc, ready_s = serve(["--max-model-len", "4096", "--tensor-parallel-size", "2"]); print(f"ready after {ready_s} s"); ladder(2); stop(proc)'),
    ("code", '# Cell 7: two T4s, two copies.\nproc, ready_s = serve(["--max-model-len", "4096", "--data-parallel-size", "2"]); print(f"ready after {ready_s} s"); ladder(2); stop(proc)'),
]


def write(path, cells):
    nb = {"cells": [{"cell_type": t, "id": f"cell-{i}", "metadata": {}, "source": src.splitlines(keepends=True), **({"outputs": [], "execution_count": None} if t == "code" else {})} for i, (t, src) in enumerate(cells)],
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
    write("labs/08-fp8-weights/kaggle/lab.ipynb", LAB08)
    write("labs/09-int4/kaggle/lab.ipynb", LAB09)
    write("labs/10-fp8-kv-cache/kaggle/lab.ipynb", LAB10)
    write("labs/11-speculative-decoding/kaggle/lab.ipynb", LAB11)
    write("labs/12-structured-output/kaggle/lab.ipynb", LAB12)
    write("labs/13-multi-lora/kaggle/lab.ipynb", LAB13)
    write("labs/14-cold-start-and-metrics/kaggle/lab.ipynb", LAB14)
    write("labs/15-two-engines/kaggle/lab.ipynb", LAB15)
    write("labs/16-two-gpus/kaggle/lab.ipynb", LAB16)
    write("labs/capstone-a-tune-to-an-slo/kaggle/lab.ipynb", CAP_A)
    write("labs/capstone-b-engineer-the-cache/kaggle/lab.ipynb", CAP_B)
    write("labs/capstone-c-cost-per-million-tokens/kaggle/lab.ipynb", CAP_C)
    print("wrote 20 notebooks")
