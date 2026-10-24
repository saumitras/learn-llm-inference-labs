"""Shared Modal setup for every lab: the image, the two caches, and small helpers.

The image is NVIDIA's CUDA developer image with vLLM pinned; the slim Debian image cannot run vLLM's compiled
mode because kernels are built at first use and need the toolkit. The model cache and the compile cache are
Modal volumes, so a lab after the first starts in seconds rather than minutes.
"""
import json
import re
import subprocess
import time
import urllib.request

import modal

VLLM_VERSION = "0.32.0"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
PORT = 8000
URL = f"http://127.0.0.1:{PORT}"

base_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}", "openai>=1.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "VLLM_LOGGING_LEVEL": "INFO"})
    # Ship this module into the container; Modal only sends the lab script itself by default.

)
# Local sources must be added last; a lab that needs extra packages builds on base_image and adds "common" itself.
image = base_image.add_local_python_source("common")
hf_cache = modal.Volume.from_name("llminference-hf-cache", create_if_missing=True)
compile_cache = modal.Volume.from_name("llminference-vllm-cache", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": compile_cache}


def serve(extra_args=(), model=MODEL, log_path="/tmp/serve.log", timeout_s=900):
    """Start `vllm serve` in the background and wait until /health answers. Returns (process, seconds to ready)."""
    args = ["vllm", "serve", model, "--port", str(PORT), *extra_args]
    log = open(log_path, "w")
    t0 = time.time()
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError("vllm serve exited; last log lines:\n" + tail(log_path, 40))
        try:
            urllib.request.urlopen(URL + "/health", timeout=2)
            return proc, round(time.time() - t0, 1)
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("vllm serve did not become ready in time")


def stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    hf_cache.commit()
    compile_cache.commit()


def tail(path, n=20):
    with open(path) as f:
        return "".join(f.readlines()[-n:])


def log_lines(path, patterns):
    """The first log line matching each pattern, with the timestamp and process prefix stripped."""
    text = open(path).read()
    out = {}
    for name, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            line = m.group(0)
            line = re.sub(r"^\(EngineCore[^)]*\) ", "", line)
            line = re.sub(r"^INFO \d\d-\d\d \d\d:\d\d:\d\d \[[^\]]+\] ", "", line)
            out[name] = line.strip()
    return out


STARTUP_PATTERNS = {
    "dtype": r"INFO[^\n]*dtype=torch\.[a-z0-9]+",
    "backend": r"Using [A-Z_]+ attention backend[^\n]*",
    "weights": r"Loading weights took [^\n]*",
    "model": r"Model loading took [^\n]*",
    "kv_memory": r"Available KV cache memory: [^\n]*",
    "kv_size": r"GPU KV cache size: [^\n]*",
    "graphs": r"Graph capturing finished[^\n]*",
}


def metrics(names):
    """Current values of the named counters from /metrics."""
    out = {}
    for line in urllib.request.urlopen(URL + "/metrics", timeout=5).read().decode().splitlines():
        if line.startswith("#"):
            continue
        for n in names:
            if line.startswith(n + "{") or line.startswith(n + " "):
                try:
                    out[n] = float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return out


def chat(messages, model=MODEL, **params):
    body = json.dumps({"model": model, "messages": messages, **params}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


def stream_chat(messages, model=MODEL, **params):
    """A streaming chat request. Returns (first_token_s, [gap_s, ...], text): time to first token and the gaps between tokens."""
    body = json.dumps({"model": model, "messages": messages, "stream": True, **params}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    last = None
    gaps = []
    text = []
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            delta = chunk["choices"][0].get("delta", {}).get("content")
            if not delta:
                continue
            now = time.time()
            if first is None:
                first = now - t0
            else:
                gaps.append(now - last)
            last = now
            text.append(delta)
    return first, gaps, "".join(text)


def bench_serve(rate, num_prompts, input_len=1024, output_len=128, max_concurrency=None, prefix_len=0, model=MODEL, extra=()):
    """Run `vllm bench serve` and return the parsed result lines as a dict of floats."""
    args = ["vllm", "bench", "serve", "--model", model, "--tokenizer", MODEL, "--port", str(PORT), "--dataset-name", "random",
            "--random-input-len", str(input_len), "--random-output-len", str(output_len), "--num-prompts", str(num_prompts),
            "--request-rate", str(rate), "--metric-percentiles", "50,95,99", "--percentile-metrics", "ttft,tpot,itl,e2el",
            "--ignore-eos", "--seed", "0", "--random-prefix-len", str(prefix_len), *extra]
    if max_concurrency:
        args += ["--max-concurrency", str(max_concurrency)]
    out = subprocess.run(args, capture_output=True, text=True)
    res = {}
    for line in out.stdout.splitlines():
        m = re.match(r"^(.*?):\s+([0-9.]+)\s*$", line)
        if m:
            res[m.group(1).strip()] = float(m.group(2))
    if out.returncode != 0:
        print(out.stderr[-1500:])
    res["_stdout"] = out.stdout
    return res


def bench_row(label, r):
    """One line of a results table from a bench_serve result."""
    g = lambda k: r.get(k, float("nan"))
    return (f"{label:<26} req/s {g('Request throughput (req/s)'):5.2f}  out tok/s {g('Output token throughput (tok/s)'):7.1f}  "
            f"TTFT p50 {g('P50 TTFT (ms)'):7.0f} p99 {g('P99 TTFT (ms)'):7.0f} ms  TPOT p50 {g('P50 TPOT (ms)'):5.1f} p99 {g('P99 TPOT (ms)'):5.1f} ms  "
            f"peak conc {g('Peak concurrent requests'):3.0f}")


models_volume = modal.Volume.from_name("llminference-models", create_if_missing=True)
VOLUMES_WITH_MODELS = {**VOLUMES, "/models": models_volume}


def random_prompt(n_tokens, seed):
    """About n_tokens of varied text, deterministic per seed, so prefix caching cannot collapse it."""
    import random
    rng = random.Random(seed)
    words = ["river", "stone", "quiet", "engine", "orange", "ledger", "window", "copper", "signal", "harbor", "meadow", "lantern", "velvet", "summit", "cinder", "fable"]
    return " ".join(rng.choice(words) for _ in range(int(n_tokens * 0.8)))


def load_test(rate, num_prompts, prompt_len=512, out_len=128, prefix_len=0, model=MODEL, concurrency=None):
    """A small load generator that works against any OpenAI-compatible server: Poisson arrivals at `rate` per
    second (or a fixed concurrency), streaming requests, p50/p99 of TTFT and TPOT, and output tokens per second."""
    import random, threading
    rng = random.Random(1)
    prefix = random_prompt(prefix_len, 99) + " " if prefix_len else ""
    prompts = [prefix + random_prompt(prompt_len - prefix_len, i) for i in range(num_prompts)]
    results = []
    lock = threading.Lock()
    def one(i):
        first, gaps, text = stream_chat([{"role": "user", "content": prompts[i]}], model=model, max_tokens=out_len, temperature=0, ignore_eos=True)
        with lock:
            results.append((first, (sum(gaps) / len(gaps)) if gaps else 0.0, len(gaps) + 1))
    t0 = time.time()
    threads = []
    if concurrency:
        sem = threading.Semaphore(concurrency)
        def gated(i):
            with sem:
                one(i)
        for i in range(num_prompts):
            th = threading.Thread(target=gated, args=(i,)); th.start(); threads.append(th)
    else:
        for i in range(num_prompts):
            th = threading.Thread(target=one, args=(i,)); th.start(); threads.append(th)
            time.sleep(rng.expovariate(rate))
    for th in threads:
        th.join()
    dur = time.time() - t0
    ttft = sorted(r[0] for r in results); tpot = sorted(r[1] for r in results)
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"req_s": round(len(results) / dur, 2), "out_tok_s": round(sum(r[2] for r in results) / dur, 1),
            "ttft_p50_ms": round(1000 * q(ttft, 0.5)), "ttft_p99_ms": round(1000 * q(ttft, 0.99)),
            "tpot_p50_ms": round(1000 * q(tpot, 0.5), 1), "tpot_p99_ms": round(1000 * q(tpot, 0.99), 1), "duration_s": round(dur, 1)}


def load_row(label, r):
    return (f"{label:<30} req/s {r['req_s']:5.2f}  out tok/s {r['out_tok_s']:7.1f}  TTFT p50 {r['ttft_p50_ms']:6d} p99 {r['ttft_p99_ms']:6d} ms  "
            f"TPOT p50 {r['tpot_p50_ms']:5.1f} p99 {r['tpot_p99_ms']:5.1f} ms")


def banner(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)
