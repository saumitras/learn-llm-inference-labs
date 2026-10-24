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

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}", "openai>=1.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "VLLM_LOGGING_LEVEL": "INFO"})
    # Ship this module into the container; Modal only sends the lab script itself by default.
    .add_local_python_source("common")
)
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


def banner(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)
