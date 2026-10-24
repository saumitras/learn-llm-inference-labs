"""Lab 04: continuous batching and the knee. A rate sweep with the running and waiting counts read from the server,
then the same sweep with the batch capped. Run: modal run labs/04-continuous-batching/modal/lab.py"""
import os, sys, time, subprocess, json, re
import modal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "modal"))
from common import *  # noqa: E402,F403
import threading

app = modal.App("llminference-lab04")
NAMES = ["vllm:num_requests_running", "vllm:num_requests_waiting"]


def sweep(rates, num_prompts=60):
    rows = []
    for rate in rates:
        peak = {"running": 0, "waiting": 0}
        stop_flag = threading.Event()
        def watch():
            while not stop_flag.is_set():
                try:
                    m = metrics(NAMES)
                    peak["running"] = max(peak["running"], int(m.get(NAMES[0], 0)))
                    peak["waiting"] = max(peak["waiting"], int(m.get(NAMES[1], 0)))
                except Exception:
                    pass
                time.sleep(0.5)
        th = threading.Thread(target=watch); th.start()
        r = bench_serve(rate=rate, num_prompts=num_prompts, input_len=512, output_len=128)
        stop_flag.set(); th.join()
        rows.append((rate, r, peak))
        print(bench_row(f"{rate} req/s offered", r) + f"  peak running {peak['running']:3d} waiting {peak['waiting']:3d}")
    return rows


@app.function(image=image, gpu="L4", timeout=2400, volumes=VOLUMES)
def run():
    banner("1. Rate sweep, 512-token prompts, 128-token answers, 60 prompts per rate")
    proc, ready = serve(["--max-model-len", "4096"]); print(f"server ready after {ready} s")
    sweep([1, 2, 4, 6, 8, 12])
    banner("2. The server's own log line during the sweep")
    for line in re.findall(r"Running: \d+ reqs, Waiting: \d+ reqs[^\n]*", open("/tmp/serve.log").read())[-4:]:
        print("  " + line[:150])
    stop(proc)
    banner("3. The same sweep with the batch capped at 8 requests per step")
    proc, ready = serve(["--max-model-len", "4096", "--max-num-seqs", "8"]); print(f"server ready after {ready} s")
    sweep([1, 2, 4, 6])
    stop(proc)


@app.local_entrypoint()
def main():
    run.remote()
