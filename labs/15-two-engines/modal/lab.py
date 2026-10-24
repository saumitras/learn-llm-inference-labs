"""Lab 15: two engines, same traffic. vLLM and SGLang one after the other on random and shared-prefix traffic, with the
course's own load generator so the client is identical. Run: modal run labs/15-two-engines/modal/lab.py"""
import os, sys, time, subprocess, json, re
import modal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "modal"))
from common import *  # noqa: E402,F403

app = modal.App("llminference-lab15")
sglang_image = (modal.Image.from_registry("lmsysorg/sglang:v0.5.22").add_local_python_source("common"))


def traffic(engine):
    for label, prefix in (("random prompts", 0), ("75% shared prefix", 768)):
        r = load_test(rate=4, num_prompts=48, prompt_len=1024, out_len=128, prefix_len=prefix)
        print("  " + load_row(f"{engine}: {label}", r))


@app.function(image=image, gpu="L4", timeout=2400, volumes=VOLUMES)
def vllm_run():
    banner("vLLM 0.32.0")
    proc, ready = serve(["--max-model-len", "4096"]); print(f"  ready after {ready} s")
    traffic("vLLM"); stop(proc)


@app.function(image=sglang_image, gpu="L4", timeout=2400, volumes={"/hf": hf_cache})   # the SGLang image already has files under /root/.cache/huggingface
def sglang_run():
    import urllib.request
    os.environ["HF_HOME"] = "/hf"
    banner("SGLang 0.5.22")
    log = open("/tmp/sglang.log", "w"); t0 = time.time()
    proc = subprocess.Popen(["python", "-m", "sglang.launch_server", "--model-path", MODEL, "--port", str(PORT), "--context-length", "4096"], stdout=log, stderr=subprocess.STDOUT)
    while time.time() - t0 < 900:
        if proc.poll() is not None:
            print("sglang exited:\n" + tail("/tmp/sglang.log", 30)); return
        try:
            urllib.request.urlopen(URL + "/health", timeout=2); break
        except Exception:
            time.sleep(0.5)
    print(f"  ready after {round(time.time() - t0, 1)} s")
    traffic("SGLang"); proc.terminate(); proc.wait()
    print("  " + tail("/tmp/sglang.log", 3)[:300])


@app.local_entrypoint()
def main():
    vllm_run.remote()
    sglang_run.remote()
