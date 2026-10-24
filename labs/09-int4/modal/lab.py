"""Lab 09: INT4, quantised by you. Quantise Qwen3-0.6B to W4A16 with LLM Compressor, score the original and the result
with lm_eval against the running server, then measure the prefill penalty. Run: modal run labs/09-int4/modal/lab.py"""
import os, sys, time, subprocess, json, re
import modal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "modal"))
from common import *  # noqa: E402,F403

app = modal.App("llminference-lab09")
SMALL = "Qwen/Qwen3-0.6B"
OUT = "/models/Qwen3-0.6B-W4A16"
quant_image = (modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
               .pip_install("llmcompressor==0.14.0.1").add_local_python_source("common"))
eval_image = base_image.pip_install("lm_eval[api]==0.4.13").add_local_python_source("common")


@app.function(image=quant_image, gpu="L4", timeout=2400, volumes=VOLUMES_WITH_MODELS)
def quantise():
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    recipe = GPTQModifier(scheme="W4A16", targets="Linear", ignore=["lm_head"])
    tok = AutoTokenizer.from_pretrained(SMALL)
    model = AutoModelForCausalLM.from_pretrained(SMALL, device_map="cuda", dtype=torch.bfloat16)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train").filter(lambda x: len(x["text"]) > 400).shuffle(seed=42).select(range(128))
    ds = ds.map(lambda x: tok(x["text"], max_length=2048, truncation=True, add_special_tokens=False), remove_columns=ds.column_names)
    t0 = time.time()
    oneshot(model=model, tokenizer=tok, dataset=ds, recipe=recipe, output_dir=OUT, max_seq_length=2048, num_calibration_samples=128, pipeline="basic")
    took = round(time.time() - t0)
    size = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.endswith(".safetensors"))
    models_volume.commit(); hf_cache.commit()
    return {"seconds": took, "safetensors_gb": round(size / 1e9, 2)}


def lm_eval(model_name, limit=100):
    args = ["lm_eval", "--model", "local-completions", "--model_args", f"model={model_name},base_url={URL}/v1/completions,num_concurrent=8,max_retries=2,tokenized_requests=False",
            "--tasks", "hellaswag", "--limit", str(limit), "--batch_size", "8"]
    out = subprocess.run(args, capture_output=True, text=True)
    m = re.search(r"\|hellaswag[^\n]*", out.stdout + out.stderr)
    acc = re.findall(r"acc_norm\s*\|[^|]*\|\s*([0-9.]+)", out.stdout + out.stderr)
    return (m.group(0) if m else out.stdout[-800:] + out.stderr[-800:]), (acc[0] if acc else "?")


@app.function(image=eval_image, gpu="L4", timeout=2400, volumes=VOLUMES_WITH_MODELS)
def evaluate(q):
    banner("1. Quantisation")
    print(f"GPTQ W4A16 on {SMALL} with 128 wikitext samples: {q['seconds']} s on an L4; checkpoint {q['safetensors_gb']} GB against 1.50 GB")
    for label, model in (("BF16 original", SMALL), ("W4A16 quantised", OUT)):
        banner(f"{label}: serve, score, measure")
        proc, ready = serve(["--max-model-len", "4096"], model=model); print(f"ready after {ready} s")
        print("  " + log_lines("/tmp/serve.log", {"q": r"quantization=[a-z0-9_]+"}).get("q", "quantization=none"))
        line, acc = lm_eval(model)
        print(f"  lm_eval hellaswag, 100 examples: acc_norm {acc}   ({line.strip()[:110]})")
        print("  " + bench_row("1 at a time, 1,024 in", bench_serve(rate="inf", num_prompts=16, input_len=1024, max_concurrency=1, model=model)))
        print("  " + bench_row("32 concurrent, 1,024 in", bench_serve(rate="inf", num_prompts=64, input_len=1024, max_concurrency=32, model=model)))
        stop(proc)


@app.local_entrypoint()
def main():
    q = quantise.remote()
    evaluate.remote(q)
