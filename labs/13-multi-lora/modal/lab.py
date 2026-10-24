"""Lab 13: many fine-tunes on one base. Two adapters behind one server; what each costs per step.
Run: modal run labs/13-multi-lora/modal/lab.py"""
import os, sys, time, subprocess, json, re
import modal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "modal"))
from common import *  # noqa: E402,F403

app = modal.App("llminference-lab13")
lora_image = base_image.pip_install("peft==0.21.2").add_local_python_source("common")


@app.function(image=lora_image, gpu="L4", timeout=2400, volumes=VOLUMES_WITH_MODELS)
def run():
    banner("1. Two rank-8 adapters, made here (random weights: this lab measures cost, not quality)")
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    for name in ("legal", "support"):
        cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.0)
        m = get_peft_model(base, cfg)
        for p in m.parameters():
            if p.requires_grad: torch.nn.init.normal_(p, std=0.02)
        m.save_pretrained(f"/models/lora-{name}"); del m
        size = sum(os.path.getsize(os.path.join(f"/models/lora-{name}", f)) for f in os.listdir(f"/models/lora-{name}"))
        print(f"  adapter {name}: {size / 1e6:.1f} MB")
    del base; torch.cuda.empty_cache(); models_volume.commit()
    banner("2. One server, the base plus both adapters")
    proc, ready = serve(["--max-model-len", "4096", "--enable-lora", "--max-lora-rank", "8", "--lora-modules", "legal=/models/lora-legal", "support=/models/lora-support"]); print(f"ready after {ready} s")
    for label, model in (("base", MODEL), ("adapter legal", "legal"), ("adapter support", "support")):
        print(bench_row(f"{label}, 1 at a time", bench_serve(rate="inf", num_prompts=16, input_len=512, max_concurrency=1, model=model)))
    banner("3. Thirty-two concurrent, base only, then half legal half support")
    print(bench_row("base x32", bench_serve(rate="inf", num_prompts=64, input_len=512, max_concurrency=32, model=MODEL)))
    import threading
    res = {}
    def go(name):
        res[name] = load_test(rate=0, num_prompts=32, prompt_len=512, out_len=128, model=name, concurrency=16)
    ts = [threading.Thread(target=go, args=(n,)) for n in ("legal", "support")]
    t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]
    tot = sum(r["out_tok_s"] for r in res.values())
    print(f"mixed x32 (16 legal + 16 support)      out tok/s {tot:7.1f}  TPOT p50 {res['legal']['tpot_p50_ms']:5.1f} / {res['support']['tpot_p50_ms']:5.1f} ms")
    stop(proc)


@app.local_entrypoint()
def main():
    run.remote()
