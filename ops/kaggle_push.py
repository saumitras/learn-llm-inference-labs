"""Push a lab's Kaggle notebook to Kaggle with the Kaggle API, run it on the T4 pair, wait, and save its log.
Needs ~/.kaggle/kaggle.json (Kaggle → Settings → API → Create New Token).
Run: python ops/kaggle_push.py 00-first-request [01-measure ...]   (or `all`)"""
import json, os, re, shutil, subprocess, sys, time

OWNER = os.environ.get("KAGGLE_OWNER", "saumitra")
LABS = sorted(d for d in os.listdir("labs") if os.path.isfile(f"labs/{d}/kaggle/lab.ipynb"))
OUT = "ops/kaggle-runs"


def slug(lab):
    return "llminf-" + re.sub(r"[^a-z0-9]+", "-", lab.lower()).strip("-")


def push(lab, timeout=3 * 3600):
    folder = f"{OUT}/{lab}"
    os.makedirs(folder, exist_ok=True)
    shutil.copy(f"labs/{lab}/kaggle/lab.ipynb", f"{folder}/lab.ipynb")
    meta = {"id": f"{OWNER}/{slug(lab)}", "title": f"Learn LLM Inference {lab}", "code_file": "lab.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    json.dump(meta, open(f"{folder}/kernel-metadata.json", "w"), indent=1)
    out = subprocess.run(["kaggle", "kernels", "push", "-p", folder, "--accelerator", "NvidiaTeslaT4", "-t", str(timeout)], capture_output=True, text=True)
    print(f"{lab}: {out.stdout.strip()} {out.stderr.strip()}")
    return f"{OWNER}/{slug(lab)}"


def wait(ref, poll=60, limit=3 * 3600):
    t0 = time.time()
    while time.time() - t0 < limit:
        out = subprocess.run(["kaggle", "kernels", "status", ref], capture_output=True, text=True).stdout
        if "complete" in out.lower() or "error" in out.lower() or "cancel" in out.lower():
            return out.strip()
        time.sleep(poll)
    return "timed out"


def fetch_log(lab, ref):
    folder = f"{OUT}/{lab}/output"
    os.makedirs(folder, exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", ref, "-p", folder, "-o", "-q"], capture_output=True, text=True)
    log = os.path.join(folder, slug(lab) + ".log")
    if os.path.exists(log):
        text = []
        for line in open(log):
            try:
                rec = json.loads(line); text.append(rec.get("data", ""))
            except ValueError:
                text.append(line)
        open(f"{OUT}/{lab}/run.txt", "w").write("".join(text))
        return f"{OUT}/{lab}/run.txt"
    return None


if __name__ == "__main__":
    labs = LABS if sys.argv[1:] == ["all"] else sys.argv[1:]
    refs = {lab: push(lab) for lab in labs}
    for lab, ref in refs.items():
        print(f"{lab}: {wait(ref)}")
        print(f"{lab}: log at {fetch_log(lab, ref)}")
