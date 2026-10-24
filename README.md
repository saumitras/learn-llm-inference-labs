# Learn LLM Inference: labs

The hands-on labs for [Learn LLM Inference](https://learnllminference.fignix.com), a free course by fignix on serving
large language models fast and cheaply. Each lab runs on one of three paths; pick yours on the course's labs page.

| Path | GPU | Cost | How a lab runs |
| --- | --- | --- | --- |
| Kaggle | 2 × T4 (16 GB each) | free, no card | open the lab's notebook in Kaggle, turn on the GPU, run the cells |
| Modal | 1 × L4 (24 GB) | pay per second, card on file | `modal run labs/<lab>/modal/lab.py` |
| AWS | g6.xlarge (L4) | about $0.81 an hour | coming |

Every lab pins vLLM 0.32.0 and uses Qwen/Qwen3-4B-Instruct-2507 unless it says otherwise. The expected output on
each path is in the lab's README and on the course page; it was captured on the date it names.

## Modal

```bash
pip install modal
modal token new            # once; approve in the browser
modal run labs/00-first-request/modal/lab.py
```

The first run builds the image (about two minutes) and downloads the model (about one minute); both are cached, so
later labs start in seconds. The model cache and the compiled kernels live in two Modal volumes. Lab 15 pulls the
SGLang image for its second half, and Lab 16 asks for a two-L4 machine (`gpu="L4:2"`). A Starter workspace runs at
most ten GPUs at once; a lab launched past that limit waits for a free one.

## Kaggle

Open the lab's notebook with the course page's button, or paste its URL into Kaggle's File → Open notebook. Then
Settings → Accelerator → GPU T4 ×2, and Run all. The first cell checks the GPU; the second installs vLLM, which takes
about nine minutes on Kaggle. The two FP8 labs are read-only on this path, because the T4 has no FP8 arithmetic.

## Layout

```
ops/modal/common.py        the image, the volumes and helpers every Modal lab imports
labs/NN-slug/README.md     what the lab does, the commands, the expected output per path
labs/NN-slug/modal/lab.py  the Modal script
labs/NN-slug/kaggle/lab.ipynb
```
