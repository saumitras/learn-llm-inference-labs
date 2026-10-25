# Lab 00: install, first request, what you downloaded

Serve the course model, send one request, read the five startup lines every later lab refers to, see the
model's confidence per token, and list what landed on disk.

- Modal: `modal run labs/00-first-request/modal/lab.py`
- Kaggle: `kaggle/lab.ipynb`, cells in order

## Expected output, Modal L4 (captured 24 October 2026, vLLM 0.32.0)

```
1. The GPU this lab got
NVIDIA L4, 23034 MiB, 610.57.04
vllm 0.32.0 · torch 2.13.0+cu130 · compute capability (8, 9)

2. Start the server
vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 4096
ready after 214.0 s
  dtype      dtype=torch.bfloat16
  backend    Using FLASH_ATTN attention backend
  weights    Loading weights took 5.39 seconds
  model      Model loading took 7.64 GiB memory and 11.4 seconds
  kv_memory  Available KV cache memory: 11.82 GiB
  kv_size    GPU KV cache size: 86,096 tokens, Maximum concurrency for 4,096 tokens per request: 21.02x
  graphs     Graph capturing finished in 11 secs, took 0.37 GiB

3. One request
An inference server performs real-time predictions or decisions by applying machine learning models to new, unseen data.
prompt tokens 19 · completion tokens 21

4. The same request, with the model's confidence per token
  'The'        chosen with 100%   alternatives: 'The' 100%, 'Paris' 0%, 'the' 0%
  ' capital'   chosen with 100%   alternatives: ' capital' 100%, 'capital' 0%, '资本' 0%
  ' of'        chosen with 100%   alternatives: ' of' 100%, ' của' 0%, 'ของ' 0%
  ' France'    chosen with 100%   alternatives: ' France' 100%, 'France' 0%, '法国' 0%
  ' is'        chosen with 100%   alternatives: ' is' 100%, '是' 0%, 'は' 0%

5. What landed on disk
  model-00001-of-00003.safetensors   3957.9 MB
  model-00002-of-00003.safetensors   3987.5 MB
  model-00003-of-00003.safetensors     99.6 MB
  tokenizer.json                       11.4 MB
  vocab.json                            2.8 MB
  merges.txt                            1.7 MB
  config.json, generation_config.json, tokenizer_config.json, model.safetensors.index.json, README.md, LICENSE, .gitattributes
  total                                 8.06 GB
```

The 214 s to ready is on a machine whose compile cache was already warm from an earlier lab; the first lab ever run
on a fresh machine takes about five minutes.

## Expected output, Kaggle T4 ×2 (captured 24 October 2026, vLLM 0.32.0)

```
Tesla T4, 15360 MiB, 580.178.04
Tesla T4, 15360 MiB, 580.178.04
Path: Kaggle T4. Labs 08 and 10 are read-only on this GPU (no FP8 arithmetic).   (the check cell's wording at the time; Lab 08 turned out to run)
vllm 0.32.0 · torch 2.13.0+cu130 · compute capability (7, 5)      (the install cell took about eleven minutes)

ready after 225.3 s
  dtype      dtype=torch.float16
  backend    Using TRITON_ATTN attention backend out of potential backends: ['TRITON_ATTN', 'FLEX_ATTENTION']
  weights    Loading weights took 6.62 seconds
  model      Model loading took 7.64 GiB memory and 52.8 seconds
  kv_memory  Available KV cache memory: 4.38 GiB
  kv_size    GPU KV cache size: 31,856 tokens, Maximum concurrency for 4,096 tokens per request: 7.78x
  graphs     Graph capturing finished in 14 secs, took 0.37 GiB

An inference server performs real-time predictions or decisions by applying machine learning models to new, unseen data.
prompt tokens 19 · completion tokens 21
  'The' ... ' is'   every token chosen with 100%, the same alternatives as on the L4
  total on disk 8.06 GB, the same files
```

The server uses one of the two T4s. The weights are the same 7.64 GiB in float16, which leaves 4.38 GiB for the
cache on a 15 GiB GPU: 31,856 tokens, seven and a half 4,096-token requests, against twenty-one on the L4.
