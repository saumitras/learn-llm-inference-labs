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
on a fresh machine takes about five minutes. On the Kaggle T4 the dtype line reads `torch.float16`, the backend is
`TRITON_ATTN`, and the KV cache is about 5 GiB.
