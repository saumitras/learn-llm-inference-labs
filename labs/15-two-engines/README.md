# Lab 15: two engines, same traffic

vLLM and SGLang one after the other, the same model, the same load generator in the client, on random prompts
and on prompts that share three quarters of their length.

- Modal: `modal run labs/15-two-engines/modal/lab.py` (vLLM on the course image, SGLang on `lmsysorg/sglang:v0.5.22`)
- Kaggle: `kaggle/lab.ipynb`

## Expected output, Modal L4 (captured 24 October 2026, vLLM 0.32.0, SGLang 0.5.22)

```
vLLM 0.32.0
  ready after 228.6 s
  vLLM: random prompts           req/s  2.61  out tok/s   333.5  TTFT p50   481 p99  1518 ms  TPOT p50  87.8 p99 103.8 ms
  vLLM: 75% shared prefix        req/s  2.99  out tok/s   382.9  TTFT p50   186 p99   269 ms  TPOT p50  51.5 p99  54.8 ms

SGLang 0.5.22
  ready after 182.0 s
  SGLang: random prompts         req/s  2.26  out tok/s   289.3  TTFT p50   530 p99  5018 ms  TPOT p50  95.8 p99 122.6 ms
  SGLang: 75% shared prefix      req/s  2.98  out tok/s   382.0  TTFT p50   147 p99   228 ms  TPOT p50  56.7 p99  60.5 ms
```

Each row is 48 prompts of 1,024 tokens with 128-token answers at 4 requests a second offered. On shared-prefix
traffic the two engines land on the same number, 382 tokens a second; on random prompts vLLM held a p99 first
token of 1.5 s where SGLang's reached 5 s, with 13 percent more tokens a second. Both are default settings.

## Expected output, Kaggle T4 ×2 (captured 25 October 2026, vLLM 0.32.0 and SGLang 0.5.22, `--dtype half`; 32 prompts at 2 req/s)

The halves ran in separate sessions. SGLang runs from its own Python 3.12 environment made by uv (Kaggle's Python 3.13
has no wheel for its pinned `outlines_core`), with the Triton attention backend and PyTorch sampling. The T4 has no
prebuilt FlashInfer kernels, so SGLang compiles some at startup; that needs the environment's bin folder on `PATH`
(for ninja) and `LIBRARY_PATH` pointing at `/usr/local/nvidia/lib64`, where Kaggle keeps `libcuda.so`.

```
vLLM ready after 222.8 s
  vLLM: random prompts       req/s 0.63  out tok/s  80.2  TTFT p50  6549 p99  11149 ms  TPOT p50 249.3 p99 309.2 ms
  vLLM: 75% shared prefix    req/s 0.98  out tok/s 124.9  TTFT p50   870 p99   2260 ms  TPOT p50 183.0 p99 195.5 ms

SGLang installed in 72 s
SGLang ready after 291 s
  Engine startup timings (s): load_weight=10.94, kv_cache_allocation=0.49, scheduler_e2e=220.63, cuda_graph={prefill=196.13, decode=3.20, ...}
  SGLang: random prompts     req/s 0.25  out tok/s  32.2  TTFT p50 51911 p99 102186 ms  TPOT p50 275.2 p99 562.8 ms
  SGLang: 75% shared prefix  req/s 0.54  out tok/s  69.0  TTFT p50 16299 p99  26353 ms  TPOT p50 285.0 p99 443.7 ms

From the logs:
  SGLang  KV Cache is allocated. dtype: torch.float16, #tokens: 20976   (chunked_prefill_size=2048)
  vLLM    GPU KV cache size: 31,856 tokens
```

On the T4 the engines are no longer the same server. SGLang keeps a cache two thirds the size of vLLM's and its
fallback attention path makes each step slower, so at two requests a second it falls behind the arrivals on random
prompts and the queue becomes the first token. The shared prefix doubles its throughput but the queue never clears.
These rows measure each engine's support for a seven-year-old GPU rather than its defaults; on the L4, where both run
their tuned kernels, they tie on shared-prefix traffic.
