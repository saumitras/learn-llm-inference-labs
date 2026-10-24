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
