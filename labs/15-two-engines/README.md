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

## Expected output, Kaggle T4 ×2 (captured 25 October 2026, vLLM 0.32.0, `--dtype half`; 32 prompts at 2 req/s)

```
vLLM ready after 222.8 s
  vLLM: random prompts       req/s 0.63  out tok/s  80.2  TTFT p50 6549 p99 11149 ms  TPOT p50 249.3 p99 309.2 ms
  vLLM: 75% shared prefix    req/s 0.98  out tok/s 124.9  TTFT p50  870 p99  2260 ms  TPOT p50 183.0 p99 195.5 ms

SGLang: pip install "sglang[all]==0.5.22" failed on Kaggle's Python 3.13 image
  error: subprocess-exited-with-error
  Building wheel for outlines_core (pyproject.toml) did not run successfully.
```

The vLLM half ran: at two requests a second the T4 is past its knee on random prompts and the shared prefix
brings the first token from 6.5 seconds to 0.9. The SGLang half could not run on this path: the pinned release
has no wheel for one of its dependencies on Kaggle's Python 3.13 and the build fails, so the comparison exists
only in the Modal column until an SGLang release ships that wheel.
