# Lab 07: prefix caching and cache hygiene

Cold and warm first token for one prompt; twenty questions on one system prompt; a timestamp at the top and at the
bottom; a sweep over the shared-prefix fraction.

- Modal: `modal run labs/07-prefix-caching/modal/lab.py`
- Kaggle: `kaggle/lab.ipynb`

## Expected output, Modal L4 (captured 24 October 2026, vLLM 0.32.0)

```
1. The same 2,000-token prompt, cold and then warm
  cold: first token   244 ms   prefix blocks queried 1453, hit    0
  warm: first token    56 ms   prefix blocks queried 1453, hit 1440

2. Twenty questions on one 600-token system prompt
  queried 10140 blocks, hit 9120: hit rate 90%   first token: first request 115 ms, median of the rest 49 ms

3. The same twenty, with a timestamp at the top of the system prompt, then at the bottom
  timestamp at the top   : hit rate   3%
  timestamp at the bottom: hit rate  93%

4. Throughput against the shared-prefix fraction (1,024-token prompts, 4 req/s, 48 prompts)
shared    0 of 1,024 ( 0%)  req/s 2.38  out tok/s 304  TTFT p50 474 p99 958 ms  TPOT p50 89 p99 103 ms  peak conc 48
shared  256 of 1,024 (25%)  req/s 2.52  out tok/s 322  TTFT p50 324 p99 650 ms  TPOT p50 75 p99  86 ms  peak conc 43
shared  512 of 1,024 (50%)  req/s 2.65  out tok/s 339  TTFT p50 216 p99 437 ms  TPOT p50 60 p99  69 ms  peak conc 38
shared  768 of 1,024 (75%)  req/s 2.74  out tok/s 351  TTFT p50 174 p99 286 ms  TPOT p50 49 p99  54 ms  peak conc 31
```

The counters count tokens, not blocks, on this engine version: 1,453 queried is the prompt's length, and 1,440 hit
is everything except the last partial block. Moving the timestamp from the top of the system prompt to the bottom
takes the hit rate from 3 percent to 93.

## Expected output, Kaggle T4 ×2 (captured 24 October 2026, vLLM 0.32.0, `--dtype half`)

```
1. The same 2,000-token prompt, cold and then warm
  cold: first token  2253 ms   prefix tokens queried 1453, hit    0
  warm: first token   102 ms   prefix tokens queried 1453, hit 1440

2. Twenty questions on one 600-token system prompt
  hit rate 90%   first token: first request 287 ms, median of the rest 80 ms

3. The same twenty, with a timestamp in the system prompt
  timestamp at the top   : hit rate   3%
  timestamp at the bottom: hit rate  93%

4. Throughput against the shared-prefix fraction (1,024-token prompts, 4 req/s, 48 prompts)
shared    0 of 1,024 ( 0%)  out tok/s  72.2  TTFT p50 22575 p99 57290 ms
shared  256 of 1,024 (25%)  out tok/s  84.2  TTFT p50 11171 p99 49571 ms
shared  512 of 1,024 (50%)  out tok/s 110.5  TTFT p50  7254 p99 14484 ms
shared  768 of 1,024 (75%)  out tok/s 138.8  TTFT p50  1824 p99  4536 ms
```

The hit rates match the L4's to the point, as they should: they depend on the prompts, not the GPU. The times
show how much more a cache hit is worth on a slow prefill: cold to warm is 2.3 seconds to 0.1, twenty-two times,
against four times on the L4. The sweep is far past the T4's knee at four requests a second, so its first-token
times are queueing times, and sharing three quarters of each prompt takes the p99 from 57 seconds to 4.5.
