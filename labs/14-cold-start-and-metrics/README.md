# Lab 14: cold start, the metrics endpoint, and the thinking switch

Time to ready with and without compilation, and again with the compile cache warm; the metrics endpoint idle and
under a burst of forty; the hybrid Qwen3-4B with thinking off and on.

- Modal: `modal run labs/14-cold-start-and-metrics/modal/lab.py`
- Kaggle: `kaggle/lab.ipynb`

## Expected output, Modal L4 (captured 24 October 2026, vLLM 0.32.0)

```
1. Cold start: default, then with compilation off, then default again (the compile cache)
  default                ready after  160.3 s   Loading weights took 5.99 s   Graph capturing finished in 8 secs
  --enforce-eager        ready after   48.6 s   Loading weights took 1.27 s   no graph capture
  default, second start  ready after   65.6 s   Loading weights took 1.58 s   Graph capturing finished in 8 secs

2. The metrics endpoint, idle and during a burst of forty
  idle: running 0, waiting 0, kv_cache_usage_perc 0, prefix hits 0, preemptions 0
  second running waiting kv%  preempt  prefix hit%  gen tokens
       1      40       0    2        0         88%        880
       3      40       0    5        0         88%       2960
       5      40       0    8        0         88%       4960
       6      27       0    6        0         88%       5796
      10      27       0    9        0         88%       8496

3. The thinking switch on the hybrid Qwen3-4B
  thinking False:    52 completion tokens in   1.7 s   answer: ... 4 × $2 = $8.  **Answer: $8**.
  thinking True :   531 completion tokens in  17.5 s   answer: ... 4 × $2 = **$8**.  **Answer:** $8.
```

The weights were already on the mounted volume, so the first start is compilation and warm-up, not download. The
third start reads the compile cache the first one wrote. Both answers are right; thinking spent ten times the tokens.

## Expected output, Kaggle T4 ×2 (captured 25 October 2026, vLLM 0.32.0, `--dtype half`)

```
1. Cold start: default, then with compilation off, then default again (the compile cache)
  default                ready after  228.8 s   Loading weights took 6.97 s   Graph capturing finished in 14 secs
  --enforce-eager        ready after   59.6 s   Loading weights took 6.99 s   no graph capture
  default, second start  ready after  110.6 s   Loading weights took 6.95 s   Graph capturing finished in 14 secs

2. The metrics endpoint, idle and during a burst of forty
  idle: running 0, waiting 0, kv_cache_usage_perc 0, prefix hits 0, preemptions 0
  second running waiting kv%  preempt  prefix hit%  gen tokens
       2      40       0    4        0         88%        241
       5      40       0    9        0         88%       1681
      10      40       0   15        0         88%       3881
      13      40       0   18        0         88%       5121
      15      34       0   17        0         88%       5861

3. The thinking switch on the hybrid Qwen3-4B
  ready after 82.6 s
  thinking False:    52 completion tokens in   1.9 s   answer: ... 4 × $2 = $8.  **Answer: $8**.
  thinking True :   395 completion tokens in  14.2 s   answer: ... 4 \times 2 = 8.  **Answer:** $8.
```

The same shape as on the L4: compilation is most of the cold start, the compile cache halves the second start,
the burst runs forty with nothing waiting and an 88 percent hit rate from the first second. The T4 produces
about 400 tokens a second during the burst against 1,000 on the L4. The thinking answer spent 395 tokens where
the L4's run spent 531: the answer at temperature zero differs between the two GPUs' kernels, and so does its
length.
