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
