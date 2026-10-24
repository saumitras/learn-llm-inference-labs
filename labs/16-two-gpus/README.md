# Lab 16: two GPUs

One L4, then two L4s with the model split across them (tensor parallel), then two L4s each holding a whole copy
(data parallel): the shelf, the batch-1 step, and a rate sweep for each.

- Modal: `modal run labs/16-two-gpus/modal/lab.py` (one `gpu="L4"` function and one `gpu="L4:2"` function)
- Kaggle: `kaggle/lab.ipynb` (two T4s)

## Expected output, Modal L4 (captured 24 October 2026, vLLM 0.32.0; each sweep row is 96 prompts of 512 tokens, 128-token answers)

```
One L4
  ready after 165.9 s
  GPU KV cache size: 86,096 tokens, Maximum concurrency for 4,096 tokens per request: 21.02x
  1 GPU: 1 at a time         req/s  0.23  out tok/s    29.6  TTFT p50   97 p99  100 ms  TPOT p50 33.3 p99 33.4 ms  peak conc  2
  1 GPU: 4 req/s             req/s  3.20  out tok/s   409.9  TTFT p50  239 p99  578 ms  TPOT p50 62.0 p99 75.8 ms  peak conc 51
  1 GPU: 8 req/s             req/s  5.18  out tok/s   662.8  TTFT p50  152 p99  229 ms  TPOT p50 57.1 p99 61.3 ms  peak conc 74
  1 GPU: 16 req/s            req/s  6.80  out tok/s   869.8  TTFT p50  173 p99  316 ms  TPOT p50 68.5 p99 71.0 ms  peak conc 96

Two L4s, tensor parallel (--tensor-parallel-size 2)
  ready after 304.7 s
  GPU KV cache size: 224,048 tokens, Maximum concurrency for 4,096 tokens per request: 54.70x
  TP=2: 1 at a time          req/s  0.41  out tok/s    52.0  TTFT p50   81 p99   98 ms  TPOT p50 18.7 p99 18.8 ms  peak conc  2
  TP=2: 4 req/s              req/s  3.53  out tok/s   451.5  TTFT p50  138 p99  259 ms  TPOT p50 31.9 p99 37.1 ms  peak conc 31
  TP=2: 8 req/s              req/s  6.18  out tok/s   791.5  TTFT p50   90 p99  124 ms  TPOT p50 30.6 p99 34.1 ms  peak conc 51
  TP=2: 16 req/s             req/s  8.98  out tok/s  1149.1  TTFT p50  117 p99  170 ms  TPOT p50 41.5 p99 43.8 ms  peak conc 88

Two L4s, data parallel (--data-parallel-size 2)
  ready after 200.9 s
  GPU KV cache size: 81,936 tokens (per copy), Maximum concurrency for 4,096 tokens per request: 20.00x
  DP=2: 1 at a time          req/s  0.23  out tok/s    29.4  TTFT p50  102 p99  107 ms  TPOT p50 33.5 p99 33.5 ms  peak conc  2
  DP=2: 4 req/s              req/s  3.33  out tok/s   425.7  TTFT p50  178 p99  297 ms  TPOT p50 43.6 p99 47.8 ms  peak conc 36
  DP=2: 8 req/s              req/s  5.51  out tok/s   705.5  TTFT p50  173 p99  293 ms  TPOT p50 48.8 p99 53.4 ms  peak conc 69
  DP=2: 16 req/s             req/s  7.89  out tok/s  1009.5  TTFT p50  157 p99  368 ms  TPOT p50 53.1 p99 56.4 ms  peak conc 96
```

Splitting the model halves the weights each GPU reads per step: 18.7 ms against 33.3 at batch 1, and a shelf of
224,048 tokens because each GPU holds 4 GB of weights instead of 8. Two whole copies keep the 33 ms step and
add capacity: 1,010 tokens a second at 16 req/s against 870 for one GPU and 1,149 for the split pair.

## Expected output, Kaggle T4 ×2 (captured 25 October 2026, vLLM 0.32.0, fp16; each sweep row is 64 prompts of 512 tokens, 128-token answers)

```
One T4
  ready after 206.3 s
  GPU KV cache size: 31,856 tokens, Maximum concurrency for 4,096 tokens per request: 7.78x
  1 GPU: 1 at a time   req/s 0.19  out tok/s  24.4  TTFT p50  276 p99  1099 ms  TPOT p50  38.6 p99  39.2 ms
  1 GPU: 2 req/s       req/s 1.23  out tok/s 157.4  TTFT p50 1296 p99  4429 ms  TPOT p50 214.0 p99 245.1 ms
  1 GPU: 4 req/s       req/s 1.31  out tok/s 167.2  TTFT p50 1802 p99 23789 ms  TPOT p50 236.2 p99 270.9 ms
  1 GPU: 8 req/s       req/s 1.32  out tok/s 169.0  TTFT p50 5011 p99 32217 ms  TPOT p50 243.2 p99 269.5 ms

Two T4s, tensor parallel (--tensor-parallel-size 2)
  ready after 154.7 s
  GPU KV cache size: 126,464 tokens, Maximum concurrency for 4,096 tokens per request: 30.88x
  TP=2: 1 at a time    req/s 0.33  out tok/s  42.7  TTFT p50  196 p99  1087 ms  TPOT p50  21.5 p99  21.8 ms
  TP=2: 2 req/s        req/s 1.62  out tok/s 207.1  TTFT p50  674 p99  4713 ms  TPOT p50 107.9 p99 153.1 ms
  TP=2: 4 req/s        req/s 2.77  out tok/s 354.8  TTFT p50  141 p99   230 ms  TPOT p50  56.6 p99  64.9 ms
  TP=2: 8 req/s        req/s 3.57  out tok/s 457.1  TTFT p50  189 p99   288 ms  TPOT p50  80.8 p99  85.4 ms

Two T4s, data parallel (--data-parallel-size 2)
  ready after 186.2 s
  GPU KV cache size: 31,856 tokens (per copy), Maximum concurrency for 4,096 tokens per request: 7.78x
  DP=2: 1 at a time    req/s 0.15  out tok/s  19.3  TTFT p50  390 p99  1275 ms  TPOT p50  44.7 p99  67.9 ms
  DP=2: 2 req/s        req/s 1.44  out tok/s 184.7  TTFT p50  785 p99  3505 ms  TPOT p50 126.1 p99 148.6 ms
  DP=2: 4 req/s        req/s 2.25  out tok/s 287.9  TTFT p50  394 p99  1553 ms  TPOT p50 102.2 p99 117.2 ms
  DP=2: 8 req/s        req/s 2.83  out tok/s 362.2  TTFT p50  278 p99   889 ms  TPOT p50 109.3 p99 128.3 ms
```

The T4 pair tells the L4 pair's story more sharply, because one T4 is so small: the split model halves the step
(38.6 to 21.5 ms), quadruples the shelf (31,856 to 126,464 tokens), and takes the 8 req/s row from 169 tokens a
second with a 32-second first token to 457 with a first token under 0.3 s. Two copies help less than the split
here, because each copy keeps the small shelf and the single T4's knee.
