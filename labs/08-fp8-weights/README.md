# Lab 08: FP8 weights

The official FP8 checkpoint against the BF16 one: start time, shelf size, the batch-1 step, and the same rate sweep
on both.

- Modal: `modal run labs/08-fp8-weights/modal/lab.py`
- Kaggle: `kaggle/lab.ipynb` (the T4 has no FP8 arithmetic: the notebook shows the refusal and runs the fp16 half)

## Expected output, Modal L4 (captured 24 October 2026, vLLM 0.32.0)

```
BF16: Qwen/Qwen3-4B-Instruct-2507
ready after 217.5 s
  GPU KV cache size: 86,096 tokens
1 at a time, 1,024 in      req/s  0.22  out tok/s    28.8  TTFT p50  174 p99  215 ms  TPOT p50 33.6 p99 33.7 ms  peak conc  2
2 req/s, 512 in            req/s  1.72  out tok/s   220.1  TTFT p50  165 p99  259 ms  TPOT p50 42.9 p99 47.3 ms  peak conc 19
4 req/s, 512 in            req/s  2.95  out tok/s   377.9  TTFT p50  124 p99  159 ms  TPOT p50 43.6 p99 46.0 ms  peak conc 33
8 req/s, 512 in            req/s  4.39  out tok/s   561.3  TTFT p50  137 p99  208 ms  TPOT p50 51.4 p99 53.6 ms  peak conc 52
12 req/s, 512 in           req/s  5.13  out tok/s   656.5  TTFT p50  148 p99  234 ms  TPOT p50 54.7 p99 56.3 ms  peak conc 60

FP8: Qwen/Qwen3-4B-Instruct-2507-FP8
ready after 162.4 s
  quantization=fp8
  GPU KV cache size: 105,920 tokens
1 at a time, 1,024 in      req/s  0.37  out tok/s    47.6  TTFT p50  202 p99  218 ms  TPOT p50 19.6 p99 19.6 ms  peak conc  2
2 req/s, 512 in            req/s  1.82  out tok/s   233.6  TTFT p50  143 p99  243 ms  TPOT p50 24.8 p99 29.7 ms  peak conc 13
4 req/s, 512 in            req/s  3.34  out tok/s   427.8  TTFT p50   71 p99   90 ms  TPOT p50 23.2 p99 25.3 ms  peak conc 22
8 req/s, 512 in            req/s  5.37  out tok/s   687.5  TTFT p50   90 p99  141 ms  TPOT p50 30.8 p99 33.0 ms  peak conc 40
12 req/s, 512 in           req/s  6.47  out tok/s   828.7  TTFT p50  109 p99  207 ms  TPOT p50 36.2 p99 38.6 ms  peak conc 55
```

Each sweep row is 60 prompts of 512 tokens with 128-token answers. The FP8 weights are 5.19 GB against 8.04, the
batch-1 step is 19.6 ms against 33.6, and the shelf grew from 86,096 to 105,920 tokens because the weights left
2.85 GB free.
