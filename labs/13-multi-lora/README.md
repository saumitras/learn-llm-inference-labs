# Lab 13: many fine-tunes on one base

Two rank-8 adapters made on the spot (random weights: the lab measures cost, not quality), served behind one base
model, and what each costs per step at batch 1 and with thirty-two requests in flight.

- Modal: `modal run labs/13-multi-lora/modal/lab.py`
- Kaggle: `kaggle/lab.ipynb`

## Expected output, Modal L4 (captured 24 October 2026, vLLM 0.32.0, peft 0.21.2)

```
1. Two rank-8 adapters, made here
  adapter legal: 23.6 MB
  adapter support: 23.6 MB

2. One server, the base plus both adapters  (--enable-lora --max-lora-rank 8 --lora-modules legal=... support=...)
ready after 221.5 s
base, 1 at a time            req/s  0.23  out tok/s  29.3  TTFT p50  111 p99  114 ms  TPOT p50  33.6 p99  33.6 ms
adapter legal, 1 at a time   req/s  0.21  out tok/s  26.8  TTFT p50  121 p99  151 ms  TPOT p50  36.7 p99  36.8 ms
adapter support, 1 at a time req/s  0.21  out tok/s  26.8  TTFT p50  121 p99  145 ms  TPOT p50  36.7 p99  36.8 ms

3. Thirty-two in flight
base x32 (benchmark tool)                 out tok/s  509.6  TTFT p50 1098 p99 1962 ms  TPOT p50  54.7 p99  63.5 ms
mixed x32, 16 legal + 16 support (client) out tok/s  368.3                              TPOT p50  49.3 / 47.1 ms
```

An adapter adds 3 ms to the 33.6 ms step at batch 1, about 9 percent, for 23.6 MB of extra weights on the
attention projections. The mixed row comes from the course's own load generator rather than the benchmark tool,
so compare its step (47 to 49 ms) with the base row's (55 ms), not its throughput.
