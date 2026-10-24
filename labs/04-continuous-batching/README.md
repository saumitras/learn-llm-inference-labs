# Lab 04: continuous batching and the knee

A rate sweep with the running and waiting counts read from the server, then the same sweep with the batch capped.

- Modal: `modal run labs/04-continuous-batching/modal/lab.py`
- Kaggle: `kaggle/lab.ipynb`

## Expected output, Modal L4 (captured 24 October 2026, vLLM 0.32.0)

512-token prompts, 128-token answers, 60 prompts per rate.

```
1. Uncapped
offered  achieved  out tok/s  TTFT p50/p99 ms  TPOT p50 ms  peak running  waiting
 1 req/s   0.93      119        171 /  279        38.0           10           0
 2 req/s   1.73      221        106 /  128        37.4           14           0
 4 req/s   2.95      378        121 /  152        43.8           28           0
 6 req/s   3.79      485        131 /  174        48.2           40           0
 8 req/s   4.39      562        143 /  194        51.4           50           0
12 req/s   5.12      655        133 /  238        54.8           60           0

2. The server's log line
Running: 35 reqs, Waiting: 0 reqs, GPU KV cache usage: 24.7%, Prefix cache hit rate: 77.5%

3. Capped at 8 requests per step (--max-num-seqs 8)
 1 req/s   0.93      119        170 /   936       38.2            8           2
 2 req/s   1.46      187       1114 /  5960       37.0            8          10
 4 req/s   1.54      197       7046 / 19450       37.1            8          34
 6 req/s   1.55      198       9503 / 24160       37.0            8          40
```

Uncapped, all sixty requests of a burst fit in one step, nothing waits, and the step stretches from 38 to 55 ms
as the batch grows to sixty: the knee here is in the step time, not in a queue. Capped at eight, the queue
appears at once and the p99 first-token time goes from under a second to twenty-four.
