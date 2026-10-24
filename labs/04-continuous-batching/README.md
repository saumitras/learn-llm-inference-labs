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

## Expected output, Kaggle T4 ×2 (captured 24 October 2026, vLLM 0.32.0, `--dtype half`)

```
6. The sweep (512-token prompts, 128-token answers, 60 prompts per rate)
ready after 216.8 s
  1 req/s offered  req/s  0.84  out tok/s   107.9  TTFT p50    508 p99   2818 ms  TPOT p50  98.5 ms  peak running  21 waiting   0
  2 req/s offered  req/s  1.24  out tok/s   158.6  TTFT p50    863 p99   2083 ms  TPOT p50 189.8 ms  peak running  52 waiting   0
  4 req/s offered  req/s  1.25  out tok/s   160.5  TTFT p50   1980 p99  12816 ms  TPOT p50 241.3 ms  peak running  59 waiting   9
  6 req/s offered  req/s  1.29  out tok/s   164.5  TTFT p50   3874 p99   7589 ms  TPOT p50 238.8 ms  peak running  60 waiting  17
  8 req/s offered  req/s  1.29  out tok/s   165.2  TTFT p50   5082 p99   9238 ms  TPOT p50 237.4 ms  peak running  60 waiting  23
 12 req/s offered  req/s  1.30  out tok/s   166.3  TTFT p50   6468 p99  11563 ms  TPOT p50 238.2 ms  peak running  60 waiting  33
  Running: 60 reqs, Waiting: 0 reqs, GPU KV cache usage: 99.5%, Prefix cache hit rate: 4.9%
  Running: 53 reqs, Waiting: 7 reqs, GPU KV cache usage: 98.4%, Prefix cache hit rate: 4.9%

7. The same sweep with the batch capped at 8
ready after 122.1 s
  1 req/s offered  req/s  0.73  out tok/s    93.5  TTFT p50   3461 p99  14925 ms  TPOT p50  75.1 ms  peak running   8 waiting  13
  2 req/s offered  req/s  0.77  out tok/s    98.1  TTFT p50  15330 p99  41951 ms  TPOT p50  75.3 ms  peak running   8 waiting  35
  4 req/s offered  req/s  0.77  out tok/s    98.6  TTFT p50  23767 p99  56594 ms  TPOT p50  73.7 ms  peak running   8 waiting  44
  6 req/s offered  req/s  0.77  out tok/s    98.8  TTFT p50  26500 p99  61507 ms  TPOT p50  71.9 ms  peak running   8 waiting  49
```

On the T4 the knee is at about 1.3 requests a second and 165 tokens a second, and here the queue does form: the
shelf is 31,856 tokens, sixty of these requests fill it to 99.5 percent, and from four a second the waiting count
climbs while the throughput does not. The cap at 8 keeps the step at 75 ms and sends everything else to the queue:
a first token of 15 to 60 seconds, the shape the L4 run also showed, with a lower ceiling.
