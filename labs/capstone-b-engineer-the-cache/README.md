# Capstone B: engineer the cache

A multi-turn support agent: a 1,000-token system prompt (policy and tool list), tool results that accumulate turn
by turn, six turns per session. One engine with the prompt laid out carelessly (timestamp and tool results first),
then carefully (everything static first, everything that changes last); then two engines, one per GPU, with 48
sessions routed round-robin per request, then session-sticky.

- Modal: `modal run labs/capstone-b-engineer-the-cache/modal/lab.py` (a two-L4 machine, about 15 minutes)
- Kaggle: `kaggle/lab.ipynb` (runs as written on the two T4s)

## Expected output, Modal L4 ×2 (captured 24 October 2026, vLLM 0.32.0)

```
1. One engine, 24 sessions, careless layout, then careful
  careless:  hit rate   1%   TTFT p50  2161 p99  3888 ms   per-turn p50 ms t0:2505 t1:1766 t2:1805 t3:1856 t4:2275 t5:3128   53 s
  careful:   hit rate  71%   TTFT p50  1126 p99  1790 ms   per-turn p50 ms t0: 408 t1:1179 t2:1409 t3:1445 t4:1442 t5:1562   21 s

2. The same careful layout, 48 sessions at once
             hit rate  75%   TTFT p50  1490 p99  3545 ms   per-turn p50 ms t0: 240 t1: 485 t2:1488 t3:1608 t4:2494 t5:2371   37 s

3. Two engines, one per GPU, 48 careful sessions
  round-robin:     hit rate  67%   TTFT p50  1976 p99  9336 ms   per-turn p50 ms t0: 446 t1:1807 t2:1615 t3:3889 t4:4066 t5:4583   40 s
  session-sticky:  hit rate  80%   TTFT p50   760 p99  2733 ms   per-turn p50 ms t0: 241 t1: 627 t2: 760 t3:1257 t4: 925 t5:1619   20 s
```

All sessions run their turns in parallel, so each first-token time includes queueing behind the other sessions'
prefills; that is the point. The careless layout hits 1 percent because the timestamp at the top changes every
block after it. Round-robin across two engines keeps the shared system prompt warm on both but loses each
session's history half the time; sticky routing keeps it, and two engines with sticky routing finish the same
work in half the time of one.
