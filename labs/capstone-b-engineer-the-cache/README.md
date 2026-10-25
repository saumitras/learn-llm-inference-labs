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

## Expected output, Kaggle T4 ×2 (captured 25 October 2026, vLLM 0.32.0, `--dtype half`)

```
1. One engine on GPU 0, 24 sessions, careless layout, then careful
  careless:  hit rate  0%   TTFT p50 19690 p99  64157 ms   per-turn p50 ms t0:12016 t1:9186 t2:13204 t3:26843 t4:39137 t5:57646   354 s
  careful:   hit rate 61%   TTFT p50 11420 p99  50021 ms   per-turn p50 ms t0:2028 t1:8668 t2:12472 t3:14601 t4:15955 t5:33952   214 s

2. The same careful layout, 48 sessions at once
             hit rate 53%   TTFT p50 25396 p99 123125 ms   per-turn p50 ms t0:2294 t1:11341 t2:16026 t3:40595 t4:66436 t5:105312   487 s

3. Two engines, one per T4, 48 careful sessions
  second engine ready after 81.6 s
  round-robin:     hit rate 52%   TTFT p50 17952 p99 88322 ms   per-turn p50 ms t0:3002 t1:12558 t2:26908 t3:24115 t4:39388 t5:22347   343 s
  session-sticky:  hit rate 64%   TTFT p50  9918 p99 50528 ms   per-turn p50 ms t0:1653 t1:6109 t2:9603 t3:13873 t4:15391 t5:29757   213 s
```

The ranking is the L4's: the careless layout hits nothing, the careful one most of the time, round-robin across
two engines loses the history and session-sticky keeps it, finishing in 213 s against 343. Every time is ten
times the L4's, because 24 to 48 agents re-sending contexts of 1,000 to 3,000 tokens is far past a T4's prefill
rate, and the hit rates are a little lower because the 31,856-token shelf evicts older sessions' blocks while
the T4 is still prefilling the newer ones.
