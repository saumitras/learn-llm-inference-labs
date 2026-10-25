# Capstone A: tune to an SLO

The support bot from the capacity module: a 600-token system prompt shared by everyone, 1,300-token prompts in all,
300-token answers, 500 users at one message a minute, which is 8.3 requests a second. Targets: first token p99 at
or under 1,000 ms, time per output token p99 at or under 75 ms. Each configuration is swept until a p99 crosses a
target; the highest achieved rate inside the targets sizes the fleet; lm_eval gates the quality of the winner.

- Modal: `modal run labs/capstone-a-tune-to-an-slo/modal/lab.py` (about 25 minutes on one L4)
- Kaggle: `kaggle/lab.ipynb` (fp16 and the scheduler knob; the FP8 rows are read-only on the T4)

## Expected output, Modal L4 (captured 24 October 2026, vLLM 0.32.0, lm_eval 0.4.13)

```
Target: TTFT p99 <= 1000 ms and TPOT p99 <= 75 ms, prompts 1300 tokens (600 shared), answers 300 tokens, 8.3 req/s needed

BF16, defaults                         ready after 223.6 s   GPU KV cache size: 86,096 tokens
  MISS 2 req/s    req/s 1.37  out tok/s  411.9  TTFT p50 281 p99  591 ms  TPOT p50 66.7 p99 76.2 ms  peak conc 52
  OK   4 req/s    req/s 1.94  out tok/s  582.3  TTFT p50 185 p99  334 ms  TPOT p50 69.1 p99 72.6 ms  peak conc 70
  MISS 6 req/s    req/s 2.18  out tok/s  654.1  TTFT p50 190 p99  341 ms  TPOT p50 73.7 p99 76.3 ms  peak conc 72
  highest rate inside the target: 1.94 req/s achieved  ->  GPUs for 8.3 req/s: 4.3

FP8 weights                            ready after 105.3 s   GPU KV cache size: 110,272 tokens
  OK   2 req/s    req/s 1.55  out tok/s  464.8  TTFT p50 245 p99  567 ms  TPOT p50 46.1 p99 55.7 ms  peak conc 43
  OK   4 req/s    req/s 2.33  out tok/s  699.7  TTFT p50 137 p99  257 ms  TPOT p50 48.5 p99 51.8 ms  peak conc 59
  OK   6 req/s    req/s 2.58  out tok/s  772.9  TTFT p50 153 p99  317 ms  TPOT p50 57.6 p99 60.9 ms  peak conc 72
  OK   8 req/s    req/s 2.72  out tok/s  815.5  TTFT p50 163 p99  334 ms  TPOT p50 61.4 p99 64.2 ms  peak conc 72
  OK   10 req/s   req/s 2.79  out tok/s  838.2  TTFT p50 168 p99  335 ms  TPOT p50 64.5 p99 66.8 ms  peak conc 72
  OK   12 req/s   req/s 2.86  out tok/s  857.6  TTFT p50 162 p99  324 ms  TPOT p50 66.1 p99 68.2 ms  peak conc 72
  highest rate inside the target: 2.86 req/s achieved  ->  GPUs for 8.3 req/s: 2.9

FP8 weights + FP8 KV cache             ready after 204.5 s   GPU KV cache size: 206,720 tokens
  OK   2 req/s    req/s 1.64  out tok/s  493.3  TTFT p50 209 p99  533 ms  TPOT p50 34.7 p99 42.5 ms  peak conc 35
  OK   4 req/s    req/s 2.70  out tok/s  810.1  TTFT p50  92 p99  123 ms  TPOT p50 31.8 p99 33.6 ms  peak conc 49
  OK   6 req/s    req/s 3.23  out tok/s  969.6  TTFT p50  94 p99  146 ms  TPOT p50 37.0 p99 38.6 ms  peak conc 64
  OK   8 req/s    req/s 3.49  out tok/s 1048.4  TTFT p50  99 p99  168 ms  TPOT p50 41.1 p99 42.5 ms  peak conc 72
  OK   10 req/s   req/s 3.65  out tok/s 1096.2  TTFT p50  99 p99  174 ms  TPOT p50 43.5 p99 44.7 ms  peak conc 72
  MISS 12 req/s   req/s 3.41  out tok/s 1022.0  TTFT p50 393 p99 1151 ms  TPOT p50 55.1 p99 57.8 ms  peak conc 72
  highest rate inside the target: 3.65 req/s achieved  ->  GPUs for 8.3 req/s: 2.3

FP8 + FP8 KV, max-num-seqs 32          ready after 162.5 s   GPU KV cache size: 217,312 tokens
  OK   2 req/s    req/s 1.64  out tok/s  492.4  TTFT p50 202 p99  523 ms  TPOT p50 34.2 p99 41.9 ms  peak conc 34
  MISS 4 req/s    req/s 2.50  out tok/s  750.7  TTFT p50  91 p99 3350 ms  TPOT p50 29.4 p99 31.1 ms  peak conc 48
  highest rate inside the target: 1.64 req/s achieved  ->  GPUs for 8.3 req/s: 5.1

Summary
  BF16, defaults                      1.94 req/s per GPU inside the target   fleet for 500 users: 5 GPUs
  FP8 weights                         2.86 req/s per GPU inside the target   fleet for 500 users: 3 GPUs
  FP8 weights + FP8 KV cache          3.65 req/s per GPU inside the target   fleet for 500 users: 3 GPUs
  FP8 + FP8 KV, max-num-seqs 32       1.64 req/s per GPU inside the target   fleet for 500 users: 6 GPUs

Quality gate: lm_eval HellaSwag, 100 examples
  BF16: acc_norm 0.58
  FP8:  acc_norm 0.57
```

Each row is 72 requests. Above about 4 offered requests a second all 72 are in flight at once (peak conc 72), so
the achieved rate is the GPU's throughput at that configuration, and the rate column stops meaning arrivals and
starts meaning capacity. That is the number the fleet is sized from.

## Expected output, Kaggle T4 ×2 (captured 25 October 2026, vLLM 0.32.0, fp16; the FP8 configurations are not run on this GPU)

```
Target: TTFT p99 <= 1000 ms and TPOT p99 <= 75 ms, prompts 1300 tokens (600 shared), answers 300 tokens, 8.3 req/s needed

fp16, defaults                         ready after 222.3 s
  MISS 1 req/s    req/s 0.31  out tok/s  91.9  TTFT p50  3267 p99  73274 ms  TPOT p50 323.3 p99 361.7 ms
  MISS 2 req/s    req/s 0.32  out tok/s  95.4  TTFT p50  7410 p99  94043 ms  TPOT p50 335.5 p99 378.8 ms
  MISS 4 req/s    req/s 0.32  out tok/s  95.8  TTFT p50 12913 p99 105470 ms  TPOT p50 329.3 p99 381.0 ms
  MISS 8 req/s    req/s 0.32  out tok/s  95.7  TTFT p50 16564 p99 111319 ms  TPOT p50 327.1 p99 380.6 ms
  highest rate inside the target: 0.00 req/s  ->  GPUs for 8.3 req/s: inf

fp16, max-num-seqs 16                  ready after 120.1 s
  MISS 1 req/s    req/s 0.30  out tok/s  90.8  TTFT p50 36082 p99  73402 ms  TPOT p50 157.1 p99 174.2 ms
  MISS 2 req/s    req/s 0.31  out tok/s  94.4  TTFT p50 44645 p99  90685 ms  TPOT p50 159.1 p99 166.2 ms
  MISS 8 req/s    req/s 0.32  out tok/s  94.9  TTFT p50 55490 p99 107785 ms  TPOT p50 153.6 p99 166.0 ms
  highest rate inside the target: 0.00 req/s  ->  GPUs for 8.3 req/s: inf

Quality gate: lm_eval HellaSwag, 100 examples
  fp16 acc_norm 0.58
```

On the T4 the support bot's targets are out of reach at every rate tried: 48 of these 1,300-token prompts are
all in flight within seconds, the step is 330 ms, and the first token is a queue of a minute or more. Capping the
batch at 16 brings the step to 157 ms and moves the whole wait to the queue. The sweep returns no passing rate,
so the fleet size is undefined rather than large: this GPU cannot serve this bot inside these targets at any
count. The quality gate matches the L4's 0.58.
