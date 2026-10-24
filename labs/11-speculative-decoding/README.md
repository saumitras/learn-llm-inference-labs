# Lab 11: speculative decoding, three ways

n-gram lookup, Qwen3-0.6B as a draft model, and an EAGLE3 head, each against the plain server: time per token at
batch 1 on a code prompt and a prose prompt, the acceptance rate from the counters, and thirty-two requests at once.

- Modal: `modal run labs/11-speculative-decoding/modal/lab.py`
- Kaggle: `kaggle/lab.ipynb`

## Expected output, Modal L4 (captured 24 October 2026, vLLM 0.32.0)

```
none
  batch 1, code : 200 tokens in  6.6 s =  30.2 tok/s, time per token  32.9 ms
  batch 1, prose: 198 tokens in  6.6 s =  30.0 tok/s, time per token  33.2 ms
  32 concurrent, 512 in      req/s  3.81  out tok/s   488.0  TTFT p50   926 p99  2517 ms  TPOT p50  57.5 p99  63.1 ms

n-gram  (num_speculative_tokens 5, prompt_lookup 2..5)
  batch 1, code : 182 tokens in  7.5 s =  24.1 tok/s, time per token  41.2 ms
  batch 1, prose: 130 tokens in  4.7 s =  27.6 tok/s, time per token  36.1 ms
  drafted 1560 tokens, accepted 90: acceptance 6%
  32 concurrent, 512 in      req/s  2.87  out tok/s   367.0  TTFT p50   498 p99  9228 ms  TPOT p50  36.5 p99  90.9 ms

draft model 0.6B  (Qwen/Qwen3-0.6B, num_speculative_tokens 5)
  batch 1, code : 62 tokens in  7.4 s =   8.4 tok/s, time per token  96.4 ms
  batch 1, prose: 101 tokens in 10.5 s =   9.6 tok/s, time per token 101.3 ms
  drafted 810 tokens, accepted 236: acceptance 29%
  32 concurrent, 512 in      req/s  2.23  out tok/s   285.8  TTFT p50  2071 p99 10426 ms  TPOT p50  42.2 p99 107.9 ms

EAGLE3 head  (taobao-mnn/Qwen3-4B-Instruct-2507-Eagle3, num_speculative_tokens 3)
  batch 1, code : 100 tokens in  4.9 s =  20.5 tok/s, time per token  37.9 ms
  batch 1, prose: 101 tokens in  3.9 s =  25.6 tok/s, time per token  38.7 ms
  drafted 600 tokens, accepted 200: acceptance 33%
  32 concurrent, 512 in      req/s  2.69  out tok/s   343.8  TTFT p50   689 p99  9154 ms  TPOT p50  42.2 p99 107.9 ms
```

None of the three beat the plain server on this GPU with these prompts. The counters say why: n-gram found
almost nothing to copy in fresh prose and code (6 percent accepted), the 0.6B draft model accepted 29 percent but
its own five steps cost more than the verification saved, and the EAGLE3 head accepted a third of its drafts and
still landed 5 ms per token behind. Token counts differ between runs because speculative decoding changes the
sampled sequence slightly at temperature zero only when ties break differently; the model stops when it stops.

## Expected output, Kaggle T4 ×2 (captured 25 October 2026, vLLM 0.32.0, `--dtype half`)

```
none              ready after 235.3 s
  batch 1, code : 200 tokens in  8.2 s =  24.5 tok/s      batch 1, prose: 198 tokens in  7.1 s =  27.8 tok/s
  32 concurrent, 512 in   req/s 1.17  out tok/s 149.6  TTFT p50  4519 p99 16657 ms  TPOT p50 154.2 p99 206.9 ms
n-gram            ready after 158.7 s
  batch 1, code : 179 tokens in  8.2 s =  21.9 tok/s      batch 1, prose: 130 tokens in  5.3 s =  24.3 tok/s
  drafted 1545 tokens, accepted 91: acceptance 6%
  32 concurrent, 512 in   req/s 1.38  out tok/s 176.7  TTFT p50  1464 p99 11093 ms  TPOT p50 103.6 p99 277.7 ms
draft model 0.6B  ready after 209.2 s
  batch 1, code :  62 tokens in  9.3 s =   6.6 tok/s      batch 1, prose:  80 tokens in 10.8 s =   7.4 tok/s
  drafted 705 tokens, accepted 261: acceptance 37%
  32 concurrent, 512 in   req/s 0.90  out tok/s 115.0  TTFT p50 10745 p99 25674 ms  TPOT p50 123.8 p99 321.4 ms
EAGLE3 head       ready after 185.2 s
  batch 1, code : 101 tokens in  5.2 s =  19.2 tok/s      batch 1, prose: 101 tokens in  4.6 s =  22.1 tok/s
  drafted 603 tokens, accepted 200: acceptance 33%
  32 concurrent, 512 in   req/s 1.36  out tok/s 173.8  TTFT p50  1682 p99 13009 ms  TPOT p50 107.7 p99 258.8 ms
```

The same verdict as on the L4, with the same acceptance rates to within a few points: nothing beat the plain
server at batch 1. The EAGLE3 head loaded and ran on the T4. The thirty-two-at-once rows are all past the T4's
knee and differ mostly in queueing; the plain server's slow first row is the one that paid the Triton kernel's
first compilation.
