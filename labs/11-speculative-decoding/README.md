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
