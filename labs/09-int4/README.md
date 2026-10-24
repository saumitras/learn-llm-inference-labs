# Lab 09: INT4, quantised by you

Quantise Qwen3-0.6B to 4-bit weights with LLM Compressor (GPTQ, W4A16), score the original and the result with
lm_eval against the running server, then measure the decode step and the prefill.

- Modal: `modal run labs/09-int4/modal/lab.py` (quantises on one L4, then serves and scores on another)
- Kaggle: `kaggle/lab.ipynb`

## Expected output, Modal L4 (captured 24 October 2026, vLLM 0.32.0, llmcompressor 0.14.0.1, lm_eval 0.4.13)

```
1. Quantisation
GPTQ W4A16 on Qwen/Qwen3-0.6B with 128 wikitext samples: 15 s on an L4; checkpoint 0.54 GB against 1.50 GB

BF16 original: serve, score, measure
  quantization=none
  lm_eval hellaswag, 100 examples: acc_norm 0.48
  1 at a time, 1,024 in      req/s  1.17  out tok/s   149.9  TTFT p50   41 p99   47 ms  TPOT p50  6.4 p99  6.4 ms
  32 concurrent, 1,024 in    req/s  8.83  out tok/s  1129.8  TTFT p50  361 p99  668 ms  TPOT p50 25.9 p99 28.1 ms

W4A16 quantised: serve, score, measure
  quantization=compressed
  lm_eval hellaswag, 100 examples: acc_norm 0.47
  1 at a time, 1,024 in      req/s  1.85  out tok/s   236.9  TTFT p50   47 p99   53 ms  TPOT p50  3.9 p99  3.9 ms
  32 concurrent, 1,024 in    req/s  9.66  out tok/s  1236.5  TTFT p50  343 p99  632 ms  TPOT p50 23.6 p99 25.7 ms
```

The weights went to 36 percent of their size and the batch-1 step to 61 percent; the score on 100 HellaSwag
examples moved by one, inside the ±5 point error of so small a sample; the first token got 6 ms slower, which is
the cost of unpacking 4-bit weights to 16 bits during the arithmetic-bound prefill.
