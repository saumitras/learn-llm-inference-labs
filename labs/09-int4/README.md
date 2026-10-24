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

## Expected output, Kaggle T4 ×2 (captured 25 October 2026, vLLM 0.32.0, fp16)

```
1. Quantisation
GPTQ W4A16 on Qwen/Qwen3-0.6B with 128 wikitext samples: 23 s on a T4; checkpoint 0.54 GB against 1.50 GB

fp16 original: ready after 120.6 s
  quantization=none
  hellaswag acc_norm 0.50
  1 at a time, 1,024 in      req/s  0.60  out tok/s   76.6  TTFT p50  248 p99   364 ms  TPOT p50  11.2 p99  11.3 ms
  32 concurrent, 1,024 in    req/s  1.36  out tok/s  174.4  TTFT p50 4494 p99  9819 ms  TPOT p50 137.0 p99 167.9 ms

W4A16 quantised: ready after 128.6 s
  quantization=compressed
  hellaswag acc_norm 0.45
  1 at a time, 1,024 in      req/s  0.76  out tok/s   97.9  TTFT p50  251 p99   380 ms  TPOT p50   8.3 p99   8.4 ms
  32 concurrent, 1,024 in    req/s  1.38  out tok/s  176.9  TTFT p50 5170 p99 11777 ms  TPOT p50 128.8 p99 171.6 ms
```

The notebook's servers run with `--gpu-memory-utilization 0.8`: after quantising in the same process the
notebook holds about two gigabytes, and the default 0.9 is refused with "Free memory on device ... is less than
desired". The step went from 11.2 to 8.3 ms, the score from 0.50 to 0.45 on 100 examples, inside the sample's
error; at thirty-two at once the T4 is far past its knee either way, and the two checkpoints tie.
