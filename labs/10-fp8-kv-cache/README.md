# Lab 10: the FP8 KV cache

The BF16 cache and the FP8 cache on the same weights: the shelf size, the step at short and long context, and
thirty-two long prompts at once.

- Modal: `modal run labs/10-fp8-kv-cache/modal/lab.py`
- Kaggle: `kaggle/lab.ipynb` (the T4 refuses the FP8 cache: the notebook shows the refusal and measures the fp16 cache)

## Expected output, Modal L4 (captured 24 October 2026, vLLM 0.32.0)

```
BF16 cache
ready after 233.0 s
  Available KV cache memory: 11.82 GiB
  GPU KV cache size: 86,096 tokens, Maximum concurrency for 8,192 tokens per request: 10.51x
1 at a time, 512 in        req/s  0.23  out tok/s    29.4  TTFT p50   107 p99   115 ms  TPOT p50  33.5 p99  33.6 ms  peak conc  2
1 at a time, 6,000 in      req/s  0.10  out tok/s    24.6  TTFT p50  1077 p99  1108 ms  TPOT p50  36.6 p99  36.7 ms  peak conc  2
32 concurrent, 2,048 in    req/s  1.63  out tok/s   208.6  TTFT p50  4010 p99  8709 ms  TPOT p50 132.3 p99 154.9 ms  peak conc 37
  sample: The sky appears blue because molecules in the Earth's atmosphere scatter shorter blue wavelengths of sunlight more than longer wavelengths, like red or yellow.

FP8 cache
ready after 195.4 s
  Available KV cache memory: 10.8 GiB
  GPU KV cache size: 157,312 tokens, Maximum concurrency for 8,192 tokens per request: 19.20x
1 at a time, 512 in        req/s  0.23  out tok/s    29.4  TTFT p50   110 p99   115 ms  TPOT p50  33.4 p99  33.4 ms  peak conc  2
1 at a time, 6,000 in      req/s  0.10  out tok/s    25.7  TTFT p50  1034 p99  1049 ms  TPOT p50  35.0 p99  35.1 ms  peak conc  2
32 concurrent, 2,048 in    req/s  1.98  out tok/s   253.7  TTFT p50  3773 p99  7714 ms  TPOT p50 106.0 p99 126.7 ms  peak conc 41
  sample: The sky appears blue because molecules in the atmosphere scatter blue light more than other colors due to its shorter wavelength.
```

Same weights, same memory: the shelf went from 86,096 to 157,312 tokens, 1.83 times, with the batch-1 step
unchanged at 33 ms. Thirty-two 2,048-token prompts at once got 22 percent more tokens a second and a step of
106 ms against 132, because each step reads half the cache bytes.

## Expected output, Kaggle T4 ×2 (captured 25 October 2026, vLLM 0.32.0, `--dtype half`)

```
--kv-cache-dtype fp8
  ValueError: FP8 KV cache is not supported by the Triton attention backend on Tesla T4 (compute capability 7.5);
  native FP8 (fp8e4nv) requires SM89+. Re-run with --kv-cache-dtype float16.
  refused: vllm serve exited

fp16 cache, --max-model-len 8192
ready after 183.7 s
  Available KV cache memory: 4.38 GiB
  GPU KV cache size: 31,856 tokens, Maximum concurrency for 8,192 tokens per request: 3.89x
1 at a time, 512 in        req/s 0.15  out tok/s 19.5  TTFT p50   302 p99  1105 ms  TPOT p50  46.1 p99  64.8 ms
1 at a time, 6,000 in      req/s 0.03  out tok/s  7.8  TTFT p50 15923 p99 16144 ms  TPOT p50  66.2 p99  67.8 ms
16 concurrent, 2,048 in    req/s 0.26  out tok/s 33.5  TTFT p50 18907 p99 48382 ms  TPOT p50 305.7 p99 455.8 ms
```

The refusal is the engine's own sentence: the 8-bit cache needs compute capability 8.9. The fp16 shelf holds
31,856 tokens, under four 8,192-token requests, and the 6,000-token prompt takes 16 seconds to prefill. With
sixteen 2,048-token prompts at once the T4 is past its shelf as well as its knee: a 306 ms step and a 49-second
first token at the tail.
