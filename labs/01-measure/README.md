# Lab 01: measure time to first token, time per output token and throughput

The engine's own benchmark tool against the running server, four ways: one request at a time with a 1,024-token
prompt, one at a time with a 64-token prompt, two requests a second, eight requests a second. Read the p50, p95 and
p99 lines, not the mean.

- Modal: `modal run labs/01-measure/modal/lab.py`
- Kaggle: `kaggle/lab.ipynb`, cells in order

Each run uses `vllm bench serve` with random prompts, 128-token answers, `--ignore-eos`, and
`--metric-percentiles 50,95,99`.

## Expected output, Modal L4 (captured 24 October 2026, vLLM 0.32.0)

| Run | Request rate | TTFT p50 / p99 | TPOT p50 / p99 | Output tok/s | Peak concurrent |
| --- | --- | --- | --- | --- | --- |
| 1,024-token prompt, one at a time | – | 181 / 186 ms | 33.8 / 33.8 ms | 29 | 1 |
| 64-token prompt, one at a time | – | 45 / 49 ms | 33.2 / 33.3 ms | 30 | 1 |
| 2 requests a second, 48 prompts | 2 | 262 / 571 ms | 50 / 70 ms | 205 | 23 |
| 8 requests a second, 96 prompts | 8 | 390 ms / 10.5 s | 129 / 154 ms | 408 | 89 |

What to see: the first-token time is the prefill, so it drops from 181 ms to 45 ms when the prompt shrinks and the
step time does not move. At two requests a second the server keeps up and everything is tight. At eight it cannot:
the p99 first-token time is ten seconds because requests queue, and the step is four times longer because the
batch is near ninety. The mean TTFT at eight per second, 1.9 s, describes nobody.

The server was ready after 299 s on this run, a cold start with an empty compile cache; later runs start faster.

## Expected output, Kaggle T4 ×2 (captured 24 October 2026, vLLM 0.32.0, `--dtype half`)

| Run | Request rate | TTFT p50 / p99 | TPOT p50 / p99 | Output tok/s | Peak concurrent |
| --- | --- | --- | --- | --- | --- |
| 1,024-token prompt, one at a time | – | 753 / 1,541 ms | 43.8 / 60.6 ms | 18.7 | 1 |
| 64-token prompt, one at a time | – | 57 / 66 ms | 66.3 / 127 ms | 13.5 | 1 |
| 2 requests a second, 48 prompts | 2 | 5.3 s / 34.2 s | 290 / 361 ms | 83 | 48 |
| 8 requests a second, 96 prompts | 8 | 54.0 s / 126.8 s | 301 / 433 ms | 82 | 96 |

The server was ready after 213 s. The first-token time on the 1,024-token prompt is 753 ms against 181 on the L4:
prefill is arithmetic, and the T4 has about a seventh of the L4's. The knee is already behind two requests a
second: the 31,856-token shelf holds about 27 of these requests, so with 48 in flight the server preempts and
re-prefills, and the first-token p99 is half a minute. The T4's step at 64-token prompts came out slower than at
1,024, 66 against 44 ms; the run is noisy at this scale and the number is what the tool printed.
