# Capstone C: cost per million tokens

Five configurations, each driven from one request at a time up to 128 at once with 1,024-token prompts and
256-token answers, and the price of a million tokens at every point from the GPU's hourly rate ($0.80 per L4-hour,
the Modal rate the capacity module uses).

- Modal: `modal run labs/capstone-c-cost-per-million-tokens/modal/lab.py` (one L4 for the first three, a two-L4 machine for the last two; about 30 minutes)
- Kaggle: `kaggle/lab.ipynb` (fp16 on one and two T4s, priced at a typical cloud T4 rate; the FP8 rows are L4-only)

## Expected output, Modal L4 (captured 24 October 2026, vLLM 0.32.0)

```
One L4, BF16                               shelf 86,096 tokens
   concurrency  out tok/s  TPOT p50  TTFT p99  $ per 1M output tokens  $ per 1M total tokens
             1       28.9     33.9ms      199ms                   7.676                  1.535
             8      201.5     39.3ms      156ms                   1.103                  0.221
            32      425.4     67.8ms     3285ms                   0.522                  0.104
            64      603.7     86.5ms    10981ms                   0.368                  0.074
           128      463.8    145.7ms    49755ms                   0.479                  0.096

One L4, FP8 weights                        shelf 110,272 tokens
             1       48.8     19.8ms      222ms                   4.557                  0.911
             8      322.4     24.5ms      113ms                   0.689                  0.138
            32      508.0     54.6ms     3972ms                   0.437                  0.087
            64      655.7     74.0ms    13218ms                   0.339                  0.068
           128      450.9    199.1ms    59871ms                   0.493                  0.099

One L4, FP8 weights + FP8 KV cache         shelf 206,720 tokens
             1       49.4     19.5ms      222ms                   4.502                  0.900
             8      359.2     21.9ms      113ms                   0.619                  0.124
            32      621.1     44.4ms     3697ms                   0.358                  0.072
            64      850.7     55.5ms    11715ms                   0.261                  0.052
           128      959.9     91.3ms    24304ms                   0.232                  0.046

Two L4s, FP8, tensor parallel ($1.60/h)
             1       87.1     11.0ms      152ms                   5.105                  1.021
             8      607.4     12.9ms      104ms                   0.732                  0.146
            32      988.6     28.8ms     2240ms                   0.450                  0.090
            64     1357.5     36.9ms     6247ms                   0.327                  0.065
           128     1500.6     62.5ms    13648ms                   0.296                  0.059

Two L4s, FP8, two copies ($1.60/h)
             1       50.1     19.3ms      198ms                   8.866                  1.773
             8      355.4     21.0ms      601ms                   1.250                  0.250
            32      907.0     30.5ms     2161ms                   0.490                  0.098
            64     1294.3     39.1ms     5329ms                   0.343                  0.069
           128     1542.0     68.3ms     9472ms                   0.288                  0.058
```

A million output tokens costs $7.68 from a GPU serving one request at a time and 23 cents from the same GPU
with FP8 weights, an FP8 cache and 128 requests in flight. The first two configurations got slower at 128 than at
64: 128 requests of 1,280 tokens need 164,000 cache tokens, more than their shelves hold, so requests were
preempted and re-prefilled. The total-token price is the output price times 256/1,280.

## Expected output, Kaggle T4 ×2 (captured 25 October 2026, vLLM 0.32.0, fp16, priced at $0.35 per T4-hour)

```
One T4                                     ready after 215.3 s
   concurrency  out tok/s  TPOT p50  TTFT p99  $ per 1M output tokens  $ per 1M total tokens
             1       19.8    47.1ms    1548ms                   4.915                  0.983
             8       96.9    78.0ms    2194ms                   1.003                  0.201
            32      107.6   207.3ms   62029ms                   0.904                  0.181
            64      108.6   227.5ms  135697ms                   0.895                  0.179

Two T4s, tensor parallel ($0.70/h)         ready after 159.7 s
             1       38.1    23.7ms    1359ms                   5.100                  1.020
             8      155.2    41.3ms    2890ms                   1.253                  0.251
            32      213.0   125.9ms   12481ms                   0.913                  0.183
            64      284.9   178.4ms   27211ms                   0.683                  0.137

Two T4s, two copies ($0.70/h)             ready after 265.4 s   (second session, same day)
             1       19.9    44.3ms    1767ms                   9.776                  1.955
             8      100.4    61.0ms    4826ms                   1.936                  0.387
            32      203.9   132.6ms   10933ms                   0.954                  0.191
            64      203.3   229.4ms   70483ms                   0.957                  0.191
```

A T4 is cheap and slow: a million output tokens costs $4.92 one at a time and about 90 cents once the GPU is
saturated at 32 in flight, where it stops gaining. Two copies are two of those T4s: double the price while one
request leaves a GPU idle, then 95 cents, because each copy tops out at one T4's hundred tokens a second and keeps
one T4's 31,856-token shelf. The split pair reaches 68 cents at 64 in flight because splitting the weights frees
memory on both GPUs: its pooled shelf of 126,464 tokens is twice what the two copies hold between them, so all 64
requests stay resident while the copies make some of theirs wait.
