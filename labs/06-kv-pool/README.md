# Lab 06: the KV pool, the length limit and preemption

Start without a length limit and read the refusal; start with one and read the shelf; then overfill it and watch
the preemption counter.

- Modal: `modal run labs/06-kv-pool/modal/lab.py`
- Kaggle: `kaggle/lab.ipynb`

Prefix caching is turned off for the overfill step, because 48 requests with the same prompt would otherwise
share one copy of its blocks and the shelf would never fill.

## Expected output, Modal L4 (captured 24 October 2026, vLLM 0.32.0)

```
1. Start with no length limit: the model card's 262,144
refused: ValueError: To serve at least one request with the model's max seq len (262144), (36.0 GiB KV cache
is needed, which is larger than the available KV cache memory (11.82 GiB). Based on the available memory, the
estimated maximum model length is 86080. Try increasing `gpu_memory_utilization` or decreasing `max_model_len`

2. Start with --max-model-len 4096: the shelf
  Available KV cache memory: 12.19 GiB
  GPU KV cache size: 88,752 tokens, Maximum concurrency for 4,096 tokens per request: 21.67x
  arithmetic: 12.2 GiB ÷ 147,456 bytes per token ≈ 88,800 tokens; 4,096 per request → 21 at once

3. Overfill it: 48 requests of ~3,000 prompt tokens plus 600 output
second  running  waiting  kv%   preemptions
     1        2       27     7            0
     5        9       39    32            0
    10       16       32    60            0
    15       23       25    86            0
    17       25       23    97            0
    25       25       23   100            0
    27       24       24    96            1
    39       23       25    96            2
worst gap between two tokens in any stream: 42776 ms
```

The scheduler admits requests as blocks allow: about 25 fit, 23 wait. As the 25 answers grow the shelf reaches
100 percent, and at second 27 a running request is paused and its blocks given away: the first preemption. The
worst gap in any stream is 43 seconds, a request that waited for the shelf.

## Expected output, Kaggle T4 ×2 (captured 24 and 25 October 2026, vLLM 0.32.0, `--dtype half`)

```
1. Start with no length limit
refused: ValueError: To serve at least one request with the model's max seq len (262144), (36.0 GiB KV cache
is needed, which is larger than the available KV cache memory (4.37 GiB). Based on the available memory, the
estimated maximum model length is 31840.

2. Start with --max-model-len 4096: the shelf
ready after 221.3 s
  Available KV cache memory: 4.74 GiB
  GPU KV cache size: 34,496 tokens, Maximum concurrency for 4,096 tokens per request: 8.42x

3. Overfill it: 48 requests of ~3,000 prompt tokens plus 600 output (every fifth second for three minutes; selected rows)
second  running  waiting  kv%   preemptions
     5        2       46    12            0
    20        5       43    42            0
    40        8       40    77            0
    50       10       38    95            0
    55       10       38   100            0
    60        9       39    91            1
    90        9       39    96            1
   115        9       39   100            1
   120        8       40    90            2
   150        7       39    81            2
   180        7       33    69            2
worst gap between two tokens in any stream: 101627 ms
```

The refusal and the shelf were captured on 24 October and the three-minute watch on 25 October. The refusal names
31,840 as the longest request this GPU could serve, against 86,080 on the L4, and the shelf is 34,496 tokens, about
nine of these requests once they have grown. The T4 prefills a 3,000-token prompt slowly, so requests are admitted
two or three at a time. At 55 seconds ten are running and the shelf is full. Five seconds later the engine preempts
one: running drops to nine, waiting rises by one and the shelf falls to 91 percent. The same happens again at 120
seconds. The worst gap in any stream, 101.6 seconds, is a preempted request waiting to be readmitted and to
recompute its cache. The streams still queued when the client's timeout ran out end in timeout errors, which is the
same queue seen from the client.
