# Lab 03: the overhead floor

Batch-1 decode step with kernels launched one by one and with the launch sequence replayed, on the 4B model and the
0.6B model, beside the bytes-over-bandwidth prediction.

- Modal: `modal run labs/03-overhead-floor/modal/lab.py` (four configurations on four GPUs in parallel)
- Kaggle: `kaggle/lab.ipynb`, one configuration per cell

## Expected output, Modal L4 (captured 24 October 2026, vLLM 0.32.0)

```
model                        launches     predicted   measured    floor
Qwen/Qwen3-4B-Instruct-2507  one by one     26.8 ms    33.9 ms   7.1 ms
Qwen/Qwen3-4B-Instruct-2507  replayed       26.8 ms    33.0 ms   6.2 ms
Qwen/Qwen3-0.6B              one by one      5.0 ms    26.5 ms  21.5 ms
Qwen/Qwen3-0.6B              replayed        5.0 ms     5.9 ms   0.9 ms
```

The 0.6B model with launches one by one takes 26.5 ms a step, five times its prediction: the step is almost all
launch overhead. Replayed, it drops to 5.9 ms, within a millisecond of the prediction. The 4B model barely notices
the replay because its step is dominated by the weight read either way.
