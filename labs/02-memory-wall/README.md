# Lab 02: the memory wall

Watch the GPU read its weights once per token, and see why the utilisation figure says nothing.

- Modal: `modal run labs/02-memory-wall/modal/lab.py`
- Kaggle: `kaggle/lab.ipynb`

## Expected output, Modal L4 (captured 24 October 2026, vLLM 0.32.0)

```
2. Decode at batch 1
decode step: 32.8 ms   tokens/s: 30
weights read per step: 8.04 GB  ->  implied read rate 245 GB/s  (spec sheet 300 GB/s: 82%)
arithmetic per step: 0.008 TFLOP  ->  0.2 TFLOP/s of 121 available (0.2%)

3. Decode at batch 32
decode step: 38.5 ms   tokens/s: 832
weights read per step: 8.04 GB  ->  implied read rate 209 GB/s  (spec sheet 300 GB/s: 70%)
arithmetic per step: 0.257 TFLOP  ->  6.7 TFLOP/s of 121 available (5.5%)

4. nvidia-smi dmon during a long batch-1 decode
second  sm%  mem%
     2   100   100
     3   100   100
     4   100   100
```

At batch 1 the GPU reads its weights at 82 percent of the spec-sheet bandwidth and uses 0.2 percent of its
arithmetic. At batch 32 the arithmetic is still at 5 percent. The two utilisation columns read 100 both ways.
