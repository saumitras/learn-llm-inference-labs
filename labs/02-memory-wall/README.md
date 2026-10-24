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

## Expected output, Kaggle T4 ×2 (captured 24 October 2026, vLLM 0.32.0, fp16)

```
2. Decode at batch 1
decode step: 37.6 ms   tokens/s: 27
weights read per step: 8.04 GB  ->  implied read rate 214 GB/s  (spec sheet 320 GB/s: 67%)
arithmetic per step: 0.008 TFLOP  ->  0.2 TFLOP/s of 65 available

3. Decode at batch 32
decode step: 66.3 ms   tokens/s: 482
weights read per step: 8.04 GB  ->  implied read rate 121 GB/s  (spec sheet 320 GB/s: 38%)
arithmetic per step: 0.257 TFLOP  ->  3.9 TFLOP/s of 65 available (6%)

4. nvidia-smi dmon during a long batch-1 decode (two rows a second: the notebook has two T4s)
second  sm%  mem%
     3     0     0      <- the idle T4
     4   100   100      <- the T4 holding the model
     5     0     0
     6   100   100
```

The engine's own warm-up on the T4: torch.compile 33 s, graph capture 13 s, 4.02 GiB of cache for 29,200 tokens.
At batch 1 the T4 reaches two thirds of its spec-sheet bandwidth, a lower fraction than the L4's 82 percent, and
at batch 32 the step nearly doubles where the L4's grew by a sixth: the T4's older attention kernel and smaller
arithmetic show as soon as the batch grows. The dmon columns still read 100 both ways on the busy GPU.
