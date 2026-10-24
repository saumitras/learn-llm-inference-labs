# Lab 05: chunked prefill

Twenty streaming chats, an 8,000-token prompt dropped in at 2.5 s, and the largest gap in anyone's stream, with
three token budgets per step (`--max-num-batched-tokens`).

- Modal: `modal run labs/05-chunked-prefill/modal/lab.py`
- Kaggle: `kaggle/lab.ipynb`

## Expected output, Modal L4 (captured 24 October 2026, vLLM 0.32.0)

```
budget 16384 tokens/step   chats: median gap 38 ms, worst gap 1287 ms   long prompt: first token 1357 ms
budget  2048 tokens/step   chats: median gap 38 ms, worst gap  399 ms   long prompt: first token 1383 ms
budget   512 tokens/step   chats: median gap 38 ms, worst gap  134 ms   long prompt: first token 1775 ms
```

With the whole prompt in one step, every chat freezes for 1.3 seconds. Sliced into 2,048-token pieces the worst
freeze is 0.4 seconds and the long prompt's first token is 26 ms later. At 512 the freeze is 0.13 seconds and
the long prompt pays 0.4 seconds more.
