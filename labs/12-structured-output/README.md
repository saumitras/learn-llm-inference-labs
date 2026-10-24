# Lab 12: structured output

A JSON schema on the request, the first-request grammar cost, what the grammar does to the step, and a failure
you should see once: the grammar lets the model pad with whitespace and never close the object.

- Modal: `modal run labs/12-structured-output/modal/lab.py`
- Kaggle: `kaggle/lab.ipynb`

## Expected output, Modal L4 (captured 24 October 2026, vLLM 0.32.0, structured outputs backend auto → xgrammar)

```
1. No schema
  first token 62 ms, time per token 32.9 ms, 200 tokens
  {   "order_id": "ORD20241005-7890",   "customer": {     "name": "Rahul Sharma",     "city": "Pune", ...

2. Simple schema: first request, then the same again
  request 1: first token 600 ms, time per token 32.9 ms, 200 tokens, INVALID: ends with '\n \n \n \n \n \n \n \n '
  request 2: first token  41 ms, time per token 32.9 ms, 200 tokens, INVALID: ends with '\n \n \n \n \n \n \n \n '
  {   "name": "Amit Sharma",   "age": 32,   "city": "Pune"

3. Nested schema: first request, then again
  request 1: first token 46 ms, time per token 32.9 ms, 167 tokens, valid JSON
  request 2: first token 70 ms, time per token 32.9 ms, 167 tokens, valid JSON
  {   "customer": {     "name": "Rahul Sharma",     "email": "rahul.sharma@example.com", ...

4. The same server with any-whitespace disabled in the grammar
  Simple schema again
  request 1: first token 726 ms, time per token 32.7 ms,  21 tokens, valid JSON
  request 2: first token  70 ms, time per token 32.9 ms,  21 tokens, valid JSON
  {"name":", order_id: ","age":18,"city":"Pune"}
  Nested schema again
  request 1: first token 69 ms, time per token 32.9 ms, 117 tokens, valid JSON
  request 2: first token 70 ms, time per token 32.9 ms, 117 tokens, valid JSON
  {"customer":{"name":"Rahul Sharma","email":"rahul.sharma@example.com","address":{"street":"123 Oak Street","city":"Pune","postcode":"411001','state':'Maharashtra','country':'India'}},"}},"items":[...
```

The first grammar a server compiles costs about 600 to 700 ms on the first token; later grammars and repeats cost
nothing visible. The step is 32.9 ms with and without a grammar. With the simple schema and the default grammar
the model wrote the three fields and then newlines until `max_tokens`, so the object never closed; with
`disable_any_whitespace` it closed in 21 tokens, but the strings it put in the fields are odd. The grammar
guarantees the shape, not the sense.
