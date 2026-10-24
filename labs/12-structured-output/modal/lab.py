"""Lab 12: structured output. A schema, the first-request compile cost, and what the grammar does to the step.
Run: modal run labs/12-structured-output/modal/lab.py"""
import os, sys, time, subprocess, json, re
import modal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "modal"))
from common import *  # noqa: E402,F403

app = modal.App("llminference-lab12")
SIMPLE = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "city": {"type": "string"}}, "required": ["name", "age", "city"], "additionalProperties": False}
NESTED = {"type": "object", "properties": {"customer": {"type": "object", "properties": {"name": {"type": "string"}, "email": {"type": "string"}, "address": {"type": "object", "properties": {"street": {"type": "string"}, "city": {"type": "string"}, "postcode": {"type": "string"}}, "required": ["street", "city", "postcode"]}}, "required": ["name", "email", "address"]},
  "items": {"type": "array", "items": {"type": "object", "properties": {"sku": {"type": "string"}, "quantity": {"type": "integer"}, "unit_price": {"type": "number"}}, "required": ["sku", "quantity", "unit_price"]}, "minItems": 2}, "total": {"type": "number"}, "paid": {"type": "boolean"}}, "required": ["customer", "items", "total", "paid"], "additionalProperties": False}
PROMPT = "Invent an order for a small online shop with a customer in Pune and two items, as JSON."


def ask(schema=None, name="s"):
    params = {"max_tokens": 200, "temperature": 0}
    if schema:
        params["response_format"] = {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}
    first, gaps, text = stream_chat([{"role": "user", "content": PROMPT}], **params)
    ok = "n/a"
    if schema:
        try: json.loads(text); ok = "valid JSON"
        except Exception: ok = "INVALID: ends with " + repr(text[-30:])
    return first, gaps, text, ok


def schema_rounds(title, schema, name):
    banner(title)
    for i in range(2):
        first, gaps, text, ok = ask(schema, name)
        print(f"  request {i + 1}: first token {1000 * first:.0f} ms, time per token {1000 * sum(gaps) / len(gaps):.1f} ms, {len(gaps) + 1} tokens, {ok}")
    print("  " + text.replace("\n", " ")[:300])


@app.function(image=image, gpu="L4", timeout=2400, volumes=VOLUMES)
def run():
    proc, ready = serve(["--max-model-len", "4096"]); print(f"ready after {ready} s")
    print("  " + log_lines("/tmp/serve.log", {"so": r"structured_outputs_config=[^ ]*"}).get("so", "")[:160])
    banner("1. No schema")
    first, gaps, text, _ = ask()
    print(f"  first token {1000 * first:.0f} ms, time per token {1000 * sum(gaps) / len(gaps):.1f} ms, {len(gaps) + 1} tokens")
    print("  " + text.replace("\n", " ")[:200])
    schema_rounds("2. Simple schema: first request, then the same again", SIMPLE, "simple")
    schema_rounds("3. Nested schema: first request, then again", NESTED, "nested")
    stop(proc)
    banner("4. The same server with any-whitespace disabled in the grammar")
    proc, ready = serve(["--max-model-len", "4096", "--structured-outputs-config", json.dumps({"backend": "xgrammar", "disable_any_whitespace": True})]); print(f"ready after {ready} s")
    schema_rounds("Simple schema again", SIMPLE, "simple")
    schema_rounds("Nested schema again", NESTED, "nested")
    stop(proc)


@app.local_entrypoint()
def main():
    run.remote()
