"""Capstone B: engineer the cache. A multi-turn support agent on one engine with a careless prompt layout, then a
careful one; then two engines behind a router that is round-robin, then session-sticky.
Run: modal run labs/capstone-b-engineer-the-cache/modal/lab.py"""
import os, sys, time, subprocess, json, re, threading
import modal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ops", "modal"))
from common import *  # noqa: E402,F403
import hashlib, random

app = modal.App("llminference-capstone-b")
NAMES = ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total"]
POLICY = "You are the support assistant for a small online shop. " + " ".join(f"Policy clause {i}: items may be returned within thirty days in their original packaging with proof of purchase, and refunds reach the original payment method within five working days." for i in range(22))
TOOLS = "Tools you may call: " + " ".join(f"tool_{i}(order_id) returns the {w} of an order." for i, w in enumerate(["status", "items", "shipping address", "carrier", "tracking events", "invoice", "refund state", "notes"] * 2))
TURNS = 6


def tool_result(session, turn):
    rng = random.Random(session * 100 + turn)
    return "Tool result: " + " ".join(f"order {rng.randint(1000, 9999)} line {i}: {rng.choice(['shipped', 'packed', 'delayed', 'delivered'])} via carrier {rng.randint(1, 9)};" for i in range(14))


def session_messages(session, turn, layout):
    """The messages for one turn. 'careless' puts a timestamp and the session id at the top of the system prompt and
    the tool results before the policy; 'careful' keeps everything static first and everything that changes last."""
    stamp = f"Session {session}, current time 2026-10-24T12:{turn:02d}:00Z."
    tools_so_far = " ".join(tool_result(session, t) for t in range(turn))
    if layout == "careless":
        system = stamp + " " + tools_so_far + " " + POLICY + " " + TOOLS
    else:
        system = POLICY + " " + TOOLS + " " + tools_so_far + " " + stamp
    history = []
    for t in range(turn):
        history += [{"role": "user", "content": f"Question {t} from session {session}: where is my order and when does it arrive?"}, {"role": "assistant", "content": f"Answer {t}: " + "your order is on its way and should arrive within two working days; " * 6}]
    return [{"role": "system", "content": system}, *history, {"role": "user", "content": f"Question {turn} from session {session}: where is my order and when does it arrive?"}]


def hits(url=URL):
    m = metrics(NAMES, url=url); return m.get(NAMES[0], 0), m.get(NAMES[1], 0)


def run_sessions(n_sessions, layout, route, urls):
    """n_sessions agents, each TURNS turns in order, all sessions in parallel. route(session, turn) picks the engine."""
    firsts = {}; lock = threading.Lock()
    before = [hits(u) for u in urls]; t0 = time.time()
    def one(session):
        for turn in range(TURNS):
            url = urls[route(session, turn, len(urls))]
            first, _, _ = stream_chat(session_messages(session, turn, layout), url=url, max_tokens=24, temperature=0)
            with lock: firsts.setdefault(turn, []).append(first)
    ths = [threading.Thread(target=one, args=(s,)) for s in range(n_sessions)]; [t.start() for t in ths]; [t.join() for t in ths]
    took = time.time() - t0; after = [hits(u) for u in urls]
    q = sum(a[0] - b[0] for a, b in zip(after, before)); h = sum(a[1] - b[1] for a, b in zip(after, before))
    allf = sorted(f for fs in firsts.values() for f in fs)
    per_turn = " ".join(f"t{t}:{1000 * sorted(firsts[t])[len(firsts[t]) // 2]:4.0f}" for t in range(TURNS))
    print(f"  hit rate {100 * h / max(1, q):3.0f}%   TTFT p50 {1000 * allf[len(allf) // 2]:5.0f} p99 {1000 * allf[int(0.99 * (len(allf) - 1))]:5.0f} ms   per-turn p50 ms {per_turn}   {took:.0f} s for {n_sessions} sessions x {TURNS} turns")


round_robin = lambda s, t, n: (s * TURNS + t) % n
sticky = lambda s, t, n: int(hashlib.md5(str(s).encode()).hexdigest(), 16) % n


@app.function(image=image, gpu="L4:2", timeout=3600, volumes=VOLUMES)
def run():
    banner("1. One engine, 24 sessions, careless layout (timestamp and tool results at the top), then careful")
    proc, ready = serve(["--max-model-len", "8192"], env={"CUDA_VISIBLE_DEVICES": "0"}); print(f"  ready after {ready} s")
    for layout in ("careless", "careful"):
        print(f"  {layout}:"); run_sessions(24, layout, round_robin, [URL])
    banner("2. The same careful layout, 48 sessions at once")
    run_sessions(48, "careful", round_robin, [URL])
    banner("3. Two engines, one per GPU, 48 careful sessions: round-robin per request, then session-sticky")
    proc2, ready2 = serve(["--max-model-len", "8192"], port=8001, log_path="/tmp/serve2.log", env={"CUDA_VISIBLE_DEVICES": "1"}); print(f"  second engine ready after {ready2} s")
    urls = [URL, "http://127.0.0.1:8001"]
    for name, policy in (("round-robin", round_robin), ("session-sticky", sticky)):
        print(f"  {name}:"); run_sessions(48, "careful", policy, urls)
    stop(proc); stop(proc2)


@app.local_entrypoint()
def main():
    run.remote()
