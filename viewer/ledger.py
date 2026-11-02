"""Log of every paid model step, so the app can show what has been spent.

One JSON line per step (shortening or judging) in viewer/cache/usage.jsonl.
It is kept apart from the results so that deleting a result doesn't hide what
it cost.
"""
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import shorten  # noqa: E402

LEDGER = os.path.join(HERE, "cache", "usage.jsonl")
_lock = threading.Lock()


def add(step, title, model, effort, batch, input_tokens, output_tokens, created=None):
    entry = {
        "created": created or time.time(), "step": step, "title": title,
        "model": model, "effort": effort, "batch": batch,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "cost": shorten.cost(model, input_tokens, output_tokens, batch),
    }
    with _lock:
        os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
        with open(LEDGER, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def add_client(step, title, client):
    """Log what a client has used so far; skipped when it made no calls."""
    if client.calls:
        add(step, title, client.model, getattr(client, "effort", None), getattr(client, "batch", False),
            client.prompt_tokens, client.completion_tokens)


def totals():
    total = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
    if not os.path.exists(LEDGER):
        return total
    with open(LEDGER) as f:
        for line in f:
            e = json.loads(line)
            total["input_tokens"] += e["input_tokens"]
            total["output_tokens"] += e["output_tokens"]
            total["cost"] += e["cost"] or 0.0
    return total
