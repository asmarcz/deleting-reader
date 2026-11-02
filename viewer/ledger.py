"""What the model steps cost.

Each result keeps a "usage" list of its paid steps (shortening, judging), and
the page shows their total. Every step is also appended as one JSON line to
viewer/cache/usage.jsonl, a log of all spending that outlives deleted results.
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


def entry(step, model, effort, batch, input_tokens, output_tokens):
    """One paid step; results keep a list of these as "usage"."""
    return {"step": step, "model": model, "effort": effort, "batch": batch,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cost": shorten.cost(model, input_tokens, output_tokens, batch)}


def client_entry(step, client):
    return entry(step, client.model, getattr(client, "effort", None), getattr(client, "batch", False),
                 client.prompt_tokens, client.completion_tokens)


def add(step, title, model, effort, batch, input_tokens, output_tokens, created=None):
    line = {"created": created or time.time(), "title": title,
            **entry(step, model, effort, batch, input_tokens, output_tokens)}
    with _lock:
        os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
        with open(LEDGER, "a") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def add_client(step, title, client):
    """Log what a client has used so far and return it as an entry (None without calls)."""
    if not client.calls:
        return None
    e = client_entry(step, client)
    add(step, title, e["model"], e["effort"], e["batch"], e["input_tokens"], e["output_tokens"])
    return e


def record_total(record):
    """Tokens and dollars spent on one result: every step in its "usage" list.

    Older results without that list only have their shortening counts.
    """
    steps = record.get("usage") or [entry("shorten", record["model"], record.get("effort"), record.get("batch", False),
                                          record.get("input_tokens", 0), record.get("output_tokens", 0))]
    return {"input_tokens": sum(s["input_tokens"] for s in steps),
            "output_tokens": sum(s["output_tokens"] for s in steps),
            "cost": sum(s["cost"] or 0.0 for s in steps),
            "estimated": any(s.get("estimated") for s in steps)}
