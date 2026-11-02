# /// script
# requires-python = ">=3.10"
# dependencies = ["flask", "anthropic", "requests", "python-dotenv"]
# ///
"""Local viewer for Deleting Reader.

Paste a text, run it through shorten.py and read it with a level slider, either
as grey levels or with the less important words removed. Results are cached in
viewer/cache/, so reopening a text costs nothing. /paper shows processed PDFs
(see paper.py).

    uv run viewer/server.py            # http://127.0.0.1:5050
"""
import hashlib
import json
import os
import re
import sys
import threading
import time

from flask import Flask, abort, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import shorten  # noqa: E402

CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)

app = Flask(__name__)

# One lock per cache key, so a double submit waits for the first run's result
# instead of paying for a second API call.
_locks = {}
_locks_guard = threading.Lock()


def _lock_for(rid):
    with _locks_guard:
        return _locks.setdefault(rid, threading.Lock())


def split_paragraphs(text):
    blocks = re.split(r"\n\s*\n", text.strip())
    if len(blocks) == 1:  # no blank lines: one paragraph per line, as app.py does
        blocks = text.strip().split("\n")
    return [" ".join(b.split()) for b in blocks if len(b.strip()) > 2]


def cache_key(title, paragraphs, model, effort):
    # The prompt is part of the key, so editing it invalidates old results.
    blob = json.dumps([title, paragraphs, model, effort, shorten.PROMPT_TEMPLATE])
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def with_display(record):
    """Add each level's display text, computed from the cached word depths.

    Not stored, so improvements to shorten.display_level reach old results
    without calling the model again.
    """
    for p in record["paragraphs"]:
        p.pop("display", None)
        levels = [[i for i, d in enumerate(p["depth"]) if d >= k] for k in range(shorten.LEVELS + 1)]
        p["display"] = [shorten.display_level(p["words"], level) for level in levels]
    return record


def summary(record):
    """Fields the result pickers show; variant is set on repaired runs."""
    out = {k: record[k] for k in ("id", "title", "model", "effort", "created")}
    out["variant"] = record.get("variant")
    return out


def read_cached(rid):
    with open(os.path.join(CACHE, f"{rid}.json")) as f:
        return json.load(f)


@app.get("/")
def index():
    return send_from_directory(HERE, "index.html")


PAPERS = os.path.join(HERE, "papers")
_ID = re.compile(r"[0-9a-f]{16}")


@app.get("/paper")
def paper_page():
    return send_from_directory(HERE, "paper.html")


@app.get("/api/papers")
def papers():
    items = []
    if os.path.isdir(PAPERS):
        for pid in os.listdir(PAPERS):
            path = os.path.join(PAPERS, pid, "result.json")
            if os.path.exists(path):
                with open(path) as f:
                    r = json.load(f)
                items.append(summary(r))
    items.sort(key=lambda r: r["created"], reverse=True)
    return jsonify(items)


@app.get("/api/papers/<pid>")
def paper_result(pid):
    path = os.path.join(PAPERS, pid, "result.json")
    if not _ID.fullmatch(pid) or not os.path.exists(path):
        abort(404)
    with open(path) as f:
        return jsonify(with_display(json.load(f)))


@app.get("/api/papers/<pid>/paper.pdf")
def paper_pdf(pid):
    if not _ID.fullmatch(pid) or not os.path.exists(os.path.join(PAPERS, pid, "paper.pdf")):
        abort(404)
    return send_from_directory(os.path.join(PAPERS, pid), "paper.pdf")


@app.get("/example.txt")
def example():
    return send_from_directory(HERE, "example.txt")


@app.get("/api/results")
def results():
    items = []
    for name in os.listdir(CACHE):
        with open(os.path.join(CACHE, name)) as f:
            r = json.load(f)
        items.append(summary(r))
    items.sort(key=lambda r: r["created"], reverse=True)
    return jsonify(items)


@app.get("/api/results/<rid>")
def result(rid):
    path = os.path.join(CACHE, f"{rid}.json")
    if not re.fullmatch(r"[0-9a-f]{16}", rid) or not os.path.exists(path):
        abort(404)
    return jsonify(with_display(read_cached(rid)))


@app.post("/api/process")
def process():
    data = request.get_json(force=True)
    title = (data.get("title") or "").strip() or "Untitled"
    paragraphs = split_paragraphs(data.get("text") or "")
    if not paragraphs:
        return jsonify({"error": "No text to process."}), 400
    model = data.get("model") or "claude-haiku-5-5"
    effort = data.get("effort") if model.startswith("claude-") else None

    rid = cache_key(title, paragraphs, model, effort)
    with _lock_for(rid):
        return _process(rid, title, paragraphs, model, effort)


def _process(rid, title, paragraphs, model, effort):
    path = os.path.join(CACHE, f"{rid}.json")
    if os.path.exists(path):
        return jsonify(with_display(read_cached(rid)))

    client = shorten.make_client(model=model, effort=effort)
    start = time.time()
    try:
        out = shorten.shorten_article(paragraphs, title, client)
    except Exception as e:  # show the reason in the page instead of a bare 500
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 502

    paras = []
    for p, r in zip(paragraphs, out):
        depth = [0] * len(r["levels"][0])
        for k, level in enumerate(r["levels"]):
            for i in level:
                depth[i] = k
        paras.append({"words": p.split(), "depth": depth})

    record = {
        "id": rid,
        "title": title,
        "model": model,
        "effort": effort,
        "created": time.time(),
        "seconds": round(time.time() - start, 1),
        "calls": client.calls,
        "input_tokens": client.prompt_tokens,
        "output_tokens": client.completion_tokens,
        "paragraphs": paras,
    }
    with open(path, "w") as f:
        json.dump(record, f, ensure_ascii=False)
    return jsonify(with_display(record))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5050")))
