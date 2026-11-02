# /// script
# requires-python = ">=3.10"
# dependencies = ["flask", "anthropic", "requests", "python-dotenv", "pymupdf"]
# ///
"""Local web app for Deleting Reader.

One page with two modes: Text (paste a text, cached in viewer/cache/) and PDF
(upload a paper, processed by paper.py in a background thread and stored in
viewer/papers/). Both are read with the same level slider, as grey levels or
with the less important words removed.

    uv run viewer/server.py            # http://127.0.0.1:5050
"""
import glob
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
sys.path.insert(0, HERE)
import shorten  # noqa: E402
import paper  # noqa: E402

CACHE = os.path.join(HERE, "cache")
PAPERS = paper.PAPERS
UPLOADS = os.path.join(HERE, "uploads")
for d in (CACHE, PAPERS, UPLOADS):
    os.makedirs(d, exist_ok=True)
_ID = shorten.ID_PATTERN

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

# One lock per cache key, so a double submit waits for the first run's result
# instead of paying for a second API call.
_locks = {}
_guard = threading.Lock()
# Running and finished PDF jobs by paper id: {"state", "message", "name"}.
_jobs = {}


def _lock_for(key):
    with _guard:
        return _locks.setdefault(key, threading.Lock())


def with_display(record):
    """Add each level's display text, computed from the stored word depths.

    Not stored, so improvements to shorten.display_level reach old results
    without calling the model again.
    """
    for p in record["paragraphs"]:
        p.pop("display", None)
        levels = [[i for i, d in enumerate(p["depth"]) if d >= k] for k in range(shorten.LEVELS + 1)]
        p["display"] = [shorten.display_level(p["words"], level) for level in levels]
    return record


def summary(record):
    """Fields the result picker shows; variant is set on repaired runs."""
    out = {k: record[k] for k in ("id", "title", "model", "effort", "created")}
    out["variant"] = record.get("variant")
    return out


def listing(paths):
    items = []
    for path in paths:
        with open(path) as f:
            items.append(summary(json.load(f)))
    items.sort(key=lambda r: r["created"], reverse=True)
    return jsonify(items)


def depths(levels):
    depth = [0] * len(levels[0])
    for k, level in enumerate(levels):
        for i in level:
            depth[i] = k
    return depth


@app.get("/")
def index():
    return send_from_directory(HERE, "app.html")


@app.get("/example.txt")
def example():
    return send_from_directory(HERE, "example.txt")


# Text mode


def split_paragraphs(text):
    blocks = re.split(r"\n\s*\n", text.strip())
    if len(blocks) == 1:  # no blank lines: one paragraph per line
        blocks = text.strip().split("\n")
    return [" ".join(b.split()) for b in blocks if len(b.strip()) > 2]


@app.get("/api/results")
def results():
    return listing(glob.glob(os.path.join(CACHE, "*.json")))


@app.get("/api/results/<rid>")
def result(rid):
    path = os.path.join(CACHE, f"{rid}.json")
    if not _ID.fullmatch(rid) or not os.path.exists(path):
        abort(404)
    with open(path) as f:
        return jsonify(with_display(json.load(f)))


@app.post("/api/results")
def process_text():
    data = request.get_json(force=True)
    title = (data.get("title") or "").strip() or "Untitled"
    paragraphs = split_paragraphs(data.get("text") or "")
    if not paragraphs:
        return jsonify({"error": "No text to process."}), 400
    model = data.get("model") or "claude-sonnet-5-5"
    effort = data.get("effort") if model.startswith("claude-") else None

    # The prompt is part of the digest, so editing it invalidates old results.
    digest = hashlib.sha256(json.dumps(
        [title, paragraphs, model, effort, shorten.PROMPT_TEMPLATE]).encode()).hexdigest()
    rid = shorten.result_id(title, model, effort, digest)
    path = os.path.join(CACHE, f"{rid}.json")
    with _lock_for(rid):
        if os.path.exists(path):
            return result(rid)
        client = shorten.make_client(model=model, effort=effort)
        start = time.time()
        try:
            out = shorten.shorten_article(paragraphs, title, client)
        except Exception as e:  # show the reason in the page instead of a bare 500
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 502
        record = {
            "id": rid, "title": title, "model": model, "effort": effort,
            "created": time.time(), "seconds": round(time.time() - start, 1),
            "calls": client.calls, "input_tokens": client.prompt_tokens,
            "output_tokens": client.completion_tokens,
            "paragraphs": [{"words": p.split(), "depth": depths(r["levels"])}
                           for p, r in zip(paragraphs, out)],
        }
        with open(path, "w") as f:
            json.dump(record, f, ensure_ascii=False)
    return jsonify(with_display(record))


# PDF mode


@app.get("/api/papers")
def papers():
    return listing(glob.glob(os.path.join(PAPERS, "*", "result.json")))


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


@app.post("/api/papers")
def upload_paper():
    """Store an uploaded PDF and process it in the background.

    Returns the paper id and job state; the page polls /api/jobs/<id>. The
    same PDF with the same settings maps to the same id, so a repeat upload
    reuses the finished result, or the running job, or the saved batch.
    """
    f = request.files.get("pdf")
    if not f or not f.filename:
        return jsonify({"error": "Choose a PDF file."}), 400
    data = f.read()
    if not data.startswith(b"%PDF"):
        return jsonify({"error": "That file isn't a PDF."}), 400
    model = request.form.get("model") or "claude-sonnet-5-5"
    effort = request.form.get("effort") or "medium"
    batch = request.form.get("batch", "1") == "1"

    stem = shorten.slugify(os.path.splitext(f.filename)[0])
    path = os.path.join(UPLOADS, f"{stem}-{hashlib.sha256(data).hexdigest()[:8]}.pdf")
    with open(path, "wb") as out:
        out.write(data)
    pid = paper.paper_id(path, model, effort, f.filename)
    if os.path.exists(os.path.join(PAPERS, pid, "result.json")):
        return jsonify({"id": pid, "state": "done", "message": "already processed"})
    with _guard:
        job = _jobs.get(pid)
        if not job or job["state"] == "error":
            job = _jobs[pid] = {"state": "running", "message": "starting", "name": f.filename}
            threading.Thread(target=_run_paper, args=(pid, path, model, effort, batch, f.filename),
                             daemon=True).start()
    return jsonify({"id": pid, **job})


def _run_paper(pid, path, model, effort, batch, name):
    def progress(msg):
        _jobs[pid]["message"] = msg

    try:
        paper.process(path, model, effort, batch=batch, name=name, progress=progress)
        _jobs[pid].update(state="done", message="done")
    except Exception as e:
        _jobs[pid].update(state="error", message=f"{type(e).__name__}: {e}")


@app.get("/api/jobs/<pid>")
def job(pid):
    if pid in _jobs:
        return jsonify({"id": pid, **_jobs[pid]})
    if _ID.fullmatch(pid) and os.path.exists(os.path.join(PAPERS, pid, "result.json")):
        return jsonify({"id": pid, "state": "done", "message": "done"})
    abort(404)


@app.get("/api/jobs")
def jobs():
    return jsonify([{"id": pid, **j} for pid, j in _jobs.items() if j["state"] == "running"])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5050")), threaded=True)
