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
import gzip
import hashlib
import hmac
import io
import json
import os
import re
import shutil
import sys
import threading
import time
from datetime import timedelta

from flask import Flask, abort, jsonify, redirect, request, send_file, send_from_directory, session

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import shorten  # noqa: E402
import paper  # noqa: E402
import ledger  # noqa: E402
import keep  # noqa: E402
import export  # noqa: E402

CACHE = os.path.join(HERE, "cache")
PAPERS = paper.PAPERS
UPLOADS = os.path.join(HERE, "uploads")
for d in (CACHE, PAPERS, UPLOADS):
    os.makedirs(d, exist_ok=True)
_ID = shorten.ID_PATTERN

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

# Optional global password (READER_PASSWORD, e.g. in .env.local). The session
# cookie is signed with a key derived from it, so changing the password logs
# everyone out. Set READER_COOKIE_SECURE=1 when served over HTTPS.
PASSWORD = os.getenv("READER_PASSWORD") or None
app.secret_key = hashlib.sha256(f"deleting-reader:{PASSWORD}".encode()).digest() if PASSWORD else os.urandom(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("READER_COOKIE_SECURE") == "1",
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
)
if not PASSWORD:
    print("READER_PASSWORD is not set: the app is open to anyone who can reach it", file=sys.stderr)


@app.before_request
def require_login():
    if not PASSWORD or request.endpoint in ("login_page", "login") or session.get("auth"):
        return None
    if request.path.startswith("/api/"):
        return jsonify({"error": "login required"}), 401
    return redirect("login")  # relative, so it keeps a proxy's path prefix


@app.get("/login")
def login_page():
    return send_from_directory(HERE, "login.html")


@app.post("/login")
def login():
    after = request.form.get("next", "")
    after = after if after.startswith("#") else ""  # only fragments: no redirects off-site
    if PASSWORD and hmac.compare_digest(request.form.get("password", "").encode(), PASSWORD.encode()):
        session.clear()
        session.permanent = True
        session["auth"] = True
        return redirect("./" + after)
    time.sleep(1)  # slows down guessing
    return redirect(f"login?error=1&next={after.replace('#', '%23')}")


@app.post("/logout")
def logout():
    session.clear()
    return redirect("login")


@app.get("/api/config")
def config():
    return jsonify({"auth": bool(PASSWORD)})

# One lock per cache key, so a double submit waits for the first run's result
# instead of paying for a second API call.
_locks = {}
_guard = threading.Lock()
# Running and finished PDF jobs by paper id: {"state", "message", "name"}.
_jobs = {}


def _lock_for(key):
    with _guard:
        return _locks.setdefault(key, threading.Lock())


def keep_args():
    """The ?keep=refs,cites toggles of the current request."""
    wanted = set(request.args.get("keep", "").split(","))
    return {"keep_refs": "refs" in wanted, "keep_cites": "cites" in wanted}


def with_display(record):
    """The record as the page needs it: kept words applied, level texts, cost.

    With ?slim=1 only what a keep toggle changes: each paragraph's depths and
    level texts. The page merges these into the result it has open, so a
    toggle doesn't download every word and box again.
    """
    keep.prepare(record, **keep_args())
    if request.args.get("slim"):
        return {"id": record["id"], "paragraphs": [{"depth": p["depth"], "display": p["display"]}
                                                   for p in record["paragraphs"]]}
    record["total"] = ledger.record_total(record)
    return record


@app.after_request
def compress(response):
    """Gzip JSON responses: results with word boxes run to megabytes."""
    if (response.mimetype == "application/json" and not response.direct_passthrough
            and "gzip" in request.headers.get("Accept-Encoding", "")
            and "Content-Encoding" not in response.headers):
        data = response.get_data()
        if len(data) > 2048:
            response.set_data(gzip.compress(data, compresslevel=5))
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Vary"] = "Accept-Encoding"
    return response


def summary(record):
    """Fields the result picker shows; variant is set on repaired runs."""
    out = {k: record[k] for k in ("id", "title", "model", "effort", "created")}
    out["variant"] = record.get("variant")
    out["kind"] = record.get("kind")
    return out


def listing(paths):
    items = []
    for path in paths:
        with open(path) as f:
            items.append(summary(json.load(f)))
    items.sort(key=lambda r: r["created"], reverse=True)
    return jsonify(items)


# Manage page: unlisted (nothing links to it), at /manage, for deleting
# results. Behind the login like everything else.


@app.get("/manage")
def manage_page():
    return send_from_directory(HERE, "manage.html")


@app.get("/api/manage")
def manage_items():
    items = []
    for kind, paths in (("text", glob.glob(os.path.join(CACHE, "*.json"))),
                        ("pdf", glob.glob(os.path.join(PAPERS, "*", "result.json")))):
        for path in paths:
            with open(path) as f:
                items.append({**summary(json.load(f)), "tab": kind})
    jobs = [{"id": pid, "tab": "pdf", "title": job.get("name"), "running": True}
            for pid, job in list(_jobs.items()) if job["state"] == "running"]
    items.sort(key=lambda r: r["created"], reverse=True)
    return jsonify(jobs + items)


@app.delete("/api/manage/<tab>/<rid>")
def manage_delete(tab, rid):
    """Delete a text result, or a paper with its folder (PDF, batches, draft).

    The usage log keeps its lines, so what the result cost stays on record.
    """
    if not _ID.fullmatch(rid) or tab not in ("text", "pdf"):
        abort(404)
    if tab == "text":
        path = os.path.join(CACHE, f"{rid}.json")
        if not os.path.exists(path):
            abort(404)
        os.remove(path)
    else:
        if _jobs.get(rid, {}).get("state") == "running":
            return jsonify({"error": "Still processing; delete it once it's done."}), 409
        folder = os.path.join(PAPERS, rid)
        if not os.path.isdir(folder):
            abort(404)
        shutil.rmtree(folder)
        _jobs.pop(rid, None)
    return jsonify({"deleted": rid})


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
            ledger.add_client("shorten", title, client)  # failed calls are paid for too
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 502
        shortened = ledger.add_client("shorten", title, client)
        record = {
            "id": rid, "title": title, "model": model, "effort": effort,
            "created": time.time(), "seconds": round(time.time() - start, 1),
            "calls": client.calls, "input_tokens": client.prompt_tokens,
            "output_tokens": client.completion_tokens, "usage": [shortened] if shortened else [],
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


# Export


def export_args():
    """Level and view of an export request, from ?level=0-4&view=grey|remove."""
    try:
        level = int(request.args.get("level", "0"))
    except ValueError:
        abort(400)
    view = request.args.get("view", "grey")
    if not 0 <= level <= shorten.LEVELS or view not in ("grey", "remove"):
        abort(400)
    return level, view


def download(data, rid, level, view, ext, mimetype):
    name = f"{rid}-{shorten.slugify(export.LABELS[level])}-{view}.{ext}"
    return send_file(io.BytesIO(data), mimetype=mimetype, as_attachment=True, download_name=name)


@app.get("/api/papers/<pid>/export.pdf")
def export_paper(pid):
    path = os.path.join(PAPERS, pid, "result.json")
    if not _ID.fullmatch(pid) or not os.path.exists(path):
        abort(404)
    level, view = export_args()
    with open(path) as f:
        record = keep.prepare(json.load(f), **keep_args())
    data = export.paper_pdf(record, os.path.join(PAPERS, pid, "paper.pdf"), level, view)
    return download(data, pid, level, view, "pdf", "application/pdf")


@app.get("/api/results/<rid>/export.<fmt>")
def export_text(rid, fmt):
    path = os.path.join(CACHE, f"{rid}.json")
    if not _ID.fullmatch(rid) or not os.path.exists(path) or fmt not in ("html", "pdf"):
        abort(404)
    level, view = export_args()
    with open(path) as f:
        record = keep.prepare(json.load(f), **keep_args())
    if fmt == "html":
        return download(export.text_html(record, level, view).encode(), rid, level, view, "html", "text/html")
    return download(export.text_pdf(record, level, view), rid, level, view, "pdf", "application/pdf")


@app.post("/api/papers")
def upload_paper():
    """Store an uploaded PDF and process it in the background.

    Returns the paper id and job state; the page polls /api/jobs/<id>. The
    same PDF maps to the same id, so a repeat upload
    reuses the finished result, or the running job, or the saved batch.
    """
    f = request.files.get("pdf")
    if not f or not f.filename:
        return jsonify({"error": "Choose a PDF file."}), 400
    data = f.read()
    if not data.startswith(b"%PDF"):
        return jsonify({"error": "That file isn't a PDF."}), 400
    batch = request.form.get("batch", "1") == "1"

    stem = shorten.slugify(os.path.splitext(f.filename)[0])
    path = os.path.join(UPLOADS, f"{stem}-{hashlib.sha256(data).hexdigest()[:8]}.pdf")
    with open(path, "wb") as out:
        out.write(data)
    pid = paper.paper_id(path, f.filename)
    if os.path.exists(os.path.join(PAPERS, pid, "result.json")):
        return jsonify({"id": pid, "state": "done", "message": "already processed"})
    with _guard:
        job = _jobs.get(pid)
        if not job or job["state"] == "error":
            job = _jobs[pid] = {"state": "running", "message": "starting", "name": f.filename}
            threading.Thread(target=_run_paper, args=(pid, path, batch, f.filename),
                             daemon=True).start()
    return jsonify({"id": pid, **job})


def _run_paper(pid, path, batch, name):
    def progress(msg):
        _jobs[pid]["message"] = msg

    try:
        paper.process(path, batch=batch, name=name, progress=progress)
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
