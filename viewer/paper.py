# /// script
# requires-python = ">=3.10"
# dependencies = ["pymupdf", "anthropic", "requests", "python-dotenv"]
# ///
"""Process a PDF paper for the viewer's PDF overlay.

Extracts body paragraphs with their word boxes, groups them by section,
shortens every section with Sonnet 5.5 at medium effort (as one Message Batch
by default, with the abstract and section heading as context), has the Opus
judge check each paragraph, repairs the flagged spans (viewer/repair.py), and
writes viewer/papers/<id>/{paper.pdf,result.json}. Only that final version is
shown. Titles, author blocks, captions, footnotes and the reference list are
left out, so they stay untouched in the PDF.

    uv run viewer/paper.py ~/Downloads/paper.pdf                 # batch
    uv run viewer/paper.py paper.pdf --no-batch
    uv run viewer/paper.py paper.pdf --dry-run                   # extraction only

Rerunning with the same PDF reuses the saved batches or result.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import sys
import time
import unicodedata

import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import shorten  # noqa: E402
import judge  # noqa: E402
import ledger  # noqa: E402
import repair  # noqa: E402

PAPERS = os.path.join(HERE, "papers")
MODEL, EFFORT = "claude-sonnet-5-5", "medium"  # one run per paper, then the judge and repair

SKIP_PREFIX = re.compile(
    r"^(CCS Concepts|Keywords|ACM Reference Format|Permission to make|©|Copyright|"
    r"Figure \d|Fig\. \d|Table \d|Listing \d|arXiv:)", re.I)
# Numbered headings ("3 Scenario", "4.2 Results", "A Appendix"), or the usual
# unnumbered ones; anything else bold and short (code, run-in labels) is not a
# section heading.
HEADING = re.compile(
    r"^((\d+(\.\d+)*|[A-H])\.?\s+[A-Z][^»{}=;]{0,80}"
    r"|(Abstract|Introduction|Related Work|Background|Discussion|Conclusions?|Future Work"
    r"|Acknowledge?ments?|References|Bibliography|Appendix|CCS Concepts|Keywords)\b.{0,40})$")
STOP_SECTIONS = re.compile(r"^(\d+\.?\s+)?(References|Bibliography|Acknowledg)", re.I)
SKIP_SECTIONS = re.compile(r"^(CCS Concepts|Keywords)", re.I)


def _font_stats(lines):
    """Dominant font size and bold share of some PyMuPDF dict lines."""
    sizes, bold, chars = [], 0, 0
    for line in lines:
        for span in line["spans"]:
            n = len(span["text"].strip())
            sizes += [round(span["size"], 1)] * n
            chars += n
            if span["flags"] & 16:
                bold += n
    if not chars:
        return None
    return statistics.mode(sizes), bold / chars


MATH_LETTER_SHARE = 0.45


def is_math(text):
    """True for display math and formula-heavy blocks, which are shortened like
    prose but shown without punctuation tidying (see server.with_display).

    Prose is mostly ordinary letters (over half of its characters, even with
    inline math); formulas are symbols, digits and mathematical italic letters
    (𝑡, 𝜙: their own Unicode block), so they fall well below.
    """
    chars = [c for c in text if not c.isspace()]
    letters = sum(1 for c in chars if unicodedata.category(c).startswith("L") and not 0x1D400 <= ord(c) <= 0x1D7FF)
    return bool(chars) and letters / len(chars) < MATH_LETTER_SHARE


def block_info(block):
    """(size, bold share) of a dict block, and of its first line alone."""
    if any(line["dir"] != (1.0, 0.0) for line in block["lines"]):
        return None  # rotated text, e.g. the arXiv margin stamp
    whole = _font_stats(block["lines"])
    if not whole:
        return None
    # A heading can span several PyMuPDF lines on one visual line ("2.2" and
    # "Set-Theoretic Interpretation" are separate lines in acmart PDFs), so
    # the candidate heading is every leading bold line, not just the first.
    lead = 1
    while lead < min(4, len(block["lines"])):
        stats = _font_stats(block["lines"][lead:lead + 1])
        if not stats or stats[1] <= 0.6:
            break
        lead += 1
    first = _font_stats(block["lines"][:lead])
    return whole, (first[0], first[1], lead) if first else None


def extract(pdf_path, name=None):
    """Body paragraphs grouped into sections, each word with its box(es).

    Returns (title, abstract_text, sections, stats); a section is
    {"heading": str, "paragraphs": [{"words": [...], "boxes": [[[page, x0, y0, x1, y1], ...], ...]}]}.
    """
    doc = pymupdf.open(pdf_path)
    flags = pymupdf.TEXTFLAGS_WORDS & ~pymupdf.TEXT_PRESERVE_LIGATURES
    pages = []
    all_sizes = []
    for pno, page in enumerate(doc):
        tp = page.get_textpage(flags=flags)
        words = page.get_text("words", textpage=tp)
        blocks = page.get_text("dict", textpage=tp)["blocks"]
        info = {}
        for b in blocks:
            if b.get("type") == 0:
                i = block_info(b)
                if i:
                    info[b["number"]] = i
        by_block = {}
        for x0, y0, x1, y1, w, bno, lno, wno in words:
            by_block.setdefault(bno, []).append((lno, wno, w, [pno, x0, y0, x1, y1]))
        for bno in sorted(by_block):
            if bno not in info:
                continue
            ws = sorted(by_block[bno])
            (size, bold), first = info[bno]
            parts = [(ws, size, bold)]
            # Some PDFs (Typst, Word) put a heading in the same block as the
            # paragraph under it; split such a first line off.
            lead = first[2] if first else 1
            head = [w for w in ws if w[0] < lead]
            if first and len(head) < len(ws) and (first[1] > 0.6 or first[0] > size + 0.5) \
                    and len(head) <= 12 and HEADING.match(" ".join(w[2] for w in head)):
                rest = [w for w in ws if w[0] >= lead]
                parts = [(head, *first[:2]), (rest, size, bold)]
            for part, psize, pbold in parts:
                all_sizes += [psize] * len(part)
                pages.append({"page": pno, "size": psize, "bold": pbold,
                              "y": min(w[3][2] for w in part), "height": page.rect.height,
                              "words": part})
    body_size = statistics.mode(all_sizes)
    title = doc_title(pdf_path, name)

    sections, current, abstract, stats = [], None, [], {"skipped": 0}
    stopped = False
    for blk in pages:
        text = " ".join(w[2] for w in blk["words"])
        n = len(blk["words"])
        near_edge = blk["y"] < 45 or blk["y"] > blk["height"] - 45
        if stopped:
            continue
        if (blk["bold"] > 0.6 or blk["size"] > body_size + 0.5) and n <= 12 and HEADING.match(text):
            if STOP_SECTIONS.match(text):
                stopped = True
                continue
            current = {"heading": text, "paragraphs": [], "skip": bool(SKIP_SECTIONS.match(text))}
            sections.append(current)
            continue
        if current is not None and current.get("skip"):
            stats["skipped"] += 1
            continue
        if (near_edge and n < 15) or SKIP_PREFIX.match(text) or abs(blk["size"] - body_size) > 0.6 or n < 12:
            stats["skipped"] += 1
            continue
        if current is None:  # body-size text before the first heading
            current = {"heading": "", "paragraphs": []}
            sections.append(current)

        # Rebuild tokens: join words hyphenated across a line break.
        words, boxes = [], []
        prev_line = None
        for lno, _, w, box in blk["words"]:
            if (words and prev_line is not None and lno != prev_line and words[-1].endswith("-")
                    and len(words[-1]) > 2 and w[:1].islower()):
                words[-1] = words[-1][:-1] + w
                boxes[-1].append(box)
            else:
                words.append(w)
                boxes.append([box])
            prev_line = lno

        paras = current["paragraphs"]
        # A paragraph broken by a column or page continues if the previous
        # block didn't end a sentence and this one starts in lowercase.
        if paras and not re.search(r"[.!?:][\"')\]]*$", paras[-1]["words"][-1]) and words[0][:1].islower():
            paras[-1]["words"] += words
            paras[-1]["boxes"] += boxes
        else:
            paras.append({"words": words, "boxes": boxes})

    for s in sections:
        if s["heading"].lower().startswith("abstract"):
            abstract = [" ".join(p["words"]) for p in s["paragraphs"]]
    sections = [s for s in sections if s["paragraphs"]]
    stats["body_size"] = body_size
    stats["paragraphs"] = sum(len(s["paragraphs"]) for s in sections)
    stats["words"] = sum(len(p["words"]) for s in sections for p in s["paragraphs"])
    return title, " ".join(abstract), sections, stats


def doc_title(pdf_path, name=None):
    """The PDF's own title, else the file name without extension."""
    with pymupdf.open(pdf_path) as doc:
        title = doc.metadata.get("title")
    return title or os.path.splitext(name or os.path.basename(pdf_path))[0]


def paper_id(pdf_path, name=None):
    """Readable id of a processed paper, e.g. "its-alive-what-a-live-object-environment-changes-1a2b3c4d".

    The hash covers the PDF and everything that shapes the result (models,
    prompts, repair rule), so changing any of them gives a new id.
    """
    with open(pdf_path, "rb") as f:
        pdf_hash = hashlib.sha256(f.read()).hexdigest()
    digest = hashlib.sha256(json.dumps(
        [pdf_hash, MODEL, EFFORT, shorten.PROMPT_TEMPLATE,
         judge.MODEL, judge.EFFORT, judge.SYSTEM, judge.PROMPT, repair.MAX_RUN]).encode()).hexdigest()
    return f"{shorten.slugify(doc_title(pdf_path, name))}-{digest[:8]}"


def process(pdf_path, batch=True, name=None, progress=None):
    """Shorten, judge and repair a paper; returns its id. Reuses finished steps.

    Only the final, repaired version is kept as result.json. The shortened
    draft is saved as draft.json, so an interrupted run can resume at the judge.
    name is the original file name (for the title when the PDF has none);
    progress, if given, is called with short status messages.
    """
    report = progress or (lambda msg: print(msg, file=sys.stderr))
    pid = paper_id(pdf_path, name)
    out_dir = os.path.join(PAPERS, pid)
    result_path = os.path.join(out_dir, "result.json")
    draft_path = os.path.join(out_dir, "draft.json")
    if os.path.exists(result_path):
        report("already processed")
        return pid
    start = time.time()

    if os.path.exists(draft_path):
        with open(draft_path) as f:
            draft = json.load(f)
    else:
        draft = shorten_paper(pdf_path, pid, out_dir, batch, name, report)
        with open(draft_path, "w") as f:
            json.dump(draft, f, ensure_ascii=False)

    report("judging grammar and meaning")
    verdicts, judged = judge.judge_record(draft, batch=batch, state_dir=out_dir, progress=report)
    report("repairing")
    paragraphs, log = repair.repair_paragraphs(draft["paragraphs"], verdicts)
    restored = sum(1 for entry in log if entry[2] == "restored")
    record = {**draft, "paragraphs": paragraphs, "judge_model": judge.MODEL, "judge_effort": judge.EFFORT,
              "restored": restored, "seconds": round(time.time() - start, 1),
              "usage": draft.get("usage", []) + ([judged] if judged else [])}
    with open(result_path, "w") as f:
        json.dump(record, f, ensure_ascii=False)
    report(f"done: {restored} dropped word runs restored, {record['seconds']} s")
    return pid


def shorten_paper(pdf_path, pid, out_dir, batch, name, report):
    """Extract the paper's paragraphs and shorten them; returns the draft record."""
    report("extracting text")
    title, abstract, sections, stats = extract(pdf_path, name)
    if not stats["paragraphs"]:
        raise ValueError("no body paragraphs found in this PDF")
    report(f"{stats['paragraphs']} paragraphs in {len(sections)} sections")
    os.makedirs(out_dir, exist_ok=True)
    shutil.copyfile(pdf_path, os.path.join(out_dir, "paper.pdf"))

    doc_sections = []
    for s in sections:
        context = (f"Abstract of the paper: {abstract}\n\n" if abstract and not s["heading"].lower().startswith("abstract") else "")
        context += f"The paragraphs below are from the section \"{s['heading']}\"." if s["heading"] else ""
        doc_sections.append({"paragraphs": [" ".join(p["words"]) for p in s["paragraphs"]],
                             "context": context or None})

    client = shorten.make_client(model=MODEL, effort=EFFORT, batch=batch)
    client.progress = report
    results = shorten.shorten_document(doc_sections, title, client, state_dir=out_dir)
    shortened = ledger.add_client("shorten", title, client)

    paragraphs = []
    for s, sec in zip(results, sections):
        for r, p in zip(s, sec["paragraphs"]):
            depth = [0] * len(p["words"])
            for k, level in enumerate(r["levels"]):
                for i in level:
                    depth[i] = k
            paragraphs.append({"section": sec["heading"], "words": p["words"],
                               "boxes": p["boxes"], "depth": depth})
    return {
        "id": pid, "kind": "paper", "title": title, "source": name or os.path.basename(pdf_path),
        "model": MODEL, "effort": EFFORT, "batch": batch, "created": time.time(),
        "calls": client.calls, "input_tokens": client.prompt_tokens,
        "output_tokens": client.completion_tokens, "paragraphs": paragraphs,
        "usage": [shortened] if shortened else [],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--no-batch", action="store_true", help="send requests directly instead of as a batch")
    ap.add_argument("--dry-run", action="store_true", help="only print the extracted structure")
    args = ap.parse_args()

    if args.dry_run:
        title, _, sections, stats = extract(args.pdf)
        print(f"{title}\n{stats['paragraphs']} paragraphs, {stats['words']} words in {len(sections)} sections, "
              f"{stats['skipped']} blocks skipped, body font {stats['body_size']} pt")
        for s in sections:
            print(f"  [{s['heading'] or '(no heading)'}]")
            for p in s["paragraphs"]:
                print(f"    {len(p['words']):4} words: {' '.join(p['words'][:12])} …")
        return
    pid = process(args.pdf, batch=not args.no_batch)
    print(os.path.join(PAPERS, pid, "result.json"))


if __name__ == "__main__":
    main()
