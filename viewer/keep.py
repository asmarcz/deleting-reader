"""Words the viewer keeps whatever the model decided.

Applied to the stored word depths when a result is shown, so they work on
results processed before they existed:

- headings: never removed. A short heading-like text paragraph (an article's
  title) is kept whole. In papers, so are a heading that older extractions
  merged into the paragraph under it ("2.2 Set-Theoretic Interpretation The
  main idea ...") and a run-in heading ("3.2.1 Subtyping of Field Types.",
  or one set in italic or bold, read from the PDF).
- figure references ("Figure 3", "Table 2", "Section 4.1", "§3") and
  citations ("[9]", "[Smith et al. 2020]", "(Smith 2020)"), on request.
  These are kept as long as their sentence still shows anything: their depth
  rises to that of the sentence's most-kept word, so a reference never
  floats on its own after its sentence is gone.
"""
import os
import re

import pymupdf

import paper
import shorten

REF_WORD = re.compile(
    r"^\(?(Figures?|Figs?\.|Tables?|Tabs?\.|Listings?|Sections?|Secs?\.|Chapters?|Ch\.|Appendix|Appendices"
    r"|Algorithms?|Alg\.|Equations?|Eqs?\.|Theorems?|Thm\.|Lemmas?|Definitions?|Def\.|Propositions?"
    r"|Corollary|Examples?|Ex\.|Lines?)$", re.I)
REF_NUMBER = re.compile(r"^\(?([A-Z]?\d+[\w.–-]*|[A-Z])\)?[.,;:)]*$")
SECTION_SIGN = re.compile(r"^\(?§+\s*\d")
JOINERS = {"and", "&", "to", "or", "–", "-"}
YEAR = re.compile(r"\b(19|20)\d{2}[a-z]?\b")
ABBREVIATIONS = {"fig.", "figs.", "sec.", "secs.", "eq.", "eqs.", "tab.", "tabs.", "ch.", "alg.", "thm.", "def.",
                 "ex.", "e.g.", "i.e.", "al.", "cf.", "vs.", "etc.", "resp.", "approx.", "dr.", "prof.", "no."}
_END = re.compile(r"[.!?][\"'”’)\]]*$")


def references(words):
    """Indices of words that form figure, table and section references."""
    out = set()
    for i, w in enumerate(words):
        if SECTION_SIGN.match(w):
            out.add(i)
            continue
        if not REF_WORD.match(w) or i + 1 >= len(words) or not REF_NUMBER.match(words[i + 1]):
            continue
        # "Figure 3", "Figures 2 and 3", "Sections 4.1–4.3"
        j = i + 1
        while j + 2 < len(words) and words[j + 1].lower() in JOINERS and REF_NUMBER.match(words[j + 2]):
            j += 2
        out.update(range(i, j + 1))
    return out


def citations(words):
    """Indices of words inside citation brackets, or author-year parentheses."""
    out = set()
    i = 0
    while i < len(words):
        w = words[i]
        close = None
        if w.startswith("["):
            close = "]"
        elif w.startswith("(") and w[1:2].isupper():
            close = ")"
        if close:
            for j in range(i, min(i + 15, len(words))):
                if close in words[j]:
                    span = words[i:j + 1]
                    if close == "]" or YEAR.search(" ".join(span)):
                        out.update(range(i, j + 1))
                        i = j
                    break
        i += 1
    return out


def sentences(words):
    """Sentence number of every word."""
    out, s = [], 0
    for w in words:
        out.append(s)
        if _END.search(w) and w.lower() not in ABBREVIATIONS:
            s += 1
    return out


def first_line(p):
    """Number of words on a paper paragraph's first visual line (0 if none)."""
    boxes = p.get("boxes")
    if not boxes:
        return 0
    first_y = boxes[0][0][2]
    n = 0
    while n < len(boxes) and abs(boxes[n][0][2] - first_y) < 2 and boxes[n][0][0] == boxes[0][0][0]:
        n += 1
    return n


def heading_lead(p):
    """Number of leading words of a paper paragraph that are a merged heading,
    judged from the word boxes alone (styled_leads also reads the fonts).

    The words of the paragraph's first visual line count as a heading when
    they read like one (paper.HEADING), stand on their own line and take
    clearly less width than the line under them.
    """
    boxes = p.get("boxes")
    if not boxes or len(boxes) < 3:
        return 0
    n = first_line(p)
    if n >= len(boxes) or n > 12:
        return 0
    second = [b[0] for b in boxes[n:] if abs(b[0][2] - boxes[n][0][2]) < 2]
    width = lambda bs: max(b[3] for b in bs) - min(b[1] for b in bs)
    if width([b[0] for b in boxes[:n]]) > 0.7 * width(second):
        return 0
    return n if paper.HEADING.match(" ".join(p["words"][:n])) else 0


def prepare(record, keep_refs=False, keep_cites=False):
    """Apply the kept words, then add each level's display text in place.

    Display texts are computed from the stored word depths on every load, not
    stored, so improvements to shorten.display_level reach old results
    without calling the model again.
    """
    apply(record, keep_refs, keep_cites)
    for p in record["paragraphs"]:
        p.pop("display", None)
        levels = [[i for i, d in enumerate(p["depth"]) if d >= k] for k in range(shorten.LEVELS + 1)]
        if record.get("kind") == "paper" and paper.is_math(" ".join(p["words"])):
            # Formulas: the tidying would move a "?" or "." onto another
            # symbol or capitalise one, so the kept symbols are shown as printed.
            p["display"] = [[(i, p["words"][i]) for i in level] for level in levels]
        elif p.get("lead"):
            # A merged heading is tidied on its own, or the sentence after it
            # would pass its full stop to the heading's last word.
            n = p["lead"]
            p["display"] = [shorten.display_level(p["words"][:n], [i for i in level if i < n])
                            + [(i + n, t) for i, t in shorten.display_level(p["words"][n:], [i - n for i in level if i >= n])]
                            for level in levels]
        else:
            p["display"] = [shorten.display_level(p["words"], level) for level in levels]
    return record


_styled = {}  # (pdf path, mtime) -> leads per paragraph


def styled_leads(record):
    """Per paragraph, how many leading words are a run-in heading set in
    italic or bold ("*Building from the debugger.* The process ..."), read
    from the paper's PDF. Cached per PDF; None without a PDF."""
    path = os.path.join(paper.PAPERS, record.get("id", ""), "paper.pdf")
    if not os.path.exists(path):
        return None
    key = (path, os.path.getmtime(path), len(record["paragraphs"]))
    if key in _styled:
        return _styled[key]
    doc = pymupdf.open(path)
    spans = {}

    def styled(b, flags=2 | 16):  # italic or bold by default
        if b[0] not in spans:
            spans[b[0]] = [s for blk in doc[b[0]].get_text("dict")["blocks"] for line in blk.get("lines", [])
                           for s in line["spans"] if s["text"].strip()]
        cx, cy = (b[1] + b[3]) / 2, (b[2] + b[4]) / 2
        for s in spans[b[0]]:
            x0, y0, x1, y1 = s["bbox"]
            if x0 - 1 <= cx <= x1 + 1 and y0 - 1 <= cy <= y1 + 1:
                return bool(s["flags"] & flags)
        return False

    leads = []
    for p in record["paragraphs"]:
        # A numbered heading merged into the paragraph: its whole first line
        # is bold, however long ("3.6 Patching Values Instead of Addresses").
        line = first_line(p)
        if (line and line < len(p["words"]) and paper.HEADING.match(" ".join(p["words"][:line]))
                and all(styled(p["boxes"][i][0], 16) for i in range(line))):
            leads.append(line)
            continue
        n = 0
        for i, (w, boxes) in enumerate(zip(p["words"][:8], p["boxes"][:8])):
            if not styled(boxes[0]):
                break
            if re.search(r"[.:]$", w):
                # A heading ends there only if plain text follows, and has a
                # real word in it (math italic like "(𝑑:" is styled too).
                if (i + 1 < len(p["words"]) and not styled(p["boxes"][i + 1][0])
                        and re.search(r"[A-Za-z]{3}", " ".join(p["words"][:i + 1]))):
                    n = i + 1
                break
        leads.append(n)
    doc.close()
    _styled[key] = leads
    return leads


def looks_like_heading(words):
    """A short text paragraph without closing punctuation, e.g. a title."""
    return 0 < len(words) <= 12 and not re.search(r"[.!?:;,]$", words[-1]) and words[0][:1].isupper()


def apply(record, keep_refs=False, keep_cites=False):
    """Raise word depths in place so the kept words survive."""
    top = shorten.LEVELS  # the depth of words that survive every level
    styled = styled_leads(record) if record.get("kind") == "paper" else None
    for n, p in enumerate(record["paragraphs"]):
        words, depth = p["words"], p["depth"]
        if record.get("kind") == "paper":
            # also tells the tidying where the heading ends
            p["lead"] = max(heading_lead(p), styled[n] if styled else 0)
            for i in range(p["lead"]):
                depth[i] = top
        elif looks_like_heading(words):
            p["depth"] = [top] * len(words)
            continue
        wanted = (references(words) if keep_refs else set()) | (citations(words) if keep_cites else set())
        if not wanted:
            continue
        sent = sentences(words)
        best = {}
        for i, s in enumerate(sent):
            best[s] = max(best.get(s, 0), depth[i])
        for i in wanted:
            depth[i] = max(depth[i], best[sent[i]])
