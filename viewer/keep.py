"""Words the viewer keeps whatever the model decided.

Applied to the stored word depths when a result is shown, so they work on
results processed before they existed:

- headings: never removed. A short heading-like text paragraph (an article's
  title) is kept whole; in papers, a heading that older extractions merged
  into the paragraph under it ("2.2 Set-Theoretic Interpretation The main
  idea ...") is kept too.
- figure references ("Figure 3", "Table 2", "Section 4.1", "§3") and
  citations ("[9]", "[Smith et al. 2020]", "(Smith 2020)"), on request.
  These are kept as long as their sentence still shows anything: their depth
  rises to that of the sentence's most-kept word, so a reference never
  floats on its own after its sentence is gone.
"""
import re

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


def heading_lead(p):
    """Number of leading words of a paper paragraph that are a merged heading.

    The words of the paragraph's first visual line count as a heading when
    they read like one (paper.HEADING), stand on their own line and take
    clearly less width than the line under them.
    """
    boxes = p.get("boxes")
    if not boxes or len(boxes) < 3:
        return 0
    first_y = boxes[0][0][2]
    n = 0
    while n < len(boxes) and abs(boxes[n][0][2] - first_y) < 2 and boxes[n][0][0] == boxes[0][0][0]:
        n += 1
    if n >= len(boxes) or n > 12:
        return 0
    second = [b[0] for b in boxes[n:] if abs(b[0][2] - boxes[n][0][2]) < 2]
    width = lambda bs: max(b[3] for b in bs) - min(b[1] for b in bs)
    if width([b[0] for b in boxes[:n]]) > 0.7 * width(second):
        return 0
    return n if paper.HEADING.match(" ".join(p["words"][:n])) else 0


def looks_like_heading(words):
    """A short text paragraph without closing punctuation, e.g. a title."""
    return 0 < len(words) <= 12 and not re.search(r"[.!?:;,]$", words[-1]) and words[0][:1].isupper()


def apply(record, keep_refs=False, keep_cites=False):
    """Raise word depths in place so the kept words survive."""
    top = shorten.LEVELS  # the depth of words that survive every level
    for p in record["paragraphs"]:
        words, depth = p["words"], p["depth"]
        if record.get("kind") == "paper":
            p["lead"] = heading_lead(p)  # also tells the tidying where the heading ends
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
