"""Export a result as the reader shows it at one level, in grey or remove mode.

- Papers: the original PDF with the cut words covered. In grey mode a
  semi-transparent white box fades each cut word, as the page does. In remove
  mode the cut words are deleted from the PDF (redaction, so they are gone
  from search and copy too, while figures and drawings stay), and the tidied
  capitals and punctuation are drawn in, in the PDF's own font when it has
  the needed glyphs.
- Text results: a standalone HTML page, or a PDF made from the same HTML.

Word boxes, depths and level texts come from keep.prepare, so the export
matches the page, keep toggles included.
"""
import html
import io

import pymupdf

import shorten

LABELS = ["Full text", "Level 1", "Level 2", "Level 3", "Core"]
# Per number of levels below the cut: GP-TSM's greys on white, the same
# values the page uses (app.html OPACITY).
OPACITY = [1, 0.54, 0.37, 0.27, 0.18]
INK = "#1c1b19"
GREYS = [INK, "#767676", "#a0a0a0", "#b9b9b9", "#d0d0d0"]
TAIL = "\"'”’)].,;:!?…"


def _split_tail(s):
    body = s.rstrip(TAIL)
    return body, s[len(body):]


def description(level, view):
    return f"{LABELS[level]}, {'greyed' if view == 'grey' else 'removed'}"


# Text results


def _text_body(record, level, view):
    paras = []
    for p in record["paragraphs"]:
        if view == "remove":
            parts = [html.escape(t) for _, t in p["display"][level]]
        else:
            parts = []
            for w, d in zip(p["words"], p["depth"]):
                below = min(level - d, shorten.LEVELS) if d < level else 0
                parts.append(f'<span style="color:{GREYS[below]}">{html.escape(w)}</span>' if below else html.escape(w))
        if parts:
            paras.append(f"<p>{' '.join(parts)}</p>")
    return "\n".join(paras)


TEXT_CSS = f"""
body {{ font-family: Georgia, "Iowan Old Style", "Times New Roman", serif; font-size: 12pt; line-height: 1.55; color: {INK}; }}
h1 {{ font-size: 18pt; line-height: 1.2; margin: 0 0 4pt; }}
.meta {{ font-family: sans-serif; font-size: 9pt; color: #6b6860; margin: 0 0 14pt; }}
p {{ margin: 0 0 9pt; }}
"""


def text_html(record, level, view):
    title = html.escape(record["title"])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
{TEXT_CSS}
body {{ max-width: 38rem; margin: 3rem auto; padding: 0 16px; background: #ffffff; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">{description(level, view)} · Deleting Reader</p>
{_text_body(record, level, view)}
</body>
</html>
"""


def text_pdf(record, level, view):
    body = (f"<h1>{html.escape(record['title'])}</h1>"
            f"<p class='meta'>{description(level, view)} · Deleting Reader</p>"
            + _text_body(record, level, view))
    story = pymupdf.Story(html=body, user_css=TEXT_CSS)
    out = io.BytesIO()
    writer = pymupdf.DocumentWriter(out)
    page = pymupdf.paper_rect("a4")
    where = page + (64, 64, -64, -64)
    more = True
    while more:
        device = writer.begin_page(page)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()
    return out.getvalue()


# Papers


class _Page:
    """Text spans of one PDF page, for baselines, fonts and colours."""

    def __init__(self, doc, page):
        self.page = page
        self.spans = [s for b in page.get_text("dict")["blocks"] for line in b.get("lines", [])
                      for s in line["spans"] if s["text"].strip()]
        self.fonts = {}  # span font name -> (fontname for insert_text, pymupdf.Font, buffer) or None
        self.inserted = set()  # embedded fonts already added to the page for drawing
        self.doc = doc

    def span_for(self, b):
        cx, cy = (b[1] + b[3]) / 2, (b[2] + b[4]) / 2
        for s in self.spans:
            x0, y0, x1, y1 = s["bbox"]
            if x0 - 1 <= cx <= x1 + 1 and y0 - 1 <= cy <= y1 + 1:
                return s
        return None

    def cover(self, b):
        """The box to cover for a word, like extent() in app.html."""
        h = b[4] - b[2]
        top, bottom = b[2] - 0.04 * h, b[4] + 0.16 * h
        s = self.span_for(b)
        if s:
            base = s["origin"][1]
            top, bottom = min(top, base - 0.8 * s["size"]), max(bottom, base + 0.26 * s["size"])
        return pymupdf.Rect(b[1] - 0.04 * h, top, b[3] + 0.1 * h, bottom)

    def font(self, name, text):
        """A font to draw text in: the embedded one when it has the glyphs."""
        if name not in self.fonts:
            self.fonts[name] = None
            for xref, _, _, basefont, _, _ in self.page.get_fonts():
                if basefont.split("+")[-1] != name:
                    continue
                _, ext, _, buffer = self.doc.extract_font(xref)
                if buffer and ext in ("ttf", "otf", "cff", "pfa", "pfb", "cid"):
                    try:
                        self.fonts[name] = (f"dr{xref}", pymupdf.Font(fontbuffer=buffer), buffer)
                    except Exception:
                        pass
                break
        found = self.fonts[name]
        if found and all(found[1].has_glyph(ord(c)) for c in text):
            return found
        return ("tiro", pymupdf.Font("tiro"), None)

    def draw(self, x, baseline, text, span, anchor_end=False):
        size = span["size"] if span else 10
        fontname, font, buffer = self.font(span["font"] if span else "", text)
        if anchor_end:
            x -= font.text_length(text, fontsize=size)
        c = span["color"] if span else 0
        color = ((c >> 16 & 255) / 255, (c >> 8 & 255) / 255, (c & 255) / 255)
        if buffer and fontname not in self.inserted:
            self.page.insert_font(fontname=fontname, fontbuffer=buffer)
            self.inserted.add(fontname)
        self.page.insert_text((x, baseline), text, fontsize=size, fontname=fontname, color=color)


def _part_width(b, word, part):
    """Width of part of a word, as a share of its box, measured in Times."""
    times = pymupdf.Font("tiro")
    whole = times.text_length(word, fontsize=10)
    return (b[3] - b[1]) * times.text_length(part, fontsize=10) / whole if whole else 0


def paper_pdf(record, pdf_path, level, view):
    doc = pymupdf.open(pdf_path)
    pages = {}
    covers = {}     # page number -> [(rect, opacity)]
    redacts = {}    # page number -> [rect]
    fixes = []      # (page number, box, word, new text)
    for p in record["paragraphs"]:
        shown = dict(p["display"][level]) if view == "remove" else {}
        for i, boxes in enumerate(p["boxes"]):
            d = p["depth"][i]
            for b in boxes:
                pno = b[0]
                if pno not in pages:
                    pages[pno] = _Page(doc, doc[pno])
                if d < level:
                    rect = pages[pno].cover(b)
                    if view == "remove":
                        covers.setdefault(pno, []).append((rect, 1))
                        # Only the word's own box, so neighbouring glyphs stay.
                        redacts.setdefault(pno, []).append(pymupdf.Rect(b[1] + 0.3, b[2] + 0.2 * (b[4] - b[2]),
                                                                        b[3] - 0.3, b[4] - 0.2 * (b[4] - b[2])))
                    else:
                        covers.setdefault(pno, []).append((rect, 1 - OPACITY[min(level - d, shorten.LEVELS)]))
            if view == "remove" and d >= level and len(boxes) == 1 and shown.get(i, p["words"][i]) != p["words"][i]:
                fixes.append((boxes[0][0], boxes[0], p["words"][i], shown[i]))

    for pno, rects in redacts.items():
        page = doc[pno]
        for r in rects:
            page.add_redact_annot(r, fill=False)
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)
    for pno, items in covers.items():
        page = doc[pno]
        for rect, opacity in items:
            page.draw_rect(rect, color=None, fill=(1, 1, 1), fill_opacity=opacity, overlay=True)

    # Remove mode: redraw what the tidying changed (as showFix in app.html).
    for pno, b, word, text in fixes:
        pg = pages[pno]
        span = pg.span_for(b)
        baseline = span["origin"][1] if span else b[4] - 0.22 * (b[4] - b[2])
        rect = pg.cover(b)
        white = lambda x0, x1: pg.page.draw_rect(pymupdf.Rect(x0, rect.y0, x1, rect.y1), color=None,
                                                fill=(1, 1, 1), overlay=True)
        old_body, old_tail = _split_tail(word)
        new_body, new_tail = _split_tail(text)
        if new_body != old_body and new_body.lower() != old_body.lower():
            white(rect.x0, rect.x1)
            pg.draw(b[1], baseline, text, span)
            continue
        if new_body != old_body and old_body:
            # A capital ends where the small letter ended, growing to the left.
            w = _part_width(b, word, old_body[0])
            white(b[1] - 0.2, b[1] + w)
            pg.draw(b[1] + w, baseline, new_body[0], span, anchor_end=True)
        if new_tail != old_tail:
            x = b[3] - _part_width(b, word, old_tail)
            if old_tail:
                white(x, b[3] + 0.5)
            if new_tail:
                pg.draw(x, baseline, new_tail, span)

    meta = doc.metadata or {}
    meta["title"] = f"{record['title']} ({description(level, view)})"
    doc.set_metadata(meta)
    return doc.tobytes(garbage=3, deflate=True)
