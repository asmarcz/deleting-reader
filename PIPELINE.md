# Pipeline

```
PDF ──extract──▶ shorten (Sonnet) ──▶ draft.json ──judge (Opus)──▶ verdicts ──repair──▶ result.json ──▶ server.py
```

1. `viewer/paper.py` extracts body paragraphs and their word boxes from the PDF, groups them by section and skips titles, captions and references.
2. `shorten.py` asks Sonnet 5.5 (medium effort) for four shorter versions of every paragraph, one request per section with the abstract as context, all in one Message Batch. Each version is checked against the previous one with `worddiff.py`, so the model can only delete words and every level is a subset of the one above. Result per word: the deepest level it survives. Saved as `draft.json`.
3. `viewer/judge.py` sends every paragraph and its versions to Opus 5.5, blind, as one batch. It grades each sentence's grammar and lists claims a version changes. Verdicts are cached in `viewer/cache/judge/`.
4. `viewer/repair.py` uses those verdicts to put back short dropped phrases that changed a claim ("most of", "often"). No model call. The repaired version is the paper's only version: `viewer/papers/<id>/result.json`.
5. `viewer/server.py` serves the single page `viewer/app.html` (Text and PDF tabs). A PDF uploaded there runs steps 1–4 in a background thread. It works out each level's tidied text (capitals, commas, full stops) from the stored levels on every request, after `viewer/keep.py` has raised the words that must stay: headings always, figure references and citations when their toggles are on. `viewer/export.py` turns the same view into a download: the PDF with cut words faded or deleted, or a text result as HTML or PDF.

Pasted text in the Text tab goes through step 2 directly, with a model of your choice and without a batch, and is cached in `viewer/cache/`. For comparing models, `judge.py` and `repair.py` can also be run on these by hand; repaired text results are kept next to the originals.

Every paid step adds a line with its tokens and price to `viewer/cache/usage.jsonl` (`viewer/ledger.py`); the page shows the total.

Everything under `viewer/cache/` and `viewer/papers/` is local and git-ignored. Rerunning a step with the same input reuses its cached result or batch.
