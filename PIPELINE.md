# Pipeline

```
PDF ──paper.py──▶ result.json ──judge.py──▶ verdicts ──repair.py──▶ repaired result.json
                       │                                                   │
                       └───────────────── server.py ◀──────────────────────┘
```

1. `viewer/paper.py` extracts body paragraphs and their word boxes from the PDF, groups them by section and skips titles, captions and references. Result: `viewer/papers/<id>/result.json`.
2. `shorten.py` asks the model for four shorter versions of every paragraph, one request per section with the abstract as context, all in one Message Batch. Each version is checked against the previous one with `worddiff.py`, so the model can only delete words and every level is a subset of the one above. Result per word: the deepest level it survives.
3. `viewer/judge.py` (optional) sends every paragraph and its versions to a second model, blind, as one batch. It grades each sentence's grammar and lists claims a version changes. Verdicts are cached in `viewer/cache/judge/`.
4. `viewer/repair.py` (optional) uses those verdicts to put back short dropped phrases that changed a claim ("most of", "often"). No model call. Writes a new result marked "repaired".
5. `viewer/server.py` serves the pasted-text view (`/`) and the PDF view (`/paper`). It works out each level's tidied text (capitals, commas, full stops) from the stored levels on every request.

Pasted text in the web page goes through step 2 directly, without a batch, and is cached in `viewer/cache/`.

Everything under `viewer/cache/` and `viewer/papers/` is local and git-ignored. Rerunning a step with the same input reuses its cached result or batch.
