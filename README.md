# Deleting Reader

Read a text at several levels of detail: a model shortens every paragraph by deleting words only, and a slider fades or removes the less important ones. Based on [GP-TSM](https://github.com/ZiweiGu/GP-TSM).

## Start

Needs [uv](https://docs.astral.sh/uv/) and an Anthropic API key in `.env.local`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

```
uv run viewer/server.py
```

Then open http://127.0.0.1:5050. To require a password, add `READER_PASSWORD=...` to `.env.local` (and `READER_COOKIE_SECURE=1` when served over HTTPS). In the Text tab, click New text, then Load example and Process.

## Example: a PDF paper

In the PDF tab, click Upload PDF and choose a paper. Sonnet 5.5 shortens it, and an Opus 5.5 judge checks the result before a repair step. With Batch on (half price) it takes a few minutes; you can keep reading meanwhile, and the paper opens when it's ready. Move the slider.

The same from the command line: `uv run viewer/paper.py ~/Downloads/paper.pdf`.

How the steps fit together: [PIPELINE.md](PIPELINE.md).
