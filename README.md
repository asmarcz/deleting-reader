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

Then open http://127.0.0.1:5050, paste a text (or click Load example) and press Process.

## Example: a PDF paper

```
uv run viewer/paper.py ~/Downloads/paper.pdf
```

This sends the paper as one batch (usually a few minutes). Open http://127.0.0.1:5050/paper and move the slider.

How the steps fit together: [PIPELINE.md](PIPELINE.md).
