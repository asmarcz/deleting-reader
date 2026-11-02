"""Shorten text into nested levels by deleting words only.

GP-TSM (Gu et al., CHI 2024) builds each level with its own round of sampling
and scoring. Here the model writes all levels of every paragraph at once, with
the document's title and context, and the method's guarantees are enforced
locally instead of by sampling:

- extractive: each level is diffed against the previous one; words the model
  rewrote are put back, words it added are dropped;
- nested: each level is a subset of the previous level's words.

Levels are tracked as sets of word indices into the original paragraph, so the
output only ever shows the paragraph's own words.

Backends: Claude models (READER_MODEL=claude-..., key ANTHROPIC_API_KEY,
effort READER_EFFORT) go through the Anthropic SDK, optionally as a Message
Batch at half price; anything else goes to an OpenAI-compatible chat
completions endpoint such as llama.cpp's llama-server (READER_BASE_URL,
READER_MODEL, OPENAI_API_KEY).
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
import sys
import time
import unicodedata

import requests
from dotenv import load_dotenv

import worddiff

# Per-project secrets, git-ignored.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.local"))

LEVELS = 4  # levels below the original
MAX_CHARS_PER_CALL = 6000  # longer sections are split into several calls

SYSTEM_PROMPT = "You shorten text by deleting words only. You answer with JSON only."

PROMPT_TEMPLATE = """Below is {where} split into numbered paragraphs. For each paragraph, write {levels} successively shorter versions.

Rules:
- Version 1 is the paragraph with some words deleted. Version 2 is version 1 with some words deleted, and so on.
- Only delete words. Never add, replace, reorder or reword anything.
- Each version keeps at least 80 percent of the words of the version before it. Never delete more than that in one step, even in the shortest versions.
- Every sentence in every version must stay grammatical.
- Delete the least important material first: asides, examples, secondary modifiers. Keep what the article as a whole depends on: its main claims, key terms, and the subject, verb and object of each sentence.
- Never delete words that change what a sentence claims: negations (not, never, no), modals (should, may, might, could), quantifiers (some, most, few, all) and conditions (if, unless). A shorter version may say less than the original, but never something different.

Title: {title}
{context}
{paragraphs}

Answer with JSON of this shape, one entry per paragraph, each with exactly {levels} versions:
{{"paragraphs": [{{"id": 1, "levels": ["version 1", "version 2", ...]}}, ...]}}"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "paragraphs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "levels": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "levels"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["paragraphs"],
    "additionalProperties": False,
}


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,120}")


def slugify(text, limit=48):
    """Lowercase ASCII words joined by hyphens, cut at a word boundary."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    words = re.findall(r"[a-z0-9]+", re.sub(r"['’]", "", text.lower()))
    out = ""
    for w in words:
        if out and len(out) + 1 + len(w) > limit:
            break
        out = f"{out}-{w}" if out else w[:limit]
    return out or "untitled"


def result_id(title, model, effort, digest):
    """Readable id for a cached result, e.g.
    "why-night-trains-are-coming-back-to-europe-sonnet-5-5-medium-74cdacf6".

    digest is a hash of everything that determines the result (text, model,
    effort, prompt), so different inputs never share a name.
    """
    parts = [slugify(title), slugify(model.removeprefix("claude-"), 24)]
    if effort:
        parts.append(effort)
    parts.append(digest[:8])
    return "-".join(parts)


class Client:
    """Minimal OpenAI-compatible chat client that counts calls and tokens."""

    batch = False
    progress = None  # optional callable for status messages, e.g. from the web server

    def _report(self, msg):
        (self.progress or _log)(msg)

    def __init__(self, base_url=None, model=None, api_key=None, temperature=0.2):
        self.base_url = (base_url or os.getenv("READER_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.getenv("READER_MODEL", "gpt-4.1-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def complete_json(self, prompt, system=SYSTEM_PROMPT, schema=RESPONSE_SCHEMA):
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "levels", "schema": schema, "strict": True},
            },
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=600)
        r.raise_for_status()
        data = r.json()
        self.calls += 1
        usage = data.get("usage") or {}
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        return _parse_json(data["choices"][0]["message"]["content"])

    def complete_json_many(self, prompts, system=SYSTEM_PROMPT, schema=RESPONSE_SCHEMA, workers=4, **_):
        """{id: prompt} -> {id: parsed answer}; failed requests are left out."""
        def one(item):
            key, prompt = item
            try:
                return key, self.complete_json(prompt, system, schema)
            except Exception as e:
                _log(f"request {key} failed: {type(e).__name__}: {e}")
                return key, None
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return {k: v for k, v in pool.map(one, prompts.items()) if v is not None}


class AnthropicClient(Client):
    """Claude through the official SDK, with structured outputs and an effort level.

    With batch=True, complete_json_many sends its prompts as one Message Batch
    (half price, asynchronous). The batch id is saved under state_dir before
    waiting, so an interrupted run resumes the same batch instead of paying for
    a new one.
    """

    def __init__(self, model=None, effort=None, batch=False):
        import anthropic  # only needed for this backend

        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or os.getenv("READER_MODEL", "claude-sonnet-5-5")
        self.effort = effort or os.getenv("READER_EFFORT", "medium")
        self.batch = batch
        self.base_url = "Claude Batch API" if batch else "Claude API"
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _params(self, prompt, system, schema, max_tokens):
        return {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        }

    def _answer(self, message):
        self.calls += 1
        self.prompt_tokens += message.usage.input_tokens
        self.completion_tokens += message.usage.output_tokens  # includes thinking
        if message.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"{self.model} stopped with {message.stop_reason}")
        return json.loads(next(b.text for b in message.content if b.type == "text"))

    def complete_json(self, prompt, system=SYSTEM_PROMPT, schema=RESPONSE_SCHEMA):
        # Streamed so a long thinking phase can't hit the HTTP timeout; at
        # medium effort Haiku has used over 16k tokens on four paragraphs.
        with self._client.messages.stream(**self._params(prompt, system, schema, 64000)) as stream:
            return self._answer(stream.get_final_message())

    def complete_json_many(self, prompts, system=SYSTEM_PROMPT, schema=RESPONSE_SCHEMA,
                           workers=4, state_dir=None, poll=20):
        if not self.batch:
            return super().complete_json_many(prompts, system, schema, workers)
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        # Custom ids must be short and plain, so map our keys onto c0, c1, ...
        keys = list(prompts)
        ids = {f"c{n}": k for n, k in enumerate(keys)}
        state = None
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)
            digest = hashlib.sha256(json.dumps(
                [self.model, self.effort, system, schema, [prompts[k] for k in keys]]
            ).encode()).hexdigest()[:16]
            state = os.path.join(state_dir, f"batch-{digest}.json")

        batch_id = None
        if state and os.path.exists(state):
            with open(state) as f:
                batch_id = json.load(f)["batch_id"]
            self._report(f"resuming batch {batch_id}")
        if not batch_id:
            batch = self._client.messages.batches.create(requests=[
                Request(custom_id=cid, params=MessageCreateParamsNonStreaming(
                    **self._params(prompts[k], system, schema, 32000)))
                for cid, k in ids.items()
            ])
            batch_id = batch.id
            if state:
                with open(state, "w") as f:
                    json.dump({"batch_id": batch_id, "created": time.time()}, f)
            self._report(f"submitted batch with {len(ids)} requests")

        start = time.time()
        while True:
            b = self._client.messages.batches.retrieve(batch_id)
            if b.processing_status == "ended":
                break
            c = b.request_counts
            self._report(f"batch {b.processing_status.replace('_', ' ')}: {c.succeeded} of {len(ids)} requests done, "
                         f"{time.time() - start:.0f} s")
            time.sleep(poll)

        out = {}
        for r in self._client.messages.batches.results(batch_id):
            key = ids.get(r.custom_id)
            if r.result.type != "succeeded":
                _log(f"request {key}: {r.result.type}")
                continue
            try:
                out[key] = self._answer(r.result.message)
            except Exception as e:
                _log(f"request {key} failed: {type(e).__name__}: {e}")
        return out


def make_client(**kwargs):
    """Claude models go through the Anthropic SDK; anything else through Client."""
    model = kwargs.get("model") or os.getenv("READER_MODEL", "")
    if model.startswith("claude-") or os.getenv("READER_PROVIDER") == "anthropic":
        return AnthropicClient(model=kwargs.get("model"), effort=kwargs.get("effort"),
                               batch=bool(kwargs.get("batch")))
    kwargs.pop("effort", None)
    kwargs.pop("batch", None)
    return Client(**kwargs)


def _parse_json(text):
    # Some endpoints ignore response_format and wrap the JSON in prose or fences.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _project(prev_words, candidate):
    """Indices of prev_words that survive in candidate, judged by a word diff.

    Equal blocks are kept, replaced blocks count as kept (the model reworded
    them, so the original stays), deleted blocks are dropped, inserted words
    are ignored.
    """
    opcodes = worddiff.opcodes(" ".join(prev_words), candidate)
    kept = []
    for tag, i1, i2, _, _ in opcodes:
        if tag in ("equal", "replace"):
            kept.extend(range(i1, i2))
    return kept


def nest_levels(paragraph, candidates):
    """Turn the model's level texts into nested lists of original word indices.

    Returns (levels, rewrites): levels[0] is every word; levels[k] is a subset
    of levels[k-1]. rewrites counts candidate words that were not plain
    deletions (added or reworded), a measure of how well the model obeyed.
    """
    words = paragraph.split()
    levels = [list(range(len(words)))]
    rewrites = 0
    for candidate in candidates[:LEVELS]:
        prev = levels[-1]
        prev_words = [words[i] for i in prev]
        opcodes = worddiff.opcodes(" ".join(prev_words), candidate)
        rewrites += sum(j2 - j1 for tag, _, _, j1, j2 in opcodes if tag in ("insert", "replace"))
        levels.append([prev[i] for i in _project(prev_words, candidate)])
    while len(levels) < LEVELS + 1:  # model returned too few levels
        levels.append(levels[-1])
    return levels, rewrites


_SENTENCE_END = re.compile(r"[.!?]+[\"'”’)\]]*$")


def display_level(words, kept):
    """Text of one level, tidied so it reads as sentences when shown on its own.

    Levels hold original words, so deleting a sentence's last word also deletes
    its full stop, and deleting its first word leaves it lowercase. Returns
    (index, display text) pairs for the kept words with the sentence's end
    punctuation moved onto its last kept word, its first kept word
    capitalised, the leftover comma of a removed aside dropped, and the comma
    that closed a removed clause moved onto the word before it.
    """
    sentence_of, starts, ends, s = [], {}, {}, 0
    for i, w in enumerate(words):
        sentence_of.append(s)
        starts.setdefault(s, i)
        ends[s] = i
        if _SENTENCE_END.search(w):
            s += 1
    shown = {i: words[i] for i in kept}
    for a, b in zip(kept, kept[1:]):
        if b == a + 1 or not words[b - 1].endswith(",") or sentence_of[a] != sentence_of[b]:
            continue
        if words[a].endswith(","):
            # An aside cut out from between two commas leaves the opening comma
            # behind ("operators, such as European Sleeper, have" -> "operators, have").
            shown[a] = shown[a][:-1]
        elif words[a][-1:].isalnum() or words[a][-1:] in "\"')”’]":
            # A cut that ended a clause takes its comma with it ("By showcasing
            # these features that evolved alongside Smalltalk, we" -> "features, we").
            shown[a] = shown[a] + ","
    by_sentence = {}
    for i in kept:
        by_sentence.setdefault(sentence_of[i], []).append(i)
    for s, idx in by_sentence.items():
        first, last = idx[0], idx[-1]
        end = _SENTENCE_END.search(words[ends[s]])
        if last != ends[s] and end:
            shown[last] = shown[last].rstrip(",;:—–-") + end.group(0)
        if first != starts[s] and shown[first][:1].islower():
            shown[first] = shown[first][0].upper() + shown[first][1:]
    return [(i, shown[i]) for i in kept]


def _chunks(paragraphs):
    chunk, size = [], 0
    for i, p in enumerate(paragraphs):
        if chunk and size + len(p) > MAX_CHARS_PER_CALL:
            yield chunk
            chunk, size = [], 0
        chunk.append(i)
        size += len(p)
    if chunk:
        yield chunk


def _prompt(paragraphs, indices, title, where, context):
    numbered = "\n\n".join(f"[{n}] {paragraphs[i]}" for n, i in enumerate(indices, 1))
    return PROMPT_TEMPLATE.format(
        where=where, levels=LEVELS, title=title or "(none)",
        context=f"\n{context}\n" if context else "", paragraphs=numbered,
    )


def _unpack(answer, indices):
    by_id = {}
    for entry in (answer or {}).get("paragraphs", []):
        try:
            by_id[int(entry["id"])] = [str(s) for s in entry["levels"]]
        except (KeyError, TypeError, ValueError):
            continue
    return {i: by_id.get(n) for n, i in enumerate(indices, 1)}


def _result(paragraph, candidates):
    levels, rewrites = nest_levels(paragraph, candidates)
    words = paragraph.split()
    display = [display_level(words, level) for level in levels]
    texts = [" ".join(t for _, t in shown) for shown in display]
    return {"levels": levels, "display": display, "texts": texts, "rewrites": rewrites}


def shorten_document(sections, title=None, client=None, state_dir=None):
    """Shorten every paragraph of a document made of sections.

    sections is a list of {"paragraphs": [...], "context": str or None}; the
    context (for a paper: the abstract and the section heading) goes into the
    prompt. Each section is split into chunks of at most MAX_CHARS_PER_CALL and
    every chunk becomes one request, all sent together (as one batch if the
    client is in batch mode). Paragraphs the model skipped get one retry on
    their own, sent directly.

    Returns, per section, a list with one dict per paragraph: 'levels' (nested
    word index lists, level 0 = all words), 'display' (per level, (index, text)
    pairs from display_level), 'texts' (each level as one string), 'rewrites'.
    """
    client = client or make_client()
    prompts, plan = {}, {}
    for s, section in enumerate(sections):
        paragraphs = section["paragraphs"]
        chunks = list(_chunks(paragraphs))
        where = "an article" if len(sections) == 1 and len(chunks) == 1 else "part of an article"
        for c, indices in enumerate(chunks):
            key = f"s{s}c{c}"
            prompts[key] = _prompt(paragraphs, indices, title, where, section.get("context"))
            plan[key] = (s, indices)

    answers = client.complete_json_many(prompts, state_dir=state_dir)
    found = {}
    for key, (s, indices) in plan.items():
        for i, candidates in _unpack(answers.get(key), indices).items():
            found[(s, i)] = candidates

    missing = [k for k, v in found.items() if not v]
    if missing:
        client._report(f"retrying {len(missing)} paragraphs directly")
    for s, i in missing:
        section = sections[s]
        prompt = _prompt(section["paragraphs"], [i], title, "one paragraph of an article", section.get("context"))
        try:
            found[(s, i)] = _unpack(client.complete_json(prompt), [i])[i] or []
        except Exception as e:
            _log(f"paragraph {s}/{i} failed again: {e}")
            found[(s, i)] = []

    return [
        [_result(p, found[(s, i)]) for i, p in enumerate(section["paragraphs"])]
        for s, section in enumerate(sections)
    ]


def shorten_article(paragraphs, title=None, client=None, state_dir=None):
    """Shorten a plain list of paragraphs; see shorten_document."""
    return shorten_document([{"paragraphs": paragraphs}], title, client, state_dir)[0]
