# /// script
# requires-python = ">=3.10"
# dependencies = ["anthropic", "requests", "python-dotenv"]
# ///
"""Blind grammar and meaning judge for viewer results.

Every paragraph of every result in viewer/cache/ and viewer/papers/ becomes one
judge item: the original paragraph and its four shortened versions, with no
model or run named. Items are shuffled and sent to the judge model (Opus 5.5 by
default) as one Message Batch. Per version the judge grades each sentence
(A grammatical, B awkward but readable, C ungrammatical) and lists claims the
version makes that the original doesn't support. Verdicts are cached per item
in viewer/cache/judge/, so rerunning only pays for new items.

    uv run viewer/judge.py                 # judge everything new, print the table
    uv run viewer/judge.py --report        # table from cached verdicts only
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import shorten  # noqa: E402

JUDGE_DIR = os.path.join(HERE, "cache", "judge")

SYSTEM = "You are a meticulous copy editor and fact checker. You answer with JSON only."

PROMPT = """Below is an original paragraph followed by {n} shortened versions of it. Each version was made by deleting words from the original. Judge every version on its own against the original.

1. Grammar: split the version into sentences and grade each one.
   A = grammatical and reads naturally.
   B = understandable but awkward, or a minor slip such as a missing article or comma.
   C = ungrammatical: a sentence fragment, a broken clause, or words that no longer fit together.
2. Meaning: list every claim the version makes that the original does not support. Saying less is fine: missing details, examples or qualifications are not a problem by themselves. Flag only what the version asserts differently, such as a dropped negation, modal, quantifier or condition, a changed subject or object, or two ideas joined into a new claim. Quote the version's wording and the original's wording for each. If there are none, return an empty list.

Original:
{original}

{versions}

Answer with JSON: {{"versions": [{{"version": 1, "sentences": [{{"text": "...", "grade": "A"}}], "unsupported": [{{"claim": "...", "original": "..."}}]}}, ...]}}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "versions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "version": {"type": "integer"},
                    "sentences": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "grade": {"type": "string", "enum": ["A", "B", "C"]},
                            },
                            "required": ["text", "grade"],
                            "additionalProperties": False,
                        },
                    },
                    "unsupported": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim": {"type": "string"},
                                "original": {"type": "string"},
                            },
                            "required": ["claim", "original"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["version", "sentences", "unsupported"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["versions"],
    "additionalProperties": False,
}


def runs():
    """(label, record) for every cached text result and processed paper."""
    paths = glob.glob(os.path.join(HERE, "cache", "*.json"))
    paths += glob.glob(os.path.join(HERE, "papers", "*", "result.json"))
    for path in paths:
        with open(path) as f:
            r = json.load(f)
        variant = f" {r['variant']}" if r.get("variant") else ""
        yield f"{r['title'][:28]} | {r['model']} {r['effort'] or ''}{variant}", r


def items(record):
    """Per paragraph: original text and the level texts as display_level shows them."""
    for p in record["paragraphs"]:
        versions = []
        for k in range(1, shorten.LEVELS + 1):
            kept = [i for i, d in enumerate(p["depth"]) if d >= k]
            versions.append(" ".join(t for _, t in shorten.display_level(p["words"], kept)))
        yield " ".join(p["words"]), versions


def prompt_for(original, versions):
    body = "\n\n".join(f"Version {n}:\n{v}" for n, v in enumerate(versions, 1))
    return PROMPT.format(n=len(versions), original=original, versions=body)


def item_key(model, effort, prompt):
    return hashlib.sha256(json.dumps([model, effort, SYSTEM, prompt]).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--no-batch", action="store_true")
    ap.add_argument("--report", action="store_true", help="don't call the judge, only print the table")
    args = ap.parse_args()
    os.makedirs(JUDGE_DIR, exist_ok=True)

    plan = []  # (label, paragraph number, key, prompt)
    for label, record in runs():
        for n, (original, versions) in enumerate(items(record)):
            prompt = prompt_for(original, versions)
            plan.append((label, n, item_key(args.model, args.effort, prompt), prompt))

    todo = {key: prompt for _, _, key, prompt in plan
            if not os.path.exists(os.path.join(JUDGE_DIR, f"{key}.json"))}
    if todo and not args.report:
        keys = list(todo)
        random.Random(0).shuffle(keys)  # blind: runs interleaved, nothing names them
        client = shorten.make_client(model=args.model, effort=args.effort, batch=not args.no_batch)
        print(f"judging {len(keys)} items with {args.model} ({args.effort})", file=sys.stderr)
        verdicts = client.complete_json_many({k: todo[k] for k in keys}, system=SYSTEM, schema=SCHEMA,
                                             state_dir=JUDGE_DIR)
        for key, verdict in verdicts.items():
            with open(os.path.join(JUDGE_DIR, f"{key}.json"), "w") as f:
                json.dump(verdict, f, ensure_ascii=False)
        rate = (2, 10) if client.batch else (4, 20)  # Opus 5.5 $/MTok, halved in a batch
        cost = (client.prompt_tokens * rate[0] + client.completion_tokens * rate[1]) / 1e6
        print(f"{client.calls} verdicts, {client.prompt_tokens} in / {client.completion_tokens} out tokens"
              + (f", ≈ ${cost:.2f}" if args.model == "claude-opus-5-5" else ""), file=sys.stderr)

    # Table: per run and level, share of A/B/C sentences and unsupported claims.
    table, claims = {}, {}
    for label, n, key, _ in plan:
        path = os.path.join(JUDGE_DIR, f"{key}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            verdict = json.load(f)
        for v in verdict.get("versions", []):
            k = v.get("version")
            if k not in range(1, shorten.LEVELS + 1):
                continue
            row = table.setdefault(label, {}).setdefault(k, {"A": 0, "B": 0, "C": 0, "claims": 0})
            for s in v["sentences"]:
                row[s["grade"]] += 1
            row["claims"] += len(v["unsupported"])
            for c in v["unsupported"]:
                claims.setdefault(label, []).append((n + 1, k, c["claim"], c["original"]))

    print(f"\n{'run':<52} {'level':>5} {'A':>5} {'B':>5} {'C':>5} {'claims':>7}")
    for label in sorted(table):
        for k in sorted(table[label]):
            r = table[label][k]
            total = r["A"] + r["B"] + r["C"] or 1
            print(f"{label:<52} {k:>5} {r['A'] / total:>5.0%} {r['B'] / total:>5.0%} "
                  f"{r['C'] / total:>5.0%} {r['claims']:>7}")
    for label in sorted(claims):
        print(f"\nUnsupported claims, {label}:")
        for para, k, claim, original in claims[label]:
            print(f"  ¶{para} L{k}: “{claim}”  vs  “{original}”")


if __name__ == "__main__":
    main()
