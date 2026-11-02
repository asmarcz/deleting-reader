# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "python-dotenv"]
# ///
"""Repair results using the judge's verdicts, without calling a model.

For every claim the judge flagged ("version k says X, the original says Y"),
find Y in the paragraph and look at the words deleted inside that span at
level k. Short runs of deleted words (at most MAX_RUN) are almost always scope
words such as "most of", "often", "short-haul" or "Let us imagine that", so
they are kept up to level k. Longer runs are whole examples or clauses, which
are legitimate cuts, and stay deleted.

Keeping a word up to level k only raises its depth, so levels stay nested.
For text results, each repaired result is saved as a new run next to the
original ("variant": "repaired"), so the viewer and the judge can compare the
two. Papers are judged and repaired as part of processing (viewer/paper.py)
and keep only the final version.

    uv run viewer/repair.py            # repair every judged text result
    uv run viewer/judge.py             # then judge the repaired paragraphs
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import worddiff  # noqa: E402
import judge  # noqa: E402

MAX_RUN = 4
JUDGE_MODEL, JUDGE_EFFORT = "claude-opus-5-5", "medium"


def find_span(words, quote):
    """(start, end) of quote in words, comparing words without case or punctuation."""
    target = [worddiff.clean(w) for w in quote.split()]
    target = [t for t in target if t]
    cleaned = [worddiff.clean(w) for w in words]
    n = len(target)
    for i in range(len(words) - n + 1):
        if cleaned[i:i + n] == target:
            return i, i + n
    return None


def repair_paragraph(p, verdict):
    """New depth list plus a log of what was restored and what couldn't be placed."""
    depth = list(p["depth"])
    restored, unplaced = [], []
    for v in verdict.get("versions", []):
        k = v.get("version")
        for claim in v.get("unsupported", []):
            span = find_span(p["words"], claim["original"])
            if not span:
                unplaced.append((k, claim["original"]))
                continue
            a, b = span
            i = a
            while i < b:  # runs of words deleted by level k inside the span
                if depth[i] >= k:
                    i += 1
                    continue
                j = i
                while j < b and depth[j] < k:
                    j += 1
                if j - i <= MAX_RUN:
                    for w in range(i, j):
                        depth[w] = max(depth[w], k)
                    restored.append((k, " ".join(p["words"][i:j])))
                i = j
    return depth, restored, unplaced


def repaired_id(rid):
    return f"{rid}-repaired"


def repair_paragraphs(paragraphs, verdicts):
    """Repaired paragraphs plus a log of (paragraph number, level, what, words).

    verdicts has one entry per paragraph; None leaves that paragraph as it is.
    """
    new_paras, log = [], []
    for n, (p, verdict) in enumerate(zip(paragraphs, verdicts), 1):
        if verdict is None:
            new_paras.append(p)
            continue
        depth, restored, unplaced = repair_paragraph(p, verdict)
        new_paras.append({**p, "depth": depth})
        log += [(n, k, "restored", t) for k, t in restored]
        log += [(n, k, "quote not found", t) for k, t in unplaced]
    return new_paras, log


def main():
    done = 0
    for label, record in judge.runs():
        if record.get("variant") == "repaired" or record.get("kind") == "paper":
            continue  # papers are judged and repaired when they are processed
        verdicts = [judge.load(judge.item_key(JUDGE_MODEL, JUDGE_EFFORT, judge.prompt_for(o, v)))
                    for o, v in judge.items(record)]
        missing_verdicts = verdicts.count(None)
        if missing_verdicts == len(verdicts):
            continue  # never judged
        new_paras, log = repair_paragraphs(record["paragraphs"], verdicts)

        rid = repaired_id(record["id"])
        out = {**record, "id": rid, "variant": "repaired", "repair_of": record["id"],
               "paragraphs": new_paras}
        with open(os.path.join(HERE, "cache", f"{rid}.json"), "w") as f:
            json.dump(out, f, ensure_ascii=False)
        done += 1

        restored = [entry for entry in log if entry[2] == "restored"]
        print(f"\n{label}: {len(restored)} runs restored"
              + (f", {missing_verdicts} paragraphs without a verdict" if missing_verdicts else ""))
        for n, k, what, text in log:
            print(f"  ¶{n} L{k} {what}: {text}")
    print(f"\n{done} repaired results written", file=sys.stderr)


if __name__ == "__main__":
    main()
