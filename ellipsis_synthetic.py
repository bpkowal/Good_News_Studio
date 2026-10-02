"""Synthetic verb-phrase gaps built from Hoosier training phrases.

Each row keeps a missing phrase that was already the spelled-out gap in a
training group. The held-out Hoosier groups and the dilemma lines are not
sources. Two patterns place that phrase under a matching auxiliary while
another finite verb sits nearby, so the label is not always the main verb
and not always the nearest verb.
"""

import json
from collections import defaultdict
from pathlib import Path

import ellipsis_proposer as proposer
from ellipsis_corpus import load_examples, rows_for_split, split_of

OUT_PATH = Path(__file__).resolve().parent / "resources" / "ellipsis_synthetic_train.jsonl"
PROBE_MARKS = (
    "pull the lever", "the medicine", "five workers", "the dog", "the child",
    "maria", "lila", "omar", "nora",
)
NEARER = ("locks the gate", "calls the office")
EARLIER = ("open the window", "paint the door")


def training_phrases(nlp):
    """Finite verb phrases that are the spelled-out gap in a training group."""
    grouped = defaultdict(list)
    for row in rows_for_split(load_examples(), "train"):
        grouped[row["group_id"]].append(row)
    found = {}
    for group_id, items in grouped.items():
        if split_of(group_id) != "train":
            continue
        gapped = next((item["text"] for item in items if item["label"] == 1), None)
        full = next((item["text"] for item in items if item["label"] == 0), None)
        if not gapped or not full or len(gapped) > 400:
            continue
        doc = nlp(proposer.surface(gapped))
        phrases = [item for item in proposer.antecedent_candidates(doc)
                   if item["kind"] == "verb_and_nominal" and item["finite"]]
        if not phrases:
            continue
        index = proposer._gold_index(phrases, proposer.missing_spans(gapped, full))
        if not index:
            continue
        gold = phrases[index - 1]
        lemmas = gold["lemmas"].split()
        if not (2 <= len(lemmas) <= 6):
            continue
        if lemmas[0] in {"do", "say", "tell", "be", "have"}:
            continue
        key = " ".join(lemmas)
        if any(mark in key.lower() for mark in PROBE_MARKS):
            continue
        if any(mark in key.lower() for mark in NEARER + EARLIER):
            continue
        found.setdefault(key, group_id)
    return found


def _rows_for_phrase(phrase, group_id):
    rows = [dict(
        pattern="main",
        text=f"Alex can {phrase}. Sam can, too.",
        gold=phrase,
        distractor=None,
        source_group=group_id,
    )]
    for nearer in NEARER:
        rows.append(dict(
            pattern="nearer_finite",
            text=f"Alex can {phrase} if Jordan {nearer}. Sam can, too.",
            gold=phrase,
            distractor=nearer,
            source_group=group_id,
        ))
    for earlier in EARLIER:
        rows.append(dict(
            pattern="inner_finite",
            text=f"Jordan will {earlier} after Alex can {phrase}. Sam can, too.",
            gold=phrase,
            distractor=earlier,
            source_group=group_id,
        ))
    return rows


def _gold_is_a_candidate(text, gold, nlp):
    doc = nlp(proposer.surface(text))
    target = proposer.normalize(gold)
    for candidate in proposer.antecedent_candidates(doc):
        if candidate["kind"] != "verb_and_nominal":
            continue
        forms = {proposer.normalize(candidate["text"]), proposer.normalize(candidate["lemmas"])}
        if target in forms:
            return True
    return False


def build_rows(nlp):
    rows = []
    for phrase, group_id in sorted(training_phrases(nlp).items()):
        for row in _rows_for_phrase(phrase, group_id):
            if not _gold_is_a_candidate(row["text"], row["gold"], nlp):
                continue
            row["source_split"] = "train"
            # A slice of the training groups, for checking a later fit. Not the Hoosier dev or test.
            row["synthetic_split"] = "train" if split_of(group_id, salt="synthetic-ellipsis-v1") == "train" else "dev"
            rows.append(row)
    return rows


def write_rows(rows, path=OUT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            handle.write(json.dumps(dict(id=f"syn-{index:04d}", **row)) + "\n")
    return path


def score_current(rows, model):
    """How often the current proposer offers the gold, and how often it offers only the gold."""
    counts = defaultdict(lambda: dict(rows=0, in_set=0, exact=0))
    for row in rows:
        offered = [proposer.normalize(item["text"]) for item in model.propose(row["text"])["proposals"]]
        gold = proposer.normalize(row["gold"])
        bucket = counts[row["pattern"]]
        bucket["rows"] += 1
        bucket["in_set"] += int(gold in offered)
        bucket["exact"] += int(offered == [gold])
    return counts


def main():
    nlp = proposer.detector.get_nlp()
    rows = build_rows(nlp)
    path = write_rows(rows)
    counts = score_current(rows, proposer.load_proposer())
    print(f"wrote {len(rows)} rows to {path.name}")
    for pattern, bucket in sorted(counts.items()):
        print(pattern, bucket)


if __name__ == "__main__":
    main()
