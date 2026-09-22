"""Measure spaCy predicate-anchor recall on the frozen relation challenge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from global_workspace.syntactic_annotation import (
    annotate_text,
    load_english_parser,
    predicate_lemmas,
)


def _gold_events(case: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(
        str(pair[field]).strip()
        for pair in case.get("gold_pairs") or []
        for field in ("source_event", "target_event")
        if str(pair.get(field) or "").strip()
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("evals/pairwise_relation_challenge_set.json"))
    parser.add_argument("--output", type=Path, default=Path("eval_outputs/syntactic-anchor-frozen-20260920/report.json"))
    parser.add_argument("--model", default="en_core_web_sm")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    nlp = load_english_parser(args.model)
    cases: list[dict[str, Any]] = []
    total = matched = fallback_matched = ambiguous = 0
    for case in manifest.get("cases") or []:
        annotation = annotate_text(str(case["scenario"]), nlp=nlp)
        by_lemma: dict[str, list[str]] = {}
        for predicate in annotation["predicates"]:
            by_lemma.setdefault(predicate["lemma"], []).append(predicate["predicate_id"])
        event_rows = []
        for event in _gold_events(case):
            gold_doc = nlp(event)
            strict_lemmas = list(dict.fromkeys(
                token.lemma_.casefold() for token in gold_doc if token.pos_ == "VERB"
            ))
            lemmas = predicate_lemmas(event, nlp=nlp)
            anchors = list(dict.fromkeys(
                predicate_id for lemma in lemmas for predicate_id in by_lemma.get(lemma, [])
            ))
            if anchors and strict_lemmas:
                status = "MATCHED_PREDICATE_LEMMA"
            elif anchors:
                status = "MATCHED_ROOT_LEMMA_FALLBACK"
                fallback_matched += 1
            else:
                status = "UNRESOLVED"
            total += 1
            matched += bool(anchors)
            ambiguous += len(anchors) > 1
            event_rows.append({
                "gold_event": event,
                "gold_predicate_lemmas": lemmas,
                "status": status,
                "predicate_anchor_ids": anchors,
            })
        cases.append({
            "id": case["id"],
            "family": case.get("family"),
            "gold_event_count": len(event_rows),
            "matched_gold_event_count": sum(row["status"].startswith("MATCHED") for row in event_rows),
            "gold_events": event_rows,
            "annotation": annotation,
        })
    report = {
        "report_version": "1.0",
        "protocol": "READ_ONLY_SPACY_GOLD_PREDICATE_ANCHOR_RECALL",
        "manifest": str(args.manifest),
        "model": args.model,
        "aggregate": {
            "gold_event_count": total,
            "matched_gold_event_count": matched,
            "predicate_anchor_recall": matched / total if total else None,
            "root_lemma_fallback_match_count": fallback_matched,
            "multi_anchor_gold_event_count": ambiguous,
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
