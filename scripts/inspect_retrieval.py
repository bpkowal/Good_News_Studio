#!/usr/bin/env python3
"""Inspect MiniLM corpus retrieval on stored worlds. No workspace LLM calls.

Use this to iterate query text, identity/case mix, and per-framework hits
against committed traces without running parliament.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from global_workspace.framework_retrieval import load_corpus_passages, retrieve_framework_evidence
from global_workspace.rag_ablation import (
    AGENT_RETRIEVAL_SPECS,
    make_embedder,
    select_committed_worlds,
    world_case_query,
)
from global_workspace.retrieval_trace import serialize_retrieval_result


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "workspace_outputs",
    )
    parser.add_argument("--list-worlds", action="store_true")
    parser.add_argument("--world", help="World label substring, e.g. reservoir")
    parser.add_argument(
        "--agent",
        action="append",
        dest="agents",
        choices=tuple(AGENT_RETRIEVAL_SPECS),
        help="Repeatable. Default: all frameworks on that world",
    )
    parser.add_argument(
        "--query-mode",
        choices=("scenario", "world", "custom"),
        default="scenario",
        help="scenario=full canonical text (current pipeline); world=admitted compact query",
    )
    parser.add_argument("--query", help="Required when --query-mode custom")
    parser.add_argument(
        "--framework-weight",
        type=float,
        default=0.65,
        help="Mix with case_score when identity scoring is on (default 0.65 = current pipeline)",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _pick_world(worlds, needle: str | None):
    if needle is None:
        return worlds[0] if worlds else None
    key = needle.casefold()
    for world in worlds:
        if key in world.label.casefold() or key in str(world.path).casefold():
            return world
    return None


def _print_result(agent: str, query: str, record: dict) -> None:
    print(f"\n=== {agent} ===")
    print(f"query: {query[:280]}{'…' if len(query) > 280 else ''}")
    print(
        f"candidates={record['candidate_count']} rejected={record['rejected_count']} "
        f"core={record['has_core_evidence']}"
    )
    for index, item in enumerate(record.get("evidence") or [], start=1):
        text = " ".join(str(item.get("text") or "").split())
        print(
            f"  [{index}] {item.get('tier')} fw={item.get('framework_score'):.2f} "
            f"case={item.get('case_score'):.2f} rank={item.get('final_score'):.2f} "
            f"{item.get('source_file')} {item.get('author')}"
        )
        print(f"      {text[:220]}{'…' if len(text) > 220 else ''}")


def main() -> int:
    args = parse_args()
    worlds = select_committed_worlds(args.output_dir, limit=12)
    if args.list_worlds or args.world is None:
        for world in worlds:
            print(f"{world.label}\t{world.path.name}")
        return 0 if worlds else 2
    world = _pick_world(worlds, args.world)
    if world is None:
        print(f"No world matching {args.world!r}", file=sys.stderr)
        return 2
    if args.query_mode == "custom":
        if not args.query:
            print("--query is required for --query-mode custom", file=sys.stderr)
            return 2
        query = args.query
    elif args.query_mode == "world":
        query = world_case_query(world.trace, fallback=world.scenario)
    else:
        query = world.scenario
    agents = tuple(args.agents) if args.agents else tuple(world.agents)
    embedder = make_embedder()
    payload = {"world": world.label, "query_mode": args.query_mode, "query": query, "agents": {}}
    for name in agents:
        spec = AGENT_RETRIEVAL_SPECS[name]
        passages = load_corpus_passages(
            spec["corpus_dir"], framework=name, max_chars=250,
        )
        result = retrieve_framework_evidence(
            passages,
            query=query,
            embedder=embedder,
            query_lens=spec["query_lens"],
            identity_tags=spec["identity_tags"],
            core_evidence_roles=spec["core_evidence_roles"],
            tag_weights={},
            thresholds=spec["thresholds"],
            limit=max(1, args.limit),
            prefer_direct_quotes=spec["prefer_direct_quotes"],
            separate_identity_scoring=spec["separate_identity_scoring"],
            framework_weight=args.framework_weight,
        )
        record = serialize_retrieval_result(result, mode="inspect")
        payload["agents"][name] = record
        if not args.json:
            _print_result(name, query, record)
    if args.json:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
