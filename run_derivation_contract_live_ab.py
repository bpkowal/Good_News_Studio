#!/usr/bin/env python3
"""Live A/B of broad branch reconstruction versus metadata-only repair cards."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from typing import Any

from dotenv import load_dotenv

from global_workspace.local_specialists import _admit_action_source_rows
from global_workspace.openai_backend import OpenAIWorkspaceLLM
from global_workspace.repair_experiment import graph_diff
from global_workspace.structured_io import call_json_llm, extract_json
from global_workspace.world_validation import (
    format_repair_guidance_for_prompt,
    repair_guidance_cards,
)


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"candidate_json": {"type": "string"}},
    "required": ["candidate_json"],
    "additionalProperties": False,
}


def _ids(candidate: dict[str, Any]) -> tuple[set[str], set[tuple[str, str, str, str]]]:
    world = candidate.get("world_model") or {}
    effects = {
        str(row.get("effect_id")) for row in world.get("effects") or []
        if isinstance(row, dict) and row.get("effect_id")
    }
    links = {
        (
            str(row.get("action_id") or ""), str(row.get("source_id") or ""),
            str(row.get("link_relation") or row.get("relation") or ""),
            str(row.get("target_id") or ""),
        )
        for row in world.get("causal_links") or [] if isinstance(row, dict)
    }
    return effects, links


def _run_arm(
    *, llm: OpenAIWorkspaceLLM, arm: str, candidate: dict[str, Any],
    guidance: str, scenario: str, actions: list[str], clauses: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt = f"""You are repairing an already substantially correct typed world graph.
The two experimental arms differ only in repair guidance. Follow the guidance below.
Do not improve, simplify, or reinterpret unrelated content. Return the complete
candidate, including its actions mapping and world_model, as a JSON-encoded string
in candidate_json. The decoded string must be one JSON object and contain no Markdown.

SCENARIO:
{scenario}

CANONICAL ACTIONS:
{json.dumps(actions, ensure_ascii=False)}

REPAIR GUIDANCE ({arm}):
{guidance}

REJECTED CANDIDATE:
{json.dumps(candidate, ensure_ascii=False, sort_keys=True)}
"""
    started = time.monotonic()
    output = call_json_llm(
        llm, prompt, max_tokens=12288, temperature=0.0,
        schema=OUTPUT_SCHEMA, call_kind="derivation_contract_live_ab",
        call_metadata={"arm": arm},
    )
    elapsed = round(time.monotonic() - started, 3)
    raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
    envelope = extract_json(raw)
    repaired = json.loads(str(envelope["candidate_json"]))
    if not isinstance(repaired, dict):
        raise ValueError(f"{arm} returned a non-object candidate")
    admission = _admit_action_source_rows(
        repaired, actions, ["A0", "A1"], clauses,
        allow_action_text_evidence=True,
    )
    before_effects, before_links = _ids(candidate)
    after_effects, after_links = _ids(repaired)
    diff = graph_diff(candidate, repaired)
    return {
        "arm": arm,
        "elapsed_seconds": elapsed,
        "status": admission.get("status"),
        "world_model_status": admission.get("world_model_status"),
        "validation_issues": admission.get("validation_issues") or [],
        "compiler_loss_telemetry": admission.get("compiler_loss_telemetry") or {},
        "semantic_nodes_preserved": before_effects <= after_effects,
        "causal_links_preserved": before_links <= after_links,
        "effects_added": sorted(after_effects - before_effects),
        "effects_removed": sorted(before_effects - after_effects),
        "links_added": sorted(after_links - before_links),
        "links_removed": sorted(before_links - after_links),
        "unrelated_changed_nodes": [
            row for row in diff.get("nodes_changed") or []
            if row.get("node") not in {"effects:E1C", "effects:E3C"}
        ],
        "graph_diff": diff,
        "candidate": repaired,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic", type=Path)
    parser.add_argument("--model", default="o3")
    parser.add_argument(
        "--output", type=Path,
        default=Path("eval_outputs/derivation-contract-live-ab.json"),
    )
    args = parser.parse_args()
    load_dotenv()
    payload = json.loads(args.diagnostic.read_text(encoding="utf-8"))
    grounding = payload["grounding"]
    candidate = copy.deepcopy(grounding["rejected_candidate"])
    clauses = list(grounding.get("clauses") or [])
    actions = [str(value) for value in payload["canonical_actions"]]
    scenario = str(payload.get("canonical_scenario") or payload.get("ethical_problem") or "")

    current = _admit_action_source_rows(
        candidate, actions, ["A0", "A1"], clauses,
        allow_action_text_evidence=True,
    )
    narrow_cards = repair_guidance_cards(
        current.get("validation_issues") or [], candidate, clauses=clauses,
    )
    saved_broad_issues = [
        row for row in grounding.get("validation_issues") or []
        if isinstance(row, dict)
        and str(row.get("code")) == "EXCLUSIVE_ALLOCATION_BRANCH_MISSING"
    ]
    broad_cards = repair_guidance_cards(
        saved_broad_issues, candidate, clauses=clauses,
    )
    if not broad_cards or not narrow_cards:
        raise ValueError("diagnostic does not provide both broad and narrow repair cards")

    llm = OpenAIWorkspaceLLM(args.model, timeout=600.0)
    arms = []
    for arm, cards in (
        ("A_LEGACY_BRANCH_RECONSTRUCTION", broad_cards),
        ("B_METADATA_ONLY", narrow_cards),
    ):
        result = _run_arm(
            llm=llm, arm=arm, candidate=candidate,
            guidance=format_repair_guidance_for_prompt(cards),
            scenario=scenario, actions=actions, clauses=clauses,
        )
        arms.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "experiment_version": "1.0", "model": args.model,
            "source_diagnostic": str(args.diagnostic), "arms": arms,
        }, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            key: value for key, value in result.items()
            if key not in {"candidate", "graph_diff", "validation_issues"}
        }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
