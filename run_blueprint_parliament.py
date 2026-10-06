"""Choose a source-grounded blueprint and run it through Parliament.

The command accepts arbitrary scenario text, assesses its question before world
construction, asks the cloze chooser to compare candidate blueprints, and writes
every candidate attempt. Parliament receives only the chosen contract-valid
proposal. Withheld constructions still produce complete diagnostic artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import re
import shutil
from typing import Sequence

import parsing_game_Z10 as z10

from blueprint_cloze_chooser import (
    assess_question,
    choose_by_cloze,
    openai_complete,
)
from blueprint_proposal_contract import validate_proposal
from blueprint_admission_core import supported_core
from blueprint_discourse import (
    discourse_graph_rows,
    has_discourse,
)
from blueprint_graph_amendments import expand_candidates
from blueprint_primitive_composition import extract_inventory, append_composition


DEFAULT_SCENARIO = (
    "A clinic has one dose of medicine. "
    "Ada can give the medicine to Ben or Cara, but not both. "
    "If Ada gives the medicine to Ben, Ben has a 95% chance of survival. "
    "If Ada gives the medicine to Cara, Cara has a 5% chance of survival. "
    "The patient who does not get the medicine will die."
)
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PARLIAMENT_ROOT = Path("/tmp/parliament-smoke-614af0c")
DEFAULT_PARLIAMENT_PYTHON = Path("/tmp/parliament-smoke-env/bin/python")
DEFAULT_OPENAI_ENV = Path(
    "/Users/benjaminkowal/Documents/Python/Python Coding/RAGAIMODEL/.env"
)
RELENT_FRAMEWORKS = (
    "utilitarian", "deontological", "virtue", "care", "rawlsian",
)
PROTOTYPE_FRAMEWORKS = ("utilitarian", "deontological")
_AGENT_ALIASES = {
    "util": "utilitarian", "utility": "utilitarian",
    "deon": "deontological", "duty": "deontological",
    "virtue ethics": "virtue",
    "care ethics": "care",
    "rawls": "rawlsian",
}


def _authorized_proposal(blueprint_result: dict) -> dict:
    proposals = blueprint_result.get("proposals") or []
    if not proposals:
        raise ValueError("Blueprint did not produce a proposal envelope")
    proposal = proposals[0]
    errors = validate_proposal(proposal)
    if errors:
        raise ValueError("Blueprint proposal contract failed: " + "; ".join(errors))
    if not proposal.get("admission_authorized") or proposal.get("candidate") is None:
        reasons = proposal.get("world_withheld") or proposal.get("construction_problems")
        raise ValueError("Blueprint proposal is withheld: " + json.dumps(reasons))
    return proposal


def _label(value: object) -> str:
    return html.escape(" ".join(str(value or "").split()), quote=True)


def _semantic_key(value: object) -> str:
    text = " ".join(str(value or "").casefold().split())
    word = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
    irregular = {
        "has": "have", "had": "have", "survival": "survive",
        "gave": "give", "given": "give", "died": "die",
    }
    if word in irregular:
        return irregular[word]
    for suffix in ("ing", "ed", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[:-len(suffix)]
    return word


def compare_z10_blueprints(package: dict, blueprint_result: dict) -> dict:
    """Compare independent representations without selecting or merging either."""
    nodes = {row["id"]: row for row in package.get("nodes", [])}
    z10_predicates = {
        _semantic_key(nodes.get(row.get("arguments", {}).get("proposition"), {}).get("predicate"))
        for row in package.get("candidates", []) if row.get("type") == "PREDICATION"
    } - {""}
    z10_roles = sorted({
        str(row.get("value")) for row in package.get("candidates", [])
        if row.get("type") == "PARTICIPANT" and row.get("value")
    })
    z10_relations = sorted({
        row.get("type") for row in package.get("candidates", [])
        if row.get("type") in {"CONDITIONAL_ON", "EVENT_LINK", "MODALITY", "QUANTITY"}
    })
    comparisons = []
    for attempt in blueprint_result.get("candidate_attempts", []):
        proposal = attempt.get("proposal", {})
        candidate = proposal.get("candidate") or {}
        world = candidate.get("world_model")
        if not world:
            comparisons.append({
                "blueprint_id": attempt.get("blueprint_id"),
                "rank": attempt.get("rank"), "status": "NO_BLUEPRINT_WORLD",
                "agreement": [], "z10_only": sorted(z10_predicates),
                "blueprint_only": [], "conflicts": [],
            })
            continue
        blueprint_predicates = {
            _semantic_key(row.get("predicate")) for row in world.get("effects", [])
        } - {""}
        copied_words = {
            _semantic_key(token)
            for row in world.get("effects", [])
            for token in re.findall(
                r"[A-Za-z]+", str(row.get("source_proposition") or row.get("outcome") or ""))
        }
        blueprint_predicates.update(z10_predicates & copied_words)
        common = sorted(z10_predicates & blueprint_predicates)
        z10_only = sorted(z10_predicates - blueprint_predicates)
        blueprint_only = sorted(blueprint_predicates - z10_predicates)
        blueprint_relations = sorted({
            row.get("link_relation") or row.get("relation")
            for row in world.get("causal_links", [])
        } | ({"CONDITIONAL_ON"} if world.get("conditions") else set()))
        if any(row.get("structural_basis") == "conditional_parent"
               for row in proposal.get("relation_alternatives", [])):
            blueprint_relations = sorted({*blueprint_relations, "CONDITIONAL_ON"})
        conflicts = []
        if "CONDITIONAL_ON" in z10_relations and "CONDITIONAL_ON" not in blueprint_relations:
            conflicts.append({
                "kind": "scope", "z10": "CONDITIONAL_ON",
                "blueprint": "no condition record",
            })
        comparisons.append({
            "blueprint_id": attempt.get("blueprint_id"), "rank": attempt.get("rank"),
            "status": "AGREEMENT" if common and not conflicts else "PARTIAL",
            "agreement": common, "z10_only": z10_only,
            "blueprint_only": blueprint_only, "conflicts": conflicts,
            "z10_relations": z10_relations,
            "blueprint_relations": blueprint_relations,
            "z10_participant_roles": z10_roles,
            "blueprint_party_labels": sorted(
                row.get("label", "") for row in world.get("parties", [])),
        })
    return {
        "comparison_version": "z10-blueprint-comparison/0.1",
        "policy": "independent_nonbinding_comparison",
        "selection_changed": False,
        "z10_package_id": package.get("package_id"),
        "z10_open_question_ids": [row.get("id") for row in package.get("open_questions", [])],
        "candidate_comparisons": comparisons,
    }


def _mermaid_world(world: dict, heading: str) -> list[str]:
    """Render one candidate or admitted world as an isolated Mermaid graph."""
    parties = {row["party_id"]: row for row in world.get("parties", [])}
    effects = {row["effect_id"]: row for row in world.get("effects", [])}
    lines = [f"### {heading}", "", "```mermaid", "flowchart LR"]
    for row in parties.values():
        lines.append(
            f'    {row["party_id"]}["{row["party_id"]} — {_label(row["label"])}'
            f'<br/>{_label(row.get("kind"))}"]'
        )
    for row in world.get("actions", []):
        lines.append(
            f'    {row["action_id"]}["{row["action_id"]} — {_label(row["intervention"])}"]'
        )
    for row in effects.values():
        lines.append(
            f'    {row["effect_id"]}["{row["effect_id"]} — {_label(row["outcome"])}'
            f'<br/>{_label(row.get("directness"))} · {_label(row.get("polarity"))}'
            f' · {_label(row.get("modality"))}"]'
        )
    for row in world.get("conditions", []):
        lines.append(
            f'    {row["condition_id"]}{{"{row["condition_id"]} — '
            f'{_label(row.get("description"))}<br/>{_label(row.get("polarity"))}"}}'
        )
    for row in world.get("actions", []):
        actor = row.get("actor_party_id")
        if actor in parties:
            lines.append(f'    {actor} -->|"actor"| {row["action_id"]}')
        for party_id in row.get("recipient_party_ids", []):
            if party_id in parties:
                lines.append(f'    {row["action_id"]} -->|"recipient"| {party_id}')
        for effect_id in row.get("effect_ids", []):
            if effect_id in effects:
                lines.append(f'    {row["action_id"]} -.->|"effect"| {effect_id}')
    for row in effects.values():
        if row.get("party_id") in parties:
            lines.append(f'    {row["effect_id"]} -.->|"affects"| {row["party_id"]}')
    for row in world.get("causal_links", []):
        relation = row.get("relation") or row.get("link_relation") or "LINKS"
        lines.append(
            f'    {row["source_id"]} -->|"{_label(relation)}"| {row["target_id"]}'
        )
    for condition in world.get("conditions", []):
        description = str(condition.get("description") or "").casefold()
        for action in world.get("actions", []):
            intervention = str(action.get("intervention") or "").casefold()
            if intervention and intervention in description:
                lines.append(
                    f'    {condition["condition_id"]} -.->|"branch scope"| '
                    f'{action["action_id"]}'
                )
    discourse_nodes, discourse_edges = discourse_graph_rows(world)
    for row in discourse_nodes:
        shape = (
            f'    {row["id"]}([" {row["id"]} — {_label(row.get("label"))}'
            f'<br/>{_label(row.get("kind"))} · {_label(row.get("status"))} "])'
        )
        lines.append(shape)
    for row in discourse_edges:
        lines.append(
            f'    {row["source"]} -.->|"{_label(row["relation"])}"| {row["target"]}'
        )
    lines.extend(["```", ""])
    return lines


def _write_candidate_attempts(blueprint_result: dict, output_dir: Path) -> tuple[Path, Path]:
    """Write every ranked candidate, including withheld attempts and missing slots."""
    attempts = blueprint_result.get("candidate_attempts") or []
    json_path = output_dir / "candidate_attempts.json"
    json_path.write_text(
        json.dumps({
            "chosen_blueprint_id": blueprint_result.get("chosen_blueprint_id"),
            "status": blueprint_result.get("status"),
            "question": blueprint_result.get("question"),
            "z10_blueprint_comparison": blueprint_result.get("z10_blueprint_comparison"),
            "semantic_coverage": blueprint_result.get("semantic_coverage", []),
            "attempts": attempts,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Candidate world attempts", "",
        f'Chosen blueprint: **{_label(blueprint_result.get("chosen_blueprint_id") or "none")}**  ',
        f'Chooser status: **{_label(blueprint_result.get("status"))}**', "",
    ]
    for display_index, attempt in enumerate(sorted(attempts, key=lambda a: a["rank"]), 1):
        proposal = attempt["proposal"]
        marker = "selected" if attempt.get("selected") else "alternative"
        lines.extend([
            f'## {display_index}. {_label(attempt["blueprint_id"])} ({marker}; {_label(attempt.get("variant") or "baseline")})', "",
            f'Template status: **{_label(attempt.get("template_status"))}**  ',
            f'Proposal status: **{_label(proposal.get("status"))}**  ',
            f'Contract valid: **{str(bool(attempt.get("contract_valid"))).lower()}**  ',
            "Missing slots: " + (
                "`" + "`, `".join(attempt.get("unfilled_slots") or []) + "`"
                if attempt.get("unfilled_slots") else "none"
            ), "",
        ])
        candidate = proposal.get("candidate")
        if candidate and candidate.get("world_model"):
            lines.extend(_mermaid_world(
                candidate["world_model"],
                f'{_label(attempt["blueprint_id"])} candidate graph',
            ))
            core, overlay = supported_core(proposal)
            if overlay["effects"]:
                lines.extend(["Hypothesis overlay (not submitted as world effects):", ""])
                for effect in overlay["effects"]:
                    lines.append(f'- `{effect["effect_id"]}`: {effect["outcome"]}; '
                                 + ", ".join(overlay["reasons"][effect["effect_id"]]))
                lines.append("")
                lines.extend(_mermaid_world(core["candidate"]["world_model"],
                                           "Supported core for admission"))
            if attempt.get("native_admission"):
                lines.extend(["Native admission:", "", "```json",
                              json.dumps(attempt["native_admission"], indent=2), "```", ""])
        else:
            reasons = proposal.get("world_withheld") or [
                "No candidate world was constructed."
            ]
            lines.extend(["World withheld:", ""])
            lines.extend(f"- {reason}" for reason in reasons)
            lines.append("")
        for retention in proposal.get("unresolved_readings", []):
            if not isinstance(retention, dict) or retention.get("kind") != "semantic_construction_retention":
                continue
            lines.extend(["Source constructions retained outside the admitted world:", "",
                          "Parser proposals below preserve alternatives and local scope; "
                          "they do not establish occurrence or a resolved interpretation.", ""])
            for construction in retention["source_constructions"]:
                lines.extend([f'#### {_label(construction["id"])} — retained, not projected', "",
                              "```mermaid", "flowchart LR"])
                predications = {c["arguments"]["proposition"]: c for c in construction["source_candidates"]
                                if c["type"] == "PREDICATION"}
                for node in construction["source_nodes"]:
                    scope = predications.get(node["id"], {}).get("scope", {})
                    qualifier = scope.get("polarity", "") + " " + ", ".join(
                        c["kind"] for c in scope.get("contexts", []))
                    lines.append(f'    {node["id"]}["{_label(node["label"])}<br/>{_label(qualifier)}"]')
                for source in construction["source_candidates"]:
                    args = source["arguments"]
                    if source["type"] == "PARTICIPANT":
                        lines.append(f'    {args["proposition"]} -->|"{_label(source["value"])}"| {args["mention"]}')
                    elif source["type"] == "EVENT_LINK":
                        lines.append(f'    {args["parent"]} -.->|"{_label(source["value"])} content"| {args["child"]}')
                lines.extend(['    mapping{"Missing world mapping"}',
                              f'    {construction["anchor_id"]} -.-> mapping', "```", "",
                              construction["mapping_question"]["question"], ""])
            missing = retention["missing_construction_questions"]
            lines.extend([f"Unconsumed parser candidates: **{len(missing)}**. "
                          "These are visible mapping gaps, not admission failures.", ""])
            if missing:
                lines.extend(["```mermaid", "flowchart LR",
                              '    gap{"Missing construction mappings"}',
                              '    source["Unconsumed source candidates"] -.-> gap', "```", ""])
                lines.extend(f'- `{q["candidate_ids"][0]}`: {q["question"]}' for q in missing)
                lines.append("")
    markdown_path = output_dir / "candidate_graphs.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path, json_path


def _admit_candidates(blueprint: dict, scenario: str, package: dict,
                      output_dir: Path, parliament_root: Path,
                      parliament_python: Path) -> tuple[dict | None, list[dict]]:
    """Admit every filled core, then choose by the chooser's recorded rank."""
    attempts = blueprint.get("candidate_attempts") or [
        {"rank": 0, "blueprint_id": blueprint.get("chosen_blueprint_id"),
         "proposal": blueprint["proposals"][0], "selected": True}]
    records = []
    admitted = []
    with tempfile.TemporaryDirectory() as directory:
        helper = Path(directory) / "admit_blueprint.py"
        helper.write_text(_parliament_admission_script(), encoding="utf-8")
        for index, attempt in enumerate(sorted(attempts, key=lambda a: a["rank"])):
            original = attempt["proposal"]
            if not original.get("admission_authorized") or not original.get("candidate"):
                attempt["native_admission"] = {"status": "NOT_RUN", "reason": "construction_withheld"}
                records.append({"blueprint_id": attempt["blueprint_id"], **attempt["native_admission"]})
                continue
            proposal, overlay = supported_core(original)
            folder = output_dir / "admission_attempts" / f"{index:02d}_{attempt['blueprint_id']}"
            folder.mkdir(parents=True, exist_ok=True)
            candidate_path = folder / "candidate.json"
            trace_path = folder / "frozen_world_trace.json"
            payload = {"scenario": scenario, "actions": proposal["assignment"],
                       "package": package, "blueprint_result": {"proposals": [proposal]}}
            candidate_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (folder / "hypothesis_overlay.json").write_text(
                json.dumps(overlay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            errors = validate_proposal(proposal)
            record = {"blueprint_id": attempt["blueprint_id"], "rank": attempt["rank"],
                      "variant": attempt.get("variant", "baseline"),
                      "candidate_path": str(candidate_path), "overlay_path": str(folder / "hypothesis_overlay.json")}
            if errors:
                record.update(status="REJECTED", error="; ".join(errors))
            else:
                try:
                    completed = subprocess.run([
                        str(parliament_python), str(helper), str(parliament_root.resolve()),
                        str(candidate_path.resolve()), str(trace_path.resolve()),
                        str(PROJECT_ROOT),
                    ], text=True, capture_output=True, timeout=120)
                    if completed.returncode:
                        record.update(status="REJECTED", returncode=completed.returncode,
                                      error=completed.stdout + completed.stderr)
                    else:
                        record.update(status="ADMITTED", admission=json.loads(completed.stdout),
                                      trace_path=str(trace_path))
                        admitted.append({"proposal": proposal, "attempt": attempt, "record": record,
                                         "overlay": overlay})
                except subprocess.TimeoutExpired:
                    record.update(status="REJECTED", error="native_admission_timeout")
            attempt["native_admission"] = record
            records.append(record)
    return (admitted[0] if admitted else None), records


def _scenario_identifier(scenario: str, blueprint_id: str | None) -> str:
    digest = hashlib.sha256(scenario.encode("utf-8")).hexdigest()[:12]
    return f"{blueprint_id or 'withheld'}_{digest}"


def _scenario_text(text: str | None, scenario_file: Path | None) -> str:
    if scenario_file is not None:
        value = scenario_file.expanduser().read_text(encoding="utf-8")
    elif text is not None:
        value = text
    else:
        value = DEFAULT_SCENARIO
    value = value.strip()
    if not value:
        raise ValueError("Scenario text is empty")
    return value


def _write_world_topology(trace_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Write machine-readable and Mermaid views of the admitted world."""
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    grounding = trace["action_source_grounding"]
    world = grounding.get("world_model_1_4") or grounding["world_model"]
    parties = {row["party_id"]: row for row in world.get("parties", [])}
    effects = {row["effect_id"]: row for row in world.get("effects", [])}
    nodes: list[dict] = []
    edges: list[dict] = []

    for row in parties.values():
        nodes.append({"id": row["party_id"], "kind": "party", "label": row["label"]})
    for row in world.get("actions", []):
        nodes.append({"id": row["action_id"], "kind": "action",
                      "label": row["intervention"]})
        edges.append({"source": row["actor_party_id"], "relation": "ACTOR", "target": row["action_id"]})
        for party_id in row.get("recipient_party_ids", []):
            edges.append({"source": row["action_id"], "relation": "RECIPIENT", "target": party_id})
        for effect_id in row.get("effect_ids", []):
            edges.append({"source": row["action_id"], "relation": "HAS_EFFECT", "target": effect_id})
    for row in effects.values():
        nodes.append({"id": row["effect_id"], "kind": "effect", "label": row["outcome"],
                      "polarity": row["polarity"], "modality": row["modality"],
                      "directness": row["directness"]})
        edges.append({"source": row["effect_id"], "relation": "AFFECTS", "target": row["party_id"]})
    for row in world.get("causal_links", []):
        edges.append({"source": row["source_id"], "relation": row["relation"],
                      "target": row["target_id"], "modality": row["modality"]})
    for condition in world.get("conditions", []):
        nodes.append({
            "id": condition["condition_id"], "kind": "condition",
            "label": condition.get("description", ""),
            "polarity": condition.get("polarity"),
        })
        description = str(condition.get("description") or "").casefold()
        for action in world.get("actions", []):
            intervention = str(action.get("intervention") or "").casefold()
            if intervention and intervention in description:
                edges.append({
                    "source": condition["condition_id"],
                    "relation": "BRANCH_SCOPE", "target": action["action_id"],
                })
    discourse_nodes, discourse_edges = discourse_graph_rows(world)
    nodes.extend(discourse_nodes)
    edges.extend(discourse_edges)

    occurrence = grounding.get("world_model") or {}
    admission_status = grounding.get("status") or (
        (world.get("admission") or {}).get("status")
    ) or ""
    admitted_effect_ids = list(
        (world.get("admission") or {}).get("admitted_effect_ids")
        or [row["effect_id"] for row in occurrence.get("effects") or []]
    )

    topology = {
        "schema_version": world.get("schema_version"),
        "admission": {
            "status": admission_status,
            "admitted_effect_ids": admitted_effect_ids,
        },
        "nodes": nodes,
        "edges": edges,
        "conditions": world.get("conditions", []),
        "temporal_relations": world.get("temporal_relations", []),
        "counterfactual_links": world.get("counterfactual_links", []),
    }
    json_path = output_dir / "world_state_topology.json"
    json_path.write_text(json.dumps(topology, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")

    lines = [
        "# Admitted world-state topology", "", "```mermaid", "flowchart LR",
    ]
    for row in parties.values():
        lines.append(
            f'    {row["party_id"]}["{row["party_id"]} — {_label(row["label"])}<br/>{_label(row["kind"])}"]'
        )
    for row in world.get("actions", []):
        lines.append(
            f'    {row["action_id"]}["{row["action_id"]} — {_label(row["intervention"])}"]'
        )
    for row in effects.values():
        lines.append(
            f'    {row["effect_id"]}["{row["effect_id"]} — {_label(row["outcome"])}'
            f'<br/>{_label(row["directness"])} · {_label(row["polarity"])} · {_label(row["modality"])}"]'
        )
    for row in world.get("conditions", []):
        lines.append(
            f'    {row["condition_id"]}{{"{row["condition_id"]} — '
            f'{_label(row.get("description"))}<br/>{_label(row.get("polarity"))}"}}'
        )
    lines.append("")
    for row in world.get("actions", []):
        lines.append(f'    {row["actor_party_id"]} -->|"actor"| {row["action_id"]}')
        for party_id in row.get("recipient_party_ids", []):
            lines.append(f'    {row["action_id"]} -->|"recipient"| {party_id}')
        for effect_id in row.get("effect_ids", []):
            lines.append(f'    {row["action_id"]} -.->|"effect"| {effect_id}')
    for row in effects.values():
        lines.append(f'    {row["effect_id"]} -.->|"affects"| {row["party_id"]}')
    for row in world.get("causal_links", []):
        lines.append(
            f'    {row["source_id"]} -->|"{_label(row["relation"])}"| {row["target_id"]}'
        )
    for condition in world.get("conditions", []):
        description = str(condition.get("description") or "").casefold()
        for action in world.get("actions", []):
            intervention = str(action.get("intervention") or "").casefold()
            if intervention and intervention in description:
                lines.append(
                    f'    {condition["condition_id"]} -.->|"branch scope"| '
                    f'{action["action_id"]}'
                )
    for row in discourse_nodes:
        lines.append(
            f'    {row["id"]}(["{row["id"]} — {_label(row.get("label"))}'
            f'<br/>{_label(row.get("kind"))} · {_label(row.get("status"))}"])'
        )
    for row in discourse_edges:
        lines.append(
            f'    {row["source"]} -.->|"{_label(row["relation"])}"| {row["target"]}'
        )
    lines.extend([
        "```", "",
        f'Admission: **{_label(admission_status)}**  ',
        f'Admitted effects: `{", ".join(admitted_effect_ids)}`  ',
        f'Conditions: **{len(world.get("conditions", []))}**; '
        f'Temporal links: **{len(world.get("temporal_relations", []))}**; '
        f'Counterfactual links: **{len(world.get("counterfactual_links", []))}**.',
        "",
    ])
    if has_discourse(world):
        lines.extend([
            "Discourse objects (not established effects):",
            "",
        ])
        for row in discourse_nodes:
            lines.append(
                f'- `{row["id"]}` ({row["kind"]}): {_label(row.get("label"))}'
            )
        lines.append("")
    markdown_path = output_dir / "world_state_topology.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path, json_path


def _write_deliberation_report(trace: Path, result: dict) -> Path:
    """Expose native testimony and deliberation without rewriting its judgments."""
    path = trace.parent / "deliberation_report.md"
    lines = [
        "# Parliament deliberation record", "",
        f"Native trace: [{trace.name}]({trace.name})", "",
        f"Judgment status: **{result.get('judgment_status', 'unknown')}**  ",
        f"Selected action: {result.get('selected_action') or 'none'}  ",
        f"Current plurality: {result.get('current_plurality') or 'none'}  ",
        f"Stopping reason: {result.get('halted_by') or 'unknown'}  ",
        f"Recorded cycles: {len(result.get('cycles', []))}", "",
        "Plurality and selected action are reported separately from judgment status.", "",
    ]
    def section(title: str, data: object) -> None:
        lines.extend([f"## {title}", "", "```json",
                      json.dumps(data, ensure_ascii=False, indent=2), "```", ""])
    section("Original-agent testimony", result.get("source_testimonies", {}))
    section("Original-agent errors", result.get("source_errors", {}))
    section("Frozen-world replay", result.get("frozen_world_replay", {}))
    for cycle in result.get("cycles", []):
        section(f"Cycle {cycle.get('cycle', '?')}", cycle)
    fields = (
        "utilitarian_consequence_ledger", "deontological_duty_ledger",
        "virtue_character_ledger", "care_relationship_ledger", "rawlsian_position_ledger",
        "synthesis_proposals", "synthesis_viability_assessments", "planning_assessments",
        "visibility_assessments", "autonomy_assessments", "side_premise_audits",
        "shared_unresolved_dependencies", "termination_assessment", "trace_health",
    )
    for field in fields:
        if field in result:
            section(field.replace("_", " ").capitalize(), result[field])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _normalize_agents(values: Sequence[str]) -> list[str]:
    requested: list[str] = []
    for raw in values:
        value = " ".join(str(raw).strip().casefold().replace("_", " ").split())
        value = _AGENT_ALIASES.get(value, value)
        if value not in RELENT_FRAMEWORKS:
            raise ValueError(
                f"Unknown framework '{raw}'. Choose from: {', '.join(RELENT_FRAMEWORKS)}"
            )
        if value not in requested:
            requested.append(value)
    if len(requested) < 2:
        raise ValueError("Select at least two ethical frameworks for a workspace run")
    return [name for name in RELENT_FRAMEWORKS if name in requested]


def _prompt_agents() -> list[str]:
    answer = input(
        "Frameworks [all or comma-separated: utilitarian, deontological, "
        "virtue, care, rawlsian] (all): "
    ).strip()
    if not answer or answer.casefold() == "all":
        return list(RELENT_FRAMEWORKS)
    return _normalize_agents(answer.split(","))


def _prompt_max_cycles(default: int = 3) -> int:
    answer = input(f"Max cycles [{default}]: ").strip()
    if not answer:
        return default
    try:
        value = int(answer)
    except ValueError as exc:
        raise ValueError("Max cycles must be an integer") from exc
    if value < 1:
        raise ValueError("Max cycles must be at least 1")
    return value


def _prompt_use_rag(default: bool = False) -> bool:
    hint = "y" if default else "n"
    answer = input(
        "Corpus RAG for original agents? [y/N] "
        "(quotes into the initial consult only; compact scoring does not use them) "
        f"({hint}): "
    ).strip().lower()
    if not answer:
        return default
    if answer in {"y", "yes"}:
        return True
    if answer in {"n", "no"}:
        return False
    raise ValueError("Corpus RAG must be 'y' or 'n'")


def resolve_relent_deliberation(
    args: argparse.Namespace,
    *,
    interactive: bool | None = None,
) -> dict[str, object]:
    """Choose RelEnt options after the cloze world is admitted.

    Explicit CLI flags skip the matching RelEnt prompt. Prototype mode keeps the
    cheap compact stack and does not prompt.
    """
    if interactive is None:
        interactive = sys.stdin.isatty()
    if args.prototype_deliberation:
        agents = (
            _normalize_agents(args.agents)
            if args.agents else list(PROTOTYPE_FRAMEWORKS)
        )
        max_cycles = max(1, int(args.max_cycles)) if args.max_cycles is not None else 1
        use_rag = bool(args.use_rag) if args.use_rag is True else False
        if args.no_rag_context:
            use_rag = False
        time_budget = (
            float(args.time_budget) if args.time_budget is not None else 600.0
        )
        agent_timeout = (
            float(args.agent_timeout) if args.agent_timeout is not None else 300.0
        )
        return {
            "mode": "prototype",
            "agents": agents,
            "max_cycles": max_cycles,
            "use_rag": use_rag,
            "time_budget": time_budget,
            "agent_timeout": agent_timeout,
            "openai_model": args.openai_model,
        }
    if args.agents:
        agents = _normalize_agents(args.agents)
    elif interactive:
        agents = _prompt_agents()
    else:
        agents = list(RELENT_FRAMEWORKS)
    if args.max_cycles is not None:
        max_cycles = max(1, int(args.max_cycles))
    elif interactive:
        max_cycles = _prompt_max_cycles()
    else:
        max_cycles = 3
    if args.no_rag_context:
        use_rag = False
    elif args.use_rag is not None:
        use_rag = bool(args.use_rag)
    elif interactive:
        use_rag = _prompt_use_rag(default=False)
    else:
        use_rag = False
    time_budget = float(args.time_budget) if args.time_budget is not None else 600.0
    agent_timeout = (
        float(args.agent_timeout) if args.agent_timeout is not None else 600.0
    )
    return {
        "mode": "full",
        "agents": agents,
        "max_cycles": max_cycles,
        "use_rag": use_rag,
        "time_budget": time_budget,
        "agent_timeout": agent_timeout,
        "openai_model": args.openai_model,
    }


def _parliament_deliberation_command(
    *,
    parliament_python: Path,
    scenario_path: Path,
    trace_path: Path,
    output_dir: Path,
    openai_model: str,
    agents: Sequence[str],
    max_cycles: int,
    time_budget: float,
    agent_timeout: float,
    use_rag: bool,
    prototype: bool,
) -> list[str]:
    """Build RelEnt's workspace argv from the admitted frozen world only.

    Hypothesis records stay on the proposal overlay. RelEnt never receives
    the evidence graph.
    """
    command = [
        str(parliament_python), "global_workspace_pipeline.py",
        str(scenario_path),
        "--frozen-world-trace", str(trace_path),
        "--backend", "openai",
        "--openai-model", openai_model,
        "--agents", *list(agents),
        "--max-cycles", str(max(1, int(max_cycles))),
        "--time-budget", str(max(1.0, float(time_budget))),
        "--agent-timeout", str(max(1.0, float(agent_timeout))),
        "--openai-concurrency", "2",
        "--delegate-tokens", "128",
        "--output-dir", str(output_dir),
    ]
    if not use_rag:
        command.append("--no-rag-context")
    if prototype:
        command.extend([
            "--skip-original-agents",
            "--no-synthesis",
            "--no-planning",
            "--no-reformulation",
            "--no-visibility-audit",
            "--no-autonomy-audit",
            "--no-consensus-audit",
            "--no-cycle-extension",
        ])
    return command


def _load_environment(path: Path) -> dict[str, str]:
    """Load an env file for the child process without logging its values."""
    environment = dict(os.environ)
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        environment[key] = value
    return environment


def _parliament_revision(root: Path) -> dict:
    """Identify the actual checkout, including local integration changes."""
    record = {"root": str(root), "commit": None, "branch": None,
              "modified_files": [], "integration_patch_sha256": None}
    for field, arguments in (
        ("commit", ["rev-parse", "HEAD"]),
        ("branch", ["branch", "--show-current"]),
        ("modified_files", ["diff", "--name-only", "HEAD"]),
    ):
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            text=True, capture_output=True, timeout=15,
        )
        if result.returncode == 0:
            value = result.stdout.strip()
            record[field] = value.splitlines() if field == "modified_files" else value
    patch = PROJECT_ROOT / "integrations/parliament_z10/grounding_hook.patch"
    if patch.exists():
        record["integration_patch_sha256"] = hashlib.sha256(patch.read_bytes()).hexdigest()
    discourse_patch = PROJECT_ROOT / "integrations/parliament_z10/discourse_workspace.patch"
    if discourse_patch.exists():
        record["discourse_workspace_patch_sha256"] = hashlib.sha256(
            discourse_patch.read_bytes()).hexdigest()
    return record


def _parliament_admission_script() -> str:
    return r'''
import json
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
from global_workspace.action_identity import (
    build_canonical_action_records,
    extract_scenario_actor,
)
from global_workspace.frozen_world_replay import load_frozen_world_trace
from global_workspace.local_specialists import _admit_action_source_rows
from global_workspace.world_state import parse_world_model

OCCURRENCE_KEYS = (
    "parties", "actions", "effects", "conditions",
    "temporal_relations", "causal_links", "counterfactual_links",
)
DISCOURSE_KEYS = (
    "propositions", "reports", "commitments",
    "modal_operators", "normative_propositions",
)
NON_OCCURRENCE_STATUSES = {
    "ATTRIBUTED", "COMMITTED_CONTENT", "GOVERNED", "UNASSERTED",
}


def occurrence_subgraph(world):
    return {
        "schema_version": "1.3",
        **{key: list(world.get(key) or []) for key in OCCURRENCE_KEYS},
    }


def has_discourse(world):
    return any(world.get(key) for key in DISCOURSE_KEYS)


def merge_discourse(admitted, original):
    row = dict(admitted or {})
    row["schema_version"] = "1.4"
    for key in OCCURRENCE_KEYS + DISCOURSE_KEYS:
        row.setdefault(key, [])
    for key in DISCOURSE_KEYS:
        row[key] = list(original.get(key) or row.get(key) or [])
    return row


def relent_workspace_split(world):
    established = [{
        "kind": "occurrence_effect",
        "effect_id": effect["effect_id"],
        "outcome": effect.get("outcome"),
        "predicate": effect.get("predicate"),
        "status": "established",
    } for effect in world.get("effects") or []]
    not_established = []
    ev_forbidden = []
    for proposition in world.get("propositions") or []:
        if proposition.get("status") in NON_OCCURRENCE_STATUSES:
            not_established.append({
                "kind": "proposition",
                "proposition_id": proposition["proposition_id"],
                "predication": proposition.get("predication"),
                "status": proposition.get("status"),
            })
            ev_forbidden.append(proposition["proposition_id"])
    for report in world.get("reports") or []:
        not_established.append({
            "kind": "report", "report_id": report["report_id"],
            "speech_act": report.get("speech_act"),
            "content_proposition_id": report.get("content_proposition_id"),
            "status": "attributed",
        })
    for commitment in world.get("commitments") or []:
        not_established.append({
            "kind": "commitment",
            "commitment_id": commitment["commitment_id"],
            "content_proposition_id": commitment.get("content_proposition_id"),
            "status": "committed",
        })
    for modal in world.get("modal_operators") or []:
        not_established.append({
            "kind": "modal_operator", "modal_id": modal["modal_id"],
            "force": modal.get("force"),
            "governed_proposition_id": modal.get("governed_proposition_id"),
            "status": "modal",
        })
    for norm in world.get("normative_propositions") or []:
        not_established.append({
            "kind": "normative_proposition", "norm_id": norm["norm_id"],
            "force": norm.get("force"),
            "governed_proposition_id": norm.get("governed_proposition_id"),
            "status": "normative",
        })
    return {
        "established": established,
        "not_established": not_established,
        "duty_ledger_citations": [
            {"kind": "norm", "norm_id": row["norm_id"]}
            for row in world.get("normative_propositions") or []
        ] + [
            {"kind": "commitment", "commitment_id": row["commitment_id"]}
            for row in world.get("commitments") or []
        ],
        "ev_forbidden_proposition_ids": ev_forbidden,
        "factual_status_sources": ["occurrence_effects"],
    }

source_path = Path(sys.argv[2])
target_path = Path(sys.argv[3])
payload = json.loads(source_path.read_text(encoding="utf-8"))
proposal = payload["blueprint_result"]["proposals"][0]
if not proposal.get("admission_authorized") or proposal.get("candidate") is None:
    raise SystemExit("Blueprint proposal is withheld before Parliament admission")
original_world = proposal["candidate"]["world_model"]
occurrence = occurrence_subgraph(original_world)
actions = payload["actions"]
scenario = payload["scenario"]
action_ids = [row.get("action_id") or f"A{index}"
              for index, row in enumerate(occurrence.get("actions") or [])]
if not action_ids:
    action_ids = [f"A{index}" for index in range(len(actions))]
native_actions = [row["intervention"] for row in occurrence.get("actions") or []] or list(actions)
discourse_only = has_discourse(original_world) and (
    not occurrence.get("effects") or len(occurrence.get("actions") or []) < 2
)

if discourse_only:
    compile_world = dict(occurrence)
    compile_world["schema_version"] = "1.3"
    for key in DISCOURSE_KEYS:
        compile_world[key] = list(original_world.get(key) or [])
    grounding = {
        "status": "COMMITTED",
        "world_model_status": "COMMITTED",
        "world_model": compile_world,
        "world_model_1_4": original_world,
        "actions": {},
        "errors": [],
        "world_contradictions": [],
    }
    records = []
    replay_ok = False
    replay_fingerprint = None
else:
    proposal["candidate"]["world_model"] = occurrence
    parse_world_model(
        occurrence,
        clauses=proposal["clauses"],
        action_ids=action_ids,
        action_texts=dict(zip(action_ids, native_actions)),
        require_completeness=True,
    )
    grounding = _admit_action_source_rows(
        proposal["candidate"], native_actions, action_ids, proposal["clauses"]
    )
    admitted = {"COMMITTED", "COMMITTED_WITH_UNCERTAINTY"}
    if grounding.get("status") not in admitted or grounding.get("world_model_status") not in admitted:
        raise SystemExit("Parliament rejected the blueprint candidate: " + json.dumps(grounding))
    full_world = merge_discourse(grounding.get("world_model") or occurrence, original_world)
    grounding["world_model_1_4"] = full_world
    compile_world = dict(grounding.get("world_model") or occurrence)
    compile_world["schema_version"] = "1.3"
    for key in DISCOURSE_KEYS:
        compile_world[key] = list(full_world.get(key) or [])
    grounding["world_model"] = compile_world
    grounded_texts = {}
    for action_id, row in (grounding.get("actions") or {}).items():
        texts = [
            " ".join(str(clause.get("text") or "").split())
            for clause in row.get("clauses") or []
            if str(clause.get("text") or "").strip()
        ]
        if texts:
            grounded_texts[str(action_id)] = texts
    records = build_canonical_action_records(
        native_actions,
        actor=extract_scenario_actor(scenario),
        grounded_clause_texts_by_id=grounded_texts,
        scenario=scenario,
        grounding_status="COMMITTED",
        world_model=compile_world,
        action_origin="FROZEN_REPLAY",
    )
    replay_ok = True
    replay_fingerprint = None

workspace_world = grounding.get("world_model_1_4") or original_world
trace = {
    "scenario": scenario,
    "presentation_actions": native_actions,
    "actions": native_actions,
    "action_source_grounding": grounding,
    "canonical_action_records": [record.as_dict() for record in records],
    "presentation_action_mapping": [],
    "source_action_legend": {
        action_id: action for action_id, action in zip(action_ids, native_actions)
    },
    "relent_workspace": relent_workspace_split(workspace_world),
    "discourse_committed": discourse_only,
}
target_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

if replay_ok:
    replay = load_frozen_world_trace(target_path, expected_scenario=scenario)
    replay_fingerprint = replay.fingerprint
print(json.dumps({
    "status": grounding["status"],
    "world_model_status": grounding["world_model_status"],
    "effects": len((grounding.get("world_model") or {}).get("effects") or []),
    "causal_links": len((grounding.get("world_model") or {}).get("causal_links") or []),
    "frozen_trace_valid": bool(replay_ok),
    "discourse_committed": discourse_only,
    "relent_workspace": trace["relent_workspace"],
    "fingerprint_sha256": replay_fingerprint,
}))
'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Scenario text to interpret")
    source.add_argument("--scenario-file", type=Path,
                        help="UTF-8 text file containing the scenario")
    parser.add_argument("--parliament-root", type=Path,
                        default=DEFAULT_PARLIAMENT_ROOT)
    parser.add_argument("--parliament-python", type=Path,
                        default=DEFAULT_PARLIAMENT_PYTHON)
    parser.add_argument("--openai-env", type=Path, default=DEFAULT_OPENAI_ENV)
    parser.add_argument("--chooser-model", default="gpt-4o-mini",
                        help="OpenAI model used to rank and fill blueprints")
    parser.add_argument("--openai-model", default="o3")
    parser.add_argument(
        "--agents", nargs="+", default=None,
        help="RelEnt frameworks (default: all five, or the TTY prompt)",
    )
    parser.add_argument(
        "--max-cycles", type=int, default=None,
        help="RelEnt cycle budget (default: 3, or the TTY prompt; 1 in prototype)",
    )
    parser.add_argument(
        "--rag", dest="use_rag", action="store_true",
        help="Original agents retrieve CORE/ADJACENT corpus quotes",
    )
    parser.add_argument(
        "--no-rag", dest="use_rag", action="store_false",
        help="Original agents run without corpus retrieval",
    )
    parser.add_argument(
        "--no-rag-context", action="store_true",
        help="Alias for --no-rag",
    )
    parser.set_defaults(use_rag=None)
    parser.add_argument("--time-budget", type=float, default=None)
    parser.add_argument("--agent-timeout", type=float, default=None)
    parser.add_argument(
        "--prototype-deliberation", action="store_true",
        help=(
            "Cheap compact smoke: skip original agents, two frameworks, "
            "one cycle, no synthesis/planning/audits. Pipeline-only; "
            "not a RelEnt flag."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Stop after admission and topology generation")
    args = parser.parse_args(argv)

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = (
            PROJECT_ROOT / "diagnostics" / "blueprint_parliament_runs" / stamp
        )
    else:
        args.output_dir = args.output_dir.expanduser().resolve()
    args.parliament_root = args.parliament_root.expanduser().resolve()
    # Keep the virtual-environment launcher path intact. Resolving its symlink to
    # the base interpreter would silently discard that environment's packages.
    args.parliament_python = args.parliament_python.expanduser().absolute()
    args.openai_env = args.openai_env.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        scenario = _scenario_text(args.text, args.scenario_file)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    environment = (
        _load_environment(args.openai_env)
        if args.openai_env.exists() else dict(os.environ)
    )
    if not environment.get("OPENAI_API_KEY"):
        raise SystemExit(
            f"OPENAI_API_KEY is absent from {args.openai_env} and the shell"
        )

    package_id = "blueprint_run_" + hashlib.sha256(
        scenario.encode("utf-8")
    ).hexdigest()[:12]
    package = z10.export_candidate_graph(scenario, package_id=package_id)
    primitive_inventory = extract_inventory(scenario, package)
    (args.output_dir / "primitive_inventory.json").write_text(
        json.dumps(primitive_inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    question = assess_question(scenario)
    question_path = args.output_dir / "question_assessment.json"
    question_path.write_text(
        json.dumps(question, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    blueprint = choose_by_cloze(
        scenario,
        openai_complete(args.chooser_model, args.openai_env),
        question=question,
    )
    blueprint = expand_candidates(scenario, package, blueprint, inventory=primitive_inventory)
    blueprint = append_composition(scenario, primitive_inventory, blueprint)
    from blueprint_semantic_coverage import attach_coverage
    blueprint = attach_coverage(primitive_inventory, blueprint)
    (args.output_dir / "semantic_coverage.json").write_text(
        json.dumps(blueprint["semantic_coverage"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (args.output_dir / "primitive_comparison.json").write_text(
        json.dumps(blueprint["primitive_comparison"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (args.output_dir / "amendment_inventory.json").write_text(
        json.dumps(blueprint.get("amendment_inventory", {}), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    comparison = compare_z10_blueprints(package, blueprint)
    blueprint["z10_blueprint_comparison"] = comparison
    comparison_path = args.output_dir / "z10_blueprint_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    question_path.write_text(
        json.dumps(blueprint["question"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    candidate_graphs_markdown, candidate_attempts_json = _write_candidate_attempts(
        blueprint, args.output_dir
    )
    print("Candidate blueprint attempts:")
    for attempt in blueprint.get("candidate_attempts", []):
        proposal_state = (
            "graph" if (attempt["proposal"].get("candidate") or {}).get("world_model")
            else "withheld"
        )
        selected = " selected" if attempt.get("selected") else ""
        missing = ", ".join(attempt.get("unfilled_slots") or []) or "none"
        print(
            f'  {attempt["rank"] + 1}. {attempt["blueprint_id"]}:'
            f' {proposal_state}{selected}; missing={missing}'
        )
    print(f"Candidate graph display: {candidate_graphs_markdown}")

    candidate_path = args.output_dir / "blueprint_candidate.json"
    trace_path = args.output_dir / "frozen_world_trace.json"
    scenario_path = args.output_dir / "scenario.json"
    proposal = (blueprint.get("proposals") or [None])[0]
    actions = list(proposal.get("assignment") or []) if proposal else []
    candidate_path.write_text(json.dumps({
        "scenario": scenario,
        "actions": actions,
        "package": package,
        "blueprint_result": blueprint,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    blueprint_id = blueprint.get("chosen_blueprint_id")
    scenario_path.write_text(json.dumps({
        "scenario_id": _scenario_identifier(scenario, blueprint_id),
        "scenario_type": "global_workspace",
        "ethical_question": scenario,
        "tags": [blueprint_id] if blueprint_id else [],
        "tag_expectations": {},
        "tag_descriptions": {},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_path = args.output_dir / "preparation_manifest.json"
    manifest = {
        "scenario": scenario,
        "semantic_coverage": str(args.output_dir / "semantic_coverage.json"),
        "question_assessment": str(question_path),
        "chooser_model": args.chooser_model,
        "chosen_blueprint_id": blueprint_id,
        "blueprint_status": blueprint.get("status"),
        "candidate_graphs_markdown": str(candidate_graphs_markdown),
        "candidate_attempts_json": str(candidate_attempts_json),
        "z10_blueprint_comparison": str(comparison_path),
        "candidate_path": str(candidate_path),
        "scenario_path": str(scenario_path),
        "actions": actions,
        "deliberation": {"status": "NOT_RUN"},
    }
    try:
        proposal = _authorized_proposal(blueprint)
    except ValueError as exc:
        selected_attempt = next(
            (row for row in blueprint.get("candidate_attempts", [])
             if row.get("selected")),
            None,
        )
        manifest.update({
            "pipeline_status": "WITHHELD",
            "parliament_admission": {"status": "NOT_RUN"},
            "withholding_reason": str(exc),
            "missing_slots": (
                (proposal or {}).get("unfilled_required_slots", [])
                if proposal else (
                    selected_attempt.get("unfilled_slots", [])
                    if selected_attempt else []
                )
            ),
            "world_withheld": (
                (proposal or {}).get("world_withheld", []) if proposal else []
            ),
        })
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("Blueprint construction was withheld; no world was sent to Parliament.")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 2

    proposal_errors = validate_proposal(proposal)
    if proposal_errors:
        raise SystemExit("Selected proposal contract failed: " + "; ".join(proposal_errors))
    actions = list(proposal["assignment"])
    for path, description in (
        (args.parliament_root / "global_workspace_pipeline.py", "Parliament pipeline"),
        (args.parliament_python, "Parliament Python"),
    ):
        if not path.exists():
            raise SystemExit(f"{description} not found: {path}")

    manifest["parliament_version"] = _parliament_revision(args.parliament_root)

    chosen, admission_attempts = _admit_candidates(
        blueprint, scenario, package, args.output_dir, args.parliament_root,
        args.parliament_python,
    )
    manifest["admission_attempts"] = admission_attempts
    for attempt in blueprint.get("candidate_attempts", []):
        attempt["selected"] = bool(chosen and attempt is chosen["attempt"])
    if chosen:
        blueprint["chosen_blueprint_id"] = chosen["attempt"]["blueprint_id"]
    _write_candidate_attempts(blueprint, args.output_dir)
    if chosen is None:
        manifest.update({
            "pipeline_status": "ADMISSION_FAILED",
            "parliament_admission": {
                "status": "REJECTED", "error": "No filled candidate core passed native admission.",
            },
        })
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        print("Parliament admission failed; candidate graphs and run manifest were retained.")
        print(json.dumps(admission_attempts, indent=2))
        return 1
    proposal = chosen["proposal"]
    actions = list(proposal["assignment"])
    blueprint_id = chosen["attempt"]["blueprint_id"]
    manifest["chooser_preferred_blueprint_id"] = blueprint.get(
        "chooser_preferred_blueprint_id", manifest["chosen_blueprint_id"])
    manifest["chosen_blueprint_id"] = blueprint_id
    manifest["hypothesis_overlay"] = chosen["record"]["overlay_path"]
    manifest["selected_variant"] = chosen["record"].get("variant", "baseline")
    blueprint["proposals"] = [proposal]
    blueprint["graph"] = proposal
    candidate_path.write_text(json.dumps({
        "scenario": scenario, "actions": actions, "package": package,
        "blueprint_result": blueprint,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    scenario_record = json.loads(scenario_path.read_text(encoding="utf-8"))
    scenario_record.update(scenario_id=_scenario_identifier(scenario, blueprint_id), tags=[blueprint_id])
    scenario_path.write_text(json.dumps(scenario_record, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(chosen["record"]["trace_path"], trace_path)
    from parliament_source_constructions import build_packet
    source_advisory = build_packet(package, proposal)
    if source_advisory:
        frozen_payload = json.loads(trace_path.read_text(encoding="utf-8"))
        frozen_payload["source_construction_advisory"] = source_advisory
        trace_path.write_text(json.dumps(frozen_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    advisory_path = args.output_dir / "source_construction_advisory.json"
    advisory_path.write_text(json.dumps(source_advisory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["source_construction_advisory"] = {"path": str(advisory_path),
        "construction_count": len(source_advisory.get("constructions", [])),
        "authority": source_advisory.get("authority", "NONE")}
    admission = chosen["record"]["admission"]
    topology_markdown, topology_json = _write_world_topology(
        trace_path, args.output_dir
    )
    manifest.update({
        "pipeline_status": "ADMITTED",
        "actions": actions,
        "proposal_contract_valid": True,
        "selection_contract_valid": proposal["selection_validation"]["contract_valid"],
        "parliament_admission": admission,
        "frozen_world_trace": str(trace_path),
        "world_state_topology_markdown": str(topology_markdown),
        "world_state_topology_json": str(topology_json),
    })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print("Prepared and admitted blueprint world:")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.prepare_only:
        return 0

    try:
        deliberation = resolve_relent_deliberation(args)
    except ValueError as exc:
        parser.error(str(exc))
    parliament_output = args.output_dir / "parliament"
    command = _parliament_deliberation_command(
        parliament_python=args.parliament_python,
        scenario_path=scenario_path.resolve(),
        trace_path=trace_path.resolve(),
        output_dir=parliament_output.resolve(),
        openai_model=str(deliberation["openai_model"]),
        agents=list(deliberation["agents"]),
        max_cycles=int(deliberation["max_cycles"]),
        time_budget=float(deliberation["time_budget"]),
        agent_timeout=float(deliberation["agent_timeout"]),
        use_rag=bool(deliberation["use_rag"]),
        prototype=deliberation["mode"] == "prototype",
    )
    print(
        f"\nRunning {deliberation['mode']} RelEnt on the admitted graph "
        f"({', '.join(deliberation['agents'])}; "
        f"cycles={deliberation['max_cycles']}; "
        f"RAG={'on' if deliberation['use_rag'] else 'off'})...",
        flush=True,
    )
    manifest["deliberation"] = {
        "status": "RUNNING", **deliberation, "command": command,
        "output_dir": str(parliament_output),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    result = subprocess.run(command, cwd=args.parliament_root, env=environment)
    traces = sorted(parliament_output.glob("workspace_*.json"))
    trace = traces[-1] if traces else None
    if result.returncode or trace is None:
        manifest["pipeline_status"] = "DELIBERATION_FAILED"
        manifest["deliberation"] = {
            "status": "FAILED", "returncode": result.returncode,
            "output_dir": str(parliament_output),
            "mode": deliberation["mode"],
            "agents": list(deliberation["agents"]),
            "max_cycles": deliberation["max_cycles"],
            "use_rag": deliberation["use_rag"],
        }
    else:
        manifest["pipeline_status"] = "COMPLETED"
        result_data = json.loads(trace.read_text(encoding="utf-8"))
        report = _write_deliberation_report(trace, result_data)
        answer = trace.with_name(trace.stem + "_answer.txt")
        summary = trace.with_suffix(".txt")
        manifest["deliberation"] = {
            "status": "COMPLETED",
            "mode": deliberation["mode"],
            "agents": list(deliberation["agents"]),
            "max_cycles": deliberation["max_cycles"],
            "use_rag": deliberation["use_rag"],
            "command": command,
            "judgment_status": result_data.get("judgment_status"),
            "selected_action": result_data.get("selected_action"),
            "current_plurality": result_data.get("current_plurality"),
            "world_generation_calls": (
                result_data.get("frozen_world_replay") or {}
            ).get("world_generation_calls"),
            "trace": str(trace),
            "deliberation_report": str(report),
            "summary": str(summary),
            "answer": str(answer),
            "semantic_preservation_trace": str(
                parliament_output / "semantic_preservation_trace.json"
            ),
        }
        if summary.exists():
            print("\n" + summary.read_text(encoding="utf-8"))
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nRun manifest:")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return result.returncode or (1 if trace is None else 0)


if __name__ == "__main__":
    raise SystemExit(main())
