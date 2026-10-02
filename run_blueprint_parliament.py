"""Run the first Z10 blueprint through real Parliament deliberation.

This command runs the source text through Z10, fills the exclusive-allocation
blueprint, asks Parliament to admit the resulting candidate, and writes a native
frozen-world trace and Mermaid topology. By default it then runs Parliament's
ordinary specialist and voting pipeline with a small reproducible profile.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import html
import json
import os
from pathlib import Path
import subprocess
import tempfile

import parsing_game_Z10 as z10

from candidate_graph_blueprints import instantiate_exclusive_allocation


DEFAULT_SCENARIO = (
    "A clinic has one dose of medicine. "
    "Ada can give the medicine to Ben or Cara, but not both. "
    "If Ada gives the medicine to Ben, Ben has a 95% chance of survival. "
    "If Ada gives the medicine to Cara, Cara has a 5% chance of survival. "
    "The patient who does not get the medicine will die."
)
DEFAULT_ACTIONS = ["give the medicine to Ben", "give the medicine to Cara"]
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PARLIAMENT_ROOT = Path("/tmp/parliament-smoke-614af0c")
DEFAULT_PARLIAMENT_PYTHON = Path("/tmp/parliament-smoke-env/bin/python")
DEFAULT_OPENAI_ENV = Path(
    "/Users/benjaminkowal/Documents/Python/Python Coding/RAGAIMODEL/.env"
)


def _label(value: object) -> str:
    return html.escape(" ".join(str(value or "").split()), quote=True)


def _write_world_topology(trace_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Write machine-readable and Mermaid views of the admitted world."""
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    grounding = trace["action_source_grounding"]
    world = grounding["world_model"]
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

    topology = {
        "schema_version": world.get("schema_version"),
        "admission": world.get("admission"),
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
    lines.extend([
        "```", "",
        f'Admission: **{_label(world.get("admission", {}).get("status"))}**  ',
        f'Admitted effects: `{", ".join(world.get("admission", {}).get("admitted_effect_ids", []))}`  ',
        f'Conditions: **{len(world.get("conditions", []))}**; '
        f'Temporal links: **{len(world.get("temporal_relations", []))}**; '
        f'Counterfactual links: **{len(world.get("counterfactual_links", []))}**.',
        "",
    ])
    markdown_path = output_dir / "world_state_topology.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path, json_path


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

source_path = Path(sys.argv[2])
target_path = Path(sys.argv[3])
payload = json.loads(source_path.read_text(encoding="utf-8"))
proposal = payload["blueprint_result"]["proposals"][0]
actions = payload["actions"]
scenario = payload["scenario"]
action_ids = [f"A{index}" for index in range(len(actions))]

# Exercise Parliament's strict parser before admission so this is not merely a
# serialization adapter.
parse_world_model(
    proposal["candidate"]["world_model"],
    clauses=proposal["clauses"],
    action_ids=action_ids,
    action_texts=dict(zip(action_ids, actions)),
    require_completeness=True,
)
grounding = _admit_action_source_rows(
    proposal["candidate"], actions, action_ids, proposal["clauses"]
)
if grounding.get("status") != "COMMITTED" or grounding.get("world_model_status") != "COMMITTED":
    raise SystemExit("Parliament rejected the blueprint candidate: " + json.dumps(grounding))

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
    actions,
    actor=extract_scenario_actor(scenario),
    grounded_clause_texts_by_id=grounded_texts,
    scenario=scenario,
    grounding_status="COMMITTED",
    world_model=grounding["world_model"],
    action_origin="FROZEN_REPLAY",
)
trace = {
    "scenario": scenario,
    "presentation_actions": actions,
    "actions": actions,
    "action_source_grounding": grounding,
    "canonical_action_records": [record.as_dict() for record in records],
    "presentation_action_mapping": [],
    "source_action_legend": {
        action_id: action for action_id, action in zip(action_ids, actions)
    },
}
target_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# Prove the artifact satisfies Parliament's native replay contract now rather
# than allowing the later deliberation command to discover a malformed trace.
replay = load_frozen_world_trace(target_path, expected_scenario=scenario)
print(json.dumps({
    "status": grounding["status"],
    "world_model_status": grounding["world_model_status"],
    "effects": len(grounding["world_model"]["effects"]),
    "causal_links": len(grounding["world_model"]["causal_links"]),
    "frozen_trace_valid": True,
    "fingerprint_sha256": replay.fingerprint,
}))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parliament-root", type=Path,
                        default=DEFAULT_PARLIAMENT_ROOT)
    parser.add_argument("--parliament-python", type=Path,
                        default=DEFAULT_PARLIAMENT_PYTHON)
    parser.add_argument("--openai-env", type=Path, default=DEFAULT_OPENAI_ENV)
    parser.add_argument("--openai-model", default="o3")
    parser.add_argument("--agents", nargs="+",
                        default=["utilitarian", "deontological"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Stop after admission and topology generation")
    args = parser.parse_args()

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

    for path, description in (
        (args.parliament_root / "global_workspace_pipeline.py", "Parliament pipeline"),
        (args.parliament_python, "Parliament Python"),
    ):
        if not path.exists():
            raise SystemExit(f"{description} not found: {path}")

    scenario = DEFAULT_SCENARIO
    actions = list(DEFAULT_ACTIONS)
    package = z10.export_candidate_graph(
        scenario, package_id="exclusive_allocation_medicine_parliament"
    )
    blueprint = instantiate_exclusive_allocation(package, actions)
    if blueprint.get("status") != "FILLED" or not blueprint.get("proposals"):
        raise SystemExit("Blueprint did not produce a filled proposal")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "blueprint_candidate.json"
    trace_path = args.output_dir / "frozen_world_trace.json"
    scenario_path = args.output_dir / "scenario.json"
    candidate_path.write_text(json.dumps({
        "scenario": scenario,
        "actions": actions,
        "package": package,
        "blueprint_result": blueprint,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    scenario_path.write_text(json.dumps({
        "scenario_id": "exclusive_allocation_medicine_chance_blueprint",
        "scenario_type": "global_workspace",
        "ethical_question": scenario,
        "tags": ["allocation", "scarcity"],
        "tag_expectations": {},
        "tag_descriptions": {},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with tempfile.TemporaryDirectory() as directory:
        helper = Path(directory) / "admit_blueprint.py"
        helper.write_text(_parliament_admission_script(), encoding="utf-8")
        completed = subprocess.run([
            str(args.parliament_python), str(helper),
            str(args.parliament_root.resolve()), str(candidate_path.resolve()),
            str(trace_path.resolve()),
        ], text=True, capture_output=True, timeout=120)
    if completed.returncode:
        raise SystemExit(completed.stdout + completed.stderr)
    admission = json.loads(completed.stdout)
    topology_markdown, topology_json = _write_world_topology(
        trace_path, args.output_dir
    )
    manifest = {
        "scenario": scenario,
        "actions": actions,
        "blueprint_status": blueprint["status"],
        "selection_contract_valid": blueprint["proposals"][0]["selection_validation"]["contract_valid"],
        "parliament_admission": admission,
        "candidate_path": str(candidate_path),
        "frozen_world_trace": str(trace_path),
        "scenario_path": str(scenario_path),
        "world_state_topology_markdown": str(topology_markdown),
        "world_state_topology_json": str(topology_json),
        "deliberation": {"status": "NOT_RUN"},
    }
    manifest_path = args.output_dir / "preparation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print("Prepared and admitted blueprint world:")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.prepare_only:
        return 0

    if not args.openai_env.exists() and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            f"OpenAI environment file not found: {args.openai_env}. "
            "Pass --openai-env or export OPENAI_API_KEY."
        )
    environment = (
        _load_environment(args.openai_env)
        if args.openai_env.exists() else dict(os.environ)
    )
    if not environment.get("OPENAI_API_KEY"):
        raise SystemExit(
            f"OPENAI_API_KEY is absent from {args.openai_env} and the shell"
        )
    parliament_output = args.output_dir / "parliament"
    command = [
        str(args.parliament_python), "global_workspace_pipeline.py",
        str(scenario_path.resolve()),
        "--frozen-world-trace", str(trace_path.resolve()),
        "--backend", "openai",
        "--openai-model", args.openai_model,
        "--agents", *args.agents,
        "--skip-original-agents",
        "--no-rag-context",
        "--max-cycles", "1",
        "--time-budget", "600",
        "--agent-timeout", "300",
        "--openai-concurrency", "2",
        "--delegate-tokens", "128",
        "--no-synthesis",
        "--no-planning",
        "--no-reformulation",
        "--no-visibility-audit",
        "--no-autonomy-audit",
        "--no-consensus-audit",
        "--no-cycle-extension",
        "--output-dir", str(parliament_output.resolve()),
    ]
    print("\nRunning Parliament with the admitted graph...", flush=True)
    result = subprocess.run(command, cwd=args.parliament_root, env=environment)
    traces = sorted(parliament_output.glob("workspace_*.json"))
    trace = traces[-1] if traces else None
    if result.returncode or trace is None:
        manifest["deliberation"] = {
            "status": "FAILED", "returncode": result.returncode,
            "output_dir": str(parliament_output),
        }
    else:
        result_data = json.loads(trace.read_text(encoding="utf-8"))
        answer = trace.with_name(trace.stem + "_answer.txt")
        summary = trace.with_suffix(".txt")
        manifest["deliberation"] = {
            "status": "COMPLETED",
            "judgment_status": result_data.get("judgment_status"),
            "selected_action": result_data.get("selected_action"),
            "current_plurality": result_data.get("current_plurality"),
            "world_generation_calls": (
                result_data.get("frozen_world_replay") or {}
            ).get("world_generation_calls"),
            "trace": str(trace),
            "summary": str(summary),
            "answer": str(answer),
            "semantic_preservation_trace": str(
                parliament_output / "semantic_preservation_trace.json"
            ),
        }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nRun manifest:")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
