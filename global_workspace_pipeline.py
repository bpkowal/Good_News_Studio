from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from global_workspace.engine import WorkspaceConfig, WorkspaceEngine
from global_workspace.legacy_bridge import AGENT_MODULES, consult_original_agents
from global_workspace.local_specialists import (
    CompactLocalSpecialist,
    FRAMEWORK_ROLES,
    extract_scenario_facts,
    infer_testimony_baseline,
    propose_actions,
)
from global_workspace.memory import EpisodicMemory
from global_workspace.models import WorkspaceBroadcast


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = (ROOT / "../mistral-7b-instruct-v0.2.Q4_K_M.gguf").resolve()


def render_summary(result) -> str:
    final = result.cycles[-1]
    dissent = final.dissent
    valid_count = sum(candidate.schema_valid for candidate in final.candidates)
    lines = [
        "Ethical Global Workspace Judgment",
        f"Decision: {result.selected_action}",
        f"Policy support: {result.confidence:.2f}",
        f"Halting condition: {result.halted_by}",
        f"Valid delegates: {valid_count}/{len(final.candidates)}",
        f"Dominant constraint: {final.broadcast.constraint}",
        f"Original agents consulted: {', '.join(result.source_testimonies) or 'none'}",
    ]
    if result.source_errors:
        lines.append(f"Unavailable original agents: {', '.join(result.source_errors)}")
    if dissent:
        lines.append(
            f"Preserved dissent: {dissent.specialist} raised {dissent.constraint}"
            + (f" ({dissent.rationale})" if dissent.rationale else "")
        )
    if result.moral_residue:
        lines.append(f"Moral residue: {', '.join(result.moral_residue)}")
    lines.append(f"Compressed rule: {result.compressed_rule}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the recurrent ethical global workspace.")
    parser.add_argument("scenario", type=Path, help="Scenario JSON containing ethical_question")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--actions", nargs="+", help="Skip local action planning and use these actions")
    parser.add_argument("--urgency", type=float, default=0.5)
    parser.add_argument("--danger", type=float, default=0.5)
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--time-budget", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "workspace_outputs")
    parser.add_argument(
        "--skip-original-agents",
        action="store_true",
        help="Use ungrounded compact specialists (diagnostic/prototype mode only)",
    )
    parser.add_argument("--agent-timeout", type=float, default=600.0)
    parser.add_argument("--n-ctx", type=int, default=768)
    parser.add_argument("--n-gpu-layers", type=int, default=8)
    parser.add_argument("--n-batch", type=int, default=32)
    parser.add_argument("--delegate-tokens", type=int, default=128)
    parser.add_argument("--accept-actions", action="store_true", help="Skip interactive action confirmation")
    return parser.parse_args()


def confirm_actions(actions: list[str]) -> list[str] | None:
    if not sys.stdin.isatty():
        print("Non-interactive input: accepting proposed actions.", flush=True)
        return actions
    answer = input("Use these actions? [Y/n/edit]: ").strip().lower()
    if answer in {"", "y", "yes"}:
        return actions
    if answer in {"n", "no"}:
        return None
    if answer in {"e", "edit"}:
        edited = input("Enter 2-5 actions separated by |: ").split("|")
        edited = [" ".join(action.split()) for action in edited if action.strip()]
        edited = list(dict.fromkeys(edited))[:5]
        if len(edited) < 2:
            print("At least two distinct actions are required.", flush=True)
            return None
        return edited
    print("Unrecognized response; actions were not accepted.", flush=True)
    return None


def main() -> int:
    args = parse_args()
    scenario_path = args.scenario.resolve()
    data = json.loads(scenario_path.read_text(encoding="utf-8"))
    scenario = str(data.get("ethical_question", "")).strip()
    if not scenario:
        raise ValueError("Scenario JSON must contain a non-empty ethical_question")
    if not args.model.exists():
        raise FileNotFoundError(f"Local GGUF model not found: {args.model}")

    if args.skip_original_agents:
        testimonies: dict[str, str] = {}
        source_errors = {name: "skipped by request" for name in AGENT_MODULES}
    else:
        consultation = consult_original_agents(
            scenario_path,
            timeout_seconds=max(1.0, args.agent_timeout),
        )
        testimonies = consultation.testimonies
        source_errors = consultation.errors
        for name, error in source_errors.items():
            print(f"Original {name} agent unavailable: {error}")
        if len(testimonies) < 2:
            raise RuntimeError(
                "Fewer than two original ethical agents produced testimony; "
                "cannot run meaningful recurrent orchestration."
            )

    from llama_cpp import Llama

    print(
        f"Loading shared local model: {args.model} "
        f"(ctx={args.n_ctx}, gpu_layers={args.n_gpu_layers}, batch={args.n_batch})",
        flush=True,
    )
    llm = Llama(
        model_path=str(args.model),
        n_ctx=max(512, args.n_ctx),
        n_threads=6,
        n_gpu_layers=max(0, args.n_gpu_layers),
        n_batch=max(8, args.n_batch),
        verbose=False,
    )
    print("Planning a shared action set...", flush=True)
    try:
        actions = args.actions or propose_actions(llm, scenario)
    except ValueError as exc:
        print(f"Action planning could not produce a safe feasible set: {exc}", flush=True)
        print("Rerun with explicit choices, for example: --actions \"first action\" \"second action\"", flush=True)
        return 2
    print("Actions:", ", ".join(actions), flush=True)
    if not args.accept_actions:
        confirmed_actions = confirm_actions(actions)
        if confirmed_actions is None:
            print("Action set rejected; no deliberation was run.", flush=True)
            return 2
        actions = confirmed_actions
        print("Confirmed actions:", ", ".join(actions), flush=True)

    scenario_facts = extract_scenario_facts(scenario)
    if scenario_facts:
        print("Scenario facts:", json.dumps(scenario_facts, sort_keys=True), flush=True)

    baselines: dict[str, dict[str, str]] = {}
    for name in (list(testimonies) if testimonies else list(FRAMEWORK_ROLES)):
        if not testimonies.get(name):
            baselines[name] = {"action_id": "NONE", "reason": "no original testimony"}
            continue
        print(f"Freezing {name} testimony baseline...", flush=True)
        baseline_id, baseline_reason = infer_testimony_baseline(
            llm, name, testimonies[name], actions
        )
        baselines[name] = {"action_id": baseline_id, "reason": baseline_reason}
        print(f"  {name} baseline: {baseline_id} ({baseline_reason})", flush=True)
    specialist_names = list(testimonies) if testimonies else list(FRAMEWORK_ROLES)
    specialists = [
        CompactLocalSpecialist(
            name,
            llm,
            testimony=testimonies.get(name, ""),
            baseline_action_id=baselines.get(name, {}).get("action_id", "NONE"),
            scenario_facts=scenario_facts,
            max_tokens=max(48, args.delegate_tokens),
        )
        for name in specialist_names
    ]
    engine = WorkspaceEngine(
        specialists,
        WorkspaceConfig(max_cycles=max(1, args.max_cycles), time_budget_seconds=max(1, args.time_budget)),
    )
    result = engine.run(
        scenario,
        actions,
        WorkspaceBroadcast(
            urgency=args.urgency,
            danger_probability=args.danger,
            unresolved="ASSESS_FACTS",
        ),
        progress=lambda message: print(message, flush=True),
    )
    result.source_testimonies = testimonies
    result.source_errors = source_errors
    result.source_baselines = baselines
    result.scenario_facts = scenario_facts

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_dir / f"workspace_{scenario_path.stem}_{stamp}.json"
    summary_path = output_path.with_suffix(".txt")
    result_data = result.to_dict()
    output_path.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
    summary = render_summary(result)
    summary_path.write_text(summary, encoding="utf-8")
    EpisodicMemory(args.output_dir / "episodic_memory.jsonl").append({
        "scenario_id": scenario_path.stem,
        "selected_action": result.selected_action,
        "confidence": result.confidence,
        "trajectory": [cycle.broadcast.constraint for cycle in result.cycles],
        "compressed_rule": result.compressed_rule,
        "reopen_conditions": result.reopen_conditions,
    })

    print("\n" + summary)
    print(f"Saved trace: {output_path}")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
