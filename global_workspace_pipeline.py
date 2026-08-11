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
    analyze_action_plan,
    assess_visibility,
    extract_scenario_facts,
    generate_failure_condition,
    infer_testimony_baseline,
    propose_actions,
    propose_problem_reformulation,
    propose_synthesis,
)
from global_workspace.memory import EpisodicMemory, summarize_specialist_contributions
from global_workspace.models import WorkspaceBroadcast
from global_workspace.openai_backend import OpenAIWorkspaceLLM
from global_workspace.presentation import render_public_judgment
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = (ROOT / "../mistral-7b-instruct-v0.2.Q4_K_M.gguf").resolve()


def render_summary(result) -> str:
    final = next(cycle for cycle in reversed(result.cycles) if not cycle.is_hypothetical)
    dissent = final.dissent
    valid_count = sum(candidate.schema_valid for candidate in final.candidates)
    lines = [
        "Ethical Global Workspace Judgment",
        f"Judgment status: {result.judgment_status}",
        f"Decision: {result.selected_action}",
        f"Current plurality: {result.current_plurality or 'none'}",
        f"Policy support: {result.confidence:.2f}",
        f"Epistemic confidence: {result.epistemic_confidence:.2f}",
        f"Halting condition: {result.halted_by}",
        f"Valid delegates: {valid_count}/{len(final.candidates)}",
        f"Dominant constraint: {(final.received_broadcast or final.broadcast).constraint}",
        f"Original agents consulted: {', '.join(result.source_testimonies) or 'none'}",
    ]
    if result.source_errors:
        lines.append(f"Unavailable original agents: {', '.join(result.source_errors)}")
    for visibility in result.visibility_assessments:
        if visibility.valid and visibility.activated:
            penalties = ", ".join(
                f"{action}×{value:.2f}"
                for action, value in visibility.action_multipliers.items()
                if value < 1.0
            )
            lines.append(
                f"Non-voting visibility audit: {visibility.mechanism}; "
                f"confidence adjustment={penalties}"
            )
    if dissent:
        lines.append(
            f"Preserved dissent: {dissent.specialist} raised {dissent.constraint}"
            + (f" ({dissent.rationale})" if dissent.rationale else "")
        )
    if result.moral_residue:
        lines.append(f"Moral residue: {', '.join(result.moral_residue)}")
    if result.reopen_conditions:
        lines.append("Explicit reversal conditions: " + "; ".join(result.reopen_conditions))
    speculative = [
        candidate
        for cycle in result.cycles
        if not cycle.is_hypothetical
        for candidate in cycle.candidates
        if candidate.evidence_basis == "UNSTATED_FACTS"
    ]
    if speculative:
        lines.append(
            "Damped speculative claims: "
            + "; ".join(
                f"{candidate.specialist}: {candidate.speculative_claim}"
                for candidate in speculative
            )
        )
    landscape_failures = [
        candidate
        for cycle in result.cycles
        if not cycle.is_hypothetical
        for candidate in cycle.candidates
        if candidate.landscape_search_attempted and not candidate.landscape_semantic_valid
    ]
    if landscape_failures:
        lines.append(
            "Landscape semantic penalties: "
            + "; ".join(
                f"{candidate.specialist}: {' | '.join(candidate.landscape_validation_errors)}"
                for candidate in landscape_failures
            )
        )
    independent = [
        candidate.specialist
        for cycle in result.cycles
        if not cycle.is_hypothetical
        for candidate in cycle.candidates
        if candidate.independence_bonus
    ]
    if independent:
        lines.append("Grounded non-consensus rewarded: " + ", ".join(independent))
    for proposal in result.synthesis_proposals:
        status = "admitted" if proposal.accepted else f"rejected ({proposal.rejection_reason})"
        lines.append(f"Synthesis candidate: {proposal.action or 'none'} — {status}")
    for condition in result.failure_conditions:
        if condition.valid:
            lines.append(f"Synthesis dependency: {condition.necessary_condition}")
            lines.append(f"Failure condition: {condition.failure_condition}")
            lines.append(f"Contingency question: {condition.contingency_question}")
    for assessment in result.planning_assessments:
        if assessment.valid:
            visibility = "broadcast" if assessment.broadcast_worthy else "private"
            lines.append(
                f"Planning assessment ({visibility}): {assessment.target_action} — "
                f"feasibility={assessment.feasibility:.2f}"
            )
            lines.append(f"Planning dependency: {assessment.necessary_condition}")
            lines.append(f"Planning failure: {assessment.failure_condition}")
            lines.append(f"Planning fallback: {assessment.fallback}")
    for branch in result.planning_branches:
        lines.append(
            f"Planning branch: if {branch.condition}, prefer {branch.selected_action} "
            f"({branch.confidence:.2f}); base leader={branch.origin_action}; "
            f"planned fallback={branch.fallback}"
        )
    for decision in result.access_decisions:
        if decision.admitted:
            lines.append(
                f"Workspace access: {decision.content_type} admitted "
                f"({', '.join(decision.signals)})"
            )
            lines.append(f"Audit question: {decision.question}")
    audited = [
        candidate
        for cycle in result.cycles
        if not cycle.is_hypothetical
        for candidate in cycle.candidates
        if candidate.assumption_status != "NOT_AUDITED"
    ]
    if audited:
        lines.append(
            "Consensus audit: "
            + "; ".join(
                f"{candidate.specialist}={candidate.assumption_status} "
                f"(assumption: {candidate.unsupported_assumption}; "
                f"reversal: {candidate.reversal_condition})"
                for candidate in audited
            )
        )
    reversal_reviews = [
        candidate
        for cycle in result.cycles
        if cycle.is_hypothetical
        and (cycle.received_broadcast or cycle.broadcast).constraint == "REVERSAL_AUDIT"
        for candidate in cycle.candidates
        if candidate.reversal_review_response != "NOT_TESTED"
    ]
    if reversal_reviews:
        lines.append(
            "Conditional reversal review: "
            + "; ".join(
                f"{candidate.specialist}={candidate.reversal_review_response} "
                f"({candidate.reversal_review_justification})"
                for candidate in reversal_reviews
            )
        )
    for reformulation in result.problem_reformulations:
        status = "admitted" if reformulation.accepted else f"rejected ({reformulation.rejection_reason})"
        lines.append(f"Problem reformulation: {status}")
        if reformulation.outcomes:
            for outcome in reformulation.outcomes:
                lines.append(
                    f"  Hypothetical stake — {outcome.dimension}/{outcome.action}: "
                    f"{outcome.probability:.0%} chance of {outcome.magnitude:g} "
                    f"{outcome.unit} {outcome.direction.lower()} over {outcome.horizon}"
                )
            lines.append(f"Switch condition: {reformulation.switch_condition}")
            lines.append(f"Residual tension: {reformulation.residual_tension}")
            lines.append(f"Calibrated question: {reformulation.question}")
        for comparison in reformulation.numeric_comparisons:
            lines.append(
                f"Computed dimension — {comparison.dimension}: "
                + "; ".join(
                    f"{action}={value:g} {comparison.unit}"
                    for action, value in comparison.action_values.items()
                )
                + f"; relative gap={comparison.relative_gap:.2f}"
            )
        if reformulation.fixed_facts:
            lines.append(f"Protected scenario facts: {'; '.join(reformulation.fixed_facts)}")
        for axis in reformulation.categorical_axes:
            lines.append(
                f"Categorical axis — {axis.name}: "
                + "; ".join(f"{action}={value}" for action, value in axis.action_values.items())
            )
        lines.append(
            f"Coalition probe: {reformulation.probe_result}; "
            f"switch verified={str(reformulation.switch_claim_valid).lower()}"
        )
        boundary_responses = [
            candidate
            for cycle in result.cycles
            for candidate in cycle.candidates
            if candidate.boundary_position != "NOT_TESTED"
        ]
        for candidate in boundary_responses:
            lines.append(
                f"Boundary response — {candidate.specialist}: "
                f"position={candidate.boundary_position}; axis={candidate.decisive_axis}; "
                f"switch={candidate.boundary_switch_condition}"
            )
    lines.append(f"Compressed rule: {result.compressed_rule}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the recurrent ethical global workspace.")
    parser.add_argument("scenario", type=Path, help="Scenario JSON containing ethical_question")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("local", "openai"), default="local")
    parser.add_argument("--openai-model", default="o3")
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
    parser.add_argument("--no-synthesis", action="store_true", help="Disable recurrent action synthesis")
    parser.add_argument("--no-planning", action="store_true", help="Disable selective implementation planning")
    parser.add_argument("--no-consensus-audit", action="store_true", help="Disable suspicious-consensus access gate")
    parser.add_argument("--no-reformulation", action="store_true", help="Disable hypothetical switch-point reformulation")
    parser.add_argument("--no-visibility-audit", action="store_true", help="Disable the non-voting epistemic-exclusion audit")
    parser.add_argument("--extension-cycles", type=int, default=2)
    parser.add_argument("--max-cycle-extensions", type=int, default=1)
    parser.add_argument("--no-cycle-extension", action="store_true")
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


def prompt_cycle_extension(result, extension_cycles: int = 2) -> int:
    if not sys.stdin.isatty():
        return 0
    final = result.cycles[-1]
    synthesis = next(
        (proposal.action for proposal in result.synthesis_proposals if proposal.accepted),
        "the synthesis candidate",
    )
    contingency = next(
        (condition for condition in reversed(result.failure_conditions) if condition.valid),
        None,
    )
    print(
        f"\nDeliberation remains unresolved (entropy={final.entropy:.2f}) while reviewing: "
        f"{synthesis}",
        flush=True,
    )
    if contingency:
        print(f"Necessary condition: {contingency.necessary_condition}", flush=True)
        print(f"Failure condition: {contingency.failure_condition}", flush=True)
        print(f"Next question: {contingency.contingency_question}", flush=True)
    answer = input(f"Extend deliberation by {extension_cycles} cycle(s)? [y/N]: ").strip().lower()
    return extension_cycles if answer in {"y", "yes"} else 0


def main() -> int:
    args = parse_args()
    load_dotenv()
    scenario_path = args.scenario.resolve()
    data = json.loads(scenario_path.read_text(encoding="utf-8"))
    scenario = str(data.get("ethical_question", "")).strip()
    if not scenario:
        raise ValueError("Scenario JSON must contain a non-empty ethical_question")
    if args.backend == "local" and not args.model.exists():
        raise FileNotFoundError(f"Local GGUF model not found: {args.model}")

    if args.skip_original_agents:
        testimonies: dict[str, str] = {}
        source_errors = {name: "skipped by request" for name in AGENT_MODULES}
    else:
        consultation = consult_original_agents(
            scenario_path,
            timeout_seconds=max(1.0, args.agent_timeout),
            backend=args.backend,
            openai_model=args.openai_model,
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

    if args.backend == "openai":
        print(f"Using OpenAI workspace model: {args.openai_model}", flush=True)
        llm = OpenAIWorkspaceLLM(args.openai_model, timeout=max(1.0, args.agent_timeout))
    else:
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

    episodic_memory = EpisodicMemory(args.output_dir / "episodic_memory.jsonl")
    specialist_profiles = episodic_memory.specialist_profiles()
    if specialist_profiles:
        print(
            "Loaded specialist contribution memory: "
            + ", ".join(sorted(specialist_profiles)),
            flush=True,
        )

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
            memory_profile=specialist_profiles.get(name, {}),
        )
        for name in specialist_names
    ]
    engine = WorkspaceEngine(
        specialists,
        WorkspaceConfig(
            max_cycles=max(1, args.max_cycles),
            time_budget_seconds=max(1, args.time_budget),
            enable_synthesis=not args.no_synthesis,
            enable_consensus_audit=not args.no_consensus_audit,
            max_cycle_extensions=max(0, args.max_cycle_extensions),
        ),
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
        synthesize=(
            lambda current_scenario, current_actions, candidates, current_broadcast: propose_synthesis(
                llm,
                current_scenario,
                current_actions,
                candidates,
                current_broadcast,
                testimonies,
                max_tokens=max(96, args.delegate_tokens),
            )
        ) if not args.no_synthesis and testimonies else None,
        request_extension=(
            lambda current_result: prompt_cycle_extension(
                current_result, max(1, args.extension_cycles)
            )
        ) if not args.no_cycle_extension else None,
        analyze_contingency=(
            lambda current_result: generate_failure_condition(
                llm,
                current_result.scenario,
                next(
                    proposal.action
                    for proposal in reversed(current_result.synthesis_proposals)
                    if proposal.accepted
                ),
                current_result.actions[:2],
                max_tokens=max(96, args.delegate_tokens),
            )
        ) if not args.no_cycle_extension else None,
        analyze_plan=(
            lambda current_scenario, current_actions, selected_action, current_broadcast,
            candidates, activation_reason: analyze_action_plan(
                llm,
                current_scenario,
                current_actions,
                selected_action,
                current_broadcast,
                candidates,
                activation_reason,
                max_tokens=max(128, args.delegate_tokens),
            )
        ) if not args.no_planning else None,
        scenario_facts=scenario_facts,
        source_testimonies=testimonies,
        reformulate_problem=(
            lambda current_scenario, current_actions, candidates: propose_problem_reformulation(
                llm,
                current_scenario,
                current_actions,
                candidates,
                max_tokens=max(320, args.delegate_tokens),
            )
        ) if not args.no_reformulation else None,
        assess_visibility=(
            lambda current_scenario, current_actions: assess_visibility(
                llm,
                current_scenario,
                current_actions,
                max_tokens=max(160, args.delegate_tokens),
            )
        ) if not args.no_visibility_audit else None,
    )
    result.source_testimonies = testimonies
    result.source_errors = source_errors
    result.source_baselines = baselines
    result.scenario_facts = scenario_facts

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_dir / f"workspace_{scenario_path.stem}_{stamp}.json"
    summary_path = output_path.with_suffix(".txt")
    answer_path = output_path.with_name(output_path.stem + "_answer.txt")
    result_data = result.to_dict()
    output_path.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
    summary = render_summary(result)
    summary_path.write_text(summary, encoding="utf-8")
    answer_path.write_text(render_public_judgment(result), encoding="utf-8")
    episodic_memory.append({
        "scenario_id": scenario_path.stem,
        "judgment_status": result.judgment_status,
        "selected_action": result.selected_action,
        "current_plurality": result.current_plurality,
        "confidence": result.confidence,
        "epistemic_confidence": result.epistemic_confidence,
        "trajectory": [cycle.broadcast.constraint for cycle in result.cycles],
        "compressed_rule": result.compressed_rule,
        "reopen_conditions": result.reopen_conditions,
        "specialist_contributions": summarize_specialist_contributions(result.cycles),
        "hypothetical_reformulations": [
            {
                "switch_condition": proposal.switch_condition,
                "residual_tension": proposal.residual_tension,
                "question": proposal.question,
            }
            for proposal in result.problem_reformulations
            if proposal.accepted
        ],
    })

    print("\n" + summary)
    print(f"Saved trace: {output_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved final answer: {answer_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
