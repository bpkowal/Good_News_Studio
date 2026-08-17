from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from global_workspace.engine import WorkspaceConfig, WorkspaceEngine
from global_workspace.autonomy_audit import assess_autonomy_and_coercion
from global_workspace.evidence_calibration import calibrate_speculative_claim
from global_workspace.contingency_feasibility import verify_contingency_feasibility
from global_workspace.legacy_bridge import AGENT_MODULES, consult_original_agents
from global_workspace.landscape_validation import verify_landscape_alignment
from global_workspace.local_specialists import (
    CompactLocalSpecialist,
    FRAMEWORK_ROLES,
    analyze_action_plan,
    extract_labeled_action_legend,
    extract_scenario_facts,
    generate_failure_condition,
    infer_testimony_stance,
    propose_actions,
    propose_problem_reformulation,
    propose_synthesis,
)
from global_workspace.memory import EpisodicMemory, summarize_specialist_contributions
from global_workspace.models import WorkspaceBroadcast
from global_workspace.openai_backend import OpenAIWorkspaceLLM
from global_workspace.presentation import render_public_judgment
from global_workspace.scenario_semantics import (
    canonicalize_action_order,
    canonicalize_deliberation_scenario,
)
from global_workspace.structured_io import reset_model_call_budget, start_model_call_budget
from global_workspace.visibility import assess_visibility
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = (ROOT / "../mistral-7b-instruct-v0.2.Q4_K_M.gguf").resolve()


def render_summary(result) -> str:
    final = next(
        (cycle for cycle in reversed(result.cycles) if not cycle.is_hypothetical), None
    )
    if final is None:
        return "\n".join([
            "Ethical Global Workspace Judgment",
            f"Judgment status: {result.judgment_status}",
            "Decision: INCONCLUSIVE",
            f"Halting condition: {result.halted_by}",
            f"Compressed rule: {result.compressed_rule}",
        ]) + "\n"
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
    if result.termination_assessment is not None:
        termination = result.termination_assessment
        lines.append(
            f"Termination semantics: {termination.termination_type}; "
            f"resource-censored={str(termination.resource_censored).lower()}; "
            f"convergence evidence={termination.convergence_evidence:.2f}"
        )
    if result.further_deliberation_estimate is not None:
        estimate = result.further_deliberation_estimate
        lines.append(
            "Further-deliberation estimate (observational only): "
            f"action-change signal={estimate.action_change_signal:.2f}; "
            f"new-constraint signal={estimate.new_material_constraint_signal:.2f}"
        )
    if result.source_errors:
        lines.append(f"Unavailable original agents: {', '.join(result.source_errors)}")
    activated_ev = next(
        (item for item in reversed(result.ev_dominance_assessments) if item.get("activated")),
        None,
    )
    if activated_ev:
        lines.append(
            "EV dominance circuit breaker: "
            f"{activated_ev['ratio']:.2f}×; majority="
            f"{activated_ev['majority_count']}/{activated_ev['valid_delegate_count']}; "
            f"unit={activated_ev['unit']}; direction={activated_ev['direction']}"
        )
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
            responses = [
                candidate
                for cycle in result.cycles
                if (cycle.received_broadcast or cycle.broadcast).constraint == "VISIBILITY_AUDIT"
                for candidate in cycle.candidates
                if candidate.visibility_response != "NOT_TESTED"
            ]
            if responses:
                lines.append(
                    "Visibility deliberation: "
                    + "; ".join(
                        f"{candidate.specialist}={candidate.visibility_response}/"
                        f"harm {candidate.visibility_harm_revision.lower()}/"
                        f"magnitude {candidate.visibility_magnitude_status.lower()} "
                        f"({candidate.visibility_justification})"
                        for candidate in responses
                    )
                )
    for autonomy in result.autonomy_assessments:
        if autonomy.valid and autonomy.activated:
            tagged = ", ".join(
                f"{action}={tag}"
                for action, tag in autonomy.action_tags.items() if tag != "NONE"
            )
            lines.append(f"Non-voting autonomy audit: {tagged}")
            if autonomy.voluntary_alternative.casefold() != "none":
                lines.append(
                    "Voluntary-exhaustion probe: " + autonomy.voluntary_alternative
                )
    if dissent:
        lines.append(
            f"Preserved dissent: {dissent.specialist} raised {dissent.constraint}"
            + (f" ({dissent.rationale})" if dissent.rationale else "")
        )
    if result.moral_residue:
        lines.append(f"Moral residue: {', '.join(result.moral_residue)}")
    if result.moral_residue_records:
        lines.append(
            "Typed residue: "
            + "; ".join(
                f"{record.constraint} affecting {record.affected_action} "
                f"[{', '.join(record.source_specialists)}]"
                for record in result.moral_residue_records
            )
        )
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
                f"{candidate.specialist}: {candidate.speculative_claim} "
                f"[{candidate.evidence_calibration_tier}, "
                f"retention={candidate.evidence_direction_retention:.2f}]"
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
    for assessment in result.synthesis_viability_assessments:
        status = "viable for contingency analysis" if assessment.viable else "not branch-worthy"
        lines.append(
            f"Post-review synthesis viability: {assessment.synthesis_action} — {status}; "
            f"recommended={assessment.recommendation_count}/{assessment.valid_delegates}; "
            f"admissible={assessment.admissible_count}/{assessment.valid_delegates}; "
            f"mean score={assessment.mean_score:.2f}"
        )
    for condition in result.failure_conditions:
        if condition.valid:
            lines.append(f"Synthesis dependency: {condition.necessary_condition}")
            lines.append(f"Failure condition: {condition.failure_condition}")
            lines.append(
                "Fallback availability after failure: "
                + ", ".join(
                    f"{action_id}={status}"
                    for action_id, status in condition.fallback_availability.items()
                )
            )
            lines.append(f"Contingency question: {condition.contingency_question}")
    for assessment in result.contingency_feasibility_assessments:
        status = "approved" if assessment.approved else "blocked"
        lines.append(
            f"Independent contingency feasibility: {status}; "
            + ", ".join(
                f"{action_id}={value}"
                for action_id, value in assessment.fallback_statuses.items()
            )
            + (f"; {assessment.error}" if assessment.error else "")
        )
    contingency_reviews = [
        (cycle, candidate)
        for cycle in result.cycles
        if (cycle.received_broadcast or cycle.broadcast).constraint == "CONTINGENCY_REVIEW"
        for candidate in cycle.candidates
        if candidate.schema_valid and candidate.contingency_choice
    ]
    if contingency_reviews:
        lines.append(
            "Contingency fallback responses: "
            + "; ".join(
                f"{candidate.specialist} chose {candidate.contingency_choice} "
                f"({candidate.contingency_justification})"
                for _, candidate in contingency_reviews
            )
        )
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
    failed_invariants = [record for record in result.semantic_invariants if not record.valid]
    if failed_invariants:
        lines.append(
            "Semantic invariant holds: "
            + "; ".join(
                f"{record.boundary} ({' | '.join(record.errors)})"
                for record in failed_invariants
            )
        )
    if result.trace_health:
        lines.append(
            "Trace health: "
            + "; ".join(
                f"{finding.severity}/{finding.code}"
                + (f"@cycle{finding.cycle}" if finding.cycle else "")
                + f" ({finding.detail})"
                for finding in result.trace_health
            )
        )
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
    parser.add_argument("--call-reserve-seconds", type=float, default=20.0)
    parser.add_argument("--max-auxiliary-calls-per-cycle", type=int, default=5)
    parser.add_argument(
        "--drop-vote-on-graph-rejection", action="store_true",
        help="Exclude a delegate vote when its decision-critical graph update is rejected",
    )
    parser.add_argument("--no-ev-dominance-breaker", action="store_true")
    parser.add_argument("--ev-dominance-ratio", type=float, default=5.0)
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
    parser.add_argument("--no-autonomy-audit", action="store_true", help="Disable the non-voting autonomy and coercion audit")
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
    # Cover planning, baseline extraction, audits, and deliberation with one wall-clock
    # allowance. A single slow stage must not silently grant later stages a fresh budget.
    budget_token = start_model_call_budget(
        max(1.0, args.time_budget),
        reserve_seconds=max(0.0, args.call_reserve_seconds),
        max_auxiliary_calls_per_cycle=max(0, args.max_auxiliary_calls_per_cycle),
    )
    print("Planning a shared action set...", flush=True)
    source_action_legend = extract_labeled_action_legend(scenario)
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

    # Preserve the author's/user's order solely as presentation provenance.
    # Delegates receive a stable hash-ordered internal mapping so swapping the
    # displayed alternatives cannot by itself swap the meanings of A0 and A1.
    presentation_actions = list(actions)
    if not source_action_legend:
        source_action_legend = {
            f"A{index}": action for index, action in enumerate(presentation_actions)
        }
    presentation_action_legend = dict(source_action_legend)
    actions = canonicalize_action_order(presentation_actions)
    scenario = canonicalize_deliberation_scenario(
        scenario, presentation_action_legend, actions,
    )
    # Original agents now receive the canonical mapping, so any A0/A1 labels in
    # their testimony refer to this legend—not to the user's display order.
    source_action_legend = {
        f"A{index}": action for index, action in enumerate(actions)
    }
    print(
        "Canonical deliberation IDs: "
        + "; ".join(f"A{index}={action}" for index, action in enumerate(actions)),
        flush=True,
    )

    # The original RAG agents are part of the same experimental treatment. Give
    # them the canonical mapping too; otherwise order bias can enter through the
    # frozen testimony before workspace safeguards ever run.
    if args.skip_original_agents:
        testimonies: dict[str, str] = {}
        source_errors = {name: "skipped by request" for name in AGENT_MODULES}
    else:
        consultation = consult_original_agents(
            scenario_path,
            timeout_seconds=max(1.0, args.agent_timeout),
            backend=args.backend,
            openai_model=args.openai_model,
            canonical_actions=tuple(actions),
            canonical_scenario=scenario,
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

    baselines: dict[str, dict[str, object]] = {}
    for name in (list(testimonies) if testimonies else list(FRAMEWORK_ROLES)):
        if not testimonies.get(name):
            baselines[name] = {
                "status": "UNAVAILABLE",
                "action_id": "NONE",
                "provisional_action_id": "NONE",
                "condition": "",
                "reason": "no original testimony",
                "rejected_action_ids": [],
            }
            continue
        print(f"Freezing {name} testimony baseline...", flush=True)
        stance = infer_testimony_stance(
            llm,
            name,
            testimonies[name],
            actions,
            source_action_legend=source_action_legend,
        )
        baselines[name] = stance.as_dict()
        stance_target = stance.action_id if stance.status == "DIRECT" else stance.provisional_action_id
        print(
            f"  {name} baseline: {stance.status} {stance_target} "
            f"({stance.reason or stance.condition})",
            flush=True,
        )
    specialist_names = list(testimonies) if testimonies else list(FRAMEWORK_ROLES)
    specialists = [
        CompactLocalSpecialist(
            name,
            llm,
            testimony=testimonies.get(name, ""),
            baseline_action_id=str(baselines.get(name, {}).get("action_id", "NONE")),
            baseline_status=str(baselines.get(name, {}).get("status", "UNAVAILABLE")),
            baseline_provisional_action_id=str(
                baselines.get(name, {}).get("provisional_action_id", "NONE")
            ),
            baseline_condition=str(baselines.get(name, {}).get("condition", "")),
            baseline_framework_commitments=dict(
                baselines.get(name, {}).get("framework_commitments", {}) or {}
            ),
            baseline_numerical_role=str(
                baselines.get(name, {}).get("numerical_role", "UNASSESSED")
            ),
            source_action_legend=source_action_legend,
            scenario_facts=scenario_facts,
            max_tokens=max(48, args.delegate_tokens),
            memory_profile=specialist_profiles.get(name, {}),
            evidence_calibrator=calibrate_speculative_claim,
            landscape_verifier=verify_landscape_alignment,
            assumption_status=(
                str(baselines.get(name, {}).get("status"))
                if baselines.get(name, {}).get("status") in {
                    "CONDITIONAL", "UNDERDETERMINED", "NORMATIVELY_CONTESTED",
                }
                else "NOT_AUDITED"
            ),
            unsupported_assumption=str(
                baselines.get(name, {}).get("condition", "")
            ),
            reversal_condition=str(
                baselines.get(name, {}).get("condition", "")
            ),
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
            graph_rejection_policy=(
                "DROP_VOTE" if args.drop_vote_on_graph_rejection else "RETAIN_VOTE"
            ),
            enable_ev_dominance_breaker=not args.no_ev_dominance_breaker,
            ev_dominance_ratio=max(1.0, args.ev_dominance_ratio),
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_dir / f"workspace_{scenario_path.stem}_{stamp}.json"
    checkpoint_path = args.output_dir / f"checkpoint_{scenario_path.stem}_{stamp}.json"

    def save_checkpoint(current_result) -> None:
        checkpoint_path.write_text(
            json.dumps(current_result.to_dict(), indent=2), encoding="utf-8"
        )

    try:
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
        verify_contingency_feasibility=(
            lambda condition: verify_contingency_feasibility(
                llm,
                scenario,
                condition,
                max_tokens=min(96, max(72, args.delegate_tokens)),
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
        source_action_legend=source_action_legend,
        presentation_actions=presentation_actions,
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
        assess_autonomy=(
            lambda current_scenario, current_actions: assess_autonomy_and_coercion(
                llm,
                current_scenario,
                current_actions,
                max_tokens=max(180, args.delegate_tokens),
            )
        ) if not args.no_autonomy_audit else None,
        checkpoint=save_checkpoint,
        )
    finally:
        reset_model_call_budget(budget_token)
    result.source_testimonies = testimonies
    result.source_errors = source_errors
    result.source_action_legend = source_action_legend
    result.source_baselines = baselines
    result.scenario_facts = scenario_facts

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
