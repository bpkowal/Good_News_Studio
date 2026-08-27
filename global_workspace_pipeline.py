from __future__ import annotations

import argparse
import json
import re
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
    ground_actions_in_scenario,
    infer_testimony_stance,
    propose_actions,
    propose_problem_reformulation,
    propose_synthesis,
    _validate_lossless_action_set,
)
from global_workspace.memory import EpisodicMemory, summarize_specialist_contributions
from global_workspace.models import WorkspaceBroadcast
from global_workspace.openai_backend import OpenAIWorkspaceLLM
from global_workspace.presentation import (
    render_public_judgment,
    summarize_problem_shape_paragraphs,
)
from global_workspace.scenario_semantics import (
    canonicalize_action_order,
    canonicalize_deliberation_scenario,
)
from global_workspace.structured_io import reset_model_call_budget, start_model_call_budget
from global_workspace.visibility import assess_visibility
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = (ROOT / "../mistral-7b-instruct-v0.2.Q4_K_M.gguf").resolve()


def _baseline_display_action(baseline: dict[str, object]) -> str:
    """Render committed actions for direct stances and provisional actions otherwise."""
    status = str(baseline.get("status", "UNAVAILABLE")).strip().upper()
    field = "action_id" if status == "DIRECT" else "provisional_action_id"
    return str(baseline.get(field, "NONE")).strip().upper() or "NONE"


def render_summary(result) -> str:
    final = next(
        (cycle for cycle in reversed(result.cycles) if not cycle.is_hypothetical), None
    )
    lines = ["# Ethical Parliament Judgment"]

    def add_section(title: str, paragraphs: list[str]) -> None:
        paragraphs = [entry for entry in paragraphs if entry]
        if not paragraphs:
            return
        lines.extend(["", f"## {title}"])
        lines.extend(paragraphs)

    def add_diagnostic_section(title: str, entries: list[str]) -> None:
        entries = [entry for entry in entries if entry]
        if not entries:
            return
        lines.extend(["", f"## {title}"])
        lines.extend(f"- {entry}" for entry in entries)

    def compact_cycles(cycles: list[int]) -> str:
        return ", ".join(dict.fromkeys(str(cycle) for cycle in cycles))

    def sentence(text: str) -> str:
        cleaned = " ".join(str(text).split()).strip()
        if not cleaned:
            return ""
        cleaned = cleaned[0].upper() + cleaned[1:]
        if cleaned[-1] not in ".?!":
            cleaned += "."
        return cleaned

    def format_quantity(number: str, unit: str = "") -> str:
        try:
            value = float(number)
        except ValueError:
            return number
        if unit.casefold() == "fraction" and 0.0 <= value <= 1.0:
            return f"{value * 100:.0f}%"
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:g}"

    def humanize_threshold(text: str) -> str:
        cleaned = " ".join(str(text).replace("_", " ").split()).strip()
        if not cleaned:
            return cleaned

        pattern = re.compile(
            r"(?P<lhs>[A-Za-z][A-Za-z ]*?)\s*(?P<op><=|>=|<|>)\s*"
            r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z%]+)?"
        )

        def replace(match: re.Match[str]) -> str:
            lhs = " ".join(match.group("lhs").split()).strip()
            op = match.group("op")
            num = format_quantity(match.group("num"), match.group("unit") or "")
            if op == "<":
                return f"{lhs} falls below {num}"
            if op == ">":
                return f"{lhs} exceeds {num}"
            if op == "<=":
                return f"{lhs} is at or below {num}"
            return f"{lhs} is at or above {num}"

        cleaned = pattern.sub(replace, cleaned)
        cleaned = cleaned.replace("count", "").replace("fraction", "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    constraint_labels = {
        "RIGHTS": "a rights-based objection",
        "DUTY": "a duty-based objection",
        "CARE": "a relational-care objection",
        "FAIRNESS": "a fairness objection",
        "CHARACTER": "a character and practical-wisdom objection",
        "UNCERTAINTY": "an unresolved consequence comparison",
        "AUTONOMY": "an autonomy objection",
    }

    def render_residue(constraint: str) -> str:
        code = str(constraint).strip().upper()
        label = constraint_labels.get(
            code, code.lower().replace("_", " ") + " consideration",
        )
        matching = [
            candidate for cycle in reversed(result.cycles)
            if not cycle.is_hypothetical
            for candidate in cycle.candidates
            if candidate.schema_valid
            and candidate.constraint == code
            and result.current_plurality
            and candidate.action_scores.get(result.current_plurality, 0.5) < 0.5
        ]
        detail = next((
            candidate.rationale for candidate in matching
            if candidate.rationale and candidate.rationale.casefold() not in {
                "none", "delegate output failed semantic validation."
            }
        ), "")
        return (
            f"{label.capitalize()} remains active"
            + (f": {detail}" if detail else "")
        )

    def render_reopen_condition(condition: str) -> str:
        code = str(condition).strip().upper()
        problem_state = (
            final.broadcast.problem_state if final is not None else {}
        ) or {}
        question = next((
            str(item.get("question", "")).strip()
            for item in problem_state.get("unresolved_questions", [])
            if str(item.get("category", "")).upper() == code
            and str(item.get("question", "")).strip()
        ), "")
        if question:
            return question[0].lower() + question[1:] if question.startswith("Whether ") else question
        labels = {
            "RESOLVE_NORMATIVE_TENSION": "the unresolved normative conflict is clarified",
            "VERIFY_FACTS": "the decision-critical factual uncertainty is resolved",
            "CHECK_FEASIBILITY": "the proposed action's feasibility is established",
            "ACTION_SET_ADEQUACY": "the adequacy of the available action set is reviewed",
            "FRAMEWORK_GROUNDING_UNCERTAINTY": "the framework-grounding uncertainty is resolved",
        }
        return labels.get(code, humanize_threshold(condition))

    def summarize_trace_health() -> list[str]:
        grouped: dict[tuple[str, str], dict[str, object]] = {}
        for finding in result.trace_health or []:
            bucket = grouped.setdefault(
                (finding.severity, finding.code),
                {"detail": finding.detail, "cycles": []},
            )
            bucket["cycles"].append(finding.cycle)
        entries: list[str] = []
        for (severity, code), bucket in sorted(grouped.items()):
            entry = f"{severity}/{code}"
            cycles = [cycle for cycle in bucket["cycles"] if cycle]
            if cycles:
                entry += f" [cycles {compact_cycles(cycles)}]"
            detail = str(bucket["detail"])
            if detail:
                entry += f" — {detail}"
            entries.append(entry)
        return entries

    if final is None:
        add_section(
            "Judgment",
            [
                "No reliable deliberative judgment was produced.",
                f"Judgment status: {result.judgment_status}. Decision: INCONCLUSIVE. "
                f"Halting condition: {result.halted_by}. Compressed rule: {result.compressed_rule}.",
            ],
        )
        if result.trace_health:
            add_diagnostic_section("Trace note", summarize_trace_health()[:4])
        return "\n".join(lines) + "\n"

    final_policy_support = final.policy.get(result.current_plurality, result.confidence)
    valid_count = sum(candidate.schema_valid for candidate in final.candidates)
    received_broadcast = final.received_broadcast or final.broadcast
    dissent = final.dissent
    supporters = [
        candidate
        for candidate in final.candidates
        if getattr(candidate, "schema_valid", False)
        and getattr(candidate, "recommended_action", "") == result.selected_action
    ]
    supporter_names = ", ".join(
        dict.fromkeys(candidate.specialist for candidate in supporters if candidate.specialist)
    )

    if result.judgment_status == "ACTION_RECOMMENDATION":
        judgment_sentence = f"The Parliament recommends **{result.selected_action}**."
    elif result.judgment_status == "CONTESTED_RECOMMENDATION":
        judgment_sentence = f"The Parliament recommends **{result.selected_action}** as a contested recommendation."
    elif result.judgment_status == "CONDITIONAL":
        judgment_sentence = (
            f"The Parliament currently leans toward **{result.current_plurality or result.selected_action}** "
            f"as a conditional judgment."
        )
    elif result.judgment_status == "UNDERDETERMINED":
        judgment_sentence = (
            f"The Parliament cannot yet justify a conclusive choice; current plurality is "
            f"{result.current_plurality or 'none'}."
        )
    else:
        judgment_sentence = f"The Parliament's status is {result.judgment_status.lower().replace('_', ' ')}."

    if str(result.halted_by or "").strip():
        halted_sentence = (
            "The run stopped because the cycle budget ran out rather than because the Parliament naturally converged."
            if result.halted_by == "cycle_budget"
            else f"The run stopped because {result.halted_by.replace('_', ' ')}."
        )
    else:
        halted_sentence = "The run stopped without a recorded halt reason."

    add_section(
        "Judgment",
        [
            judgment_sentence,
            f"Final policy support: {final_policy_support:.2f}. Epistemic confidence: {result.epistemic_confidence:.2f}. "
            f"{halted_sentence} Valid delegates: {valid_count}/{len(final.candidates)}.",
        ],
    )

    problem_shape_paragraphs = summarize_problem_shape_paragraphs(result.to_dict())
    if problem_shape_paragraphs:
        add_section(
            "What the case turns on",
            problem_shape_paragraphs[:4],
        )
    else:
        add_section(
            "What the case turns on",
            [
                "The authoritative state did not expose a compact problem-shape relation, so the Parliament relied mainly on framework pressure and the scenario itself.",
            ],
        )

    if final.winner is None:
        leading_paragraphs = [
            sentence(
                f"The final cycle produced no valid deliberative winner. System status: "
                f"{final.system_error}; the last valid ProblemState was retained."
            ),
        ]
    else:
        leading_paragraphs = [
            sentence(
                f"The final cycle's winning constraint was {final.winner.constraint}; the previous broadcast context was {received_broadcast.constraint}; "
                f"the last emitted broadcast was {final.broadcast.constraint}."
            ),
            sentence(
                f"Final winning constraint: {final.winner.constraint}. Previous broadcast context: {received_broadcast.constraint}. "
                f"Last emitted broadcast: {final.broadcast.constraint}. The integrator kept the original agents in view: {', '.join(result.source_testimonies) or 'none'}."
            ),
        ]
    if supporter_names:
        leading_paragraphs.append(
            sentence(
                f"The supportive voices were {supporter_names}, which kept the recommendation alive despite the remaining objection."
            )
        )
    if dissent:
        dissent_line = (
            f"The strongest preserved objection came from {dissent.specialist} under {dissent.constraint}"
        )
        if dissent.rationale:
            dissent_line += f" ({dissent.rationale})"
        leading_paragraphs.append(sentence(dissent_line))
    add_section("Why the Parliament lands there", leading_paragraphs)

    unresolved_paragraphs: list[str] = []
    if dissent:
        unresolved_paragraphs.append(
            sentence(
                f"The strongest preserved objection remains active: {dissent.specialist} preserved {dissent.constraint}"
                + (f" ({dissent.rationale})" if dissent.rationale else "")
            )
        )
    if result.moral_residue:
        unresolved_paragraphs.extend(
            sentence(render_residue(constraint))
            for constraint in result.moral_residue[:4]
        )
    substantive_reopen = [
        condition for condition in result.reopen_conditions
        if str(condition).strip().upper() != "RETRY_MODEL_CALL"
    ]
    if substantive_reopen:
        unresolved_paragraphs.append(sentence(
            "The judgment should be reconsidered when "
            + "; or ".join(
                render_reopen_condition(condition)
                for condition in substantive_reopen[:3]
            )
        ))
    if unresolved_paragraphs:
        add_section("What remains unresolved", unresolved_paragraphs)

    if result.termination_assessment is not None:
        termination = result.termination_assessment
        add_section(
            "Process note",
            [
                sentence(
                    f"The run ended as {termination.termination_type.lower().replace('_', ' ')}; resource-censored={str(termination.resource_censored).lower()}; "
                    f"convergence evidence={termination.convergence_evidence:.2f}"
                )
            ],
        )
    if result.further_deliberation_estimate is not None:
        estimate = result.further_deliberation_estimate
        add_section(
            "Further-deliberation estimate",
            [
                sentence(f"Action-change signal: {estimate.action_change_signal:.2f}."),
                sentence(f"New-constraint signal: {estimate.new_material_constraint_signal:.2f}."),
            ],
        )

    appendix_paragraphs: list[str] = []
    activated_ev = next(
        (item for item in reversed(result.ev_dominance_assessments) if item.get("activated")),
        None,
    )
    if activated_ev:
        appendix_paragraphs.append(
            sentence(
                f"EV dominance circuit breaker: {float(activated_ev['ratio']):.2f}× with majority "
                f"{activated_ev['majority_count']}/{activated_ev['valid_delegate_count']} on unit "
                f"{activated_ev['unit']} ({activated_ev['direction']})"
            )
        )
    active_visibilities = [
        visibility
        for visibility in result.visibility_assessments
        if visibility.valid and visibility.activated
    ]
    if active_visibilities:
        visibility = active_visibilities[-1]
        penalties = ", ".join(
            f"{action}×{value:.2f}"
            for action, value in visibility.action_multipliers.items()
            if value < 1.0
        )
        appendix_paragraphs.append(
            sentence(
                f"Visibility audit: mechanism={visibility.mechanism}; confidence adjustment={penalties or 'none'}"
            )
        )
    active_autonomy = [
        autonomy
        for autonomy in result.autonomy_assessments
        if autonomy.valid and autonomy.activated
    ]
    if active_autonomy:
        autonomy = active_autonomy[-1]
        tagged = ", ".join(
            f"{action}={tag}"
            for action, tag in autonomy.action_tags.items() if tag != "NONE"
        )
        autonomy_line = f"Autonomy audit: {tagged or 'none'}"
        if autonomy.voluntary_alternative.casefold() != "none":
            autonomy_line += f"; voluntary-exhaustion probe: {autonomy.voluntary_alternative}"
        appendix_paragraphs.append(sentence(autonomy_line))
    if result.synthesis_proposals:
        proposal = next(
            (proposal for proposal in reversed(result.synthesis_proposals) if proposal.accepted),
            result.synthesis_proposals[-1],
        )
        if proposal.accepted:
            addressed = ", ".join(proposal.addressed_constraints) or "none recorded"
            appendix_paragraphs.append(
                sentence(
                    f"Synthesis: {proposal.action or 'none'} was admitted; addressed constraints: {addressed}; "
                    f"feasibility={proposal.feasibility:.2f}; rationale: {proposal.rationale or 'none recorded'}"
                )
            )
        else:
            appendix_paragraphs.append(
                sentence(
                    f"Synthesis: {proposal.action or 'none'} was rejected because {proposal.rejection_reason or 'no rejection reason recorded'}"
                )
            )
    if result.contingency_feasibility_assessments:
        assessment = result.contingency_feasibility_assessments[-1]
        fallback_text = ", ".join(
            f"{action}={status}" for action, status in (assessment.fallback_statuses or {}).items()
        ) or "unavailable"
        basis_text = ", ".join(
            f"{action}={value}" for action, value in (assessment.evidence_bases or {}).items()
        ) or "unavailable"
        shared_note = (
            "shared failure: the synthesis and at least one fallback lose a common capability"
            if assessment.shared_failure
            else "no shared failure detected"
        )
        appendix_paragraphs.append(
            sentence(
                f"Contingency feasibility: Fallback availability: {fallback_text}. Shared-failure check: {shared_note}. "
                f"Basis: {basis_text}. Reason: {assessment.error or 'none'}"
            )
        )
    if result.planning_assessments:
        assessment = result.planning_assessments[-1]
        appendix_paragraphs.append(
            sentence(
                f"Planning assessment: {'broadcast' if assessment.broadcast_worthy else 'private'}; "
                f"target={assessment.target_action}; feasibility={assessment.feasibility:.2f}"
            )
        )
    if result.access_decisions:
        admitted = [decision for decision in result.access_decisions if decision.admitted]
        if admitted:
            decision = admitted[-1]
            audit_variable = decision.audit_variable or {}
            parts = [
                f"{key}={audit_variable.get(key, 'NONE')}"
                for key in ("entity", "relation", "focus_action")
                if audit_variable.get(key)
            ]
            if audit_variable.get("possible_values"):
                parts.append(
                    "possible_values="
                    + ",".join(map(str, audit_variable.get("possible_values", [])))
                )
            if isinstance(audit_variable.get("required_response"), dict):
                anchor = audit_variable["required_response"].get("counterfactual_anchor", "")
                if anchor:
                    parts.append(f"counterfactual_anchor={anchor}")
            access_line = f"{decision.content_type} admitted ({', '.join(decision.signals)})"
            if parts:
                access_line += " | typed audit variable: " + ", ".join(parts)
            appendix_paragraphs.append(sentence(f"{access_line}. Audit question: {decision.question}"))
    if result.source_baselines:
        baseline_lines: list[str] = []
        for specialist, baseline in result.source_baselines.items():
            status = str(baseline.get("status", "UNKNOWN"))
            action_id = _baseline_display_action(baseline)
            reason = str(baseline.get("reason", "")).strip()
            if status and action_id:
                entry = f"{specialist}: {status} {action_id}"
                if reason:
                    entry += f" ({reason})"
                baseline_lines.append(entry)
        if baseline_lines:
            appendix_paragraphs.append("Framework baselines: " + "; ".join(baseline_lines[:5]) + ".")
    if result.action_source_grounding:
        grounding = result.action_source_grounding
        status = str(grounding.get("status", "UNKNOWN"))
        mapping_lines: list[str] = []
        for action_id, mapping in (grounding.get("actions", {}) or {}).items():
            if not isinstance(mapping, dict):
                continue
            cited = ",".join(str(value) for value in mapping.get("clause_ids", [])) or "NONE"
            action = str(mapping.get("action", action_id))
            reason = str(mapping.get("reason", "")).strip()
            mapping_lines.append(
                f"{action_id}={action} <- {cited}" + (f" ({reason})" if reason else "")
            )
        errors = " | ".join(str(value) for value in grounding.get("errors", []))
        grounding_line = f"Action-source grounding: {status}"
        if mapping_lines:
            grounding_line += "; " + "; ".join(mapping_lines)
        if errors:
            grounding_line += f"; errors={errors}"
        appendix_paragraphs.append(sentence(grounding_line))
    audited = [
        (
            (
                candidate.specialist,
                candidate.assumption_status,
                candidate.unsupported_assumption,
                candidate.reversal_condition,
            ),
            f"{candidate.specialist}={candidate.assumption_status} "
            f"(assumption: {candidate.unsupported_assumption}; reversal: {candidate.reversal_condition})",
            cycle.cycle,
        )
        for cycle in result.cycles
        if not cycle.is_hypothetical
        for candidate in cycle.candidates
        if candidate.assumption_status != "NOT_AUDITED"
    ]
    if audited:
        grouped: dict[tuple[str, ...], dict[str, object]] = {}
        for key, entry, cycle in audited:
            bucket = grouped.setdefault(key, {"entry": entry, "cycles": []})
            if cycle is not None:
                bucket["cycles"].append(cycle)
        for bucket in list(grouped.values())[:4]:
            entry = str(bucket["entry"])
            cycles = [str(cycle) for cycle in bucket["cycles"] if cycle is not None]
            if cycles:
                entry = f"{entry} [cycles {', '.join(dict.fromkeys(cycles))}]"
            appendix_paragraphs.append(entry + ".")
    reversal_reviews = [
        candidate
        for cycle in result.cycles
        if cycle.is_hypothetical
        and (cycle.received_broadcast or cycle.broadcast).constraint == "REVERSAL_AUDIT"
        for candidate in cycle.candidates
        if candidate.reversal_review_response != "NOT_TESTED"
    ]
    if reversal_reviews:
        appendix_paragraphs.append(
            "Conditional reversal review: "
            + "; ".join(
                sentence(
                    f"{candidate.specialist}={candidate.reversal_review_response} ({candidate.reversal_review_justification})"
                )
                for candidate in reversal_reviews[:3]
            )
        )
    if result.problem_reformulations:
        reformulation = result.problem_reformulations[-1]
        reformulation_text = "admitted" if reformulation.accepted else f"rejected ({reformulation.rejection_reason})"
        if reformulation.accepted:
            reformulation_text += (
                f"; Switch condition: {reformulation.switch_condition or 'none'}; "
                f"Residual tension: {reformulation.residual_tension or 'none'}"
            )
        if reformulation.question:
            reformulation_text += f"; Question: {reformulation.question}"
        appendix_paragraphs.append("Problem reformulation: " + reformulation_text + ".")
    if result.compressed_rule:
        appendix_paragraphs.append("Compressed rule: " + sentence(result.compressed_rule))
    failed_invariants = [record for record in result.semantic_invariants if not record.valid]
    if failed_invariants:
        appendix_paragraphs.append(
            "Semantic invariants: "
            + "; ".join(
                sentence(f"{record.boundary} ({' | '.join(record.errors)})")
                for record in failed_invariants[:3]
            )
        )
    if result.source_errors:
        appendix_paragraphs.append(
            "Unavailable original agents: " + ", ".join(result.source_errors) + "."
        )
    if appendix_paragraphs:
        add_section("Appendix: guardrails and review notes", appendix_paragraphs[:10])
    if result.trace_health:
        add_diagnostic_section("Trace note", summarize_trace_health()[:6])
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
    if source_action_legend:
        _validate_lossless_action_set(list(source_action_legend.values()), scenario)
    try:
        actions = args.actions or propose_actions(llm, scenario)
    except ValueError as exc:
        print(f"Action planning could not produce a safe feasible set: {exc}", flush=True)
        print("Rerun with explicit choices, for example: --actions \"first action\" \"second action\"", flush=True)
        return 2
    _validate_lossless_action_set(actions, scenario)
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
    action_source_grounding = ground_actions_in_scenario(
        llm, scenario, actions, max_tokens=max(128, args.delegate_tokens),
    )
    print(
        "Action-source grounding: "
        + json.dumps(action_source_grounding, ensure_ascii=False, sort_keys=True),
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
        action_source_grounding=action_source_grounding,
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
