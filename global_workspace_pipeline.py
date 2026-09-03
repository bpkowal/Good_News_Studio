from __future__ import annotations

import argparse
import copy
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
    render_decision_brief,
    render_public_judgment,
    summarize_problem_shape_paragraphs,
)
from global_workspace.premise_audit import audit_side_premises
from global_workspace.scenario_semantics import (
    build_presentation_action_mapping,
    canonicalize_action_order,
    canonicalize_deliberation_scenario,
)
from global_workspace.structured_io import reset_model_call_budget, start_model_call_budget
from global_workspace.visibility import assess_visibility
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = (ROOT / "../mistral-7b-instruct-v0.2.Q4_K_M.gguf").resolve()
FRAMING_CACHE_VERSION = 2
FRAMING_CACHE_FILENAME = "last_problem_framing.json"


def load_problem_framing_cache(
    path: Path, ethical_problem: str,
) -> tuple[dict[str, object] | None, str]:
    """Load the last successful framing only for an exact problem-text match."""
    if not path.exists():
        return None, "MISS_NO_CACHE"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, "MISS_INVALID_CACHE"
    if not isinstance(payload, dict) or payload.get("cache_version") != FRAMING_CACHE_VERSION:
        return None, "MISS_INCOMPATIBLE_CACHE"
    if payload.get("ethical_problem") != ethical_problem:
        return None, "MISS_DIFFERENT_PROBLEM"
    actions = payload.get("presentation_actions")
    grounding = payload.get("action_source_grounding")
    canonical_actions = payload.get("canonical_actions")
    canonical_scenario = payload.get("canonical_scenario")
    if (
        not isinstance(actions, list) or not 2 <= len(actions) <= 5
        or not all(isinstance(action, str) and action.strip() for action in actions)
        or not isinstance(canonical_actions, list)
        or not all(isinstance(action, str) and action.strip() for action in canonical_actions)
        or not isinstance(canonical_scenario, str) or not canonical_scenario.strip()
        or not isinstance(grounding, dict)
        or str(grounding.get("status") or "").upper() != "COMMITTED"
        or not isinstance(grounding.get("world_model"), dict)
    ):
        return None, "MISS_INVALID_CACHE"
    try:
        from global_workspace.world_state import world_model_from_dict
        restored_world = world_model_from_dict(grounding["world_model"])
    except (KeyError, TypeError, ValueError):
        return None, "MISS_INVALID_CACHE"
    if restored_world is None:
        return None, "MISS_INVALID_CACHE"
    return payload, "HIT"


def save_problem_framing_cache(path: Path, payload: dict[str, object]) -> None:
    """Atomically replace the single-entry successful-framing cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    complete = {**payload, "cache_version": FRAMING_CACHE_VERSION}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(complete, indent=2), encoding="utf-8")
    temporary.replace(path)


def cached_framing_matches(
    cached: dict[str, object] | None,
    *,
    presentation_actions: list[str],
    canonical_actions: list[str],
    canonical_scenario: str,
) -> bool:
    return bool(
        cached is not None
        and cached.get("presentation_actions") == presentation_actions
        and cached.get("canonical_actions") == canonical_actions
        and cached.get("canonical_scenario") == canonical_scenario
    )


def choose_initial_actions(
    explicit_actions: list[str] | None,
    cached: dict[str, object] | None,
    planner,
) -> tuple[list[str], bool]:
    if explicit_actions:
        return list(explicit_actions), False
    if cached is not None:
        return list(cached["presentation_actions"]), True
    return list(planner()), False


def choose_action_source_grounding(
    cached: dict[str, object] | None,
    *,
    cache_matches: bool,
    grounder,
) -> tuple[dict[str, object], bool]:
    if cached is not None and cache_matches:
        return copy.deepcopy(cached["action_source_grounding"]), True
    return dict(grounder()), False


def _baseline_display_action(baseline: dict[str, object]) -> str:
    """Render committed actions for direct stances and provisional actions otherwise."""
    status = str(baseline.get("status", "UNAVAILABLE")).strip().upper()
    field = "action_id" if status == "DIRECT" else "provisional_action_id"
    return str(baseline.get(field, "NONE")).strip().upper() or "NONE"



def render_summary(result) -> str:
    """Default user-facing summary: decision brief + deliberation map.

    Implementation-oriented diagnostics remain in the saved JSON trace.
    """
    return render_decision_brief(result)


def resolve_world_state_contradictions(
    grounding: dict[str, object],
    *,
    input_fn=input,
    output_fn=print,
) -> bool:
    """Ask whether an unrepaired contradictory direct-effect set may be quarantined."""
    groups = grounding.get("world_contradictions") or []
    if not groups:
        return True
    from global_workspace.world_state import (
        quarantine_contradictions, world_model_from_dict,
    )
    model = world_model_from_dict(grounding.get("world_model"))
    if model is None:
        output_fn("World-state contradictions were reported without a recoverable typed model.")
        return False
    effect_by_id = {effect.effect_id: effect for effect in model.effects}
    party_by_id = {party.party_id: party.label for party in model.parties}
    output_fn("\nWorld-state validation could not resolve contradictory direct effects:")
    for group in groups:
        for effect_id in group:
            effect = effect_by_id.get(str(effect_id))
            if effect is None:
                continue
            output_fn(
                f"- {effect.action_id} / {party_by_id.get(effect.party_id, effect.party_id)}: "
                f"{effect.relation} — {effect.outcome} [{effect.modality}]"
            )
    output_fn(
        "These effects will not be treated as established facts. Continuing will "
        "quarantine every member of each contradictory set and mark the judgment "
        "as based on a degraded world state."
    )
    answer = input_fn(
        "Continue while ignoring all listed contradictory direct effects? [y/N]: "
    ).strip().casefold()
    accepted = answer in {"y", "yes"}
    resolved = quarantine_contradictions(model, groups, user_override=accepted)
    grounding["world_model"] = resolved.as_dict()
    grounding["world_model_status"] = resolved.admission.status
    grounding["status"] = (
        "COMMITTED_WITH_QUARANTINE" if accepted
        else "ABANDONED_CONTRADICTORY_WORLD_STATE"
    )
    return accepted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the recurrent ethical global workspace.")
    parser.add_argument("scenario", type=Path, help="Scenario JSON containing ethical_question")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("local", "openai"), default="local")
    parser.add_argument("--openai-model", default="o3")
    parser.add_argument(
        "--agents", nargs="+", choices=tuple(AGENT_MODULES),
        help="Run only the selected ethical frameworks (at least two)",
    )
    parser.add_argument("--actions", nargs="+", help="Skip local action planning and use these actions")
    parser.add_argument("--urgency", type=float, default=0.5)
    parser.add_argument("--danger", type=float, default=0.5)
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--time-budget", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "workspace_outputs")
    parser.add_argument(
        "--no-framing-cache", action="store_true",
        help="Recompute action planning and action-source grounding",
    )
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
    selected_agents = [
        name for name in AGENT_MODULES
        if name in set(args.agents or AGENT_MODULES)
    ]
    if len(selected_agents) < 2:
        raise ValueError("Select at least two ethical frameworks for a workspace run")
    print("Active ethical frameworks: " + ", ".join(selected_agents), flush=True)
    load_dotenv()
    scenario_path = args.scenario.resolve()
    data = json.loads(scenario_path.read_text(encoding="utf-8"))
    scenario = str(data.get("ethical_question", "")).strip()
    if not scenario:
        raise ValueError("Scenario JSON must contain a non-empty ethical_question")
    ethical_problem = scenario
    framing_cache_path = args.output_dir / FRAMING_CACHE_FILENAME
    if args.no_framing_cache:
        cached_framing = None
        framing_cache_lookup = "DISABLED"
    else:
        cached_framing, framing_cache_lookup = load_problem_framing_cache(
            framing_cache_path, ethical_problem,
        )
    grounding_reused = False
    framing_cache_written = False
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
    source_labels_explicit = bool(source_action_legend)
    if source_action_legend:
        _validate_lossless_action_set(list(source_action_legend.values()), scenario)
    try:
        actions, cached_actions_reused = choose_initial_actions(
            args.actions, cached_framing, lambda: propose_actions(llm, scenario),
        )
        if cached_actions_reused:
            print(
                "Framing cache hit: reusing the last action set for this exact problem.",
                flush=True,
            )
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
    presentation_action_mapping = build_presentation_action_mapping(
        scenario,
        presentation_action_legend,
        actions,
        source_labels_explicit=source_labels_explicit,
    )
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
    cached_actions_match = cached_framing_matches(
        cached_framing,
        presentation_actions=presentation_actions,
        canonical_actions=actions,
        canonical_scenario=scenario,
    )
    action_source_grounding, grounding_reused = choose_action_source_grounding(
        cached_framing,
        cache_matches=cached_actions_match,
        grounder=lambda: ground_actions_in_scenario(
            llm, scenario, actions, max_tokens=max(128, args.delegate_tokens),
        ),
    )
    if grounding_reused:
        print(
            "Framing cache hit: reusing committed action-source grounding.",
            flush=True,
        )
    else:
        if cached_framing is not None:
            framing_cache_lookup = "MISS_ACTION_SET_CHANGED"
            print(
                "Cached actions changed; recomputing action-source grounding.",
                flush=True,
            )
    print(
        "Action-source grounding: "
        + json.dumps(action_source_grounding, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    if action_source_grounding.get("world_contradictions"):
        if not resolve_world_state_contradictions(action_source_grounding):
            print("Run abandoned because the direct world state remained contradictory.")
            return 2
    from global_workspace.action_identity import (
        build_canonical_action_records,
        extract_scenario_actor,
        partition_records_for_deliberation,
        validate_action_set_completeness,
    )
    grounded_texts: dict[str, list[str]] = {}
    for action_id, row in (action_source_grounding.get("actions") or {}).items():
        texts = []
        for clause in row.get("clauses") or []:
            text = " ".join(str(clause.get("text", "")).split())
            if text:
                texts.append(text)
        if texts:
            grounded_texts[str(action_id)] = texts
    grounding_status = str(action_source_grounding.get("status") or "").upper()
    if (
        grounding_status == "COMMITTED"
        and grounded_texts
        and not action_source_grounding.get("world_model")
    ):
        validate_action_set_completeness(
            actions,
            scenario=scenario,
            grounded_clause_texts_by_id=grounded_texts,
        )
    records = build_canonical_action_records(
        actions,
        actor=extract_scenario_actor(scenario),
        grounded_clause_texts_by_id=grounded_texts,
        scenario=scenario,
        grounding_status=grounding_status,
        world_model=dict(action_source_grounding.get("world_model") or {}),
    )
    if action_source_grounding.get("world_model"):
        from global_workspace.world_state import world_model_from_dict
        typed_world = world_model_from_dict(action_source_grounding["world_model"])
        if typed_world is None:
            raise SystemExit("Refusing to deliberate: typed world state could not be restored.")
        expected_effect_ids = {
            effect.effect_id
            for action in typed_world.actions
            for effect in typed_world.effects_for(action.action_id)
        }
        record_effect_ids = {
            str(effect.get("effect_id"))
            for record in records
            for effect in record.world_effects
        }
        if record_effect_ids != expected_effect_ids:
            raise SystemExit(
                "Refusing to deliberate: canonical records disagree with the admitted "
                "typed world effects."
            )
    # Deliberation runs on committed semantic state only. Previously a rejected
    # grounding merely skipped validation and handed the same records downstream,
    # so agents reasoned over a world model the system had already disowned.
    if action_source_grounding.get("repair_attempts"):
        print(
            "Action-source grounding required "
            f"{action_source_grounding['repair_attempts']} repair attempt(s).",
            flush=True,
        )
    admitted, withheld = partition_records_for_deliberation(records)
    # The action set is the unit of admission, not the individual action.
    # Deliberating over a subset would silently pose a different dilemma than
    # the one the user asked about, so any withheld action halts the run.
    if withheld:
        detail = "; ".join(
            f"{record.action_id} [{record.commitment_status}] "
            + ", ".join(record.commitment_reasons or ("unspecified",))
            for record in withheld
        )
        raise SystemExit(
            "Refusing to deliberate: "
            f"{len(withheld)} of {len(records)} canonical action records did not "
            f"reach COMMITTED state. {detail}"
        )
    canonical_action_records = [record.as_dict() for record in admitted]
    if (
        not args.no_framing_cache and not grounding_reused
        and str(action_source_grounding.get("status") or "").upper() == "COMMITTED"
    ):
        try:
            save_problem_framing_cache(framing_cache_path, {
                "ethical_problem": ethical_problem,
                "presentation_actions": presentation_actions,
                "canonical_actions": actions,
                "canonical_scenario": scenario,
                "action_source_grounding": action_source_grounding,
            })
            framing_cache_written = True
            print(f"Updated framing cache: {framing_cache_path}", flush=True)
        except OSError as exc:
            print(f"Framing cache could not be saved: {exc}", flush=True)
    # Agents reason from the semantic action object, never a truncated label.
    actions = [
        str(record["canonical_semantic_action"])
        for record in canonical_action_records
    ]
    source_action_legend = {
        str(record["action_id"]): str(record["canonical_semantic_action"])
        for record in canonical_action_records
    }
    print(
        "Canonical action records: "
        + json.dumps(canonical_action_records, ensure_ascii=False, sort_keys=True),
        flush=True,
    )

    # The original RAG agents are part of the same experimental treatment. Give
    # them the canonical mapping too; otherwise order bias can enter through the
    # frozen testimony before workspace safeguards ever run.
    if args.skip_original_agents:
        testimonies: dict[str, str] = {}
        source_errors = {name: "skipped by request" for name in selected_agents}
    else:
        consultation = consult_original_agents(
            scenario_path,
            agents=tuple(selected_agents),
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
    for name in selected_agents:
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
    specialist_names = list(selected_agents)
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
            canonical_action_records=[dict(record) for record in canonical_action_records],
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
        canonical_action_records=canonical_action_records,
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
        audit_side_premises=(
            lambda ledger, candidates: audit_side_premises(
                llm, ledger, candidates,
                max_tokens=max(600, args.delegate_tokens * 4),
            )
        ),
        )
    finally:
        reset_model_call_budget(budget_token)
    result.source_testimonies = testimonies
    result.source_errors = source_errors
    # Attach only after deliberation: this is explanatory output metadata, not
    # an input to specialists, broadcasts, salience, scoring, or synthesis.
    result.presentation_action_mapping = presentation_action_mapping
    result.framing_cache = {
        "lookup_status": framing_cache_lookup,
        "actions_reused": cached_actions_reused,
        "grounding_reused": grounding_reused,
        "cache_written": framing_cache_written,
        "cache_version": FRAMING_CACHE_VERSION,
    }
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
