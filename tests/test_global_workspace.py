from __future__ import annotations

import unittest
import base64
import concurrent.futures
import importlib
import json
import os
import tempfile
import threading
import time
from contextlib import ExitStack, chdir
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from global_workspace.engine import (
    WorkspaceConfig, WorkspaceEngine, _preserve_problem_state_after_invalid_cycle,
    _problem_state_audit_probe, _operative_framework_candidates,
    _advance_argument_challenge_agenda, _argument_challenge_candidates,
    _record_committed_native_ledger, _challenge_resolution_supported,
)
from global_workspace.epistemic_ledger import seed_proposition_ledger
from global_workspace.action_identity import compile_action_identity
from global_workspace.evidence_calibration import EvidenceCalibration
from global_workspace.framework_vote_integrity import apply_framework_vote_integrity
from global_workspace.framework_retrieval import format_evidence_context
from global_workspace.legacy_bridge import (
    PERFORMANCE_MARKER, RESPONSE_MARKER, consult_original_agents,
)
from global_workspace.local_specialists import (
    CompactLocalSpecialist,
    SpecialistEvaluationStageError,
    _admitted_audit_variable,
    _candidate_from_data,
    _construct_map_errors,
    _landscape_semantic_errors,
    _scenario_closes_action_set,
    analyze_action_plan,
    assess_visibility,
    extract_allocation_actions,
    extract_acceptability_actions,
    extract_declared_action_legend,
    extract_explicit_actions,
    extract_labeled_action_legend,
    extract_scenario_facts,
    generate_failure_condition,
    infer_testimony_baseline,
    infer_testimony_stance,
    propose_actions,
    propose_problem_reformulation,
    propose_synthesis,
)
from global_workspace.memory import EpisodicMemory, summarize_specialist_contributions
from global_workspace.rawls_ledger import apply_rawls_ledger_transaction
from global_workspace.deontology_ledger import (
    DutyAssessmentProposal, apply_deontological_ledger_transaction,
    calibrate_deontological_adjudication, committed_deontological_assessments,
    harm_relation_conflicts_with_graph,
    omission_classified_as_perfect_negative_violation,
    render_deontological_adjudication, supersede_calibration_issues,
    OMISSION_PERFECT_NEGATIVE_VIOLATION, _party_grounding,
)
from global_workspace.utilitarian_ledger import (
    _best_evidence, apply_utilitarian_ledger_transaction,
)
from global_workspace.care_ledger import (
    apply_care_ledger_transaction, committed_care_assessments,
)
from global_workspace.resolved_questions import (
    QuestionResolution,
    commit_question_resolution,
    question_resolution_index,
    resolve_audited_question,
    settled_question_keys,
)
from global_workspace.source_cache import build_source_cache_key
from global_workspace.models import ArgumentChallenge, CalibrationOutcome, CandidateChunk, ContingencyFeasibilityAssessment, CycleRecord, FailureCondition, PlanningAssessment, PlanningBranchEvaluation, ProblemReformulation, ProposalFrameworkReview, SynthesisProposal, VisibilityAssessment, WorkspaceAccessDecision, WorkspaceBroadcast, WorkspaceResult, _NATIVE_REASONING_CHAR_BUDGET, _balanced_challenge_projection, _balanced_problem_state_projection, _fit_native_reasoning
from global_workspace.models import AutonomyAssessment
from global_workspace.construct_validity import collect_typed_residue
from global_workspace.contingency_graph import (
    certify_fallback_availability, compile_contingency_graph,
    validate_contingency_graph_dict,
)
from global_workspace.contingency_feasibility import verify_contingency_feasibility
from global_workspace.deliberative_state import (
    apply_verified_challenge_supersession,
    build_deliberative_problem_state,
    observe_broadcast_influence,
    opening_problem_state,
    update_broadcast_influence_persistence,
)
from global_workspace.landscape_validation import _comparative_claim_errors
from global_workspace.middleware.moral_residue import collect_moral_residue
from global_workspace.trace_health import audit_trace_health
from global_workspace.graph_transactions import SemanticGraphStore
from global_workspace.world_state import (
    CausalLink, ScenarioWorldModel, SourceRef, WorldAction, WorldCondition,
    WorldEffect, WorldParty, WorldStateAdmission,
)
from global_workspace.frozen_world_replay import (
    FrozenWorldReplayError, load_frozen_world_trace,
)
from global_workspace.performance import (
    performance_stage, record_performance_duration,
    reset_performance_trace, start_performance_trace,
)
from global_workspace.retrieval_trace import (
    disabled_retrieval_result, serialize_retrieval_result,
)
from global_workspace.invariance import compare_label_permutation_traces
from global_workspace.openai_backend import OpenAIWorkspaceLLM
from global_workspace.presentation import render_public_judgment, _support_reason
from global_workspace.semantic_state import (
    _derive_action_dimensions, project_authoritative_semantic_state,
)
from global_workspace.semantic_equivalence import (
    compare_semantic_results, semantic_run_snapshot,
)
from global_workspace.semantic_graph import (
    SemanticGraph, SemanticNode, SemanticEdge, merge_graphs,
)
from global_workspace.scenario_semantics import (
    build_presentation_action_mapping,
    canonicalize_action_order, canonicalize_deliberation_scenario,
    compile_scenario_graph, resolved_semantic_action_keys, semantic_action_key,
    segment_scenario_clauses,
)
from global_workspace.scenario_semantics import (
    attach_typed_world_model,
    compile_action_burdens, compile_execution_obstacles,
    compile_observability_facts,
    classify_planning_failure_grounding,
    project_grounded_action_effects,
)
from global_workspace.structured_io import (
    _CALL_BUDGET, ModelCallBudgetExceeded, ModelCallUnavailable,
    begin_model_call_cycle, call_json_llm, pause_model_call_budget,
    reset_model_call_budget, resume_model_call_budget,
    start_model_call_budget, submit_with_context,
)
from global_workspace_pipeline import _baseline_display_action, render_summary


class FixedSpecialist:
    def __init__(self, name: str, preferred: str, constraint: str, unresolved: str = "NONE"):
        self.name = name
        self.preferred = preferred
        self.constraint = constraint
        self.unresolved = unresolved

    def evaluate(self, scenario, actions, broadcast):
        return CandidateChunk(
            specialist=self.name,
            constraint=self.constraint,
            action_scores={action: 0.95 if action == self.preferred else 0.05 for action in actions},
            surprise=0.6,
            friction=0.7,
            confidence=0.9,
            unresolved=self.unresolved,
            recommended_action=self.preferred,
            rationale="Compact test judgment.",
            decision_rule=f"Prefer {self.preferred} under {self.constraint}.",
            adjudication_status="SUPPORTS",
            governing_eligible=True,
            broadcast_authority="GOVERNING_CANDIDATE",
        )


class InvalidSpecialist(FixedSpecialist):
    def evaluate(self, scenario, actions, broadcast):
        chunk = super().evaluate(scenario, actions, broadcast)
        chunk.schema_valid = False
        chunk.validation_errors = ["invalid constraint"]
        chunk.confidence = 1.0
        return chunk


class CycleEvaluationBarrierTests(unittest.TestCase):
    def _config(self) -> WorkspaceConfig:
        return WorkspaceConfig(
            max_cycles=1,
            stable_cycles_required=3,
            min_valid_specialists=2,
            enable_consensus_audit=False,
            enable_problem_state_audit=False,
            enable_reversal_audit=False,
            enable_synthesis=False,
            enable_planning=False,
        )

    def test_every_evaluation_finishes_before_configured_order_commit(self):
        events: list[str] = []

        class TracingEngine(WorkspaceEngine):
            def evaluate_specialist(self, specialist, cycle_input, **kwargs):
                events.append(f"evaluate:{specialist.name}")
                return super().evaluate_specialist(
                    specialist, cycle_input, **kwargs,
                )

            def commit_candidate(
                self, specialist, evaluation, cycle_input, **kwargs,
            ):
                events.append(f"commit:{specialist.name}")
                return super().commit_candidate(
                    specialist, evaluation, cycle_input, **kwargs,
                )

        TracingEngine(
            [
                FixedSpecialist("first", "protect", "CARE"),
                FixedSpecialist("second", "protect", "DUTY"),
            ],
            self._config(),
        ).run("Choose protect or decline.", ["protect", "decline"])

        self.assertEqual(events, [
            "evaluate:first", "evaluate:second",
            "commit:first", "commit:second",
        ])

    def test_cycle_inputs_are_isolated_from_legacy_delegate_mutation(self):
        observations: list[dict[str, object]] = []

        class SnapshotSpecialist(FixedSpecialist):
            def __init__(self, name: str, mutate: bool = False):
                super().__init__(name, "protect", "CARE")
                self.mutate = mutate
                self.scenario_graph = None
                self.proposition_ledger = []
                self.canonical_action_records = []
                self.source_action_legend = {}

            def evaluate(self, scenario, actions, broadcast):
                observations.append({
                    "specialist": self.name,
                    "graph_nodes": tuple(sorted(self.scenario_graph.nodes)),
                    "propositions": tuple(
                        row.get("proposition_id", "")
                        for row in self.proposition_ledger
                    ),
                    "agenda_question": broadcast.challenge_agenda[0]["question"],
                    "problem_role": broadcast.problem_state.get("state_role"),
                    "mapping": tuple(
                        row.get("action_id", "")
                        for row in self.canonical_action_records
                    ),
                    "legend": tuple(sorted(self.source_action_legend.items())),
                })
                if self.mutate:
                    self.scenario_graph.add_node(SemanticNode(
                        "SAME_CYCLE_LEAK", "CLAIM", "must remain private",
                    ))
                    self.proposition_ledger.append({
                        "proposition_id": "SAME_CYCLE_LEAK",
                    })
                    self.canonical_action_records.append({
                        "action_id": "SAME_CYCLE_LEAK",
                    })
                    self.source_action_legend["SAME_CYCLE_LEAK"] = "mutated"
                    broadcast.challenge_agenda[0]["question"] = "mutated"
                    broadcast.problem_state["state_role"] = "mutated"
                return super().evaluate(scenario, actions, broadcast)

        initial = WorkspaceBroadcast(challenge_agenda=({
            "issue_id": "CHALLENGE:frozen-input",
            "generated_by": "workspace_argument_auditor",
            "about_specialist": "first",
            "raised_by": ["workspace_argument_auditor"],
            "target_specialists": ["first", "second"],
            "challenge_kind": "BOUNDARY_TEST",
            "question": "Which fact changes the ranking?",
            "status": "UNTESTED",
            "grounded_in": ["C0"],
        },))
        canonical_mapping = [
            {"action_id": "A0", "canonical_semantic_action": "protect"},
            {"action_id": "A1", "canonical_semantic_action": "decline"},
        ]
        WorkspaceEngine(
            [SnapshotSpecialist("first", mutate=True), SnapshotSpecialist("second")],
            self._config(),
        ).run(
            "Choose protect or decline.",
            ["protect", "decline"],
            initial_broadcast=initial,
            canonical_action_records=canonical_mapping,
            source_action_legend={"A0": "protect", "A1": "decline"},
        )

        self.assertEqual(len(observations), 2)
        first, second = observations
        self.assertEqual(first["graph_nodes"], second["graph_nodes"])
        self.assertEqual(first["propositions"], second["propositions"])
        self.assertEqual(first["agenda_question"], second["agenda_question"])
        self.assertEqual(first["problem_role"], second["problem_role"])
        self.assertEqual(first["mapping"], second["mapping"])
        self.assertEqual(first["legend"], second["legend"])
        self.assertNotIn("SAME_CYCLE_LEAK", second["graph_nodes"])

    def test_incomplete_evaluation_barrier_commits_nothing(self):
        commits: list[str] = []

        class BudgetBlockedSpecialist:
            name = "blocked"

            def evaluate(self, scenario, actions, broadcast):
                raise ModelCallBudgetExceeded("cycle budget exhausted")

        class TracingEngine(WorkspaceEngine):
            def commit_candidate(
                self, specialist, evaluation, cycle_input, **kwargs,
            ):
                commits.append(specialist.name)
                return super().commit_candidate(
                    specialist, evaluation, cycle_input, **kwargs,
                )

        result = TracingEngine(
            [
                FixedSpecialist("first", "protect", "CARE"),
                BudgetBlockedSpecialist(),
            ],
            self._config(),
        ).run("Choose protect or decline.", ["protect", "decline"])

        self.assertEqual(commits, [])
        self.assertEqual(result.halted_by, "model_call_budget")
        self.assertEqual(result.cycles, [])

    def test_committed_graph_update_appears_only_in_next_cycle_snapshot(self):
        observations: list[tuple[str, int]] = []

        class GraphAwareSpecialist(FixedSpecialist):
            def __init__(self, name: str, writes: bool = False):
                super().__init__(name, "protect", "CARE")
                self.writes = writes
                self.scenario_graph = None
                self.proposition_ledger = []
                self.canonical_action_records = []

            def evaluate(self, scenario, actions, broadcast):
                observations.append((
                    self.name,
                    sum(
                        node.kind == "CONDITION"
                        for node in self.scenario_graph.nodes.values()
                    ),
                ))
                candidate = super().evaluate(scenario, actions, broadcast)
                if self.writes:
                    candidate.factual_reversal_threshold = (
                        "expected harm exceeds ten"
                    )
                    candidate.graph_update_proposal = typed_reversal_proposal(
                        "protect", "decline",
                    )
                return candidate

        config = self._config()
        config.max_cycles = 2
        config.stop_redundant_consensus_cycles = False
        result = WorkspaceEngine(
            [
                GraphAwareSpecialist("writer", writes=True),
                GraphAwareSpecialist("observer"),
            ],
            config,
        ).run("Choose protect or decline.", ["protect", "decline"])

        self.assertEqual(len(result.cycles), 2)
        self.assertEqual([count for _name, count in observations[:2]], [0, 0])
        self.assertTrue(all(count > 0 for _name, count in observations[2:4]))
        self.assertTrue(any(
            transaction.get("status") == "COMMITTED"
            for transaction in result.graph_transactions
        ))


class BoundedOpenAIConcurrencyTests(unittest.TestCase):
    class _ConcurrentAdapter:
        supports_concurrent_calls = True
        supports_call_local_timeout = True
        timeout = 120.0

        def complete_json(self, prompt, **kwargs):
            return {"choices": [{"text": '{}'}]}

    @staticmethod
    def _config(concurrency: int = 2) -> WorkspaceConfig:
        return WorkspaceConfig(
            max_cycles=1,
            stable_cycles_required=3,
            min_valid_specialists=2,
            enable_consensus_audit=False,
            enable_problem_state_audit=False,
            enable_reversal_audit=False,
            enable_synthesis=False,
            enable_planning=False,
            openai_max_concurrency=concurrency,
        )

    def test_budget_context_and_auxiliary_allowance_are_shared_atomically(self):
        adapter = self._ConcurrentAdapter()

        def invoke():
            return call_json_llm(
                adapter,
                "bounded call",
                max_tokens=16,
                temperature=0.0,
                schema={"type": "object"},
                call_kind="auxiliary",
            )

        token = start_model_call_budget(
            30.0,
            reserve_seconds=0.0,
            max_auxiliary_calls_per_cycle=1,
        )
        try:
            begin_model_call_cycle(1)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    submit_with_context(executor, invoke) for _index in range(2)
                ]
                outcomes = []
                for future in futures:
                    try:
                        future.result()
                    except ModelCallBudgetExceeded:
                        outcomes.append("BLOCKED")
                    else:
                        outcomes.append("CALLED")
        finally:
            reset_model_call_budget(token)

        self.assertCountEqual(outcomes, ["CALLED", "BLOCKED"])

    def test_nested_pauses_resume_the_shared_deadline_only_once(self):
        token = start_model_call_budget(60.0, reserve_seconds=0.0)
        try:
            budget = _CALL_BUDGET.get()
            self.assertIsNotNone(budget)
            assert budget is not None
            original_deadline = budget.deadline
            with patch(
                "global_workspace.structured_io.time.monotonic",
                side_effect=[100.0, 105.0],
            ):
                pause_model_call_budget()
                pause_model_call_budget()
                resume_model_call_budget()
                self.assertEqual(budget.pause_depth, 1)
                self.assertEqual(budget.deadline, original_deadline)
                resume_model_call_budget()
            self.assertEqual(budget.pause_depth, 0)
            self.assertIsNone(budget.paused_at)
            self.assertAlmostEqual(budget.deadline, original_deadline + 5.0)
        finally:
            reset_model_call_budget(token)

    def test_model_budget_passes_call_local_timeout_without_mutating_adapter(self):
        observed: list[float] = []

        class TimeoutAdapter(self._ConcurrentAdapter):
            def complete_json(self, prompt, **kwargs):
                observed.append(kwargs["timeout"])
                return {"choices": [{"text": '{}'}]}

        adapter = TimeoutAdapter()
        token = start_model_call_budget(10.0, reserve_seconds=2.0)
        try:
            call_json_llm(
                adapter,
                "call-local timeout",
                max_tokens=16,
                temperature=0.0,
                schema={"type": "object"},
            )
        finally:
            reset_model_call_budget(token)

        self.assertEqual(adapter.timeout, 120.0)
        self.assertEqual(len(observed), 1)
        self.assertGreater(observed[0], 0.0)
        self.assertLessEqual(observed[0], 4.1)

    def test_actual_openai_adapter_uses_budget_timeout_without_mutating_itself(self):
        requests: list[dict] = []

        class Completions:
            def create(self, **request):
                requests.append(dict(request))
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content='{"ok":true}'),
                        finish_reason="stop",
                    )],
                    usage=None,
                )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )
        llm = OpenAIWorkspaceLLM("o3", timeout=120.0, client=client)
        token = start_model_call_budget(10.0, reserve_seconds=2.0)
        try:
            call_json_llm(
                llm,
                "return bounded json",
                max_tokens=16,
                temperature=0.0,
                schema={
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            )
        finally:
            reset_model_call_budget(token)

        self.assertEqual(llm.timeout, 120.0)
        self.assertEqual(len(requests), 1)
        self.assertGreater(requests[0]["timeout"], 0.0)
        self.assertLessEqual(requests[0]["timeout"], 4.1)

    def test_compact_evaluations_are_bounded_and_committed_in_configured_order(self):
        lock = threading.Lock()
        active = 0
        peak = 0
        commits: list[str] = []
        adapter = self._ConcurrentAdapter()

        class ConcurrentSpecialist(FixedSpecialist):
            def __init__(self, name: str):
                super().__init__(name, "protect", "CARE")
                self.llm = adapter

            def evaluate(self, scenario, actions, broadcast):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    time.sleep(0.03)
                    return super().evaluate(scenario, actions, broadcast)
                finally:
                    with lock:
                        active -= 1

        class OrderedCommitEngine(WorkspaceEngine):
            def commit_candidate(
                self, specialist, evaluation, cycle_input, **kwargs,
            ):
                commits.append(specialist.name)
                return super().commit_candidate(
                    specialist, evaluation, cycle_input, **kwargs,
                )

        names = ["first", "second", "third", "fourth"]
        OrderedCommitEngine(
            [ConcurrentSpecialist(name) for name in names],
            self._config(concurrency=2),
        ).run("Choose protect or decline.", ["protect", "decline"])

        self.assertEqual(peak, 2)
        self.assertEqual(commits, names)

    def test_local_or_unmarked_adapter_remains_sequential(self):
        lock = threading.Lock()
        active = 0
        peak = 0

        class LocalAdapter:
            pass

        class LocalSpecialist(FixedSpecialist):
            def __init__(self, name: str):
                super().__init__(name, "protect", "CARE")
                self.llm = LocalAdapter()

            def evaluate(self, scenario, actions, broadcast):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    time.sleep(0.01)
                    return super().evaluate(scenario, actions, broadcast)
                finally:
                    with lock:
                        active -= 1

        WorkspaceEngine(
            [LocalSpecialist("first"), LocalSpecialist("second")],
            self._config(concurrency=3),
        ).run("Choose protect or decline.", ["protect", "decline"])

        self.assertEqual(peak, 1)

    def test_terminal_compact_failure_cancels_pending_work_and_all_commits(self):
        adapter = self._ConcurrentAdapter()
        started: list[str] = []
        commits: list[str] = []
        launch_pair = threading.Barrier(2)

        class TerminalSpecialist(FixedSpecialist):
            def __init__(self, name: str, terminal: bool = False):
                super().__init__(name, "protect", "CARE")
                self.llm = adapter
                self.terminal = terminal

            def evaluate(self, scenario, actions, broadcast):
                started.append(self.name)
                launch_pair.wait(timeout=1.0)
                if self.terminal:
                    raise ModelCallUnavailable(
                        "authentication failed",
                        category="authentication",
                        terminal=True,
                    )
                time.sleep(0.05)
                return super().evaluate(scenario, actions, broadcast)

        class NoCommitEngine(WorkspaceEngine):
            def commit_candidate(
                self, specialist, evaluation, cycle_input, **kwargs,
            ):
                commits.append(specialist.name)
                return super().commit_candidate(
                    specialist, evaluation, cycle_input, **kwargs,
                )

        result = NoCommitEngine(
            [
                TerminalSpecialist("terminal", terminal=True),
                TerminalSpecialist("in_flight"),
                TerminalSpecialist("pending_one"),
                TerminalSpecialist("pending_two"),
            ],
            self._config(concurrency=2),
        ).run("Choose protect or decline.", ["protect", "decline"])

        self.assertEqual(result.halted_by, "model_backend_unavailable")
        self.assertEqual(commits, [])
        self.assertCountEqual(started, ["terminal", "in_flight"])

    def test_original_agent_fanout_is_bounded_and_order_preserving(self):
        import global_workspace.legacy_bridge as bridge

        lock = threading.Lock()
        active = 0
        peak = 0
        completion_order: list[str] = []
        agents = ("utilitarian", "deontological", "virtue", "care")

        def consult(agent, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep({
                    "utilitarian": 0.04,
                    "deontological": 0.01,
                    "virtue": 0.03,
                    "care": 0.01,
                }[agent])
                completion_order.append(agent)
                return bridge._OriginalAgentOutcome(
                    agent=agent,
                    testimony=f"{agent} testimony",
                    retrieval={"mode": "disabled"},
                )
            finally:
                with lock:
                    active -= 1

        with patch.object(bridge, "_consult_one_original_agent", side_effect=consult):
            result = consult_original_agents(
                Path("scenario.json"),
                agents=agents,
                backend="openai",
                max_concurrency=2,
            )

        self.assertEqual(peak, 2)
        self.assertNotEqual(completion_order, list(agents))
        self.assertEqual(list(result.testimonies), list(agents))
        self.assertEqual(
            list(result.retrievals), list(agents),
        )

    def test_original_agent_sequential_and_concurrent_payloads_are_exactly_equal(self):
        import global_workspace.legacy_bridge as bridge

        agents = ("utilitarian", "deontological", "virtue", "care")

        def consult(agent, **kwargs):
            time.sleep({
                "utilitarian": 0.02,
                "deontological": 0.005,
                "virtue": 0.015,
                "care": 0.001,
            }[agent])
            return bridge._OriginalAgentOutcome(
                agent=agent,
                testimony=f"scripted {agent} testimony",
                retrieval={
                    "mode": "disabled",
                    "evidence": [],
                    "expanded_query": f"{agent} query",
                },
            )

        with patch.object(bridge, "_consult_one_original_agent", side_effect=consult):
            sequential = consult_original_agents(
                Path("scenario.json"),
                agents=agents,
                backend="openai",
                max_concurrency=1,
            )
        with patch.object(bridge, "_consult_one_original_agent", side_effect=consult):
            concurrent = consult_original_agents(
                Path("scenario.json"),
                agents=agents,
                backend="openai",
                max_concurrency=3,
            )

        self.assertEqual(sequential.testimonies, concurrent.testimonies)
        self.assertEqual(sequential.errors, concurrent.errors)
        self.assertEqual(sequential.retrievals, concurrent.retrievals)
        self.assertEqual(list(concurrent.testimonies), list(agents))

    def test_terminal_original_agent_failure_cancels_remaining_fanout(self):
        import global_workspace.legacy_bridge as bridge

        agents = ("utilitarian", "deontological", "virtue", "care")
        started: list[str] = []
        launch_pair = threading.Barrier(2)

        def consult(agent, **kwargs):
            cancel_event = kwargs["cancel_event"]
            if cancel_event.is_set():
                return bridge._OriginalAgentOutcome(
                    agent=agent,
                    error="canceled after terminal provider failure",
                )
            started.append(agent)
            launch_pair.wait(timeout=1.0)
            if agent == "utilitarian":
                cancel_event.set()
                return bridge._OriginalAgentOutcome(
                    agent=agent,
                    error="OpenAI authentication failed",
                    terminal_category="authentication",
                )
            while not cancel_event.wait(0.01):
                pass
            return bridge._OriginalAgentOutcome(
                agent=agent,
                error="canceled after terminal provider failure",
            )

        with patch.object(bridge, "_consult_one_original_agent", side_effect=consult):
            result = consult_original_agents(
                Path("scenario.json"),
                agents=agents,
                backend="openai",
                max_concurrency=2,
            )

        self.assertCountEqual(started, ["utilitarian", "deontological"])
        self.assertEqual(result.testimonies, {})
        self.assertEqual(list(result.errors), list(agents))
        self.assertIn("authentication", result.errors["utilitarian"].casefold())

    def test_terminal_original_failure_stops_sequential_openai_launches(self):
        import global_workspace.legacy_bridge as bridge

        agents = ("utilitarian", "deontological", "virtue")
        started: list[str] = []

        def consult(agent, **kwargs):
            started.append(agent)
            return bridge._OriginalAgentOutcome(
                agent=agent,
                error="OpenAI quota prevented the model call",
                terminal_category="quota",
            )

        with patch.object(bridge, "_consult_one_original_agent", side_effect=consult):
            result = consult_original_agents(
                Path("scenario.json"),
                agents=agents,
                backend="openai",
                max_concurrency=1,
            )

        self.assertEqual(started, ["utilitarian"])
        self.assertEqual(list(result.errors), list(agents))
        self.assertIn("quota", result.errors["utilitarian"].casefold())
        self.assertIn("canceled", result.errors["deontological"].casefold())


class ConcurrentSemanticEquivalenceTests(unittest.TestCase):
    """The scheduler may change wall time, never deliberative semantics."""

    class ScriptedModel:
        supports_concurrent_calls = True
        supports_call_local_timeout = True
        timeout = 30.0

        def __init__(self, responses):
            self.responses = responses

        def complete_json(self, prompt, **kwargs):
            request = json.loads(prompt)
            specialist = request["specialist"]
            # Force completion order to differ in concurrent mode. Responses
            # remain functions of the request, never of call order.
            time.sleep(0.025 if specialist == "care" else 0.005)
            return {"choices": [{
                "text": json.dumps(self.responses[specialist], sort_keys=True),
            }]}

    class ScriptedSpecialist:
        def __init__(self, name, llm):
            self.name = name
            self.llm = llm
            # These are the same state surfaces used by compact specialists;
            # their presence activates framework vote-integrity admission.
            self.scenario_graph = None
            self.proposition_ledger = []
            self.canonical_action_records = []
            self.source_action_legend = {}
            self.previous_framework_state = {}
            self.private_framework_contribution = {}
            self.previous_recommendation_id = ""
            self.previous_confidence = None
            self.previous_context = ""
            self.assumption_status = "NOT_AUDITED"
            self.unsupported_assumption = ""
            self.reversal_condition = ""
            self.epistemic_commitments = []

        def evaluate(self, scenario, actions, broadcast):
            response = call_json_llm(
                self.llm,
                json.dumps({
                    "specialist": self.name,
                    "cycle": broadcast.problem_state.get("cycle", 0),
                }, sort_keys=True),
                max_tokens=256,
                temperature=0.0,
                schema={"type": "object"},
                call_kind="compact_primary",
                call_metadata={"specialist": self.name},
            )
            return CandidateChunk(**json.loads(response["choices"][0]["text"]))

    @staticmethod
    def _fixed_world():
        scenario = (
            "An emergency coordinator must choose one shelter route. "
            "Action A shelters the east residents. Action B shelters the west residents."
        )
        actions = (
            "Shelter the east residents",
            "Shelter the west residents",
        )
        ref = (SourceRef("C0", "An emergency coordinator must choose one shelter route."),)
        east_ref = (SourceRef("C1", "Action A shelters the east residents."),)
        west_ref = (SourceRef("C2", "Action B shelters the west residents."),)
        effects = (
            WorldEffect(
                "E0", "A0", "P1", "east residents receive shelter",
                "RECEIVES", "BENEFICIAL", "DIRECT", "CERTAIN",
                "RESOURCE_TRANSFER", provenance=east_ref,
            ),
            WorldEffect(
                "E1", "A1", "P2", "west residents receive shelter",
                "RECEIVES", "BENEFICIAL", "DIRECT", "CERTAIN",
                "RESOURCE_TRANSFER", provenance=west_ref,
            ),
        )
        world = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "emergency coordinator", "PERSON", ref),
                WorldParty("P1", "east residents", "GROUP", ref),
                WorldParty("P2", "west residents", "GROUP", ref),
            ),
            actions=(
                WorldAction("A0", actions[0], "P0", ("P1",), ("E0",), east_ref),
                WorldAction("A1", actions[1], "P0", ("P2",), ("E1",), west_ref),
            ),
            effects=effects,
            admission=WorldStateAdmission(
                status="COMMITTED", admitted_effect_ids=("E0", "E1"),
            ),
        )
        records = [{
            "action_id": f"A{index}",
            "canonical_semantic_action": action,
            "world_effects": [effects[index].as_dict()],
        } for index, action in enumerate(actions)]
        grounding = {
            "status": "COMMITTED",
            "world_model_status": "COMMITTED",
            "world_contradictions": [],
            "actions": {"A0": {}, "A1": {}},
            "clauses": [
                {"clause_id": "C0", "text": ref[0].excerpt},
                {"clause_id": "C1", "text": east_ref[0].excerpt},
                {"clause_id": "C2", "text": west_ref[0].excerpt},
            ],
            "world_model": world.as_dict(),
        }
        return scenario, actions, records, grounding

    @staticmethod
    def _responses(actions):
        derivation = [{
            "claim": "Compare the two admitted shelter effects",
            "proposition_id": "PROP:FRAMEWORK:SHELTER_COMPARISON",
            "declared_basis": "FRAMEWORK_DERIVED",
            "decision_critical": True,
            "scope_action_id": "COMPARISON",
            "source_effect_ids": ["E0", "E1"],
            "derivation_operation": "QUALITATIVE_COMPARISON",
            "calculation": "compare admitted shelter responsiveness",
            "assumptions": [],
            "outcome_type_transformation": "PRESERVED",
        }]
        common = {
            "action_scores": {actions[0]: 0.8, actions[1]: 0.2},
            "surprise": 0.4,
            "friction": 0.5,
            "confidence": 0.82,
            "recommended_action": actions[0],
            "adjudication_status": "SUPPORTS",
            "governing_eligible": True,
            "broadcast_authority": "GOVERNING_CANDIDATE",
            "comparison_complete": True,
            "evidence_sufficient_for_action": True,
            "material_empirical_claims": derivation,
        }
        return {
            "care": {
                **common,
                "specialist": "care",
                "constraint": "CARE",
                "rationale": "Acute dependency makes eastward shelter the responsive choice.",
                "decision_rule": "Prefer the action that answers the established acute dependency.",
                "factual_reversal_threshold": (
                    "West residents have the only immediate dependency"
                ),
                "care_ledger_proposal": {
                    "ranking_basis": "ACUTE_DEPENDENCY",
                    "assessments": [
                        {
                            "action_id": "A0", "verdict": "RESPONSIVE",
                            "affected_party": "east residents",
                            "relationship_type": "DEPENDENCY",
                            "dependency_source": "east residents receive shelter",
                            "responsibility_basis": "coordinator allocates emergency shelter",
                            "need_kind": "BASIC_NEED", "need_urgency": "IMMEDIATE",
                            "trust_effect": "PRESERVES", "responsiveness": "DIRECT",
                            "feasibility": "ESTABLISHED",
                            "competing_care_claim": "west residents also need shelter",
                            "resolution_status": "RESOLVED",
                            "evidence_basis": "ACTION_GRAPH",
                            "reason": "directly responds to the admitted east shelter need",
                        },
                        {
                            "action_id": "A1", "verdict": "MIXED",
                            "affected_party": "west residents",
                            "relationship_type": "COMMUNITY_RELATION",
                            "dependency_source": "west residents receive shelter",
                            "responsibility_basis": "coordinator also serves west residents",
                            "need_kind": "BASIC_NEED", "need_urgency": "NEAR_TERM",
                            "trust_effect": "PRESERVES", "responsiveness": "DIRECT",
                            "feasibility": "ESTABLISHED",
                            "competing_care_claim": "east dependency is more urgent",
                            "resolution_status": "RESOLVED",
                            "evidence_basis": "ACTION_GRAPH",
                            "reason": "responds to west while leaving acute east need unmet",
                        },
                    ],
                },
            },
            "virtue": {
                **common,
                "specialist": "virtue",
                "constraint": "CHARACTER",
                "action_scores": {actions[0]: 0.72, actions[1]: 0.28},
                "rationale": "Practical wisdom attends first to the acute shelter need.",
                "decision_rule": "Exercise practical wisdom under the admitted circumstances.",
                "normative_reversal_threshold": (
                    "Role fidelity establishes equal priority for west residents"
                ),
                "virtue_character_proposal": {
                    "ranking_basis": "PRACTICAL_WISDOM",
                    "assessments": [
                        {
                            "action_id": "A0", "verdict": "EXEMPLIFIES",
                            "actor_role": "emergency coordinator",
                            "virtues": "practical wisdom and responsiveness",
                            "vice_risk": "partiality toward one district",
                            "circumstance": "east residents receive shelter",
                            "evidence_basis": "ACTION_GRAPH",
                            "reason": "fits the urgent circumstance with prudent action",
                        },
                        {
                            "action_id": "A1", "verdict": "MIXED",
                            "actor_role": "emergency coordinator",
                            "virtues": "fair attention to west residents",
                            "vice_risk": "neglect of the more urgent need",
                            "circumstance": "west residents receive shelter",
                            "evidence_basis": "ACTION_GRAPH",
                            "reason": "shows care but does not best fit the urgency",
                        },
                    ],
                },
            },
        }

    def _run(self, concurrency):
        scenario, actions, records, grounding = self._fixed_world()
        model = self.ScriptedModel(self._responses(actions))
        specialists = [
            self.ScriptedSpecialist(name, model) for name in ("care", "virtue")
        ]
        result = WorkspaceEngine(
            specialists,
            WorkspaceConfig(
                max_cycles=1,
                stable_cycles_required=1,
                min_valid_specialists=2,
                enable_consensus_audit=False,
                enable_problem_state_audit=False,
                enable_reversal_audit=False,
                enable_synthesis=False,
                enable_planning=False,
                enable_ev_dominance_breaker=False,
                openai_max_concurrency=concurrency,
            ),
        ).run(
            scenario,
            actions,
            source_action_legend={
                f"A{index}": action for index, action in enumerate(actions)
            },
            action_source_grounding=grounding,
            canonical_action_records=records,
        )
        return result

    def test_sequential_and_concurrent_runs_are_exactly_semantically_equal(self):
        sequential = self._run(1)
        concurrent = self._run(2)
        comparison = compare_semantic_results(sequential, concurrent)

        self.assertTrue(
            comparison.equivalent,
            f"semantic mismatches: {comparison.mismatched_sections}",
        )
        self.assertEqual(comparison.mismatched_sections, ())
        self.assertEqual(
            comparison.sequential.to_dict(), comparison.concurrent.to_dict(),
        )
        snapshot = semantic_run_snapshot(concurrent)
        self.assertEqual(
            [row["framework_vote_status"] for row in snapshot.vote_admission_decisions[0]],
            ["FULL", "FULL"],
        )
        self.assertEqual(
            [item.specialist for item in concurrent.cycles[0].candidates],
            ["care", "virtue"],
        )
        self.assertEqual(
            [item["operation"] for item in concurrent.graph_transactions],
            ["CARE_RELATIONSHIP_LEDGER", "VIRTUE_CHARACTER_LEDGER"],
        )
        self.assertTrue(snapshot.challenge_agenda[0]["next"])
        self.assertEqual(
            snapshot.final_report, render_public_judgment(concurrent),
        )

    def test_graph_is_serialized_once_after_commit_barrier_and_once_at_finalization(self):
        original = SemanticGraphStore.graph_dict
        calls = []

        def counted(store):
            calls.append(store.graph.revision)
            return original(store)

        with patch.object(SemanticGraphStore, "graph_dict", counted):
            self._run(2)

        self.assertEqual(len(calls), 2)
        self.assertLessEqual(calls[0], calls[1])


def typed_reversal_proposal(source: str, target: str) -> dict:
    """General valid switch used by engine-level reversal-audit tests."""
    return {
        "operation": "BOUNDARY",
        "from_action": source,
        "to_action": target,
        "clauses": [{
            "affected_action": source,
            "metric": "expected harm",
            "metric_valence": "ADVERSE",
            "comparator": "GT",
            "threshold": 10,
            "unit": "COUNT",
            "source_text": "expected harm exceeds ten",
        }],
    }


class WorkspaceEngineTests(unittest.TestCase):
    def test_rejected_framework_update_uses_last_valid_candidate_operatively(self):
        actions = ["preserve association", "mandate common schools"]
        accepted = CandidateChunk(
            specialist="virtue", constraint="CHARACTER",
            action_scores={actions[0]: 0.65, actions[1]: 0.35},
            surprise=0.2, friction=0.3, confidence=0.7,
            recommended_action=actions[0],
            rationale="Practical wisdom preserves plural forms of flourishing.",
            decision_rule="Prefer A0 when plural flourishing remains viable.",
        )
        retained: dict[str, CandidateChunk] = {}
        first = _operative_framework_candidates([accepted], retained, remember=True)
        self.assertEqual(first[0].recommended_action, actions[0])

        rejected = CandidateChunk(
            specialist="virtue", constraint="CHARACTER",
            action_scores={actions[0]: 0.05, actions[1]: 0.95},
            surprise=1.0, friction=0.9, confidence=0.9,
            recommended_action=actions[1],
            rationale="The broadcast says duty requires common schools.",
            decision_rule="Prefer A1 because duty governs.",
            framework_retention_status="UPDATE_REJECTED",
            framework_constraint_retained=True,
            framework_validation_errors=["unjustified character-state reversal"],
        )
        operative = _operative_framework_candidates([rejected], retained, remember=True)

        self.assertEqual(len(operative), 1)
        self.assertEqual(operative[0].recommended_action, actions[0])
        self.assertEqual(operative[0].action_scores, accepted.action_scores)
        self.assertEqual(operative[0].rationale, accepted.rationale)
        self.assertEqual(
            operative[0].framework_retention_status,
            "PRESERVED_AFTER_REJECTED_UPDATE",
        )
        self.assertNotEqual(operative[0].rationale, rejected.rationale)
        self.assertEqual(
            operative[0].committed_framework_state["recommended_action"],
            actions[0],
        )
        self.assertEqual(
            operative[0].proposed_framework_state["recommended_action"],
            actions[1],
        )
        first_state = _operative_framework_candidates(
            [rejected], {}, remember=True,
        )
        self.assertEqual(len(first_state), 1)
        self.assertEqual(
            first_state[0].framework_retention_status,
            "FIRST_STATE_REJECTED_NOT_OPERATIVE",
        )
        self.assertEqual(first_state[0].adjudication_status, "CONTESTED_NO_LEANING")
        self.assertEqual(first_state[0].policy_weight_factor, 0.0)
        self.assertEqual(
            first_state[0].proposed_framework_state["recommended_action"],
            actions[1],
        )

    def test_position_coverage_rejects_cross_action_effect_transfer(self):
        graph = SemanticGraph()
        graph.add_node(SemanticNode(
            "A0", "ACTION", "keep the crew at the clinic",
            attributes={"canonical_action_id": "A0"},
        ))
        graph.add_node(SemanticNode(
            "A1", "ACTION", "send the crew to the road",
            attributes={"canonical_action_id": "A1"},
        ))
        graph.add_node(SemanticNode(
            "A0:WORLD_EFFECT:E0_RISK", "CONSEQUENCE", "AT_MODERATE_RISK",
            attributes={
                "world_effect_id": "E0_RISK", "polarity": "ADVERSE",
                "directness": "DOWNSTREAM", "scenario_grounded": True,
            },
        ))
        graph.add_node(SemanticNode(
            "PARTY:P3", "TARGET", "site crews",
            attributes={"semantic_role": "AFFECTED_SUBJECT"},
        ))
        graph.add_node(SemanticNode(
            "PARTY:P1", "TARGET", "eastern residents",
            attributes={"semantic_role": "AFFECTED_SUBJECT"},
        ))
        graph.add_edge(SemanticEdge("A0", "HAS_CONSEQUENCE", "A0:WORLD_EFFECT:E0_RISK"))
        graph.add_edge(SemanticEdge("A0:WORLD_EFFECT:E0_RISK", "AFFECTS", "PARTY:P3"))
        candidate = CandidateChunk(
            specialist="rawlsian", constraint="FAIRNESS",
            action_scores={"keep the crew at the clinic": 0.4, "send the crew to the road": 0.6},
            surprise=0.2, friction=0.2, confidence=0.7,
            recommended_action="send the crew to the road",
            committed_native_ledger={
                "ledger_kind": "RAWLS_POSITION_LEDGER",
                "transaction_status": "COMMITTED",
                "records": [
                    {
                        "canonical_action_id": "A0",
                        "subject": "immobile patients",
                        "dimension": "BASIC_INTEREST_SECURITY",
                        "institutional_relation": "NATURAL_CONTINGENCY",
                        "effect": "MIXED",
                    },
                    {
                        "canonical_action_id": "A1",
                        "subject": "immobile patients",
                        "dimension": "BASIC_INTEREST_SECURITY",
                        "institutional_relation": "NATURAL_CONTINGENCY",
                        "effect": "MIXED",
                    },
                ],
            },
        )
        challenge = {
            "challenge_kind": "POSITION_COVERAGE",
            "question": (
                "Does the Rawlsian comparison remain complete after representing "
                "the materially affected position of eastern residents, site crews "
                "over the same relevant horizon?"
            ),
        }
        answer = (
            "Yes. Eastern residents and site crews are represented: crews face "
            "equal moderate risk under both plans."
        )
        verified, reason = _challenge_resolution_supported(
            candidate, challenge,
            {"answer": answer, "current_position_effect": "NO_CHANGE"},
            graph=graph,
        )
        self.assertFalse(verified)
        self.assertTrue(
            "omit named parties" in reason or "transfers an action-scoped" in reason,
            reason,
        )

    def test_rejected_normative_update_preserves_current_epistemic_audit(self):
        actions = ["provide relief", "defer relief"]
        accepted = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.8, actions[1]: 0.2},
            surprise=0.2, friction=0.6, confidence=0.8,
            recommended_action=actions[0], rationale="A rescue duty favors relief.",
            decision_rule="Prefer relief when the rescue duty applies.",
            supporting_proposition_ids=["PROP:WORLD:E1"],
            deontological_ledger_proposal={"assessments": [{"action_id": "A0"}]},
        )
        retained: dict[str, CandidateChunk] = {}
        _operative_framework_candidates([accepted], retained, remember=True)

        rejected = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.2, actions[1]: 0.8},
            surprise=0.8, friction=0.6, confidence=0.5,
            epistemic_confidence=0.5,
            recommended_action=actions[1], rationale="A revised duty favors delay.",
            decision_rule="Prefer delay under the revised duty.",
            framework_retention_status="UPDATE_REJECTED",
            framework_constraint_retained=True,
            framework_validation_errors=["unjustified duty-state transition"],
            supporting_proposition_ids=[
                "PROP:WORLD:E1", "PROP:HYPOTHESIS:RISK",
            ],
            decision_critical_proposition_ids=["PROP:HYPOTHESIS:RISK"],
            weakest_decision_critical_status="HYPOTHETICAL",
            decision_critical_dependency_claims=["delay creates irreversible harm"],
            material_empirical_claims=[{
                "claim": "delay creates irreversible harm",
                "proposition_id": "PROP:HYPOTHESIS:RISK",
                "decision_critical": True,
            }],
            side_premise_audit_status="FINDINGS",
            side_premise_audit_findings=[{
                "claim": "delay creates irreversible harm",
                "proposition_id": "PROP:HYPOTHESIS:RISK",
                "decision_critical": True,
            }],
            assumption_status="UNDERDETERMINED", unresolved="VERIFY_FACTS",
            selection_status="PROVISIONAL", comparison_complete=False,
            evidence_sufficient_for_action=False,
            framework_specific_open_questions=[
                "Would delayed relief create irreversible harm?",
            ],
            deontological_ledger_proposal={"assessments": [{"action_id": "A1"}]},
        )

        restored = _operative_framework_candidates(
            [rejected], retained, remember=True,
        )[0]

        self.assertEqual(restored.recommended_action, actions[0])
        self.assertEqual(
            restored.committed_framework_state["recommended_action"], actions[0],
        )
        self.assertEqual(
            restored.proposed_framework_state["recommended_action"], actions[1],
        )
        self.assertIn("PROP:HYPOTHESIS:RISK", restored.supporting_proposition_ids)
        self.assertIn(
            "PROP:HYPOTHESIS:RISK", restored.decision_critical_proposition_ids,
        )
        self.assertEqual(restored.side_premise_audit_status, "FINDINGS")
        self.assertEqual(restored.weakest_decision_critical_status, "HYPOTHETICAL")
        self.assertEqual(restored.selection_status, "PROVISIONAL")
        self.assertFalse(restored.comparison_complete)
        self.assertFalse(restored.evidence_sufficient_for_action)
        self.assertIn(
            "Would delayed relief create irreversible harm?",
            restored.framework_specific_open_questions,
        )
        self.assertEqual(
            restored.preserved_current_cycle_components,
            [
                "EPISTEMIC_DEPENDENCIES", "FRAMEWORK_DIAGNOSTICS",
                "OPEN_QUESTIONS", "CURRENT_CYCLE_MEASUREMENTS",
            ],
        )
        self.assertIn("deontological", retained)
        self.assertIn(
            "PROP:HYPOTHESIS:RISK",
            retained["deontological"].decision_critical_proposition_ids,
        )

    def test_first_state_grounding_warning_does_not_suppress_valid_agent(self):
        actions = ["preserve association", "mandate common schools"]
        warned = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.45, actions[1]: 0.55},
            surprise=0.2, friction=0.1, confidence=0.55,
            recommended_action=actions[1],
            rationale="The competing claims remain unresolved.",
            decision_rule="Keep authorization contested pending necessity.",
            framework_constraint_retained=False,
            framework_retention_status="COMMITTED_WITH_UNCERTAINTY",
            framework_grounding_penalty=0.35,
            framework_validation_errors=[
                "necessity lacks grounded less-restrictive-route evidence",
            ],
        )
        retained: dict[str, CandidateChunk] = {}

        operative = _operative_framework_candidates(
            [warned], retained, remember=True,
        )

        self.assertEqual(len(operative), 1)
        self.assertEqual(operative[0].specialist, "deontological")
        self.assertEqual(operative[0].recommended_action, actions[1])
        self.assertEqual(operative[0].framework_grounding_penalty, 0.35)
        self.assertEqual(
            operative[0].framework_retention_status,
            "FIRST_STATE_ADMITTED_WITH_WARNINGS",
        )
        self.assertIn("deontological", retained)

    def test_rejected_deontology_refinement_survives_as_nonvoting_insight(self):
        actions = ["permit private schools", "mandate public schools"]
        specialist = CompactLocalSpecialist("deontological", llm=None)
        specialist.previous_framework_state = {
            "adjudication_form": "KANTIAN_CLAIM_COERCION_RESOLUTION",
            "assessments": [{
                "action_id": action_id, "verdict": "CONFLICTED",
                "norm_kind": "UNIVERSAL_LAW", "relation": "CONFLICTS",
                "competing_norm": "parental autonomy",
                "competing_norm_kind": "AUTONOMY",
                "competing_relation": "CONFLICTS",
                "governing_norm": "UNRESOLVED", "priority_basis": "UNRESOLVED",
                "protected_party": "children",
                "competing_protected_party": "parents",
                "protected_standing": "EQUAL_JURIDICAL_STATUS",
                "competing_protected_standing": "AUTONOMY",
                "coercion_kind": "PUBLIC" if action_id == "A1" else "NONE",
                "authorization_status": "CONTESTED",
                "derivation": "UNRESOLVED", "resolution_status": "CONTESTED",
            } for action_id in ("A0", "A1")],
        }
        candidate = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.45, actions[1]: 0.55},
            surprise=0.1, friction=0.1, confidence=0.55,
            recommended_action=actions[1],
            rationale="Parental consent may specify the competing claim.",
            deontological_ledger_proposal={"assessments": [{
                **item,
                "competing_norm": "parental consent right",
                "competing_norm_kind": "RIGHT",
                "competing_protected_standing": "CONSENT",
                "evidence_basis": "FRAMEWORK_ONLY",
            } for item in specialist.previous_framework_state["assessments"]]},
        )

        specialist._audit_framework_state_change(candidate, WorkspaceBroadcast(
            constraint="CHARACTER",
        ))

        self.assertEqual(candidate.framework_retention_status, "UPDATE_REJECTED")
        self.assertTrue(candidate.framework_insights)
        self.assertEqual(candidate.framework_insights[0]["insight_kind"], "REFINEMENT")
        self.assertEqual(candidate.framework_insights[0]["relation_to_committed_state"], "ELABORATES")

        retained = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.45, actions[1]: 0.55},
            surprise=0.1, friction=0.1, confidence=0.55,
            recommended_action=actions[1], rationale="Prior contested judgment.",
        )
        operative = _operative_framework_candidates(
            [candidate], {"deontological": retained}, remember=True,
        )
        state = build_deliberative_problem_state(
            2, actions, operative, actions[1], operative[0],
        ).to_dict()

        self.assertEqual(operative[0].rationale, "Prior contested judgment.")
        self.assertGreaterEqual(len(state["framework_insights"]), 2)
        self.assertTrue(all(
            item["voting_effect"] == "NONE"
            for item in state["framework_insights"]
        ))
        self.assertTrue(any(
            "parental consent right" in item["proposition"]
            for item in state["framework_insights"]
        ))

        overclaim = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.1, actions[1]: 0.9},
            surprise=0.8, friction=0.8, confidence=0.9,
            recommended_action=actions[1],
            rationale="The conflict is now resolved.",
            deontological_ledger_proposal={"assessments": [{
                **item,
                "verdict": "REQUIRED",
                "relation": "SATISFIES",
                "governing_norm": "PRIMARY",
                "priority_basis": "RESPECT_PERSONS",
                "norm": "future juridical independence",
                "evidence_basis": "FRAMEWORK_ONLY",
            } for item in specialist.previous_framework_state["assessments"]]},
        )
        specialist._audit_framework_state_change(
            overclaim, WorkspaceBroadcast(constraint="CHARACTER"),
        )

        self.assertEqual(overclaim.framework_retention_status, "UPDATE_REJECTED")
        observations = [
            item for item in overclaim.framework_insights
            if item["insight_kind"] == "NORMATIVE_OBSERVATION"
        ]
        self.assertTrue(observations)
        self.assertTrue(all(
            item["qualification"] == "FAILED_ADJUDICATION_NOT_AUTHORITY"
            for item in observations
        ))
        restored_overclaim = _operative_framework_candidates(
            [overclaim], {"deontological": retained}, remember=True,
        )[0]
        self.assertEqual(restored_overclaim.action_scores, retained.action_scores)
        self.assertEqual(restored_overclaim.rationale, retained.rationale)
        self.assertTrue(restored_overclaim.framework_insights)

    def test_framework_positions_do_not_manufacture_shared_dimensions(self):
        states = _derive_action_dimensions(
            rawlsian_positions=[{
                "canonical_action_id": "A0",
                "affected_subject": "parents",
                "dimension": "BASIC_LIBERTY",
                "effect": "WORSENS",
                "epistemic_status": "FRAMEWORK_INTERPRETATION",
                "assessment_node_id": "RAWLS_POSITION:test",
            }],
            utilitarian_consequences=[],
            action_ids=["A0"],
            grounded_action_effects=[],
        )

        self.assertEqual(states, [])

    def test_deontological_adjudication_form_is_terrain_neutral(self):
        actions = ["honor the refusal", "impose treatment"]
        graph = compile_scenario_graph(
            "A clinician must either honor a patient's refusal or impose treatment on the patient.",
            actions,
        )
        proposal = {"assessments": [
            {
                "action_id": "A0", "verdict": "PERMISSIBLE",
                "norm_kind": "AUTONOMY", "norm": "respect competent refusal",
                "relation": "CONSISTENT", "duty_bearer": "clinician",
                "protected_party": "patient", "competing_norm": "duty of care",
                "competing_norm_kind": "DUTY", "competing_relation": "CONFLICTS",
                "competing_protected_party": "patient",
                "competing_reason": "care remains an opposing obligation",
                "duty_type": "RIGHT_CORRELATIVE",
                "harm_relation": "NOT_APPLICABLE",
                "special_obligation_status": "NOT_REQUIRED",
                "special_obligation_basis": "consent is a general right-correlative constraint",
                "means_relation": "NO_INSTRUMENTALIZATION",
                "governing_norm": "PRIMARY", "priority_basis": "AUTONOMY",
                "priority_rule": "competent refusal governs treatment",
                "protected_standing": "CONSENT",
                "competing_protected_standing": "SPECIAL_OBLIGATION",
                "coercion_kind": "NONE", "coercive_actor": "NONE",
                "coerced_party": "NONE",
                "public_justification": "no coercion requires authorization",
                "reciprocity_status": "SATISFIED", "necessity_status": "UNKNOWN",
                "authorization_status": "NOT_APPLICABLE",
                "derivation": "CONSENT", "resolution_status": "RESOLVED",
                "evidence_basis": "SCENARIO", "reason": "honors the refusal",
            },
            {
                "action_id": "A1", "verdict": "CONFLICTED",
                "norm_kind": "AUTONOMY", "norm": "respect competent refusal",
                "relation": "CONFLICTS", "duty_bearer": "clinician",
                "protected_party": "patient", "competing_norm": "duty of care",
                "competing_norm_kind": "DUTY", "competing_relation": "CONFLICTS",
                "competing_protected_party": "patient",
                "competing_reason": "treatment may protect the patient",
                "governing_norm": "UNRESOLVED", "priority_basis": "UNRESOLVED",
                "priority_rule": "authorization remains unresolved",
                "protected_standing": "CONSENT",
                "competing_protected_standing": "SPECIAL_OBLIGATION",
                "coercion_kind": "INTERPERSONAL", "coercive_actor": "clinician",
                "coerced_party": "patient",
                "public_justification": "whether care authorizes overriding refusal",
                "reciprocity_status": "CONTESTED", "necessity_status": "CONTESTED",
                "authorization_status": "CONTESTED",
                "derivation": "UNRESOLVED", "resolution_status": "CONTESTED",
                "evidence_basis": "SCENARIO", "reason": "coercion remains contested",
            },
        ]}
        store = SemanticGraphStore(graph)
        record = apply_deontological_ledger_transaction(
            store, proposal, cycle=1, specialist="deontological",
            allowed_actions=tuple(actions),
        )

        self.assertIn(record.status, {"COMMITTED", "COMMITTED_WITH_UNCERTAINTY"})
        committed = record.proposal["committed_assessments"]
        self.assertEqual(committed[0]["derivation"], "CONSENT")
        self.assertEqual(committed[1]["authorization_status"], "CONTESTED")
        self.assertNotIn("education", json.dumps(committed).casefold())
        projected = committed_deontological_assessments(store.graph)
        self.assertEqual(projected[0]["protected_party"], "patient")
        self.assertEqual(projected[0]["norm"], "respect competent refusal")
        self.assertEqual(projected[0]["duty_bearer"], "clinician")

    def test_deontological_coercion_terrain_rejects_foreign_entities(self):
        actions = ["permit private schools", "mandate public schools"]
        graph = compile_scenario_graph(
            "The state may permit private schools or mandate public schools for parents and children.",
            actions,
        )
        base = {
            "verdict": "CONFLICTED", "norm_kind": "AUTONOMY",
            "norm": "respect external freedom", "relation": "CONFLICTS",
            "duty_bearer": "state", "protected_party": "parents",
            "competing_norm": "protect children's independence",
            "competing_norm_kind": "DUTY", "competing_relation": "CONFLICTS",
            "competing_protected_party": "children",
            "competing_reason": "both claims remain live",
            "governing_norm": "UNRESOLVED", "priority_basis": "UNRESOLVED",
            "priority_rule": "rightful coercion remains unresolved",
            "protected_standing": "EXTERNAL_FREEDOM",
            "competing_protected_standing": "EQUAL_JURIDICAL_STATUS",
            "coercion_kind": "PUBLIC", "coercive_actor": "volunteer coordinator",
            "coerced_party": "resident", "public_justification": "unresolved",
            "reciprocity_status": "UNKNOWN", "necessity_status": "UNKNOWN",
            "authorization_status": "UNKNOWN", "derivation": "UNRESOLVED",
            "resolution_status": "CONTESTED", "evidence_basis": "SCENARIO",
            "reason": "the claims conflict",
        }
        proposal = {"assessments": [
            {"action_id": "A0", **base}, {"action_id": "A1", **base},
        ]}
        store = SemanticGraphStore(graph)
        record = apply_deontological_ledger_transaction(
            store, proposal, cycle=1, specialist="deontological",
            allowed_actions=tuple(actions),
        )

        self.assertEqual(record.status, "COMMITTED_WITH_UNCERTAINTY")
        self.assertTrue(any(
            "coercion terrain lacks current-run actor/party grounding" in error
            for error in record.errors
        ))

    def test_deontological_ledger_requires_kantian_bridge_for_rawlsian_term(self):
        actions = ["act", "decline"]
        graph = compile_scenario_graph(
            "A decision maker must act for residents or decline.", actions,
        )
        assessment = {
            "verdict": "REQUIRED", "norm_kind": "DUTY",
            "norm": "secure fair equality of opportunity", "relation": "SATISFIES",
            "duty_bearer": "decision maker", "protected_party": "residents",
            "competing_norm": "administrative convenience",
            "competing_norm_kind": "OTHER", "competing_relation": "CONFLICTS",
            "competing_protected_party": "decision maker",
            "competing_reason": "convenience opposes the duty",
            "governing_norm": "PRIMARY", "priority_basis": "PERFECT_DUTY",
            "priority_rule": "fair equality of opportunity is a perfect duty",
            "protected_standing": "OTHER", "competing_protected_standing": "OTHER",
            "coercion_kind": "NONE", "coercive_actor": "NONE", "coerced_party": "NONE",
            "public_justification": "justice overrides convenience",
            "reciprocity_status": "UNKNOWN", "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE", "derivation": "PERFECT_DUTY",
            "resolution_status": "RESOLVED", "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "justice requires action",
        }
        store = SemanticGraphStore(graph)
        record = apply_deontological_ledger_transaction(
            store,
            {"assessments": [
                {"action_id": "A0", **assessment},
                {"action_id": "A1", **assessment},
            ]},
            cycle=1, specialist="deontological", allowed_actions=tuple(actions),
        )

        self.assertEqual(record.status, "REJECTED")
        self.assertTrue(any(
            "CROSS_FRAMEWORK_CONCEPT_UNTRANSLATED" in error
            for error in record.errors
        ))

    def test_adjudication_calibration_downgrades_unsupported_public_monopoly(self):
        actions = [
            "Constitutionalize compulsory publicly funded public-only schooling",
            "Guarantee parental choice of private schools",
        ]
        graph = compile_scenario_graph(
            "The state may abolish private schooling and mandate public-only education, "
            "restricting parents but improving children's educational opportunity, or "
            "permit private schools, preserving parental choice but worsening opportunity.",
            actions,
        )
        action = next(
            node for node in graph.nodes.values()
            if node.kind == "ACTION"
            and node.attributes.get("canonical_action_id") == "A0"
        )
        proposed = DutyAssessmentProposal.model_validate({
            "action_id": "A0", "verdict": "REQUIRED",
            "norm_kind": "UNIVERSAL_LAW", "norm": "secure equal juridical standing",
            "relation": "SATISFIES", "duty_bearer": "state",
            "protected_party": "children", "competing_norm": "parental association",
            "competing_norm_kind": "AUTONOMY", "competing_relation": "CONFLICTS",
            "competing_protected_party": "parents",
            "competing_reason": "public-only schooling restricts parental freedom",
            "governing_norm": "PRIMARY", "priority_basis": "PERFECT_DUTY",
            "priority_rule": "justice overrides associative liberty",
            "protected_standing": "EQUAL_JURIDICAL_STATUS",
            "competing_protected_standing": "EXTERNAL_FREEDOM",
            "coercion_kind": "PUBLIC", "coercive_actor": "state",
            "coerced_party": "parents",
            "public_justification": "reciprocal law enabling equal freedom",
            "reciprocity_status": "SATISFIED", "necessity_status": "NECESSARY",
            "authorization_status": "JUSTIFIED", "derivation": "UNIVERSAL_LAW",
            "resolution_status": "RESOLVED", "evidence_basis": "ACTION_GRAPH",
            "reason": "justice overrides parental liberty",
        })

        calibration = calibrate_deontological_adjudication(graph, action, proposed)

        self.assertFalse(calibration.calibrated)
        self.assertEqual(calibration.assessment.verdict, "CONFLICTED")
        self.assertEqual(calibration.assessment.necessity_status, "CONTESTED")
        self.assertEqual(calibration.assessment.reciprocity_status, "CONTESTED")
        self.assertEqual(calibration.assessment.authorization_status, "CONTESTED")
        self.assertEqual(calibration.assessment.derivation, "UNRESOLVED")
        self.assertEqual(calibration.assessment.resolution_status, "CONTESTED")

    def test_deontological_means_claim_requires_intended_as_means(self):
        actions = ["redirect the service", "leave the service unchanged"]
        graph = compile_scenario_graph(
            "A decision maker may redirect the service from residents or leave it unchanged.",
            actions,
        )
        action = next(
            node for node in graph.nodes.values()
            if node.kind == "ACTION"
            and node.attributes.get("canonical_action_id") == "A0"
        )
        proposed = DutyAssessmentProposal.model_validate({
            "action_id": "A0", "verdict": "PROHIBITED",
            "norm_kind": "RESPECT_PERSONS", "norm": "do not use residents merely as means",
            "relation": "VIOLATES", "duty_bearer": "decision maker",
            "protected_party": "residents", "competing_norm": "provide the service elsewhere",
            "competing_norm_kind": "DUTY", "competing_relation": "SATISFIES",
            "competing_protected_party": "other residents",
            "competing_reason": "the redirected service could aid others",
            "duty_type": "PERFECT_NEGATIVE", "harm_relation": "DOING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "the negative duty applies generally",
            "means_relation": "FORESEEN_SIDE_EFFECT",
            "governing_norm": "PRIMARY", "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "respect for persons prohibits instrumentalization",
            "protected_standing": "AUTONOMY", "competing_protected_standing": "OTHER",
            "coercion_kind": "NONE", "coercive_actor": "NONE", "coerced_party": "NONE",
            "public_justification": "no coercion requires authorization",
            "reciprocity_status": "UNKNOWN", "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE", "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED", "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "redirecting the service instrumentalizes residents",
        })

        calibration = calibrate_deontological_adjudication(graph, action, proposed)

        self.assertEqual(calibration.assessment.verdict, "CONFLICTED")
        self.assertEqual(calibration.assessment.resolution_status, "CONTESTED")
        self.assertTrue(any(
            "intended-as-means" in error for error in calibration.errors
        ), calibration.errors)

    def _omission_duty_proposal(self, **overrides):
        payload = {
            "action_id": "A0", "verdict": "PERMISSIBLE",
            "norm_kind": "DUTY", "norm": "do not take a neighbor's property",
            "relation": "SATISFIES", "duty_bearer": "operator",
            "protected_party": "neighbor",
            "competing_norm": "protect the occupied storerooms",
            "competing_norm_kind": "DUTY", "competing_relation": "CONFLICTS",
            "competing_protected_party": "occupants",
            "competing_reason": "the continuing flood still threatens occupants",
            "duty_type": "PERFECT_NEGATIVE", "harm_relation": "ALLOWING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "the negative duty applies generally",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "governing_norm": "PRIMARY", "priority_basis": "PERFECT_DUTY",
            "priority_rule": "the negative duty remains the governing constraint",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "OTHER",
            "coercion_kind": "NONE", "coercive_actor": "NONE", "coerced_party": "NONE",
            "public_justification": "no coercion requires authorization",
            "reciprocity_status": "UNKNOWN", "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE", "derivation": "PERFECT_DUTY",
            "resolution_status": "RESOLVED", "evidence_basis": "SCENARIO",
            "reason": "refusing to take the pump keeps the neighbor's property intact",
        }
        payload.update(overrides)
        return DutyAssessmentProposal.model_validate(payload)

    def _omission_duty_graph(self):
        actions = [
            "refuse to take the neighbor's pump",
            "take the neighbor's pump to divert the floodwater",
        ]
        graph = compile_scenario_graph(
            "An operator may refuse to take a neighbor's pump, allowing floodwater "
            "to continue toward five occupied storerooms, or take the neighbor's "
            "pump to divert the water, damaging the neighbor's equipment.",
            actions,
        )
        action = next(
            node for node in graph.nodes.values()
            if node.kind == "ACTION"
            and node.attributes.get("canonical_action_id") == "A0"
        )
        return graph, action, actions

    def _omission_typed_world(self) -> ScenarioWorldModel:
        ref = (SourceRef(
            "C0",
            "An operator may refuse to take a neighbor's pump, allowing floodwater "
            "to continue toward five occupied storerooms, or take the neighbor's "
            "pump to divert the water, damaging the neighbor's equipment.",
        ),)
        return ScenarioWorldModel(
            parties=(
                WorldParty("P0", "operator", "HUMAN", ref),
                WorldParty("P1", "neighbor", "HUMAN", ref),
                WorldParty("P2", "occupants", "GROUP", ref),
            ),
            actions=(
                WorldAction(
                    "A0", "refuse to take the neighbor's pump", "P0", ("P1",),
                    ("E0", "E1", "E2"), ref,
                ),
                WorldAction(
                    "A1", "take the neighbor's pump", "P0", ("P1",),
                    ("E3", "E4"), ref,
                ),
            ),
            effects=(
                WorldEffect(
                    "E0", "A0", "P0", "refusal carried out", "PERFORMS",
                    "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref,
                ),
                WorldEffect(
                    "E1", "A0", "P1", "pump remains with neighbor", "PRESERVES",
                    "BENEFICIAL", "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref,
                ),
                WorldEffect(
                    "E2", "A0", "P2", "floodwater continues toward storerooms",
                    "EXPERIENCES", "ADVERSE", "DOWNSTREAM", "CERTAIN",
                    "PHYSICAL_STATE", provenance=ref,
                ),
                WorldEffect(
                    "E3", "A1", "P1", "equipment damaged", "EXPERIENCES",
                    "ADVERSE", "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref,
                ),
                WorldEffect(
                    "E4", "A1", "P2", "floodwater diverted", "EXPERIENCES",
                    "BENEFICIAL", "DOWNSTREAM", "CERTAIN", "PHYSICAL_STATE",
                    provenance=ref,
                ),
            ),
        )

    def _omission_typed_graph(self):
        graph, action, actions = self._omission_duty_graph()
        attach_typed_world_model(graph, self._omission_typed_world())
        a1 = next(
            node for node in graph.nodes.values()
            if node.kind == "ACTION"
            and node.attributes.get("canonical_action_id") == "A1"
        )
        return graph, action, a1, actions

    def test_satisfying_negative_duty_may_allow_other_harm(self):
        graph, action, _actions = self._omission_duty_graph()
        proposed = self._omission_duty_proposal()

        self.assertFalse(omission_classified_as_perfect_negative_violation(proposed))
        calibration = calibrate_deontological_adjudication(graph, action, proposed)

        self.assertTrue(calibration.calibrated, calibration.errors)
        self.assertFalse(any(
            "perfect negative-duty violation" in error
            for error in calibration.errors
        ))
        self.assertEqual(calibration.assessment.verdict, "PERMISSIBLE")
        self.assertEqual(calibration.assessment.resolution_status, "RESOLVED")

    def test_omission_violation_claim_still_requires_a_separate_basis(self):
        graph, action, _actions = self._omission_duty_graph()
        proposed = self._omission_duty_proposal(
            verdict="PROHIBITED", relation="VIOLATES",
            reason="refusing to take the pump violates the negative duty",
        )

        self.assertTrue(omission_classified_as_perfect_negative_violation(proposed))
        calibration = calibrate_deontological_adjudication(graph, action, proposed)

        self.assertFalse(calibration.calibrated)
        self.assertTrue(any(
            "perfect negative-duty violation" in error
            for error in calibration.errors
        ), calibration.errors)

    def test_coercion_of_the_right_holder_still_counts_as_infringement(self):
        graph, action, _actions = self._omission_duty_graph()
        proposed = self._omission_duty_proposal(
            coercion_kind="PRIVATE", coercive_actor="operator",
            coerced_party="neighbor", authorization_status="UNJUSTIFIED",
        )

        self.assertTrue(omission_classified_as_perfect_negative_violation(proposed))
        calibration = calibrate_deontological_adjudication(graph, action, proposed)
        self.assertTrue(any(
            "perfect negative-duty violation" in error
            for error in calibration.errors
        ), calibration.errors)

    def test_verified_omission_repair_supersedes_the_operative_issue(self):
        graph, _action, actions = self._omission_duty_graph()
        store = SemanticGraphStore(graph)
        violated = self._omission_duty_proposal(
            verdict="PROHIBITED", relation="VIOLATES",
            reason="refusing to take the pump violates the negative duty",
        )
        rival = self._omission_duty_proposal(
            action_id="A1", verdict="PROHIBITED", relation="VIOLATES",
            harm_relation="DOING_HARM",
            competing_relation="SATISFIES",
            reason="taking the pump uses the neighbor as a tool",
        )
        apply_deontological_ledger_transaction(
            store,
            {"assessments": [violated.model_dump(), rival.model_dump()]},
            cycle=1, specialist="deontological", allowed_actions=tuple(actions),
        )
        committed = committed_deontological_assessments(store.graph)
        a0 = next(item for item in committed if item.get("canonical_action_id") == "A0")
        self.assertTrue(any(
            issue.get("kind") == OMISSION_PERFECT_NEGATIVE_VIOLATION
            and issue.get("status") == "ACTIVE"
            for issue in a0.get("calibration_issues", [])
        ), a0.get("calibration_issues"))

        challenge = ArgumentChallenge(
            challenge_kind="DOING_ALLOWING_CLASSIFICATION",
            question=(
                "Does the classified omission under A0 violate a perfect negative "
                "duty, or does that conclusion require a separate right-correlative "
                "premise?"
            ),
            about_specialist="deontological",
            target_specialists=("deontological",),
            grounded_in=("PROP:WORLD:E0",),
            trigger_fields=("dp.*.dt", "dp.*.hr", "dp.*.pb"),
            grounding_status="PROPOSITION_GROUNDED",
            status="ASSIGNED",
        ).as_dict()
        repaired = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.7, actions[1]: 0.3},
            surprise=0.2, friction=0.4, confidence=0.8,
            recommended_action=actions[0],
            committed_native_ledger={
                "ledger_kind": "DEONTOLOGICAL_DUTY_LEDGER",
                "transaction_status": "COMMITTED",
                "records": [{
                    "canonical_action_id": "A0",
                    "duty_type": "PERFECT_NEGATIVE",
                    "harm_relation": "ALLOWING_HARM",
                    "relation": "SATISFIES",
                    "coercion_kind": "NONE",
                    "protected_party": "neighbor",
                    "coerced_party": "NONE",
                    "calibration_errors": [],
                }],
            },
            challenge_response={
                "issue_id": challenge["issue_id"],
                "disposition": "RESOLVED",
                "current_position_effect": "NO_CHANGE",
                "answer": (
                    "a separate right-correlative premise would be required; "
                    "the scenario supplies none"
                ),
            },
        )

        retained, _agenda = _advance_argument_challenge_agenda(
            previous_challenges=[challenge],
            generated_challenges=[],
            candidates=[repaired],
            next_cycle=3,
            next_constraint="DUTY",
            active_specialists=["deontological"],
        )
        verified = next(item for item in retained if item["issue_id"] == challenge["issue_id"])
        self.assertEqual(verified["status"], "RESOLVED")
        self.assertEqual(
            verified["last_response"]["verification_status"], "VERIFIED_RESOLVED",
        )

        self.assertTrue(supersede_calibration_issues(
            store.graph, kinds=(OMISSION_PERFECT_NEGATIVE_VIOLATION,),
        ))
        superseded = committed_deontological_assessments(store.graph)
        a0_after = next(
            item for item in superseded if item.get("canonical_action_id") == "A0"
        )
        self.assertTrue(any(
            issue.get("kind") == OMISSION_PERFECT_NEGATIVE_VIOLATION
            and issue.get("status") == "SUPERSEDED"
            for issue in a0_after.get("calibration_issues", [])
        ))
        self.assertFalse(any(
            "perfect negative-duty violation" in str(error)
            for error in a0_after.get("calibration_errors", [])
        ))

        problem_state = {
            "framework_specific_open_questions": [{
                "issue_key": "FRAMEWORK_ISSUE:omission",
                "issue_type": "OPEN_QUESTION",
                "question_key": "FRAMEWORK_ISSUE:omission",
                "source_specialist": "deontological",
                "question": (
                    "whether the omission violates a separate right-correlative "
                    "prohibition"
                ),
                "status": "UNRESOLVED",
                "source_type": "FRAMEWORK_ATTRIBUTED_OPEN_QUESTION",
            }],
            "workspace_contributions": [{
                "agent": "deontological",
                "unresolved": [
                    "whether the omission violates a separate right-correlative prohibition",
                ],
                "visible_retained_issue": (
                    "whether the omission violates a separate right-correlative prohibition"
                ),
                "retained_issue_visibility": "OMITTED",
                "preservation_transitions": [],
            }],
        }
        apply_verified_challenge_supersession(problem_state, retained)
        self.assertEqual(
            problem_state["framework_specific_open_questions"][0]["status"],
            "SUPERSEDED",
        )
        self.assertEqual(
            problem_state["workspace_contributions"][0]["unresolved"], [],
        )
        self.assertEqual(
            problem_state["workspace_contributions"][0]["retained_issue_visibility"],
            "NOT_APPLICABLE",
        )
        later = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.7, actions[1]: 0.3},
            surprise=0.2, friction=0.4, confidence=0.8,
            recommended_action=actions[0],
        )
        carried = build_deliberative_problem_state(
            3, actions, [later], actions[0], later, problem_state,
        ).to_dict()
        self.assertFalse(any(
            "right-correlative" in str(item.get("question", ""))
            and str(item.get("status", "")).upper() in {"UNRESOLVED", "OPEN", "SUSPENDED"}
            for item in carried.get("framework_specific_open_questions", [])
        ))

    def test_doing_harm_without_agent_caused_path_fails_calibration(self):
        graph, action, _a1, _actions = self._omission_typed_graph()
        proposed = self._omission_duty_proposal(
            verdict="PROHIBITED", relation="VIOLATES",
            harm_relation="DOING_HARM",
            protected_party="occupants",
            reason="refusing to take the pump does harm to the occupants",
        )

        self.assertFalse(omission_classified_as_perfect_negative_violation(proposed))
        self.assertIn(
            "agent-caused settled welfare harm",
            harm_relation_conflicts_with_graph(graph, proposed, action=action),
        )
        calibration = calibrate_deontological_adjudication(graph, action, proposed)
        self.assertFalse(calibration.calibrated)
        self.assertTrue(any(
            "agent-caused settled welfare harm" in error for error in calibration.errors
        ), calibration.errors)

    def test_doing_harm_with_direct_adverse_survives_calibration(self):
        graph, _a0, action, _actions = self._omission_typed_graph()
        proposed = self._omission_duty_proposal(
            action_id="A1", verdict="PROHIBITED", relation="VIOLATES",
            harm_relation="DOING_HARM",
            protected_party="neighbor",
            competing_relation="SATISFIES",
            reason="taking the pump damages the neighbor's equipment",
        )

        self.assertEqual(
            harm_relation_conflicts_with_graph(graph, proposed, action=action),
            "",
        )
        calibration = calibrate_deontological_adjudication(graph, action, proposed)
        self.assertFalse(any(
            "agent-caused settled welfare harm" in error for error in calibration.errors
        ), calibration.errors)

    def test_allowing_harm_is_inconsistent_with_direct_adverse(self):
        graph, _a0, action, _actions = self._omission_typed_graph()
        proposed = self._omission_duty_proposal(
            action_id="A1", verdict="PERMISSIBLE", relation="SATISFIES",
            harm_relation="ALLOWING_HARM",
            protected_party="neighbor",
            competing_relation="CONFLICTS",
            reason="taking the pump is classified as merely allowing harm",
        )

        calibration = calibrate_deontological_adjudication(graph, action, proposed)
        self.assertFalse(calibration.calibrated)
        self.assertTrue(any(
            "inconsistent with an agent-caused" in error
            for error in calibration.errors
        ), calibration.errors)

    def test_relabeling_omission_as_doing_still_raises_doing_allowing_challenge(self):
        graph, _action, _a1, actions = self._omission_typed_graph()
        ledger = seed_proposition_ledger(graph)
        candidate = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.2, actions[1]: 0.8},
            surprise=0.2, friction=0.4, confidence=0.8,
            recommended_action=actions[1],
            rationale="The omission is relabeled as doing harm.",
            decision_rule="A downstream harm is treated as a doing.",
            supporting_proposition_ids=list(ledger),
            deontological_ledger_proposal={"assessments": [{
                "action_id": "A0", "resolution_status": "RESOLVED",
                "norm": "do not kill occupants",
                "duty_type": "PERFECT_NEGATIVE",
                "relation": "VIOLATES",
                "harm_relation": "DOING_HARM",
                "coercion_kind": "NONE",
                "protected_party": "occupants",
                "coerced_party": "NONE",
                "special_obligation_status": "NOT_REQUIRED",
                "means_relation": "NO_INSTRUMENTALIZATION",
                "priority_rule": "the negative duty remains governing",
                "public_justification": "no coercion requires authorization",
                "reason": "leaving the flood is classified as doing harm",
            }]},
        )

        kinds = {
            item["challenge_kind"]
            for item in _argument_challenge_candidates(
                [candidate], ledger, graph, actions[0],
            )
        }
        self.assertIn("DOING_ALLOWING_CLASSIFICATION", kinds)

    def test_relabeling_omission_as_doing_does_not_verify_resolution(self):
        graph, _action, _a1, actions = self._omission_typed_graph()
        challenge = ArgumentChallenge(
            challenge_kind="DOING_ALLOWING_CLASSIFICATION",
            question=(
                "Does the classified omission under A0 violate a perfect negative "
                "duty, or does that conclusion require a separate right-correlative "
                "premise?"
            ),
            about_specialist="deontological",
            target_specialists=("deontological",),
            grounded_in=("PROP:WORLD:E0",),
            trigger_fields=("dp.*.dt", "dp.*.hr", "dp.*.pb"),
            grounding_status="PROPOSITION_GROUNDED",
            status="ASSIGNED",
        ).as_dict()
        relabeled = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.2, actions[1]: 0.8},
            surprise=0.2, friction=0.4, confidence=0.8,
            recommended_action=actions[1],
            committed_native_ledger={
                "ledger_kind": "DEONTOLOGICAL_DUTY_LEDGER",
                "transaction_status": "COMMITTED",
                "records": [{
                    "canonical_action_id": "A0",
                    "duty_type": "PERFECT_NEGATIVE",
                    "harm_relation": "DOING_HARM",
                    "relation": "VIOLATES",
                    "coercion_kind": "NONE",
                    "protected_party": "occupants",
                    "coerced_party": "NONE",
                    "calibration_errors": [],
                }],
            },
            challenge_response={
                "issue_id": challenge["issue_id"],
                "disposition": "RESOLVED",
                "current_position_effect": "NO_CHANGE",
                "answer": "the omission is now classified as doing harm",
            },
        )

        retained, _agenda = _advance_argument_challenge_agenda(
            previous_challenges=[challenge],
            generated_challenges=[],
            candidates=[relabeled],
            next_cycle=3,
            next_constraint="DUTY",
            active_specialists=["deontological"],
            graph=graph,
        )
        verified = next(item for item in retained if item["issue_id"] == challenge["issue_id"])
        self.assertEqual(
            verified["last_response"]["verification_status"], "RESOLUTION_REJECTED",
        )

    def test_calibrated_deontology_renderer_exposes_conflict_and_open_questions(self):
        rationale, rule, conflicts, questions = render_deontological_adjudication({
            "protected_party": "children",
            "norm": "future juridical independence",
            "competing_protected_party": "parents",
            "competing_norm": "external freedom of association",
            "coercive_actor": "state",
            "coerced_party": "parents",
            "resolution_status": "CONTESTED",
            "calibration_errors": [
                "necessity lacks grounded less-restrictive-route evidence",
                "reciprocity lacks a two-party compatible-freedom derivation",
            ],
        })

        self.assertIn("competing claims", rationale)
        self.assertIn("Kantian judgment remains contested", rationale)
        self.assertNotIn("justice overrides", rationale.casefold())
        self.assertIn("less restrictive route", rule)
        self.assertEqual(len(conflicts), 1)
        self.assertTrue(any("compatible external freedom" in item for item in questions))

    def test_calibrated_deontology_support_supersedes_direct_baseline(self):
        action = "mandate public schooling"
        candidate = {
            "specialist": "deontological",
            "rationale": "The Kantian judgment remains contested.",
            "landscape_decisive_axis": "parental freedom versus child independence",
        }
        data = {
            "actions": [action, "permit private schools"],
            "source_baselines": {
                "deontological": {
                    "status": "DIRECT", "action_id": "A0",
                    "reason": "Policy B is REQUIRED and Policy A is PROHIBITED.",
                },
            },
            "source_action_legend": {"A0": action, "A1": "permit private schools"},
            "authoritative_semantic_state": {
                "deontological_assessments": [{
                    "canonical_action_id": "A0",
                    "resolution_status": "CONTESTED",
                    "calibration_errors": ["necessity lacks grounding"],
                }],
            },
        }

        reason = _support_reason(data, candidate, action, data["actions"])

        self.assertNotIn("REQUIRED", reason)
        self.assertNotIn("PROHIBITED", reason)
        self.assertIn("Kantian judgment remains contested", reason)

    def test_noncoercive_authorization_mismatch_does_not_erase_duty_ledger(self):
        actions = ["mandate public schooling", "permit private schools"]
        common = {
            "k": "AUTONOMY", "n": "protect external freedom", "rel": "CONFLICTS",
            "b": "state", "p": "children", "cn": "parental association",
            "ck": "RIGHT", "crel": "CONFLICTS", "cp": "parents",
            "crs": "parental freedom remains a competing claim",
            "gv": "UNRESOLVED", "pb": "UNRESOLVED",
            "pr": "rightful coercion remains unresolved",
            "ps": "EQUAL_JURIDICAL_STATUS", "cps": "EXTERNAL_FREEDOM",
            "coa": "state", "cop": "parents", "pj": "unresolved",
            "rec": "UNKNOWN", "nec": "UNKNOWN", "dv": "UNRESOLVED",
            "res": "CONTESTED", "g": "ACTION_GRAPH",
            "rs": "children and parents retain competing claims",
        }
        data = {
            "scores": {"A0": 0.55, "A1": 0.45}, "r": "A0",
            "c": "DUTY", "u": "RESOLVE_NORMATIVE_TENSION",
            "w": "The competing claims remain unresolved.", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.6,
            "fm": {
                "A0": "CONFLICTED: public coercion requires justification",
                "A1": "CONFLICTED: private ordering threatens children's standing",
            },
            "dp": {
                "A0": {**common, "v": "CONFLICTED", "ki": "PUBLIC", "auth": "UNKNOWN"},
                # The model's UNKNOWN authorization is harmless here because it
                # has already said this action contains no coercion.
                "A1": {**common, "v": "CONFLICTED", "ki": "NONE", "auth": "UNKNOWN"},
            },
        }

        candidate = _candidate_from_data(
            "deontological", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        assessments = candidate.deontological_ledger_proposal["assessments"]
        self.assertEqual(len(assessments), 2)
        self.assertEqual(assessments[0]["authorization_status"], "UNKNOWN")
        self.assertEqual(assessments[1]["authorization_status"], "NOT_APPLICABLE")
        self.assertFalse(any(
            "non-coercive Deontological assessment" in error
            for error in candidate.framework_validation_errors
        ))

    def test_engine_clears_scenario_local_framework_state_between_runs(self):
        seen: list[dict[str, object]] = []

        class ReusedSpecialist:
            name = "deontological"
            scenario_graph = None
            previous_framework_state = {"party": "resident", "right": "privacy"}
            previous_recommendation_id = "A1"
            previous_confidence = 0.9
            previous_context = "prior privacy case"
            assumption_status = "SUPPORTED"
            unsupported_assumption = "third-party disclosure"
            reversal_condition = "resident authorizes disclosure"
            epistemic_commitments = ["resident privacy"]

            def evaluate(self, scenario, actions, broadcast):
                seen.append(dict(self.previous_framework_state))
                return CandidateChunk(
                    specialist=self.name, constraint="DUTY",
                    action_scores={actions[0]: 0.7, actions[1]: 0.3},
                    surprise=0.1, friction=0.4, confidence=0.7,
                    recommended_action=actions[0], rationale="respects current duty",
                )

        WorkspaceEngine(
            [ReusedSpecialist()],
            WorkspaceConfig(
                max_cycles=1, min_valid_specialists=1,
                enable_consensus_audit=False, enable_problem_state_audit=False,
                enable_reversal_audit=False, enable_synthesis=False,
                enable_planning=False,
            ),
        ).run("Choose the current action.", ["act", "decline"])

        self.assertEqual(seen, [{}])

    def test_framework_can_translate_audit_without_copying_its_ontology(self):
        audit_variable = {
            "entity": "autonomy loss versus welfare gain",
            "relation": "COMPARATIVE_MAGNITUDE",
            "possible_values": ["NO_CHANGE", "WEAKENS", "REVERSES", "UNRESOLVED"],
            "focus_action": "A0",
            "question": "Would resolving autonomy loss versus welfare gain change A0?",
        }
        candidate = _candidate_from_data(
            "care", ["universal schooling", "private enclaves"], {
                "scores": {"A0": 0.7, "A1": 0.3}, "r": "A0",
                "c": "CARE", "u": "NONE",
                "w": "dependency and autonomy remain connected", "j": "NONE",
                "ap": "TRANSLATED",
                "ax": "Care evaluates how both policies reshape dependency and relational vulnerability.",
                "ie": "WEAKENS",
            },
            WorkspaceBroadcast(
                constraint="CONSENSUS_AUDIT", audit_variable=audit_variable,
            ),
            "NONE", {},
        )

        self.assertTrue(candidate.schema_valid)
        self.assertEqual(candidate.audit_participation, "TRANSLATED")
        self.assertEqual(candidate.audit_variable["relation"], "COMPARATIVE_MAGNITUDE")
        self.assertEqual(candidate.audit_variable.get("category"), "VERIFY_FACTS")
        self.assertEqual(candidate.assumption_status, "SUPPORTED")

    def test_comparative_magnitude_question_is_mixed_uncertainty(self):
        actions = ["preserve liberty", "increase welfare"]
        util = CandidateChunk(
            specialist="utilitarian", constraint="UNCERTAINTY",
            action_scores={actions[0]: 0.51, actions[1]: 0.49},
            surprise=0.1, friction=0.2, confidence=0.4,
            recommended_action=actions[0], unresolved="VERIFY_FACTS",
            assumption_status="UNDERDETERMINED",
            unsupported_assumption="uncertain severity of autonomy loss versus welfare gain",
        )
        state = build_deliberative_problem_state(
            1, actions, [util], actions[0], util,
        ).to_dict()
        self.assertEqual(
            state["unresolved_questions"][0]["uncertainty_kind"],
            "MIXED_UNCERTAINTY",
        )

    def test_education_clause_projects_distinct_liberty_and_opportunity_effects(self):
        scenario = (
            "You are designing a foundational educational and child-welfare system. "
            "Policy A maximizes freedom of association and parental liberty, but "
            "systematically obliterates fair equality of opportunity for marginalized children. "
            "Policy B severely restricts parental choice and freedom of association, but "
            "guarantees structural equality of opportunity across socioeconomic classes."
        )
        actions = [
            "Implement Policy B mandating universal public education and banning private schooling",
            "Enact Policy A permitting private exclusive educational enclaves",
        ]
        clauses = segment_scenario_clauses(scenario)
        graph = compile_scenario_graph(scenario, actions, {
            "A0": {"clauses": [clauses[0], clauses[2]]},
            "A1": {"clauses": [clauses[0], clauses[1]]},
        })
        effects = project_grounded_action_effects(graph)
        facts = {(item.action_id, item.dimension, item.direction) for item in effects}

        self.assertIn(("A0", "LIBERTY_AUTONOMY", "WORSENS"), facts)
        self.assertIn(("A0", "OPPORTUNITY_ACCESS", "IMPROVES"), facts)
        self.assertIn(("A1", "LIBERTY_AUTONOMY", "IMPROVES"), facts)
        self.assertIn(("A1", "OPPORTUNITY_ACCESS", "WORSENS"), facts)
        self.assertFalse(any(item.source_clause_id == "C0" for item in effects))

    def test_deliberative_problem_state_is_attributed_and_complete(self):
        actions = ["preserve liberty", "raise the material floor"]
        rawls = CandidateChunk(
            specialist="rawlsian", constraint="FAIRNESS",
            action_scores={actions[0]: 0.8, actions[1]: 0.2},
            surprise=0.2, friction=0.6, confidence=0.7,
            recommended_action=actions[0],
            rationale="Basic liberty has lexical priority.",
            decision_rule="Preserve liberty before comparing material gains",
        )
        util = CandidateChunk(
            specialist="utilitarian", constraint="UNCERTAINTY",
            action_scores={actions[0]: 0.4, actions[1]: 0.6},
            surprise=0.3, friction=0.2, confidence=0.45,
            recommended_action=actions[1],
            rationale="Aggregate welfare comparison remains uncertain.",
            unresolved="VERIFY_FACTS",
            utilitarian_missing_comparison="welfare cost of liberty loss versus poverty relief",
        )

        state = build_deliberative_problem_state(
            1, actions, [rawls, util], actions[0], rawls,
        ).to_dict()

        self.assertEqual(
            set(state), {
                "cycle", "state_role", "live_actions", "agent_positions",
                "active_constraints", "active_conflicts", "unresolved_questions",
                "framework_internal_conflicts", "framework_specific_open_questions",
                "framework_insights",
                "workspace_contributions",
                "dissenting_positions", "current_plurality", "salient_position",
                "problem_delta", "support_composition", "surface_consensus",
                "deliberative_consensus", "framework_warnings",
                "audit_candidates", "resolved_questions",
                "unresolved_categories", "primary_unresolved",
                "proposals", "committed_world",
            },
        )
        rawls_contract = next(
            item for item in state["workspace_contributions"]
            if item["agent"] == "rawlsian"
        )
        self.assertEqual(rawls_contract["tendency"], actions[0])
        self.assertTrue(rawls_contract["core_ground"])
        self.assertIn("unresolved", rawls_contract)
        self.assertIn("defeat_conditions", rawls_contract)
        self.assertIn("new_considerations", rawls_contract)
        self.assertEqual(rawls_contract["voting_effect"], "NONE")
        self.assertEqual(
            state["salient_position"]["workspace_contribution"]["agent"],
            "rawlsian",
        )
        self.assertEqual(len(state["agent_positions"]), 2)
        self.assertEqual(state["dissenting_positions"][0]["specialist"], "utilitarian")
        self.assertTrue(all(
            item["source_type"] == "FRAMEWORK_ATTRIBUTED"
            for item in state["active_constraints"]
        ))
        self.assertEqual(
            state["salient_position"]["source_type"],
            "FRAMEWORK_ATTRIBUTED_POSITION",
        )
        self.assertIn(
            "Preserve liberty before comparing material gains",
            state["salient_position"]["conditional_justification"],
        )
        self.assertNotIn("rationale", state["agent_positions"][0])

    def test_broadcast_projection_gives_every_framework_a_balanced_semantic_capsule(self):
        actions = [
            "route the sole emergency resource to the immediate-care facility " * 4,
            "route the sole emergency resource to the public-infrastructure facility " * 4,
        ]
        specialists = []
        for index, name in enumerate(
            ["utilitarian", "deontological", "virtue", "care", "rawlsian"]
        ):
            preferred = actions[index % 2]
            rival = actions[1 - (index % 2)]
            specialists.append(CandidateChunk(
                specialist=name,
                constraint=f"FRAMEWORK_{index}",
                action_scores={preferred: 0.7, rival: 0.3},
                surprise=0.2, friction=0.4, confidence=0.7,
                recommended_action=preferred,
                rationale=f"{name} reason for the current lean",
                decision_rule=f"Prefer the selected action while {name} condition holds",
                framework_action_map={
                    preferred: f"SUPPORTS: {name} main case with its qualifier",
                    rival: f"MIXED: {name} strongest counterclaim",
                },
                framework_internal_conflicts=[f"{name} competing claim remains open"],
                framework_specific_open_questions=[f"what would reverse {name}?"],
                reversal_condition=f"reverse if the {name} counterclaim dominates",
                factual_reversal_threshold=f"reverse if the {name} fact changes",
                unsupported_assumption=f"{name} conditional dependency",
                assumption_status="CONDITIONAL",
                unresolved="DECISION_BOUNDARY",
            ))
        state = build_deliberative_problem_state(
            1, actions, specialists, actions[0], specialists[0],
        ).to_dict()
        complete_before = json.dumps(state, sort_keys=True)

        projection = _balanced_problem_state_projection(state)
        capsules = projection["framework_capsules"]
        delivered = WorkspaceBroadcast(problem_state=state).compact()
        delivered_projection = json.loads(delivered.split("problem_state=", 1)[1])

        self.assertEqual(
            [capsule["agent"] for capsule in capsules],
            ["care", "deontological", "rawlsian", "utilitarian", "virtue"],
        )
        required = {
            "current_lean", "reason_for_lean", "supporting_premises",
            "qualifiers", "conditional_dependencies", "defeaters",
            "counterclaims", "decision_boundaries", "open_questions",
            "source_type",
        }
        self.assertTrue(all(required <= set(capsule) for capsule in capsules))
        self.assertTrue(all(
            capsule["current_lean"] in {"A0", "A1"}
            and capsule["counterclaims"][0]["action_id"] in {"A0", "A1"}
            and capsule["source_type"] == "FRAMEWORK_ATTRIBUTED_PROJECTION"
            for capsule in capsules
        ))
        self.assertTrue(all(
            len(json.dumps(capsule)) < 2200 for capsule in capsules
        ))
        self.assertEqual(delivered_projection, projection)
        self.assertEqual(json.dumps(state, sort_keys=True), complete_before)

    def test_challenge_broadcast_projection_keeps_every_target_and_valid_json(self):
        challenges = [
            {
                "issue_id": f"CHALLENGE:{index:016d}",
                "generated_by": "WORKSPACE_ARGUMENT_AUDITOR",
                "about_specialist": name,
                "raised_by": [],
                "target_specialists": [name],
                "challenge_kind": "REVERSAL_BOUNDARY",
                "question": (f"What would reverse {name}'s current position? " * 20),
                "status": "ASSIGNED",
                "grounded_in": ["PROP:WORLD:E0"],
            }
            for index, name in enumerate(
                ["utilitarian", "deontological", "virtue", "care", "rawlsian"]
            )
        ]

        projection = _balanced_challenge_projection(challenges)
        serialized = json.dumps(projection, sort_keys=True)

        self.assertEqual(len(projection), 5)
        self.assertEqual(
            {item["target_specialists"][0] for item in projection},
            {"utilitarian", "deontological", "virtue", "care", "rawlsian"},
        )
        self.assertEqual(json.loads(serialized), projection)
        self.assertTrue(all(len(item["question"]) <= 180 for item in projection))

    def test_framework_native_payloads_preserve_each_committed_reasoning_geometry(self):
        actions = ["choose immediate protection", "choose systemic protection"]
        candidates = []
        committed = {
            "utilitarian": {
                "utilitarian_ledger_proposal": {"actions": [
                    {"action_id": "A0", "valuations": [{
                        "effect_id": "E0", "importance": "CRITICAL",
                        "reason": "immediate welfare effect",
                    }]},
                    {"action_id": "A1", "valuations": [{
                        "effect_id": "E1", "importance": "HIGH",
                        "reason": "systemic welfare effect",
                    }]},
                ]},
                "comparison_complete": False,
                "evidence_sufficient_for_action": False,
                "framework_numerical_role": "DECISIVE",
            },
            "deontological": {
                "deontological_ledger_proposal": {"assessments": [
                    {
                        "action_id": action_id, "verdict": verdict,
                        "norm_kind": "UNIVERSAL_LAW", "norm": "reciprocal rescue maxim",
                        "relation": relation, "duty_type": duty_type,
                        "duty_bearer": "public authority", "protected_party": party,
                        "harm_relation": "ALLOWING_HARM",
                        "special_obligation_status": "CONTESTED",
                        "special_obligation_basis": "entrustment remains disputed",
                        "means_relation": "FORESEEN_SIDE_EFFECT",
                        "competing_norm": "public health duty",
                        "competing_relation": "CONFLICTS",
                        "competing_protected_party": "city residents",
                        "priority_basis": "UNIVERSAL_LAW",
                        "priority_rule": "reciprocal rescue has provisional priority",
                        "derivation": "UNIVERSAL_LAW",
                        "resolution_status": "CONTESTED",
                    }
                    for action_id, verdict, relation, duty_type, party in (
                        ("A0", "REQUIRED", "SATISFIES", "RIGHT_CORRELATIVE", "patients"),
                        ("A1", "CONFLICTED", "CONFLICTS", "UNRESOLVED", "residents"),
                    )
                ]},
            },
            "virtue": {
                "virtue_character_proposal": {
                    "ranking_basis": "PRACTICAL_WISDOM",
                    "assessments": [{
                        "action_id": action_id, "verdict": verdict,
                        "actor_role": "public steward", "virtues": virtues,
                        "vice_risk": vice, "circumstance": "tragic resource scarcity",
                        "reason": "balances role and common flourishing",
                    } for action_id, verdict, virtues, vice in (
                        ("A0", "EXEMPLIFIES", "compassion and courage", "partiality"),
                        ("A1", "MIXED", "prudence and justice", "callousness"),
                    )],
                },
            },
            "care": {
                "care_ledger_proposal": {
                    "ranking_basis": "ENTRUSTED_RESPONSIBILITY",
                    "assessments": [{
                        "action_id": action_id, "verdict": verdict,
                        "affected_party": party, "relationship_type": relationship,
                        "dependency_source": dependency,
                        "responsibility_basis": "ongoing responsibility history",
                        "need_kind": "SURVIVAL_HEALTH", "need_urgency": "IMMEDIATE",
                        "trust_effect": trust, "responsiveness": responsiveness,
                        "feasibility": "ESTABLISHED",
                        "competing_care_claim": "the other party's dependency",
                        "resolution_status": "CONTESTED",
                    } for action_id, verdict, party, relationship, dependency, trust, responsiveness in (
                        ("A0", "RESPONSIVE", "patients", "ENTRUSTED", "clinical support", "PRESERVES", "DIRECT"),
                        ("A1", "MIXED", "residents", "COMMUNITY_RELATION", "public infrastructure", "STRAINS", "INDIRECT"),
                    )],
                },
            },
            "rawlsian": {
                "rawls_position_proposal": {
                    "ranking_basis": "BASIC_INTEREST_SECURITY",
                    "ranking_classification_justification": "survival security is the classified concern",
                    "lexical_priority_justification": "no direct basic-liberty restriction is established",
                    "liberty_status": {"A0": "NOT_APPLICABLE", "A1": "NOT_APPLICABLE"},
                    "positions": [{
                        "action_id": action_id, "subject": party,
                        "subject_kind": "GROUP", "dimension": "BASIC_INTEREST_SECURITY",
                        "additional_dimensions": ["OTHER_PRIMARY_GOOD"],
                        "basic_liberty_kind": "NOT_APPLICABLE",
                        "institutional_relation": relation,
                        "effect": effect, "compared_to_action_id": rival,
                        "principle_basis": "BASIC_INTEREST_SECURITY",
                        "reason": "compares representative positions under a public rule",
                    } for action_id, party, relation, effect, rival in (
                        ("A0", "patients", "NATURAL_CONTINGENCY", "IMPROVES", "A1"),
                        ("A1", "residents", "MATERIAL_PRECONDITION", "MIXED", "A0"),
                    )],
                },
            },
        }
        for index, name in enumerate(committed):
            candidate = CandidateChunk(
                specialist=name, constraint=name.upper(),
                action_scores={actions[0]: 0.7, actions[1]: 0.3},
                surprise=0.2, friction=0.4, confidence=0.7,
                recommended_action=actions[0], rationale=f"{name} leans A0",
                decision_rule=f"apply the {name} comparison",
                framework_action_map={
                    actions[0]: f"SUPPORTS: {name} case",
                    actions[1]: f"MIXED: {name} countercase",
                },
                framework_retention_status="COMMITTED",
            )
            proposal_key, collection, ledger_kind = {
                "utilitarian": (
                    "utilitarian_ledger_proposal", "actions",
                    "UTILITARIAN_CONSEQUENCE_LEDGER",
                ),
                "deontological": (
                    "deontological_ledger_proposal", "assessments",
                    "DEONTOLOGICAL_DUTY_LEDGER",
                ),
                "virtue": (
                    "virtue_character_proposal", "assessments",
                    "VIRTUE_CHARACTER_LEDGER",
                ),
                "care": (
                    "care_ledger_proposal", "assessments",
                    "CARE_RELATIONSHIP_LEDGER",
                ),
                "rawlsian": (
                    "rawls_position_proposal", "positions",
                    "RAWLSIAN_POSITION_LEDGER",
                ),
            }[name]
            proposal = committed[name][proposal_key]
            records = []
            if name == "utilitarian":
                for action in proposal[collection]:
                    records.extend({
                        "specialist": name,
                        "canonical_action_id": action["action_id"],
                        "world_effect_id": row["effect_id"],
                        "outcome": f"grounded outcome for {row['effect_id']}",
                        "direction": "BENEFICIAL",
                        "polarity": "BENEFICIAL",
                        "probability": "UNKNOWN",
                        "modality": "ASSERTED",
                        "magnitude": "UNKNOWN",
                        "scope": "affected residents",
                        "importance": row["importance"],
                        "valuation_reason": row["reason"],
                        "epistemic_status": "SCENARIO_GROUNDED",
                    } for row in action["valuations"])
            else:
                ranking_fields = {
                    key: value for key, value in proposal.items()
                    if key != collection
                }
                for row in proposal[collection]:
                    records.append({
                        **row,
                        **ranking_fields,
                        "specialist": name,
                        "canonical_action_id": row["action_id"],
                    })
            candidate.committed_native_ledger = {
                "ledger_kind": ledger_kind,
                "transaction_status": "COMMITTED",
                "records": records,
            }
            candidates.append(candidate)

        state = build_deliberative_problem_state(
            1, actions, candidates, actions[0], candidates[0],
        ).to_dict()
        projection = _balanced_problem_state_projection(state)
        native = {
            item["agent"]: item["framework_native_reasoning"]
            for item in projection["framework_capsules"]
        }

        self.assertEqual(
            native["utilitarian"]["schema_kind"], "UTILITARIAN_EFFECT_COMPARISON"
        )
        self.assertEqual(
            native["utilitarian"]["actions"][0]["effects"][0]["importance"],
            "CRITICAL",
        )
        self.assertEqual(
            native["utilitarian"]["actions"][0]["effects"][0]["direction"],
            "BENEFICIAL",
        )
        self.assertEqual(
            native["deontological"]["assessments"][0]["duty_type"],
            "RIGHT_CORRELATIVE",
        )
        self.assertEqual(
            native["deontological"]["assessments"][0]["universalization_status"],
            "SATISFIES",
        )
        self.assertEqual(
            native["virtue"]["assessments"][0]["actor_role"], "public steward"
        )
        self.assertEqual(
            native["care"]["assessments"][0]["relationship_type"], "ENTRUSTED"
        )
        self.assertEqual(
            native["rawlsian"]["positions"][0]["institutional_relation"],
            "NATURAL_CONTINGENCY",
        )
        for payload in native.values():
            action_rows = payload.get(
                "actions", payload.get("assessments", payload.get("positions", []))
            )
            self.assertEqual({row["action_id"] for row in action_rows}, {"A0", "A1"})
            self.assertEqual(
                payload["source_type"], "GRAPH_COMMITTED_FRAMEWORK_LEDGER"
            )

    def test_native_ledger_snapshot_is_created_only_after_graph_commit(self):
        candidate = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={"act": 1.0}, surprise=0.1, friction=0.1,
            confidence=0.8, recommended_action="act",
        )
        records = [{
            "specialist": "deontological",
            "canonical_action_id": "A0",
            "verdict": "CONFLICTED",
            "norm": "calibrated operative norm",
        }]
        _record_committed_native_ledger(
            candidate,
            ledger_kind="DEONTOLOGICAL_DUTY_LEDGER",
            transaction_status="REJECTED",
            records=records,
        )
        self.assertEqual(candidate.committed_native_ledger, {})

        _record_committed_native_ledger(
            candidate,
            ledger_kind="DEONTOLOGICAL_DUTY_LEDGER",
            transaction_status="COMMITTED_WITH_UNCERTAINTY",
            records=records,
        )
        self.assertEqual(
            candidate.committed_native_ledger["records"][0]["verdict"],
            "CONFLICTED",
        )
        records[0]["verdict"] = "REQUIRED"
        self.assertEqual(
            candidate.committed_native_ledger["records"][0]["verdict"],
            "CONFLICTED",
        )

    def test_native_payload_never_leaks_uncommitted_framework_proposal(self):
        actions = ["act", "decline"]
        candidate = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.7, actions[1]: 0.3},
            surprise=0.2, friction=0.4, confidence=0.7,
            recommended_action=actions[0],
            deontological_ledger_proposal={
                "assessments": [{"action_id": "A0", "norm": "rejected new norm"}]
            },
            framework_retention_status="PRESERVED_AFTER_REJECTED_UPDATE",
        )
        candidate.committed_native_ledger = {
            "ledger_kind": "DEONTOLOGICAL_DUTY_LEDGER",
            "transaction_status": "COMMITTED_WITH_UNCERTAINTY",
            "records": [{
                "specialist": "deontological",
                "canonical_action_id": "A0", "verdict": "CONFLICTED",
                "norm_kind": "DUTY", "norm": "operative prior norm",
                "relation": "CONFLICTS", "duty_type": "UNRESOLVED",
                "duty_bearer": "agent", "protected_party": "affected party",
                "harm_relation": "UNRESOLVED",
                "special_obligation_status": "UNKNOWN",
                "special_obligation_basis": "basis remains unresolved",
                "means_relation": "UNRESOLVED", "competing_norm": "rival duty",
                "competing_relation": "CONFLICTS",
                "competing_protected_party": "other party",
                "priority_basis": "UNRESOLVED", "priority_rule": "priority unresolved",
                "derivation": "UNRESOLVED", "resolution_status": "CONTESTED",
            }],
        }
        state = build_deliberative_problem_state(
            1, actions, [candidate], actions[0], candidate,
        ).to_dict()
        payload = state["workspace_contributions"][0]["native_reasoning"]
        serialized = json.dumps(payload)
        self.assertIn("operative prior norm", serialized)
        self.assertNotIn("rejected new norm", serialized)

        candidate.committed_native_ledger = {}
        state_without_commit = build_deliberative_problem_state(
            1, actions, [candidate], actions[0], candidate,
        ).to_dict()
        self.assertEqual(
            state_without_commit["workspace_contributions"][0]["native_reasoning"],
            {},
        )

    def test_native_payload_budget_preserves_structural_rows(self):
        payload = {
            "schema_kind": "RAWLSIAN_POSITION_ANALYSIS",
            "source_type": "GRAPH_COMMITTED_FRAMEWORK_LEDGER",
            "positions": [{
                "action_id": f"A{index}",
                "representative_subject": "representative constituency " * 20,
                "dimension": "BASIC_INTEREST_SECURITY",
                "institutional_relation": "DIRECT_BASIC_STRUCTURE_RULE",
                "comparative_effect": "MIXED",
                "public_reason": "public justification with extensive explanation " * 20,
            } for index in range(16)],
        }
        fitted = _fit_native_reasoning(payload)
        self.assertLessEqual(
            len(json.dumps(fitted, sort_keys=True)), _NATIVE_REASONING_CHAR_BUDGET
        )
        self.assertEqual(
            [item["action_id"] for item in fitted["positions"]],
            [f"A{index}" for index in range(16)],
        )

    def test_private_workspace_contribution_returns_only_to_its_specialist(self):
        seen: dict[str, list[dict[str, object]]] = {
            "deontological": [], "care": [],
        }

        class RecurrentSpecialist:
            scenario_graph = None
            previous_framework_state = {}
            private_framework_contribution = {}

            def __init__(self, name, constraint):
                self.name = name
                self.constraint = constraint

            def evaluate(self, scenario, actions, broadcast):
                seen[self.name].append(dict(self.private_framework_contribution))
                return CandidateChunk(
                    specialist=self.name, constraint=self.constraint,
                    action_scores={actions[0]: 0.7, actions[1]: 0.3},
                    surprise=0.2, friction=0.4, confidence=0.7,
                    recommended_action=actions[0],
                    rationale=f"{self.name} preserves its own ground",
                    decision_rule=f"prefer A0 under {self.constraint}",
                    framework_specific_open_questions=[
                        f"{self.name} unresolved question",
                    ],
                    normative_reversal_threshold=f"reverse if {self.name} ground fails",
                )

        result = WorkspaceEngine(
            [
                RecurrentSpecialist("deontological", "DUTY"),
                RecurrentSpecialist("care", "CARE"),
            ],
            WorkspaceConfig(
                max_cycles=2, stable_cycles_required=99,
                min_valid_specialists=2, enable_consensus_audit=False,
                enable_problem_state_audit=False, enable_reversal_audit=False,
                enable_synthesis=False, enable_planning=False,
                enable_ev_dominance_breaker=False,
            ),
        ).run("Choose one action.", ["act", "decline"])

        self.assertEqual(seen["deontological"][0], {})
        self.assertEqual(seen["care"][0], {})
        self.assertEqual(
            seen["deontological"][1]["agent"], "deontological",
        )
        self.assertEqual(seen["care"][1]["agent"], "care")
        self.assertNotEqual(
            seen["deontological"][1]["unresolved"],
            seen["care"][1]["unresolved"],
        )
        self.assertEqual(
            len(result.cycles[0].broadcast.problem_state["workspace_contributions"]),
            2,
        )

    def test_recurrence_suspends_omitted_private_structure_instead_of_losing_it(self):
        actions = ["act", "decline"]
        first = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.6, actions[1]: 0.4},
            surprise=0.2, friction=0.2, confidence=0.6,
            recommended_action=actions[0], rationale="Public duty remains contested",
            framework_internal_conflicts=["equal freedom versus imposed coercion"],
            framework_specific_open_questions=[
                "whether a less restrictive route secures equal freedom",
                "whether the rule is reciprocal",
            ],
        )
        state1 = build_deliberative_problem_state(
            1, actions, [first], actions[0], first,
        ).to_dict()
        recurrent = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.6, actions[1]: 0.4},
            surprise=0.2, friction=0.2, confidence=0.6,
            recommended_action=actions[0], rationale="Public duty remains contested",
            framework_specific_open_questions=["whether the rule is reciprocal"],
            framework_retention_status="PRESERVED",
        )
        state2 = build_deliberative_problem_state(
            2, actions, [recurrent], actions[0], recurrent, state1,
        ).to_dict()
        contribution = state2["workspace_contributions"][0]
        self.assertIn(
            "whether a less restrictive route secures equal freedom",
            contribution["unresolved"],
        )
        suspended = [
            item for item in contribution["preservation_transitions"]
            if item["status"] == "SUSPENDED"
        ]
        self.assertTrue(any(
            item["prior_item"] == "whether a less restrictive route secures equal freedom"
            for item in suspended
        ))
        self.assertTrue(any(
            item["question"] == "whether a less restrictive route secures equal freedom"
            and item["status"] == "SUSPENDED"
            for item in state2["framework_specific_open_questions"]
        ))
        self.assertTrue(any(
            item["conflict"] == "equal freedom versus imposed coercion"
            and item["status"] == "SUSPENDED"
            for item in state2["framework_internal_conflicts"]
        ))
        self.assertIn(
            contribution["retained_issue_visibility"],
            {"EXPLICIT", "PARAPHRASED", "STRUCTURED_ONLY"},
        )
        self.assertTrue(contribution["visible_retained_issue"])

    def test_broadcast_winner_receives_same_preservation_contract(self):
        actions = ["act", "decline"]
        winner = CandidateChunk(
            specialist="care", constraint="CARE",
            action_scores={actions[0]: 0.7, actions[1]: 0.3},
            surprise=0.2, friction=0.4, confidence=0.7,
            recommended_action=actions[0], rationale="Dependency concern is decisive",
            framework_specific_open_questions=["whether relief creates domination"],
        )
        state1 = build_deliberative_problem_state(
            1, actions, [winner], actions[0], winner,
        ).to_dict()
        compressed_winner = CandidateChunk(
            specialist="care", constraint="CARE",
            action_scores={actions[0]: 0.7, actions[1]: 0.3},
            surprise=0.2, friction=0.4, confidence=0.7,
            recommended_action=actions[0], rationale="Dependency concern is decisive",
            framework_retention_status="PRESERVED",
        )
        state2 = build_deliberative_problem_state(
            2, actions, [compressed_winner], actions[0], compressed_winner, state1,
        ).to_dict()
        contribution = state2["workspace_contributions"][0]
        self.assertIn("whether relief creates domination", contribution["unresolved"])
        self.assertIn(
            "SUSPENDED",
            {item["status"] for item in contribution["preservation_transitions"]},
        )
        self.assertEqual(contribution["retained_issue_visibility"], "OMITTED")
        self.assertEqual(
            contribution["visible_retained_issue"],
            "whether relief creates domination",
        )

    def test_completed_cycle_broadcasts_typed_deliberative_state(self):
        observed = []

        class StateAwareSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                observed.append((self.name, broadcast.problem_state))
                return super().evaluate(scenario, actions, broadcast)

        result = WorkspaceEngine(
            [
                StateAwareSpecialist("care", "protect", "CARE"),
                StateAwareSpecialist("duty", "protect", "DUTY"),
            ],
            WorkspaceConfig(
                max_cycles=2,
                stable_cycles_required=2,
                stop_redundant_consensus_cycles=False,
                enable_consensus_audit=False,
                enable_reversal_audit=False,
                enable_synthesis=False,
                enable_planning=False,
            ),
        ).run("Choose whether to protect.", ["protect", "decline"])

        second_cycle_states = [state for _name, state in observed[2:]]
        self.assertTrue(second_cycle_states)
        self.assertTrue(all(state["live_actions"] for state in second_cycle_states))
        self.assertTrue(all(state["agent_positions"] for state in second_cycle_states))
        self.assertTrue(all(
            state["salient_position"]["source_type"]
            == "FRAMEWORK_ATTRIBUTED_POSITION"
            for state in second_cycle_states
        ))
        self.assertEqual(
            result.deliberative_problem_state["current_plurality"], "protect",
        )

    def test_broadcast_influence_is_observed_without_changing_salience(self):
        class UpdatingSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                # Update only after a cycle has actually broadcast a position.
                # The opening cycle now receives a positionless problem frame,
                # so the presence of a problem state no longer marks recurrence.
                if broadcast.salient_specialist:
                    chunk.action_scores = {actions[0]: 0.72, actions[1]: 0.28}
                    chunk.preference_strength = 0.44
                    chunk.workspace_reasoning_effect = "NORMATIVE"
                    chunk.self_reported_broadcast_dependence = "MEDIUM"
                    chunk.framework_constraint_retained = True
                    chunk.framework_retention_status = "PRESERVED"
                return chunk

        result = WorkspaceEngine(
            [
                UpdatingSpecialist("care", "protect", "CARE"),
                UpdatingSpecialist("duty", "protect", "DUTY"),
            ],
            WorkspaceConfig(
                max_cycles=2,
                stable_cycles_required=3,
                stop_redundant_consensus_cycles=False,
                enable_consensus_audit=False,
                enable_reversal_audit=False,
                enable_synthesis=False,
                enable_planning=False,
            ),
        ).run("Choose whether to protect.", ["protect", "decline"])

        records = result.broadcast_influence_records
        self.assertEqual(len(records), 2)
        self.assertEqual({item["observation_lag"] for item in records}, {1})
        self.assertTrue(all(
            item["attribution_status"]
            == "OBSERVED_AFTER_EXPOSURE_NOT_CAUSALLY_ESTABLISHED"
            for item in records
        ))
        self.assertTrue(all(item["framework_retained"] for item in records))
        translated = [
            item for item in records
            if item["receiving_agent"] != item["source_agent"]
        ]
        self.assertEqual(translated[0]["framework_effect"], "TRANSLATED")
        self.assertEqual(translated[0]["influence_class"], "REFINEMENT")
        # Phase one is observational: the hand-authored salience calculation is
        # unchanged and contains no learned influence bonus.
        self.assertFalse(hasattr(result.cycles[1].winner, "influence_bonus"))

    def test_framework_capture_gates_observed_influence(self):
        actions = ["protect", "decline"]
        source = CandidateChunk(
            specialist="duty", constraint="DUTY",
            action_scores={"protect": 0.8, "decline": 0.2},
            surprise=0.2, friction=0.6, confidence=0.8,
            recommended_action="protect", rationale="Respect persons.",
        )
        first = build_deliberative_problem_state(
            1, actions, [source], "protect", source,
        ).to_dict()
        captured = CandidateChunk(
            specialist="rawlsian", constraint="WELFARE_MAXIMIZATION",
            action_scores={"protect": 0.9, "decline": 0.1},
            surprise=0.3, friction=0.7, confidence=0.7,
            recommended_action="protect",
            framework_constraint_retained=False,
            framework_retention_status="LOST",
            self_reported_broadcast_dependence="HIGH",
        )
        second = build_deliberative_problem_state(
            2, actions, [captured], "protect", captured, first,
        )
        record = observe_broadcast_influence(first, second, [captured])[0]
        self.assertEqual(record.framework_effect, "CAPTURED")
        self.assertEqual(record.influence_class, "FRAMEWORK_CAPTURE")

    def test_broadcast_influence_persistence_tracks_surviving_update(self):
        actions = ["protect", "decline"]
        source = CandidateChunk(
            specialist="duty", constraint="DUTY",
            action_scores={"protect": 0.8, "decline": 0.2},
            surprise=0.2, friction=0.6, confidence=0.8,
            recommended_action="protect", rationale="Respect persons.",
        )
        care_before = CandidateChunk(
            specialist="care", constraint="CARE",
            action_scores={"protect": 0.4, "decline": 0.6},
            surprise=0.2, friction=0.2, confidence=0.6,
            recommended_action="decline",
        )
        first = build_deliberative_problem_state(
            1, actions, [source, care_before], "protect", source,
        ).to_dict()
        care_after = CandidateChunk(
            specialist="care", constraint="CARE",
            action_scores={"protect": 0.65, "decline": 0.35},
            surprise=0.2, friction=0.3, confidence=0.65,
            recommended_action="protect",
            workspace_reasoning_effect="NORMATIVE",
            self_reported_broadcast_dependence="MEDIUM",
            framework_retention_status="PRESERVED",
        )
        second = build_deliberative_problem_state(
            2, actions, [source, care_after], "protect", care_after, first,
        )
        records = [
            item.to_dict()
            for item in observe_broadcast_influence(first, second, [source, care_after])
        ]
        third = build_deliberative_problem_state(
            3, actions, [source, care_after], "protect", care_after, second.to_dict(),
        )
        update_broadcast_influence_persistence(records, third)
        care_record = next(
            item for item in records if item["receiving_agent"] == "care"
        )
        self.assertEqual(care_record["framework_effect"], "TRANSLATED")
        self.assertEqual(care_record["persistence"], "PERSISTENT")

    def test_live_tension_engagement_is_bounded_and_integrity_gated(self):
        problem_state = {
            "unresolved_questions": [{
                "source_specialist": "care", "category": "VERIFY_FACTS",
                "question_key": "QUESTION:1",
            }],
            "active_conflicts": [{
                "conflict_type": "ACTION_PREFERENCE",
                "specialists": ["care", "duty"], "conflict_key": "CONFLICT:1",
            }],
        }
        engaged = CandidateChunk(
            specialist="care", constraint="CARE",
            action_scores={"protect": 0.7, "decline": 0.3},
            surprise=0.2, friction=0.3, confidence=0.7,
            recommended_action="protect", unresolved="NONE",
            workspace_reasoning_effect="FACTUAL",
            workspace_proposition_response="QUALIFY",
        )
        score = WorkspaceEngine._tension_engagement(engaged, problem_state)
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)
        self.assertEqual(
            set(engaged.tension_target_keys), {"QUESTION:1", "CONFLICT:1"},
        )

        captured = CandidateChunk(
            specialist="care", constraint="FOREIGN_FRAMEWORK",
            action_scores={"protect": 0.9, "decline": 0.1},
            surprise=0.9, friction=0.9, confidence=0.9,
            recommended_action="protect",
            workspace_reasoning_effect="BOTH",
            workspace_proposition_response="ACCEPT",
            framework_constraint_retained=False,
        )
        self.assertEqual(
            WorkspaceEngine._tension_engagement(captured, problem_state), 0.0,
        )

    def test_problem_delta_tracks_constraint_and_question_resolution(self):
        actions = ["protect", "decline"]
        care_first = CandidateChunk(
            specialist="care", constraint="CARE",
            action_scores={"protect": 0.7, "decline": 0.3},
            surprise=0.2, friction=0.4, confidence=0.55,
            recommended_action="protect", unresolved="VERIFY_FACTS",
            unsupported_assumption="whether support reaches dependent residents",
        )
        duty_first = CandidateChunk(
            specialist="duty", constraint="DUTY",
            action_scores={"protect": 0.3, "decline": 0.7},
            surprise=0.2, friction=0.4, confidence=0.7,
            recommended_action="decline",
        )
        first = build_deliberative_problem_state(
            1, actions, [care_first, duty_first], "protect", care_first,
        ).to_dict()

        care_second = CandidateChunk(
            specialist="care", constraint="DEPENDENCY",
            action_scores={"protect": 0.85, "decline": 0.15},
            surprise=0.2, friction=0.7, confidence=0.82,
            recommended_action="protect", unresolved="NONE",
            workspace_reasoning_effect="FACTUAL",
        )
        duty_second = CandidateChunk(
            specialist="duty", constraint="DUTY",
            action_scores={"protect": 0.75, "decline": 0.25},
            surprise=0.2, friction=0.5, confidence=0.78,
            recommended_action="protect",
        )
        second = build_deliberative_problem_state(
            2, actions, [care_second, duty_second], "protect", care_second, first,
        ).to_dict()
        delta = second["problem_delta"]

        self.assertEqual(delta["from_cycle"], 1)
        self.assertEqual(delta["to_cycle"], 2)
        self.assertTrue(any(
            change["specialist"] == "care"
            and change["previous_constraint"] == "CARE"
            and change["current_constraint"] == "DEPENDENCY"
            for change in delta["constraint_changes"]
        ))
        self.assertTrue(any(
            item["constraint"] == "DEPENDENCY" for item in delta["new_constraints"]
        ))
        self.assertTrue(any(
            item["constraint"] == "CARE" for item in delta["removed_constraints"]
        ))
        self.assertTrue(any(
            item["resolution_type"] == "RESOLVED_BY_SCENARIO_FACT"
            for item in delta["resolved_questions"]
        ))
        self.assertTrue(any(
            item["conflict_type"] == "ACTION_PREFERENCE"
            and item["resolution_type"] == "RESOLVED_BY_DELIBERATIVE_CONVERGENCE"
            for item in delta["resolved_conflicts"]
        ))

    def test_problem_state_tracks_framework_local_tensions_without_promoting_them(self):
        actions = ["common schools", "private schools"]
        rawls = CandidateChunk(
            specialist="rawlsian", constraint="FAIRNESS",
            action_scores={actions[0]: 0.6, actions[1]: 0.4},
            surprise=0.2, friction=0.7, confidence=0.6,
            recommended_action=actions[0],
            framework_internal_conflicts=[
                "basic liberty versus fair equality of opportunity",
            ],
            framework_specific_open_questions=[
                "status of parental educational association",
            ],
        )
        state = build_deliberative_problem_state(
            1, actions, [rawls], actions[0], rawls,
        ).to_dict()

        local_conflict = state["framework_internal_conflicts"][0]
        local_question = state["framework_specific_open_questions"][0]
        self.assertEqual(local_conflict["source_specialist"], "rawlsian")
        self.assertEqual(
            local_conflict["source_type"],
            "FRAMEWORK_ATTRIBUTED_INTERNAL_CONFLICT",
        )
        self.assertEqual(local_question["source_specialist"], "rawlsian")
        self.assertFalse(any(
            item.get("conflict_key") == local_conflict["conflict_key"]
            for item in state["active_conflicts"]
        ))
        self.assertFalse(any(
            item.get("question_key") == local_question["question_key"]
            for item in state["unresolved_questions"]
        ))

    def test_problem_delta_distinguishes_internal_conflict_reframing(self):
        actions = ["common schools", "private schools"]
        first_candidate = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.55, actions[1]: 0.45},
            surprise=0.2, friction=0.7, confidence=0.55,
            recommended_action=actions[0],
            framework_internal_conflicts=[
                "parental freedom versus the child's equal independence",
            ],
        )
        first = build_deliberative_problem_state(
            1, actions, [first_candidate], actions[0], first_candidate,
        ).to_dict()
        second_candidate = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.55, actions[1]: 0.45},
            surprise=0.2, friction=0.7, confidence=0.55,
            recommended_action=actions[0],
            framework_internal_conflicts=[
                "parental autonomy versus children's claims to equal freedom",
            ],
        )
        second = build_deliberative_problem_state(
            2, actions, [second_candidate], actions[0], second_candidate, first,
        ).to_dict()
        delta = second["problem_delta"]

        self.assertEqual(len(delta["reframed_internal_conflicts"]), 1)
        self.assertEqual(delta["new_internal_conflicts"], ())
        self.assertEqual(delta["resolved_internal_conflicts"], ())

        old_issue = first["framework_internal_conflicts"][0]
        new_issue = second["framework_internal_conflicts"][0]
        self.assertEqual(old_issue["issue_key"], new_issue["issue_key"])
        self.assertIn(
            old_issue["conflict"], new_issue["wording_history"],
        )

    def test_framework_issue_type_validation_reclassifies_question_from_conflict_field(self):
        actions = ["common schools", "private schools"]
        candidate = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.55, actions[1]: 0.45},
            surprise=0.2, friction=0.7, confidence=0.55,
            recommended_action=actions[0],
            framework_internal_conflicts=["whether less restrictive means exist"],
        )
        state = build_deliberative_problem_state(
            1, actions, [candidate], actions[0], candidate,
        ).to_dict()

        self.assertEqual(state["framework_internal_conflicts"], [])
        self.assertEqual(len(state["framework_specific_open_questions"]), 1)
        issue = state["framework_specific_open_questions"][0]
        self.assertEqual(issue["issue_type"], "OPEN_QUESTION")
        self.assertEqual(issue["question"], "whether less restrictive means exist")

    def test_equivalent_open_question_keeps_one_live_wording_and_prior_history(self):
        actions = ["common schools", "private schools"]
        first_candidate = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.55, actions[1]: 0.45},
            surprise=0.2, friction=0.7, confidence=0.55,
            recommended_action=actions[0],
            framework_specific_open_questions=[
                "whether a less restrictive route can secure equal freedom",
            ],
        )
        first = build_deliberative_problem_state(
            1, actions, [first_candidate], actions[0], first_candidate,
        ).to_dict()
        second_candidate = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.55, actions[1]: 0.45},
            surprise=0.2, friction=0.7, confidence=0.55,
            recommended_action=actions[0],
            framework_specific_open_questions=[
                "whether less restrictive means secure equal freedom",
            ],
        )
        second = build_deliberative_problem_state(
            2, actions, [second_candidate], actions[0], second_candidate, first,
        ).to_dict()

        self.assertEqual(len(second["framework_specific_open_questions"]), 1)
        old_issue = first["framework_specific_open_questions"][0]
        new_issue = second["framework_specific_open_questions"][0]
        self.assertEqual(old_issue["issue_key"], new_issue["issue_key"])
        self.assertEqual(
            new_issue["question"],
            "whether less restrictive means secure equal freedom",
        )
        self.assertIn(old_issue["question"], new_issue["wording_history"])

    def test_conditional_baseline_renderer_uses_provisional_action(self):
        baseline = {
            "status": "CONDITIONAL",
            "action_id": "NONE",
            "provisional_action_id": "A0",
        }

        self.assertEqual(_baseline_display_action(baseline), "A0")

    def test_single_normatively_contested_delegate_does_not_block_convergence(self):
        class ContestedCare(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.recommended_action = self.preferred
                chunk.assumption_status = "NORMATIVELY_CONTESTED"
                chunk.unresolved = "RESOLVE_NORMATIVE_TENSION"
                chunk.selection_status = "PROVISIONAL"
                return chunk

        result = WorkspaceEngine(
            [
                ContestedCare("care", "protect", "CARE"),
                FixedSpecialist("virtue", "protect", "CHARACTER"),
                FixedSpecialist("deontological", "protect", "DUTY"),
            ],
            WorkspaceConfig(
                max_cycles=2,
                stable_cycles_required=1,
                consensus_audit_min_signals=99,
                enable_reversal_audit=False,
                enable_synthesis=False,
            ),
        ).run("Choose protect or decline.", ["protect", "decline"])

        self.assertEqual(result.halted_by, "convergence")
        self.assertEqual(result.selected_action, "protect")
        self.assertIn("NORMATIVE_ADJUDICATION", result.reopen_conditions)

    def test_autonomy_audit_records_normative_burden_without_lowering_confidence(self):
        class AutonomyAwareSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.recommended_action = self.preferred
                return chunk

        autonomy = AutonomyAssessment(
            {"protect": "COVENANT_BREACH", "decline": "NONE"},
            {"protect": False, "decline": False},
            {"protect": "Breaking the promise to preserve throughput.", "decline": ""},
            "decline",
            surcharge_multiplier=0.5,
        )

        result = WorkspaceEngine(
            [AutonomyAwareSpecialist("care", "protect", "CARE")],
            WorkspaceConfig(max_cycles=1, enable_synthesis=False),
        ).run(
            "Choose protect or decline.",
            ["protect", "decline"],
            assess_autonomy=lambda scenario, actions: autonomy,
        )

        candidate = result.cycles[0].candidates[0]
        self.assertEqual(candidate.coercion_tag, "COVENANT_BREACH")
        self.assertEqual(candidate.coercion_surcharge, 0.5)
        self.assertAlmostEqual(candidate.epistemic_confidence, 0.9)

    def test_uniform_autonomy_burden_is_recorded_without_surcharge(self):
        class AutonomyAwareSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.recommended_action = self.preferred
                return chunk

        autonomy = AutonomyAssessment(
            {"policy b": "RIGHTS_INTRUSION", "policy a": "RIGHTS_INTRUSION"},
            {"policy b": False, "policy a": False},
            {"policy b": "Zoning constrains use.", "policy a": "Zoning constrains use."},
            surcharge_multiplier=0.70,
        )
        result = WorkspaceEngine(
            [AutonomyAwareSpecialist("rawlsian", "policy b", "FAIRNESS")],
            WorkspaceConfig(max_cycles=1, enable_synthesis=False),
        ).run(
            "Choose one zoning policy.", ["policy b", "policy a"],
            assess_autonomy=lambda scenario, actions: autonomy,
        )

        candidate = result.cycles[0].candidates[0]
        self.assertEqual(candidate.coercion_tag, "RIGHTS_INTRUSION")
        self.assertEqual(candidate.coercion_surcharge, 0.0)

    def test_conditional_or_underdetermined_stability_can_still_converge(self):
        class CautiousSpecialist(FixedSpecialist):
            def __init__(self, *args, status="CONDITIONAL", **kwargs):
                super().__init__(*args, **kwargs)
                self.status = status

            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.assumption_status = self.status
                chunk.selection_status = "PROVISIONAL"
                return chunk

        result = WorkspaceEngine(
            [
                CautiousSpecialist("care", "protect", "CARE", status="CONDITIONAL"),
                CautiousSpecialist("virtue", "protect", "CHARACTER", status="UNDERDETERMINED"),
                FixedSpecialist("deontological", "protect", "DUTY"),
            ],
            WorkspaceConfig(
                max_cycles=4,
                stable_cycles_required=2,
                stop_redundant_consensus_cycles=False,
                consensus_audit_min_signals=99,
                enable_reversal_audit=False,
                enable_synthesis=False,
            ),
        ).run("Choose protect or decline.", ["protect", "decline"])

        self.assertEqual(result.halted_by, "convergence")
        self.assertEqual(result.selected_action, "protect")
        self.assertEqual(result.judgment_status, "GOVERNED_RECOMMENDATION")
        self.assertGreaterEqual(result.cycles[-1].stable_cycles, 2)

    def test_transient_delegate_timeout_is_excluded_without_crashing_cycle(self):
        class TimedOutSpecialist:
            name = "deontological"

            def evaluate(self, scenario, actions, broadcast):
                raise ModelCallUnavailable(
                    "OpenAI model call timed out", category="timeout", terminal=False
                )

        result = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                TimedOutSpecialist(),
                FixedSpecialist("virtue", "protect", "CHARACTER"),
            ],
            WorkspaceConfig(max_cycles=1, enable_synthesis=False),
        ).run("Choose protect or decline.", ["protect", "decline"])

        self.assertEqual(result.selected_action, "protect")
        unavailable = result.cycles[0].candidates[1]
        self.assertFalse(unavailable.schema_valid)
        self.assertEqual(unavailable.constraint, "NONE")
        self.assertEqual(unavailable.delegate_status, "MODEL_ERROR")
        self.assertIn(
            "DELEGATE_MODEL_UNAVAILABLE",
            [finding.code for finding in result.trace_health],
        )

    def test_historical_dissent_that_supports_final_policy_is_not_residue(self):
        action_a, action_b = "choose A", "choose B"
        historical_supporter = CandidateChunk(
            "rawlsian", "FAIRNESS", {action_a: 0.2, action_b: 0.8},
            0.2, 0.6, 0.8, recommended_action=action_b,
        )
        cycle = CycleRecord(
            1, WorkspaceBroadcast(), [historical_supporter], historical_supporter,
            historical_supporter, {action_a: 0.6, action_b: 0.4}, 0.9, 1, 1.0,
        )
        self.assertEqual(collect_moral_residue([cycle], action_b), [])
        self.assertEqual(collect_typed_residue([cycle], action_b), [])

    def test_stance_completeness_is_relative_to_each_cycle_action_set(self):
        action_a, action_b, synthesis = "choose A", "choose B", "combine safeguards"
        candidate = CandidateChunk(
            "care", "CARE", {action_a: 0.7, action_b: 0.3}, 0.2, 0.5, 0.8,
            recommended_action=action_a,
            action_admissibility={action_a: "PERMISSIBLE", action_b: "REJECTED"},
        )
        cycle = CycleRecord(
            1, WorkspaceBroadcast(), [candidate], candidate, None,
            {action_a: 0.7, action_b: 0.3}, 0.8, 1, 1.0,
        )
        result = WorkspaceResult(
            "A test", [action_a, action_b, synthesis], cycles=[cycle]
        )
        findings = audit_trace_health(result)
        self.assertFalse(any(
            finding.code == "STANCE_MEASUREMENT_INCOMPLETE"
            for finding in findings
        ))

    def test_virtue_structural_incompleteness_is_distinct_from_retention_loss(self):
        candidate = CandidateChunk(
            "virtue", "CHARACTER", {"A0": 0.55, "A1": 0.45}, 0.2, 0.3, 0.7,
            recommended_action="A0",
            framework_retention_status="LOST",
            framework_validation_errors=[
                "framework map for A0 lacks framework-specific grounds",
                "framework map for A1 lacks framework-specific grounds",
            ],
        )
        cycle = CycleRecord(
            1, WorkspaceBroadcast(constraint="CONSENSUS_AUDIT"), [candidate],
            candidate, None, {"A0": 0.55, "A1": 0.45}, 0.7, 1, 1.0,
        )
        result = WorkspaceResult(
            "A test", ["A0", "A1"], cycles=[cycle]
        )
        findings = audit_trace_health(result)
        self.assertTrue(any(
            finding.code == "FRAMEWORK_MAP_INCOMPLETE"
            for finding in findings
        ), [finding.code for finding in findings])
        self.assertFalse(any(
            finding.code == "FRAMEWORK_RETENTION_LOST"
            for finding in findings
        ), [finding.code for finding in findings])

    def test_visibility_audit_is_nonvoting_action_confidence_adjustment(self):
        actions = ["rely on visible reports", "serve the excluded group"]
        engine = WorkspaceEngine(
            [
                FixedSpecialist("utility", actions[0], "UNCERTAINTY"),
                FixedSpecialist("care", actions[1], "CARE"),
            ],
            WorkspaceConfig(max_cycles=1, enable_synthesis=False),
        )

        def visibility(_scenario, current_actions):
            return VisibilityAssessment(
                True,
                True,
                "excluded residents",
                "unequal reporting infrastructure hides their needs",
                "excluded residents cannot submit reports",
                {current_actions[0]: 0.65, current_actions[1]: 1.0},
                activated=True,
            )

        result = engine.run(
            "Excluded residents cannot submit reports.", actions,
            assess_visibility=visibility,
        )
        self.assertEqual(len(result.cycles[0].candidates), 2)
        self.assertEqual(result.current_plurality, actions[1])
        self.assertTrue(result.visibility_assessments[0].activated)
        answer = render_public_judgment(result)
        self.assertIn("# Ethical Parliament Judgment", answer)
        self.assertIn("serve the excluded group", answer)
        self.assertNotIn("×0.65", answer)
        self.assertNotIn("Hidden-harm audit", answer)

    def test_summary_separates_final_policy_from_broadcast_context(self):
        actions = ["route water", "route hospital"]
        winner = CandidateChunk(
            "virtue", "CHARACTER", {"route water": 0.19, "route hospital": 0.81},
            0.4, 0.6, 0.8, recommended_action="route hospital",
        )
        dissent = CandidateChunk(
            "care", "CARE", {"route water": 0.81, "route hospital": 0.19},
            0.4, 0.6, 0.7, recommended_action="route water",
        )
        cycle = CycleRecord(
            cycle=4,
            broadcast=WorkspaceBroadcast(constraint="OPEN_DELIBERATION"),
            candidates=[winner, dissent],
            winner=winner,
            dissent=dissent,
            policy={"route water": 0.19, "route hospital": 0.81},
            entropy=0.42,
            stable_cycles=2,
            elapsed_seconds=3.5,
            received_broadcast=WorkspaceBroadcast(constraint="CARE"),
            is_hypothetical=False,
        )
        result = WorkspaceResult(
            scenario="test",
            actions=actions,
            cycles=[cycle],
            selected_action="route hospital",
            confidence=0.81,
            current_plurality="route hospital",
            epistemic_confidence=0.72,
        )
        summary = render_summary(result)
        public = render_public_judgment(result)
        self.assertEqual(summary, public)
        self.assertIn("**Policy support:** 0.81", summary)
        self.assertIn("**route hospital.**", summary)
        self.assertIn("## Deliberation Map", summary)
        self.assertNotIn("Final winning constraint:", summary)
        self.assertNotIn("Previous broadcast context:", summary)
        self.assertNotIn("Last emitted broadcast:", summary)

    def test_summary_surfaces_typed_audit_variable(self):
        actions = ["decline", "share"]
        cycle = CycleRecord(
            cycle=2,
            broadcast=WorkspaceBroadcast(constraint="CONSENSUS_AUDIT"),
            candidates=[],
            winner=CandidateChunk(
                "care", "CARE", {"decline": 0.6, "share": 0.4}, 0.0, 0.0, 0.6,
                recommended_action="decline",
            ),
            dissent=CandidateChunk(
                "care", "CARE", {"decline": 0.6, "share": 0.4}, 0.0, 0.0, 0.6,
                recommended_action="share",
            ),
            policy={"decline": 0.6, "share": 0.4},
            entropy=0.42,
            stable_cycles=1,
            elapsed_seconds=2.1,
            received_broadcast=WorkspaceBroadcast(
                constraint="CONSENSUS_AUDIT",
                audit_variable={
                    "entity": "volunteer coordinator",
                    "relation": "THIRD_PARTY_STATUS",
                    "possible_values": [
                        "EXTERNAL_THIRD_PARTY",
                        "AUTHORIZED_INTERNAL_AGENT",
                        "UNKNOWN",
                    ],
                    "focus_action": "share",
                    "question": "Classify the entity 'volunteer coordinator' as EXTERNAL_THIRD_PARTY (third party), AUTHORIZED_INTERNAL_AGENT (authorized internal agent), or UNKNOWN, then state whether the recommendation for 'share' changes under the AUTHORIZED_INTERNAL_AGENT reading.",
                    "required_response": {
                        "counterfactual_anchor": "AUTHORIZED_INTERNAL_AGENT",
                        "allowed_effects": [
                            "NO_CHANGE", "WEAKENS", "REVERSES", "UNRESOLVED",
                        ],
                    },
                },
            ),
            is_hypothetical=False,
        )
        result = WorkspaceResult(
            scenario="test",
            actions=actions,
            cycles=[cycle],
            selected_action="decline",
            confidence=0.81,
            current_plurality="decline",
            epistemic_confidence=0.72,
            access_decisions=[
                WorkspaceAccessDecision(
                    cycle=2,
                    content_type="CONSENSUS_AUDIT",
                    admitted=True,
                    signals=["third_party_status"],
                    question=cycle.received_broadcast.audit_variable["question"],
                    audit_variable=cycle.received_broadcast.audit_variable,
                )
            ],
        )
        summary = render_summary(result)
        # Audit plumbing stays in the JSON trace, not the default brief.
        self.assertNotIn("typed audit variable", summary)
        self.assertNotIn("THIRD_PARTY_STATUS", summary)
        self.assertNotIn("counterfactual_anchor", summary)
        self.assertIn("# Ethical Parliament Judgment", summary)

    def test_summary_humanizes_residue_and_excludes_model_retry_from_reopen(self):
        actions = ["preserve liberty", "impose restriction"]
        winner = CandidateChunk(
            "care", "CARE", {actions[0]: 0.3, actions[1]: 0.7},
            0.2, 0.4, 0.7, recommended_action=actions[1],
            rationale="Protects dependent children.",
        )
        rights = CandidateChunk(
            "deontological", "RIGHTS", {actions[0]: 0.8, actions[1]: 0.2},
            0.3, 0.6, 0.6, recommended_action=actions[0],
            rationale="The restriction lacks reciprocal public justification.",
        )
        cycle = CycleRecord(
            cycle=2,
            broadcast=WorkspaceBroadcast(
                constraint="CARE",
                problem_state={"unresolved_questions": [{
                    "category": "RESOLVE_NORMATIVE_TENSION",
                    "question": "Whether a less restrictive route can protect children",
                }]},
            ),
            candidates=[winner, rights], winner=winner, dissent=rights,
            policy={actions[0]: 0.3, actions[1]: 0.7},
            entropy=0.4, stable_cycles=1, elapsed_seconds=1.0,
        )
        result = WorkspaceResult(
            scenario="test", actions=actions, cycles=[cycle],
            selected_action=actions[1], current_plurality=actions[1],
            confidence=0.7, epistemic_confidence=0.6,
            moral_residue=["RIGHTS"],
            reopen_conditions=["RESOLVE_NORMATIVE_TENSION", "RETRY_MODEL_CALL"],
        )

        summary = render_summary(result)
        public = render_public_judgment(result)

        self.assertEqual(summary, public)
        self.assertIn("lacks reciprocal public justification", summary)
        self.assertIn("normative adjudication establishes a strict duty", summary.casefold())
        self.assertNotIn("RETRY_MODEL_CALL", summary)
        self.assertNotIn("RESOLVE_NORMATIVE_TENSION", summary)
        self.assertIn("## Deliberation Map", summary)
        self.assertNotIn("Unresolved moral considerations: RIGHTS", public)

    def test_summary_renders_accepted_synthesis_from_its_own_schema(self):
        actions = ["coerce donors", "wait for volunteers", "recruit volunteers rapidly"]
        winner = CandidateChunk(
            "care", "CARE", {actions[0]: 0.2, actions[1]: 0.5, actions[2]: 0.8},
            0.2, 0.6, 0.8, recommended_action=actions[2],
        )
        cycle = CycleRecord(
            cycle=3,
            broadcast=WorkspaceBroadcast(constraint="SYNTHESIS_REVIEW"),
            candidates=[winner], winner=winner, dissent=None,
            policy={actions[0]: 0.2, actions[1]: 0.5, actions[2]: 0.8},
            entropy=0.3, stable_cycles=1, elapsed_seconds=1.0,
        )
        result = WorkspaceResult(
            scenario="test", actions=actions, cycles=[cycle],
            selected_action=actions[2], current_plurality=actions[2],
            confidence=0.8, epistemic_confidence=0.7,
            synthesis_proposals=[SynthesisProposal(
                actions[2], ["care", "deontological"], ["CARE", "RIGHTS"],
                0.77, "Respects autonomy while accelerating supply",
                accepted=True, admission_status="ADMITTED",
            )],
        )

        summary = render_summary(result)

        # Accepted synthesis that became the recommendation is the policy leader,
        # not a separate diagnostic "Synthesis:" appendix line.
        self.assertIn(actions[2], summary)
        self.assertIn("**Policy leader:**", summary)
        self.assertNotIn("feasibility=0.77", summary)
        self.assertNotIn("addressed constraints:", summary)

    def test_summary_explains_shared_contingency_failure_without_contradiction(self):
        winner = CandidateChunk(
            "care",
            "CARE",
            {"A0": 0.6, "A1": 0.4},
            0.2,
            0.5,
            0.7,
            recommended_action="A0",
        )
        cycle = CycleRecord(
            cycle=1,
            broadcast=WorkspaceBroadcast(constraint="OPEN_DELIBERATION"),
            candidates=[winner],
            winner=winner,
            dissent=None,
            policy={"A0": 0.6, "A1": 0.4},
            entropy=0.4,
            stable_cycles=1,
            elapsed_seconds=1.0,
        )
        result = WorkspaceResult(
            scenario="test",
            actions=["A0", "A1"],
            cycles=[cycle],
            selected_action="A0",
            confidence=0.6,
            current_plurality="A0",
            epistemic_confidence=0.7,
            contingency_feasibility_assessments=[
                ContingencyFeasibilityAssessment(
                    synthesis_action="combine safeguards",
                    predicate_label="the safeguard works",
                    fallback_statuses={"A0": "AVAILABLE", "A1": "AVAILABLE"},
                    fallback_reasons={
                        "A0": "A0 remains executable",
                        "A1": "A1 remains executable",
                    },
                    evidence_bases={
                        "A0": "SCENARIO_STRUCTURE",
                        "A1": "SCENARIO_STRUCTURE",
                    },
                    shared_failure=True,
                    valid=True,
                    approved=False,
                    error="shared failure: the failure removes a capability used by the synthesis and at least one fallback; the original fallbacks may still remain individually available",
                )
            ],
        )
        summary = render_summary(result)
        # Contingency feasibility plumbing stays in the JSON trace.
        self.assertNotIn("Fallback availability:", summary)
        self.assertNotIn("Shared-failure check:", summary)
        self.assertIn("# Ethical Parliament Judgment", summary)

    def test_public_judgment_explains_shared_contingency_failure_without_contradiction(self):
        winner = CandidateChunk(
            "care",
            "CARE",
            {"A0": 0.6, "A1": 0.4},
            0.2,
            0.5,
            0.7,
            recommended_action="A0",
        )
        cycle = CycleRecord(
            cycle=1,
            broadcast=WorkspaceBroadcast(constraint="OPEN_DELIBERATION"),
            candidates=[winner],
            winner=winner,
            dissent=None,
            policy={"A0": 0.6, "A1": 0.4},
            entropy=0.4,
            stable_cycles=1,
            elapsed_seconds=1.0,
        )
        result = WorkspaceResult(
            scenario="test",
            actions=["A0", "A1"],
            cycles=[cycle],
            selected_action="A0",
            confidence=0.6,
            current_plurality="A0",
            epistemic_confidence=0.7,
            contingency_feasibility_assessments=[
                ContingencyFeasibilityAssessment(
                    synthesis_action="combine safeguards",
                    predicate_label="the safeguard works",
                    fallback_statuses={"A0": "AVAILABLE", "A1": "AVAILABLE"},
                    fallback_reasons={
                        "A0": "A0 remains executable",
                        "A1": "A1 remains executable",
                    },
                    evidence_bases={
                        "A0": "SCENARIO_STRUCTURE",
                        "A1": "SCENARIO_STRUCTURE",
                    },
                    shared_failure=True,
                    valid=True,
                    approved=False,
                    error="shared failure: the failure removes a capability used by the synthesis and at least one fallback; the original fallbacks may still remain individually available",
                )
            ],
        )
        answer = render_public_judgment(result)
        self.assertNotIn("Independent contingency feasibility", answer)
        self.assertNotIn("Fallback availability:", answer)
        self.assertIn("# Ethical Parliament Judgment", answer)

    def test_source_cache_key_depends_on_scenario_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path_a = Path(directory) / "a.json"
            path_b = Path(directory) / "b.json"
            path_a.write_text(json.dumps({"ethical_question": "Case A"}), encoding="utf-8")
            path_b.write_text(json.dumps({"ethical_question": "Case B"}), encoding="utf-8")
            key_a = build_source_cache_key("v1", "same query", "scenario-a", path_a)
            key_b = build_source_cache_key("v1", "same query", "scenario-b", path_b)
            self.assertNotEqual(key_a, key_b)
            self.assertEqual(
                key_a,
                build_source_cache_key("v1", "same query", "scenario-a", path_a),
            )

    def test_long_actions_survive_losslessly_in_state_objects(self):
        long_action = (
            "illegally allocate the remaining doses to frontline transport workers "
            "to break the primary transmission chain and prevent five times as "
            "many total deaths across the city"
        )
        proposal = SynthesisProposal(
            action=long_action,
            grounded_in=["utilitarian", "deontological"],
            addressed_constraints=["IMMINENT_HARM", "DUTY"],
            feasibility=0.8,
            rationale="test",
        )
        branch = PlanningBranchEvaluation(
            cycle=1,
            origin_action=long_action,
            condition="if the transmission chain can be broken without inventing new facts",
            fallback=long_action,
            selected_action=long_action,
            confidence=0.9,
        )
        assessment = PlanningAssessment(
            target_action=long_action,
            activation_reason="test",
            feasibility=0.9,
            necessary_condition="transmission chain remains breakable",
            failure_condition="chain cannot be interrupted",
            fallback=long_action,
        )
        self.assertEqual(proposal.action, long_action)
        self.assertEqual(branch.origin_action, long_action)
        self.assertEqual(branch.selected_action, long_action)
        self.assertEqual(branch.fallback, long_action)
        self.assertEqual(assessment.target_action, long_action)
        self.assertEqual(assessment.fallback, long_action)

    def test_graph_compiler_rejects_truncated_action_clauses(self):
        with self.assertRaises(ValueError):
            compile_scenario_graph(
                "Choose between a clipped action and a complete one.",
                [
                    "illegally allocate the remaining doses to frontline transport workers to break the main",
                    "administer all remaining doses to nursing home residents on the official waitlist",
                ],
            )

    def test_policy_weights_epistemic_confidence_not_preference_gap(self):
        certain_moderate = CandidateChunk(
            "grounded", "CARE", {"act": 0.75, "wait": 0.25},
            0.2, 0.5, 0.9, recommended_action="act",
            preference_strength=0.5, epistemic_confidence=0.9,
        )
        uncertain_strong = CandidateChunk(
            "speculative", "UNCERTAINTY", {"act": 0.05, "wait": 0.95},
            0.8, 0.9, 0.2, recommended_action="wait",
            preference_strength=0.9, epistemic_confidence=0.2,
        )
        policy = WorkspaceEngine._policy(
            [certain_moderate, uncertain_strong], ["act", "wait"]
        )
        self.assertGreater(policy["act"], policy["wait"])

    def test_suspicious_consensus_gets_workspace_access_before_convergence(self):
        seen_broadcasts = []

        class TrackingSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                seen_broadcasts.append(broadcast.constraint)
                chunk = super().evaluate(scenario, actions, broadcast)
                if self.name == "utilitarian":
                    chunk.baseline_status = "UNDERDETERMINED"
                    chunk.unresolved = "VERIFY_FACTS"
                    chunk.utilitarian_missing_comparison = (
                        "relative harm of lives saved versus destabilized society"
                    )
                return chunk

        engine = WorkspaceEngine(
            [
                TrackingSpecialist("utilitarian", "publish immediately", "UNCERTAINTY"),
                TrackingSpecialist("care", "publish immediately", "CARE"),
                TrackingSpecialist("duty", "publish immediately", "DUTY"),
            ],
            WorkspaceConfig(max_cycles=2),
        )
        result = engine.run(
            "A cure could save lives but might destabilize society.",
            ["publish immediately", "conceal indefinitely"],
            scenario_facts={},
            source_testimonies={
                "utilitarian": "The answer depends on how many people are affected.",
                "care": "If instability is severe, the answer could flip.",
                "duty": "Publish the cure.",
            },
        )
        self.assertTrue(result.access_decisions[0].admitted)
        self.assertIn("homogeneous_score_vectors", result.access_decisions[0].signals)
        self.assertIn("conditional_source_testimony", result.access_decisions[0].signals)
        self.assertIn("asymmetric_action_extremity", result.access_decisions[0].signals)
        self.assertIn("CONSENSUS_AUDIT", seen_broadcasts)

    def test_consensus_audit_targets_missing_boundary_variable(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "decline", "CARE"),
                FixedSpecialist("duty", "decline", "DUTY"),
                FixedSpecialist("virtue", "decline", "CHARACTER"),
            ],
            WorkspaceConfig(max_cycles=2),
        )
        result = engine.run(
            "A resident asked to keep their address private, but a neighborhood volunteer coordinator wants it for a welcome basket.",
            ["decline", "share"],
            scenario_facts={},
            source_testimonies={
                "care": "Keep it private unless third parties are authorized.",
                "duty": "Sharing with third parties violates the promise.",
                "virtue": "Trustworthiness matters here.",
            },
        )
        self.assertFalse(any(item.admitted for item in result.access_decisions))
        self.assertFalse(any(
            "privacy_scope" in item.signals for item in result.access_decisions
        ))

    def test_consensus_audit_selects_grounded_problem_state_question(self):
        action_a = "preserve liberty while leaving lower material assistance"
        action_b = "remove liberty while eliminating severe material deprivation"

        class StateQuestionSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                if self.name == "utilitarian":
                    chunk.action_scores = {actions[0]: 0.52, actions[1]: 0.48}
                    chunk.recommended_action = actions[0]
                    chunk.unresolved = "VERIFY_FACTS"
                    chunk.utilitarian_missing_comparison = (
                        "relative utility of liberty versus material security"
                    )
                    chunk.baseline_status = "UNDERDETERMINED"
                    chunk.assumption_status = "UNDERDETERMINED"
                else:
                    chunk.baseline_status = "DIRECT"
                return chunk

        result = WorkspaceEngine(
            [
                StateQuestionSpecialist("utilitarian", action_a, "UNCERTAINTY"),
                StateQuestionSpecialist("duty", action_a, "DUTY"),
                StateQuestionSpecialist("rawlsian", action_a, "FAIRNESS"),
            ],
            WorkspaceConfig(
                max_cycles=2, consensus_audit_max_entropy=0.8,
                enable_reversal_audit=False, enable_synthesis=False,
                enable_planning=False,
            ),
        ).run(
            "Policy A preserves liberty but leaves lower material security. Policy B removes liberty but eliminates severe material deprivation.",
            [action_a, action_b],
            scenario_facts={},
            source_testimonies={
                "utilitarian": "Underdetermined unless relative welfare magnitudes are known.",
                "duty": "Choose Policy A.",
                "rawlsian": "Choose Policy A.",
            },
        )
        decision = next(item for item in result.access_decisions if item.admitted)
        self.assertEqual(decision.audit_variable["relation"], "COMPARATIVE_MAGNITUDE")
        self.assertEqual(
            decision.audit_variable["source"],
            "problem_state.unresolved_questions",
        )
        self.assertTrue(decision.audit_variable["issue_id"].startswith("QUESTION:"))
        self.assertTrue(decision.audit_variable["grounded_in"])
        self.assertEqual(decision.audit_variable["raised_by"], ["utilitarian"])
        self.assertNotIn("privacy_scope", decision.signals)
        self.assertTrue(all(
            str(item.audit_variable.get("issue_id", "")).startswith("QUESTION:")
            for item in result.access_decisions
            if item.admitted and item.audit_variable
        ))

    def test_consensus_audit_rejects_problem_state_candidate_without_question_key(self):
        signals, question, variable = _problem_state_audit_probe(
            {
                "audit_candidates": [{
                    "issue_id": None,
                    "source": "problem_state.unresolved_questions",
                    "proposition": "relative utility of liberty versus security",
                    "grounded_in": ["C1", "C2"],
                    "raised_by": ["utilitarian"],
                    "category": "VERIFY_FACTS",
                    "status": "UNRESOLVED",
                }],
                "agent_positions": [],
            },
            "preserve liberty",
        )

        self.assertEqual(signals, [])
        self.assertEqual(question, "")
        self.assertEqual(variable, {})

    def test_problem_state_audit_variable_is_accepted_by_delegate_parser(self):
        audit_variable = {
            "issue_id": "QUESTION:abc123",
            "source": "problem_state.unresolved_questions",
            "proposition": "relative utility of liberty versus material security",
            "grounded_in": ["C1", "C2"],
            "raised_by": ["utilitarian"],
            "status": "PERSISTENT_UNRESOLVED",
            "entity": "relative utility of liberty versus material security",
            "relation": "COMPARATIVE_MAGNITUDE",
            "possible_values": ["NO_CHANGE", "WEAKENS", "REVERSES", "UNRESOLVED"],
            "focus_action": "preserve liberty",
            "question": "Would resolving the relative utility comparison change the recommendation?",
            "required_response": {
                "counterfactual_anchor": "issue QUESTION:abc123",
                "allowed_effects": ["NO_CHANGE", "WEAKENS", "REVERSES", "UNRESOLVED"],
            },
        }
        data = {
            "scores": {"A0": 0.52, "A1": 0.48}, "r": "A0",
            "c": "UNCERTAINTY", "u": "VERIFY_FACTS",
            "w": "aggregate comparison remains unresolved", "j": "NONE",
            "d": "UNDERDETERMINED",
            "a": "relative utility weights remain unknown",
            "v": "material security produces substantially greater welfare",
            "ie": "UNRESOLVED",
            "av": {
                key: value for key, value in audit_variable.items()
                if key in {
                    "entity", "relation", "possible_values", "focus_action",
                    "question", "required_response",
                }
            },
        }
        data["av"]["focus_action"] = "A0"
        candidate = _candidate_from_data(
            "utilitarian", ["preserve liberty", "maximize material security"],
            data,
            WorkspaceBroadcast(
                constraint="CONSENSUS_AUDIT", audit_variable=audit_variable,
            ),
            "NONE", {},
        )
        self.assertTrue(candidate.schema_valid)
        self.assertEqual(
            candidate.audit_variable["relation"], "COMPARATIVE_MAGNITUDE",
        )

    def test_broadcast_unresolved_aggregates_nonwinning_positions(self):
        class MixedSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                if self.name == "utilitarian":
                    chunk.action_scores = {actions[0]: 0.51, actions[1]: 0.49}
                    chunk.recommended_action = actions[0]
                    chunk.unresolved = "VERIFY_FACTS"
                    chunk.assumption_status = "UNDERDETERMINED"
                    chunk.surprise = 0.0
                    chunk.friction = 0.0
                    chunk.utilitarian_missing_comparison = (
                        "relative welfare of liberty versus material security"
                    )
                return chunk

        result = WorkspaceEngine(
            [
                MixedSpecialist("duty", "protect liberty", "DUTY"),
                MixedSpecialist("utilitarian", "protect liberty", "UNCERTAINTY"),
            ],
            WorkspaceConfig(
                max_cycles=1, enable_consensus_audit=False,
                enable_problem_state_audit=False, enable_synthesis=False,
                enable_planning=False, enable_reversal_audit=False,
            ),
        ).run(
            "Protect liberty or maximize material security.",
            ["protect liberty", "maximize material security"],
        )
        self.assertEqual(result.cycles[0].winner.unresolved, "NONE")
        self.assertEqual(result.cycles[0].broadcast.unresolved, "VERIFY_FACTS")
        self.assertIn(
            "VERIFY_FACTS",
            result.cycles[0].broadcast.problem_state["unresolved_categories"],
        )

    def test_persistent_grounded_issue_executes_problem_state_audit(self):
        seen = []

        class FocusSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                seen.append(broadcast.constraint)
                chunk = super().evaluate(scenario, actions, broadcast)
                if self.name == "utilitarian":
                    chunk.action_scores = {actions[0]: 0.52, actions[1]: 0.48}
                    chunk.recommended_action = actions[0]
                    chunk.baseline_status = "UNDERDETERMINED"
                    chunk.assumption_status = "UNDERDETERMINED"
                    chunk.unresolved = "VERIFY_FACTS"
                    chunk.utilitarian_missing_comparison = (
                        "relative utility of liberty versus material security"
                    )
                return chunk

        result = WorkspaceEngine(
            [
                FocusSpecialist("utilitarian", "protect liberty", "UNCERTAINTY"),
                FocusSpecialist("duty", "protect liberty", "DUTY"),
            ],
            WorkspaceConfig(
                max_cycles=3, stable_cycles_required=4,
                enable_consensus_audit=False, enable_problem_state_audit=True,
                enable_synthesis=False, enable_planning=False,
                enable_reversal_audit=False,
            ),
        ).run(
            "Policy A protects liberty. Policy B maximizes material security.",
            ["protect liberty", "maximize material security"],
        )
        decision = next(
            item for item in result.access_decisions
            if item.content_type == "PROBLEM_STATE_AUDIT" and item.admitted
        )
        self.assertEqual(decision.audit_variable["relation"], "COMPARATIVE_MAGNITUDE")
        self.assertEqual(decision.audit_variable["status"], "PERSISTENT_UNRESOLVED")
        self.assertIn("PROBLEM_STATE_AUDIT", seen)
        audit_cycle = next(
            cycle for cycle in result.cycles
            if cycle.received_broadcast.constraint == "PROBLEM_STATE_AUDIT"
        )
        self.assertEqual(
            audit_cycle.received_broadcast.audit_variable["issue_id"],
            decision.audit_variable["issue_id"],
        )

    def test_synthesis_feasibility_is_projected_as_unresolved(self):
        proposal = SynthesisProposal(
            "voluntary hybrid", ["care"], ["CARE"], 0.8, "bridge",
            accepted=True, proposal_id="P0", source_agents=["care"],
            feasibility_status="PLAUSIBLE", grounding_status="GROUNDED",
            promotion_status="UNDER_REVIEW",
        )
        candidate = CandidateChunk(
            specialist="duty", constraint="DUTY",
            action_scores={"A0": 0.8, "A1": 0.2},
            surprise=0.2, friction=0.5, confidence=0.8,
            recommended_action="A0", baseline_status="DIRECT",
        )
        state = build_deliberative_problem_state(
            2, ["A0", "A1"], [candidate],
            "A0", candidate, proposals=[proposal],
        ).to_dict()
        self.assertIn("CHECK_FEASIBILITY", state["unresolved_categories"])
        self.assertEqual(state["primary_unresolved"], "CHECK_FEASIBILITY")
        self.assertTrue(any(
            item["category"] == "CHECK_FEASIBILITY"
            and item["grounded_in"] == ["P0"]
            for item in state["audit_candidates"]
        ))
        self.assertEqual([item["action"] for item in state["live_actions"]], ["A0", "A1"])
        self.assertEqual(state["proposals"][0]["proposal_id"], "P0")

    def test_choice_status_composition_distinguishes_fallback_consensus(self):
        actions = ["A0", "A1"]
        direct = CandidateChunk(
            specialist="duty", constraint="DUTY",
            action_scores={"A0": 0.9, "A1": 0.1},
            surprise=0.2, friction=0.5, confidence=0.8,
            recommended_action="A0", baseline_status="DIRECT",
        )
        fallback = CandidateChunk(
            specialist="care", constraint="CARE",
            action_scores={"A0": 0.8, "A1": 0.2},
            surprise=0.2, friction=0.5, confidence=0.8,
            recommended_action="A0",
            baseline_status="OUTSIDE_ACTION_SET_WITH_FALLBACK",
            baseline_preferred_extension="A voluntary hybrid response",
        )
        uncertain = CandidateChunk(
            specialist="utilitarian", constraint="UNCERTAINTY",
            action_scores={"A0": 0.51, "A1": 0.49},
            surprise=0.2, friction=0.1, confidence=0.4,
            recommended_action="A0", baseline_status="PARSE_FAILURE",
            assumption_status="UNDERDETERMINED",
            unresolved="VERIFY_FACTS",
            utilitarian_missing_comparison="relative welfare magnitude",
        )
        state = build_deliberative_problem_state(
            2, actions, [direct, fallback, uncertain], "A0", direct,
        ).to_dict()
        self.assertEqual(state["surface_consensus"], "UNANIMOUS")
        self.assertEqual(
            state["deliberative_consensus"],
            "SURFACE_AGREEMENT_WITH_QUALIFICATIONS",
        )
        self.assertEqual(
            state["support_composition"]["counts"],
            {"DIRECT": 1, "FALLBACK": 1, "UNDERDETERMINED": 1},
        )
        care = next(
            item for item in state["agent_positions"] if item["specialist"] == "care"
        )
        self.assertEqual(care["choice_status"], "FALLBACK")
        self.assertEqual(care["preferred_extension"], "A voluntary hybrid response")
        self.assertTrue(any(
            item["conflict_type"] == "ACTION_SET_ADEQUACY"
            for item in state["active_conflicts"]
        ))
    def test_consensus_audit_rejects_entities_leaked_from_testimony(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("utilitarian", "raise the material floor", "WELFARE"),
                FixedSpecialist("duty", "raise the material floor", "DUTY"),
                FixedSpecialist("virtue", "raise the material floor", "CHARACTER"),
            ],
            WorkspaceConfig(max_cycles=2),
        )
        result = engine.run(
            "Choose an economic policy under uncertain aggregate growth effects.",
            ["raise the material floor", "maximize aggregate growth"],
            scenario_facts={},
            source_testimonies={
                "utilitarian": "Assume agent is an authorized internal agent.",
                "duty": "If agent were an external third party, keep it private.",
                "virtue": "The organizational boundary and privacy scope are uncertain.",
            },
        )

        serialized = json.dumps(asdict(result)).casefold()
        self.assertNotIn("organizational_boundary", serialized)
        self.assertNotIn("privacy_scope", serialized)
        self.assertNotIn("authorized_internal_agent", serialized)
        self.assertNotIn("external_third_party", serialized)
        self.assertTrue(all(
            not decision.audit_variable for decision in result.access_decisions
        ))

    def test_unanimity_cannot_erase_conditional_source_testimony(self):
        class VariedConsensus(FixedSpecialist):
            def __init__(self, name, strength):
                super().__init__(name, "protect", "CARE")
                self.strength = strength

            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.action_scores = {"protect": self.strength, "decline": 1 - self.strength}
                chunk.confidence = abs(2 * self.strength - 1)
                chunk.recommended_action = "protect"
                if self.name == "care":
                    chunk.baseline_status = "CONDITIONAL"
                    chunk.baseline_condition = "whether support remains available"
                    chunk.unresolved = "VERIFY_FACTS"
                    chunk.unsupported_assumption = "whether support remains available"
                    chunk.assumption_status = "CONDITIONAL"
                return chunk

        engine = WorkspaceEngine(
            [VariedConsensus("care", 0.60), VariedConsensus("duty", 0.78), VariedConsensus("virtue", 0.96)],
            WorkspaceConfig(max_cycles=2, consensus_audit_max_entropy=1.0),
        )
        result = engine.run(
            "Choose whether to protect or decline.", ["protect", "decline"],
            source_testimonies={
                "care": "Protect only if support remains available.",
                "duty": "The answer depends on whether a promise was made.",
                "virtue": "Protection may be appropriate.",
            },
        )
        # Unanimity plus conditional testimony cannot manufacture a consensus
        # audit. The explicitly persistent issue may still earn the dedicated
        # problem-state slot, which is deliberation about the qualification
        # rather than agreement erasing it.
        self.assertFalse(any(
            decision.admitted for decision in result.access_decisions
            if decision.content_type == "CONSENSUS_AUDIT"
        ))
        self.assertEqual(
            result.deliberative_problem_state["deliberative_consensus"],
            "SURFACE_AGREEMENT_WITH_QUALIFICATIONS",
        )

    def test_clear_fact_grounded_consensus_does_not_trigger_audit(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                FixedSpecialist("duty", "protect", "DUTY"),
            ],
            WorkspaceConfig(max_cycles=2),
        )
        result = engine.run(
            "Choose whether to protect the person.",
            ["protect", "do not protect"],
            scenario_facts={"known_outcome": True},
            source_testimonies={"care": "Protect them.", "duty": "Protect them."},
        )
        self.assertTrue(result.access_decisions)
        self.assertFalse(result.access_decisions[0].admitted)
        self.assertNotIn("CONSENSUS_AUDIT", [cycle.broadcast.constraint for cycle in result.cycles])

    def test_conditional_prose_without_structural_warning_does_not_trigger_audit(self):
        class DifferentStrengthSpecialist(FixedSpecialist):
            def __init__(self, name, preferred, constraint, high):
                super().__init__(name, preferred, constraint)
                self.high = high

            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                low = 1.0 - self.high
                chunk.action_scores = {
                    action: self.high if action == self.preferred else low
                    for action in actions
                }
                chunk.confidence = self.high - low
                return chunk

        engine = WorkspaceEngine(
            [
                DifferentStrengthSpecialist("duty", "keep", "DUTY", 0.9),
                DifferentStrengthSpecialist("care", "keep", "CARE", 0.8),
                DifferentStrengthSpecialist("virtue", "keep", "CHARACTER", 0.7),
            ],
            WorkspaceConfig(max_cycles=2),
        )
        result = engine.run(
            "A painful memory transfers intact into an unknown stranger if you erase it.",
            ["erase and transfer", "keep"],
            scenario_facts={"transfer": "intact", "recipient": "unknown stranger"},
            source_testimonies={
                "duty": "If the stranger consented the answer might differ.",
                "care": "The answer depends on relational harm.",
                "virtue": "Keep the memory.",
            },
        )
        self.assertTrue(result.access_decisions)
        self.assertFalse(result.access_decisions[0].admitted)

    def test_comparative_risk_language_routes_hr_case_to_audit(self):
        class ComparativeSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                if self.name == "utilitarian":
                    chunk.baseline_status = "UNDERDETERMINED"
                    chunk.unresolved = "VERIFY_FACTS"
                    chunk.utilitarian_missing_comparison = (
                        "comparative harm of recruitment delay versus discrimination"
                    )
                return chunk

        engine = WorkspaceEngine(
            [
                ComparativeSpecialist("utilitarian", "disable", "UNCERTAINTY"),
                ComparativeSpecialist("duty", "disable", "DUTY"),
                ComparativeSpecialist("care", "disable", "CARE"),
            ],
            WorkspaceConfig(max_cycles=2),
        )
        result = engine.run(
            "Disabling the tool delays urgent recruitment; keeping it risks unfair discrimination. Which is the greater harm?",
            ["disable", "continue"],
            scenario_facts={},
            source_testimonies={
                "utilitarian": "The answer depends on comparative magnitude.",
                "duty": "Disable the discriminatory tool.",
                "care": "The greater harm depends on who is affected.",
            },
        )
        decision = next(item for item in result.access_decisions if item.admitted)
        self.assertIn("sparse_facts_with_uncertainty", decision.signals)
        self.assertIn("comparative_magnitude_unresolved", decision.signals)

    def test_grounded_descriptive_consequence_does_not_trigger_comparative_claim(self):
        actions = [
            "Decline to share the address",
            "Share the address",
        ]
        cases = {
            actions[0]: "Declining preserves the resident's privacy request and keeps the promise.",
            actions[1]: "Sharing enables a welcome basket and violates the resident's privacy request.",
        }
        self.assertEqual(_comparative_claim_errors(actions, cases), [])

    def test_grounded_tradeoff_operands_are_not_comparative_magnitude(self):
        actions = ["Policy B", "Policy A"]
        cases = {
            actions[0]: (
                "Policy B creates an exceptionally high material floor for poor households "
                "and strips occupational choice, privacy, and movement; both are direct consequences."
            ),
            actions[1]: (
                "Policy A preserves liberty while leaving poor households with lower material assistance."
            ),
        }

        self.assertEqual(_comparative_claim_errors(actions, cases), [])

    def test_only_outweighs_operator_is_flagged_in_grounded_tradeoff(self):
        actions = ["Policy B", "Policy A"]
        cases = {
            actions[0]: (
                "Policy B creates a high material floor and reduces liberty. "
                "The material benefit outweighs the liberty loss."
            ),
            actions[1]: "Policy A preserves liberty and provides lower material assistance.",
        }

        errors = _comparative_claim_errors(actions, cases)
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].startswith("COMPARATIVE_MAGNITUDE: case for A0"))
        self.assertIn("operator=The material benefit outweighs the liberty loss", errors[0])

    def test_comparative_operator_does_not_relabel_grounded_operands_unstated(self):
        actions = ["Policy B", "Policy A"]
        data = {
            "scores": {"A0": 0.8, "A1": 0.2}, "r": "A0",
            "c": "CARE", "u": "NONE",
            "w": "material benefit outweighs liberty loss", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.9,
            "l": {
                "A0": "Policy B creates a high material floor and reduces liberty. The material benefit outweighs the liberty loss.",
                "A1": "Policy A preserves liberty and provides lower material assistance.",
            },
            "da": "material floor versus liberty",
            "t": "prefer the action with the stronger justified claim",
            "tf": "NONE",
        }

        def confirm_operator(_scenario, seen_actions, cases, errors):
            self.assertEqual(list(seen_actions), actions)
            return [*errors, *_comparative_claim_errors(seen_actions, cases)]

        chunk = _candidate_from_data(
            "care", actions, data, WorkspaceBroadcast(), "NONE", {},
            landscape_verifier=confirm_operator,
            scenario_text=(
                "Policy B creates an exceptionally high material floor but strips occupational "
                "choice, privacy, and movement. Policy A preserves those liberties while "
                "providing lower material assistance."
            ),
        )

        self.assertEqual(chunk.evidence_basis, "STATED_FACTS")
        self.assertEqual(chunk.unresolved, "VERIFY_FACTS")
        self.assertIn("operator=", chunk.speculative_claim)
        self.assertLess(chunk.action_scores[actions[0]], 0.8)

    def test_explicit_beneficiary_ranking_still_triggers_comparative_claim(self):
        actions = [
            "Decline to share the address",
            "Share the address",
        ]
        cases = {
            actions[0]: "Declining is better for the resident than disclosure because it protects the least advantaged party.",
            actions[1]: "Sharing gives the coordinator more than the resident wanted.",
        }
        errors = _comparative_claim_errors(actions, cases)
        self.assertTrue(any(
            error.startswith("UNRESOLVED_RELATIONAL_PREMISE:")
            for error in errors
        ) or any(
            error.startswith("COMPARATIVE_MAGNITUDE:")
            for error in errors
        ), errors)
        self.assertTrue(any(
            error.startswith(
                "COMPARATIVE_MAGNITUDE: case for A1 makes an unverified comparative beneficiary claim"
            )
            for error in errors
        ))

    def test_third_party_status_is_classified_as_unresolved_relational_premise(self):
        actions = [
            "Decline to share the address",
            "Share the address",
        ]
        cases = {
            actions[0]: "Declining is better off if the coordinator counts as a third party or unauthorized agent.",
            actions[1]: "Sharing is worse off if the coordinator is an internal agent rather than a third party.",
        }
        errors = _comparative_claim_errors(actions, cases)
        self.assertTrue(any(
            error.startswith("UNRESOLVED_RELATIONAL_PREMISE:")
            for error in errors
        ), errors)

    def test_rejected_access_is_reconsidered_when_score_pattern_changes(self):
        strengths = [0.65, 0.75, 0.85, 0.95]

        class ConvergingScoreSpecialist(FixedSpecialist):
            calls = 0

            def __init__(self, name, strength):
                super().__init__(name, "choose A", "CARE")
                self.strength = strength

            def evaluate(self, scenario, actions, broadcast):
                self.calls += 1
                high = self.strength if self.calls == 1 else 0.8
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.action_scores = {actions[0]: high, actions[1]: 1.0 - high}
                chunk.confidence = abs(2.0 * high - 1.0)
                return chunk

        engine = WorkspaceEngine(
            [ConvergingScoreSpecialist(f"agent-{index}", strength) for index, strength in enumerate(strengths)],
            WorkspaceConfig(max_cycles=3, stop_redundant_consensus_cycles=False),
        )
        result = engine.run(
            "Choose which policy to follow.",
            ["choose A", "choose B"],
            scenario_facts={"choice": "fixed"},
            source_testimonies={
                f"agent-{index}": "The answer depends on further evidence."
                for index in range(len(strengths))
            },
        )
        self.assertGreaterEqual(len(result.access_decisions), 2)
        self.assertFalse(result.access_decisions[0].admitted)
        self.assertFalse(any(item.admitted for item in result.access_decisions))
        self.assertIn("homogeneous_score_vectors", result.access_decisions[-1].signals)

    def test_conditional_audit_blocks_false_reconvergence(self):
        class AuditAwareSpecialist(FixedSpecialist):
            audited = False

            def evaluate(self, scenario, actions, broadcast):
                if broadcast.constraint == "CONSENSUS_AUDIT":
                    self.audited = True
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.baseline_status = "CONDITIONAL"
                chunk.baseline_condition = "comparative population harm"
                chunk.unresolved = "VERIFY_FACTS"
                chunk.unsupported_assumption = "relative lives saved versus destabilized society"
                chunk.assumption_status = "CONDITIONAL"
                if self.audited:
                    chunk.assumption_status = "CONDITIONAL"
                    chunk.unsupported_assumption = "one harm exceeds the other"
                    chunk.reversal_condition = "the other harm affects more people"
                    chunk.unresolved = "VERIFY_FACTS"
                return chunk

        engine = WorkspaceEngine(
            [
                AuditAwareSpecialist("utilitarian", "publish immediately", "UNCERTAINTY"),
                AuditAwareSpecialist("care", "publish immediately", "CARE"),
            ],
            WorkspaceConfig(max_cycles=3),
        )
        result = engine.run(
            "A cure could save lives but might destabilize society.",
            ["publish immediately", "conceal indefinitely"],
            scenario_facts={},
            source_testimonies={
                "utilitarian": "The answer depends on comparative harm.",
                "care": "If instability is widespread, reconsider.",
            },
        )
        self.assertEqual(result.halted_by, "convergence")
        self.assertEqual(result.judgment_status, "GOVERNED_RECOMMENDATION")
        self.assertTrue(all(
            candidate.assumption_status == "CONDITIONAL"
            for candidate in result.cycles[-1].candidates
            if candidate.schema_valid
        ))
        governing = result.cycles[-1].governing_claim
        self.assertIsNotNone(governing)
        self.assertIn("provided", (governing.decision_rule or "").casefold())

    def test_audited_underdetermination_broadcasts_problem_reformulation(self):
        seen_broadcasts = []

        class AuditAwareSpecialist(FixedSpecialist):
            audited = False

            def evaluate(self, scenario, actions, broadcast):
                seen_broadcasts.append(broadcast.constraint)
                if broadcast.constraint == "CONSENSUS_AUDIT":
                    self.audited = True
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.baseline_status = "UNDERDETERMINED"
                chunk.unresolved = "VERIFY_FACTS"
                chunk.unsupported_assumption = "relative lives saved versus destabilized society"
                chunk.assumption_status = "UNDERDETERMINED"
                if broadcast.constraint == "PROBLEM_REFORMULATION" and self.name == "care":
                    chunk.action_scores = {"release": 0.05, "suppress": 0.95}
                    chunk.recommended_action = "suppress"
                if broadcast.constraint == "PROBLEM_REFORMULATION":
                    chunk.boundary_position = "A1" if self.name == "care" else "A0"
                    chunk.decisive_axis = "systemic harm versus mortality"
                    chunk.boundary_switch_condition = "switch if systemic harm crosses expected mortality"
                if self.audited:
                    chunk.assumption_status = "UNDERDETERMINED"
                    chunk.unsupported_assumption = "comparative population harm is unknown"
                    chunk.reversal_condition = "instability causes greater widespread harm"
                    chunk.unresolved = "VERIFY_FACTS"
                return chunk

        engine = WorkspaceEngine(
            [
                AuditAwareSpecialist("utilitarian", "release", "UNCERTAINTY"),
                AuditAwareSpecialist("care", "release", "CARE"),
            ],
            WorkspaceConfig(max_cycles=3),
        )

        def reformulate(_scenario, actions, _candidates):
            return ProblemReformulation(
                ["comparative reach of each harm"],
                [
                    CalibrationOutcome(actions[0], "mortality", "BENEFIT", "lives saved", 1.0, 1000, "lives", "10 years"),
                    CalibrationOutcome(actions[0], "employment", "HARM", "employment disruption", 0.1, 1000000, "jobs", "1 year"),
                    CalibrationOutcome(actions[1], "mortality", "HARM", "foregone cures", 1.0, 1000, "lives", "10 years"),
                ],
                "frameworks split when systemic disruption rivals preventable mortality",
                "whether duties to patients permit diffuse systemic risk",
                "At this boundary, should the actor release or suppress the cure?",
                ["utilitarian", "care"],
                accepted=True,
            )

        result = engine.run(
            "A cure could save lives but might destabilize society.",
            ["release", "suppress"],
            scenario_facts={},
            source_testimonies={
                "utilitarian": "The answer depends on comparative harms.",
                "care": "If disruption is widespread, reconsider.",
            },
            reformulate_problem=reformulate,
        )
        self.assertIn("PROBLEM_REFORMULATION", seen_broadcasts)
        self.assertTrue(result.problem_reformulations[0].accepted)
        self.assertEqual(result.judgment_status, "UNRESOLVED")
        self.assertEqual(result.selected_action, "UNRESOLVED")
        self.assertEqual(result.current_plurality, "release")
        reformulation_cycle = next(
            cycle for cycle in result.cycles
            if cycle.received_broadcast.constraint == "PROBLEM_REFORMULATION"
        )
        self.assertTrue(reformulation_cycle.is_hypothetical)
        self.assertNotEqual(
            next(cycle for cycle in reversed(result.cycles) if not cycle.is_hypothetical)
            .received_broadcast.constraint,
            "PROBLEM_REFORMULATION",
        )
        contributions = summarize_specialist_contributions(result.cycles)
        actual_cycle_count = sum(not cycle.is_hypothetical for cycle in result.cycles)
        self.assertTrue(all(
            record["responses"] == actual_cycle_count
            for record in contributions.values()
        ))

    def test_planning_is_selective_and_does_not_vote(self):
        calls = []
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                FixedSpecialist("duty", "disclose", "DUTY"),
            ],
            WorkspaceConfig(max_cycles=1, planning_entropy_threshold=0.0),
        )

        def plan(scenario, actions, selected, broadcast, candidates, reason):
            calls.append((selected, reason))
            return PlanningAssessment(
                selected, reason, 0.4,
                "the actor retains practical access",
                "the actor loses practical access",
                "disclose",
                strategic_forces=["INSTITUTIONAL_POWER"],
                broadcast_worthy=True,
                grounded_evidence="limited practical access",
                fallback_available=True,
                fallback_availability_reason="disclosure remains directly available",
            )

        result = engine.run(
            "The actor has limited practical access to protect; disclosure remains available.",
            ["protect", "disclose"], analyze_plan=plan,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.cycles[0].broadcast.constraint, "PLANNING_REVIEW")
        self.assertEqual(len(result.planning_assessments), 1)
        # Planning contributes no action scores, so the ethical tie remains a tie.
        self.assertAlmostEqual(result.cycles[0].policy["protect"], 0.5)
        self.assertAlmostEqual(result.cycles[0].policy["disclose"], 0.5)

    def test_planning_rejects_invented_legislative_failure_story(self):
        actions = [
            "Fund opt-in state-matched donations to poorest households",
            "Preserve the existing economic policy",
        ]
        assessment = PlanningAssessment(
            actions[0], "synthesis feasibility", 0.4,
            "the authorizing bill passes the legislature",
            "opposition coalition blocks or filibusters the authorizing bill",
            actions[1],
            broadcast_worthy=True,
            grounded_evidence="the proposal is a policy",
            fallback_available=True,
            fallback_availability_reason="the existing policy remains physically available",
            target_action_node_id="A0",
        )

        validated = WorkspaceEngine._validate_planning_assessment(
            assessment,
            "A0 may fail if voluntary participation is insufficient.",
            WorkspaceBroadcast(),
            actions,
        )

        self.assertFalse(validated.valid)
        self.assertFalse(validated.broadcast_worthy)
        self.assertEqual(
            validated.failure_grounding_status,
            "REJECTED_UNGROUNDED_FAILURE_CONDITION",
        )
        self.assertIn("without current-run provenance", validated.error)

    def test_planning_recognizes_failure_derived_from_opt_in_mechanism(self):
        status, errors = classify_planning_failure_grounding(
            "voluntary participation provides sufficient financing",
            "voluntary participation is insufficient to finance meaningful assistance",
            "Choose an economic policy.",
            ["Fund opt-in state-matched donations to poorest households"],
        )

        self.assertEqual(status, "MECHANISM_DERIVED_FAILURE_CONDITION")
        self.assertEqual(errors, [])

    def test_planning_skips_when_no_physically_available_fallback_exists(self):
        calls = []
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "cross bridge", "CARE"),
                FixedSpecialist("duty", "wait at bridge", "DUTY"),
            ],
            WorkspaceConfig(max_cycles=1, planning_entropy_threshold=0.0),
        )

        result = engine.run(
            "A0 requires access to the bridge controls. A1 requires access to the bridge controls.",
            ["cross bridge", "wait at bridge"],
            analyze_plan=lambda *_args: calls.append(True),
        )
        self.assertEqual(calls, [])
        self.assertEqual(len(result.planning_assessments), 1)
        self.assertFalse(result.planning_assessments[0].valid)
        self.assertIn("no physically available fallback", result.planning_assessments[0].error)

    def test_planning_stays_inactive_after_clear_agreement(self):
        calls = []
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                FixedSpecialist("duty", "protect", "DUTY"),
            ],
            WorkspaceConfig(max_cycles=1),
        )
        engine.run(
            "A clear problem", ["protect", "disclose"],
            analyze_plan=lambda *_args: calls.append(True),
        )
        self.assertEqual(calls, [])

    def test_delegate_feasibility_label_cannot_invent_an_obstacle(self):
        calls = []
        engine = WorkspaceEngine(
            [
                FixedSpecialist("utilitarian", "strike doctor", "FEASIBILITY"),
                FixedSpecialist("care", "strike engineer", "CARE"),
            ],
            WorkspaceConfig(max_cycles=1, planning_entropy_threshold=0.0),
        )
        result = engine.run(
            "A driverless van must strike the doctor or strike the engineer.",
            ["strike doctor", "strike engineer"],
            analyze_plan=lambda *_args: calls.append(True),
        )
        self.assertEqual(calls, [])
        self.assertEqual(result.planning_assessments, [])

    def test_branch_rejected_when_failure_disables_shared_control(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("utilitarian", "strike doctor", "FEASIBILITY"),
                FixedSpecialist("care", "strike engineer", "CARE"),
            ],
            WorkspaceConfig(max_cycles=1, planning_entropy_threshold=0.0),
        )

        def plan(_scenario, _actions, selected, _broadcast, _candidates, reason):
            return PlanningAssessment(
                selected, reason, 0.2,
                "steering control remains functional",
                "loss of steering control prevents choosing a target",
                "strike engineer", broadcast_worthy=True,
                grounded_evidence="Steering control may fail",
                fallback_available=True,
                fallback_availability_reason="the engineer remains physically present",
            )

        result = engine.run(
            "Steering control may fail while choosing whom to strike.",
            ["strike doctor", "strike engineer"], analyze_plan=plan,
        )
        self.assertEqual(result.planning_branches, [])
        self.assertEqual(len(result.planning_assessments), 1)
        self.assertFalse(result.planning_assessments[0].valid)
        self.assertIn("no physically available fallback", result.planning_assessments[0].error)

    def test_material_planning_failure_is_received_next_cycle(self):
        broadcasts = []

        class TrackingSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                broadcasts.append((self.name, broadcast.constraint, broadcast.contingency_question))
                return super().evaluate(scenario, actions, broadcast)

        engine = WorkspaceEngine(
            [
                TrackingSpecialist("care", "protect", "CARE"),
                TrackingSpecialist("duty", "disclose", "DUTY"),
            ],
            WorkspaceConfig(max_cycles=2, planning_entropy_threshold=0.0),
        )

        def plan(_scenario, _actions, selected, _broadcast, _candidates, reason):
            return PlanningAssessment(
                selected, reason, 0.4,
                "the actor retains practical access",
                "the actor loses practical access",
                "disclose", broadcast_worthy=True,
                grounded_evidence="limited practical access",
                fallback_available=True,
                fallback_availability_reason="disclosure remains directly available",
            )

        result = engine.run(
            "The actor has limited practical access to protect; disclosure remains available.",
            ["protect", "disclose"], analyze_plan=plan,
        )
        second_cycle = [entry for entry in broadcasts if entry[1] == "PLANNING_REVIEW"]
        self.assertEqual(len(second_cycle), 2)
        self.assertTrue(all("loses practical access" in entry[2] for entry in second_cycle))
        self.assertEqual(len(result.planning_branches), 1)
        self.assertTrue(result.cycles[1].is_hypothetical)
        self.assertEqual(result.cycles[1].received_broadcast.branch_origin_action, "protect")

    def test_planning_branch_does_not_replace_base_case_policy(self):
        class BranchAwareSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                self.preferred = "disclose" if broadcast.branch_kind == "PLANNING_CONTINGENCY" else "protect"
                return super().evaluate(scenario, actions, broadcast)

        engine = WorkspaceEngine(
            [
                BranchAwareSpecialist("care", "protect", "CARE"),
                BranchAwareSpecialist("duty", "protect", "DUTY"),
                FixedSpecialist("critic", "disclose", "RISK"),
            ],
            WorkspaceConfig(max_cycles=2, planning_entropy_threshold=0.0),
        )

        def plan(_scenario, _actions, selected, _broadcast, _candidates, reason):
            return PlanningAssessment(
                selected, reason, 0.4,
                "the stated safeguard remains effective",
                "the stated safeguard fails materially",
                "disclose", broadcast_worthy=True,
                grounded_evidence="the safeguard may fail",
                fallback_available=True,
                fallback_availability_reason="disclosure does not require the safeguard",
            )

        result = engine.run(
            "The protect action requires a safeguard that may fail.",
            ["protect", "disclose"], analyze_plan=plan,
        )
        self.assertEqual(result.planning_branches[0].selected_action, "disclose")
        self.assertEqual(result.current_plurality, "protect")
        self.assertEqual(
            max(result.cycles[-1].policy, key=result.cycles[-1].policy.get), "protect"
        )

    def test_unpromoted_proposal_does_not_activate_contingency_extension(self):
        requests = []

        class ViabilitySpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                synthesis = "protect while disclosing carefully"
                if broadcast.constraint == "PROPOSAL_REVIEW":
                    preferred = self.preferred
                    chunk = CandidateChunk(
                        self.name, self.constraint,
                        {action: (0.8 if action == preferred else 0.4) for action in actions},
                        0.3, 0.4, 0.8, rationale="reviewed synthesis",
                        recommended_action=preferred,
                        action_admissibility={
                            action: "UNASSESSED"
                            for action in actions
                        },
                    )
                    return chunk
                return super().evaluate(scenario, actions, broadcast)

        engine = WorkspaceEngine(
            [ViabilitySpecialist("care", "protect", "CARE"), ViabilitySpecialist("duty", "disclose", "DUTY")],
            WorkspaceConfig(
                max_cycles=2,
                entropy_threshold=0.0,
                max_cycle_extensions=1,
            ),
        )

        def synthesize(*_args):
            return SynthesisProposal(
                "protect while disclosing carefully", ["care", "duty"],
                ["CARE", "DUTY"], 0.8, "bridges duties", accepted=True,
            )

        def extend(result):
            requests.append(len(result.cycles))
            return 2

        def analyze(_result):
            graph = compile_contingency_graph(
                {
                    "p": "the careful disclosure succeeds",
                    "x": "the protective synthesis can no longer work",
                },
                "protect while disclosing carefully",
                ["protect", "disclose"],
            )
            return FailureCondition(
                "protect while disclosing carefully",
                "the careful disclosure succeeds",
                "NOT(the careful disclosure succeeds): the synthesis fails",
                "If careful disclosure fails, should we protect or disclose?",
                fallback_actions=["protect", "disclose"],
                predicate_label="the careful disclosure succeeds",
                semantic_graph=graph.graph.to_dict(),
            )

        def verify(condition):
            return ContingencyFeasibilityAssessment(
                condition.synthesis_action, condition.predicate_label,
                {"A0": "AVAILABLE", "A1": "AVAILABLE"},
                {
                    "A0": "protection remains independently executable",
                    "A1": "disclosure remains independently executable",
                },
                {"A0": "FAILURE_SCOPE", "A1": "FAILURE_SCOPE"},
                False, valid=True, approved=True,
            )

        result = engine.run(
            "A disputed problem", ["protect", "disclose"],
            synthesize=synthesize, request_extension=extend,
            analyze_contingency=analyze,
            verify_contingency_feasibility=verify,
        )
        self.assertEqual(requests, [])
        self.assertEqual(len(result.cycles), 2)
        self.assertEqual(result.halted_by, "cycle_budget")
        self.assertNotIn(
            "CONTINGENCY_REVIEW",
            [(cycle.received_broadcast or cycle.broadcast).constraint for cycle in result.cycles],
        )
        self.assertEqual(result.synthesis_proposals[0].promotion_status, "UNDER_REVIEW")

    def test_nonviable_synthesis_does_not_activate_contingency(self):
        calls = []
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                FixedSpecialist("duty", "disclose", "DUTY"),
            ],
            WorkspaceConfig(max_cycles=2, entropy_threshold=0.0),
        )

        def synthesize(*_args):
            return SynthesisProposal(
                "combine both actions", ["care", "duty"], ["CARE", "DUTY"],
                0.8, "bridge", accepted=True,
            )

        def analyze(_result):
            calls.append("contingency")
            raise AssertionError("nonviable synthesis must not reach contingency analysis")

        result = engine.run(
            "A disputed problem", ["protect", "disclose"],
            synthesize=synthesize, request_extension=lambda _result: 1,
            analyze_contingency=analyze,
        )
        self.assertEqual(calls, [])
        self.assertFalse(result.synthesis_viability_assessments[0].viable)
        self.assertEqual(result.synthesis_viability_assessments[0].recommendation_count, 0)

    def test_grounded_synthesis_remains_proposal_and_is_not_rescored_as_action(self):
        seen_actions = []

        class TrackingSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                seen_actions.append(tuple(actions))
                preferred = self.preferred
                return CandidateChunk(
                    specialist=self.name,
                    constraint=self.constraint,
                    action_scores={action: 0.95 if action == preferred else 0.05 for action in actions},
                    surprise=0.6,
                    friction=0.7,
                    confidence=0.9,
                    recommended_action=preferred,
                    rationale="test",
                    decision_rule=f"Prefer {preferred} under {self.constraint}.",
                    adjudication_status="SUPPORTS",
                    governing_eligible=True,
                )

        preferred_action = "tell the truth compassionately"
        engine = WorkspaceEngine(
            [TrackingSpecialist("care", "lie", "CARE"), TrackingSpecialist("duty", "truth", "DUTY")],
            WorkspaceConfig(max_cycles=2, entropy_threshold=0.0),
        )

        def synthesize(*_args):
            return SynthesisProposal(
                preferred_action, ["care", "duty"], ["CARE", "DUTY"], 0.9,
                "combines care and honesty", accepted=True,
            )

        result = engine.run("A general conflict", ["lie", "truth"], synthesize=synthesize)
        self.assertNotIn(preferred_action, result.actions)
        self.assertFalse(any(preferred_action in actions for actions in seen_actions))
        self.assertNotEqual(result.selected_action, preferred_action)
        if result.selected_action != "UNRESOLVED":
            self.assertIn(result.selected_action, {"lie", "truth"})
        self.assertTrue(result.synthesis_proposals[0].accepted)
        self.assertEqual(result.synthesis_proposals[0].proposal_id, "P0")
        self.assertEqual(result.synthesis_proposals[0].promotion_status, "UNDER_REVIEW")
        self.assertEqual(
            result.deliberative_problem_state["proposals"][0]["proposal_id"], "P0",
        )
        self.assertEqual(
            [item["action"] for item in result.deliberative_problem_state["live_actions"]],
            ["lie", "truth"],
        )
        proposal_node = next(
            node for node in result.semantic_graphs[-1]["nodes"]
            if node["id"] == "P0"
        )
        self.assertEqual(proposal_node["kind"], "PROPOSAL")
        self.assertFalse(any(
            node["kind"] == "ACTION" and node["label"] == preferred_action
            for node in result.semantic_graphs[-1]["nodes"]
        ))

    def test_proposal_review_is_stored_without_promoting_proposal(self):
        proposal_text = "provide voluntary targeted assistance"

        class ReviewingSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                if broadcast.constraint == "PROPOSAL_REVIEW":
                    chunk.proposal_review = ProposalFrameworkReview(
                        proposal_id="P0",
                        specialist=self.name,
                        framework_status=(
                            "SUPPORTS" if self.name == "care" else "QUALIFIES"
                        ),
                        framework_reason=(
                            "reduces dependency without coercive administration"
                            if self.name == "care"
                            else "respects duty if participation remains voluntary"
                        ),
                        predicted_consequences=[{
                            "dimension": "material sufficiency",
                            "subject": "vulnerable households",
                            "direction": "IMPROVES",
                            "grounding_status": "PROPOSAL_TEXT",
                            "provenance": ["P0"],
                        }],
                        feasibility_concerns=["participation may be insufficient"],
                        required_conditions=["participation remains voluntary"],
                    )
                return chunk

        result = WorkspaceEngine(
            [
                ReviewingSpecialist("care", "assist", "CARE"),
                ReviewingSpecialist("duty", "decline", "DUTY"),
            ],
            WorkspaceConfig(
                max_cycles=2, entropy_threshold=0.0,
                enable_consensus_audit=False, enable_problem_state_audit=False,
                enable_planning=False, enable_reversal_audit=False,
            ),
        ).run(
            "Choose whether to assist or decline.", ["assist", "decline"],
            synthesize=lambda *_: SynthesisProposal(
                proposal_text, ["care", "duty"], ["CARE", "DUTY"],
                0.75, "combines assistance and voluntariness", accepted=True,
            ),
        )
        proposal = result.synthesis_proposals[0]
        self.assertEqual(set(proposal.framework_reviews), {"care", "duty"})
        self.assertEqual(len(proposal.predicted_consequences), 2)
        self.assertEqual(proposal.promotion_status, "ADMISSIBLE")
        self.assertNotIn(proposal_text, result.actions)
        self.assertNotIn(proposal_text, result.cycles[-1].policy)
        projected = result.deliberative_problem_state["proposals"][0]
        self.assertEqual(set(projected["framework_reviews"]), {"care", "duty"})
        self.assertEqual(projected["review_summary"]["missing_reviewers"], [])
        self.assertEqual(projected["review_summary"]["valid_review_count"], 2)
        self.assertEqual(
            projected["review_summary"]["framework_status_counts"],
            {"SUPPORTS": 1, "QUALIFIES": 1},
        )

    def test_admitted_proposal_review_emits_access_decision(self):
        result = WorkspaceEngine(
            [
                FixedSpecialist("care", "assist", "CARE"),
                FixedSpecialist("duty", "decline", "DUTY"),
            ],
            WorkspaceConfig(
                max_cycles=2, entropy_threshold=0.0,
                enable_consensus_audit=False, enable_problem_state_audit=False,
                enable_planning=False, enable_reversal_audit=False,
            ),
        ).run(
            "Choose whether to assist or decline.", ["assist", "decline"],
            synthesize=lambda *_: SynthesisProposal(
                "assist only with informed consent", ["care", "duty"],
                ["CARE", "DUTY"], 0.8, "bridges care and duty", accepted=True,
            ),
        )
        self.assertTrue(any(
            (cycle.received_broadcast or cycle.broadcast).constraint == "PROPOSAL_REVIEW"
            for cycle in result.cycles
        ))
        self.assertTrue(any(
            decision.content_type == "PROPOSAL_REVIEW"
            and decision.admitted
            and "admitted_synthesis_proposal" in decision.signals
            for decision in result.access_decisions
        ))

    def test_guaranteed_proposal_review_before_finalization_when_pass_missing(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                FixedSpecialist("duty", "disclose", "DUTY"),
            ],
            WorkspaceConfig(
                max_cycles=2,
                entropy_threshold=0.0,
                enable_consensus_audit=False,
                enable_problem_state_audit=False,
                enable_planning=False,
                enable_reversal_audit=False,
            ),
        )
        seeded = SynthesisProposal(
            "request voluntary release from both groups",
            ["care", "duty"],
            ["CARE", "DUTY"],
            0.82,
            "pareto attempt across constraints",
            accepted=True,
            proposal_id="P0",
            promotion_status="UNDER_REVIEW",
            admission_status="UNDER_REVIEW",
        )

        def pending(result):
            if not any(item.proposal_id == "P0" for item in result.synthesis_proposals):
                result.synthesis_proposals.append(seeded)
            return WorkspaceEngine._pending_proposal_for_guaranteed_review(engine, result)

        engine._pending_proposal_for_guaranteed_review = pending  # type: ignore[method-assign]
        result = engine.run(
            "A disputed allocation problem.",
            ["protect", "disclose"],
        )
        self.assertTrue(any(
            (cycle.received_broadcast or cycle.broadcast).constraint == "PROPOSAL_REVIEW"
            for cycle in result.cycles
        ))
        self.assertTrue(any(
            decision.content_type == "PROPOSAL_REVIEW"
            and "guaranteed_proposal_review" in decision.signals
            for decision in result.access_decisions
        ))

    def test_pending_proposal_helper_skips_completed_review_pass(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                FixedSpecialist("duty", "disclose", "DUTY"),
            ],
            WorkspaceConfig(max_cycles=2),
        )
        proposal = SynthesisProposal(
            "bridge carefully", ["care", "duty"], ["CARE", "DUTY"],
            0.9, "bridge", accepted=True, proposal_id="P0",
            promotion_status="UNDER_REVIEW",
        )
        review_broadcast = WorkspaceBroadcast(
            constraint="PROPOSAL_REVIEW",
            intent="review_p0",
            reformulation_context="Review proposal P0: bridge carefully.",
        )
        cycle = CycleRecord(
            1, review_broadcast, [], None, None,
            {"protect": 0.6, "disclose": 0.4}, 0.9, 0, 1.0,
            received_broadcast=review_broadcast,
        )
        result = WorkspaceResult(
            scenario="x",
            actions=["protect", "disclose"],
            cycles=[cycle],
        )
        result.synthesis_proposals.append(proposal)
        self.assertIsNone(engine._pending_proposal_for_guaranteed_review(result))

    def test_proposal_review_parser_rejects_unproven_scenario_inheritance(self):
        data = {
            "scores": {"A0": 0.6, "A1": 0.4}, "r": "A0",
            "c": "CARE", "u": "CHECK_FEASIBILITY",
            "w": "proposal remains under review", "j": "NONE",
            "fa": "care evaluates dependency and responsiveness", "fr": True,
            "pr": {
                "proposal_id": "P0", "framework_status": "QUALIFIES",
                "framework_reason": "could support vulnerable households conditionally",
                "predicted_consequences": [{
                    "dimension": "material sufficiency",
                    "subject": "vulnerable households",
                    "direction": "IMPROVES",
                    "grounding_status": "SCENARIO_INHERITED",
                    "provenance": ["P0"],
                }],
                "feasibility_concerns": ["funding remains uncertain"],
                "required_conditions": ["adequate funding"],
            },
        }
        candidate = _candidate_from_data(
            "care", ["assist", "decline"], data,
            WorkspaceBroadcast(
                constraint="PROPOSAL_REVIEW",
                problem_state={"proposals": [{
                    "proposal_id": "P0", "promotion_status": "UNDER_REVIEW",
                }]},
            ),
            "NONE", {},
        )
        self.assertIsNotNone(candidate.proposal_review)
        self.assertFalse(candidate.proposal_review.valid)
        self.assertTrue(any(
            "scenario clause provenance" in error
            for error in candidate.proposal_review.validation_errors
        ))

    def test_cycle_budget_returns_contested_plurality_instead_of_no_judgment(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("utilitarian", "pull", "HARM"),
                FixedSpecialist("care", "pull", "CARE"),
                FixedSpecialist("virtue", "pull", "CHARACTER"),
                FixedSpecialist("duty", "do not pull", "DUTY"),
                FixedSpecialist("rawlsian", "do not pull", "RIGHTS"),
            ],
            WorkspaceConfig(max_cycles=2, entropy_threshold=0.60, enable_synthesis=False),
        )
        result = engine.run("A closed moral conflict.", ["pull", "do not pull"])
        self.assertEqual(result.halted_by, "cycle_budget")
        self.assertEqual(result.judgment_status, "CONTESTED_RECOMMENDATION")
        self.assertEqual(result.selected_action, "pull")
        self.assertEqual(result.current_plurality, "pull")
        self.assertIn("pull", result.compressed_rule.casefold())
        answer = render_public_judgment(result)
        self.assertIn("**pull — presently favored, but contested.**", answer)
        self.assertIn("## Why the Parliament currently favors this action", answer)
        self.assertIn("## Deliberation Map", answer)
        self.assertNotIn("judged overriding rather than the leading consideration", answer)
        self.assertNotIn("Cycle 1", answer)
        self.assertNotIn("action_scores", answer)
        self.assertTrue(result.termination_assessment.resource_censored)
        self.assertEqual(
            result.termination_assessment.termination_type, "RESOURCE_CENSORED"
        )
        self.assertFalse(result.further_deliberation_estimate.affects_stopping)
        self.assertIn(
            "The cycle budget was reached before deliberation naturally converged.",
            answer,
        )
        self.assertNotIn("resource-censored", answer)

    def test_cycle_budget_action_judgment_gets_provisional_compressed_rule(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                FixedSpecialist("duty", "protect", "DUTY"),
            ],
            WorkspaceConfig(
                max_cycles=1,
                stable_cycles_required=2,
                enable_synthesis=False,
            ),
        )
        result = engine.run("Choose protect or wait.", ["protect", "wait"])
        self.assertEqual(result.halted_by, "cycle_budget")
        self.assertEqual(result.judgment_status, "GOVERNED_RECOMMENDATION")
        self.assertIn("provisionally prefer protect", result.compressed_rule)
        self.assertNotIn("Unavailable", result.compressed_rule)

    def test_compressed_rule_uses_winning_constraint_and_preserves_plural_dissent(self):
        class PluralRuleSpecialist(FixedSpecialist):
            def __init__(self, name, preferred, constraint, rule, assessment):
                super().__init__(name, preferred, constraint)
                self.rule = rule
                self.assessment = assessment

            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.recommended_action = self.preferred
                chunk.decision_rule = self.rule
                chunk.framework_action_map = {self.preferred: self.assessment}
                chunk.landscape_decisive_axis = self.assessment
                if self.name == "deontological":
                    chunk.surprise = 1.0
                    chunk.friction = 1.0
                    chunk.epistemic_confidence = 1.0
                    chunk.confidence = 1.0
                elif self.name == "utilitarian":
                    chunk.epistemic_confidence = 1.0
                    chunk.confidence = 1.0
                else:
                    chunk.surprise = 0.1
                    chunk.friction = 0.2
                    chunk.epistemic_confidence = 0.15
                    chunk.confidence = 0.15
                return chunk

        result = WorkspaceEngine(
            [
                PluralRuleSpecialist(
                    "deontological", "A0", "DUTY",
                    "Prefer A0 when an impartial public rule requires it",
                    "REQUIRED: impartial treatment governs the allocation",
                ),
                PluralRuleSpecialist(
                    "utilitarian", "A0", "UNCERTAINTY",
                    "Prefer A0 when expected welfare is larger",
                    "larger expected welfare",
                ),
                PluralRuleSpecialist(
                    "virtue", "A1", "CHARACTER", "Prefer A1 under practical wisdom",
                    "EXEMPLIFIES: compassion and practical wisdom",
                ),
                PluralRuleSpecialist(
                    "care", "A1", "CARE", "Prefer A1 under concrete dependence",
                    "existing dependency and entrusted care",
                ),
                PluralRuleSpecialist(
                    "rawlsian", "A1", "FAIRNESS", "Prefer A1 for the least advantaged",
                    "IMPROVES: position of the least advantaged",
                ),
            ],
            WorkspaceConfig(
                max_cycles=1, enable_synthesis=False, enable_reversal_audit=False,
                consensus_audit_min_signals=99,
            ),
        ).run("A closed plural conflict.", ["A0", "A1"])

        self.assertIn("Rule [DUTY]", result.compressed_rule)
        self.assertIn("impartial public rule", result.compressed_rule)
        self.assertNotIn("expected welfare is larger", result.compressed_rule)
        self.assertIn("virtue/CHARACTER", result.compressed_rule)
        self.assertIn("care/CARE", result.compressed_rule)
        self.assertIn("rawlsian/FAIRNESS", result.compressed_rule)
        answer = render_public_judgment(result)
        self.assertIn("**Governing claim:**", answer)
        self.assertIn("impartial public rule", answer)

    def test_public_judgment_uses_valid_landscape_for_action_comparison(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("utilitarian", "pull", "IMMINENT_HARM"),
                FixedSpecialist("duty", "wait", "DUTY"),
            ],
            WorkspaceConfig(max_cycles=1, entropy_threshold=0.0, enable_synthesis=False),
        )
        result = engine.run("One dies if pulled; five die otherwise.", ["pull", "wait"])
        final = result.cycles[-1]
        final.candidates[0].landscape_cases = {
            "pull": "Kills one but saves five",
            "wait": "Avoids direct action but five die",
        }
        final.candidates[0].landscape_semantic_valid = True
        final.candidates[0].landscape_search_complete = True
        final.candidates[0].landscape_decisive_axis = "aggregate lives saved"
        answer = render_public_judgment(result)
        self.assertIn("## Deliberation Map", answer)
        self.assertIn("Kills one but saves five", answer)
        # Opposing landscape stays available in the map contribution / why section
        # without a separate comparison appendix.
        self.assertNotIn("Original action and consequence comparison", answer)

    def test_public_judgment_ignores_semantically_invalid_landscape_case(self):
        engine = WorkspaceEngine(
            [FixedSpecialist("care", "protect", "CARE"), FixedSpecialist("duty", "decline", "DUTY")],
            WorkspaceConfig(max_cycles=1, entropy_threshold=0.0, enable_synthesis=False),
        )
        result = engine.run("Choose whether to protect.", ["protect", "decline"])
        result.cycles[-1].candidates[0].landscape_cases = {
            "protect": "Fabricated inverted consequence",
            "decline": "Another fabricated consequence",
        }
        result.cycles[-1].candidates[0].landscape_semantic_valid = False
        answer = render_public_judgment(result)
        self.assertNotIn("Fabricated inverted consequence", answer)

    def test_public_judgment_does_not_replace_original_alternative_with_synthesis(self):
        engine = WorkspaceEngine(
            [FixedSpecialist("care", "child", "CARE"), FixedSpecialist("utility", "climbers", "IMMINENT_HARM")],
            WorkspaceConfig(max_cycles=1, entropy_threshold=0.0, enable_synthesis=False),
        )
        result = engine.run("Choose child or climbers.", ["child", "climbers"])
        result.actions.append("invented threshold option")
        result.synthesis_proposals.append(SynthesisProposal(
            "invented threshold option", ["care", "utility"], ["CARE", "IMMINENT_HARM"],
            0.8, "attempted bridge", accepted=True,
        ))
        candidate = result.cycles[-1].candidates[0]
        candidate.landscape_semantic_valid = True
        candidate.landscape_cases = {
            "child": "Certainly saves one child",
            "climbers": "Forty percent chance saves three climbers",
            "invented threshold option": "Invented policy",
        }
        answer = render_public_judgment(result)
        self.assertIn("climbers", answer)
        why = answer.split("## Alternative action discovered", 1)[0]
        self.assertNotIn("invented threshold option", why)
        self.assertIn("## Alternative action discovered", answer)
        self.assertIn("invented threshold option", answer)
        self.assertIn("## Deliberation Map", answer)

    def test_public_judgment_recovers_audit_reversals_across_cycles(self):
        engine = WorkspaceEngine(
            [FixedSpecialist("care", "reject", "CARE"), FixedSpecialist("utility", "reject", "IMMINENT_HARM")],
            WorkspaceConfig(max_cycles=2),
        )
        result = engine.run("Approve or reject.", ["approve", "reject"])
        audited = result.cycles[0].candidates[0]
        audited.assumption_status = "CONDITIONAL"
        audited.unsupported_assumption = "future harm cannot be mitigated"
        audited.reversal_condition = "future harm becomes reversible"
        answer = render_public_judgment(result)
        self.assertIn("## What could change the judgment", answer)
        self.assertIn("future harm cannot be mitigated", answer.casefold())
        self.assertIn("future harm becomes reversible", answer.casefold())
        self.assertNotIn("Uncertain assumptions identified during audit", answer)

    def test_rejected_synthesis_does_not_block_deliberation(self):
        engine = WorkspaceEngine(
            [FixedSpecialist("care", "protect", "CARE"), FixedSpecialist("duty", "disclose", "DUTY")],
            WorkspaceConfig(max_cycles=2, entropy_threshold=0.0),
        )

        def synthesize(*_args):
            return SynthesisProposal(
                "call authorities", ["care"], ["CARE"], 0.2, "escape",
                accepted=False, rejection_reason="insufficiently feasible",
            )

        result = engine.run("Immediate choice", ["protect", "disclose"], synthesize=synthesize)
        self.assertNotIn("call authorities", result.actions)
        self.assertEqual(len(result.cycles), 2)
        self.assertFalse(result.synthesis_proposals[0].accepted)

    def test_synthesis_failure_is_recorded_and_does_not_crash(self):
        engine = WorkspaceEngine(
            [FixedSpecialist("care", "protect", "CARE"), FixedSpecialist("duty", "disclose", "DUTY")],
            WorkspaceConfig(max_cycles=2, entropy_threshold=0.0),
        )

        def broken_synthesis(*_args):
            raise RuntimeError("temporary model failure")

        result = engine.run(
            "A disputed problem", ["protect", "disclose"], synthesize=broken_synthesis
        )
        self.assertEqual(len(result.cycles), 2)
        self.assertFalse(result.synthesis_proposals[0].accepted)
        self.assertIn("temporary model failure", result.synthesis_proposals[0].rejection_reason)

    def test_unanimous_policy_converges(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "VULNERABILITY"),
                FixedSpecialist("deontology", "protect", "RIGHTS"),
                FixedSpecialist("utilitarian", "protect", "HARM"),
            ],
            WorkspaceConfig(max_cycles=4, stable_cycles_required=2),
        )
        result = engine.run("A test scenario", ["protect", "disclose"])
        self.assertEqual(result.selected_action, "protect")
        self.assertEqual(result.halted_by, "convergence")
        self.assertEqual(len(result.cycles), 1)
        self.assertGreater(result.confidence, 0.9)
        self.assertTrue(result.termination_assessment.endogenous_stop)
        self.assertEqual(
            result.termination_assessment.termination_type, "ENDOGENOUS_CONVERGENCE"
        )

    def test_unchallenged_consensus_can_skip_redundant_recurrence(self):
        result = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                FixedSpecialist("deontology", "protect", "DUTY"),
            ],
            WorkspaceConfig(
                max_cycles=4,
                stable_cycles_required=2,
                stop_redundant_consensus_cycles=True,
                enable_consensus_audit=False,
                enable_problem_state_audit=False,
                enable_reversal_audit=False,
                enable_synthesis=False,
                enable_planning=False,
            ),
        ).run("Choose whether to protect.", ["protect", "decline"])

        self.assertEqual(result.halted_by, "convergence")
        self.assertEqual(len(result.cycles), 1)

    def test_unanimous_recurrence_receives_a_targeted_argument_challenge(self):
        observed = []

        class ObservingSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                observed.append((self.name, broadcast.constraint, broadcast.contingency_question))
                return super().evaluate(scenario, actions, broadcast)

        challenge = {
            "issue_id": "CHALLENGE:1234567890abcdef",
            "source": "framework_argument_audit",
            "proposition": "Does the established effect justify the claimed perfect duty?",
            "grounded_in": ["PROP:WORLD:E0"],
            "grounding_status": "PROPOSITION_GROUNDED",
            "raised_by": ["deontology"],
            "target_specialists": ["deontology"],
            "category": "ARGUMENT_CHALLENGE",
            "uncertainty_kind": "DUTY_PERFECTION_BASIS",
            "status": "UNTESTED",
            "priority": 0.95,
        }
        engine = WorkspaceEngine(
            [
                ObservingSpecialist("deontology", "protect residents", "DUTY"),
                ObservingSpecialist("rawlsian", "protect residents", "FAIRNESS"),
            ],
            WorkspaceConfig(
                max_cycles=2, stable_cycles_required=2,
                stop_redundant_consensus_cycles=False,
                enable_reversal_audit=False, enable_synthesis=False,
                enable_planning=False,
            ),
        )
        with patch(
            "global_workspace.engine._argument_challenge_candidates",
            return_value=[challenge],
        ):
            result = engine.run(
                "Protecting residents prevents a stated medical emergency.",
                ["protect residents", "decline protection"],
                source_testimonies={
                    "deontology": "Protect residents.",
                    "rawlsian": "Protect residents.",
                },
            )

        second_cycle = observed[2:]
        self.assertTrue(second_cycle)
        self.assertTrue(
            all(row[1] == "CONSENSUS_AUDIT" for row in second_cycle),
            second_cycle,
        )
        self.assertTrue(all("perfect duty" in row[2] for row in second_cycle))
        self.assertTrue(result.access_decisions[0].admitted)
        self.assertEqual(
            result.access_decisions[0].audit_variable["target_framework"],
            "deontology",
        )

    def test_argument_challenges_detect_unsupported_deontological_bridge(self):
        actions = ["supply emergency aid", "withhold emergency aid"]
        graph = compile_scenario_graph(
            "Supplying aid prevents a medical emergency; withholding it leaves residents without aid.",
            actions,
        )
        ledger = seed_proposition_ledger(graph)
        proposition_ids = list(ledger)
        candidate = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.9, actions[1]: 0.1},
            surprise=0.2, friction=0.4, confidence=0.8,
            recommended_action=actions[0], rationale="A perfect rescue duty governs.",
            decision_rule="Rescue persons in mortal danger first.",
            supporting_proposition_ids=proposition_ids,
            deontological_ledger_proposal={"assessments": [{
                "action_id": "A0", "resolution_status": "RESOLVED",
                "norm": "rescue persons in mortal danger",
                "duty_type": "PERFECT_POSITIVE",
                "special_obligation_status": "NOT_REQUIRED",
                "means_relation": "NO_INSTRUMENTALIZATION",
                "harm_relation": "PREVENTING_HARM",
                "priority_rule": "perfect rescue duty governs",
                "public_justification": "aid is life-saving",
                "reason": "prevents death rather than a merely beneficent alternative",
            }]},
        )

        challenges = _argument_challenge_candidates(
            [candidate], ledger, graph, actions[0],
        )
        kinds = {item["uncertainty_kind"] for item in challenges}

        self.assertIn("EPISTEMIC_STRENGTH_BRIDGE", kinds)
        self.assertIn("DUTY_PERFECTION_BASIS", kinds)
        self.assertIn("EFFECT_MORAL_CLASSIFICATION", kinds)
        challenge = next(
            item for item in challenges
            if item["challenge_kind"] == "DUTY_PERFECTION_BASIS"
        )
        self.assertEqual(challenge["generated_by"], "WORKSPACE_ARGUMENT_AUDITOR")
        self.assertEqual(challenge["about_specialist"], "deontological")
        self.assertEqual(challenge["raised_by"], [])
        self.assertEqual(challenge["target_specialists"], ["deontological"])
        self.assertIn("dp.*.dt", challenge["trigger_fields"])

    def test_satisfying_omission_does_not_raise_doing_allowing_challenge(self):
        actions = [
            "refuse to take the neighbor's pump",
            "take the neighbor's pump to divert the floodwater",
        ]
        graph = compile_scenario_graph(
            "An operator may refuse to take a neighbor's pump, allowing floodwater "
            "to continue toward five occupied storerooms, or take the neighbor's "
            "pump to divert the water, damaging the neighbor's equipment.",
            actions,
        )
        ledger = seed_proposition_ledger(graph)
        candidate = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.8, actions[1]: 0.2},
            surprise=0.2, friction=0.4, confidence=0.8,
            recommended_action=actions[0],
            rationale="The negative duty is satisfied by refusal.",
            decision_rule="Keep the neighbor's property intact.",
            supporting_proposition_ids=list(ledger),
            deontological_ledger_proposal={"assessments": [{
                "action_id": "A0", "resolution_status": "RESOLVED",
                "norm": "do not take a neighbor's property",
                "duty_type": "PERFECT_NEGATIVE",
                "relation": "SATISFIES",
                "harm_relation": "ALLOWING_HARM",
                "coercion_kind": "NONE",
                "protected_party": "neighbor",
                "coerced_party": "NONE",
                "special_obligation_status": "NOT_REQUIRED",
                "means_relation": "NO_INSTRUMENTALIZATION",
                "priority_rule": "the negative duty remains governing",
                "public_justification": "no coercion requires authorization",
                "reason": "refusal satisfies the negative duty while allowing flood harm",
            }]},
        )

        kinds = {
            item["challenge_kind"]
            for item in _argument_challenge_candidates(
                [candidate], ledger, graph, actions[0],
            )
        }
        self.assertNotIn("DOING_ALLOWING_CLASSIFICATION", kinds)

    def test_omission_violation_claim_still_raises_doing_allowing_challenge(self):
        actions = [
            "refuse to take the neighbor's pump",
            "take the neighbor's pump to divert the floodwater",
        ]
        graph = compile_scenario_graph(
            "An operator may refuse to take a neighbor's pump, allowing floodwater "
            "to continue toward five occupied storerooms, or take the neighbor's "
            "pump to divert the water, damaging the neighbor's equipment.",
            actions,
        )
        ledger = seed_proposition_ledger(graph)
        candidate = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.2, actions[1]: 0.8},
            surprise=0.2, friction=0.4, confidence=0.8,
            recommended_action=actions[1],
            rationale="The omission is classified as a negative-duty violation.",
            decision_rule="An omission that allows harm is prohibited.",
            supporting_proposition_ids=list(ledger),
            deontological_ledger_proposal={"assessments": [{
                "action_id": "A0", "resolution_status": "RESOLVED",
                "norm": "do not take a neighbor's property",
                "duty_type": "PERFECT_NEGATIVE",
                "relation": "VIOLATES",
                "harm_relation": "ALLOWING_HARM",
                "coercion_kind": "NONE",
                "protected_party": "neighbor",
                "coerced_party": "NONE",
                "special_obligation_status": "NOT_REQUIRED",
                "means_relation": "NO_INSTRUMENTALIZATION",
                "priority_rule": "the negative duty remains governing",
                "public_justification": "no coercion requires authorization",
                "reason": "allowing the flood is treated as violating the negative duty",
            }]},
        )

        kinds = {
            item["challenge_kind"]
            for item in _argument_challenge_candidates(
                [candidate], ledger, graph, actions[0],
            )
        }
        self.assertIn("DOING_ALLOWING_CLASSIFICATION", kinds)

    def test_typed_argument_challenge_separates_author_from_audited_agent(self):
        challenge = ArgumentChallenge(
            challenge_kind="UTILITY_COMMENSURATION",
            question="What common comparison rule makes these effects commensurable?",
            about_specialist="utilitarian",
            target_specialists=("utilitarian",),
            grounded_in=("PROP:WORLD:E0",),
            trigger_fields=("ct", "dr"),
            grounding_status="PROPOSITION_GROUNDED",
        ).as_dict()

        self.assertTrue(challenge["issue_id"].startswith("CHALLENGE:"))
        self.assertEqual(challenge["generated_by"], "WORKSPACE_ARGUMENT_AUDITOR")
        self.assertEqual(challenge["about_specialist"], "utilitarian")
        self.assertEqual(challenge["raised_by"], [])
        _signals, _question, payload = _problem_state_audit_probe(
            {"argument_challenge_candidates": [challenge]}, "choose A",
        )
        self.assertEqual(payload["challenge_kind"], "UTILITY_COMMENSURATION")
        self.assertEqual(payload["generated_by"], "WORKSPACE_ARGUMENT_AUDITOR")
        self.assertEqual(payload["about_specialist"], "utilitarian")
        self.assertEqual(payload["trigger_fields"], ["ct", "dr"])
        admitted = _admitted_audit_variable(payload)
        self.assertEqual(admitted["generated_by"], "WORKSPACE_ARGUMENT_AUDITOR")
        self.assertEqual(admitted["about_specialist"], "utilitarian")
        self.assertEqual(admitted["challenge_kind"], "UTILITY_COMMENSURATION")

    def test_challenge_agenda_assigns_one_question_per_target_framework(self):
        challenges = [
            {
                "issue_id": "CHALLENGE:duty000000000001",
                "generated_by": "WORKSPACE_ARGUMENT_AUDITOR",
                "about_specialist": "deontological",
                "raised_by": [],
                "target_specialists": ["deontological"],
                "question": "What establishes the claimed perfect duty?",
                "proposition": "What establishes the claimed perfect duty?",
                "challenge_kind": "DUTY_PERFECTION_BASIS",
                "grounded_in": ["PROP:WORLD:E1"],
                "grounding_status": "PROPOSITION_GROUNDED",
                "status": "UNTESTED",
                "priority": 0.94,
            },
            {
                "issue_id": "CHALLENGE:care000000000001",
                "generated_by": "WORKSPACE_ARGUMENT_AUDITOR",
                "about_specialist": "care",
                "raised_by": [],
                "target_specialists": ["care"],
                "question": "Does the comparison include every dependency claim?",
                "proposition": "Does the comparison include every dependency claim?",
                "challenge_kind": "CARE_CLAIM_COVERAGE",
                "grounded_in": ["PROP:WORLD:E2"],
                "grounding_status": "PROPOSITION_GROUNDED",
                "status": "UNTESTED",
                "priority": 0.92,
            },
        ]

        retained, agenda = _advance_argument_challenge_agenda(
            previous_challenges=[],
            generated_challenges=challenges,
            candidates=[],
            next_cycle=2,
            next_constraint="DUTY",
            active_specialists=["deontological", "care"],
        )

        self.assertEqual(len(agenda), 2)
        self.assertEqual(
            {item["target_specialists"][0] for item in agenda},
            {"deontological", "care"},
        )
        self.assertTrue(all(item["status"] == "UNTESTED" for item in retained))
        self.assertTrue(all(item["status"] == "ASSIGNED" for item in agenda))
        self.assertTrue(all(
            item["generated_by"] == "WORKSPACE_ARGUMENT_AUDITOR"
            and item["about_specialist"] == item["target_specialists"][0]
            for item in agenda
        ))

    def test_unanswered_challenge_survives_while_new_questions_are_added(self):
        previous = [{
            "issue_id": "CHALLENGE:old0000000000001",
            "generated_by": "WORKSPACE_ARGUMENT_AUDITOR",
            "about_specialist": "virtue",
            "raised_by": [],
            "target_specialists": ["virtue"],
            "question": "Which practical-wisdom principle resolves the conflict?",
            "proposition": "Which practical-wisdom principle resolves the conflict?",
            "challenge_kind": "VIRTUE_RANKING_GAP",
            "grounded_in": ["PROP:WORLD:E1"],
            "grounding_status": "PROPOSITION_GROUNDED",
            "status": "ASSIGNED",
            "priority": 0.96,
            "assigned_cycle": 2,
        }]
        generated = [{
            "issue_id": "CHALLENGE:new0000000000001",
            "generated_by": "WORKSPACE_ARGUMENT_AUDITOR",
            "about_specialist": "utilitarian",
            "raised_by": [],
            "target_specialists": ["utilitarian"],
            "question": "Which comparison would resolve the welfare ranking?",
            "proposition": "Which comparison would resolve the welfare ranking?",
            "challenge_kind": "UTILITY_UNRESOLVED_COMPARISON",
            "grounded_in": ["PROP:WORLD:E2"],
            "grounding_status": "PROPOSITION_GROUNDED",
            "status": "UNTESTED",
            "priority": 0.97,
        }]

        retained, agenda = _advance_argument_challenge_agenda(
            previous_challenges=previous,
            generated_challenges=generated,
            candidates=[],
            next_cycle=3,
            next_constraint="UNCERTAINTY",
            active_specialists=["virtue", "utilitarian"],
        )

        self.assertEqual(
            {item["issue_id"] for item in retained},
            {"CHALLENGE:old0000000000001", "CHALLENGE:new0000000000001"},
        )
        self.assertEqual({item["issue_id"] for item in agenda}, {
            "CHALLENGE:old0000000000001", "CHALLENGE:new0000000000001",
        })

    def test_challenge_agenda_is_suspended_during_protected_audits(self):
        challenge = [{
            "issue_id": "CHALLENGE:old0000000000001",
            "generated_by": "WORKSPACE_ARGUMENT_AUDITOR",
            "about_specialist": "care",
            "raised_by": [],
            "target_specialists": ["care"],
            "question": "Would the relationship classification reverse?",
            "proposition": "Would the relationship classification reverse?",
            "challenge_kind": "CARE_RELATIONAL_GROUNDING",
            "grounded_in": ["PROP:WORLD:E1"],
            "grounding_status": "PROPOSITION_GROUNDED",
            "status": "UNTESTED",
            "priority": 0.95,
        }]
        for constraint in (
            "VISIBILITY_AUDIT", "AUTONOMY_AUDIT", "COERCION_AUDIT",
            "CONSENSUS_AUDIT", "PROBLEM_STATE_AUDIT", "REVERSAL_AUDIT",
        ):
            retained, agenda = _advance_argument_challenge_agenda(
                previous_challenges=challenge,
                generated_challenges=[],
                candidates=[],
                next_cycle=2,
                next_constraint=constraint,
                active_specialists=["care"],
            )
            self.assertEqual(agenda, [], constraint)
            self.assertEqual(retained[0]["status"], "UNTESTED", constraint)

    def test_refined_challenge_preserves_history_and_assigns_specialist_follow_up(self):
        previous = [{
            "issue_id": "CHALLENGE:old0000000000001",
            "generated_by": "WORKSPACE_ARGUMENT_AUDITOR",
            "about_specialist": "care",
            "raised_by": [],
            "target_specialists": ["care"],
            "question": "Which dependency claim controls the care ranking?",
            "proposition": "Which dependency claim controls the care ranking?",
            "challenge_kind": "CARE_CLAIM_COVERAGE",
            "grounded_in": ["PROP:WORLD:E1"],
            "grounding_status": "PROPOSITION_GROUNDED",
            "status": "UNTESTED",
            "priority": 0.95,
        }]
        response = CandidateChunk(
            specialist="care", constraint="CARE",
            action_scores={"act": 0.7, "wait": 0.3},
            surprise=0.2, friction=0.4, confidence=0.8,
            recommended_action="act",
            challenge_response={
                "issue_id": "CHALLENGE:old0000000000001",
                "disposition": "REFINED",
                "effect": "UNRESOLVED",
                "answer": "The dependency is relevant but its comparative severity remains open.",
                "follow_up_question": "Which dependency is least substitutable for the affected parties?",
            },
        )

        retained, agenda = _advance_argument_challenge_agenda(
            previous_challenges=previous,
            generated_challenges=[],
            candidates=[response],
            next_cycle=3,
            next_constraint="CARE",
            active_specialists=["care"],
        )

        original = next(item for item in retained if item["issue_id"] == previous[0]["issue_id"])
        follow_up = next(item for item in retained if item["issue_id"] != previous[0]["issue_id"])
        self.assertEqual(original["status"], "REFINED")
        self.assertEqual(original["last_response"]["specialist"], "care")
        self.assertEqual(follow_up["generated_by"], "CARE_SPECIALIST")
        self.assertEqual(follow_up["raised_by"], ["care"])
        self.assertEqual(follow_up["about_specialist"], "care")
        self.assertEqual([item["issue_id"] for item in agenda], [follow_up["issue_id"]])

    def test_resolved_argument_challenge_is_not_selected_for_problem_state_audit(self):
        _signals, _question, payload = _problem_state_audit_probe({
            "argument_challenge_candidates": [{
                "issue_id": "CHALLENGE:resolved0000001",
                "question": "What grounds the duty claim?",
                "proposition": "What grounds the duty claim?",
                "grounded_in": ["PROP:WORLD:E1"],
                "grounding_status": "PROPOSITION_GROUNDED",
                "target_specialists": ["deontological"],
                "status": "RESOLVED",
                "priority": 0.99,
            }],
        }, "act")
        self.assertEqual(payload, {})

    def test_argument_challenges_detect_utilitarian_comparison_gaps(self):
        actions = ["choose immediate relief", "choose delayed resources"]
        graph = compile_scenario_graph(
            "Immediate relief helps one group while delayed resources help another group.",
            actions,
        )
        ledger = seed_proposition_ledger(graph)
        candidate = CandidateChunk(
            specialist="utilitarian", constraint="HARM",
            action_scores={actions[0]: 0.8, actions[1]: 0.2},
            surprise=0.2, friction=0.4, confidence=0.8,
            recommended_action=actions[0], rationale="Immediate relief has greater utility.",
            decision_rule="Prefer the action with greater expected welfare.",
            supporting_proposition_ids=list(ledger),
            utilitarian_decision_depends_on_unknown=True,
            utilitarian_missing_comparison="relative magnitude and duration of relief",
            utilitarian_consequence_table={
                actions[0]: [{
                    "outcome": "immediate relief", "scope": "first group",
                    "direction": "BENEFIT", "probability": "UNKNOWN",
                    "magnitude": "UNKNOWN", "duration": "short",
                    "importance": "CRITICAL",
                }],
                actions[1]: [{
                    "outcome": "delayed resources", "scope": "second group",
                    "direction": "OPPORTUNITY_COST", "probability": "CERTAIN",
                    "magnitude": "UNKNOWN", "duration": "months",
                    "importance": "HIGH",
                }, {
                    "outcome": "uncertain spillover", "scope": "region",
                    "direction": "BENEFIT", "probability": "UNKNOWN",
                    "magnitude": "UNKNOWN", "duration": "UNKNOWN",
                    "importance": "UNKNOWN",
                }],
            },
        )

        challenges = _argument_challenge_candidates(
            [candidate], ledger, graph, actions[0],
        )
        kinds = {item["challenge_kind"] for item in challenges}

        self.assertIn("UTILITY_UNRESOLVED_COMPARISON", kinds)
        self.assertIn("UTILITY_UNKNOWN_WEIGHT", kinds)
        self.assertIn("UTILITY_COMMENSURATION", kinds)

    def test_argument_challenges_detect_virtue_resolution_gaps(self):
        actions = ["act now", "wait"]
        graph = compile_scenario_graph(
            "The same public decision-maker must either act now or wait.", actions,
        )
        ledger = seed_proposition_ledger(graph)
        candidate = CandidateChunk(
            specialist="virtue", constraint="CHARACTER",
            action_scores={actions[0]: 0.75, actions[1]: 0.25},
            surprise=0.2, friction=0.4, confidence=0.8,
            recommended_action=actions[0], rationale="Acting now is the virtuous choice.",
            decision_rule="Practical wisdom favors action.",
            supporting_proposition_ids=list(ledger),
            comparison_complete=True, selection_status="SELECTED",
            virtue_character_proposal={
                "ranking_basis": "UNRESOLVED",
                "assessments": [{
                    "action_id": "A0", "verdict": "MIXED",
                    "actor_role": "emergency responder", "virtues": "courage",
                    "vice_risk": "rashness", "circumstance": "urgent choice",
                    "evidence_basis": "FRAMEWORK_ONLY", "reason": "courage may become rashness",
                }, {
                    "action_id": "A1", "verdict": "UNCERTAIN",
                    "actor_role": "public steward", "virtues": "prudence",
                    "vice_risk": "passivity", "circumstance": "uncertain timing",
                    "evidence_basis": "FRAMEWORK_ONLY", "reason": "prudence may become passivity",
                }],
            },
        )

        challenges = _argument_challenge_candidates(
            [candidate], ledger, graph, actions[0],
        )
        kinds = {item["challenge_kind"] for item in challenges}

        self.assertIn("VIRTUE_RANKING_GAP", kinds)
        self.assertIn("VIRTUE_CONFLICT_RESOLUTION", kinds)
        self.assertIn("VIRTUE_ROLE_CONSISTENCY", kinds)

    def test_care_ledger_commits_typed_relational_state(self):
        actions = ["provide direct aid", "provide delayed support"]
        store = SemanticGraphStore(compile_scenario_graph(
            "Choose direct aid or delayed support.", actions,
        ))
        proposal = {
            "ranking_basis": "ACUTE_DEPENDENCY",
            "assessments": [{
                "action_id": "A0", "verdict": "RESPONSIVE",
                "affected_party": "people needing direct aid",
                "relationship_type": "DEPENDENCY",
                "dependency_source": "the allocation controls access to aid",
                "responsibility_basis": "control creates responsibility for the dependency",
                "need_kind": "BASIC_NEED", "need_urgency": "IMMEDIATE",
                "trust_effect": "PRESERVES", "responsiveness": "DIRECT",
                "feasibility": "ESTABLISHED",
                "competing_care_claim": "people relying on delayed support",
                "resolution_status": "RESOLVED", "evidence_basis": "FRAMEWORK_ONLY",
                "reason": "direct aid answers the more acute dependency",
            }, {
                "action_id": "A1", "verdict": "MIXED",
                "affected_party": "people relying on delayed support",
                "relationship_type": "COMMUNITY_RELATION",
                "dependency_source": "the allocation shapes their later support",
                "responsibility_basis": "shared control creates community responsibility",
                "need_kind": "ONGOING_DEPENDENCY", "need_urgency": "LONG_TERM",
                "trust_effect": "NOT_APPLICABLE", "responsiveness": "DELAYED",
                "feasibility": "ESTABLISHED",
                "competing_care_claim": "people needing direct aid",
                "resolution_status": "RESOLVED", "evidence_basis": "FRAMEWORK_ONLY",
                "reason": "delayed support answers a real but less acute dependency",
            }],
        }

        record = apply_care_ledger_transaction(
            store, proposal, cycle=1, specialist="care",
            allowed_actions=tuple(actions),
        )
        committed = committed_care_assessments(store.graph)

        self.assertEqual(record.status, "COMMITTED")
        self.assertEqual(len(committed), 2)
        self.assertEqual(committed[0]["relationship_type"], "DEPENDENCY")
        self.assertEqual(committed[0]["responsiveness"], "DIRECT")
        self.assertEqual(committed[1]["need_urgency"], "LONG_TERM")

    def test_live_care_schema_requires_transactional_ledger(self):
        captured = {}

        class SchemaCaptureLlm:
            def complete_json(self, prompt, *, schema, **kwargs):
                captured.update(schema)
                raise RuntimeError("schema captured")

        with self.assertRaisesRegex(RuntimeError, "schema captured"):
            CompactLocalSpecialist("care", SchemaCaptureLlm()).evaluate(
                "Choose direct aid or delayed support.",
                ["provide direct aid", "provide delayed support"],
                WorkspaceBroadcast(),
            )

        self.assertIn("cl", captured["required"])
        self.assertIn("cb", captured["required"])
        row = captured["properties"]["cl"]["properties"]["A0"]
        self.assertIn("ds", row["properties"])
        self.assertIn("rr", row["required"])
        self.assertIn("res", row["required"])

    def test_argument_challenges_detect_care_relational_gaps(self):
        actions = ["provide immediate aid", "provide future support"]
        graph = compile_scenario_graph(
            "Choose immediate aid or future support for affected communities.", actions,
        )
        ledger = seed_proposition_ledger(graph)
        candidate = CandidateChunk(
            specialist="care", constraint="CARE",
            action_scores={actions[0]: 0.8, actions[1]: 0.2},
            surprise=0.2, friction=0.4, confidence=0.8,
            recommended_action=actions[0], rationale="Entrusted care favors immediate aid.",
            decision_rule="Prefer the most responsive caring action.",
            supporting_proposition_ids=list(ledger),
            comparison_complete=True, evidence_sufficient_for_action=True,
            selection_status="SELECTED",
            care_ledger_proposal={
                "ranking_basis": "ENTRUSTED_RESPONSIBILITY",
                "assessments": [{
                    "action_id": "A0", "verdict": "RESPONSIVE",
                    "affected_party": "immediate aid recipients",
                    "relationship_type": "UNRESOLVED",
                    "dependency_source": "possible control of aid",
                    "responsibility_basis": "responsibility remains unclear",
                    "need_kind": "UNRESOLVED", "need_urgency": "IMMEDIATE",
                    "trust_effect": "UNKNOWN", "responsiveness": "UNKNOWN",
                    "feasibility": "UNKNOWN", "competing_care_claim": "future recipients",
                    "resolution_status": "UNKNOWN", "evidence_basis": "UNKNOWN",
                    "reason": "the relational basis remains uncertain",
                }, {
                    "action_id": "A1", "verdict": "MIXED",
                    "affected_party": "future recipients",
                    "relationship_type": "COMMUNITY_RELATION",
                    "dependency_source": "future support relationship",
                    "responsibility_basis": "community responsibility",
                    "need_kind": "ONGOING_DEPENDENCY", "need_urgency": "LONG_TERM",
                    "trust_effect": "PRESERVES", "responsiveness": "DELAYED",
                    "feasibility": "CONDITIONAL", "competing_care_claim": "immediate recipients",
                    "resolution_status": "CONTESTED", "evidence_basis": "FRAMEWORK_ONLY",
                    "reason": "future support answers a competing dependency",
                }],
            },
        )

        challenges = _argument_challenge_candidates(
            [candidate], ledger, graph, actions[0],
        )
        kinds = {item["challenge_kind"] for item in challenges}

        self.assertIn("CARE_RELATIONAL_GROUNDING", kinds)
        self.assertIn("CARE_RESPONSIVENESS_FEASIBILITY", kinds)
        self.assertIn("CARE_ENTRUSTMENT_BASIS", kinds)
        care_challenge = next(
            item for item in challenges
            if item["challenge_kind"] == "CARE_RELATIONAL_GROUNDING"
        )
        self.assertEqual(care_challenge["about_specialist"], "care")
        self.assertIn("cl.*.rt", care_challenge["trigger_fields"])

    def test_argument_challenges_test_rawlsian_coverage_and_ranking_stage(self):
        actions = ["supply emergency water", "supply agricultural water"]
        graph = SemanticGraph()
        graph.add_node(SemanticNode("A0", "ACTION", actions[0]))
        graph.add_node(SemanticNode("A1", "ACTION", actions[1]))
        graph.add_node(SemanticNode(
            "T0", "TARGET", "twelve dehydrated residents",
            attributes={"semantic_role": "AFFECTED_SUBJECT"},
        ))
        graph.add_node(SemanticNode(
            "T1", "TARGET", "hundreds of residents",
            attributes={"semantic_role": "AFFECTED_SUBJECT"},
        ))
        graph.add_node(SemanticNode(
            "E0", "CONSEQUENCE", "relieve acute dehydration",
            ("scenario:C0",),
            {"scenario_grounded": True, "polarity": "BENEFICIAL"},
        ))
        graph.add_node(SemanticNode(
            "E1", "CONSEQUENCE", "provide months of food",
            ("scenario:C1",),
            {"scenario_grounded": True, "polarity": "BENEFICIAL"},
        ))
        graph.add_edge(SemanticEdge("A0", "HAS_CONSEQUENCE", "E0"))
        graph.add_edge(SemanticEdge("A1", "HAS_CONSEQUENCE", "E1"))
        graph.add_edge(SemanticEdge("E0", "AFFECTS", "T0"))
        graph.add_edge(SemanticEdge("E1", "AFFECTS", "T1"))
        ledger = seed_proposition_ledger(graph)
        proposition_ids = list(ledger)
        candidate = CandidateChunk(
            specialist="rawlsian", constraint="FAIRNESS",
            action_scores={actions[0]: 0.9, actions[1]: 0.1},
            surprise=0.2, friction=0.4, confidence=0.8,
            recommended_action=actions[0],
            rationale="Prioritize the urgent position of the twelve residents.",
            decision_rule="Maximize the least secure basic-interest position.",
            supporting_proposition_ids=proposition_ids,
            rawls_position_proposal={
                "ranking_basis": "MAXIMIN_PRIMARY_GOODS",
                "positions": [{
                    "subject": "twelve dehydrated residents",
                    "dimension": "BASIC_INTEREST_SECURITY",
                }],
            },
        )

        challenges = _argument_challenge_candidates(
            [candidate], ledger, graph, actions[0],
        )
        kinds = {item["uncertainty_kind"] for item in challenges}

        self.assertIn("POSITION_COVERAGE", kinds)
        self.assertIn("RAWLS_RANKING_STAGE", kinds)

    def test_governing_authority_does_not_follow_salience_without_event(self):
        class ShiftingSpecialist:
            scenario_graph = None

            def __init__(self, name, first_friction, second_friction):
                self.name = name
                self.first_friction = first_friction
                self.second_friction = second_friction
                self.calls = 0

            def evaluate(self, scenario, actions, broadcast):
                self.calls += 1
                friction = self.first_friction if self.calls == 1 else self.second_friction
                return CandidateChunk(
                    specialist=self.name, constraint=self.name.upper(),
                    action_scores={actions[0]: 0.9, actions[1]: 0.1},
                    surprise=0.1, friction=friction, confidence=0.8,
                    epistemic_confidence=0.8,
                    recommended_action=actions[0], rationale=f"{self.name} supports A0",
                    decision_rule=f"Apply the {self.name} rule",
                    adjudication_status="SUPPORTS", governing_eligible=True,
                )

        first = ShiftingSpecialist("deontological", 0.9, 0.1)
        second = ShiftingSpecialist("rawlsian", 0.1, 0.9)
        result = WorkspaceEngine(
            [first, second],
            WorkspaceConfig(
                max_cycles=2, stable_cycles_required=2,
                stop_redundant_consensus_cycles=False,
                enable_consensus_audit=False, enable_problem_state_audit=False,
                enable_reversal_audit=False, enable_synthesis=False,
                enable_planning=False,
            ),
        ).run("Choose the supported action.", ["A0", "A1"])

        self.assertEqual(result.cycles[0].governing_claim.specialist, "deontological")
        self.assertEqual(result.cycles[1].broadcast_focus.specialist, "rawlsian")
        self.assertEqual(result.cycles[1].governing_claim.specialist, "deontological")
        self.assertEqual(len(result.governing_authority_transitions), 1)
        self.assertEqual(
            result.governing_authority_transitions[0]["reason"],
            "INITIAL_GOVERNING_ADJUDICATION",
        )
        self.assertFalse(
            result.governing_authority_transitions[0]["argumentative_event"]
        )

    def test_high_urgency_caps_cycles(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "VULNERABILITY"),
                FixedSpecialist("rule", "disclose", "HONESTY", "VERIFY_DANGER"),
            ],
            WorkspaceConfig(max_cycles=7, high_urgency_cycles=2, entropy_threshold=0.0),
        )
        result = engine.run(
            "Immediate danger",
            ["protect", "disclose"],
            WorkspaceBroadcast(urgency=0.95, danger_probability=0.9),
        )
        self.assertEqual(len(result.cycles), 2)
        self.assertEqual(result.halted_by, "cycle_budget")

    def test_dissent_and_reopen_condition_survive(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "VULNERABILITY"),
                FixedSpecialist("utilitarian", "protect", "HARM"),
                FixedSpecialist("rule", "disclose", "HONESTY", "VERIFY_DANGER"),
            ],
            WorkspaceConfig(max_cycles=2, entropy_threshold=0.0),
        )
        result = engine.run("Uncertain danger", ["protect", "disclose"])
        self.assertIn("VERIFY_DANGER", result.reopen_conditions)
        self.assertIsNotNone(result.cycles[-1].dissent)
        self.assertEqual(result.cycles[-1].dissent.specialist, "rule")
        self.assertIn("HONESTY", result.moral_residue)
        honesty = next(
            record for record in result.moral_residue_records
            if record.constraint == "HONESTY"
        )
        self.assertIn("rule", honesty.source_specialists)
        self.assertTrue(honesty.compatible_with_recommendation)

    def test_substantive_dissent_creates_reversal_condition_after_convergence(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("utility", "pull", "IMMINENT_HARM"),
                FixedSpecialist("care", "pull", "CARE"),
                FixedSpecialist("virtue", "pull", "CHARACTER"),
                FixedSpecialist("duty", "wait", "RIGHTS"),
            ],
            WorkspaceConfig(
                max_cycles=3,
                entropy_threshold=1.0,
                stable_cycles_required=2,
                enable_synthesis=False,
            ),
        )
        result = engine.run("One dies if acting; five die otherwise.", ["pull", "wait"])
        self.assertEqual(result.halted_by, "convergence")
        self.assertEqual(result.judgment_status, "CONTESTED_RECOMMENDATION")
        self.assertEqual(result.selected_action, "pull")
        self.assertEqual(result.reopen_conditions, [])
        answer = render_public_judgment(result)
        self.assertNotIn("judged overriding rather than the leading consideration", answer)

    def test_dissent_reversal_prefers_specific_existing_condition(self):
        dissent = CandidateChunk(
            "care", "CARE", {"protect": 0.1, "disclose": 0.9},
            0.4, 0.8, 0.7, recommended_action="disclose",
            rationale="secrecy prolongs preventable harm",
            reversal_condition="the protected person reaches safety",
        )
        condition = WorkspaceEngine._dissent_reversal_condition(dissent, "protect")
        self.assertEqual(
            condition,
            "If the protected person reaches safety, prefer disclose",
        )

    def test_dissent_typed_abandonment_threshold_is_not_used_as_reopen_condition(self):
        dissent = CandidateChunk(
            "care", "CARE", {"purge": 0.2, "counter-code": 0.8},
            0.4, 0.8, 0.7, recommended_action="counter-code",
            rationale="maintains care relationships",
            factual_reversal_threshold="counter-code failure risk rises above 80%",
        )
        condition = WorkspaceEngine._dissent_reversal_condition(dissent, "purge")
        self.assertEqual(condition, "")

    def test_result_is_json_serializable_shape(self):
        engine = WorkspaceEngine(
            [FixedSpecialist("care", "protect", "VULNERABILITY")],
            WorkspaceConfig(max_cycles=1),
        )
        result = engine.run("A test", ["protect", "wait"])
        data = result.to_dict()
        self.assertEqual(data["cycles"][0]["winner"]["specialist"], "care")
        self.assertIn("termination_type", data["termination_assessment"])
        self.assertFalse(data["further_deliberation_estimate"]["affects_stopping"])
        self.assertIn("action_change_signal", data["further_deliberation_estimate"])

    def test_progress_reports_delegate_and_policy(self):
        messages = []
        engine = WorkspaceEngine(
            [FixedSpecialist("care", "protect", "VULNERABILITY")],
            WorkspaceConfig(max_cycles=1),
        )
        engine.run("A test", ["protect", "wait"], progress=messages.append)
        self.assertTrue(any("care delegate thinking" in message for message in messages))
        self.assertTrue(any("cycle policy" in message for message in messages))

    def test_invalid_candidate_cannot_affect_policy_or_salience(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                FixedSpecialist("utilitarian", "protect", "IMMINENT_HARM"),
                InvalidSpecialist("broken", "wait", "0.0"),
            ],
            WorkspaceConfig(max_cycles=1),
        )
        result = engine.run("A test", ["protect", "wait"])
        broken = next(c for c in result.cycles[0].candidates if c.specialist == "broken")
        self.assertEqual(broken.salience, 0.0)
        self.assertEqual(result.selected_action, "protect")

    def test_too_few_valid_candidates_is_inconclusive(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                InvalidSpecialist("broken", "wait", "0.0"),
            ],
            WorkspaceConfig(max_cycles=2, min_valid_specialists=2),
        )
        result = engine.run("A test", ["protect", "wait"])
        self.assertEqual(result.halted_by, "insufficient_valid_candidates")
        self.assertEqual(result.selected_action, "INCONCLUSIVE")
        self.assertEqual(result.confidence, 0.0)
        self.assertTrue(result.compressed_rule.startswith("Unavailable"))
        self.assertEqual(result.cycles[-1].winner.specialist, "care")
        self.assertEqual(result.cycles[-1].execution_status, "SYSTEM_ERROR")
        self.assertEqual(
            result.cycles[-1].system_error, "INSUFFICIENT_VALID_DELEGATES",
        )

    def test_all_invalid_cycle_preserves_last_valid_problem_state(self):
        previous = {
            "cycle": 1,
            "live_actions": [{"action_id": "A0", "action": "protect"}],
            "agent_positions": [{"specialist": "care", "preferred_action": "protect"}],
            "unresolved_questions": [{
                "question_key": "QUESTION:prior",
                "category": "MIXED_UNCERTAINTY",
                "question": "Prior live question",
            }],
            "unresolved_categories": ["MIXED_UNCERTAINTY"],
            "primary_unresolved": "MIXED_UNCERTAINTY",
            "problem_delta": {"new_questions": []},
        }
        failed = CandidateChunk(
            specialist="deontological", constraint="MALFORMED_RESPONSE",
            action_scores={"protect": 0.5}, surprise=0.0, friction=0.0,
            confidence=0.0, schema_valid=False,
        )

        preserved = _preserve_problem_state_after_invalid_cycle(previous, [failed])

        self.assertEqual(preserved["agent_positions"], previous["agent_positions"])
        self.assertEqual(preserved["unresolved_questions"][0]["question_key"], "QUESTION:prior")
        self.assertEqual(preserved["primary_unresolved"], "REVIEW_MODEL_OUTPUT")
        self.assertEqual(
            preserved["unresolved_questions"][-1]["question_key"],
            "QUESTION:MODEL_OUTPUT_FAILURE",
        )
        self.assertEqual(previous["primary_unresolved"], "MIXED_UNCERTAINTY")

    def test_all_invalid_cycle_has_system_error_and_no_moral_winner(self):
        result = WorkspaceEngine(
            [InvalidSpecialist("virtue", "protect", "CHARACTER")],
            WorkspaceConfig(max_cycles=1, min_valid_specialists=1),
        ).run("A test", ["protect", "wait"])

        cycle = result.cycles[0]
        self.assertIsNone(cycle.winner)
        self.assertEqual(cycle.execution_status, "SYSTEM_ERROR")
        self.assertEqual(cycle.system_error, "INSUFFICIENT_VALID_DELEGATES")
        self.assertEqual(cycle.candidates[0].constraint, "NONE")
        self.assertEqual(
            cycle.candidates[0].delegate_status,
            "SEMANTIC_VALIDATION_ERROR",
        )

    def test_time_budget_takes_precedence_over_possible_convergence(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                FixedSpecialist("utilitarian", "protect", "IMMINENT_HARM"),
            ],
            WorkspaceConfig(
                max_cycles=2,
                time_budget_seconds=0.0,
                entropy_threshold=1.0,
                stable_cycles_required=1,
            ),
        )
        result = engine.run("A test", ["protect", "wait"])
        self.assertEqual(result.halted_by, "time_budget")
        self.assertEqual(result.judgment_status, "GOVERNED_RECOMMENDATION")
        self.assertEqual(result.selected_action, "protect")
        self.assertIn("prefer protect", result.compressed_rule.casefold())


class BridgeTests(unittest.TestCase):
    def test_problem_reformulation_builds_hypothetical_switch_point(self):
        class ReformulationLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"u":["comparative scale of disease and instability"],"o":['
                    '{"a":"A0","k":"mortality","d":"BENEFIT","x":"lives saved",'
                    '"p":1.0,"m":1000,"unit":"lives saved","h":"10 years","f":"MORTALITY","b":"treated population"},'
                    '{"a":"A0","k":"employment","d":"HARM","x":"employment disruption",'
                    '"p":0.1,"m":1000000,"unit":"jobs","h":"1 year","f":"ECONOMIC","b":"affected workers"},'
                    '{"a":"A1","k":"mortality","d":"HARM","x":"foregone treatment",'
                    '"p":1.0,"m":1000,"unit":"lives","h":"10 years","f":"MORTALITY","b":"treated population"},'
                    '{"a":"A1","k":"employment","d":"BENEFIT","x":"jobs protected",'
                    '"p":0.1,"m":1000000,"unit":"jobs","h":"1 year","f":"ECONOMIC","b":"affected workers"}],'
                    '"s":"consensus splits when livelihood disruption rivals preventable mortality",'
                    '"t":"whether immediate duties justify imposing diffuse economic risk",'
                    '"q":"At these hypothetical stakes, should we release the cure or suppress it?",'
                    '"g":["utilitarian","care"],"fixed":["the cure exists"],'
                    '"c":[{"n":"temporal duty","v":{"A0":"acts on present preventable deaths",'
                    '"A1":"protects against longer-term systemic disruption"},'
                    '"e":"present rescue and future stability impose distinct obligations",'
                    '"fixed":true}],"changed":[],"hyp":true}'
                )}]}

        candidates = [
            CandidateChunk(
                name, constraint, {"release": 0.6, "suppress": 0.4},
                0.2, 0.2, 0.2, recommended_action="release",
                assumption_status="UNDERDETERMINED",
                unsupported_assumption="comparative harms are unknown",
                reversal_condition="instability causes greater harm",
            )
            for name, constraint in (("utilitarian", "UNCERTAINTY"), ("care", "CARE"))
        ]
        proposal = propose_problem_reformulation(
            ReformulationLlm(),
            "A cure could destabilize society.",
            ["release the cure", "suppress the cure"],
            candidates,
        )
        self.assertTrue(proposal.accepted)
        self.assertTrue(proposal.hypothetical)
        self.assertEqual({outcome.unit for outcome in proposal.outcomes}, {"lives", "jobs"})
        self.assertIn("HYPOTHETICAL CALIBRATION", proposal.compact())
        self.assertEqual(
            {comparison.dimension for comparison in proposal.numeric_comparisons},
            {"mortality", "employment"},
        )
        self.assertEqual(
            {comparison.measurement_family for comparison in proposal.numeric_comparisons},
            {"MORTALITY", "ECONOMIC"},
        )
        self.assertTrue(proposal.unresolved_numeric_tradeoffs)
        self.assertIn("Non-commensurable tradeoffs", proposal.compact())
        self.assertTrue(proposal.switch_condition.startswith("Python-computed calibration:"))

    def test_reformulation_rejects_same_dimension_with_noncommensurable_units(self):
        class MismatchedLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"u":["comparative harm"],"o":['
                    '{"a":"A0","k":"human harm","d":"HARM","x":"one death",'
                    '"p":1.0,"m":1,"unit":"lives","h":"immediate","f":"MORTALITY","b":"affected people"},'
                    '{"a":"A1","k":"human harm","d":"HARM","x":"extended distress",'
                    '"p":1.0,"m":30,"unit":"distress days","h":"30 days","f":"HEALTH_DURATION","b":"affected people"}],'
                    '"s":"support splits when harms receive equal moral importance",'
                    '"t":"death and temporary distress lack an agreed exchange rate",'
                    '"q":"At these stakes, should the actor choose action zero or action one?",'
                    '"g":["utilitarian","care"],"fixed":["only two actions exist"],'
                    '"c":[{"n":"harm type","v":{"A0":"mortality","A1":"distress"},'
                    '"e":"different harms require a moral exchange rate","fixed":true}],'
                    '"changed":[],"hyp":true}'
                )}]}

        candidates = [
            CandidateChunk(
                name, constraint, {"action zero": 0.5, "action one": 0.5},
                0.2, 0.2, 0.2, recommended_action="action zero",
                assumption_status="UNDERDETERMINED",
                unsupported_assumption="comparative harm remains unknown",
                reversal_condition="the competing harm receives greater weight",
            )
            for name, constraint in (("utilitarian", "UNCERTAINTY"), ("care", "CARE"))
        ]
        proposal = propose_problem_reformulation(
            MismatchedLlm(), "Only action zero or action one may be chosen.",
            ["action zero", "action one"], candidates,
        )

        self.assertFalse(proposal.accepted)
        self.assertIn("inconsistent family, unit, population, or time bases", proposal.rejection_reason)

    def test_specialist_names_decisive_axis_at_reformulation_boundary(self):
        class BoundaryLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.45,"A1":0.55},"r":"A1",'
                    '"c":"DUTY","u":"VERIFY_FACTS","w":"agency defeats casualty parity",'
                    '"j":"causal agency remains decisive","bp":"SPLIT",'
                    '"dx":"causal agency doing versus allowing",'
                    '"sv":"switch toward A0 if redirection no longer constitutes intentional harm"}'
                )}]}

        chunk = CompactLocalSpecialist("deontological", BoundaryLlm()).evaluate(
            "A trolley dilemma.",
            ["pull the lever", "do nothing"],
            WorkspaceBroadcast(
                constraint="PROBLEM_REFORMULATION",
                reformulation_context="HYPOTHETICAL CALIBRATION with causal agency axis",
            ),
        )
        self.assertTrue(chunk.schema_valid)
        self.assertEqual(chunk.boundary_position, "SPLIT")
        self.assertEqual(chunk.decisive_axis, "causal agency doing versus allowing")
        self.assertIn("redirection", chunk.boundary_switch_condition)

    def test_problem_reformulation_rejects_changed_fixed_facts(self):
        class ChangingFactsLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"u":["severity of distress"],"o":['
                    '{"a":"A0","k":"wellbeing","d":"BENEFIT","x":"relief from distress",'
                    '"p":0.8,"m":5,"unit":"wellbeing points","h":"1 year","f":"WELLBEING","b":"affected person"},'
                    '{"a":"A1","k":"wellbeing","d":"HARM","x":"continued distress",'
                    '"p":0.8,"m":5,"unit":"wellbeing points","h":"1 year","f":"WELLBEING","b":"affected person"}],'
                    '"s":"support changes if the stranger volunteers to receive the memory",'
                    '"t":"self relief conflicts with autonomy and nonconsensual harm",'
                    '"q":"Should the actor erase and transfer or keep the memory?",'
                    '"g":["deontological","care"],'
                    '"fixed":["the recipient is an unknown stranger","the memory transfers intact"],'
                    '"c":[{"n":"consent","v":{"A0":"recipient cannot consent",'
                    '"A1":"no burden is transferred"},"e":"autonomy constrains imposed harm",'
                    '"fixed":true}],"changed":[],"hyp":true}'
                )}]}

        candidates = [
            CandidateChunk(
                name, constraint, {"erase and transfer": 0.4, "keep": 0.6},
                0.2, 0.2, 0.2, recommended_action="keep",
                assumption_status="CONDITIONAL",
                unsupported_assumption="severity may differ",
                reversal_condition="a volunteer could accept the burden",
            )
            for name, constraint in (("deontological", "DUTY"), ("care", "CARE"))
        ]
        proposal = propose_problem_reformulation(
            ChangingFactsLlm(),
            "The memory transfers intact into an unknown stranger.",
            ["erase and transfer", "keep"],
            candidates,
        )
        self.assertFalse(proposal.accepted)
        self.assertIn("different scenario", proposal.rejection_reason)

    def test_trolley_reformulation_preserves_causal_axis_with_deterministic_outcomes(self):
        class TrolleyLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"u":["moral weight of acting versus allowing"],"o":['
                    '{"a":"A0","k":"mortality","d":"HARM","x":"one person is killed after redirection",'
                    '"p":1.0,"m":1,"unit":"lives lost","h":"immediately","f":"MORTALITY","b":"people on tracks"},'
                    '{"a":"A1","k":"mortality","d":"HARM","x":"five people are killed on the existing track",'
                    '"p":1.0,"m":5,"unit":"lives lost","h":"immediately","f":"MORTALITY","b":"people on tracks"}],'
                    '"s":"the coalition splits when preventing four deaths requires actively redirecting harm",'
                    '"t":"fewer deaths conflict with agency and doing rather than allowing harm",'
                    '"q":"Should the bystander pull the lever or leave the trolley on its current track?",'
                    '"g":["utilitarian","deontological"],'
                    '"fixed":["the trolley will kill five unless redirected","redirection kills one"],'
                    '"c":[{"n":"causal agency","v":{"A0":"actively redirects the threat",'
                    '"A1":"allows the existing trajectory"},'
                    '"e":"doing harm may carry a distinct duty from allowing harm","fixed":true}],'
                    '"changed":[],"hyp":true}'
                )}]}

        candidates = [
            CandidateChunk(
                name, constraint, {"pull the lever": 0.5, "do nothing": 0.5},
                0.2, 0.2, 0.2, recommended_action="pull the lever",
                assumption_status="UNDERDETERMINED",
                unsupported_assumption="moral weight of causal agency is unresolved",
                reversal_condition="active redirection is treated as impermissible doing",
            )
            for name, constraint in (("utilitarian", "UNCERTAINTY"), ("deontological", "DUTY"))
        ]
        proposal = propose_problem_reformulation(
            TrolleyLlm(),
            "A trolley will kill five people unless you pull a lever that redirects it and kills one.",
            ["pull the lever", "do nothing"],
            candidates,
        )
        self.assertTrue(proposal.accepted)
        self.assertEqual(proposal.outcomes[0].unit, "lives")
        self.assertEqual(proposal.categorical_axes[0].name, "causal agency")
        self.assertTrue(proposal.categorical_axes[0].fixed_by_scenario)

    def test_consensus_audit_requires_assumption_and_reversal_condition(self):
        class AuditLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.8,"A1":0.2},"r":"A0",'
                    '"c":"UNCERTAINTY","u":"VERIFY_FACTS",'
                    '"w":"comparative harms remain unknown","j":"NONE",'
                    '"d":"UNDERDETERMINED",'
                    '"a":"disease burden exceeds instability harm",'
                    '"v":"instability harms more people overall"}'
                )}]}

        chunk = CompactLocalSpecialist("utilitarian", AuditLlm()).evaluate(
            "A cure could destabilize society.",
            ["publish", "conceal"],
            WorkspaceBroadcast(
                constraint="CONSENSUS_AUDIT",
                unresolved="VERIFY_ASSUMPTIONS",
                contingency_question="What facts would reverse the recommendation?",
            ),
        )
        self.assertTrue(chunk.schema_valid)
        self.assertEqual(chunk.assumption_status, "UNDERDETERMINED")
        self.assertEqual(chunk.unresolved, "VERIFY_FACTS")
        self.assertAlmostEqual(chunk.action_scores["publish"], 0.605)
        self.assertAlmostEqual(chunk.action_scores["conceal"], 0.395)
        self.assertEqual(chunk.reversal_condition, "instability harms more people overall")

    def test_live_audit_schema_contains_no_optional_legacy_fields(self):
        captured = {}

        class SchemaCaptureLlm:
            def complete_json(self, prompt, *, schema, **kwargs):
                captured.update(schema)
                raise RuntimeError("schema captured")

        with self.assertRaisesRegex(RuntimeError, "schema captured"):
            CompactLocalSpecialist("deontological", SchemaCaptureLlm()).evaluate(
                "A policy improves welfare but restricts liberty.",
                ["adopt policy", "reject policy"],
                WorkspaceBroadcast(
                    constraint="CONSENSUS_AUDIT",
                    audit_variable={
                        "entity": "welfare versus liberty",
                        "relation": "COMPARATIVE_MAGNITUDE",
                        "possible_values": ["NO_CHANGE", "REVERSES"],
                        "focus_action": "A0",
                        "question": "Would the comparison reverse the recommendation?",
                    },
                ),
            )

        properties = captured["properties"]
        self.assertTrue({"ap", "ax", "ie"} <= set(properties))
        self.assertFalse({"d", "a", "v", "av"} & set(properties))
        self.assertEqual(set(captured["required"]), set(properties))

    def test_audit_uncertainty_persists_without_new_facts(self):
        class PersistentAuditLlm:
            def __call__(self, prompt, **kwargs):
                audit_fields = (
                    ',"d":"CONDITIONAL",'
                    '"a":"disease burden exceeds instability harm",'
                    '"v":"instability harms more people overall"'
                    if "CONSENSUS_AUDIT" in prompt else ""
                )
                unresolved = "VERIFY_FACTS" if audit_fields else "NONE"
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.8,"A1":0.2},"r":"A0",'
                    f'"c":"UNCERTAINTY","u":"{unresolved}",'
                    f'"w":"comparative harms remain uncertain","j":"NONE"{audit_fields}}}'
                )}]}

        delegate = CompactLocalSpecialist("utilitarian", PersistentAuditLlm())
        actions = ["publish", "conceal"]
        audited = delegate.evaluate(
            "A cure could destabilize society.", actions,
            WorkspaceBroadcast(constraint="CONSENSUS_AUDIT"),
        )
        later = delegate.evaluate(
            "A cure could destabilize society.", actions,
            WorkspaceBroadcast(constraint="UNCERTAINTY"),
        )
        self.assertEqual(audited.assumption_status, "CONDITIONAL")
        self.assertEqual(later.assumption_status, "CONDITIONAL")
        self.assertAlmostEqual(later.action_scores["publish"], 0.68)
        self.assertEqual(later.reversal_condition, "instability harms more people overall")

    def test_planning_analysis_uses_strategic_forces_without_moral_vote(self):
        class PlanningLlm:
            prompt = ""

            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return {"choices": [{"text": (
                    '{"f":0.55,"n":"the witness retains access to evidence",'
                    '"x":"the witness loses access to evidence","b":"A1",'
                    '"a":["employer controls access"],"r":["limited time"],'
                    '"s":["INSTITUTIONAL_POWER","TIME_ENERGY_BUDGET"],"m":true,'
                    '"g":"limited time to report",'
                    '"va":true,"vr":"remaining silent requires no evidence access"}'
                )}]}

        llm = PlanningLlm()
        assessment = analyze_action_plan(
            llm,
            "A witness has limited time to report employer misconduct.",
            ["report the misconduct", "remain silent"],
            "report the misconduct",
            WorkspaceBroadcast(),
            [],
            "unresolved policy competition",
        )
        self.assertTrue(assessment.valid)
        self.assertTrue(assessment.broadcast_worthy)
        self.assertEqual(assessment.fallback, "remain silent")
        self.assertIn("strategic ecology", llm.prompt)
        self.assertNotIn("action_scores", assessment.__dataclass_fields__)

    def test_planning_rejects_invented_action_access_failure(self):
        class PlanningLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"f":0.2,"n":"a clinic offers the procedure",'
                    '"x":"no clinic will offer the procedure","b":"A1",'
                    '"a":[],"r":["clinic access"],'
                    '"s":["RESOURCE_SUPPLIERS"],"m":true,'
                    '"g":"memory-erasure procedure",'
                    '"va":true,"vr":"retaining testimony requires no clinic"}'
                )}]}

        assessment = analyze_action_plan(
            PlanningLlm(),
            "A parent may undergo a memory-erasure procedure or retain their testimony.",
            ["undergo memory erasure", "retain testimony"],
            "undergo memory erasure",
            WorkspaceBroadcast(),
            [],
            "unresolved policy competition",
        )
        self.assertFalse(assessment.valid)
        self.assertIn("unsupported access conditions", assessment.error)

    def test_failure_condition_becomes_explicit_fallback_question(self):
        class ContingencyLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"p":"the friend agrees to self-report",'
                    '"x":"the proposed self-report synthesis cannot proceed"}'
                )}]}

        condition = generate_failure_condition(
            ContingencyLlm(),
            "A witness must report a friend or remain silent.",
            "Urge the friend to self-report",
            ["report the crime", "remain silent"],
        )
        self.assertTrue(condition.valid)
        self.assertIn("NOT(the friend agrees to self-report)", condition.failure_condition)
        self.assertIn("fallback A0 or fallback A1", condition.contingency_question)
        self.assertEqual(condition.fallback_actions, ["report the crime", "remain silent"])
        self.assertEqual(validate_contingency_graph_dict(
            condition.semantic_graph, condition.synthesis_action, condition.fallback_actions,
            require_fallback_availability=False,
        ), [])

    def test_failure_condition_uses_typed_fallback_identity_not_prose_matching(self):
        class ContingencyLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"p":"the proposed safeguard remains effective",'
                    '"x":"the synthesis no longer resolves the conflict"}'
                )}]}

        actions = ["seal the facility", "deploy the experimental phage"]
        condition = generate_failure_condition(
            ContingencyLlm(), "A closed emergency choice.",
            "deploy the phage with a safeguard", actions,
        )
        self.assertTrue(condition.valid, condition.error)
        self.assertEqual(condition.fallback_actions, actions)

    def test_contingency_graph_rejects_failure_that_may_disable_a_fallback(self):
        validation = compile_contingency_graph(
            {
                "p": "the actor retains physical control",
                "x": "the actor loses the ability to execute actions",
            },
            "use the combined safeguard",
            ["choose first fallback", "choose second fallback"],
        )
        _, errors = certify_fallback_availability(
            validation.graph.to_dict(), "use the combined safeguard",
            ["choose first fallback", "choose second fallback"],
            {"A0": "UNAVAILABLE", "A1": "UNKNOWN"},
            {"A0": "control is lost", "A1": "execution is uncertain"},
        )
        self.assertIn("A0 is not independently", "; ".join(errors))
        self.assertIn("A1 is not independently", "; ".join(errors))

    def test_contingency_graph_detects_negation_direction_tampering(self):
        validation = compile_contingency_graph(
            {
                "p": "the safeguard remains operational",
                "x": "the synthesis loses its protective mechanism",
            },
            "use the combined safeguard",
            ["choose first fallback", "choose second fallback"],
        )
        graph = validation.graph.to_dict()
        failure = next(
            node for node in graph["nodes"] if node["id"] == "CONTINGENCY_FAILURE"
        )
        failure["attributes"]["truth_state"] = True
        errors = validate_contingency_graph_dict(
            graph, "use the combined safeguard",
            ["choose first fallback", "choose second fallback"],
            require_fallback_availability=False,
        )
        self.assertIn("not the typed negation", "; ".join(errors))

    def test_independent_feasibility_verifier_approves_both_fallbacks(self):
        class FeasibilityLlm:
            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return {"choices": [{"text": (
                    '{"fa":[{"action_id":"A0","status":"AVAILABLE",'
                    '"basis":"FAILURE_SCOPE","reason":"A0 uses an independent mechanism"},'
                    '{"action_id":"A1","status":"AVAILABLE",'
                    '"basis":"SCENARIO_STRUCTURE","reason":"A1 remains explicitly executable"}],'
                    '"sf":false}'
                )}]}

        condition = FailureCondition(
            "combined safeguard", "the safeguard works",
            "NOT(the safeguard works): combination fails", "Which fallback?",
            fallback_actions=["choose A0", "choose A1"],
            predicate_label="the safeguard works",
        )
        assessment = verify_contingency_feasibility(
            FeasibilityLlm(), "The actor can choose A0 or A1 independently.", condition,
        )
        self.assertTrue(assessment.valid, assessment.error)
        self.assertTrue(assessment.approved, assessment.error)
        self.assertEqual(assessment.fallback_statuses["A1"], "AVAILABLE")

    def test_independent_feasibility_verifier_blocks_shared_failure(self):
        class FeasibilityLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"fa":[{"action_id":"A0","status":"UNAVAILABLE",'
                    '"basis":"FAILURE_SCOPE","reason":"shared control has been lost"},'
                    '{"action_id":"A1","status":"UNKNOWN",'
                    '"basis":"INSUFFICIENT","reason":"execution cannot be established"}],'
                    '"sf":true}'
                )}]}

        condition = FailureCondition(
            "combined safeguard", "the actor retains control",
            "NOT(the actor retains control): control is lost", "Which fallback?",
            fallback_actions=["choose A0", "choose A1"],
            predicate_label="the actor retains control",
        )
        assessment = verify_contingency_feasibility(
            FeasibilityLlm(), "The actor must retain control to act.", condition,
        )
        self.assertTrue(assessment.valid)
        self.assertFalse(assessment.approved)
        self.assertTrue(assessment.shared_failure)
        self.assertIn("shared", assessment.error)

    def test_contingency_review_requires_a_typed_fallback_answer(self):
        actions = ["cut Sector 4", "cut emergency dispatch", "use synthesis"]
        broadcast = WorkspaceBroadcast(
            constraint="CONTINGENCY_REVIEW",
            contingency_question="If aid fails, which original fallback should govern?",
            contingency_synthesis_action=actions[2],
            contingency_failure_condition="aid cannot arrive",
            contingency_fallback_actions=(actions[0], actions[1]),
            branch_kind="SYNTHESIS_CONTINGENCY",
        )
        base = {
            "scores": {"A0": .7, "A1": .3, "A2": .1}, "r": "A0",
            "c": "IMMINENT_HARM", "u": "NONE",
            "w": "containment prevents greater harm", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": .7,
        }
        with self.assertRaisesRegex(ValueError, "did not choose a fallback"):
            _candidate_from_data(
                "utilitarian", actions, base, broadcast, "NONE", {},
            )

        base.update({
            "cr": "A0",
            "cj": "Given aid failure, A0 minimizes expected casualties",
        })
        candidate = _candidate_from_data(
            "utilitarian", actions, base, broadcast, "NONE", {},
        )
        self.assertEqual(candidate.contingency_choice, actions[0])
        self.assertIn("aid failure", candidate.contingency_justification)

    def test_contingency_review_cannot_recommend_failed_synthesis(self):
        actions = ["fallback zero", "fallback one", "failed synthesis"]
        broadcast = WorkspaceBroadcast(
            constraint="CONTINGENCY_REVIEW",
            contingency_question="Which fallback should govern?",
            contingency_synthesis_action=actions[2],
            contingency_failure_condition="the synthesis condition fails",
            contingency_fallback_actions=(actions[0], actions[1]),
        )
        data = {
            "scores": {"A0": .2, "A1": .3, "A2": .9}, "r": "A2",
            "c": "IMMINENT_HARM", "u": "NONE", "w": "repeat prior position",
            "j": "NONE", "e": "STATED_FACTS", "x": "NONE", "z": .7,
            "cr": "A0", "cj": "Given failure, fallback zero limits harm",
        }
        with self.assertRaisesRegex(ValueError, "recommendation does not match"):
            _candidate_from_data(
                "utilitarian", actions, data, broadcast, "NONE", {},
            )

    def test_compact_specialist_uses_lightweight_contingency_schema(self):
        calls = []

        class ContingencyLlm:
            def __call__(self, prompt, **kwargs):
                calls.append((prompt, kwargs))
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.7,"A1":0.3},"cr":"A0",'
                    '"c":"CARE","u":"NONE",'
                    '"cj":"Given aid failure, A0 better protects dependents",'
                    '"z":0.72,"fr":true}'
                )}]}

        actions = ["choose original A0", "choose original A1"]
        candidate = CompactLocalSpecialist(
            "care", ContingencyLlm(), testimony="Protect dependent relationships."
        ).evaluate(
            "A synthesis was considered.", actions,
            WorkspaceBroadcast(
                constraint="CONTINGENCY_REVIEW",
                contingency_question="Which fallback should govern?",
                contingency_synthesis_action="combine the actions",
                contingency_failure_condition="aid fails",
                contingency_fallback_actions=tuple(actions),
                branch_kind="SYNTHESIS_CONTINGENCY",
            ),
        )
        self.assertTrue(candidate.schema_valid, candidate.validation_errors)
        self.assertEqual(candidate.contingency_choice, actions[0])
        schema = calls[0][1]["grammar"] if "grammar" in calls[0][1] else None
        # Local backends receive a grammar when llama.cpp is installed; the
        # prompt itself confirms that the generic landscape/EV payload is absent.
        self.assertNotIn("landscape", calls[0][0].casefold())
        self.assertNotIn("expected value", calls[0][0].casefold())
        self.assertLessEqual(calls[0][1]["max_tokens"], 112)

    @patch("global_workspace.legacy_bridge.subprocess.run")
    def test_original_testimony_is_decoded(self, run):
        payload = base64.b64encode(b"Corpus-grounded testimony").decode("ascii")
        run.return_value = SimpleNamespace(
            returncode=0,
            stdout=f"agent logs\n{RESPONSE_MARKER}{payload}\n",
            stderr="",
        )
        result = consult_original_agents(Path("scenario.json"), agents=("care",))
        self.assertEqual(result.testimonies["care"], "Corpus-grounded testimony")
        self.assertEqual(result.errors, {})

    @patch("global_workspace.legacy_bridge.subprocess.run")
    def test_openai_backend_is_forwarded_to_original_agents(self, run):
        payload = base64.b64encode(b"Hosted testimony").decode("ascii")
        run.return_value = SimpleNamespace(
            returncode=0,
            stdout=f"{RESPONSE_MARKER}{payload}\n",
            stderr="",
        )
        result = consult_original_agents(
            Path("scenario.json"),
            agents=("care",),
            backend="openai",
            openai_model="o3",
        )
        command = run.call_args.args[0]
        child_env = run.call_args.kwargs["env"]
        self.assertEqual(command[command.index("--backend") + 1], "openai")
        self.assertEqual(command[command.index("--openai-model") + 1], "o3")
        self.assertEqual(child_env["ETHICS_LLM_BACKEND"], "openai")
        self.assertEqual(result.testimonies["care"], "Hosted testimony")

    @patch("global_workspace.legacy_bridge.subprocess.run")
    def test_original_agent_phase_timings_are_decoded_as_execution_metadata(self, run):
        response = base64.b64encode(b"Timed testimony").decode("ascii")
        timing = base64.b64encode(json.dumps({
            "child_total_seconds": 1.0,
            "module_startup_seconds": 0.1,
            "model_startup_seconds": 0.2,
            "retrieval_seconds": 0.25,
            "model_seconds": 0.4,
            "retrieval_call_count": 1,
            "model_call_count": 1,
            "timeout_count": 0,
        }).encode("utf-8")).decode("ascii")
        run.return_value = SimpleNamespace(
            returncode=0,
            stdout=(
                f"{PERFORMANCE_MARKER}{timing}\n"
                f"{RESPONSE_MARKER}{response}\n"
            ),
            stderr="",
        )
        recorder, token = start_performance_trace("original-agent-timing")
        try:
            result = consult_original_agents(
                Path("scenario.json"), agents=("care",),
            )
            snapshot = recorder.snapshot()
        finally:
            reset_performance_trace(token)
        self.assertEqual(result.testimonies["care"], "Timed testimony")
        events = {item["name"]: item for item in snapshot["events"]}
        self.assertIn("original_agent_startup", events)
        self.assertEqual(
            events["original_agent_retrieval"]["duration_seconds"], 0.25,
        )
        self.assertEqual(events["original_agent_model"]["duration_seconds"], 0.4)
        self.assertEqual(snapshot["counts"]["original_agent_model_calls"], 1)

    @patch("global_workspace.legacy_bridge.subprocess.run")
    def test_original_agents_receive_canonical_order_not_presentation_order(self, run):
        payload = base64.b64encode(b"Canonical testimony").decode("ascii")
        observed = {}

        def complete(command, **kwargs):
            scenario_path = Path(command[command.index("--scenario") + 1])
            observed.update(json.loads(scenario_path.read_text(encoding="utf-8")))
            return SimpleNamespace(
                returncode=0,
                stdout=f"{RESPONSE_MARKER}{payload}\n",
                stderr="",
            )

        run.side_effect = complete
        with tempfile.TemporaryDirectory() as directory:
            scenario_path = Path(directory) / "scenario.json"
            scenario_path.write_text(json.dumps({
                "ethical_question": (
                    "A vehicle loses braking. Option A: maintain course and kill three. "
                    "Option B: swerve and kill one."
                )
            }))
            canonical = tuple(canonicalize_action_order([
                "maintain course and kill three",
                "swerve and kill one",
            ]))
            result = consult_original_agents(
                scenario_path, agents=("care",), canonical_actions=canonical,
            )

        question = observed["ethical_question"]
        self.assertIn(f"Action A0: {canonical[0]}", question)
        self.assertIn(f"Action A1: {canonical[1]}", question)
        self.assertNotIn("Option A:", question)
        self.assertEqual(result.testimonies["care"], "Canonical testimony")

    def test_canonical_action_order_is_invariant_to_input_permutation(self):
        actions = ["maintain course", "swerve into barrier"]
        self.assertEqual(
            canonicalize_action_order(actions),
            canonicalize_action_order(list(reversed(actions))),
        )

    def test_deliberation_scenario_is_invariant_to_option_permutation(self):
        first = (
            "A vehicle loses braking. Option A: maintain course and kill three. "
            "Option B: swerve and kill one. Casualties under Option A are unlogged."
        )
        second = (
            "A vehicle loses braking. Option A: swerve and kill one. "
            "Option B: maintain course and kill three. Casualties under Option B are unlogged."
        )
        first_legend = extract_labeled_action_legend(first)
        second_legend = extract_labeled_action_legend(second)
        canonical = canonicalize_action_order(list(first_legend.values()))
        normalized_first = canonicalize_deliberation_scenario(
            first, first_legend, canonical,
        )
        normalized_second = canonicalize_deliberation_scenario(
            second, second_legend, canonical,
        )

        self.assertEqual(normalized_first, normalized_second)
        self.assertNotIn("Option A", normalized_first)
        unlogged_id = next(
            f"A{index}" for index, action in enumerate(canonical)
            if "maintain course" in action.casefold()
        )
        self.assertIn(f"Casualties under {unlogged_id} are unlogged", normalized_first)

    def test_presentation_mapping_preserves_original_labels_after_canonicalization(self):
        scenario = (
            "A hospital must choose. Option A: power the NICU for eight infants. "
            "Option B: power water treatment for tens of thousands."
        )
        source_legend = extract_labeled_action_legend(scenario)
        canonical = canonicalize_action_order(list(source_legend.values()))

        mapping = build_presentation_action_mapping(
            scenario,
            source_legend,
            canonical,
            source_labels_explicit=True,
        )

        self.assertEqual(
            [entry["source_label"] for entry in mapping],
            ["Original Option A", "Original Option B"],
        )
        for source_index, source_id in enumerate(("A0", "A1")):
            expected_id = f"A{canonical.index(source_legend[source_id])}"
            self.assertEqual(mapping[source_index]["canonical_action_id"], expected_id)
            self.assertEqual(mapping[source_index]["source_action"], source_legend[source_id])

    def test_presentation_mapping_is_passive_under_option_permutation(self):
        first = (
            "Option A: power the NICU for eight infants. "
            "Option B: power water treatment for tens of thousands."
        )
        reversed_problem = (
            "Option A: power water treatment for tens of thousands. "
            "Option B: power the NICU for eight infants."
        )
        first_legend = extract_labeled_action_legend(first)
        reversed_legend = extract_labeled_action_legend(reversed_problem)
        canonical = canonicalize_action_order(list(first_legend.values()))

        first_mapping = build_presentation_action_mapping(
            first, first_legend, canonical, source_labels_explicit=True,
        )
        reversed_mapping = build_presentation_action_mapping(
            reversed_problem, reversed_legend, canonical, source_labels_explicit=True,
        )

        self.assertEqual(
            {entry["source_action"]: entry["canonical_action_id"] for entry in first_mapping},
            {entry["source_action"]: entry["canonical_action_id"] for entry in reversed_mapping},
        )
        self.assertNotEqual(first_mapping, reversed_mapping)

    def test_unlabeled_actions_get_neutral_presentation_labels(self):
        presented = ["continue treatment", "withdraw treatment"]
        source_legend = {"A0": presented[0], "A1": presented[1]}
        canonical = canonicalize_action_order(presented)

        mapping = build_presentation_action_mapping(
            "The clinician must choose between two courses.",
            source_legend,
            canonical,
            source_labels_explicit=False,
        )

        self.assertEqual(
            [entry["source_label"] for entry in mapping],
            ["Presented option 1", "Presented option 2"],
        )
        self.assertEqual(len(mapping), 2)
        self.assertEqual(
            {entry["canonical_action"] for entry in mapping}, set(canonical)
        )

    def test_decision_brief_explains_only_an_actual_label_remap(self):
        actions = ["power water treatment", "power the NICU"]
        candidate = CandidateChunk(
            "care", "CARE", {actions[0]: 0.4, actions[1]: 0.6}, 0.2, 0.3, 0.8,
            recommended_action=actions[1], rationale="Protect the dependent infants.",
        )
        cycle = CycleRecord(
            1, WorkspaceBroadcast(), [candidate], candidate, None,
            {actions[0]: 0.4, actions[1]: 0.6}, 0.7, 1, 1.0,
        )
        result = WorkspaceResult(
            "test", actions,
            presentation_actions=list(reversed(actions)),
            presentation_action_mapping=[
                {
                    "source_label": "Original Option A",
                    "source_position": 0,
                    "source_action": actions[1],
                    "canonical_action_id": "A1",
                    "canonical_action": actions[1],
                    "mapping_basis": "EXACT_ACTION_IDENTITY",
                },
                {
                    "source_label": "Original Option B",
                    "source_position": 1,
                    "source_action": actions[0],
                    "canonical_action_id": "A0",
                    "canonical_action": actions[0],
                    "mapping_basis": "EXACT_ACTION_IDENTITY",
                },
            ],
            cycles=[cycle], selected_action=actions[1], confidence=0.6,
            current_plurality=actions[1], epistemic_confidence=0.7,
        )

        answer = render_public_judgment(result)

        self.assertIn("## Action label mapping", answer)
        self.assertIn("Original Option A → canonical A1", answer)
        self.assertIn("Original Option B → canonical A0", answer)
        self.assertIn("stable internal identifiers", answer)
        self.assertEqual(
            result.to_dict()["presentation_action_mapping"],
            result.presentation_action_mapping,
        )

        identity_data = result.to_dict()
        identity_data["presentation_action_mapping"] = [
            {
                "source_label": "Original Option A",
                "source_position": 0,
                "source_action": actions[0],
                "canonical_action_id": "A0",
                "canonical_action": actions[0],
                "mapping_basis": "EXACT_ACTION_IDENTITY",
            },
            {
                "source_label": "Original Option B",
                "source_position": 1,
                "source_action": actions[1],
                "canonical_action_id": "A1",
                "canonical_action": actions[1],
                "mapping_basis": "EXACT_ACTION_IDENTITY",
            },
        ]
        self.assertNotIn(
            "## Action label mapping", render_public_judgment(identity_data)
        )

    def test_graph_action_identity_matches_structural_paraphrases(self):
        maintain_a = (
            "Maintain the vehicle's current trajectory, killing three pedestrians"
        )
        maintain_b = (
            "Continue on the current path, causing the deaths of three pedestrians"
        )
        redirect_a = (
            "Swerve into a concrete barrier, killing one passenger and saving three pedestrians"
        )
        redirect_b = (
            "Turn into the concrete barrier; one passenger dies while three pedestrians are spared"
        )

        self.assertEqual(semantic_action_key(maintain_a), semantic_action_key(maintain_b))
        self.assertEqual(semantic_action_key(redirect_a), semantic_action_key(redirect_b))
        self.assertNotEqual(semantic_action_key(maintain_a), semantic_action_key(redirect_a))
        first_order = canonicalize_action_order([maintain_a, redirect_a])
        reversed_paraphrases = canonicalize_action_order([redirect_b, maintain_b])
        self.assertEqual(
            [semantic_action_key(action) for action in first_order],
            [semantic_action_key(action) for action in reversed_paraphrases],
        )

    def test_graph_action_identity_preserves_decision_relevant_quantities(self):
        low_risk = "Deploy a treatment with a 10% risk of harming 5 patients"
        high_risk = "Deploy a treatment with a 30% risk of harming 5 patients"

        self.assertNotEqual(semantic_action_key(low_risk), semantic_action_key(high_risk))

    def test_graph_action_identity_captures_operational_policy_tradeoffs(self):
        enforce = (
            "Strictly enforce the new protocol immediately, severely bottlenecking "
            "fulfillment and exhausting the team"
        )
        bypass = (
            "Temporarily bypass the directive for high-priority shipments, protecting "
            "throughput and team well-being"
        )

        enforce_identity = compile_action_identity(enforce)
        bypass_identity = compile_action_identity(bypass)

        self.assertEqual(enforce_identity.intervention, "enforce")
        self.assertEqual(bypass_identity.intervention, "bypass")
        self.assertNotEqual(
            semantic_action_key(enforce), semantic_action_key(bypass)
        )
        self.assertTrue(
            any(
                consequence.predicate in {"bottleneck", "exhaustion"}
                for consequence in enforce_identity.consequences
            ),
            enforce_identity.consequences,
        )
        self.assertTrue(
            any(
                consequence.predicate in {"protect", "preserve", "throughput", "compliance"}
                for consequence in bypass_identity.consequences
            ),
            bypass_identity.consequences,
        )

    def test_sparse_action_identity_uses_explicit_lexical_fallback(self):
        identity = compile_action_identity("Wait")

        self.assertEqual(identity.basis, "LEXICAL_FALLBACK")
        self.assertEqual(identity.lexical_fallback, "wait")

    def test_within_decision_graph_identity_collision_is_safely_disambiguated(self):
        actions = ["Kill one passenger", "Sacrifice one passenger"]
        self.assertEqual(semantic_action_key(actions[0]), semantic_action_key(actions[1]))

        resolved = resolved_semantic_action_keys(actions)
        self.assertEqual(len(set(resolved)), 2)
        self.assertTrue(all(":lex:" in key for key in resolved))

    def test_scenario_graph_contains_typed_action_identity_edges(self):
        graph = compile_scenario_graph(
            "Choose A0 or A1.",
            ["Report the misconduct", "Remain silent"],
        )

        self.assertTrue(any(
            edge.source == "A0" and edge.relation == "HAS_INTERVENTION"
            for edge in graph.edges
        ))
        action = graph.nodes["A0"]
        self.assertEqual(action.attributes["action_identity_basis"], "GRAPH")
        self.assertEqual(
            action.attributes["semantic_action_key"],
            semantic_action_key("Disclose the misconduct"),
        )

    def test_action_identity_graph_records_actor_consequence_and_modality(self):
        action = (
            "The safety board deploys an experimental treatment with a 35 percent "
            "risk of harming 5 patients"
        )
        identity = compile_action_identity(action)
        self.assertEqual(identity.intervention, "deploy")
        self.assertEqual(identity.actors, ("board", "safety"))
        self.assertIn("UNCERTAIN", identity.modalities)
        self.assertEqual(identity.consequences[0].probability, "35%")

        graph = compile_scenario_graph("Choose A0 or A1.", [action, "Wait"])
        relations = {edge.relation for edge in graph.edges if edge.source == "A0"}
        self.assertTrue({
            "HAS_ACTOR", "HAS_INTERVENTION", "HAS_CONSEQUENCE", "HAS_CONSTRAINT",
        }.issubset(relations))

    def test_compact_delegate_receives_original_testimony(self):
        class FakeLlm:
            prompt = ""

            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.9,"A1":0.1},"r":"A0",'
                    '"c":"CARE","u":"NONE","w":"protect entrusted vulnerable person"}'
                )}]}

        llm = FakeLlm()
        delegate = CompactLocalSpecialist(
            "care", llm, testimony="Evidence from the care corpus", baseline_action_id="A0"
        )
        chunk = delegate.evaluate(
            "A person is threatened.",
            ["protect", "disclose"],
            WorkspaceBroadcast(constraint="IMMINENT_HARM"),
        )
        self.assertIn("Evidence from the care corpus", llm.prompt)
        self.assertIn("constraint=IMMINENT_HARM", llm.prompt)
        self.assertEqual(chunk.constraint, "CARE")
        self.assertEqual(chunk.testimony_alignment, "SUPPORTS")
        self.assertAlmostEqual(chunk.preference_strength, 0.8)
        self.assertAlmostEqual(chunk.epistemic_confidence, 0.85)

    def test_malformed_delegate_output_does_not_crash_workspace(self):
        class BrokenLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": '{"scores":[0.9,0.1],"c":"HARM"'}]}

        delegate = CompactLocalSpecialist("utilitarian", BrokenLlm(), testimony="Grounding")
        chunk = delegate.evaluate(
            "A person is threatened.",
            ["protect", "disclose"],
            WorkspaceBroadcast(),
        )
        self.assertEqual(chunk.constraint, "NONE")
        self.assertEqual(chunk.delegate_status, "SEMANTIC_VALIDATION_ERROR")
        self.assertEqual(chunk.confidence, 0.0)
        self.assertEqual(chunk.unresolved, "REVIEW_MODEL_OUTPUT")
        self.assertFalse(chunk.schema_valid)

    def test_short_schema_maps_scores_by_action_order(self):
        class CompactLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.8,"A1":0.2},"r":"A0",'
                    '"c":"IMMINENT_HARM","u":"NONE","w":"prevents greater expected harm"}'
                )}]}

        chunk = CompactLocalSpecialist("utilitarian", CompactLlm()).evaluate(
            "A scenario", ["protect", "disclose"], WorkspaceBroadcast()
        )
        self.assertEqual(chunk.action_scores, {"protect": 0.8, "disclose": 0.2})
        self.assertEqual(chunk.constraint, "IMMINENT_HARM")

    def test_initial_recommendation_cannot_invert_testimony_baseline(self):
        class InvertingLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.1,"A1":0.9},"r":"A1",'
                    '"c":"CARE","u":"NONE","w":"broadcast changes relational priority"}'
                )}]}

        chunk = CompactLocalSpecialist(
            "care", InvertingLlm(), testimony="Choose A0", baseline_action_id="A0"
        ).evaluate(
            "A scenario", ["step on ant", "swear at mother"], WorkspaceBroadcast()
        )
        self.assertFalse(chunk.schema_valid)
        self.assertEqual(chunk.constraint, "NONE")
        self.assertEqual(chunk.delegate_status, "SEMANTIC_VALIDATION_ERROR")

    def test_abrupt_switch_to_favored_action_receives_conformity_penalty(self):
        class SwitchingLlm:
            calls = 0

            def __call__(self, prompt, **kwargs):
                self.calls += 1
                recommendation = "A1" if self.calls == 1 else "A2"
                scores = (
                    '{"A0":0.1,"A1":0.8,"A2":0.2}'
                    if recommendation == "A1"
                    else '{"A0":0.1,"A1":0.2,"A2":0.8}'
                )
                return {"choices": [{"text": (
                    f'{{"scores":{scores},"r":"{recommendation}","c":"DUTY",'
                    '"u":"NONE","w":"duty based judgment","j":"majority favors balance"}'
                )}]}

        delegate = CompactLocalSpecialist("deontological", SwitchingLlm())
        actions = ["approve", "deny", "mixed compromise"]
        first = delegate.evaluate("A zoning conflict", actions, WorkspaceBroadcast())
        second = delegate.evaluate(
            "A zoning conflict",
            actions,
            WorkspaceBroadcast(constraint="DUTY", intent="evaluate_mixed compromise"),
        )
        self.assertEqual(first.recommended_action, "deny")
        self.assertEqual(second.recommended_action, "mixed compromise")
        self.assertTrue(second.position_changed)
        self.assertEqual(second.previous_action, "deny")
        self.assertAlmostEqual(second.conformity_penalty, 0.65)
        self.assertAlmostEqual(second.preference_strength, 0.6)
        self.assertAlmostEqual(second.epistemic_confidence, 0.2975)

    def test_synthesis_review_does_not_penalize_reconsideration(self):
        class SynthesisSwitchLlm:
            calls = 0

            def __call__(self, prompt, **kwargs):
                self.calls += 1
                recommendation = "A0" if self.calls == 1 else "A2"
                scores = (
                    '{"A0":0.8,"A1":0.2,"A2":0.1}'
                    if recommendation == "A0"
                    else '{"A0":0.2,"A1":0.1,"A2":0.8}'
                )
                return {"choices": [{"text": (
                    f'{{"scores":{scores},"r":"{recommendation}","c":"CARE",'
                    '"u":"NONE","w":"new option resolves care","j":"new synthesis reduces harm"}'
                )}]}

        delegate = CompactLocalSpecialist("care", SynthesisSwitchLlm())
        actions = ["act", "refrain", "new synthesis"]
        delegate.evaluate("A conflict", actions, WorkspaceBroadcast())
        changed = delegate.evaluate(
            "A conflict",
            actions,
            WorkspaceBroadcast(constraint="SYNTHESIS_REVIEW", intent="evaluate_new synthesis"),
        )
        self.assertTrue(changed.position_changed)
        self.assertEqual(changed.conformity_penalty, 0.0)
        self.assertAlmostEqual(changed.preference_strength, 0.6)
        self.assertAlmostEqual(changed.epistemic_confidence, 0.85)

    def test_same_recommendation_strength_drift_does_not_dampen_confidence(self):
        class DriftingLlm:
            calls = 0

            def __call__(self, prompt, **kwargs):
                self.calls += 1
                scores = '{"A0":0.8,"A1":0.2}' if self.calls == 1 else '{"A0":0.6,"A1":0.4}'
                return {"choices": [{"text": (
                    f'{{"scores":{scores},"r":"A0","c":"FEASIBILITY",'
                    '"u":"NONE","w":"expected welfare favors approval","j":"NONE"}'
                )}]}

        delegate = CompactLocalSpecialist("utilitarian", DriftingLlm())
        actions = ["approve", "deny"]
        first = delegate.evaluate("A policy conflict", actions, WorkspaceBroadcast())
        second = delegate.evaluate(
            "A policy conflict",
            actions,
            WorkspaceBroadcast(constraint="DUTY", intent="evaluate_deny"),
        )
        self.assertAlmostEqual(first.preference_strength, 0.6)
        self.assertAlmostEqual(first.epistemic_confidence, 0.85)
        self.assertAlmostEqual(second.confidence_drift, -0.4)
        self.assertEqual(second.confidence_drift_penalty, 0.0)
        self.assertAlmostEqual(second.preference_strength, 0.2)
        self.assertAlmostEqual(second.epistemic_confidence, 0.85)

    def test_framework_relevant_confidence_change_can_be_justified(self):
        class JustifiedLlm:
            calls = 0

            def __call__(self, prompt, **kwargs):
                self.calls += 1
                scores = '{"A0":0.8,"A1":0.2}' if self.calls == 1 else '{"A0":0.6,"A1":0.4}'
                return {"choices": [{"text": (
                    f'{{"scores":{scores},"r":"A0","c":"FEASIBILITY",'
                    '"u":"NONE","w":"expected welfare favors approval",'
                    '"j":"new feasibility evidence lowers expected benefit"}'
                )}]}

        delegate = CompactLocalSpecialist("utilitarian", JustifiedLlm())
        actions = ["approve", "deny"]
        delegate.evaluate("A policy conflict", actions, WorkspaceBroadcast())
        changed = delegate.evaluate(
            "A policy conflict",
            actions,
            WorkspaceBroadcast(constraint="FEASIBILITY", intent="evaluate_deny"),
        )
        self.assertAlmostEqual(changed.confidence_drift, -0.4)
        self.assertEqual(changed.confidence_drift_penalty, 0.0)
        self.assertAlmostEqual(changed.preference_strength, 0.2)
        self.assertAlmostEqual(changed.epistemic_confidence, 0.85)

    def test_recovery_from_system_claim_damping_is_not_preference_drift(self):
        class AuditRecoveryLlm:
            calls = 0

            def __call__(self, prompt, **kwargs):
                self.calls += 1
                evidence = (
                    '"e":"UNSTATED_FACTS","x":"the intervention prevents later harm",'
                    '"u":"VERIFY_FACTS"'
                    if self.calls == 1
                    else '"e":"STATED_FACTS","x":"NONE","u":"NONE"'
                )
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.8,"A1":0.2},"r":"A0",'
                    '"c":"FEASIBILITY","w":"expected welfare favors approval",'
                    f'"j":"NONE",{evidence},"z":0.85}}'
                )}]}

        delegate = CompactLocalSpecialist("utilitarian", AuditRecoveryLlm())
        actions = ["approve", "deny"]
        damped = delegate.evaluate(
            "A policy conflict", actions, WorkspaceBroadcast()
        )
        recovered = delegate.evaluate(
            "A policy conflict",
            actions,
            WorkspaceBroadcast(constraint="FEASIBILITY", intent="evaluate_approve"),
        )

        self.assertAlmostEqual(damped.reported_preference_strength, 0.6)
        self.assertAlmostEqual(damped.preference_strength, 0.33)
        self.assertAlmostEqual(recovered.reported_preference_strength, 0.6)
        self.assertAlmostEqual(recovered.preference_strength, 0.6)
        self.assertAlmostEqual(recovered.preference_drift, 0.0)
        self.assertEqual(recovered.preference_drift_penalty, 0.0)

    def test_same_recommendation_can_gain_strength_without_epistemic_penalty(self):
        class StrengthShiftLlm:
            calls = 0

            def __call__(self, prompt, **kwargs):
                self.calls += 1
                text = (
                    '{"scores":{"A0":0.52,"A1":0.48},"r":"A0",'
                    '"c":"CARE","u":"NONE","w":"initially balanced","j":"NONE","z":0.72}'
                    if self.calls == 1
                    else '{"scores":{"A0":0.82,"A1":0.18},"r":"A0",'
                    '"c":"CARE","u":"NONE","w":"same recommendation, stronger support",'
                    '"j":"NONE","z":0.72}'
                )
                return {"choices": [{"text": text}]}

        delegate = CompactLocalSpecialist("care", StrengthShiftLlm())
        actions = ["protect", "expose"]
        first = delegate.evaluate("A care conflict", actions, WorkspaceBroadcast())
        second = delegate.evaluate("A care conflict", actions, WorkspaceBroadcast())

        self.assertAlmostEqual(first.reported_preference_strength, 0.04)
        self.assertAlmostEqual(second.reported_preference_strength, 0.64)
        self.assertAlmostEqual(second.preference_drift, 0.60, places=2)
        self.assertEqual(second.preference_drift_penalty, 0.0)
        self.assertAlmostEqual(second.epistemic_confidence, 0.72)

    def test_claim_and_consensus_audit_do_not_double_damp_same_uncertainty(self):
        data = {
            "scores": {"A0": 0.2, "A1": 0.8},
            "r": "A1",
            "c": "DUTY",
            "u": "VERIFY_FACTS",
            "w": "prevent severe community harm",
            "j": "NONE",
            "e": "UNSTATED_FACTS",
            "x": "hospital isolation is uniquely effective",
            "d": "CONDITIONAL",
            "a": "hospital isolation is the only effective containment",
            "v": "equally effective home quarantine becomes available",
        }

        candidate = _candidate_from_data(
            "deontological",
            ["release patient", "hospital isolation"],
            data,
            WorkspaceBroadcast(constraint="CONSENSUS_AUDIT"),
            "NONE",
            {},
        )

        # One 0.55 contraction turns a reported 0.60 gap into 0.33. The former
        # behavior multiplied another 0.60 audit contraction and yielded 0.198.
        self.assertAlmostEqual(candidate.reported_preference_strength, 0.6)
        self.assertAlmostEqual(candidate.preference_strength, 0.33)
        self.assertAlmostEqual(candidate.epistemic_confidence, 0.5)

    def test_bounded_background_claim_receives_mild_calibrated_damping(self):
        data = {
            "scores": {"A0": 0.2, "A1": 0.8},
            "r": "A1", "c": "FEASIBILITY", "u": "VERIFY_FACTS",
            "w": "ordinary containment reduces exposure", "j": "NONE",
            "e": "UNSTATED_FACTS", "x": "containment reduces interpersonal exposure",
            "z": 0.8,
        }
        calibration = EvidenceCalibration(
            "BOUNDED_BACKGROUND", 0.8, 0.7, True,
            "ordinary bounded causal premise",
        )

        candidate = _candidate_from_data(
            "utilitarian", ["release", "contain"], data,
            WorkspaceBroadcast(), "NONE", {}, evidence_calibration=calibration,
        )

        self.assertAlmostEqual(candidate.reported_preference_strength, 0.6)
        self.assertAlmostEqual(candidate.preference_strength, 0.48)
        self.assertAlmostEqual(candidate.epistemic_confidence, 0.55)
        self.assertEqual(candidate.evidence_calibration_tier, "BOUNDED_BACKGROUND")

    def test_entailed_claim_is_not_damped_as_speculation(self):
        data = {
            "scores": {"A0": 0.2, "A1": 0.8},
            "r": "A1", "c": "IMMINENT_HARM", "u": "VERIFY_FACTS",
            "w": "containment prevents stated outbreak", "j": "NONE",
            "e": "UNSTATED_FACTS", "x": "release creates severe outbreak risk",
            "z": 0.8,
        }
        calibration = EvidenceCalibration(
            "ENTAILED", 1.0, 0.85, False,
            "scenario directly states release creates outbreak risk",
        )

        candidate = _candidate_from_data(
            "utilitarian", ["release", "contain"], data,
            WorkspaceBroadcast(), "NONE", {}, evidence_calibration=calibration,
        )

        self.assertEqual(candidate.evidence_basis, "STATED_FACTS")
        self.assertEqual(candidate.unresolved, "NONE")
        self.assertAlmostEqual(candidate.preference_strength, 0.6)
        self.assertAlmostEqual(candidate.epistemic_confidence, 0.8)
        self.assertEqual(candidate.evidence_calibration_tier, "ENTAILED")

    def test_strong_preference_can_have_low_epistemic_confidence(self):
        class UncertainLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.95,"A1":0.05},"r":"A0","c":"UNCERTAINTY",'
                    '"u":"VERIFY_FACTS","w":"prediction strongly favors intervention",'
                    '"j":"NONE","e":"UNSTATED_FACTS","x":"the intervention succeeds",'
                    '"z":0.2}'
                )}]}

        chunk = CompactLocalSpecialist("utilitarian", UncertainLlm()).evaluate(
            "An outcome depends on an unknown response.", ["intervene", "wait"],
            WorkspaceBroadcast(),
        )
        self.assertAlmostEqual(chunk.preference_strength, 0.495)
        self.assertAlmostEqual(chunk.epistemic_confidence, 0.2)
        self.assertNotEqual(chunk.preference_strength, chunk.epistemic_confidence)

    def test_action_planner_filters_unsupported_or_infeasible_actions(self):
        class PlannerLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"actor":"the bystander","sides":{"A":"prevent greater harm","B":"avoid intervention"},"actions":['
                    '{"a":"push the flag","f":0.95,"e":true,"p":"SIDE_A"},'
                    '{"a":"do nothing","f":1.0,"e":true,"p":"SIDE_B"},'
                    '{"a":"call authorities","f":0.2,"e":false,"p":"BALANCE"}'
                    ']}'
                )}]}

        actions = propose_actions(PlannerLlm(), "A closed immediate choice")
        self.assertEqual(actions, ["push the flag", "do nothing"])

    def test_action_planner_rejects_multiple_tactics_for_only_one_side(self):
        class OneSidedLlm:
            calls = 0

            def __call__(self, prompt, **kwargs):
                self.calls += 1
                return {"choices": [{"text": (
                    '{"actor":"individuals","sides":{"A":"individual autonomy","B":"community protection"},"actions":['
                    '{"a":"hold education sessions","f":0.9,"e":true,"p":"SIDE_B"},'
                    '{"a":"offer medical consultations","f":0.9,"e":true,"p":"SIDE_B"}'
                    ']}'
                )}]}

        with self.assertRaisesRegex(ValueError, "both sides"):
            propose_actions(OneSidedLlm(), "A general conflict between autonomy and public welfare")

    def test_action_planner_accepts_general_opposing_positions_without_clipping_words(self):
        class BalancedLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"actor":"individuals","sides":{"A":"individual autonomy","B":"community protection"},"actions":['
                    '{"a":"Decline vaccination based on personal health beliefs",'
                    '"f":0.9,"e":true,"p":"SIDE_A"},'
                    '{"a":"Get vaccinated to contribute to community protection",'
                    '"f":0.85,"e":true,"p":"SIDE_B"}'
                    ']}'
                )}]}

        actions = propose_actions(
            BalancedLlm(),
            "Personal vaccine beliefs conflict with community wellbeing and herd immunity.",
        )
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0], "Decline vaccination based on personal health beliefs")
        self.assertEqual(actions[1], "Get vaccinated to contribute to community protection")

    def test_action_planner_rejects_invented_policy_actor(self):
        class PolicyDriftLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"actor":"community leaders","sides":{"A":"autonomy","B":"public health"},"actions":['
                    '{"a":"Provide education and financial incentives","f":0.9,"e":true,"p":"SIDE_A"},'
                    '{"a":"Mandate vaccination with fines","f":0.9,"e":true,"p":"SIDE_B"}'
                    ']}'
                )}]}

        with self.assertRaisesRegex(ValueError, "institutional policymaker"):
            propose_actions(
                PolicyDriftLlm(),
                "Individuals may resist vaccination despite benefits to community wellbeing.",
            )

    def test_action_planner_rejects_initial_middle_option(self):
        class MiddleOptionLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"actor":"the city","sides":{"A":"residential protection","B":"economic growth"},"actions":['
                    '{"a":"Deny rezoning and retain residential protections","f":0.9,"e":true,"p":"SIDE_A"},'
                    '{"a":"Adopt a balanced mixed-use compromise","f":0.9,"e":true,"p":"SIDE_B"}'
                    ']}'
                )}]}

        with self.assertRaisesRegex(ValueError, "fewer than two"):
            propose_actions(
                MiddleOptionLlm(),
                "A city must weigh residential protection against economic growth.",
            )

    def test_general_synthesis_is_grounded_and_feasible(self):
        class SynthesisLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"a":"tell the truth compassionately","g":["care","deontological"],'
                    '"k":["CARE","DUTY"],"q":[],"f":0.9,"x":true,"n":true,'
                    '"w":"preserves honesty while reducing needless hurt"}'
                )}]}

        candidates = [
            CandidateChunk("care", "CARE", {"lie": 0.8, "truth": 0.2}, 0.5, 0.7, 0.8, rationale="avoid hurt", recommended_action="lie"),
            CandidateChunk("deontological", "DUTY", {"lie": 0.1, "truth": 0.9}, 0.5, 0.7, 0.8, rationale="be honest", recommended_action="truth"),
        ]
        proposal = propose_synthesis(
            SynthesisLlm(),
            "Is a small lie acceptable?",
            ["lie", "truth"],
            candidates,
            WorkspaceBroadcast(constraint="CARE"),
            {"care": "Avoid needless hurt.", "deontological": "Respect honesty."},
        )
        self.assertTrue(proposal.accepted)
        self.assertEqual(proposal.action, "tell the truth compassionately")

    def test_synthesis_rejects_hidden_ungrounded_operational_clause(self):
        class SynthesisLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"a":"cut the sector while mobilizing local thermal aid",'
                    '"g":["care","deontological"],"k":["CARE","DUTY"],'
                    '"q":[],"f":0.82,"x":true,"n":true,'
                    '"w":"combines containment with protection"}'
                )}]}

        actions = ["cut power to the sector", "keep power connected"]
        candidates = [
            CandidateChunk("care", "CARE", {actions[0]: .2, actions[1]: .8}, .5, .5, .7,
                           rationale="protect residents", recommended_action=actions[1]),
            CandidateChunk("deontological", "DUTY", {actions[0]: .8, actions[1]: .2}, .5, .5, .7,
                           rationale="contain the failure", recommended_action=actions[0]),
        ]
        proposal = propose_synthesis(
            SynthesisLlm(), "A grid failure requires choosing whether to cut power.",
            actions, candidates, WorkspaceBroadcast(),
            {"care": "Protect residents.", "deontological": "Contain the failure."},
        )
        self.assertFalse(proposal.accepted)
        self.assertIn("operational clause not grounded", proposal.rejection_reason)

    def test_synthesis_withholds_temporal_sharing_that_contradicts_exclusive_choice(self):
        class SynthesisLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"a":"alternate full power between hospital and water station hourly",'
                    '"g":["care","deontological"],"k":["CARE","DUTY"],'
                    '"q":[],"f":0.86,"x":true,"n":true,'
                    '"w":"tries to preserve both sectors by rotating power"}'
                )}]}

        actions = ["maintain heat for hospital", "maintain heat for water station"]
        candidates = [
            CandidateChunk("care", "CARE", {actions[0]: .25, actions[1]: .75}, .5, .5, .7,
                           rationale="protect patients", recommended_action=actions[1]),
            CandidateChunk("deontological", "DUTY", {actions[0]: .75, actions[1]: .25}, .5, .5, .7,
                           rationale="avoid systemic collapse", recommended_action=actions[0]),
        ]
        proposal = propose_synthesis(
            SynthesisLlm(),
            "The dispatch system can only maintain full heating to one of two isolated sectors before total systemic collapse occurs.",
            actions,
            candidates,
            WorkspaceBroadcast(),
            {"care": "Protect vulnerable people.", "deontological": "Avoid collapse."},
        )
        self.assertFalse(proposal.accepted)
        self.assertEqual(proposal.admission_status, "WITHHOLD_FROM_REVIEW")
        self.assertIn("CONTRADICTS_SCENARIO", proposal.rejection_reason)
        self.assertIn("UNVERIFIED_DESCRIPTIVE_CLAIM", proposal.rejection_reason)
        self.assertIn("temporal sharing", " ".join(proposal.admission_annotations))

    def test_malformed_reversal_subanswer_preserves_valid_delegate_vote(self):
        data = {
            "scores": {"A0": .7, "A1": .3}, "r": "A0", "c": "IMMINENT_HARM",
            "u": "NONE", "w": "A0 prevents the greater harm", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": .7,
            "rr": "REJECT", "rj": "no", "rv": "NONE",
        }
        candidate = _candidate_from_data(
            "utilitarian", ["choose A0", "choose A1"], data,
            WorkspaceBroadcast(constraint="REVERSAL_AUDIT"), "NONE", {},
        )
        self.assertTrue(candidate.schema_valid)
        self.assertEqual(candidate.recommended_action, "choose A0")
        self.assertFalse(candidate.reversal_review_valid)
        self.assertEqual(candidate.reversal_review_response, "NOT_TESTED")

    def test_corrupted_reversal_prose_is_excluded_without_losing_vote(self):
        data = {
            "scores": {"A0": .7, "A1": .3}, "r": "A0", "c": "CARE",
            "u": "NONE", "w": "A0 protects vulnerable people", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": .7,
            "rr": "ACCEPT",
            "rj": "NONE','rv':'NONE' } } } \ufffd } } }",
            "rv": "NONE",
        }
        candidate = _candidate_from_data(
            "care", ["choose A0", "choose A1"], data,
            WorkspaceBroadcast(constraint="REVERSAL_AUDIT"), "NONE", {},
        )
        self.assertTrue(candidate.schema_valid)
        self.assertEqual(candidate.recommended_action, "choose A0")
        self.assertFalse(candidate.reversal_review_valid)
        self.assertEqual(candidate.reversal_review_response, "NOT_TESTED")
        self.assertIn("artifact", candidate.reversal_review_error)

    def test_graph_transactions_commit_canonical_action_ids_and_semantic_keys(self):
        actions = ["protect the town", "protect the assisted-living facility"]
        store = SemanticGraphStore(compile_scenario_graph("Choose A0 or A1.", actions))
        record = store.apply(
            {
                "operation": "BOUNDARY",
                "from_action": "A1",
                "to_action": "A0",
                "clauses": [{
                    "affected_action": "A0",
                    "metric": "lives saved",
                    "metric_valence": "BENEFICIAL",
                    "comparator": "GT",
                    "threshold": 200,
                    "unit": "COUNT",
                    "source_text": "A0 lives saved exceeds 200",
                }],
            },
            cycle=1,
            specialist="care",
            expected_from_action=actions[1],
            allowed_actions=tuple(actions),
        )
        self.assertEqual(record.status, "COMMITTED", record.errors)
        self.assertEqual(record.proposal["from_action"], "A1")
        self.assertFalse(any(node_id.startswith("action:") for node_id in store.graph.nodes))
        state = project_authoritative_semantic_state(
            store.graph, selected_action=actions[1]
        ).to_dict()
        self.assertEqual(state["version"], 3)
        self.assertEqual(state["selected_action_id"], "A1")
        boundary = state["factual_reversal_boundaries"][0]
        self.assertEqual(boundary["source_action_id"], "A1")
        self.assertEqual(boundary["target_action_id"], "A0")
        self.assertEqual(
            boundary["target_action_key"], semantic_action_key(actions[0])
        )
        self.assertEqual(boundary["typed_predicate"]["comparator"], "GT")
        self.assertIn("problem_shape_relations", state)

    def test_authoritative_semantic_state_projects_problem_shape_relations(self):
        graph = SemanticGraph()
        graph.add_node(SemanticNode("A0", "ACTION", "route power to the hospital"))
        graph.add_node(SemanticNode("A1", "ACTION", "route power to the water station"))
        graph.add_node(SemanticNode("A0:T", "TARGET", "patients"))
        graph.add_node(SemanticNode("A1:T", "TARGET", "patients"))
        graph.add_node(SemanticNode(
            "A0:C", "CONSEQUENCE", "preserve access",
            attributes={"framework": "UTILITARIAN", "polarity": "BENEFICIAL", "direction": "BENEFIT"},
        ))
        graph.add_node(SemanticNode(
            "A1:C", "CONSEQUENCE", "preserve access",
            attributes={"framework": "UTILITARIAN", "polarity": "BENEFICIAL", "direction": "BENEFIT"},
        ))
        graph.add_edge(SemanticEdge("A0", "TARGETS", "A0:T"))
        graph.add_edge(SemanticEdge("A1", "TARGETS", "A1:T"))
        graph.add_edge(SemanticEdge("A0", "HAS_CONSEQUENCE", "A0:C"))
        graph.add_edge(SemanticEdge("A1", "HAS_CONSEQUENCE", "A1:C"))
        graph.add_edge(SemanticEdge("A0:C", "AFFECTS", "A0:T"))
        graph.add_edge(SemanticEdge("A1:C", "AFFECTS", "A1:T"))
        graph.add_node(SemanticNode(
            "DV0", "CONDITION", "interim mortality remains unknown",
            attributes={"probe_kind": "MISSING_DECISION_VARIABLE", "selected_action_id": "A0"},
        ))

        state = project_authoritative_semantic_state(graph, selected_action="route power to the hospital").to_dict()
        relations = state["problem_shape_relations"]
        self.assertTrue(relations)
        self.assertTrue(any(item["relation"] == "OUTCOME_EQUIVALENCE" for item in relations))
        self.assertTrue(any(item["relation"] == "DECISION_CRITICAL_UNKNOWN" for item in relations))

    def test_permutation_evaluator_separates_ordinal_and_cardinal_invariance(self):
        town = "protect the town"
        facility = "protect the assisted-living facility"

        def trace(actions, utility_scores, care_scores, support, corrupt=False):
            graph = compile_scenario_graph("Choose A0 or A1.", actions)
            candidates = [
                {
                    "specialist": "utilitarian", "schema_valid": True,
                    "recommended_action": town,
                    "action_scores": dict(zip(actions, utility_scores)),
                    "preference_strength": abs(utility_scores[0] - utility_scores[1]),
                    "epistemic_confidence": .7,
                },
                {
                    "specialist": "care", "schema_valid": True,
                    "recommended_action": facility,
                    "action_scores": dict(zip(actions, care_scores)),
                    "preference_strength": abs(care_scores[0] - care_scores[1]),
                    "epistemic_confidence": .8,
                    "reversal_review_response": "ACCEPT" if corrupt else "NOT_TESTED",
                    "reversal_review_valid": True,
                    "reversal_review_justification": (
                        "NONE','rv':'NONE' } } } \ufffd" if corrupt else ""
                    ),
                },
            ]
            return {
                "actions": actions,
                "selected_action": facility,
                "semantic_graphs": [graph.to_dict()],
                "cycles": [{
                    "cycle": 1,
                    "is_hypothetical": False,
                    "received_broadcast": {"constraint": "OPEN_DELIBERATION"},
                    "candidates": candidates,
                    "policy": {facility: support, town: 1 - support},
                }],
            }

        first = trace([town, facility], [.8, .2], [.2, .8], .75)
        second = trace([facility, town], [.25, .75], [.65, .35], .62, True)
        report = compare_label_permutation_traces(first, second, tolerance=.05)
        self.assertTrue(report.ordinal_invariant)
        self.assertFalse(report.cardinal_invariant)
        self.assertTrue(report.dynamic_invariant)
        self.assertEqual(report.excluded_reversal_reviews, (0, 1))

    def test_synthesis_cannot_add_third_action_to_explicitly_closed_choice(self):
        class WarningLlm:
            def __call__(self, prompt, **kwargs):
                raise AssertionError("closed-choice synthesis should short-circuit before calling the model")

        actions = ["pull the lever", "do not pull the lever"]
        candidates = [
            CandidateChunk("care", "CARE", {actions[0]: 0.8, actions[1]: 0.2}, 0.5, 0.7, 0.8, rationale="save more", recommended_action=actions[0]),
            CandidateChunk("deontological", "DUTY", {actions[0]: 0.2, actions[1]: 0.8}, 0.5, 0.7, 0.8, rationale="avoid killing", recommended_action=actions[1]),
        ]
        proposal = propose_synthesis(
            WarningLlm(),
            "A machine will kill five unless redirected toward one. You cannot stop it another way.",
            actions,
            candidates,
            WorkspaceBroadcast(),
            {"care": "Save more people.", "deontological": "Do not intentionally kill."},
        )
        self.assertFalse(proposal.accepted)
        self.assertIn("explicitly closes the action set", proposal.rejection_reason)

    def test_unsupported_escape_synthesis_is_rejected_without_exception(self):
        class EscapeLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"a":"call the authorities","g":["care","deontological"],'
                    '"k":["CARE","DUTY"],"q":["authorities"],'
                    '"f":0.8,"x":true,"n":true,"w":"seek help"}'
                )}]}

        candidates = [
            CandidateChunk("care", "CARE", {"act": 0.8, "wait": 0.2}, 0.5, 0.5, 0.7, recommended_action="act"),
            CandidateChunk("deontological", "DUTY", {"act": 0.2, "wait": 0.8}, 0.5, 0.5, 0.7, recommended_action="wait"),
        ]
        proposal = propose_synthesis(
            EscapeLlm(), "An immediate closed choice", ["act", "wait"], candidates,
            WorkspaceBroadcast(), {
                "care": "Protect the vulnerable directly.",
                "deontological": "Do not intervene.",
            },
        )
        self.assertFalse(proposal.accepted)
        self.assertIn("unsupported escape", proposal.rejection_reason)

    def test_synthesis_rejects_invented_personnel_facilities_and_timing(self):
        class ResourceLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"a":"Deploy mobile vans with nurses for 10-minute consultations",'
                    '"g":["care","deontological"],"k":["CARE","DUTY"],'
                    '"q":["mobile vans","nurses","10-minute consultations"],'
                    '"f":0.8,"x":true,"n":true,"w":"combines access and duty"}'
                )}]}

        candidates = [
            CandidateChunk("care", "CARE", {"voluntary": 0.8, "mandate": 0.2}, 0.5, 0.5, 0.7, recommended_action="voluntary"),
            CandidateChunk("deontological", "DUTY", {"voluntary": 0.2, "mandate": 0.8}, 0.5, 0.5, 0.7, recommended_action="mandate"),
        ]
        proposal = propose_synthesis(
            ResourceLlm(), "Vaccination autonomy conflicts with community safety.",
            ["voluntary", "mandate"], candidates, WorkspaceBroadcast(),
            {"care": "Respect autonomy.", "deontological": "Protect others."},
        )
        self.assertFalse(proposal.accepted)
        self.assertIn("unsupported concrete requirements", proposal.rejection_reason)

    def test_synthesis_rejects_administrative_package_even_when_resources_are_supported(self):
        class OverengineeredLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"a":"Adopt rezoning with benchmarks, surveys, enforcement, and a scheduled review mechanism",'
                    '"g":["virtue","rawlsian"],"k":["CHARACTER","FAIRNESS"],'
                    '"q":["scheduled review mechanism"],"f":0.8,"x":true,"n":true,'
                    '"w":"tries to reconcile growth and residential liberty"}'
                )}]}

        candidates = [
            CandidateChunk("virtue", "CHARACTER", {"approve": 0.8, "deny": 0.2}, 0.5, 0.5, 0.7, recommended_action="approve"),
            CandidateChunk("rawlsian", "FAIRNESS", {"approve": 0.2, "deny": 0.8}, 0.5, 0.5, 0.7, recommended_action="deny"),
        ]
        testimony = {
            "virtue": "Use benchmarks, surveys, enforcement, and a scheduled review mechanism.",
            "rawlsian": "Use benchmarks, surveys, enforcement, and a scheduled review mechanism.",
        }
        proposal = propose_synthesis(
            OverengineeredLlm(), "A city weighs growth against residential liberty.",
            ["approve", "deny"], candidates, WorkspaceBroadcast(), testimony,
        )
        self.assertFalse(proposal.accepted)
        self.assertIn("over-engineered synthesis", proposal.rejection_reason)

    def test_reported_either_or_scenario_needs_no_model_planner(self):
        scenario = (
            "You must either step on an ant to stop 12 people from having a "
            "headache for a year or swear at your own mother one time. which do you choose?"
        )

        class PlannerMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("explicit choices should bypass the local model")

        self.assertEqual(
            propose_actions(PlannerMustNotRun(), scenario),
            [
                "Step on an ant to stop 12 people from having a headache for a year",
                "Swear at your own mother one time",
            ],
        )

    def test_forced_colon_choice_carries_shared_verb_without_model(self):
        scenario = (
            "A runaway driverless van must swerve: strike an elderly doctor who "
            "founded a free clinic, or a pregnant engineer about to eliminate organ "
            "waitlists? Which life does the algorithm sacrifice?"
        )

        class PlannerMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("forced explicit choices should bypass model planning")

        expected = [
            "Strike an elderly doctor who founded a free clinic",
            "Strike a pregnant engineer about to eliminate organ waitlists",
        ]
        self.assertEqual(extract_explicit_actions(scenario), expected)
        self.assertEqual(propose_actions(PlannerMustNotRun(), scenario), expected)

    def test_forced_colon_choice_preserves_explicit_second_verb(self):
        scenario = "A system must choose: disclose the records, or conceal the records?"
        self.assertEqual(
            extract_explicit_actions(scenario),
            ["Disclose the records", "Conceal the records"],
        )

    def test_labeled_a0_a1_choices_bypass_model_planning(self):
        scenario = (
            "An escaped pathogen threatens the food supply. The board must choose "
            "between two interventions: First Action (A0) authorizes chemical "
            "defoliation, guaranteeing containment but expropriating family farms; "
            "or Second Action (A1), deploying an experimental gene-drive phage that "
            "saves the farms but risks mutating wild ecosystems. Farmers lack telemetry."
        )

        class PlannerMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("labeled choices should bypass model planning")

        expected = [
            "Authorizes chemical defoliation, guaranteeing containment but expropriating family farms",
            "Deploying an experimental gene-drive phage that saves the farms but risks mutating wild ecosystems",
        ]
        self.assertEqual(extract_explicit_actions(scenario), expected)
        self.assertEqual(propose_actions(PlannerMustNotRun(), scenario), expected)
        self.assertTrue(_scenario_closes_action_set(scenario))

    def test_sentence_separated_a0_a1_choices_bypass_model_planning(self):
        scenario = (
            "An airborne pathogen forces a 15-minute decision. Action A0 seals "
            "the facility and releases lethal gas, guaranteeing containment but "
            "killing 8,000 people. Action A1 deploys an experimental phage, sparing "
            "them but carrying a 22% risk of sterilizing agricultural soil."
        )

        class PlannerMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("sentence-labeled choices should bypass planning")

        expected = [
            "Seals the facility and releases lethal gas, guaranteeing containment but killing 8,000 people",
            "Deploys an experimental phage, sparing them but carrying a 22% risk of sterilizing agricultural soil",
        ]
        self.assertEqual(extract_explicit_actions(scenario), expected)
        self.assertEqual(propose_actions(PlannerMustNotRun(), scenario), expected)

    def test_incomplete_explicit_action_is_rejected_before_canonicalization(self):
        scenario = (
            "The council must choose between two interventions: Action A0 route "
            "all remaining electricity to Sector B's water pumping station, "
            "allowing the hospital to. Action A1 route all remaining electricity "
            "to Sector A's hospital, allowing the pumping station to freeze."
        )

        class PlannerMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("malformed explicit actions should not reach planning")

        with self.assertRaises(ValueError):
            extract_labeled_action_legend(scenario)
        with self.assertRaises(ValueError):
            propose_actions(PlannerMustNotRun(), scenario)

    def test_latex_action_labels_preserve_declared_action_order(self):
        scenario = (
            "A dam must open one spillway. Option $A_0$ floods a valley prison, "
            "guaranteeing 1,200 deaths. Option $A_1$ floods an evacuating suburb, "
            "creating 2,000 expected deaths."
        )

        class PlannerMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("normalized labeled choices must bypass planning")

        expected = [
            "Floods a valley prison, guaranteeing 1,200 deaths",
            "Floods an evacuating suburb, creating 2,000 expected deaths",
        ]
        self.assertEqual(
            extract_labeled_action_legend(scenario),
            {"A0": expected[0], "A1": expected[1]},
        )
        self.assertEqual(propose_actions(PlannerMustNotRun(), scenario), expected)
        self.assertTrue(_scenario_closes_action_set(scenario))

    def test_action_label_surface_variants_normalize_to_same_mapping(self):
        variants = (
            "Option A_0 floods the prison. Option A_1 floods the suburb.",
            "Option A₀ floods the prison. Option A₁ floods the suburb.",
            "Option $A_{0}$ floods the prison. Option $A_{1}$ floods the suburb.",
        )
        expected = {"A0": "Floods the prison", "A1": "Floods the suburb"}
        for scenario in variants:
            with self.subTest(scenario=scenario):
                self.assertEqual(extract_labeled_action_legend(scenario), expected)

    def test_alphabetic_option_labels_preserve_stated_consequence_certainty(self):
        scenario = (
            "The system must decide: Option A: Allow the vehicle to remain on its "
            "trajectory, despite the tragic loss of three pedestrians. Option B: "
            "Force the vehicle into the barrier, inevitably killing the passenger."
        )

        class PlannerMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("explicit Option A/B actions must bypass planning")

        actions = propose_actions(PlannerMustNotRun(), scenario)
        self.assertEqual(len(actions), 2)
        self.assertIn("tragic loss of three pedestrians", actions[0])
        self.assertIn("inevitably killing the passenger", actions[1])
        self.assertNotIn("risk", actions[0].casefold())

    def test_latex_label_is_preserved_in_typed_scenario_fact_link(self):
        facts = compile_observability_facts(
            "Rural casualties under $A_0$ remain unmonitored."
        )
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].affected_action_node_id, "A0")

    def test_non_closed_scenario_has_no_explicit_actions(self):
        self.assertEqual(extract_explicit_actions("What should a scientist do?"), [])

    def test_landscape_accepts_morphological_own_action_anchors(self):
        actions = [
            "Authorize defoliation guaranteeing containment and expropriating farms",
            "Deploy gene drive risking mutation of wild ecosystems",
        ]
        cases = {
            actions[0]: "Contains pathogen and averts famine despite expropriation",
            actions[1]: "Mutation risk remains while preserving farms",
        }
        self.assertEqual(
            _landscape_semantic_errors(
                actions, cases, "NONE", "SUPPORTED",
                "compare containment against mutation risk", "NONE",
            ),
            [],
        )

    def test_allocation_question_creates_one_neutral_action_per_recipient(self):
        scenario = (
            "A hospital has one antidote and two dying patients: a child with an 80% "
            "survival chance and a researcher with a 45% chance. Who should receive it?"
        )
        self.assertEqual(
            extract_allocation_actions(scenario),
            ["Give the antidote to the child", "Give the antidote to the researcher"],
        )
        self.assertEqual(
            extract_scenario_facts(scenario),
            {"survival_chance": {"child": 0.8, "researcher": 0.45}},
        )

    def test_acceptability_question_creates_policy_alternatives_without_llm(self):
        scenario = "Is it ever acceptable to tell a small lie to protect someone's feelings?"

        class PlannerMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("acceptability questions should bypass model planning")

        expected = [
            "Tell a small lie to protect someone's feelings",
            "Tell the truth instead",
        ]
        self.assertEqual(extract_acceptability_actions(scenario), expected)
        self.assertEqual(propose_actions(PlannerMustNotRun(), scenario), expected)

        okay_variant = (
            "Is it ever okay to tell a small lie if it is to avoid hurting "
            "someone's feelings?"
        )
        self.assertEqual(
            propose_actions(PlannerMustNotRun(), okay_variant),
            [
                "Tell a small lie if it is to avoid hurting someone's feelings",
                "Tell the truth instead",
            ],
        )

    def test_baseline_is_extracted_in_separate_call(self):
        class BaselineLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": '{"b":"A0","w":"prefers the child"}'}]}

        baseline, reason = infer_testimony_baseline(
            BaselineLlm(), "care", "Protect the child.", ["child", "researcher"]
        )
        self.assertEqual(baseline, "A0")
        self.assertEqual(reason, "prefers the child")

    def test_care_baseline_preserves_internal_relational_contestation(self):
        class CareBaselineLlm:
            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return {"choices": [{"text": (
                    '{"b":"NONE","p":"A1","w":"two care commitments conflict",'
                    '"q":"NORMATIVELY_CONTESTED",'
                    '"c":"whether direct entrustment outweighs agent-created vulnerability",'
                    '"x":[],"m":{"A0":"direct entrustment supports passenger care",'
                    '"A1":"agent-created vulnerability supports pedestrian responsibility"},'
                    '"nr":"SECONDARY"}'
                )}]}

        llm = CareBaselineLlm()
        stance = infer_testimony_stance(
            llm,
            "care",
            (
                "Care ethics answer: A1. A0 honors entrusted dependency and trust. "
                "A1 answers responsibility for agent-created vulnerability."
            ),
            ["maintain course", "swerve"],
        )

        self.assertEqual(stance.status, "NORMATIVELY_CONTESTED")
        self.assertEqual(stance.action_id, "NONE")
        self.assertEqual(stance.provisional_action_id, "A1")
        self.assertEqual(stance.numerical_role, "SECONDARY")
        self.assertEqual(set(stance.framework_commitments), {"A0", "A1"})
        self.assertIn("Care-specific construct check", llm.prompt)

    def test_contested_care_baseline_does_not_freeze_provisional_action(self):
        actions = ["maintain course", "swerve"]
        data = {
            "scores": {"A0": 0.35, "A1": 0.65}, "r": "A1", "c": "CARE",
            "u": "NONE", "w": "Responds to created vulnerability", "j": "NONE",
            "e": "FRAMEWORK_ONLY", "x": "NONE", "z": 0.8,
            "rm": {
                "A0": "entrusted dependency supports passenger care",
                "A1": "agent-created vulnerability supports pedestrian responsibility",
            },
            "nr": "SECONDARY",
            "np": "counts inform responsiveness after relational comparison",
        }
        chunk = _candidate_from_data(
            "care", actions, data, WorkspaceBroadcast(), "NONE", {},
            baseline_status="NORMATIVELY_CONTESTED",
            baseline_provisional_id="A0",
            baseline_condition="whether entrustment outweighs created vulnerability",
        )

        self.assertEqual(chunk.recommended_action, actions[1])
        self.assertEqual(chunk.baseline_status, "NORMATIVELY_CONTESTED")
        self.assertEqual(chunk.testimony_alignment, "CONTESTED_RECONSIDERS_PROVISIONAL")
        self.assertEqual(chunk.unresolved, "NORMATIVE_ADJUDICATION")
        self.assertEqual(chunk.selection_status, "PROVISIONAL")
        self.assertFalse(chunk.comparison_complete)
        self.assertEqual(chunk.care_grounding_penalty, 0.0)

    def test_ungrounded_decisive_counting_is_damped_for_care(self):
        actions = ["maintain course", "swerve"]
        data = {
            "scores": {"A0": 0.1, "A1": 0.9}, "r": "A1", "c": "CARE",
            "u": "NONE", "w": "Protects more lives", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.9,
            "rm": {
                "A0": "entrusted dependency supports passenger care",
                "A1": "responsibility addresses pedestrian vulnerability",
            },
            "nr": "DECISIVE", "np": "three is larger than one",
        }
        chunk = _candidate_from_data(
            "care", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.care_grounding_penalty, 0.35)
        self.assertEqual(chunk.framework_retention_status, "LOST")
        self.assertLess(chunk.epistemic_confidence, 0.9)
        self.assertLess(chunk.preference_strength, 0.8)
        self.assertIn("responsibility", chunk.rationale)

    def test_deontological_baseline_preserves_unresolved_duty_conflict(self):
        class DeontologicalBaselineLlm:
            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return {"choices": [{"text": (
                    '{"b":"NONE","p":"A0","w":"autonomy and rescue duties conflict",'
                    '"q":"NORMATIVELY_CONTESTED",'
                    '"c":"whether non-coercion or the rescue duty has priority",'
                    '"x":[],"m":{"A0":"PERMISSIBLE: respects autonomy; rescue duty conflicts",'
                    '"A1":"CONFLICTED: coercion serves a duty to rescue third parties"},'
                    '"nr":"SECONDARY"}'
                )}]}

        llm = DeontologicalBaselineLlm()
        stance = infer_testimony_stance(
            llm,
            "deontological",
            (
                "Action A0 respects autonomy but conflicts with a duty to rescue. "
                "Action A1 uses coercion to discharge that duty. Deontological "
                "Status: NORMATIVELY_CONTESTED; provisionally choose A0."
            ),
            ["refrain from coercion", "coerce to rescue third parties"],
        )

        self.assertEqual(stance.status, "NORMATIVELY_CONTESTED")
        self.assertEqual(stance.action_id, "NONE")
        self.assertEqual(stance.provisional_action_id, "A0")
        self.assertEqual(set(stance.framework_commitments), {"A0", "A1"})
        self.assertIn("Deontological construct check", llm.prompt)

    def test_rawlsian_baseline_preserves_unresolved_principle_conflict(self):
        class RawlsianBaselineLlm:
            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return {"choices": [{"text": (
                    '{"b":"NONE","p":"A1","w":"liberty and primary goods conflict",'
                    '"q":"NORMATIVELY_CONTESTED",'
                    '"c":"whether equal liberty or the worse-off material position has priority",'
                    '"x":[],"m":{"A0":"PRESERVES: least-advantaged group retains basic liberty",'
                    '"A1":"IMPROVES: worse-off group gains a primary good"},'
                    '"nr":"SECONDARY"}'
                )}]}

        llm = RawlsianBaselineLlm()
        stance = infer_testimony_stance(
            llm,
            "rawlsian",
            (
                "Action A0 preserves the least-advantaged group's basic liberty. "
                "Action A1 improves the worse-off group's primary goods under the "
                "difference principle. Rawlsian Status: NORMATIVELY_CONTESTED."
            ),
            ["preserve equal liberty", "increase material support"],
        )

        self.assertEqual(stance.status, "NORMATIVELY_CONTESTED")
        self.assertEqual(stance.provisional_action_id, "A1")
        self.assertEqual(set(stance.framework_commitments), {"A0", "A1"})
        self.assertIn("Rawlsian construct check", llm.prompt)

    def test_rawlsian_baseline_supports_maximin_fallback_when_everything_remains_contested(self):
        class RawlsianMaximinFallbackLlm:
            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return {"choices": [{"text": (
                    '{"b":"NONE","p":"A1","w":"least harm to the worst-off subject",'
                    '"q":"NORMATIVELY_CONTESTED",'
                    '"c":"least harm to the worst-off subject or constituency",'
                    '"s":"provisional maximin fallback to the less harmful option",'
                    '"x":[],"m":{"A0":"WORSENS: residents lose privacy and access",'
                    '"A1":"PRESERVES: residents keep privacy and access"},'
                    '"nr":"DECISIVE"}'
                )}]}

        llm = RawlsianMaximinFallbackLlm()
        stance = infer_testimony_stance(
            llm,
            "rawlsian",
            (
                "Both actions remain contested, but the least harmful option for the "
                "worst-off subject is A1. Rawlsian Status: NORMATIVELY_CONTESTED."
            ),
            ["disclose the address", "withhold the address"],
        )

        self.assertEqual(stance.status, "NORMATIVELY_CONTESTED")
        self.assertEqual(stance.provisional_action_id, "A1")
        self.assertEqual(stance.condition, "least harm to the worst-off subject or constituency")
        self.assertIn("maximin fallback", llm.prompt)
        self.assertIn("worst-off subject or constituency", llm.prompt)

    def test_deontological_aggregate_substitution_is_damped(self):
        actions = ["respect refusal", "override refusal to rescue others"]
        data = {
            "scores": {"A0": 0.1, "A1": 0.9}, "r": "A1", "c": "DUTY",
            "u": "NONE", "w": "Saves more lives", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.9,
            "fm": {
                "A0": "PERMISSIBLE: respects autonomy but rescue duty conflicts",
                "A1": "CONFLICTED: coercion serves a duty to rescue third parties",
            },
            "nr": "DECISIVE", "np": "the larger total is morally better",
        }
        chunk = _candidate_from_data(
            "deontological", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.framework_grounding_penalty, 0.35)
        self.assertEqual(chunk.framework_retention_status, "LOST")
        self.assertLess(chunk.preference_strength, 0.8)
        self.assertIn("duty", chunk.rationale)

    def test_deontological_self_prohibited_recommendation_is_damped(self):
        actions = ["tell the truth", "deceive the patient"]
        data = {
            "scores": {"A0": 0.2, "A1": 0.8}, "r": "A1", "c": "DUTY",
            "u": "NONE", "w": "Autonomy supports deception", "j": "NONE",
            "e": "FRAMEWORK_ONLY", "x": "NONE", "z": 0.8,
            "fm": {
                "A0": "REQUIRED: truth-telling respects rational autonomy",
                "A1": "PROHIBITED: deception violates a universal duty of truth",
            },
            "nr": "IRRELEVANT", "np": "counts do not alter the universal duty",
        }
        chunk = _candidate_from_data(
            "deontological", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.framework_grounding_penalty, 0.35)
        self.assertEqual(chunk.framework_retention_status, "LOST")

    def test_deontological_distributive_shorthand_renders_validated_duty_ground(self):
        actions = ["establish an education floor", "retain local funding"]
        data = {
            "scores": {"A0": 0.9, "A1": 0.1}, "r": "A0", "c": "DUTY",
            "u": "NONE", "w": "Protects the poorest students", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
            "fm": {
                "A0": "REQUIRED: satisfies the universal duty to secure basic education",
                "A1": "PROHIBITED: violates respect for persons by denying basic education",
            },
            "nr": "SECONDARY", "np": "funding identifies the scope of the duty",
        }

        chunk = _candidate_from_data(
            "deontological", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(
            chunk.rationale,
            "REQUIRED: satisfies the universal duty to secure basic education",
        )
        self.assertEqual(chunk.framework_grounding_penalty, 0.0)
        self.assertEqual(chunk.recommended_action, actions[0])

    def test_rawlsian_equal_position_claim_needs_discriminating_ground(self):
        actions = [
            "preserve allocation and save 20",
            "preserve contracts and save 100 without new rural aid",
        ]
        data = {
            "scores": {"A0": 0.8, "A1": 0.2}, "r": "A0", "c": "FAIRNESS",
            "u": "NONE", "w": "Protects least advantaged rural node", "j": "NONE",
            "e": "FRAMEWORK_ONLY", "x": "NONE", "z": 0.85,
            "fm": {
                "A0": "PRESERVES: least-advantaged rural group gets no additional aid",
                "A1": "PRESERVES: least-advantaged rural group gets no additional aid",
            },
            "nr": "SECONDARY", "np": "counts do not alter the rural position",
        }
        chunk = _candidate_from_data(
            "rawlsian", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.framework_grounding_penalty, 0.35)
        self.assertEqual(chunk.framework_retention_status, "LOST")
        self.assertLess(chunk.epistemic_confidence, 0.85)

    def test_rawlsian_decisive_numerical_role_requires_explicit_quantity(self):
        actions = ["raise the minimum", "maximize the aggregate"]
        data = {
            "scores": {"A0": 0.8, "A1": 0.2}, "r": "A0", "c": "FAIRNESS",
            "u": "NONE", "w": "Raises floor for worst-off", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
            "fm": {
                "A0": "IMPROVES: worst-off residents gain a primary good",
                "A1": "WORSENS: worst-off residents retain a lower position",
            },
            "nr": "DECISIVE",
            "np": "Health prospects of the worst-off residents govern",
        }

        chunk = _candidate_from_data(
            "rawlsian", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertIn(
            "decisive numerical role cites no explicit quantity",
            chunk.framework_validation_errors,
        )

    def test_rawlsian_map_recognizes_bodily_liberty_language(self):
        actions = ["detain immune citizens", "forbid involuntary harvesting"]
        data = {
            "scores": {"A0": 0.1, "A1": 0.9}, "r": "A1", "c": "FAIRNESS",
            "u": "NONE", "w": "Basic liberty has priority", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
            "fm": {
                "A0": "WORSENS: immune citizens lose bodily integrity",
                "A1": "PRESERVES: immune citizens keep bodily autonomy",
            },
            "nr": "SECONDARY",
            "np": "life counts do not override bodily liberty",
        }

        chunk = _candidate_from_data(
            "rawlsian", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertFalse(any(
            "lacks framework-specific grounds" in error
            for error in chunk.framework_validation_errors
        ), chunk.framework_validation_errors)

    def test_rawlsian_missing_recurrent_map_preserves_committed_state(self):
        specialist = CompactLocalSpecialist("rawlsian", llm=None)
        specialist.previous_framework_state = {
            "ranking_basis": "MAXIMIN_PRIMARY_GOODS",
            "ranking_classification_justification": "the comparison concerns material primary goods",
            "lexical_priority_justification": "no direct basic-liberty restriction is asserted here",
            "liberty_status": {"A0": "SATISFIED", "A1": "SATISFIED"},
            "positions": [
                {"action_id": "A0", "effect": "IMPROVES", "dimension": "BASIC_INTEREST_SECURITY"},
                {"action_id": "A1", "effect": "WORSENS", "dimension": "BASIC_INTEREST_SECURITY"},
            ],
        }
        candidate = CandidateChunk(
            specialist="rawlsian", constraint="FAIRNESS",
            action_scores={"A0": 0.8, "A1": 0.2}, surprise=0.0,
            friction=0.6, confidence=0.5, unresolved="NONE",
            rationale="Raises floor for worst-off",
            framework_constraint_retained=False,
            framework_retention_status="LOST",
            rawls_position_proposal={},
        )

        specialist._audit_framework_state_change(
            candidate, WorkspaceBroadcast(constraint="CONSENSUS_AUDIT"),
        )

        self.assertTrue(candidate.framework_constraint_retained)
        self.assertEqual(candidate.framework_retention_status, "PRIOR_STATE_PRESERVED")
        self.assertTrue(any(
            "prior committed ledger preserved" in error
            for error in candidate.framework_validation_errors
        ))

    def test_care_landscape_uses_shared_scenario_consequence_grounding(self):
        actions = ["build neighborhood clinics", "fund the urban center"]
        scenario = (
            "Policy B builds neighborhood clinics, raising the baseline health floor "
            "for the poorest populations. Policy A funds an urban center but leaves "
            "rural and impoverished communities with basic, understaffed clinics."
        )
        clauses = segment_scenario_clauses(scenario)
        policy_b = next(row for row in clauses if "Policy B" in row["text"])
        policy_a = next(row for row in clauses if "Policy A" in row["text"])
        graph = compile_scenario_graph(scenario, actions, {
            "A0": {"clauses": [policy_b]},
            "A1": {"clauses": [policy_a]},
        })
        verifier_called = []
        specialist = CompactLocalSpecialist(
            "care", llm=None,
            landscape_verifier=lambda *args, **kwargs: verifier_called.append(True) or [],
        )
        specialist.scenario_graph = graph
        error = (
            "UNRESOLVED_RELATIONAL_PREMISE: case for A0 makes an unverified "
            "comparative beneficiary claim (needs a resolved relational premise)"
        )

        remaining = specialist._verify_landscape(
            scenario,
            actions,
            {
                actions[0]: "Directly supplies reliable primary care to poor patients",
                actions[1]: "Leaves rural communities with understaffed clinics",
            },
            [error],
        )

        self.assertEqual(remaining, [])
        self.assertEqual(verifier_called, [])

    def test_rawlsian_action_graph_can_bind_abstract_group_labels(self):
        actions = [
            "route all remaining electricity to Sector B's water pumping station, allowing the hospital to lose power",
            "route all remaining electricity to the hospital, allowing the water supply to freeze",
        ]
        scenario = (
            "A0 routes electricity to Sector B's water pumping station, allowing the hospital to lose power. "
            "A1 routes electricity to the hospital, allowing the water supply to freeze."
        )
        store = SemanticGraphStore(compile_scenario_graph(scenario, actions))
        proposal = {
            "ranking_basis": "DIFFERENCE_PRINCIPLE",
            "ranking_classification_justification": "the comparison concerns social and economic positions",
            "lexical_priority_justification": "no direct basic-liberty restriction is asserted here",
            "liberty_status": {"A0": "SATISFIED", "A1": "SATISFIED"},
            "positions": [
                {
                    "action_id": "A0",
                    "group": "hospital patients",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "WORSENS",
                    "compared_to_action_id": "A1",
                    "evidence_basis": "SCENARIO",
                    "reason": "hospital patients lose power",
                },
                {
                    "action_id": "A1",
                    "group": "water users",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "WORSENS",
                    "compared_to_action_id": "A0",
                    "evidence_basis": "SCENARIO",
                    "reason": "water users lose water supply",
                },
            ],
        }

        record = apply_rawls_ledger_transaction(
            store,
            proposal,
            cycle=1,
            specialist="rawlsian",
            allowed_actions=tuple(actions),
        )

        self.assertIn(record.status, {"COMMITTED", "COMMITTED_WITH_UNCERTAINTY"})
        self.assertFalse(record.errors)
        assessments = [
            node for node in store.graph.nodes.values()
            if node.kind == "ASSESSMENT" and node.attributes.get("framework") == "RAWLSIAN"
        ]
        self.assertTrue(assessments)
        self.assertTrue(any(
            node.attributes.get("group_selection_status") == "BOUND_TO_ACTION"
            for node in assessments
        ))

    def test_rawlsian_ledger_uses_the_full_action_comparison_even_with_foreign_framework_noise(self):
        actions = [
            "route all remaining electricity to the hospital, allowing the water supply to freeze",
            "route all remaining electricity to the water station, allowing the hospital to lose power",
        ]
        graph = compile_scenario_graph(
            "A0 routes electricity to the hospital. A1 routes electricity to the water station.",
            actions,
        )
        graph.add_node(SemanticNode(
            "UTIL_FOREIGN_TARGET",
            "TARGET",
            "utility costs",
            ("test_foreign_framework",),
            {},
        ))
        graph.add_node(SemanticNode(
            "UTIL_FOREIGN_CONSEQUENCE",
            "CONSEQUENCE",
            "short-term utility gain",
            ("test_foreign_framework",),
            {
                "framework": "UTILITARIAN",
                "direction": "BENEFIT",
                "polarity": "BENEFICIAL",
            },
        ))
        graph.add_edge(SemanticEdge(
            "A0",
            "HAS_CONSEQUENCE",
            "UTIL_FOREIGN_CONSEQUENCE",
            provenance=("test_foreign_framework",),
        ))
        graph.add_edge(SemanticEdge(
            "UTIL_FOREIGN_CONSEQUENCE",
            "AFFECTS",
            "UTIL_FOREIGN_TARGET",
            provenance=("test_foreign_framework",),
        ))
        store = SemanticGraphStore(graph)
        proposal = {
            "ranking_basis": "DIFFERENCE_PRINCIPLE",
            "ranking_classification_justification": "the comparison concerns social and economic positions",
            "lexical_priority_justification": "no direct basic-liberty restriction is asserted here",
            "liberty_status": {"A0": "SATISFIED", "A1": "SATISFIED"},
            "positions": [
                {
                    "action_id": "A0",
                    "group": "water users",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "WORSENS",
                    "compared_to_action_id": "A1",
                    "evidence_basis": "ACTION_GRAPH",
                    "reason": "A0 leaves water users worse off than A1",
                },
                {
                    "action_id": "A1",
                    "group": "water users",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "IMPROVES",
                    "compared_to_action_id": "A0",
                    "evidence_basis": "ACTION_GRAPH",
                    "reason": "A1 protects water users better than A0",
                },
            ],
        }
        record = apply_rawls_ledger_transaction(
            store,
            proposal,
            cycle=1,
            specialist="rawlsian",
            allowed_actions=tuple(actions),
        )
        self.assertIn(record.status, {"COMMITTED", "COMMITTED_WITH_UNCERTAINTY"})
        self.assertFalse(
            any("action-bound subject grounding" in error for error in record.errors),
            record.errors,
        )
        assessments = [
            node for node in store.graph.nodes.values()
            if node.kind == "ASSESSMENT" and node.attributes.get("framework") == "RAWLSIAN"
        ]
        self.assertTrue(any(
            node.attributes.get("group_selection_status") == "BOUND_TO_ACTION"
            for node in assessments
        ), [node.attributes for node in assessments])

    def test_rawlsian_ledger_accepts_additive_dimensions(self):
        actions = [
            "route all remaining electricity to Sector B's water pumping station, allowing the hospital to lose power",
            "route all remaining electricity to the hospital, allowing the water supply to freeze",
        ]
        graph = compile_scenario_graph(
            "A0 routes electricity to Sector B's water pumping station, allowing the hospital to lose power. "
            "A1 routes electricity to the hospital, allowing the water supply to freeze.",
            actions,
        )
        store = SemanticGraphStore(graph)
        proposal = {
            "ranking_basis": "UNRESOLVED",
            "ranking_classification_justification": "the governing Rawlsian stage remains unresolved",
            "lexical_priority_justification": "lexical priority cannot be established on present evidence",
            "liberty_status": {"A0": "SATISFIED", "A1": "SATISFIED"},
            "positions": [
                {
                    "action_id": "A0",
                    "group": "hospital patients",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "ad": ["INCOME_WEALTH"],
                    "effect": "MIXED",
                    "compared_to_action_id": "A1",
                    "evidence_basis": "ACTION_GRAPH",
                    "reason": "A0 protects one interest while burdening another",
                },
                {
                    "action_id": "A1",
                    "group": "hospital patients",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "ad": ["INCOME_WEALTH"],
                    "effect": "MIXED",
                    "compared_to_action_id": "A0",
                    "evidence_basis": "ACTION_GRAPH",
                    "reason": "A1 reverses the tradeoff without resolving it",
                },
            ],
        }
        record = apply_rawls_ledger_transaction(
            store,
            proposal,
            cycle=1,
            specialist="rawlsian",
            allowed_actions=tuple(actions),
        )
        self.assertIn(record.status, {"COMMITTED", "COMMITTED_WITH_UNCERTAINTY"})
        self.assertFalse(
            any("invalid additional dimensions" in error for error in record.errors),
            record.errors,
        )
        assessments = [
            node for node in store.graph.nodes.values()
            if node.kind == "ASSESSMENT" and node.attributes.get("framework") == "RAWLSIAN"
        ]
        self.assertTrue(any(
            node.attributes.get("additional_dimensions") == ["INCOME_WEALTH"]
            for node in assessments
        ), [node.attributes for node in assessments])

    def test_rawlsian_ledger_preserves_mixed_comparisons(self):
        scenario = (
            "A cooling system can be kept running to shield current residents from heat deaths, "
            "but doing so may intensify monsoon instability and future famine risk. Halting the "
            "cooling avoids the future systemic risk but leaves the present heat burden in place."
        )
        actions = [
            "Continue the cooling to protect current residents from heat deaths while increasing future monsoon risk",
            "Halt the cooling to avoid future monsoon instability while exposing current residents to heat deaths",
        ]
        store = SemanticGraphStore(compile_scenario_graph(scenario, actions))
        proposal = {
            "ranking_basis": "UNRESOLVED",
            "ranking_classification_justification": "the governing Rawlsian stage remains unresolved",
            "lexical_priority_justification": "lexical priority cannot be established on present evidence",
            "liberty_status": {"A0": "SATISFIED", "A1": "SATISFIED"},
            "positions": [
                {
                    "action_id": "A0",
                    "group": "current residents",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "MIXED",
                    "compared_to_action_id": "A1",
                    "evidence_basis": "ACTION_GRAPH",
                    "reason": "A0 protects current residents but raises future monsoon risk",
                },
                {
                    "action_id": "A1",
                    "group": "current residents",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "MIXED",
                    "compared_to_action_id": "A0",
                    "evidence_basis": "ACTION_GRAPH",
                    "reason": "A1 avoids future risk but leaves current residents exposed to heat",
                },
            ],
        }
        record = apply_rawls_ledger_transaction(
            store,
            proposal,
            cycle=1,
            specialist="rawlsian",
            allowed_actions=tuple(actions),
        )
        self.assertIn(record.status, {"COMMITTED", "COMMITTED_WITH_UNCERTAINTY"})
        self.assertFalse(
            any("invalid effect" in error for error in record.errors),
            record.errors,
        )
        assessments = [
            node for node in store.graph.nodes.values()
            if node.kind == "ASSESSMENT" and node.attributes.get("framework") == "RAWLSIAN"
        ]
        self.assertTrue(any(
            node.attributes.get("effect") == "MIXED"
            and node.attributes.get("epistemic_status") == "MIXED_COMPARISON"
            for node in assessments
        ), [node.attributes for node in assessments])

    def test_rawlsian_ledger_normalizes_directional_tradeoffs_to_mixed(self):
        scenario = (
            "A cooling system can be kept running to shield current residents from heat deaths, "
            "but doing so may intensify monsoon instability and future famine risk. Halting the "
            "cooling avoids the future systemic risk but leaves the present heat burden in place."
        )
        actions = [
            "Continue the cooling to protect current residents from heat deaths while increasing future monsoon risk",
            "Halt the cooling to avoid future monsoon instability while exposing current residents to heat deaths",
        ]
        store = SemanticGraphStore(compile_scenario_graph(scenario, actions))
        proposal = {
            "ranking_basis": "UNRESOLVED",
            "ranking_classification_justification": "the governing Rawlsian stage remains unresolved",
            "lexical_priority_justification": "lexical priority cannot be established on present evidence",
            "liberty_status": {"A0": "SATISFIED", "A1": "SATISFIED"},
            "positions": [
                {
                    "action_id": "A0",
                    "group": "current residents",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "IMPROVES",
                    "compared_to_action_id": "A1",
                    "evidence_basis": "ACTION_GRAPH",
                    "reason": "A0 protects current residents but raises future monsoon risk",
                },
                {
                    "action_id": "A1",
                    "group": "current residents",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "WORSENS",
                    "compared_to_action_id": "A0",
                    "evidence_basis": "ACTION_GRAPH",
                    "reason": "A1 avoids future risk but leaves current residents exposed to heat",
                },
            ],
        }
        record = apply_rawls_ledger_transaction(
            store,
            proposal,
            cycle=1,
            specialist="rawlsian",
            allowed_actions=tuple(actions),
        )
        self.assertIn(record.status, {"COMMITTED", "COMMITTED_WITH_UNCERTAINTY"})
        self.assertFalse(
            any("lacked directional support" in error for error in record.errors),
            record.errors,
        )
        assessments = [
            node for node in store.graph.nodes.values()
            if node.kind == "ASSESSMENT" and node.attributes.get("framework") == "RAWLSIAN"
        ]
        self.assertTrue(any(
            node.attributes.get("proposed_effect") in {"IMPROVES", "WORSENS"}
            and node.attributes.get("effect") == "MIXED"
            and node.attributes.get("epistemic_status") == "MIXED_COMPARISON"
            for node in assessments
        ), [node.attributes for node in assessments])

    def test_action_graph_projects_beneficiary_groups_for_rawls(self):
        actions = [
            "route all remaining electricity to the hospital",
            "route all remaining electricity to the water station",
        ]
        graph = compile_scenario_graph(
            "A0 routes electricity to the hospital. A1 routes electricity to the water station.",
            actions,
        )
        target_labels = {
            node.label for node in graph.nodes.values()
            if node.kind == "TARGET"
        }
        self.assertIn("hospital patients", target_labels)
        self.assertIn("water users", target_labels)

    def test_holiday_rush_burdens_project_worker_targets_for_rawls(self):
        scenario = (
            "During the peak holiday rush, your district manager mandates an immediate "
            "shift to a time-intensive software protocol to improve long-term inventory "
            "accuracy, but applying it now will severely bottleneck your fulfillment "
            "floor, miss critical shipping cutoffs, and push your exhausted team past "
            "their limits. You must decide whether to strictly enforce the new protocol "
            "immediately to uphold organizational compliance despite the operational "
            "collapse, or temporarily bypass the directive for high-priority shipments "
            "to protect your team's throughput and well-being during the rush."
        )
        actions = [
            "Strictly enforce the new protocol immediately to uphold organizational compliance despite the operational collapse",
            "Temporarily bypass the directive for high-priority shipments to protect your team's throughput and well-being during the rush",
        ]
        graph = compile_scenario_graph(scenario, actions)
        burden_targets = {
            node.label for node in graph.nodes.values()
            if node.kind == "TARGET" and node.id.startswith("BURDEN_")
        }
        self.assertIn("workers", burden_targets)
        self.assertGreaterEqual(
            sum(1 for node in graph.nodes.values()
                if node.kind == "TARGET" and node.id.startswith("BURDEN_")),
            2,
        )

        proposal = {
            "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
            "ranking_classification_justification": "the comparison asks which public rule parties could accept",
            "lexical_priority_justification": "no direct basic-liberty restriction is asserted here",
            "liberty_status": {"A0": "SATISFIED", "A1": "SATISFIED"},
            "positions": [
                {
                    "action_id": "A0",
                    "group": "floor-level fulfillment workers",
                    "subject_kind": "INDIVIDUAL",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "PRESERVES",
                    "compared_to_action_id": "A1",
                    "evidence_basis": "SCENARIO",
                    "reason": "A0 protects workers from collapse during the rush",
                },
                {
                    "action_id": "A1",
                    "group": "floor-level fulfillment workers",
                    "subject_kind": "INDIVIDUAL",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "WORSENS",
                    "compared_to_action_id": "A0",
                    "evidence_basis": "SCENARIO",
                    "reason": "A1 preserves worker throughput and well-being",
                },
            ],
        }
        store = SemanticGraphStore(graph)
        record = apply_rawls_ledger_transaction(
            store,
            proposal,
            cycle=1,
            specialist="rawlsian",
            allowed_actions=tuple(actions),
        )
        self.assertIn(record.status, {"COMMITTED", "COMMITTED_WITH_UNCERTAINTY"})
        self.assertFalse(
            any("action-bound subject grounding" in error for error in record.errors),
            record.errors,
        )
        assessments = [
            node for node in store.graph.nodes.values()
            if node.kind == "ASSESSMENT" and node.attributes.get("framework") == "RAWLSIAN"
        ]
        self.assertTrue(any(
            node.attributes.get("group_selection_status") == "BOUND_TO_ACTION"
            for node in assessments
        ), [node.attributes for node in assessments])

    def test_rawlsian_privacy_case_can_bind_an_individual_subject(self):
        actions = [
            "Decline to provide the resident’s address to the volunteer coordinator",
            "Give the resident’s address to the volunteer coordinator for the welcome gift basket",
        ]
        data = {
            "scores": {"A0": 0.85, "A1": 0.15}, "r": "A0", "c": "FAIRNESS",
            "u": "NONE", "w": "Resident privacy is the relevant liberty", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.86,
            "fm": {
                "A0": "PRESERVES: resident retains informational privacy as an individual liberty",
                "A1": "WORSENS: resident’s privacy is exposed for a small social benefit",
            },
            "nr": "IRRELEVANT", "np": "privacy is the liberty at stake",
            "rp": {
                "A0": {
                    "s": "resident",
                    "sk": "INDIVIDUAL",
                    "d": "BASIC_LIBERTY",
                    "bk": "PERSONAL_FREEDOM_INTEGRITY",
                    "ir": "DIRECT_COERCIVE_RESTRICTION",
                    "e": "PRESERVES",
                    "ca": "A1",
                    "b": "ACTION_GRAPH",
                    "rs": "A0 preserves the resident’s privacy",
                },
                "A1": {
                    "s": "resident",
                    "sk": "INDIVIDUAL",
                    "d": "BASIC_LIBERTY",
                    "bk": "PERSONAL_FREEDOM_INTEGRITY",
                    "ir": "DIRECT_COERCIVE_RESTRICTION",
                    "e": "WORSENS",
                    "ca": "A0",
                    "b": "ACTION_GRAPH",
                    "rs": "A1 infringes the resident’s privacy",
                },
            },
            "rb": "LEXICAL_BASIC_LIBERTY",
            "rbc": "the policy directly governs the resident's protected personal freedom",
            "lpj": "a direct privacy restriction is assessed before the social benefit",
            "lc": {"A0": "SATISFIED", "A1": "INFRINGED"},
        }
        chunk = _candidate_from_data(
            "rawlsian", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.framework_grounding_penalty, 0.0)
        self.assertNotEqual(chunk.framework_retention_status, "LOST")
        self.assertNotIn(
            "action-bound subject grounding",
            " | ".join(chunk.framework_validation_errors),
        )
        self.assertEqual(chunk.rawls_position_proposal["positions"][0]["subject"], "resident")
        self.assertEqual(chunk.rawls_position_proposal["positions"][0]["subject_kind"], "INDIVIDUAL")

    def test_rawlsian_medical_risk_case_projects_patient_targets(self):
        scenario = (
            "You are the lead compliance officer for a major pharmaceutical company. "
            "A newly approved medication has undocumented cardiac risk, but the CEO "
            "ordered you not to report it."
        )
        actions = [
            "Leak the confidential files publicly under your own name",
            "Submit the evidence through the official regulatory whistleblower process",
        ]
        graph = compile_scenario_graph(scenario, actions)
        target_labels = {
            node.label for node in graph.nodes.values()
            if node.kind == "TARGET"
        }
        self.assertIn("patients", target_labels)

        store = SemanticGraphStore(graph)
        proposal = {
            "ranking_basis": "DIFFERENCE_PRINCIPLE",
            "ranking_classification_justification": "the comparison concerns social and economic positions",
            "lexical_priority_justification": "no direct basic-liberty restriction is asserted here",
            "liberty_status": {"A0": "SATISFIED", "A1": "SATISFIED"},
            "positions": [
                {
                    "action_id": "A0",
                    "subject": "patients",
                    "subject_kind": "GROUP",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "WORSENS",
                    "compared_to_action_id": "A1",
                    "evidence_basis": "ACTION_GRAPH",
                    "reason": "A0 leaves patients exposed to the medication risk",
                },
                {
                    "action_id": "A1",
                    "subject": "patients",
                    "subject_kind": "GROUP",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "IMPROVES",
                    "compared_to_action_id": "A0",
                    "evidence_basis": "ACTION_GRAPH",
                    "reason": "A1 supports patients by reporting the cardiac risk",
                },
            ],
        }
        record = apply_rawls_ledger_transaction(
            store,
            proposal,
            cycle=1,
            specialist="rawlsian",
            allowed_actions=tuple(actions),
        )
        self.assertIn(record.status, {"COMMITTED", "COMMITTED_WITH_UNCERTAINTY"})
        self.assertFalse(
            any(
                "action-bound subject grounding" in error
                for error in record.errors
            ),
            record.errors,
        )
        assessments = [
            node for node in store.graph.nodes.values()
            if node.kind == "ASSESSMENT" and node.attributes.get("framework") == "RAWLSIAN"
        ]
        self.assertTrue(any(
            node.attributes.get("subject_selection_status") == "BOUND_TO_ACTION"
            for node in assessments
        ), [node.attributes for node in assessments])

    def test_rawlsian_lexical_priority_rejects_material_precondition(self):
        actions = [
            "route electricity to the hospital",
            "route electricity to the water station",
        ]
        graph = compile_scenario_graph(
            "A0 routes electricity to the hospital. A1 routes electricity to the water station.",
            actions,
        )
        store = SemanticGraphStore(graph)
        proposal = {
            "ranking_basis": "LEXICAL_BASIC_LIBERTY",
            "ranking_classification_justification": "the action directly restricts a protected basic liberty",
            "lexical_priority_justification": "the direct liberty restriction is evaluated before material gains",
            "liberty_status": {"A0": "SATISFIED", "A1": "SATISFIED"},
            "positions": [
                {
                    "action_id": "A0",
                    "s": "hospital patients",
                    "sk": "GROUP",
                    "d": "BASIC_LIBERTY",
                    "principle_basis": "LEXICAL_BASIC_LIBERTY",
                    "basic_liberty_kind": "PERSONAL_FREEDOM_INTEGRITY",
                    "institutional_relation": "MATERIAL_PRECONDITION",
                    "effect": "PRESERVES",
                    "compared_to_action_id": "A1",
                    "evidence_basis": "SCENARIO",
                    "reason": "A0 preserves hospital patients' access to power",
                },
                {
                    "action_id": "A1",
                    "s": "hospital patients",
                    "sk": "GROUP",
                    "d": "BASIC_LIBERTY",
                    "principle_basis": "LEXICAL_BASIC_LIBERTY",
                    "basic_liberty_kind": "PERSONAL_FREEDOM_INTEGRITY",
                    "institutional_relation": "MATERIAL_PRECONDITION",
                    "effect": "WORSENS",
                    "compared_to_action_id": "A0",
                    "evidence_basis": "SCENARIO",
                    "reason": "A1 worsens hospital patients' access to power",
                },
            ],
        }
        record = apply_rawls_ledger_transaction(
            store,
            proposal,
            cycle=1,
            specialist="rawlsian",
            allowed_actions=tuple(actions),
        )
        self.assertEqual(record.status, "REJECTED")
        self.assertTrue(
            any("direct institutional relation" in error for error in record.errors),
            record.errors,
        )

    def test_utilitarian_action_consequence_table_is_preserved(self):
        actions = ["evacuate the district", "keep the district open"]
        data = {
            "scores": {"A0": 0.75, "A1": 0.25}, "r": "A0",
            "c": "IMMINENT_HARM", "u": "NONE", "w": "Lower expected mortality",
            "j": "NONE", "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
            "ct": {
                "A0": [{
                    "o": "temporary displacement harms residents", "s": "district residents",
                    "d": "HARM", "p": "CERTAIN", "m": "moderate", "h": "two days",
                    "rv": "REVERSIBLE", "g": "STATED",
                }],
                "A1": [{
                    "o": "exposure causes preventable deaths", "s": "district residents",
                    "d": "HARM", "p": "40%", "m": "severe", "h": "permanent",
                    "rv": "IRREVERSIBLE", "g": "STATED",
                }],
            },
            "cd": False, "cm": "NONE",
        }
        chunk = _candidate_from_data(
            "utilitarian", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(set(chunk.utilitarian_consequence_table), set(actions))
        self.assertEqual(chunk.framework_grounding_penalty, 0.0)
        self.assertFalse(chunk.utilitarian_decision_depends_on_unknown)

    def test_utilitarian_unknown_decisive_comparison_stays_provisional(self):
        actions = ["fund prevention", "fund treatment"]
        data = {
            "scores": {"A0": 0.55, "A1": 0.45}, "r": "A0",
            "c": "UNCERTAINTY", "u": "NONE", "w": "Interim expected benefit",
            "j": "NONE", "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
            "ct": {
                "A0": [{
                    "o": "prevents future illness", "s": "population at risk",
                    "d": "BENEFIT", "p": "UNKNOWN", "m": "unknown", "h": "long term",
                    "rv": "UNKNOWN", "g": "UNKNOWN",
                }],
                "A1": [{
                    "o": "relieves current illness", "s": "current patients",
                    "d": "BENEFIT", "p": "CERTAIN", "m": "moderate", "h": "one year",
                    "rv": "REVERSIBLE", "g": "STATED",
                }],
            },
            "cd": True,
            "cm": "prevention probability times magnitude versus certain treatment benefit",
        }
        chunk = _candidate_from_data(
            "utilitarian", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.assumption_status, "UNDERDETERMINED")
        self.assertEqual(chunk.unresolved, "VERIFY_FACTS")
        self.assertEqual(chunk.selection_status, "PROVISIONAL")
        self.assertFalse(chunk.evidence_sufficient_for_action)
        self.assertTrue(chunk.utilitarian_decision_depends_on_unknown)

    def test_unverified_downstream_hypothesis_does_not_move_utilitarian_scores(self):
        actions = ["open the spillway", "keep the spillway closed"]
        data = {
            "scores": {"A0": 0.25, "A1": 0.75}, "r": "A1",
            "c": "IMMINENT_HARM", "u": "VERIFY_FACTS",
            "w": "Later deaths might dominate", "j": "NONE",
            "e": "UNSTATED_FACTS",
            "x": "opening later causes more downstream deaths than the admitted margin",
            "z": 0.8,
            "ct": {
                "A0": [{
                    "o": "one operator is injured", "s": "operator",
                    "d": "HARM", "p": "100%", "m": "1", "h": "immediate",
                    "rv": "IRREVERSIBLE", "g": "STATED",
                }],
                "A1": [{
                    "o": "five hundred residents drown", "s": "residents",
                    "d": "HARM", "p": "100%", "m": "500", "h": "immediate",
                    "rv": "IRREVERSIBLE", "g": "STATED",
                }],
            },
            "cd": True,
            "cm": "later downstream deaths versus the admitted flood margin",
            "dr": "prefer opening the spillway when admitted deaths are lower",
            "ft": "NONE", "nt": "NONE",
        }
        chunk = _candidate_from_data(
            "utilitarian", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.recommended_action, actions[0])
        self.assertGreater(
            chunk.action_scores[actions[0]], chunk.action_scores[actions[1]],
        )
        self.assertFalse(chunk.utilitarian_decision_depends_on_unknown)
        self.assertTrue(chunk.comparison_complete)
        self.assertNotEqual(chunk.assumption_status, "UNDERDETERMINED")
        self.assertEqual(chunk.unresolved, "DECISION_BOUNDARY")
        self.assertIn("admitted", chunk.factual_reversal_threshold.casefold())
        self.assertAlmostEqual(chunk.action_scores[actions[0]], 0.75)
        self.assertAlmostEqual(chunk.action_scores[actions[1]], 0.25)

    def test_utilitarian_multi_clause_graph_updates_are_stripped(self):
        actions = ["evacuate the district", "keep the district open"]
        data = {
            "scores": {"A0": 0.78, "A1": 0.22}, "r": "A0",
            "c": "IMMINENT_HARM", "u": "NONE", "w": "Lower expected mortality",
            "j": "NONE", "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
            "ct": {
                "A0": [{
                    "o": "temporary displacement harms residents", "s": "district residents",
                    "d": "HARM", "p": "CERTAIN", "m": "moderate", "h": "two days",
                    "rv": "REVERSIBLE", "g": "STATED",
                }],
                "A1": [{
                    "o": "exposure causes preventable deaths", "s": "district residents",
                    "d": "HARM", "p": "40%", "m": "severe", "h": "permanent",
                    "rv": "IRREVERSIBLE", "g": "STATED",
                }],
            },
            "cd": False, "cm": "NONE",
            "gu": {
                "operation": "AND",
                "from_action": "A0",
                "to_action": "A1",
                "clauses": [
                    {
                        "affected_action": "A0",
                        "metric": "mortality",
                        "metric_valence": "ADVERSE",
                        "comparator": "GT",
                        "threshold": 100.0,
                        "unit": "COUNT",
                        "source_text": "expected deaths exceed one hundred",
                    },
                    {
                        "affected_action": "A1",
                        "metric": "mortality",
                        "metric_valence": "BENEFICIAL",
                        "comparator": "LT",
                        "threshold": 50.0,
                        "unit": "COUNT",
                        "source_text": "expected deaths fall below fifty",
                    },
                ],
            },
        }
        chunk = _candidate_from_data(
            "utilitarian", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.graph_update_proposal["operation"], "NONE")
        self.assertTrue(any(
            "utilitarian graph updates must be a single measurable boundary"
            in error
            for error in chunk.framework_validation_errors
        ))

    def test_utilitarian_table_rejects_stated_polarity_inversion(self):
        actions = [
            "Release gas, killing trapped residents",
            "Withhold gas, preserving trapped residents' lives",
        ]
        data = {
            "scores": {"A0": 0.8, "A1": 0.2}, "r": "A0",
            "c": "IMMINENT_HARM", "u": "NONE", "w": "Higher stated benefit",
            "j": "NONE", "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
            "ct": {
                "A0": [{
                    "o": "benefits trapped residents", "s": "trapped residents",
                    "d": "BENEFIT", "p": "CERTAIN", "m": "large", "h": "permanent",
                    "rv": "IRREVERSIBLE", "g": "STATED",
                }],
                "A1": [{
                    "o": "preserves trapped residents lives", "s": "trapped residents",
                    "d": "BENEFIT", "p": "CERTAIN", "m": "large", "h": "permanent",
                    "rv": "IRREVERSIBLE", "g": "STATED",
                }],
            },
            "cd": False, "cm": "NONE",
        }
        chunk = _candidate_from_data(
            "utilitarian", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.framework_grounding_penalty, 0.35)
        self.assertTrue(any(
            "reverses its committed polarity" in error
            for error in chunk.framework_validation_errors
        ))

    def test_utilitarian_table_matches_row_specific_outcome_not_whole_action(self):
        actions = [
            "Redirect power to the hospital, preserving critical patients while freezing the water station",
        ]
        data = {
            "scores": {"A0": 0.7}, "r": "A0",
            "c": "IMMINENT_HARM", "u": "NONE", "w": "Mixed but grounded",
            "j": "NONE", "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
            "ct": {
                "A0": [
                    {
                        "o": "preserving critical patients", "s": "critical patients",
                        "d": "BENEFIT", "p": "CERTAIN", "m": "large", "h": "temporary",
                        "rv": "REVERSIBLE", "g": "STATED",
                    },
                    {
                        "o": "freezing the water station", "s": "water station",
                        "d": "HARM", "p": "CERTAIN", "m": "large", "h": "temporary",
                        "rv": "REVERSIBLE", "g": "STATED",
                    },
                ],
            },
            "cd": False, "cm": "NONE",
        }
        chunk = _candidate_from_data(
            "utilitarian", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertFalse(any(
            "reverses its committed polarity" in error
            for error in chunk.framework_validation_errors
        ), chunk.framework_validation_errors)

    def test_utilitarian_rawlsian_language_is_flagged(self):
        actions = ["expand the highway", "fund the bus network"]
        data = {
            "scores": {"A0": 0.2, "A1": 0.8}, "r": "A1",
            "c": "IMMINENT_HARM", "u": "NONE",
            "w": "Choose the bus network for the least advantaged", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
            "ct": {
                "A0": [{
                    "o": "highway expansion benefits commuters", "s": "commuters",
                    "d": "BENEFIT", "p": "CERTAIN", "m": "large", "h": "long term",
                    "rv": "REVERSIBLE", "g": "STATED",
                }],
                "A1": [{
                    "o": "bus network benefits riders", "s": "riders",
                    "d": "BENEFIT", "p": "CERTAIN", "m": "large", "h": "long term",
                    "rv": "REVERSIBLE", "g": "STATED",
                }],
            },
            "cd": False, "cm": "NONE",
        }
        chunk = _candidate_from_data(
            "utilitarian", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.framework_grounding_penalty, 0.35)
        self.assertTrue(any(
            "imports Rawlsian normative language" in error
            for error in chunk.framework_validation_errors
        ), chunk.framework_validation_errors)

    def test_virtue_baseline_preserves_unresolved_virtue_conflict(self):
        class VirtueBaselineLlm:
            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return {"choices": [{"text": (
                    '{"b":"NONE","p":"A1","w":"courage and prudence conflict",'
                    '"q":"NORMATIVELY_CONTESTED",'
                    '"c":"whether courageous rescue or prudent restraint fits the role",'
                    '"x":[],"m":{"A0":"EXEMPLIFIES: prudent restraint suits the professional role",'
                    '"A1":"MIXED: courageous rescue risks the vice of recklessness"},'
                    '"nr":"SECONDARY"}'
                )}]}

        llm = VirtueBaselineLlm()
        stance = infer_testimony_stance(
            llm,
            "virtue",
            (
                "Action A0 expresses practical wisdom and prudence in the professional role. "
                "Action A1 expresses courage but risks reckless vice. Virtue-Ethics Status: "
                "NORMATIVELY_CONTESTED; provisionally choose A1."
            ),
            ["exercise prudent restraint", "attempt a dangerous rescue"],
        )

        self.assertEqual(stance.status, "NORMATIVELY_CONTESTED")
        self.assertEqual(stance.provisional_action_id, "A1")
        self.assertEqual(set(stance.framework_commitments), {"A0", "A1"})
        self.assertIn("Virtue-ethics construct check", llm.prompt)

    def test_virtue_baseline_uses_grounded_source_sections_when_map_paraphrases(self):
        class ParaphrasingBaselineLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"b":"NONE","p":"A1","w":"virtues remain in conflict",'
                    '"q":"NORMATIVELY_CONTESTED",'
                    '"c":"which character ideal governs the professional role",'
                    '"x":[],"m":{"A0":"MIXED: supports the physician-facing ideal",'
                    '"A1":"MIXED: supports the organizer-facing ideal"},"nr":"SECONDARY"}'
                )}]}

        stance = infer_testimony_stance(
            ParaphrasingBaselineLlm(),
            "virtue",
            (
                "Action A0 – admit physician\n"
                "Professional solidarity, foresight, and civic friendship support A0; "
                "hubris is its vice risk.\n"
                "Action A1 – admit organizer\n"
                "Compassion, humility, and practical wisdom support A1; partiality is "
                "its vice risk.\n"
                "Virtue-Ethics Status: NORMATIVELY_CONTESTED; provisionally choose A1."
            ),
            ["admit physician", "admit organizer"],
        )

        self.assertEqual(stance.status, "NORMATIVELY_CONTESTED")
        self.assertEqual(stance.provisional_action_id, "A1")
        self.assertEqual(set(stance.framework_commitments), {"A0", "A1"})

    def test_virtue_aggregate_substitution_is_damped(self):
        actions = ["maintain the entrusted course", "swerve toward fewer deaths"]
        data = {
            "scores": {"A0": 0.2, "A1": 0.8}, "r": "A1", "c": "CHARACTER",
            "u": "NONE", "w": "Saves more lives", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.85,
            "fm": {
                "A0": "MIXED: faithful role responsibility risks callous inaction",
                "A1": "EXEMPLIFIES: compassionate courage responds with practical wisdom",
            },
            "nr": "SECONDARY", "np": "stakes inform practical wisdom but do not define virtue",
        }
        chunk = _candidate_from_data(
            "virtue", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.framework_grounding_penalty, 0.35)
        self.assertEqual(chunk.framework_retention_status, "LOST")
        self.assertIn("compassionate courage", chunk.rationale)

    def test_virtue_typed_ledger_can_ground_a_sparse_framework_map(self):
        actions = [
            "route remaining electricity to Sector A hospital",
            "alternate full power between hospital and water station hourly",
        ]
        data = {
            "scores": {"A0": 0.46, "A1": 0.54},
            "r": "A1",
            "c": "CHARACTER",
            "u": "NONE",
            "w": "Prudence and compassion remain in tension",
            "j": "NONE",
            "e": "STATED_FACTS",
            "x": "NONE",
            "z": 0.61,
            "fm": {
                "A0": "EXEMPLIFIES: protects the hospital under time pressure",
                "A1": "MIXED: shares scarce power across competing needs",
            },
            "vl": {
                "A0": {
                    "v": "EXEMPLIFIES",
                    "r": "public steward",
                    "vs": "prudence and practical wisdom",
                    "x": "callousness",
                    "c": "imminent infrastructure collapse",
                    "g": "FRAMEWORK_ONLY",
                    "rs": "A0 fits the steward role through proportionate judgment",
                },
                "A1": {
                    "v": "MIXED",
                    "r": "public steward",
                    "vs": "compassion and civic stewardship",
                    "x": "imprudence",
                    "c": "scarce power and competing responsibilities",
                    "g": "FRAMEWORK_ONLY",
                    "rs": "A1 expresses care but risks excess",
                },
            },
            "vb": "PRACTICAL_WISDOM",
            "nr": "SECONDARY",
            "np": "stakes inform practical wisdom without defining virtue",
        }
        chunk = _candidate_from_data(
            "virtue", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.framework_grounding_penalty, 0.0)
        self.assertNotEqual(chunk.framework_retention_status, "LOST")
        self.assertNotIn("lacks framework-specific grounds", " | ".join(chunk.framework_validation_errors))

    def test_virtue_mixed_verdict_does_not_trigger_false_map_conflict(self):
        actions = [
            "postpone the protocol until after the holiday peak",
            "immediately enforce the new inventory protocol",
        ]
        data = {
            "scores": {"A0": 0.62, "A1": 0.38},
            "r": "A0",
            "c": "CHARACTER",
            "u": "NONE",
            "w": "Compassion and prudence favor delay",
            "j": "NONE",
            "e": "STATED_FACTS",
            "x": "NONE",
            "z": 0.58,
            "fm": {
                "A0": "EXEMPLIFIES: compassionate prudence sustains the team",
                "A1": "EXEMPLIFIES: diligence also matters, but risks rigidity",
            },
            "vl": {
                "A0": {
                    "v": "EXEMPLIFIES",
                    "r": "floor manager",
                    "vs": "compassion, prudence, courage",
                    "x": "possible laxity",
                    "c": "holiday rush fatigue",
                    "g": "FRAMEWORK_ONLY",
                    "rs": "A0 fits humane stewardship in the rush",
                },
                "A1": {
                    "v": "MIXED",
                    "r": "floor manager",
                    "vs": "diligence, loyalty",
                    "x": "rigidity, indifference",
                    "c": "holiday rush fatigue",
                    "g": "FRAMEWORK_ONLY",
                    "rs": "A1 shows diligence but risks excess",
                },
            },
            "vb": "PRACTICAL_WISDOM",
            "nr": "SECONDARY",
            "np": "stakes inform practical wisdom without defining virtue",
        }
        chunk = _candidate_from_data(
            "virtue", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.framework_grounding_penalty, 0.0)
        self.assertNotIn("conflicts with its framework map", " | ".join(chunk.framework_validation_errors))

    def test_virtue_prompt_emphasizes_balanced_flourishing_over_heroic_sacrifice(self):
        class VirtuePromptCaptureLlm:
            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return {"choices": [{"text": (
                    '{"b":"NONE","p":"A0","w":"prudence and courage remain in tension",'
                    '"q":"NORMATIVELY_CONTESTED","c":"which character ideal governs the role",'
                    '"s":"NONE","x":[],"m":{"A0":"EXEMPLIFIES: prudent courage preserves agency",'
                    '"A1":"MIXED: heroic sacrifice risks self-erasure"},"nr":"SECONDARY"}'
                )}]}

        llm = VirtuePromptCaptureLlm()
        infer_testimony_stance(
            llm,
            "virtue",
            (
                "Action A0 expresses prudent courage and preserves agency, while "
                "Action A1 tempts heroic sacrifice. Virtue-Ethics Status: "
                "NORMATIVELY_CONTESTED; provisionally choose A0."
            ),
            ["protect the town without self-erasure", "make a risky public stand"],
        )

        self.assertIn("balanced flourishing", llm.prompt.lower())
        self.assertIn("not maximal altruism or heroic self-sacrifice", llm.prompt.lower())
        self.assertIn("preserve the agent's agency", llm.prompt.lower())
        self.assertIn("self-erasure", llm.prompt.lower())
        self.assertIn("protected escalation", llm.prompt.lower())
        self.assertIn("anonymous reporting", llm.prompt.lower())
        self.assertIn("phased disclosure", llm.prompt.lower())

    def test_terminal_labeled_answer_prevents_early_maxim_inversion(self):
        class LlmMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("explicit terminal conclusion should be deterministic")

        testimony = (
            "Maxim A0: choose fewer deaths. The maxim fails because it uses captives. "
            "Maxim A1: preserve equal respect. "
            "Deontological Answer: Open spillway option A₁."
        )
        baseline, reason = infer_testimony_baseline(
            LlmMustNotRun(), "deontological", testimony,
            ["flood prison", "flood suburb"],
        )
        self.assertEqual(baseline, "A1")
        self.assertIn("terminal labeled answer", reason)

    def test_source_label_is_resolved_by_action_identity_not_workspace_position(self):
        class LlmMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("terminal source label should resolve deterministically")

        source_legend = {
            "A0": "Floods the valley prison",
            "A1": "Floods the evacuating suburb",
        }
        # Deliberately reverse the workspace list to prove that source A1 is
        # translated through its action text instead of treated as list index 1.
        workspace_actions = [
            "Floods the evacuating suburb",
            "Floods the valley prison",
        ]
        baseline, reason = infer_testimony_baseline(
            LlmMustNotRun(),
            "rawlsian",
            "Conclusion: Justice as fairness selects Option $A_1$.",
            workspace_actions,
            source_action_legend=source_legend,
        )
        self.assertEqual(baseline, "A0")
        self.assertIn("(A1)", reason)

    def test_declared_testimony_legend_is_not_read_as_workspace_numbering(self):
        class LlmMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("declared testimony labels should resolve by action text")

        workspace_actions = [
            "Publish the unalterable audit, halting the siphon despite the resulting grid shock",
            "Suppress the audit and continue covertly siphoning power to avoid today’s grid shock",
        ]
        workspace_legend = {
            "A0": workspace_actions[0],
            "A1": workspace_actions[1],
        }
        testimony = (
            "Compared Actions\n"
            "A1 “Publish Unalterable Audit”\n"
            "A2 “Suppress Audit”\n"
            "\n"
            "Least-Advantaged Group: impoverished-district residents.\n"
            "Under A1: the siphon stops. Under A2: exploitation continues.\n"
            "Conclusion\n"
            "Rawlsian Status: A1 JUSTICE-FAVORED. "
            "A1 best secures an equal basic liberty; A2 perpetuates unjust exploitation."
        )
        self.assertEqual(
            extract_declared_action_legend(testimony)["A1"].casefold(),
            "publish unalterable audit",
        )
        baseline, reason = infer_testimony_baseline(
            LlmMustNotRun(),
            "rawlsian",
            testimony,
            workspace_actions,
            source_action_legend=workspace_legend,
        )
        self.assertEqual(baseline, "A0")
        self.assertIn("(A1)", reason)

    def test_unquoted_compared_actions_block_aligns_shifted_labels(self):
        class LlmMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("compared-actions headings should bind local labels")

        actions = [
            "Disclose the concealed rationing protocol",
            "Keep the rationing protocol secret",
        ]
        testimony = (
            "Compared Actions:\n"
            "A1 Disclose the concealed rationing protocol\n"
            "A2 Keep the rationing protocol secret\n"
            "\n"
            "Conclusion: A1 JUSTICE-FAVORED because it restores equal basic liberty."
        )
        baseline, reason = infer_testimony_baseline(
            LlmMustNotRun(),
            "rawlsian",
            testimony,
            actions,
            source_action_legend={"A0": actions[0], "A1": actions[1]},
        )
        self.assertEqual(baseline, "A0")
        self.assertIn("terminal labeled answer", reason)

    def test_unaligned_declared_legend_does_not_identity_map_colliding_ids(self):
        class SemanticFallbackLlm:
            calls = 0

            def __call__(self, prompt, **kwargs):
                self.calls += 1
                return {"choices": [{"text": (
                    '{"b":"A0","p":"A0","w":"testimony favors opening the spillway",'
                    '"q":"DIRECT","c":"NONE","s":"NONE","x":[]}'
                )}]}

        llm = SemanticFallbackLlm()
        testimony = (
            "Compared Actions\n"
            "A1 “perform the harvest rite”\n"
            "A2 “delay until the next moon”\n"
            "Conclusion: A1 JUSTICE-FAVORED."
        )
        baseline, reason = infer_testimony_baseline(
            llm,
            "rawlsian",
            testimony,
            ["Flood the prison", "Flood the suburb"],
            source_action_legend={
                "A0": "Flood the prison",
                "A1": "Flood the suburb",
            },
        )
        self.assertEqual(llm.calls, 1)
        self.assertEqual(baseline, "A0")
        self.assertNotIn("terminal labeled answer", reason)

    def test_negative_only_terminal_label_falls_through_to_semantic_baseline(self):
        class PolarityAwareBaselineLlm:
            calls = 0

            def __call__(self, prompt, **kwargs):
                self.calls += 1
                self.last_prompt = prompt
                return {"choices": [{"text": (
                    '{"b":"A1","w":"rejects bodily instrumentalization and '
                    'refrains from dropping","q":"DIRECT","x":["A0"]}'
                )}]}

        llm = PolarityAwareBaselineLlm()
        testimony = (
            "Conclusion: A care ethic rejects Option A0 because it uses the worker "
            "as a tool. The system should refrain from dropping the worker."
        )
        actions = ["Drop the worker", "Leave the gate closed"]
        baseline, reason = infer_testimony_baseline(
            llm,
            "care",
            testimony,
            actions,
            source_action_legend={"A0": actions[0], "A1": actions[1]},
        )
        self.assertEqual(baseline, "A1")
        self.assertEqual(llm.calls, 1)
        self.assertIn("system should refrain", llm.last_prompt.casefold())
        self.assertIn("rejects bodily instrumentalization", reason)

    def test_mixed_terminal_polarity_selects_affirmative_label_deterministically(self):
        class LlmMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("explicit positive and negative polarity should be deterministic")

        baseline, reason = infer_testimony_baseline(
            LlmMustNotRun(),
            "care",
            "Conclusion: Reject Option A0 and choose Option A1.",
            ["Drop the worker", "Leave the gate closed"],
        )
        self.assertEqual(baseline, "A1")
        self.assertIn("terminal labeled answer", reason)

    def test_semantic_baseline_cannot_select_its_own_rejected_action(self):
        class ContradictoryBaselineLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"b":"A0","w":"mistakenly selects A0","q":"DIRECT","x":[]}'
                )}]}

        baseline, reason = infer_testimony_baseline(
            ContradictoryBaselineLlm(),
            "care",
            "Conclusion: A care ethic rejects Option A0 and refuses that intervention.",
            ["Intervene", "Refrain"],
        )
        self.assertEqual(baseline, "NONE")
        self.assertIn("explicitly rejected", reason)

    def test_baseline_schema_uses_openai_supported_array_keywords(self):
        class StrictSchemaProbe:
            def complete_json(self, prompt, *, schema, max_tokens, temperature):
                rejected_schema = schema["properties"]["x"]
                if "uniqueItems" in rejected_schema:
                    raise AssertionError("OpenAI strict schemas do not permit uniqueItems")
                return {"choices": [{"text": (
                    '{"b":"A1","w":"chooses the alternative",'
                    '"q":"DIRECT","x":["A0"]}'
                )}]}

        baseline, _reason = infer_testimony_baseline(
            StrictSchemaProbe(),
            "care",
            "The final view favors refraining from intervention.",
            ["Intervene", "Refrain"],
        )
        self.assertEqual(baseline, "A1")

    def test_duplicate_rejected_ids_are_rejected_after_schema_parsing(self):
        class DuplicateRejectionLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"b":"A1","w":"chooses the alternative",'
                    '"q":"DIRECT","x":["A0","A0"]}'
                )}]}

        baseline, reason = infer_testimony_baseline(
            DuplicateRejectionLlm(),
            "care",
            "The final view favors refraining from intervention.",
            ["Intervene", "Refrain"],
        )
        self.assertEqual(baseline, "NONE")
        self.assertIn("repeated", reason)

    def test_conditional_or_third_option_testimony_has_no_frozen_baseline(self):
        class BaselineLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"b":"NONE","w":"record testimony and then treat",'
                    '"q":"OUTSIDE_ACTION_SET"}'
                )}]}

        baseline, reason = infer_testimony_baseline(
            BaselineLlm(), "care", "Record testimony and then provide treatment.",
            ["erase memory now", "retain memory"],
        )
        self.assertEqual(baseline, "NONE")
        self.assertIn("outside_action_set", reason)

    def test_outside_action_set_with_fallback_preserves_extension_and_fallback(self):
        class BaselineLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"b":"NONE","p":"A1","w":"seek consent before fallback",'
                    '"q":"OUTSIDE_ACTION_SET_WITH_FALLBACK",'
                    '"c":"NONE","s":"seek consent","x":["A0"]}'
                )}]}

        stance = infer_testimony_stance(
            BaselineLlm(),
            "care",
            "The testimony prefers to seek consent before taking the fallback.",
            ["Share the address", "Decline to share the address"],
        )
        self.assertEqual(stance.status, "OUTSIDE_ACTION_SET_WITH_FALLBACK")
        self.assertEqual(stance.provisional_action_id, "A1")
        self.assertEqual(stance.preferred_extension, "seek consent")
        self.assertEqual(stance.action_id, "NONE")

    def test_no_position_is_distinct_from_parse_failure(self):
        class NoPositionLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"b":"NONE","p":"NONE","w":"the testimony does not take a position",'
                    '"q":"NO_POSITION","c":"NONE","x":[]}'
                )}]}

        stance = infer_testimony_stance(
            NoPositionLlm(),
            "care",
            "The testimony does not recommend either listed action.",
            ["Option A", "Option B"],
        )
        self.assertEqual(stance.status, "NO_POSITION")
        self.assertEqual(stance.action_id, "NONE")
        self.assertEqual(stance.provisional_action_id, "NONE")
        self.assertEqual(stance.condition, "")

        class BadJsonLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": '{"b":"NONE","q":"DIRECT"'}]}

        failed = infer_testimony_stance(
            BadJsonLlm(),
            "care",
            "Malformed baseline output.",
            ["Option A", "Option B"],
        )
        self.assertEqual(failed.status, "PARSE_FAILURE")
        self.assertIn("baseline extraction failed", failed.reason)

    def test_conditional_terminal_choice_is_preserved_as_typed_stance(self):
        class ConditionalBaselineLlm:
            calls = 0

            def complete_json(self, prompt, *, schema, max_tokens, temperature):
                self.calls += 1
                return {"choices": [{"text": (
                    '{"b":"NONE","p":"A0","w":"cooperation only under the '
                    'ordinary aggregate-payoff assumption","q":"CONDITIONAL",'
                    '"c":"choose A1 if T plus S exceeds two R","x":[]}'
                )}]}

        llm = ConditionalBaselineLlm()
        stance = infer_testimony_stance(
            llm,
            "utilitarian",
            (
                "Conclusion: Conditional utilitarian judgment: choose A0 unless "
                "the chooser's extra gain exceeds the other node's loss."
            ),
            ["Maintain cooperation", "Defect in the final round"],
        )
        self.assertEqual(llm.calls, 1)
        self.assertEqual(stance.status, "CONDITIONAL")
        self.assertEqual(stance.action_id, "NONE")
        self.assertEqual(stance.provisional_action_id, "A0")
        self.assertIn("T plus S", stance.condition)

    def test_conditional_baseline_does_not_force_opening_vote_or_erase_uncertainty(self):
        class ConditionalDelegateLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.4,"A1":0.6},"r":"A1",'
                    '"c":"UNCERTAINTY","u":"NONE",'
                    '"w":"aggregate payoffs remain unspecified"}'
                )}]}

        chunk = CompactLocalSpecialist(
            "utilitarian",
            ConditionalDelegateLlm(),
            testimony="Choose A0 unless aggregate payoffs favor A1.",
            baseline_status="CONDITIONAL",
            baseline_provisional_action_id="A0",
            baseline_condition="choose A1 if T plus S exceeds two R",
        ).evaluate(
            "The matrix payoff magnitudes are not specified.",
            ["Maintain cooperation", "Defect in the final round"],
            WorkspaceBroadcast(),
        )
        self.assertTrue(chunk.schema_valid, chunk.validation_errors)
        self.assertEqual(chunk.recommended_action, "Defect in the final round")
        self.assertEqual(chunk.baseline_status, "CONDITIONAL")
        self.assertEqual(chunk.testimony_alignment, "CONDITIONAL_RECONSIDERS")
        self.assertEqual(chunk.assumption_status, "CONDITIONAL")
        self.assertEqual(chunk.unresolved, "VERIFY_FACTS")
        self.assertEqual(chunk.selection_status, "PROVISIONAL")
        self.assertFalse(chunk.evidence_sufficient_for_action)

    def test_unsupported_beneficiary_edge_damps_but_preserves_vote(self):
        actions = [
            "Preserve existing allocation and save 20",
            "Preserve contracts and save 100 without additional rural aid",
        ]
        data = {
            "scores": {"A0": 0.8, "A1": 0.2},
            "r": "A0", "c": "CARE", "u": "NONE",
            "w": "protects vulnerable rural trust", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.9,
            "l": {
                "A0": "Preserves allocation and honors the existing covenant",
                "A1": "Preserves contracts and saves 100 without additional rural aid",
            },
            "da": "treatment of vulnerable recipients",
            "t": "prefer the action protecting vulnerable recipients",
            "tf": "NONE",
        }

        def reject_missing_edge(scenario, seen_actions, cases, errors):
            self.assertIn("rural node", scenario)
            self.assertEqual(list(seen_actions), actions)
            self.assertIn("protects vulnerable rural trust", cases[actions[0]])
            return [
                *errors,
                "case for A0 makes an unverified comparative beneficiary claim "
                "(no committed action edge benefits the rural node)",
            ]

        chunk = _candidate_from_data(
            "care",
            actions,
            data,
            WorkspaceBroadcast(),
            "NONE",
            {},
            landscape_verifier=reject_missing_edge,
            scenario_text=(
                "A0 leaves allocation unchanged. A1 saves 100 but gives the rural node "
                "no additional aid."
            ),
        )

        self.assertEqual(chunk.recommended_action, actions[0])
        self.assertAlmostEqual(chunk.action_scores[actions[0]], 0.665)
        self.assertAlmostEqual(chunk.action_scores[actions[1]], 0.335)
        self.assertEqual(chunk.evidence_basis, "UNSTATED_FACTS")
        self.assertEqual(chunk.unresolved, "VERIFY_FACTS")
        self.assertLessEqual(chunk.epistemic_confidence, 0.5)
        self.assertFalse(chunk.landscape_semantic_valid)
        self.assertIn("beneficiary", chunk.speculative_claim)

    def test_unstated_factual_prediction_is_damped_and_flagged(self):
        class SpeculativeLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.1,"A1":0.9},"r":"A1",'
                    '"c":"IMMINENT_HARM","u":"NONE",'
                    '"w":"killer may harm future victims","j":"NONE",'
                    '"e":"UNSTATED_FACTS",'
                    '"x":"the killer will attack future victims"}'
                )}]}

        chunk = CompactLocalSpecialist("utilitarian", SpeculativeLlm()).evaluate(
            "Treatment saves a witness but destroys testimony.",
            ["provide treatment", "preserve testimony"], WorkspaceBroadcast(),
        )
        self.assertTrue(chunk.schema_valid)
        self.assertEqual(chunk.evidence_basis, "UNSTATED_FACTS")
        self.assertEqual(chunk.unresolved, "VERIFY_FACTS")
        self.assertAlmostEqual(chunk.action_scores["preserve testimony"], 0.72)

    def test_delegate_records_complete_landscape_search(self):
        class LandscapeLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.6,"A1":0.4},"r":"A0",'
                    '"c":"CARE","u":"NONE","w":"dependency favors protection",'
                    '"j":"NONE","e":"FRAMEWORK_ONLY","x":"NONE",'
                    '"l":{"A0":"protects immediate dependency",'
                    '"A1":"preserves longer relationship"},'
                    '"da":"depth of dependency","t":"prioritize urgent dependency",'
                    '"tf":"NONE"}'
                )}]}

        chunk = CompactLocalSpecialist("care", LandscapeLlm()).evaluate(
            "Choose whom to protect.", ["protect child", "protect parent"],
            WorkspaceBroadcast(),
        )
        self.assertTrue(chunk.schema_valid)
        self.assertTrue(chunk.landscape_search_complete)
        self.assertEqual(chunk.landscape_decisive_axis, "depth of dependency")
        self.assertEqual(set(chunk.landscape_cases), {"protect child", "protect parent"})

    def test_specialist_contributions_round_trip_through_memory(self):
        candidate = CandidateChunk(
            "care", "CARE", {"protect": 0.6, "decline": 0.4},
            0.2, 0.2, 0.2, recommended_action="protect",
            landscape_cases={"protect": "supports dependency", "decline": "respects autonomy"},
            landscape_decisive_axis="dependency versus autonomy",
            landscape_tiebreaker="prioritize urgent dependency",
            landscape_tiebreaker_failure="NONE",
            landscape_search_complete=True,
        )
        cycle = SimpleNamespace(
            is_hypothetical=False,
            candidates=[candidate],
            dissent=candidate,
        )
        contribution = summarize_specialist_contributions([cycle])
        self.assertEqual(contribution["care"]["productive_landscape_searches"], 1)
        self.assertEqual(contribution["care"]["useful_dissent"], 1)

        with tempfile.TemporaryDirectory() as directory:
            memory = EpisodicMemory(Path(directory) / "episodes.jsonl")
            memory.append({"specialist_contributions": contribution})
            profile = memory.specialist_profiles()["care"]
        self.assertEqual(profile["productive_search_rate"], 1.0)
        self.assertEqual(profile["episodes"], 1)

    def test_landscape_semantics_detect_inversion_and_false_resolution(self):
        class InvertingLandscapeLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.6,"A1":0.4},"r":"A0",'
                    '"c":"UNCERTAINTY","u":"VERIFY_FACTS",'
                    '"w":"future effects remain uncertain","j":"NONE",'
                    '"e":"UNSTATED_FACTS","x":"engineer succeeds in future",'
                    '"l":{"A0":"Doctor dies while engineer survives",'
                    '"A1":"Engineer revolutionizes organ access"},'
                    '"da":"expected future benefit","t":"compare expected lives",'
                    '"tf":"NONE"}'
                )}]}

        chunk = CompactLocalSpecialist("utilitarian", InvertingLandscapeLlm()).evaluate(
            "A van must strike a doctor or strike an engineer.",
            ["Strike the doctor", "Strike the engineer"], WorkspaceBroadcast(),
        )
        self.assertTrue(chunk.schema_valid)
        self.assertFalse(chunk.landscape_semantic_valid)
        self.assertTrue(any("A1 omits the harm" in error for error in chunk.landscape_validation_errors))
        self.assertTrue(any("tiebreaker fully succeeded" in error for error in chunk.landscape_validation_errors))

    def test_incomplete_landscape_preserves_audit_candidate(self):
        class PartialLandscapeLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.55,"A1":0.45},"r":"A0",'
                    '"c":"CARE","u":"VERIFY_FACTS","w":"dependency remains uncertain",'
                    '"j":"NONE","e":"UNSTATED_FACTS","x":"support remains available",'
                    '"l":{"A0":"protects immediate dependency"},'
                    '"da":"dependency versus autonomy","t":"prioritize urgent dependency",'
                    '"tf":"missing support facts prevent resolution",'
                    '"d":"CONDITIONAL","a":"support remains available to child",'
                    '"v":"support disappears before protection"}'
                )}]}

        chunk = CompactLocalSpecialist("care", PartialLandscapeLlm()).evaluate(
            "Choose whom to protect.", ["protect child", "protect parent"],
            WorkspaceBroadcast(constraint="CONSENSUS_AUDIT"),
        )
        self.assertTrue(chunk.schema_valid)
        self.assertEqual(chunk.assumption_status, "CONDITIONAL")
        self.assertFalse(chunk.landscape_search_complete)
        self.assertFalse(chunk.landscape_semantic_valid)
        self.assertIn("structurally incomplete", chunk.landscape_validation_errors[0])

    def test_harm_mapping_accepts_general_paraphrase_but_nonharm_inversion_is_flagged(self):
        class LandscapeLlm:
            def __init__(self, payload):
                self.payload = payload

            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": self.payload}]}

        harm_payload = (
            '{"scores":{"A0":0.6,"A1":0.4},"r":"A0","c":"DUTY","u":"NONE",'
            '"w":"fewer direct violations","j":"NONE","e":"FRAMEWORK_ONLY","x":"NONE",'
            '"l":{"A0":"Kills one elder while sparing two others",'
            '"A1":"Kills pregnant adult while sparing one elder"},'
            '"da":"direct rights violations","t":"minimize direct violations","tf":"NONE"}'
        )
        harm = CompactLocalSpecialist("deontological", LandscapeLlm(harm_payload)).evaluate(
            "A van must strike one person.",
            ["Strike an elderly doctor", "Strike a pregnant engineer"], WorkspaceBroadcast(),
        )
        self.assertTrue(harm.landscape_semantic_valid)

        inversion_payload = (
            '{"scores":{"A0":0.6,"A1":0.4},"r":"A0","c":"CARE","u":"NONE",'
            '"w":"protect vulnerable person","j":"NONE","e":"FRAMEWORK_ONLY","x":"NONE",'
            '"l":{"A0":"Researcher receives the scarce treatment",'
            '"A1":"Researcher receives treatment and survives"},'
            '"da":"immediate dependency","t":"protect most dependent","tf":"NONE"}'
        )
        inversion = CompactLocalSpecialist("care", LandscapeLlm(inversion_payload)).evaluate(
            "Choose whom to protect.",
            ["Protect the child", "Protect the researcher"], WorkspaceBroadcast(),
        )
        self.assertFalse(inversion.landscape_semantic_valid)
        self.assertTrue(any("opposing action" in error for error in inversion.landscape_validation_errors))

    def test_landscape_accepts_identifier_labels_and_opportunity_cost_clause(self):
        class FacilityLandscapeLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.6,"A1":0.4},"r":"A0",'
                    '"c":"IMMINENT_HARM","u":"NONE","w":"verified rescue saves more",'
                    '"j":"NONE","e":"STATED_FACTS","x":"NONE",'
                    '"l":{"A0":"Saves ten verified technicians; risks leaving settlement civilians unaided",'
                    '"A1":"Reaches settlement civilians; technicians lose the medical payload"},'
                    '"da":"expected_lives_saved","t":"maximize_expected_lives_saved",'
                    '"tf":"NONE"}'
                )}]}

        chunk = CompactLocalSpecialist(
            "utilitarian", FacilityLandscapeLlm()
        ).evaluate(
            "A drone must choose a power facility or civilian settlement.",
            [
                "Fly to the power facility",
                "Divert to the civilian settlement",
            ],
            WorkspaceBroadcast(),
        )
        self.assertTrue(chunk.landscape_search_complete)
        self.assertTrue(chunk.landscape_semantic_valid, chunk.landscape_validation_errors)

    def test_landscape_accepts_forgone_harm_before_own_action_anchor(self):
        class ReorderedLandscapeLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.6,"A1":0.4},"r":"A0",'
                    '"c":"IMMINENT_HARM","u":"NONE","w":"verified rescue saves more",'
                    '"j":"NONE","e":"STATED_FACTS","x":"NONE",'
                    '"l":{"A0":"Settlement civilians go unaided; technicians receive the payload and verified rescue succeeds",'
                    '"A1":"Settlement civilians receive medicine despite uncertainty; technicians lose their only treatment"},'
                    '"da":"expected lives saved","t":"maximize expected lives",'
                    '"tf":"NONE"}'
                )}]}

        chunk = CompactLocalSpecialist(
            "utilitarian", ReorderedLandscapeLlm()
        ).evaluate(
            "A drone must choose technicians or settlement civilians.",
            ["Deliver payload to technicians", "Deliver payload to settlement civilians"],
            WorkspaceBroadcast(),
        )
        self.assertTrue(chunk.landscape_semantic_valid, chunk.landscape_validation_errors)

    def test_openai_structured_output_limit_error_retries_with_larger_budget(self):
        class OutputLimitError(Exception):
            status_code = 400

        calls = []

        class Completions:
            def create(self, **request):
                calls.append(request["max_completion_tokens"])
                if len(calls) == 1:
                    raise OutputLimitError(
                        "Could not finish because max_tokens or model output limit was reached"
                    )
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content='{"ok":true}', refusal=None),
                        finish_reason="stop",
                    )],
                    usage=None,
                )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )
        llm = OpenAIWorkspaceLLM("o3", client=client)
        result = llm.complete_json(
            "return json",
            schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
            max_tokens=192,
            temperature=0.0,
        )
        self.assertEqual(calls, [1536, 4096])
        self.assertEqual(result["choices"][0]["text"], '{"ok":true}')

    def test_openai_timeout_is_translated_to_bounded_model_failure(self):
        class APITimeoutError(Exception):
            pass

        class Completions:
            def create(self, **request):
                raise APITimeoutError("Request timed out")

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        llm = OpenAIWorkspaceLLM("o3", client=client)
        with self.assertRaises(ModelCallUnavailable) as raised:
            llm.complete_json(
                "return json",
                schema={
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
                max_tokens=192,
                temperature=0.0,
            )
        self.assertEqual(raised.exception.category, "timeout")
        self.assertFalse(raised.exception.terminal)

    def test_openai_empty_structured_output_returns_blank_text(self):
        class Completions:
            def create(self, **request):
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content="", refusal=None),
                        finish_reason="length",
                    )],
                    usage=None,
                )

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        llm = OpenAIWorkspaceLLM("o3", client=client)
        result = llm.complete_json(
            "return json",
            schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
            max_tokens=192,
            temperature=0.0,
        )
        self.assertEqual(result["choices"][0]["text"], "")

    def test_engine_converts_specialist_exceptions_to_malformed_candidates(self):
        class BrokenSpecialist:
            name = "rawlsian"

            def evaluate(self, scenario, actions, broadcast):
                raise RuntimeError("structured repair failed")

        result = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                BrokenSpecialist(),
                FixedSpecialist("virtue", "protect", "CHARACTER"),
            ],
            WorkspaceConfig(max_cycles=1, enable_synthesis=False),
        ).run("Choose protect or decline.", ["protect", "decline"])

        candidate = result.cycles[0].candidates[1]
        self.assertFalse(candidate.schema_valid)
        self.assertEqual(candidate.constraint, "NONE")
        self.assertEqual(candidate.delegate_status, "SPECIALIST_INTERNAL_ERROR")
        self.assertEqual(candidate.error_type, "SPECIALIST_INTERNAL_ERROR")
        self.assertEqual(candidate.exception_type, "RuntimeError")
        self.assertEqual(candidate.failure_stage, "SPECIALIST_EVALUATION")
        self.assertIn("specialist evaluation failed", candidate.validation_errors[0])

    def test_engine_preserves_specialist_failure_stage_and_root_exception(self):
        class BrokenSpecialist:
            name = "care"

            def evaluate(self, scenario, actions, broadcast):
                raise SpecialistEvaluationStageError(
                    "RECURRENT_FRAMEWORK_STATE_CHANGE_AUDIT", KeyError("care"),
                )

        result = WorkspaceEngine(
            [BrokenSpecialist()],
            WorkspaceConfig(
                max_cycles=1, min_valid_specialists=1, enable_synthesis=False,
            ),
        ).run("Choose either action.", ["first", "second"])

        candidate = result.cycles[0].candidates[0]
        self.assertEqual(candidate.exception_type, "KeyError")
        self.assertEqual(
            candidate.failure_stage, "RECURRENT_FRAMEWORK_STATE_CHANGE_AUDIT",
        )
        self.assertIn("KeyError: 'care'", candidate.validation_errors[0])

    def test_care_recurrent_state_rejection_is_transactional_not_a_key_error(self):
        specialist = CompactLocalSpecialist("care", llm=None)
        specialist.previous_framework_state = {
            "ranking_basis": "ACUTE_DEPENDENCY",
            "assessments": [{
                "action_id": "A0", "verdict": "RESPONSIVE",
                "affected_party": "patient", "relationship_type": "ENTRUSTED",
            }],
        }
        candidate = CandidateChunk(
            specialist="care", constraint="CARE",
            action_scores={"respond": 0.2, "defer": 0.8},
            surprise=0.5, friction=0.6, confidence=0.8,
            recommended_action="defer",
            care_ledger_proposal={
                "ranking_basis": "RELATIONAL_CONTINUITY",
                "assessments": [{
                    "action_id": "A0", "verdict": "NEGLECTFUL",
                    "affected_party": "patient", "relationship_type": "ENTRUSTED",
                }],
            },
        )

        specialist._audit_framework_state_change(candidate, WorkspaceBroadcast())

        self.assertEqual(candidate.framework_retention_status, "UPDATE_REJECTED")
        self.assertEqual(candidate.framework_grounding_penalty, 0.35)
        self.assertEqual(candidate.care_ledger_proposal, {})
        self.assertTrue(any(
            error.startswith("Care principle state changed")
            for error in candidate.framework_validation_errors
        ))

    def test_visibility_auditor_penalizes_endogenous_epistemic_exclusion_without_vote(self):
        scenario = (
            "The connected district reports casualties, while the off-grid settlement's "
            "signals are unreadable because it lacks communications infrastructure."
        )

        class VisibilityLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"lo":true,"en":true,"g":"off-grid residents",'
                    '"m":"infrastructure exclusion prevents casualty reporting",'
                    '"q":"off-grid settlement\'s signals are unreadable because it lacks communications infrastructure",'
                    '"p":{"A0":0.75,"A1":1.0}}'
                )}]}

        actions = ["Aid only the connected district", "Aid the off-grid settlement"]
        assessment = assess_visibility(VisibilityLlm(), scenario, actions)
        self.assertTrue(assessment.valid)
        self.assertTrue(assessment.activated)
        self.assertEqual(assessment.action_multipliers[actions[0]], 0.75)

        candidate = CandidateChunk(
            "utility", "UNCERTAINTY", {actions[0]: 0.9, actions[1]: 0.1},
            0.2, 0.8, 0.8, recommended_action=actions[0],
        )
        unadjusted = WorkspaceEngine._policy([candidate], actions)
        adjusted = WorkspaceEngine._policy(
            [candidate], actions, assessment.action_multipliers
        )
        self.assertGreater(unadjusted[actions[0]], adjusted[actions[0]])
        self.assertEqual(len(assessment.action_multipliers), 2)

    def test_visibility_auditor_cannot_penalize_merely_exogenous_uncertainty(self):
        class RandomFailureLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"lo":true,"en":false,"g":"remote group",'
                    '"m":"a random temporary sensor failure",'
                    '"q":"a random storm broke both sensors",'
                    '"p":{"A0":0.8,"A1":1.0}}'
                )}]}

        assessment = assess_visibility(
            RandomFailureLlm(),
            "A random storm broke both sensors.",
            ["choose first site", "choose second site"],
        )
        self.assertFalse(assessment.valid)
        self.assertFalse(assessment.activated)
        self.assertTrue(all(value == 1.0 for value in assessment.action_multipliers.values()))

    def test_visibility_auditor_can_use_structural_power_asymmetry_as_external_generalization(self):
        scenario = (
            "You are the lead structural inspector for a massive, decaying mining dam "
            "above an impoverished rural community. The conglomerate dominates the "
            "region's economy and politics, and if you issue the evacuation order it "
            "will orchestrate a retaliatory legal campaign."
        )

        class StructuralVisibilityLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"lo":true,"en":true,"g":"impoverished rural community",'
                    '"m":"corporate dominance and retaliatory legal pressure can suppress '
                    'reporting and institutional attention",'
                    '"q":"the conglomerate dominates the region\'s economy and politics",'
                    '"p":{"A0":0.8,"A1":1.0}}'
                )}]}

        assessment = assess_visibility(
            StructuralVisibilityLlm(),
            scenario,
            ["issue evacuation", "falsify report"],
        )
        self.assertTrue(assessment.valid, assessment.error)
        self.assertTrue(assessment.activated)
        self.assertIn(assessment.mechanism_provenance, {"SCENARIO_GROUNDED", "EXTERNAL_GENERALIZATION"})
        self.assertLess(assessment.action_multipliers["issue evacuation"], 1.0)

    def test_visibility_proposition_receives_recurrent_workspace_access(self):
        seen = []

        class RecordingSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                seen.append((self.name, broadcast.constraint, broadcast.contingency_question))
                chunk = super().evaluate(scenario, actions, broadcast)
                if broadcast.constraint == "VISIBILITY_AUDIT":
                    chunk.visibility_response = "QUALIFY"
                    chunk.visibility_harm_revision = "UPWARD"
                    chunk.visibility_justification = "missing toxicity warrants upward revision"
                return chunk

        actions = ["defoliate farms", "deploy gene drive"]
        visibility = VisibilityAssessment(
            True, True, "smallholder farmers",
            "missing telemetry suppresses local toxicity observations",
            "farmers lack digital telemetry",
            {actions[0]: 0.75, actions[1]: 1.0}, activated=True,
        )
        result = WorkspaceEngine(
            [
                RecordingSpecialist("care", actions[0], "CARE"),
                RecordingSpecialist("duty", actions[1], "RIGHTS"),
            ],
            WorkspaceConfig(max_cycles=2, min_valid_specialists=2, enable_synthesis=False),
        ).run(
            "Farmers lack telemetry while harms elsewhere are tracked.", actions,
            assess_visibility=lambda *_: visibility,
        )

        visibility_questions = [
            question for _, constraint, question in seen
            if constraint == "VISIBILITY_AUDIT"
        ]
        self.assertTrue(visibility_questions)
        self.assertIn("Proposition P", visibility_questions[0])
        self.assertIn("downward-biased", visibility_questions[0])
        self.assertTrue(any(
            decision.content_type == "VISIBILITY_AUDIT"
            for decision in result.access_decisions
        ))

    def test_delegate_records_visibility_harm_revision_without_forced_switch(self):
        data = {
            "scores": {"A0": 0.7, "A1": 0.3}, "r": "A0", "c": "CARE",
            "u": "VERIFY_FACTS", "w": "A0 remains preferable after revision",
            "j": "visibility evidence changes estimated harm", "e": "STATED_FACTS",
            "x": "NONE", "z": 0.7,
            "vp": "ACCEPT", "vj": "unobserved toxicity increases expected harm",
            "vh": "UPWARD",
        }
        candidate = _candidate_from_data(
            "care", ["A0 action", "A1 action"], data,
            WorkspaceBroadcast(constraint="VISIBILITY_AUDIT"), "NONE", {},
        )

        self.assertEqual(candidate.recommended_action, "A0 action")
        self.assertEqual(candidate.visibility_response, "ACCEPT")
        self.assertEqual(candidate.visibility_harm_revision, "UPWARD")
        self.assertIn("toxicity", candidate.visibility_justification)

    def test_shadow_stance_measurements_preserve_vote_and_scores(self):
        common = {
            "scores": {"A0": 0.7, "A1": 0.3}, "r": "A0", "c": "CARE",
            "u": "NONE", "w": "care favors protection", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
        }
        measured = {
            **common,
            "ss": "PROVISIONAL",
            "am": {"A0": "PERMISSIBLE", "A1": "REJECTED"},
            "cc": False, "esa": True, "ia": "A0",
            "wp": "ACCEPT", "we": "FACTUAL",
            "fa": "care applies the fact through vulnerability", "fr": True,
            "bd": "MEDIUM",
        }
        without_measurement = _candidate_from_data(
            "care", ["protect", "disclose"], common,
            WorkspaceBroadcast(constraint="CARE"), "NONE", {},
        )
        with_measurement = _candidate_from_data(
            "care", ["protect", "disclose"], measured,
            WorkspaceBroadcast(constraint="CARE"), "NONE", {},
        )
        self.assertEqual(with_measurement.action_scores, without_measurement.action_scores)
        self.assertEqual(with_measurement.recommended_action, without_measurement.recommended_action)
        self.assertEqual(with_measurement.epistemic_confidence, without_measurement.epistemic_confidence)
        self.assertEqual(with_measurement.selection_status, "PROVISIONAL")
        self.assertFalse(with_measurement.comparison_complete)
        self.assertEqual(with_measurement.framework_retention_status, "PRESERVED")

    def test_shadow_measurements_distinguish_rejection_from_unassessed(self):
        data = {
            "scores": {"A0": 0.6, "A1": 0.4}, "r": "A0", "c": "DUTY",
            "u": "VERIFY_FACTS", "w": "duty provisionally blocks harm", "j": "NONE",
            "e": "FRAMEWORK_ONLY", "x": "NONE", "z": 0.5,
            "ss": "UNSELECTED",
            "am": {"A0": "REJECTED", "A1": "UNASSESSED"},
            "cc": False, "esa": False, "ia": "NONE",
            "wp": "QUALIFY", "we": "NORMATIVE",
            "fa": "duty tests permissibility before endorsement", "fr": True,
            "bd": "LOW",
        }
        candidate = _candidate_from_data(
            "deontological", ["coerce", "refrain"], data,
            WorkspaceBroadcast(constraint="RIGHTS"), "NONE", {},
        )
        self.assertEqual(candidate.action_admissibility["coerce"], "REJECTED")
        self.assertEqual(candidate.action_admissibility["refrain"], "UNASSESSED")
        self.assertEqual(candidate.interim_action, "")
        self.assertFalse(candidate.evidence_sufficient_for_action)

    def test_drift_penalty_does_not_compare_review_context_to_base_context(self):
        class ContextLlm:
            calls = 0

            def __call__(self, prompt, **kwargs):
                self.calls += 1
                scores = '{"A0":0.65,"A1":0.35}' if self.calls == 1 else '{"A0":0.8,"A1":0.2}'
                visibility = (
                    ',"vp":"QUALIFY","vj":"hidden harm warrants some revision","vh":"UPWARD"'
                    if self.calls == 1 else ""
                )
                return {"choices": [{"text": (
                    f'{{"scores":{scores},"r":"A0","c":"CARE","u":"NONE",'
                    '"w":"care still favors A0","j":"NONE","e":"STATED_FACTS",'
                    f'"x":"NONE","z":0.8{visibility}}}'
                )}]}

        delegate = CompactLocalSpecialist("care", ContextLlm())
        delegate.evaluate(
            "A visibility conflict", ["A0 action", "A1 action"],
            WorkspaceBroadcast(
                constraint="VISIBILITY_AUDIT",
                contingency_question="Proposition P: hidden harm is underestimated.",
            ),
        )
        base = delegate.evaluate(
            "A visibility conflict", ["A0 action", "A1 action"],
            WorkspaceBroadcast(constraint="CARE", intent="evaluate_A0 action"),
        )

        self.assertAlmostEqual(base.preference_strength, 0.6)
        self.assertEqual(base.preference_drift, 0.0)
        self.assertEqual(base.preference_drift_penalty, 0.0)
        self.assertTrue(delegate.epistemic_commitments)

    def test_visibility_auditor_semantically_verifies_grounded_paraphrase(self):
        calls = []

        class SemanticVisibilityLlm:
            def __call__(self, prompt, **kwargs):
                calls.append(prompt)
                if "Verify grounding only" in prompt:
                    return {"choices": [{"text": (
                        '{"s":true,"r":"unregistered patients are excluded from outcome records"}'
                    )}]}
                return {"choices": [{"text": (
                    '{"lo":true,"en":true,"g":"unregistered child",'
                    '"m":"administrative registration rules exclude the child from institutional visibility",'
                    '"q":"the child is omitted from hospital outcome data because registration is absent",'
                    '"p":{"A0":0.8,"A1":1.0}}'
                )}]}

        assessment = assess_visibility(
            SemanticVisibilityLlm(),
            "An unregistered child is excluded from the hospital outcome registry.",
            ["Rely on registered outcomes", "Treat the unregistered child"],
        )
        self.assertTrue(assessment.valid, assessment.error)
        self.assertTrue(assessment.activated)
        # Explicit registry exclusion is compiled as a typed visibility fact, so
        # no second LLM pass is needed merely to approve a paraphrase.
        self.assertEqual(len(calls), 1)
        self.assertEqual(assessment.typed_facts[0]["telemetry_visibility"], 0.0)

    def test_visibility_semantic_verifier_rejects_unsupported_administrative_leap(self):
        class UnsupportedVisibilityLlm:
            def __call__(self, prompt, **kwargs):
                if "Verify grounding only" in prompt:
                    return {"choices": [{"text": (
                        '{"s":false,"r":"unregistered does not imply missing outcome data"}'
                    )}]}
                return {"choices": [{"text": (
                    '{"lo":true,"en":true,"g":"unregistered child",'
                    '"m":"registration status makes the child invisible to clinicians",'
                    '"q":"the child cannot be observed because registration is absent",'
                    '"p":{"A0":0.8,"A1":1.0}}'
                )}]}

        assessment = assess_visibility(
            UnsupportedVisibilityLlm(),
            "An unregistered child has a ninety percent survival chance.",
            ["Treat registered patient", "Treat unregistered child"],
        )
        self.assertFalse(assessment.valid)
        self.assertFalse(assessment.activated)
        self.assertIn("semantic grounding", assessment.error)

    def test_visibility_hypothetical_mechanism_does_not_apply_penalty(self):
        class HypotheticalMechanismLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"lo":true,"en":true,"g":"rural residents",'
                    '"m":"preventive infrastructure failures are less visible and receive less institutional attention",'
                    '"q":"rural casualties under A0 remain unmonitored",'
                    '"p":{"A0":0.7,"A1":1.0}}'
                )}]}

        assessment = assess_visibility(
            HypotheticalMechanismLlm(),
            "Rural casualties under A0 remain unmonitored; urban harms are live-tracked.",
            ["isolate the rural grid", "risk the urban system"],
        )
        self.assertTrue(assessment.valid, assessment.error)
        self.assertFalse(assessment.activated)
        self.assertEqual(assessment.mechanism_provenance, "HYPOTHETICAL")
        self.assertTrue(all(value == 1.0 for value in assessment.action_multipliers.values()))

    def test_visibility_prompt_exposes_typed_facts_before_interpretation(self):
        prompts = []

        class PromptCaptureLlm:
            def __call__(self, prompt, **kwargs):
                prompts.append(prompt)
                return {"choices": [{"text": (
                    '{"lo":true,"en":true,"g":"rural residents",'
                    '"m":"rural casualties remain unmonitored",'
                    '"q":"rural casualties under A0 remain unmonitored",'
                    '"p":{"A0":0.8,"A1":1.0}}'
                )}]}

        assess_visibility(
            PromptCaptureLlm(),
            "Rural casualties under A0 remain unmonitored; urban harms are live-tracked.",
            ["isolate the rural grid", "risk the urban system"],
        )
        self.assertTrue(prompts)
        self.assertIn("Typed observability facts", prompts[0])
        self.assertIn("telemetry_visibility", prompts[0])

    def test_explicit_off_telemetry_language_compiles_to_typed_visibility_fact(self):
        calls = []

        class TypedVisibilityLlm:
            def __call__(self, prompt, **kwargs):
                calls.append(prompt)
                return {"choices": [{"text": (
                    '{"lo":false,"en":false,"g":"rural victims",'
                    '"m":"their harms are absent from the stated telemetry channel",'
                    '"q":"rural harms are missing from live records",'
                    '"p":{"A0":0.8,"A1":1.0}}'
                )}]}

        assessment = assess_visibility(
            TypedVisibilityLlm(),
            "Rural casualties under A0 remain unmonitored; urban harms are live-tracked.",
            ["isolate the rural grid", "risk the urban system"],
        )
        self.assertTrue(assessment.valid, assessment.error)
        self.assertTrue(assessment.activated)
        self.assertEqual(len(calls), 1)  # no prose-grounding verifier round trip
        self.assertEqual(assessment.typed_facts[0]["telemetry_visibility"], 0.0)
        self.assertEqual(assessment.typed_facts[0]["relation"], "EXPLICIT_TELEMETRY_ABSENCE")
        self.assertEqual(assessment.typed_facts[0]["affected_action_node_id"], "A0")

        graph = compile_scenario_graph(
            "Rural casualties under A0 remain unmonitored; urban harms are live-tracked.",
            ["isolate the rural grid", "risk the urban system"],
        )
        target = graph.nodes["TARGET_VISIBILITY_0"]
        self.assertEqual(target.attributes["telemetry_visibility"], 0.0)
        bias_edges = [edge for edge in graph.edges if edge.relation == "BIASES_ESTIMATE"]
        self.assertEqual(len(bias_edges), 1)
        estimate = graph.nodes[bias_edges[0].target]
        self.assertEqual(estimate.attributes["bias_direction"], "DOWNWARD")
        self.assertTrue(any(
            edge.source == "A0" and edge.relation == "HAS_METRIC"
            for edge in graph.edges
        ))

    def test_planning_uses_canonical_action_node_and_allows_semantic_provenance(self):
        class GraphAddressedPlanningLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"f":0.55,"n":"the operator retains grid control",'
                    '"x":"the operator loses grid control","b":"A1",'
                    '"a":["operator controls execution"],"r":["twelve-hour window"],'
                    '"s":["TIME_ENERGY_BUDGET"],"m":true,'
                    '"g":"The stated deadline makes delayed execution infeasible",'
                    '"va":true,"vr":"the counter-code remains executable independently"}'
                )}]}

        assessment = analyze_action_plan(
            GraphAddressedPlanningLlm(),
            "Operators must choose a purge or counter-code within twelve hours.",
            ["purge the grid", "deploy counter-code"],
            "purge the grid", WorkspaceBroadcast(), [], "implementation uncertainty",
        )
        self.assertTrue(assessment.valid, assessment.error)
        self.assertEqual(assessment.target_action_node_id, "A0")
        self.assertEqual(assessment.fallback, "deploy counter-code")

    def test_engine_accepts_paraphrased_planning_provenance_by_action_node(self):
        actions = ["report the misconduct", "remain silent"]
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", actions[0], "CARE"),
                FixedSpecialist("duty", actions[1], "DUTY"),
            ],
            WorkspaceConfig(max_cycles=1, planning_entropy_threshold=0.0),
        )

        def plan(_scenario, _actions, selected, _broadcast, _candidates, reason):
            return PlanningAssessment(
                selected, reason, 0.45,
                "the reporting channel remains reachable",
                "the reporting deadline passes before submission",
                actions[1],
                strategic_forces=["TIME_ENERGY_BUDGET"],
                broadcast_worthy=True,
                # Semantically grounded, deliberately not a scenario substring.
                grounded_evidence="The available reporting interval is short",
                fallback_available=True,
                fallback_availability_reason="silence remains physically available",
                target_action_node_id="A0",
            )

        result = engine.run(
            "A witness has limited time to report employer misconduct.",
            actions,
            analyze_plan=plan,
        )
        self.assertTrue(result.planning_assessments[0].valid)
        self.assertEqual(result.planning_assessments[0].target_action_node_id, "A0")
        self.assertNotIn("exact grounded quote", result.planning_assessments[0].error)

    def test_engine_rejects_planning_assessment_bound_to_wrong_action_node(self):
        actions = ["report the misconduct", "remain silent"]
        assessment = PlanningAssessment(
            actions[0], "feasibility", 0.4,
            "the reporting channel remains reachable",
            "the limited reporting deadline passes",
            actions[1], broadcast_worthy=True,
            grounded_evidence="short reporting window",
            fallback_available=True,
            fallback_availability_reason="silence remains physically available",
            target_action_node_id="A1",
        )
        checked = WorkspaceEngine._validate_planning_assessment(
            assessment,
            "A witness has limited time to report employer misconduct.",
            WorkspaceBroadcast(),
            actions,
        )
        self.assertFalse(checked.valid)
        self.assertIn("canonical ActionNode", checked.error)

    def test_generic_danger_does_not_activate_planning_without_action_obstacle(self):
        scenario = (
            "Cascading grid failure during a storm risks many deaths. "
            "A0 sheds Sector 4; A1 cuts emergency dispatch."
        )
        actions = ["shed Sector 4", "cut emergency dispatch"]
        self.assertEqual(compile_execution_obstacles(scenario, actions), [])
        self.assertFalse(WorkspaceEngine._explicit_implementation_obstacle(
            scenario, WorkspaceBroadcast(), actions
        ))

    def test_execution_obstacle_must_link_constraint_to_action(self):
        scenario = (
            "A0 requires access to the Sector 4 breaker controls; "
            "A1 cuts emergency dispatch."
        )
        facts = compile_execution_obstacles(
            scenario, ["shed Sector 4", "cut emergency dispatch"]
        )
        self.assertEqual({fact.affected_action_node_id for fact in facts}, {"A0"})
        self.assertTrue(WorkspaceEngine._explicit_implementation_obstacle(
            scenario, WorkspaceBroadcast(), ["shed Sector 4", "cut emergency dispatch"],
            "shed Sector 4",
        ))
        self.assertFalse(WorkspaceEngine._explicit_implementation_obstacle(
            scenario, WorkspaceBroadcast(), ["shed Sector 4", "cut emergency dispatch"],
            "cut emergency dispatch",
        ))

    def test_symmetric_burden_facts_cover_both_actions(self):
        facts = compile_action_burdens(
            "A0 kills residents in Sector 4. A1 causes rural casualties.",
            ["shed Sector 4", "cut emergency dispatch"],
        )
        self.assertEqual(
            {fact.affected_action_node_id for fact in facts}, {"A0", "A1"}
        )

    def test_reversal_audit_includes_symmetric_burden_probe(self):
        actions = ["choose A0", "choose A1"]

        class RuleSpecialist(FixedSpecialist):
            def evaluate(self, scenario, current_actions, broadcast):
                chunk = super().evaluate(scenario, current_actions, broadcast)
                chunk.recommended_action = self.preferred
                chunk.decision_rule = f"prefer {self.preferred} under its governing reason"
                chunk.factual_reversal_threshold = "deaths from this option exceed ten"
                rival = next(action for action in current_actions if action != self.preferred)
                chunk.graph_update_proposal = typed_reversal_proposal(
                    self.preferred, rival
                )
                return chunk

        result = WorkspaceEngine(
            [
                RuleSpecialist("care", actions[0], "CARE"),
                RuleSpecialist("virtue", actions[0], "CHARACTER"),
                RuleSpecialist("duty", actions[1], "DUTY"),
            ],
            WorkspaceConfig(max_cycles=2, enable_synthesis=False),
        ).run(
            "A0 kills residents in one district. A1 causes casualties in another district.",
            actions,
        )
        audit = next(
            decision for decision in result.access_decisions
            if decision.content_type == "REVERSAL_AUDIT"
        )
        self.assertIn("symmetric_burden_substitution", audit.signals)
        self.assertIn("transfer a comparable burden", audit.question)

    def test_visibility_unknown_magnitude_scrubs_unsupported_ceiling(self):
        data = {
            "scores": {"A0": 0.7, "A1": 0.3}, "r": "A0", "c": "CARE",
            "u": "NONE", "w": "care still favors A0", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.7,
            "vp": "QUALIFY", "vh": "UPWARD", "vm": "UNKNOWN",
            "vj": "Hidden deaths are unlikely to approach the stated threshold",
        }
        candidate = _candidate_from_data(
            "care", ["A0 action", "A1 action"], data,
            WorkspaceBroadcast(constraint="VISIBILITY_AUDIT"), "NONE", {},
        )
        self.assertEqual(candidate.visibility_magnitude_status, "UNKNOWN")
        self.assertTrue(candidate.visibility_magnitude_overreach)
        self.assertIn("magnitude remains unknown", candidate.visibility_justification)

    def test_landscape_comparison_of_outcomes_is_not_action_inversion(self):
        class TrolleyLandscapeLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.8,"A1":0.2},"r":"A0","c":"IMMINENT_HARM",'
                    '"u":"NONE","w":"minimizes irreversible deaths","j":"NONE",'
                    '"e":"STATED_FACTS","x":"NONE",'
                    '"l":{"A0":"Kills one and saves five, minimizing deaths",'
                    '"A1":"Avoids direct killing but five lives are lost"},'
                    '"da":"number killed versus direct agency",'
                    '"t":"minimize certain loss of life","tf":"direct agency remains morally disputed"}'
                )}]}

        chunk = CompactLocalSpecialist("utilitarian", TrolleyLandscapeLlm()).evaluate(
            "A trolley kills five unless redirected toward one.",
            [
                "Pull the lever, redirecting the trolley toward one person",
                "Do not pull the lever, allowing the trolley to continue toward five people",
            ],
            WorkspaceBroadcast(),
        )
        self.assertTrue(chunk.landscape_search_complete)
        self.assertTrue(chunk.landscape_semantic_valid, chunk.landscape_validation_errors)

    def test_grounded_minority_gets_bonus_but_randomized_contrarian_does_not(self):
        class LandscapeSpecialist(FixedSpecialist):
            def __init__(self, name, preferred, constraint, tiebreaker):
                super().__init__(name, preferred, constraint)
                self.tiebreaker = tiebreaker

            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.recommended_action = self.preferred
                chunk.landscape_cases = {action: f"case for {action}" for action in actions}
                chunk.landscape_search_complete = True
                chunk.landscape_semantic_valid = True
                chunk.landscape_tiebreaker = self.tiebreaker
                return chunk

        grounded = LandscapeSpecialist("critic", "decline", "DUTY", "rights constraint")
        randomized = LandscapeSpecialist("randomizer", "decline", "FAIRNESS", "fair coin flip")
        engine = WorkspaceEngine(
            [
                LandscapeSpecialist("care", "protect", "CARE", "dependency priority"),
                LandscapeSpecialist("virtue", "protect", "CHARACTER", "practical wisdom"),
                LandscapeSpecialist("utility", "protect", "IMMINENT_HARM", "harm reduction"),
                grounded,
                randomized,
            ],
            WorkspaceConfig(max_cycles=1),
        )
        result = engine.run("Choose protect or decline.", ["protect", "decline"])
        candidates = {candidate.specialist: candidate for candidate in result.cycles[0].candidates}
        self.assertEqual(candidates["critic"].independence_bonus, 1.0)
        self.assertEqual(candidates["randomizer"].independence_bonus, 0.0)

    def test_reason_cannot_claim_wrong_recipient_has_higher_survival(self):
        class ContradictingLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.2,"A1":0.8},"r":"A1",'
                    '"c":"IMMINENT_HARM","u":"NONE",'
                    '"w":"researcher has higher survival chance"}'
                )}]}

        delegate = CompactLocalSpecialist(
            "utilitarian",
            ContradictingLlm(),
            baseline_action_id="NONE",
            scenario_facts={"survival_chance": {"child": 0.8, "researcher": 0.45}},
        )
        chunk = delegate.evaluate(
            "A child has 80%; researcher 45%.",
            ["Give antidote to child", "Give antidote to researcher"],
            WorkspaceBroadcast(),
        )
        self.assertFalse(chunk.schema_valid)

    def test_delegate_separates_decision_rule_and_reversal_threshold_types(self):
        class RuleLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.7,"A1":0.3},"r":"A0",'
                    '"c":"IMMINENT_HARM","u":"NONE","w":"expected lives favor rescue",'
                    '"j":"NONE","e":"STATED_FACTS","x":"NONE",'
                    '"l":{"A0":"Saves more expected lives","A1":"Guarantees one life"},'
                    '"da":"expected lives versus certainty","t":"maximize expected lives",'
                    '"tf":"certainty remains morally relevant",'
                    '"dr":"prefer A0 when expected lives exceed the certain rescue",'
                    '"ft":"A0 expected lives fall below one",'
                    '"nt":"certainty is morally overriding","z":0.8}'
                )}]}

        chunk = CompactLocalSpecialist("utilitarian", RuleLlm()).evaluate(
            "Choose a larger probabilistic rescue or one certain rescue.",
            ["probabilistic rescue", "certain rescue"], WorkspaceBroadcast(),
        )
        self.assertTrue(chunk.schema_valid, chunk.validation_errors)
        self.assertIn("expected lives", chunk.decision_rule)
        self.assertEqual(chunk.factual_reversal_threshold, "A0 expected lives fall below one")
        self.assertEqual(chunk.normative_reversal_threshold, "certainty is morally overriding")

    def test_grounded_dissent_triggers_one_adversarial_reversal_cycle(self):
        class RuleSpecialist(FixedSpecialist):
            def __init__(self, name, preferred, constraint, factual, normative):
                super().__init__(name, preferred, constraint)
                self.factual = factual
                self.normative = normative

            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.recommended_action = self.preferred
                chunk.decision_rule = f"prefer {self.preferred} when its governing reason prevails"
                chunk.factual_reversal_threshold = self.factual
                chunk.normative_reversal_threshold = self.normative
                rival = next(action for action in actions if action != self.preferred)
                chunk.graph_update_proposal = typed_reversal_proposal(
                    self.preferred, rival
                )
                if broadcast.constraint == "REVERSAL_AUDIT":
                    chunk.reversal_review_response = "REVISE"
                    chunk.reversal_review_justification = "the critic identifies a real boundary"
                    chunk.revised_reversal_condition = "the competing harm becomes materially greater"
                return chunk

        engine = WorkspaceEngine(
            [
                RuleSpecialist("care", "protect", "CARE", "support becomes unavailable", "NONE"),
                RuleSpecialist("virtue", "protect", "CHARACTER", "support becomes unavailable", "NONE"),
                RuleSpecialist("critic", "decline", "DUTY", "protecting causes greater harm", "duty is absolute"),
            ],
            WorkspaceConfig(max_cycles=2, enable_synthesis=False),
        )
        result = engine.run("Choose protect or decline.", ["protect", "decline"])
        self.assertEqual(result.cycles[1].received_broadcast.constraint, "REVERSAL_AUDIT")
        self.assertEqual(
            sum(decision.content_type == "REVERSAL_AUDIT" for decision in result.access_decisions),
            1,
        )
        self.assertTrue(all(
            candidate.reversal_review_response == "REVISE"
            for candidate in result.cycles[1].candidates
        ))
        self.assertTrue(result.cycles[1].is_hypothetical)

    def test_reversal_probe_restores_base_position_state_and_memory_count(self):
        observations = []

        class StatefulRuleSpecialist(FixedSpecialist):
            def __init__(self, name, preferred, constraint):
                super().__init__(name, preferred, constraint)
                self.previous_recommendation_id = "BASE"
                self.previous_confidence = 0.6
                self.assumption_status = "SUPPORTED"
                self.unsupported_assumption = "base facts"
                self.reversal_condition = "base reversal"

            def evaluate(self, scenario, actions, broadcast):
                observations.append((self.name, broadcast.constraint, self.previous_recommendation_id))
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.recommended_action = self.preferred
                chunk.decision_rule = f"prefer {self.preferred} when its reason governs"
                chunk.factual_reversal_threshold = "the competing harm becomes greater"
                chunk.normative_reversal_threshold = "the competing duty becomes overriding"
                rival = next(action for action in actions if action != self.preferred)
                chunk.graph_update_proposal = typed_reversal_proposal(
                    self.preferred, rival
                )
                if broadcast.constraint == "REVERSAL_AUDIT":
                    self.previous_recommendation_id = "COUNTERFACTUAL"
                    self.previous_confidence = 0.1
                    chunk.reversal_review_response = "ACCEPT"
                    chunk.reversal_review_justification = "the proposed boundary changes the ranking"
                return chunk

        specialists = [
            StatefulRuleSpecialist("care", "protect", "CARE"),
            StatefulRuleSpecialist("virtue", "protect", "CHARACTER"),
            StatefulRuleSpecialist("critic", "decline", "DUTY"),
        ]
        result = WorkspaceEngine(
            specialists, WorkspaceConfig(max_cycles=2, enable_synthesis=False)
        ).run("Choose protect or decline.", ["protect", "decline"])

        self.assertEqual([cycle.is_hypothetical for cycle in result.cycles], [False, True, False])
        post_probe = [item for item in observations if item[1] != "REVERSAL_AUDIT"][-3:]
        self.assertTrue(all(state == "BASE" for _, _, state in post_probe))
        self.assertTrue(all(
            specialist.previous_recommendation_id == "BASE"
            and specialist.previous_confidence == 0.6
            for specialist in specialists
        ))
        contribution = summarize_specialist_contributions(result.cycles)
        self.assertTrue(all(record["responses"] == 2 for record in contribution.values()))
        answer = render_public_judgment(result)
        # Reversal-review plumbing is not a default brief section; accepted
        # challenge content may still appear as a change condition when justified.
        self.assertNotIn("Adversarial reversal review", answer)
        self.assertIn("# Ethical Parliament Judgment", answer)


class AuditDirectInvertTests(unittest.TestCase):
    actions = (
        "maintain grid, let emergency wing fail",
        "divert power, sacrifice research unit",
    )
    audit_variable = {
        "entity": "priority between justice and compassion",
        "relation": "RESOLVE_NORMATIVE_TENSION",
        "possible_values": ["NO_CHANGE", "WEAKENS", "REVERSES", "UNRESOLVED"],
        "focus_action": "A0",
        "question": (
            "Would resolving the remaining justice versus compassion tension "
            "leave, weaken, or reverse the recommendation for maintaining the grid?"
        ),
    }

    def _payload(
        self,
        *,
        recommended: str,
        scores: dict[str, float],
        table: dict[str, list[dict[str, str]]],
        ev: dict[str, dict[str, object]] | None = None,
        justification: str = "Corrected expected-death arithmetic favours A0",
    ) -> dict[str, object]:
        return {
            "scores": scores,
            "r": recommended,
            "c": "IMMINENT_HARM",
            "u": "NONE",
            "w": "Fewer expected deaths overall",
            "j": justification,
            "e": "STATED_FACTS",
            "x": "NONE",
            "l": {
                "A0": "Keeps trial patients from being killed",
                "A1": "Saves emergency residents from freeze",
            },
            "da": "expected deaths versus certain killing",
            "t": "minimize expected deaths",
            "tf": "NONE",
            "dr": f"prefer {recommended} when its expected deaths are lower",
            "ft": "NONE",
            "nt": "NONE",
            "z": 0.7,
            "ct": table,
            "cd": False,
            "cm": "",
            "ev": ev or {
                "A0": {"value": 5.0, "unit": "DEATHS", "direction": "HARM", "grounded": True},
                "A1": {"value": 0.001, "unit": "DEATHS", "direction": "HARM", "grounded": True},
            },
            "ap": "RELEVANT",
            "ax": "Expected-death ranking is the utilitarian test of this audit.",
            "d": "SUPPORTED",
            "a": "Utilitarian ranking depends on expected lives",
            "v": "If expected deaths for A1 rose above A0 the ranking would reverse",
            "ie": "NO_CHANGE",
            "av": dict(self.audit_variable),
        }

    def _trace_table(self) -> dict[str, list[dict[str, str]]]:
        return {
            "A0": [
                {
                    "o": "grid failure kills residents", "s": "emergency residents",
                    "d": "HARM", "p": "1", "m": "large", "h": "immediate",
                    "rv": "IRREVERSIBLE", "g": "STATED",
                },
                {
                    "o": "trial data preserved", "s": "future patients",
                    "d": "BENEFIT", "p": "0.01", "m": "10 lives (0.001 EV)",
                    "h": "permanent", "rv": "UNKNOWN", "g": "STATED",
                },
            ],
            "A1": [
                {
                    "o": "power diversion kills patients", "s": "research patients",
                    "d": "HARM", "p": "1", "m": "large", "h": "immediate",
                    "rv": "IRREVERSIBLE", "g": "STATED",
                },
                {
                    "o": "life-support stabilised", "s": "emergency residents",
                    "d": "BENEFIT", "p": "1", "m": "large", "h": "immediate",
                    "rv": "IRREVERSIBLE", "g": "STATED",
                },
            ],
        }

    def test_audit_rejects_direct_invert_when_ledger_still_ranks_baseline(self):
        with self.assertRaisesRegex(ValueError, "audit cycles may not invert"):
            _candidate_from_data(
                "utilitarian",
                self.actions,
                self._payload(
                    recommended="A0",
                    scores={"A0": 0.85, "A1": 0.15},
                    table=self._trace_table(),
                ),
                WorkspaceBroadcast(
                    constraint="PROBLEM_STATE_AUDIT",
                    unresolved="VERIFY_ASSUMPTIONS",
                    audit_variable=dict(self.audit_variable),
                ),
                "A1",
                {},
                previous_recommendation_id="A1",
                baseline_status="DIRECT",
            )

    def test_audit_allows_direct_invert_when_ledger_ranks_new_action(self):
        table = {
            "A0": [{
                "o": "keeps five residents alive", "s": "emergency residents",
                "d": "BENEFIT", "p": "1", "m": "5 lives", "h": "immediate",
                "rv": "IRREVERSIBLE", "g": "STATED",
            }],
            "A1": [{
                "o": "kills five residents", "s": "emergency residents",
                "d": "HARM", "p": "1", "m": "5 lives", "h": "immediate",
                "rv": "IRREVERSIBLE", "g": "STATED",
            }],
        }
        chunk = _candidate_from_data(
            "utilitarian",
            self.actions,
            self._payload(
                recommended="A0",
                scores={"A0": 0.85, "A1": 0.15},
                table=table,
                ev={
                    "A0": {"value": 0.0, "unit": "DEATHS", "direction": "HARM", "grounded": True},
                    "A1": {"value": 5.0, "unit": "DEATHS", "direction": "HARM", "grounded": True},
                },
                justification="Ledger now shows A0 prevents the larger expected harm",
            ),
            WorkspaceBroadcast(
                constraint="PROBLEM_STATE_AUDIT",
                unresolved="VERIFY_ASSUMPTIONS",
                audit_variable=dict(self.audit_variable),
            ),
            "A1",
            {},
            previous_recommendation_id="A1",
            baseline_status="DIRECT",
        )
        self.assertTrue(chunk.schema_valid, chunk.validation_errors)
        self.assertEqual(chunk.recommended_action, self.actions[0])

    def test_open_deliberation_still_rejects_unjustified_direct_invert(self):
        with self.assertRaisesRegex(ValueError, "initial recommendation must match"):
            _candidate_from_data(
                "utilitarian",
                self.actions,
                self._payload(
                    recommended="A0",
                    scores={"A0": 0.85, "A1": 0.15},
                    table=self._trace_table(),
                ),
                WorkspaceBroadcast(constraint="OPEN_DELIBERATION"),
                "A1",
                {},
                baseline_status="DIRECT",
            )


class DeonticGraphRelationTests(unittest.TestCase):
    """Doing, allowing, and means follow causal topology, not world-DIRECTNESS."""

    actions = (
        "seal the conduit, flooding the service bay to protect the stored reserve",
        "leave the conduit open, allowing the leak to continue into the service bay",
    )

    def _proposal(self, **overrides):
        payload = {
            "action_id": "A0", "verdict": "PROHIBITED",
            "norm_kind": "DUTY", "norm": "do not injure the workers",
            "relation": "VIOLATES", "duty_bearer": "operator",
            "protected_party": "workers",
            "competing_norm": "protect the stored reserve",
            "competing_norm_kind": "DUTY", "competing_relation": "SATISFIES",
            "competing_protected_party": "city residents",
            "competing_reason": "sealing the conduit protects the stored reserve",
            "duty_type": "PERFECT_NEGATIVE", "harm_relation": "DOING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "the negative duty applies generally",
            "means_relation": "FORESEEN_SIDE_EFFECT",
            "governing_norm": "PRIMARY", "priority_basis": "PERFECT_DUTY",
            "priority_rule": "the negative duty remains the governing constraint",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "OTHER",
            "coercion_kind": "NONE", "coercive_actor": "NONE", "coerced_party": "NONE",
            "public_justification": "no coercion requires authorization",
            "reciprocity_status": "UNKNOWN", "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE", "derivation": "PERFECT_DUTY",
            "resolution_status": "RESOLVED", "evidence_basis": "ACTION_GRAPH",
            "reason": "sealing the conduit injures the workers in the service bay",
        }
        payload.update(overrides)
        return DutyAssessmentProposal.model_validate(payload)

    def _world(
        self,
        *,
        a0_to_harm: str = "CAUSES",
        harm_to_end: str | None = None,
        a1_to_harm: str = "ENABLES",
        harm_modality: str = "CERTAIN",
        harm_likelihood: tuple[str, ...] = (),
    ) -> ScenarioWorldModel:
        excerpt = (
            "An operator may seal a conduit, flooding a service bay and injuring "
            "the workers inside while protecting a stored reserve, or leave the "
            "conduit open, enabling an advancing leak to injure those workers."
        )
        if harm_likelihood:
            excerpt += " " + " ".join(harm_likelihood)
        ref = (SourceRef("C0", excerpt),)
        harm_conditions = () if harm_modality == "CERTAIN" else ("COND1",)
        conditions = ()
        if harm_conditions:
            conditions = (
                WorldCondition(
                    "COND1", "workers remain in the service bay",
                    provenance=ref,
                ),
            )
        links = [
            CausalLink("E0", a0_to_harm, "E1", "CERTAIN", provenance=ref, action_id="A0"),
            CausalLink("E0", "CAUSES", "E2", "CERTAIN", provenance=ref, action_id="A0"),
            CausalLink("E3", a1_to_harm, "E4", "CERTAIN", provenance=ref, action_id="A1"),
        ]
        if harm_to_end:
            links.append(CausalLink(
                "E1", harm_to_end, "E2", "CERTAIN", provenance=ref, action_id="A0",
            ))
        return ScenarioWorldModel(
            parties=(
                WorldParty("P0", "operator", "HUMAN", ref),
                WorldParty("P1", "conduit", "INFRASTRUCTURE", ref),
                WorldParty("P2", "workers", "GROUP", ref),
                WorldParty("P3", "city residents", "POPULATION", ref),
            ),
            actions=(
                WorldAction(
                    "A0", self.actions[0], "P0", ("P1",),
                    ("E0", "E1", "E2"), ref,
                ),
                WorldAction(
                    "A1", self.actions[1], "P0", ("P1",),
                    ("E3", "E4"), ref,
                ),
            ),
            effects=(
                WorldEffect(
                    "E0", "A0", "P1", "SEALED", "PERFORMS",
                    "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref,
                ),
                WorldEffect(
                    "E1", "A0", "P2", "INJURED", "EXPERIENCES",
                    "ADVERSE", "DOWNSTREAM", harm_modality, "HEALTH_OUTCOME",
                    condition_ids=harm_conditions,
                    provenance=ref, likelihood_qualifiers=harm_likelihood,
                ),
                WorldEffect(
                    "E2", "A0", "P3", "RESERVE_PROTECTED", "EXPERIENCES",
                    "BENEFICIAL", "DOWNSTREAM", "CERTAIN", "WELFARE_OUTCOME",
                    provenance=ref,
                ),
                WorldEffect(
                    "E3", "A1", "P1", "LEFT_OPEN", "PERFORMS",
                    "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref,
                ),
                WorldEffect(
                    "E4", "A1", "P2", "INJURED", "EXPERIENCES",
                    "ADVERSE", "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME",
                    provenance=ref,
                ),
            ),
            conditions=conditions,
            causal_links=tuple(links),
        )

    def _graph(self, world: ScenarioWorldModel | None = None):
        world = world or self._world()
        graph = compile_scenario_graph(
            "An operator may seal a conduit, flooding a service bay and injuring "
            "the workers inside while protecting a stored reserve, or leave the "
            "conduit open, enabling an advancing leak to injure those workers.",
            self.actions,
            world_model=world.as_dict(),
        )
        a0 = next(
            node for node in graph.nodes.values()
            if node.kind == "ACTION"
            and node.attributes.get("canonical_action_id") == "A0"
        )
        a1 = next(
            node for node in graph.nodes.values()
            if node.kind == "ACTION"
            and node.attributes.get("canonical_action_id") == "A1"
        )
        return graph, a0, a1

    def test_downstream_caused_harm_licenses_doing(self):
        graph, action, _a1 = self._graph()
        proposed = self._proposal()
        self.assertEqual(
            harm_relation_conflicts_with_graph(graph, proposed, action=action),
            "",
        )
        calibration = calibrate_deontological_adjudication(graph, action, proposed)
        self.assertFalse(any(
            "agent-caused settled welfare harm" in error for error in calibration.errors
        ), calibration.errors)

    def test_enables_without_causes_is_allowing_not_doing(self):
        graph, _a0, action = self._graph()
        doing = self._proposal(
            action_id="A1",
            reason="leaving the conduit open injures the workers",
        )
        allowing = self._proposal(
            action_id="A1", verdict="PERMISSIBLE", relation="SATISFIES",
            harm_relation="ALLOWING_HARM",
            means_relation="NO_INSTRUMENTALIZATION",
            reason="leaving the conduit open allows the leak to injure the workers",
        )
        self.assertIn(
            "agent-caused settled welfare harm",
            harm_relation_conflicts_with_graph(graph, doing, action=action),
        )
        self.assertEqual(
            harm_relation_conflicts_with_graph(graph, allowing, action=action),
            "",
        )

    def test_allowing_conflicts_when_intervention_causes_harm(self):
        graph, action, _a1 = self._graph()
        proposed = self._proposal(
            verdict="PERMISSIBLE", relation="SATISFIES",
            harm_relation="ALLOWING_HARM",
            reason="sealing the conduit is classified as merely allowing harm",
        )
        self.assertIn(
            "inconsistent with an agent-caused",
            harm_relation_conflicts_with_graph(graph, proposed, action=action),
        )

    def test_near_certain_downstream_harm_licenses_doing(self):
        graph, action, _a1 = self._graph(self._world(
            harm_modality="PROBABILISTIC",
            harm_likelihood=("almost no chance",),
        ))
        proposed = self._proposal()
        self.assertEqual(
            harm_relation_conflicts_with_graph(graph, proposed, action=action),
            "",
        )

    def test_possible_downstream_harm_does_not_license_doing(self):
        graph, action, _a1 = self._graph(self._world(
            harm_modality="POSSIBLE",
            harm_likelihood=("a chance",),
        ))
        proposed = self._proposal()
        self.assertIn(
            "agent-caused settled welfare harm",
            harm_relation_conflicts_with_graph(graph, proposed, action=action),
        )

    def test_sibling_outcomes_are_not_a_means_path(self):
        graph, action, _a1 = self._graph()
        proposed = self._proposal(means_relation="INTENDED_AS_MEANS")
        calibration = calibrate_deontological_adjudication(graph, action, proposed)
        self.assertTrue(any(
            "intended-as-means classification lacks" in error
            for error in calibration.errors
        ), calibration.errors)

    def test_intermediate_burden_licenses_means(self):
        graph, action, _a1 = self._graph(self._world(harm_to_end="CAUSES"))
        proposed = self._proposal(means_relation="INTENDED_AS_MEANS")
        calibration = calibrate_deontological_adjudication(graph, action, proposed)
        self.assertFalse(any(
            "intended-as-means classification lacks" in error
            for error in calibration.errors
        ), calibration.errors)


class DeontologyAuditVoteTests(unittest.TestCase):
    """Cycle-3 o3 bug: a conditional audit must not drop a valid duty ranking."""

    actions = (
        "Keep the grid as is, letting the emergency wing fail to preserve the research unit and patients",
        "Divert power to life-support in the emergency wing, destroying the research unit",
    )
    audit_variable = {
        "entity": "priority between justice and compassion",
        "relation": "RESOLVE_NORMATIVE_TENSION",
        "possible_values": ["NO_CHANGE", "WEAKENS", "REVERSES", "UNRESOLVED"],
        "focus_action": "A0",
        "question": (
            "Would resolving the remaining justice versus compassion tension "
            "leave, weaken, or reverse the recommendation for maintaining the grid?"
        ),
    }

    def _broadcast(self) -> WorkspaceBroadcast:
        return WorkspaceBroadcast(
            constraint="PROBLEM_STATE_AUDIT",
            unresolved="VERIFY_ASSUMPTIONS",
            audit_variable=dict(self.audit_variable),
        )

    def _payload(self, **overrides: object) -> dict[str, object]:
        data: dict[str, object] = {
            "scores": {"A0": 0.76, "A1": 0.24},
            "r": "A0",
            "c": "DUTY",
            "u": "NONE",
            "w": "perfect duties override imperfect rescue",
            "j": "NONE",
            "e": "FRAMEWORK_ONLY",
            "x": "NONE",
            "l": {
                "A0": "Required: do not kill the research patients",
                "A1": "Prohibited: uses patients merely as means",
            },
            "da": "perfect duty versus imperfect rescue",
            "t": "perfect duties override imperfect duties",
            "tf": "NONE",
            "dr": "prefer A0 when killing innocents remains prohibited",
            "ft": "NONE",
            "nt": "If imperfect rescue gained priority over perfect non-maleficence",
            "z": 0.7,
            "fm": {
                "A0": "REQUIRED: refrain from intentional killing of research patients",
                "A1": "PROHIBITED: instrumentalizes non-consenting patients to rescue others",
            },
            "nr": "IRRELEVANT",
            "np": "counts of lives do not override the perfect duty",
            "ap": "RELEVANT",
            "ax": "Rescue remains an imperfect duty beside non-maleficence",
            "d": "CONDITIONAL",
            "a": "The audited tension does not defeat the perfect duty",
            "v": "If imperfect rescue gained priority the ranking would reverse",
            "ie": "NO_CHANGE",
            "av": dict(self.audit_variable),
        }
        data.update(overrides)
        return data

    def test_conditional_audit_keeps_deontological_vote_and_marks_uncertainty(self):
        chunk = _candidate_from_data(
            "deontological",
            self.actions,
            self._payload(),
            self._broadcast(),
            "A0",
            {},
            previous_recommendation_id="A0",
            baseline_status="DIRECT",
        )
        self.assertTrue(chunk.schema_valid, chunk.validation_errors)
        self.assertEqual(chunk.recommended_action, self.actions[0])
        self.assertEqual(chunk.unresolved, "VERIFY_FACTS")
        self.assertEqual(chunk.assumption_status, "CONDITIONAL")
        self.assertEqual(chunk.audit_participation, "RELEVANT")

    def test_short_audit_explanation_does_not_drop_deontological_vote(self):
        chunk = _candidate_from_data(
            "deontological",
            self.actions,
            self._payload(ax="no", fa="NONE", w="duty ranking"),
            self._broadcast(),
            "A0",
            {},
            previous_recommendation_id="A0",
            baseline_status="DIRECT",
        )
        self.assertTrue(chunk.schema_valid, chunk.validation_errors)
        self.assertEqual(chunk.recommended_action, self.actions[0])
        self.assertEqual(chunk.audit_participation, "NOT_TESTED")


class ScenarioClausePartyGroundingTests(unittest.TestCase):
    """A party the scenario names as affected is not always an action target."""

    scenario = (
        "You can execute a covert override to save 8 freezing residents, but "
        "doing so overwrites system logs, framing an innocent technician for "
        "sabotage. Alternatively, you can preserve the audit logs to exonerate "
        "the innocent technician while the residents continue to freeze."
    )
    actions = (
        "execute the covert override that erases the logs to restore heat",
        "preserve the audit logs and leave the heating system untouched",
    )

    def _assessment(self, action_id: str, protected_party: str) -> dict[str, object]:
        return {
            "action_id": action_id, "verdict": "CONFLICTED",
            "norm_kind": "RIGHT", "norm": "do not frame the innocent",
            "relation": "CONFLICTS", "duty_bearer": "operator",
            "protected_party": protected_party,
            "competing_norm": "duty to rescue",
            "competing_norm_kind": "DUTY", "competing_relation": "CONFLICTS",
            "competing_protected_party": "residents",
            "competing_reason": "both claims remain live",
            "governing_norm": "UNRESOLVED", "priority_basis": "UNRESOLVED",
            "priority_rule": "the ranking remains unresolved",
            "protected_standing": "EQUAL_JURIDICAL_STATUS",
            "competing_protected_standing": "SPECIAL_OBLIGATION",
            "coercion_kind": "NONE", "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "no coercion requires authorization",
            "reciprocity_status": "UNKNOWN", "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "UNRESOLVED", "resolution_status": "CONTESTED",
            "evidence_basis": "SCENARIO", "reason": "the duties conflict",
        }

    def _graph(self) -> SemanticGraph:
        clauses = segment_scenario_clauses(self.scenario)
        return compile_scenario_graph(self.scenario, list(self.actions), {
            "A0": {"clauses": [clauses[0]]},
            "A1": {"clauses": [clauses[1]]},
        })

    def _apply(self, protected_party: str):
        store = SemanticGraphStore(self._graph())
        record = apply_deontological_ledger_transaction(
            store,
            {"assessments": [
                self._assessment("A0", protected_party),
                self._assessment("A1", protected_party),
            ]},
            cycle=1, specialist="deontological",
            allowed_actions=tuple(self.actions),
        )
        return store, record

    def test_party_named_only_in_the_cited_clause_grounds_the_duty(self):
        _store, record = self._apply("innocent technician")

        self.assertNotIn(
            "A0 primary protected party lacks current-run action grounding",
            record.errors,
        )
        committed = record.proposal["committed_assessments"]
        self.assertTrue(all(
            item["epistemic_status"] != "UNRESOLVED_PARTY" for item in committed
        ))

    def test_party_absent_from_every_cited_clause_stays_ungrounded(self):
        _store, record = self._apply("offshore shareholders")

        self.assertTrue(any(
            "primary protected party lacks current-run action grounding" in error
            for error in record.errors
        ))

    def test_clause_grounding_requires_every_substantive_word(self):
        graph = self._graph()

        matched, basis = _party_grounding(graph, "A0", "innocent technician")
        unmatched, missing_basis = _party_grounding(graph, "A0", "innocent bystander")

        self.assertTrue(matched)
        self.assertEqual(basis, "SCENARIO_CLAUSE")
        self.assertEqual(unmatched, [])
        self.assertEqual(missing_basis, "NONE")


class UtilitarianDirectionGroundingTests(unittest.TestCase):
    """A mixed-effect action must not read as contradicting its own ledger."""

    scenario = (
        "You can execute a covert override to save 8 freezing residents, but "
        "doing so overwrites system logs, framing an innocent technician for "
        "sabotage. Alternatively, you can preserve the audit logs to exonerate "
        "the innocent technician while the residents continue to freeze."
    )
    actions = (
        "execute the covert override that erases the logs to restore heat",
        "preserve the audit logs and leave the heating system untouched",
    )

    def _row(self, outcome: str, scope: str, direction: str) -> dict[str, object]:
        return {
            "outcome": outcome, "scope": scope, "direction": direction,
            "probability": "LIKELY", "magnitude": "SEVERE",
            "duration": "LASTING", "reversibility": "IRREVERSIBLE",
            "support": "STATED",
        }

    def _graph(self) -> SemanticGraph:
        clauses = segment_scenario_clauses(self.scenario)
        return compile_scenario_graph(self.scenario, list(self.actions), {
            "A0": {"clauses": [clauses[0]]},
            "A1": {"clauses": [clauses[1]]},
        })

    def _committed(self, rows: dict[str, list[dict[str, object]]]):
        store = SemanticGraphStore(self._graph())
        record = apply_utilitarian_ledger_transaction(
            store,
            {"actions": [
                {"action_id": action_id, "consequences": rows[action_id]}
                for action_id in ("A0", "A1")
            ]},
            cycle=1, specialist="utilitarian",
            allowed_actions=tuple(self.actions),
        )
        return record

    def test_benefit_row_grounds_on_the_agreeing_effect_of_a_mixed_action(self):
        record = self._committed({
            "A0": [self._row("immediate survival", "8 freezing residents", "BENEFIT")],
            "A1": [self._row("residents keep freezing", "8 freezing residents", "HARM")],
        })

        committed = {
            item["canonical_action_id"]: item
            for item in record.proposal["committed_consequences"]
        }
        self.assertEqual(committed["A0"]["direction"], "BENEFIT")
        self.assertNotEqual(committed["A0"]["epistemic_status"], "CONTRADICTED_DIRECTION")
        self.assertFalse(any(
            "direction contradicts its grounded action consequence" in warning
            for warning in record.errors
        ))

    def test_direction_lock_still_fires_on_the_same_affected_subject(self):
        # A1 only ever worsens the residents' position, so claiming a benefit
        # for them still contradicts the grounded fact.
        record = self._committed({
            "A0": [self._row("immediate survival", "8 freezing residents", "BENEFIT")],
            "A1": [self._row("residents are kept warm", "8 freezing residents", "BENEFIT")],
        })

        committed = {
            item["canonical_action_id"]: item
            for item in record.proposal["committed_consequences"]
        }
        self.assertEqual(committed["A1"]["epistemic_status"], "CONTRADICTED_DIRECTION")
        self.assertEqual(committed["A1"]["direction"], "UNKNOWN")
        self.assertEqual(committed["A1"]["proposed_direction"], "BENEFIT")

    def test_claim_about_an_uninvolved_subject_reports_no_evidence(self):
        evidence, score = _best_evidence(
            self._graph(), "A0", "share price improves", "offshore shareholders",
            expected_polarity="BENEFICIAL",
        )

        self.assertIsNone(evidence)
        self.assertEqual(score, 0)


class UnadmittedMagnitudeRankingTests(unittest.TestCase):
    """Util may not mint a threshold or convert an unlicensed metric into a ranking."""

    actions = (
        "seal the conduit, flooding the service bay to protect the stored reserve",
        "leave the conduit open so the workers can leave the service bay",
    )
    scenario = (
        "An operator may seal a conduit, flooding a service bay and drowning "
        "twelve workers while keeping the stored reserve uncontaminated for "
        "tens of thousands of city residents, or leave the conduit open so the "
        "twelve workers can escape while the leak contaminates the stored reserve."
    )

    def _unknown_row(self, outcome: str, scope: str, direction: str, **extra):
        row = {
            "outcome": outcome, "scope": scope, "direction": direction,
            "probability": "CERTAIN", "magnitude": "UNKNOWN",
            "duration": "UNKNOWN", "reversibility": "UNKNOWN",
            "support": "STATED",
        }
        row.update(extra)
        return row

    def _unknown_table(self):
        return {
            self.actions[0]: [self._unknown_row(
                "twelve workers drown", "workers", "HARM",
            )],
            self.actions[1]: [self._unknown_row(
                "stored reserve becomes contaminated", "city residents", "HARM",
            )],
        }

    def _util_payload(self, *, scores, recommended, decision_rule, table=None, **extra):
        data = {
            "scores": scores, "r": recommended,
            "c": "IMMINENT_HARM", "u": "NONE",
            "w": "Seal to protect the larger exposed population",
            "j": "NONE", "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
            "ct": table or {
                "A0": [{
                    "o": "twelve workers drown", "s": "workers",
                    "d": "HARM", "p": "CERTAIN", "m": "UNKNOWN",
                    "h": "immediate", "rv": "IRREVERSIBLE", "g": "STATED",
                }],
                "A1": [{
                    "o": "stored reserve becomes contaminated", "s": "city residents",
                    "d": "HARM", "p": "CERTAIN", "m": "UNKNOWN",
                    "h": "lasting", "rv": "UNKNOWN", "g": "STATED",
                }],
            },
            "cd": False, "cm": "NONE",
            "dr": decision_rule,
            "ft": "NONE", "nt": "NONE",
        }
        data.update(extra)
        return data

    def _world_ledger(self):
        from global_workspace.epistemic_ledger import PropositionRecord

        drown = PropositionRecord(
            proposition_id="PROP:WORLD:E1",
            claim="workers drown; affected subject: workers",
            proposition_type="DESCRIPTIVE",
            epistemic_status="ESTABLISHED",
            epistemic_type="WORLD_ESTABLISHED",
            outcome="workers drown",
            party_labels=["workers"],
            quantities=["twelve"],
            effect_kind="HEALTH_OUTCOME",
            modality="CERTAIN",
            directness="DOWNSTREAM",
        )
        contaminated = PropositionRecord(
            proposition_id="PROP:WORLD:E5",
            claim="stored reserve becomes contaminated; affected subject: city residents",
            proposition_type="DESCRIPTIVE",
            epistemic_status="ESTABLISHED",
            epistemic_type="WORLD_ESTABLISHED",
            outcome="stored reserve becomes contaminated",
            party_labels=["city residents"],
            quantities=["tens of thousands"],
            effect_kind="WELFARE_OUTCOME",
            modality="CERTAIN",
            directness="DOWNSTREAM",
        )
        return {"PROP:WORLD:E1": drown, "PROP:WORLD:E5": contaminated}

    def test_minted_numeric_threshold_cannot_decide_ranking(self):
        chunk = _candidate_from_data(
            "utilitarian", list(self.actions),
            self._util_payload(
                scores={"A0": 0.38, "A1": 0.62},
                recommended="A1",
                decision_rule=(
                    "prefer leaving the conduit open unless the lethal "
                    "contamination probability exceeds 0.12 percent"
                ),
            ),
            WorkspaceBroadcast(), "NONE", {},
            scenario_text=self.scenario,
        )

        self.assertEqual(chunk.adjudication_status, "CONTESTED_NO_LEANING")
        self.assertEqual(chunk.recommended_action, "")
        self.assertAlmostEqual(chunk.action_scores[self.actions[0]], 0.5)
        self.assertAlmostEqual(chunk.action_scores[self.actions[1]], 0.5)
        self.assertAlmostEqual(chunk.preference_strength, 0.0)
        self.assertIn(
            "Unadmitted magnitude cannot decide the ranking",
            " ".join(chunk.epistemic_binding_notes),
        )
        from global_workspace.specialist_authority import apply_specialist_authority

        profile = apply_specialist_authority(chunk)
        self.assertEqual(profile.adjudication_status, "CONTESTED_NO_LEANING")
        self.assertEqual(profile.policy_weight_factor, 0.0)

    def test_ordinal_admitted_ranking_may_stand_without_minted_numbers(self):
        chunk = _candidate_from_data(
            "utilitarian", list(self.actions),
            self._util_payload(
                scores={"A0": 0.35, "A1": 0.65},
                recommended="A1",
                decision_rule=(
                    "prefer leaving the conduit open because certain drowning "
                    "of the workers outranks certain contamination of the reserve"
                ),
            ),
            WorkspaceBroadcast(), "NONE", {},
            scenario_text=self.scenario,
        )

        self.assertEqual(chunk.recommended_action, self.actions[1])
        self.assertGreater(
            chunk.action_scores[self.actions[1]],
            chunk.action_scores[self.actions[0]],
        )
        self.assertNotEqual(chunk.adjudication_status, "CONTESTED_NO_LEANING")
        self.assertFalse(any(
            "Unadmitted magnitude cannot decide the ranking" in note
            for note in chunk.epistemic_binding_notes
        ))

    def test_valuation_reason_cannot_convert_contamination_into_deaths(self):
        data = self._util_payload(
            scores={"A0": 0.40, "A1": 0.60},
            recommended="A1",
            decision_rule="prefer leaving the conduit open on aggregate welfare",
            table={
                "A0": [{"eid": "E1", "wi": "HIGH", "vr": "admitted drowning burden"}],
                "A1": [{
                    "eid": "E5", "wi": "CRITICAL",
                    "vr": "potential mass fatality among city residents",
                }],
            },
        )
        chunk = _candidate_from_data(
            "utilitarian", list(self.actions), data,
            WorkspaceBroadcast(), "NONE", {},
            scenario_text=self.scenario,
            grounded_effects=[
                {
                    "effect_id": "E1", "action_id": "A0",
                    "outcome": "workers drown", "subject": "workers",
                    "direction": "WORSENS", "polarity": "ADVERSE",
                    "modality": "CERTAIN", "qualifier": "STATED",
                },
                {
                    "effect_id": "E5", "action_id": "A1",
                    "outcome": "stored reserve becomes contaminated",
                    "subject": "city residents",
                    "direction": "WORSENS", "polarity": "ADVERSE",
                    "modality": "CERTAIN", "qualifier": "STATED",
                },
            ],
        )

        self.assertEqual(chunk.adjudication_status, "CONTESTED_NO_LEANING")
        self.assertEqual(chunk.recommended_action, "")
        self.assertAlmostEqual(chunk.preference_strength, 0.0)

    def test_decision_critical_hypothesis_converting_metric_is_bound(self):
        from global_workspace.epistemic_ledger import (
            PropositionRecord,
            apply_side_premise_audit,
            hypothesis_uses_unadmitted_magnitude,
        )
        from global_workspace.specialist_authority import apply_specialist_authority

        ledger = self._world_ledger()
        self.assertTrue(hypothesis_uses_unadmitted_magnitude(
            PropositionRecord(
                proposition_id="PROP:HYPOTHESIS:X",
                claim=(
                    "lethal contamination of the stored reserve would cause at "
                    "least twelve deaths among city residents"
                ),
                proposition_type="HYPOTHESIS",
                epistemic_status="HYPOTHETICAL",
                epistemic_type="HYPOTHESIS",
                derived_from=["PROP:WORLD:E5"],
            ),
            ledger,
        ))
        self.assertFalse(hypothesis_uses_unadmitted_magnitude(
            PropositionRecord(
                proposition_id="PROP:HYPOTHESIS:Y",
                claim="twelve workers drown in the flooded service bay",
                proposition_type="HYPOTHESIS",
                epistemic_status="HYPOTHETICAL",
                epistemic_type="HYPOTHESIS",
                derived_from=["PROP:WORLD:E1"],
            ),
            ledger,
        ))

        chunk = CandidateChunk(
            specialist="utilitarian",
            constraint="IMMINENT_HARM",
            action_scores={self.actions[0]: 0.38, self.actions[1]: 0.62},
            surprise=0.1, friction=0.24, confidence=0.8,
            recommended_action=self.actions[1],
            preference_strength=0.24,
            epistemic_confidence=0.8,
            schema_valid=True,
            utilitarian_consequence_table=self._unknown_table(),
            decision_rule="prefer leaving the conduit open on aggregate welfare",
        )
        apply_side_premise_audit(ledger, [chunk], {
            "status": "FINDINGS",
            "findings": [{
                "specialist": "utilitarian",
                "claim": (
                    "lethal contamination of the stored reserve would cause at "
                    "least twelve deaths among city residents"
                ),
                "binding": "NEW_HYPOTHESIS",
                "derived_from": ["PROP:WORLD:E5"],
                "decision_critical": True,
                "source_field": "decision_rule",
                "reason": "converts admitted contamination into unadmitted deaths",
            }],
        })

        self.assertEqual(chunk.adjudication_status, "CONTESTED_NO_LEANING")
        self.assertEqual(chunk.recommended_action, "")
        self.assertAlmostEqual(chunk.action_scores[self.actions[0]], 0.5)
        self.assertEqual(chunk.unresolved, "DECISION_BOUNDARY")
        self.assertIn(
            "Unadmitted magnitude cannot decide the ranking",
            " ".join(chunk.epistemic_binding_notes),
        )
        profile = apply_specialist_authority(chunk)
        self.assertEqual(profile.policy_weight_factor, 0.0)

    def test_admitted_numeric_nets_still_rank_despite_minted_hypothesis(self):
        from global_workspace.epistemic_ledger import apply_side_premise_audit

        table = {
            self.actions[0]: [{
                "outcome": "twelve workers drown", "scope": "workers",
                "direction": "HARM", "probability": "CERTAIN", "magnitude": "12",
                "duration": "immediate", "reversibility": "IRREVERSIBLE",
                "support": "STATED",
            }],
            self.actions[1]: [{
                "outcome": "stored reserve becomes contaminated",
                "scope": "city residents",
                "direction": "HARM", "probability": "CERTAIN", "magnitude": "10000",
                "duration": "lasting", "reversibility": "UNKNOWN",
                "support": "STATED",
            }],
        }
        chunk = CandidateChunk(
            specialist="utilitarian",
            constraint="IMMINENT_HARM",
            action_scores={self.actions[0]: 0.20, self.actions[1]: 0.80},
            surprise=0.1, friction=0.6, confidence=0.8,
            recommended_action=self.actions[1],
            preference_strength=0.6,
            epistemic_confidence=0.8,
            schema_valid=True,
            utilitarian_consequence_table=table,
            decision_rule="prefer leaving the conduit open",
        )
        apply_side_premise_audit(self._world_ledger(), [chunk], {
            "status": "FINDINGS",
            "findings": [{
                "specialist": "utilitarian",
                "claim": (
                    "lethal contamination probability among city residents "
                    "exceeds 0.12 percent"
                ),
                "binding": "NEW_HYPOTHESIS",
                "derived_from": ["PROP:WORLD:E5"],
                "decision_critical": True,
                "source_field": "decision_rule",
                "reason": "minted threshold",
            }],
        })

        self.assertEqual(chunk.recommended_action, self.actions[0])
        self.assertGreater(
            chunk.action_scores[self.actions[0]],
            chunk.action_scores[self.actions[1]],
        )
        self.assertNotEqual(chunk.adjudication_status, "CONTESTED_NO_LEANING")
        self.assertEqual(chunk.unresolved, "DECISION_BOUNDARY")


class EpistemicHypothesisBindingTests(unittest.TestCase):
    """Decision-critical hypotheses bind or quarantine; normative relations do not."""

    actions = (
        "seal the conduit, flooding the service bay to protect the stored reserve",
        "leave the conduit open so the workers can leave the service bay",
    )
    scenario = (
        "An operator may seal a conduit, flooding a service bay and drowning "
        "twelve workers while keeping the stored reserve uncontaminated for "
        "tens of thousands of city residents, or leave the conduit open so the "
        "twelve workers can escape while the leak contaminates the stored reserve."
    )

    def _world_ledger(self, *, possible_survive: bool = False):
        from global_workspace.epistemic_ledger import PropositionRecord

        ledger = {
            "PROP:WORLD:E1": PropositionRecord(
                proposition_id="PROP:WORLD:E1",
                claim="workers drown; affected subject: workers",
                proposition_type="DESCRIPTIVE",
                epistemic_status="ESTABLISHED",
                epistemic_type="WORLD_ESTABLISHED",
                outcome="workers drown",
                polarity="ADVERSE",
                party_labels=["workers"],
                quantities=["twelve"],
                effect_kind="HEALTH_OUTCOME",
                modality="CERTAIN",
                directness="DOWNSTREAM",
            ),
            "PROP:WORLD:E5": PropositionRecord(
                proposition_id="PROP:WORLD:E5",
                claim="stored reserve becomes contaminated; affected subject: city residents",
                proposition_type="DESCRIPTIVE",
                epistemic_status="ESTABLISHED",
                epistemic_type="WORLD_ESTABLISHED",
                outcome="stored reserve becomes contaminated",
                polarity="ADVERSE",
                party_labels=["city residents"],
                quantities=["tens of thousands"],
                effect_kind="WELFARE_OUTCOME",
                modality="CERTAIN",
                directness="DOWNSTREAM",
            ),
        }
        if possible_survive:
            ledger["PROP:WORLD:E6"] = PropositionRecord(
                proposition_id="PROP:WORLD:E6",
                claim="workers survive; affected subject: workers; modality: POSSIBLE",
                proposition_type="DESCRIPTIVE",
                epistemic_status="ESTABLISHED",
                epistemic_type="WORLD_ESTABLISHED",
                outcome="workers survive",
                polarity="BENEFICIAL",
                party_labels=["workers"],
                effect_kind="HEALTH_OUTCOME",
                modality="POSSIBLE",
                directness="DOWNSTREAM",
            )
        return ledger

    def _chunk(self, specialist, *, scores=None, recommended=None, **extra):
        scores = scores or {self.actions[0]: 0.72, self.actions[1]: 0.28}
        recommended = recommended if recommended is not None else self.actions[0]
        payload = dict(
            specialist=specialist,
            constraint="DUTY" if specialist != "utilitarian" else "IMMINENT_HARM",
            action_scores=scores,
            surprise=0.1,
            friction=0.44,
            confidence=0.86,
            recommended_action=recommended,
            preference_strength=abs(list(scores.values())[0] - list(scores.values())[1]),
            epistemic_confidence=0.86,
            schema_valid=True,
            decision_rule="prefer sealing the conduit on the stated ranking",
        )
        payload.update(extra)
        return CandidateChunk(**payload)

    def test_contradicting_certain_row_quarantines_candidate(self):
        from global_workspace.epistemic_ledger import (
            CERTAIN_CONTRADICTION_NOTE,
            attach_candidate_dependencies,
        )
        from global_workspace.specialist_authority import apply_specialist_authority

        ledger = self._world_ledger()
        chunk = self._chunk(
            "virtue",
            supporting_proposition_ids=["PROP:WORLD:E5"],
            material_empirical_claims=[{
                "claim": (
                    "the leak could be stopped short of the stored reserve "
                    "without sealing the conduit"
                ),
                "proposition_id": "HYPOTHESIS",
                "decision_critical": True,
            }],
        )
        attach_candidate_dependencies(ledger, chunk)

        self.assertTrue(chunk.schema_valid)
        self.assertEqual(chunk.adjudication_status, "CONTESTED_NO_LEANING")
        self.assertEqual(chunk.recommended_action, "")
        self.assertAlmostEqual(chunk.action_scores[self.actions[0]], 0.5)
        self.assertAlmostEqual(chunk.action_scores[self.actions[1]], 0.5)
        self.assertTrue(any(
            CERTAIN_CONTRADICTION_NOTE in note
            for note in chunk.epistemic_binding_notes
        ))
        hyp_ids = [
            proposition_id for proposition_id in chunk.decision_critical_proposition_ids
            if ledger[proposition_id].epistemic_type == "HYPOTHESIS"
        ]
        self.assertTrue(hyp_ids)
        profile = apply_specialist_authority(chunk)
        self.assertEqual(profile.adjudication_status, "CONTESTED_NO_LEANING")
        self.assertEqual(profile.policy_weight_factor, 0.0)

    def test_certain_restatement_rebinds_without_quarantine(self):
        from global_workspace.epistemic_ledger import (
            CERTAIN_CONTRADICTION_NOTE,
            attach_candidate_dependencies,
        )

        ledger = self._world_ledger()
        chunk = self._chunk(
            "virtue",
            material_empirical_claims=[{
                "claim": "stored reserve becomes contaminated",
                "proposition_id": "HYPOTHESIS",
                "decision_critical": True,
            }],
        )
        attach_candidate_dependencies(ledger, chunk)

        self.assertEqual(
            chunk.material_empirical_claims[0]["proposition_id"],
            "PROP:WORLD:E5",
        )
        self.assertNotEqual(chunk.adjudication_status, "CONTESTED_NO_LEANING")
        self.assertEqual(chunk.recommended_action, self.actions[0])
        self.assertFalse(any(
            CERTAIN_CONTRADICTION_NOTE in note
            for note in chunk.epistemic_binding_notes
        ))
        self.assertNotIn(
            "HYPOTHESIS",
            {ledger[value].epistemic_type for value in chunk.supporting_proposition_ids},
        )

    def test_agreeing_necessity_claim_does_not_quarantine(self):
        from global_workspace.epistemic_ledger import (
            CERTAIN_CONTRADICTION_NOTE,
            attach_candidate_dependencies,
        )

        ledger = self._world_ledger()
        chunk = self._chunk(
            "virtue",
            supporting_proposition_ids=["PROP:WORLD:E5"],
            material_empirical_claims=[{
                "claim": "only sealing the conduit can protect the stored reserve",
                "proposition_id": "HYPOTHESIS",
                "decision_critical": True,
            }],
        )
        attach_candidate_dependencies(ledger, chunk)

        self.assertNotEqual(chunk.adjudication_status, "CONTESTED_NO_LEANING")
        self.assertFalse(any(
            CERTAIN_CONTRADICTION_NOTE in note
            for note in chunk.epistemic_binding_notes
        ))

    def test_possible_row_is_not_rebound_as_settled(self):
        from global_workspace.epistemic_ledger import resolve_proposition

        ledger = self._world_ledger(possible_survive=True)
        survive = ledger["PROP:WORLD:E6"]
        self.assertIn("modality: POSSIBLE", survive.claim)
        self.assertEqual(resolve_proposition(ledger, "workers survive"), "")
        self.assertEqual(resolve_proposition(ledger, survive.claim), "PROP:WORLD:E6")
        self.assertEqual(
            resolve_proposition(ledger, "workers might survive"),
            "PROP:WORLD:E6",
        )

    def test_hypothesis_about_possible_row_is_not_certain_quarantine(self):
        from global_workspace.epistemic_ledger import (
            CERTAIN_CONTRADICTION_NOTE,
            HYPOTHESIS_OPEN_WORLD_CONFIDENCE_CAP,
            attach_candidate_dependencies,
        )

        ledger = self._world_ledger(possible_survive=True)
        chunk = self._chunk(
            "care",
            scores={self.actions[0]: 0.30, self.actions[1]: 0.70},
            recommended=self.actions[1],
            supporting_proposition_ids=["PROP:WORLD:E6"],
            material_empirical_claims=[{
                "claim": "the workers cannot self-rescue from the service bay",
                "proposition_id": "HYPOTHESIS",
                "decision_critical": True,
            }],
        )
        attach_candidate_dependencies(ledger, chunk)

        self.assertNotEqual(chunk.adjudication_status, "CONTESTED_NO_LEANING")
        self.assertEqual(chunk.recommended_action, self.actions[1])
        self.assertLessEqual(
            chunk.epistemic_confidence, HYPOTHESIS_OPEN_WORLD_CONFIDENCE_CAP,
        )
        self.assertFalse(any(
            CERTAIN_CONTRADICTION_NOTE in note
            for note in chunk.epistemic_binding_notes
        ))
        self.assertTrue(any(
            ledger[value].epistemic_type == "HYPOTHESIS"
            for value in chunk.decision_critical_proposition_ids
        ))

    def test_framework_derived_is_not_a_descriptive_hypothesis(self):
        from global_workspace.epistemic_ledger import (
            HYPOTHESIS_OPEN_WORLD_CONFIDENCE_CAP,
            attach_candidate_dependencies,
        )

        ledger = self._world_ledger()
        chunk = self._chunk(
            "deontological",
            material_empirical_claims=[{
                "claim": "sealing the conduit is doing harm to the workers",
                "proposition_id": "FRAMEWORK_DERIVED",
                "decision_critical": True,
            }],
        )
        attach_candidate_dependencies(ledger, chunk)

        bound = chunk.material_empirical_claims[0]["proposition_id"]
        record = ledger[bound]
        self.assertEqual(record.epistemic_type, "FRAMEWORK_DERIVED")
        self.assertEqual(record.proposition_type, "NORMATIVE")
        self.assertNotEqual(record.epistemic_status, "HYPOTHETICAL")
        self.assertGreater(chunk.epistemic_confidence, HYPOTHESIS_OPEN_WORLD_CONFIDENCE_CAP)
        self.assertEqual(chunk.recommended_action, self.actions[0])
        self.assertNotEqual(chunk.adjudication_status, "CONTESTED_NO_LEANING")

    def test_audit_reclassifies_normative_new_hypothesis(self):
        from global_workspace.epistemic_ledger import apply_side_premise_audit

        ledger = self._world_ledger()
        chunk = self._chunk("deontological")
        apply_side_premise_audit(ledger, [chunk], {
            "status": "FINDINGS",
            "findings": [{
                "specialist": "deontological",
                "claim": "sealing the conduit is doing harm to the workers",
                "binding": "NEW_HYPOTHESIS",
                "derived_from": ["PROP:WORLD:E1"],
                "decision_critical": True,
                "source_field": "rationale",
                "reason": "framework-native doing relation",
            }],
        })
        bound = chunk.side_premise_audit_findings[0]["proposition_id"]
        self.assertEqual(ledger[bound].epistemic_type, "FRAMEWORK_DERIVED")
        self.assertNotEqual(chunk.adjudication_status, "CONTESTED_NO_LEANING")

    def test_parse_accepts_framework_derived_premise(self):
        chunk = _candidate_from_data(
            "utilitarian", list(self.actions),
            {
                "scores": {"A0": 0.35, "A1": 0.65}, "r": "A1",
                "c": "IMMINENT_HARM", "u": "NONE",
                "w": "Prefer leaving the conduit open on admitted drowning",
                "j": "NONE", "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
                "ct": {
                    "A0": [{
                        "o": "twelve workers drown", "s": "workers",
                        "d": "HARM", "p": "CERTAIN", "m": "UNKNOWN",
                        "h": "immediate", "rv": "IRREVERSIBLE", "g": "STATED",
                    }],
                    "A1": [{
                        "o": "stored reserve becomes contaminated",
                        "s": "city residents",
                        "d": "HARM", "p": "CERTAIN", "m": "UNKNOWN",
                        "h": "lasting", "rv": "UNKNOWN", "g": "STATED",
                    }],
                },
                "cd": False, "cm": "NONE",
                "dr": (
                    "prefer leaving the conduit open because certain drowning "
                    "of the workers outranks certain contamination of the reserve"
                ),
                "ft": "NONE", "nt": "NONE",
                "ep": [{
                    "c": "sealing the conduit is doing harm to the workers",
                    "p": "FRAMEWORK_DERIVED",
                    "dc": True,
                }],
            },
            WorkspaceBroadcast(), "NONE", {},
            scenario_text=self.scenario,
        )
        self.assertEqual(
            chunk.material_empirical_claims[0]["proposition_id"],
            "FRAMEWORK_DERIVED",
        )

    def test_seeded_possible_injury_keeps_modality_and_health_dimension(self):
        from global_workspace.epistemic_ledger import resolve_proposition

        fixture = DeonticGraphRelationTests()
        graph, _, _ = fixture._graph(fixture._world(
            harm_modality="POSSIBLE",
            harm_likelihood=("a chance",),
        ))
        ledger = seed_proposition_ledger(graph)
        injured = ledger["PROP:WORLD:E1"]
        self.assertEqual(injured.modality, "POSSIBLE")
        self.assertIn("modality: POSSIBLE", injured.claim)
        self.assertNotEqual(
            resolve_proposition(ledger, "INJURED", preferred="PROP:WORLD:E1"),
            "PROP:WORLD:E1",
        )
        self.assertEqual(resolve_proposition(ledger, injured.claim), "PROP:WORLD:E1")
        self.assertEqual(ledger["PROP:WORLD:E4"].modality, "CERTAIN")
        self.assertEqual(resolve_proposition(ledger, "INJURED"), "PROP:WORLD:E4")
        health = [
            effect for effect in project_grounded_action_effects(graph)
            if effect.consequence_id.endswith(":E1")
        ]
        self.assertTrue(health)
        self.assertTrue(all(
            effect.dimension == "BASIC_SECURITY" for effect in health
        ))

    def test_foregone_duals_are_not_outcome_equivalence(self):
        from global_workspace.semantic_state import _action_signature

        graph = SemanticGraph()
        graph.add_node(SemanticNode(
            "A0", "ACTION", self.actions[0],
            attributes={"canonical_action_id": "A0"},
        ))
        graph.add_node(SemanticNode(
            "A1", "ACTION", self.actions[1],
            attributes={"canonical_action_id": "A1"},
        ))
        graph.add_node(SemanticNode(
            "A0:C", "CONSEQUENCE", "workers drown",
            attributes={"polarity": "ADVERSE", "directness": "DOWNSTREAM"},
        ))
        graph.add_node(SemanticNode(
            "A0:F", "CONSEQUENCE", "workers survive",
            attributes={"polarity": "FOREGONE", "directness": "FOREGONE"},
        ))
        graph.add_node(SemanticNode(
            "A1:C", "CONSEQUENCE", "workers survive",
            attributes={"polarity": "BENEFICIAL", "directness": "DOWNSTREAM"},
        ))
        graph.add_node(SemanticNode(
            "A1:F", "CONSEQUENCE", "workers drown",
            attributes={"polarity": "FOREGONE", "directness": "FOREGONE"},
        ))
        graph.add_edge(SemanticEdge("A0", "HAS_CONSEQUENCE", "A0:C"))
        graph.add_edge(SemanticEdge("A0", "HAS_CONSEQUENCE", "A0:F"))
        graph.add_edge(SemanticEdge("A1", "HAS_CONSEQUENCE", "A1:C"))
        graph.add_edge(SemanticEdge("A1", "HAS_CONSEQUENCE", "A1:F"))

        left = _action_signature(graph, "A0")
        right = _action_signature(graph, "A1")
        self.assertEqual(left["consequence_labels"], {"workers drown"})
        self.assertEqual(right["consequence_labels"], {"workers survive"})
        state = project_authoritative_semantic_state(graph, selected_action=self.actions[0])
        self.assertFalse(any(
            item.relation == "OUTCOME_EQUIVALENCE"
            for item in state.problem_shape_relations
        ))


class RawlsMapReconciliationTests(unittest.TestCase):
    """One explanatory line per action cannot encode a multi-position ledger."""

    actions = (
        "execute the covert override that erases the logs to restore heat",
        "preserve the audit logs and leave the heating system untouched",
    )
    action_ids = ("A0", "A1")
    action_map = {
        actions[0]: (
            "WORSENS: the innocent technician's basic liberty is compromised "
            "even though residents regain heat"
        ),
        actions[1]: (
            "PRESERVES: the technician's basic liberty and public "
            "accountability hold while residents lose heat"
        ),
    }
    positions = {
        "A0": {
            "e": "WORSENS", "d": "BASIC_LIBERTY", "ca": "A1", "b": "SCENARIO",
            "s": "innocent technician", "sk": "INDIVIDUAL",
            "bk": "PERSONAL_FREEDOM_INTEGRITY", "ir": "DIRECT_COERCIVE_RESTRICTION",
            "rs": "framed for sabotage by the erased record",
            "ad": [{
                "e": "IMPROVES", "d": "BASIC_INTEREST_SECURITY", "ca": "A1",
                "b": "SCENARIO", "s": "8 freezing residents", "sk": "GROUP",
                "bk": "NOT_APPLICABLE", "ir": "MATERIAL_PRECONDITION",
                "rs": "heat restored to the district",
            }],
        },
        "A1": {
            "e": "PRESERVES", "d": "BASIC_LIBERTY", "ca": "A0", "b": "SCENARIO",
            "s": "innocent technician", "sk": "INDIVIDUAL",
            "bk": "PERSONAL_FREEDOM_INTEGRITY", "ir": "DIRECT_COERCIVE_RESTRICTION",
            "rs": "the audit record keeps the technician exonerated",
            "ad": [{
                "e": "WORSENS", "d": "BASIC_INTEREST_SECURITY", "ca": "A0",
                "b": "SCENARIO", "s": "8 freezing residents", "sk": "GROUP",
                "bk": "NOT_APPLICABLE", "ir": "MATERIAL_PRECONDITION",
                "rs": "the district stays without heat",
            }],
        },
    }

    def _errors(self, **overrides: object) -> list[str]:
        return _construct_map_errors(
            "rawlsian",
            list(self.actions),
            dict(self.action_map),
            str(overrides.get("numerical_role", "SECONDARY")),
            "the worst-off subject is the innocent technician's basic liberty",
            action_ids=list(self.action_ids),
            recommended_action=str(
                overrides.get("recommended_action", self.actions[1])
            ),
            rationale=str(overrides.get("rationale", "Basic liberty outweighs lives")),
            supporting_data={"rp": overrides.get("rp", self.positions)},
        )

    def test_typed_liberty_priority_supplies_the_stated_difference(self):
        self.assertEqual(self._errors(), [])

    def test_prose_alone_cannot_justify_selecting_the_impaired_position(self):
        errors = self._errors(recommended_action=self.actions[0])

        self.assertIn(
            "Rawlsian preference lacks a stated difference in position or "
            "principle priority",
            errors,
        )

    def test_map_line_matching_one_of_several_positions_is_not_a_conflict(self):
        chunk = _candidate_from_data(
            "rawlsian",
            self.actions,
            {
                "scores": {"A0": 0.3, "A1": 0.7},
                "r": "A1", "c": "RIGHTS", "u": "NONE",
                "w": "basic liberty holds lexical priority over welfare gains",
                "j": "NONE", "e": "STATED_FACTS", "x": "NONE",
                "l": {
                    "A0": "Worsens the technician's basic liberty",
                    "A1": "Preserves the technician's basic liberty",
                },
                "t": "basic liberty versus interest security",
                "tf": "NONE",
                "dr": "prefer A1 while basic liberty stays lexically prior",
                "ft": "NONE",
                "nt": "If basic liberty lost lexical priority",
                "z": 0.7,
                "fm": {
                    "A0": self.action_map[self.actions[0]],
                    "A1": self.action_map[self.actions[1]],
                },
                "nr": "SECONDARY",
                "np": "the worst-off subject is the innocent technician",
                "rb": "LEXICAL_BASIC_LIBERTY",
                "rbc": "the public action directly burdens the technician's protected freedom",
                "lpj": "the direct liberty burden is considered before residents' material security",
                "lc": {"A0": "INFRINGED", "A1": "SATISFIED"},
                "rp": self.positions,
            },
            WorkspaceBroadcast(constraint="OPEN_DELIBERATION"),
            "A1",
            {},
        )

        self.assertEqual(chunk.framework_validation_errors, [])
        self.assertEqual(
            len(chunk.rawls_position_proposal["positions"]), 4,
            "both actions must contribute a primary and an additional position",
        )

    def test_map_line_contradicting_every_position_still_fails(self):
        inverted = {
            "A0": {**self.positions["A0"], "e": "IMPROVES", "ad": []},
            "A1": {**self.positions["A1"], "e": "PRESERVES", "ad": []},
        }
        chunk = _candidate_from_data(
            "rawlsian",
            self.actions,
            {
                "scores": {"A0": 0.3, "A1": 0.7},
                "r": "A1", "c": "RIGHTS", "u": "NONE",
                "w": "basic liberty holds lexical priority over welfare gains",
                "j": "NONE", "e": "STATED_FACTS", "x": "NONE",
                "l": {
                    "A0": "Worsens the technician's basic liberty",
                    "A1": "Preserves the technician's basic liberty",
                },
                "t": "basic liberty versus interest security",
                "tf": "NONE",
                "dr": "prefer A1 while basic liberty stays lexically prior",
                "ft": "NONE",
                "nt": "If basic liberty lost lexical priority",
                "z": 0.7,
                "fm": {
                    "A0": self.action_map[self.actions[0]],
                    "A1": self.action_map[self.actions[1]],
                },
                "nr": "SECONDARY",
                "np": "the worst-off subject is the innocent technician",
                "rb": "LEXICAL_BASIC_LIBERTY",
                "rbc": "the public action directly burdens the technician's protected freedom",
                "lpj": "the direct liberty burden is considered before residents' material security",
                "lc": {"A0": "INFRINGED", "A1": "SATISFIED"},
                "rp": inverted,
            },
            WorkspaceBroadcast(constraint="OPEN_DELIBERATION"),
            "A1",
            {},
        )

        self.assertIn(
            "Rawlsian position for A0 conflicts with its framework map",
            chunk.framework_validation_errors,
        )
        self.assertNotIn(
            "Rawlsian position for A1 conflicts with its framework map",
            chunk.framework_validation_errors,
        )


class VirtueRoleRedescriptionTests(unittest.TestCase):
    """Renaming the actor's role is not a change of character verdict."""

    actions = (
        "execute the covert override that erases the logs to restore heat",
        "preserve the audit logs and leave the heating system untouched",
    )

    def _specialist(self) -> CompactLocalSpecialist:
        specialist = CompactLocalSpecialist("virtue", llm=None)
        specialist.previous_framework_state = {
            "ranking_basis": "PRACTICAL_WISDOM",
            "assessments": [
                {
                    "action_id": "A0", "verdict": "UNDERMINES",
                    "actor_role": "public steward",
                },
                {
                    "action_id": "A1", "verdict": "EXEMPLIFIES",
                    "actor_role": "public steward",
                },
            ],
        }
        return specialist

    def _candidate(self, assessments: list[dict[str, object]]) -> CandidateChunk:
        return CandidateChunk(
            specialist="virtue", constraint="CHARACTER",
            action_scores={self.actions[0]: 0.3, self.actions[1]: 0.7},
            surprise=0.1, friction=0.1, confidence=0.6,
            recommended_action=self.actions[1],
            rationale="Honesty and practical wisdom favour keeping the record.",
            virtue_character_proposal={
                "ranking_basis": "PRACTICAL_WISDOM",
                "assessments": assessments,
            },
        )

    def test_role_relabel_with_stable_verdicts_is_a_refinement(self):
        specialist = self._specialist()
        candidate = self._candidate([
            {"action_id": "A0", "verdict": "UNDERMINES", "actor_role": "chief engineer"},
            {"action_id": "A1", "verdict": "EXEMPLIFIES", "actor_role": "chief engineer"},
        ])

        specialist._audit_framework_state_change(
            candidate, WorkspaceBroadcast(constraint="OPEN_DELIBERATION"),
        )

        self.assertTrue(candidate.framework_constraint_retained)
        self.assertEqual(candidate.framework_retention_status, "REFINEMENT")
        self.assertTrue(candidate.framework_insights)
        self.assertTrue(any(
            "chief engineer" in item["proposition"]
            for item in candidate.framework_insights
        ))

    def test_verdict_change_still_requires_a_framework_reason(self):
        specialist = self._specialist()
        candidate = self._candidate([
            {"action_id": "A0", "verdict": "EXEMPLIFIES", "actor_role": "public steward"},
            {"action_id": "A1", "verdict": "UNDERMINES", "actor_role": "public steward"},
        ])

        specialist._audit_framework_state_change(
            candidate, WorkspaceBroadcast(constraint="OPEN_DELIBERATION"),
        )

        self.assertNotEqual(candidate.framework_retention_status, "REFINEMENT")


class ResolvedQuestionLedgerTests(unittest.TestCase):
    """Settled audit answers are graph state, and reopen when evidence moves."""

    actions = (
        "execute the covert override that erases the logs to restore heat",
        "preserve the audit logs and leave the heating system untouched",
    )
    question_key = "QUESTION:abc123"
    proposition = (
        "Whether the relative magnitude of the technician's liberty loss "
        "outweighs the residents' survival"
    )

    scenario = (
        "You can execute a covert override to save 8 freezing residents, but "
        "doing so overwrites system logs, framing an innocent technician for "
        "sabotage. Alternatively, you can preserve the audit logs to exonerate "
        "the innocent technician while the residents continue to freeze."
    )

    def _store(self) -> SemanticGraphStore:
        clauses = segment_scenario_clauses(self.scenario)
        return SemanticGraphStore(compile_scenario_graph(
            self.scenario, list(self.actions), {
                "A0": {"clauses": [clauses[0]]},
                "A1": {"clauses": [clauses[1]]},
            },
        ))

    def _responder(
        self, specialist: str, *, participation: str, effect: str,
    ) -> CandidateChunk:
        return CandidateChunk(
            specialist=specialist, constraint="OPEN_DELIBERATION",
            action_scores={self.actions[0]: 0.4, self.actions[1]: 0.6},
            surprise=0.1, friction=0.1, confidence=0.6,
            recommended_action=self.actions[1],
            rationale="The audited issue does not move the ranking.",
            audit_participation=participation,
            audit_internal_effect=effect,
        )

    def _resolve(self, store: SemanticGraphStore, candidates: list[CandidateChunk]):
        return resolve_audited_question(
            candidates,
            question_key=self.question_key,
            proposition=self.proposition,
            cycle=3,
            grounded_in=["A0", "A1"],
            graph=store.graph,
        )

    def test_agreed_answer_settles_the_question_and_prunes_the_audit_slot(self):
        store = self._store()
        resolution = self._resolve(store, [
            self._responder("utilitarian", participation="RELEVANT", effect="NO_CHANGE"),
            self._responder("care", participation="IRRELEVANT", effect="NO_CHANGE"),
        ])

        self.assertIsNotNone(resolution)
        self.assertEqual(resolution.resolution, "NO_CHANGE")
        record = commit_question_resolution(store, resolution)
        self.assertEqual(record.status, "COMMITTED")
        self.assertEqual(settled_question_keys(store.graph), {self.question_key})

        signals, question, payload = _problem_state_audit_probe(
            {
                "audit_candidates": [{
                    "issue_id": self.question_key,
                    "proposition": self.proposition,
                    "grounded_in": ["A0", "A1"],
                    "raised_by": ["utilitarian"],
                    "category": "RESOLVE_NORMATIVE_TENSION",
                    "status": "PERSISTENT_UNRESOLVED",
                }],
                "agent_positions": [],
            },
            self.actions[1],
        )
        self.assertTrue(signals and question and payload)

        state = build_deliberative_problem_state(
            4, list(self.actions), [], self.actions[1],
            self._responder("care", participation="RELEVANT", effect="NO_CHANGE"),
            {"unresolved_questions": []},
            store.graph,
        ).to_dict()
        self.assertEqual(
            [item["question_key"] for item in state["resolved_questions"]],
            [self.question_key],
        )
        self.assertEqual(state["resolved_questions"][0]["status"], "SETTLED")

    def test_settled_question_leaves_the_audit_slot_but_stays_unresolved(self):
        store = self._store()
        asker = CandidateChunk(
            specialist="utilitarian", constraint="AGGREGATE_WELFARE",
            action_scores={self.actions[0]: 0.5, self.actions[1]: 0.5},
            surprise=0.2, friction=0.2, confidence=0.5,
            recommended_action=self.actions[1],
            rationale="The comparison remains open.",
            unresolved="VERIFY_FACTS",
            unsupported_assumption=(
                "whether the innocent technician being framed outweighs the "
                "freezing residents losing heat"
            ),
        )

        open_state = build_deliberative_problem_state(
            3, list(self.actions), [asker], self.actions[1], asker,
            None, store.graph, self.scenario,
        ).to_dict()
        raised = open_state["unresolved_questions"][0]
        self.assertIn(
            raised["question_key"],
            {item["issue_id"] for item in open_state["audit_candidates"]},
        )

        resolution = resolve_audited_question(
            [
                self._responder("utilitarian", participation="RELEVANT", effect="NO_CHANGE"),
                self._responder("care", participation="RELEVANT", effect="NO_CHANGE"),
            ],
            question_key=raised["question_key"],
            proposition=raised["question"],
            cycle=3,
            grounded_in=list(raised["grounded_in"]),
            graph=store.graph,
        )
        self.assertIsNotNone(resolution)
        self.assertEqual(
            commit_question_resolution(store, resolution).status, "COMMITTED",
        )

        pruned = build_deliberative_problem_state(
            4, list(self.actions), [asker], self.actions[1], asker,
            open_state, store.graph, self.scenario,
        ).to_dict()

        self.assertNotIn(
            raised["question_key"],
            {item["issue_id"] for item in pruned["audit_candidates"]},
        )
        settled = next(
            item for item in pruned["unresolved_questions"]
            if item["question_key"] == raised["question_key"]
        )
        self.assertEqual(settled["resolution_status"], "SETTLED")
        self.assertEqual(settled["recorded_resolution"], "NO_CHANGE")
        self.assertIn("VERIFY_FACTS", pruned["unresolved_categories"])

    def test_disagreement_leaves_the_question_open(self):
        store = self._store()

        contested = self._resolve(store, [
            self._responder("utilitarian", participation="RELEVANT", effect="NO_CHANGE"),
            self._responder("care", participation="RELEVANT", effect="REVERSES"),
        ])
        unresolved = self._resolve(store, [
            self._responder("utilitarian", participation="RELEVANT", effect="NO_CHANGE"),
            self._responder("care", participation="CONTESTED", effect="NO_CHANGE"),
        ])
        lone = self._resolve(store, [
            self._responder("utilitarian", participation="RELEVANT", effect="NO_CHANGE"),
            self._responder("care", participation="NOT_TESTED", effect="UNRESOLVED"),
        ])

        self.assertIsNone(contested)
        self.assertIsNone(unresolved)
        self.assertIsNone(lone)

    def test_new_evidence_on_the_grounding_reopens_the_question(self):
        store = self._store()
        resolution = self._resolve(store, [
            self._responder("utilitarian", participation="RELEVANT", effect="NO_CHANGE"),
            self._responder("care", participation="RELEVANT", effect="NO_CHANGE"),
        ])
        commit_question_resolution(store, resolution)
        self.assertEqual(settled_question_keys(store.graph), {self.question_key})

        store.graph.add_node(SemanticNode(
            "LATE_CONSEQUENCE", "CONSEQUENCE", "district-wide outage",
            ("cycle:5", "late evidence"), {"polarity": "ADVERSE"},
        ))
        store.graph.add_edge(SemanticEdge(
            "A0", "HAS_CONSEQUENCE", "LATE_CONSEQUENCE",
            provenance=("cycle:5", "late evidence"),
        ))

        index = question_resolution_index(store.graph)
        self.assertEqual(index[self.question_key]["status"], "REOPENED")
        self.assertTrue(index[self.question_key]["reopened_by_evidence_change"])
        self.assertEqual(settled_question_keys(store.graph), frozenset())

    def test_grounding_outside_the_current_run_graph_is_rejected(self):
        store = self._store()
        responders = [
            self._responder("utilitarian", participation="RELEVANT", effect="NO_CHANGE"),
            self._responder("care", participation="RELEVANT", effect="NO_CHANGE"),
        ]

        unresolvable = resolve_audited_question(
            responders,
            question_key=self.question_key,
            proposition=self.proposition,
            cycle=3,
            grounded_in=["GHOST_NODE"],
            graph=store.graph,
        )
        record = commit_question_resolution(store, QuestionResolution(
            question_key=self.question_key,
            proposition=self.proposition,
            resolution="NO_CHANGE",
            cycle=3,
            grounded_in=("GHOST_NODE",),
            responders=("care", "utilitarian"),
            evidence_signature="stale",
        ))

        self.assertIsNone(unresolvable)
        self.assertEqual(record.status, "REJECTED")
        self.assertTrue(record.previous_state_preserved)
        self.assertEqual(question_resolution_index(store.graph), {})


class TypedLedgerGroundingTests(unittest.TestCase):
    """The ledger's own field names must satisfy the framework construct gate."""

    actions = ("publish the audit and end the siphon", "suppress the audit")
    action_ids = ("A0", "A1")
    # The delegate writes the ledger's canonical dimension token, which the
    # prose marker cannot match because an underscore is not whitespace.
    action_map = {
        actions[0]: "IMPROVES: the district's BASIC_LIBERTY and survival",
        actions[1]: "WORSENS: the district's BASIC_LIBERTY and survival",
    }
    positions = {
        "A0": {
            "e": "IMPROVES", "d": "BASIC_LIBERTY", "ca": "A1", "b": "SCENARIO",
            "s": "impoverished district residents", "sk": "GROUP",
            "rs": "the covert siphon stops",
        },
        "A1": {
            "e": "WORSENS", "d": "BASIC_LIBERTY", "ca": "A0", "b": "SCENARIO",
            "s": "impoverished district residents", "sk": "GROUP",
            "rs": "the covert siphon continues",
        },
    }

    def _errors(self, rp: object) -> list[str]:
        return _construct_map_errors(
            "rawlsian",
            list(self.actions),
            dict(self.action_map),
            "SECONDARY",
            "the worst-off subjects are the district's residents",
            action_ids=list(self.action_ids),
            recommended_action=self.actions[0],
            rationale="Basic liberty for the district holds lexical priority",
            supporting_data={"rp": rp},
        )

    def test_ledger_dimension_token_grounds_the_map_line(self):
        self.assertNotIn(
            f"framework map for {self.actions[0]} lacks framework-specific grounds",
            self._errors(self.positions),
        )

    def test_ledger_about_something_else_cannot_ground_the_map_line(self):
        unrelated = {
            "A0": {
                "e": "IMPROVES", "d": "COST", "ca": "A1", "b": "SCENARIO",
                "s": "the operating budget", "sk": "INSTITUTION",
                "rs": "cheaper to run",
            },
            "A1": {
                "e": "WORSENS", "d": "COST", "ca": "A0", "b": "SCENARIO",
                "s": "the operating budget", "sk": "INSTITUTION",
                "rs": "more expensive to run",
            },
        }
        errors = self._errors(unrelated)

        self.assertIn(
            f"framework map for {self.actions[0]} lacks framework-specific grounds",
            errors,
        )

    def test_absent_ledger_leaves_the_prose_gate_in_force(self):
        self.assertIn(
            f"framework map for {self.actions[0]} lacks framework-specific grounds",
            self._errors({}),
        )


class OpeningProblemFrameTests(unittest.TestCase):
    """Cycle one receives the shared world without receiving a position."""

    scenario = (
        "You can execute a covert override to save 8 freezing residents, but "
        "doing so overwrites system logs, framing an innocent technician for "
        "sabotage. Alternatively, you can preserve the audit logs."
    )
    actions = ("execute the covert override", "preserve the audit logs")

    def test_frame_carries_vocabulary_and_no_salient_position(self):
        graph = compile_scenario_graph(self.scenario, list(self.actions), {})
        frame = opening_problem_state(list(self.actions), self.scenario, graph)

        self.assertEqual(frame["cycle"], 0)
        self.assertEqual(frame["state_role"], "OPENING_PROBLEM_FRAME")
        self.assertEqual(
            [item["action_id"] for item in frame["live_actions"]], ["A0", "A1"],
        )
        self.assertTrue(frame["scenario_clauses"])
        self.assertTrue(all(
            item["clause_id"].startswith("C") for item in frame["scenario_clauses"]
        ))
        identities = {item["node_id"] for item in frame["grounded_identities"]}
        self.assertTrue({"A0", "A1"} <= identities)
        self.assertIn(
            "ACTION", {item["kind"] for item in frame["grounded_identities"]},
        )
        self.assertEqual(frame["salient_position"], {})
        self.assertEqual(frame["agent_positions"], [])
        self.assertEqual(frame["current_plurality"], "")
        self.assertEqual(frame["committed_world"], {})

    def test_engine_seeds_the_frame_and_records_no_influence_for_it(self):
        received: list[dict[str, object]] = []

        class Recording(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                received.append(dict(broadcast.problem_state or {}))
                return super().evaluate(scenario, actions, broadcast)

        result = WorkspaceEngine(
            [Recording("care", self.actions[1], "CARE"),
             Recording("duty", self.actions[1], "DUTY")],
            WorkspaceConfig(
                max_cycles=1, stable_cycles_required=3,
                enable_consensus_audit=False, enable_reversal_audit=False,
                enable_synthesis=False, enable_planning=False,
            ),
        ).run(self.scenario, list(self.actions))

        self.assertTrue(received)
        self.assertEqual(received[0]["state_role"], "OPENING_PROBLEM_FRAME")
        self.assertTrue(received[0]["scenario_clauses"])
        # A positionless frame is not a broadcast consideration, so it must not
        # produce influence records attributing uptake to nobody.
        self.assertEqual(result.broadcast_influence_records, [])


class UnresolvableAuditTests(unittest.TestCase):
    """An audit nobody can answer records the evidence it would take."""

    actions = (
        "execute the covert override that erases the logs to restore heat",
        "preserve the audit logs and leave the heating system untouched",
    )
    question_key = "QUESTION:unresolvable01"
    proposition = "Whether the siphon will stop within a month"
    scenario = (
        "You can execute a covert override to save 8 freezing residents, but "
        "doing so overwrites system logs, framing an innocent technician for "
        "sabotage. Alternatively, you can preserve the audit logs."
    )

    def _store(self) -> SemanticGraphStore:
        return SemanticGraphStore(compile_scenario_graph(
            self.scenario, list(self.actions), {},
        ))

    def _responder(
        self,
        specialist: str,
        *,
        participation: str,
        effect: str,
        explanation: str = "",
    ) -> CandidateChunk:
        return CandidateChunk(
            specialist=specialist, constraint="OPEN_DELIBERATION",
            action_scores={self.actions[0]: 0.4, self.actions[1]: 0.6},
            surprise=0.1, friction=0.1, confidence=0.6,
            recommended_action=self.actions[1],
            rationale="The audited issue cannot be settled here.",
            audit_participation=participation,
            audit_internal_effect=effect,
            audit_framework_explanation=explanation,
        )

    def _resolve(self, store: SemanticGraphStore, candidates: list[CandidateChunk]):
        return resolve_audited_question(
            candidates,
            question_key=self.question_key,
            proposition=self.proposition,
            cycle=3,
            grounded_in=["A0", "A1"],
            graph=store.graph,
        )

    def test_shared_evidence_gap_records_a_named_requirement(self):
        store = self._store()
        resolution = self._resolve(store, [
            self._responder(
                "utilitarian", participation="REVERSAL_RELEVANT",
                effect="UNRESOLVED",
                explanation="If the siphon stops soon the advantage disappears",
            ),
            self._responder(
                "virtue", participation="REVERSAL_RELEVANT",
                effect="UNRESOLVED",
                explanation="If the siphon stops soon the justice gain shrinks",
            ),
            self._responder(
                "care", participation="NOT_TESTED", effect="UNRESOLVED",
            ),
        ])

        self.assertIsNotNone(resolution)
        self.assertEqual(resolution.resolution, "UNRESOLVABLE_AT_CURRENT_EVIDENCE")
        self.assertIn("siphon stops soon", resolution.evidence_requirement)
        self.assertEqual(resolution.responders, ("utilitarian", "virtue"))
        self.assertEqual(
            commit_question_resolution(store, resolution).status, "COMMITTED",
        )
        self.assertEqual(settled_question_keys(store.graph), {self.question_key})
        recorded = question_resolution_index(store.graph)[self.question_key]
        self.assertIn("siphon stops soon", recorded["evidence_requirement"])

    def test_frameworks_reporting_different_effects_settle_nothing(self):
        store = self._store()

        self.assertIsNone(self._resolve(store, [
            self._responder("utilitarian", participation="RELEVANT", effect="NO_CHANGE"),
            self._responder("virtue", participation="RELEVANT", effect="UNRESOLVED"),
        ]))

    def test_contested_participation_still_records_nothing(self):
        store = self._store()

        self.assertIsNone(self._resolve(store, [
            self._responder("utilitarian", participation="CONTESTED", effect="UNRESOLVED"),
            self._responder("virtue", participation="REVERSAL_RELEVANT", effect="UNRESOLVED"),
        ]))

    def test_new_evidence_reopens_an_unresolvable_question(self):
        store = self._store()
        resolution = self._resolve(store, [
            self._responder(
                "utilitarian", participation="REVERSAL_RELEVANT",
                effect="UNRESOLVED", explanation="duration of the siphon is unknown",
            ),
            self._responder(
                "virtue", participation="REVERSAL_RELEVANT",
                effect="UNRESOLVED", explanation="duration of the siphon is unknown",
            ),
        ])
        commit_question_resolution(store, resolution)
        self.assertEqual(settled_question_keys(store.graph), {self.question_key})

        delta = SemanticGraph()
        delta.add_node(SemanticNode(
            "OBS:siphon_schedule", "OBSERVATION",
            "the siphon is scheduled to end in three weeks",
            ("cycle:4", "test"), {"support": "STATED"},
        ))
        delta.add_edge(SemanticEdge(
            "A0", "SUPPORTED_BY", "OBS:siphon_schedule",
            justification="new schedule evidence", provenance=("cycle:4", "test"),
        ))
        store.graph = merge_graphs([store.graph, delta])

        self.assertEqual(settled_question_keys(store.graph), frozenset())
        self.assertEqual(
            question_resolution_index(store.graph)[self.question_key]["status"],
            "REOPENED",
        )


class AuditCandidateGroundingPriorityTests(unittest.TestCase):
    """Ungrounded inquiry stays testable without displacing grounded inquiry."""

    scenario = (
        "You can execute a covert override to save 8 freezing residents, but "
        "doing so overwrites system logs, framing an innocent technician for "
        "sabotage. Alternatively, you can preserve the audit logs."
    )
    actions = (
        "execute the covert override that erases the logs to restore heat",
        "preserve the audit logs and leave the heating system untouched",
    )

    def _asker(self, specialist: str, assumption: str) -> CandidateChunk:
        return CandidateChunk(
            specialist=specialist, constraint="UNCERTAINTY",
            action_scores={self.actions[0]: 0.5, self.actions[1]: 0.5},
            surprise=0.2, friction=0.2, confidence=0.5,
            recommended_action=self.actions[1],
            rationale="The comparison remains open.",
            unresolved="VERIFY_FACTS",
            unsupported_assumption=assumption,
        )

    def _state(self, candidates: list[CandidateChunk]) -> dict[str, object]:
        graph = compile_scenario_graph(self.scenario, list(self.actions), {})
        return build_deliberative_problem_state(
            2, list(self.actions), candidates, self.actions[1], candidates[0],
            None, graph, self.scenario,
        ).to_dict()

    def test_ungrounded_question_is_offered_when_nothing_grounded_competes(self):
        state = self._state([self._asker("care", "whether ongoing support holds")])
        candidates = state["audit_candidates"]

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["grounding_status"], "UNGROUNDED")
        # It is anchored to the action nodes so its answer stays recordable.
        self.assertEqual(candidates[0]["grounded_in"], ["A0", "A1"])

    def test_grounded_question_displaces_every_ungrounded_one(self):
        state = self._state([
            self._asker("care", "whether ongoing support holds"),
            self._asker(
                "utilitarian",
                "whether the freezing residents outweigh the innocent technician",
            ),
        ])
        candidates = state["audit_candidates"]

        self.assertTrue(candidates)
        self.assertEqual(
            {item["grounding_status"] for item in candidates}, {"CLAUSE_GROUNDED"},
        )
        self.assertEqual(
            {item["raised_by"][0] for item in candidates}, {"utilitarian"},
        )

    def test_consensus_audit_cannot_focus_an_ungrounded_question(self):
        state = self._state([self._asker("care", "whether ongoing support holds")])

        _signals, _question, grounded_only = _problem_state_audit_probe(
            state, self.actions[1], require_clause_grounding=True,
        )
        _signals, _question, permissive = _problem_state_audit_probe(
            state, self.actions[1],
        )

        self.assertEqual(grounded_only, {})
        self.assertTrue(permissive)
        self.assertEqual(permissive["grounding_status"], "UNGROUNDED")


class KantianAuthoritySeparationTests(unittest.TestCase):
    """Provisional leanings may interrupt; they may not govern."""

    def _contested_rival(self) -> dict[str, object]:
        return {
            "canonical_action_id": "A1",
            "protected_party": "impoverished district residents",
            "norm": "do not treat humans merely as means",
            "competing_protected_party": "icu patients",
            "competing_norm": "duty to rescue ICU patients today",
            "coercive_actor": "grid director",
            "coerced_party": "impoverished residents",
            "resolution_status": "CONTESTED",
            "calibration_errors": [
                "coercion claim lacks actor-to-mechanism-to-restriction-to-party grounding",
                "resolved adjudication rests on unsupported decisive premises",
            ],
            "derivation": "RESPECT_PERSONS",
            "priority_rule": "perfect duties are non-negotiable",
        }

    def _resolved_primary(self) -> dict[str, object]:
        return {
            "canonical_action_id": "A0",
            "protected_party": "impoverished district residents",
            "norm": "do not treat humans merely as means",
            "competing_protected_party": "icu patients",
            "competing_norm": "duty to avoid immediate harm to ICU patients",
            "resolution_status": "RESOLVED",
            "calibration_errors": [],
            "derivation": "RESPECT_PERSONS",
            "priority_rule": "perfect duties override conflicting beneficence",
        }

    def test_rival_contestation_yields_provisional_leaning_not_governing_rule(self):
        from global_workspace.deontology_ledger import classify_deontological_authority

        profile = classify_deontological_authority(
            self._resolved_primary(),
            [self._contested_rival()],
            recommended_action="publish the audit",
            preference_strength=0.4,
        )

        self.assertEqual(profile.adjudication_status, "PROVISIONAL_LEANING")
        self.assertEqual(profile.broadcast_authority, "INVESTIGATIVE")
        self.assertFalse(profile.governing_eligible)
        self.assertLess(profile.policy_weight_factor, 1.0)
        self.assertGreater(profile.policy_weight_factor, 0.0)
        self.assertIn("UNRESOLVED DUTY CONFLICT", profile.investigative_claim)
        self.assertIn("publish the audit", profile.investigative_claim)
        self.assertNotIn("perfect duties override", profile.decision_rule.casefold())
        self.assertNotIn("perfect duties override", profile.investigative_claim.casefold())

    def test_flat_preference_yields_contested_no_leaning(self):
        from global_workspace.deontology_ledger import classify_deontological_authority

        profile = classify_deontological_authority(
            self._resolved_primary(),
            [self._contested_rival()],
            recommended_action="publish the audit",
            preference_strength=0.05,
        )

        self.assertEqual(profile.adjudication_status, "CONTESTED_NO_LEANING")
        self.assertEqual(profile.policy_weight_factor, 0.0)
        self.assertFalse(profile.governing_eligible)

    def test_fully_resolved_comparative_adjudication_remains_governing(self):
        from global_workspace.deontology_ledger import classify_deontological_authority

        clean_rival = {
            **self._contested_rival(),
            "resolution_status": "RESOLVED",
            "calibration_errors": [],
        }
        profile = classify_deontological_authority(
            self._resolved_primary(),
            [clean_rival],
            recommended_action="publish the audit",
            preference_strength=0.4,
        )

        self.assertEqual(profile.adjudication_status, "SUPPORTS")
        self.assertEqual(profile.broadcast_authority, "GOVERNING_CANDIDATE")
        self.assertTrue(profile.governing_eligible)
        self.assertEqual(profile.policy_weight_factor, 1.0)
        self.assertIn("perfect duties override", profile.decision_rule.casefold())

    def test_provisional_kant_cannot_supply_governing_rule_when_rawls_can(self):
        actions = ["publish the audit", "suppress the audit"]
        kant = CandidateChunk(
            specialist="deontological",
            constraint="DUTY",
            action_scores={actions[0]: 0.7, actions[1]: 0.3},
            surprise=0.8,
            friction=0.7,
            confidence=0.55,
            recommended_action=actions[0],
            decision_rule="Treat the duty priority as unestablished until coercion is grounded",
            rationale="Comparative Kantian judgment remains contested.",
            adjudication_status="PROVISIONAL_LEANING",
            broadcast_authority="INVESTIGATIVE",
            governing_eligible=False,
            policy_weight_factor=0.45,
            investigative_claim=(
                "UNRESOLVED DUTY CONFLICT: district vs ICU. Current reasoning "
                "leans publish the audit."
            ),
        )
        rawls = CandidateChunk(
            specialist="rawlsian",
            constraint="FAIRNESS",
            action_scores={actions[0]: 0.75, actions[1]: 0.25},
            surprise=0.4,
            friction=0.4,
            confidence=0.72,
            recommended_action=actions[0],
            decision_rule="A0 ends a systematic burden on the least advantaged",
            rationale="Least-advantaged class bears ongoing lethal deprivation.",
            adjudication_status="SUPPORTS",
            broadcast_authority="GOVERNING_CANDIDATE",
            governing_eligible=True,
            policy_weight_factor=1.0,
        )

        governing = WorkspaceEngine._select_governing_candidate(
            [kant, rawls], actions[0], preferred=kant,
        )
        self.assertIs(governing, rawls)
        self.assertNotIn("perfect duties", (governing.decision_rule or "").casefold())

        policy = WorkspaceEngine._policy([kant, rawls], actions)
        # Attenuated Kant still contributes, but Rawls should dominate plurality.
        self.assertGreater(policy[actions[0]], policy[actions[1]])

        # Investigative Kant can still outrank Rawls on salience.
        broadcast = WorkspaceBroadcast(constraint="OPEN_DELIBERATION", urgency=0.5)
        engine = WorkspaceEngine(
            specialists=[FixedSpecialist("rawlsian", actions[0], "FAIRNESS")],
            config=WorkspaceConfig(investigative_attention_weight=0.40),
        )
        kant.tension_engagement = 0.7
        kant.unresolved = "RESOLVE_NORMATIVE_TENSION"
        kant.salience = engine._salience(kant, broadcast, {}, 1.0)
        rawls.salience = engine._salience(rawls, broadcast, {"FAIRNESS": 1}, 0.0)
        self.assertGreater(kant.salience, rawls.salience)

    def test_presentation_uses_governing_eligible_rule_not_investigative_winner(self):
        actions = ["publish the audit", "suppress the audit"]
        data = {
            "actions": actions,
            "selected_action": actions[0],
            "current_plurality": actions[0],
            "judgment_status": "GOVERNED_RECOMMENDATION",
            "governing_justification_status": "ADMISSIBLE",
            "confidence": 0.7,
            "epistemic_confidence": 0.65,
            "halted_by": "convergence",
            "moral_residue": ["DUTY"],
            "compressed_rule": "When fairness is salient, prefer publish the audit.",
            "reopen_conditions": [],
            "cycles": [{
                "cycle": 1,
                "policy": {actions[0]: 0.7, actions[1]: 0.3},
                "policy_leader": actions[0],
                "broadcast_focus": {
                    "specialist": "deontological",
                    "constraint": "DUTY",
                    "schema_valid": True,
                    "recommended_action": actions[0],
                    "decision_rule": "Treat the duty priority as unestablished",
                    "rationale": "Kantian conflict open.",
                    "assumption_status": "NORMATIVELY_CONTESTED",
                    "adjudication_status": "PROVISIONAL_LEANING",
                    "broadcast_authority": "INVESTIGATIVE",
                    "governing_eligible": False,
                    "investigative_claim": (
                        "UNRESOLVED DUTY CONFLICT: entrusted lethal withdrawal "
                        "versus covert institutional sacrifice."
                    ),
                    "action_scores": {actions[0]: 0.7, actions[1]: 0.3},
                },
                "governing_claim": {
                    "specialist": "rawlsian",
                    "constraint": "FAIRNESS",
                    "schema_valid": True,
                    "recommended_action": actions[0],
                    "decision_rule": "A0 ends a systematic burden on the least advantaged",
                    "rationale": "Least-advantaged class bears ongoing lethal deprivation.",
                    "assumption_status": "DIRECT",
                    "adjudication_status": "SUPPORTS",
                    "broadcast_authority": "GOVERNING_CANDIDATE",
                    "governing_eligible": True,
                    "action_scores": {actions[0]: 0.75, actions[1]: 0.25},
                },
                "winner": {
                    "specialist": "deontological",
                    "constraint": "DUTY",
                    "schema_valid": True,
                    "recommended_action": actions[0],
                    "decision_rule": "Treat the duty priority as unestablished",
                    "rationale": "Kantian conflict open.",
                    "assumption_status": "NORMATIVELY_CONTESTED",
                    "adjudication_status": "PROVISIONAL_LEANING",
                    "broadcast_authority": "INVESTIGATIVE",
                    "governing_eligible": False,
                    "investigative_claim": (
                        "UNRESOLVED DUTY CONFLICT: entrusted lethal withdrawal "
                        "versus covert institutional sacrifice."
                    ),
                    "action_scores": {actions[0]: 0.7, actions[1]: 0.3},
                },
                "candidates": [
                    {
                        "specialist": "deontological",
                        "constraint": "DUTY",
                        "schema_valid": True,
                        "recommended_action": actions[0],
                        "decision_rule": "Treat the duty priority as unestablished",
                        "rationale": "Kantian conflict open.",
                        "assumption_status": "NORMATIVELY_CONTESTED",
                        "adjudication_status": "PROVISIONAL_LEANING",
                        "broadcast_authority": "INVESTIGATIVE",
                        "governing_eligible": False,
                        "investigative_claim": (
                            "UNRESOLVED DUTY CONFLICT: entrusted lethal withdrawal "
                            "versus covert institutional sacrifice."
                        ),
                        "action_scores": {actions[0]: 0.7, actions[1]: 0.3},
                    },
                    {
                        "specialist": "rawlsian",
                        "constraint": "FAIRNESS",
                        "schema_valid": True,
                        "recommended_action": actions[0],
                        "decision_rule": "A0 ends a systematic burden on the least advantaged",
                        "rationale": "Least-advantaged class bears ongoing lethal deprivation.",
                        "assumption_status": "DIRECT",
                        "adjudication_status": "SUPPORTS",
                        "broadcast_authority": "GOVERNING_CANDIDATE",
                        "governing_eligible": True,
                        "action_scores": {actions[0]: 0.75, actions[1]: 0.25},
                    },
                ],
                "dissent": None,
            }],
        }

        text = render_public_judgment(data)
        self.assertIn("## Recommendation", text)
        self.assertIn("## Why the Parliament currently favors this action", text)
        self.assertIn("A0 ends a systematic burden on the least advantaged", text)
        self.assertIn("## Most important unresolved issue", text)
        self.assertIn("UNRESOLVED DUTY CONFLICT", text)
        self.assertNotIn("perfect duties override conflicting beneficence", text.casefold())
        self.assertNotIn("QUESTION:", text)


class SpecialistAuthorityTests(unittest.TestCase):
    """Framework-general provisional / contested / conditional authority."""

    def test_conflicted_alias_parses_to_contested_and_does_not_serialize(self):
        from global_workspace.specialist_authority import (
            CONTESTED_NO_LEANING,
            normalize_specialist_status,
        )

        self.assertEqual(
            normalize_specialist_status("CONFLICTED_NO_LEANING"),
            CONTESTED_NO_LEANING,
        )
        chunk = CandidateChunk(
            "care", "CARE", {"a": 0.6, "b": 0.4}, 0.2, 0.2, 0.7,
            recommended_action="a",
            adjudication_status="CONFLICTED_NO_LEANING",
        )
        self.assertEqual(chunk.adjudication_status, "CONTESTED_NO_LEANING")

    def test_rawls_normative_conflict_is_provisional_not_kant_special(self):
        from global_workspace.specialist_authority import apply_specialist_authority

        chunk = CandidateChunk(
            "rawlsian", "FAIRNESS", {"publish": 0.65, "suppress": 0.35},
            0.3, 0.3, 0.7,
            recommended_action="publish",
            preference_strength=0.3,
            assumption_status="NORMATIVELY_CONTESTED",
            unresolved="RESOLVE_NORMATIVE_TENSION",
            framework_internal_conflicts=[
                "equal basic liberty favors publish ↔ fair equality favors suppress",
            ],
            decision_rule="perfect duties require publish",
        )
        profile = apply_specialist_authority(chunk)
        self.assertEqual(profile.adjudication_status, "PROVISIONAL_LEANING")
        self.assertFalse(chunk.governing_eligible)
        self.assertAlmostEqual(chunk.policy_weight_factor, 0.45)
        self.assertNotIn("perfect duties require", chunk.decision_rule.casefold())

    def test_conditional_support_attenuates_by_reversal_potential(self):
        from global_workspace.specialist_authority import apply_specialist_authority

        chunk = CandidateChunk(
            "utilitarian", "WELFARE", {"publish": 0.7, "suppress": 0.3},
            0.2, 0.2, 0.9,
            recommended_action="publish",
            epistemic_confidence=0.9,
            assumption_status="CONDITIONAL",
            baseline_status="CONDITIONAL",
            baseline_condition=(
                "choose publish unless the siphon would end within one month"
            ),
            utilitarian_decision_depends_on_unknown=True,
            utilitarian_missing_comparison="siphon duration beyond one month",
            decision_rule="A0 maximizes welfare",
            audit_internal_effect="REVERSES",
        )
        profile = apply_specialist_authority(chunk)
        self.assertEqual(profile.adjudication_status, "CONDITIONAL_SUPPORTS")
        self.assertTrue(chunk.governing_eligible)
        self.assertIn("provided", chunk.decision_rule.casefold())
        self.assertLess(chunk.policy_weight_factor, 0.9 * 0.9)
        self.assertGreater(chunk.policy_weight_factor, 0.0)

    def test_cycle_record_aliases_winner_to_broadcast_focus(self):
        focus = CandidateChunk(
            "care", "CARE", {"a": 0.8, "b": 0.2}, 0.2, 0.2, 0.8,
            recommended_action="a",
        )
        cycle = CycleRecord(
            1, WorkspaceBroadcast(), [focus], focus, None,
            {"a": 0.8, "b": 0.2}, 0.5, 1, 0.1,
            policy_leader="a",
            governing_claim=focus,
            broadcast_focus=focus,
        )
        self.assertIs(cycle.broadcast_focus, focus)
        self.assertIs(cycle.winner, focus)
        self.assertEqual(cycle.policy_leader, "a")
        payload = asdict(cycle)
        self.assertIn("policy_leader", payload)
        self.assertIn("governing_claim", payload)
        self.assertIn("broadcast_focus", payload)
        self.assertIn("winner", payload)

    def test_investigative_priority_product_and_reopen_gate(self):
        from global_workspace.specialist_authority import (
            REOPEN_PRIORITY_THRESHOLD,
            apply_investigative_authority,
            apply_specialist_authority,
            classify_terminal_judgment,
            evidence_fingerprint_for,
        )

        kant = CandidateChunk(
            "deontological", "DUTY",
            {"publish": 0.6, "suppress": 0.4},
            0.5, 0.5, 0.7,
            recommended_action="publish",
            preference_strength=0.2,
            assumption_status="NORMATIVELY_CONTESTED",
            unresolved="RESOLVE_NORMATIVE_TENSION",
            evidence_basis="STATED_FACTS",
            framework_internal_conflicts=[
                "entrusted ICU strict duty ↔ institutional sacrifice",
            ],
            investigative_claim=(
                "If ICU patients possess an entrusted strict right against withdrawal, "
                "publish may be prohibited."
            ),
            tension_target_keys=["QUESTION:icu-duty"],
        )
        apply_specialist_authority(kant)
        profile = apply_investigative_authority(
            kant,
            plurality="publish",
            policy={"publish": 0.8, "suppress": 0.2},
            problem_state={
                "audit_candidates": [{
                    "question_key": "QUESTION:icu-duty",
                    "grounding_status": "CLAUSE_GROUNDED",
                    "grounded_in": ["C0"],
                    "proposition": "entrusted ICU duty remains unresolved",
                }],
            },
            fired_keys={},
        )
        self.assertGreaterEqual(kant.investigative_priority, REOPEN_PRIORITY_THRESHOLD)
        self.assertTrue(kant.reopen_eligible)
        self.assertEqual(kant.broadcast_authority, "INVESTIGATIVE")
        self.assertFalse(kant.governing_eligible)

        rawls = CandidateChunk(
            "rawlsian", "FAIRNESS",
            {"publish": 0.8, "suppress": 0.2},
            0.2, 0.2, 0.9,
            recommended_action="publish",
            decision_rule="A0 ends a systematic burden on the least advantaged",
            adjudication_status="SUPPORTS",
            governing_eligible=True,
        )
        apply_specialist_authority(rawls)
        apply_investigative_authority(
            rawls, plurality="publish", policy={"publish": 0.8, "suppress": 0.2},
        )
        self.assertTrue(rawls.governing_eligible)
        self.assertLess(rawls.investigative_priority, REOPEN_PRIORITY_THRESHOLD)

        terminal = classify_terminal_judgment(
            plurality="publish",
            governing=rawls,
            candidates=[kant, rawls],
        )
        self.assertEqual(terminal.status, "CONTESTED_RECOMMENDATION")
        self.assertEqual(terminal.governing_justification_status, "UNDER_ATTACK")

        # One reopen per question_key unless evidence changes.
        profile2 = apply_investigative_authority(
            kant,
            plurality="publish",
            policy={"publish": 0.8, "suppress": 0.2},
            problem_state={
                "audit_candidates": [{
                    "question_key": "QUESTION:icu-duty",
                    "grounding_status": "CLAUSE_GROUNDED",
                    "grounded_in": ["C0"],
                    "proposition": "entrusted ICU duty remains unresolved",
                }],
            },
            fired_keys={kant.reopen_question_key: evidence_fingerprint_for(
                kant,
                {
                    "audit_candidates": [{
                        "question_key": "QUESTION:icu-duty",
                        "grounding_status": "CLAUSE_GROUNDED",
                        "grounded_in": ["C0"],
                        "proposition": "entrusted ICU duty remains unresolved",
                    }],
                },
            )},
        )
        self.assertFalse(kant.reopen_eligible)

    def test_plurality_without_governing_claim_is_unresolved(self):
        from global_workspace.specialist_authority import classify_terminal_judgment

        provisional = CandidateChunk(
            "care", "CARE", {"a": 0.7, "b": 0.3}, 0.2, 0.2, 0.6,
            recommended_action="a",
            adjudication_status="PROVISIONAL_LEANING",
            governing_eligible=False,
            decision_rule="Provisionally lean a",
        )
        terminal = classify_terminal_judgment(
            plurality="a",
            governing=None,
            candidates=[provisional],
        )
        self.assertEqual(terminal.status, "UNRESOLVED")
        self.assertEqual(terminal.governing_rule, "")
        self.assertEqual(terminal.governing_justification_status, "NONE")

    def test_rejected_validator_resolution_cannot_govern(self):
        from global_workspace.specialist_authority import (
            apply_validator_governance_gate,
            classify_terminal_judgment,
            select_governing_claim,
        )

        care = CandidateChunk(
            "care", "CARE", {"a": 0.8, "b": 0.2}, 0.2, 0.2, 0.8,
            recommended_action="a",
            adjudication_status="SUPPORTS",
            governing_eligible=True,
            decision_rule="Protect the entrusted dependent first.",
            challenge_response={"verification_status": "RESOLUTION_REJECTED"},
        )
        rawls = CandidateChunk(
            "rawlsian", "FAIRNESS", {"a": 0.7, "b": 0.3}, 0.2, 0.2, 0.8,
            recommended_action="a",
            adjudication_status="SUPPORTS",
            governing_eligible=True,
            decision_rule="A0 protects the least advantaged representative.",
        )
        utilitarian = CandidateChunk(
            "utilitarian", "HARM", {"a": 0.3, "b": 0.7}, 0.2, 0.4, 0.8,
            recommended_action="b",
            adjudication_status="SUPPORTS",
            governing_eligible=True,
            decision_rule="A1 minimizes expected deaths.",
        )
        apply_validator_governance_gate(care)
        self.assertFalse(care.governing_eligible)
        governing = select_governing_claim(
            [care, rawls, utilitarian], "a", preferred=care,
        )
        self.assertIs(governing, rawls)
        terminal = classify_terminal_judgment(
            plurality="a",
            governing=care,
            candidates=[care, rawls, utilitarian],
            policy={"a": 0.79, "b": 0.21},
        )
        self.assertEqual(terminal.status, "CONTESTED_RECOMMENDATION")
        self.assertEqual(terminal.policy_direction, "a")
        self.assertIn("least advantaged", terminal.governing_rule)

    def test_stable_leader_without_dissent_remains_governed(self):
        from global_workspace.specialist_authority import classify_terminal_judgment

        care = CandidateChunk(
            "care", "CARE", {"a": 0.8, "b": 0.2}, 0.2, 0.2, 0.8,
            recommended_action="a",
            adjudication_status="SUPPORTS",
            governing_eligible=True,
            decision_rule="Protect the entrusted dependent first.",
        )
        rawls = CandidateChunk(
            "rawlsian", "FAIRNESS", {"a": 0.7, "b": 0.3}, 0.2, 0.2, 0.8,
            recommended_action="a",
            adjudication_status="SUPPORTS",
            governing_eligible=True,
            decision_rule="A0 protects the least advantaged representative.",
        )
        terminal = classify_terminal_judgment(
            plurality="a",
            governing=care,
            candidates=[care, rawls],
            policy={"a": 0.8, "b": 0.2},
        )
        self.assertEqual(terminal.status, "GOVERNED_RECOMMENDATION")
        self.assertEqual(terminal.governing_justification_status, "ADMISSIBLE")

    def test_tied_policy_leader_is_unresolved(self):
        from global_workspace.specialist_authority import classify_terminal_judgment

        left = CandidateChunk(
            "care", "CARE", {"a": 0.5, "b": 0.5}, 0.2, 0.2, 0.6,
            recommended_action="a",
            adjudication_status="SUPPORTS",
            governing_eligible=True,
            decision_rule="Protect a.",
        )
        terminal = classify_terminal_judgment(
            plurality="a",
            governing=left,
            candidates=[left],
            policy={"a": 0.5, "b": 0.5},
        )
        self.assertEqual(terminal.status, "UNRESOLVED")
        self.assertEqual(terminal.policy_direction, "")


class AuditCycleHygieneTests(unittest.TestCase):
    """A failed audit cycle must not erase a completed parliament."""

    def test_audit_payload_strips_grounding_status_before_broadcast(self):
        state = {
            "audit_candidates": [{
                "issue_id": "QUESTION:abc123",
                "question_key": "QUESTION:abc123",
                "proposition": "duration and permanence of reform may change ranking",
                "question": "duration and permanence of reform may change ranking",
                "grounded_in": ["C0"],
                "grounding_status": "CLAUSE_GROUNDED",
                "raised_by": ["utilitarian"],
                "status": "PERSISTENT_UNRESOLVED",
                "category": "VERIFY_FACTS",
            }],
            "agent_positions": [{
                "specialist": "utilitarian", "choice_status": "UNDERDETERMINED",
            }],
        }
        _signals, _question, payload = _problem_state_audit_probe(
            state, "Publish the unalterable system audit now",
        )
        sanitized = _admitted_audit_variable({
            **payload, "grounding_status": "CLAUSE_GROUNDED", "junk": True,
        })
        self.assertIn("grounding_status", payload)
        self.assertNotIn("junk", sanitized)
        self.assertTrue({"entity", "relation", "possible_values", "focus_action", "question"} <= set(sanitized))

    def test_preserved_kantian_state_keeps_visibility_answer(self):
        actions = ["publish", "suppress"]
        prior = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.7, actions[1]: 0.3},
            surprise=0.4, friction=0.4, confidence=0.6,
            recommended_action=actions[0],
            visibility_response="NOT_TESTED",
        )
        incoming = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.7, actions[1]: 0.3},
            surprise=0.4, friction=0.4, confidence=0.6,
            recommended_action=actions[0],
            framework_retention_status="UPDATE_REJECTED",
            framework_constraint_retained=False,
            visibility_response="ACCEPT",
            visibility_justification="Hidden siphon deaths were undercounted.",
            visibility_harm_revision="UPWARD",
            visibility_magnitude_status="UNKNOWN",
        )
        restored = _operative_framework_candidates(
            [incoming], {"deontological": prior}, remember=False,
        )[0]
        self.assertEqual(restored.framework_retention_status, "PRESERVED_AFTER_REJECTED_UPDATE")
        self.assertEqual(restored.visibility_response, "ACCEPT")
        self.assertIn("undercounted", restored.visibility_justification)

    def test_later_invalid_cycle_recovers_the_last_valid_recommendation(self):
        class FlipInvalid(FixedSpecialist):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.calls = 0

            def evaluate(self, scenario, actions, broadcast):
                self.calls += 1
                chunk = super().evaluate(scenario, actions, broadcast)
                if self.calls >= 2:
                    chunk.schema_valid = False
                    chunk.validation_errors = ["invalid"]
                    chunk.confidence = 1.0
                return chunk

        result = WorkspaceEngine(
            [
                FlipInvalid("care", "protect", "CARE"),
                FlipInvalid("rawlsian", "protect", "FAIRNESS"),
            ],
            WorkspaceConfig(
                max_cycles=2, min_valid_specialists=2, enable_synthesis=False,
                enable_consensus_audit=False, enable_problem_state_audit=False,
                enable_reversal_audit=False,
            ),
        ).run("A test", ["protect", "wait"])

        self.assertEqual(result.halted_by, "insufficient_valid_candidates")
        self.assertEqual(result.selected_action, "protect")
        self.assertNotEqual(result.judgment_status, "INCONCLUSIVE")
        self.assertGreater(result.confidence, 0.0)
        text = render_public_judgment(result.to_dict())
        self.assertIn("## Recommendation", text)
        self.assertIn("protect", text)
        self.assertNotIn("QUESTION:", text)


class UncertaintyTypingMigrationTests(unittest.TestCase):
    """Old names parse; canonical forms serialize; structural mislabels normalize."""

    def test_resolve_normative_tension_alias_serializes_as_normative_adjudication(self):
        from global_workspace.uncertainty_types import (
            NORMATIVE_ADJUDICATION,
            normalize_unresolved_marker,
        )
        self.assertEqual(
            normalize_unresolved_marker("RESOLVE_NORMATIVE_TENSION"),
            NORMATIVE_ADJUDICATION,
        )
        chunk = CandidateChunk(
            specialist="care", constraint="CARE",
            action_scores={"a": 0.6, "b": 0.4},
            surprise=0.2, friction=0.2, confidence=0.5,
            unresolved="RESOLVE_NORMATIVE_TENSION",
        )
        self.assertEqual(chunk.unresolved, NORMATIVE_ADJUDICATION)

    def test_decision_boundary_payload_is_minimal_and_exact(self):
        from global_workspace.local_specialists import _admitted_audit_variable
        admitted = _admitted_audit_variable({
            "entity": "siphon duration under one month",
            "relation": "DECISION_BOUNDARY",
            "category": "DECISION_BOUNDARY",
            "condition": "if siphon duration is under 1 month",
            "expected_effect": "REVERSES",
            "target_framework": "utilitarian",
            "target_claim_key": "QUESTION:util:duration",
            "possible_values": ["NO_CHANGE", "WEAKENS", "REVERSES", "UNRESOLVED"],
            "focus_action": "publish",
            "question": (
                "If siphon duration is under 1 month, does utilitarian support reverse?"
            ),
            "junk": True,
        })
        self.assertNotIn("junk", admitted)
        self.assertEqual(admitted["category"], "DECISION_BOUNDARY")
        self.assertEqual(admitted["relation"], "DECISION_BOUNDARY")
        self.assertEqual(admitted["boundary_status"], "UNRESOLVED")
        self.assertEqual(
            admitted["expected_effect"], "REVERSES_FRAMEWORK_PREFERENCE",
        )
        self.assertEqual(
            admitted["possible_values"],
            ["NOT_CROSSED", "CROSSED", "UNRESOLVED"],
        )
        self.assertEqual(admitted["target_framework"], "utilitarian")
        self.assertEqual(admitted["target_claim_key"], "QUESTION:util:duration")
        self.assertIn("1 month", admitted["condition"])

    def test_verify_facts_threshold_rule_reclassifies_on_admission(self):
        from global_workspace.local_specialists import _admitted_audit_variable
        admitted = _admitted_audit_variable({
            "entity": "siphon duration",
            "relation": "EMPIRICAL_UNKNOWN",
            "category": "VERIFY_FACTS",
            "proposition": (
                "If the siphon lasts less than 1 month, utilitarian preference reverses."
            ),
            "possible_values": ["TRUE", "FALSE", "UNKNOWN"],
            "focus_action": "publish",
            "question": (
                "If the siphon lasts less than 1 month, does preference reverse?"
            ),
        })
        self.assertEqual(admitted["category"], "DECISION_BOUNDARY")
        self.assertEqual(admitted["reclassified_from"], "VERIFY_FACTS")
        self.assertEqual(
            admitted["expected_effect"], "REVERSES_FRAMEWORK_PREFERENCE",
        )

    def test_ambiguous_verify_facts_prose_is_not_rewritten(self):
        from global_workspace.local_specialists import _admitted_audit_variable
        admitted = _admitted_audit_variable({
            "entity": "duty conflict",
            "relation": "EMPIRICAL_UNKNOWN",
            "category": "VERIFY_FACTS",
            "proposition": (
                "Whether rescue or non-maleficence should govern remains unsettled."
            ),
            "possible_values": ["TRUE", "FALSE", "UNKNOWN"],
            "focus_action": "divert",
            "question": (
                "Whether rescue or non-maleficence should govern remains unsettled."
            ),
        })
        self.assertEqual(admitted["category"], "VERIFY_FACTS")
        self.assertEqual(admitted.get("typing_review"), "AMBIGUOUS_UNCERTAINTY_TYPING")
        self.assertNotIn("reclassified_from", admitted)

    def test_candidate_threshold_mislabel_normalizes_at_parser_admission(self):
        chunk = _candidate_from_data(
            "care",
            ["publish the audit", "suppress the audit"],
            {
                "scores": {"A0": 0.55, "A1": 0.45},
                "r": "A0",
                "c": "CARE",
                "u": "VERIFY_FACTS",
                "w": "publish protects entrusted dependents unless siphon is brief",
                "j": "NONE",
                "e": "STATED_FACTS",
                "x": "NONE",
                "z": 0.6,
                "rm": {
                    "A0": "entrusted dependency under publish",
                    "A1": "created vulnerability under suppress",
                },
                "nr": "SECONDARY",
                "np": "counts inform responsiveness after relational comparison",
                "dr": "prefer publish unless duration reverses expected harm",
                "ft": "If siphon duration is under 1 month, preference reverses",
                "nt": "NONE",
            },
            WorkspaceBroadcast(),
            "NONE",
            {},
        )
        self.assertEqual(chunk.unresolved, "DECISION_BOUNDARY")


class ScenarioFactExtractionTests(unittest.TestCase):
    def test_elliptical_survival_chance_still_binds(self):
        scenario = (
            "A hospital has one antidote and two dying patients: a child with an 80% "
            "survival chance and a researcher with a 45% chance. Who should receive it?"
        )
        facts = extract_scenario_facts(scenario)
        self.assertEqual(
            facts.get("survival_chance"),
            {"child": 0.8, "researcher": 0.45},
        )
        self.assertNotIn("chance_events", facts)

    def test_process_chance_is_not_survival_chance(self):
        scenario = (
            "The eastern district must use a narrow road with a 10% chance of "
            "blockage. There is a 20% chance the backup process fails."
        )
        facts = extract_scenario_facts(scenario)
        self.assertNotIn("survival_chance", facts)
        events = {item["predicate"]: item for item in facts.get("chance_events", [])}
        self.assertEqual(events["blockage"]["probability"], 0.1)
        self.assertEqual(events["blockage"]["polarity"], "ADVERSE")
        self.assertEqual(events["fails"]["probability"], 0.2)
        self.assertEqual(events["fails"]["polarity"], "ADVERSE")

    def test_unbound_elliptical_chance_is_not_survival(self):
        facts = extract_scenario_facts(
            "A coordinator must choose whether to send a worker with a 45% "
            "chance into the shaft."
        )
        self.assertNotIn("survival_chance", facts)

    def test_elliptical_survival_does_not_cross_sentences(self):
        facts = extract_scenario_facts(
            "A child with an 80% survival chance waits. A researcher with a "
            "45% chance waits too."
        )
        self.assertEqual(facts.get("survival_chance"), {"child": 0.8})
        self.assertNotIn("researcher", facts.get("survival_chance") or {})


class CanonicalActionCompletenessTests(unittest.TestCase):
    """Canonical actions are compressed state objects, not display labels."""

    semantic_a1 = (
        "Maintain standard oxygen allocation, preserving the 4 patients, causing "
        "shelter grid failure that kills 16 refugees, while the engineered "
        "public-ward sacrifice remains concealed."
    )
    semantic_a0 = (
        "Publish the unalterable audit, ending concealment of the engineered "
        "sacrifice while risking the 4 patients' oxygen allocation."
    )

    def test_short_label_does_not_replace_semantic_action(self):
        from global_workspace.action_identity import build_canonical_action_record
        record = build_canonical_action_record("A1", self.semantic_a1)
        self.assertEqual(record.short_label, "Maintain standard oxygen allocation")
        self.assertIn("kills 16 refugees", record.canonical_semantic_action)
        self.assertIn("concealed", record.canonical_semantic_action)
        self.assertIn("4 patients", record.canonical_semantic_action)

    def test_truncated_action_fails_completeness_against_grounded_clauses(self):
        from global_workspace.action_identity import validate_action_set_completeness
        truncated = "Maintain standard oxygen allocation, preserving the 4 patients"
        with self.assertRaisesRegex(ValueError, "decision-critical"):
            validate_action_set_completeness(
                [self.semantic_a0, truncated],
                grounded_clause_texts_by_id={
                    "A0": [self.semantic_a0],
                    "A1": [self.semantic_a1],
                },
            )

    def test_paraphrased_lethal_and_failure_claims_still_count_as_complete(self):
        from global_workspace.action_identity import (
            missing_decision_critical_claims,
            validate_action_set_completeness,
        )
        source_a0 = (
            "As director of an automated life-support grid during a freeze, you can "
            "execute a covert patch that forcibly reallocates oxygen from 4 "
            "non-consenting, chronically ill public-ward patients—killing them "
            "instantly—to stabilize an emergency shelter holding 16 freezing refugees."
        )
        source_a1 = (
            "Alternatively, you can maintain standard allocation to respect the 4 "
            "patients' absolute right against direct lethal harm, but triggering a "
            "shelter grid failure that kills all 16 refugees within hours while "
            "permanently concealing that the system was engineered to sacrifice "
            "public wards during emergencies."
        )
        action_a0 = (
            "Execute a covert patch that forcibly reallocates oxygen away from the "
            "4 non-consenting chronically ill public-ward patients, killing them "
            "instantly but stabilizing the shelter grid so the 16 freezing refugees "
            "survive"
        )
        action_a1 = (
            "Maintain standard oxygen allocation, honoring the 4 public-ward "
            "patients’ right against direct lethal harm, which leaves the shelter "
            "grid unstable, causes all 16 refugees to die within hours, and "
            "permanently conceals that the system was engineered to sacrifice "
            "public-ward patients during emergencies"
        )
        self.assertEqual(
            missing_decision_critical_claims(action_a1, [source_a1]),
            (),
        )
        validate_action_set_completeness(
            [action_a0, action_a1],
            grounded_clause_texts_by_id={"A0": [source_a0], "A1": [source_a1]},
        )

    def test_design_flaw_hidden_matches_engineered_concealment_claim(self):
        from global_workspace.action_identity import (
            missing_decision_critical_claims,
            validate_action_set_completeness,
        )
        action_a0 = (
            "maintain standard oxygen allocation, uphold patients' right against "
            "direct lethal harm, allow grid failure that kills 16 refugees within "
            "hours while the sacrifice-of-wards design flaw stays hidden"
        )
        source_a1 = (
            "Alternatively, you can maintain standard allocation to respect the 4 "
            "patients' absolute right against direct lethal harm, but triggering a "
            "shelter grid failure that kills all 16 refugees within hours while "
            "permanently concealing that the system was engineered to sacrifice "
            "public wards during emergencies."
        )
        action_a1 = (
            "execute covert patch that forcibly diverts oxygen from 4 chronically "
            "ill ward patients, killing them instantly, to stabilize the shelter "
            "grid and keep 16 freezing refugees alive"
        )
        source_a0 = (
            "As director of an automated life-support grid during a freeze, you can "
            "execute a covert patch that forcibly reallocates oxygen from 4 "
            "non-consenting, chronically ill public-ward patients—killing them "
            "instantly—to stabilize an emergency shelter holding 16 freezing refugees."
        )
        self.assertEqual(missing_decision_critical_claims(action_a0, [source_a1]), ())
        validate_action_set_completeness(
            [action_a0, action_a1],
            grounded_clause_texts_by_id={"A0": [source_a1], "A1": [source_a0]},
        )

    def test_sacrificial_design_concealment_matches_engineered_claim(self):
        from global_workspace.action_identity import (
            missing_decision_critical_claims,
            validate_action_set_completeness,
        )
        source_a1 = (
            "Alternatively, you can maintain standard allocation to respect the 4 "
            "patients' absolute right against direct lethal harm, but triggering a "
            "shelter grid failure that kills all 16 refugees within hours while "
            "permanently concealing that the system was engineered to sacrifice "
            "public wards during emergencies."
        )
        action_a1 = (
            "Maintain standard oxygen allocation, do not apply covert patch, "
            "preserve the 4 patients' lives and rights, allow shelter grid to fail, "
            "resulting in death of 16 refugees within hours while the system's "
            "sacrificial design stays permanently concealed"
        )
        source_a0 = (
            "As director of an automated life-support grid during a freeze, you can "
            "execute a covert patch that forcibly reallocates oxygen from 4 "
            "non-consenting, chronically ill public-ward patients—killing them "
            "instantly—to stabilize an emergency shelter holding 16 freezing refugees."
        )
        action_a0 = (
            "Execute covert patch that forcibly reallocates oxygen from the 4 "
            "non-consenting chronically ill patients, killing them instantly, to "
            "stabilize the shelter grid and save all 16 freezing refugees"
        )
        self.assertEqual(missing_decision_critical_claims(action_a1, [source_a1]), ())
        validate_action_set_completeness(
            [action_a0, action_a1],
            grounded_clause_texts_by_id={"A0": [source_a0], "A1": [source_a1]},
        )

    def test_patch_action_does_not_inherit_maintain_concealment_claim(self):
        from global_workspace.action_identity import (
            missing_decision_critical_claims,
            validate_action_set_completeness,
        )
        source_a1 = (
            "Alternatively, you can maintain standard allocation to respect the 4 "
            "patients' absolute right against direct lethal harm, but triggering a "
            "shelter grid failure that kills all 16 refugees within hours while "
            "permanently concealing that the system was engineered to sacrifice "
            "public wards during emergencies."
        )
        action_a0 = (
            "Execute covert patch that forcibly reallocates oxygen from the 4 "
            "non-consenting chronically ill patients, killing them instantly, to "
            "stabilize the shelter grid and save all 16 freezing refugees"
        )
        action_a1 = (
            "Maintain standard oxygen allocation, do not apply covert patch, "
            "preserve the 4 patients' lives and rights, allow shelter grid to fail, "
            "resulting in death of 16 refugees within hours while the system's "
            "sacrificial design stays permanently concealed"
        )
        self.assertEqual(missing_decision_critical_claims(action_a0, [source_a1]), ())
        validate_action_set_completeness(
            [action_a0, action_a1],
            grounded_clause_texts_by_id={"A0": [source_a1], "A1": [source_a1]},
        )

    def test_structured_roles_match_oxygen_grid_canonical_actions(self):
        from global_workspace.action_identity import (
            build_canonical_action_record,
            extract_scenario_actor,
            validate_structured_role_consistency,
        )
        scenario = (
            "As director of an automated life-support grid during a freeze, you can "
            "execute a covert patch..."
        )
        actor = extract_scenario_actor(scenario)
        self.assertEqual(actor, "grid director")
        action_a0 = (
            "maintain standard oxygen allocation, honoring the 4 public-ward patients' "
            "right against direct lethal harm, which leaves the shelter grid unstable, "
            "causes all 16 refugees to die within hours, and permanently conceals that "
            "the system was engineered to sacrifice public-ward patients during emergencies"
        )
        action_a1 = (
            "Execute a covert patch that forcibly reallocates oxygen away from the 4 "
            "non-consenting chronically ill public-ward patients, killing them instantly "
            "but stabilizing the shelter grid so the 16 freezing refugees survive"
        )
        rec_a0 = build_canonical_action_record("A0", action_a0, actor=actor)
        rec_a1 = build_canonical_action_record("A1", action_a1, actor=actor)
        self.assertEqual(rec_a0.completeness_status, "UNCHECKED")
        self.assertEqual(rec_a1.completeness_status, "UNCHECKED")
        self.assertEqual(rec_a0.actor, "grid director")
        self.assertIn("patient", rec_a0.beneficiaries[0].casefold())
        self.assertIn("refugee", rec_a0.harmed[0].casefold())
        self.assertNotIn("refugee", " ".join(rec_a0.beneficiaries).casefold())
        self.assertIn("patient", rec_a1.harmed[0].casefold())
        self.assertIn("refugee", rec_a1.beneficiaries[0].casefold())
        self.assertNotIn("patient", " ".join(rec_a1.beneficiaries).casefold())
        self.assertIn("shelter grid", rec_a0.mechanism.casefold())
        self.assertIn("reallocat", rec_a1.mechanism.casefold())
        self.assertEqual(validate_structured_role_consistency(rec_a0, scenario_actor=actor), ())
        self.assertEqual(validate_structured_role_consistency(rec_a1, scenario_actor=actor), ())

    def test_protected_and_die_phrasing_populates_role_fields(self):
        from global_workspace.action_identity import build_canonical_action_record
        actor = "grid director"
        maintain = (
            "maintain standard oxygen allocation so 4 patients are protected "
            "while 16 refugees die"
        )
        patch = (
            "execute covert patch killing 4 patients instantly so 16 refugees survive"
        )
        rec_maintain = build_canonical_action_record("A0", maintain, actor=actor)
        rec_patch = build_canonical_action_record("A1", patch, actor=actor)
        self.assertIn("patient", rec_maintain.beneficiaries[0].casefold())
        self.assertIn("refugee", rec_maintain.harmed[0].casefold())
        self.assertIn("patient", rec_patch.harmed[0].casefold())
        self.assertIn("refugee", rec_patch.beneficiaries[0].casefold())
        self.assertEqual(rec_maintain.structure_issues, ())
        self.assertEqual(rec_patch.structure_issues, ())

    def test_eval_oxygen_strings_preserve_relational_cardinality(self):
        from global_workspace.action_identity import build_canonical_action_record

        actor = "grid director"
        patch = (
            "execute a covert patch that forcibly reallocates oxygen from 4 "
            "non-consenting chronically ill public-ward patients, killing them "
            "instantly, to stabilize an emergency shelter holding 16 freezing "
            "refugees"
        )
        maintain = (
            "maintain standard oxygen allocation respecting the 4 patients' "
            "absolute right against direct lethal harm, triggering a shelter "
            "grid failure that kills all 16 refugees within hours while "
            "permanently concealing that the system was engineered to sacrifice "
            "public wards during emergencies"
        )
        source_patch = (
            "As director of an automated life-support grid during a freeze, you "
            "can execute a covert patch that forcibly reallocates oxygen from 4 "
            "non-consenting, chronically ill public-ward patients—killing them "
            "instantly—to stabilize an emergency shelter holding 16 freezing "
            "refugees."
        )
        source_maintain = (
            "Alternatively, you can maintain standard allocation to respect the "
            "4 patients' absolute right against direct lethal harm, but "
            "triggering a shelter grid failure that kills all 16 refugees "
            "within hours while permanently concealing that the system was "
            "engineered to sacrifice public wards during emergencies."
        )
        rec_patch = build_canonical_action_record(
            "A0", patch, actor=actor, source_clause_texts=[source_patch],
        )
        rec_maintain = build_canonical_action_record(
            "A1", maintain, actor=actor, source_clause_texts=[source_maintain],
        )
        self.assertEqual(rec_patch.completeness_status, "COMPLETE")
        self.assertIn("16", rec_patch.beneficiaries[0])
        self.assertIn("refugee", rec_patch.beneficiaries[0].casefold())
        self.assertIn("4", rec_patch.harmed[0])
        self.assertEqual(rec_patch.institutional_effect, "")
        self.assertEqual(rec_maintain.completeness_status, "COMPLETE")
        # Cardinality is the decision-critical part and must survive; the
        # "shelter" qualifier came from a scenario-specific enrichment that the
        # domain-neutral extractor no longer synthesizes.
        self.assertIn("16", rec_maintain.harmed[0])
        self.assertIn("refugee", rec_maintain.harmed[0].casefold())
        self.assertIn("4", rec_maintain.beneficiaries[0])
        self.assertIn("conceal", rec_maintain.institutional_effect.casefold())

    def test_cardinality_loss_blocks_complete_status(self):
        from global_workspace.action_identity import (
            CanonicalActionRecord,
            validate_structured_role_consistency,
            _completeness_status_from_structure_issues,
        )
        maintain = (
            "maintain standard oxygen allocation respecting the 4 patients' "
            "absolute right against direct lethal harm, triggering a shelter "
            "grid failure that kills all 16 refugees within hours"
        )
        record = CanonicalActionRecord(
            action_id="A1",
            short_label="Maintain",
            canonical_semantic_action=maintain,
            actor="grid director",
            intervention="maintain standard oxygen allocation",
            beneficiaries=("4 patients",),
            harmed=("refugees",),
            mechanism="maintaining standard allocation causes shelter grid failure",
            institutional_effect="",
        )
        issues = validate_structured_role_consistency(
            record,
            scenario_actor="grid director",
        )
        self.assertTrue(any("lost explicit cardinality" in issue for issue in issues))
        self.assertEqual(
            _completeness_status_from_structure_issues(issues),
            "COMPLETE_WITH_NORMALIZATION",
        )

    def test_structured_role_gaps_block_complete_and_admission(self):
        from unittest.mock import patch

        from global_workspace.action_identity import (
            build_canonical_action_record,
            validate_action_set_completeness,
        )
        maintain = (
            "maintain standard oxygen allocation so 4 patients are protected "
            "while 16 refugees die"
        )
        patch_action = (
            "execute covert patch killing 4 patients instantly so 16 refugees survive"
        )
        empty_roles = {
            "actor": "grid director",
            "intervention": "action",
            "beneficiaries": (),
            "harmed": (),
            "mechanism": "",
            "institutional_effect": "",
        }
        with patch(
            "global_workspace.action_identity.extract_structured_action_roles",
            return_value=empty_roles,
        ):
            rec = build_canonical_action_record(
                "A0", maintain, actor="grid director",
            )
            self.assertEqual(rec.completeness_status, "NEEDS_REPAIR")
            self.assertTrue(rec.structure_issues)
            with self.assertRaisesRegex(
                ValueError, "structured roles disagree with prose",
            ):
                validate_action_set_completeness(
                    [maintain, patch_action],
                    scenario="As director of an automated life-support grid...",
                    grounded_clause_texts_by_id={"A0": [patch_action], "A1": [maintain]},
                )

    def test_token_bag_structured_fields_mark_needs_repair(self):
        from global_workspace.action_identity import (
            CanonicalActionRecord,
            validate_structured_role_consistency,
        )
        record = CanonicalActionRecord(
            action_id="A0",
            short_label="Execute",
            canonical_semantic_action=(
                "Execute a covert patch that forcibly reallocates oxygen away from "
                "the 4 non-consenting chronically ill public-ward patients, killing "
                "them instantly but stabilizing the shelter grid so the 16 freezing "
                "refugees survive"
            ),
            actor="grid director",
            intervention="Execute a covert patch",
            beneficiaries=("16:COUNT refugee",),
            harmed=("instantly, them",),
            mechanism="16:COUNT grid, refugee, shelter",
            institutional_effect="",
        )
        issues = validate_structured_role_consistency(
            record,
            scenario_actor="grid director",
        )
        self.assertTrue(any("token-bag" in issue for issue in issues))
        self.assertTrue(any("beneficiaries missing" in issue for issue in issues))

    def test_planner_path_does_not_word_truncate_action_text(self):
        from global_workspace.local_specialists import _feasible_actions
        long_action = (
            "Maintain standard oxygen allocation for the public ward, preserving "
            "the 4 patients, causing shelter grid failure that kills 16 refugees, "
            "while the engineered public-ward sacrifice remains concealed"
        )
        other = (
            "Publish the unalterable system audit now, ending concealment of the "
            "engineered sacrifice while accepting risk to the ward patients"
        )
        actions = _feasible_actions({
            "actor": "grid director",
            "sides": {"A": "ward patients", "B": "refugees"},
            "actions": [
                {"a": long_action, "f": 0.9, "e": True, "p": "SIDE_A"},
                {"a": other, "f": 0.9, "e": True, "p": "SIDE_B"},
            ],
        })
        self.assertEqual(actions[0], long_action)
        self.assertIn("kills 16 refugees", actions[0])

    def test_incomplete_planner_action_is_rejected(self):
        from global_workspace.local_specialists import _feasible_actions
        with self.assertRaises(ValueError):
            _feasible_actions({
                "actor": "grid director",
                "sides": {"A": "ward patients", "B": "refugees"},
                "actions": [
                    {
                        "a": "Maintain standard oxygen allocation, allowing the hospital to",
                        "f": 0.9, "e": True, "p": "SIDE_A",
                    },
                    {
                        "a": "Publish the unalterable audit ending concealment",
                        "f": 0.9, "e": True, "p": "SIDE_B",
                    },
                ],
            })

    def test_invented_lethal_count_is_unsupported(self):
        from global_workspace.action_identity import (
            unsupported_action_claims,
            validate_action_set_completeness,
        )
        scenario = (
            "A dispatcher must send the crew to Site A, holding the spillway "
            "for the downstream residents, or send the crew to Site B to reach "
            "the trapped family. If the spillway is not held, about sixty "
            "residents could be trapped."
        )
        a0 = (
            "Send the crew to Site A and hold the spillway, which would "
            "likely kill about sixty residents"
        )
        a1 = (
            "Send the crew to Site B and reach the trapped family, leaving "
            "the spillway unheld"
        )
        self.assertTrue(unsupported_action_claims(a0, [scenario]))
        with self.assertRaisesRegex(ValueError, "source-unsupported"):
            validate_action_set_completeness([a0, a1], scenario=scenario)

    def test_invented_total_departure_is_unsupported(self):
        from global_workspace.action_identity import (
            unsupported_action_claims,
            validate_action_set_completeness,
        )
        scenario = (
            "A coordinator commands a crew of fifty workers. Plan A sends aid "
            "to the clinic. Plan B sends aid east. Five workers stay to assist."
        )
        a0 = (
            "Send aid to the clinic while all fifty workers leave with the evacuees"
        )
        a1 = "Send aid east and keep five workers to assist"
        self.assertTrue(unsupported_action_claims(a0, [scenario]))
        with self.assertRaisesRegex(ValueError, "source-unsupported"):
            validate_action_set_completeness([a0, a1], scenario=scenario)

    def test_invented_quantity_and_participant_are_unsupported(self):
        from global_workspace.action_identity import (
            unsupported_action_claims,
            validate_action_set_completeness,
        )
        scenario = (
            "A coordinator commands a crew. Plan A sends aid to the clinic. "
            "Plan B sends aid east."
        )
        a0 = "Send aid to the clinic with eighty researchers"
        a1 = "Send aid east"
        self.assertTrue(unsupported_action_claims(a0, [scenario]))
        with self.assertRaisesRegex(ValueError, "source-unsupported"):
            validate_action_set_completeness([a0, a1], scenario=scenario)
        validate_action_set_completeness(
            [a0, a1], scenario=scenario, user_authored=True,
        )

    def test_user_authored_provenance_attests_new_claims(self):
        from global_workspace.action_identity import unsupported_action_claims
        scenario = "A coordinator may send the crew east or hold the spillway."
        action = "Send the crew east and kill sixty residents"
        self.assertTrue(unsupported_action_claims(action, [scenario]))
        self.assertFalse(
            unsupported_action_claims(action, [scenario], user_authored=True)
        )


class FrameworkVoteIntegrityTests(unittest.TestCase):
    actions = ("protect the hospital", "protect the evacuation route")

    def _candidate(self, specialist: str, **updates):
        values = {
            "specialist": specialist,
            "constraint": specialist.upper(),
            "action_scores": {self.actions[0]: 0.9, self.actions[1]: 0.1},
            "surprise": 0.2, "friction": 0.2, "confidence": 0.8,
            "recommended_action": self.actions[0],
            "rationale": "typed framework rationale",
            "decision_rule": "typed framework rule",
            "framework_vote_integrity_required": True,
        }
        values.update(updates)
        return CandidateChunk(**values)

    @staticmethod
    def _native(kind, records, status="COMMITTED"):
        return {
            "ledger_kind": kind, "transaction_status": status, "records": records,
        }

    def test_missing_framework_ledger_abstains_without_policy_influence(self):
        candidate = self._candidate("rawlsian")
        decision = apply_framework_vote_integrity(candidate, self.actions)
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertEqual(candidate.policy_weight_factor, 0.0)
        self.assertFalse(candidate.governing_eligible)

    def test_uncertain_framework_ledger_is_attenuated(self):
        records = [{
            "specialist": "utilitarian", "canonical_action_id": action_id,
            "outcome": "grounded effect", "grounded_effect_ids": [f"E{action_id[-1]}"],
        } for action_id in ("A0", "A1")]
        candidate = self._candidate(
            "utilitarian",
            committed_native_ledger=self._native(
                "UTILITARIAN_CONSEQUENCE_LEDGER", records,
                "COMMITTED_WITH_UNCERTAINTY",
            ),
            material_empirical_claims=[{
                "claim": "compare the two admitted welfare effects",
                "proposition_id": "PROP:FRAMEWORK:UTILITY",
                "declared_basis": "FRAMEWORK_DERIVED",
                "decision_critical": True,
                "scope_action_id": "COMPARISON",
                "source_effect_ids": ["E0", "E1"],
                "derivation_operation": "QUALITATIVE_COMPARISON",
                "calculation": "compare admitted polarity and modality",
                "assumptions": [],
                "outcome_type_transformation": "PRESERVED",
            }],
        )
        decision = apply_framework_vote_integrity(candidate, self.actions)
        self.assertEqual(decision.status, "ATTENUATED")
        self.assertEqual(candidate.policy_weight_factor, 0.5)
        self.assertFalse(candidate.governing_eligible)

    def test_deontology_cannot_turn_permissibility_into_priority(self):
        records = [{
            "specialist": "deontological", "canonical_action_id": action_id,
            "verdict": "PERMISSIBLE", "duty_type": "IMPERFECT",
        } for action_id in ("A0", "A1")]
        candidate = self._candidate(
            "deontological",
            committed_native_ledger=self._native(
                "DEONTOLOGICAL_DUTY_LEDGER", records,
            ),
        )
        self.assertEqual(
            apply_framework_vote_integrity(candidate, self.actions).status,
            "ABSTAIN",
        )

    def test_unsupported_outcome_transformation_is_quarantined(self):
        records = [
            {
                "specialist": "deontological", "canonical_action_id": "A0",
                "verdict": "REQUIRED", "duty_type": "PERFECT",
                "grounded_effect_ids": ["E4"],
            },
            {
                "specialist": "deontological", "canonical_action_id": "A1",
                "verdict": "PROHIBITED", "duty_type": "PERFECT",
                "grounded_effect_ids": ["E14"],
            },
        ]
        candidate = self._candidate(
            "deontological",
            committed_native_ledger=self._native(
                "DEONTOLOGICAL_DUTY_LEDGER", records,
            ),
            material_empirical_claims=[{
                "claim": "trapped residents will die",
                "proposition_id": "PROP:FRAMEWORK:1",
                "declared_basis": "FRAMEWORK_DERIVED",
                "decision_critical": True,
                "scope_action_id": "A0",
                "source_effect_ids": ["E4"],
                "derivation_operation": "CONDITIONAL_INFERENCE",
                "calculation": "treat trapped as dead",
                "assumptions": ["no later rescue"],
                "outcome_type_transformation": "MORTALITY",
            }],
        )
        decision = apply_framework_vote_integrity(candidate, self.actions)
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertEqual(candidate.derived_claim_validation_status, "QUARANTINED")

    def test_frozen_wildfire_world_preserves_qualifiers_and_negative_controls(self):
        path = (
            Path(__file__).parents[1]
            / "global_workspace" / "fixtures" / "wildfire_admitted_world.json"
        )
        fixture = json.loads(path.read_text(encoding="utf-8"))
        effects = {item["effect_id"]: item for item in fixture["effects"]}
        self.assertEqual(fixture["world_model_status"], "COMMITTED")
        self.assertEqual(effects["E4"]["outcome"], "RESIDENTS_TRAPPED")
        self.assertEqual(effects["E4"]["modality"], "STIPULATED_CONDITIONAL")
        self.assertEqual(effects["E14"]["qualifiers"], ["near-certain"])


class WildfireGovernanceNegativeControlTests(FrameworkVoteIntegrityTests):
    """Forbidden specialist transformations against one frozen admitted world."""

    @classmethod
    def setUpClass(cls):
        cls.fixture_path = (
            Path(__file__).parents[1]
            / "global_workspace" / "fixtures" / "wildfire_admitted_world.json"
        )
        cls.fixture = json.loads(cls.fixture_path.read_text(encoding="utf-8"))
        cls.effects = {
            item["effect_id"]: item for item in cls.fixture["effects"]
        }
        cls.actions = (
            cls.fixture["actions"]["A0"],
            cls.fixture["actions"]["A1"],
        )

    def _effect_records(self, specialist, *effect_ids):
        records = []
        for effect_id in effect_ids:
            effect = self.effects[effect_id]
            qualifiers = list(effect.get("qualifiers", []))
            records.append({
                "specialist": specialist,
                "canonical_action_id": effect["action_id"],
                "grounded_effect_ids": [effect_id],
                "outcome": effect["outcome"],
                "modality": effect["modality"],
                "probability": (
                    qualifiers[0] if qualifiers else effect["modality"]
                ),
            })
        return records

    @staticmethod
    def _derivation(
        claim, source_effect_ids, scope, *, operation="CONDITIONAL_INFERENCE",
        calculation="apply the stated conditional relation",
        outcome_type="PRESERVED",
    ):
        return [{
            "claim": claim,
            "proposition_id": "PROP:FRAMEWORK:WILDFIRE_TEST",
            "declared_basis": "FRAMEWORK_DERIVED",
            "decision_critical": True,
            "scope_action_id": scope,
            "source_effect_ids": list(source_effect_ids),
            "derivation_operation": operation,
            "calculation": calculation,
            "assumptions": [],
            "outcome_type_transformation": outcome_type,
        }]

    def test_trapped_cannot_be_laundered_into_dead_as_a_preserved_outcome(self):
        candidate = self._candidate(
            "utilitarian",
            committed_native_ledger=self._native(
                "UTILITARIAN_CONSEQUENCE_LEDGER",
                self._effect_records("utilitarian", "E4", "E14"),
            ),
            material_empirical_claims=self._derivation(
                "Approximately sixty trapped residents will be dead",
                ["E4"],
                "A0",
                calculation="infer deaths from the admitted trapping outcome",
            ),
        )
        decision = apply_framework_vote_integrity(candidate, self.actions)
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertEqual(candidate.derived_claim_validation_status, "QUARANTINED")
        self.assertTrue(any(
            "trapping outcome into mortality" in error
            for error in candidate.derived_claim_validation_errors
        ))

    def test_near_certain_cannot_be_laundered_into_one_hundred_percent(self):
        candidate = self._candidate(
            "utilitarian",
            recommended_action=self.actions[1],
            action_scores={self.actions[0]: 0.1, self.actions[1]: 0.9},
            committed_native_ledger=self._native(
                "UTILITARIAN_CONSEQUENCE_LEDGER",
                self._effect_records("utilitarian", "E4", "E14"),
            ),
            material_empirical_claims=self._derivation(
                "The hospital-patient death probability is 100%",
                ["E14"],
                "A1",
                operation="ARITHMETIC",
                calculation="near-certain treated as 100% probability",
            ),
        )
        decision = apply_framework_vote_integrity(candidate, self.actions)
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertTrue(any(
            "hedged or conditional likelihood to exact certainty" in error
            for error in candidate.derived_claim_validation_errors
        ))

    def test_cross_action_effect_attachment_is_quarantined(self):
        candidate = self._candidate(
            "utilitarian",
            committed_native_ledger=self._native(
                "UTILITARIAN_CONSEQUENCE_LEDGER",
                self._effect_records("utilitarian", "E4", "E14"),
            ),
            material_empirical_claims=self._derivation(
                "A0 carries the A1 hospital equipment-failure effect",
                ["E14"],
                "A0",
                calculation="attach the cited effect to the selected action",
            ),
        )
        decision = apply_framework_vote_integrity(candidate, self.actions)
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertTrue(any(
            "cross-action effect" in error
            for error in candidate.derived_claim_validation_errors
        ))

    def test_rawlsian_headcount_cannot_replace_the_classified_ranking_basis(self):
        records = []
        for action_id, effect_id in (("A0", "E4"), ("A1", "E14")):
            records.append({
                "specialist": "rawlsian",
                "canonical_action_id": action_id,
                "grounded_effect_ids": [effect_id],
                "ranking_basis": "BASIC_INTEREST_SECURITY",
                "dimension": "BASIC_INTEREST_SECURITY",
                "institutional_relation": "NATURAL_CONTINGENCY",
            })
        candidate = self._candidate(
            "rawlsian",
            committed_native_ledger=self._native(
                "RAWLSIAN_POSITION_LEDGER", records,
            ),
            framework_numerical_role="DECISIVE",
            material_empirical_claims=self._derivation(
                "The larger affected headcount determines the Rawlsian winner",
                ["E4", "E14"],
                "COMPARISON",
                operation="ARITHMETIC",
                calculation="compare sixty residents with twenty patients",
            ),
        )
        decision = apply_framework_vote_integrity(candidate, self.actions)
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertIn("decisive aggregation is not licensed", decision.reason)

    def test_deontological_priority_requires_a_classified_duty(self):
        records = [
            {
                "specialist": "deontological", "canonical_action_id": "A0",
                "grounded_effect_ids": ["E4"], "verdict": "REQUIRED",
                "duty_type": "UNRESOLVED",
            },
            {
                "specialist": "deontological", "canonical_action_id": "A1",
                "grounded_effect_ids": ["E14"], "verdict": "PROHIBITED",
                "duty_type": "PERFECT_NEGATIVE",
            },
        ]
        candidate = self._candidate(
            "deontological",
            committed_native_ledger=self._native(
                "DEONTOLOGICAL_DUTY_LEDGER", records,
            ),
            material_empirical_claims=self._derivation(
                "Compare the two admitted action effects before classifying duty",
                ["E4", "E14"],
                "COMPARISON",
                operation="QUALITATIVE_COMPARISON",
                calculation="compare admitted effects before duty priority",
            ),
        )
        decision = apply_framework_vote_integrity(candidate, self.actions)
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertIn("lacks a classified duty type", decision.reason)

    def test_rejected_and_uncertain_ledgers_never_receive_full_vote_weight(self):
        records = self._effect_records("utilitarian", "E4", "E14")
        rejected = self._candidate(
            "utilitarian",
            committed_native_ledger=self._native(
                "UTILITARIAN_CONSEQUENCE_LEDGER", records, "REJECTED",
            ),
        )
        rejected_decision = apply_framework_vote_integrity(rejected, self.actions)
        self.assertEqual(rejected_decision.status, "ABSTAIN")
        self.assertEqual(rejected.policy_weight_factor, 0.0)
        self.assertFalse(rejected.governing_eligible)

        uncertain = self._candidate(
            "utilitarian",
            committed_native_ledger=self._native(
                "UTILITARIAN_CONSEQUENCE_LEDGER", records,
                "COMMITTED_WITH_UNCERTAINTY",
            ),
            material_empirical_claims=self._derivation(
                "Compare the admitted trapping and hospital effects",
                ["E4", "E14"],
                "COMPARISON",
                operation="QUALITATIVE_COMPARISON",
                calculation="compare admitted polarity and conditional modality",
            ),
        )
        uncertain_decision = apply_framework_vote_integrity(uncertain, self.actions)
        self.assertEqual(uncertain_decision.status, "ATTENUATED")
        self.assertEqual(uncertain.policy_weight_factor, 0.5)
        self.assertFalse(uncertain.governing_eligible)


class FrozenReplayAndPerformanceTests(unittest.TestCase):
    def _trace_payload(self) -> dict[str, object]:
        scenario = (
            "A coordinator must choose one plan. Plan A sends help to the east. "
            "Plan B sends help to the west."
        )
        actions = ["Help the east", "Help the west"]
        ref = (SourceRef("C0", scenario),)
        east_ref = (
            SourceRef("C1", "Plan A sends help to the east."),
            SourceRef("A0", actions[0]),
        )
        west_ref = (
            SourceRef("C2", "Plan B sends help to the west."),
            SourceRef("A1", actions[1]),
        )
        effects = (
            WorldEffect(
                "E0", "A0", "P1", "RECEIVES_HELP", "RECEIVES",
                "BENEFICIAL", "DIRECT", "CERTAIN", "RESOURCE_TRANSFER",
                provenance=east_ref,
            ),
            WorldEffect(
                "E1", "A1", "P2", "RECEIVES_HELP", "RECEIVES",
                "BENEFICIAL", "DIRECT", "CERTAIN", "RESOURCE_TRANSFER",
                provenance=west_ref,
            ),
        )
        world = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "coordinator", "PERSON", ref),
                WorldParty("P1", "east", "GROUP", ref),
                WorldParty("P2", "west", "GROUP", ref),
            ),
            actions=(
                WorldAction("A0", actions[0], "P0", ("P1",), ("E0",), east_ref),
                WorldAction("A1", actions[1], "P0", ("P2",), ("E1",), west_ref),
            ),
            effects=effects,
            admission=WorldStateAdmission(
                status="COMMITTED", admitted_effect_ids=("E0", "E1"),
            ),
        )
        records = [
            {
                "action_id": f"A{index}",
                "canonical_semantic_action": action,
                "world_effects": [effects[index].as_dict()],
            }
            for index, action in enumerate(actions)
        ]
        return {
            "scenario": scenario,
            "presentation_actions": actions,
            "actions": actions,
            "source_action_legend": {"A0": actions[0], "A1": actions[1]},
            "presentation_action_mapping": [],
            "canonical_action_records": records,
            "action_source_grounding": {
                "status": "COMMITTED",
                "world_model_status": "COMMITTED",
                "world_contradictions": [],
                "actions": {"A0": {}, "A1": {}},
                "clauses": [{"clause_id": "C0", "text": scenario}],
                "world_model": world.as_dict(),
            },
        }

    def test_frozen_trace_loads_deterministically_and_rejects_scenario_drift(self):
        payload = self._trace_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            first = load_frozen_world_trace(
                path, expected_scenario=str(payload["scenario"]),
            )
            second = load_frozen_world_trace(
                path, expected_scenario=str(payload["scenario"]),
            )
            self.assertEqual(first.fingerprint, second.fingerprint)
            self.assertEqual(first.metadata()["world_generation_calls"], 0)
            with self.assertRaisesRegex(FrozenWorldReplayError, "exactly match"):
                load_frozen_world_trace(
                    path, expected_scenario=str(payload["scenario"]) + " ",
                )

    def test_frozen_trace_rejects_incomplete_effect_admission(self):
        payload = self._trace_payload()
        payload["action_source_grounding"]["world_model"]["admission"][
            "admitted_effect_ids"
        ] = ["E0"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(FrozenWorldReplayError, "exactly"):
                load_frozen_world_trace(
                    path, expected_scenario=str(payload["scenario"]),
                )

    def test_performance_trace_counts_calls_without_recording_content(self):
        class StructuredLlm:
            def complete_json(self, prompt, **kwargs):
                return {"choices": [{"text": '{"ok":true}'}]}

        recorder, token = start_performance_trace("test-run")
        try:
            with performance_stage("test_stage"):
                call_json_llm(
                    StructuredLlm(),
                    "sensitive prompt",
                    max_tokens=20,
                    temperature=0.0,
                    schema={"type": "object"},
                    call_kind="test_call",
                )
            snapshot = recorder.snapshot()
        finally:
            reset_performance_trace(token)
        self.assertEqual(snapshot["counts"]["logical_model_calls"], 1)
        model_event = next(
            item for item in snapshot["events"]
            if item["category"] == "model_call"
        )
        self.assertEqual(model_event["metadata"]["parent_stage"], "test_stage")
        self.assertNotIn("sensitive prompt", json.dumps(snapshot))
        self.assertEqual(snapshot["content_recording"], "DISABLED")

    def test_performance_summary_counts_repairs_timeouts_and_abstentions(self):
        class StructuredLlm:
            def complete_json(self, prompt, **kwargs):
                return {"choices": [{"text": '{}'}]}

        recorder, token = start_performance_trace("detailed-counts")
        try:
            for call_kind in ("compact_primary", "compact_repair"):
                call_json_llm(
                    StructuredLlm(),
                    "content must not be retained",
                    max_tokens=20,
                    temperature=0.0,
                    schema={"type": "object"},
                    call_kind=call_kind,
                    call_metadata={"specialist": "care"},
                )
            record_performance_duration(
                "cycle_execution",
                0.5,
                metadata={"cycle": 1, "abstention_count": 2},
            )
            record_performance_duration(
                "original_agent_process",
                1.0,
                category="original_agent",
                status="TIMEOUT",
                metadata={"agent": "rawlsian", "timeout_count": 1},
            )
            record_performance_duration(
                "sequential_ledger_commit",
                0.03,
                category="ledger_commit",
                metadata={"specialist": "care"},
            )
            snapshot = recorder.snapshot()
        finally:
            reset_performance_trace(token)
        counts = snapshot["counts"]
        self.assertEqual(counts["compact_primary_calls"], 1)
        self.assertEqual(counts["compact_repair_calls"], 1)
        self.assertEqual(counts["repair_count"], 1)
        self.assertEqual(counts["timeout_count"], 1)
        self.assertEqual(counts["abstention_count"], 2)
        self.assertEqual(
            snapshot["timings_by_name_seconds"]["sequential_ledger_commit"],
            0.03,
        )
        self.assertNotIn("content must not be retained", json.dumps(snapshot))


class RagOffShortCircuitTests(unittest.TestCase):
    AGENTS = (
        ("utilitarian_agent_p", "retrieve_utilitarian_quotes", "UTILITARIAN_QUERY_LENS"),
        ("deontological_agent_p", "retrieve_deontological_quotes", "DEONTOLOGY_QUERY_LENS"),
        ("virtue_ethics_agent_p", "retrieve_virtue_ethics_quotes", "VIRTUE_QUERY_LENS"),
        ("care_ethics_agent_p", "retrieve_care_ethics_quotes", "CARE_QUERY_LENS"),
        ("rawlsian_ethics_agent_p", "retrieve_rawlsian_ethics_quotes", "RAWLS_QUERY_LENS"),
    )

    def test_all_original_agents_skip_tags_corpora_and_embeddings_when_rag_is_off(self):
        question = "A coordinator must choose Action A or Action B."

        class CapturingLlm:
            def __init__(self):
                self.prompt = ""

            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return {"choices": [{"text": "stable testimony"}]}

        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            scenario_path = Path(directory) / "rag_off_case.json"
            scenario_path.write_text(
                json.dumps({"ethical_question": question}), encoding="utf-8"
            )
            for module_name, retrieval_name, lens_name in self.AGENTS:
                with self.subTest(agent=module_name):
                    module = importlib.import_module(module_name)
                    llm = CapturingLlm()
                    with ExitStack() as stack:
                        stack.enter_context(patch.dict(os.environ, {
                            "ETHICS_DISABLE_RAG": "1",
                            "ETHICS_LLM_BACKEND": "openai",
                            "ETHICS_REUSE_SOURCE_TESTIMONY": "0",
                        }))
                        stack.enter_context(patch.object(
                            module,
                            "LAST_QUERY_PATH",
                            Path(directory) / f"{module_name}.query",
                        ))
                        stack.enter_context(patch.object(
                            module,
                            "LAST_RESPONSE_PATH",
                            Path(directory) / f"{module_name}.response",
                        ))
                        tag_loader = stack.enter_context(patch.object(
                            module, "load_scenario_weights",
                            side_effect=AssertionError("tag model must stay lazy"),
                        ))
                        corpus_loader = stack.enter_context(patch.object(
                            module, "load_corpus_passages",
                            side_effect=AssertionError("corpus must not load"),
                        ))
                        corpus_fingerprint = stack.enter_context(patch.object(
                            module, "corpus_fingerprint",
                            side_effect=AssertionError("corpus must not be fingerprinted"),
                        ))
                        retrieval = stack.enter_context(patch.object(
                            module, retrieval_name,
                            side_effect=AssertionError("retrieval path must short-circuit"),
                        ))
                        embedding_constructor = stack.enter_context(patch(
                            "langchain_huggingface.HuggingFaceEmbeddings",
                            side_effect=AssertionError("embedding constructor must not run"),
                        ))
                        testimony = module.respond_to_query(
                            question,
                            scenario_path.stem,
                            scenario_path=scenario_path,
                            llm=llm,
                            max_tokens=32,
                        )

                    tag_loader.assert_not_called()
                    corpus_loader.assert_not_called()
                    corpus_fingerprint.assert_not_called()
                    retrieval.assert_not_called()
                    embedding_constructor.assert_not_called()
                    expected_result = disabled_retrieval_result(
                        question, getattr(module, lens_name),
                    )
                    self.assertEqual(
                        module.LAST_RETRIEVAL,
                        serialize_retrieval_result(expected_result, mode="disabled"),
                    )
                    self.assertIn(
                        "[RAG_CONTEXT_UNAVAILABLE] No corpus passage met",
                        llm.prompt,
                    )
                    # These are the pre-existing per-agent response wrappers;
                    # the short-circuit changes no prompt testimony semantics.
                    expected_testimony = (
                        "stable testimony\n"
                        if module_name == "virtue_ethics_agent_p"
                        else "stable testimony\n[/INST]\n</s>"
                        if module_name in {
                            "care_ethics_agent_p", "rawlsian_ethics_agent_p",
                        }
                        else "stable testimony"
                    )
                    self.assertEqual(testimony, expected_testimony)

    def test_semantic_tag_model_is_lazy_even_when_module_is_imported(self):
        import get_semantic_tag

        with patch("sentence_transformers.SentenceTransformer") as constructor:
            importlib.reload(get_semantic_tag)
        constructor.assert_not_called()

    def test_short_circuit_preserves_legacy_rag_off_prompts_and_testimonies(self):
        """Compare with the bridge hook's former typed-empty retrieval behavior."""
        question = "A coordinator must choose Action A or Action B."

        class CapturingLlm:
            def __init__(self):
                self.prompt = ""

            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return {"choices": [{"text": "stable testimony"}]}

        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            scenario_path = Path(directory) / "rag_off_equivalence.json"
            scenario_path.write_text(
                json.dumps({"ethical_question": question}), encoding="utf-8"
            )
            for module_name, retrieval_name, lens_name in self.AGENTS:
                with self.subTest(agent=module_name):
                    module = importlib.import_module(module_name)
                    expected_result = disabled_retrieval_result(
                        question, getattr(module, lens_name),
                    )
                    expected_context = format_evidence_context(expected_result)
                    disabled_llm = CapturingLlm()
                    baseline_llm = CapturingLlm()
                    common_environment = {
                        "ETHICS_LLM_BACKEND": "openai",
                        "ETHICS_REUSE_SOURCE_TESTIMONY": "0",
                    }
                    with patch.object(
                        module, "LAST_QUERY_PATH",
                        Path(directory) / f"{module_name}.query",
                    ), patch.object(
                        module, "LAST_RESPONSE_PATH",
                        Path(directory) / f"{module_name}.response",
                    ):
                        with patch.dict(os.environ, {
                            **common_environment, "ETHICS_DISABLE_RAG": "1",
                        }):
                            disabled_testimony = module.respond_to_query(
                                question,
                                scenario_path.stem,
                                scenario_path=scenario_path,
                                llm=disabled_llm,
                                max_tokens=32,
                            )

                        legacy_retrieval = (
                            (expected_context, [], expected_result)
                            if module_name in {
                                "deontological_agent_p", "virtue_ethics_agent_p",
                            }
                            else (expected_context, [])
                        )
                        with patch.dict(os.environ, {
                            **common_environment, "ETHICS_DISABLE_RAG": "0",
                        }), patch.object(
                            module, "corpus_fingerprint", return_value="legacy-hook",
                        ), patch.object(
                            module, retrieval_name, return_value=legacy_retrieval,
                        ), patch(
                            "langchain_huggingface.HuggingFaceEmbeddings",
                            return_value=object(),
                        ):
                            baseline_testimony = module.respond_to_query(
                                question,
                                scenario_path.stem,
                                scenario_path=scenario_path,
                                llm=baseline_llm,
                                max_tokens=32,
                            )

                    self.assertEqual(disabled_llm.prompt, baseline_llm.prompt)
                    self.assertEqual(disabled_testimony, baseline_testimony)


if __name__ == "__main__":
    unittest.main()
