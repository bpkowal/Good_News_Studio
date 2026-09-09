"""Habitat consumption invariants over Hypothesis-generated labels.

Oracles are declared case kinds. Production is only asked whether chance
counted as obtained IMPROVES or WORSENS, whether unsettled or FOREGONE
rows rendered as obtained facts, whether polarity counts ignored modality,
or whether averted risk counted as obtained benefit.
"""
from __future__ import annotations

import re
import unittest

from hypothesis import given, settings

from global_workspace.epistemic_ledger import seed_proposition_ledger
from global_workspace.presentation import (
    _factual_status_lines,
    _row_is_settled_world_fact,
)
from global_workspace.scenario_semantics import (
    compile_scenario_graph,
    project_grounded_action_effects,
)
from global_workspace.semantic_state import (
    _action_signature,
    project_authoritative_semantic_state,
)
from global_workspace.utilitarian_ledger import utilitarian_accounting
from global_workspace.world_state import project_world_action_roles
from strategies.consumption import (
    ChanceOutcomeCase,
    ForegoneObtainedCase,
    ModalityBlindCostCase,
    PresentationModalityCase,
    chance_outcome_cases,
    foregone_obtained_cases,
    modality_blind_cost_cases,
    presentation_modality_cases,
)
from strategies.worlds import AvertedRiskCase, averted_risk_cases


def _established_block(lines: list[str]) -> list[str]:
    established_block: list[str] = []
    in_established = False
    for line in lines:
        if line.startswith("**Established or derived"):
            in_established = True
            continue
        if line.startswith("**") and in_established:
            break
        if in_established:
            established_block.append(line)
    return established_block


def _world_consequence(graph, effect_id: str):
    for node in graph.nodes.values():
        if (
            node.kind == "CONSEQUENCE"
            and str(node.attributes.get("world_effect_id", "")) == effect_id
        ):
            return node
    return None


def _compile_averted(case: AvertedRiskCase):
    return compile_scenario_graph(
        case.scenario, case.actions, world_model=case.world.as_dict(),
    )


class ChanceOutcomeTests(unittest.TestCase):
    @given(chance_outcome_cases())
    @settings(max_examples=40, deadline=None)
    def test_hedged_chance_is_not_an_obtained_outcome(
        self, case: ChanceOutcomeCase,
    ):
        effects = [
            effect for effect in project_grounded_action_effects(case.graph)
            if effect.consequence_id == "A0:C"
        ]
        self.assertTrue(effects)
        obtained = any(
            effect.direction == case.obtained_direction for effect in effects
        )
        signature = _action_signature(case.graph, "A0")
        count_key = (
            "beneficial_count" if case.polarity == "BENEFICIAL" else "adverse_count"
        )
        counted = signature.get(count_key, 0) >= 1
        self.assertEqual(obtained, case.should_count_as_obtained)
        self.assertEqual(counted, case.should_count_as_obtained)
        if not case.should_count_as_obtained:
            self.assertNotIn(
                effects[0].direction, {"IMPROVES", "WORSENS", "PRESERVES"},
            )


class PresentationModalityTests(unittest.TestCase):
    @given(presentation_modality_cases())
    @settings(max_examples=40, deadline=None)
    def test_unsettled_row_is_not_listed_as_bare_established_fact(
        self, case: PresentationModalityCase,
    ):
        lines = _factual_status_lines(
            {"proposition_ledger": list(case.ledger_rows)},
            [],
        )
        outcome = case.claim.split(";")[0].casefold()
        listed = any(
            outcome in line.casefold() for line in _established_block(lines)
        )
        self.assertEqual(listed, case.should_list_as_established_fact)


class ForegoneObtainedTests(unittest.TestCase):
    @given(foregone_obtained_cases())
    @settings(max_examples=40, deadline=None)
    def test_foregone_overlay_is_not_listed_as_an_obtained_event(
        self, case: ForegoneObtainedCase,
    ):
        lines = _factual_status_lines(
            {"proposition_ledger": list(case.ledger_rows)},
            [],
        )
        listed = any(
            case.outcome in line.casefold()
            for line in _established_block(lines)
        )
        self.assertEqual(listed, case.should_list_as_established_fact)


class ModalityBlindCostTests(unittest.TestCase):
    @given(modality_blind_cost_cases())
    @settings(max_examples=40, deadline=None)
    def test_polarity_counts_are_not_modality_blind(
        self, case: ModalityBlindCostCase,
    ):
        state = project_authoritative_semantic_state(
            case.graph, selected_action=case.actions[0],
        )
        a0_more_cost = any(
            item.relation == "ASYMMETRIC_COST"
            and item.source_action_ids
            and item.source_action_ids[0] == "A0"
            for item in state.problem_shape_relations
        )
        self.assertEqual(a0_more_cost, case.should_treat_as_more_cost)


class AvertedRiskTests(unittest.TestCase):
    @given(averted_risk_cases())
    @settings(max_examples=40, deadline=None)
    def test_averted_risk_compact_party_is_not_a_certain_beneficiary(
        self, case: AvertedRiskCase,
    ):
        graph = _compile_averted(case)
        welfare = [
            effect for effect in project_grounded_action_effects(graph)
            if str(graph.nodes[effect.consequence_id].attributes.get(
                "world_effect_id", "",
            )) == "E2"
        ]
        self.assertTrue(welfare)
        obtained = any(effect.direction == "IMPROVES" for effect in welfare)
        self.assertEqual(obtained, case.should_count_as_obtained)
        if not case.should_count_as_obtained:
            self.assertNotIn(
                welfare[0].direction, {"IMPROVES", "WORSENS", "PRESERVES"},
            )
        roles = project_world_action_roles(case.world, "A0")
        self.assertEqual(
            case.group in roles.beneficiaries,
            case.should_count_as_obtained,
        )
        if not case.should_count_as_obtained:
            self.assertIn(case.group, roles.conditionally_benefited)

    @given(averted_risk_cases())
    @settings(max_examples=40, deadline=None)
    def test_averted_risk_util_prevention_is_not_certain_benefit(
        self, case: AvertedRiskCase,
    ):
        graph = _compile_averted(case)
        welfare = _world_consequence(graph, "E2")
        self.assertIsNotNone(welfare)
        direction, probability = utilitarian_accounting(graph, welfare.id)
        certain_benefit = direction == "BENEFIT" and probability == "CERTAIN"
        self.assertEqual(certain_benefit, case.should_count_as_obtained)
        if not case.should_count_as_obtained:
            prevented = _world_consequence(graph, "E1")
            self.assertIsNotNone(prevented)
            process_direction, process_probability = utilitarian_accounting(
                graph, prevented.id,
            )
            self.assertFalse(
                process_direction == "BENEFIT"
                and process_probability == "CERTAIN"
            )

    @given(averted_risk_cases())
    @settings(max_examples=40, deadline=None)
    def test_averted_risk_presentation_is_not_established_welfare_gain(
        self, case: AvertedRiskCase,
    ):
        graph = _compile_averted(case)
        ledger = seed_proposition_ledger(graph)
        welfare = ledger.get("PROP:WORLD:E2")
        self.assertIsNotNone(welfare)
        settled = _row_is_settled_world_fact(welfare.to_dict())
        self.assertEqual(settled, case.should_count_as_obtained)
        lines = _factual_status_lines(
            {"proposition_ledger": [
                record.to_dict() for record in ledger.values()
            ]},
            [],
        )
        outcome = re.compile(rf"\b{re.escape(welfare.outcome)}\b", re.I)
        listed = any(outcome.search(line) for line in _established_block(lines))
        self.assertEqual(listed, case.should_count_as_obtained)
