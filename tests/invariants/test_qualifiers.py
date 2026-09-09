"""Qualifier-head binding over Hypothesis-generated shared clauses.

The oracle is QualifierHeadCase.owner_id. Production is only asked whether
that span binds to the owner, is filled/stripped by parse, omit-checked on
admission, and copied onto the graph from the bound tuples.
"""
from __future__ import annotations

import unittest

from dataclasses import replace

from hypothesis import given, settings

from global_workspace.scenario_semantics import attach_typed_world_model
from global_workspace.semantic_graph import SemanticGraph, SemanticNode
from global_workspace.world_state import (
    _closed_class_qualifiers,
    _supporting_source_is_hedged,
    effect_expected_qualifiers,
    explicit_likelihood_spans,
    explicit_scope_spans,
    explicit_temporal_spans,
    parse_world_model,
    validate_world_completeness,
    validate_world_model,
    world_model_as_parse_payload,
)
from strategies.qualifiers import QualifierHeadCase, qualifier_head_cases


_EXTRACTORS = {
    "LIKELIHOOD": explicit_likelihood_spans,
    "SCOPE": explicit_scope_spans,
    "TEMPORAL": explicit_temporal_spans,
}


def _admission_errors(world) -> list[str]:
    structural, _contradictions = validate_world_model(world, action_ids=["A0"])
    complete = validate_world_completeness(world, action_ids=["A0"])
    return [*structural, *complete]


def _recorded(effect, field: str) -> tuple[str, ...]:
    return getattr(effect, field)


class QualifierHeadBindingTests(unittest.TestCase):
    @given(qualifier_head_cases())
    @settings(max_examples=50, deadline=None)
    def test_qualifier_binds_only_the_modified_head(self, case: QualifierHeadCase):
        extractor = _EXTRACTORS[case.kind]
        self.assertEqual(
            effect_expected_qualifiers(case.owner, extractor),
            (case.span,),
        )
        self.assertEqual(
            effect_expected_qualifiers(case.sibling, extractor),
            (),
        )
        self.assertEqual(
            _recorded(_closed_class_qualifiers(case.owner_without_copy), case.field),
            (case.span,),
        )
        self.assertEqual(
            _recorded(_closed_class_qualifiers(case.sibling_with_leak), case.field),
            (),
        )
        self.assertEqual(_admission_errors(case.world), [])
        omitted = _admission_errors(replace(
            case.world,
            effects=tuple(
                case.owner_without_copy if effect.effect_id == case.owner_id
                else effect
                for effect in case.world.effects
            ),
        ))
        self.assertTrue(any(
            case.owner_id in error
            and "omits" in error
            and case.span in error
            for error in omitted
        ))
        self.assertFalse(any(
            case.sibling_id in error
            and "omits" in error
            and case.span in error
            for error in omitted
        ))
        if case.kind == "LIKELIHOOD":
            self.assertTrue(_supporting_source_is_hedged(case.owner_without_copy))
            self.assertFalse(_supporting_source_is_hedged(case.sibling))

        payload = world_model_as_parse_payload(case.world)
        for row in payload["effects"]:
            if row["effect_id"] == case.owner_id:
                row[case.field] = []
            if row["effect_id"] == case.sibling_id:
                row[case.field] = [case.span]
        parsed = parse_world_model(
            payload,
            clauses=[{"clause_id": "C0", "text": case.source}],
            action_ids=["A0"],
        )
        parsed_owner = next(
            effect for effect in parsed.effects if effect.effect_id == case.owner_id
        )
        parsed_sibling = next(
            effect for effect in parsed.effects if effect.effect_id == case.sibling_id
        )
        self.assertEqual(_recorded(parsed_owner, case.field), (case.span,))
        self.assertEqual(_recorded(parsed_sibling, case.field), ())

        graph = SemanticGraph()
        graph.add_node(SemanticNode(
            "A0", "ACTION", parsed.actions[0].intervention, ("test",), {},
        ))
        attach_typed_world_model(graph, parsed)
        owner_node = graph.nodes[f"A0:WORLD_EFFECT:{case.owner_id}"]
        sibling_node = graph.nodes[f"A0:WORLD_EFFECT:{case.sibling_id}"]
        self.assertEqual(
            tuple(owner_node.attributes.get(case.field, ())),
            (case.span,),
        )
        self.assertEqual(
            tuple(sibling_node.attributes.get(case.field, ())),
            (),
        )
