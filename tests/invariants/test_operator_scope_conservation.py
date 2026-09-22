"""OPERATOR_SCOPE_CONSERVATION over RelEnt Conditional / ExceptionRule."""
from __future__ import annotations

import unittest

from hypothesis import given, settings

from global_workspace.relent_adapt import project_operator_to_effect_gates
from relent.operators import (
    Conditional,
    Fact,
    Modal,
    Temporal,
    consequent_licensed,
    evaluate,
    operator_from_dict,
    operator_to_dict,
)
from relent.scope import (
    and_not_inheritance_errors,
    assignment_variants_for_and_not,
    build_and_not_conditional,
    build_unless_rule,
    exception_scope_errors,
    modal_scope_errors,
    operator_scope_conservation_errors,
)
from strategies.operator_scope import OperatorScopeCase, operator_scope_cases


class OperatorScopeConservationTests(unittest.TestCase):
    def test_json_round_trip(self):
        valve = Fact(entity="valve", predicate="state", state="open")
        pump = Fact(entity="backup_pump", predicate="state", state="active")
        pressure = Fact(entity="pressure", predicate="change", change="increase")
        rule = build_and_not_conditional(
            positive=valve,
            negated=pump,
            consequent=pressure,
            modality=Modal(strength="PROBABLE", body=pressure),
        )
        restored = operator_from_dict(operator_to_dict(rule))
        self.assertEqual(operator_to_dict(restored), operator_to_dict(rule))

    def test_a_and_not_b_licenses_only_matching_world(self):
        valve = Fact(entity="valve", predicate="state", state="open")
        pump = Fact(entity="backup_pump", predicate="state", state="active")
        pressure = Fact(entity="pressure", predicate="change", change="increase")
        rule = build_and_not_conditional(
            positive=valve, negated=pump, consequent=pressure,
        )
        worlds = assignment_variants_for_and_not(valve, pump)
        self.assertTrue(consequent_licensed(rule, worlds["A_and_not_B"]))
        self.assertFalse(consequent_licensed(rule, worlds["A_and_B"]))
        self.assertFalse(consequent_licensed(rule, worlds["not_A_and_not_B"]))
        self.assertEqual(
            and_not_inheritance_errors(rule, positive=valve, negated=pump),
            [],
        )

    def test_modal_drop_and_antecedent_move_are_rejected(self):
        valve = Fact(entity="valve", predicate="state", state="open")
        pump = Fact(entity="backup_pump", predicate="state", state="active")
        pressure = Fact(entity="pressure", predicate="change", change="increase")
        source = build_and_not_conditional(
            positive=valve,
            negated=pump,
            consequent=pressure,
            modality=Modal(strength="PROBABLE", body=pressure),
        )
        dropped = Conditional(if_=source.if_, then_=pressure, modality=None)
        moved = Conditional(
            if_=Modal(strength="PROBABLE", body=source.if_),
            then_=pressure,
            modality=None,
        )
        drop_errors = modal_scope_errors(source, dropped)
        move_errors = modal_scope_errors(source, moved)
        self.assertTrue(drop_errors)
        self.assertIn("drop or escalate", drop_errors[0])
        self.assertTrue(move_errors)
        self.assertTrue(any("antecedent" in err for err in move_errors))

    def test_unless_exception_blocks_consequent(self):
        valve = Fact(entity="valve", predicate="state", state="open")
        pump = Fact(entity="backup_pump", predicate="state", state="active")
        pressure = Fact(entity="pressure", predicate="change", change="increase")
        rule = build_unless_rule(
            antecedent=valve, consequent=pressure, unless=pump,
        )
        base = {valve.key(): True, pump.key(): False}
        exception = {valve.key(): True, pump.key(): True}
        self.assertEqual(
            exception_scope_errors(
                rule, base_assignment=base, exception_assignment=exception,
            ),
            [],
        )
        self.assertTrue(consequent_licensed(rule, base))
        self.assertFalse(consequent_licensed(rule, exception))

    def test_parliament_adapter_projects_condition_gates(self):
        valve = Fact(entity="valve", predicate="state", state="open")
        pump = Fact(entity="backup_pump", predicate="state", state="active")
        pressure = Fact(entity="pressure", predicate="change", change="increase")
        rule = build_and_not_conditional(
            positive=valve,
            negated=pump,
            consequent=pressure,
            modality=Modal(strength="PROBABLE", body=pressure),
        )
        gates = project_operator_to_effect_gates(rule)
        self.assertEqual(gates["condition_join"], "AND")
        self.assertEqual(gates["modality"], "PROBABILISTIC")
        self.assertTrue(gates["condition_ids"])
        self.assertTrue(gates["negated_condition_ids"])
        self.assertTrue(
            set(gates["negated_condition_ids"]) <= set(gates["condition_ids"])
        )
        self.assertIn("operator", gates)
        self.assertEqual(gates["operator"]["kind"], "Conditional")

    def test_temporal_operator_round_trip_preserves_direction(self):
        opening = Fact(entity="valve", predicate="state", state="open")
        rise = Fact(entity="pressure", predicate="change", change="increase")
        temporal = Temporal(relation="BEFORE", left=opening, right=rise)
        restored = operator_from_dict(operator_to_dict(temporal))
        self.assertEqual(restored, temporal)
        self.assertEqual(operator_to_dict(restored)["relation"], "BEFORE")


class OperatorScopeHypothesisTests(unittest.TestCase):
    @settings(max_examples=40, deadline=None)
    @given(operator_scope_cases())
    def test_oracle_agrees_with_detector(self, case: OperatorScopeCase):
        if case.mode == "unless_ok":
            valve = operator_from_dict(case.positive)
            pump = operator_from_dict(case.negated)
            assert isinstance(valve, Fact) and isinstance(pump, Fact)
            errors = operator_scope_conservation_errors(
                rule=case.rule,
                base_assignment={valve.key(): True, pump.key(): False},
                exception_assignment={valve.key(): True, pump.key(): True},
            )
        elif case.claim is not None:
            errors = operator_scope_conservation_errors(
                rule=case.rule,
                claim=case.claim,
                positive=case.positive,
                negated=case.negated,
            )
        else:
            errors = operator_scope_conservation_errors(
                rule=case.rule,
                positive=case.positive,
                negated=case.negated,
            )
        if case.expect_errors:
            self.assertTrue(errors, msg=case.mode)
            if case.error_substring:
                self.assertTrue(
                    any(case.error_substring in err for err in errors),
                    msg=f"{case.mode}: {errors}",
                )
        else:
            self.assertEqual(errors, [], msg=f"{case.mode}: {errors}")


if __name__ == "__main__":
    unittest.main()
