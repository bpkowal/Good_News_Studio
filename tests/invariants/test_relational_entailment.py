"""RELATIONAL_ENTAILMENT over RelEnt relation algebra."""
from __future__ import annotations

import unittest

from hypothesis import given, settings

from global_workspace.relent_adapt import (
    averted_alternative_relational_errors,
    directionality_relational_errors,
    identity_relational_errors,
    project_alternative_of_edges,
    project_averted_derived_edges,
    project_before_edges,
    project_causes_edges,
    project_same_entity_edges,
)
from relent.algebra import (
    RelEdge,
    closure_edges,
    derived_only,
    directionality_errors,
    forbidden_composites,
    function_transfer_errors,
    relational_entailment_errors,
)
from relent.relations import (
    RELATION_SPECS,
    RELATION_TAGS,
    relation_properties,
    status_transfers,
)
from strategies.averted_alternative_harm import (
    AvertedAlternativeHarmCase,
    averted_alternative_harm_cases,
)
from strategies.directionality import (
    DirectionalityCase,
    directionality_cases,
)
from strategies.identity_closure import (
    IdentityClosureCase,
    identity_closure_cases,
)
from strategies.relational_algebra import (
    RelationalAlgebraCase,
    relational_algebra_cases,
)


class RelationSpecTableTests(unittest.TestCase):
    def test_every_tag_has_a_spec(self):
        for tag in RELATION_TAGS:
            spec = relation_properties(tag)
            self.assertIsNotNone(spec, tag)
            self.assertEqual(spec.tag, tag)
            self.assertIn(tag, RELATION_SPECS)

    def test_identity_is_rst(self):
        spec = relation_properties("SAME_ENTITY_AS")
        self.assertTrue(spec.reflexive)
        self.assertTrue(spec.symmetric)
        self.assertEqual(spec.transitive, True)

    def test_alternative_is_symmetric_not_transitive(self):
        spec = relation_properties("ALTERNATIVE_OF")
        self.assertFalse(spec.reflexive)
        self.assertTrue(spec.symmetric)
        self.assertEqual(spec.transitive, False)
        self.assertTrue(spec.action_scoped)

    def test_causes_restricted_transitivity(self):
        spec = relation_properties("CAUSES")
        self.assertFalse(spec.symmetric)
        self.assertEqual(spec.transitive, "restricted")

    def test_status_transfers_still_gated(self):
        self.assertTrue(status_transfers("PARAPHRASE_OF"))
        self.assertFalse(status_transfers("APPROXIMATES"))
        self.assertFalse(status_transfers("DERIVED_FROM"))
        self.assertTrue(status_transfers("DERIVED_FROM", derived_marked=True))


class RelationalAlgebraTests(unittest.TestCase):
    def test_identity_closure_a_b_c(self):
        edges = (
            RelEdge("A", "B", "SAME_ENTITY_AS"),
            RelEdge("B", "C", "EQUIVALENT_TO"),
        )
        closed = closure_edges(edges)
        keys = {(e.source, e.target, e.relation) for e in closed}
        self.assertIn(("A", "C", "SAME_ENTITY_AS"), keys)
        self.assertIn(("B", "A", "SAME_ENTITY_AS"), keys)
        self.assertEqual(
            relational_entailment_errors(
                edges,
                expect_derived=(RelEdge("A", "C", "SAME_ENTITY_AS"),),
            ),
            [],
        )

    def test_alternative_symmetry(self):
        edges = (RelEdge("A0", "A1", "ALTERNATIVE_OF"),)
        derived = derived_only(edges)
        self.assertTrue(
            any(
                e.source == "A1" and e.target == "A0" and e.relation == "ALTERNATIVE_OF"
                for e in derived
            )
        )

    def test_causes_does_not_auto_transit(self):
        edges = (
            RelEdge("A", "B", "CAUSES"),
            RelEdge("B", "C", "CAUSES"),
        )
        derived = derived_only(edges)
        self.assertFalse(
            any(e.source == "A" and e.target == "C" for e in derived)
        )
        self.assertEqual(
            relational_entailment_errors(
                edges,
                forbid_derived=(RelEdge("A", "C", "CAUSES"),),
            ),
            [],
        )

    def test_alternative_plus_identity_is_forbidden(self):
        edges = (
            RelEdge("A0", "A1", "ALTERNATIVE_OF"),
            RelEdge("A0", "A1", "SAME_ENTITY_AS"),
        )
        errors = forbidden_composites(edges)
        self.assertTrue(errors)
        self.assertIn("RELATIONAL_NON_TRANSFER", errors[0])

    def test_quantity_transfer_policies(self):
        self.assertEqual(
            function_transfer_errors("SAME_ENTITY_AS", "quantity"),
            [],
        )
        self.assertTrue(
            function_transfer_errors("ALTERNATIVE_OF", "quantity")
        )
        self.assertEqual(
            function_transfer_errors(
                "ALTERNATIVE_OF", "quantity", derivation_marked=True,
            ),
            [],
        )
        self.assertTrue(
            function_transfer_errors(
                "ALTERNATIVE_OF",
                "harm_under_action",
                source_action="A1",
                target_action="A0",
            )
        )


class AvertedRelationalAdapterTests(unittest.TestCase):
    @given(averted_alternative_harm_cases())
    @settings(max_examples=30, deadline=None)
    def test_alternative_of_projects_symmetric_closure(
        self, case: AvertedAlternativeHarmCase,
    ):
        edges = project_alternative_of_edges(case.world)
        self.assertTrue(edges, msg=case.mutation)
        closed = closure_edges(edges)
        keys = {(e.source, e.target, e.relation) for e in closed}
        self.assertIn(("A0", "A1", "ALTERNATIVE_OF"), keys)
        self.assertIn(("A1", "A0", "ALTERNATIVE_OF"), keys)

    @given(
        averted_alternative_harm_cases().filter(
            lambda c: c.mutation == "derived_averted",
        ),
    )
    @settings(max_examples=20, deadline=None)
    def test_derived_averted_quantity_transfer_is_licensed(
        self, case: AvertedAlternativeHarmCase,
    ):
        derived = project_averted_derived_edges(case.world)
        self.assertTrue(
            any(e.relation == "DERIVED_FROM" and e.derived_marked for e in derived),
            case.mutation,
        )
        errors = averted_alternative_relational_errors(case.world)
        self.assertEqual(errors, [], msg=errors)

    @given(
        averted_alternative_harm_cases().filter(
            lambda c: c.mutation == "silent_copy_survival",
        ),
    )
    @settings(max_examples=20, deadline=None)
    def test_silent_copy_is_relational_non_transfer(
        self, case: AvertedAlternativeHarmCase,
    ):
        errors = averted_alternative_relational_errors(case.world)
        self.assertTrue(errors, msg=case.mutation)
        self.assertTrue(
            any(
                "RELATIONAL_FUNCTION_TRANSFER" in e or "RELATIONAL_NON_TRANSFER" in e
                for e in errors
            ),
            errors,
        )


class IdentityClosureAdapterTests(unittest.TestCase):
    @given(
        identity_closure_cases().filter(
            lambda c: c.mutation in {"alias_pair", "quantity_along_identity"},
        ),
    )
    @settings(max_examples=20, deadline=None)
    def test_label_alias_projects_symmetric_identity(
        self, case: IdentityClosureCase,
    ):
        edges = project_same_entity_edges(
            case.world, licensed_edges=case.licensed_identity_edges,
        )
        self.assertTrue(edges, msg=case.mutation)
        closed = closure_edges(edges)
        keys = {(e.source, e.target) for e in closed if e.relation == "SAME_ENTITY_AS"}
        for left, right in case.expect_closed_pairs:
            self.assertIn((left, right), keys, msg=case.mutation)

    @given(
        identity_closure_cases().filter(lambda c: c.mutation == "chain_three"),
    )
    @settings(max_examples=15, deadline=None)
    def test_chain_closes_a_to_c(self, case: IdentityClosureCase):
        edges = project_same_entity_edges(
            case.world, licensed_edges=case.licensed_identity_edges,
        )
        closed = closure_edges(edges)
        keys = {(e.source, e.target) for e in closed if e.relation == "SAME_ENTITY_AS"}
        self.assertIn(("P_a", "P_c"), keys)
        self.assertIn(("P_c", "P_a"), keys)
        errors = identity_relational_errors(
            case.world, licensed_identity_edges=case.licensed_identity_edges,
        )
        self.assertEqual(errors, [], msg=errors)

    @given(
        identity_closure_cases().filter(
            lambda c: c.mutation == "quantity_along_identity",
        ),
    )
    @settings(max_examples=15, deadline=None)
    def test_quantity_may_transfer_along_identity(
        self, case: IdentityClosureCase,
    ):
        self.assertTrue(case.expect_quantity_transfer_ok)
        self.assertEqual(
            function_transfer_errors("SAME_ENTITY_AS", "quantity"),
            [],
        )
        errors = identity_relational_errors(
            case.world, licensed_identity_edges=case.licensed_identity_edges,
        )
        self.assertEqual(errors, [], msg=errors)

    @given(
        identity_closure_cases().filter(
            lambda c: c.mutation == "harm_cross_branch",
        ),
    )
    @settings(max_examples=15, deadline=None)
    def test_harm_cross_branch_is_non_transfer(
        self, case: IdentityClosureCase,
    ):
        self.assertFalse(case.expect_harm_cross_action_ok)
        self.assertTrue(
            function_transfer_errors(
                "SAME_ENTITY_AS",
                "harm_under_action",
                source_action="A1",
                target_action="A0",
            )
        )
        errors = identity_relational_errors(
            case.world, licensed_identity_edges=case.licensed_identity_edges,
        )
        self.assertTrue(errors, msg=case.mutation)
        self.assertTrue(
            any("RELATIONAL_NON_TRANSFER" in e for e in errors),
            errors,
        )


class IdentityClosureHypothesisTests(unittest.TestCase):
    @given(identity_closure_cases())
    @settings(max_examples=40, deadline=None)
    def test_oracle_agrees_with_adapter(self, case: IdentityClosureCase):
        errors = identity_relational_errors(
            case.world, licensed_identity_edges=case.licensed_identity_edges,
        )
        if case.expect_relational_errors:
            self.assertTrue(errors, msg=case.mutation)
        else:
            self.assertEqual(errors, [], msg=f"{case.mutation}: {errors}")
        closed = {
            (e.source, e.target)
            for e in closure_edges(project_same_entity_edges(
                case.world, licensed_edges=case.licensed_identity_edges,
            ))
            if e.relation in {"SAME_ENTITY_AS", "EQUIVALENT_TO"}
        }
        for left, right in case.expect_closed_pairs:
            self.assertIn((left, right), closed, msg=case.mutation)


class DirectionalityAdapterTests(unittest.TestCase):
    @given(
        directionality_cases().filter(
            lambda c: c.mutation in {"causes_chain_no_skip", "causes_no_reverse"},
        ),
    )
    @settings(max_examples=15, deadline=None)
    def test_causes_projects_without_reverse(self, case: DirectionalityCase):
        edges = project_causes_edges(case.world)
        self.assertTrue(edges, msg=case.mutation)
        derived = derived_only(edges)
        self.assertFalse(
            any(
                e.relation == "CAUSES" and e.source == "E_b" and e.target == "E_a"
                for e in derived
            ),
            case.mutation,
        )
        self.assertEqual(directionality_errors(edges), [])

    @given(
        directionality_cases().filter(lambda c: c.mutation == "causes_chain_no_skip"),
    )
    @settings(max_examples=15, deadline=None)
    def test_causes_chain_does_not_skip(self, case: DirectionalityCase):
        edges = project_causes_edges(case.world)
        derived = derived_only(edges)
        self.assertFalse(
            any(
                e.relation == "CAUSES" and e.source == "E_a" and e.target == "E_c"
                for e in derived
            ),
            case.mutation,
        )
        self.assertEqual(
            directionality_relational_errors(case.world),
            [],
            msg=case.mutation,
        )

    @given(
        directionality_cases().filter(lambda c: c.mutation == "before_transits"),
    )
    @settings(max_examples=15, deadline=None)
    def test_before_chain_transits(self, case: DirectionalityCase):
        edges = project_before_edges(
            case.world, licensed_edges=case.licensed_before_edges,
        )
        closed = closure_edges(edges)
        keys = {(e.source, e.target) for e in closed if e.relation == "BEFORE"}
        self.assertIn(("T0", "T2"), keys)
        self.assertNotIn(("T1", "T0"), keys)
        self.assertEqual(
            directionality_relational_errors(
                case.world, licensed_before_edges=case.licensed_before_edges,
            ),
            [],
        )

    @given(
        directionality_cases().filter(
            lambda c: c.mutation == "before_antisymmetric",
        ),
    )
    @settings(max_examples=15, deadline=None)
    def test_before_antisymmetric_base_is_rejected(
        self, case: DirectionalityCase,
    ):
        errors = directionality_relational_errors(
            case.world, licensed_before_edges=case.licensed_before_edges,
        )
        self.assertTrue(errors, msg=case.mutation)
        self.assertTrue(
            any("RELATIONAL_DIRECTIONALITY" in e for e in errors),
            errors,
        )


class DirectionalityHypothesisTests(unittest.TestCase):
    @given(directionality_cases())
    @settings(max_examples=40, deadline=None)
    def test_oracle_agrees_with_adapter(self, case: DirectionalityCase):
        errors = directionality_relational_errors(
            case.world,
            licensed_causes_edges=case.licensed_causes_edges,
            licensed_before_edges=case.licensed_before_edges,
        )
        if case.expect_relational_errors:
            self.assertTrue(errors, msg=case.mutation)
        else:
            self.assertEqual(errors, [], msg=f"{case.mutation}: {errors}")
        if case.expect_before_transit:
            closed = {
                (e.source, e.target)
                for e in closure_edges(project_before_edges(
                    case.world, licensed_edges=case.licensed_before_edges,
                ))
                if e.relation == "BEFORE"
            }
            self.assertIn(("T0", "T2"), closed)


class RelationalAlgebraHypothesisTests(unittest.TestCase):
    @settings(max_examples=50, deadline=None)
    @given(relational_algebra_cases())
    def test_oracle_agrees_with_algebra(self, case: RelationalAlgebraCase):
        errors = relational_entailment_errors(
            case.edges,
            expect_derived=case.expect_derived,
            forbid_derived=case.forbid_derived,
        )
        transfer_errors = function_transfer_errors(
            case.transfer_relation,
            case.transfer_kind,  # type: ignore[arg-type]
            derivation_marked=case.transfer_derivation_marked,
            source_action=case.transfer_source_action,
            target_action=case.transfer_target_action,
        )
        if case.expect_transfer_ok:
            self.assertEqual(transfer_errors, [], msg=case.mode)
        else:
            self.assertTrue(transfer_errors, msg=case.mode)
        if case.expect_errors:
            self.assertTrue(errors, msg=f"{case.mode}: {errors}")
        else:
            self.assertEqual(errors, [], msg=f"{case.mode}: {errors}")


if __name__ == "__main__":
    unittest.main()
