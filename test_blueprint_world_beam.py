import unittest

from blueprint_world_beam import (
    COLLAPSE_INVARIANT,
    COLLAPSE_SEPARATED,
    COLLAPSE_UNRESOLVED,
    collapse_admitted,
    graph_identity,
    normalize_weights,
    occurrence_signature,
    raw_support,
    slot_hypothesis,
)


def _admitted(blueprint_id, rank, assignment, effects, *, variant="baseline",
              core_filled=5, template_status="FILLED"):
    proposal = {
        "assignment": assignment,
        "candidate": {"world_model": {
            "actions": [{"intervention": name} for name in assignment],
            "parties": [{"party_id": "P1", "kind": "PERSON"}],
            "effects": effects,
        }},
    }
    attempt = {
        "blueprint_id": blueprint_id, "rank": rank, "variant": variant,
        "template_status": template_status, "core_filled": core_filled,
        "optional_filled": 0, "proposal": proposal,
    }
    return {
        "proposal": proposal, "attempt": attempt,
        "record": {"status": "ADMITTED", "blueprint_id": blueprint_id,
                   "rank": rank, "variant": variant},
        "overlay": {"hypotheses": []},
    }


def _effect(relation, polarity="ADVERSE", directness="DOWNSTREAM"):
    return {"party_id": "P1", "relation": relation, "polarity": polarity,
            "directness": directness}


class WorldBeamTests(unittest.TestCase):
    def test_slot_hypothesis_records_span_type_and_copy_status(self):
        row = slot_hypothesis(
            text="Ada can give the medicine to Ben.",
            slot="assignment", check="assignment", span="give the medicine",
            producer="cloze_copy", relent_compatibility="copy_ok",
        )
        self.assertEqual(row["source_span"], "give the medicine")
        self.assertEqual(row["semantic_type"], "assignment")
        self.assertEqual(row["relent_compatibility"], "copy_ok")
        self.assertEqual(row["provenance"], "SOURCE_ASSERTED")
        self.assertEqual(row["producer"], "cloze_copy")
        self.assertTrue(row["clause_ids"])

    def test_close_admitted_graphs_stay_a_beam(self):
        first = _admitted("b1", 1, ["act"], [_effect("DIE")])
        second = _admitted("b2", 2, ["other"], [_effect("LIVE", "BENEFICIAL")])
        result = collapse_admitted([first, second])
        self.assertEqual(result["collapse_status"], COLLAPSE_UNRESOLVED)
        self.assertIsNone(result["chosen"])
        weights = [row["relative_interpretive_support"] for row in result["members"]]
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertTrue(all(weight > 0 for weight in weights))

    def test_rank0_exclusive_separates_from_flexible_fallback(self):
        exclusive = _admitted("exclusive_allocation", 0, ["give to Ben"], [_effect("DIE")])
        flexible = _admitted(
            "flexible_graph", 3, ["give to Cara"], [_effect("LIVE", "BENEFICIAL")],
            variant="flexible", core_filled=0,
        )
        result = collapse_admitted([exclusive, flexible])
        self.assertEqual(result["collapse_status"], COLLAPSE_SEPARATED)
        self.assertEqual(result["chosen"]["attempt"]["blueprint_id"], "exclusive_allocation")
        self.assertEqual(result['collapse_reason'], 'SOLE_NONFALLBACK')
        self.assertEqual([r['relative_interpretive_support'] for r in result['members']], [1.0])
        self.assertFalse(result['noncompeting_members'][0]['competing'])
        self.assertIsNone(result['noncompeting_members'][0]['relative_interpretive_support'])
        self.assertGreaterEqual(
            exclusive_support := raw_support(exclusive["attempt"], "ADMITTED"),
            2 * raw_support(flexible["attempt"], "ADMITTED"),
            exclusive_support,
        )

    def test_invariant_occurrence_graphs_collapse_to_preferred_fill(self):
        effects = [_effect("DIE")]
        first = _admitted("exclusive_allocation", 0, ["give to Ben"], effects)
        second = _admitted("omission_harm", 1, ["give to Ben"], effects)
        self.assertEqual(occurrence_signature(first["proposal"]),
                         occurrence_signature(second["proposal"]))
        result = collapse_admitted([first, second])
        self.assertEqual(result["collapse_status"], COLLAPSE_INVARIANT)
        self.assertEqual(result["chosen"]["attempt"]["blueprint_id"], "exclusive_allocation")

    def test_identity_key_ignores_optional_process_wording(self):
        self.assertEqual(
            graph_identity("exclusive_allocation", ["give to Ben"], False),
            graph_identity("exclusive_allocation", ["give to Ben"], False),
        )

    def test_single_admitted_world_separates(self):
        only = _admitted("deontic_rule", 0, ["must deliver water"], [_effect("OBLIGATION")])
        result = collapse_admitted([only])
        self.assertEqual(result["collapse_status"], COLLAPSE_SEPARATED)
        self.assertEqual(result["chosen"]["attempt"]["blueprint_id"], "deontic_rule")

    def test_same_interventions_do_not_hide_amended_effects(self):
        first = _admitted("conditional_outcome", 0, ["pulls the lever"], [_effect("DIE")])
        second = _admitted(
            "conditional_outcome", -0.1, ["pulls the lever"],
            [_effect("DIE"), _effect("STOP")], variant="amended",
        )
        result = collapse_admitted([first, second])
        self.assertEqual(result["collapse_status"], COLLAPSE_UNRESOLVED)
        self.assertIsNone(result["chosen"])

    def test_normalize_weights_are_relative_support_not_empty(self):
        self.assertEqual(normalize_weights([2, 1]), [2 / 3, 1 / 3])
        self.assertEqual(normalize_weights([0, 0]), [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
