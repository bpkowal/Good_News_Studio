import json
import unittest
from pathlib import Path

from global_workspace.world_state import (
    CausalLink,
    ScenarioWorldModel,
    SourceRef,
    WorldAction,
    WorldCondition,
    WorldEffect,
    WorldParty,
    WorldTemporalRelation,
    effect_expected_qualifiers,
    extract_binary_contrast_stipulations,
    extract_quantity_bearing_consequences,
    explicit_likelihood_spans,
    parse_world_model,
    validate_world_model,
    validate_world_completeness,
)
from global_workspace.world_validation import (
    repair_guidance_cards,
    validation_issues_from_messages,
)
from global_workspace.graph_transactions import SemanticGraphStore
from global_workspace.scenario_semantics import attach_typed_world_model
from global_workspace.semantic_graph import SemanticGraph, SemanticNode
from global_workspace.utilitarian_ledger import (
    apply_utilitarian_ledger_transaction,
    utilitarian_accounting,
    utilitarian_scored_grounded_effects,
)


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "hospital_malware_compiler_regression.json"
)


class HospitalMalwareCompilerRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_rejected_candidate_now_commits_without_semantic_loss(self):
        data = self.fixture
        model = parse_world_model(
            data["world_model"],
            clauses=data["clauses"],
            action_ids=data["action_ids"],
            require_completeness=True,
        )
        by_id = {effect.effect_id: effect for effect in model.effects}
        self.assertTrue(
            set(data["expected"]["preserved_effect_ids"]) <= set(by_id)
        )
        for effect_id, outcome in data["expected"]["atomic_outcomes"].items():
            self.assertEqual(by_id[effect_id].outcome, outcome)

        self.assertEqual(by_id["E03"].quantities, ("450",))
        self.assertEqual(by_id["E03"].source_effect_ids, ("E01",))
        self.assertEqual(by_id["E15"].quantities, ("12,000",))
        self.assertEqual(by_id["E15"].polarity, "BENEFICIAL")
        self.assertEqual(by_id["E15"].modality, "CERTAIN")

        # The malware shutdown remains a background risk.  It gates the
        # action-mediated death path instead of becoming an action-caused fact.
        self.assertEqual(by_id["E16"].source_effect_ids, ())
        condition_by_id = {
            condition.condition_id: condition for condition in model.conditions
        }
        self.assertTrue(by_id["E17"].condition_ids)
        self.assertEqual(by_id["E17"].source_effect_ids, ("E11",))
        self.assertEqual(
            {
                condition_by_id[condition_id].event_effect_id
                for condition_id in by_id["E17"].condition_ids
            },
            {"E16"},
        )
        death_links = [
            link for link in model.causal_links
            if link.target_id == "E17"
        ]
        self.assertTrue(death_links)
        self.assertTrue(all(
            link.condition_ids == by_id["E17"].condition_ids
            for link in death_links
        ))
        self.assertTrue(all(
            link.modality == "STIPULATED_CONDITIONAL"
            for link in death_links
        ))
        errors = validate_world_completeness(
            model,
            action_ids=data["action_ids"],
            source_texts=[row["text"] for row in data["clauses"]],
        )
        self.assertEqual(errors, [])

    def test_mixed_benefit_and_risk_are_atomic(self):
        text = self.fixture["clauses"][-1]["text"]
        stipulations = extract_binary_contrast_stipulations([text])
        observed = {
            (item.side_cue.casefold(), item.consequence_span.casefold()):
            item.polarity
            for item in stipulations
        }
        self.assertEqual(
            observed[("refrain", "data that will save 12,000 future patients")],
            "BENEFICIAL",
        )
        self.assertEqual(
            observed[("refrain", "risking immediate deaths today")],
            "ADVERSE",
        )
        quantities = extract_quantity_bearing_consequences([text])
        future = next(
            item for item in quantities if "12,000" in item.quantity_spans
        )
        self.assertEqual(
            future.consequence_span,
            "data that will save 12,000 future patients",
        )
        self.assertEqual(future.polarity, "BENEFICIAL")

    def test_background_population_risk_is_not_an_action_consequence(self):
        extracted = extract_quantity_bearing_consequences([
            "Site B holds a trapped family of a dozen residents facing acute risk."
        ])
        self.assertEqual(extracted, ())

    def test_coordinated_quantity_keeps_its_local_polarity(self):
        extracted = extract_quantity_bearing_consequences([
            "A1 diverts the surge, submerging forty farms and ruining the "
            "livelihoods of three hundred residents, but protecting the plant."
        ])
        livelihoods = next(
            item for item in extracted if "three hundred" in item.quantity_spans
        )
        self.assertEqual(
            livelihoods.consequence_span,
            "ruining the livelihoods of three hundred residents",
        )
        self.assertEqual(livelihoods.polarity, "ADVERSE")

    def test_atomic_temporal_qualifiers_do_not_cross_contrast_segments(self):
        data = self.fixture
        model = parse_world_model(
            data["world_model"],
            clauses=data["clauses"],
            action_ids=data["action_ids"],
            require_completeness=True,
        )
        by_id = {effect.effect_id: effect for effect in model.effects}
        self.assertNotIn("future", by_id["E03"].temporal_qualifiers)
        self.assertNotIn("future", by_id["F0"].temporal_qualifiers)
        self.assertIn("future", by_id["E15"].temporal_qualifiers)
        self.assertIn("immediate", by_id["E17"].temporal_qualifiers)

    def test_atomic_segment_keeps_a_leading_likelihood_qualifier(self):
        source = (
            "A0 exposes patients to a near-certain fatal equipment failure, "
            "while A1 preserves future treatment access."
        )
        effect = WorldEffect(
            effect_id="E_TEST",
            action_id="A0",
            party_id="P_TEST",
            outcome="fatal equipment failure",
            relation="FAILS",
            polarity="ADVERSE",
            directness="DOWNSTREAM",
            modality="PROBABILISTIC",
            effect_kind="HEALTH_OUTCOME",
            provenance=(SourceRef("C_TEST", source),),
            source_proposition="fatal equipment failure",
        )
        self.assertEqual(
            effect_expected_qualifiers(effect, explicit_likelihood_spans),
            ("near-certain",),
        )

    def test_current_schema_rejects_unsynchronized_conditional_edge(self):
        ref = (SourceRef(
            "C_TEST", "If the backup fails, patients are harmed."
        ),)
        model = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "operator", "HUMAN", ref),
                WorldParty("P1", "patients", "POPULATION", ref),
            ),
            actions=(WorldAction(
                "A0", "wait", "P0", ("P1",), ("E0", "E1"), ref,
            ),),
            effects=(
                WorldEffect(
                    "E0", "A0", "P0", "waits", "PERFORMS",
                    "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION",
                    provenance=ref,
                ),
                WorldEffect(
                    "E1", "A0", "P1", "patients are harmed", "HARMS",
                    "ADVERSE", "DOWNSTREAM", "STIPULATED_CONDITIONAL",
                    "HEALTH_OUTCOME", condition_ids=("COND0",),
                    provenance=ref, source_effect_ids=("E0",),
                ),
            ),
            conditions=(WorldCondition(
                "COND0", "the backup fails", provenance=ref,
            ),),
            causal_links=(CausalLink(
                "E0", "CAUSES", "E1", "CERTAIN", (), ref, "A0",
            ),),
            schema_version="1.3",
        )
        errors, _issues = validate_world_model(model, action_ids=["A0"])
        self.assertTrue(any(
            "omits target effect conditions" in error for error in errors
        ))
        self.assertTrue(any(
            "unconditional but target effect" in error for error in errors
        ))

    def test_missing_terminal_effect_guidance_is_downstream(self):
        issues = validation_issues_from_messages([
            "A0 omits source-stipulated outcome 'the survival of 450 current "
            "patients' from binary contrast; admit a matching effect (or "
            "UNRESOLVED quarantine)"
        ])
        cards = repair_guidance_cards(
            issues,
            {"world_model": {"effects": [], "actions": [{"action_id": "A0"}]}},
            clauses=self.fixture["clauses"],
        )
        additions = [
            patch["value"]
            for patch in cards[0]["concrete_patches"]
            if patch.get("op") == "add_effect"
        ]
        self.assertTrue(additions)
        self.assertTrue(all(
            item["directness"] == "DOWNSTREAM" for item in additions
        ))

    def test_exception_gate_projects_as_not_operand(self):
        ref = (SourceRef("C_TEST", "Pressure rises unless the pump activates."),)
        model = ScenarioWorldModel(
            parties=(WorldParty("P0", "pressure", "PROCESS", ref),),
            actions=(WorldAction("A0", "leave valve open", "P0", (), ("E0",), ref),),
            effects=(WorldEffect(
                "E0", "A0", "P0", "pressure rises", "INCREASES",
                "ADVERSE", "DIRECT", "STIPULATED_CONDITIONAL", "PHYSICAL_STATE",
                condition_ids=("COND0",), provenance=ref,
            ),),
            conditions=(WorldCondition(
                "COND0", "backup pump activates", provenance=ref,
                polarity="NEGATED", operator="UNLESS",
            ),),
            schema_version="1.0",
        )
        graph = SemanticGraph()
        graph.add_node(SemanticNode(
            "A0", "ACTION", "leave valve open",
            attributes={"canonical_action_id": "A0"},
        ))
        attach_typed_world_model(graph, model)
        consequence = graph.nodes["A0:WORLD_EFFECT:E0"]
        self.assertEqual(consequence.attributes["condition_ids"], ["COND0"])
        gate = list(graph.outgoing(consequence.id, "CONDITIONAL_ON"))[0].target
        self.assertEqual(graph.nodes[gate].attributes["operator"], "NOT")
        self.assertEqual(graph.nodes["COND0"].attributes["operator"], "UNLESS")

    def test_temporal_relations_project_and_cycles_are_rejected(self):
        ref = (SourceRef("C_TEST", "E0 occurs before E1."),)
        base = ScenarioWorldModel(
            parties=(WorldParty("P0", "system", "PROCESS", ref),),
            actions=(WorldAction("A0", "operate", "P0", (), ("E0", "E1"), ref),),
            effects=(
                WorldEffect("E0", "A0", "P0", "starts", "STARTS", "NEUTRAL", "DIRECT", "CERTAIN", provenance=ref),
                WorldEffect("E1", "A0", "P0", "stops", "STOPS", "NEUTRAL", "DOWNSTREAM", "CERTAIN", provenance=ref),
            ),
            temporal_relations=(WorldTemporalRelation(
                "T0", "E0", "BEFORE", "E1", ref,
            ),),
            schema_version="1.0",
        )
        graph = SemanticGraph()
        graph.add_node(SemanticNode(
            "A0", "ACTION", "operate", attributes={"canonical_action_id": "A0"},
        ))
        attach_typed_world_model(graph, base)
        self.assertTrue(any(edge.relation == "BEFORE" for edge in graph.edges))
        reversed_model = ScenarioWorldModel(
            parties=base.parties, actions=base.actions, effects=base.effects,
            temporal_relations=(
                *base.temporal_relations,
                WorldTemporalRelation("T1", "E1", "BEFORE", "E0", ref),
            ),
            schema_version="1.0",
        )
        errors, _ = validate_world_model(reversed_model, action_ids=["A0"])
        self.assertTrue(any("temporal" in error for error in errors))

    def _admitted_graph(self):
        data = self.fixture
        model = parse_world_model(
            data["world_model"],
            clauses=data["clauses"],
            action_ids=data["action_ids"],
            require_completeness=True,
        )
        graph = SemanticGraph()
        for action in model.actions:
            graph.add_node(SemanticNode(
                action.action_id,
                "ACTION",
                action.intervention,
                attributes={"canonical_action_id": action.action_id},
            ))
        attach_typed_world_model(graph, model)
        return model, graph

    def test_guaranteed_survival_remains_a_utilitarian_benefit(self):
        _model, graph = self._admitted_graph()
        self.assertEqual(
            utilitarian_accounting(graph, "A0:WORLD_EFFECT:E03"),
            ("BENEFIT", "CERTAIN"),
        )

    def test_semantic_causal_edge_retains_the_world_gate(self):
        _model, graph = self._admitted_graph()
        links = [
            edge for edge in graph.edges
            if edge.source == "A1:WORLD_EFFECT:E11"
            and edge.target == "A1:WORLD_EFFECT:E17"
            and edge.relation == "CAUSES"
        ]
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].condition, "CND2")

    def test_effect_valuation_transaction_commits_the_admitted_world(self):
        _model, graph = self._admitted_graph()
        valuations = {action_id: [] for action_id in self.fixture["action_ids"]}
        for effect in utilitarian_scored_grounded_effects(graph):
            valuations[effect.action_id].append({
                "effect_id": effect.effect_id,
                "importance": "HIGH",
                "reason": "value the admitted effect without changing direction",
            })
        store = SemanticGraphStore(graph)
        record = apply_utilitarian_ledger_transaction(
            store,
            {"actions": [
                {"action_id": action_id, "valuations": valuations[action_id]}
                for action_id in self.fixture["action_ids"]
            ]},
            cycle=1,
            specialist="utilitarian",
            allowed_actions=tuple(self.fixture["action_ids"]),
        )
        self.assertEqual(record.status, "COMMITTED", record.errors)
        committed = {
            item["grounded_effect_id"]: item
            for item in record.proposal["committed_consequences"]
        }
        self.assertEqual(committed["E03"]["direction"], "BENEFIT")
        self.assertEqual(committed["E03"]["polarity"], "BENEFICIAL")
        self.assertEqual(
            committed["E03"]["grounded_world_polarity"], "BENEFICIAL",
        )

    def test_derived_averted_risk_is_unknown_without_rejecting_ledger(self):
        _model, graph = self._admitted_graph()
        effect_id = "A0:WORLD_EFFECT:E03"
        effect = graph.nodes[effect_id]
        graph.nodes[effect_id] = SemanticNode(
            effect.id,
            effect.kind,
            effect.label,
            effect.provenance,
            {
                **effect.attributes,
                "derivation_operation": "RISK_PREVENTION",
            },
        )
        self.assertEqual(
            utilitarian_accounting(graph, effect_id),
            ("UNKNOWN", "CERTAIN"),
        )

        valuations = {action_id: [] for action_id in self.fixture["action_ids"]}
        for projected in utilitarian_scored_grounded_effects(graph):
            valuations[projected.action_id].append({
                "effect_id": projected.effect_id,
                "importance": "HIGH",
                "reason": "value the admitted effect without changing direction",
            })
        store = SemanticGraphStore(graph)
        record = apply_utilitarian_ledger_transaction(
            store,
            {"actions": [
                {"action_id": action_id, "valuations": valuations[action_id]}
                for action_id in self.fixture["action_ids"]
            ]},
            cycle=1,
            specialist="utilitarian",
            allowed_actions=tuple(self.fixture["action_ids"]),
        )
        self.assertEqual(record.status, "COMMITTED", record.errors)
        committed = {
            item["grounded_effect_id"]: item
            for item in record.proposal["committed_consequences"]
        }
        self.assertEqual(committed["E03"]["direction"], "UNKNOWN")
        self.assertEqual(committed["E03"]["polarity"], "UNKNOWN")
        self.assertEqual(
            committed["E03"]["grounded_world_polarity"], "BENEFICIAL",
        )


if __name__ == "__main__":
    unittest.main()
