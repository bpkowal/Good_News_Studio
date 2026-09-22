from __future__ import annotations

import unittest
from unittest.mock import patch

from global_workspace.local_specialists import ground_actions_in_scenario
from global_workspace.repair_experiment import (
    aggregate_repair_events,
    build_repair_event,
    graph_diff,
)


def _candidate(effect_outcome: str = "fails", *, extra: bool = False) -> dict:
    effects = [{
        "effect_id": "E1", "action_id": "A0", "party_id": "P1",
        "outcome": effect_outcome, "modality": "CERTAIN",
        "clause_ids": ["C1"],
    }]
    if extra:
        effects.append({
            "effect_id": "E2", "action_id": "A0", "party_id": "P2",
            "outcome": "is harmed", "modality": "POSSIBLE",
            "clause_ids": ["C2"],
        })
    return {"world_model": {
        "parties": [{"party_id": "P1"}, {"party_id": "P2"}],
        "actions": [{"action_id": "A0", "effect_ids": ["E1"]}],
        "effects": effects,
        "conditions": [], "temporal_relations": [],
        "causal_links": [], "counterfactual_links": [],
    }}


def _issue(code: str, entity: str = "E1") -> dict:
    return {
        "code": code, "entity_kind": "effect", "entity_id": entity,
        "field": "outcome", "message": f"{entity} has {code}",
        "repair_class": "SEMANTIC_PATCH",
    }


class RepairExperimentTests(unittest.TestCase):
    def test_graph_diff_records_local_and_collateral_changes(self) -> None:
        diff = graph_diff(_candidate(), _candidate("stops", extra=True))
        self.assertEqual(diff["collections"]["effects"]["added"], ["E2"])
        self.assertEqual(
            diff["collections"]["effects"]["changed"],
            [{"id": "E1", "fields": ["outcome"]}],
        )
        self.assertNotEqual(diff["before_fingerprint"], diff["after_fingerprint"])

    def test_event_records_cards_issue_instances_and_clean_yield(self) -> None:
        event = build_repair_event(
            event_index=1,
            execution_mode="LLM_REPAIR",
            repair_scope="LOCAL_PATCH",
            before_issues=[_issue("BROKEN")],
            after_issues=[],
            before_candidate=_candidate(),
            after_candidate=_candidate("stops"),
            repair_contract={
                "allowed_entity_ids": ["E1"],
                "guidance_cards": [{
                    "card_id": "BROKEN_V2", "card_version": 2,
                    "code": "BROKEN", "entity_id": "E1",
                }],
            },
            terminal_status="COMMITTED",
            context={"model": "test-model"},
        )
        self.assertEqual(event["selected_cards"][0]["card_id"], "BROKEN_V2")
        self.assertTrue(event["target_issues"][0]["issue_instance_id"])
        self.assertTrue(event["outcome"]["target_fixed"])
        self.assertTrue(event["outcome"]["clean_repair"])
        self.assertTrue(event["outcome"]["final_world_committed"])

    def test_event_distinguishes_target_fix_from_collateral_failure(self) -> None:
        event = build_repair_event(
            event_index=2,
            execution_mode="LLM_REPAIR",
            repair_scope="LOCAL_PATCH",
            before_issues=[_issue("BROKEN")],
            after_issues=[_issue("NEW_ERROR", "E2")],
            before_candidate=_candidate(),
            after_candidate=_candidate("stops", extra=True),
            repair_contract={
                "allowed_entity_ids": ["E1"],
                "guidance_cards": [{"code": "BROKEN", "entity_id": "E1"}],
            },
            terminal_status="REJECTED",
        )
        self.assertTrue(event["outcome"]["target_fixed"])
        self.assertFalse(event["outcome"]["clean_repair"])
        self.assertEqual(event["outcome"]["introduced_issue_codes"], ["NEW_ERROR"])
        self.assertIn("E2", event["outside_target_entity_changes"])

    def test_same_invariant_on_a_new_entity_is_collateral_not_persistent(self) -> None:
        event = build_repair_event(
            event_index=3,
            execution_mode="LLM_REPAIR",
            repair_scope="LOCAL_PATCH",
            before_issues=[_issue("BROKEN", "E1")],
            after_issues=[_issue("BROKEN", "E2")],
            before_candidate=_candidate(),
            after_candidate=_candidate("stops", extra=True),
            repair_contract={
                "allowed_entity_ids": ["E1"],
                "guidance_cards": [{"code": "BROKEN", "entity_id": "E1"}],
            },
            terminal_status="REJECTED",
        )
        self.assertTrue(event["outcome"]["target_fixed"])
        self.assertEqual(event["outcome"]["introduced_issue_codes"], ["BROKEN"])
        self.assertFalse(event["outcome"]["clean_repair"])

    def test_permitted_node_addition_is_not_automatically_collateral(self) -> None:
        event = build_repair_event(
            event_index=4,
            execution_mode="LLM_REPAIR",
            repair_scope="LOCAL_PATCH",
            before_issues=[_issue("MISSING", "E1")],
            after_issues=[],
            before_candidate=_candidate(),
            after_candidate=_candidate(extra=True),
            repair_contract={
                "allowed_entity_ids": ["E1"],
                "allowed_operations": ["add"],
                "guidance_cards": [{"code": "MISSING", "entity_id": "E1"}],
            },
            terminal_status="COMMITTED",
        )
        self.assertEqual(event["outside_target_entity_changes"], [])
        self.assertTrue(event["outcome"]["clean_repair"])

    def test_suppressed_cards_are_explicit(self) -> None:
        event = build_repair_event(
            event_index=3,
            execution_mode="DETERMINISTIC_PATCH",
            repair_scope="DETERMINISTIC_LOCAL_PATCH",
            before_issues=[_issue("BROKEN")], after_issues=[],
            before_candidate=_candidate(), after_candidate=_candidate("stops"),
            repair_contract={
                "allowed_entity_ids": ["E1"],
                "guidance_cards": [{"code": "BROKEN", "entity_id": "E1"}],
            },
            cards_presented=False,
            terminal_status="COMMITTED",
        )
        self.assertFalse(event["selected_cards"])
        self.assertEqual(event["suppressed_cards"][0]["card_id"], "BROKEN_V1")

    def test_grounding_loop_emits_explicit_llm_repair_event(self) -> None:
        before = {
            "actions": {"A0": {}, "A1": {}},
            **_candidate("fails"),
        }
        after = {
            "actions": {"A0": {}, "A1": {}},
            **_candidate("stops"),
        }
        issue = _issue("BROKEN")
        committed = {
            "status": "COMMITTED", "actions": {"A0": {}, "A1": {}},
            "errors": [], "validation_issues": [], "clauses": [],
            "world_contradictions": [],
        }

        def fake_call(_llm, _prompt, **_kwargs):
            import json
            return {"choices": [{"text": json.dumps(after)}]}

        with patch(
            "global_workspace.local_specialists._call_json_llm",
            side_effect=fake_call,
        ), patch(
            "global_workspace.local_specialists._admit_action_source_rows",
            return_value=committed,
        ):
            result = ground_actions_in_scenario(
                object(),
                "Option A: stop it. Option B: leave it running.",
                ["stop it", "leave it running"],
                max_attempts=1,
                prior_errors=["E1 has BROKEN"],
                prior_issues=[issue],
                prior_candidate=before,
            )

        self.assertEqual(result["status"], "COMMITTED")
        self.assertEqual(len(result["repair_experiment_events"]), 1)
        event = result["repair_experiment_events"][0]
        self.assertEqual(event["execution_mode"], "LLM_REPAIR")
        self.assertEqual(event["target_issues"][0]["code"], "BROKEN")
        self.assertEqual(event["selected_cards"][0]["card_id"], "BROKEN_V1")
        self.assertTrue(event["outcome"]["target_fixed"])
        self.assertTrue(event["outcome"]["final_world_committed"])

    def test_scorecards_preserve_target_and_clean_repair_distinction(self) -> None:
        clean = build_repair_event(
            event_index=1, execution_mode="LLM_REPAIR", repair_scope="LOCAL_PATCH",
            before_issues=[_issue("BROKEN")], after_issues=[],
            before_candidate=_candidate(), after_candidate=_candidate("stops"),
            repair_contract={"allowed_entity_ids": ["E1"], "guidance_cards": [{
                "card_id": "BROKEN_V1", "code": "BROKEN", "entity_id": "E1",
            }]}, terminal_status="COMMITTED",
        )
        collateral = build_repair_event(
            event_index=2, execution_mode="LLM_REPAIR", repair_scope="LOCAL_PATCH",
            before_issues=[_issue("BROKEN")], after_issues=[_issue("NEW", "E2")],
            before_candidate=_candidate(), after_candidate=_candidate("stops", extra=True),
            repair_contract={"allowed_entity_ids": ["E1"], "guidance_cards": [{
                "card_id": "BROKEN_V1", "code": "BROKEN", "entity_id": "E1",
            }]}, terminal_status="REJECTED",
        )
        report = aggregate_repair_events([[clean, collateral]])
        score = report["card_scorecards"]["BROKEN_V1"]
        self.assertEqual(score["target_repair_rate"], 1.0)
        self.assertEqual(score["clean_repair_rate"], 0.5)
        self.assertEqual(score["collateral_issue_codes"], {"NEW": 1})


if __name__ == "__main__":
    unittest.main()
