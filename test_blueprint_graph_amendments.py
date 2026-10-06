from copy import deepcopy
import json
from pathlib import Path
import tempfile
import contextlib
import io
import unittest
from unittest import mock

import parsing_game_Z10 as z10
from blueprint_cloze_chooser import _graph, assess_question
from blueprint_graph_amendments import conditional_inventory, expand_candidates, merge_world
from run_blueprint_parliament import _admit_candidates, main, DEFAULT_PARLIAMENT_ROOT, DEFAULT_PARLIAMENT_PYTHON


TEXT = ("Maria can pull the lever or press the button. "
        "If Maria pulls the lever, one worker will die. "
        "If Maria does not pull the lever, five workers will die. ")


def blueprint(text):
    slots = {"actor": "Maria", "condition": "If Maria pulls the lever",
             "bearer": "one worker", "outcome": "one worker will die",
             "second_condition": "If Maria does not pull the lever",
             "second_bearer": "five workers", "second_outcome": "five workers will die"}
    question = assess_question(text)
    seed = {"blueprint_id": "conditional_outcome", "slots": slots,
            "accepted_evidence": slots, "items": []}
    proposal = _graph(text, seed, question)
    return {"question": question, "proposals": [proposal], "chosen_blueprint_id": "conditional_outcome",
            "status": "FILLED", "candidate_attempts": [{"rank": 0, "blueprint_id": "conditional_outcome",
            "selected": True, "template_status": "FILLED", "contract_valid": True,
            "unfilled_slots": [], "proposal": proposal}]}


class AmendmentTests(unittest.TestCase):
    @unittest.skipUnless(DEFAULT_PARLIAMENT_PYTHON.exists() and DEFAULT_PARLIAMENT_ROOT.exists(),
                         "native Parliament integration checkout unavailable")
    def test_runner_prepares_amended_candidate_without_model_call(self):
        text = TEXT + "If Maria presses the button, two workers will survive."
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("run_blueprint_parliament.choose_by_cloze", return_value=blueprint(text)), \
                mock.patch("run_blueprint_parliament.openai_complete"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--text", text, "--prepare-only", "--output-dir", tmp]), 0)
            manifest = json.loads((Path(tmp) / "preparation_manifest.json").read_text())
            self.assertEqual(manifest["pipeline_status"], "ADMITTED")
            self.assertEqual(manifest["selected_variant"], "amended")
            self.assertEqual(manifest["parliament_admission"]["effects"], 9)
            self.assertTrue((Path(tmp) / "world_state_topology.md").exists())
            self.assertTrue((Path(tmp) / "amendment_inventory.json").exists())

    @unittest.skipUnless(DEFAULT_PARLIAMENT_PYTHON.exists() and DEFAULT_PARLIAMENT_ROOT.exists(),
                         "native Parliament integration checkout unavailable")
    def test_native_admission_baseline_and_expansion_preserve_supported_effects(self):
        for text, expected in ((TEXT, 6), (TEXT + "If Maria presses the button, two workers will survive.", 9)):
            with self.subTest(effects=expected), tempfile.TemporaryDirectory() as tmp:
                package = z10.export_candidate_graph(text)
                result = expand_candidates(text, package, blueprint(text))
                chosen, records = _admit_candidates(result, text, package, Path(tmp),
                                                    DEFAULT_PARLIAMENT_ROOT, DEFAULT_PARLIAMENT_PYTHON)
                self.assertIsNotNone(chosen, records)
                self.assertEqual(chosen["record"]["admission"]["effects"], expected)
                trace = json.loads(Path(chosen["record"]["trace_path"]).read_text())
                effects = trace["action_source_grounding"]["world_model"]["effects"]
                self.assertFalse(any(e["derivation_operation"] == "AVERTED_ALTERNATIVE_HARM" for e in effects))
                self.assertTrue(chosen["record"]["admission"]["frozen_trace_valid"])

    def test_original_and_existing_topology_unchanged_when_covered(self):
        original = blueprint(TEXT)
        before = deepcopy(original)
        result = expand_candidates(TEXT, z10.export_candidate_graph(TEXT), original)
        self.assertEqual(original, before)
        self.assertEqual(result["candidate_attempts"][0], original["candidate_attempts"][0])
        self.assertFalse(any(a.get("variant") == "amended" for a in result["candidate_attempts"]))

    def test_third_branch_adds_nodes_contiguous_actions_and_evidence(self):
        text = TEXT + "If Maria presses the button, two workers will survive."
        original = blueprint(text)
        result = expand_candidates(text, z10.export_candidate_graph(text), original)
        amended = next(a for a in result["candidate_attempts"] if a.get("variant") == "amended")
        world = amended["proposal"]["candidate"]["world_model"]
        self.assertEqual([a["action_id"] for a in world["actions"]], ["A0", "A1", "A2"])
        self.assertEqual(len(world["effects"]), 9)
        self.assertEqual(len(world["causal_links"]), 6)
        extra = next(e for e in world["effects"] if e["outcome"] == "two workers will survive")
        self.assertTrue(extra["clause_ids"])
        self.assertEqual(extra["quantities"], ["two"])
        self.assertEqual(result["candidate_attempts"][0], original["candidate_attempts"][0])

    def test_same_action_additional_party_reuses_action_and_process(self):
        text = TEXT + "If Maria pulls the lever, Ben will die."
        result = expand_candidates(text, z10.export_candidate_graph(text), blueprint(text))
        world = next(a for a in result["candidate_attempts"] if a.get("variant") == "amended")["proposal"]["candidate"]["world_model"]
        self.assertEqual(len(world["actions"]), 2)
        self.assertEqual(len(world["effects"]), 7)
        self.assertEqual(sum(p["label"].casefold() == "the lever" for p in world["parties"]), 1)

    def test_none_option_can_use_source_backed_fallback(self):
        question = assess_question(TEXT)
        result = expand_candidates(TEXT, z10.export_candidate_graph(TEXT), {
            "question": question, "proposals": [], "candidate_attempts": [],
            "chosen_blueprint_id": "none", "status": "WITHHELD"})
        self.assertEqual(result["chosen_blueprint_id"], "flexible_graph")
        self.assertEqual(len(result["proposals"][0]["candidate"]["world_model"]["actions"]), 2)

    def test_unrepresented_predicate_stays_visible(self):
        text = "If Maria pulls the lever, the warning signal will blink."
        branches, unresolved = conditional_inventory(text, z10.export_candidate_graph(text))
        self.assertEqual(branches, [])
        self.assertTrue(unresolved)

    def test_process_state_reuses_branch_and_preserves_original(self):
        text = TEXT + "If Maria pulls the lever, the warning signal will stop."
        original = blueprint(text)
        before = deepcopy(original)
        result = expand_candidates(text, z10.export_candidate_graph(text), original)
        amended = next(a for a in result["candidate_attempts"] if a.get("variant") == "amended")
        world = amended["proposal"]["candidate"]["world_model"]
        self.assertEqual(original, before)
        self.assertEqual(len(world["actions"]), 2)
        self.assertEqual(len(world["effects"]), 7)
        state = next(e for e in world["effects"] if e["predicate"] == "stop")
        self.assertEqual(state["effect_kind"], "PHYSICAL_STATE")
        self.assertEqual(state["modality"], "CERTAIN")
        self.assertEqual(state["source_proposition"], "the warning signal will stop")
        self.assertEqual(state["polarity"], "NEUTRAL")
        self.assertEqual(next(p for p in world["parties"] if p["party_id"] == state["party_id"])["kind"], "PROCESS")

    def test_uncertain_or_negated_process_state_is_not_promoted(self):
        for outcome in ("the warning signal may stop", "the warning signal will probably stop",
                        "the warning signal will not stop"):
            text = "If Maria pulls the lever, " + outcome + "."
            with self.subTest(outcome=outcome):
                branches, unresolved = conditional_inventory(text, z10.export_candidate_graph(text))
                self.assertEqual(branches, [])
                self.assertTrue(unresolved)

    def test_process_only_flexible_candidate_has_no_generic_placeholder(self):
        text = "If Maria pulls the lever, the warning signal will stop."
        result = expand_candidates(text, z10.export_candidate_graph(text), {
            "question": assess_question(text), "proposals": [], "candidate_attempts": [],
            "chosen_blueprint_id": "none", "status": "WITHHELD"})
        effects = result["proposals"][0]["candidate"]["world_model"]["effects"]
        self.assertEqual(sum(e["predicate"] == "stop" for e in effects), 1)
        self.assertFalse(any(e["predicate"] == "outcome" for e in effects))

    @unittest.skipUnless(DEFAULT_PARLIAMENT_PYTHON.exists() and DEFAULT_PARLIAMENT_ROOT.exists(),
                         "native Parliament integration checkout unavailable")
    def test_native_process_amendment_and_original_both_admit(self):
        text = TEXT + "If Maria pulls the lever, the warning signal will stop."
        package = z10.export_candidate_graph(text)
        result = expand_candidates(text, package, blueprint(text))
        with tempfile.TemporaryDirectory() as tmp:
            chosen, records = _admit_candidates(result, text, package, Path(tmp),
                                               DEFAULT_PARLIAMENT_ROOT, DEFAULT_PARLIAMENT_PYTHON)
            self.assertIsNotNone(chosen, records)
            self.assertEqual(chosen["record"]["variant"], "amended")
            self.assertEqual(chosen["record"]["admission"]["effects"], 7)
            self.assertTrue(all(r["status"] == "ADMITTED" for r in records))
            trace = json.loads(Path(chosen["record"]["trace_path"]).read_text())
            effects = trace["action_source_grounding"]["world_model"]["effects"]
            self.assertEqual(sum(e["outcome"] == "the warning signal will stop" for e in effects), 1)
            self.assertEqual(sum("will die" in e["outcome"] for e in effects), 2)

    def test_negative_consequence_is_not_reinterpreted_as_positive_harm(self):
        text = "If Maria pulls the lever, Ben will not die."
        branches, unresolved = conditional_inventory(text, z10.export_candidate_graph(text))
        self.assertEqual(branches, [])
        self.assertTrue(unresolved)

    def test_generic_merge_retains_extra_relations(self):
        world = {"parties": [], "actions": [], "effects": [], "conditions": [],
                 "causal_links": [], "temporal_relations": [], "counterfactual_links": []}
        extra = deepcopy(world)
        extra["temporal_relations"] = [{"source_id": "X", "target_id": "Y", "relation": "BEFORE"}]
        merged, changes = merge_world(world, extra)
        self.assertEqual(merged["temporal_relations"], extra["temporal_relations"])
        self.assertEqual(world["temporal_relations"], [])


if __name__ == "__main__":
    unittest.main()
