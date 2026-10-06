from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import contextlib
import io

import parsing_game_Z10 as z10
from blueprint_primitive_composition import extract_inventory, compose, append_composition
from blueprint_graph_amendments import expand_candidates
from blueprint_proposal_contract import validate_proposal
from test_blueprint_graph_amendments import TEXT, blueprint
from run_blueprint_parliament import main, _admit_candidates, DEFAULT_PARLIAMENT_ROOT, DEFAULT_PARLIAMENT_PYTHON


class CompositionTests(unittest.TestCase):
    def test_baseline_matches_without_macro_dispatch(self):
        package = z10.export_candidate_graph(TEXT)
        inventory = extract_inventory(TEXT, package)
        original = blueprint(TEXT)
        before = deepcopy(original)
        with mock.patch("blueprint_cloze_chooser._graph", side_effect=AssertionError("macro dispatch")):
            result = append_composition(TEXT, inventory, original)
        self.assertEqual(original, before)
        self.assertEqual(result["candidate_attempts"][0], before["candidate_attempts"][0])
        self.assertTrue(result["primitive_comparison"][0]["effect_readings_match"])
        self.assertTrue(result["primitive_comparison"][0]["topology_matches"])
        self.assertEqual(validate_proposal(result["candidate_attempts"][-1]["proposal"]), [])
        choices = [r for r in inventory["constructions"] if r["type"] == "choice"]
        self.assertEqual(choices[0]["selection_rule"], "any_subset")
        self.assertFalse(choices[0]["exhaustive"])
        self.assertEqual({o["scope"]["polarity"] for o in choices[0]["options"]}, {"positive", "negative"})
        sources = {c["id"] for c in inventory["supporting_candidates"]}
        self.assertTrue(all(set(c["requires"]) <= sources for c in inventory["supporting_candidates"]))
        for evidence in inventory["evidence"]:
            self.assertEqual(TEXT[evidence["start"]:evidence["end"]], evidence["text"])

    def test_extra_state_is_one_added_effect_not_a_new_family(self):
        text = TEXT + "If Maria pulls the lever, the warning signal will stop."
        package = z10.export_candidate_graph(text)
        inventory = extract_inventory(text, package)
        result = append_composition(text, inventory, expand_candidates(text, package, blueprint(text), inventory))
        composed = result["candidate_attempts"][-1]["proposal"]
        world = composed["candidate"]["world_model"]
        self.assertEqual(len(world["actions"]), 2)
        self.assertEqual(len(world["effects"]), 7)
        self.assertEqual(sum(e["predicate"] == "stop" for e in world["effects"]), 1)
        self.assertTrue(next(r for r in result["primitive_comparison"] if r["proposal_id"].endswith("_amended"))["topology_matches"])
        self.assertFalse(result["primitive_comparison"][0]["effect_readings_match"])
        self.assertEqual(composed["exclusivity_proof"]["status"], "UNKNOWN")

    def test_pilot_limit_does_not_change_existing_candidates(self):
        text = TEXT + "If Maria presses the button, two workers will survive."
        inventory = extract_inventory(text, z10.export_candidate_graph(text))
        result = append_composition(text, inventory, blueprint(text))
        self.assertEqual(result["candidate_attempts"][-1]["proposal"]["status"], "WITHHELD")
        self.assertEqual(result["candidate_attempts"][0], blueprint(text)["candidate_attempts"][0])

    @unittest.skipUnless(DEFAULT_PARLIAMENT_PYTHON.exists() and DEFAULT_PARLIAMENT_ROOT.exists(), "native checkout unavailable")
    def test_native_admission_both_paths_baseline_and_state(self):
        for text, count in ((TEXT, 6), (TEXT + "If Maria pulls the lever, the warning signal will stop.", 7)):
            with self.subTest(effects=count), tempfile.TemporaryDirectory() as tmp:
                package = z10.export_candidate_graph(text)
                inventory = extract_inventory(text, package)
                result = append_composition(text, inventory, expand_candidates(text, package, blueprint(text), inventory))
                chosen, records = _admit_candidates(result, text, package, Path(tmp), DEFAULT_PARLIAMENT_ROOT, DEFAULT_PARLIAMENT_PYTHON)
                self.assertIsNotNone(chosen, records)
                composed = next(r for r in records if r.get("variant") == "composition")
                self.assertEqual(composed["status"], "ADMITTED", composed)
                self.assertEqual(composed["admission"]["effects"], count)
                self.assertTrue(composed["admission"]["frozen_trace_valid"])
                self.assertTrue(all(r["status"] == "ADMITTED" for r in records), records)

    @unittest.skipUnless(DEFAULT_PARLIAMENT_PYTHON.exists() and DEFAULT_PARLIAMENT_ROOT.exists(), "native checkout unavailable")
    def test_runner_writes_inventory_comparison_and_all_graphs(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("run_blueprint_parliament.choose_by_cloze", return_value=blueprint(TEXT)), mock.patch("run_blueprint_parliament.openai_complete"), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--text", TEXT, "--prepare-only", "--output-dir", tmp]), 0)
            root = Path(tmp)
            self.assertTrue((root / "primitive_inventory.json").exists())
            self.assertTrue(json.loads((root / "primitive_comparison.json").read_text())[0]["topology_matches"])
            self.assertIn("primitive_composition", (root / "candidate_graphs.md").read_text())


if __name__ == "__main__":
    unittest.main()
