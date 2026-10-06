from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock
import contextlib
import io

import parsing_game_Z10 as z10
from blueprint_primitive_composition import extract_inventory, append_composition, compose
from blueprint_semantic_coverage import attach_coverage
from blueprint_proposal_contract import validate_proposal
from test_blueprint_graph_amendments import blueprint
from run_blueprint_parliament import main, DEFAULT_PARLIAMENT_ROOT, DEFAULT_PARLIAMENT_PYTHON

BASE = ("Maria must choose whether to pull the lever. "
        "If Maria pulls the lever, one worker will die. "
        "If Maria does not pull the lever, five workers will die.")
PROMISE = "Maria promised Anna to pull the lever. "


class SemanticCoverageTests(unittest.TestCase):
    def test_inventory_accounts_for_every_candidate_and_preserves_alternatives(self):
        text = PROMISE + BASE + " Ben gives Anna a token."
        package = z10.export_candidate_graph(text)
        before = deepcopy(package)
        inventory = extract_inventory(text, package)
        self.assertEqual(package, before)
        self.assertEqual(inventory["source_candidates"], package["candidates"])
        self.assertEqual(inventory["source_choice_sets"], package["choice_sets"])
        self.assertEqual(inventory["source_open_questions"], package["open_questions"])
        self.assertEqual(inventory["source_coverage"], package["coverage"])
        self.assertEqual({r["candidate_id"] for r in inventory["candidate_accounting"]},
                         {c["id"] for c in package["candidates"]})
        gaps = [r for r in inventory["candidate_accounting"] if r["status"] == "unconsumed"]
        self.assertTrue(gaps)
        self.assertTrue(all(r["question"]["candidate_ids"] == [r["candidate_id"]] for r in gaps))

    def test_promise_scope_roles_content_and_evidence_are_copied_without_resolution(self):
        for prefix in (PROMISE, "Maria did not promise Anna to pull the lever. ",
                       "If Maria promises Anna to pull the lever, Anna will survive. "):
            with self.subTest(prefix=prefix):
                text = prefix + BASE
                package = z10.export_candidate_graph(text)
                inventory = extract_inventory(text, package)
                promise = next(c for c in inventory["constructions"] if c["type"] == "promise")
                sources = {c["id"]: c for c in package["candidates"]}
                self.assertEqual(promise["projection_status"], "retained_not_projected")
                self.assertTrue(any(c["type"] == "EVENT_LINK" for c in promise["source_candidates"]))
                self.assertTrue(any(c["type"] == "PARTICIPANT" for c in promise["source_candidates"]))
                for c in promise["source_candidates"]:
                    self.assertEqual(c, sources[c["id"]])
                for e in promise["source_evidence"]:
                    self.assertEqual(text[e["start"]:e["end"]], e["text"])
                pred = next(c for c in promise["source_candidates"]
                            if c["type"] == "PREDICATION" and c["arguments"]["proposition"] == promise["anchor_id"])
                self.assertEqual(promise["scope"], pred["scope"])

    def test_coverage_annotations_do_not_change_worlds_rank_or_authorization(self):
        text = PROMISE + BASE
        inventory = extract_inventory(text, z10.export_candidate_graph(text))
        result = append_composition(text, inventory, blueprint(text))
        before = deepcopy(result)
        annotated = attach_coverage(inventory, result)
        self.assertEqual(result, before)
        for old, new in zip(before["candidate_attempts"], annotated["candidate_attempts"]):
            self.assertEqual(old["rank"], new["rank"])
            self.assertEqual(old["selected"], new["selected"])
            self.assertEqual(old["proposal"]["candidate"], new["proposal"]["candidate"])
            self.assertEqual(old["proposal"]["admission_authorized"], new["proposal"]["admission_authorized"])
            self.assertEqual(validate_proposal(new["proposal"]), [])
        composed = annotated["candidate_attempts"][-1]["proposal"]
        world = composed["candidate"]["world_model"]
        self.assertEqual(len(world["actions"]), 2)
        self.assertEqual(len(world["effects"]), 6)
        self.assertFalse(any(e["predicate"] == "promise" for e in world["effects"]))
        self.assertTrue(annotated["semantic_coverage"][-1]["retained_not_projected_construction_ids"])
        self.assertEqual(len(annotated["semantic_coverage"][-1]["source_aligned_conditional_ids"]), 2)

    def test_withheld_composition_still_retains_promise_and_questions(self):
        package = z10.export_candidate_graph(PROMISE)
        inventory = extract_inventory(PROMISE, package)
        composed = compose(PROMISE, inventory, blueprint(BASE)["question"])
        self.assertEqual(composed["status"], "WITHHELD")
        self.assertTrue(composed["unresolved_readings"][-1]["source_constructions"])
        result = attach_coverage(inventory, {"candidate_attempts": [{"proposal": composed}]})
        retention = result["candidate_attempts"][0]["proposal"]["unresolved_readings"][-1]
        self.assertTrue(retention["source_constructions"])
        self.assertEqual(retention["source_open_questions"], package["open_questions"])
        self.assertEqual(validate_proposal(result["candidate_attempts"][0]["proposal"]), [])
        self.assertEqual(sum(r.get("kind") == "semantic_construction_retention"
                             for r in result["candidate_attempts"][0]["proposal"]["unresolved_readings"]
                             if isinstance(r, dict)), 1)

    def test_mapped_promise_is_retained_in_construction_not_occurrence(self):
        from blueprint_discourse import build_promise_reliance
        package = z10.export_candidate_graph(PROMISE)
        inventory = extract_inventory(PROMISE, package)
        world, _ = build_promise_reliance(
            PROMISE,
            {"promisor": "Maria", "promisee": "Anna",
             "commitment_event": "promised",
             "commitment_content": "to pull the lever"},
        )
        proposal = {
            "proposal_id": "promise_map",
            "candidate": {"world_model": world},
            "unresolved_readings": [],
        }
        result = attach_coverage(inventory, {"candidate_attempts": [{"proposal": proposal}]})
        coverage = result["semantic_coverage"][0]
        self.assertTrue(coverage["mapped_commitment_construction_ids"])
        self.assertFalse(coverage["retained_not_projected_construction_ids"])
        constructions = result["candidate_attempts"][0]["proposal"][
            "unresolved_readings"][-1]["source_constructions"]
        self.assertEqual(constructions[0]["status"], "retained_in_construction")
        self.assertEqual(
            constructions[0]["projection_status"], "mapped_commitment_not_occurrence")
        self.assertFalse(any(
            "deliver" in (row.get("predicate") or "").casefold()
            for row in world.get("effects") or []))

    @unittest.skipUnless(DEFAULT_PARLIAMENT_ROOT.exists() and DEFAULT_PARLIAMENT_PYTHON.exists(),
                         "native Parliament checkout unavailable")
    def test_runner_admits_two_action_world_with_visible_promise_gap(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("run_blueprint_parliament.choose_by_cloze", return_value=blueprint(PROMISE + BASE)), \
                mock.patch("run_blueprint_parliament.openai_complete"), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--text", PROMISE + BASE, "--prepare-only", "--output-dir", tmp]), 0)
            root = Path(tmp)
            manifest = json.loads((root / "preparation_manifest.json").read_text())
            self.assertEqual(manifest["pipeline_status"], "ADMITTED")
            coverage = json.loads((root / "semantic_coverage.json").read_text())
            self.assertTrue(all(c["retained_not_projected_construction_ids"] for c in coverage))
            graph = (root / "candidate_graphs.md").read_text()
            self.assertIn("Missing world mapping", graph)
            self.assertIn("Unconsumed parser candidates", graph)
            self.assertIn("complement content", graph)
            attempts = json.loads((root / "candidate_attempts.json").read_text())["attempts"]
            composition = next(a for a in attempts if a.get("variant") == "composition")
            self.assertEqual(composition["native_admission"]["status"], "ADMITTED")
            self.assertEqual(composition["native_admission"]["admission"]["effects"], 6)


if __name__ == "__main__":
    unittest.main()
