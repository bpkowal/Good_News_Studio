from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from global_workspace.world_graph_tikz import (
    compile_world_graph_tikz_bundle,
    render_action_world_tikz,
    write_world_graph_tikz_bundle,
)


def _world() -> dict:
    return {
        "actions": [
            {"action_id": "A0", "intervention": "Give medicine to Maria"},
            {"action_id": "A1", "intervention": "Give medicine to David"},
        ],
        "parties": [
            {"party_id": "P_MARIA", "label": "Maria"},
            {"party_id": "P_DAVID", "label": "David"},
        ],
        "effects": [
            {
                "effect_id": "E1", "action_id": "A0", "party_id": "P_MARIA",
                "outcome": "receives medicine", "polarity": "NEUTRAL",
                "modality": "CERTAIN", "clause_ids": ["C1"],
            },
            {
                "effect_id": "E2", "action_id": "A0", "party_id": "P_DAVID",
                "outcome": "receives no medicine", "polarity": "NEUTRAL",
                "modality": "CERTAIN", "clause_ids": ["C1"],
            },
            {
                "effect_id": "E3", "action_id": "A0", "party_id": "P_DAVID",
                "outcome": "dies", "polarity": "ADVERSE",
                "modality": "CONDITIONAL", "clause_ids": ["C2"],
            },
            {
                "effect_id": "F1", "action_id": "A1", "party_id": "P_DAVID",
                "outcome": "does not die", "polarity": "BENEFICIAL",
                "modality": "CERTAIN", "clause_ids": ["C2"],
            },
        ],
        "causal_links": [
            {"source_id": "E1", "target_id": "E2", "link_relation": "CAUSES"},
            {
                "source_id": "E2", "target_id": "E3", "link_relation": "CAUSES",
                "condition_ids": ["C2"],
            },
        ],
        "counterfactual_links": [{
            "source_effect_id": "E3", "alternative_effect_id": "F1",
            "counterfactual_relation": "PRECLUDES_ALTERNATIVE_EFFECT",
        }],
    }


class WorldGraphTikzTests(unittest.TestCase):
    def test_action_figure_encodes_topology_conditions_and_provenance(self) -> None:
        rendered = render_action_world_tikz(_world(), "A0")
        self.assertIn(r"\draw[causal]", rendered)
        self.assertIn(r"\draw[conditional]", rendered)
        self.assertIn(r"CAUSES if C2", rendered)
        self.assertIn(r"source: C1", rendered)
        self.assertIn("David · ADVERSE · CONDITIONAL", rendered)

    def test_bundle_is_non_mutating_and_marks_representation_stage(self) -> None:
        world = _world()
        original = copy.deepcopy(world)
        clauses = [
            {"clause_id": "C1", "text": "There is one dose."},
            {"clause_id": "C2", "text": "Without medicine, the patient dies."},
        ]
        with tempfile.TemporaryDirectory() as raw_directory:
            destination = Path(raw_directory)
            manifest = write_world_graph_tikz_bundle(
                destination,
                world_model=world,
                clauses=clauses,
                representation_stage="admitted_compiled_world",
            )
            self.assertEqual(world, original)
            self.assertEqual(manifest["representation_stage"], "ADMITTED_COMPILED_WORLD")
            self.assertTrue((destination / "action_A0.tex").is_file())
            self.assertTrue((destination / "action_A1.tex").is_file())
            self.assertTrue((destination / "source_bindings.tex").is_file())
            self.assertTrue((destination / "counterfactual_topology.tex").is_file())
            saved = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(saved["canonical_representation"], "JSON_WORLD_MODEL")
            self.assertFalse(saved["latex_compiled"])

    def test_optional_compiler_has_explicit_result(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            result = compile_world_graph_tikz_bundle(Path(raw_directory))
        self.assertIn(result["status"], {"LATEX_NOT_AVAILABLE", "COMPILED"})
        self.assertIn("compiled", result)
        self.assertIn("failed", result)

    def test_quarantined_comparative_is_visibly_withheld(self) -> None:
        world = _world()
        world["effects"].append({
            "effect_id": "E4", "action_id": "A0", "party_id": "P_MARIA",
            "outcome": "has a better chance", "polarity": "BENEFICIAL",
            "modality": "PROBABILISTIC", "clause_ids": ["C3"],
        })
        world["causal_links"].append({
            "source_id": "E1", "target_id": "E4", "link_relation": "CAUSES",
        })
        world["admission"] = {
            "status": "COMMITTED_WITH_QUARANTINE",
            "quarantined_effects": [{
                "effect_id": "E4",
                "contradiction_type": "IMPLICIT_COMPARATIVE_CAUSALIZATION_UNRESOLVED",
            }],
        }
        rendered = render_action_world_tikz(world, "A0")
        self.assertIn(r"\node[quarantined] (eE4)", rendered)
        self.assertIn("QUARANTINED: IMPLICIT\\_COMPARATIVE", rendered)
        self.assertIn(r"\draw[quarantineedge]", rendered)
        self.assertIn("WITHHELD INFERENCE: CAUSES", rendered)


if __name__ == "__main__":
    unittest.main()
