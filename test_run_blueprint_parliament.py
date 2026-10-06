import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from blueprint_discourse import build_disputed_report
from blueprint_proposal_contract import withheld_proposal
from run_blueprint_parliament import (
    RELENT_FRAMEWORKS,
    _mermaid_world,
    _normalize_agents,
    _parliament_revision,
    _parliament_deliberation_command,
    _scenario_text,
    _write_candidate_attempts,
    _write_deliberation_report,
    _write_world_topology,
    compare_z10_blueprints,
    main,
    resolve_relent_deliberation,
)


def _deliberation_args(**overrides):
    values = dict(
        prototype_deliberation=False,
        agents=None,
        max_cycles=None,
        use_rag=None,
        no_rag_context=False,
        time_budget=None,
        agent_timeout=None,
        openai_model="o3",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _command(**overrides):
    values = dict(
        parliament_python=Path("/tmp/parliament-smoke-env/bin/python"),
        scenario_path=Path("/tmp/scenario.json"),
        trace_path=Path("/tmp/frozen_world_trace.json"),
        output_dir=Path("/tmp/parliament"),
        openai_model="o3",
        agents=list(RELENT_FRAMEWORKS),
        max_cycles=3,
        time_budget=600.0,
        agent_timeout=600.0,
        use_rag=False,
        prototype=False,
    )
    values.update(overrides)
    return _parliament_deliberation_command(**values)


class BlueprintRunnerArtifactsTests(unittest.TestCase):
    def test_report_preserves_testimony_dissent_and_unresolved_judgment(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "workspace_test.json"
            data = {
                "judgment_status": "UNRESOLVED", "selected_action": "none",
                "current_plurality": "pull", "halted_by": "cycle_budget",
                "source_testimonies": {"care": "No decisive recommendation."},
                "source_errors": {"virtue": "timeout"},
                "cycles": [{"cycle": 1, "dissent": ["Do not pull."]}],
                "deontological_duty_ledger": [{"status": "uncertain"}],
            }
            report = _write_deliberation_report(trace, data).read_text()
            for text in ("UNRESOLVED", "Current plurality: pull", "No decisive recommendation.",
                         "timeout", "Do not pull.", "uncertain", "workspace_test.json"):
                self.assertIn(text, report)

    def test_topology_draws_discourse_as_stadium_nodes_not_effects(self):
        world, _ = build_disputed_report(
            "Ada claims the medicine is safe.",
            {"source": "Ada", "report_words": "claims",
             "reported_content": "the medicine is safe"},
        )
        mermaid = "\n".join(_mermaid_world(world, "attributed report"))
        compact = mermaid.replace(" ", "")
        self.assertIn("R0([", compact)
        self.assertIn("PR1([", compact)
        self.assertIn("CLAIMS", mermaid)
        self.assertNotIn('["E', mermaid)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "frozen_world_trace.json"
            trace.write_text(json.dumps({
                "action_source_grounding": {
                    "status": "COMMITTED",
                    "world_model": {
                        "schema_version": "1.3",
                        "parties": world["parties"],
                        "actions": world["actions"],
                        "effects": [],
                        "conditions": [],
                        "causal_links": [],
                    },
                    "world_model_1_4": world,
                },
            }), encoding="utf-8")
            markdown_path, json_path = _write_world_topology(trace, root)
            markdown = markdown_path.read_text(encoding="utf-8")
            topology = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertIn("Admission: **COMMITTED**", markdown)
        self.assertIn("Discourse objects (not established effects)", markdown)
        self.assertIn("`R0`", markdown)
        self.assertIn("`PR1`", markdown)
        self.assertTrue(any(row["id"] == "R0" and row["kind"] == "report"
                            for row in topology["nodes"]))
        self.assertFalse(any(row.get("kind") == "effect"
                             for row in topology["nodes"]))

    def test_z10_comparison_preserves_differences_without_changing_selection(self):
        package = {
            "package_id": "pkg", "open_questions": [{"id": "q0"}],
            "nodes": [{"id": "p0", "kind": "proposition", "predicate": "give"},
                      {"id": "p1", "kind": "proposition", "predicate": "survive"}],
            "candidates": [
                {"type": "PREDICATION", "arguments": {"proposition": "p0"}},
                {"type": "PREDICATION", "arguments": {"proposition": "p1"}},
                {"type": "CONDITIONAL_ON", "arguments": {}},
                {"type": "PARTICIPANT", "value": "agent", "arguments": {}},
            ],
        }
        world = {
            "parties": [{"label": "Ada"}],
            "effects": [{"predicate": "gives"}, {"predicate": "die"}],
            "conditions": [], "causal_links": [],
        }
        result = {"candidate_attempts": [{
            "blueprint_id": "exclusive_allocation", "rank": 0,
            "proposal": {"candidate": {"world_model": world}},
        }]}
        comparison = compare_z10_blueprints(package, result)
        row = comparison["candidate_comparisons"][0]
        self.assertIn("give", row["agreement"])
        self.assertIn("survive", row["z10_only"])
        self.assertIn("die", row["blueprint_only"])
        self.assertEqual(row["conflicts"][0]["kind"], "scope")
        self.assertFalse(comparison["selection_changed"])

    def test_candidate_display_keeps_graphs_and_withheld_missing_slots(self):
        world = {
            "schema_version": "1.3",
            "parties": [{"party_id": "P1", "label": "Ada", "kind": "PERSON"}],
            "actions": [{
                "action_id": "A0", "intervention": "act",
                "actor_party_id": "P1", "recipient_party_ids": [],
                "effect_ids": ["E1"], "clause_ids": ["C0"],
            }],
            "effects": [{
                "effect_id": "E1", "action_id": "A0", "party_id": "P1",
                "outcome": "Ada acts", "directness": "DIRECT",
                "polarity": "NEUTRAL", "modality": "CERTAIN",
            }],
            "causal_links": [], "conditions": [{
                "condition_id": "CND1", "description": "If Ada acts",
                "polarity": "POSITIVE",
            }], "temporal_relations": [],
            "counterfactual_links": [],
        }
        result = {
            "chosen_blueprint_id": "conditional_outcome",
            "status": "FILLED",
            "question": {"ethical_question": "whether Ada acts"},
            "candidate_attempts": [
                {
                    "blueprint_id": "conditional_outcome", "rank": 0,
                    "template_status": "FILLED", "selected": True,
                    "contract_valid": True, "contract_errors": [],
                    "core_filled": 3, "core_count": 3, "optional_filled": 0,
                    "unfilled_slots": [],
                    "proposal": {
                        "status": "FILLED", "candidate": {"world_model": world},
                        "world_withheld": [],
                    },
                },
                {
                    "blueprint_id": "exclusive_allocation", "rank": 1,
                    "template_status": "PARTIAL", "selected": False,
                    "contract_valid": True, "contract_errors": [],
                    "core_filled": 1, "core_count": 6, "optional_filled": 0,
                    "unfilled_slots": ["resource", "second_recipient"],
                    "proposal": {
                        "status": "WITHHELD", "candidate": None,
                        "world_withheld": ["Required slots remain unfilled."],
                    },
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            markdown, data = _write_candidate_attempts(result, Path(directory))
            rendered = markdown.read_text(encoding="utf-8")
            payload = json.loads(data.read_text(encoding="utf-8"))
        self.assertIn("```mermaid", rendered)
        self.assertIn("Ada acts", rendered)
        self.assertIn("CND1", rendered)
        self.assertIn("branch scope", rendered)
        self.assertIn("resource", rendered)
        self.assertIn("Required slots remain unfilled", rendered)
        self.assertEqual(len(payload["attempts"]), 2)

    def test_scenario_text_accepts_inline_and_file_input(self):
        self.assertEqual(_scenario_text("  Ada must decide.  ", None),
                         "Ada must decide.")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario.txt"
            path.write_text(" Ben may act.\n", encoding="utf-8")
            self.assertEqual(_scenario_text(None, path), "Ben may act.")
        with self.assertRaisesRegex(ValueError, "empty"):
            _scenario_text("  ", None)

    def test_withheld_run_writes_diagnostics_without_calling_parliament(self):
        proposal = withheld_proposal(
            proposal_id="promise_cloze", blueprint_id="promise_reliance",
            assignment=[], slot_bindings={"copied_spans": {"promisor": "Ada"}},
            clauses=[{"clause_id": "C0", "text": "Ada promised."}],
            unfilled_required_slots=["commitment_content"],
            unresolved_readings=[],
            construction_problems=[{
                "code": "incomplete_template",
                "message": "Required slots remain unfilled.",
            }],
            accepted_evidence={"promisor": "Ada"},
        )
        result = {
            "chosen_blueprint_id": "promise_reliance",
            "status": "PARTIAL", "ranking": ["promise_reliance"],
            "left_out": [], "question": {"ethical_question": ""},
            "world_withheld": ["Required slots remain unfilled."],
            "considered": [], "proposals": [proposal], "graph": None,
            "candidate_attempts": [{
                "blueprint_id": "promise_reliance", "rank": 0,
                "template_status": "PARTIAL", "selected": True,
                "contract_valid": True, "contract_errors": [],
                "core_filled": 1, "core_count": 2, "optional_filled": 0,
                "unfilled_slots": ["commitment_content"], "proposal": proposal,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = root / "api.env"
            env.write_text("OPENAI_API_KEY=test-only\n", encoding="utf-8")
            output = root / "output"
            with mock.patch("run_blueprint_parliament.z10.export_candidate_graph",
                            return_value={"package_id": "test"}), \
                 mock.patch("run_blueprint_parliament.assess_question",
                            return_value={"ethical_question": ""}), \
                 mock.patch("run_blueprint_parliament.openai_complete",
                            return_value=lambda messages: "{}"), \
                 mock.patch("run_blueprint_parliament.choose_by_cloze",
                            return_value=result), \
                 mock.patch("run_blueprint_parliament.subprocess.run") as run:
                status = main([
                    "--text", "Ada promised.", "--prepare-only",
                    "--openai-env", str(env), "--output-dir", str(output),
                    "--parliament-root", str(root / "missing-parliament"),
                    "--parliament-python", str(root / "missing-python"),
                ])
            manifest = json.loads(
                (output / "preparation_manifest.json").read_text(encoding="utf-8")
            )
        self.assertEqual(status, 2)
        self.assertEqual(manifest["pipeline_status"], "WITHHELD")
        self.assertEqual(manifest["missing_slots"], ["commitment_content"])
        self.assertEqual(manifest["parliament_admission"]["status"], "NOT_RUN")
        run.assert_not_called()


class RelEntHandoffTests(unittest.TestCase):
    def test_revision_records_checkout_and_local_integration_changes(self):
        responses = [
            mock.Mock(returncode=0, stdout="614af0c\n"),
            mock.Mock(returncode=0, stdout="relent-framework\n"),
            mock.Mock(returncode=0, stdout="global_workspace/world_state.py\n"),
        ]
        with mock.patch("run_blueprint_parliament.subprocess.run", side_effect=responses):
            record = _parliament_revision(Path("/tmp/parliament"))
        self.assertEqual(record["commit"], "614af0c")
        self.assertEqual(record["branch"], "relent-framework")
        self.assertEqual(record["modified_files"], ["global_workspace/world_state.py"])
        self.assertEqual(len(record["integration_patch_sha256"]), 64)
        self.assertEqual(len(record["discourse_workspace_patch_sha256"]), 64)

    def test_prototype_deliberation_command_keeps_cheap_smoke_flags(self):
        command = _command(
            agents=["utilitarian", "deontological"],
            max_cycles=1,
            agent_timeout=300.0,
            prototype=True,
        )
        self.assertIn("--skip-original-agents", command)
        self.assertEqual(command[command.index("--max-cycles") + 1], "1")
        self.assertIn("--no-synthesis", command)
        self.assertIn("--no-planning", command)
        self.assertIn("--no-cycle-extension", command)
        self.assertIn("--no-rag-context", command)
        self.assertEqual(command[command.index("--agents") + 1], "utilitarian")
        self.assertEqual(command[command.index("--agents") + 2], "deontological")
        self.assertNotIn("virtue", command)

    def test_full_deliberation_command_uses_relent_defaults(self):
        command = _command(prototype=False, max_cycles=3)
        self.assertNotIn("--skip-original-agents", command)
        self.assertEqual(command[command.index("--max-cycles") + 1], "3")
        self.assertNotIn("--no-synthesis", command)
        self.assertNotIn("--no-planning", command)
        self.assertNotIn("--no-cycle-extension", command)
        self.assertIn("--frozen-world-trace", command)
        for name in RELENT_FRAMEWORKS:
            self.assertIn(name, command)
        self.assertNotIn("evidence_graph", " ".join(command))

    def test_script_full_defaults_are_all_five_frameworks_and_three_cycles(self):
        options = resolve_relent_deliberation(
            _deliberation_args(), interactive=False,
        )
        self.assertEqual(options["mode"], "full")
        self.assertEqual(options["agents"], list(RELENT_FRAMEWORKS))
        self.assertEqual(options["max_cycles"], 3)
        self.assertFalse(options["use_rag"])
        self.assertEqual(options["agent_timeout"], 600.0)

    def test_prototype_defaults_skip_prompts_and_keep_two_frameworks(self):
        options = resolve_relent_deliberation(
            _deliberation_args(prototype_deliberation=True), interactive=True,
        )
        self.assertEqual(options["mode"], "prototype")
        self.assertEqual(options["agents"], ["utilitarian", "deontological"])
        self.assertEqual(options["max_cycles"], 1)
        self.assertEqual(options["agent_timeout"], 300.0)

    def test_explicit_flags_skip_the_matching_relent_prompt(self):
        options = resolve_relent_deliberation(
            _deliberation_args(
                agents=["care", "rawls"],
                max_cycles=4,
                use_rag=True,
            ),
            interactive=True,
        )
        self.assertEqual(options["agents"], ["care", "rawlsian"])
        self.assertEqual(options["max_cycles"], 4)
        self.assertTrue(options["use_rag"])

    def test_agent_aliases_keep_canonical_relent_order(self):
        self.assertEqual(
            _normalize_agents(["rawls", "util", "duty"]),
            ["utilitarian", "deontological", "rawlsian"],
        )


if __name__ == "__main__":
    unittest.main()
