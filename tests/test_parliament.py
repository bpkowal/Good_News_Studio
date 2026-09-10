from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import parliament
import global_workspace_pipeline
from global_workspace.structured_io import (
    ModelCallBudgetExceeded,
    call_json_llm,
    model_call_budget_paused,
    reset_model_call_budget,
    start_model_call_budget,
)


class ParliamentLauncherTests(unittest.TestCase):
    def test_workspace_scenario_contains_question(self):
        with tempfile.TemporaryDirectory() as directory:
            path = parliament.create_workspace_scenario(
                "Should a scientist disclose a dangerous program?",
                Path(directory),
            )
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["scenario_type"], "global_workspace")
            self.assertEqual(
                data["ethical_question"],
                "Should a scientist disclose a dangerous program?",
            )

    def test_workspace_command_uses_active_interpreter_and_actions(self):
        args = parliament.parse_args([
            "--mode", "workspace",
            "--question", "A sufficiently long ethical question",
            "--actions", "act now", "wait",
        ])
        command = parliament.workspace_command(args, Path("scenario.json"))
        self.assertEqual(command[0], parliament.sys.executable)
        self.assertIn("--actions", command)
        start = command.index("--actions") + 1
        self.assertEqual(command[start:start + 2], ["act now", "wait"])

    def test_workspace_command_can_select_openai_backend(self):
        args = parliament.parse_args([
            "--mode", "workspace",
            "--question", "A sufficiently long ethical question",
            "--backend", "openai",
            "--openai-model", "o3",
            "--openai-concurrency", "3",
        ])
        command = parliament.workspace_command(args, Path("scenario.json"))
        self.assertIn("--backend", command)
        self.assertEqual(command[command.index("--backend") + 1], "openai")
        self.assertEqual(command[command.index("--openai-model") + 1], "o3")
        self.assertEqual(
            command[command.index("--openai-concurrency") + 1], "3",
        )

    def test_workspace_command_caps_openai_concurrency_at_three(self):
        args = parliament.parse_args([
            "--mode", "workspace",
            "--question", "A sufficiently long ethical question",
            "--backend", "openai",
            "--openai-concurrency", "20",
        ])
        command = parliament.workspace_command(args, Path("scenario.json"))
        self.assertEqual(
            command[command.index("--openai-concurrency") + 1], "3",
        )

    def test_workspace_command_runs_only_selected_frameworks(self):
        args = parliament.parse_args([
            "--mode", "workspace",
            "--question", "A sufficiently long ethical question",
            "--agents", "rawlsian", "deontological",
        ])
        command = parliament.workspace_command(args, Path("scenario.json"))
        start = command.index("--agents") + 1
        self.assertEqual(command[start:start + 2], ["deontological", "rawlsian"])
        self.assertNotIn("utilitarian", command[start:])

    def test_workspace_command_can_bypass_framing_cache(self):
        args = parliament.parse_args([
            "--mode", "workspace",
            "--question", "A sufficiently long ethical question",
            "--no-framing-cache",
        ])
        command = parliament.workspace_command(args, Path("scenario.json"))
        self.assertIn("--no-framing-cache", command)

    def test_workspace_command_forwards_world_checkpoint_flags(self):
        args = parliament.parse_args([
            "--mode", "workspace",
            "--question", "A sufficiently long ethical question",
            "--stop-after-world",
        ])
        command = parliament.workspace_command(args, Path("scenario.json"))
        self.assertIn("--stop-after-world", command)
        args = parliament.parse_args([
            "--mode", "workspace",
            "--question", "A sufficiently long ethical question",
            "--accept-world",
        ])
        command = parliament.workspace_command(args, Path("scenario.json"))
        self.assertIn("--accept-world", command)

    def test_workspace_command_forwards_world_escalation_flags(self):
        args = parliament.parse_args([
            "--mode", "workspace",
            "--question", "A sufficiently long ethical question",
            "--escalate-world-model",
            "--world-escalation-model", "gpt-5.6-sol",
        ])
        command = parliament.workspace_command(args, Path("scenario.json"))
        self.assertIn("--escalate-world-model", command)
        self.assertEqual(
            command[command.index("--world-escalation-model") + 1],
            "gpt-5.6-sol",
        )
        args = parliament.parse_args([
            "--mode", "workspace",
            "--question", "A sufficiently long ethical question",
            "--no-world-escalation",
        ])
        command = parliament.workspace_command(args, Path("scenario.json"))
        self.assertIn("--no-world-escalation", command)
        self.assertNotIn("--escalate-world-model", command)

    def test_agent_aliases_are_normalized_in_canonical_order(self):
        self.assertEqual(
            parliament.normalize_agents(["Rawls", "deon"]),
            ["deontological", "rawlsian"],
        )

    def test_single_agent_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            parliament.normalize_agents(["rawlsian"])

    def test_exact_problem_framing_cache_hit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_problem_framing.json"
            payload = {
                "ethical_problem": "Exactly this ethical problem",
                "presentation_actions": ["act", "wait"],
                "canonical_actions": ["wait", "act"],
                "canonical_scenario": "canonical problem",
                "action_source_grounding": {
                    "status": "COMMITTED", "world_model": {"parties": [{}]},
                },
            }
            global_workspace_pipeline.save_problem_framing_cache(path, payload)
            with patch(
                "global_workspace.world_state.world_model_from_dict",
                return_value=object(),
            ):
                cached, status = global_workspace_pipeline.load_problem_framing_cache(
                    path, "Exactly this ethical problem",
                )
            self.assertEqual(status, "HIT")
            self.assertEqual(cached["presentation_actions"], ["act", "wait"])

    def test_framing_cache_requires_exact_problem_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_problem_framing.json"
            global_workspace_pipeline.save_problem_framing_cache(path, {
                "ethical_problem": "Problem A",
                "presentation_actions": ["act", "wait"],
                "canonical_actions": ["wait", "act"],
                "canonical_scenario": "canonical problem",
                "action_source_grounding": {
                    "status": "COMMITTED", "world_model": {"parties": [{}]},
                },
            })
            cached, status = global_workspace_pipeline.load_problem_framing_cache(
                path, "Problem A ",
            )
            self.assertIsNone(cached)
            self.assertEqual(status, "MISS_DIFFERENT_PROBLEM")

    def test_pre_qualifier_schema_framing_cache_is_invalidated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_problem_framing.json"
            path.write_text(json.dumps({
                "cache_version": 1,
                "ethical_problem": "Exactly this ethical problem",
            }), encoding="utf-8")
            cached, status = global_workspace_pipeline.load_problem_framing_cache(
                path, "Exactly this ethical problem",
            )
            self.assertIsNone(cached)
            self.assertEqual(status, "MISS_INCOMPATIBLE_CACHE")

    def test_edited_actions_invalidate_cached_grounding(self):
        cached = {
            "presentation_actions": ["act", "wait"],
            "canonical_actions": ["wait", "act"],
            "canonical_scenario": "canonical problem",
        }
        self.assertTrue(global_workspace_pipeline.cached_framing_matches(
            cached,
            presentation_actions=["act", "wait"],
            canonical_actions=["wait", "act"],
            canonical_scenario="canonical problem",
        ))
        self.assertFalse(global_workspace_pipeline.cached_framing_matches(
            cached,
            presentation_actions=["act differently", "wait"],
            canonical_actions=["wait", "act differently"],
            canonical_scenario="changed canonical problem",
        ))

    def test_cache_hit_skips_planner_and_grounder_calls(self):
        cached = {
            "presentation_actions": ["act", "wait"],
            "action_source_grounding": {
                "status": "COMMITTED", "world_model": {"parties": [{}]},
            },
        }

        def must_not_run():
            raise AssertionError("expensive model call should have been skipped")

        actions, actions_reused = global_workspace_pipeline.choose_initial_actions(
            None, cached, must_not_run,
        )
        grounding, grounding_reused = (
            global_workspace_pipeline.choose_action_source_grounding(
                cached, cache_matches=True, grounder=must_not_run,
            )
        )
        self.assertEqual(actions, ["act", "wait"])
        self.assertEqual(grounding["status"], "COMMITTED")
        self.assertTrue(actions_reused)
        self.assertTrue(grounding_reused)

    def test_malformed_framing_cache_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_problem_framing.json"
            path.write_text("not-json", encoding="utf-8")
            cached, status = global_workspace_pipeline.load_problem_framing_cache(
                path, "Any problem",
            )
            self.assertIsNone(cached)
            self.assertEqual(status, "MISS_INVALID_CACHE")

    def test_question_validation(self):
        with self.assertRaises(ValueError):
            parliament.main(["--mode", "workspace", "--question", "short"])

    @patch("parliament.run_workspace", return_value=0)
    @patch("builtins.input", side_effect=[
        "Should I return a dangerous person's lost property?",
        "local",
    ])
    def test_question_at_mode_prompt_defaults_to_workspace(self, _input, run_workspace):
        self.assertEqual(parliament.main([]), 0)
        args, question = run_workspace.call_args.args
        self.assertEqual(question, "Should I return a dangerous person's lost property?")

    @patch("parliament.run_workspace", return_value=0)
    @patch("builtins.input", side_effect=["workspace", "A sufficiently long ethical question", "openai"])
    def test_interactive_workspace_prompts_for_backend(self, _input, run_workspace):
        self.assertEqual(parliament.main([]), 0)
        args, _question = run_workspace.call_args.args
        self.assertEqual(args.backend, "openai")

    @patch("parliament.run_workspace", return_value=0)
    @patch("builtins.input", side_effect=[
        "workspadce", "A sufficiently long ethical question", "openai",
    ])
    def test_workspace_typo_still_prompts_for_question_and_backend(
        self, _input, run_workspace,
    ):
        self.assertEqual(parliament.main([]), 0)
        args, question = run_workspace.call_args.args
        self.assertEqual(question, "A sufficiently long ethical question")
        self.assertEqual(args.backend, "openai")

    def test_real_question_is_not_mistaken_for_mode_typo(self):
        self.assertEqual(parliament.correct_mode_typo("workspace safety dilemma"), "")
        self.assertEqual(parliament.correct_mode_typo("Should workers report danger?"), "")

    @patch("parliament.run_workspace", return_value=0)
    @patch("builtins.input", side_effect=["openai", "A sufficiently long ethical question"])
    def test_backend_entered_at_mode_prompt_is_recovered(self, _input, run_workspace):
        self.assertEqual(parliament.main([]), 0)
        args, question = run_workspace.call_args.args
        self.assertEqual(args.backend, "openai")
        self.assertEqual(question, "A sufficiently long ethical question")

    @patch("parliament.run_workspace", return_value=0)
    @patch("parliament.sys.stdin.isatty", return_value=True)
    @patch("builtins.input", side_effect=[
        "workspace",
        "A sufficiently long ethical question for cycle selection",
        "openai",
        "7",
        "deon, rawls",
        "",
    ])
    def test_interactive_startup_prompts_for_max_cycles(self, _input, _isatty, run_workspace):
        self.assertEqual(parliament.main([]), 0)
        args, _question = run_workspace.call_args.args
        self.assertEqual(args.max_cycles, 7)
        self.assertEqual(args.backend, "openai")
        self.assertEqual(args.agents, ["deontological", "rawlsian"])
        self.assertFalse(args.use_rag)

    @patch("parliament.run_workspace", return_value=0)
    @patch("parliament.sys.stdin.isatty", return_value=True)
    @patch("builtins.input", side_effect=[
        "workspace",
        "A sufficiently long ethical question for cycle default",
        "local",
        "",
        "",
        "",
    ])
    def test_interactive_max_cycles_default_is_three(self, _input, _isatty, run_workspace):
        self.assertEqual(parliament.main([]), 0)
        args, _question = run_workspace.call_args.args
        self.assertEqual(args.max_cycles, 3)
        self.assertEqual(args.agents, list(parliament.AGENT_MODULES))
        self.assertFalse(args.use_rag)

    def test_prompt_max_cycles_rejects_non_positive(self):
        with patch("builtins.input", return_value="0"):
            with self.assertRaises(ValueError):
                parliament.prompt_max_cycles()

    def test_cli_max_cycles_skips_interactive_prompt(self):
        args = parliament.parse_args([
            "--mode", "workspace",
            "--question", "A sufficiently long ethical question",
            "--max-cycles", "10",
        ])
        self.assertEqual(
            parliament.resolve_max_cycles(args, interactive=True),
            10,
        )

    def test_workspace_command_defaults_to_no_rag_context(self):
        args = parliament.parse_args([
            "--mode", "workspace",
            "--question", "A sufficiently long ethical question",
        ])
        command = parliament.workspace_command(args, Path("scenario.json"))
        self.assertIn("--no-rag-context", command)

    def test_workspace_command_can_opt_in_to_rag(self):
        args = parliament.parse_args([
            "--mode", "workspace",
            "--question", "A sufficiently long ethical question",
            "--rag",
        ])
        command = parliament.workspace_command(args, Path("scenario.json"))
        self.assertNotIn("--no-rag-context", command)

    def test_cli_rag_skips_interactive_prompt(self):
        args = parliament.parse_args([
            "--mode", "workspace",
            "--question", "A sufficiently long ethical question",
            "--rag",
        ])
        self.assertTrue(parliament.resolve_use_rag(args, interactive=True))

    def test_cli_no_rag_skips_interactive_prompt(self):
        args = parliament.parse_args([
            "--mode", "workspace",
            "--question", "A sufficiently long ethical question",
            "--no-rag",
        ])
        self.assertFalse(parliament.resolve_use_rag(args, interactive=True))

    def test_skip_original_agents_forces_rag_off(self):
        args = parliament.parse_args([
            "--mode", "workspace",
            "--question", "A sufficiently long ethical question",
            "--rag",
            "--skip-original-agents",
        ])
        self.assertFalse(parliament.resolve_use_rag(args, interactive=True))

    @patch("parliament.run_workspace", return_value=0)
    @patch("parliament.sys.stdin.isatty", return_value=True)
    @patch("builtins.input", side_effect=[
        "workspace",
        "A sufficiently long ethical question for rag opt in",
        "local",
        "",
        "",
        "y",
    ])
    def test_interactive_startup_can_enable_corpus_rag(self, _input, _isatty, run_workspace):
        self.assertEqual(parliament.main([]), 0)
        args, _question = run_workspace.call_args.args
        self.assertTrue(args.use_rag)

    @patch("global_workspace_pipeline.sys.stdin.isatty", return_value=True)
    @patch("builtins.input", side_effect=["edit", "give to child | give to researcher"])
    def test_actions_can_be_reviewed_and_edited(self, _input, _isatty):
        result = global_workspace_pipeline.confirm_actions(["bad action", "other action"])
        self.assertEqual(result, ["give to child", "give to researcher"])

    @patch("global_workspace_pipeline.sys.stdin.isatty", return_value=True)
    @patch("builtins.input", return_value="yes")
    def test_user_can_extend_unresolved_deliberation(self, _input, _isatty):
        result = SimpleNamespace(
            cycles=[SimpleNamespace(entropy=0.91)],
            synthesis_proposals=[SimpleNamespace(action="a grounded third action", accepted=True)],
            failure_conditions=[SimpleNamespace(
                valid=True,
                necessary_condition="the third action succeeds",
                failure_condition="the third action fails",
                contingency_question="If it fails, should we choose the first or second action?",
            )],
        )
        self.assertEqual(global_workspace_pipeline.prompt_cycle_extension(result, 2), 2)

    def test_world_checkpoint_summary_names_actual_and_foregone_rows(self):
        text = global_workspace_pipeline.render_admitted_world(
            [{
                "action_id": "A0",
                "short_label": "Refuse to frame",
                "beneficiaries": ["innocent citizen"],
                "harmed": ["city residents"],
                "mechanism": "CONTINUES causes DIES",
                "causal_links": [
                    {"source_id": "E0_2", "relation": "CAUSES", "target_id": "E0_3"},
                ],
                "world_effects": [
                    {
                        "effect_id": "E0_3", "directness": "DOWNSTREAM",
                        "polarity": "ADVERSE", "outcome": "DIES", "party_id": "P3",
                        "quantities": ["over five hundred"],
                    },
                ],
                "counterfactual_effects": [
                    {
                        "source_effect_id": "E0_4",
                        "alternative_action_id": "A1",
                        "alternative_effect_id": "E1_3",
                    },
                ],
            }],
            {"status": "COMMITTED", "repair_attempts": 0},
        )
        self.assertIn("A0: Refuse to frame", text)
        self.assertIn("E0_2 CAUSES E0_3", text)
        self.assertIn("over five hundred", text)
        self.assertIn("FOREGONE E0_4 -> A1 E1_3", text)
        self.assertIn("at risk:", text)
        self.assertIn("conditionally benefited:", text)

    @patch("global_workspace_pipeline.sys.stdin.isatty", return_value=True)
    def test_world_checkpoint_stops_unless_user_continues(self, _isatty):
        records = [{"action_id": "A0", "short_label": "wait", "world_effects": []}]
        grounding = {"status": "COMMITTED"}
        captured: list[str] = []
        self.assertFalse(global_workspace_pipeline.confirm_continue_after_world(
            records, grounding, input_fn=lambda _: "n", output_fn=captured.append,
        ))
        self.assertTrue(global_workspace_pipeline.confirm_continue_after_world(
            records, grounding, input_fn=lambda _: "yes", output_fn=captured.append,
        ))
        self.assertFalse(global_workspace_pipeline.confirm_continue_after_world(
            records, grounding, stop_after_world=True, output_fn=captured.append,
        ))
        self.assertTrue(global_workspace_pipeline.confirm_continue_after_world(
            records, grounding, accept_world=True, output_fn=captured.append,
        ))

    @patch("global_workspace_pipeline.sys.stdin.isatty", return_value=False)
    def test_noninteractive_world_checkpoint_continues(self, _isatty):
        self.assertTrue(global_workspace_pipeline.confirm_continue_after_world(
            [{"action_id": "A0", "short_label": "wait", "world_effects": []}],
            {"status": "COMMITTED"},
            output_fn=lambda _message: None,
        ))
        self.assertFalse(global_workspace_pipeline.confirm_continue_after_world(
            [{"action_id": "A0", "short_label": "wait", "world_effects": []}],
            {"status": "COMMITTED"},
            stop_after_world=True,
            output_fn=lambda _message: None,
        ))

    def test_world_escalation_is_offered_only_after_live_openai_failure(self):
        rejected = {"status": "REJECTED", "errors": ["missing unique clause"]}
        openai_live = dict(backend="openai", current_model="o3", reused=False)
        self.assertTrue(global_workspace_pipeline.should_offer_world_escalation(
            rejected, **openai_live,
        ))
        self.assertFalse(global_workspace_pipeline.should_offer_world_escalation(
            {"status": "COMMITTED"}, **openai_live,
        ))
        self.assertFalse(global_workspace_pipeline.should_offer_world_escalation(
            rejected, backend="openai", current_model="o3", reused=True,
        ))
        self.assertFalse(global_workspace_pipeline.should_offer_world_escalation(
            rejected, backend="local", current_model="o3", reused=False,
        ))
        self.assertFalse(global_workspace_pipeline.should_offer_world_escalation(
            rejected, backend="openai", current_model="gpt-5.6-sol", reused=False,
        ))
        self.assertFalse(global_workspace_pipeline.should_offer_world_escalation(
            rejected, backend="openai", current_model="gpt-5.6", reused=False,
        ))
        self.assertFalse(global_workspace_pipeline.should_offer_world_escalation(
            rejected, **openai_live, no_world_escalation=True,
        ))
        self.assertFalse(global_workspace_pipeline.should_offer_world_escalation(
            {"status": "UNAVAILABLE", "errors": ["scenario has no source clauses"]},
            **openai_live,
        ))

    @patch("global_workspace_pipeline.sys.stdin.isatty", return_value=True)
    def test_world_escalation_asks_before_a_sol_retry(self, _isatty):
        grounding = {"status": "REJECTED", "errors": ["missing unique clause"]}
        captured: list[str] = []
        self.assertTrue(global_workspace_pipeline.confirm_world_model_escalation(
            grounding,
            from_model="o3",
            input_fn=lambda _: "y",
            output_fn=captured.append,
        ))
        self.assertTrue(any("missing unique clause" in line for line in captured))
        self.assertTrue(any("GPT-5.6 Sol" in line for line in captured))
        captured.clear()
        self.assertFalse(global_workspace_pipeline.confirm_world_model_escalation(
            grounding,
            from_model="o3",
            input_fn=lambda _: "n",
            output_fn=captured.append,
        ))
        self.assertTrue(any("Skipping" in line for line in captured))

    @patch("global_workspace_pipeline.sys.stdin.isatty", return_value=False)
    def test_noninteractive_world_escalation_requires_flag(self, _isatty):
        grounding = {"status": "REJECTED", "errors": ["missing unique clause"]}
        captured: list[str] = []
        self.assertFalse(global_workspace_pipeline.confirm_world_model_escalation(
            grounding,
            from_model="o3",
            output_fn=captured.append,
        ))
        self.assertTrue(any("--escalate-world-model" in line for line in captured))
        captured.clear()
        self.assertTrue(global_workspace_pipeline.confirm_world_model_escalation(
            grounding,
            from_model="o3",
            escalate_world_model=True,
            output_fn=captured.append,
        ))
        self.assertTrue(any("Retrying once" in line for line in captured))

    def test_world_escalation_keeps_primary_attempts(self):
        merged = global_workspace_pipeline.attach_world_escalation(
            {
                "status": "REJECTED",
                "attempts": [{"attempt": 1, "errors": ["primary failed"]}],
            },
            {
                "status": "COMMITTED",
                "attempts": [{"attempt": 1, "errors": []}],
                "world_model": {"schema_version": "1.3"},
            },
            from_model="o3",
            to_model="gpt-5.6-sol",
        )
        self.assertEqual(merged["status"], "COMMITTED")
        self.assertEqual(len(merged["attempts"]), 2)
        self.assertEqual(merged["escalation"]["from_model"], "o3")
        self.assertEqual(merged["escalation"]["to_model"], "gpt-5.6-sol")
        self.assertEqual(merged["escalation"]["attempts"], 1)

    def test_quota_failure_continues_to_compact_specialists(self):
        captured: list[str] = []
        testimonies, errors = global_workspace_pipeline.original_testimonies_or_continue(
            {},
            {
                "utilitarian": "OpenAI quota prevented the model call",
                "deontological": "canceled after terminal provider failure",
            },
            ["utilitarian", "deontological", "virtue"],
            output_fn=captured.append,
        )
        self.assertEqual(testimonies, {})
        self.assertEqual(errors["utilitarian"], "OpenAI quota prevented the model call")
        self.assertEqual(errors["virtue"], "no original testimony")
        self.assertTrue(any("compact specialists only" in line for line in captured))

    def test_two_original_testimonies_are_kept(self):
        testimonies, errors = global_workspace_pipeline.original_testimonies_or_continue(
            {"utilitarian": "prefer A0", "care": "prefer A1"},
            {},
            ["utilitarian", "care", "virtue"],
            output_fn=lambda _message: None,
        )
        self.assertEqual(testimonies, {"utilitarian": "prefer A0", "care": "prefer A1"})
        self.assertEqual(errors["virtue"], "no original testimony")

    def test_rejected_world_exits_zero_on_tty(self):
        record = SimpleNamespace(
            action_id="A0",
            commitment_status="REJECTED",
            commitment_reasons=("incomplete",),
            as_dict=lambda: {
                "action_id": "A0", "short_label": "wait", "world_effects": [],
            },
        )
        captured: list[str] = []
        code = global_workspace_pipeline.report_withheld_world(
            [record],
            {"status": "REJECTED", "repair_attempts": 3},
            tty=True,
            output_fn=captured.append,
        )
        self.assertEqual(code, 0)
        self.assertTrue(any("REJECTED" in message for message in captured))
        self.assertTrue(any("not running expert agents" in message for message in captured))
        rendered = "\n".join(captured)
        self.assertIn("World grounding rejected", rendered)
        self.assertNotIn("Admitted world", rendered)
        self.assertNotIn("benefits:", rendered)

    def test_rejected_world_never_prints_fallback_role_inferences(self):
        text = global_workspace_pipeline.render_admitted_world(
            [{
                "action_id": "A1",
                "short_label": "Leave facility exposed",
                "beneficiaries": ["residents named in a conditional harm"],
                "harmed": ["another fallback guess"],
                "at_risk": [],
            }],
            {"status": "REJECTED", "rejected_candidate": {"world_model": {}}},
        )
        self.assertIn("World grounding rejected", text)
        self.assertIn("A1: Leave facility exposed", text)
        self.assertIn("No beneficiary, harm, risk", text)
        self.assertNotIn("residents named in a conditional harm", text)
        self.assertNotIn("another fallback guess", text)

    def test_rejected_world_exits_two_when_not_a_tty(self):
        record = SimpleNamespace(
            action_id="A0",
            commitment_status="REJECTED",
            commitment_reasons=("incomplete",),
            as_dict=lambda: {
                "action_id": "A0", "short_label": "wait", "world_effects": [],
            },
        )
        code = global_workspace_pipeline.report_withheld_world(
            [record],
            {"status": "REJECTED"},
            tty=False,
            output_fn=lambda _message: None,
        )
        self.assertEqual(code, 2)

    def test_paused_budget_survives_a_wait_longer_than_the_deadline(self):
        token = start_model_call_budget(0.05, reserve_seconds=0.0)
        try:
            with model_call_budget_paused():
                time.sleep(0.12)

            class _LLM:
                def complete_json(self, *_args, **_kwargs):
                    return {"ok": True}

            self.assertEqual(
                call_json_llm(
                    _LLM(), "prompt", max_tokens=8, temperature=0.0,
                    schema={"type": "object"},
                ),
                {"ok": True},
            )
        finally:
            reset_model_call_budget(token)

    def test_unpaused_budget_expires_during_a_wait(self):
        token = start_model_call_budget(0.05, reserve_seconds=0.0)
        try:
            time.sleep(0.12)

            class _LLM:
                def complete_json(self, *_args, **_kwargs):
                    return {"ok": True}

            with self.assertRaises(ModelCallBudgetExceeded):
                call_json_llm(
                    _LLM(), "prompt", max_tokens=8, temperature=0.0,
                    schema={"type": "object"},
                )
        finally:
            reset_model_call_budget(token)


if __name__ == "__main__":
    unittest.main()
