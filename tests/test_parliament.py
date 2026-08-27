from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import parliament
import global_workspace_pipeline


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
        self.assertEqual(command[-2:], ["act now", "wait"])

    def test_workspace_command_can_select_openai_backend(self):
        args = parliament.parse_args([
            "--mode", "workspace",
            "--question", "A sufficiently long ethical question",
            "--backend", "openai",
            "--openai-model", "o3",
        ])
        command = parliament.workspace_command(args, Path("scenario.json"))
        self.assertIn("--backend", command)
        self.assertEqual(command[command.index("--backend") + 1], "openai")
        self.assertEqual(command[command.index("--openai-model") + 1], "o3")

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
    ])
    def test_interactive_startup_prompts_for_max_cycles(self, _input, _isatty, run_workspace):
        self.assertEqual(parliament.main([]), 0)
        args, _question = run_workspace.call_args.args
        self.assertEqual(args.max_cycles, 7)
        self.assertEqual(args.backend, "openai")

    @patch("parliament.run_workspace", return_value=0)
    @patch("parliament.sys.stdin.isatty", return_value=True)
    @patch("builtins.input", side_effect=[
        "workspace",
        "A sufficiently long ethical question for cycle default",
        "local",
        "",
    ])
    def test_interactive_max_cycles_default_is_three(self, _input, _isatty, run_workspace):
        self.assertEqual(parliament.main([]), 0)
        args, _question = run_workspace.call_args.args
        self.assertEqual(args.max_cycles, 3)

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


if __name__ == "__main__":
    unittest.main()
