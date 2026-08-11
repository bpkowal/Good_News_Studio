from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive entry point for the Ethical Parliament.")
    parser.add_argument("--mode", choices=("workspace", "legacy"))
    parser.add_argument("--question", help="Ethical problem; prompts interactively when omitted")
    parser.add_argument("--urgency", type=float, default=0.5)
    parser.add_argument("--danger", type=float, default=0.5)
    parser.add_argument("--actions", nargs="+", help="Optional action choices for workspace mode")
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--time-budget", type=float, default=600.0)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--backend", choices=("local", "openai"))
    parser.add_argument("--openai-model", default="o3")
    parser.add_argument("--agent-timeout", type=float, default=600.0)
    parser.add_argument("--n-ctx", type=int, default=768)
    parser.add_argument("--n-gpu-layers", type=int, default=8)
    parser.add_argument("--n-batch", type=int, default=32)
    parser.add_argument("--delegate-tokens", type=int, default=128)
    parser.add_argument("--accept-actions", action="store_true")
    parser.add_argument("--no-synthesis", action="store_true")
    parser.add_argument("--extension-cycles", type=int, default=2)
    parser.add_argument("--max-cycle-extensions", type=int, default=1)
    parser.add_argument("--no-cycle-extension", action="store_true")
    parser.add_argument("--no-visibility-audit", action="store_true")
    return parser.parse_args(argv)


def prompt_mode() -> str:
    answer = input("Mode [workspace/legacy] (workspace): ").strip().lower()
    if not answer:
        return "workspace"
    if answer not in {"workspace", "legacy"}:
        raise ValueError("Mode must be 'workspace' or 'legacy'")
    return answer


def prompt_question() -> str:
    question = input("Enter an ethical problem: ").strip()
    if len(question) < 10:
        raise ValueError("The ethical problem must contain at least 10 characters")
    return question


def prompt_backend() -> str:
    answer = input("Workspace backend [local/openai] (local): ").strip().lower()
    if not answer:
        return "local"
    if answer not in {"local", "openai"}:
        raise ValueError("Backend must be 'local' or 'openai'")
    return answer


def create_workspace_scenario(question: str, scenario_dir: Path = ROOT / "scenarios") -> Path:
    scenario_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    scenario_id = f"workspace_{stamp}"
    path = scenario_dir / f"{scenario_id}.json"
    data = {
        "scenario_id": scenario_id,
        "scenario_type": "global_workspace",
        "ethical_question": question,
        "tags": [],
        "tag_expectations": {},
        "tag_descriptions": {},
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def workspace_command(args: argparse.Namespace, scenario_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "global_workspace_pipeline.py"),
        str(scenario_path),
        "--urgency", str(args.urgency),
        "--danger", str(args.danger),
        "--max-cycles", str(max(1, args.max_cycles)),
        "--time-budget", str(max(1.0, args.time_budget)),
        "--agent-timeout", str(max(1.0, args.agent_timeout)),
        "--n-ctx", str(max(512, args.n_ctx)),
        "--n-gpu-layers", str(max(0, args.n_gpu_layers)),
        "--n-batch", str(max(8, args.n_batch)),
        "--delegate-tokens", str(max(48, args.delegate_tokens)),
        "--backend", args.backend or "local",
        "--openai-model", args.openai_model,
        "--extension-cycles", str(max(1, args.extension_cycles)),
        "--max-cycle-extensions", str(max(0, args.max_cycle_extensions)),
    ]
    if args.model:
        command.extend(["--model", str(args.model)])
    if args.actions:
        command.append("--actions")
        command.extend(args.actions)
    if args.accept_actions:
        command.append("--accept-actions")
    if args.no_synthesis:
        command.append("--no-synthesis")
    if args.no_cycle_extension:
        command.append("--no-cycle-extension")
    if args.no_visibility_audit:
        command.append("--no-visibility-audit")
    return command


def run_workspace(args: argparse.Namespace, question: str) -> int:
    scenario_path = create_workspace_scenario(question)
    print(f"Scenario saved: {scenario_path}")
    return subprocess.run(workspace_command(args, scenario_path), cwd=ROOT, check=False).returncode


def run_legacy(question: str) -> int:
    print("Starting the legacy Parliament pipeline...")
    return subprocess.run(
        [sys.executable, str(ROOT / "ethics_synthesis_agent.py")],
        cwd=ROOT,
        input=question + "\n",
        text=True,
        check=False,
    ).returncode


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode:
        mode = args.mode
        question = (args.question or prompt_question()).strip()
    elif args.question:
        mode = "workspace"
        question = args.question.strip()
    else:
        first_answer = input("Mode [workspace/legacy] (workspace): ").strip()
        normalized = first_answer.lower()
        if normalized in {"", "workspace", "legacy"}:
            mode = normalized or "workspace"
            question = prompt_question()
        else:
            mode = "workspace"
            question = first_answer
            print("Treating that entry as the ethical problem (workspace mode).")
    if len(question) < 10:
        raise ValueError("The ethical problem must contain at least 10 characters")
    if mode == "legacy":
        return run_legacy(question)
    if args.backend is None and args.question is None:
        args.backend = prompt_backend()
    else:
        args.backend = args.backend or "local"
    return run_workspace(args, question)


if __name__ == "__main__":
    raise SystemExit(main())
