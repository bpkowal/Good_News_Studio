from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

from global_workspace.legacy_bridge import AGENT_MODULES


ROOT = Path(__file__).resolve().parent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive entry point for the Ethical Parliament.")
    parser.add_argument("--mode", choices=("workspace", "legacy"))
    parser.add_argument("--question", help="Ethical problem; prompts interactively when omitted")
    parser.add_argument("--urgency", type=float, default=0.5)
    parser.add_argument("--danger", type=float, default=0.5)
    parser.add_argument("--actions", nargs="+", help="Optional action choices for workspace mode")
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Workspace deliberation cycle budget (prompts interactively when omitted)",
    )
    parser.add_argument("--time-budget", type=float, default=600.0)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--backend", choices=("local", "openai"))
    parser.add_argument("--openai-model", default="o3")
    parser.add_argument(
        "--agents", nargs="+", choices=tuple(AGENT_MODULES),
        help="Run only the selected ethical frameworks (at least two)",
    )
    parser.add_argument("--agent-timeout", type=float, default=600.0)
    parser.add_argument("--n-ctx", type=int, default=768)
    parser.add_argument("--n-gpu-layers", type=int, default=8)
    parser.add_argument("--n-batch", type=int, default=32)
    parser.add_argument("--delegate-tokens", type=int, default=128)
    parser.add_argument("--accept-actions", action="store_true")
    parser.add_argument(
        "--accept-world",
        action="store_true",
        help="Skip the post-world checkpoint and continue to expert agents",
    )
    parser.add_argument(
        "--stop-after-world",
        action="store_true",
        help="Halt after the admitted world is printed; do not run expert agents",
    )
    parser.add_argument(
        "--skip-original-agents",
        action="store_true",
        help="Skip the original-agent consult (diagnostic)",
    )
    parser.add_argument(
        "--rag",
        dest="use_rag",
        action="store_true",
        help="Original agents retrieve CORE/ADJACENT corpus quotes (MiniLM)",
    )
    parser.add_argument(
        "--no-rag",
        dest="use_rag",
        action="store_false",
        help="Original agents run without corpus retrieval",
    )
    parser.add_argument(
        "--no-rag-context",
        action="store_true",
        help="Alias for --no-rag",
    )
    parser.set_defaults(use_rag=None)
    parser.add_argument(
        "--no-framing-cache", action="store_true",
        help="Recompute action planning and action-source grounding",
    )
    parser.add_argument("--no-synthesis", action="store_true")
    parser.add_argument("--extension-cycles", type=int, default=2)
    parser.add_argument("--max-cycle-extensions", type=int, default=1)
    parser.add_argument("--no-cycle-extension", action="store_true")
    parser.add_argument("--no-visibility-audit", action="store_true")
    parser.add_argument("--no-autonomy-audit", action="store_true")
    parser.add_argument("--drop-vote-on-graph-rejection", action="store_true")
    parser.add_argument("--no-ev-dominance-breaker", action="store_true")
    parser.add_argument("--ev-dominance-ratio", type=float, default=5.0)
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


def prompt_max_cycles(default: int = 3) -> int:
    """Ask how many recurrent workspace cycles to allow before halt."""
    answer = input(f"Max cycles [{default}]: ").strip()
    if not answer:
        return default
    try:
        value = int(answer)
    except ValueError as exc:
        raise ValueError("Max cycles must be an integer") from exc
    if value < 1:
        raise ValueError("Max cycles must be at least 1")
    return value


_AGENT_ALIASES = {
    "util": "utilitarian", "utility": "utilitarian",
    "deon": "deontological", "duty": "deontological",
    "virtue ethics": "virtue",
    "care ethics": "care",
    "rawls": "rawlsian",
}


def normalize_agents(values: Sequence[str]) -> list[str]:
    requested: list[str] = []
    for raw in values:
        value = " ".join(str(raw).strip().casefold().replace("_", " ").split())
        value = _AGENT_ALIASES.get(value, value)
        if value not in AGENT_MODULES:
            raise ValueError(
                f"Unknown framework '{raw}'. Choose from: {', '.join(AGENT_MODULES)}"
            )
        if value not in requested:
            requested.append(value)
    if len(requested) < 2:
        raise ValueError("Select at least two ethical frameworks for a workspace run")
    # Stable canonical order keeps selection order from becoming an experimental variable.
    return [name for name in AGENT_MODULES if name in requested]


def prompt_agents() -> list[str]:
    answer = input(
        "Frameworks [all or comma-separated: utilitarian, deontological, virtue, care, rawlsian] (all): "
    ).strip()
    if not answer or answer.casefold() == "all":
        return list(AGENT_MODULES)
    return normalize_agents(answer.split(","))


def prompt_use_rag(default: bool = False) -> bool:
    """Ask whether original agents should retrieve corpus quotes this run."""
    hint = "y" if default else "n"
    answer = input(
        "Corpus RAG for original agents? [y/N] "
        "(quotes into the initial consult only; compact scoring does not use them) "
        f"({hint}): "
    ).strip().lower()
    if not answer:
        return default
    if answer in {"y", "yes"}:
        return True
    if answer in {"n", "no"}:
        return False
    raise ValueError("Corpus RAG must be 'y' or 'n'")


def resolve_use_rag(args: argparse.Namespace, *, interactive: bool) -> bool:
    """CLI flags win; otherwise prompt on a TTY, else leave RAG off."""
    if args.skip_original_agents or args.stop_after_world:
        return False
    if args.no_rag_context:
        return False
    if args.use_rag is not None:
        return bool(args.use_rag)
    if interactive:
        return prompt_use_rag(default=False)
    return False


def resolve_max_cycles(args: argparse.Namespace, *, interactive: bool) -> int:
    """Use CLI value when set; otherwise prompt on a TTY, else default to 3."""
    if args.max_cycles is not None:
        return max(1, int(args.max_cycles))
    if interactive:
        return prompt_max_cycles()
    return 3


def correct_mode_typo(value: str) -> str:
    """Recognize close mode-name typos without consuming real questions."""
    normalized = value.strip().lower()
    for mode, prefix in (("workspace", "works"), ("legacy", "lega")):
        if (
            normalized.startswith(prefix)
            and abs(len(normalized) - len(mode)) <= 2
            and SequenceMatcher(None, normalized, mode).ratio() >= 0.80
        ):
            return mode
    return ""


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
        "--max-cycles", str(max(1, args.max_cycles if args.max_cycles is not None else 3)),
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
        "--ev-dominance-ratio", str(max(1.0, args.ev_dominance_ratio)),
    ]
    if args.model:
        command.extend(["--model", str(args.model)])
    if args.actions:
        command.append("--actions")
        command.extend(args.actions)
    if args.agents:
        command.append("--agents")
        command.extend(normalize_agents(args.agents))
    if args.accept_actions:
        command.append("--accept-actions")
    if args.accept_world:
        command.append("--accept-world")
    if args.stop_after_world:
        command.append("--stop-after-world")
    if args.skip_original_agents:
        command.append("--skip-original-agents")
    # Default is off: MiniLM consult is opt-in per run.
    if args.use_rag is True:
        pass
    else:
        command.append("--no-rag-context")
    if args.no_framing_cache:
        command.append("--no-framing-cache")
    if args.no_synthesis:
        command.append("--no-synthesis")
    if args.no_cycle_extension:
        command.append("--no-cycle-extension")
    if args.no_visibility_audit:
        command.append("--no-visibility-audit")
    if args.no_autonomy_audit:
        command.append("--no-autonomy-audit")
    if args.drop_vote_on_graph_rejection:
        command.append("--drop-vote-on-graph-rejection")
    if args.no_ev_dominance_breaker:
        command.append("--no-ev-dominance-breaker")
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
        elif corrected_mode := correct_mode_typo(normalized):
            mode = corrected_mode
            print(
                f"Interpreting '{first_answer}' as mode '{corrected_mode}'."
            )
            question = prompt_question()
        elif normalized in {"local", "openai"}:
            # A common interactive slip is answering the backend question one
            # prompt early. Preserve that intent instead of treating a backend
            # name as a seven-character ethical problem.
            mode = "workspace"
            args.backend = normalized
            print(f"Using {normalized} as the workspace backend.")
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
    # Interactive startups choose the cycle budget unless --max-cycles was passed.
    args.max_cycles = resolve_max_cycles(
        args, interactive=sys.stdin.isatty() and args.question is None,
    )
    if args.agents is not None:
        args.agents = normalize_agents(args.agents)
    elif sys.stdin.isatty() and args.question is None:
        args.agents = prompt_agents()
    else:
        args.agents = list(AGENT_MODULES)
    interactive = sys.stdin.isatty() and args.question is None
    args.use_rag = resolve_use_rag(args, interactive=interactive)
    args.no_rag_context = not args.use_rag
    if args.skip_original_agents:
        print("Original agents skipped; corpus RAG is off.", flush=True)
    elif args.use_rag:
        print("Corpus RAG: on (original-agent consult only).", flush=True)
    else:
        print("Corpus RAG: off.", flush=True)
    return run_workspace(args, question)


if __name__ == "__main__":
    raise SystemExit(main())
