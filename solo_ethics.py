"""Run one ethical question through a single, unassisted OpenAI model.

Interactive use:
    python solo_ethics.py

Non-interactive use:
    python solo_ethics.py --question "Should ...?"

The solo runner is intentionally not given Parliament schemas, graph validators,
specialist prompts, or semantic middleware. Its only architectural concession is
a completion-token ceiling comparable to prior Parliament runs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

from dotenv import load_dotenv

from global_workspace.openai_backend import OpenAIWorkspaceLLM


ROOT = Path(__file__).resolve().parent
DEFAULT_PARLIAMENT_EQUIVALENT_TOKENS = 16_384
MODEL_COMPLETION_TOKEN_CAP = 100_000


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a solo OpenAI ethical analysis.")
    parser.add_argument("--question", help="Ethical problem; prompts when omitted")
    parser.add_argument("--model", default="o3")
    parser.add_argument(
        "--reasoning-effort", choices=("low", "medium", "high"), default="high",
        help="Reasoning effort for supported OpenAI reasoning models",
    )
    parser.add_argument(
        "--max-tokens", type=int,
        help="Completion-token ceiling; default is the measured Parliament-run average",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "solo_outputs")
    parser.add_argument(
        "--parliament-usage-root", type=Path, default=ROOT / "eval_outputs",
        help="Directory searched for Parliament usage logs when calculating the default budget",
    )
    return parser.parse_args(argv)


def prompt_question() -> str:
    question = input("Enter an ethical problem: ").strip()
    if len(question) < 10:
        raise ValueError("The ethical problem must contain at least 10 characters")
    return question


def build_prompt(question: str) -> str:
    """Use an ordinary user-level prompt without Parliament scaffolding."""
    return (
        "Analyze the following ethical problem and give your best answer. "
        "Explain your reasoning clearly.\n\n"
        f"{question}"
    )


def _completion_tokens(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            total += int((record.get("usage") or {}).get("completion_tokens") or 0)
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return total


def parliament_average_completion_tokens(
    root: Path, *, fallback: int = DEFAULT_PARLIAMENT_EQUIVALENT_TOKENS,
) -> tuple[int, int]:
    """Return mean completion usage per recorded Parliament run and sample count."""
    samples = [
        value for value in (
            _completion_tokens(path)
            for path in root.rglob("parliament_usage.jsonl")
        ) if value > 0
    ] if root.exists() else []
    if not samples:
        return fallback, 0
    average = math.ceil(sum(samples) / len(samples))
    return min(average, MODEL_COMPLETION_TOKEN_CAP), len(samples)


@contextmanager
def _usage_log(path: Path) -> Iterator[None]:
    previous = os.environ.get("ETHICS_USAGE_LOG")
    os.environ["ETHICS_USAGE_LOG"] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("ETHICS_USAGE_LOG", None)
        else:
            os.environ["ETHICS_USAGE_LOG"] = previous


def render(result: dict[str, object]) -> str:
    answer = str(result.get("answer", "")).strip()
    return answer + ("\n" if answer else "")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv()
    question = (args.question or prompt_question()).strip()
    if len(question) < 10:
        raise ValueError("The ethical problem must contain at least 10 characters")
    measured_budget, sample_count = parliament_average_completion_tokens(
        args.parliament_usage_root.resolve()
    )
    budget = min(
        max(1024, args.max_tokens if args.max_tokens is not None else measured_budget),
        MODEL_COMPLETION_TOKEN_CAP,
    )
    basis = (
        f"historical Parliament average across {sample_count} runs"
        if args.max_tokens is None and sample_count
        else "fallback Parliament-equivalent budget"
        if args.max_tokens is None
        else "explicit --max-tokens override"
    )

    print(f"Using OpenAI solo model: {args.model}", flush=True)
    print(f"Solo completion-token ceiling: {budget} ({basis})", flush=True)
    llm = OpenAIWorkspaceLLM(args.model, timeout=max(1.0, args.timeout))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = args.output_dir / f"solo_{stamp}.json"
    text_path = json_path.with_suffix(".txt")
    usage_path = args.output_dir / f"solo_{stamp}_usage.jsonl"
    with _usage_log(usage_path):
        raw = llm(
            build_prompt(question),
            max_tokens=budget,
            temperature=0.0,
            reasoning_effort=args.reasoning_effort,
            retry_on_empty=False,
        )
    answer = str(raw["choices"][0]["text"]).strip()
    result: dict[str, object] = {
        "ethical_question": question,
        "model": args.model,
        "answer": answer,
        "max_completion_tokens": budget,
        "reasoning_effort": args.reasoning_effort,
        "token_budget_basis": basis,
        "actual_completion_tokens": _completion_tokens(usage_path),
        "response_protocol": "plain_text_single_call_v1",
    }
    answer = render(result)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    text_path.write_text(answer, encoding="utf-8")
    print("\n" + answer)
    print(f"Saved structured result: {json_path}")
    print(f"Saved readable answer: {text_path}")
    print(f"Saved token usage: {usage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
