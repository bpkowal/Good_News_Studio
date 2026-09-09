from __future__ import annotations

import argparse
import base64
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from global_workspace.framework_retrieval import RetrievalResult, retrieve_framework_evidence
from global_workspace.retrieval_trace import (
    RETRIEVAL_MARKER,
    annotate_retrieval_use,
    rag_cache_token,
    serialize_retrieval_result,
    skipped_retrieval_record,
)
from global_workspace.source_cache import build_source_cache_key


AGENT_MODULES = {
    "utilitarian": "utilitarian_agent_p",
    "deontological": "deontological_agent_p",
    "virtue": "virtue_ethics_agent_p",
    "care": "care_ethics_agent_p",
    "rawlsian": "rawlsian_ethics_agent_p",
}
RESPONSE_MARKER = "WORKSPACE_RESPONSE_B64="


@dataclass(slots=True)
class LegacyConsultation:
    testimonies: dict[str, str]
    errors: dict[str, str]
    retrievals: dict[str, dict] = field(default_factory=dict)


def _install_retrieval_hooks(module, *, disable_rag: bool) -> None:
    """Capture MiniLM hits in the child and keep RAG on/off out of the LLM cache key."""

    def hooked_retrieve(*args, **kwargs):
        if disable_rag:
            query = str(kwargs.get("query") or "")
            lens = str(kwargs.get("query_lens") or "")
            empty = RetrievalResult(
                (), 0, 0, None, f"{lens}\nCase: {query}".strip(),
            )
            module.LAST_RETRIEVAL = serialize_retrieval_result(empty, mode="disabled")
            return empty
        result = retrieve_framework_evidence(*args, **kwargs)
        module.LAST_RETRIEVAL = serialize_retrieval_result(result, mode="retrieved")
        return result

    def hooked_cache_key(*args, **kwargs):
        return build_source_cache_key(*args, **kwargs) + "\n" + rag_cache_token(
            disabled=disable_rag
        )

    module.retrieve_framework_evidence = hooked_retrieve
    module.build_source_cache_key = hooked_cache_key


def _child_consult(
    agent: str,
    scenario_path: Path,
    max_tokens: int,
    backend: str = "local",
    openai_model: str = "o3",
    disable_rag: bool = False,
) -> int:
    if agent not in AGENT_MODULES:
        raise ValueError(f"Unknown original agent: {agent}")
    data = json.loads(scenario_path.read_text(encoding="utf-8"))
    question = str(data.get("ethical_question", "")).strip()
    if not question:
        raise ValueError("Scenario has no ethical_question")
    module = importlib.import_module(AGENT_MODULES[agent])
    _install_retrieval_hooks(module, disable_rag=disable_rag)
    llm = None
    if backend == "openai":
        from .openai_backend import OpenAIWorkspaceLLM

        llm = OpenAIWorkspaceLLM(openai_model)
        # Keep hosted answers from overwriting caches used by local-model runs.
        for cache_name in ("LAST_QUERY_PATH", "LAST_RESPONSE_PATH"):
            cache_path = getattr(module, cache_name, None)
            if isinstance(cache_path, Path):
                setattr(
                    module,
                    cache_name,
                    cache_path.with_name(f"{cache_path.stem}_openai{cache_path.suffix}"),
                )
    response = module.respond_to_query(
        query=question,
        scenario_id=scenario_path.stem,
        scenario_path=scenario_path,
        max_tokens=max_tokens,
        llm=llm,
    )
    encoded = base64.b64encode(str(response).encode("utf-8")).decode("ascii")
    print(f"{RESPONSE_MARKER}{encoded}")
    retrieval = getattr(module, "LAST_RETRIEVAL", None)
    if not isinstance(retrieval, dict):
        retrieval = skipped_retrieval_record(
            "disabled" if disable_rag else "cache_hit_untraced"
        )
    retrieval_encoded = base64.b64encode(
        json.dumps(retrieval, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    print(f"{RETRIEVAL_MARKER}{retrieval_encoded}")
    return 0


def _decode_marked_payload(stdout: str, marker: str) -> str | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(marker):
            payload = line[len(marker):]
            return base64.b64decode(payload).decode("utf-8").strip()
    return None


def _decode_response(stdout: str) -> str:
    payload = _decode_marked_payload(stdout, RESPONSE_MARKER)
    if payload is None:
        raise ValueError("Original agent produced no bridge response marker")
    return payload


def _decode_retrieval(stdout: str) -> dict:
    payload = _decode_marked_payload(stdout, RETRIEVAL_MARKER)
    if not payload:
        return skipped_retrieval_record("untraced")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return skipped_retrieval_record("untraced")
    return data if isinstance(data, dict) else skipped_retrieval_record("untraced")


def consult_original_agents(
    scenario_path: Path,
    *,
    agents: tuple[str, ...] = tuple(AGENT_MODULES),
    max_tokens: int = 180,
    timeout_seconds: float = 600.0,
    backend: str = "local",
    openai_model: str = "o3",
    canonical_actions: tuple[str, ...] = (),
    canonical_scenario: str = "",
    disable_rag: bool = False,
) -> LegacyConsultation:
    testimonies: dict[str, str] = {}
    errors: dict[str, str] = {}
    retrievals: dict[str, dict] = {}
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    effective_scenario_path = scenario_path
    if canonical_actions:
        original = json.loads(scenario_path.read_text(encoding="utf-8"))
        question = str(original.get("ethical_question", "")).strip()
        # When the author supplied Option A/B, the canonical action descriptions
        # already retain both branches and consequences. Remove the order-bearing
        # option block from the background so source agents do not see two rival
        # label systems. For other scenario forms, retain the full factual text.
        option_start = re.search(
            r"\b(?:action|option)\s+A\s*:", question, flags=re.IGNORECASE
        )
        background = question[:option_start.start()].strip() if option_start else question
        canonical_question = canonical_scenario.strip() or "\n".join([
            background,
            "Presentation order has no ethical significance. Use this authoritative action mapping:",
            *(
                f"Action A{index}: {action}"
                for index, action in enumerate(canonical_actions)
            ),
            "Evaluate these physical actions; do not reuse labels from any earlier presentation.",
        ]).strip()
        original["ethical_question"] = canonical_question
        original["canonical_action_legend"] = {
            f"A{index}": action for index, action in enumerate(canonical_actions)
        }
        temporary_directory = tempfile.TemporaryDirectory(prefix="ethics-canonical-")
        effective_scenario_path = Path(temporary_directory.name) / scenario_path.name
        effective_scenario_path.write_text(
            json.dumps(original, indent=2), encoding="utf-8"
        )
    try:
        for agent in agents:
            print(f"Consulting original {agent} agent...", flush=True)
            command = [
                sys.executable,
                "-m",
                "global_workspace.legacy_bridge",
                "--agent",
                agent,
                "--scenario",
                str(effective_scenario_path),
                "--max-tokens",
                str(max_tokens),
                "--backend",
                backend,
                "--openai-model",
                openai_model,
            ]
            if disable_rag:
                command.append("--disable-rag")
            try:
                child_env = os.environ.copy()
                # Avoid remote HEAD requests when the embedding model is already cached.
                child_env.setdefault("HF_HUB_OFFLINE", "1")
                child_env.setdefault("TRANSFORMERS_OFFLINE", "1")
                child_env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
                child_env["ETHICS_LLM_BACKEND"] = backend
                child_env["ETHICS_DISABLE_RAG"] = "1" if disable_rag else "0"
                completed = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parent.parent,
                    env=child_env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                if completed.returncode != 0:
                    detail = completed.stderr.strip().splitlines()
                    errors[agent] = detail[-1][:500] if detail else f"exit code {completed.returncode}"
                    continue
                testimony = _decode_response(completed.stdout)
                retrievals[agent] = annotate_retrieval_use(
                    _decode_retrieval(completed.stdout),
                    testimony=testimony,
                )
                if testimony:
                    testimonies[agent] = testimony
                else:
                    errors[agent] = "empty response"
            except (subprocess.TimeoutExpired, ValueError, UnicodeError) as exc:
                errors[agent] = str(exc)[:500]
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()
    return LegacyConsultation(
        testimonies=testimonies, errors=errors, retrievals=retrievals,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=tuple(AGENT_MODULES))
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--backend", choices=("local", "openai"), default="local")
    parser.add_argument("--openai-model", default="o3")
    parser.add_argument("--disable-rag", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    child_args = _parse_args()
    raise SystemExit(_child_consult(
        child_args.agent,
        child_args.scenario.resolve(),
        child_args.max_tokens,
        child_args.backend,
        child_args.openai_model,
        child_args.disable_rag,
    ))
