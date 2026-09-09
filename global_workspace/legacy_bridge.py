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
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event

from global_workspace.framework_retrieval import retrieve_framework_evidence
from global_workspace.retrieval_trace import (
    RETRIEVAL_MARKER,
    annotate_retrieval_use,
    disabled_retrieval_result,
    rag_cache_token,
    serialize_retrieval_result,
    skipped_retrieval_record,
)
from global_workspace.source_cache import build_source_cache_key
from global_workspace.performance import record_performance_duration
from global_workspace.structured_io import ModelCallUnavailable, submit_with_context


AGENT_MODULES = {
    "utilitarian": "utilitarian_agent_p",
    "deontological": "deontological_agent_p",
    "virtue": "virtue_ethics_agent_p",
    "care": "care_ethics_agent_p",
    "rawlsian": "rawlsian_ethics_agent_p",
}
RESPONSE_MARKER = "WORKSPACE_RESPONSE_B64="
PERFORMANCE_MARKER = "WORKSPACE_PERFORMANCE_B64="
ERROR_MARKER = "WORKSPACE_ERROR_B64="


@dataclass(slots=True)
class LegacyConsultation:
    testimonies: dict[str, str]
    errors: dict[str, str]
    retrievals: dict[str, dict] = field(default_factory=dict)


@dataclass(slots=True)
class _OriginalAgentOutcome:
    agent: str
    testimony: str = ""
    error: str = ""
    retrieval: dict = field(default_factory=dict)
    terminal_category: str = ""


def _timed_model_call(model, timing: dict[str, float | int], *args, **kwargs):
    started = time.monotonic()
    timing["model_call_count"] = int(timing.get("model_call_count", 0)) + 1
    try:
        return model(*args, **kwargs)
    except Exception as exc:
        category = str(getattr(exc, "category", "") or "").casefold()
        if (
            category == "timeout"
            or "timeout" in type(exc).__name__.casefold()
            or "timed out" in str(exc).casefold()
        ):
            timing["timeout_count"] = int(timing.get("timeout_count", 0)) + 1
        raise
    finally:
        timing["model_seconds"] = float(timing.get("model_seconds", 0.0)) + (
            time.monotonic() - started
        )


class _TimedModel:
    def __init__(self, model, timing: dict[str, float | int]):
        self._model = model
        self._timing = timing

    def __call__(self, *args, **kwargs):
        return _timed_model_call(self._model, self._timing, *args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._model, name)


class _LazyLocalTimedModel:
    """Preserve source-response cache hits without eagerly loading llama.cpp."""

    def __init__(self, module, timing: dict[str, float | int]):
        self._module = module
        self._timing = timing
        self._model = None

    def _load(self):
        if self._model is None:
            started = time.monotonic()
            from llama_cpp import Llama

            self._model = Llama(
                model_path=str(self._module.MODEL_PATH),
                n_ctx=768,
                n_threads=6,
                n_gpu_layers=60,
                n_batch=64,
                verbose=False,
            )
            self._timing["model_startup_seconds"] = float(
                self._timing.get("model_startup_seconds", 0.0)
            ) + (time.monotonic() - started)
        return self._model

    def __call__(self, *args, **kwargs):
        return _timed_model_call(
            self._load(), self._timing, *args, **kwargs,
        )


def _install_retrieval_hooks(
    module,
    *,
    disable_rag: bool,
    timing: dict[str, float | int],
) -> None:
    """Capture MiniLM hits in the child and keep RAG on/off out of the LLM cache key."""

    def hooked_retrieve(*args, **kwargs):
        started = time.monotonic()
        timing["retrieval_call_count"] = int(
            timing.get("retrieval_call_count", 0)
        ) + 1
        try:
            if disable_rag:
                query = str(kwargs.get("query") or "")
                lens = str(kwargs.get("query_lens") or "")
                empty = disabled_retrieval_result(query, lens)
                module.LAST_RETRIEVAL = serialize_retrieval_result(empty, mode="disabled")
                return empty
            result = retrieve_framework_evidence(*args, **kwargs)
            module.LAST_RETRIEVAL = serialize_retrieval_result(result, mode="retrieved")
            return result
        finally:
            timing["retrieval_seconds"] = float(
                timing.get("retrieval_seconds", 0.0)
            ) + (time.monotonic() - started)

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
    child_started = time.monotonic()
    timing: dict[str, float | int] = {
        "module_startup_seconds": 0.0,
        "model_startup_seconds": 0.0,
        "retrieval_seconds": 0.0,
        "model_seconds": 0.0,
        "retrieval_call_count": 0,
        "model_call_count": 0,
        "timeout_count": 0,
    }
    if agent not in AGENT_MODULES:
        raise ValueError(f"Unknown original agent: {agent}")
    module_started = time.monotonic()
    data = json.loads(scenario_path.read_text(encoding="utf-8"))
    question = str(data.get("ethical_question", "")).strip()
    if not question:
        raise ValueError("Scenario has no ethical_question")
    module = importlib.import_module(AGENT_MODULES[agent])
    timing["module_startup_seconds"] = time.monotonic() - module_started
    _install_retrieval_hooks(
        module, disable_rag=disable_rag, timing=timing,
    )
    llm = None
    if backend == "openai":
        from .openai_backend import OpenAIWorkspaceLLM

        model_started = time.monotonic()
        llm = _TimedModel(OpenAIWorkspaceLLM(openai_model), timing)
        timing["model_startup_seconds"] = time.monotonic() - model_started
        # Keep hosted answers from overwriting caches used by local-model runs.
        for cache_name in ("LAST_QUERY_PATH", "LAST_RESPONSE_PATH"):
            cache_path = getattr(module, cache_name, None)
            if isinstance(cache_path, Path):
                setattr(
                    module,
                    cache_name,
                    cache_path.with_name(f"{cache_path.stem}_openai{cache_path.suffix}"),
                )
    else:
        llm = _LazyLocalTimedModel(module, timing)
    try:
        response = module.respond_to_query(
            query=question,
            scenario_id=scenario_path.stem,
            scenario_path=scenario_path,
            max_tokens=max_tokens,
            llm=llm,
        )
    finally:
        timing["child_total_seconds"] = time.monotonic() - child_started
        performance_encoded = base64.b64encode(
            json.dumps(timing, sort_keys=True).encode("utf-8")
        ).decode("ascii")
        print(f"{PERFORMANCE_MARKER}{performance_encoded}")
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


def _decode_child_error(stdout: str) -> dict[str, object]:
    payload = _decode_marked_payload(stdout, ERROR_MARKER)
    if not payload:
        return {}
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _decode_retrieval(stdout: str) -> dict:
    payload = _decode_marked_payload(stdout, RETRIEVAL_MARKER)
    if not payload:
        return skipped_retrieval_record("untraced")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return skipped_retrieval_record("untraced")
    return data if isinstance(data, dict) else skipped_retrieval_record("untraced")


def _decode_performance(stdout: str) -> dict[str, float | int]:
    payload = _decode_marked_payload(stdout, PERFORMANCE_MARKER)
    if not payload:
        return {}
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _record_original_agent_performance(
    agent: str,
    *,
    process_seconds: float,
    timing: dict[str, float | int],
    status: str,
    timed_out: bool = False,
) -> None:
    child_total = float(timing.get("child_total_seconds", 0.0) or 0.0)
    process_startup = max(0.0, process_seconds - child_total)
    module_startup = float(timing.get("module_startup_seconds", 0.0) or 0.0)
    model_startup = float(timing.get("model_startup_seconds", 0.0) or 0.0)
    retrieval_seconds = float(timing.get("retrieval_seconds", 0.0) or 0.0)
    model_seconds = float(timing.get("model_seconds", 0.0) or 0.0)
    startup_seconds = (
        process_seconds
        if not timing
        else process_startup + module_startup + model_startup
    )
    accounted = startup_seconds + retrieval_seconds + model_seconds
    overhead_seconds = max(0.0, process_seconds - accounted)
    common = {"agent": agent}
    record_performance_duration(
        "original_agent_process",
        process_seconds,
        category="original_agent",
        status=status,
        metadata={
            **common,
            "timeout_count": (
                1 if timed_out else int(timing.get("timeout_count", 0) or 0)
            ),
        },
    )
    record_performance_duration(
        "original_agent_startup",
        startup_seconds,
        category="original_agent_phase",
        status=status,
        metadata={
            **common,
            "process_startup_seconds": round(process_startup, 6),
            "module_startup_seconds": round(module_startup, 6),
            "model_startup_seconds": round(model_startup, 6),
        },
    )
    record_performance_duration(
        "original_agent_retrieval",
        retrieval_seconds,
        category="original_agent_phase",
        status=status,
        metadata={
            **common,
            "retrieval_call_count": int(
                timing.get("retrieval_call_count", 0) or 0
            ),
        },
    )
    record_performance_duration(
        "original_agent_model",
        model_seconds,
        category="original_agent_phase",
        status=status,
        metadata={
            **common,
            "model_call_count": int(timing.get("model_call_count", 0) or 0),
        },
    )
    record_performance_duration(
        "original_agent_overhead",
        overhead_seconds,
        category="original_agent_phase",
        status=status,
        metadata=common,
    )


class _OriginalConsultCancelled(RuntimeError):
    pass


def _run_cancellable_subprocess(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: float,
    cancel_event: Event,
) -> subprocess.CompletedProcess[str]:
    """Run one child while allowing a sibling's terminal error to stop it."""
    if cancel_event.is_set():
        raise _OriginalConsultCancelled(
            "canceled before launch after terminal provider failure"
        )
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + max(0.1, timeout_seconds)
    while True:
        if cancel_event.is_set():
            process.terminate()
            try:
                process.communicate(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
            raise _OriginalConsultCancelled(
                "canceled after terminal provider failure"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(
                command,
                timeout_seconds,
                output=stdout,
                stderr=stderr,
            )
        try:
            stdout, stderr = process.communicate(timeout=min(0.10, remaining))
        except subprocess.TimeoutExpired:
            continue
        return subprocess.CompletedProcess(
            command,
            process.returncode,
            stdout,
            stderr,
        )


def _consult_one_original_agent(
    agent: str,
    *,
    scenario_path: Path,
    max_tokens: int,
    timeout_seconds: float,
    backend: str,
    openai_model: str,
    disable_rag: bool,
    cancel_event: Event | None = None,
    cancellable: bool = False,
) -> _OriginalAgentOutcome:
    command = [
        sys.executable,
        "-m",
        "global_workspace.legacy_bridge",
        "--agent",
        agent,
        "--scenario",
        str(scenario_path),
        "--max-tokens",
        str(max_tokens),
        "--backend",
        backend,
        "--openai-model",
        openai_model,
    ]
    if disable_rag:
        command.append("--disable-rag")
    process_started = time.monotonic()
    child_env = os.environ.copy()
    child_env.setdefault("HF_HUB_OFFLINE", "1")
    child_env.setdefault("TRANSFORMERS_OFFLINE", "1")
    child_env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    child_env["ETHICS_LLM_BACKEND"] = backend
    child_env["ETHICS_DISABLE_RAG"] = "1" if disable_rag else "0"
    try:
        if cancellable and cancel_event is not None:
            completed = _run_cancellable_subprocess(
                command,
                cwd=Path(__file__).resolve().parent.parent,
                env=child_env,
                timeout_seconds=timeout_seconds,
                cancel_event=cancel_event,
            )
        else:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parent.parent,
                env=child_env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        process_seconds = time.monotonic() - process_started
        child_timing = _decode_performance(completed.stdout)
        _record_original_agent_performance(
            agent,
            process_seconds=process_seconds,
            timing=child_timing,
            status=("OK" if completed.returncode == 0 else "ERROR"),
        )
        if completed.returncode != 0:
            child_error = _decode_child_error(completed.stdout)
            category = str(child_error.get("category", "")).casefold()
            terminal = bool(child_error.get("terminal", False)) and category in {
                "authentication", "quota", "quota_or_rate_limit",
            }
            if terminal and cancel_event is not None:
                cancel_event.set()
            detail = completed.stderr.strip().splitlines()
            message = str(child_error.get("message", "")).strip()
            return _OriginalAgentOutcome(
                agent=agent,
                error=(
                    message[:500]
                    if message else detail[-1][:500]
                    if detail else f"exit code {completed.returncode}"
                ),
                terminal_category=category if terminal else "",
            )
        testimony = _decode_response(completed.stdout)
        retrieval = annotate_retrieval_use(
            _decode_retrieval(completed.stdout),
            testimony=testimony,
        )
        return _OriginalAgentOutcome(
            agent=agent,
            testimony=testimony,
            error="" if testimony else "empty response",
            retrieval=retrieval,
        )
    except subprocess.TimeoutExpired as exc:
        _record_original_agent_performance(
            agent,
            process_seconds=time.monotonic() - process_started,
            timing={},
            status="TIMEOUT",
            timed_out=True,
        )
        return _OriginalAgentOutcome(agent=agent, error=str(exc)[:500])
    except _OriginalConsultCancelled as exc:
        _record_original_agent_performance(
            agent,
            process_seconds=time.monotonic() - process_started,
            timing={},
            status="CANCELED",
        )
        return _OriginalAgentOutcome(agent=agent, error=str(exc)[:500])
    except (ValueError, UnicodeError) as exc:
        return _OriginalAgentOutcome(agent=agent, error=str(exc)[:500])


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
    max_concurrency: int = 2,
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
        worker_count = (
            min(max(1, int(max_concurrency)), 3, len(agents))
            if backend == "openai" and agents else 1
        )
        outcomes: list[_OriginalAgentOutcome | None] = [None for _ in agents]
        if worker_count == 1:
            for index, agent in enumerate(agents):
                outcome = _consult_one_original_agent(
                    agent,
                    scenario_path=effective_scenario_path,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                    backend=backend,
                    openai_model=openai_model,
                    disable_rag=disable_rag,
                )
                outcomes[index] = outcome
                if outcome.terminal_category:
                    for remaining_index in range(index + 1, len(agents)):
                        outcomes[remaining_index] = _OriginalAgentOutcome(
                            agent=agents[remaining_index],
                            error="canceled after terminal provider failure",
                        )
                    break
        else:
            cancel_event = Event()
            executor = ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="original-agent",
            )
            future_indexes = {}
            try:
                for index, agent in enumerate(agents):
                    future = submit_with_context(
                        executor,
                        _consult_one_original_agent,
                        agent,
                        scenario_path=effective_scenario_path,
                        max_tokens=max_tokens,
                        timeout_seconds=timeout_seconds,
                        backend=backend,
                        openai_model=openai_model,
                        disable_rag=disable_rag,
                        cancel_event=cancel_event,
                        cancellable=True,
                    )
                    future_indexes[future] = index
                for future in as_completed(future_indexes):
                    index = future_indexes[future]
                    try:
                        outcome = future.result()
                    except CancelledError:
                        continue
                    except Exception as exc:
                        outcome = _OriginalAgentOutcome(
                            agent=agents[index],
                            error=(
                                "parallel original-agent worker failed: "
                                f"{type(exc).__name__}: {exc}"
                            )[:500],
                        )
                    outcomes[index] = outcome
                    if outcome.terminal_category:
                        cancel_event.set()
                        for remaining in future_indexes:
                            if remaining is not future:
                                remaining.cancel()
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
            for index, outcome in enumerate(outcomes):
                if outcome is None:
                    outcomes[index] = _OriginalAgentOutcome(
                        agent=agents[index],
                        error="canceled after terminal provider failure",
                    )

        # Insert results only in configured order, regardless of completion order.
        for outcome in outcomes:
            if outcome is None:
                continue
            if outcome.testimony:
                testimonies[outcome.agent] = outcome.testimony
            if outcome.retrieval:
                retrievals[outcome.agent] = outcome.retrieval
            if outcome.error:
                errors[outcome.agent] = outcome.error
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
    try:
        child_status = _child_consult(
            child_args.agent,
            child_args.scenario.resolve(),
            child_args.max_tokens,
            child_args.backend,
            child_args.openai_model,
            child_args.disable_rag,
        )
    except ModelCallUnavailable as exc:
        error_payload = base64.b64encode(json.dumps({
            "category": exc.category,
            "terminal": exc.terminal,
            "message": str(exc),
        }, sort_keys=True).encode("utf-8")).decode("ascii")
        print(f"{ERROR_MARKER}{error_payload}")
        child_status = 3
    raise SystemExit(child_status)
