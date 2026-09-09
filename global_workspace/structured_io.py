from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token, copy_context
from dataclasses import dataclass, field
from functools import lru_cache
from threading import Event, RLock
from typing import Any, Iterator

from .performance import record_performance_event


_CACHE_MISS = object()


class ModelCallBudgetExceeded(RuntimeError):
    """Raised before a model call that cannot safely fit in the run budget."""


class ModelCallUnavailable(RuntimeError):
    """Bounded provider failure that must not escape as an orchestration crash."""

    def __init__(
        self, message: str, *, category: str = "transport", terminal: bool = False,
    ) -> None:
        super().__init__(message)
        self.category = str(category).strip().lower() or "transport"
        self.terminal = bool(terminal)


@dataclass
class ModelCallBudget:
    deadline: float
    reserve_seconds: float = 20.0
    max_auxiliary_calls_per_cycle: int = 5
    cycle: int = 0
    auxiliary_calls: int = 0
    epistemic_audit_calls: int = 0
    paused_at: float | None = None
    pause_depth: int = 0
    cache: dict[str, Any] = field(default_factory=dict)
    lock: RLock = field(default_factory=RLock, repr=False)
    terminal_event: Event = field(default_factory=Event, repr=False)


_CALL_BUDGET: ContextVar[ModelCallBudget | None] = ContextVar(
    "workspace_model_call_budget", default=None
)


def start_model_call_budget(
    seconds: float, *, reserve_seconds: float = 20.0,
    max_auxiliary_calls_per_cycle: int = 5,
) -> Token:
    """Install a run-scoped deadline and auxiliary-call allowance."""
    return _CALL_BUDGET.set(ModelCallBudget(
        deadline=time.monotonic() + max(0.0, seconds),
        reserve_seconds=max(0.0, reserve_seconds),
        max_auxiliary_calls_per_cycle=max(0, max_auxiliary_calls_per_cycle),
    ))


def reset_model_call_budget(token: Token) -> None:
    _CALL_BUDGET.reset(token)


def pause_model_call_budget() -> None:
    """Stop the wall clock during interactive waits or out-of-process RAG consults."""
    budget = _CALL_BUDGET.get()
    if budget is None:
        return
    with budget.lock:
        if budget.pause_depth == 0:
            budget.paused_at = time.monotonic()
        budget.pause_depth += 1


def resume_model_call_budget() -> None:
    budget = _CALL_BUDGET.get()
    if budget is None:
        return
    with budget.lock:
        if budget.pause_depth <= 0:
            return
        budget.pause_depth -= 1
        if budget.pause_depth > 0:
            return
        if budget.paused_at is None:
            return
        budget.deadline += time.monotonic() - budget.paused_at
        budget.paused_at = None


@contextmanager
def model_call_budget_paused() -> Iterator[None]:
    pause_model_call_budget()
    try:
        yield
    finally:
        resume_model_call_budget()


def begin_model_call_cycle(cycle: int) -> None:
    budget = _CALL_BUDGET.get()
    if budget is not None:
        with budget.lock:
            if budget.cycle != cycle:
                budget.cycle = cycle
                budget.auxiliary_calls = 0
                budget.epistemic_audit_calls = 0


def submit_with_context(executor: Any, callback: Any, /, *args: Any, **kwargs: Any):
    """Submit a worker with an independent copy of the current context.

    Context values such as the run budget and performance recorder are thereby
    available in the worker. The budget object itself is intentionally shared;
    its lock protects counters and cache across those copied contexts.
    """
    context = copy_context()
    return executor.submit(context.run, callback, *args, **kwargs)


@lru_cache(maxsize=8)
def json_grammar(schema_text: str) -> Any | None:
    """Build a llama.cpp grammar when supported by the active backend."""
    try:
        from llama_cpp import LlamaGrammar

        return LlamaGrammar.from_json_schema(schema_text)
    except (ImportError, AttributeError, ValueError):
        return None


def call_json_llm(
    llm: Any,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    schema: dict[str, Any],
    call_kind: str = "primary",
    cache: bool = False,
    call_metadata: dict[str, Any] | None = None,
) -> Any:
    """Call either the OpenAI structured backend or llama.cpp compatibility API."""
    call_started = time.monotonic()
    budget = _CALL_BUDGET.get()
    if budget is not None:
        with budget.lock:
            budget_cycle = int(budget.cycle)
    else:
        budget_cycle = 0
    event_metadata = {
        "call_kind": call_kind,
        "backend": type(llm).__name__,
        "max_tokens": int(max_tokens),
        "structured": True,
        "cycle": budget_cycle,
        **dict(call_metadata or {}),
    }
    cache_key = json.dumps(
        [prompt, schema, max_tokens, temperature], sort_keys=True, separators=(",", ":")
    )
    if budget is not None and cache:
        with budget.lock:
            cached = budget.cache.get(cache_key, _CACHE_MISS)
        if cached is not _CACHE_MISS:
            record_performance_event(
                call_kind,
                call_started,
                category="model_call",
                status="CACHE_HIT",
                metadata=event_metadata,
            )
            return cached
    call_timeout: float | None = None
    try:
        if budget is not None:
            with budget.lock:
                if budget.terminal_event.is_set():
                    raise ModelCallUnavailable(
                        "model call canceled after a terminal provider failure",
                        category="canceled",
                        terminal=True,
                    )
                remaining = budget.deadline - time.monotonic()
                if budget.paused_at is not None:
                    remaining += time.monotonic() - budget.paused_at
                if remaining <= budget.reserve_seconds:
                    raise ModelCallBudgetExceeded(
                        f"model-call deadline reached ({remaining:.1f}s remaining; "
                        f"{budget.reserve_seconds:.1f}s reserved for finalization)"
                    )
                if call_kind == "auxiliary":
                    if (
                        budget.auxiliary_calls
                        >= budget.max_auxiliary_calls_per_cycle
                    ):
                        raise ModelCallBudgetExceeded(
                            "auxiliary model-call allowance exhausted for this cycle"
                        )
                    budget.auxiliary_calls += 1
                elif call_kind == "epistemic_audit":
                    if budget.epistemic_audit_calls >= 1:
                        raise ModelCallBudgetExceeded(
                            "epistemic side-audit allowance exhausted for this cycle"
                        )
                    budget.epistemic_audit_calls += 1
                # A reasoning-model adapter may retry once. Reserve half of the
                # usable wall clock per attempt without mutating the shared LLM.
                if getattr(llm, "supports_call_local_timeout", False):
                    usable = max(1.0, remaining - budget.reserve_seconds)
                    configured_timeout = float(getattr(llm, "timeout", usable))
                    call_timeout = min(
                        configured_timeout, max(1.0, usable / 2.0)
                    )
        if hasattr(llm, "complete_json"):
            call_kwargs = {
                "schema": schema,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if call_timeout is not None:
                call_kwargs["timeout"] = call_timeout
            result = llm.complete_json(
                prompt,
                **call_kwargs,
            )
        else:
            kwargs = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }
            grammar = json_grammar(json.dumps(schema, sort_keys=True))
            if grammar is not None:
                kwargs["grammar"] = grammar
            result = llm(prompt, **kwargs)
    except Exception as exc:
        if (
            budget is not None
            and isinstance(exc, ModelCallUnavailable)
            and exc.terminal
        ):
            budget.terminal_event.set()
        error_category = str(getattr(exc, "category", "") or "")
        record_performance_event(
            call_kind,
            call_started,
            category="model_call",
            status=(
                "BUDGET_BLOCKED"
                if isinstance(exc, ModelCallBudgetExceeded)
                else "ERROR"
            ),
            metadata={
                **event_metadata,
                "exception_type": type(exc).__name__,
                **({"error_category": error_category} if error_category else {}),
            },
        )
        raise
    if budget is not None and cache:
        with budget.lock:
            budget.cache[cache_key] = result
    record_performance_event(
        call_kind,
        call_started,
        category="model_call",
        status="OK",
        metadata=event_metadata,
    )
    return result


def extract_json(text: str) -> dict[str, Any]:
    """Extract one JSON object from strict or lightly wrapped model output."""
    text = text.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Local model returned no JSON object: {text[:160]!r}")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Local model response must be a JSON object")
    return value


def number(value: Any, default: float = 0.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def strict_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return parsed


def structured_text_error(value: Any, *, minimum_words: int = 3) -> str:
    """Detect corrupted prose that can survive a JSON-schema string field.

    Structured output guarantees the outer JSON shape, not that a string value
    contains prose.  The checks are deliberately narrow: replacement glyphs,
    leaked key/value syntax, repeated closing delimiters, and strings dominated
    by punctuation. Ordinary ethical prose and mathematical notation remain valid.
    """
    text = " ".join(str(value).split())
    words = re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?", text)
    if len(words) < minimum_words:
        return "contains too little explanatory prose"
    if "\ufffd" in text or "\ufffc" in text:
        return "contains Unicode replacement artifacts"
    if re.search(r"(?:[}\]])(?:\s*[,;:]?\s*[}\]]){2,}", text):
        return "contains repeated serialized delimiters"
    if re.search(r"['\"]\s*[,}]\s*['\"]?[A-Za-z_]{1,12}['\"]?\s*:", text):
        return "contains leaked serialized fields"
    nonspace = [char for char in text if not char.isspace()]
    punctuation = [
        char for char in nonspace
        if not char.isalnum() and char not in "'’-–—%<>=+./()"
    ]
    if nonspace and len(punctuation) / len(nonspace) > 0.22:
        return "is dominated by serialization punctuation"
    return ""
