from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterator


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
    cache: dict[str, Any] = field(default_factory=dict)


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
    if budget is None or budget.paused_at is not None:
        return
    budget.paused_at = time.monotonic()


def resume_model_call_budget() -> None:
    budget = _CALL_BUDGET.get()
    if budget is None or budget.paused_at is None:
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
    if budget is not None and budget.cycle != cycle:
        budget.cycle = cycle
        budget.auxiliary_calls = 0
        budget.epistemic_audit_calls = 0


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
) -> Any:
    """Call either the OpenAI structured backend or llama.cpp compatibility API."""
    budget = _CALL_BUDGET.get()
    cache_key = json.dumps(
        [prompt, schema, max_tokens, temperature], sort_keys=True, separators=(",", ":")
    )
    if budget is not None:
        if cache and cache_key in budget.cache:
            return budget.cache[cache_key]
        remaining = budget.deadline - time.monotonic()
        if budget.paused_at is not None:
            remaining += time.monotonic() - budget.paused_at
        if remaining <= budget.reserve_seconds:
            raise ModelCallBudgetExceeded(
                f"model-call deadline reached ({remaining:.1f}s remaining; "
                f"{budget.reserve_seconds:.1f}s reserved for finalization)"
            )
        if call_kind == "auxiliary":
            if budget.auxiliary_calls >= budget.max_auxiliary_calls_per_cycle:
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
        # The OpenAI adapter may retry a length-limited response once. Bound each
        # attempt to half the usable remainder so retries cannot consume the
        # finalization reserve. Local llama.cpp backends have no timeout attribute.
        if hasattr(llm, "timeout"):
            usable = max(1.0, remaining - budget.reserve_seconds)
            llm.timeout = min(float(llm.timeout), max(1.0, usable / 2.0))
    if hasattr(llm, "complete_json"):
        result = llm.complete_json(
            prompt,
            schema=schema,
            max_tokens=max_tokens,
            temperature=temperature,
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
    if budget is not None and cache:
        budget.cache[cache_key] = result
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
