from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .structured_io import ModelCallUnavailable
from .performance import record_performance_event


def uses_hidden_reasoning(model: str) -> bool:
    """Chat Completions models that reject temperature and use a reasoning budget."""
    name = str(model or "").casefold()
    return name.startswith(("o1", "o3", "o4", "gpt-5"))


class OpenAIWorkspaceLLM:
    """Expose an OpenAI chat model through the small callable API used by delegates."""

    supports_call_local_timeout = True
    supports_concurrent_calls = True

    def __init__(self, model: str = "o3", *, timeout: float = 120.0, client: Any | None = None):
        if client is None:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY is required for --backend openai; add it to .env or the shell."
                )
            client = OpenAI(api_key=api_key)
        self.client = client
        self.model = model
        self.timeout = timeout
        self._usage_lock = Lock()

    @staticmethod
    def _plain(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, dict):
            return value
        return {
            name: getattr(value, name)
            for name in (
                "prompt_tokens", "completion_tokens", "total_tokens",
                "prompt_tokens_details", "completion_tokens_details",
            )
            if hasattr(value, name)
        }

    def _record_usage(self, response: Any, *, call_kind: str, attempt: int) -> None:
        """Append token counts only; prompts and model outputs are deliberately excluded."""
        destination = os.getenv("ETHICS_USAGE_LOG", "").strip()
        if not destination:
            return
        usage = self._plain(getattr(response, "usage", None)) or {}
        choice = response.choices[0] if getattr(response, "choices", None) else None
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "call_kind": call_kind,
            "attempt": attempt,
            "finish_reason": getattr(choice, "finish_reason", "unknown"),
            "usage": usage,
        }
        path = Path(destination).expanduser()
        # Compact specialists share one OpenAI adapter during fan-out. Keep its
        # optional telemetry line-oriented when several responses finish at once.
        with self._usage_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _is_output_limit_error(error: Exception) -> bool:
        message = str(error).casefold()
        status = getattr(error, "status_code", None)
        return bool(
            (status in {None, 400})
            and (
                "max_tokens" in message
                or "max_completion_tokens" in message
                or "output limit was reached" in message
            )
        )

    @staticmethod
    def _bounded_provider_failure(error: Exception) -> ModelCallUnavailable | None:
        """Classify transport/service failures without masking request/code errors."""
        name = type(error).__name__.casefold()
        status = getattr(error, "status_code", None)
        message = str(error).casefold()
        if status in {401, 403} or "authentication" in name:
            return ModelCallUnavailable(
                "OpenAI authentication failed", category="authentication", terminal=True,
            )
        if "insufficient_quota" in message or "quota" in message:
            return ModelCallUnavailable(
                "OpenAI quota prevented the model call",
                category="quota", terminal=True,
            )
        if status == 429 or "ratelimit" in name:
            return ModelCallUnavailable(
                "OpenAI rate limit prevented the model call",
                category="rate_limit", terminal=False,
            )
        if status == 408 or "timeout" in name or "timed out" in message:
            return ModelCallUnavailable(
                "OpenAI model call timed out", category="timeout", terminal=False,
            )
        if "connection" in name or "connect" in name:
            return ModelCallUnavailable(
                "OpenAI connection failed", category="connection", terminal=False,
            )
        if isinstance(status, int) and status >= 500:
            return ModelCallUnavailable(
                f"OpenAI service returned HTTP {status}",
                category="service", terminal=False,
            )
        return None

    def _create(self, request: dict[str, Any]) -> Any:
        started = time.monotonic()
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as error:
            bounded = self._bounded_provider_failure(error)
            record_performance_event(
                "openai.chat.completions",
                started,
                category="provider_attempt",
                status="ERROR",
                metadata={
                    "model": self.model,
                    "exception_type": type(error).__name__,
                    "error_category": bounded.category if bounded is not None else "request",
                },
            )
            if bounded is not None:
                raise bounded from error
            raise
        record_performance_event(
            "openai.chat.completions",
            started,
            category="provider_attempt",
            status="OK",
            metadata={"model": self.model},
        )
        return response

    def complete_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        max_tokens: int,
        temperature: float,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,
            "timeout": (
                self.timeout if timeout is None else max(0.1, float(timeout))
            ),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "workspace_response",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        # Reasoning models such as o3 and GPT-5.6 Sol do not accept arbitrary temperatures.
        if uses_hidden_reasoning(self.model):
            # Structured synthesis needs room for hidden reasoning before the
            # schema-constrained answer. A tiny llama.cpp output budget is not
            # a suitable total budget for a reasoning model.
            request["max_completion_tokens"] = max(1536, int(max_tokens) * 2)
            request["reasoning_effort"] = "low"
        else:
            request["temperature"] = temperature
        try:
            response = self._create(request)
        except Exception as error:
            if "reasoning_effort" not in request or not self._is_output_limit_error(error):
                raise
            request["max_completion_tokens"] = max(
                4096, int(request["max_completion_tokens"]) * 2
            )
            response = self._create(request)
            self._record_usage(response, call_kind="structured", attempt=2)
            retried_after_error = True
        else:
            retried_after_error = False
        if not retried_after_error:
            self._record_usage(response, call_kind="structured", attempt=1)
        message = response.choices[0].message
        content = message.content or ""
        finish_reason = getattr(response.choices[0], "finish_reason", "unknown")
        if (
            finish_reason == "length"
            and "reasoning_effort" in request
            and not retried_after_error
        ):
            request["max_completion_tokens"] = max(
                4096, int(request["max_completion_tokens"]) * 2
            )
            response = self._create(request)
            self._record_usage(response, call_kind="structured", attempt=2)
            message = response.choices[0].message
            content = message.content or ""
            finish_reason = getattr(response.choices[0], "finish_reason", "unknown")
        if not content.strip():
            # Some reasoning-model completions exhaust their budget without
            # emitting usable structured text. Return the empty payload so the
            # caller can treat it as a malformed response and recover softly.
            return {"choices": [{"text": content}]}
        return {"choices": [{"text": content}]}

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Fallback compatibility for callers that do not provide a schema."""
        requested_tokens = int(kwargs.get("max_tokens", 128))
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": requested_tokens,
            "timeout": max(
                0.1, float(kwargs.get("timeout", self.timeout))
            ),
        }
        if uses_hidden_reasoning(self.model):
            # This limit includes invisible reasoning tokens. The legacy agents'
            # 180-token llama.cpp allowance otherwise leaves o3 no room to answer.
            request["max_completion_tokens"] = max(2048, requested_tokens)
            request["reasoning_effort"] = str(kwargs.get("reasoning_effort", "low"))
        else:
            request["temperature"] = float(kwargs.get("temperature", 0.5))
        response = self._create(request)
        self._record_usage(response, call_kind="text", attempt=1)
        content = response.choices[0].message.content or ""
        finish_reason = getattr(response.choices[0], "finish_reason", "unknown")
        retry_on_empty = bool(kwargs.get("retry_on_empty", True))
        if (
            retry_on_empty
            and not content.strip()
            and finish_reason == "length"
            and "reasoning_effort" in request
        ):
            request["max_completion_tokens"] = max(4096, int(request["max_completion_tokens"]) * 2)
            response = self._create(request)
            self._record_usage(response, call_kind="text", attempt=2)
            content = response.choices[0].message.content or ""
            finish_reason = getattr(response.choices[0], "finish_reason", "unknown")
        if not content.strip():
            # Preserve the empty text response so higher layers can decide
            # whether to retry, repair, or downgrade the candidate.
            return {"choices": [{"text": content}]}
        return {"choices": [{"text": content}]}
