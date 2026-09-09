from __future__ import annotations

import time
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Iterator


_CURRENT_RECORDER: ContextVar["PerformanceRecorder | None"] = ContextVar(
    "workspace_performance_recorder", default=None
)
_CURRENT_STAGE: ContextVar[str] = ContextVar(
    "workspace_performance_stage", default=""
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PerformanceRecorder:
    """Run-local timings and call counts, deliberately excluding model content."""

    run_id: str
    started_at: str = field(default_factory=_utc_now)
    started_monotonic: float = field(default_factory=time.monotonic)
    events: list[dict[str, Any]] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(
        self,
        name: str,
        started: float,
        *,
        category: str = "stage",
        status: str = "OK",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.record_duration(
            name,
            max(0.0, time.monotonic() - started),
            category=category,
            status=status,
            metadata=metadata,
        )

    def record_duration(
        self,
        name: str,
        duration_seconds: float,
        *,
        category: str = "stage",
        status: str = "OK",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an already-measured duration, including child-process phases."""
        event = {
            "category": str(category),
            "name": str(name),
            "status": str(status).upper(),
            "duration_seconds": round(max(0.0, float(duration_seconds)), 6),
            "finished_at": _utc_now(),
            "metadata": dict(metadata or {}),
        }
        with self._lock:
            self.events.append(event)

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        category: str = "stage",
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        started = time.monotonic()
        status = "OK"
        try:
            yield
        except Exception:
            status = "ERROR"
            raise
        finally:
            self.record(
                name, started, category=category, status=status, metadata=metadata,
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            events = [dict(event) for event in self.events]
        categories = Counter(event["category"] for event in events)
        statuses = Counter(event["status"] for event in events)
        logical = [event for event in events if event["category"] == "model_call"]
        provider = [event for event in events if event["category"] == "provider_attempt"]
        original_model_calls = sum(
            int(event["metadata"].get("model_call_count", 0) or 0)
            for event in events
            if event["name"] == "original_agent_model"
        )
        compact_primary = [
            event for event in logical if event["name"] == "compact_primary"
        ]
        compact_repairs = [
            event for event in logical if event["name"] == "compact_repair"
        ]
        grounded_projections = [
            event for event in events
            if event["name"] == "grounded_effect_projection"
        ]
        semantic_repairs = [
            event for event in logical
            if "repair" in event["name"].casefold()
            and event["status"] in {"OK", "ERROR"}
        ]
        provider_timeout_count = sum(
            event["status"] == "ERROR"
            and str(event["metadata"].get("error_category", "")).casefold()
            == "timeout"
            for event in provider
        )
        non_provider_logical_timeout_count = sum(
            event["status"] == "ERROR"
            and str(event["metadata"].get("error_category", "")).casefold()
            == "timeout"
            and str(event["metadata"].get("backend", ""))
            != "OpenAIWorkspaceLLM"
            for event in logical
        )
        original_timeout_count = sum(
            int(event["metadata"].get("timeout_count", 0) or 0)
            for event in events
            if event["name"] == "original_agent_process"
        )
        durations: dict[str, float] = {}
        for event in events:
            durations[event["name"]] = durations.get(event["name"], 0.0) + float(
                event["duration_seconds"]
            )
        original_agents: dict[str, dict[str, float | int]] = {}
        for event in events:
            if event["name"] not in {
                "original_agent_process", "original_agent_startup",
                "original_agent_retrieval", "original_agent_model",
                "original_agent_overhead",
            }:
                continue
            agent = str(event["metadata"].get("agent", "unknown"))
            row = original_agents.setdefault(agent, {})
            phase = event["name"].removeprefix("original_agent_")
            row[f"{phase}_seconds"] = round(
                float(row.get(f"{phase}_seconds", 0.0))
                + float(event["duration_seconds"]),
                6,
            )
            if event["name"] == "original_agent_model":
                row["model_calls"] = int(row.get("model_calls", 0)) + int(
                    event["metadata"].get("model_call_count", 0) or 0
                )
            if event["name"] == "original_agent_retrieval":
                row["retrieval_calls"] = int(row.get("retrieval_calls", 0)) + int(
                    event["metadata"].get("retrieval_call_count", 0) or 0
                )
        compact_specialists: dict[str, dict[str, float | int]] = {}
        for event in (*compact_primary, *compact_repairs):
            specialist = str(event["metadata"].get("specialist", "unknown"))
            row = compact_specialists.setdefault(specialist, {
                "primary_calls": 0,
                "primary_seconds": 0.0,
                "repair_calls": 0,
                "repair_seconds": 0.0,
            })
            phase = "repair" if event["name"] == "compact_repair" else "primary"
            row[f"{phase}_calls"] = int(row[f"{phase}_calls"]) + int(
                event["status"] in {"OK", "ERROR"}
            )
            row[f"{phase}_seconds"] = round(
                float(row[f"{phase}_seconds"]) + float(event["duration_seconds"]),
                6,
            )
        ledger_commits: dict[str, dict[str, float | int]] = {}
        for event in events:
            if event["name"] != "sequential_ledger_commit":
                continue
            specialist = str(event["metadata"].get("specialist", "unknown"))
            row = ledger_commits.setdefault(
                specialist, {"commit_batches": 0, "commit_seconds": 0.0},
            )
            row["commit_batches"] = int(row["commit_batches"]) + 1
            row["commit_seconds"] = round(
                float(row["commit_seconds"]) + float(event["duration_seconds"]),
                6,
            )
        return {
            "schema_version": 2,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "elapsed_seconds": round(
                max(0.0, time.monotonic() - self.started_monotonic), 6
            ),
            "counts": {
                "events": len(events),
                "by_category": dict(sorted(categories.items())),
                "by_status": dict(sorted(statuses.items())),
                "logical_model_calls": sum(
                    event["status"] in {"OK", "ERROR"} for event in logical
                ),
                "model_call_count": (
                    sum(event["status"] in {"OK", "ERROR"} for event in logical)
                    + original_model_calls
                ),
                "original_agent_model_calls": original_model_calls,
                "compact_primary_calls": sum(
                    event["status"] in {"OK", "ERROR"} for event in compact_primary
                ),
                "compact_repair_calls": sum(
                    event["status"] in {"OK", "ERROR"} for event in compact_repairs
                ),
                "repair_count": len(semantic_repairs),
                "timeout_count": (
                    provider_timeout_count
                    + non_provider_logical_timeout_count
                    + original_timeout_count
                ),
                "abstention_count": sum(
                    int(event["metadata"].get("abstention_count", 0) or 0)
                    for event in events
                    if event["name"] == "cycle_execution"
                ),
                "budget_blocked_calls": sum(
                    event["status"] == "BUDGET_BLOCKED" for event in logical
                ),
                "model_cache_hits": sum(
                    event["status"] == "CACHE_HIT" for event in logical
                ),
                "grounded_projection_materializations": sum(
                    event["status"] == "MATERIALIZED"
                    for event in grounded_projections
                ),
                "grounded_projection_cache_hits": sum(
                    event["status"] == "CACHE_HIT"
                    for event in grounded_projections
                ),
                "provider_attempts": len(provider),
                "provider_failures": sum(
                    event["status"] != "OK" for event in provider
                ),
            },
            "timings_by_name_seconds": {
                name: round(value, 6) for name, value in sorted(durations.items())
            },
            "original_agents": dict(sorted(original_agents.items())),
            "compact_specialists": dict(sorted(compact_specialists.items())),
            "ledger_commits": dict(sorted(ledger_commits.items())),
            "events": events,
            "content_recording": "DISABLED",
        }


def start_performance_trace(run_id: str) -> tuple[PerformanceRecorder, Token]:
    recorder = PerformanceRecorder(run_id=run_id)
    return recorder, _CURRENT_RECORDER.set(recorder)


def reset_performance_trace(token: Token) -> None:
    _CURRENT_RECORDER.reset(token)


def record_performance_event(
    name: str,
    started: float,
    *,
    category: str = "stage",
    status: str = "OK",
    metadata: dict[str, Any] | None = None,
) -> None:
    recorder = _CURRENT_RECORDER.get()
    if recorder is not None:
        enriched_metadata = dict(metadata or {})
        current_stage = _CURRENT_STAGE.get()
        if current_stage and "parent_stage" not in enriched_metadata:
            enriched_metadata["parent_stage"] = current_stage
        recorder.record(
            name,
            started,
            category=category,
            status=status,
            metadata=enriched_metadata,
        )


def record_performance_duration(
    name: str,
    duration_seconds: float,
    *,
    category: str = "stage",
    status: str = "OK",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record a duration measured outside this process without exposing content."""
    recorder = _CURRENT_RECORDER.get()
    if recorder is not None:
        enriched_metadata = dict(metadata or {})
        current_stage = _CURRENT_STAGE.get()
        if current_stage and "parent_stage" not in enriched_metadata:
            enriched_metadata["parent_stage"] = current_stage
        recorder.record_duration(
            name,
            duration_seconds,
            category=category,
            status=status,
            metadata=enriched_metadata,
        )


@contextmanager
def performance_stage(
    name: str,
    *,
    category: str = "stage",
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    recorder = _CURRENT_RECORDER.get()
    if recorder is None:
        yield
        return
    token = _CURRENT_STAGE.set(name)
    try:
        with recorder.stage(name, category=category, metadata=metadata):
            yield
    finally:
        _CURRENT_STAGE.reset(token)
