from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class WorkspaceBroadcast:
    constraint: str = "OPEN_DELIBERATION"
    intent: str = "identify_a_defensible_action"
    urgency: float = 0.5
    danger_probability: float = 0.5
    unresolved: str = "NONE"

    def __post_init__(self) -> None:
        self.constraint = self.constraint.strip().upper()[:48] or "OPEN_DELIBERATION"
        self.intent = self.intent.strip().lower()[:80] or "identify_a_defensible_action"
        self.unresolved = self.unresolved.strip().upper()[:48] or "NONE"
        self.urgency = clamp(self.urgency)
        self.danger_probability = clamp(self.danger_probability)

    def compact(self) -> str:
        return (
            f"constraint={self.constraint}; intent={self.intent}; "
            f"urgency={self.urgency:.2f}; danger={self.danger_probability:.2f}; "
            f"unresolved={self.unresolved}"
        )


@dataclass(slots=True)
class CandidateChunk:
    specialist: str
    constraint: str
    action_scores: dict[str, float]
    surprise: float
    friction: float
    confidence: float
    unresolved: str = "NONE"
    rationale: str = ""
    salience: float = 0.0
    schema_valid: bool = True
    validation_errors: list[str] = field(default_factory=list)
    recommended_action: str = ""
    baseline_action: str = ""
    testimony_alignment: str = "UNCLEAR"

    def __post_init__(self) -> None:
        self.specialist = self.specialist.strip()[:32]
        self.constraint = self.constraint.strip().upper()[:48] or "UNSPECIFIED"
        self.unresolved = self.unresolved.strip().upper()[:48] or "NONE"
        self.rationale = " ".join(self.rationale.split())[:180]
        self.surprise = clamp(self.surprise)
        self.friction = clamp(self.friction)
        self.confidence = clamp(self.confidence)
        self.action_scores = {str(k): clamp(v) for k, v in self.action_scores.items()}


@dataclass(slots=True)
class CycleRecord:
    cycle: int
    broadcast: WorkspaceBroadcast
    candidates: list[CandidateChunk]
    winner: CandidateChunk
    dissent: CandidateChunk | None
    policy: dict[str, float]
    entropy: float
    stable_cycles: int
    elapsed_seconds: float


@dataclass(slots=True)
class WorkspaceResult:
    scenario: str
    actions: list[str]
    source_testimonies: dict[str, str] = field(default_factory=dict)
    source_errors: dict[str, str] = field(default_factory=dict)
    source_baselines: dict[str, dict[str, str]] = field(default_factory=dict)
    scenario_facts: dict[str, Any] = field(default_factory=dict)
    cycles: list[CycleRecord] = field(default_factory=list)
    selected_action: str = ""
    confidence: float = 0.0
    halted_by: str = ""
    moral_residue: list[str] = field(default_factory=list)
    compressed_rule: str = ""
    reopen_conditions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
