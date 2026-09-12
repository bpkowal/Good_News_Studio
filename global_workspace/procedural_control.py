"""Deterministic scheduling for recurrent workspace operations.

The controller is deliberately non-normative.  It does not rank actions, alter
salience, or interpret ethical claims.  It only remembers admitted procedural
work, records whether that work received a substantive response, and chooses
the next eligible operation by a stable priority rule.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence


BASE_DELIBERATION = "BASE_DELIBERATION"
ANSWER_ARGUMENT_CHALLENGE = "ANSWER_ARGUMENT_CHALLENGE"
CHECK_PROPOSAL_FEASIBILITY = "CHECK_PROPOSAL_FEASIBILITY"
AUDIT_VISIBILITY = "AUDIT_VISIBILITY"
AUDIT_AUTONOMY = "AUDIT_AUTONOMY"
AUDIT_CONSENSUS = "AUDIT_CONSENSUS"
AUDIT_REVERSAL = "AUDIT_REVERSAL"
TEST_CONTINGENCY = "TEST_CONTINGENCY"
TEST_REFORMULATION = "TEST_REFORMULATION"
TEST_PLANNING = "TEST_PLANNING"

ACTIVE_TASK_STATUSES = {"PENDING", "ASSIGNED", "UNANSWERED", "UNRESOLVED"}
TERMINAL_TASK_STATUSES = {"RESOLVED", "COMPLETED", "REJECTED", "WITHHELD"}

_CONSTRAINT_OPERATIONS = {
    "VISIBILITY_AUDIT": AUDIT_VISIBILITY,
    "AUTONOMY_AUDIT": AUDIT_AUTONOMY,
    "COERCION_AUDIT": AUDIT_AUTONOMY,
    "CONSENSUS_AUDIT": AUDIT_CONSENSUS,
    "PROBLEM_STATE_AUDIT": AUDIT_CONSENSUS,
    "REVERSAL_AUDIT": AUDIT_REVERSAL,
    "CONTINGENCY_REVIEW": TEST_CONTINGENCY,
    "PROBLEM_REFORMULATION": TEST_REFORMULATION,
    "PLANNING_REVIEW": TEST_PLANNING,
    "PROPOSAL_REVIEW": CHECK_PROPOSAL_FEASIBILITY,
}


@dataclass(slots=True)
class ProcedureTask:
    task_id: str
    operation: str
    source_kind: str
    source_id: str
    question: str
    priority: float = 0.5
    target_specialists: tuple[str, ...] = ()
    grounded_in: tuple[str, ...] = ()
    grounding_status: str = "UNGROUNDED"
    status: str = "PENDING"
    first_seen_cycle: int = 1
    assigned_cycle: int = 0
    attempts: int = 0
    last_attempt_cycle: int = 0
    completion_reason: str = ""
    blocking: bool = True
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProcedureDecision:
    cycle: int
    operation: str
    task_ids: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProceduralController:
    """Persistent, deterministic agenda and completion controller."""

    def __init__(self) -> None:
        self._tasks: dict[str, ProcedureTask] = {}
        self._decisions: list[ProcedureDecision] = []

    @staticmethod
    def operation_for_constraint(constraint: str) -> str:
        return _CONSTRAINT_OPERATIONS.get(
            str(constraint or "").strip().upper(), BASE_DELIBERATION,
        )

    def register_challenges(
        self,
        challenges: Sequence[dict[str, Any]],
        *,
        cycle: int,
    ) -> None:
        """Merge challenge records without resetting their history."""
        for raw in challenges:
            item = copy.deepcopy(dict(raw))
            issue_id = str(item.get("issue_id", "")).strip()
            if not issue_id.startswith("CHALLENGE:"):
                continue
            incoming_status = str(item.get("status", "UNTESTED")).upper()
            status = {
                "UNTESTED": "PENDING",
                "ASSIGNED": "ASSIGNED",
                "UNANSWERED": "UNANSWERED",
                "UNRESOLVED": "UNRESOLVED",
                "REFINED": "UNRESOLVED",
                "RESOLVED": "RESOLVED",
            }.get(incoming_status, "PENDING")
            existing = self._tasks.get(issue_id)
            if existing is None:
                first_seen = int(item.get("first_seen_cycle", cycle) or cycle)
                self._tasks[issue_id] = ProcedureTask(
                    task_id=issue_id,
                    operation=ANSWER_ARGUMENT_CHALLENGE,
                    source_kind="ARGUMENT_CHALLENGE",
                    source_id=issue_id,
                    question=" ".join(str(item.get("question", "")).split())[:500],
                    priority=max(0.0, min(1.0, float(item.get("priority", 0.5)))),
                    target_specialists=tuple(
                        str(value).casefold()
                        for value in item.get("target_specialists", []) or []
                        if str(value).strip()
                    ),
                    grounded_in=tuple(
                        str(value) for value in item.get("grounded_in", []) or []
                        if str(value).strip()
                    ),
                    grounding_status=str(
                        item.get("grounding_status", "UNGROUNDED")
                    ).upper(),
                    status=status,
                    first_seen_cycle=max(1, first_seen),
                    assigned_cycle=int(item.get("assigned_cycle", 0) or 0),
                    blocking=(
                        str(item.get("grounding_status", "UNGROUNDED")).upper()
                        not in {"UNGROUNDED", "REJECTED"}
                    ),
                    payload=item,
                )
                continue

            existing.priority = max(
                existing.priority,
                max(0.0, min(1.0, float(item.get("priority", 0.5)))),
            )
            existing.question = (
                " ".join(str(item.get("question", existing.question)).split())[:500]
                or existing.question
            )
            existing.target_specialists = tuple(dict.fromkeys((
                *existing.target_specialists,
                *(
                    str(value).casefold()
                    for value in item.get("target_specialists", []) or []
                    if str(value).strip()
                ),
            )))
            existing.grounded_in = tuple(dict.fromkeys((
                *existing.grounded_in,
                *(
                    str(value) for value in item.get("grounded_in", []) or []
                    if str(value).strip()
                ),
            )))
            incoming_grounding = str(
                item.get("grounding_status", existing.grounding_status)
            ).upper()
            if incoming_grounding not in {"UNGROUNDED", "REJECTED"}:
                existing.grounding_status = incoming_grounding
                existing.blocking = True
            if status in TERMINAL_TASK_STATUSES:
                existing.status = status
                existing.completion_reason = "challenge resolution verified"
            elif existing.status not in TERMINAL_TASK_STATUSES:
                existing.status = status
            # The newest representation is retained while controller-owned
            # identity, first-seen cycle, and attempt history remain immutable.
            existing.payload = item

    def challenge_history(self) -> list[dict[str, Any]]:
        """Return challenges in the format consumed by the agenda adapter."""
        records: list[dict[str, Any]] = []
        for task in self._ordered_tasks(operation=ANSWER_ARGUMENT_CHALLENGE):
            item = copy.deepcopy(task.payload)
            item["issue_id"] = task.source_id
            item["first_seen_cycle"] = task.first_seen_cycle
            item["priority"] = task.priority
            item["status"] = {
                "PENDING": "UNTESTED",
                "ASSIGNED": "ASSIGNED",
                "UNANSWERED": "UNANSWERED",
                "UNRESOLVED": "UNRESOLVED",
                "RESOLVED": "RESOLVED",
            }.get(task.status, task.status)
            if task.assigned_cycle:
                item["assigned_cycle"] = task.assigned_cycle
            records.append(item)
        return records

    def register_proposal(self, proposal: Any, *, cycle: int) -> str:
        proposal_id = str(getattr(proposal, "proposal_id", "")).strip().upper()
        task_id = f"PROCEDURE:PROPOSAL:{proposal_id}"
        if not proposal_id:
            return ""
        accepted = bool(getattr(proposal, "accepted", False))
        promotion = str(getattr(proposal, "promotion_status", "")).upper()
        status = (
            "REJECTED" if not accepted or promotion == "REJECTED"
            else "COMPLETED" if promotion == "ADMISSIBLE"
            else "PENDING"
        )
        existing = self._tasks.get(task_id)
        if existing is None:
            self._tasks[task_id] = ProcedureTask(
                task_id=task_id,
                operation=CHECK_PROPOSAL_FEASIBILITY,
                source_kind="SYNTHESIS_PROPOSAL",
                source_id=proposal_id,
                question=" ".join(str(getattr(proposal, "action", "")).split())[:500],
                priority=0.70,
                grounding_status=str(
                    getattr(proposal, "grounding_status", "PENDING")
                ).upper(),
                status=status,
                first_seen_cycle=max(1, cycle),
                blocking=accepted and promotion in {"UNDER_REVIEW", "PROPOSED"},
                payload={"proposal_id": proposal_id},
            )
        else:
            existing.status = status
            existing.blocking = status == "PENDING"
        return task_id

    def record_proposal_review(
        self,
        proposal_id: str,
        *,
        cycle: int,
        substantive_complete: bool,
        reason: str,
    ) -> None:
        task = self._tasks.get(
            f"PROCEDURE:PROPOSAL:{str(proposal_id).strip().upper()}"
        )
        if task is None:
            return
        if task.last_attempt_cycle != max(1, cycle):
            task.attempts += 1
            task.last_attempt_cycle = max(1, cycle)
        if substantive_complete:
            task.status = "COMPLETED"
            task.blocking = False
            task.completion_reason = "substantive framework review completed"
        else:
            task.status = "UNRESOLVED"
            task.blocking = True
            task.completion_reason = " ".join(str(reason).split())[:240]

    def record_cycle_start(
        self,
        *,
        cycle: int,
        operation: str,
        task_ids: Sequence[str] = (),
    ) -> None:
        for task_id in task_ids:
            task = self._tasks.get(str(task_id))
            if task is None:
                continue
            task.attempts += 1
            task.last_attempt_cycle = max(1, cycle)
            task.assigned_cycle = max(1, cycle)
            if task.status not in TERMINAL_TASK_STATUSES:
                task.status = "ASSIGNED"

    def select_next(
        self,
        *,
        cycle: int,
        challenge_agenda: Sequence[dict[str, Any]] = (),
        proposal_id: str = "",
        fallback_constraint: str = "OPEN_DELIBERATION",
    ) -> ProcedureDecision:
        """Choose the next ordinary operation using stable priority ordering."""
        challenge_ids = tuple(
            str(item.get("issue_id", ""))
            for item in challenge_agenda
            if str(item.get("issue_id", "")) in self._tasks
        )
        if challenge_ids:
            decision = ProcedureDecision(
                cycle=max(1, cycle),
                operation=ANSWER_ARGUMENT_CHALLENGE,
                task_ids=challenge_ids,
                reason="highest-priority grounded argument challenges remain unanswered",
            )
        else:
            proposal_task_id = (
                f"PROCEDURE:PROPOSAL:{str(proposal_id).strip().upper()}"
                if proposal_id else ""
            )
            proposal_task = self._tasks.get(proposal_task_id)
            if (
                proposal_task is not None
                and proposal_task.status in ACTIVE_TASK_STATUSES
            ):
                decision = ProcedureDecision(
                    cycle=max(1, cycle),
                    operation=CHECK_PROPOSAL_FEASIBILITY,
                    task_ids=(proposal_task_id,),
                    reason="admitted synthesis proposal requires substantive review",
                )
            else:
                decision = ProcedureDecision(
                    cycle=max(1, cycle),
                    operation=self.operation_for_constraint(fallback_constraint),
                    task_ids=(),
                    reason="no higher-priority persistent procedural task is eligible",
                )
        self._remember_decision(decision)
        return decision

    def record_forced_operation(
        self,
        *,
        cycle: int,
        constraint: str,
        task_ids: Sequence[str] = (),
        reason: str = "",
    ) -> ProcedureDecision:
        """Record an existing hard-audit operation without reprioritizing it."""
        decision = ProcedureDecision(
            cycle=max(1, cycle),
            operation=self.operation_for_constraint(constraint),
            task_ids=tuple(str(value) for value in task_ids if str(value)),
            reason=(
                " ".join(reason.split())[:240]
                or f"existing {str(constraint).upper()} trigger has procedural priority"
            ),
        )
        self._remember_decision(decision)
        return decision

    def pending_blockers(self) -> list[dict[str, Any]]:
        return [
            task.to_dict()
            for task in self._ordered_tasks()
            if task.blocking and task.status in ACTIVE_TASK_STATUSES
        ]

    def may_converge(self) -> bool:
        return not self.pending_blockers()

    @staticmethod
    def contributes_to_policy_stability(operation: str) -> bool:
        return str(operation).upper() in {
            BASE_DELIBERATION, ANSWER_ARGUMENT_CHALLENGE,
        }

    def snapshot(self) -> dict[str, Any]:
        blockers = self.pending_blockers()
        return {
            "controller_kind": "DETERMINISTIC_PROCEDURAL_CONTROLLER_V1",
            "non_normative": True,
            "tasks": [task.to_dict() for task in self._ordered_tasks()],
            "decisions": [decision.to_dict() for decision in self._decisions],
            "convergence_permitted": not blockers,
            "convergence_blockers": blockers,
        }

    def _ordered_tasks(self, *, operation: str = "") -> list[ProcedureTask]:
        tasks = [
            task for task in self._tasks.values()
            if not operation or task.operation == operation
        ]
        return sorted(
            tasks,
            key=lambda task: (
                task.status in TERMINAL_TASK_STATUSES,
                -task.priority,
                task.first_seen_cycle,
                task.task_id,
            ),
        )

    def _remember_decision(self, decision: ProcedureDecision) -> None:
        # Scheduling may first produce an ordinary cycle and then admit a
        # protected audit or proposal in the same deterministic commit phase.
        # Keep only the final decision for a cycle.
        self._decisions = [
            prior for prior in self._decisions if prior.cycle != decision.cycle
        ]
        self._decisions.append(decision)
