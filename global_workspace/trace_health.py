from __future__ import annotations

from dataclasses import dataclass

from .models import WorkspaceResult


@dataclass(frozen=True, slots=True)
class TraceHealthFinding:
    severity: str
    code: str
    cycle: int
    detail: str


def _hypothetical_context_errors(received) -> list[str]:
    """Validate context according to the hypothetical operation's own type."""
    if received.branch_kind == "PLANNING_CONTINGENCY":
        return [
            name for name, value in (
                ("origin", received.branch_origin_action),
                ("condition", received.branch_condition),
                ("fallback", received.branch_fallback),
            ) if not value
        ]
    if received.constraint == "REVERSAL_AUDIT":
        return [] if received.reversal_challenge else ["reversal challenge"]
    if received.constraint == "PROBLEM_REFORMULATION":
        return [] if received.reformulation_context else ["reformulation context"]
    if received.constraint == "CONTINGENCY_REVIEW":
        missing = []
        if not received.contingency_question:
            missing.append("contingency question")
        if not received.contingency_failure_condition:
            missing.append("failure condition")
        if not received.contingency_predicate:
            missing.append("typed contingency predicate")
        if len(received.contingency_fallback_actions) != 2:
            missing.append("typed fallback actions")
        return missing
    return ["typed hypothetical context"]


def _framework_retention_finding(candidate) -> tuple[str, str] | None:
    """Split structural map incompleteness from substantive framework loss."""
    if candidate.framework_retention_status != "LOST":
        return None
    validation_errors = " | ".join(candidate.framework_validation_errors)
    if candidate.specialist in {"virtue", "care"} and (
        "framework-specific grounds" in validation_errors
        or "conflicts with its framework map" in validation_errors
    ):
        return "WARNING", "FRAMEWORK_MAP_INCOMPLETE"
    return "WARNING", "FRAMEWORK_RETENTION_LOST"


def audit_trace_health(result: WorkspaceResult) -> list[TraceHealthFinding]:
    """Detect representation failures without judging the ethical conclusion."""
    findings: list[TraceHealthFinding] = []
    for cycle in result.cycles:
        received = cycle.received_broadcast or cycle.broadcast
        for candidate in cycle.candidates:
            if candidate.delegate_status == "MODEL_ERROR":
                findings.append(TraceHealthFinding(
                    "WARNING", "DELEGATE_MODEL_UNAVAILABLE", cycle.cycle,
                    f"{candidate.specialist} provider call failed; response excluded from policy",
                ))
            if candidate.schema_valid and candidate.action_scores:
                maximum = max(candidate.action_scores.values())
                if candidate.action_scores.get(candidate.recommended_action) != maximum:
                    findings.append(TraceHealthFinding(
                        "ERROR", "RECOMMENDATION_SCORE_MISMATCH", cycle.cycle,
                        f"{candidate.specialist} recommendation is not its highest score",
                    ))
            if received.constraint == "VISIBILITY_AUDIT" and candidate.schema_valid:
                if candidate.visibility_response == "NOT_TESTED":
                    findings.append(TraceHealthFinding(
                        "ERROR", "VISIBILITY_RESPONSE_MISSING", cycle.cycle,
                        f"{candidate.specialist} did not evaluate the admitted proposition",
                    ))
                if candidate.visibility_magnitude_overreach:
                    findings.append(TraceHealthFinding(
                        "WARNING", "VISIBILITY_MAGNITUDE_OVERREACH", cycle.cycle,
                        f"{candidate.specialist} supplied an unsupported bound on hidden harm; "
                        "direction was retained and magnitude reset to unknown",
                    ))
            if (
                received.constraint == "REVERSAL_AUDIT"
                and candidate.schema_valid
                and not candidate.reversal_review_valid
            ):
                findings.append(TraceHealthFinding(
                    "WARNING", "REVERSAL_REVIEW_INCOMPLETE", cycle.cycle,
                    f"{candidate.specialist}: {candidate.reversal_review_error}; "
                    "ethical evaluation retained but audit response excluded",
                ))
            if received.constraint == "CONTINGENCY_REVIEW" and candidate.schema_valid:
                if not candidate.contingency_choice:
                    findings.append(TraceHealthFinding(
                        "ERROR", "CONTINGENCY_RESPONSE_MISSING", cycle.cycle,
                        f"{candidate.specialist} did not select a typed fallback action",
                    ))
            if (
                candidate.schema_valid
                and received.constraint != "OPEN_DELIBERATION"
                and candidate.framework_retention_status == "UPDATE_REJECTED"
            ):
                transition_error = next(
                    (
                        error for error in candidate.framework_validation_errors
                        if "principle state changed" in error
                    ),
                    "unexplained typed framework update",
                )
                findings.append(TraceHealthFinding(
                    "HELD", "FRAMEWORK_UPDATE_REJECTED", cycle.cycle,
                    f"{candidate.specialist}: {transition_error}; authoritative "
                    "prior framework state remained operative",
                ))
            if candidate.schema_valid and received.constraint != "OPEN_DELIBERATION":
                retention_finding = _framework_retention_finding(candidate)
                if retention_finding is not None:
                    severity, code = retention_finding
                    loss_reason = (
                        "; ".join(candidate.framework_validation_errors[:2])
                        or "delegate reported that its prior framework constraint was not retained"
                    )
                    detail = (
                        f"{candidate.specialist} did not demonstrate retained framework "
                        f"constraints: {loss_reason}"
                    )
                    if code == "FRAMEWORK_MAP_INCOMPLETE":
                        detail = (
                            f"{candidate.specialist} framework structure was incomplete: "
                            f"{loss_reason}"
                        )
                    findings.append(TraceHealthFinding(
                        severity, code, cycle.cycle, detail,
                    ))
            if (
                candidate.schema_valid
                and received.constraint != "OPEN_DELIBERATION"
                and candidate.framework_retention_status == "COMMITTED_WITH_UNCERTAINTY"
            ):
                findings.append(TraceHealthFinding(
                    "WARNING", "FRAMEWORK_LEDGER_UNCERTAIN", cycle.cycle,
                    f"{candidate.specialist} framework update committed only after "
                    "its unsupported direction was replaced with uncertainty: "
                    + (
                        "; ".join(candidate.framework_validation_errors[:2])
                        or "typed direction lacked graph support"
                    ),
                ))
            if (
                candidate.schema_valid
                # Completeness is cycle-relative. Synthesis may add an action in
                # a later cycle; early delegates could not classify an option
                # that did not yet exist. action_scores is the authoritative
                # snapshot of the shared action set received by this candidate.
                and set(candidate.action_admissibility) != set(candidate.action_scores)
            ):
                findings.append(TraceHealthFinding(
                    "WARNING", "STANCE_MEASUREMENT_INCOMPLETE", cycle.cycle,
                    f"{candidate.specialist} did not classify every action's admissibility",
                ))
        if cycle.is_hypothetical:
            missing = _hypothetical_context_errors(received)
            if missing:
                code = (
                    "INCOMPLETE_PLANNING_BRANCH"
                    if received.branch_kind == "PLANNING_CONTINGENCY"
                    else "HYPOTHETICAL_CONTEXT_UNLABELED"
                )
                findings.append(TraceHealthFinding(
                    "ERROR", code, cycle.cycle,
                    "missing " + ", ".join(missing),
                ))
    for record in result.semantic_invariants:
        if not record.valid:
            findings.append(TraceHealthFinding(
                "HELD", "SEMANTIC_TRANSFORMATION_WITHHELD", 0,
                f"{record.boundary}: {'; '.join(record.errors)}",
            ))
    activated_visibility = any(
        assessment.valid and assessment.activated
        for assessment in result.visibility_assessments
    )
    visibility_reviewed = any(
        (cycle.received_broadcast or cycle.broadcast).constraint == "VISIBILITY_AUDIT"
        for cycle in result.cycles
    )
    if activated_visibility and not visibility_reviewed:
        findings.append(TraceHealthFinding(
            "WARNING", "VISIBILITY_NOT_DELIBERATED", 0,
            "activated visibility audit did not receive a recurrent response cycle",
        ))
    return findings
