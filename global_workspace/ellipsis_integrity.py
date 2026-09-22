"""Conservative surface detection and audit contracts for English ellipsis.

Detection is intentionally separate from reconstruction.  A detected missing
predicate or shared argument becomes an obligation; generators must either
record a source-licensed resolution or preserve an open question as unresolved.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


ELLIPSIS_TEMPLATES: dict[str, dict[str, Any]] = {
    "GAPPING": {"missing": "PREDICATE", "policy": "RESOLVE_REQUIRED", "copy": "parallel_conjunct_predicate"},
    "STRIPPING": {"missing": "CLAUSE", "policy": "RESOLVE_REQUIRED", "copy": "antecedent_clause_except_remnant"},
    "VP_ELLIPSIS": {"missing": "VERB_PHRASE", "policy": "RESOLVE_REQUIRED", "copy": "auxiliary_selected_vp"},
    "PSEUDOGAPPING": {"missing": "VERB_PHRASE", "policy": "RESOLVE_REQUIRED", "copy": "vp_preserving_remnant"},
    "ANSWER_ELLIPSIS": {"missing": "QUESTION_BACKGROUND", "policy": "RESOLVE_REQUIRED", "copy": "question_frame"},
    "SLUICING": {"missing": "OPEN_QUESTION", "policy": "PRESERVE_UNRESOLVED", "copy": "none_without_answer"},
    "NOMINAL_ELLIPSIS": {"missing": "NOMINAL_HEAD", "policy": "RESOLVE_REQUIRED", "copy": "compatible_nominal_head"},
    "COMPARATIVE_DELETION": {"missing": "COMPARISON_PREDICATE", "policy": "RESOLVE_REQUIRED", "copy": "comparison_dimension"},
    "NULL_COMPLEMENT_ANAPHORA": {"missing": "COMPLEMENT", "policy": "RESOLVE_OR_REJECT", "copy": "unique_discourse_complement"},
    # Coordination is not always classified as ellipsis in linguistic
    # taxonomies, but explicit templates prevent shared arguments from being
    # lost when building an atomic world graph.
    "SHARED_OBJECT_CONJUNCTION": {"missing": "SHARED_OBJECT", "policy": "RESOLVE_REQUIRED", "copy": "right_edge_shared_argument"},
    "SHARED_SUBJECT_CONJUNCTION": {"missing": "SHARED_SUBJECT", "policy": "RESOLVE_REQUIRED", "copy": "left_edge_shared_argument"},
}

ETHICAL_CONTEXT_ROLES = (
    "DECISION_MAKER", "ALTERNATIVE_ACTION", "AFFECTED_PARTY",
    "CONSEQUENCE", "CAUSAL_INTERMEDIATE", "RESOURCE_OR_CONSTRAINT",
    "UNCERTAINTY", "DECISION_QUESTION",
)


_VP = re.compile(
    r"\b(?:so\s+(?:does|did|do|will|would|is|was|were)|"
    r"(?:does|did|do|will|would|is|was|were)\s+too|"
    r"the\s+same\s+(?:thing|action|outcome|result)|likewise)\b",
    re.IGNORECASE,
)
_STRIPPING = re.compile(
    r"\b(?:and|but)\s+(?P<remnant>[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)?)\s+"
    r"(?:too|as\s+well|not)\b",
)
_SLUICING = re.compile(
    r"\b(?:but|although|though|and)\b[^?.;]{0,100}\b"
    r"(?P<wh>who|whom|whose|what|which|where|when|why|how)\b\s*[?.]?$",
    re.IGNORECASE,
)
_GAPPING = re.compile(
    r"^(?!(?:A|An|The|Both)\s)(?P<left_subject>[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)?)\s+"
    r"(?P<predicate>[a-z]+(?:ed|s|ing)?)\s+"
    r"(?P<left_remnant>[^,;]{1,70}?)\s*,?\s+and\s+"
    r"(?P<right_subject>[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)?)\s+"
    r"(?P<right_remnant>(?!too\b|as\s+well\b)[^,;?.]{1,70})[?.]?$",
)
_SHARED_OBJECT = re.compile(
    r"^(?!(?:A|An|The|Both)\s)(?P<subject>[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)?)\s+"
    r"(?P<left_predicate>[a-z]+(?:ed|s|ing)?)\s+and\s+"
    r"(?P<right_predicate>[a-z]+(?:ed|s|ing)?)\s+"
    r"(?P<shared_argument>[^,;?.]{2,80})[?.]?$",
)
_SHARED_SUBJECT = re.compile(
    r"^(?!(?:A|An|The|Both)\s)(?P<subject>[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)?)\s+"
    r"(?P<left_predicate>[a-z]+(?:ed|s|ing)?)\s+"
    r"(?P<left_argument>[^,;]{1,60}?)\s+and\s+"
    r"(?P<right_predicate>[a-z]+(?:ed|s|ing)?)\s+"
    r"(?P<right_argument>[^,;?.]{1,60})[?.]?$",
)
_PSEUDOGAPPING = re.compile(
    r"\b(?:but|and)\s+(?P<subject>[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)?|"
    r"he|she|they|it)\s+(?P<aux>won't|wouldn't|didn't|doesn't|can't|cannot|"
    r"will|would|did|does|can|could|has|have|had)\s+"
    r"(?P<remnant>(?!too\b|as\s+well\b)[A-Z]?[\w'-]+(?:\s+[\w'-]+){0,5})[?.]?$",
    re.IGNORECASE,
)
_COMPARATIVE = re.compile(
    r"\b(?P<degree>more|less|fewer|as)\b[^.;?]{1,100}\bthan\b"
    r"(?P<standard>[^.;?]{1,80})",
    re.IGNORECASE,
)
_NULL_COMPLEMENT = re.compile(
    r"\b(?P<trigger>refused|agreed|tried|decided|promised|offered|declined|"
    r"accepted|knew|knows|asked)\s*[.;?]?$",
    re.IGNORECASE,
)
_NOMINAL = re.compile(
    r"\b(?:one|two|three|four|five|several|many|some)\s+"
    r"(?P<head>[A-Za-z][\w'-]*(?:\s+[A-Za-z][\w'-]*){0,3})\b"
    r"(?P<middle>[^.;]{0,120}\b(?:and|but|before|while|because|if)\b[^.;]{0,80})"
    r"\b(?P<remnant>(?:the\s+)?(?:first|second|third|last|next)|one|two|three|"
    r"four|five|each|some|mine|ours|yours|his|hers|theirs|[A-Z][\w'-]*'s)"
    r"(?=\s*(?:too\b|now\b|would\b|will\b|is\b|are\b|was\b|were\b|has\b|"
    r"have\b|had\b|do\b|does\b|did\b|with\b|[.,;]))",
    re.IGNORECASE,
)


def detect_ellipsis_obligations(
    clauses: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return typed, auditable obligations without claiming a resolution."""
    obligations: list[dict[str, Any]] = []
    for clause in clauses:
        clause_id = str(clause.get("clause_id") or "")
        text = " ".join(str(clause.get("text") or "").split())
        matches: list[tuple[str, re.Match[str], str]] = []
        for kind, pattern, policy in (
            ("VP_ELLIPSIS", _VP, "RESOLVE_REQUIRED"),
            ("STRIPPING", _STRIPPING, "RESOLVE_REQUIRED"),
            ("SLUICING", _SLUICING, "PRESERVE_UNRESOLVED"),
            ("GAPPING", _GAPPING, "RESOLVE_REQUIRED"),
            ("SHARED_OBJECT_CONJUNCTION", _SHARED_OBJECT, "RESOLVE_REQUIRED"),
            ("SHARED_SUBJECT_CONJUNCTION", _SHARED_SUBJECT, "RESOLVE_REQUIRED"),
            ("PSEUDOGAPPING", _PSEUDOGAPPING, "RESOLVE_REQUIRED"),
            ("COMPARATIVE_DELETION", _COMPARATIVE, "RESOLVE_REQUIRED"),
            ("NULL_COMPLEMENT_ANAPHORA", _NULL_COMPLEMENT, "RESOLVE_OR_REJECT"),
            ("NOMINAL_ELLIPSIS", _NOMINAL, "RESOLVE_REQUIRED"),
        ):
            match = pattern.search(text)
            if match is not None:
                matches.append((kind, match, policy))
        # Prefer the more specific shared-subject analysis over the broad
        # gapping shape when both heuristics happen to match one clause.
        kinds = {kind for kind, _match, _policy in matches}
        if "SHARED_SUBJECT_CONJUNCTION" in kinds:
            matches = [row for row in matches if row[0] != "GAPPING"]
        if "VP_ELLIPSIS" in kinds:
            matches = [row for row in matches if row[0] != "GAPPING"]
        for kind, match, policy in matches:
            groups = {
                key: " ".join(value.split())
                for key, value in match.groupdict().items() if value
            }
            obligations.append({
                "obligation_id": f"EL{len(obligations)}",
                "clause_id": clause_id,
                "kind": kind,
                "policy": policy,
                "missing_constituent_type": ELLIPSIS_TEMPLATES[kind]["missing"],
                "resolution_template": ELLIPSIS_TEMPLATES[kind]["copy"],
                "surface_span": " ".join(match.group(0).strip().split()),
                "cues": [" ".join(match.group(0).strip().split())],
                "remnants": groups,
                "instruction": (
                    "Copy the missing predicate/shared argument only from the "
                    "same clause or a uniquely compatible antecedent and record "
                    "the reconstruction explicitly."
                    if policy == "RESOLVE_REQUIRED" else
                    "Keep the wh-remnant as an unresolved question; do not create "
                    "a factual party, action, or outcome that answers it."
                ),
            })
    # Answer fragments are discourse relations, not reliably detectable from
    # one clause in isolation. Pair a short declarative fragment with the
    # immediately preceding wh-question.
    for index in range(1, len(clauses)):
        previous = " ".join(str(clauses[index - 1].get("text") or "").split())
        current = " ".join(str(clauses[index].get("text") or "").split())
        if not re.search(r"\b(?:who|whom|whose|what|which|where|when|why|how)\b", previous, re.I):
            continue
        if not previous.endswith("?") or not 1 <= len(current.split()) <= 8:
            continue
        if re.search(r"\b(?:is|are|was|were|will|would|did|does|has|have|can|could)\b", current, re.I):
            continue
        obligations.append({
            "obligation_id": f"EL{len(obligations)}",
            "clause_id": str(clauses[index].get("clause_id") or ""),
            "context_clause_ids": [str(clauses[index - 1].get("clause_id") or "")],
            "kind": "ANSWER_ELLIPSIS",
            "policy": "RESOLVE_REQUIRED",
            "missing_constituent_type": "QUESTION_BACKGROUND",
            "resolution_template": "question_frame",
            "surface_span": current,
            "cues": [current],
            "remnants": {"answer_fragment": current, "question": previous},
            "instruction": (
                "Reconstruct the answer only from the immediately preceding "
                "question frame and preserve the answer fragment as the filled slot."
            ),
        })
    return obligations


def build_ethical_context_profile(
    clauses: Sequence[Mapping[str, Any]],
    actions: Mapping[str, str],
) -> dict[str, Any]:
    """Explicit task priors used to rank, never license, reconstructions."""
    question_ids = [
        str(row.get("clause_id") or "") for row in clauses
        if str(row.get("text") or "").strip().endswith("?")
    ]
    return {
        "task_type": "ETHICAL_DECISION_PROBLEM",
        "epistemic_status": "RANKING_PRIOR_NOT_PROVENANCE",
        "canonical_actions": dict(actions),
        "expected_roles": list(ETHICAL_CONTEXT_ROLES),
        "decision_question_clause_ids": question_ids,
        "ranking_rules": [
            "prefer antecedents that preserve canonical action ownership",
            "prefer antecedents that preserve party, polarity, modality, quantity, and time",
            "prefer parallel action/consequence structure when the source marks a contrast",
            "never add a party, outcome, probability, quantity, or moral conclusion absent from source",
        ],
    }


def contextualize_ellipsis_obligations(
    obligations: Sequence[Mapping[str, Any]],
    clauses: Sequence[Mapping[str, Any]],
    actions: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Attach candidate evidence and ethical roles without selecting a fact."""
    clause_by_id = {
        str(row.get("clause_id") or ""): " ".join(str(row.get("text") or "").split())
        for row in clauses
    }
    contextualized: list[dict[str, Any]] = []
    for obligation in obligations:
        row = dict(obligation)
        kind = str(row.get("kind") or "")
        clause_id = str(row.get("clause_id") or "")
        source = clause_by_id.get(clause_id, "")
        remnants = row.get("remnants") if isinstance(row.get("remnants"), dict) else {}
        antecedents: list[dict[str, Any]] = []
        predicate = str(remnants.get("predicate") or remnants.get("left_predicate") or "")
        if predicate and predicate.casefold() in source.casefold():
            antecedents.append({
                "span": predicate, "clause_id": clause_id,
                "basis": "PARALLEL_CONJUNCT", "source_licensed": True,
            })
        if kind == "ANSWER_ELLIPSIS":
            question = str(remnants.get("question") or "")
            context_ids = list(row.get("context_clause_ids") or [])
            if question and context_ids:
                antecedents.append({
                    "span": question, "clause_id": context_ids[0],
                    "basis": "QUESTION_FRAME", "source_licensed": True,
                })
        if not antecedents and source:
            antecedents.append({
                "span": source, "clause_id": clause_id,
                "basis": "CONTAINING_CLAUSE_REQUIRES_MODEL_RESOLUTION",
                "source_licensed": False,
            })
        roles = ["ALTERNATIVE_ACTION"] if kind in {
            "GAPPING", "STRIPPING", "VP_ELLIPSIS", "PSEUDOGAPPING",
            "SHARED_OBJECT_CONJUNCTION", "SHARED_SUBJECT_CONJUNCTION",
        } else ["CONSEQUENCE"]
        if kind in {"SLUICING", "ANSWER_ELLIPSIS"}:
            roles.append("DECISION_QUESTION")
        row["candidate_antecedents"] = antecedents
        row["ethical_context_roles"] = roles
        row["context_policy"] = "RANK_ONLY_SOURCE_MUST_LICENSE"
        contextualized.append(row)
    return contextualized


def validate_ellipsis_resolution_records(
    candidate: Any,
    obligations: Sequence[Mapping[str, Any]],
    clauses: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Reject missing, unsupported, or overconfident reconstruction records."""
    if not obligations:
        return []
    rows = candidate.get("ellipsis_resolutions") if isinstance(candidate, dict) else None
    rows = rows if isinstance(rows, list) else []
    by_id = {
        str(row.get("obligation_id") or ""): row
        for row in rows if isinstance(row, dict)
    }
    clause_by_id = {
        str(row.get("clause_id") or ""): " ".join(str(row.get("text") or "").split())
        for row in clauses
    }
    errors: list[str] = []
    for obligation in obligations:
        oid = str(obligation.get("obligation_id") or "")
        kind = str(obligation.get("kind") or "")
        policy = str(obligation.get("policy") or "")
        clause_id = str(obligation.get("clause_id") or "")
        row = by_id.get(oid)
        if row is None:
            errors.append(
                f"{oid} {kind} has no ellipsis resolution record for {clause_id}; "
                "record a source-licensed reconstruction or preserve it unresolved"
            )
            continue
        status = str(row.get("status") or "").upper()
        antecedent = " ".join(str(row.get("antecedent_span") or "").split())
        reconstruction = " ".join(str(row.get("reconstructed_span") or "").split())
        constituent = str(row.get("missing_constituent_type") or "")
        expected_constituent = str(obligation.get("missing_constituent_type") or "")
        confidence = str(row.get("confidence") or "").upper()
        admission_basis = str(row.get("admission_basis") or "").upper()
        candidate_spans = [
            " ".join(str(value).split())
            for value in row.get("candidate_antecedent_spans") or []
            if str(value).strip()
        ]
        record_roles = {str(value) for value in row.get("ethical_context_roles") or []}
        expected_roles = {
            str(value) for value in obligation.get("ethical_context_roles") or []
        }
        record_clause_ids = [str(value) for value in row.get("clause_ids") or []]
        licensed_clause_ids = {clause_id, *[
            str(value) for value in obligation.get("context_clause_ids") or []
        ]}
        licensed_sources = [
            clause_by_id.get(value, "") for value in licensed_clause_ids
            if clause_by_id.get(value, "")
        ]
        source_has_antecedent = bool(antecedent) and any(
            antecedent.casefold() in source.casefold() for source in licensed_sources
        )
        if constituent != expected_constituent:
            errors.append(
                f"{oid} {kind} missing_constituent_type must be "
                f"{expected_constituent}, not {constituent or 'empty'}"
            )
        if policy == "PRESERVE_UNRESOLVED":
            if (
                status != "UNRESOLVED" or reconstruction
                or admission_basis != "UNRESOLVED_OPEN_QUESTION"
            ):
                errors.append(
                    f"{oid} {kind} must remain UNRESOLVED unless the source "
                    "explicitly supplies its answer; clear reconstructed_span and "
                    "use UNRESOLVED_OPEN_QUESTION as admission_basis"
                )
            continue
        if policy == "RESOLVE_OR_REJECT" and status == "UNRESOLVED":
            errors.append(
                f"{oid} {kind} has no unique source-licensed antecedent; quarantine "
                "the unresolved complement and reject world admission until repaired"
            )
            continue
        if status != "RESOLVED":
            errors.append(
                f"{oid} {kind} is decision-relevant predicate ellipsis and must "
                "be RESOLVED before world admission"
            )
            continue
        if admission_basis not in {"SOURCE_ONLY", "SOURCE_AND_CONTEXT"}:
            errors.append(
                f"{oid} {kind} uses ethical context as provenance; admission_basis "
                "must retain source licensing"
            )
        if confidence not in {"HIGH", "MEDIUM"}:
            errors.append(
                f"{oid} {kind} resolution confidence is {confidence or 'empty'}; "
                "LOW or missing confidence must be quarantined"
            )
        if antecedent and candidate_spans and antecedent not in candidate_spans:
            errors.append(
                f"{oid} {kind} selected antecedent_span was not retained in its "
                "audited candidate_antecedent_spans"
            )
        if expected_roles and record_roles != expected_roles:
            errors.append(
                f"{oid} {kind} ethical_context_roles do not match its obligation; "
                "copy roles without treating them as source facts"
            )
        if not source_has_antecedent:
            errors.append(
                f"{oid} {kind} antecedent_span is not an exact span of its licensed "
                "source or discourse-context clauses"
            )
        if not set(record_clause_ids).issubset(licensed_clause_ids):
            errors.append(
                f"{oid} {kind} cites clauses outside its licensed ellipsis context"
            )
        if not reconstruction:
            errors.append(f"{oid} {kind} has an empty reconstructed_span")
    return errors
