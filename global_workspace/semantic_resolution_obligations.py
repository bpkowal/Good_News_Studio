"""Convert spaCy observations into non-authoritative semantic questions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from .syntactic_annotation import annotate_text, load_english_parser, predicate_lemmas


ELLIPSIS_CUE = re.compile(
    r"\b(?:do|does|did|would do)\s+(?:the\s+)?same\b|\bdoes\s+too\b|\bso\s+does\b",
    re.IGNORECASE,
)
CONDITIONAL_CUE = re.compile(
    r"\b(?:if|unless|provided(?: that)?|only if|only when)\b", re.IGNORECASE,
)


def build_resolution_obligations(
    clauses: Sequence[Mapping[str, Any]], *, nlp: Any | None = None,
) -> dict[str, Any]:
    """Identify locations requiring semantics without supplying the answer."""
    parser = nlp or load_english_parser()
    obligations: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []

    def add(
        clause_id: str, obligation_type: str, span: str,
        predicate_ids: list[str], predicate_lemma_values: list[str], question: str,
    ) -> None:
        obligations.append({
            "obligation_id": f"SRO_{len(obligations) + 1}",
            "type": obligation_type,
            "clause_ids": [clause_id],
            "source_span": span,
            "predicate_ids": predicate_ids,
            "predicate_lemmas": list(dict.fromkeys(predicate_lemma_values)),
            "question": question,
            "required_handling": "REPRESENT_OR_QUARANTINE",
            "authority": "SYNTACTIC_CUE_ONLY",
        })

    for clause in clauses:
        clause_id = str(clause.get("clause_id") or "")
        text = str(clause.get("text") or "")
        annotation = annotate_text(text, nlp=parser)
        annotations.append({"clause_id": clause_id, **annotation})
        predicates = annotation["predicates"]
        by_id = {row["predicate_id"]: row for row in predicates}

        conjoined = [
            row for row in predicates
            if row.get("dependency") == "conj"
            or any(child.get("dependency") == "conj" for child in row.get("children") or [])
        ]
        if conjoined:
            add(
                clause_id, "SHARED_ARGUMENT_CONJUNCTION", text,
                [row["predicate_id"] for row in conjoined],
                [row["lemma"] for row in conjoined],
                "Which subjects or objects are inherited across conjuncts, and which atomic events remain distinct?",
            )
        if ELLIPSIS_CUE.search(text):
            add(
                clause_id, "ELLIPSIS_OR_LIGHT_VERB", text,
                [row["predicate_id"] for row in predicates],
                [row["lemma"] for row in predicates],
                "What source predicate and arguments, if uniquely licensed, fill the ellipsis?",
            )
        appositives = [
            token for token in annotation["tokens"] if token.get("dependency") == "appos"
        ]
        if appositives:
            add(
                clause_id, "APPOSITIVE_IDENTITY", text, [], [],
                "Does the appositive identify an existing party without creating a second entity or event?",
            )
        embedded = [
            row for row in predicates if row.get("dependency") in {"ccomp", "xcomp"}
        ]
        if embedded:
            add(
                clause_id, "EMBEDDED_CLAUSE", text,
                [row["predicate_id"] for row in embedded],
                [row["lemma"] for row in embedded],
                "Is the embedded predicate asserted, merely enabled, requested, intended, or otherwise modal?",
            )
        negated = [row for row in predicates if row.get("negations")]
        if negated and embedded:
            add(
                clause_id, "NEGATED_SUBORDINATE_SCOPE", text,
                [row["predicate_id"] for row in [*negated, *embedded]],
                [row["lemma"] for row in [*negated, *embedded]],
                "Which relation or proposition is inside the negation scope, and which subordinate events remain asserted?",
            )
        conditional = [row for row in predicates if row.get("conditional_markers")]
        if conditional or CONDITIONAL_CUE.search(text):
            add(
                clause_id, "CONDITIONAL_SCOPE", text,
                [row["predicate_id"] for row in conditional],
                [row["lemma"] for row in conditional],
                "Which consequent is conditional, what is its antecedent, and must unconditional admission be withheld?",
            )
        # Multiple occurrences of one predicate lemma need semantic identity,
        # even though each occurrence is grammatically visible.
        lemma_groups: dict[str, list[dict[str, Any]]] = {}
        for row in predicates:
            lemma_groups.setdefault(str(row.get("lemma") or ""), []).append(row)
        repeated = [rows for lemma, rows in lemma_groups.items() if lemma and len(rows) > 1]
        for rows in repeated:
            add(
                clause_id, "REPEATED_PREDICATE_IDENTITY", text,
                [row["predicate_id"] for row in rows],
                [row["lemma"] for row in rows],
                "Which predicate occurrence corresponds to each semantic event and action branch?",
            )
    return {
        "version": "1.0",
        "mode": "READ_ONLY_RESOLUTION_QUESTIONS",
        "obligations": obligations,
        "annotations": annotations,
    }


def assess_obligation_coverage(
    obligations: Mapping[str, Any], neutral: Mapping[str, Any], *, nlp: Any,
) -> list[dict[str, Any]]:
    """Report candidate coverage without claiming semantic resolution.

    Syntax can establish that candidate nodes exist, but cannot establish that
    ellipsis, identity, conditional, or negation scope was resolved correctly.
    Those cases remain explicitly marked for semantic review.
    """
    propositions = [row for row in neutral.get("propositions") or [] if isinstance(row, Mapping)]
    unresolved = " ".join(str(value) for value in neutral.get("unresolved_source_spans") or [])
    results = []
    for obligation in obligations.get("obligations") or []:
        clause_ids = set(str(value) for value in obligation.get("clause_ids") or [])
        relevant = [
            row for row in propositions
            if clause_ids & set(str(value) for value in row.get("clause_ids") or [])
        ]
        represented_lemmas = {
            lemma
            for row in relevant
            for lemma in predicate_lemmas(
                str(row.get("source_proposition") or row.get("outcome") or ""), nlp=nlp,
            )
        }
        required_lemmas = set(str(value) for value in obligation.get("predicate_lemmas") or [])
        span = str(obligation.get("source_span") or "")
        quarantined = bool(span and span in unresolved)
        candidate_nodes_present = bool(relevant) and (
            not required_lemmas or required_lemmas <= represented_lemmas
        )
        status = (
            "QUARANTINED" if quarantined
            else "CANDIDATE_NODES_PRESENT_REVIEW_REQUIRED" if candidate_nodes_present
            else "UNADDRESSED"
        )
        results.append({
            "obligation_id": obligation.get("obligation_id"),
            "type": obligation.get("type"),
            "status": status,
            "matching_proposition_ids": [row.get("proposition_id") for row in relevant],
            "missing_predicate_lemmas": sorted(required_lemmas - represented_lemmas),
        })
    return results
