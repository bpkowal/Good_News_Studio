"""Conservative pre-Stage-2 conditional-rule extraction and instantiation."""
from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from typing import Any


_PREFIX_RULES = (
    (re.compile(r"^\s*if\s+(?P<a>[^,;]+)[,;]\s*(?P<c>.+)$", re.I), "IF", "POSITIVE"),
    (re.compile(r"^\s*unless\s+(?P<a>[^,;]+)[,;]\s*(?P<c>.+)$", re.I), "UNLESS", "NEGATED"),
    (re.compile(r"^\s*provided\s+that\s+(?P<a>[^,;]+)[,;]\s*(?P<c>.+)$", re.I), "IF", "POSITIVE"),
    (re.compile(r"^\s*without\s+(?P<a>[^,;]+)[,;]\s*(?P<c>.+)$", re.I), "WITHOUT", "NEGATED_STATE"),
)
_SUFFIX_RULES = (
    (re.compile(r"^(?P<c>.+?)\s+only\s+(?:if|when)\s+(?P<a>.+)$", re.I), "ONLY_IF", "POSITIVE"),
    (re.compile(r"^(?P<c>.+?)\s+if\s+(?P<a>.+)$", re.I), "IF", "POSITIVE"),
    (re.compile(r"^(?P<c>.+?)\s+unless\s+(?P<a>.+)$", re.I), "UNLESS", "NEGATED"),
)
_TOKEN = re.compile(r"[a-z0-9]+", re.I)
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "if", "in", "is", "it", "of", "on", "or", "that", "the", "then",
    "to", "will", "with", "without",
}
_NEGATED_STATE = re.compile(r"\b(?:not|no|without|no\s+longer|does\s+not|is\s+not)\b", re.I)
_CESSATION = re.compile(r"\b(?:stop|stops|stopped|cease|ceases|ceased|end|ends|ended|discontinue[sd]?)\b", re.I)
_UNIVERSAL_RULE = re.compile(
    r"^\s*(?P<q>anyone|whoever|every\s+(?:person|patient|individual|worker)|"
    r"any\s+(?:person|patient|individual|worker))\s+(?:who\s+)?"
    r"(?P<a>.+?)\s+(?P<c>(?:will\s+|would\s+)?(?:die|dies|survive|survives|"
    r"recover|recovers|suffers?|is\s+harmed|is\s+injured|becomes?\s+[^.]+).*)$",
    re.IGNORECASE,
)
_GENERIC_TERMS = {
    "anyone", "whoever", "every", "any", "person", "patient", "individual",
    "worker", "who", "that",
}
_NEGATION_CUE = re.compile(r"\b(?:no|not|never|without|does\s+not|is\s+not|are\s+not)\b", re.I)


def _fold(value: Any) -> str:
    return " ".join(str(value or "").casefold().strip(" .,:;").split())


def _tokens(value: Any) -> set[str]:
    values = set()
    for token in _TOKEN.findall(_fold(value)):
        if len(token) <= 2 or token in _STOP:
            continue
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
            token = token[:-1]
        values.add(token)
    return values


def extract_conditional_rules(
    clauses: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Extract explicit surface conditionals without deciding that they obtain."""
    rules: list[dict[str, Any]] = []
    for clause in clauses:
        clause_id = str(clause.get("clause_id") or "")
        text = " ".join(str(clause.get("text") or "").split())
        match = None
        operator = polarity = ""
        for pattern, candidate_operator, candidate_polarity in (*_PREFIX_RULES, *_SUFFIX_RULES):
            match = pattern.match(text)
            if match:
                operator, polarity = candidate_operator, candidate_polarity
                break
        universal = _UNIVERSAL_RULE.match(text)
        if universal:
            match = universal
            operator, polarity = "UNIVERSAL", "VARIABLE_BOUND"
        if not match:
            continue
        antecedent = " ".join(match.group("a").strip(" .,:;").split())
        consequent = " ".join(match.group("c").strip(" .,:;").split())
        if not antecedent or not consequent:
            continue
        rules.append({
            "rule_id": f"CR_{clause_id}_{len(rules) + 1}",
            "clause_id": clause_id,
            "source_span": text,
            "operator": operator,
            "antecedent_polarity": polarity,
            "antecedent_span": antecedent,
            "consequent_span": consequent,
            "relation": "CAUSES",
            "status": "UNRESOLVED",
            "rule_kind": (
                "UNIVERSAL_IMPLICATION" if operator == "UNIVERSAL"
                else "CONDITIONAL_IMPLICATION"
            ),
        })
    return rules


def _span_score(proposition: Mapping[str, Any], span: str) -> float:
    span_folded = _fold(span)
    values = (
        _fold(proposition.get("source_proposition")),
        _fold(proposition.get("outcome")),
    )
    if any(value and (value in span_folded or span_folded in value) for value in values):
        return 1.0
    wanted = _tokens(span)
    if not wanted:
        return 0.0
    observed = set().union(*(_tokens(value) for value in values))
    return len(wanted & observed) / len(wanted)


def _cessation_entails_state(
    candidate: Mapping[str, Any], antecedent: Mapping[str, Any],
) -> bool:
    candidate_text = " ".join((
        str(candidate.get("outcome") or ""),
        str(candidate.get("source_proposition") or ""),
    ))
    antecedent_text = " ".join((
        str(antecedent.get("outcome") or ""),
        str(antecedent.get("source_proposition") or ""),
    ))
    if not _CESSATION.search(candidate_text) or not _NEGATED_STATE.search(antecedent_text):
        return False
    structural = {
        "stop", "stops", "stopped", "cease", "ceases", "ceased", "end",
        "ends", "ended", "does", "not", "without", "longer",
    }
    candidate_tokens = _tokens(candidate_text) - structural
    antecedent_tokens = _tokens(antecedent_text) - structural
    shared = candidate_tokens & antecedent_tokens
    return bool(shared) and (
        len(shared) / max(1, len(antecedent_tokens)) >= 0.5
    )


def instantiate_conditional_rules(
    skeleton: Mapping[str, Any], clauses: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Instantiate a rule only when its antecedent is an admitted actual state.

    Ordinary conditionals never create propositions. Universal rules may create
    a named consequent only when a CERTAIN action-owned antecedent binds the
    universal variable and an extracted rule-template consequent supplies every
    semantic field and the exact source provenance.
    """
    result = copy.deepcopy(dict(skeleton))
    propositions = [
        row for row in result.get("propositions") or [] if isinstance(row, dict)
    ]
    rules = extract_conditional_rules(clauses)
    instantiations: list[dict[str, Any]] = []
    for rule in rules:
        clause_id = rule["clause_id"]
        cited = [
            row for row in propositions
            if clause_id in {str(value) for value in row.get("clause_ids") or []}
        ]
        if rule.get("rule_kind") == "UNIVERSAL_IMPLICATION":
            _instantiate_universal_rule(
                result, propositions, rule, cited, instantiations,
            )
            continue
        actions = sorted({str(row.get("action_id") or "") for row in cited if row.get("action_id")})
        for action_id in actions:
            own = [row for row in cited if str(row.get("action_id") or "") == action_id]
            antecedents = sorted(
                own, key=lambda row: _span_score(row, rule["antecedent_span"]), reverse=True,
            )
            consequents = sorted(
                own, key=lambda row: _span_score(row, rule["consequent_span"]), reverse=True,
            )
            antecedent = antecedents[0] if antecedents else None
            consequent = consequents[0] if consequents else None
            antecedent_score = _span_score(antecedent or {}, rule["antecedent_span"])
            consequent_score = _span_score(consequent or {}, rule["consequent_span"])
            record = {
                **rule,
                "action_id": action_id,
                "antecedent_proposition_id": (
                    str(antecedent.get("proposition_id") or "") if antecedent else None
                ),
                "consequent_proposition_id": (
                    str(consequent.get("proposition_id") or "") if consequent else None
                ),
                "antecedent_score": antecedent_score,
                "consequent_score": consequent_score,
            }
            distinct = antecedent is not None and consequent is not None and antecedent is not consequent
            antecedent_obtains = bool(
                distinct
                and antecedent_score >= 0.6
                and str(antecedent.get("modality") or "") == "CERTAIN"
            )
            antecedent_support = None
            if distinct and not antecedent_obtains and antecedent is not None:
                antecedent_support = next((
                    row for row in propositions
                    if row is not antecedent
                    and str(row.get("action_id") or "") == action_id
                    and str(row.get("modality") or "") == "CERTAIN"
                    and _cessation_entails_state(row, antecedent)
                ), None)
                if antecedent_support is not None:
                    antecedent["modality"] = "CERTAIN"
                    antecedent.setdefault("state_entailments", []).append({
                        "operation": "CESSATION_TO_NEGATED_STATE",
                        "source_proposition_id": str(
                            antecedent_support.get("proposition_id") or ""
                        ),
                    })
                    antecedent_obtains = True
            consequent_bound = bool(distinct and consequent_score >= 0.6)
            if antecedent_obtains and consequent_bound:
                if str(consequent.get("modality") or "") == "STIPULATED_CONDITIONAL":
                    consequent["modality"] = "CERTAIN"
                consequent.setdefault("conditional_instantiations", []).append({
                    "rule_id": rule["rule_id"],
                    "antecedent_proposition_id": str(antecedent.get("proposition_id") or ""),
                    "clause_id": clause_id,
                })
                record["status"] = "INSTANTIATED"
                record["reason"] = "SAME_ACTION_CERTAIN_ANTECEDENT"
                record["antecedent_support_proposition_id"] = (
                    str(antecedent_support.get("proposition_id") or "")
                    if antecedent_support is not None else None
                )
            else:
                record["status"] = "UNRESOLVED"
                record["reason"] = (
                    "ANTECEDENT_NOT_OBTAINING" if not antecedent_obtains
                    else "CONSEQUENT_NOT_BOUND"
                )
            instantiations.append(record)
    result["conditional_rules"] = instantiations
    result["propositions"] = propositions
    return result, instantiations


def _predicate_tokens(value: Any) -> set[str]:
    return _tokens(value) - _GENERIC_TERMS


def _is_negated(value: Any) -> bool:
    return bool(_NEGATION_CUE.search(str(value or "")))


def _instantiate_universal_rule(
    skeleton: dict[str, Any], propositions: list[dict[str, Any]],
    rule: dict[str, Any], cited: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> None:
    antecedent_templates = sorted(
        cited, key=lambda row: _span_score(row, rule["antecedent_span"]), reverse=True,
    )
    consequent_templates = sorted(
        cited, key=lambda row: _span_score(row, rule["consequent_span"]), reverse=True,
    )
    antecedent_template = antecedent_templates[0] if antecedent_templates else None
    consequent_template = consequent_templates[0] if consequent_templates else None
    fused_template = bool(
        antecedent_template is not None
        and antecedent_template is consequent_template
        and _span_score(antecedent_template, rule["antecedent_span"]) >= 0.6
        and _span_score(antecedent_template, rule["consequent_span"]) >= 0.6
    )
    if (
        antecedent_template is None or consequent_template is None
        or (
            not fused_template
            and (
                antecedent_template is consequent_template
                or _span_score(antecedent_template, rule["antecedent_span"]) < 0.6
                or _span_score(consequent_template, rule["consequent_span"]) < 0.6
            )
        )
    ):
        records.append({
            **rule, "status": "UNRESOLVED",
            "reason": "UNIVERSAL_TEMPLATE_NOT_BOUND",
        })
        return

    template_ids = {
        str(row.get("proposition_id") or "")
        for row in cited
        if (
            _span_score(row, rule["antecedent_span"]) >= 0.6
            or _span_score(row, rule["consequent_span"]) >= 0.6
        )
    }
    wanted = _predicate_tokens(rule["antecedent_span"])
    antecedent_is_negated = _is_negated(rule["antecedent_span"])
    actuals = [
        row for row in propositions
        if str(row.get("proposition_id") or "") not in template_ids
        and str(row.get("modality") or "") == "CERTAIN"
        and row.get("action_id")
        and _is_negated(" ".join((
            str(row.get("outcome") or ""),
            str(row.get("source_proposition") or ""),
        ))) == antecedent_is_negated
        and wanted
        and len(wanted & _predicate_tokens(" ".join((
            str(row.get("outcome") or ""),
            str(row.get("source_proposition") or ""),
        )))) / len(wanted) >= 0.75
    ]
    parties = {
        str(row.get("party_id") or ""): row
        for row in skeleton.get("parties") or [] if isinstance(row, Mapping)
    }
    existing_ids = {
        str(row.get("proposition_id") or "") for row in propositions
    }
    instantiated = 0
    for actual in actuals:
        action_id = str(actual.get("action_id") or "")
        party_id = str(actual.get("party_id") or "")
        if not party_id:
            continue
        base_id = f"UR_{rule['clause_id']}_{action_id}_{party_id}"
        proposition_id = base_id
        suffix = 2
        while proposition_id in existing_ids:
            proposition_id = f"{base_id}_{suffix}"
            suffix += 1
        existing_ids.add(proposition_id)
        party_label = str((parties.get(party_id) or {}).get("label") or party_id)
        predicate = (
            str(rule["consequent_span"])
            if fused_template else
            str(consequent_template.get("source_proposition") or rule["consequent_span"])
        ).strip(" .")
        outcome = str(consequent_template.get("outcome") or predicate)
        outcome = re.sub(
            r"^(?:a|that|the)\s+(?:person|patient|individual|worker)\b",
            party_label, outcome, flags=re.IGNORECASE,
        )
        label_present = bool(
            _fold(party_label) and _fold(party_label) in _fold(outcome)
        )
        if (
            _GENERIC_TERMS & _tokens(outcome)
            or not label_present
        ):
            outcome = f"{party_label} {predicate}".strip()
        derived = copy.deepcopy(consequent_template)
        derived.update({
            "proposition_id": proposition_id,
            "neutral_proposition_id": None,
            "action_id": action_id,
            "party_id": party_id,
            "outcome": outcome,
            "directness": "DOWNSTREAM",
            "modality": "CERTAIN",
            "source_proposition": str(
                consequent_template.get("source_proposition") or rule["consequent_span"]
            ),
            "clause_ids": [rule["clause_id"]],
            "source_rule_id": rule["rule_id"],
            "source_effect_ids": [str(actual.get("proposition_id") or "")],
            "derivation_operation": "UNIVERSAL_RULE_INSTANTIATION",
        })
        propositions.append(derived)
        records.append({
            **rule,
            "status": "INSTANTIATED",
            "reason": "CERTAIN_NAMED_ANTECEDENT_BINDS_UNIVERSAL",
            "action_id": action_id,
            "bound_party_id": party_id,
            "antecedent_proposition_id": str(actual.get("proposition_id") or ""),
            "consequent_proposition_id": proposition_id,
            "antecedent_template_proposition_id": str(
                antecedent_template.get("proposition_id") or ""
            ),
            "consequent_template_proposition_id": str(
                consequent_template.get("proposition_id") or ""
            ),
            "template_form": "FUSED" if fused_template else "SPLIT",
        })
        instantiated += 1

    ledger = [row for row in skeleton.get("node_evidence_ledger") or [] if isinstance(row, dict)]
    for row in ledger:
        materialized = {
            str(value) for value in row.get("materialized_proposition_ids") or []
        }
        if materialized & template_ids:
            row.setdefault("normalization_annotations", []).append({
                "reason": "UNIVERSAL_RULE_TEMPLATE",
                "detail": "Generic rule evidence is retained outside the action graph.",
                "proposition_ids": sorted(materialized & template_ids),
            })
    propositions[:] = [
        row for row in propositions
        if str(row.get("proposition_id") or "") not in template_ids
    ]
    if not instantiated:
        records.append({
            **rule, "status": "UNRESOLVED",
            "reason": "NO_CERTAIN_NAMED_ANTECEDENT",
            "antecedent_template_proposition_id": str(
                antecedent_template.get("proposition_id") or ""
            ),
            "consequent_template_proposition_id": str(
                consequent_template.get("proposition_id") or ""
            ),
        })
