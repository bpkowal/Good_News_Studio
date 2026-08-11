from __future__ import annotations

import re
from typing import Any


def _data(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict"):
        return result.to_dict()
    raise TypeError("result must be a WorkspaceResult or its dictionary representation")


def _sentence(text: str) -> str:
    cleaned = " ".join(str(text).split()).strip()
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    if cleaned and cleaned[-1] not in ".?!":
        cleaned += "."
    return cleaned


def _evidence_score(text: str) -> tuple[int, int]:
    lowered = text.casefold()
    concrete = len(re.findall(
        r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+|"
        r"kill\w*|die\w*|death\w*|save\w*|harm\w*|injur\w*|loss|lost|"
        r"prevent\w*|risk\w*|chance|certain\w*|expected|net|total)\b",
        lowered,
    ))
    return concrete, min(len(text.split()), 80)


def _looks_truncated(text: str) -> bool:
    cleaned = text.replace("\u200b", "").strip()
    if not cleaned:
        return True
    if re.search(r"[\u3400-\u9fff]", cleaned):
        return True
    if cleaned.endswith(("-", "–", "—")):
        return True
    # Delegate fields have hard character ceilings. Hitting a round ceiling
    # without terminal punctuation is stronger evidence of clipping than a
    # naturally short phrase that simply omitted its full stop.
    if len(cleaned) >= 100 and len(cleaned) % 10 == 0 and cleaned[-1] not in ".?!":
        return True
    last = re.findall(r"[A-Za-z]+", cleaned)
    harmless_short = {"die", "risk", "life", "job", "jobs", "harm", "aid"}
    return bool(
        last
        and len(last[-1]) <= 3
        and last[-1].casefold() not in harmless_short
        and cleaned[-1] not in ".?!"
    )


def _clean_fragment(text: str) -> str:
    cleaned = " ".join(text.replace("\u200b", "").split()).strip()
    if not _looks_truncated(cleaned):
        return cleaned
    if "—" in cleaned:
        head = cleaned.split("—", 1)[0].rstrip(" ,;:")
        if len(head.split()) >= 5:
            return head
    for separator in (",", ";"):
        if separator in cleaned:
            head = cleaned.rsplit(separator, 1)[0].rstrip(" ,;:")
            if len(head.split()) >= 5:
                return head
    return cleaned


def _candidate_recommendation(candidate: dict[str, Any]) -> str:
    explicit = candidate.get("recommended_action", "")
    scores = candidate.get("action_scores") or {}
    return explicit or (max(scores, key=scores.get) if scores else "")


def _readable_condition(value: str) -> str:
    normalized = " ".join(str(value).replace("_", " ").split()).strip()
    if not normalized or re.fullmatch(r"[A-Z ]+", normalized):
        return ""
    return normalized


def _original_actions(data: dict[str, Any]) -> list[str]:
    synthesized = {
        proposal.get("action")
        for proposal in data.get("synthesis_proposals", [])
        if proposal.get("accepted") and proposal.get("action")
    }
    original = [action for action in data.get("actions", []) if action not in synthesized]
    return original or list(data.get("actions", []))


def _best_action_case(candidates: list[dict[str, Any]], action: str) -> str:
    options = []
    for candidate in candidates:
        if not candidate.get("landscape_semantic_valid"):
            continue
        case = (candidate.get("landscape_cases") or {}).get(action, "")
        if case:
            options.append(case)
    return max(
        options,
        key=lambda text: (not _looks_truncated(text), _evidence_score(text)),
    ) if options else ""


def _baseline_reason(
    data: dict[str, Any], specialist: str, recommendation: str, original_actions: list[str]
) -> str:
    baseline = (data.get("source_baselines") or {}).get(specialist, {})
    action_id = str(baseline.get("action_id", ""))
    if not re.fullmatch(r"A\d+", action_id):
        return ""
    index = int(action_id[1:])
    if index >= len(original_actions) or original_actions[index] != recommendation:
        return ""
    return " ".join(str(baseline.get("reason", "")).split())


def _support_reason(
    data: dict[str, Any], candidate: dict[str, Any], recommendation: str,
    original_actions: list[str],
) -> str:
    baseline = _baseline_reason(
        data, candidate.get("specialist", ""), recommendation, original_actions
    )
    if len(baseline) >= 170 or _looks_truncated(baseline):
        baseline = ""
    rationale = _sentence(candidate.get("rationale", ""))
    axis = _sentence(candidate.get("landscape_decisive_axis", "").replace("_", " "))
    details = [_sentence(baseline), rationale, axis]
    unique: list[str] = []
    for part in details:
        if not part:
            continue
        normalized = part.casefold().rstrip(".")
        if any(normalized in prior.casefold() or prior.casefold().rstrip(".") in normalized for prior in unique):
            continue
        unique.append(part)
    return " ".join(unique[:2])


def render_public_judgment(result: Any) -> str:
    """Render the full deliberative trajectory without exposing internal machinery."""
    data = _data(result)
    recorded_cycles = data.get("cycles", [])
    cycles = [cycle for cycle in recorded_cycles if not cycle.get("is_hypothetical")]
    counterfactual_cycles = [
        cycle for cycle in recorded_cycles if cycle.get("is_hypothetical")
    ]
    if not cycles:
        return "Ethical judgment\n\nNo reliable deliberative judgment was produced.\n"
    final = cycles[-1]
    all_candidates = [
        candidate for cycle in cycles for candidate in cycle.get("candidates", [])
        if candidate.get("schema_valid")
    ]
    counterfactual_candidates = [
        candidate
        for cycle in counterfactual_cycles
        for candidate in cycle.get("candidates", [])
        if candidate.get("schema_valid")
    ]
    final_candidates = [candidate for candidate in final.get("candidates", []) if candidate.get("schema_valid")]
    original_actions = _original_actions(data)
    status = data.get("judgment_status", "INCONCLUSIVE")
    selected = data.get("selected_action", "")
    plurality = data.get("current_plurality", "")
    actionable = status in {"ACTION_RECOMMENDATION", "CONTESTED_RECOMMENDATION"}
    recommendation = selected if actionable else plurality

    # Use the latest valid position per specialist, but recover that specialist's
    # most concrete grounded explanation from any cycle supporting the same action.
    latest_by_specialist: dict[str, dict[str, Any]] = {}
    for candidate in all_candidates:
        latest_by_specialist[candidate.get("specialist", "")] = candidate
    supporter_names = {
        name for name, candidate in latest_by_specialist.items()
        if _candidate_recommendation(candidate) == recommendation
    }
    supporters: list[dict[str, Any]] = []
    for name in supporter_names:
        options = [
            candidate for candidate in all_candidates
            if candidate.get("specialist") == name
            and _candidate_recommendation(candidate) == recommendation
        ]
        supporters.append(max(
            options,
            key=lambda candidate: _evidence_score(" ".join([
                _baseline_reason(data, name, recommendation, original_actions),
                candidate.get("rationale", ""),
                (candidate.get("landscape_cases") or {}).get(recommendation, ""),
                candidate.get("landscape_decisive_axis", ""),
            ])),
        ))
    supporters.sort(
        key=lambda candidate: _evidence_score(_support_reason(
            data, candidate, recommendation, original_actions
        )),
        reverse=True,
    )

    dissent = final.get("dissent")
    if not dissent and recommendation:
        opponents = [
            candidate for candidate in final_candidates
            if _candidate_recommendation(candidate) not in {"", recommendation}
        ]
        dissent = max(
            opponents,
            key=lambda candidate: (
                candidate.get("epistemic_confidence", candidate.get("confidence", 0)),
                candidate.get("preference_strength", candidate.get("friction", 0)),
            ),
            default=None,
        )

    lines = ["Ethical judgment", ""]
    if actionable:
        qualifier = "Contested recommendation" if status == "CONTESTED_RECOMMENDATION" else "Recommendation"
        lines.append(f"{qualifier}: {selected}")
    elif status == "CONDITIONAL":
        lines.append(f"Conditional judgment: the current leading action is {plurality or 'not established'}.")
    elif status == "UNDERDETERMINED":
        lines.append(f"Underdetermined: the stated facts do not justify choosing conclusively; current plurality is {plurality or 'none'}.")
    else:
        lines.append("No sufficiently reliable recommendation was produced.")
    lines.append(f"Policy support: {float(data.get('confidence', 0)):.2f}")
    lines.append(
        f"Epistemic confidence: {float(data.get('epistemic_confidence', 0)):.2f} "
        "(likelihood the judgment survives further factual inquiry and scrutiny)"
    )
    visibility = next(
        (
            assessment for assessment in data.get("visibility_assessments", [])
            if assessment.get("valid") and assessment.get("activated")
        ),
        None,
    )
    if visibility:
        penalties = [
            f"{action}: ×{float(multiplier):.2f}"
            for action, multiplier in visibility.get("action_multipliers", {}).items()
            if float(multiplier) < 1.0
        ]
        lines.extend([
            "",
            "Visibility audit (non-voting):",
            f"- Affected group: {visibility.get('affected_group') or 'not specified'}.",
            f"- Epistemic-exclusion mechanism: {_sentence(visibility.get('mechanism', ''))}",
            f"- Confidence adjustment: {', '.join(penalties)}.",
        ])

    policy = final.get("policy") or {}
    alternatives = [action for action in original_actions if action != recommendation]
    alternatives.sort(key=lambda action: policy.get(action, 0), reverse=True)
    compared_actions = ([recommendation] if recommendation else []) + alternatives[:1]
    comparisons = [(action, _best_action_case(all_candidates, action)) for action in compared_actions]
    if comparisons and any(case for _, case in comparisons):
        lines.extend(["", "Original action and consequence comparison:"])
        for action, case in comparisons:
            description = _sentence(_clean_fragment(case)) or "No reliable consequence summary was produced."
            lines.append(f"- {action}: {description}")

    if supporters:
        heading = "Reasons supporting the judgment:" if actionable else "Leading considerations:"
        lines.extend(["", heading])
        seen: set[str] = set()
        for supporter in supporters:
            reason = _support_reason(data, supporter, recommendation, original_actions)
            if not reason or reason.casefold() in seen:
                continue
            seen.add(reason.casefold())
            lines.append(f"- {supporter.get('specialist', 'supporting perspective')}: {reason}")
            if len(seen) == 3:
                break
        rules = [
            _sentence(supporter.get("decision_rule", ""))
            for supporter in supporters
            if supporter.get("decision_rule")
        ]
        if rules:
            lines.extend(["", f"Decision rule: {rules[0]}"])

    if dissent and dissent.get("rationale"):
        opposing_action = _candidate_recommendation(dissent)
        objection_case = (dissent.get("landscape_cases") or {}).get(opposing_action, "")
        objection = _sentence(dissent["rationale"])
        if objection_case:
            objection += " " + _sentence(objection_case)
        lines.extend(["", f"Strongest objection ({dissent.get('specialist', 'dissenting')}): {objection}"])
        if supporters:
            supporting_axis = _readable_condition(supporters[0].get("landscape_decisive_axis", ""))
            opposing_axis = _readable_condition(dissent.get("landscape_decisive_axis", ""))
            if supporting_axis or opposing_axis:
                lines.append(
                    "Why disagreement remains: the judgment gives greater weight to "
                    f"{supporting_axis or 'the leading consideration'} than to "
                    f"{opposing_axis or 'the preserved objection'}, without treating the objection as resolved."
                )
    elif alternatives:
        alternative_case = _best_action_case(all_candidates, alternatives[0])
        if alternative_case:
            lines.extend(["", f"Strongest case for the alternative: {_sentence(_clean_fragment(alternative_case))}"])

    residue = data.get("moral_residue") or []
    if residue:
        lines.append("Unresolved moral considerations: " + ", ".join(residue) + ".")

    assumptions: list[str] = []
    reversals: list[str] = []
    factual_reversals: list[str] = []
    normative_reversals: list[str] = []
    reversal_reviews: list[str] = []
    # Counterfactual candidates may contribute conditional thresholds and
    # review verdicts, but never reasons, votes, or policy support for the base
    # scenario.
    for candidate in [*all_candidates, *counterfactual_candidates]:
        assumption = _readable_condition(candidate.get("unsupported_assumption", ""))
        reversal = _readable_condition(candidate.get("reversal_condition", ""))
        if assumption and candidate.get("assumption_status") in {"CONDITIONAL", "UNDERDETERMINED"} and assumption not in assumptions:
            assumptions.append(assumption)
        if reversal and reversal not in reversals:
            reversals.append(reversal)
        factual = _readable_condition(candidate.get("factual_reversal_threshold", ""))
        normative = _readable_condition(candidate.get("normative_reversal_threshold", ""))
        revised = _readable_condition(candidate.get("revised_reversal_condition", ""))
        if factual and factual not in factual_reversals:
            factual_reversals.append(factual)
        if normative and normative not in normative_reversals:
            normative_reversals.append(normative)
        response = candidate.get("reversal_review_response", "NOT_TESTED")
        justification = _readable_condition(
            candidate.get("reversal_review_justification", "")
        )
        if response in {"ACCEPT", "REVISE", "REJECT"} and justification:
            verb = {"ACCEPT": "accepted", "REVISE": "revised", "REJECT": "rejected"}[response]
            reviewed = f"{candidate.get('specialist', 'specialist')} {verb} the challenge: {justification}"
            if revised:
                reviewed += f" Revised condition: {revised}"
            if reviewed not in reversal_reviews:
                reversal_reviews.append(reviewed)
    for condition in data.get("reopen_conditions") or []:
        readable = _readable_condition(condition)
        if readable and readable not in reversals:
            reversals.append(readable)
    if assumptions:
        lines.extend(["", "Uncertain assumptions identified during audit:"])
        lines.extend(f"- {_sentence(assumption)}" for assumption in assumptions[:4])
    if factual_reversals:
        lines.extend(["", "Factual reversal thresholds:"])
        lines.extend(f"- {_sentence(condition)}" for condition in factual_reversals[:3])
    if normative_reversals:
        lines.extend(["", "Normative reversal thresholds:"])
        lines.extend(f"- {_sentence(condition)}" for condition in normative_reversals[:3])
    if reversal_reviews:
        lines.extend(["", "Adversarial reversal review:"])
        lines.extend(f"- {_sentence(review)}" for review in reversal_reviews[:3])
    residual_reversals = [
        condition for condition in reversals
        if condition not in factual_reversals and condition not in normative_reversals
    ]
    if residual_reversals:
        lines.extend(["", "Reconsider if:"])
        for condition in residual_reversals[:5]:
            clause = re.sub(r"^if\s+", "", condition, flags=re.I)
            lines.append(f"- {_sentence(clause)}")

    tensions = [
        _sentence(_clean_fragment(proposal.get("residual_tension", "")))
        for proposal in data.get("problem_reformulations", [])
        if proposal.get("residual_tension")
    ]
    if tensions:
        lines.extend(["", "Residual tension identified by the boundary audit:", f"- {tensions[0]}"])

    considered_syntheses = [
        proposal.get("action") for proposal in data.get("synthesis_proposals", [])
        if proposal.get("accepted") and proposal.get("action") != recommendation
    ]
    if considered_syntheses:
        lines.extend([
            "",
            "Additional synthesis considered during deliberation (not treated as an original option here):",
        ])
        lines.extend(f"- {action}" for action in considered_syntheses[:3])
    return "\n".join(lines) + "\n"
