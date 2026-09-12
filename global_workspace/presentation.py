from __future__ import annotations

import re
from typing import Any


def _data(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict"):
        return result.to_dict()
    raise TypeError("result must be a WorkspaceResult or its dictionary representation")


def _public_claim(text: str) -> str:
    """Strip internal graph IDs from text that reaches the public judgment."""
    cleaned = re.sub(r"\bAudit QUESTION:[0-9a-f]+:\s*", "", str(text or ""))
    cleaned = re.sub(r"\bQUESTION:[0-9a-f]+\b", "the live unresolved issue", cleaned)
    return " ".join(cleaned.split()).strip()


def _judgment_cycle(cycles: list[dict[str, Any]]) -> dict[str, Any]:
    """Use the last cycle that still had a valid parliament, not a failed audit."""
    for cycle in reversed(cycles):
        if any(
            candidate.get("schema_valid")
            for candidate in cycle.get("candidates", []) or []
        ):
            return cycle
    return cycles[-1] if cycles else {}


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


def _action_text_by_id(data: dict[str, Any]) -> dict[str, str]:
    """Resolve run-local canonical IDs without exposing presentation order as identity."""
    return {
        f"A{index}": action
        for index, action in enumerate(data.get("actions") or [])
        if str(action).strip()
    }


def _graph_node_label(data: dict[str, Any], node_id: str, fallback: str) -> str:
    return next(
        (
            str(node.get("label", ""))
            for graph in data.get("semantic_graphs") or []
            for node in graph.get("nodes", [])
            if node.get("id") == node_id and str(node.get("label", "")).strip()
        ),
        fallback,
    )


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


def _sentence_or_blank(text: str) -> str:
    cleaned = " ".join(str(text).split()).strip()
    if not cleaned:
        return ""
    cleaned = cleaned[0].upper() + cleaned[1:]
    if cleaned[-1] not in ".?!":
        cleaned += "."
    return cleaned


def _format_quantity(number: str, unit: str = "") -> str:
    try:
        value = float(number)
    except ValueError:
        return number
    if unit.casefold() == "fraction" and 0.0 <= value <= 1.0:
        return f"{value * 100:.0f}%"
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:g}"


def _humanize_threshold(text: str) -> str:
    cleaned = " ".join(str(text).replace("_", " ").split()).strip()
    if not cleaned:
        return ""

    pattern = re.compile(
        r"(?P<lhs>[A-Za-z][A-Za-z ]*?)\s*(?P<op><=|>=|<|>|≤|≥)\s*"
        r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z%]+)?"
    )

    def replace(match: re.Match[str]) -> str:
        lhs = " ".join(match.group("lhs").split()).strip()
        op = match.group("op")
        unit = match.group("unit") or ""
        if op == "≤":
            op = "<="
        elif op == "≥":
            op = ">="
        probability_like = unit.casefold() in {"probability", "chance", "risk", "likelihood"}
        if probability_like:
            try:
                value = float(match.group("num"))
            except ValueError:
                num = _format_quantity(match.group("num"), unit)
            else:
                num = f"{value * 100:.0f}%" if 0.0 <= value <= 1.0 else _format_quantity(match.group("num"), unit)
        else:
            num = _format_quantity(match.group("num"), unit)
        if probability_like:
            if op == "<":
                return f"{lhs} falls below around {num}"
            if op == ">":
                return f"{lhs} exceeds around {num}"
            return f"{lhs} falls to around {num}"
        unit_text = f" {unit}" if unit else ""
        if op == "<":
            return f"{lhs} falls below {num}{unit_text}"
        if op == ">":
            return f"{lhs} exceeds {num}{unit_text}"
        if op == "<=":
            return f"{lhs} is at or below {num}{unit_text}"
        return f"{lhs} is at or above {num}{unit_text}"

    cleaned = pattern.sub(replace, cleaned)
    cleaned = cleaned.replace("count", "").replace("fraction", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def summarize_problem_shape_paragraphs(data: dict[str, Any]) -> list[str]:
    semantic_state = data.get("authoritative_semantic_state") or {}
    relations = semantic_state.get("problem_shape_relations") or []
    paragraphs: list[str] = []
    dimension_states = semantic_state.get("action_dimension_states") or []
    framework_priorities = semantic_state.get("framework_dimension_priorities") or []
    substantive = {
        str(item.get("dimension", "")).upper()
        for item in dimension_states
        if str(item.get("dimension", "")).upper()
        not in {"FEASIBILITY", "UNKNOWN", "OTHER_PRIMARY_GOOD", "OTHER"}
    }
    if len(substantive) >= 2:
        lexical = any(
            str(item.get("framework", "")).upper() == "RAWLSIAN"
            and str(item.get("dimension", "")).upper() == "LIBERTY_AUTONOMY"
            and str(item.get("relation", "")).upper() == "LEXICAL_PRIORITY_OVER"
            for item in framework_priorities
        )
        heading = "The actions form a multidimensional tradeoff"
        if lexical:
            heading += "; Rawlsian priority ranks basic liberty lexically over material position"
        paragraphs.append(_sentence_or_blank(heading))
        by_action: dict[str, list[dict[str, Any]]] = {}
        for item in dimension_states:
            action_id = str(item.get("action_id", "")).strip()
            if action_id:
                by_action.setdefault(action_id, []).append(item)
        dimension_names = {
            "LIBERTY_AUTONOMY": "liberty and autonomy interests",
            "MATERIAL_FLOOR": "material floor",
            "OPPORTUNITY_ACCESS": "opportunity and access",
            "BASIC_SECURITY": "basic security",
            "FEASIBILITY": "feasibility",
        }
        direction_names = {
            "IMPROVES": "improves", "WORSENS": "worsens",
            "PRESERVES": "preserved", "MIXED": "mixed",
            "UNCERTAIN": "uncertain", "ESTABLISHED": "high",
        }
        for action_id in sorted(by_action):
            cells = []
            for item in sorted(
                by_action[action_id],
                key=lambda value: {
                    "MATERIAL_FLOOR": 0, "LIBERTY_AUTONOMY": 1, "FEASIBILITY": 2,
                }.get(str(value.get("dimension", "")).upper(), 9),
            ):
                dimension = str(item.get("dimension", "UNKNOWN")).upper()
                direction = str(item.get("direction", "UNKNOWN")).upper()
                qualifier = str(item.get("magnitude_or_qualifier", "")).upper()
                value = direction_names.get(direction, direction.lower().replace("_", " "))
                if qualifier not in {
                    "", "UNKNOWN", "HIGHER", "LOWER", "PRESERVED", "MIXED", "HIGH",
                }:
                    value += f" ({qualifier.lower().replace('_', ' ')})"
                cells.append(
                    f"{dimension_names.get(dimension, dimension.lower().replace('_', ' '))}: {value}"
                )
            paragraphs.append(_sentence_or_blank(f"{action_id} — " + "; ".join(cells)))
            if len(paragraphs) >= 4:
                return paragraphs

    ordered = sorted(
        relations,
        key=lambda relation: {
            "AGGREGATE_VS_DISTRIBUTIVE": 0,
            "TEMPORAL_RISK_ASYMMETRY": 1,
            "EPISTEMIC_ASYMMETRY": 2,
            "DECISION_BOUNDARY": 4,
            "DECISION_CRITICAL_UNKNOWN": 5,
            "OUTCOME_EQUIVALENCE": 6,
            "ASYMMETRIC_COST": 7,
        }.get(str(relation.get("relation", "")).upper(), 99),
    )
    for relation in ordered:
        relation_name = str(relation.get("relation", "")).strip().upper()
        statement = str(relation.get("statement", "")).strip()
        if not statement:
            continue
        if relation_name == "MULTIDIMENSIONAL_TRADEOFF" and dimension_states:
            continue
        if relation_name == "DECISION_BOUNDARY":
            boundary = statement.split(" if ", 1)[-1] if " if " in statement.lower() else statement
            cleaned = _humanize_threshold(boundary)
            if cleaned:
                paragraphs.append(
                    _sentence_or_blank(
                        f"The current recommendation would change if {cleaned}"
                    )
                )
        elif relation_name == "DECISION_CRITICAL_UNKNOWN":
            variable = statement.split(":", 1)[-1].strip() if ":" in statement else statement
            paragraphs.append(
                _sentence_or_blank(
                    "A decision-critical uncertainty remains unresolved: "
                    + _public_claim(variable)
                )
            )
        elif relation_name in {
            "AGGREGATE_VS_DISTRIBUTIVE",
            "TEMPORAL_RISK_ASYMMETRY",
            "EPISTEMIC_ASYMMETRY",
            "OUTCOME_EQUIVALENCE",
            "ASYMMETRIC_COST",
        }:
            paragraphs.append(_sentence_or_blank(statement))
        else:
            paragraphs.append(_sentence_or_blank(statement))
        if len(paragraphs) >= 4:
            break

    if not paragraphs:
        reformulations = data.get("problem_reformulations") or []
        accepted = next((item for item in reformulations if item.get("accepted")), None)
        if accepted:
            residual = _sentence_or_blank(str(accepted.get("residual_tension", "")))
            question = _sentence_or_blank(str(accepted.get("question", "")))
            switch = _sentence_or_blank(str(accepted.get("switch_condition", "")))
            if residual:
                paragraphs.append(residual)
            if switch:
                paragraphs.append(switch)
            if question and question not in paragraphs:
                paragraphs.append(question)

    decision_variables = semantic_state.get("decision_variables") or []
    if decision_variables and len(paragraphs) < 4:
        variable = decision_variables[0]
        label = str(variable.get("label", "")).strip()
        if label:
            paragraphs.append(
                _sentence_or_blank(
                    "The authoritative state keeps one decision-critical variable open: "
                    + _public_claim(label)
                )
            )

    return paragraphs


def _baseline_reason(
    data: dict[str, Any], specialist: str, recommendation: str, original_actions: list[str]
) -> str:
    baseline = (data.get("source_baselines") or {}).get(specialist, {})
    action_id = str(baseline.get("action_id", ""))
    status = str(baseline.get(
        "status", "DIRECT" if re.fullmatch(r"A\d+", action_id) else "UNAVAILABLE"
    )).upper()
    if status != "DIRECT":
        return ""
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
    if candidate.get("specialist") == "deontological":
        selected_id = next((
            action_id for action_id, action in _action_text_by_id(data).items()
            if action == recommendation
        ), "")
        assessments = (
            (data.get("authoritative_semantic_state") or {})
            .get("deontological_assessments") or []
        )
        calibrated = next((
            assessment for assessment in assessments
            if assessment.get("canonical_action_id") == selected_id
        ), None)
        if calibrated and (
            calibrated.get("calibration_errors")
            or calibrated.get("resolution_status") != "RESOLVED"
        ):
            # The frozen testimony remains visible in the appendix, but once
            # adjudication calibration supersedes REQUIRED/PROHIBITED with a
            # contested operative judgment it must not be repeated as current
            # moral support in the public rationale.
            baseline = ""
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
    reason = " ".join(unique[:2])
    retained = _latest_workspace_contribution(
        data, str(candidate.get("specialist", ""))
    )
    visibility = str(retained.get("retained_issue_visibility", "NOT_APPLICABLE"))
    issue = _sentence(str(retained.get("visible_retained_issue", "")))
    if issue and visibility in {"STRUCTURED_ONLY", "OMITTED"}:
        # Presentation repair only: this does not modify the candidate, vote,
        # salience, confidence, or authoritative framework state.
        reason = " ".join(filter(None, (
            reason,
            f"Retained unresolved issue: {_lower_initial(issue)}",
        )))
    return reason


def _latest_workspace_contribution(
    data: dict[str, Any], specialist: str,
) -> dict[str, Any]:
    for cycle in reversed(data.get("cycles", []) or []):
        state = ((cycle.get("broadcast") or {}).get("problem_state") or {})
        for contribution in state.get("workspace_contributions", []) or []:
            if str(contribution.get("agent", "")) == specialist:
                return dict(contribution)
    return {}


def _agreement_class(data: dict[str, Any], candidate: dict[str, Any]) -> str:
    """Classify support without changing the underlying policy calculation."""
    contribution = _latest_workspace_contribution(
        data, str(candidate.get("specialist", ""))
    )
    choice = str(
        contribution.get("choice_status")
        or candidate.get("baseline_status")
        or candidate.get("selection_status")
        or "DIRECT"
    ).upper()
    assumption = str(candidate.get("assumption_status", "")).upper()
    unresolved = str(candidate.get("unresolved", "NONE")).upper()
    retention = str(candidate.get("framework_retention_status", "")).upper()
    if "OUTSIDE_ACTION_SET" in choice or choice in {"FALLBACK", "FORCED"}:
        return "FALLBACK"
    if assumption == "NORMATIVELY_CONTESTED" or unresolved in {
        "NORMATIVE_ADJUDICATION", "RESOLVE_NORMATIVE_TENSION",
    }:
        return "CONTESTED"
    if choice == "CONDITIONAL" or assumption == "CONDITIONAL":
        return "CONDITIONAL"
    if choice in {"PROVISIONAL", "UNDERDETERMINED"}:
        return "PROVISIONAL"
    if candidate.get("utilitarian_decision_depends_on_unknown"):
        return "CONDITIONAL"
    if str(candidate.get("selection_status", "")).upper() == "PROVISIONAL":
        return "PROVISIONAL"
    if retention in {"PRESERVED_AFTER_REJECTED_UPDATE", "PRIOR_STATE_PRESERVED"}:
        return "PRESERVED"
    if retention in {"FIRST_STATE_ADMITTED_WITH_WARNINGS", "COMMITTED_WITH_UNCERTAINTY"}:
        return "QUALIFIED"
    return "DIRECT"


def _agreement_profile(
    data: dict[str, Any],
    latest_by_specialist: dict[str, dict[str, Any]],
    recommendation: str,
) -> dict[str, list[str]]:
    profile = {
        "DIRECT": [], "CONDITIONAL": [], "PROVISIONAL": [],
        "FALLBACK": [], "CONTESTED": [], "PRESERVED": [],
        "QUALIFIED": [], "OPPOSED": [],
    }
    for specialist, candidate in sorted(latest_by_specialist.items()):
        if _candidate_recommendation(candidate) != recommendation:
            profile["OPPOSED"].append(specialist)
        else:
            profile[_agreement_class(data, candidate)].append(specialist)
    return profile


def _agreement_profile_sentence(profile: dict[str, list[str]]) -> str:
    labels = {
        "DIRECT": "direct",
        "CONDITIONAL": "conditional",
        "PROVISIONAL": "provisional",
        "FALLBACK": "fallback within the stated action set",
        "CONTESTED": "internally contested",
        "PRESERVED": "supported by the last validated state after a rejected update",
        "QUALIFIED": "admitted with grounding qualifications",
        "OPPOSED": "opposed",
    }
    parts = [
        f"{labels[kind]}: {', '.join(names)}"
        for kind, names in profile.items() if names
    ]
    return "; ".join(parts)


def _lower_initial(text: str) -> str:
    text = text.strip()
    return text[:1].lower() + text[1:] if text else text


def _dimensional_synthesis_paragraphs(
    data: dict[str, Any],
    *,
    recommendation: str,
    latest_by_specialist: dict[str, dict[str, Any]],
) -> list[str]:
    """Render moral terrain first, keeping orchestration metadata secondary."""
    semantic = data.get("authoritative_semantic_state") or {}
    states = semantic.get("action_dimension_states") or []
    if not states:
        return []
    action_text = _action_text_by_id(data)
    by_action: dict[str, list[dict[str, Any]]] = {}
    for state in states:
        action_id = str(state.get("action_id", "")).strip()
        if action_id:
            by_action.setdefault(action_id, []).append(state)
    substantive = {
        str(state.get("dimension", "")).upper()
        for state in states
        if str(state.get("dimension", "")).upper()
        not in {"FEASIBILITY", "UNKNOWN", "OTHER_PRIMARY_GOOD", "OTHER"}
    }
    if len(substantive) < 2 or len(by_action) < 2:
        return []

    dimension_names = {
        "LIBERTY_AUTONOMY": "liberty and autonomy interests",
        "MATERIAL_FLOOR": "the material floor",
        "OPPORTUNITY_ACCESS": "opportunity and access",
        "BASIC_SECURITY": "basic security",
        "FEASIBILITY": "feasibility",
    }

    def state_phrase(state: dict[str, Any]) -> str:
        dimension = str(state.get("dimension", "UNKNOWN")).upper()
        direction = str(state.get("direction", "UNKNOWN")).upper()
        qualifier = str(state.get("magnitude_or_qualifier", "")).upper()
        subject = _clean_fragment(str(state.get("affected_subject", "")))
        if dimension == "FEASIBILITY":
            return (
                "has uncertain feasibility"
                if direction == "UNCERTAIN" else "has comparatively established feasibility"
            )
        strength = ""
        if qualifier in {"LARGE", "HIGH", "STRONG", "SEVERE", "MODERATE-HIGH"}:
            strength = "substantially "
        verb = {
            "IMPROVES": f"{strength}improves",
            "WORSENS": f"{strength}worsens",
            "PRESERVES": "preserves",
            "MIXED": "has mixed effects on",
            "UNCERTAIN": "has an uncertain effect on",
        }.get(direction, direction.lower().replace("_", " "))
        phrase = f"{verb} {dimension_names.get(dimension, dimension.lower().replace('_', ' '))}"
        if subject and subject != "implementation":
            phrase += f" for {subject}"
        return phrase

    action_sentences = []
    for action_id in sorted(by_action):
        action = _clean_fragment(action_text.get(action_id, action_id))
        ordered = sorted(
            by_action[action_id],
            key=lambda state: {
                "MATERIAL_FLOOR": 0, "LIBERTY_AUTONOMY": 1, "FEASIBILITY": 2,
            }.get(str(state.get("dimension", "")).upper(), 9),
        )
        phrases = [state_phrase(state) for state in ordered]
        if not phrases:
            continue
        if len(phrases) == 1:
            comparison = phrases[0]
        else:
            comparison = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"
        action_sentences.append(f"{action_id} ({action}) {comparison}")
    first = (
        "The Parliament sees the case as a conflict between "
        + " and ".join(
            dimension_names.get(dimension, dimension.lower().replace("_", " "))
            for dimension in sorted(substantive)
        )
        + ". "
        + ". ".join(action_sentences)
        + "."
    )

    interpretations: list[str] = []
    priorities = semantic.get("framework_dimension_priorities") or []
    if any(
        str(priority.get("framework", "")).upper() == "RAWLSIAN"
        and str(priority.get("relation", "")).upper() == "LEXICAL_PRIORITY_OVER"
        for priority in priorities
    ):
        interpretations.append(
            "Rawlsian reasoning gives basic liberty lexical priority over material advantage"
        )
    if any(
        str(priority.get("framework", "")).upper() == "CARE"
        and str(priority.get("relation", "")).upper() == "RELATIONAL_TENSION_WITH"
        for priority in priorities
    ):
        interpretations.append(
            "care reasoning treats dependency relief and domination or autonomy as a context-sensitive tension"
        )
    deon = semantic.get("deontological_assessments") or []
    prohibited = next((
        assessment for assessment in deon
        if str(assessment.get("verdict", "")).upper() == "PROHIBITED"
    ), None)
    if prohibited:
        action_id = str(prohibited.get("canonical_action_id", ""))
        interpretations.append(
            f"deontological reasoning treats {action_id or 'the coercive option'} as prohibited"
        )
    util = latest_by_specialist.get("utilitarian", {})
    if util.get("utilitarian_decision_depends_on_unknown"):
        missing = _clean_fragment(str(util.get("utilitarian_missing_comparison", "")))
        interpretations.append(
            "utilitarian reasoning remains unresolved"
            + (f" because {missing}" if missing else " because the aggregate comparison is not quantified")
        )
    profile = _agreement_profile(data, latest_by_specialist, recommendation)
    if profile["DIRECT"]:
        interpretations.append(
            "Direct support comes from " + " and ".join(profile["DIRECT"])
        )
    qualified: list[str] = []
    for kind, label in (
            ("CONDITIONAL", "conditional"),
            ("PROVISIONAL", "provisional"),
            ("FALLBACK", "only fallback support within the stated action set"),
            ("CONTESTED", "internally contested"),
            ("PRESERVED", "the last validated position after a rejected update"),
            ("QUALIFIED", "admitted with grounding qualifications"),
    ):
        names = profile[kind]
        if not names:
            continue
        joined = names[0] if len(names) == 1 else " and ".join(names)
        verb = "is" if len(names) == 1 else "are"
        qualified.append(f"{joined} {verb} {label}")
    if qualified:
        interpretations.append("; ".join(qualified))
    second = _sentence_or_blank("; ".join(interpretations)) if interpretations else ""

    selected_id = next(
        (action_id for action_id, action in action_text.items() if action == recommendation),
        "",
    )
    selected_states = by_action.get(selected_id, [])
    unresolved_material = any(
        str(state.get("dimension", "")).upper() == "MATERIAL_FLOOR"
        and str(state.get("direction", "")).upper() in {"WORSENS", "UNCERTAIN"}
        for state in selected_states
    )
    synthesis_candidate = next((
        (action_id, action_states)
        for action_id, action_states in by_action.items()
        if action_id != selected_id
        and any(str(state.get("dimension", "")).upper() == "FEASIBILITY"
                and str(state.get("direction", "")).upper() == "UNCERTAIN"
                for state in action_states)
        and any(str(state.get("dimension", "")).upper() == "LIBERTY_AUTONOMY"
                and str(state.get("direction", "")).upper() == "PRESERVES"
                for state in action_states)
        and any(str(state.get("dimension", "")).upper() == "MATERIAL_FLOOR"
                and str(state.get("direction", "")).upper() == "IMPROVES"
                for state in action_states)
    ), None)
    qualified_agreement = any(profile[kind] for kind in (
        "CONDITIONAL", "PROVISIONAL", "FALLBACK", "CONTESTED",
        "PRESERVED", "QUALIFIED",
    ))
    third = (
        f"Within the stated action set, the Parliament provisionally prefers {recommendation}."
        if qualified_agreement
        else f"The Parliament therefore prefers {recommendation}."
    )
    if unresolved_material:
        third += (
            " This preserves an unresolved demand to improve the material position "
            "of vulnerable people through less liberty-destructive means."
        )
    if synthesis_candidate:
        action_id, _ = synthesis_candidate
        third += (
            f" {action_id} expresses that combined aim, but its feasibility remains uncertain."
        )
    return [paragraph for paragraph in (first, second, third) if paragraph]



def _short_action(action: str, limit: int = 72, records: list[dict[str, Any]] | None = None) -> str:
    cleaned = " ".join(str(action or "").split())
    for record in records or []:
        semantic = " ".join(str(record.get("canonical_semantic_action", "")).split())
        short = " ".join(str(record.get("short_label", "")).split())
        if short and cleaned and (
            cleaned == semantic or cleaned == short or cleaned.casefold() == semantic.casefold()
        ):
            return short
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _shorten_actions_in_text(
    text: str, records: list[dict[str, Any]] | None = None,
) -> str:
    """Replace canonical action strings, including truncated prefixes, with short labels."""
    cleaned = str(text or "")
    if not cleaned or not records:
        return cleaned
    ordered = sorted(
        records,
        key=lambda row: len(" ".join(str(row.get("canonical_semantic_action", "")).split())),
        reverse=True,
    )
    for record in ordered:
        semantic = " ".join(str(record.get("canonical_semantic_action", "")).split())
        short = " ".join(str(record.get("short_label", "")).split())
        if not semantic or not short or short == semantic:
            continue
        if semantic in cleaned:
            cleaned = cleaned.replace(semantic, short)
            continue
        for length in range(len(semantic), 47, -1):
            prefix = semantic[:length].rstrip()
            if len(prefix) < 48:
                break
            if prefix in cleaned:
                cleaned = cleaned.replace(prefix, short)
                break
    return cleaned


def _epistemic_status(candidate: dict[str, Any]) -> str:
    from global_workspace.specialist_authority import normalize_specialist_status
    vote_status = str(candidate.get("framework_vote_status", "")).upper()
    if vote_status == "ABSTAIN":
        return "ABSTAINS"
    if vote_status == "ATTENUATED":
        return "CONDITIONAL_SUPPORTS"
    raw = str(candidate.get("adjudication_status", "") or "").strip().upper()
    if raw:
        return normalize_specialist_status(raw)
    assumption = str(candidate.get("assumption_status", "")).upper()
    if assumption == "CONDITIONAL" or candidate.get("utilitarian_decision_depends_on_unknown"):
        return "CONDITIONAL_SUPPORTS"
    if assumption == "NORMATIVELY_CONTESTED":
        return "PROVISIONAL_LEANING"
    if assumption == "UNDERDETERMINED":
        return "PROVISIONAL_LEANING"
    return "SUPPORTS"


_UNESTABLISHED_PROPOSITION_STATUSES = {"HYPOTHETICAL", "UNRESOLVED", "REJECTED"}


def _proposition_index(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("proposition_id")): row
        for row in (data.get("proposition_ledger") or [])
        if isinstance(row, dict) and str(row.get("proposition_id") or "").strip()
    }


def _claim_is_established_component(
    claim: str, ledger: dict[str, dict[str, Any]],
) -> bool:
    from .epistemic_ledger import claim_matches_established
    if claim_matches_established(claim, ledger):
        return True
    normalized = " ".join(str(claim).casefold().split()).strip(" .")
    if not normalized:
        return False
    for row in ledger.values():
        if str(row.get("epistemic_status") or "").upper() not in {
            "ESTABLISHED", "DERIVED", "STIPULATED",
        }:
            continue
        components = {
            " ".join(value.casefold().split()).strip(" .")
            for value in str(row.get("claim") or "").split(";")
            if " ".join(value.split())
        }
        if normalized in components:
            return True
        aliases = row.get("aliases") or []
        if any(
            normalized == " ".join(str(alias).casefold().split()).strip(" .")
            for alias in aliases
        ):
            return True
    return False


def _candidate_unestablished_dependencies(
    data: dict[str, Any], candidate: dict[str, Any], *, critical_only: bool = False,
) -> list[dict[str, Any]]:
    """Return typed dependencies without inferring status from claim wording."""
    ledger = _proposition_index(data)
    critical_order = [
        str(value) for value in candidate.get("decision_critical_proposition_ids", [])
    ]
    critical_ids = set(critical_order)
    supporting_ids = [
        str(value) for value in candidate.get("supporting_proposition_ids", [])
    ]
    ordered_ids = critical_order + [
        value for value in supporting_ids if value not in critical_ids
    ]
    dependencies: list[dict[str, Any]] = []
    seen: set[str] = set()
    for proposition_id in ordered_ids:
        row = ledger.get(proposition_id)
        if row is None:
            continue
        status = str(row.get("epistemic_status") or "UNRESOLVED").upper()
        is_critical = proposition_id in critical_ids
        if status not in _UNESTABLISHED_PROPOSITION_STATUSES:
            continue
        claim = str(row.get("claim") or "")
        if _claim_is_established_component(claim, ledger):
            continue
        from .epistemic_ledger import claim_changes_admitted_outcome_type
        if claim_changes_admitted_outcome_type(claim, ledger, row.get("derived_from") or []):
            continue
        if critical_only and not is_critical:
            continue
        seen.add(proposition_id)
        dependencies.append({**row, "decision_critical": is_critical})

    # Audit findings are retained as a fallback for older traces whose result-level
    # ledger projection predates the audit update.
    for finding in candidate.get("side_premise_audit_findings", []) or []:
        if not isinstance(finding, dict):
            continue
        proposition_id = str(finding.get("proposition_id") or "")
        is_critical = finding.get("decision_critical") is True
        if proposition_id in seen or (critical_only and not is_critical):
            continue
        row = ledger.get(proposition_id, {})
        status = str(row.get("epistemic_status") or "HYPOTHETICAL").upper()
        if status not in _UNESTABLISHED_PROPOSITION_STATUSES:
            continue
        claim = str(row.get("claim") or finding.get("claim") or "").strip()
        if claim and not _claim_is_established_component(claim, ledger):
            from .epistemic_ledger import claim_changes_admitted_outcome_type
            if claim_changes_admitted_outcome_type(
                claim, ledger, row.get("derived_from") or finding.get("derived_from") or [],
            ):
                continue
            dependencies.append({
                **row, "proposition_id": proposition_id, "claim": claim,
                "epistemic_status": status, "decision_critical": is_critical,
            })
            seen.add(proposition_id)
    return dependencies


def _candidate_epistemic_qualification(
    data: dict[str, Any], candidate: dict[str, Any], *, compact: bool = False,
) -> str:
    """Make empirical provenance visible without suppressing specific outcomes."""
    audit_unavailable = (
        str(candidate.get("side_premise_audit_status") or "").upper() == "UNAVAILABLE"
    )
    dependencies = _candidate_unestablished_dependencies(data, candidate)
    critical = [row for row in dependencies if row.get("decision_critical")]
    rows = critical or dependencies
    if not rows:
        if audit_unavailable:
            return (
                "Premise coverage unverified"
                if compact else
                "Independent empirical-premise coverage could not be verified."
            )
        return ""
    rendered = []
    for row in rows[:2]:
        status = str(row.get("epistemic_status") or "UNRESOLVED").lower()
        claim = _public_claim(_clean_fragment(str(row.get("claim") or "")))
        if claim:
            rendered.append(f"{status}: {claim}")
    if not rendered:
        return ""
    # A settled util lean with residual hypothesis still stands as a lean.
    # comparison_complete=false only attenuates; it does not erase the ranking.
    admitted_stands = bool(candidate.get("comparison_complete", True)) or (
        str(candidate.get("specialist", "")).casefold() == "utilitarian"
        and not bool(candidate.get("utilitarian_decision_depends_on_unknown", False))
        and bool(candidate.get("evidence_sufficient_for_action", True))
    )
    prefix = "Reversal boundary" if (
        critical and admitted_stands
    ) else ("Conditional on" if critical else "Uses an unestablished premise")
    if compact:
        qualification = f"{prefix} [{'; '.join(rendered)}]"
        if audit_unavailable:
            qualification += "; other premise coverage unverified"
        return qualification
    noun = "proposition" if len(rendered) == 1 else "propositions"
    if critical and admitted_stands:
        verb = "exceeds" if len(rendered) == 1 else "exceed"
        qualification = (
            f"Admitted ranking stands. Reversal boundary if the unestablished "
            f"{noun} {'; '.join(rendered)} {verb} the admitted welfare margin."
        )
    elif critical:
        qualification = (
            f"This conclusion is conditional on the unestablished {noun}: "
            f"{'; '.join(rendered)}."
        )
    else:
        qualification = f"Additional unestablished {noun}: {'; '.join(rendered)}."
    if audit_unavailable:
        qualification += " Independent coverage of other empirical premises was unavailable."
    return qualification


def _row_is_settled_world_fact(row: dict[str, Any]) -> bool:
    """Matching ESTABLISHED on a chance or FOREGONE row is not an obtained event."""
    status = str(row.get("epistemic_status") or "").upper()
    proposition_type = str(row.get("proposition_type") or "").upper()
    if status not in {"ESTABLISHED", "DERIVED"} or proposition_type != "DESCRIPTIVE":
        return False
    if not str(row.get("claim") or "").strip():
        return False
    polarity = str(row.get("polarity") or "").upper()
    directness = str(row.get("directness") or "").upper()
    if polarity == "FOREGONE" or directness == "FOREGONE":
        return False
    if polarity == "BENEFICIAL" and row.get("obtained_welfare") is False:
        return False
    modality = str(row.get("modality") or "").upper()
    return modality not in {"POSSIBLE", "PROBABILISTIC", "UNKNOWN", "STIPULATED_CONDITIONAL"}


def _row_is_stipulated_world_fact(row: dict[str, Any]) -> bool:
    """Admitted chance or gated facts about an action, including alternatives."""
    status = str(row.get("epistemic_status") or "").upper()
    if status != "STIPULATED":
        return False
    if str(row.get("polarity") or "").upper() == "FOREGONE":
        return False
    if str(row.get("directness") or "").upper() == "FOREGONE":
        return False
    return bool(str(row.get("claim") or "").strip())


_PRIMARY_FACT_KINDS = frozenset({"HEALTH_OUTCOME", "WELFARE_OUTCOME"})


def _row_is_primary_world_fact(row: dict[str, Any]) -> bool:
    """Obtained harm, or a people-welfare benefit, not an intermediate process."""
    polarity = str(row.get("polarity") or "").upper()
    kind = str(row.get("effect_kind") or "").upper()
    if polarity == "ADVERSE":
        return True
    return polarity == "BENEFICIAL" and kind in _PRIMARY_FACT_KINDS


def _order_factual_status_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep every obtained welfare/harm visible before filler process rows.

    A fixed six-row window that kept A0 spare and A1 plant-protect, and dropped
    A1's certain farm submergence, is an incomplete listing.
    """
    primary = [row for row in rows if _row_is_primary_world_fact(row)]
    secondary = [row for row in rows if not _row_is_primary_world_fact(row)]
    return [*primary, *secondary]


def _prefer_protective_relations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    protects = [
        row for row in rows
        if ":PROTECTS:" in str(row.get("proposition_id") or "")
    ]
    if not protects:
        return rows
    covered = {
        str(value)
        for row in protects
        for value in (row.get("support_ids") or [])
        if str(value).startswith("PROP:WORLD:")
    }
    return [
        row for row in rows
        if ":PROTECTS:" in str(row.get("proposition_id") or "")
        or str(row.get("proposition_id") or "") not in covered
    ]


def _factual_status_lines(
    data: dict[str, Any], candidates: list[dict[str, Any]],
) -> list[str]:
    ledger = _proposition_index(data)
    established = _order_factual_status_rows(_prefer_protective_relations([
        row for row in ledger.values()
        if _row_is_settled_world_fact(row)
    ]))
    stipulated = _order_factual_status_rows([
        row for row in ledger.values()
        if _row_is_stipulated_world_fact(row)
        and not re.search(
            r":(?:TEMPORAL|SCOPE|LIKELIHOOD|OVERALL_LIKELIHOOD):",
            str(row.get("proposition_id") or ""),
        )
    ])
    dependents: dict[str, set[str]] = {}
    relevant_ids: list[str] = []
    for candidate in candidates:
        specialist = _framework_display_name(str(candidate.get("specialist") or ""))
        for row in _candidate_unestablished_dependencies(data, candidate):
            proposition_id = str(row.get("proposition_id") or "")
            if not proposition_id:
                continue
            if proposition_id not in relevant_ids:
                relevant_ids.append(proposition_id)
            if specialist:
                dependents.setdefault(proposition_id, set()).add(specialist)
    unestablished = [ledger[value] for value in relevant_ids if value in ledger]
    if not established and not stipulated and not unestablished and not any(
        str(candidate.get("side_premise_audit_status") or "").upper() == "UNAVAILABLE"
        for candidate in candidates
    ):
        return []

    lines = ["## Factual Status", ""]
    if established:
        lines.extend(["**Established or derived from the admitted world model:**", ""])
        for row in established:
            status = str(row.get("epistemic_status") or "").lower()
            lines.append(f"- {_sentence(_public_claim(str(row.get('claim') or '')))} ({status})")
        lines.append("")
    if stipulated:
        lines.extend([
            "**Stipulated in the admitted world (including alternative-action facts):**",
            "",
        ])
        for row in stipulated:
            status = str(row.get("epistemic_status") or "").lower()
            modality = str(row.get("modality") or "").replace("_", " ").lower()
            suffix = f"{status}"
            if modality:
                suffix = f"{status}; {modality}"
            lines.append(
                f"- {_sentence(_public_claim(str(row.get('claim') or '')))} ({suffix})"
            )
        lines.append("")
    if unestablished:
        lines.extend(["**Unestablished premises used in the deliberation:**", ""])
        for row in unestablished[:6]:
            proposition_id = str(row.get("proposition_id") or "")
            status = str(row.get("epistemic_status") or "UNRESOLVED").lower()
            names = sorted(dependents.get(proposition_id, set()))
            dependency = f" Used by {', '.join(names)}." if names else ""
            lines.append(
                f"- {_sentence(_public_claim(str(row.get('claim') or '')))} "
                f"**Status: {status}.**{dependency}"
            )
        lines.extend([
            "",
            "Repeated use can increase a premise's salience, but does not make it established. "
            "Hypothetical is reserved for agent-introduced propositions absent from the admitted world.",
            "",
        ])
    if any(
        str(candidate.get("side_premise_audit_status") or "").upper() == "UNAVAILABLE"
        for candidate in candidates
    ):
        lines.extend([
            "> **Premise-audit warning:** Independent coverage of empirical side "
            "premises could not be verified; affected conclusions remain conditional.",
            "",
        ])
    return lines


def _map_position_label(
    candidate: dict[str, Any], recommendation: str,
    records: list[dict[str, Any]] | None = None,
) -> str:
    explicit = str(candidate.get("recommended_action") or "").strip()
    confidence = float(
        candidate.get("epistemic_confidence")
        if candidate.get("epistemic_confidence") not in (None, -1)
        else (candidate.get("confidence") or 0.0)
    )
    if not bool(candidate.get("schema_valid", True)):
        return "No position"
    if (
        (not explicit or explicit.upper() in {"?", "NONE", "UNRESOLVED", "INCONCLUSIVE", "UNDERDETERMINED", "CONDITIONAL"})
        and confidence <= 0.0
    ):
        return "No position"
    action = _candidate_recommendation(candidate)
    if not action:
        return "No position"
    short = _short_action(action, 48, records=records)
    alignment = str(candidate.get("testimony_alignment", "")).upper()
    if action == recommendation and alignment.startswith("RECONSIDER"):
        return f"{short} (reconsidered)"
    return short


def _framework_display_name(specialist: str) -> str:
    return {
        "utilitarian": "Utilitarian",
        "deontological": "Deontological",
        "virtue": "Virtue",
        "care": "Care",
        "rawlsian": "Rawlsian",
    }.get(str(specialist or "").strip().lower(), str(specialist or "Framework").title())


def _main_contribution(
    data: dict[str, Any],
    candidate: dict[str, Any],
    recommendation: str,
    original_actions: list[str],
    *,
    qualify: bool = False,
    records: list[dict[str, Any]] | None = None,
) -> str:
    status = _epistemic_status(candidate)
    claim = ""
    if status == "PROVISIONAL_LEANING":
        claim = str(candidate.get("investigative_claim") or "").strip()
    landscape = ""
    if candidate.get("landscape_semantic_valid", True):
        cases = candidate.get("landscape_cases") or {}
        action = _candidate_recommendation(candidate) or recommendation
        landscape = _clean_fragment(str(cases.get(action, "") or ""))
    if not claim:
        claim = _support_reason(data, candidate, recommendation, original_actions)
    if landscape and landscape.casefold() not in claim.casefold():
        claim = " ".join(part for part in (claim, landscape) if part)
    if not claim:
        claim = str(candidate.get("rationale") or candidate.get("decision_rule") or "").strip()
    claim = _public_claim(_clean_fragment(_shorten_actions_in_text(claim, records)))
    # Authority invariant: never upgrade provisional language into categorical
    # "REQUIRED" / "perfect duties require" slogans in the public map.
    if status in {"PROVISIONAL_LEANING", "CONTESTED_NO_LEANING"}:
        claim = re.sub(
            r"\b(?:REQUIRED|PROHIBITED|perfect duties require)\b",
            "provisionally favored",
            claim,
            flags=re.IGNORECASE,
        )
    claim = claim or "No compact contribution recorded"
    qualification = _candidate_epistemic_qualification(
        data, candidate, compact=True,
    ) if qualify else ""
    return f"{qualification}: {claim}" if qualification else claim


def _recommendation_headline(
    status: str, action: str, records: list[dict[str, Any]] | None = None,
) -> str:
    short = _short_action(action, records=records) or "none"
    if status == "GOVERNED_RECOMMENDATION":
        return f"**{short}.**"
    if status == "CONTESTED_RECOMMENDATION":
        return f"**{short} — presently favored, but contested.**"
    if status == "PROVISIONAL_RECOMMENDATION_DEGRADED_WORLD_STATE":
        return f"**{short} — provisional because direct world facts were quarantined.**"
    if status in {"UNRESOLVED", "UNDERDETERMINED", "INCONCLUSIVE"}:
        return f"**No conclusive recommendation yet.** Current plurality: {short}."
    return f"**{short}.**"


def _convergence_label(data: dict[str, Any]) -> str:
    halted = str(data.get("halted_by") or "").strip().lower()
    termination = data.get("termination_assessment") or {}
    if termination.get("resource_censored") or halted == "cycle_budget":
        return "Incomplete (cycle budget reached)"
    if halted == "convergence":
        return "Reached"
    if halted:
        return f"Stopped ({halted.replace('_', ' ')})"
    return "Incomplete"


def _primary_investigative_focus(
    latest_by_specialist: dict[str, dict[str, Any]],
    data: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    ranked = sorted(
        (
            candidate for candidate in latest_by_specialist.values()
            if candidate.get("schema_valid", True)
        ),
        key=lambda c: (
            1 if c.get("reopen_eligible") else 0,
            float(c.get("investigative_priority", 0) or 0),
            1 if _epistemic_status(c) in {
                "PROVISIONAL_LEANING", "CONTESTED_NO_LEANING", "CONDITIONAL_SUPPORTS",
            } else 0,
        ),
        reverse=True,
    )
    for candidate in ranked:
        claim = " ".join(str(
            candidate.get("investigative_claim")
            or candidate.get("reopen_reason")
            or candidate.get("unsupported_assumption")
            or ""
        ).split())
        if claim and (
            candidate.get("reopen_eligible")
            or float(candidate.get("investigative_priority", 0) or 0) >= 0.40
            or _epistemic_status(candidate) in {
                "PROVISIONAL_LEANING", "CONTESTED_NO_LEANING",
            }
        ):
            return candidate, _public_claim(claim)
    # Fall back to deliberative problem-state questions.
    state = data.get("deliberative_problem_state") or {}
    for item in state.get("unresolved_questions") or []:
        question = " ".join(str(item.get("question") or "").split())
        if question:
            return None, _public_claim(question)
    return None, ""


def _decision_boundary_lines(data: dict[str, Any], candidates: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    semantic = data.get("authoritative_semantic_state") or {}
    for boundary in semantic.get("factual_reversal_boundaries") or []:
        predicate = _readable_condition(str(boundary.get("predicate", "")))
        if predicate:
            clause = re.sub(r"^(?:if|when)\s+", "", predicate, flags=re.I)
            rendered = f"the recommendation changes if {_lower_initial(clause)}"
            if rendered not in lines:
                lines.append(rendered)
    for candidate in candidates:
        unresolved = str(candidate.get("unresolved", "")).upper()
        if unresolved not in {"DECISION_BOUNDARY"}:
            # Still surface factual thresholds that look like boundaries.
            factual = _readable_condition(str(candidate.get("factual_reversal_threshold", "")))
            if factual and factual.upper() != "NONE":
                clause = re.sub(r"^(?:if|when)\s+", "", factual, flags=re.I)
                rendered = f"the recommendation changes if {_lower_initial(clause)}"
                if rendered not in lines:
                    lines.append(rendered)
            continue
        for field in (
            "factual_reversal_threshold",
            "reversal_condition",
            "unsupported_assumption",
            "boundary_switch_condition",
        ):
            text = _readable_condition(str(candidate.get(field, "")))
            if text and text.upper() != "NONE":
                clause = re.sub(r"^(?:if|when)\s+", "", text, flags=re.I)
                rendered = f"the recommendation changes if {_lower_initial(clause)}"
                if rendered not in lines:
                    lines.append(rendered)
                break
    return lines[:4]


def _change_condition_lines(
    data: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[str]:
    lines = list(_decision_boundary_lines(data, candidates))
    for candidate in candidates:
        assumption = _readable_condition(str(candidate.get("unsupported_assumption", "")))
        if (
            assumption
            and str(candidate.get("assumption_status", "")).upper() in {
                "CONDITIONAL", "UNDERDETERMINED", "NORMATIVELY_CONTESTED",
            }
        ):
            rendered = _lower_initial(assumption)
            if rendered and rendered not in lines:
                lines.append(rendered)
        reversal = _readable_condition(str(candidate.get("reversal_condition", "")))
        if reversal and reversal.upper() != "NONE":
            clause = re.sub(r"^(?:if|when)\s+", "", reversal, flags=re.I)
            rendered = _lower_initial(clause)
            if rendered and rendered not in lines:
                lines.append(rendered)
        review = str(candidate.get("reversal_review_response", "")).upper()
        justification = _readable_condition(
            str(candidate.get("reversal_review_justification", ""))
        )
        if review in {"ACCEPT", "REVISE"} and justification:
            rendered = _lower_initial(justification)
            if rendered and rendered not in lines:
                lines.append(rendered)
    for condition in data.get("reopen_conditions") or []:
        code = str(condition).strip().upper()
        if code in {"NONE", "RETRY_MODEL_CALL"}:
            continue
        if code == "DECISION_BOUNDARY":
            continue
        if code == "VERIFY_FACTS":
            label = "publication is shown to cause substantially greater immediate harm than currently stipulated"
        elif code in {"NORMATIVE_ADJUDICATION", "RESOLVE_NORMATIVE_TENSION"}:
            label = (
                "normative adjudication establishes a strict duty incompatible "
                "with the current recommendation"
            )
        elif code == "CHECK_FEASIBILITY":
            label = "a feasibility review shows the recommended action cannot be carried out as stipulated"
        elif code == "ACTION_SET_ADEQUACY":
            label = "a feasible alternative action avoids the central conflict"
        else:
            readable = _readable_condition(code)
            if not readable or readable == code.replace("_", " ").casefold():
                continue
            label = _lower_initial(readable)
        if label and label not in lines:
            lines.append(label)
    # Normative reversal thresholds as reopen conditions (not factual unknowns).
    for candidate in candidates:
        normative = _readable_condition(str(candidate.get("normative_reversal_threshold", "")))
        if normative and normative.upper() != "NONE":
            clause = re.sub(r"^(?:if|when)\s+", "", normative, flags=re.I)
            rendered = _lower_initial(clause)
            if rendered and rendered not in lines:
                lines.append(rendered)
    return lines[:5]


def _alternative_proposals(data: dict[str, Any], recommendation: str) -> list[dict[str, Any]]:
    proposals = []
    original = {
        " ".join(str(action).split()).casefold()
        for action in (data.get("actions") or [])
    }
    recommendation_key = " ".join(str(recommendation or "").split()).casefold()
    for proposal in data.get("synthesis_proposals") or []:
        action = " ".join(str(proposal.get("action") or "").split())
        if not action:
            continue
        key = action.casefold()
        if key == recommendation_key:
            continue
        # Stipulated original options are compared in the deliberation map, not
        # re-listed as "discovered alternatives".
        if key in original and not proposal.get("accepted"):
            continue
        status = str(proposal.get("promotion_status") or "").upper()
        accepted = bool(proposal.get("accepted"))
        proposals.append({
            "action": action,
            "accepted": accepted,
            "promotion_status": status,
            "rationale": " ".join(str(proposal.get("rationale") or "").split()),
            "is_original": key in original,
        })
    return proposals


def _governing_claim_text(
    data: dict[str, Any],
    final: dict[str, Any],
    supporters: list[dict[str, Any]],
    recommendation: str,
) -> str:
    governing = final.get("governing_claim") or {}
    if isinstance(governing, dict) and governing:
        if governing.get("governing_eligible") is False:
            return "NONE"
        status = _epistemic_status(governing)
        if status in {"PROVISIONAL_LEANING", "CONTESTED_NO_LEANING"}:
            return "NONE"
        rule = _sentence(_public_claim(str(governing.get("decision_rule") or "")))
        qualification = _candidate_epistemic_qualification(
            data, governing, compact=True,
        )
        if rule and (status == "CONDITIONAL_SUPPORTS" or qualification):
            # Conditional governing claims must keep their factual condition attached.
            return f"{qualification}: {rule}" if qualification else rule
        if rule and status == "SUPPORTS":
            return rule
    for supporter in supporters:
        if supporter.get("governing_eligible") is False:
            continue
        if _epistemic_status(supporter) in {"PROVISIONAL_LEANING", "CONTESTED_NO_LEANING"}:
            continue
        if _candidate_recommendation(supporter) != recommendation:
            continue
        rule = _sentence(_public_claim(str(supporter.get("decision_rule") or "")))
        if rule:
            qualification = _candidate_epistemic_qualification(
                data, supporter, compact=True,
            )
            return f"{qualification}: {rule}" if qualification else rule
    return "NONE"


def render_decision_brief(result: Any) -> str:
    """Default user-facing answer: decision brief + compact deliberation map.

    Full internal machinery remains in the saved JSON trace. This renderer must
    never increase epistemic or normative authority beyond upstream state.
    """
    data = _data(result)
    action_records = list(data.get("canonical_action_records") or [])

    def short_action(action: str, limit: int = 72) -> str:
        return _short_action(action, limit=limit, records=action_records)

    recorded_cycles = data.get("cycles", []) or []
    cycles = [cycle for cycle in recorded_cycles if not cycle.get("is_hypothetical")]
    if not cycles:
        return (
            "# Ethical Parliament Judgment\n\n"
            "## Recommendation\n\n"
            "No reliable deliberative judgment was produced.\n"
        )

    from global_workspace.specialist_authority import (
        CONTESTED_RECOMMENDATION,
        DEGRADED_WORLD_STATE_RECOMMENDATION,
        GOVERNED_RECOMMENDATION,
        normalize_judgment_status,
    )

    final = _judgment_cycle(cycles)
    all_candidates = [
        candidate for cycle in cycles for candidate in cycle.get("candidates", [])
        if candidate.get("schema_valid")
    ]
    final_candidates = [
        candidate for candidate in final.get("candidates", [])
        if candidate.get("schema_valid")
    ]
    original_actions = _original_actions(data)
    status = normalize_judgment_status(data.get("judgment_status", "UNRESOLVED"))
    selected = str(data.get("selected_action") or "")
    plurality = str(data.get("current_plurality") or "")
    actionable = status in {
        GOVERNED_RECOMMENDATION, CONTESTED_RECOMMENDATION,
        DEGRADED_WORLD_STATE_RECOMMENDATION,
    }
    recommendation = selected if actionable and selected not in {
        "UNRESOLVED", "INCONCLUSIVE", "UNDERDETERMINED", "CONDITIONAL", "",
    } else (plurality or selected)

    latest_by_specialist: dict[str, dict[str, Any]] = {}
    for candidate in all_candidates:
        name = str(candidate.get("specialist") or "")
        if name:
            latest_by_specialist[name] = candidate

    framework_order = {
        "utilitarian": 0, "deontological": 1, "virtue": 2, "care": 3, "rawlsian": 4,
    }
    ordered_specialists = sorted(
        latest_by_specialist.items(),
        key=lambda item: framework_order.get(item[0], 9),
    )

    policy = final.get("policy") or {}
    policy_support = float(
        policy.get(recommendation, data.get("confidence", 0.0)) or 0.0
    )
    epistemic = float(data.get("epistemic_confidence", data.get("confidence", 0.0)) or 0.0)

    supporting = [
        candidate for _, candidate in ordered_specialists
        if _candidate_recommendation(candidate) == recommendation
    ]
    supporting_names = [
        _framework_display_name(name)
        for name, candidate in ordered_specialists
        if _candidate_recommendation(candidate) == recommendation
    ]

    lines: list[str] = ["# Ethical Parliament Judgment", "", "## Recommendation", ""]
    lines.append(_recommendation_headline(status, recommendation, records=action_records))
    lines.append("")

    if actionable and supporting_names:
        lean = (
            "presently favored, but contested"
            if status == CONTESTED_RECOMMENDATION
            else "favored"
        )
        if supporting_names:
            if len(supporting_names) == 1:
                support_clause = f"{supporting_names[0]} supports this action"
            else:
                support_clause = (
                    f"{', '.join(supporting_names[:-1])} and "
                    f"{supporting_names[-1]} support this action"
                )
            intro = (
                f"The Parliament broadly favors {short_action(recommendation)}. "
                f"{support_clause}"
            )
        provisional = [
            c for c in supporting
            if _epistemic_status(c) in {"PROVISIONAL_LEANING", "CONDITIONAL_SUPPORTS"}
        ]
        if status == CONTESTED_RECOMMENDATION or provisional:
            intro += (
                ", but the deliberation did not fully converge and at least one "
                "important framework conflict or condition remains unresolved"
            )
        else:
            intro += " after comparing the stipulated options"
        lines.append(_sentence(intro))
    elif status in {"UNRESOLVED", "UNDERDETERMINED", "INCONCLUSIVE"}:
        lines.append(
            "No stable governing recommendation is available yet. The values below "
            "describe the current plurality, not a final settled judgment."
        )
    else:
        lines.append(
            "The Parliament recorded a recommendation, but supporting framework "
            "detail was incomplete in this run."
        )

    lines.extend([
        "",
        f"**Policy support:** {policy_support:.2f}",
        f"**Epistemic confidence:** {epistemic:.2f}",
        f"**Judgment status:** {status}",
        f"**Convergence:** {_convergence_label(data)}",
    ])
    presentation_mapping = list(data.get("presentation_action_mapping") or [])
    remapped = any(
        str(entry.get("canonical_action_id") or "")
        != f"A{entry.get('source_position')}"
        for entry in presentation_mapping
    )
    if presentation_mapping and remapped:
        lines.extend([
            "",
            "## Action label mapping",
            "",
            "Canonical IDs are stable internal identifiers; they do not preserve "
            "the original presentation order.",
            "",
        ])
        for entry in presentation_mapping:
            source_label = str(entry.get("source_label") or "Presented option")
            canonical_id = str(entry.get("canonical_action_id") or "")
            action = str(
                entry.get("canonical_action") or entry.get("source_action") or ""
            )
            lines.append(
                f"- {source_label} → canonical {canonical_id}: {short_action(action)}"
            )
    if status == DEGRADED_WORLD_STATE_RECOMMENDATION:
        quarantined = (
            (((data.get("action_source_grounding") or {}).get("world_model") or {})
             .get("admission") or {}).get("quarantined_effects") or []
        )
        lines.extend([
            "",
            "> **World-state warning:** The user elected to continue after "
            f"{len(quarantined)} contradictory direct effect(s) were quarantined. "
            "The recommendation does not treat those effects as established facts.",
        ])
    termination = data.get("termination_assessment") or {}
    if termination.get("resource_censored") or str(data.get("halted_by") or "") == "cycle_budget":
        lines.extend([
            "",
            "The cycle budget was reached before deliberation naturally converged.",
        ])

    factual_lines = _factual_status_lines(data, final_candidates)
    if factual_lines:
        lines.extend(["", *factual_lines])

    # Why favored
    lines.extend(["", "## Why the Parliament currently favors this action", ""])
    why_added = False
    for name, candidate in ordered_specialists:
        if _candidate_recommendation(candidate) != recommendation:
            continue
        reason = _main_contribution(
            data, candidate, recommendation, original_actions, records=action_records,
        )
        if not reason:
            continue
        label = _framework_display_name(name)
        status_code = _epistemic_status(candidate)
        if status_code == "PROVISIONAL_LEANING":
            lead = f"{label} provisionally favors this action"
        elif status_code == "CONDITIONAL_SUPPORTS":
            lead = f"{label} conditionally supports this action"
        elif status_code == "CONTESTED_NO_LEANING":
            continue
        else:
            lead = f"{label} favors this action"
        lines.append(_sentence(f"{lead}: {_lower_initial(reason)}"))
        qualification = _candidate_epistemic_qualification(data, candidate)
        if qualification:
            lines.append(_sentence(qualification))
        lines.append("")
        why_added = True
    if not why_added:
        shape = summarize_problem_shape_paragraphs(data)
        if shape:
            lines.append(_sentence(_public_claim(shape[0])))
        else:
            lines.append(
                "Framework-specific supporting reasons were not available in a "
                "form safe to summarize without overstating authority."
            )
        lines.append("")

    # Most important unresolved issue
    focus_candidate, focus_text = _primary_investigative_focus(latest_by_specialist, data)
    focus_text = _shorten_actions_in_text(focus_text, action_records)
    lines.extend(["## Most important unresolved issue", ""])
    if focus_text:
        if focus_candidate is not None:
            label = _framework_display_name(str(focus_candidate.get("specialist", "")))
            status_code = _epistemic_status(focus_candidate)
            action = _candidate_recommendation(focus_candidate)
            if status_code == "PROVISIONAL_LEANING" and action == recommendation:
                lines.append(
                    f"**{label} reasoning currently leans toward "
                    f"{short_action(action, 40)} but remains unadjudicated.**"
                )
            elif status_code == "CONTESTED_NO_LEANING":
                lines.append(
                    f"**{label} remains contested without a current preference.**"
                )
            else:
                lines.append(
                    f"**Open question requiring further review ({label}).**"
                )
            lines.append("")
        lines.append(_sentence(focus_text))
        # Surface internal conflict horns when present, without IDs.
        conflicts = []
        if focus_candidate is not None:
            conflicts = list(focus_candidate.get("framework_internal_conflicts") or [])
        if len(conflicts) >= 2:
            lines.extend([
                "",
                "The unresolved conflict is:",
                "",
                f"**{_public_claim(conflicts[0])}**",
                "versus",
                f"**{_public_claim(conflicts[1])}**",
            ])
        elif conflicts:
            lines.extend(["", f"**{_public_claim(conflicts[0])}**"])
        lines.append("")
        lines.append(
            "An investigative claim may be the most salient issue in the workspace "
            "without becoming the governing justification for the recommendation."
        )
    else:
        residue = data.get("moral_residue") or []
        if residue:
            lines.append(
                "Moral residue remains, but no single reopen-eligible investigative "
                "focus was typed for this run."
            )
        else:
            lines.append("No primary investigative focus remains open in this run.")
    lines.append("")

    from .epistemic_ledger import claim_changes_admitted_outcome_type

    ledger = _proposition_index(data)
    shared_dependencies = [
        dependency for dependency in list(data.get("shared_unresolved_dependencies") or [])
        if isinstance(dependency, dict)
        and not _claim_is_established_component(str(dependency.get("claim") or ""), ledger)
        and not claim_changes_admitted_outcome_type(
            str(dependency.get("claim") or ""),
            ledger,
            dependency.get("derived_from") or [],
        )
    ]
    if shared_dependencies:
        lines.extend(["## Shared epistemic dependencies", ""])
        for dependency in shared_dependencies[:3]:
            claim = _sentence(_public_claim(str(dependency.get("claim") or "")))
            specialists = [
                _framework_display_name(str(name))
                for name in dependency.get("dependent_specialists") or []
            ]
            status_label = str(
                dependency.get("epistemic_status") or "UNRESOLVED"
            ).replace("_", " ").lower()
            if len(specialists) > 1:
                agent_text = ", ".join(specialists[:-1]) + f" and {specialists[-1]}"
                lines.append(
                    f"- **{claim}** Status: {status_label}. The positions of "
                    f"{agent_text} depend materially on this same proposition; "
                    "their recurrence does not provide independent factual support."
                )
            elif specialists:
                lines.append(
                    f"- **{claim}** Status: {status_label}. "
                    f"{specialists[0]}'s position depends materially on it."
                )
        lines.append("")

    # What could change
    change_lines = _change_condition_lines(data, all_candidates)
    lines.extend(["## What could change the judgment", ""])
    if change_lines:
        lines.append("The recommendation should be reconsidered if:")
        lines.append("")
        for index, condition in enumerate(change_lines):
            suffix = ";" if index < len(change_lines) - 1 else "."
            if index == len(change_lines) - 1 and len(change_lines) > 1:
                lines.append(f"- or {_lower_initial(condition).rstrip('.')}{suffix}")
            else:
                lines.append(f"- {_lower_initial(condition).rstrip('.')}{suffix}")
        lines.extend([
            "",
            "Where a condition is a hypothetical threshold rather than an unknown "
            "fact, it is a **decision boundary**, not factual uncertainty.",
        ])
    else:
        lines.append(
            "No compact decision boundary or reopen condition was safe to present "
            "without overstating what the run established."
        )
    lines.append("")

    # Alternatives
    alternatives = _alternative_proposals(data, recommendation)
    if alternatives:
        lines.extend(["## Alternative action discovered", ""])
        for proposal in alternatives[:2]:
            lines.append("The Parliament identified a possible alternative:")
            lines.append("")
            lines.append(f"**{short_action(proposal['action'])}.**")
            lines.append("")
            if proposal.get("rationale"):
                lines.append(_sentence(_public_claim(proposal["rationale"])))
                lines.append("")
            if proposal.get("accepted") and proposal.get("promotion_status") == "PROMOTED":
                lines.append("**Status:** Reviewed and admitted as a live option.")
            elif (
                proposal.get("accepted")
                and proposal.get("promotion_status") == "ADMISSIBLE"
                and any(
                    isinstance(review, dict)
                    and review.get("valid") is True
                    and str(review.get("framework_status", "")).upper()
                    != "UNDERDETERMINED"
                    for review in dict(
                        proposal.get("framework_reviews") or {}
                    ).values()
                )
            ):
                lines.append(
                    "**Status:** Reviewed by specialists — retained as a candidate "
                    "(not promoted into the live action set)."
                )
            else:
                lines.append(
                    "**Status:** Candidate only — no completed substantive specialist "
                    "review during this run."
                )
                lines.append("")
                lines.append(
                    "Do not replace the stipulated recommendation with this action "
                    "unless proposal review is successfully completed."
                )
            lines.append("")

    # Deliberation map
    lines.extend([
        "## Deliberation Map",
        "",
        "| Framework | Current position | Vote admission | Epistemic status | Main contribution |",
        "|---|---|---|---|---|",
    ])
    for name, candidate in ordered_specialists:
        position = _map_position_label(candidate, recommendation, records=action_records)
        vote_status = str(candidate.get("framework_vote_status", "NOT_APPLICABLE")).upper()
        vote_reason = _sentence(_public_claim(str(
            candidate.get("framework_vote_reason", "")
        )))
        vote_admission = vote_status.replace("_", " ")
        if vote_status in {"ABSTAIN", "ATTENUATED"} and vote_reason:
            vote_admission += f": {vote_reason}"
        vote_admission = vote_admission.replace("|", "/")
        if len(vote_admission) > 110:
            vote_admission = vote_admission[:107].rstrip() + "..."
        epistemic_status = _epistemic_status(candidate)
        # Table may show RECONSIDERED_SUPPORT when the specialist changed from baseline.
        alignment = str(candidate.get("testimony_alignment", "")).upper()
        if (
            epistemic_status == "SUPPORTS"
            and _candidate_recommendation(candidate) == recommendation
            and "RECONSIDER" in alignment
        ):
            epistemic_status = "RECONSIDERED_SUPPORT"
        contribution = _main_contribution(
            data, candidate, recommendation, original_actions, qualify=True,
            records=action_records,
        ).replace("|", "/")
        if len(contribution) > 140:
            contribution = contribution[:137].rstrip() + "..."
        lines.append(
            f"| {_framework_display_name(name)} | {position} | {vote_admission} | "
            f"{epistemic_status} | {contribution} |"
        )
    lines.append("")

    # Governing and investigative state
    governing_text = _governing_claim_text(data, final, supporting, recommendation)
    if status == CONTESTED_RECOMMENDATION and governing_text != "NONE":
        # Contested with under-attack still shows the claim but marks it.
        if data.get("governing_justification_status") == "UNDER_ATTACK":
            governing_text = (
                governing_text.rstrip(".")
                + " (governing justification currently under attack)"
            )
        elif data.get("governing_justification_status") == "CONTESTED":
            governing_text = (
                governing_text.rstrip(".")
                + " (available governing claim remains contested)"
            )
    if status == CONTESTED_RECOMMENDATION and not supporting:
        governing_text = "NONE"
    if status == CONTESTED_RECOMMENDATION and governing_text == "NONE":
        pass  # expected
    elif status == GOVERNED_RECOMMENDATION and governing_text == "NONE":
        # Prefer compressed_rule only when it does not invent categorical force.
        compressed = _sentence(_public_claim(str(data.get("compressed_rule") or "")))
        if compressed and not re.search(
            r"\b(?:REQUIRED|PROHIBITED|perfect duties require)\b",
            compressed,
            re.I,
        ):
            governing_text = compressed

    lines.extend([
        "## Governing and Investigative State",
        "",
        f"**Policy leader:** {short_action(recommendation) or 'NONE'}",
        "",
        f"**Governing claim:** {governing_text}",
        "",
    ])
    if focus_text:
        lines.append(f"**Primary investigative focus:** {_sentence(focus_text)}")
    else:
        lines.append("**Primary investigative focus:** NONE")
    lines.extend([
        "",
        "An investigative claim may be the most salient issue in the workspace "
        "without becoming the governing justification for the recommendation.",
        "",
        "## Bottom Line",
        "",
    ])
    if actionable:
        if status == CONTESTED_RECOMMENDATION:
            lines.append(
                f"**Within the stipulated action set, {short_action(recommendation)} "
                "is currently the best-supported action.**"
            )
            lines.append("")
            lines.append(
                "This is a broad policy plurality rather than complete normative "
                "convergence. Unresolved conflicts remain genuine moral residue and "
                "should stay visible in the judgment rather than being compressed away."
            )
        else:
            lines.append(
                f"**Within the stipulated action set, {short_action(recommendation)} "
                "is the Parliament's governing recommendation.**"
            )
            lines.append("")
            lines.append(
                "Frameworks that completed their own derivations may supply the "
                "governing rationale; unadjudicated conflict is retained as residue "
                "rather than treated as a settled maxim."
            )
    else:
        lines.append(
            "**No governing recommendation is safe to present from this run.**"
        )
        lines.append("")
        lines.append(
            "The detailed trace explains how the Parliament reached this unresolved state."
        )
    lines.append("")
    return "\n".join(lines)


def render_public_judgment(result: Any) -> str:
    """Default user-facing result: readable decision brief + deliberation map."""
    return render_decision_brief(result)
