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
                    f"A decision-critical uncertainty remains unresolved: {variable}"
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
                    f"The authoritative state keeps one decision-critical variable open: {label}"
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
    if assumption == "NORMATIVELY_CONTESTED" or unresolved == "RESOLVE_NORMATIVE_TENSION":
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
    agreement_profile = _agreement_profile(
        data, latest_by_specialist, recommendation
    )
    direct_supporters = [
        candidate for candidate in supporters
        if _agreement_class(data, candidate) == "DIRECT"
    ]
    qualified_supporters = [
        candidate for candidate in supporters
        if _agreement_class(data, candidate) != "DIRECT"
    ]

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
    final_policy_support = float(
        (final.get("policy") or {}).get(plurality, data.get("confidence", 0))
    )
    lines.append(f"Final policy support: {final_policy_support:.2f} (aggregate policy score)")
    agreement_summary = _agreement_profile_sentence(agreement_profile)
    if agreement_summary:
        lines.append(f"Agreement profile: {agreement_summary}.")
    lines.append(
        f"Epistemic confidence: {float(data.get('epistemic_confidence', 0)):.2f} "
        "(likelihood the judgment survives further factual inquiry and scrutiny)"
    )
    synthesis_paragraphs = _dimensional_synthesis_paragraphs(
        data,
        recommendation=recommendation,
        latest_by_specialist=latest_by_specialist,
    )
    if synthesis_paragraphs:
        lines.extend(["", "Synthesis:", ""])
        for paragraph in synthesis_paragraphs:
            lines.extend([paragraph, ""])
        lines.append("Deliberation details:")
    final_winner = final.get("winner")
    if isinstance(final_winner, dict):
        lines.append(f"Final winning constraint: {final_winner.get('constraint', 'unknown')}")
    else:
        lines.append(
            "Deliberative winner: NONE "
            f"(system status: {final.get('system_error', 'UNKNOWN')})"
        )
    received_broadcast = final.get("received_broadcast") or final.get("broadcast") or {}
    lines.append(f"Previous broadcast context: {received_broadcast.get('constraint', 'unknown')}")
    lines.append(f"Last emitted broadcast: {final.get('broadcast', {}).get('constraint', 'unknown')}")
    termination = data.get("termination_assessment") or {}
    if termination:
        if termination.get("resource_censored"):
            lines.append(
                "Deliberation status: resource-censored—the recommendation records "
                "the position when observation stopped, not natural convergence."
            )
        else:
            lines.append(
                "Deliberation status: "
                + str(termination.get("termination_type", "unknown")).lower().replace("_", " ")
                + "."
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
        visibility_responses = [
            candidate
            for cycle in cycles
            if (cycle.get("received_broadcast") or cycle.get("broadcast") or {}).get("constraint")
            == "VISIBILITY_AUDIT"
            for candidate in cycle.get("candidates", [])
            if candidate.get("visibility_response") not in {None, "", "NOT_TESTED"}
        ]
        if visibility_responses:
            lines.append("- Specialist reconsideration:")
            for candidate in visibility_responses:
                lines.append(
                    f"  - {candidate.get('specialist', 'specialist')}: "
                    f"{candidate.get('visibility_response')} proposition; harm estimate "
                    f"{str(candidate.get('visibility_harm_revision', 'UNCHANGED')).lower()}; "
                    f"magnitude {str(candidate.get('visibility_magnitude_status', 'UNKNOWN')).lower()} — "
                    f"{_sentence(candidate.get('visibility_justification', ''))}"
                )

    semantic_state = data.get("authoritative_semantic_state") or {}
    problem_shape = summarize_problem_shape_paragraphs(data)
    if problem_shape and not synthesis_paragraphs:
        lines.extend(["", "Problem shape:"])
        for paragraph in problem_shape[:4]:
            lines.append(f"- {paragraph}")

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

    # Framework-specific prose is rendered only from committed graph state.
    # Raw delegate claims that failed or never reached the transaction boundary
    # remain available in the trace but cannot be promoted into this summary.
    rawls_positions = semantic_state.get("rawlsian_positions") or []
    util_consequences = semantic_state.get("utilitarian_consequences") or []
    deon_assessments = semantic_state.get("deontological_assessments") or []
    virtue_assessments = semantic_state.get("virtue_assessments") or []
    if rawls_positions or util_consequences or deon_assessments or virtue_assessments:
        action_text = _action_text_by_id(data)
        selected_id = next(
            (action_id for action_id, text in action_text.items() if text == recommendation),
            "",
        )
        lines.extend(["", "Committed framework checks:"])
    if rawls_positions:
        ordered_positions = sorted(
            rawls_positions,
            key=lambda item: item.get("canonical_action_id") != selected_id,
        )
        for position in ordered_positions[:2]:
            action_id = str(position.get("canonical_action_id", ""))
            rival_id = str(position.get("compared_to_action_id", ""))
            action = action_text.get(action_id, action_id or "the action")
            rival = action_text.get(rival_id, rival_id or "the alternative")
            effect = str(position.get("effect", "UNCERTAIN")).lower()
            subject = str(
                position.get("subject_node_id")
                or position.get("group_node_id")
                or position.get("affected_subject")
                or position.get("subject")
                or "the affected subject"
            )
            subject_label = _graph_node_label(
                data, subject, "the affected subject"
            )
            possessive_subject = (
                f"{subject_label}'" if subject_label.casefold().endswith("s")
                else f"{subject_label}'s"
            )
            dimension = str(position.get("dimension_node_id", "UNKNOWN")).split(":")[-1]
            additional_dimensions = [
                str(value).split(":")[-1]
                for value in position.get("additional_dimensions", [])
                if str(value).strip()
            ]
            epistemic_status = str(position.get("epistemic_status", ""))
            status_note = (
                "grounded"
                if epistemic_status == "GROUNDED"
                else "mixed comparison"
                if epistemic_status == "MIXED_COMPARISON"
                else "direction unresolved"
            )
            bundle_note = (
                f"; additional dimensions: {', '.join(dim.lower().replace('_', ' ') for dim in additional_dimensions)}"
                if additional_dimensions else ""
            )
            lines.append(
                f"- Rawlsian — {action}: {effect} {possessive_subject} "
                f"{dimension.lower().replace('_', ' ')} relative to {rival} "
                f"({status_note}{bundle_note})."
            )
    if util_consequences:
        ordered_consequences = sorted(
            util_consequences,
            key=lambda item: item.get("canonical_action_id") != selected_id,
        )
        for consequence in ordered_consequences[:2]:
            action_id = str(consequence.get("canonical_action_id", ""))
            action = action_text.get(action_id, action_id or "the action")
            direction = str(consequence.get("direction", "UNKNOWN")).lower()
            scope = _graph_node_label(
                data,
                str(consequence.get("scope_node_id", "")),
                "the affected population",
            )
            status_note = str(
                consequence.get("epistemic_status", "UNKNOWN")
            ).lower().replace("_", " ")
            lines.append(
                f"- Utilitarian — {action}: {direction} to {scope}: "
                f"{_clean_fragment(str(consequence.get('outcome', '')))} "
                f"({status_note})."
            )
    if deon_assessments:
        ordered_assessments = sorted(
            deon_assessments,
            key=lambda item: item.get("canonical_action_id") != selected_id,
        )
        for assessment in ordered_assessments[:2]:
            action_id = str(assessment.get("canonical_action_id", ""))
            action = action_text.get(action_id, action_id or "the action")
            norm = _graph_node_label(
                data, str(assessment.get("norm_node_id", "")), "the relevant norm"
            )
            party = _graph_node_label(
                data, str(assessment.get("party_node_id", "")), "the protected party"
            )
            verdict = str(assessment.get("verdict", "UNCERTAIN")).lower()
            relation = str(assessment.get("relation", "UNCERTAIN")).lower()
            lines.append(
                f"- Deontological — {action}: {verdict}; {relation} {norm} "
                f"for {party}."
            )
    if virtue_assessments:
        ordered_virtue = sorted(
            virtue_assessments,
            key=lambda item: item.get("canonical_action_id") != selected_id,
        )
        for assessment in ordered_virtue[:2]:
            action_id = str(assessment.get("canonical_action_id", ""))
            action = action_text.get(action_id, action_id or "the action")
            verdict = str(assessment.get("verdict", "UNCERTAIN")).lower()
            role = str(assessment.get("actor_role", "the actor"))
            virtues = str(assessment.get("virtues", "unspecified virtues"))
            vice = str(assessment.get("vice_risk", "an unspecified excess"))
            lines.append(
                f"- Virtue — {action}: {verdict} for {role}; expresses {virtues}; "
                f"risks {vice}."
            )

    if supporters:
        heading = (
            "Reasons supporting the judgment (classified by commitment):"
            if actionable else "Leading considerations (classified by commitment):"
        )
        lines.extend(["", heading])
    if direct_supporters:
        lines.append("Direct support:")
        seen: set[str] = set()
        for supporter in direct_supporters:
            reason = _support_reason(data, supporter, recommendation, original_actions)
            if not reason or reason.casefold() in seen:
                continue
            seen.add(reason.casefold())
            lines.append(f"- {supporter.get('specialist', 'supporting perspective')}: {reason}")
            if len(seen) == 3:
                break
    if qualified_supporters:
        lines.append("Qualified positions selecting the same action:")
        for supporter in qualified_supporters:
            reason = _support_reason(data, supporter, recommendation, original_actions)
            if not reason:
                continue
            status_label = _agreement_class(data, supporter).lower().replace("_", " ")
            lines.append(
                f"- {supporter.get('specialist', 'perspective')} ({status_label}): {reason}"
            )
    if supporters:
        winner = final.get("winner") or {}
        # Governing rationale comes from an adjudicated claim, not from whoever
        # won investigative attention. Provisional Kantian leanings corroborate
        # but do not supply the compressed rule.
        def _governing_eligible(candidate: dict[str, Any]) -> bool:
            if not candidate.get("schema_valid", True):
                return False
            if candidate.get("governing_eligible") is False:
                return False
            if str(candidate.get("broadcast_authority", "")).upper() == "INVESTIGATIVE":
                return False
            status = str(candidate.get("adjudication_status", "NOT_APPLICABLE")).upper()
            return status in {"", "NOT_APPLICABLE", "ADJUDICATED_SUPPORTS"}

        governing_rule = (
            _sentence(winner.get("decision_rule", ""))
            if _governing_eligible(winner)
            and _candidate_recommendation(winner) == recommendation
            else ""
        )
        if not governing_rule:
            governing_rule = next(
                (
                    _sentence(supporter.get("decision_rule", ""))
                    for supporter in supporters
                    if supporter.get("decision_rule") and _governing_eligible(supporter)
                ),
                "",
            )
        if governing_rule:
            governing_source = next(
                (
                    candidate for candidate in [winner, *supporters]
                    if _governing_eligible(candidate)
                    and _sentence(candidate.get("decision_rule", "")) == governing_rule
                ),
                winner if _governing_eligible(winner) else {},
            )
            governing_constraint = str(
                governing_source.get("constraint", winner.get("constraint", ""))
            ).strip().upper()
            winner_class = _agreement_class(data, governing_source or winner)
            label = (
                f"Governing decision rule ({governing_constraint})"
                if governing_constraint else "Governing decision rule"
            )
            if winner_class != "DIRECT":
                label = f"Governing decision rule ({governing_constraint}) — under review"
            lines.extend(["", f"{label}: {governing_rule}"])

        provisional = [
            candidate for candidate in supporters
            if str(candidate.get("adjudication_status", "")).upper() == "PROVISIONAL_LEANING"
            or (
                str(candidate.get("broadcast_authority", "")).upper() == "INVESTIGATIVE"
                and candidate.get("investigative_claim")
            )
        ]
        if provisional:
            lines.append("Provisional corroboration (not governing):")
            for candidate in provisional[:3]:
                claim = candidate.get("investigative_claim") or candidate.get("decision_rule") or ""
                if not claim:
                    continue
                lines.append(
                    f"- {candidate.get('specialist', 'perspective')}: {_sentence(claim)}"
                )

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

    residue_labels = {
        "RIGHTS": "rights and individual freedom",
        "DUTY": "duties and principled constraints",
        "CARE": "dependency and relational responsibility",
        "FAIRNESS": "fairness and equal standing",
        "CHARACTER": "character and practical wisdom",
        "UNCERTAINTY": "uncertain consequences",
        "AUTONOMY": "autonomy and valid consent",
    }

    def residue_explanation(constraint: str, sources: list[str] | None = None) -> str:
        source_set = set(sources or [])
        matching = [
            candidate for candidate in reversed(all_candidates)
            if str(candidate.get("constraint", "")).upper() == constraint.upper()
            and (not source_set or candidate.get("specialist") in source_set)
            and candidate.get("rationale")
        ]
        return _sentence(str(matching[0].get("rationale", ""))) if matching else ""

    residue = data.get("moral_residue") or []
    typed_residue = data.get("moral_residue_records") or []
    if typed_residue:
        lines.extend(["", "Preserved moral claims:"])
        for record in typed_residue[:4]:
            source_list = list(record.get("source_specialists") or [])
            sources = ", ".join(source_list) or "unspecified"
            constraint = str(record.get("constraint", "")).upper()
            label = residue_labels.get(
                constraint, constraint.lower().replace("_", " ") + " concerns",
            )
            explanation = residue_explanation(constraint, source_list)
            lines.append(
                f"- Concerns about {label} remain relevant to "
                f"{record.get('affected_action', 'the alternative')} "
                f"(raised by {sources})"
                + (f": {explanation}" if explanation else ".")
            )
    elif residue:
        lines.extend(["", "Preserved moral claims:"])
        for constraint in residue[:4]:
            code = str(constraint).upper()
            label = residue_labels.get(
                code, code.lower().replace("_", " ") + " concerns",
            )
            explanation = residue_explanation(code)
            lines.append(
                f"- Concerns about {label} remain unresolved"
                + (f": {explanation}" if explanation else ".")
            )

    assumptions: list[str] = []
    reversals: list[str] = []
    factual_reversals: list[str] = []
    normative_reversals: list[str] = []
    reversal_reviews: list[str] = []
    semantic_state = data.get("authoritative_semantic_state") or {}
    has_authoritative_state = int(semantic_state.get("version") or 0) >= 1
    if has_authoritative_state:
        for boundary in semantic_state.get("factual_reversal_boundaries") or []:
            predicate = _readable_condition(boundary.get("predicate", ""))
            target = _clean_fragment(boundary.get("target_action", ""))
            if predicate and target:
                rendered = f"Switch to {target} if {predicate}"
                if rendered not in factual_reversals:
                    factual_reversals.append(rendered)

    # Counterfactual candidates may contribute assumptions, normative thresholds,
    # and review verdicts. Factual thresholds come only from the authoritative
    # committed-graph projection when that projection is available.
    for candidate in [*all_candidates, *counterfactual_candidates]:
        assumption = _readable_condition(candidate.get("unsupported_assumption", ""))
        reversal = _readable_condition(candidate.get("reversal_condition", ""))
        if assumption and candidate.get("assumption_status") in {
            "CONDITIONAL", "UNDERDETERMINED", "NORMATIVELY_CONTESTED",
        } and assumption not in assumptions:
            assumptions.append(assumption)
        if reversal and reversal not in reversals:
            reversals.append(reversal)
        factual = (
            "" if has_authoritative_state
            else _readable_condition(candidate.get("factual_reversal_threshold", ""))
        )
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
        if (
            candidate.get("reversal_review_valid", True)
            and response in {"ACCEPT", "REVISE", "REJECT"}
            and justification
        ):
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
    contingency_reviews = [
        candidate
        for cycle in data.get("cycles", [])
        if (cycle.get("received_broadcast") or cycle.get("broadcast") or {}).get("constraint")
        == "CONTINGENCY_REVIEW"
        for candidate in cycle.get("candidates", [])
        if candidate.get("schema_valid") and candidate.get("contingency_choice")
    ]
    if contingency_reviews:
        lines.extend(["", "If the admitted synthesis fails:"])
        lines.extend(
            f"- {candidate.get('specialist', 'specialist')} would choose "
            f"{candidate['contingency_choice']}: "
            f"{_sentence(candidate.get('contingency_justification', ''))}"
            for candidate in contingency_reviews[:5]
        )
    contingency_feasibility = data.get("contingency_feasibility_assessments") or []
    if contingency_feasibility:
        lines.extend(["", "Independent contingency feasibility:"])
        for assessment in contingency_feasibility[:5]:
            fallback_statuses = assessment.get("fallback_statuses") or {}
            evidence_bases = assessment.get("evidence_bases") or {}
            shared_failure = bool(assessment.get("shared_failure"))
            lines.append(
                "- "
                + ("approved" if assessment.get("approved") else "blocked")
                + "; Fallback availability: "
                + ", ".join(
                    f"{action_id}={value}"
                    for action_id, value in fallback_statuses.items()
                )
            )
            lines.append(
                "- Shared-failure check: "
                + (
                    "shared failure: the synthesis and at least one fallback lose a common capability"
                    if shared_failure
                    else "no shared failure detected"
                )
            )
            lines.append(
                "- Basis: "
                + ", ".join(
                    f"{action_id}={value}"
                    for action_id, value in evidence_bases.items()
                )
            )
            error = _sentence(assessment.get("error", ""))
            if error:
                lines.append(f"- Reason: {error}")
    residual_reversals = [
        condition for condition in reversals
        if condition not in factual_reversals and condition not in normative_reversals
    ]
    if residual_reversals:
        lines.extend(["", "Reconsider if:"])
        for condition in residual_reversals[:5]:
            clause = re.sub(r"^if\s+", "", condition, flags=re.I)
            lines.append(f"- {_sentence(clause)}")

    further = data.get("further_deliberation_estimate") or {}
    if further:
        lines.extend([
            "",
            "Further deliberation (uncalibrated trace signals):",
            "- Action-change signal: "
            f"{float(further.get('action_change_signal', 0)):.2f}.",
            "- New-material-constraint signal: "
            f"{float(further.get('new_material_constraint_signal', 0)):.2f}.",
            "- These are not probabilities or measures of moral correctness and did not control stopping.",
        ])

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
