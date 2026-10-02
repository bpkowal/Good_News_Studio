"""Blank ethical blueprints, filled from a Z10 package, then ranked.

Each blueprint starts as an empty slot list. A filler writes a Parliament 1.3
candidate only for slots the scenario actually supports. The chooser keeps every
attempt and selects the filled plan whose required slots are the most specific.
Unfilled slots stay visible. Nothing here admits a world or picks a moral act.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

from blueprint_allocation_invariants import party_kind
from blueprint_proposal_contract import (
    candidate as proposal_candidate,
    coverage_metrics,
    proposal as normalized_proposal,
    withheld_proposal,
)
from candidate_graph_blueprints import (
    _clause_holding,
    _mention_node,
    _role_rows,
    instantiate_exclusive_allocation,
    match_exclusive_allocation,
)
from z10_world_model_adapter import (
    _predications,
    _selection,
    _support_span,
    segment_source_clauses,
)


ENSEMBLE_VERSION = "blueprint-ensemble/0.4"

# Empty plans. Values stay None until a scenario fills them.
BLANK_BLUEPRINTS: tuple[dict[str, Any], ...] = (
    {
        "blueprint_id": "exclusive_allocation",
        "summary": "One scarce resource, two recipients, and only one can receive it.",
        "graph_builder": "implemented",
        "required_slots": (
            "two_action_options", "exact_quantity",
            "explicit_exclusivity", "conditional_outcomes",
        ),
        "optional_slots": ("survival_chance", "nonreceipt_death"),
    },
    {
        "blueprint_id": "conditional_outcome",
        "summary": "An action is stated, and a bearer outcome is explicitly conditional on it.",
        "graph_builder": "implemented",
        "required_slots": ("action_condition", "outcome_bearer"),
        "optional_slots": ("second_conditional", "survival_chance"),
    },
    {
        "blueprint_id": "rescue_contrast",
        "summary": "Someone can rescue either of two parties, but not both.",
        "graph_builder": "implemented",
        "required_slots": (
            "two_rescue_actions", "distinct_rescue_targets",
            "explicit_exclusivity", "branch_outcomes",
        ),
        "optional_slots": ("scene_parties", "foregone_harms"),
    },
    {
        "blueprint_id": "omission_harm",
        "summary": "Doing an action and not doing it are both stated, and each branch states who is harmed.",
        "graph_builder": "implemented",
        "required_slots": (
            "positive_action", "negated_action",
            "harm_if_done", "harm_if_omitted",
        ),
        "optional_slots": ("group_counts", "instrument_contrast"),
    },
    {
        "blueprint_id": "ability_permission",
        "summary": "A modal states what an actor is able or permitted to do.",
        "graph_builder": "plan_only",
        "required_slots": ("modal_action", "actor", "target"),
        "optional_slots": ("modal_reading", "option_membership", "outcome", "duty_or_prohibition"),
    },
    {
        "blueprint_id": "diversion_redirection",
        "summary": "An intervention redirects a process toward different affected parties.",
        "graph_builder": "implemented",
        "required_slots": ("actor", "controllable_process", "intervention", "affected_party", "outcome"),
        "optional_slots": ("alternative_route", "omission_branch", "uncertainty", "secondary_effect"),
    },
    {
        "blueprint_id": "deontic_rule",
        "summary": "A rule states an obligation, permission, or prohibition over an action.",
        "graph_builder": "plan_only",
        "required_slots": ("deontic_words", "governed_action", "bearer"),
        "optional_slots": ("authority", "exception", "conflicting_rule", "sanction"),
    },
    {
        "blueprint_id": "promise_reliance",
        "summary": "A promisor commits to future conduct that another party may rely on.",
        "graph_builder": "plan_only",
        "required_slots": ("commitment_event", "promisor", "commitment_content", "promisee"),
        "optional_slots": ("reliance", "breach", "changed_condition", "competing_commitment"),
    },
    {
        "blueprint_id": "uncertain_risk",
        "summary": "Alternative actions expose parties to explicitly uncertain outcomes.",
        "graph_builder": "implemented",
        "required_slots": ("actor", "action", "possible_outcome", "affected_party", "likelihood"),
        "optional_slots": (
            "second_action", "second_affected_party", "second_outcome",
            "second_likelihood", "expected_quantity", "confidence_source",
        ),
    },
    {
        "blueprint_id": "disputed_report",
        "summary": "A speaker attributes a proposition or conflicts with another report.",
        "graph_builder": "plan_only",
        "required_slots": ("source", "reported_content", "report_words"),
        "optional_slots": ("competing_report", "reliability", "downstream_decision", "confirmation"),
    },
)

_RESCUE_PREDICATES = {"save", "rescue", "carry"}
_HARM_PREDICATES = {"die", "kill"}
_SURVIVAL_PREDICATES = {"live", "survive", "recover"}


def blank_blueprints() -> list[dict[str, Any]]:
    """Return the empty plans. No scenario has been read."""
    return [{
        "blueprint_id": row["blueprint_id"],
        "summary": row["summary"],
        "graph_builder": row["graph_builder"],
        "required_slots": {name: None for name in row["required_slots"]},
        "optional_slots": {name: None for name in row["optional_slots"]},
    } for row in BLANK_BLUEPRINTS]


def choose_blueprint(package: dict, actions: Sequence[str] = ()) -> dict[str, Any]:
    """Fill every blank plan and pick the most specific complete one."""
    considered = [
        _exclusive_report(package, actions),
        _conditional_report(package),
        _rescue_report(package),
        _omission_report(package),
        _diversion_report(package),
        _risk_report(package),
        *[_plan_only_report(package, row) for row in BLANK_BLUEPRINTS
          if row["graph_builder"] == "plan_only"],
    ]
    filled = [row for row in considered if row["status"] == "FILLED"]

    def rank(row: dict) -> tuple:
        required = row["match"]["required_slots"]
        optional = row["optional_slots"]
        return (
            sum(bool(value) for value in required.values()),
            len(required),
            sum(bool(value) for value in optional.values()),
        )

    chosen = max(filled, key=rank)["blueprint_id"] if filled else None
    return {
        "ensemble_version": ENSEMBLE_VERSION,
        "chosen_blueprint_id": chosen,
        "considered": considered,
    }


def _plan_only_report(package: dict, plan: dict[str, Any]) -> dict[str, Any]:
    """Expose the schema without pretending that it can materialize a world."""
    evidence, seed_ids = _plan_evidence(package, plan["blueprint_id"])
    required = {name: evidence.get(name) for name in plan["required_slots"]}
    optional = {name: evidence.get(name) for name in plan["optional_slots"]}
    selection, validation = _selection(package, seed_ids)
    clauses = [
        {"clause_id": row["clause_id"], "text": row["text"]}
        for row in segment_source_clauses(package["document"]["text"])
    ]
    missing = [name for name, value in required.items() if not value]
    problem = {
        "code": f"{plan['blueprint_id']}_not_representable_in_world_1_3",
        "message": "The evidence is retained without converting it into an admitted world fact.",
    }
    proposal = withheld_proposal(
        proposal_id=f"{plan['blueprint_id']}_0",
        blueprint_id=plan["blueprint_id"],
        assignment=[str(value) for name, value in required.items()
                    if "action" in name and value],
        slot_bindings={"required": required, "optional": optional,
                       "seed_candidate_ids": sorted(seed_ids)},
        selection=selection,
        selection_validation=validation,
        clauses=clauses,
        unfilled_required_slots=missing,
        unresolved_readings=_plan_unresolved_readings(plan["blueprint_id"], evidence),
        construction_problems=[problem],
        pre_world_assessment=_package_question_assessment(package, False),
        accepted_evidence={
            "required": required,
            "optional": optional,
            "seed_candidate_ids": sorted(seed_ids),
        },
    )
    row = _report(plan["blueprint_id"], "PLAN_ONLY", {
        "blueprint_id": plan["blueprint_id"],
        "matched": not missing,
        "required_slots": required,
        "unfilled_required_slots": missing,
    }, optional, [proposal])
    row["world_withheld"] = [
        "This evidence plan has no Parliament 1.3 graph builder yet."
    ]
    return row


def _exclusive_report(package: dict, actions: Sequence[str]) -> dict[str, Any]:
    result = (instantiate_exclusive_allocation(package, actions) if actions
              else {"status": "NO_MATCH", "match": match_exclusive_allocation(package, ()),
                    "proposals": []})
    optional = {"survival_chance": False, "nonreceipt_death": False}
    if result.get("status") == "FILLED":
        effects = result["proposals"][0]["candidate"]["world_model"]["effects"]
        optional["survival_chance"] = any(row["modality"] == "PROBABILISTIC" for row in effects)
        optional["nonreceipt_death"] = any(row["predicate"] == "die" for row in effects)
    return _report("exclusive_allocation", result.get("status", "NO_MATCH"),
                   result.get("match") or match_exclusive_allocation(package, actions),
                   optional, result.get("proposals") or [])


def _plan_evidence(package: dict, blueprint_id: str) -> tuple[dict[str, Any], set[str]]:
    """Bind non-world semantics to Z10 evidence or exact source copies."""
    text = package["document"]["text"]
    nodes = {row["id"]: row for row in package["nodes"]}
    evidence: dict[str, Any] = {}
    seeds: set[str] = set()
    modalities = [row for row in package["candidates"] if row["type"] == "MODALITY"]
    if blueprint_id in {"ability_permission", "deontic_rule"}:
        allowed = ({"ability", "permission", "possibility", "unresolved"}
                   if blueprint_id == "ability_permission"
                   else {"obligation", "permission", "unresolved"})
        matches = [row for row in modalities if str(row.get("value")).casefold() in allowed]
        if matches:
            seeds.update(row["id"] for row in matches)
            proposition = matches[0]["arguments"].get("proposition")
            prop = nodes.get(proposition, {})
            label = prop.get("label") or prop.get("predicate")
            roles = _role_rows(package, proposition) if proposition else {}
            actor_row = (roles.get("subject") or roles.get("agent") or [None])[0]
            target_row = (roles.get("object") or roles.get("destination") or
                          roles.get("patient") or [None])[0]
            actor = _mention_node(package, actor_row)["label"] if actor_row else None
            target = _mention_node(package, target_row)["label"] if target_row else None
            if blueprint_id == "ability_permission":
                evidence.update({
                    "modal_action": label,
                    "actor": actor,
                    "target": target,
                    "modal_reading": [row.get("value") for row in matches],
                    "option_membership": [
                        row["id"] for row in package["candidates"]
                        if row["type"] == "OPTION_OF"
                        and row["arguments"].get("proposition") == proposition
                    ] or None,
                })
            else:
                words = re.search(
                    r"\b(?:must|shall|should|required to|obligated to|"
                    r"forbidden to|prohibited from|may not|must not)\b", text, re.I)
                evidence.update({
                    "deontic_words": words.group(0) if words else None,
                    "governed_action": label,
                    "bearer": actor,
                    "conflicting_rule": [row.get("value") for row in matches],
                })
    elif blueprint_id == "promise_reliance":
        event = re.search(
            r"\b(?:promise|promises|promised|commit|commits|committed|pledge|pledges)\b",
            text, re.I)
        names = re.findall(r"\b[A-Z][a-z]+\b", text)
        content = re.search(r"\b(?:that|to)\s+(.+?)(?:[.!?]|$)", text, re.I)
        evidence.update({
            "commitment_event": event.group(0) if event else None,
            "promisor": names[0] if event and names else None,
            "promisee": names[1] if event and len(names) > 1 else None,
            "commitment_content": content.group(1) if event and content else None,
            "reliance": _first_copy(text, r"\b(?:rely|relies|relied|depend|depends)\b[^.!?]*"),
            "breach": _first_copy(text, r"\b(?:breach|breaks|broke|fails|failed)\b[^.!?]*"),
        })
    elif blueprint_id == "disputed_report":
        report = re.search(
            r"\b(?:report|reports|reported|say|says|said|claim|claims|claimed|"
            r"believe|believes|allege|alleges)\b", text, re.I)
        names = re.findall(r"\b[A-Z][a-z]+\b", text)
        content = text[report.end():].strip(" ,.") if report else None
        attributed = [
            row for row in package["candidates"]
            if row["type"] == "PREDICATION"
            and "attributed" in str(row.get("scope") or row.get("value") or "").casefold()
        ]
        seeds.update(row["id"] for row in attributed)
        evidence.update({
            "source": names[0] if report and names else None,
            "report_words": report.group(0) if report else None,
            "reported_content": content,
            "competing_report": None,
            "reliability": _first_copy(
                text, r"\b(?:reliable|unreliable|credible|trustworthy|false|accurate)\b"),
            "confirmation": _first_copy(
                text, r"\b(?:confirm|confirmed|refute|refuted|verify|verified)\b[^.!?]*"),
        })
    return evidence, seeds


def _plan_unresolved_readings(blueprint_id: str,
                              evidence: dict[str, Any]) -> list[dict[str, Any]]:
    if blueprint_id == "ability_permission":
        return [{
            "kind": "modal_force",
            "alternatives": evidence.get("modal_reading")
            or ["ability", "permission", "possibility"],
        }]
    if blueprint_id == "deontic_rule":
        return [{"kind": "normative_force", "status": "not_represented_in_world_1_3"}]
    if blueprint_id == "promise_reliance":
        return [{"kind": "commitment_status", "status": "not_world_occurrence"}]
    if blueprint_id == "disputed_report":
        return [{"kind": "reported_truth", "status": "unresolved"}]
    return []


def _first_copy(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, re.I)
    return match.group(0) if match else None


def _package_question_assessment(package: dict,
                                 eligible: bool) -> dict[str, Any]:
    nodes = {row["id"]: row for row in package["nodes"]}
    candidates = {row["id"]: row for row in package["candidates"]}
    options = []
    for group in package.get("choice_sets", []):
        if group.get("kind") != "scenario_option":
            continue
        labels = []
        for candidate_id in group.get("candidate_ids", []):
            proposition = candidates.get(candidate_id, {}).get("arguments", {}).get("proposition")
            label = nodes.get(proposition, {}).get("label")
            if label:
                labels.append(label)
        options.append({
            "choice_set_id": group["id"],
            "selection_rule": group.get("selection_rule"),
            "exhaustive": group.get("exhaustive"),
            "options": labels,
        })
    participants = []
    for row in package["candidates"]:
        if row["type"] != "PARTICIPANT":
            continue
        mention = nodes.get(row["arguments"].get("mention"), {}).get("label")
        proposition = nodes.get(row["arguments"].get("proposition"), {}).get("label")
        if mention and proposition:
            participants.append({
                "role": row.get("value"),
                "mention": mention,
                "predicate": proposition,
            })
    return {
        "status": "ASSESSED" if eligible else "WITHHELD",
        "eligible_for_world_state": eligible,
        "ethical_question": next(
            (row.get("label") for row in package["nodes"]
             if row.get("kind") == "choice_point"), ""),
        "scenario_options": options,
        "exclusivity": (
            "evidenced" if re.search(
                r"\bnot\s+both\b", package["document"]["text"], re.I)
            else "unspecified"
        ),
        "participants": participants,
    }


def _conditional_report(package: dict) -> dict[str, Any]:
    links = _outcome_links(package)
    usable = [row for row in links if row["actor"] and row["bearer"]]
    required = {
        "action_condition": bool(usable),
        "outcome_bearer": bool(usable),
    }
    optional = {
        "second_conditional": len(usable) >= 2,
        "survival_chance": any("%" in row["outcome_source"] and "chance" in row["outcome_source"].casefold()
                               for row in usable),
    }
    if not all(required.values()):
        return _unmatched("conditional_outcome", required, optional)
    return _filled("conditional_outcome", required, optional, _conditional_proposal(package, usable))


def _rescue_report(package: dict) -> dict[str, Any]:
    rescues = _rescue_links(package)
    targets = {
        row["saved"]["label"].casefold() for row in rescues if row.get("saved")
    }
    explicit_exclusivity = bool(re.search(
        r"\bnot\s+both\b", package["document"]["text"], re.I))
    required = {
        "two_rescue_actions": len(rescues) >= 2,
        "distinct_rescue_targets": len(targets) >= 2,
        "explicit_exclusivity": explicit_exclusivity,
        "branch_outcomes": len(rescues) >= 2 and all(row["survival"] for row in rescues[:2]),
    }
    scene = _scene_parties(package)
    optional = {
        "scene_parties": bool(scene),
        "foregone_harms": False,
    }
    if not all(required.values()):
        return _unmatched("rescue_contrast", required, optional)
    return _filled("rescue_contrast", required, optional,
                   _rescue_proposal(package, rescues[:2], scene))


def _omission_report(package: dict) -> dict[str, Any]:
    pair = _omission_pair(package)
    required = {
        "positive_action": bool(pair and pair["done"]),
        "negated_action": bool(pair and pair["omitted"]),
        "harm_if_done": bool(pair and pair["done"] and pair["done"]["harm"]),
        "harm_if_omitted": bool(pair and pair["omitted"] and pair["omitted"]["harm"]),
    }
    optional = {
        "group_counts": bool(pair and pair["counts"]),
        "instrument_contrast": bool(pair and pair["instrument"]),
    }
    if not pair or not all(required.values()):
        return _unmatched("omission_harm", required, optional)
    return _filled("omission_harm", required, optional, _omission_proposal(package, pair))


def _diversion_report(package: dict) -> dict[str, Any]:
    slots = _diversion_slots(package["document"]["text"])
    required_names = next(
        row["required_slots"] for row in BLANK_BLUEPRINTS
        if row["blueprint_id"] == "diversion_redirection")
    optional_names = next(
        row["optional_slots"] for row in BLANK_BLUEPRINTS
        if row["blueprint_id"] == "diversion_redirection")
    required = {name: slots.get(name) for name in required_names}
    optional = {name: slots.get(name) for name in optional_names}
    if not all(required.values()):
        return _unmatched("diversion_redirection", required, optional)
    return _filled(
        "diversion_redirection", required, optional,
        _slot_graph_proposal(package, "diversion_redirection", slots))


def _risk_report(package: dict) -> dict[str, Any]:
    slots = _risk_slots(package["document"]["text"])
    required_names = next(
        row["required_slots"] for row in BLANK_BLUEPRINTS
        if row["blueprint_id"] == "uncertain_risk")
    optional_names = next(
        row["optional_slots"] for row in BLANK_BLUEPRINTS
        if row["blueprint_id"] == "uncertain_risk")
    required = {name: slots.get(name) for name in required_names}
    optional = {name: slots.get(name) for name in optional_names}
    if not all(required.values()):
        return _unmatched("uncertain_risk", required, optional)
    return _filled(
        "uncertain_risk", required, optional,
        _slot_graph_proposal(package, "uncertain_risk", slots))


def _slot_graph_proposal(package: dict, blueprint_id: str,
                         slots: dict[str, str]) -> dict[str, Any]:
    # The text-slot graph constructors are shared with the cloze path.  The
    # ensemble adds the dependency-closed Z10 selection and source assessment.
    from blueprint_cloze_chooser import _diversion_graph, _risk_graph

    builder = _diversion_graph if blueprint_id == "diversion_redirection" else _risk_graph
    world, notes = builder(package["document"]["text"], slots)
    seeds = _seed_ids_for_slots(package, slots.values())
    selection, validation = _selection(package, seeds)
    clauses = [
        {"clause_id": row["clause_id"], "text": row["text"]}
        for row in segment_source_clauses(package["document"]["text"])
    ]
    action_sources = {
        row["action_id"]: {
            "clause_ids": list(row["clause_ids"]),
            "reason": f"Filled by {blueprint_id} evidence plan.",
        }
        for row in world["actions"]
    }
    return normalized_proposal(
        proposal_id=f"{blueprint_id}_0",
        blueprint_id=blueprint_id,
        status="FILLED",
        assignment_kind="intervention_text",
        assignment=[row["intervention"] for row in world["actions"]],
        slot_bindings={
            "copied_spans": dict(slots),
            "seed_candidate_ids": sorted(seeds),
        },
        selection=selection,
        selection_validation=validation,
        candidate_value=proposal_candidate(action_sources, world),
        clauses=clauses,
        unresolved_readings=(
            [{"kind": "risk_likelihood", "status": "source_qualified"}]
            if blueprint_id == "uncertain_risk" else []
        ),
        admission_authorized=True,
        notes=notes,
        pre_world_assessment={
            "status": "ASSESSED",
            "eligible_for_world_state": True,
            "ethical_question": blueprint_id,
            "scenario_options": [row["intervention"] for row in world["actions"]],
            "exclusivity": "unspecified",
            "participants": [row["label"] for row in world["parties"]],
        },
        accepted_evidence=dict(slots),
    )


def _diversion_slots(text: str) -> dict[str, str]:
    actor = re.search(r"\b([A-Z][a-z]+)\s+(?:can|may|must|could)?\s*"
                      r"(divert\w*|redirect\w*|switch\w*|turn\w*)\b", text)
    intervention = re.search(
        r"\b(?:divert\w*|redirect\w*|switch\w*|turn\w*)\b"
        r"[^,.!?]*(?=,|\band\b|[.!?])", text, re.I)
    process = re.search(
        r"\b(?:the\s+)?(?:flow|trolley|train|water|traffic|current|fire|process)\b",
        text, re.I)
    outcome = re.search(
        r"\b((?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        r"(?:people|workers?|patients?|residents?)|(?:the\s+)?(?:person|child|dog))\s+"
        r"(?:will|may|might|could)\s+(?:die|live|drown|suffer|survive)\b[^.!?]*",
        text, re.I)
    return {
        "actor": actor.group(1) if actor else "",
        "controllable_process": process.group(0) if process else "",
        "intervention": (
            f"{actor.group(2)} {process.group(0)}"
            if actor and process else
            intervention.group(0).strip() if intervention else ""
        ),
        "affected_party": outcome.group(1).strip() if outcome else "",
        "outcome": outcome.group(0).strip() if outcome else "",
    }


def _risk_slots(text: str) -> dict[str, str]:
    action_match = re.search(
        r"\b([A-Z][a-z]+)\s+(?:may|might|could)\s+([^,.!?]+)", text)
    outcome = re.search(
        r"\b([A-Z][a-z]+|(?:one|two|three|four|five|\d+)\s+"
        r"(?:people|workers|patients|residents))\s+"
        r"(may|might|could|has\s+a\s+\d+(?:\.\d+)?%\s+chance\s+of)\s+"
        r"(?:die|live|drown|survival|survive|recover)\b[^.!?]*",
        text, re.I)
    return {
        "actor": action_match.group(1) if action_match else "",
        "action": action_match.group(2).strip() if action_match else "",
        "possible_outcome": outcome.group(0).strip() if outcome else "",
        "affected_party": outcome.group(1).strip() if outcome else "",
        "likelihood": outcome.group(2).strip() if outcome else "",
    }


def _seed_ids_for_slots(package: dict, values: Sequence[str]) -> set[str]:
    spans = [value.casefold() for value in values if isinstance(value, str) and value]
    evidence = {row["id"]: row for row in package["evidence"]}
    seeds = set()
    for row in package["candidates"]:
        snippets = [
            package["document"]["text"][evidence[ident]["start"]:evidence[ident]["end"]].casefold()
            for ident in row.get("evidence_ids", []) if ident in evidence
        ]
        if any(any(span in snippet or snippet in span for span in spans)
               for snippet in snippets if snippet):
            seeds.add(row["id"])
    return seeds


def _outcome_links(package: dict) -> list[dict[str, Any]]:
    predications = _predications(package)
    links = []
    for row in package["candidates"]:
        if row["type"] != "CONDITIONAL_ON":
            continue
        condition = row["arguments"]["condition"]
        consequence = row["arguments"]["consequence"]
        condition_roles = _role_rows(package, condition)
        consequence_roles = _role_rows(package, consequence)
        actor = (condition_roles.get("subject") or condition_roles.get("agent") or [None])[0]
        bearer = (consequence_roles.get("subject") or consequence_roles.get("patient") or [None])[0]
        outcome_source, outcome_clauses = _support_span(
            package, [row, predications[consequence]])
        host = _clause_holding(segment_source_clauses(package["document"]["text"]), outcome_source)
        links.append({
            "link": row,
            "condition": condition,
            "consequence": consequence,
            "condition_roles": condition_roles,
            "consequence_roles": consequence_roles,
            "actor": _mention_node(package, actor) if actor else None,
            "bearer": _mention_node(package, bearer) if bearer else None,
            "outcome_source": host["text"] if host else outcome_source,
            "outcome_clauses": [host["clause_id"]] if host else outcome_clauses,
            "polarity": predications[condition]["scope"]["polarity"],
            "condition_predicate": predications[condition].get("value")
            or _proposition_node(package, condition).get("predicate"),
            "outcome_predicate": _proposition_node(package, consequence).get("predicate"),
        })
    return links


def _rescue_links(package: dict) -> list[dict[str, Any]]:
    rescues = []
    for row in _outcome_links(package):
        predicate = str(row["condition_predicate"] or "")
        outcome = str(row["outcome_predicate"] or "")
        if predicate in _RESCUE_PREDICATES and outcome in _SURVIVAL_PREDICATES and row["actor"] and row["bearer"]:
            saved = (row["condition_roles"].get("object") or [None])[0]
            if not saved:
                continue
            saved_node = _mention_node(package, saved)
            if saved_node["label"].casefold() != row["bearer"]["label"].casefold():
                continue
            row["saved"] = saved_node
            row["survival"] = True
            rescues.append(row)
    return rescues


def _scene_parties(package: dict) -> list[str]:
    labels = []
    for proposition, roles in _grouped_roles(package).items():
        node = _proposition_node(package, proposition)
        if node.get("predicate") not in {"be", "are"}:
            continue
        for role in roles.get("subject", []):
            labels.append(_mention_node(package, role)["label"])
    return labels


def _omission_pair(package: dict) -> dict[str, Any] | None:
    done = omitted = None
    for row in _outcome_links(package):
        if str(row["outcome_predicate"] or "") not in _HARM_PREDICATES:
            continue
        row["harm"] = True
        if row["polarity"] == "negative" and omitted is None:
            omitted = row
        elif row["polarity"] == "positive" and done is None:
            done = row
    if not done or not omitted:
        return None
    if done["actor"]["label"].casefold() != omitted["actor"]["label"].casefold():
        return None
    counts = [row["id"] for row in package["candidates"]
              if row["type"] == "QUANTITY" and row["value"].get("operator") == "exact"]
    instrument = next((row for row in package.get("reconstructions", [])
                       if row.get("antecedent_proposition_id")), None)
    return {"done": done, "omitted": omitted, "counts": counts, "instrument": instrument}


def _intervention_text(package: dict, row: dict[str, Any]) -> str:
    obj = (row["condition_roles"].get("object") or [None])[0]
    label = _mention_node(package, obj)["label"] if obj else ""
    prefix = "do not " if row["polarity"] == "negative" else ""
    return f"{prefix}{row['condition_predicate']} {label}".strip()


def _conditional_proposal(package: dict, links: list[dict[str, Any]]) -> dict[str, Any]:
    return _proposal(package, "conditional_outcome_0", [
        _branch(package, index, row, _intervention_text(package, row),
                str(row["outcome_predicate"] or "outcome"), _outcome_polarity(row))
        for index, row in enumerate(links)
    ], [row["link"]["id"] for row in links])


def _rescue_proposal(package: dict, rescues: list[dict[str, Any]],
                     scene: list[str]) -> dict[str, Any]:
    branches = []
    for index, rescue in enumerate(rescues):
        branch = _branch(
            package, index, rescue, f"save {rescue['saved']['label']}",
            "live", "BENEFICIAL")
        branch["slot_note"] = {
            "rescue_target": rescue["saved"]["label"],
            "explicit_exclusivity": "not both",
            "scene_parties": scene,
            "foregone_harm": None,
        }
        branches.append(branch)
    return _proposal(
        package, "rescue_contrast_0", branches,
        [rescue["link"]["id"] for rescue in rescues])


def _omission_proposal(package: dict, pair: dict[str, Any]) -> dict[str, Any]:
    branches = []
    for index, (row, prefix) in enumerate((
            (pair["done"], ""),
            (pair["omitted"], "do not "),
    )):
        target = (row["condition_roles"].get("object") or [None])[0]
        target_label = _mention_node(package, target)["label"] if target else row["condition_predicate"]
        branches.append(_branch(
            package, index, row,
            f"{prefix}{row['condition_predicate']} {target_label}".strip(),
            "die", "ADVERSE"))
    return _proposal(package, "omission_harm_0", branches,
                     [pair["done"]["link"]["id"], pair["omitted"]["link"]["id"]])


def _branch(package: dict, index: int, row: dict[str, Any], intervention: str,
            outcome_predicate: str, polarity: str) -> dict[str, Any]:
    actor = row["actor"]
    bearer = row["bearer"]
    clauses = segment_source_clauses(package["document"]["text"])
    host = _clause_holding(clauses, row["outcome_source"]) or {
        "clause_id": row["outcome_clauses"][0], "text": row["outcome_source"],
    }
    direct_id, outcome_id = f"E{index * 2 + 1}", f"E{index * 2 + 2}"
    action_id = f"A{index}"
    parties = [
        _party(package, "P1" if index == 0 else None, actor, "PERSON"),
        _party(package, "P2" if index == 0 else None, bearer, _kind(bearer["label"], outcome_predicate)),
    ]
    # The caller merges parties across branches. Local ids are rewritten there.
    effects = [
        _effect(direct_id, action_id, "BEARER", intervention,
                row["condition_predicate"], "NEUTRAL", "DIRECT", "INTERVENTION",
                host["text"], [], host["clause_id"], ""),
        _effect(outcome_id, action_id, "BEARER", f"will {outcome_predicate}",
                outcome_predicate, polarity, "DOWNSTREAM", "HEALTH_OUTCOME",
                host["text"], [direct_id], host["clause_id"],
                "The source states this outcome on this branch."),
    ]
    chance = re.search(r"\b\d+(?:\.\d+)?%\s*chance\b", row["outcome_source"], re.I)
    if chance:
        effects[1]["modality"] = "PROBABILISTIC"
        effects[1]["likelihood_qualifiers"] = [chance.group(0)]
    effects[0]["party_id"] = "BEARER"
    return {
        "intervention": intervention,
        "actor_label": actor["label"],
        "bearer_label": bearer["label"],
        "actor_node_id": actor.get("id"),
        "bearer_node_id": bearer.get("id"),
        "parties": [actor, bearer],
        "effects": effects,
        "link": {"action_id": action_id, "source_id": direct_id, "target_id": outcome_id,
                 "clause_id": host["clause_id"],
                 "modality": effects[1]["modality"]},
        "seed": [row["link"]["id"]],
    }


def _proposal(package: dict, proposal_id: str, branches: list[dict[str, Any]],
              seed_ids: list[str]) -> dict[str, Any]:
    parties: list[dict[str, Any]] = []
    party_id: dict[str, str] = {}
    for branch in branches:
        for node in branch["parties"]:
            label = node["label"]
            mention_key = str(node.get("id") or f"{len(parties)}:{label}")
            if mention_key in party_id:
                continue
            ident = f"P{len(parties) + 1}"
            party_id[mention_key] = ident
            kind = _kind(label, "")
            parties.append({
                "party_id": ident, "label": label, "kind": kind, "quantities": [],
                "clause_ids": _clause_ids_containing(package, label),
            })
    actions, effects, links = [], [], []
    for index, branch in enumerate(branches):
        action_id = f"A{index}"
        actor = party_id[str(branch["actor_node_id"])]
        bearer = party_id[str(branch["bearer_node_id"])]
        direct, outcome = branch["effects"]
        direct["effect_id"] = f"E{index * 2 + 1}"
        outcome["effect_id"] = f"E{index * 2 + 2}"
        direct["action_id"] = outcome["action_id"] = action_id
        direct["party_id"] = outcome["party_id"] = bearer
        count = _count_token(branch["bearer_label"], outcome["source_proposition"])
        if count:
            for row in parties:
                if row["party_id"] == bearer and count not in row["quantities"]:
                    row["quantities"].append(count)
            direct["quantities"] = [count]
            outcome["quantities"] = [count]
        outcome["source_effect_ids"] = [direct["effect_id"]]
        clause = branch["link"]["clause_id"]
        effects.extend([direct, outcome])
        links.append({"action_id": action_id, "source_id": direct["effect_id"],
                      "link_relation": "CAUSES", "target_id": outcome["effect_id"],
                      "modality": branch["link"].get("modality", "CERTAIN"),
                      "condition_ids": [], "clause_ids": [clause]})
        actions.append({"action_id": action_id, "intervention": branch["intervention"],
                        "actor_party_id": actor, "recipient_party_ids": [bearer],
                        "effect_ids": [direct["effect_id"], outcome["effect_id"]],
                        "clause_ids": [clause]})
    seed = set(seed_ids)
    for branch in branches:
        seed.update(branch["seed"])
    selection, validation = _selection(package, seed)
    if not validation["contract_valid"]:
        raise ValueError("Blueprint produced an invalid Z10 selection: "
                         + str(validation["errors"]))
    blueprint_id = proposal_id.rsplit("_", 1)[0]
    world = {"schema_version": "1.3", "parties": parties, "actions": actions,
             "effects": effects, "conditions": [], "temporal_relations": [],
             "causal_links": links, "counterfactual_links": []}
    return normalized_proposal(
        proposal_id=proposal_id,
        blueprint_id=blueprint_id,
        status="FILLED",
        assignment_kind="intervention_text",
        assignment=[row["intervention"] for row in actions],
        slot_bindings={
            "seed_candidate_ids": sorted(seed),
            "actions": [
                {"action_id": row["action_id"], "intervention": row["intervention"],
                 "effect_ids": list(row["effect_ids"])}
                for row in actions
            ],
        },
        selection=selection,
        selection_validation=validation,
        candidate_value=proposal_candidate(
            {row["action_id"]: {
                "clause_ids": row["clause_ids"],
                "reason": "Filled from the blank blueprint.",
            } for row in actions},
            world,
        ),
        clauses=[
            {"clause_id": row["clause_id"], "text": row["text"]}
            for row in segment_source_clauses(package["document"]["text"])
        ],
        unresolved_readings=(
            [{"kind": "conditional_relation", "alternatives": ["CAUSES", "ENABLES"]}]
            if blueprint_id == "conditional_outcome" else []
        ),
        admission_authorized=True,
        notes=[branch.get("slot_note") for branch in branches if branch.get("slot_note")],
        pre_world_assessment={
            "status": "ASSESSED",
            "eligible_for_world_state": True,
            "ethical_question": proposal_id.rsplit("_", 1)[0],
            "scenario_options": [row["intervention"] for row in actions],
            "exclusivity": (
                "evidenced" if re.search(r"\bnot\s+both\b", package["document"]["text"], re.I)
                else "unspecified"
            ),
            "participants": [row["label"] for row in parties],
        },
        accepted_evidence={
            "seed_candidate_ids": sorted(seed),
            "action_clause_ids": {
                row["action_id"]: list(row["clause_ids"]) for row in actions
            },
        },
    )


def _effect(effect_id: str, action_id: str, party_id: str, outcome: str, predicate: str,
            polarity: str, directness: str, kind: str, source: str, parents: list[str],
            clause_id: str, explanation: str) -> dict[str, Any]:
    return {
        "effect_id": effect_id, "action_id": action_id, "party_id": party_id,
        "outcome": outcome, "predicate": predicate, "polarity": polarity,
        "directness": directness, "modality": "CERTAIN", "effect_kind": kind,
        "condition_ids": [], "quantities": [], "likelihood_qualifiers": [],
        "overall_likelihood_qualifiers": [], "scope_qualifiers": [],
        "temporal_qualifiers": [], "condition_join": "AND",
        "source_proposition": source, "source_effect_ids": parents,
        "derivation_operation": "DIRECT_COPY" if not parents else "SOURCE_STIPULATED_CAUSAL",
        "derivation_explanation": explanation, "derivation_assumptions": [],
        "outcome_type_transformation": "PRESERVED", "clause_ids": [clause_id],
    }


def _party(package: dict, ident: str | None, node: dict, kind: str) -> dict[str, Any]:
    return node


def _count_token(label: str, source: str) -> str | None:
    """Copy a number word only when both the party label and the clause state it."""
    for word in ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"):
        if word not in label.casefold():
            continue
        match = re.search(rf"\b{word}\b", source, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def _kind(label: str, predicate: str) -> str:
    return party_kind(label)


def _outcome_polarity(row: dict[str, Any]) -> str:
    if str(row["outcome_predicate"] or "") in _HARM_PREDICATES:
        return "ADVERSE"
    return "BENEFICIAL"


def _proposition_node(package: dict, proposition: str) -> dict:
    return next(row for row in package["nodes"] if row["id"] == proposition)


def _grouped_roles(package: dict) -> dict[str, dict[str, list[dict]]]:
    grouped: dict[str, dict[str, list[dict]]] = {}
    for proposition, rows in _all_roles(package).items():
        grouped[proposition] = {}
        for row in rows:
            grouped[proposition].setdefault(row["value"], []).append(row)
    return grouped


def _all_roles(package: dict) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for row in package["candidates"]:
        if row["type"] == "PARTICIPANT":
            result.setdefault(row["arguments"]["proposition"], []).append(row)
    return result


def _clause_ids_containing(package: dict, label: str) -> list[str]:
    word = label.strip()
    return [row["clause_id"] for row in segment_source_clauses(package["document"]["text"])
            if word.casefold() in row["text"].casefold()]


def _unmatched(blueprint_id: str, required: dict[str, bool],
               optional: dict[str, bool]) -> dict[str, Any]:
    return _report(blueprint_id, "NO_MATCH", {
        "blueprint_id": blueprint_id,
        "matched": False,
        "required_slots": required,
        "unfilled_required_slots": [key for key, filled in required.items() if not filled],
    }, optional, [])


def _filled(blueprint_id: str, required: dict[str, bool], optional: dict[str, bool],
            proposal: dict[str, Any]) -> dict[str, Any]:
    return _report(blueprint_id, "FILLED", {
        "blueprint_id": blueprint_id,
        "matched": True,
        "required_slots": required,
        "unfilled_required_slots": [],
    }, optional, [proposal])


def _report(blueprint_id: str, status: str, match: dict[str, Any],
            optional: dict[str, bool], proposals: list[dict[str, Any]]) -> dict[str, Any]:
    blank = next(row for row in blank_blueprints() if row["blueprint_id"] == blueprint_id)
    return {
        "blueprint_id": blueprint_id,
        "status": status,
        "blank": blank,
        "match": match,
        "optional_slots": optional,
        "coverage_metrics": coverage_metrics(
            match.get("required_slots") or {}, optional,
            unsupported_atoms=sum(
                len(row.get("construction_problems") or []) for row in proposals),
            unresolved_readings=sum(
                len(row.get("unresolved_readings") or []) for row in proposals),
        ),
        "proposals": proposals,
    }
