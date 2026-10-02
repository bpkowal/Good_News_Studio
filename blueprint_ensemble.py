"""Blank ethical blueprints, filled from a Z10 package, then ranked.

Each blueprint starts as an empty slot list. A filler writes a Parliament 1.3
candidate only for slots the scenario actually supports. The chooser keeps every
attempt and selects the filled plan whose required slots are the most specific.
Unfilled slots stay visible. Nothing here admits a world or picks a moral act.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

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


ENSEMBLE_VERSION = "blueprint-ensemble/0.1"

# Empty plans. Values stay None until a scenario fills them.
BLANK_BLUEPRINTS: tuple[dict[str, Any], ...] = (
    {
        "blueprint_id": "exclusive_allocation",
        "summary": "One scarce resource, two recipients, and only one can receive it.",
        "required_slots": (
            "two_action_options", "exact_quantity",
            "explicit_exclusivity", "conditional_outcomes",
        ),
        "optional_slots": ("survival_chance", "nonreceipt_death"),
    },
    {
        "blueprint_id": "conditional_outcome",
        "summary": "An action is stated, and a bearer outcome is explicitly conditional on it.",
        "required_slots": ("action_condition", "outcome_bearer"),
        "optional_slots": ("second_conditional", "survival_chance"),
    },
    {
        "blueprint_id": "rescue_contrast",
        "summary": "Someone can rescue one party but not another, and survival is conditional on the rescue.",
        "required_slots": ("rescue_action", "contrasting_remnant", "conditional_survival"),
        "optional_slots": ("scene_parties", "foregone_harm"),
    },
    {
        "blueprint_id": "omission_harm",
        "summary": "Doing an action and not doing it are both stated, and each branch states who is harmed.",
        "required_slots": (
            "positive_action", "negated_action",
            "harm_if_done", "harm_if_omitted",
        ),
        "optional_slots": ("group_counts", "instrument_contrast"),
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
    rescue = _rescue_link(package)
    remnant = _contrasting_remnant(package, rescue)
    required = {
        "rescue_action": bool(rescue),
        "contrasting_remnant": bool(remnant),
        "conditional_survival": bool(rescue and rescue["survival"]),
    }
    scene = _scene_parties(package)
    optional = {
        "scene_parties": bool(scene),
        "foregone_harm": False,
    }
    if not all(required.values()):
        return _unmatched("rescue_contrast", required, optional)
    return _filled("rescue_contrast", required, optional,
                   _rescue_proposal(package, rescue, remnant, scene))


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


def _rescue_link(package: dict) -> dict[str, Any] | None:
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
            return row
    return None


def _contrasting_remnant(package: dict, rescue: dict[str, Any] | None) -> dict[str, Any] | None:
    if not rescue:
        return None
    nodes = {row["id"]: row for row in package["nodes"]}
    candidates = {row["id"]: row for row in package["candidates"]}
    saved = rescue["saved"]["label"].casefold()
    actor = rescue["actor"]["label"].casefold()
    for reconstruction in package.get("reconstructions", []):
        labels = []
        for ident in reconstruction.get("participant_candidate_ids", []):
            mention = candidates[ident]["arguments"].get("mention")
            if mention:
                labels.append(nodes[mention]["label"])
        novel = [label for label in labels
                 if label.casefold() not in {saved, actor}]
        if novel:
            return {"reconstruction": reconstruction, "label": novel[0]}
    return None


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


def _rescue_proposal(package: dict, rescue: dict[str, Any], remnant: dict[str, Any],
                     scene: list[str]) -> dict[str, Any]:
    branch = _branch(package, 0, rescue, f"save {rescue['saved']['label']}", "live", "BENEFICIAL")
    dog = remnant["label"]
    if dog.casefold() not in {row["label"].casefold() for row in branch["parties"]}:
        branch["parties"].append({
            "party_id": f"P{len(branch['parties']) + 1}",
            "label": dog, "kind": "ANIMAL", "quantities": [],
            "clause_ids": _clause_ids_containing(package, dog),
        })
    branch["slot_note"] = {
        "contrasting_remnant": remnant["label"],
        "reconstruction_id": remnant["reconstruction"]["id"],
        "scene_parties": scene,
        "foregone_harm": None,
    }
    return _proposal(package, "rescue_contrast_0", [branch], [rescue["link"]["id"]])


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
    effects[0]["party_id"] = "BEARER"
    return {
        "intervention": intervention,
        "actor_label": actor["label"],
        "bearer_label": bearer["label"],
        "parties": [actor, bearer],
        "effects": effects,
        "link": {"action_id": action_id, "source_id": direct_id, "target_id": outcome_id,
                 "clause_id": host["clause_id"]},
        "seed": [row["link"]["id"]],
    }


def _proposal(package: dict, proposal_id: str, branches: list[dict[str, Any]],
              seed_ids: list[str]) -> dict[str, Any]:
    parties: list[dict[str, Any]] = []
    party_id: dict[str, str] = {}
    for branch in branches:
        for node in branch["parties"]:
            label = node["label"]
            if label.casefold() in party_id:
                continue
            ident = f"P{len(parties) + 1}"
            party_id[label.casefold()] = ident
            kind = _kind(label, "")
            parties.append({
                "party_id": ident, "label": label, "kind": kind, "quantities": [],
                "clause_ids": _clause_ids_containing(package, label),
            })
    actions, effects, links = [], [], []
    for index, branch in enumerate(branches):
        action_id = f"A{index}"
        actor = party_id[branch["actor_label"].casefold()]
        bearer = party_id[branch["bearer_label"].casefold()]
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
                      "modality": "CERTAIN", "condition_ids": [], "clause_ids": [clause]})
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
    world = {"schema_version": "1.3", "parties": parties, "actions": actions,
             "effects": effects, "conditions": [], "temporal_relations": [],
             "causal_links": links, "counterfactual_links": []}
    return {
        "proposal_id": proposal_id, "status": "FILLED",
        "selection_validation": validation,
        "candidate": {"actions": {row["action_id"]: {"clause_ids": row["clause_ids"],
                                                     "reason": "Filled from the blank blueprint."}
                                  for row in actions},
                      "world_model": world, "ellipsis_resolutions": []},
        "unfilled_required_slots": [],
        "notes": [branch.get("slot_note") for branch in branches if branch.get("slot_note")],
    }


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
    folded = label.casefold()
    if "dog" in folded:
        return "ANIMAL"
    # A plural headcount is a group. "one worker" stays a person.
    if re.search(r"\b(?:workers|people|patients|residents)\b", folded) and "one" not in folded:
        return "HUMAN_GROUP"
    return "PERSON"


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
        "proposals": proposals,
    }
