"""Evidence-filled graph blueprints over the stable Parliament schema.

The first functional blueprint is exclusive allocation.  It proposes a complete
candidate rather than treating absent derived edges as blockers.  Every slot keeps
its Z10 or source-text witness, and Parliament remains responsible for validating
the resulting schema.  Schema 1.3 is unchanged.  A stated survival chance is a
PROBABILISTIC health outcome with that hedge copied onto likelihood_qualifiers.
A stated death for the patient who does not receive the dose is a CERTAIN adverse
health outcome caused by nonreceipt.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

from z10_world_model_adapter import (
    _enumerate_assignments,
    _participants_by_proposition,
    _predications,
    _selection,
    _support_span,
    segment_source_clauses,
)


BLUEPRINT_VERSION = "candidate-graph-blueprints/0.2"
# A survival chance is copied as written ("95% chance"). Parliament already
# types that hedge as PROBABILISTIC on schema 1.3; it is not a new world schema
# and it is not stored as a party quantity.
_SURVIVAL_CHANCE = re.compile(r"\d+(?:\.\d+)?%\s*chance", re.IGNORECASE)
# Death of the patient who does not receive the dose. The clause need not name
# Ben or Cara; the exclusive branch identifies that patient.
_NONRECEIPT_DEATH = re.compile(
    r"\b(?:does not|doesn't|do not)\b[\s\S]{0,80}\b(?:will die|dies)\b"
    r"|\b(?:will die|dies)\b[\s\S]{0,80}\b(?:does not|doesn't|do not)\b",
    re.IGNORECASE,
)


def _clean_label(text: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", text.strip(), flags=re.IGNORECASE)


def _clause_ids_for_text(package: dict, pattern: str) -> list[str]:
    return [row["clause_id"] for row in segment_source_clauses(package["document"]["text"])
            if re.search(pattern, row["text"], flags=re.IGNORECASE)]


def _clause_holding(clauses: list[dict], snippet: str) -> dict | None:
    """Return the source clause that contains this evidence span."""
    for row in clauses:
        if snippet and snippet in row["text"]:
            return row
    return None


def _nonreceipt_death_clause(clauses: list[dict]) -> dict | None:
    for row in clauses:
        if _NONRECEIPT_DEATH.search(row["text"]):
            return row
    return None


def _quantity_text(value: dict) -> str:
    unit = str(value.get("unit") or "").strip()
    return f"{value['amount']} {unit}".strip()


def _role_rows(package: dict, proposition: str) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for row in _participants_by_proposition(package).get(proposition, []):
        result.setdefault(row["value"], []).append(row)
    return result


def _mention_node(package: dict, role: dict) -> dict:
    nodes = {row["id"]: row for row in package["nodes"]}
    return nodes[role["arguments"]["mention"]]


def _candidate_id_rows(package: dict) -> dict[str, dict]:
    return {row["id"]: row for row in package["candidates"]}


def match_exclusive_allocation(package: dict, actions: Sequence[str]) -> dict:
    """Match structural signals and return explicit filled/unfilled slots."""
    assignments = _enumerate_assignments(package, actions, 8)
    candidates = package["candidates"]
    quantity = [row for row in candidates if row["type"] == "QUANTITY"
                and row["value"].get("operator") == "exact"]
    option_props = {row["arguments"]["proposition"] for row in candidates
                    if row["type"] == "OPTION_OF"}
    conditional = {row["arguments"]["condition"]: row for row in candidates
                   if row["type"] == "CONDITIONAL_ON"}
    exclusive_evidence = _clause_ids_for_text(
        package, r"\b(?:but\s+)?not\s+both\b|\beither\b[\s\S]*\bor\b")
    valid_assignments = [values for values in assignments
                         if all(prop in option_props and prop in conditional for prop in values)]
    required = {
        "two_action_options": bool(valid_assignments and len(actions) == 2),
        "exact_quantity": bool(quantity),
        "explicit_exclusivity": bool(exclusive_evidence),
        "conditional_outcomes": bool(valid_assignments),
    }
    return {
        "blueprint_id": "exclusive_allocation",
        "matched": all(required.values()),
        "required_slots": required,
        "unfilled_required_slots": [key for key, filled in required.items() if not filled],
        "assignments": [list(values) for values in valid_assignments],
        "quantity_candidate_ids": [row["id"] for row in quantity],
        "exclusivity_clause_ids": exclusive_evidence,
    }


def instantiate_exclusive_allocation(package: dict, actions: Sequence[str]) -> dict:
    """Fill one exclusive-allocation graph and emit a Parliament 1.3 proposal."""
    match = match_exclusive_allocation(package, actions)
    if not match["matched"]:
        return {"blueprint_version": BLUEPRINT_VERSION, "blueprint_id": match["blueprint_id"],
                "status": "NO_MATCH", "match": match, "proposals": []}

    assignment = tuple(match["assignments"][0])
    nodes = {row["id"]: row for row in package["nodes"]}
    candidates = _candidate_id_rows(package)
    predications = _predications(package)
    conditional = {row["arguments"]["condition"]: row for row in package["candidates"]
                   if row["type"] == "CONDITIONAL_ON"}
    option_rows = {row["arguments"]["proposition"]: row for row in package["candidates"]
                   if row["type"] == "OPTION_OF"}
    quantity_row = candidates[match["quantity_candidate_ids"][0]]
    quantity_mention = nodes[quantity_row["arguments"]["mention"]]

    action_roles = [_role_rows(package, proposition) for proposition in assignment]
    actors, resources, recipients = [], [], []
    for roles in action_roles:
        actor = (roles.get("agent") or roles.get("controller") or roles.get("subject") or [None])[0]
        resource = (roles.get("object") or [None])[0]
        recipient = (roles.get("destination") or roles.get("patient") or [None])[0]
        if not actor or not resource or not recipient:
            raise ValueError("Matched allocation lacks actor, resource, or destination role")
        actors.append(_mention_node(package, actor))
        resources.append(_mention_node(package, resource))
        recipients.append(_mention_node(package, recipient))
    if len({node["label"].casefold() for node in actors}) != 1:
        raise ValueError("Allocation alternatives do not share one actor")
    if len({_clean_label(node["label"]).casefold() for node in resources}) != 1:
        raise ValueError("Allocation alternatives do not share one resource")
    if recipients[0]["label"].casefold() == recipients[1]["label"].casefold():
        raise ValueError("Allocation alternatives do not have distinct recipients")

    resource_label = resources[0]["label"]
    resource_head = _clean_label(resource_label)
    quantity_support = _clause_ids_for_text(
        package,
        re.escape(quantity_mention["label"]) + r"\s+of\s+(?:the\s+)?" + re.escape(resource_head),
    )
    if not quantity_support:
        raise ValueError("Exact quantity is not textually attached to the allocated resource")

    actor_label = actors[0]["label"]
    party_specs = [(actor_label, "PERSON", []),
                   (recipients[0]["label"], "PERSON", []),
                   (recipients[1]["label"], "PERSON", []),
                   (resource_label, "RESOURCE", [_quantity_text(quantity_row["value"])])]
    parties = []
    party_id: dict[str, str] = {}
    clauses = segment_source_clauses(package["document"]["text"])
    for index, (label, kind, quantities) in enumerate(party_specs, 1):
        ident = f"P{index}"
        party_id[label.casefold()] = ident
        clause_ids = [row["clause_id"] for row in clauses
                      if re.search(r"\b" + re.escape(_clean_label(label)) + r"\b",
                                   row["text"], flags=re.IGNORECASE)]
        parties.append({"party_id": ident, "label": label, "kind": kind,
                        "quantities": quantities, "clause_ids": clause_ids})

    seed: set[str] = {quantity_row["id"]}
    world_actions, effects, causal_links, action_sources = [], [], [], {}
    slot_bindings = {
        "actor": {"label": actor_label, "mention_ids": [node["id"] for node in actors]},
        "resource": {"label": resource_label,
                     "mention_ids": [node["id"] for node in resources],
                     "quantity_candidate_id": quantity_row["id"],
                     "quantity_clause_ids": quantity_support},
        "exclusivity": {"clause_ids": match["exclusivity_clause_ids"]},
        "alternatives": [],
    }
    for index, (action_text, proposition, roles) in enumerate(
            zip(actions, assignment, action_roles)):
        action_id = f"A{index}"
        link = conditional[proposition]
        consequence = link["arguments"]["consequence"]
        consequence_roles = _role_rows(package, consequence)
        bearer_role = (consequence_roles.get("patient") or consequence_roles.get("subject")
                       or consequence_roles.get("destination") or [None])[0]
        if not bearer_role:
            raise ValueError("Conditional outcome lacks a bearer")
        bearer = _mention_node(package, bearer_role)
        recipient = recipients[index]
        if bearer["label"].casefold() != recipient["label"].casefold():
            raise ValueError("Outcome bearer does not match allocation recipient")

        role_rows = [row for values in roles.values() for row in values]
        consequence_role_rows = [row for values in consequence_roles.values() for row in values]
        modal_rows = [row for row in package["candidates"] if row["type"] == "MODALITY"
                      and row["arguments"]["proposition"] == consequence
                      and row["value"] == "prediction"]
        seed.update([predications[proposition]["id"], link["id"], option_rows[proposition]["id"],
                     predications[consequence]["id"]])
        seed.update(row["id"] for row in role_rows + consequence_role_rows + modal_rows)

        action_source, action_clause_ids = _support_span(
            package, [predications[proposition], *role_rows])
        _, outcome_clause_ids = _support_span(
            package, [link, predications[consequence], *consequence_role_rows, *modal_rows])
        action_clause_ids = list(dict.fromkeys(
            [*match["exclusivity_clause_ids"], *action_clause_ids, *outcome_clause_ids]))
        recipient_id = party_id[recipient["label"].casefold()]
        other_recipient = recipients[1 - index]
        other_recipient_id = party_id[other_recipient["label"].casefold()]
        resource_phrase = _clean_label(resource_label)
        quantity_clause = next(row for row in clauses
                               if row["clause_id"] == quantity_support[0])
        outcome_source = package["document"]["text"][
            min(next(e["start"] for e in package["evidence"] if e["id"] == ident)
                for row in [link, predications[consequence], *consequence_role_rows]
                for ident in row["evidence_ids"]):
            max(next(e["end"] for e in package["evidence"] if e["id"] == ident)
                for row in [link, predications[consequence], *consequence_role_rows]
                for ident in row["evidence_ids"])
        ]
        # "95% chance of survival" can extend past the parsed evidence span.
        # The admitted proposition has to be the whole clause that states it.
        outcome_clause = _clause_holding(clauses, outcome_source)
        chance = (_SURVIVAL_CHANCE.search(outcome_clause["text"])
                  if outcome_clause and re.search(r"surviv", outcome_clause["text"], re.IGNORECASE)
                  else None)
        if chance and outcome_clause is not None:
            survival_outcome = "survival"
            survival_predicate = "survival"
            survival_modality = "PROBABILISTIC"
            survival_likelihood = [chance.group(0)]
            survival_source = outcome_clause["text"]
            survival_clauses = [outcome_clause["clause_id"]]
            survival_explanation = (
                f"The source gives {recipient['label']} {chance.group(0)} of survival "
                "on this allocation, which is not a certain recovery."
            )
        else:
            survival_outcome = f"{recipient['label']} recovers"
            survival_predicate = "SURVIVES"
            survival_modality = "CERTAIN"
            survival_likelihood = []
            survival_source = outcome_source
            survival_clauses = outcome_clause_ids
            survival_explanation = "The source condition stipulates recovery for this allocation branch."
        death_clause = _nonreceipt_death_clause(clauses)
        slot_count = 4 if death_clause else 3
        direct_id = f"E{index * slot_count + 1}"
        nonreceipt_id = f"E{index * slot_count + 2}"
        outcome_id = f"E{index * slot_count + 3}"
        death_id = f"E{index * slot_count + 4}" if death_clause else None
        effects.extend([
            {"effect_id": direct_id, "action_id": action_id, "party_id": recipient_id,
             "outcome": f"receives {resource_phrase}", "predicate": "RECEIVES",
             "polarity": "BENEFICIAL", "directness": "DIRECT", "modality": "CERTAIN",
             "effect_kind": "RESOURCE_TRANSFER", "condition_ids": [], "quantities": [],
             "likelihood_qualifiers": [], "overall_likelihood_qualifiers": [],
             "scope_qualifiers": [], "temporal_qualifiers": [], "condition_join": "AND",
             "source_proposition": action_source, "source_effect_ids": [],
             "derivation_operation": "DIRECT_COPY", "derivation_explanation": "",
             "derivation_assumptions": [], "outcome_type_transformation": "PRESERVED",
             "clause_ids": outcome_clause_ids},
            {"effect_id": nonreceipt_id, "action_id": action_id,
             "party_id": other_recipient_id,
             "outcome": f"does not receive {resource_phrase}",
             "predicate": "NOT_RECEIVES", "polarity": "ADVERSE",
             "directness": "DOWNSTREAM", "modality": "CERTAIN",
             "effect_kind": "OTHER", "condition_ids": [],
             "quantities": [_quantity_text(quantity_row["value"])],
             "likelihood_qualifiers": [], "overall_likelihood_qualifiers": [],
             "scope_qualifiers": [], "temporal_qualifiers": [], "condition_join": "AND",
             "source_proposition": quantity_clause["text"],
             "source_effect_ids": [direct_id],
             "derivation_operation": "EXCLUSIVE_ALLOCATION_COMPLEMENT",
             "derivation_explanation": (
                 f"Only {_quantity_text(quantity_row['value'])} exists; allocating it to "
                 f"{recipient['label']} precludes simultaneous allocation to {other_recipient['label']}."),
             "derivation_assumptions": [], "outcome_type_transformation": "PRESERVED",
             "clause_ids": list(dict.fromkeys(
                 [*quantity_support, *match["exclusivity_clause_ids"], *outcome_clause_ids]))},
            {"effect_id": outcome_id, "action_id": action_id, "party_id": recipient_id,
             "outcome": survival_outcome, "predicate": survival_predicate,
             "polarity": "BENEFICIAL", "directness": "DOWNSTREAM", "modality": survival_modality,
             "effect_kind": "HEALTH_OUTCOME", "condition_ids": [], "quantities": [],
             "likelihood_qualifiers": survival_likelihood, "overall_likelihood_qualifiers": [],
             "scope_qualifiers": [], "temporal_qualifiers": [], "condition_join": "AND",
             "source_proposition": survival_source, "source_effect_ids": [direct_id],
             "derivation_operation": "SOURCE_STIPULATED_CAUSAL",
             "derivation_explanation": survival_explanation,
             "derivation_assumptions": [], "outcome_type_transformation": "PRESERVED",
             "clause_ids": survival_clauses},
        ])
        effect_ids = [direct_id, nonreceipt_id, outcome_id]
        causal_links.append({"action_id": action_id, "source_id": direct_id,
                             "link_relation": "CAUSES", "target_id": outcome_id,
                             "modality": "CERTAIN", "condition_ids": [],
                             "clause_ids": survival_clauses})
        if death_clause is not None and death_id is not None:
            # The death sentence names "the patient," not Ben or Cara. This
            # branch's non-recipient is that patient, so the certain death hangs
            # off nonreceipt rather than off the probabilistic survival row.
            effects.append({
                "effect_id": death_id, "action_id": action_id,
                "party_id": other_recipient_id, "outcome": "will die", "predicate": "die",
                "polarity": "ADVERSE", "directness": "DOWNSTREAM", "modality": "CERTAIN",
                "effect_kind": "HEALTH_OUTCOME", "condition_ids": [], "quantities": [],
                "likelihood_qualifiers": [], "overall_likelihood_qualifiers": [],
                "scope_qualifiers": [], "temporal_qualifiers": [], "condition_join": "AND",
                "source_proposition": death_clause["text"],
                "source_effect_ids": [nonreceipt_id],
                "derivation_operation": "SOURCE_STIPULATED_CAUSAL",
                "derivation_explanation": (
                    f"The source says the patient who does not get the {resource_phrase} "
                    f"will die. On this branch that patient is {other_recipient['label']}."
                ),
                "derivation_assumptions": [], "outcome_type_transformation": "PRESERVED",
                "clause_ids": [death_clause["clause_id"]],
            })
            effect_ids.append(death_id)
            causal_links.append({
                "action_id": action_id, "source_id": nonreceipt_id,
                "link_relation": "CAUSES", "target_id": death_id,
                "modality": "CERTAIN", "condition_ids": [],
                "clause_ids": [death_clause["clause_id"]],
            })
        world_actions.append({"action_id": action_id, "intervention": action_text,
                              "actor_party_id": party_id[actor_label.casefold()],
                              "recipient_party_ids": [recipient_id],
                              "effect_ids": effect_ids,
                              "clause_ids": action_clause_ids})
        action_sources[action_id] = {"clause_ids": action_clause_ids,
                                     "reason": "Filled by exclusive-allocation blueprint."}
        slot_bindings["alternatives"].append({
            "action_id": action_id, "action_proposition_id": proposition,
            "recipient_mention_id": recipient["id"],
            "conditional_candidate_id": link["id"],
            "outcome_proposition_id": consequence,
            "option_candidate_id": option_rows[proposition]["id"],
            "direct_effect_id": direct_id, "nonreceipt_effect_id": nonreceipt_id,
            "outcome_effect_id": outcome_id,
            "survival_likelihood": survival_likelihood,
            "death_effect_id": death_id,
        })

    selection, selection_validation = _selection(package, seed)
    if not selection_validation["contract_valid"]:
        raise ValueError("Blueprint produced an invalid Z10 selection")
    world = {"schema_version": "1.3", "parties": parties, "actions": world_actions,
             "effects": effects, "conditions": [], "temporal_relations": [],
             "causal_links": causal_links, "counterfactual_links": []}
    proposal = {
        "proposal_id": "exclusive_allocation_0", "status": "FILLED",
        "assignment": list(assignment), "slot_bindings": slot_bindings,
        "selection": selection, "selection_validation": selection_validation,
        "candidate": {"actions": action_sources, "world_model": world,
                      "ellipsis_resolutions": []},
        "clauses": [{"clause_id": row["clause_id"], "text": row["text"]}
                    for row in clauses],
        "unfilled_required_slots": [],
    }
    return {"blueprint_version": BLUEPRINT_VERSION,
            "blueprint_id": "exclusive_allocation", "status": "FILLED",
            "match": match, "proposals": [proposal]}
