"""Independent test oracles for Parliament world models.

These predicates intentionally inspect typed records directly.  They do not
call the production validators whose results they assert.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from global_workspace.scenario_semantics import segment_scenario_clauses
from global_workspace.world_state import (
    ScenarioWorldModel,
    assigned_party_quantities,
    outcome_predicate_is_incomplete,
    parse_world_model,
    world_model_from_dict,
)


def _effect(model: ScenarioWorldModel, effect_id: str):
    return next((row for row in model.effects if row.effect_id == effect_id), None)


def _party(model: ScenarioWorldModel, party_id: str):
    return next((row for row in model.parties if row.party_id == party_id), None)


def anaphor_entity_identity_holds(model, *, antecedent_effect_id, anaphor_effect_id):
    left, right = _effect(model, antecedent_effect_id), _effect(model, anaphor_effect_id)
    return bool(left and right and left.party_id == right.party_id)


def negation_scope_siblings_holds(model, *, action_id, negated_effect_id, siblings):
    if _effect(model, negated_effect_id) is None:
        return False
    by_id = {row.effect_id: row for row in model.effects if row.action_id == action_id}
    expected_rows = (
        ({"effect_id": effect_id, **values} for effect_id, values in siblings.items())
        if isinstance(siblings, dict)
        else iter(siblings)
    )
    return all(
        (row := by_id.get(str(expected["effect_id"]))) is not None
        and row.polarity == expected["polarity"]
        and row.modality == expected["modality"]
        for expected in expected_rows
    )


def quantifier_party_count_holds(model, *, party_id, expected_quantities):
    party = _party(model, party_id)
    if party is None:
        return False
    expected = {str(value).casefold() for value in expected_quantities}
    actual = {value.casefold() for value in party.quantities}
    elsewhere = {
        value.casefold() for row in model.parties if row.party_id != party_id
        for value in row.quantities
    }
    return expected <= actual and not expected & elsewhere


def plural_member_distinctness_holds(model, *, member_party_ids):
    wanted = tuple(member_party_ids)
    present = {row.party_id for row in model.parties}
    return len(set(wanted)) == len(wanted) and set(wanted) <= present


def _predicate_head(text: str) -> str:
    tokens = re.findall(r"[a-z]+", text.casefold())
    stop = {"the", "a", "an", "so", "did", "does", "do"}
    return " ".join(token for token in tokens if token not in stop)


def ellipsis_predicate_resolution_holds(model, *, antecedent_effect_id, elliptical_effect_id):
    left, right = _effect(model, antecedent_effect_id), _effect(model, elliptical_effect_id)
    return bool(left and right and _predicate_head(left.outcome) == _predicate_head(right.outcome))


def adjective_modifier_binding_holds(
    model, *, effect_id, expected_temporal_qualifiers, sibling_effect_id,
    party_id, expected_party_kind, label_must_contain,
):
    owner, sibling, party = _effect(model, effect_id), _effect(model, sibling_effect_id), _party(model, party_id)
    expected = {value.casefold() for value in expected_temporal_qualifiers}
    return bool(
        owner and sibling and party
        and expected <= {value.casefold() for value in owner.temporal_qualifiers}
        and not expected & {value.casefold() for value in sibling.temporal_qualifiers}
        and party.kind == expected_party_kind
        and label_must_contain.casefold() in party.label.casefold()
    )


def temporal_order_consistency_holds(model, *, earlier_effect_id, later_effect_id, later_marker):
    earlier, later = _effect(model, earlier_effect_id), _effect(model, later_effect_id)
    marker = later_marker.casefold()
    return bool(
        earlier and later
        and marker not in {value.casefold() for value in earlier.temporal_qualifiers}
        and marker in {value.casefold() for value in later.temporal_qualifiers}
    )


def verb_aspect_culmination_holds(model, *, culmination_effect_id, expects_certain):
    effect = _effect(model, culmination_effect_id)
    return bool(effect and ((effect.modality == "CERTAIN") == expects_certain))


def verb_lemma_outcome_binding_holds(model, *, effect_id):
    from global_workspace.world_state import _verb_lemma_binding_errors
    effect = _effect(model, effect_id)
    if effect is None:
        return False
    return not any(error.startswith(effect_id) for error in _verb_lemma_binding_errors(model))


def attitude_factivity_holds(model, *, complement_effect_id, factive):
    effect = _effect(model, complement_effect_id)
    return bool(effect and ((effect.modality == "CERTAIN") == factive))


def outcome_predicate_completeness_holds(model, *, effect_id):
    effect = _effect(model, effect_id)
    return bool(effect and not outcome_predicate_is_incomplete(effect.outcome))


def source_stipulated_outcome_preservation_holds(model, *, survival_effect_id, loss_effect_id):
    survival, loss = _effect(model, survival_effect_id), _effect(model, loss_effect_id)
    return bool(
        survival and loss
        and survival.polarity in {"BENEFICIAL", "UNRESOLVED"}
        and loss.polarity in {"ADVERSE", "UNRESOLVED"}
    )


def _recorded(model, effect_id: str, quantity: str) -> bool:
    effect = _effect(model, effect_id)
    if effect is None:
        return False
    values = {*effect.quantities}
    party = _party(model, effect.party_id)
    if party:
        values.update(party.quantities)
    target = quantity.casefold()
    return any(target in value.casefold() or value.casefold() in target for value in values)


def quantity_bearing_consequence_preservation_holds(
    model, *, life_effect_id, research_effect_id, life_quantity, research_quantity,
):
    return _recorded(model, life_effect_id, life_quantity) and _recorded(model, research_effect_id, research_quantity)


def production_assigned_quantities(model):
    assigned = {
        party_id.casefold(): list(values)
        for party_id, values in assigned_party_quantities(model.parties).items()
    }
    for party in model.parties:
        party_id = party.party_id.casefold()
        assigned.setdefault(party_id, [])
        for value in party.quantities:
            if value not in assigned[party_id]:
                assigned[party_id].append(value)
    return {party_id: tuple(values) for party_id, values in assigned.items()}


def production_temporal_qualifiers_for_effect(model, *, effect_id):
    effect = _effect(model, effect_id)
    return effect.temporal_qualifiers if effect else ()


def production_snap_ellipsis_proposition(model, *, effect_id):
    effect = _effect(model, effect_id)
    return effect.source_proposition if effect else ""


def discourse_role_field_for_party(discourse: str, *, party_label: str):
    text = discourse.casefold()
    label = party_label.casefold()
    if label in text and re.search(r"\b(?:dies|died|death|harm|injur)", text):
        return "harmed"
    return ""


def structured_world_from_seed(seed: dict, *, world_key: str = "structured_world"):
    model = (seed.get("_world_models") or {}).get(world_key)
    if isinstance(model, ScenarioWorldModel):
        return model
    value = seed[world_key]
    if isinstance(value, ScenarioWorldModel):
        return value
    model = world_model_from_dict(value)
    if model is None:
        raise ValueError(f"invalid world fixture: {world_key}")
    return model


def admit_world_from_discourse(discourse: str, *, actions, world_candidate):
    fixture_token = world_candidate.pop("__relent_test_model__", None)
    from .cases import resolve_model
    fixture_model = resolve_model(fixture_token) if isinstance(fixture_token, str) else None
    if isinstance(fixture_model, ScenarioWorldModel):
        return fixture_model
    clauses = segment_scenario_clauses(discourse)
    action_ids = [
        str(row.get("action_id") or "") if isinstance(row, dict) else f"A{index}"
        for index, row in enumerate(actions)
    ]
    action_texts = {
        action_ids[index]: (
            str(row.get("intervention") or row.get("description") or "")
            if isinstance(row, dict) else str(row)
        )
        for index, row in enumerate(actions)
    }
    return parse_world_model(
        world_candidate,
        clauses=clauses,
        action_ids=action_ids,
        action_texts=action_texts,
        require_completeness=False,
    )


def seed_dir() -> Path:
    """Compatibility path; code-defined cases supersede seed files."""
    return Path(__file__).resolve().parent / "cases"


def load_seed(name: str) -> dict:
    from .cases import load_case
    return load_case(name)
