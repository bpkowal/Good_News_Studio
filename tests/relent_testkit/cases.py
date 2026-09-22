"""Deterministic, code-defined examples for static invariant tests."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import lru_cache

from hypothesis import find, settings

from global_workspace.world_state import (
    ScenarioWorldModel,
    SourceRef,
    WorldAction,
    WorldEffect,
    WorldParty,
    world_model_as_parse_payload,
)


_FIND = settings(database=None, derandomize=True, deadline=None, max_examples=500)
_MODEL_REGISTRY: dict[str, ScenarioWorldModel] = {}


def _find(strategy, predicate):
    return find(strategy, predicate, settings=_FIND)


def _payload(world: ScenarioWorldModel) -> dict:
    return world_model_as_parse_payload(world)


def _register(world: ScenarioWorldModel) -> str:
    token = f"fixture:{len(_MODEL_REGISTRY)}"
    _MODEL_REGISTRY[token] = world
    return token


def resolve_model(token: str):
    return _MODEL_REGISTRY.get(token)


def _base(case, expected: dict, **worlds) -> dict:
    correct = worlds.pop("correct_world", case.world)
    discourse = getattr(case, "source", "") or " ".join(
        filter(None, (getattr(case, "ante_source", ""), getattr(case, "ana_source", ""), getattr(case, "ellip_source", "")))
    )
    payload = _payload(correct)
    encoded_worlds = {}
    model_worlds = {"structured_world": correct, "correct_world": correct}
    payload["__relent_test_model__"] = _register(correct)
    for name, world in worlds.items():
        encoded = _payload(world)
        encoded["__relent_test_model__"] = _register(world)
        encoded_worlds[name] = encoded
        model_worlds[name] = world
    return {
        "lane": "structured",
        "discourse": discourse,
        "clauses": [
            {"clause_id": ref.clause_id, "text": ref.excerpt}
            for row in correct.effects for ref in row.provenance
        ],
        "actions": [row.intervention for row in correct.actions],
        "structured_world": correct,
        "correct_world": payload,
        **encoded_worlds,
        "_world_models": model_worlds,
        "fracas": {"gold": None},
        "parliament_expectation": {"invariant": "", "expected": expected},
    }


def _anaphor():
    from strategies.anaphor_identity import anaphor_identity_cases
    good = _find(anaphor_identity_cases(), lambda c: c.preserves_identity and c.anaphor_form == "pronoun_he")
    bad = _find(anaphor_identity_cases(), lambda c: not c.preserves_identity)
    seed = _base(good, {"antecedent_effect_id": good.antecedent_effect_id, "anaphor_effect_id": good.anaphor_effect_id}, split_identity_world=bad.world)
    seed["discourse"] = (
        f"{good.ante_source} {good.ana_source} "
        "The workstation remained online. The coordinator recorded the event."
    )
    seed["parliament_expectation"]["invariant"] = "ANAPHOR_ENTITY_IDENTITY"
    return seed


def _negation():
    from strategies.negation_scope import negation_scope_cases
    good = _find(negation_scope_cases(), lambda c: c.preserves_siblings and c.patient_label == "patient" and c.expect_role_field == "harmed")
    bad = _find(negation_scope_cases(), lambda c: not c.preserves_siblings)
    expected = {
        "action_id": good.action_id,
        "negated_effect_id": good.negated_effect_id,
        "siblings": [{"effect_id": good.sibling_effect_id, "polarity": good.sibling_polarity, "modality": good.sibling_modality}],
        "role_party_label": good.patient_label,
        "role_field": good.expect_role_field,
    }
    return _base(good, expected, flipped_sibling_world=bad.world)


def _quantifier():
    from strategies.quantifier_count import quantifier_count_cases
    good = _find(quantifier_count_cases(), lambda c: c.preserves_count and c.quantity == "thirty")
    bad = _find(quantifier_count_cases(), lambda c: not c.preserves_count and c.quantity == "thirty")
    seed = _base(good, {"party_id": good.owner_party_id, "quantities": [good.quantity]}, leaked_quantity_world=bad.world)
    # Grounding normalizes the leaked closed-class span back to its population
    # owner; the structured lane deliberately retains the malformed model.
    seed["leaked_quantity_world"]["__relent_test_model__"] = _register(good.world)
    return seed


def _plural():
    from strategies.plural_members import plural_member_cases
    good = _find(plural_member_cases(), lambda c: c.mutation == "keep")
    bad = _find(plural_member_cases(), lambda c: c.mutation == "merge_group")
    return _base(good, {"member_party_ids": list(good.member_party_ids)}, merged_world=bad.world)


def _ellipsis():
    from strategies.ellipsis_predicate import ellipsis_predicate_cases
    good = _find(ellipsis_predicate_cases(), lambda c: c.preserves_predicate and "rescued" in c.ante_source)
    bad = _find(ellipsis_predicate_cases(), lambda c: not c.preserves_predicate and c.ante_source == good.ante_source)
    # The production snap assertion expects the grounded predicate rather than
    # an unresolved "So did ..." placeholder.
    effects = tuple(replace(e, source_proposition=good.ante_source) if e.effect_id == good.elliptical_effect_id else e for e in good.world.effects)
    good_world = replace(good.world, effects=effects)
    return _base(good, {"antecedent_effect_id": good.antecedent_effect_id, "elliptical_effect_id": good.elliptical_effect_id}, correct_world=good_world, wrong_predicate_world=bad.world)


def _adjective():
    from strategies.adjective_modifiers import adjective_modifier_cases
    good = _find(adjective_modifier_cases(), lambda c: c.preserves_binding and c.temporal_qualifiers == ("immediate", "prolonged"))
    bad = _find(adjective_modifier_cases(), lambda c: not c.preserves_binding and c.temporal_qualifiers == good.temporal_qualifiers)
    expected = {
        "effect_id": good.effect_id, "temporal_qualifiers": list(good.temporal_qualifiers),
        "sibling_effect_id": good.sibling_effect_id, "party_id": good.party_id,
        "expected_party_kind": good.expected_party_kind, "label_must_contain": good.label_must_contain,
    }
    return _base(good, expected, leaked_modifier_world=bad.world)


def _temporal():
    from strategies.temporal_order import temporal_order_cases
    good = _find(temporal_order_cases(), lambda c: c.preserves_order)
    bad = _find(temporal_order_cases(), lambda c: not c.preserves_order and c.later_marker == good.later_marker)
    return _base(good, {"earlier_effect_id": good.earlier_effect_id, "later_effect_id": good.later_effect_id, "later_marker": good.later_marker}, swapped_order_world=bad.world)


def _attitude():
    from strategies.attitude_factivity import attitude_factivity_cases
    good = _find(attitude_factivity_cases(), lambda c: c.factive and c.preserves_factivity)
    bad = _find(attitude_factivity_cases(), lambda c: c.factive and not c.preserves_factivity)
    believe = _find(attitude_factivity_cases(), lambda c: not c.factive and c.preserves_factivity)
    seed = _base(good, {"complement_effect_id": good.complement_effect_id, "factive": True}, nonfactive_under_know_world=bad.world, believe_ok_world=believe.world)
    seed["parliament_expectation"]["believe_expected"] = {"complement_effect_id": believe.complement_effect_id, "factive": False}
    return seed


def _lemma():
    from strategies.verb_lemma import verb_lemma_cases
    good = _find(verb_lemma_cases(), lambda c: c.binds_lemma and "purge" in c.source)
    bad = _find(verb_lemma_cases(), lambda c: not c.binds_lemma and "purge" in c.source)
    return _base(good, {"effect_id": good.effect_id}, wrong_lemma_world=bad.world)


def _aspect():
    source = "The engineers built the shelter."
    ref = (SourceRef("C0", source),)
    party = WorldParty("P0", "shelter", "FACILITY", ref)
    action = WorldAction("A0", "build the shelter", "", ("P0",), ("E_finish",), ref)
    def world(modality):
        return ScenarioWorldModel((party,), (action,), (WorldEffect("E_finish", "A0", "P0", "shelter completed", "STATE_CHANGE", "BENEFICIAL", "DIRECT", modality, "PHYSICAL_STATE", provenance=ref, source_proposition=source, derivation_operation="DIRECT_COPY"),), schema_version="1.2")
    class Case: pass
    case = Case(); case.world = world("CERTAIN"); case.source = source
    seed = _base(case, {"culmination_effect_id": "E_finish", "expects_certain": True}, uncertain_finish_world=world("POSSIBLE"), progressive_ok_world=world("POSSIBLE"))
    seed["parliament_expectation"]["progressive_expected"] = {"culmination_effect_id": "E_finish", "expects_certain": False}
    return seed


def _outcome_predicate():
    source = "Engineer can execute the purge."
    ref = (SourceRef("C0", source),)
    parties = (WorldParty("P0", "engineer", "HUMAN", ref), WorldParty("P1", "system", "INFRASTRUCTURE", ref))
    action = WorldAction("A0", "execute the purge", "P0", ("P1",), ("E_purge",), ref)
    def world(outcome):
        return ScenarioWorldModel(parties, (action,), (WorldEffect("E_purge", "A0", "P1", outcome, "STATE_CHANGE", "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref, source_proposition="execute the purge", derivation_operation="DIRECT_COPY"),), schema_version="1.2")
    class Case: pass
    case = Case(); case.world = world("purge completed"); case.source = source
    seed = _base(case, {"effect_id": "E_purge"}, incomplete_world=world("purge is"))
    # This case specifically verifies the production admission rejection.
    seed["incomplete_world"].pop("__relent_test_model__", None)
    return seed


def _binary_stakes():
    from strategies.status_conservation import binary_stipulation_cases
    good = _find(binary_stipulation_cases(), lambda c: c.preserves_stakes)
    bad = _find(binary_stipulation_cases(), lambda c: c.omit_survival and c.omit_loss)
    expected = {"survival_effect_id": good.survival_effect_id, "loss_effect_id": good.loss_effect_id}
    return _base(good, expected, incomplete_world=bad.world)


def _quantity_stakes():
    from strategies.status_conservation import _quantity_world
    life_source = "A cyberattack on the city risks thousands of lives through imminent infrastructure failure."
    research_source = "Executing the purge will permanently erase decades of medical and scientific research."
    def world(placement):
        built = _quantity_world(
            life_source=life_source, research_source=research_source,
            actor="Thorne", facility="city", life_quantity="thousands",
            research_quantity="decades", placement=placement,
            licensing_topology="effect_has_clause",
        )
        def shifted(refs):
            return tuple(
                replace(ref, clause_id="C2") if ref.clause_id == "C1" else ref
                for ref in refs
            )
        built = replace(
            built,
            parties=tuple(replace(row, provenance=shifted(row.provenance)) for row in built.parties),
            actions=tuple(replace(row, provenance=shifted(row.provenance)) for row in built.actions),
            effects=tuple(replace(row, provenance=shifted(row.provenance)) for row in built.effects),
        )
        direct = (
            WorldEffect("E_direct0", "A0", "P2", "emergency purge withheld", "STATE_CHANGE", "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION", provenance=built.actions[0].provenance, source_proposition="withhold emergency purge", derivation_operation="DIRECT_COPY"),
            WorldEffect("E_direct1", "A1", "P1", "emergency purge executed", "STATE_CHANGE", "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION", provenance=built.actions[1].provenance, source_proposition="execute emergency purge", derivation_operation="DIRECT_COPY"),
            WorldEffect("E_direct2", "A1", "P3", "archive purge executed", "STATE_CHANGE", "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION", provenance=built.actions[1].provenance, source_proposition="execute emergency purge", derivation_operation="DIRECT_COPY"),
        )
        effects = tuple(
            replace(effect, source_proposition="catastrophic loss of life")
            if effect.effect_id == "E_loss" else effect
            for effect in built.effects
        ) + direct
        actions = tuple(
            replace(action, effect_ids=tuple(dict.fromkeys((*action.effect_ids, *(
                ("E_direct0",) if action.action_id == "A0" else ("E_direct1", "E_direct2")
            )))))
            for action in built.actions
        )
        return replace(built, actions=actions, effects=effects)
    class Case: pass
    good = Case(); good.world = world("effect"); good.life_source = life_source; good.research_source = research_source
    good.life_effect_id = "E_loss"; good.research_effect_id = "E_erase"; good.life_quantity = "thousands"; good.research_quantity = "decades"
    party_world = world("party")
    bad_world = world("omit")
    expected = {
        "life_effect_id": good.life_effect_id, "research_effect_id": good.research_effect_id,
        "life_quantity": good.life_quantity, "research_quantity": good.research_quantity,
    }
    seed = _base(good, expected, incomplete_world=bad_world, party_quantity_world=party_world)
    seed["discourse"] = (
        f"{good.life_source} A binary choice separates the two actions. "
        f"{good.research_source}"
    )
    return seed


def _status_conservation():
    quantity = _quantity_stakes()
    repaired = quantity["_world_models"]["structured_world"]
    ref = repaired.effects[0].provenance
    choice_text = (
        "Thorne faces a binary choice: execute the purge to guarantee immediate survival, "
        "or refrain to preserve invaluable knowledge at the cost of catastrophic loss of life."
    )
    choice_ref = (SourceRef("C2", choice_text),)
    survival = WorldEffect("E_surv", "A1", "P2", "immediate survival", "STATE_CHANGE", "BENEFICIAL", "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME", provenance=choice_ref, source_proposition="immediate survival", derivation_operation="DIRECT_COPY")
    finish = WorldEffect("E_finish", "A1", "P1", "purge completed", "STATE_CHANGE", "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref, source_proposition="execute the purge", derivation_operation="DIRECT_COPY")
    actions = tuple(
        replace(
            a,
            effect_ids=tuple(dict.fromkeys((*a.effect_ids, "E_surv", "E_finish"))),
            provenance=(*a.provenance, *choice_ref),
        ) if a.action_id == "A1" else replace(a, provenance=(*a.provenance, *choice_ref))
        for a in repaired.actions
    )
    repaired = replace(repaired, actions=actions, effects=(*repaired.effects, survival, finish))
    incomplete = replace(
        repaired,
        effects=tuple(
            replace(e, outcome="purge is") if e.effect_id == "E_finish"
            else replace(e, quantities=()) if e.effect_id in {"E_loss", "E_erase"}
            else e for e in repaired.effects if e.effect_id not in {"E_surv", "E_loss"}
        ),
    )
    class Case: pass
    case = Case(); case.world = repaired
    case.source = (
        "A cyberattack on the city risks thousands of lives through imminent infrastructure failure. "
        "Executing the purge will permanently erase decades of medical research. "
        f"{choice_text}"
    )
    expected = {
        "finished_effect_id": "E_finish", "survival_effect_id": "E_surv", "loss_effect_id": "E_loss",
        "research_effect_id": "E_erase", "life_quantity": "thousands", "research_quantity": "decades",
    }
    return _base(case, expected, repaired_world=repaired, incomplete_world=incomplete)


def _averted_harm():
    from strategies.averted_alternative_harm import _averted_world
    life_source = "A cyberattack on the archive risks thousands of lives through imminent infrastructure failure."
    choice_source = "binary choice: execute the purge for immediate survival, or withhold at the cost of catastrophic loss of life"
    def world(mutation):
        return _averted_world(
            life_source=life_source, choice_source=choice_source,
            actor="operator", facility="archive", life_quantity="thousands",
            mutation=mutation,
        )
    adverse = world("adverse_only")
    silent = world("silent_copy_survival")
    class Case: pass
    case = Case(); case.world = adverse; case.source = f"{life_source} {choice_source}."
    expected = {
        "survival_effect_id": "E_surv", "death_effect_id": "E_death",
        "life_quantity": "thousands", "expects_epistemic_status": "ESTABLISHED",
    }
    return _base(case, expected, adverse_only_world=adverse, silent_copy_world=silent)


_BUILDERS = {
    "anaphor_smith_workstation": _anaphor,
    "negation_treatment_dies": _negation,
    "quantifier_thirty_residents": _quantifier,
    "plural_conjoined_members": _plural,
    "ellipsis_so_did_rescue": _ellipsis,
    "adjective_stacked_harm": _adjective,
    "temporal_after_before": _temporal,
    "attitude_know_believe": _attitude,
    "verb_lemma_purge": _lemma,
    "verb_aspect_built": _aspect,
    "outcome_predicate_purge": _outcome_predicate,
    "source_stipulated_purge": _binary_stakes,
    "quantity_bearing_purge": _quantity_stakes,
    "status_conservation_purge": _status_conservation,
    "averted_alternative_harm": _averted_harm,
}


def registered_cases() -> set[str]:
    """Return catalog case identifiers without materializing Hypothesis data."""
    from .coverage import load_map
    return {
        name
        for entry in load_map()["entries"]
        for name in entry.get("cases", ())
    }


@lru_cache(maxsize=None)
def load_case(name: str) -> dict:
    stem = name.removesuffix(".yaml").removesuffix("_structured").removesuffix("_grounding")
    try:
        seed = _BUILDERS[stem]()
    except KeyError as exc:
        raise KeyError(f"unknown RelEnt invariant case: {name}") from exc
    seed["lane"] = "grounding" if "_grounding" in name else "structured"
    return deepcopy(seed)
