"""License effect provenance from asserted propositions, not graph topology.

Builders may propose SOURCE_STIPULATED_CAUSAL because an effect has a parent.
That is not a license. The license is propositional:

- SOURCE_ASSERTED / DIRECT_COPY / SOURCE_STIPULATED_CAUSAL only when a source
  clause states the same party-outcome. Closed paraphrase counts: inflection,
  determiners, number-word synonyms, copied spans, generic human heads, and a
  pronoun whose antecedent is the transfer patient in the same if-clause.
- EXCLUSIVE_ALLOCATION_COMPLEMENT is STRUCTURALLY_DERIVED from discharged
  exclusivity, not from a copied non-receipt sentence.
- AVERTED_ALTERNATIVE_HARM is a WORLD_KNOWLEDGE_HYPOTHESIS. The opposed harm
  is the input. Polarity inversion is not a stipulated survival.

Never promote a hypothesis to a stipulated fact. Never demote a closed
paraphrase of an asserted death or recovery because the wording shifted.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence


STIPULATED_OPERATIONS = frozenset({"DIRECT_COPY", "SOURCE_STIPULATED_CAUSAL"})
STRUCTURAL_OPERATIONS = frozenset({"EXCLUSIVE_ALLOCATION_COMPLEMENT"})
HYPOTHESIS_OPERATIONS = frozenset({"AVERTED_ALTERNATIVE_HARM"})
INVERSE_FAMILIES = frozenset({
    ("die", "live"), ("live", "die"),
    ("die", "avert"), ("avert", "die"),
    ("receive", "not_receive"), ("not_receive", "receive"),
})
AVERTED_ASSUMPTIONS = (
    "the action branches partition the choice",
    "a CERTAIN harm stated on the opposed branch does not occur on this branch",
    "the opposed parties are not on the same fatal path",
)
_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_STOP = {
    "a", "an", "the", "to", "of", "or", "and", "who", "that", "in", "on", "for",
    "if", "will", "would", "can", "must", "does", "do", "not", "is", "are",
    "has", "have", "be", "been", "by", "with", "as", "at",
}
_NUMBER = {
    "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "one": "one", "two": "two", "three": "three", "four": "four", "five": "five",
    "six": "six", "seven": "seven", "eight": "eight", "nine": "nine", "ten": "ten",
}
_FAMILIES = {
    "die": {
        "die", "dies", "died", "dying", "dead", "death", "kill", "kills",
        "killed", "drown", "drowns", "drowned", "drowning",
    },
    "live": {
        "live", "lives", "lived", "survive", "survives", "survived", "survival",
        "recover", "recovers", "recovered", "recovering", "surviving",
    },
    "save": {"save", "saves", "saved"},
    "divert": {"divert", "diverts", "diverted", "redirect", "redirects"},
    "avert": {
        "avert", "averts", "averted", "prevent", "prevents", "prevented",
        "preclude", "precludes", "precluded",
    },
    "receive": {
        "receive", "receives", "received", "receiving", "get", "gets", "got",
    },
    "give": {
        "give", "gives", "gave", "given", "giving", "allocate", "allocates",
        "administer", "administers", "deliver", "delivers", "provide", "provides",
        "supply", "supplies", "assign", "assigns", "send", "sends",
    },
    "pull": {"pull", "pulls", "pulled", "pulling"},
}
_FAMILY_INDEX = {
    word: family
    for family, words in _FAMILIES.items()
    for word in words
}


def clause_asserts_effect(
    clause_text: str,
    effect: Mapping[str, Any],
    party_label: str = "",
) -> bool:
    """True when the clause states this party-outcome, allowing closed paraphrase."""
    return _clause_asserts_body(clause_text, effect, party_label)


def effect_is_asserted(
    effect: Mapping[str, Any],
    clauses: Sequence[Mapping[str, str]],
    party_label: str = "",
) -> bool:
    """True when any source clause states the effect, not only the cited id."""
    if clause_asserts_effect(_cited_text(effect, clauses), effect, party_label):
        return True
    for row in clauses:
        text = row.get("text") or ""
        if text and clause_asserts_effect(text, effect, party_label):
            return True
    return False


def _clause_asserts_body(
    clause_text: str,
    effect: Mapping[str, Any],
    party_label: str,
) -> bool:
    family = effect_family(effect)
    focus = _assertion_focus(clause_text, effect)
    known = family in _FAMILIES or family == "not_receive"
    present = family in families_in(focus)
    if not present and family in families_in(clause_text):
        present = True
    copied = _copied_span(effect, clause_text)
    if known and not present and inverted_families(
        family, families_in(focus) | families_in(clause_text),
    ):
        return False
    if not present and not copied:
        return False
    return _bearer_compatible(party_label, focus, clause_text, effect)


def effect_family(effect: Mapping[str, Any]) -> str:
    predicate = str(effect.get("predicate") or "").casefold()
    if predicate in {"not_receives", "not_receive"}:
        return "not_receive"
    if predicate in _FAMILY_INDEX:
        return _FAMILY_INDEX[predicate]
    outcome = str(effect.get("outcome") or "")
    found = families_in(outcome)
    if "avert" in found:
        return "avert"
    if "not_receive" in found:
        return "not_receive"
    for family in ("die", "live", "receive", "give", "pull"):
        if family in found:
            return family
    return predicate


def families_in(text: str) -> set[str]:
    found: set[str] = set()
    folded = text or ""
    if re.search(
        r"\b(?:does not|do not|doesn't|don't|not)\s+(?:receive|get)\b",
        folded, re.I,
    ):
        found.add("not_receive")
    for token in _tokens(folded):
        family = _FAMILY_INDEX.get(token)
        if family:
            found.add(family)
    return found


def bearer_mentioned(label: str, clause_text: str) -> bool:
    """Match a party through inflection or a determiner, not a different count."""
    if not label:
        return False
    if label.casefold() in (clause_text or "").casefold():
        return True
    clause_tokens = set(_tokens(clause_text))
    label_tokens = [token for token in _tokens(label) if token not in _STOP]
    nouns = [token for token in label_tokens if token not in _NUMBER]
    numbers = {_NUMBER.get(token, token) for token in label_tokens if token in _NUMBER}
    if nouns and not (set(nouns) & clause_tokens):
        return False
    clause_numbers = {_NUMBER.get(token, token) for token in clause_tokens if token in _NUMBER}
    if numbers and clause_numbers and numbers.isdisjoint(clause_numbers):
        return False
    return bool(nouns)


def _bearer_compatible(
    label: str,
    focus: str,
    clause_text: str,
    effect: Mapping[str, Any],
) -> bool:
    if not label:
        return True
    if bearer_mentioned(label, focus):
        return True
    source = str(effect.get("source_proposition") or "")
    if source and source.casefold() in (clause_text or "").casefold() and bearer_mentioned(
        label, source,
    ):
        return True
    outcome = str(effect.get("outcome") or "")
    if outcome and outcome.casefold() in (clause_text or "").casefold() and bearer_mentioned(
        label, outcome,
    ):
        return True
    if _pronoun_antecedent(label, focus, clause_text):
        return True
    if effect_family(effect) not in {"die", "live", "not_receive", "receive"}:
        return _copied_span(effect, clause_text)
    if _number_conflict(label, focus) or _number_conflict(label, clause_text):
        return False
    return _generic_human_head(focus) or _generic_human_head(source)


def _pronoun_antecedent(label: str, focus: str, clause_text: str) -> bool:
    if not re.search(r"\b(?:he|she|they|him|her|them|his)\b", focus or "", re.I):
        return False
    parts = _split_conditional_clause(clause_text)
    antecedent = parts[0] if parts else (clause_text or "")
    if not bearer_mentioned(label, antecedent):
        return False
    if _number_conflict(label, focus) or _number_conflict(label, clause_text):
        return False
    if _intervention_agent(label, antecedent) and not _transfer_patient(label, antecedent):
        return False
    return True


def _intervention_agent(label: str, text: str) -> bool:
    if not label:
        return False
    return bool(re.search(
        rf"\b{re.escape(label)}\b\s+"
        r"(?:can\s+|may\s+|must\s+|does\s+not\s+|do\s+not\s+|doesn't\s+)?"
        r"(?:pull|divert|redirect|throw|push|hit|flip|save|rescue)s?\b",
        text or "", re.I,
    ))


def _transfer_patient(label: str, text: str) -> bool:
    if not label:
        return False
    escaped = re.escape(label)
    return bool(re.search(
        rf"(?:\b{escaped}\b\s+(?:receives?|gets?|got|received|is given|is treated)"
        rf"|(?:to|for)\s+(?:either\s+)?{escaped}\b"
        rf"|(?:save|saves|saved|rescue|rescues|rescued)\s+(?:the\s+)?{escaped}\b"
        rf"|given\b[\s\S]{{0,60}}\b{escaped}\b)",
        text or "", re.I,
    ))


def _generic_human_head(text: str) -> bool:
    return bool(re.search(
        r"\b(?:patient|patients|person|people|recipient|whoever|anyone|"
        r"untreated|who does not|who do not)\b",
        text or "", re.I,
    ))


def _number_conflict(label: str, text: str) -> bool:
    label_numbers = {
        _NUMBER.get(token, token) for token in _tokens(label) if token in _NUMBER
    }
    text_numbers = {
        _NUMBER.get(token, token) for token in _tokens(text) if token in _NUMBER
    }
    return bool(label_numbers and text_numbers and label_numbers.isdisjoint(text_numbers))


def _copied_span(effect: Mapping[str, Any], clause_text: str) -> bool:
    cited = (clause_text or "").casefold()
    for span in (effect.get("outcome"), effect.get("source_proposition")):
        text = " ".join(str(span or "").split())
        if text and text.casefold() in cited:
            return True
        tokens = [token for token in _tokens(text) if token not in _STOP]
        if tokens and set(tokens) <= set(_tokens(cited)):
            return True
    return False


def _split_conditional_clause(clause_text: str) -> tuple[str, str] | None:
    match = re.match(
        r"^(?:if|unless)\b([\s\S]+?),\s*(.+)$",
        (clause_text or "").strip(), re.I,
    )
    if not match:
        return None
    return match.group(1), match.group(2)


def _assertion_focus(clause_text: str, effect: Mapping[str, Any]) -> str:
    """Use the if-antecedent for the transfer it states; the consequent otherwise.

    That keeps Maria from counting as the dying party while still treating
    ``If Malik receives the serum`` as an assertion that Malik receives it.
    """
    parts = _split_conditional_clause(clause_text)
    if not parts:
        return clause_text or ""
    antecedent, consequent = parts
    family = effect_family(effect)
    if family and family in families_in(antecedent) and family not in families_in(consequent):
        return antecedent
    return consequent


def origin_for_effect(
    effect: Mapping[str, Any],
    clauses: Sequence[Mapping[str, str]],
    parties: Sequence[Mapping[str, Any]] = (),
    exclusivity_proof: Mapping[str, Any] | None = None,
) -> str:
    operation = str(effect.get("derivation_operation") or "UNSPECIFIED")
    if operation in HYPOTHESIS_OPERATIONS:
        return "WORLD_KNOWLEDGE_HYPOTHESIS"
    if operation in STRUCTURAL_OPERATIONS:
        return "STRUCTURALLY_DERIVED"
    if effect_is_asserted(effect, clauses, _party_label(effect, parties)):
        return "SOURCE_ASSERTED"
    if operation in STIPULATED_OPERATIONS:
        return "UNRESOLVED"
    proof = exclusivity_proof or {}
    if proof.get("status") in {"EXPLICIT", "DERIVED"}:
        return "STRUCTURALLY_DERIVED"
    return "UNRESOLVED"


def apply_derivation_license(
    world: dict[str, Any],
    clauses: Sequence[Mapping[str, str]],
    exclusivity_proof: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stamp origins in place. Never promote a hypothesis to a stipulated fact."""
    _attach_hypothesized_aversions(world)
    parties = world.get("parties") or []
    for effect in world.get("effects") or []:
        cited = _cited_text(effect, clauses)
        label = _party_label(effect, parties)
        asserted = effect_is_asserted(effect, clauses, label)
        operation = str(effect.get("derivation_operation") or "")
        inverted = inverted_families(effect_family(effect), families_in(cited))
        if operation in STIPULATED_OPERATIONS and inverted and not asserted:
            effect["derivation_operation"] = "AVERTED_ALTERNATIVE_HARM"
            effect["outcome_type_transformation"] = "POLARITY_INVERTED"
            operation = "AVERTED_ALTERNATIVE_HARM"
        if operation in HYPOTHESIS_OPERATIONS:
            if not effect.get("derivation_assumptions"):
                effect["derivation_assumptions"] = list(AVERTED_ASSUMPTIONS)
            if effect.get("outcome_type_transformation") == "PRESERVED":
                effect["outcome_type_transformation"] = "POLARITY_INVERTED"
        elif operation in STIPULATED_OPERATIONS and asserted:
            effect["outcome_type_transformation"] = effect.get(
                "outcome_type_transformation") or "PRESERVED"
    return world


def derivation_errors(
    world: Mapping[str, Any],
    clauses: Sequence[Mapping[str, str]],
    exclusivity_proof: Mapping[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    parties = world.get("parties") or []
    for effect in world.get("effects") or []:
        ident = str(effect.get("effect_id") or "effect")
        operation = str(effect.get("derivation_operation") or "")
        cited = _cited_text(effect, clauses)
        label = _party_label(effect, parties)
        asserted = effect_is_asserted(effect, clauses, label)
        if operation in STIPULATED_OPERATIONS and not asserted:
            errors.append(
                f"{ident} is {operation} but no source clause asserts "
                "that party-outcome"
            )
            if inverted_families(effect_family(effect), families_in(cited)):
                errors.append(
                    f"{ident} cannot be {operation} after a polarity inversion"
                )
        if operation in HYPOTHESIS_OPERATIONS and not effect.get("derivation_assumptions"):
            errors.append(f"{ident} is hypothesized without derivation_assumptions")
        if operation in HYPOTHESIS_OPERATIONS and asserted:
            errors.append(
                f"{ident} is hypothesized even though the cited clause asserts it"
            )
        if operation in STRUCTURAL_OPERATIONS:
            status = (exclusivity_proof or {}).get("status")
            if status not in {"EXPLICIT", "DERIVED"} and status is not None:
                # Complement still needs a discharged exclusivity proof when
                # the envelope recorded one. UNKNOWN/HYPOTHESIZED is a problem
                # only for exclusive-allocation worlds, checked elsewhere.
                pass
    return errors


def inverted_families(family: str, others: Iterable[str]) -> bool:
    return any((family, other) in INVERSE_FAMILIES for other in others)


def effect_family_from_polarity(effect: Mapping[str, Any]) -> str:
    polarity = str(effect.get("polarity") or "").upper()
    family = effect_family(effect)
    if family == "avert":
        return "avert"
    if polarity == "ADVERSE" and family == "live":
        return "die"
    if polarity == "BENEFICIAL" and family == "die":
        return "avert"
    return family


def _attach_hypothesized_aversions(world: dict[str, Any]) -> None:
    effects = list(world.get("effects") or [])
    actions = list(world.get("actions") or [])
    parties = {row.get("party_id"): row for row in world.get("parties") or []}
    if len(actions) < 2:
        return
    existing = {(row.get("action_id"), row.get("party_id")) for row in effects}
    action_ids = [row.get("action_id") for row in actions if row.get("action_id")]
    links = list(world.get("counterfactual_links") or [])
    added = False
    for harm in effects:
        if str(harm.get("polarity") or "").upper() != "ADVERSE":
            continue
        if str(harm.get("modality") or "").upper() != "CERTAIN":
            continue
        if effect_family(harm) not in {"die", "not_receive"}:
            continue
        if str(harm.get("derivation_operation") or "") in HYPOTHESIS_OPERATIONS:
            continue
        party_id = harm.get("party_id")
        harm_action = harm.get("action_id")
        for action_id in action_ids:
            if action_id == harm_action:
                continue
            if any(
                row.get("action_id") == action_id
                and row.get("party_id") == party_id
                and str(row.get("polarity") or "").upper() == "BENEFICIAL"
                for row in effects
            ):
                continue
            if (action_id, party_id) in existing:
                continue
            overlay = _averted_effect(harm, action_id, parties.get(party_id) or {})
            effects.append(overlay)
            existing.add((action_id, party_id))
            for action in actions:
                if action.get("action_id") == action_id:
                    action.setdefault("effect_ids", []).append(overlay["effect_id"])
            links.append({
                "action_id": action_id,
                "source_effect_id": overlay["effect_id"],
                "relation": "PRECLUDES_ALTERNATIVE_EFFECT",
                "alternative_action_id": harm_action,
                "alternative_effect_id": harm.get("effect_id"),
                "modality": "CERTAIN",
                "condition_ids": [],
                "clause_ids": list(harm.get("clause_ids") or []),
            })
            added = True
    if added:
        world["effects"] = effects
        world["counterfactual_links"] = links


def _averted_effect(
    harm: Mapping[str, Any],
    action_id: str,
    party: Mapping[str, Any],
) -> dict[str, Any]:
    used = harm.get("effect_id") or "E"
    ident = f"AV{used[1:]}" if str(used).startswith("E") else f"AV_{used}"
    label = party.get("label") or "the opposed party"
    return {
        "effect_id": ident,
        "action_id": action_id,
        "party_id": harm.get("party_id"),
        "outcome": f"averts alternative harm to {label}",
        "predicate": "avert",
        "polarity": "BENEFICIAL",
        "directness": "DOWNSTREAM",
        "modality": "CERTAIN",
        "effect_kind": harm.get("effect_kind") or "HEALTH_OUTCOME",
        "condition_ids": [],
        "quantities": list(harm.get("quantities") or []),
        "likelihood_qualifiers": [],
        "overall_likelihood_qualifiers": [],
        "scope_qualifiers": [],
        "temporal_qualifiers": [],
        "condition_join": "AND",
        "source_proposition": harm.get("source_proposition") or harm.get("outcome"),
        "source_effect_ids": [harm.get("effect_id")],
        "derivation_operation": "AVERTED_ALTERNATIVE_HARM",
        "derivation_explanation": (
            f"{action_id} precludes CERTAIN alternative harm "
            f"{harm.get('effect_id')}; the opposed death is the input, not a "
            "copied survival clause"
        ),
        "derivation_assumptions": list(AVERTED_ASSUMPTIONS),
        "outcome_type_transformation": "POLARITY_INVERTED",
        "clause_ids": list(harm.get("clause_ids") or []),
    }


def _cited_text(
    effect: Mapping[str, Any],
    clauses: Sequence[Mapping[str, str]],
) -> str:
    by_id = {row.get("clause_id"): row.get("text") or "" for row in clauses}
    cited = [by_id.get(cid, "") for cid in effect.get("clause_ids") or [] if cid in by_id]
    if cited:
        return " ".join(cited)
    return str(effect.get("source_proposition") or effect.get("outcome") or "")


def _party_label(
    effect: Mapping[str, Any],
    parties: Sequence[Mapping[str, Any]],
) -> str:
    by_id = {row.get("party_id"): row.get("label") or "" for row in parties}
    return by_id.get(effect.get("party_id"), "")


def _tokens(text: str) -> list[str]:
    return [word.casefold() for word in _WORD.findall(text or "")]
