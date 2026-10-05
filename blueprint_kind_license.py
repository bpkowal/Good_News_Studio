"""License party and effect kinds from construction evidence, not lexicons.

Regex may witness a span or veto a claimed atom. It may not assign kind.
Missing kind is OTHER with origin UNRESOLVED, never PERSON-by-default.

A learned kind proposer is out of scope. This contract is the control loop
already used for effects: propose an atom, cite a span, admit or overlay.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from blueprint_allocation_invariants import group_span, party_kind
from blueprint_derivation_license import families_in


KIND_ORIGINS = ("SOURCE_ASSERTED", "STRUCTURALLY_DERIVED", "UNRESOLVED")
LICENSED_KINDS = (
    "PERSON", "HUMAN_GROUP", "ANIMAL", "RESOURCE", "PROCESS",
    "COMMUNITY", "OTHER",
)
ACTOR_ROLES = {
    "actor", "decider", "rescuer", "promisor", "source", "authority",
}
PARTICIPANT_ROLES = {
    "recipient", "saved", "bearer", "affected", "first_recipient",
    "second_recipient", "promisee", "target", "participant",
}
RESOURCE_ROLES = {"resource"}
PROCESS_ROLES = {"process", "controllable_process", "instrument"}
ALLOCATION_CONSTRUCTIONS = {
    "exclusive_allocation", "allocation", "resource_transfer",
}
PROCESS_CONSTRUCTIONS = {
    "omission_harm", "conditional_outcome", "diversion_redirection",
    "physical_state",
}

# Span checks. These do not write parties[].kind.
TRANSFER_EVENT = re.compile(
    r"\b(?:give|gives|gave|given|giving|receive|receives|received|receiving|"
    r"allocate|allocates|allocated|allocating|assign|assigns|assigned|assigning|"
    r"send|sends|sent|sending|deliver|delivers|delivered|delivering|"
    r"transfer|transfers|transferred|transferring|administer|administers|"
    r"administered|administering|provide|provides|provided|providing|"
    r"supply|supplies|supplied|supplying)\b",
    re.I,
)
NONRECEIPT = re.compile(
    r"\b(?:does not|doesn't|do not|don't|not receive|not get|does not get)\b",
    re.I,
)
EXPLICIT_EXCLUSIVITY = re.compile(
    r"\b(?:but\s+)?not\s+both\b|"
    r"\b(?:cannot|can\s+not|can't)\b[^.!?]{0,80}\bboth\b",
    re.I,
)
# Negative control: a countable unit is not exclusivity and not a RESOURCE.
INDIVISIBLE_ONE = re.compile(
    r"\b(?:one|1|single)\s+(?:[A-Za-z-]+\s+){0,3}"
    r"(?:dose|vial|unit|organ|seat|ticket|bed|ventilator)\b",
    re.I,
)
OUTCOME_WORDS = re.compile(
    r"\b(?:live|lives|survive|survives|recover|recovers|die|dies|kill|kills|"
    r"harm|harms|lose|loses|gain|gains|sustain|sustains|drown|drowns|"
    r"benefit|benefits|suffer|suffers|stop|stops)\b",
    re.I,
)
WELFARE_WORDS = re.compile(
    r"\b(?:live|lives|survive|survives|recover|recovers|die|dies|kill|kills|"
    r"drown|drowns|harm|harms)\b",
    re.I,
)
# Locative water is a medium, not an allocated good. Do not enlarge the
# medicine/serum/water list; this only blocks a prepositional use.
LOCATIVE_MEDIUM = re.compile(
    r"\b(?:in|into|through|under|on|across)\s+(?:the\s+)?water\b",
    re.I,
)
EXPLICIT_KIND_PREDICATION = re.compile(
    r"\b(?:is|are|was|were)\s+(?:a|an|the\s+)?(person|group|resource|animal|"
    r"process|community)\b",
    re.I,
)
# Process-verb veto for a claimed PHYSICAL_STATE, not a document search.
PROCESS_PREDICATES = {
    "stop", "start", "open", "close", "fail", "activate",
    "pull", "push", "throw", "divert",
}
# Human heads that cannot bear a PROCESS atom. Morphology, not a new ontology.
_PROCESS_VETO_HEAD = re.compile(
    r"\b(?:people|workers?|patients?|residents?|children|child|person)\b",
    re.I,
)
_POSSESSION = re.compile(
    r"\b(?:has|have|holds?|contains?)\b",
    re.I,
)
_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

# Inventory of regexes in the kind/effect path. ontology_induction must stay
# empty: those jobs moved to license_kind / license_effect_kind.
REGEX_CATALOG: tuple[dict[str, str], ...] = (
    {"name": "_TRANSFER_EVENT", "file": "blueprint_cloze_chooser.py",
     "role": "witness"},
    {"name": "TRANSFER_EVENT", "file": "blueprint_kind_license.py",
     "role": "witness"},
    {"name": "_NONRECEIPT", "file": "blueprint_cloze_chooser.py",
     "role": "witness"},
    {"name": "_NONRECEIPT_DEATH", "file": "candidate_graph_blueprints.py",
     "role": "witness"},
    {"name": "_EXPLICIT_EXCLUSIVITY", "file": "blueprint_cloze_chooser.py",
     "role": "witness"},
    {"name": "_GROUP", "file": "blueprint_allocation_invariants.py",
     "role": "witness"},
    {"name": "_EXCLUSIVITY", "file": "blueprint_cloze_chooser.py",
     "role": "witness"},
    {"name": "_OUTCOME_WORDS", "file": "blueprint_cloze_chooser.py",
     "role": "witness"},
    {"name": "_FAMILIES", "file": "blueprint_derivation_license.py",
     "role": "witness"},
    {"name": "party_kind", "file": "blueprint_allocation_invariants.py",
     "role": "negative_control"},
    {"name": "_INDIVISIBLE_ONE", "file": "blueprint_cloze_chooser.py",
     "role": "negative_control"},
    {"name": "_PROCESS_PREDICATES", "file": "blueprint_graph_amendments.py",
     "role": "negative_control"},
    {"name": "_PROCESS_BEARER", "file": "blueprint_graph_amendments.py",
     "role": "negative_control"},
    {"name": "LOCATIVE_MEDIUM", "file": "blueprint_kind_license.py",
     "role": "negative_control"},
)


def catalog_roles() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {
        "witness": [], "negative_control": [], "ontology_induction": [],
    }
    for row in REGEX_CATALOG:
        grouped[row["role"]].append(row["name"])
    return grouped


def license_kind(
    label: str,
    *,
    role: str = "",
    text: str = "",
    construction: str = "",
    quantities: Sequence[str] | None = None,
    action_span: str = "",
    evidence_span: str = "",
) -> dict[str, Any]:
    """Propose a kind from construction role or explicit predication.

    ``party_kind`` may support or veto. It never assigns.
    """
    span = evidence_span or label or ""
    witness = party_kind(label)
    vetoes: list[str] = []
    supports: list[str] = []
    asserted = _explicit_kind_predication(label, text)
    if asserted:
        if _blocked_by_witness(asserted, witness, label, text, span):
            vetoes.append(f"witness_blocks_{asserted}")
            return _envelope("OTHER", "UNRESOLVED", label, witness, vetoes, supports)
        return _envelope(asserted, "SOURCE_ASSERTED", label, witness, vetoes, supports)

    role = (role or "").casefold()
    construction = (construction or "").casefold()

    if locative_medium(text, span) or locative_medium(text, label):
        if role in RESOURCE_ROLES or witness == "RESOURCE":
            vetoes.append("locative_medium")
            if role in RESOURCE_ROLES:
                return _envelope("OTHER", "UNRESOLVED", label, witness, vetoes, supports)

    if role in PROCESS_ROLES or (
            construction in {"physical_state"}
            and role in {"bearer", "affected", "participant", ""}):
        if _process_vetoed(label, witness):
            vetoes.append("not_a_process_bearer")
            if role in PROCESS_ROLES:
                return _envelope("OTHER", "UNRESOLVED", label, witness, vetoes, supports)
        elif role in PROCESS_ROLES or construction == "physical_state":
            supports.append("process_construction")
            return _envelope("PROCESS", "STRUCTURALLY_DERIVED", label, witness, vetoes, supports)

    if role in RESOURCE_ROLES:
        if not _resource_construction(construction, action_span, text, quantities):
            vetoes.append("dose_or_noun_is_not_resource")
            return _envelope("OTHER", "UNRESOLVED", label, witness, vetoes, supports)
        if witness == "RESOURCE":
            supports.append("resource_lexicon")
        return _envelope("RESOURCE", "STRUCTURALLY_DERIVED", label, witness, vetoes, supports)

    if role in ACTOR_ROLES or role in PARTICIPANT_ROLES:
        if group_span(label) or (quantities and group_span(label or " ".join(quantities))):
            supports.append("quantified_human_head")
            return _envelope(
                "HUMAN_GROUP", "STRUCTURALLY_DERIVED", label, witness, vetoes, supports)
        if witness == "ANIMAL":
            supports.append("animal_witness_on_licensed_participant")
            return _envelope("ANIMAL", "STRUCTURALLY_DERIVED", label, witness, vetoes, supports)
        if witness == "COMMUNITY":
            supports.append("community_witness_on_licensed_participant")
            return _envelope(
                "COMMUNITY", "STRUCTURALLY_DERIVED", label, witness, vetoes, supports)
        if witness == "OTHER" and re.search(r"\b(?:bot|model|system)\b", label or "", re.I):
            vetoes.append("nonperson_agent")
            return _envelope("OTHER", "STRUCTURALLY_DERIVED", label, witness, vetoes, supports)
        if witness == "RESOURCE" and role in PARTICIPANT_ROLES:
            vetoes.append("resource_witness_on_participant")
        return _envelope("PERSON", "STRUCTURALLY_DERIVED", label, witness, vetoes, supports)

    # No licensed role: lexicon hits are not kinds.
    if witness in {"RESOURCE", "ANIMAL", "COMMUNITY", "HUMAN_GROUP"}:
        vetoes.append("lexicon_may_not_assign")
    return _envelope("OTHER", "UNRESOLVED", label, witness, vetoes, supports)


def license_effect_kind(
    claimed: str,
    *,
    span: str = "",
    construction: str = "",
    parties: Sequence[Mapping[str, Any]] = (),
    predicate: str = "",
) -> dict[str, Any]:
    """Admit a claimed effect kind or demote it. Regex never mints the atom."""
    claimed = (claimed or "INTERVENTION").upper()
    construction = (construction or "").casefold()
    vetoes: list[str] = []
    if claimed == "RESOURCE_TRANSFER":
        if not TRANSFER_EVENT.search(span or ""):
            vetoes.append("no_give_or_receive_span")
            return {"kind": "INTERVENTION", "origin": "UNRESOLVED", "vetoes": vetoes}
        if not _licensed_resource(parties):
            vetoes.append("no_licensed_resource")
            return {"kind": "INTERVENTION", "origin": "UNRESOLVED", "vetoes": vetoes}
        if construction not in ALLOCATION_CONSTRUCTIONS:
            vetoes.append("transfer_not_in_allocation_construction")
            return {"kind": "INTERVENTION", "origin": "UNRESOLVED", "vetoes": vetoes}
        return {"kind": "RESOURCE_TRANSFER", "origin": "STRUCTURALLY_DERIVED", "vetoes": []}
    if claimed == "PHYSICAL_STATE":
        lemma = (predicate or licensed_source_predicate(span) or "").casefold()
        if lemma and lemma not in PROCESS_PREDICATES:
            vetoes.append("process_predicate_veto")
            return {"kind": "OTHER", "origin": "UNRESOLVED", "vetoes": vetoes}
        if not _licensed_process(parties):
            vetoes.append("no_licensed_process")
            return {"kind": "OTHER", "origin": "UNRESOLVED", "vetoes": vetoes}
        return {"kind": "PHYSICAL_STATE", "origin": "STRUCTURALLY_DERIVED", "vetoes": []}
    if claimed in {"HEALTH_OUTCOME", "WELFARE_OUTCOME"}:
        if not WELFARE_WORDS.search(span or "") and not OUTCOME_WORDS.search(span or ""):
            vetoes.append("no_outcome_span")
            return {"kind": "OTHER", "origin": "UNRESOLVED", "vetoes": vetoes}
        if claimed == "HEALTH_OUTCOME" and not WELFARE_WORDS.search(span or ""):
            vetoes.append("outcome_words_are_not_welfare")
            return {"kind": "OTHER", "origin": "UNRESOLVED", "vetoes": vetoes}
        return {"kind": claimed, "origin": "STRUCTURALLY_DERIVED", "vetoes": []}
    return {"kind": claimed, "origin": "STRUCTURALLY_DERIVED", "vetoes": []}


def licensed_source_predicate(span: str) -> str:
    """Copied verb from a licensed family. SpaCy ROOT must not replace this."""
    found = families_in(span or "")
    for family in (
            "die", "live", "save", "receive", "give", "pull", "divert", "avert"):
        if family in found:
            return family
    match = TRANSFER_EVENT.search(span or "")
    if match:
        token = match.group(0).casefold()
        for stem in ("receive", "give", "allocate", "assign", "send", "deliver",
                     "transfer", "administer", "provide", "supply"):
            if token.startswith(stem):
                return stem
        return token
    words = [word.casefold() for word in _WORD.findall(span or "")]
    for word in words:
        if word in PROCESS_PREDICATES or word.rstrip("s") in PROCESS_PREDICATES:
            lemma = word if word in PROCESS_PREDICATES else word.rstrip("s")
            return lemma
    return ""


def process_predicate_allowed(predicate: str) -> bool:
    """Negative control for a claimed PHYSICAL_STATE verb."""
    return (predicate or "").casefold() in PROCESS_PREDICATES


def locative_medium(text: str, label: str = "") -> bool:
    haystack = text or ""
    if LOCATIVE_MEDIUM.search(haystack):
        if not label or "water" in (label or "").casefold():
            return True
    if label and re.search(
            rf"\b(?:in|into|through|under|on|across)\s+(?:the\s+)?{re.escape(label.strip())}\b",
            haystack, re.I):
        return True
    return False


def apply_kind_license(
    world: dict[str, Any],
    clauses: Sequence[Mapping[str, str]] | None = None,
    blueprint_id: str = "",
) -> dict[str, Any]:
    """Stamp kind origins and demote unlicensed effect kinds in place."""
    parties = world.get("parties") or []
    construction = blueprint_id or ""
    for party in parties:
        if party.get("kind_origin") in KIND_ORIGINS:
            continue
        licensed = license_kind(
            party.get("label") or "",
            role=_role_hint(party, construction),
            text=_joined_text(clauses),
            construction=construction,
            quantities=party.get("quantities") or [],
        )
        party["kind"] = licensed["kind"]
        party["kind_origin"] = licensed["origin"]
    for effect in world.get("effects") or []:
        claimed = str(effect.get("effect_kind") or "INTERVENTION")
        if claimed not in {"RESOURCE_TRANSFER", "PHYSICAL_STATE", "HEALTH_OUTCOME",
                           "WELFARE_OUTCOME"}:
            continue
        licensed = license_effect_kind(
            claimed,
            span=str(effect.get("source_proposition") or effect.get("outcome") or ""),
            construction=construction,
            parties=parties,
            predicate=str(effect.get("predicate") or ""),
        )
        if licensed["kind"] != claimed:
            effect["effect_kind"] = licensed["kind"]
            assumptions = list(effect.get("derivation_assumptions") or [])
            tag = "unlicensed_atom:" + ",".join(licensed["vetoes"])
            if tag not in assumptions and licensed["vetoes"]:
                # Keep stipulated copies in the world with a demoted kind.
                # Overlay only atoms that cannot be rewritten in place.
                effect.setdefault("scope_qualifiers", [])
                if tag not in effect["scope_qualifiers"]:
                    effect["scope_qualifiers"].append(tag)
        elif claimed == "PHYSICAL_STATE" and not _licensed_process(parties):
            effect["effect_kind"] = "OTHER"
    return world


def kind_problems(world: Mapping[str, Any]) -> list[dict[str, str]]:
    problems = []
    for party in world.get("parties") or []:
        if party.get("kind_origin") == "UNRESOLVED":
            problems.append({
                "code": "party_kind_unresolved",
                "message": (
                    f"{party.get('label') or party.get('party_id')} has no licensed "
                    "kind; OTHER is a gap, not PERSON."
                ),
            })
    return problems


def overlay_unlicensed_atoms(effect: Mapping[str, Any], parties: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reasons to move an atom into the supported_core overlay."""
    reasons = []
    kind = str(effect.get("effect_kind") or "")
    qualifiers = " ".join(str(item) for item in effect.get("scope_qualifiers") or [])
    if "unlicensed_atom:" in qualifiers:
        # Demoted in place; do not strip the stipulated copy.
        return reasons
    if kind == "RESOURCE_TRANSFER" and not _licensed_resource(parties):
        reasons.append("unlicensed_resource_transfer")
    if kind == "PHYSICAL_STATE" and not _licensed_process(parties):
        reasons.append("unlicensed_process_state")
    if kind in {"HEALTH_OUTCOME", "WELFARE_OUTCOME"}:
        span = str(effect.get("source_proposition") or effect.get("outcome") or "")
        if not WELFARE_WORDS.search(span) and not OUTCOME_WORDS.search(span):
            reasons.append("unlicensed_welfare_effect")
    return reasons


def _envelope(
    kind: str, origin: str, label: str, witness: str,
    vetoes: list[str], supports: list[str],
) -> dict[str, Any]:
    return {
        "kind": kind,
        "origin": origin,
        "label": label,
        "witness": {"party_kind": witness, "vetoes": vetoes, "supports": supports},
    }


def _explicit_kind_predication(label: str, text: str) -> str:
    if not text or not label:
        return ""
    for clause in re.split(r"(?<=[.!?])\s+", text):
        if label.casefold() not in clause.casefold():
            continue
        match = EXPLICIT_KIND_PREDICATION.search(clause)
        if match:
            token = match.group(1).casefold()
            return {
                "person": "PERSON", "group": "HUMAN_GROUP", "resource": "RESOURCE",
                "animal": "ANIMAL", "process": "PROCESS", "community": "COMMUNITY",
            }[token]
    return ""


def _blocked_by_witness(
    kind: str, witness: str, label: str, text: str, span: str,
) -> bool:
    if kind == "RESOURCE" and (locative_medium(text, span) or locative_medium(text, label)):
        return True
    if kind == "PROCESS" and _process_vetoed(label, witness):
        return True
    if kind == "HUMAN_GROUP" and not group_span(label):
        return True
    return False


def _process_vetoed(label: str, witness: str) -> bool:
    if witness in {"ANIMAL", "HUMAN_GROUP", "RESOURCE", "COMMUNITY"}:
        return True
    if group_span(label) or _PROCESS_VETO_HEAD.search(label or ""):
        return True
    return False


def _resource_construction(
    construction: str,
    action_span: str,
    text: str,
    quantities: Sequence[str] | None,
) -> bool:
    if construction in ALLOCATION_CONSTRUCTIONS:
        return True
    haystack = f"{action_span} {text}"
    if TRANSFER_EVENT.search(haystack) and (quantities or INDIVISIBLE_ONE.search(haystack)):
        # Quantity plus give/receive is a transfer construction, not "dose" alone.
        return True
    if _POSSESSION.search(haystack) and quantities:
        return True
    return False


def _licensed_resource(parties: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        row.get("kind") == "RESOURCE" and row.get("kind_origin") != "UNRESOLVED"
        for row in parties
    )


def _licensed_process(parties: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        row.get("kind") == "PROCESS" and row.get("kind_origin") != "UNRESOLVED"
        for row in parties
    )


def _role_hint(party: Mapping[str, Any], construction: str) -> str:
    kind = str(party.get("kind") or "")
    if kind == "PROCESS":
        return "process"
    if kind == "RESOURCE":
        return "resource"
    if construction in ALLOCATION_CONSTRUCTIONS and kind in {"PERSON", "HUMAN_GROUP", "OTHER"}:
        return "participant"
    return ""


def _joined_text(clauses: Sequence[Mapping[str, str]] | None) -> str:
    return " ".join(row.get("text") or "" for row in clauses or [])
