"""Parliament adapters onto the portable RelEnt kernel.

Kernel code lives in ``relent/`` and must not import this module (or engine).
"""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

from relent import (
    RELATION_TAGS,
    RelEdge,
    action_families_conflict,
    closure_edges,
    directionality_errors,
    forbidden_composites,
    function_transfer_errors,
    licensed_action_paraphrase,
    quantity_precision_escalation_errors,
    relation_properties,
    relational_entailment_errors,
    status_conserving_paraphrase_errors,
    status_transfers,
)
from relent.operators import (
    AllOf,
    AnyOf,
    Conditional,
    ExceptionRule,
    Fact,
    Modal,
    Not,
    Operator,
    collect_facts,
    modal_strength_of,
    operator_from_dict,
    operator_to_dict,
)
from relent.relations import RelationSpec, RelationTag
from relent.scope import (
    and_not_inheritance_errors,
    exception_scope_errors,
    modal_scope_errors,
    operator_scope_conservation_errors,
)

__all__ = [
    "RELATION_TAGS",
    "RelEdge",
    "RelationSpec",
    "RelationTag",
    "action_families_conflict",
    "licensed_action_paraphrase",
    "quantity_precision_escalation_errors",
    "status_conserving_paraphrase_errors",
    "status_transfers",
    "relation_properties",
    "closure_edges",
    "directionality_errors",
    "forbidden_composites",
    "function_transfer_errors",
    "relational_entailment_errors",
    "precision_errors_for_claim",
    "action_paraphrase_binds",
    "action_branch_conflicts",
    "project_operator_to_effect_gates",
    "project_alternative_of_edges",
    "project_averted_derived_edges",
    "averted_alternative_relational_errors",
    "project_same_entity_edges",
    "identity_relational_errors",
    "project_causes_edges",
    "project_before_edges",
    "directionality_relational_errors",
    "and_not_inheritance_errors",
    "exception_scope_errors",
    "modal_scope_errors",
    "operator_scope_conservation_errors",
]


def action_paraphrase_binds(
    claim: str,
    *,
    outcome: str,
    polarity: str,
    action_id: str = "",
    action_glosses: Sequence[str] = (),
) -> bool:
    """Adapter for epistemic ledger / Hypothesis hosts."""
    return licensed_action_paraphrase(
        claim,
        outcome=outcome,
        polarity=polarity,
        action_id=action_id,
        action_glosses=action_glosses,
    )


def action_branch_conflicts(
    claim: str,
    *,
    action_glosses: Sequence[str],
) -> bool:
    """True when claim cues the opposed action family from licensed glosses."""
    return action_families_conflict(claim, action_glosses)


def precision_errors_for_claim(
    claim_text: str,
    *,
    source_texts: Sequence[str],
) -> list[str]:
    """Adapter-shaped entry for util / challenge / epistemic call sites."""
    return quantity_precision_escalation_errors(
        source_texts=source_texts,
        claim_text=claim_text,
    )


def _stable_condition_id(fact: Fact) -> str:
    digest = hashlib.sha256(fact.key().encode("utf-8")).hexdigest()[:10]
    label = (fact.entity or fact.predicate or "COND").upper().replace(" ", "_")
    return f"COND:{label[:24]}:{digest}"


def _join_for(node: Operator) -> str:
    if isinstance(node, AllOf):
        return "AND"
    if isinstance(node, AnyOf):
        return "OR"
    if isinstance(node, Not):
        return _join_for(node.child) or "AND"
    if isinstance(node, Conditional):
        return _join_for(node.if_)
    if isinstance(node, ExceptionRule):
        return _join_for(node.rule)
    return ""


def project_operator_to_effect_gates(
    rule: Operator | dict[str, Any],
) -> dict[str, Any]:
    """Compile a RelEnt rule into Parliament-shaped gate fields.

    Output keys mirror WorldEffect channels without writing the world model:
    ``condition_ids``, ``condition_join``, ``modality``, plus descriptions and
    the source operator JSON for provenance.
    """
    node = operator_from_dict(rule)
    if isinstance(node, Conditional):
        antecedent: Operator = node.if_
    elif isinstance(node, ExceptionRule):
        antecedent = AllOf(children=(node.rule.if_, Not(child=node.unless)))
    else:
        antecedent = node
    facts = collect_facts(antecedent)
    positive_ids: list[str] = []
    descriptions: list[str] = []
    negated_ids: list[str] = []

    def walk(item: Operator, *, negated: bool = False) -> None:
        if isinstance(item, Fact):
            cid = _stable_condition_id(item)
            desc = " ".join(
                part for part in (
                    item.entity, item.predicate, item.state or item.change,
                    item.quantity,
                ) if part
            ).strip()
            descriptions.append(("NOT " if negated else "") + desc)
            if negated:
                negated_ids.append(cid)
            else:
                positive_ids.append(cid)
            return
        if isinstance(item, Not):
            walk(item.child, negated=not negated)
            return
        if isinstance(item, (AllOf, AnyOf)):
            for child in item.children:
                walk(child, negated=negated)
            return
        if isinstance(item, Modal):
            walk(item.body, negated=negated)
            return
        if isinstance(item, Conditional):
            walk(item.if_, negated=negated)
            return
        if isinstance(item, ExceptionRule):
            walk(item.rule.if_, negated=negated)
            walk(item.unless, negated=True)

    walk(antecedent)
    strength = modal_strength_of(node) or "CERTAIN"
    return {
        "condition_ids": list(dict.fromkeys(positive_ids)),
        "negated_condition_ids": list(dict.fromkeys(negated_ids)),
        "condition_join": _join_for(antecedent) or "AND",
        "modality": strength if strength != "PROBABLE" else "PROBABILISTIC",
        "condition_descriptions": list(dict.fromkeys(descriptions)),
        "operator": operator_to_dict(node),
        "fact_keys": [fact.key() for fact in facts],
    }


def _is_averted_row(effect: Any) -> bool:
    return str(getattr(effect, "derivation_operation", "") or "").upper() == (
        "AVERTED_ALTERNATIVE_HARM"
    )


def _is_actual_welfare(effect: Any) -> bool:
    directness = str(getattr(effect, "directness", "") or "").upper()
    polarity = str(getattr(effect, "polarity", "") or "").upper()
    return directness != "FOREGONE" and polarity in {"BENEFICIAL", "ADVERSE"}


def project_alternative_of_edges(model: Any) -> tuple[RelEdge, ...]:
    """Project CF pairs and opposed welfare exclusivity onto ``ALTERNATIVE_OF``.

    Duck-types ScenarioWorldModel. Does not write world state.
    """
    indexed: dict[tuple[str, str], RelEdge] = {}

    def add(left: str, right: str) -> None:
        a = str(left or "").strip()
        b = str(right or "").strip()
        if not a or not b or a == b:
            return
        key = (a, b) if a < b else (b, a)
        if key in indexed:
            return
        indexed[key] = RelEdge(source=key[0], target=key[1], relation="ALTERNATIVE_OF")

    for link in getattr(model, "counterfactual_links", ()) or ():
        add(
            getattr(link, "action_id", ""),
            getattr(link, "alternative_action_id", ""),
        )

    # Opposed CERTAIN welfare on the same party ⇒ mutually exclusive actions.
    by_party: dict[str, dict[str, list[Any]]] = {}
    for effect in getattr(model, "effects", ()) or ():
        if not _is_actual_welfare(effect) or _is_averted_row(effect):
            continue
        party = str(getattr(effect, "party_id", "") or "").strip()
        action = str(getattr(effect, "action_id", "") or "").strip()
        if not party or not action:
            continue
        by_party.setdefault(party, {}).setdefault(action, []).append(effect)

    for action_map in by_party.values():
        action_ids = sorted(action_map)
        for i, left_id in enumerate(action_ids):
            for right_id in action_ids[i + 1:]:
                left_rows = action_map[left_id]
                right_rows = action_map[right_id]
                left_ben = any(
                    str(getattr(e, "polarity", "")).upper() == "BENEFICIAL"
                    for e in left_rows
                )
                right_ben = any(
                    str(getattr(e, "polarity", "")).upper() == "BENEFICIAL"
                    for e in right_rows
                )
                left_adv = any(
                    str(getattr(e, "polarity", "")).upper() == "ADVERSE"
                    and str(getattr(e, "modality", "")).upper() == "CERTAIN"
                    for e in left_rows
                )
                right_adv = any(
                    str(getattr(e, "polarity", "")).upper() == "ADVERSE"
                    and str(getattr(e, "modality", "")).upper() == "CERTAIN"
                    for e in right_rows
                )
                if (left_ben and right_adv) or (right_ben and left_adv):
                    add(left_id, right_id)

    return tuple(indexed.values())


def project_averted_derived_edges(model: Any) -> tuple[RelEdge, ...]:
    """Project ``AVERTED_ALTERNATIVE_HARM`` rows onto derived-marked ``DERIVED_FROM``."""
    edges: list[RelEdge] = []
    for effect in getattr(model, "effects", ()) or ():
        if not _is_averted_row(effect):
            continue
        eid = str(getattr(effect, "effect_id", "") or "").strip()
        if not eid:
            continue
        for source_id in getattr(effect, "source_effect_ids", ()) or ():
            sid = str(source_id or "").strip()
            if not sid:
                continue
            edges.append(RelEdge(
                source=eid,
                target=sid,
                relation="DERIVED_FROM",
                derived_marked=True,
            ))
    return tuple(edges)


def averted_alternative_relational_errors(model: Any) -> list[str]:
    """RelEnt post-check for ALTERNATIVE_OF symmetry and AVERTED quantity transfer.

    - ``RELATIONAL_SYMMETRY``: each projected ``ALTERNATIVE_OF`` closes to its reverse.
    - ``RELATIONAL_FUNCTION_TRANSFER``: AV* quantity moves require derivation marks.
    - ``RELATIONAL_NON_TRANSFER``: silent DIRECT_COPY of opposed CERTAIN harm
      quantities onto a survival row is forbidden; harm_under_action must not
      cross action-scoped alternatives.
    """
    errors: list[str] = []
    alt_edges = project_alternative_of_edges(model)
    derived_edges = project_averted_derived_edges(model)
    all_edges = (*alt_edges, *derived_edges)

    for edge in alt_edges:
        errors.extend(relational_entailment_errors(
            (edge,),
            expect_derived=(RelEdge(
                source=edge.target,
                target=edge.source,
                relation="ALTERNATIVE_OF",
            ),),
        ))
    errors.extend(forbidden_composites(all_edges))

    closed_alts = {
        (e.source, e.target)
        for e in closure_edges(alt_edges)
        if e.relation == "ALTERNATIVE_OF" and e.source != e.target
    }
    effect_by_id = {
        str(getattr(effect, "effect_id", "")): effect
        for effect in getattr(model, "effects", ()) or ()
        if str(getattr(effect, "effect_id", "") or "").strip()
    }

    for effect in getattr(model, "effects", ()) or ():
        if not _is_averted_row(effect):
            continue
        quantities = tuple(getattr(effect, "quantities", ()) or ())
        if not quantities:
            continue
        action_id = str(getattr(effect, "action_id", "") or "").strip()
        for source_id in getattr(effect, "source_effect_ids", ()) or ():
            source = effect_by_id.get(str(source_id))
            if source is None:
                continue
            source_action = str(getattr(source, "action_id", "") or "").strip()
            for msg in function_transfer_errors(
                "ALTERNATIVE_OF",
                "quantity",
                derivation_marked=True,
                source_action=source_action,
                target_action=action_id,
            ):
                errors.append(f"{getattr(effect, 'effect_id', '?')} {msg}")
            for msg in function_transfer_errors(
                "DERIVED_FROM",
                "quantity",
                derivation_marked=True,
            ):
                errors.append(f"{getattr(effect, 'effect_id', '?')} {msg}")

    for effect in getattr(model, "effects", ()) or ():
        op = str(getattr(effect, "derivation_operation", "") or "").upper()
        if op != "DIRECT_COPY":
            continue
        if str(getattr(effect, "polarity", "") or "").upper() != "BENEFICIAL":
            continue
        if str(getattr(effect, "directness", "") or "").upper() == "FOREGONE":
            continue
        qty = {
            str(span).casefold()
            for span in (getattr(effect, "quantities", ()) or ())
            if str(span).strip()
        }
        if not qty:
            continue
        action_id = str(getattr(effect, "action_id", "") or "").strip()
        party_id = str(getattr(effect, "party_id", "") or "").strip()
        eid = str(getattr(effect, "effect_id", "") or "").strip() or "?"
        for other in getattr(model, "effects", ()) or ():
            if str(getattr(other, "party_id", "") or "").strip() != party_id:
                continue
            other_action = str(getattr(other, "action_id", "") or "").strip()
            if (action_id, other_action) not in closed_alts:
                continue
            if str(getattr(other, "polarity", "") or "").upper() != "ADVERSE":
                continue
            if str(getattr(other, "modality", "") or "").upper() != "CERTAIN":
                continue
            if str(getattr(other, "directness", "") or "").upper() == "FOREGONE":
                continue
            other_qty = {
                str(span).casefold()
                for span in (getattr(other, "quantities", ()) or ())
                if str(span).strip()
            }
            shared = sorted(qty & other_qty)
            if not shared:
                continue
            for msg in function_transfer_errors(
                "ALTERNATIVE_OF",
                "quantity",
                derivation_marked=False,
                source_action=other_action,
                target_action=action_id,
            ):
                errors.append(
                    f"{eid} {msg}; spans {shared} belong on "
                    f"AVERTED_ALTERNATIVE_HARM via {getattr(other, 'effect_id', '?')}, "
                    "not DIRECT_COPY on the survival row"
                )
            for msg in function_transfer_errors(
                "ALTERNATIVE_OF",
                "harm_under_action",
                source_action=other_action,
                target_action=action_id,
            ):
                errors.append(f"{eid} {msg}")

    return list(dict.fromkeys(errors))


def _normalized_party_label(label: str) -> str:
    return " ".join(str(label or "").casefold().split())


def project_same_entity_edges(
    model: Any,
    *,
    licensed_edges: Sequence[RelEdge | dict] | None = None,
) -> tuple[RelEdge, ...]:
    """Project party aliases onto ``SAME_ENTITY_AS``.

    Hosts may pass ``licensed_edges`` (Hypothesis / anaphor bridge). Matching
    normalized party labels also license undirected identity edges. Does not
    parse discourse or invent party merges.
    """
    indexed: dict[tuple[str, str, str], RelEdge] = {}

    def add(left: str, right: str, *, tag: str = "SAME_ENTITY_AS") -> None:
        a = str(left or "").strip()
        b = str(right or "").strip()
        if not a or not b or a == b:
            return
        edge = RelEdge(source=a, target=b, relation=tag).normalized()
        if edge is None:
            return
        indexed[_edge_pair_key(edge)] = edge

    for raw in licensed_edges or ():
        if isinstance(raw, RelEdge):
            edge = raw.normalized()
        elif isinstance(raw, dict):
            edge = RelEdge(
                source=str(raw.get("source") or ""),
                target=str(raw.get("target") or ""),
                relation=str(
                    raw.get("relation") or raw.get("tag") or "SAME_ENTITY_AS"
                ),
                derived_marked=bool(raw.get("derived_marked") or raw.get("derived")),
            ).normalized()
        else:
            edge = None
        if edge is None:
            continue
        if edge.relation not in {"SAME_ENTITY_AS", "EQUIVALENT_TO"}:
            continue
        indexed[_edge_pair_key(edge)] = edge

    by_label: dict[str, list[str]] = {}
    for party in getattr(model, "parties", ()) or ():
        pid = str(getattr(party, "party_id", "") or "").strip()
        label = _normalized_party_label(getattr(party, "label", ""))
        if not pid or not label:
            continue
        by_label.setdefault(label, []).append(pid)
    for party_ids in by_label.values():
        unique = list(dict.fromkeys(party_ids))
        for i, left in enumerate(unique):
            for right in unique[i + 1:]:
                add(left, right)

    return tuple(indexed.values())


def _edge_pair_key(edge: RelEdge) -> tuple[str, str, str]:
    return (edge.source, edge.target, edge.relation)


def identity_relational_errors(
    model: Any,
    *,
    licensed_identity_edges: Sequence[RelEdge | dict] | None = None,
) -> list[str]:
    """RelEnt post-check for ``SAME_ENTITY_AS`` closure and action-scoped non-transfer.

    - ``RELATIONAL_IDENTITY_CLOSURE``: licensed identity edges close under
      symmetry / transitivity (identity family).
    - Quantity may move along identity (no error when derivation unmarked).
    - ``RELATIONAL_NON_TRANSFER``: ``harm_under_action`` must not cross distinct
      actions even when parties are identity-linked; identity must not compose
      with ``ALTERNATIVE_OF`` into shared-effect equality.
    """
    errors: list[str] = []
    identity_edges = project_same_entity_edges(
        model, licensed_edges=licensed_identity_edges,
    )
    alt_edges = project_alternative_of_edges(model)
    all_edges = (*identity_edges, *alt_edges)
    errors.extend(forbidden_composites(all_edges))

    if identity_edges:
        expect: list[RelEdge] = []
        for edge in identity_edges:
            expect.append(RelEdge(
                source=edge.target,
                target=edge.source,
                relation="SAME_ENTITY_AS",
            ))
        base = list(identity_edges)
        for left in base:
            for right in base:
                if left.target != right.source:
                    continue
                if left.source == right.target:
                    continue
                expect.append(RelEdge(
                    source=left.source,
                    target=right.target,
                    relation="SAME_ENTITY_AS",
                ))
        for msg in relational_entailment_errors(
            identity_edges,
            expect_derived=tuple(expect),
        ):
            if msg.startswith("RELATIONAL_IDENTITY_CLOSURE"):
                errors.append(msg)
            elif "expected derived" in msg:
                detail = msg.split(": ", 1)[-1] if ": " in msg else msg
                errors.append(f"RELATIONAL_IDENTITY_CLOSURE: {detail}")
            else:
                errors.append(msg)

    # Cross-branch harm via identity-linked parties.
    closed_identity = {
        frozenset((e.source, e.target))
        for e in closure_edges(identity_edges)
        if e.relation in {"SAME_ENTITY_AS", "EQUIVALENT_TO"} and e.source != e.target
    }
    effects = list(getattr(model, "effects", ()) or ())
    for left in effects:
        if str(getattr(left, "polarity", "") or "").upper() != "ADVERSE":
            continue
        if str(getattr(left, "modality", "") or "").upper() != "CERTAIN":
            continue
        if str(getattr(left, "directness", "") or "").upper() == "FOREGONE":
            continue
        left_party = str(getattr(left, "party_id", "") or "").strip()
        left_action = str(getattr(left, "action_id", "") or "").strip()
        left_qty = {
            str(span).casefold()
            for span in (getattr(left, "quantities", ()) or ())
            if str(span).strip()
        }
        if not left_party or not left_action:
            continue
        for right in effects:
            right_party = str(getattr(right, "party_id", "") or "").strip()
            right_action = str(getattr(right, "action_id", "") or "").strip()
            if not right_party or not right_action or right_action == left_action:
                continue
            if frozenset((left_party, right_party)) not in closed_identity:
                continue
            right_qty = {
                str(span).casefold()
                for span in (getattr(right, "quantities", ()) or ())
                if str(span).strip()
            }
            shared = sorted(left_qty & right_qty)
            if not shared:
                continue
            # Same magnitude moved onto an identity-linked party under another action.
            for msg in function_transfer_errors(
                "SAME_ENTITY_AS",
                "harm_under_action",
                source_action=left_action,
                target_action=right_action,
            ):
                rid = str(getattr(right, "effect_id", "") or "").strip() or "?"
                errors.append(
                    f"{rid} {msg}; identity-linked parties "
                    f"{left_party}/{right_party} must not share opposed-action "
                    f"harm spans {shared}"
                )

    return list(dict.fromkeys(errors))


# CausalLink relations that project onto RelEnt CAUSES (directed, restricted).
_CAUSES_PROJECTED = frozenset({
    "CAUSES", "ENABLES", "INCREASES", "ACCELERATES",
})


def project_causes_edges(
    model: Any,
    *,
    licensed_edges: Sequence[RelEdge | dict] | None = None,
) -> tuple[RelEdge, ...]:
    """Project allowing causal links onto directed ``CAUSES`` RelEnt edges."""
    indexed: dict[tuple[str, str, str], RelEdge] = {}

    for raw in licensed_edges or ():
        if isinstance(raw, RelEdge):
            edge = raw.normalized()
        elif isinstance(raw, dict):
            edge = RelEdge(
                source=str(raw.get("source") or ""),
                target=str(raw.get("target") or ""),
                relation=str(raw.get("relation") or raw.get("tag") or "CAUSES"),
                derived_marked=bool(raw.get("derived_marked") or raw.get("derived")),
            ).normalized()
        else:
            edge = None
        if edge is None or edge.relation != "CAUSES":
            continue
        indexed[(edge.source, edge.target, edge.relation)] = edge

    for link in getattr(model, "causal_links", ()) or ():
        rel = str(getattr(link, "relation", "") or "").upper()
        if rel not in _CAUSES_PROJECTED:
            continue
        src = str(getattr(link, "source_id", "") or "").strip()
        tgt = str(getattr(link, "target_id", "") or "").strip()
        if not src or not tgt or src == tgt:
            continue
        edge = RelEdge(source=src, target=tgt, relation="CAUSES").normalized()
        if edge is not None:
            indexed[(edge.source, edge.target, edge.relation)] = edge
    return tuple(indexed.values())


def project_before_edges(
    model: Any,
    *,
    licensed_edges: Sequence[RelEdge | dict] | None = None,
) -> tuple[RelEdge, ...]:
    """Project host-licensed temporal ``BEFORE`` edges (no discourse clock).

    World models do not yet own a temporal-link table; Hypothesis / adapters
    pass ``licensed_edges``. Duck-types an optional ``temporal_links`` iterable
    with ``source_id`` / ``target_id`` when present.
    """
    indexed: dict[tuple[str, str, str], RelEdge] = {}

    for raw in licensed_edges or ():
        if isinstance(raw, RelEdge):
            edge = raw.normalized()
        elif isinstance(raw, dict):
            edge = RelEdge(
                source=str(raw.get("source") or ""),
                target=str(raw.get("target") or ""),
                relation=str(raw.get("relation") or raw.get("tag") or "BEFORE"),
                derived_marked=bool(raw.get("derived_marked") or raw.get("derived")),
            ).normalized()
        else:
            edge = None
        if edge is None or edge.relation != "BEFORE":
            continue
        indexed[(edge.source, edge.target, edge.relation)] = edge

    for link in getattr(model, "temporal_links", ()) or ():
        src = str(
            getattr(link, "source_id", None)
            or getattr(link, "before_id", None)
            or ""
        ).strip()
        tgt = str(
            getattr(link, "target_id", None)
            or getattr(link, "after_id", None)
            or ""
        ).strip()
        if not src or not tgt or src == tgt:
            continue
        edge = RelEdge(source=src, target=tgt, relation="BEFORE").normalized()
        if edge is not None:
            indexed[(edge.source, edge.target, edge.relation)] = edge
    return tuple(indexed.values())


def directionality_relational_errors(
    model: Any,
    *,
    licensed_causes_edges: Sequence[RelEdge | dict] | None = None,
    licensed_before_edges: Sequence[RelEdge | dict] | None = None,
) -> list[str]:
    """RelEnt post-check for CAUSES/BEFORE directionality and restricted hops.

    - ``RELATIONAL_DIRECTIONALITY``: no invented reverse; no BEFORE/DERIVED_FROM
      base antisymmetry violations.
    - ``RELATIONAL_TRANSITIVITY``: CAUSES must not auto-derive A→C; BEFORE may.
    """
    causes = project_causes_edges(model, licensed_edges=licensed_causes_edges)
    before = project_before_edges(model, licensed_edges=licensed_before_edges)
    derived = project_averted_derived_edges(model)
    edges = (*causes, *before, *derived)
    if not edges:
        return []
    errors = list(forbidden_composites(edges))
    skip_forbids = tuple(
        RelEdge(source=a.source, target=c.target, relation="CAUSES")
        for a in causes
        for c in causes
        if a.target == c.source and a.source != c.target
    )
    if skip_forbids:
        for msg in relational_entailment_errors(
            causes,
            forbid_derived=skip_forbids,
        ):
            errors.append(msg)
    before_expect: list[RelEdge] = []
    for left in before:
        for right in before:
            if left.target != right.source or left.source == right.target:
                continue
            before_expect.append(RelEdge(
                source=left.source, target=right.target, relation="BEFORE",
            ))
    if before_expect:
        for msg in relational_entailment_errors(
            before,
            expect_derived=tuple(before_expect),
        ):
            if "expected derived" in msg:
                detail = msg.split(": ", 1)[-1] if ": " in msg else msg
                errors.append(f"RELATIONAL_TRANSITIVITY: {detail}")
            else:
                errors.append(msg)
    return list(dict.fromkeys(errors))
