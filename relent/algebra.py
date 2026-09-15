"""Typed relational entailment: closure, forbidden composites, function transfer.

Pure calculator over licensed edges. Does not own world state. Hosts supply
edges; RelEnt derives what the RelationSpec table allows and rejects illegal
function moves (quantity / status / harm-under-action).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from .relations import (
    RELATION_TAGS,
    identity_family,
    normalize_relation_tag,
    quantity_may_transfer,
    relation_properties,
    status_transfers,
)

FunctionKind = Literal["quantity", "status", "harm_under_action"]


@dataclass(frozen=True, slots=True)
class RelEdge:
    """One directed licensed edge. Symmetric specs also imply the reverse."""

    source: str
    target: str
    relation: str
    derived_marked: bool = False

    def normalized(self) -> RelEdge | None:
        tag = normalize_relation_tag(self.relation)
        if not tag or not str(self.source).strip() or not str(self.target).strip():
            return None
        return RelEdge(
            source=str(self.source).strip(),
            target=str(self.target).strip(),
            relation=tag,
            derived_marked=bool(self.derived_marked),
        )

    def key(self) -> tuple[str, str, str, bool]:
        return (self.source, self.target, self.relation, self.derived_marked)


def _edge_key(edge: RelEdge) -> tuple[str, str, str]:
    return (edge.source, edge.target, edge.relation)


def _as_edge(item: RelEdge | dict) -> RelEdge | None:
    if isinstance(item, RelEdge):
        return item.normalized()
    if isinstance(item, dict):
        return RelEdge(
            source=str(item.get("source") or ""),
            target=str(item.get("target") or ""),
            relation=str(item.get("relation") or item.get("tag") or ""),
            derived_marked=bool(item.get("derived_marked") or item.get("derived")),
        ).normalized()
    return None


def _index_edges(edges: Sequence[RelEdge | dict]) -> dict[tuple[str, str, str], RelEdge]:
    indexed: dict[tuple[str, str, str], RelEdge] = {}
    for raw in edges:
        edge = _as_edge(raw)
        if edge is None:
            continue
        indexed[_edge_key(edge)] = edge
    return indexed


def closure_edges(
    edges: Sequence[RelEdge | dict],
    *,
    max_rounds: int = 32,
) -> tuple[RelEdge, ...]:
    """Add reflexive / symmetric / transitive edges allowed by RelationSpec.

    ``transitive=\"restricted\"`` relations are **not** auto-closed; hosts must
    license each hop. Identity-family tags (``SAME_ENTITY_AS``, ``EQUIVALENT_TO``)
    close together as one undirected equivalence class.
    """
    pool = dict(_index_edges(edges))

    def add(edge: RelEdge) -> bool:
        key = _edge_key(edge)
        if key in pool:
            return False
        pool[key] = edge
        return True

    # Reflexive closures over nodes that already appear.
    nodes = {edge.source for edge in pool.values()} | {
        edge.target for edge in pool.values()
    }
    for node in nodes:
        for tag, spec in (
            (tag, relation_properties(tag)) for tag in RELATION_TAGS
        ):
            if spec is None or not spec.reflexive:
                continue
            # Only introduce reflexive identity when some identity edge exists,
            # or when the tag already appears in the graph.
            if identity_family(tag):
                if not any(identity_family(e.relation) for e in pool.values()):
                    continue
            elif not any(e.relation == tag for e in pool.values()):
                continue
            add(RelEdge(source=node, target=node, relation=tag))

    changed = True
    rounds = 0
    while changed and rounds < max_rounds:
        changed = False
        rounds += 1
        snapshot = list(pool.values())
        # Symmetry.
        for edge in snapshot:
            spec = relation_properties(edge.relation)
            if spec is None or not spec.symmetric:
                continue
            if add(RelEdge(
                source=edge.target,
                target=edge.source,
                relation=edge.relation,
                derived_marked=edge.derived_marked,
            )):
                changed = True
        # Transitivity (true only — not restricted).
        for left in snapshot:
            left_spec = relation_properties(left.relation)
            if left_spec is None or left_spec.transitive is not True:
                continue
            for right in snapshot:
                if left.target != right.source:
                    continue
                if identity_family(left.relation) and identity_family(right.relation):
                    out_tag = "SAME_ENTITY_AS"
                elif left.relation != right.relation:
                    continue
                else:
                    out_tag = left.relation
                out_spec = relation_properties(out_tag)
                if out_spec is None or out_spec.transitive is not True:
                    continue
                if add(RelEdge(
                    source=left.source,
                    target=right.target,
                    relation=out_tag,
                    derived_marked=left.derived_marked and right.derived_marked,
                )):
                    changed = True
    return tuple(sorted(pool.values(), key=lambda e: e.key()))


def derived_only(
    edges: Sequence[RelEdge | dict],
) -> tuple[RelEdge, ...]:
    """Edges present in closure but not in the input set (ignoring derived flag)."""
    base = {_edge_key(edge) for edge in _index_edges(edges).values()}
    return tuple(
        edge for edge in closure_edges(edges)
        if _edge_key(edge) not in base
    )


def forbidden_composites(
    edges: Sequence[RelEdge | dict],
) -> list[str]:
    """Reject illegal composites visible in the licensed edge set.

    - ``ALTERNATIVE_OF`` must not co-occur as identity / equivalence.
    - Restricted relations must not appear as auto-closed transitive hops
      in ``closure_edges`` (sanity check against the algebra itself).
    """
    indexed = _index_edges(edges)
    errors: list[str] = []
    alt_pairs = {
        frozenset((e.source, e.target))
        for e in indexed.values()
        if e.relation == "ALTERNATIVE_OF" and e.source != e.target
    }
    for pair in alt_pairs:
        a, b = tuple(pair)
        for tag in ("SAME_ENTITY_AS", "EQUIVALENT_TO"):
            if (a, b, tag) in indexed or (b, a, tag) in indexed:
                errors.append(
                    "RELATIONAL_NON_TRANSFER: ALTERNATIVE_OF "
                    f"{a}/{b} must not imply {tag}"
                )
    # Closure must never invent restricted-transitive hops.
    base_keys = set(indexed)
    for edge in derived_only(edges):
        spec = relation_properties(edge.relation)
        if spec is not None and spec.transitive == "restricted":
            if _edge_key(edge) not in base_keys:
                errors.append(
                    "RELATIONAL_TRANSITIVITY: restricted relation "
                    f"{edge.relation} must not auto-derive "
                    f"{edge.source}->{edge.target}"
                )
    errors.extend(directionality_errors(edges))
    return list(dict.fromkeys(errors))


def directionality_errors(
    edges: Sequence[RelEdge | dict],
) -> list[str]:
    """Reject illegal symmetry / reverse invention for directed relations.

    ``CAUSES``, ``BEFORE``, and ``DERIVED_FROM`` are not symmetric: closure must
    not invent the reverse. Licensing both directions in the base set is an
    error for ``BEFORE`` / ``DERIVED_FROM`` (contradiction). Mutual ``CAUSES`` in
    the base is left to hosts (feedback deferred); only invented reverses are
    rejected. Free ``CAUSES`` A→C skips stay under ``RELATIONAL_TRANSITIVITY``.
    """
    indexed = _index_edges(edges)
    errors: list[str] = []
    directed = ("CAUSES", "BEFORE", "DERIVED_FROM")
    antisymmetric_base = ("BEFORE", "DERIVED_FROM")
    seen_pairs: set[tuple[str, str, str]] = set()
    for edge in indexed.values():
        if edge.relation not in antisymmetric_base or edge.source == edge.target:
            continue
        reverse = (edge.target, edge.source, edge.relation)
        if reverse not in indexed:
            continue
        key = (edge.relation, *sorted((edge.source, edge.target)))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        errors.append(
            "RELATIONAL_DIRECTIONALITY: "
            f"{edge.relation} is not symmetric; both directions licensed "
            f"between {key[1]} and {key[2]}"
        )
    base_keys = set(indexed)
    for edge in derived_only(edges):
        if edge.relation not in directed:
            continue
        reverse_base = (edge.target, edge.source, edge.relation)
        if reverse_base in base_keys and _edge_key(edge) not in base_keys:
            errors.append(
                "RELATIONAL_DIRECTIONALITY: must not derive reverse "
                f"{edge.relation}({edge.source},{edge.target})"
            )
    return list(dict.fromkeys(errors))


def function_transfer_errors(
    relation: str,
    function_kind: FunctionKind,
    *,
    derivation_marked: bool = False,
    source_action: str = "",
    target_action: str = "",
) -> list[str]:
    """Whether a function may move along ``relation``.

    - ``quantity`` / ``status`` follow RelationSpec transfer flags.
    - ``harm_under_action`` never transfers across ``action_scoped`` relations
      when source and target actions differ (including ALTERNATIVE_OF).
    """
    tag = normalize_relation_tag(relation)
    if not tag:
        return [f"RELATIONAL_FUNCTION_TRANSFER: unknown relation {relation!r}"]
    spec = relation_properties(tag)
    assert spec is not None
    errors: list[str] = []
    kind = str(function_kind or "").strip().lower()
    if kind == "status":
        if not status_transfers(tag, derived_marked=derivation_marked):
            errors.append(
                f"RELATIONAL_FUNCTION_TRANSFER: status may not move along {tag}"
            )
    elif kind == "quantity":
        if not quantity_may_transfer(tag, derivation_marked=derivation_marked):
            errors.append(
                "RELATIONAL_FUNCTION_TRANSFER: quantity may not move along "
                f"{tag} without a licensed derivation"
                if spec.quantity_transfers == "via_derivation_only"
                else f"RELATIONAL_FUNCTION_TRANSFER: quantity may not move along {tag}"
            )
    elif kind == "harm_under_action":
        src = str(source_action or "").strip()
        dst = str(target_action or "").strip()
        if spec.action_scoped and src and dst and src != dst:
            errors.append(
                "RELATIONAL_NON_TRANSFER: harm_under_action must not transfer "
                f"from {src} to {dst} along action-scoped {tag}"
            )
        elif tag == "ALTERNATIVE_OF":
            errors.append(
                "RELATIONAL_NON_TRANSFER: harm_under_action must not transfer "
                "along ALTERNATIVE_OF"
            )
    else:
        errors.append(
            f"RELATIONAL_FUNCTION_TRANSFER: unknown function kind {function_kind!r}"
        )
    return errors


def relational_entailment_errors(
    edges: Sequence[RelEdge | dict],
    *,
    expect_derived: Sequence[RelEdge | dict] = (),
    forbid_derived: Sequence[RelEdge | dict] = (),
) -> list[str]:
    """Oracle helper: required derived edges present; forbidden absent; composites clean."""
    closed = {_edge_key(edge) for edge in closure_edges(edges)}
    errors = list(forbidden_composites(edges))
    for raw in expect_derived:
        edge = _as_edge(raw)
        if edge is None:
            errors.append("RELATIONAL_ENTAILMENT: malformed expect_derived edge")
            continue
        # Symmetry/transitivity may normalize identity to SAME_ENTITY_AS.
        candidates = {_edge_key(edge)}
        if identity_family(edge.relation):
            candidates.add((edge.source, edge.target, "SAME_ENTITY_AS"))
            candidates.add((edge.source, edge.target, "EQUIVALENT_TO"))
        if not candidates & closed:
            errors.append(
                "RELATIONAL_IDENTITY_CLOSURE: expected derived "
                f"{edge.relation}({edge.source},{edge.target})"
            )
    for raw in forbid_derived:
        edge = _as_edge(raw)
        if edge is None:
            continue
        if _edge_key(edge) in closed and _edge_key(edge) not in {
            _edge_key(e) for e in _index_edges(edges).values()
        }:
            errors.append(
                "RELATIONAL_NON_TRANSFER: must not derive "
                f"{edge.relation}({edge.source},{edge.target})"
            )
    return list(dict.fromkeys(errors))
