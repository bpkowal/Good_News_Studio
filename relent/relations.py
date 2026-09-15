"""Relation vocabulary and typed property table for RelEnt.

Tags are data, not English parsers. Hosts license edges; RelEnt answers
closure, symmetry, and which functions may move. Stimulus equivalence is one
family (``SAME_ENTITY_AS`` / ``EQUIVALENT_TO``), not the whole algebra.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

RelationTag = Literal[
    "EQUIVALENT_TO",
    "SAME_ENTITY_AS",
    "PARAPHRASE_OF",
    "DERIVED_FROM",
    "SUBSET_OF",
    "APPROXIMATES",
    "CAUSES",
    "ALTERNATIVE_OF",
    "BEFORE",
]

BoolOrRestricted = Literal[True, False, "restricted"]
QuantityTransfer = Literal["none", "along_edge", "via_derivation_only"]

RELATION_TAGS: Final[frozenset[str]] = frozenset({
    "EQUIVALENT_TO",
    "SAME_ENTITY_AS",
    "PARAPHRASE_OF",
    "DERIVED_FROM",
    "SUBSET_OF",
    "APPROXIMATES",
    "CAUSES",
    "ALTERNATIVE_OF",
    "BEFORE",
})


@dataclass(frozen=True, slots=True)
class RelationSpec:
    """Declared algebraic properties for one relation tag."""

    tag: str
    reflexive: bool
    symmetric: bool
    transitive: BoolOrRestricted
    status_transfers: bool = False
    status_transfers_if_derived: bool = False
    quantity_transfers: QuantityTransfer = "none"
    action_scoped: bool = False
    notes: str = ""


# Identity family: SAME_ENTITY_AS is the preferred name; EQUIVALENT_TO aliases it.
_IDENTITY = RelationSpec(
    tag="SAME_ENTITY_AS",
    reflexive=True,
    symmetric=True,
    transitive=True,
    status_transfers=True,
    quantity_transfers="along_edge",
    action_scoped=True,
    notes="Party/alias identity. Harm-under-action does not cross branches.",
)
_EQUIVALENT = RelationSpec(
    tag="EQUIVALENT_TO",
    reflexive=True,
    symmetric=True,
    transitive=True,
    status_transfers=True,
    quantity_transfers="along_edge",
    action_scoped=True,
    notes="Alias of SAME_ENTITY_AS for claim-level equivalence.",
)
_PARAPHRASE = RelationSpec(
    tag="PARAPHRASE_OF",
    reflexive=False,
    symmetric=True,
    transitive="restricted",
    status_transfers=True,
    quantity_transfers="none",
    notes="Licensed surface paraphrase; status may transfer.",
)
_DERIVED = RelationSpec(
    tag="DERIVED_FROM",
    reflexive=False,
    symmetric=False,
    transitive="restricted",
    status_transfers_if_derived=True,
    quantity_transfers="via_derivation_only",
    notes="Host must mark derived; quantity only via named derivation.",
)
_SUBSET = RelationSpec(
    tag="SUBSET_OF",
    reflexive=True,
    symmetric=False,
    transitive=True,
    quantity_transfers="none",
    notes="Inclusion; does not escalate precision.",
)
_APPROX = RelationSpec(
    tag="APPROXIMATES",
    reflexive=False,
    symmetric=True,
    transitive=False,
    quantity_transfers="none",
    notes="Neighbor of QUANTITY_PRECISION_NON_ESCALATION.",
)
_CAUSES = RelationSpec(
    tag="CAUSES",
    reflexive=False,
    symmetric=False,
    transitive="restricted",
    action_scoped=True,
    notes="No auto-symmetry; no free A→C without host license.",
)
_ALTERNATIVE = RelationSpec(
    tag="ALTERNATIVE_OF",
    reflexive=False,
    symmetric=True,
    transitive=False,
    quantity_transfers="via_derivation_only",
    action_scoped=True,
    notes="Mutually exclusive actions/effects; not effect equality.",
)
_BEFORE = RelationSpec(
    tag="BEFORE",
    reflexive=False,
    symmetric=False,
    transitive=True,
    notes="Temporal order; feedback cycles deferred to Temporal operator.",
)

RELATION_SPECS: Final[dict[str, RelationSpec]] = {
    spec.tag: spec
    for spec in (
        _IDENTITY,
        _EQUIVALENT,
        _PARAPHRASE,
        _DERIVED,
        _SUBSET,
        _APPROX,
        _CAUSES,
        _ALTERNATIVE,
        _BEFORE,
    )
}

# Tags that share identity-family closure (treated as one undirected class).
_IDENTITY_FAMILY: Final[frozenset[str]] = frozenset({
    "SAME_ENTITY_AS",
    "EQUIVALENT_TO",
})


def relation_properties(tag: str) -> RelationSpec | None:
    """Return the RelationSpec for a tag, or None if unknown."""
    key = str(tag or "").strip().upper()
    return RELATION_SPECS.get(key)


def normalize_relation_tag(tag: str) -> str:
    """Uppercase known tags; empty string if unknown."""
    key = str(tag or "").strip().upper()
    return key if key in RELATION_SPECS else ""


def identity_family(tag: str) -> bool:
    return normalize_relation_tag(tag) in _IDENTITY_FAMILY


def status_transfers(
    relation: str,
    *,
    derived_marked: bool = False,
) -> bool:
    """True when epistemic status may move along this licensed relation."""
    spec = relation_properties(relation)
    if spec is None:
        return False
    if spec.status_transfers:
        return True
    if spec.status_transfers_if_derived and derived_marked:
        return True
    return False


def quantity_may_transfer(
    relation: str,
    *,
    derivation_marked: bool = False,
) -> bool:
    """True when a magnitude span may move along this relation."""
    spec = relation_properties(relation)
    if spec is None:
        return False
    if spec.quantity_transfers == "along_edge":
        return True
    if spec.quantity_transfers == "via_derivation_only" and derivation_marked:
        return True
    return False
