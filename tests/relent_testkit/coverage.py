"""Coverage metadata derived from the invariant catalog.

The former semantic-integrity scaffold duplicated this information in YAML.
Keeping the catalog authoritative prevents test discovery from depending on
untracked data files.
"""
from __future__ import annotations

from invariants.catalog import INVARIANTS


_SECTIONS = {
    "ANAPHOR_ENTITY_IDENTITY": 3,
    "QUANTIFIER_PARTY_COUNT": 1,
    "PLURAL_MEMBER_DISTINCTNESS": 2,
    "ELLIPSIS_PREDICATE_RESOLUTION": 4,
    "ADJECTIVE_MODIFIER_BINDING": 5,
    "TEMPORAL_ORDER_CONSISTENCY": 7,
    "VERB_ASPECT_CULMINATION": 8,
    "VERB_LEMMA_OUTCOME_BINDING": 8,
    "ATTITUDE_FACTIVITY": 9,
}


def _phenomenon_id(invariant_id: str) -> str:
    return invariant_id.casefold()


def _semantic_items():
    return tuple(
        item for item in INVARIANTS
        if item.layer == "semantic_integrity" and item.integrity_layers is not None
    )


def load_taxonomy() -> dict:
    phenomena = []
    for item in _semantic_items():
        layers = item.integrity_layers
        lanes = []
        if layers.structured_property:
            lanes.append("structured")
        if layers.grounding_property:
            lanes.append("grounding")
        source = {"type": item.source_type or "parliament_extension"}
        if item.id in _SECTIONS:
            source["section"] = _SECTIONS[item.id]
        row = {
            "id": _phenomenon_id(item.id),
            "source": source,
            "lanes": lanes,
        }
        if item.id == "SEMANTIC_STATUS_CONSERVATION":
            row["family"] = "status_conservation"
        phenomena.append(row)
    return {"phenomena": phenomena}


def load_map() -> dict:
    entries = []
    for item in _semantic_items():
        layers = item.integrity_layers
        tests = tuple(item.tests)
        module = ""
        for test in tests:
            head = test.split(".")
            if head and head[0] == "invariants" and len(head) >= 2:
                module = ".".join(head[:2])
                break
        seed_count = 2 if layers.seed_case else 0
        if item.id == "AVERTED_ALTERNATIVE_HARM":
            seed_count = 1
        seed_names = [f"{_phenomenon_id(item.id)}:{i}" for i in range(seed_count)]
        if item.id == "AVERTED_ALTERNATIVE_HARM":
            seed_names = ["averted_alternative_harm_grounding.yaml"]
        entries.append({
            "phenomenon_id": _phenomenon_id(item.id),
            "invariant_id": item.id,
            "enforcement": item.enforcement or "cataloged",
            "cases": [f"{_phenomenon_id(item.id)}:{i}" for i in range(seed_count)],
            # Transitional alias while the static tests move from file names to
            # the code-defined registry.
            "seeds": seed_names,
            "test_module": module,
        })
    return {"entries": entries}


def enforced_invariant_ids() -> tuple[str, ...]:
    return tuple(
        item.id for item in _semantic_items()
        if item.enforcement == "enforced"
    )


def coverage_report() -> str:
    lines = ["RelEnt semantic invariant coverage"]
    for row in load_map()["entries"]:
        lines.append(
            f"{row['phenomenon_id']}: {row['enforcement']} "
            f"({len(row['cases'])} cases; structured/grounding lanes)"
        )
    return "\n".join(lines)
