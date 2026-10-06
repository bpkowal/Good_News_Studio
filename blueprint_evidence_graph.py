"""Hypothesis records on the existing proposal envelope.

Producers attach the same record shape. They do not mint schema 1.3. RelEnt
still weights the admitted frozen world, not these pre-world guesses.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from blueprint_discourse import discourse_bind_rows, discourse_citeable_atoms
from z10_world_model_adapter import segment_source_clauses


ATTACH_TYPES = {
    "PREDICATION", "PARTICIPANT", "CONDITIONAL_ON", "QUANTITY", "OPTION_OF",
    "MODALITY", "EVENT_LINK",
}
PRODUCERS = {"z10", "cloze_copy", "llm_note", "amr"}
PROVENANCES = {
    "SOURCE_ASSERTED", "STRUCTURALLY_DERIVED",
    "WORLD_KNOWLEDGE_HYPOTHESIS", "UNRESOLVED",
}
LICENSED_PROVENANCE = {"SOURCE_ASSERTED", "STRUCTURALLY_DERIVED"}
HYPOTHESIS_KEYS = {
    "hypothesis_id", "producer", "span", "clause_ids", "z10_candidate_ids",
    "hypothesis", "scope", "provenance", "atom_ids",
}

# Catalog of hops that already drop meaning. This is not a sixth IR.
IR_HOPS = (
    {
        "name": "z10_package",
        "objects": ("candidates", "nodes", "scope", "evidence"),
        "does_not_carry": ("schema_1_3_kinds", "admitted_world"),
    },
    {
        "name": "cloze_slots",
        "objects": ("copied_spans", "semantic_recoveries"),
        "does_not_carry": ("licensed_1_3", "unconsumed_z10_unless_attached"),
    },
    {
        "name": "blueprint_slot_bindings",
        "objects": ("template_fills",),
        "does_not_carry": ("shared_hypothesis_records",),
    },
    {
        "name": "proposal_construction_provenance",
        "objects": ("atom_origin", "source_span"),
        "does_not_carry": ("z10_type_and_scope",),
    },
    {
        "name": "schema_1_3_world",
        "objects": ("parties", "actions", "effects"),
        "does_not_carry": ("pre_world_hypotheses",),
    },
    {
        "name": "relent_workspace",
        "objects": ("admitted_frozen_world_trace",),
        "does_not_carry": ("evidence_graph_hypotheses",),
    },
)

_EXCLUSIVITY_SPAN = (
    r"\b(?:but\s+)?not\s+both\b|"
    r"\b(?:cannot|can\s+not|can't)\b[^.!?]{0,80}\bboth\b"
)


def source_copy_validation() -> dict[str, Any]:
    return {
        "contract_valid": True,
        "status": "source_copy",
        "reason": "Authorized from copied source spans with clause evidence.",
    }


def hypothesis(*, hypothesis_id: str, producer: str, hypothesis: Mapping[str, Any],
               span: str = "", clause_ids: Sequence[str] = (),
               z10_candidate_ids: Sequence[str] = (),
               scope: Mapping[str, Any] | None = None,
               provenance: str = "UNRESOLVED",
               atom_ids: Sequence[str] = ()) -> dict[str, Any]:
    if producer not in PRODUCERS:
        raise ValueError(f"unknown hypothesis producer: {producer}")
    if provenance not in PROVENANCES:
        raise ValueError(f"unknown hypothesis provenance: {provenance}")
    return {
        "hypothesis_id": hypothesis_id,
        "producer": producer,
        "span": span or "",
        "clause_ids": list(clause_ids),
        "z10_candidate_ids": list(z10_candidate_ids),
        "hypothesis": dict(hypothesis),
        "scope": dict(scope) if scope else {"contexts": [{"kind": "UNRESOLVED"}]},
        "provenance": provenance,
        "atom_ids": list(atom_ids),
    }


def is_licensed(row: Mapping[str, Any]) -> bool:
    if row.get("provenance") not in LICENSED_PROVENANCE:
        return False
    if row.get("z10_candidate_ids"):
        return True
    return bool(row.get("span") and row.get("clause_ids"))


def clause_ids_for_span(clauses: Sequence[Mapping[str, Any]], span: str) -> list[str]:
    needle = (span or "").strip().casefold()
    if not needle:
        return []
    ids = []
    for row in clauses:
        ident = row.get("clause_id") or row.get("id")
        text = (row.get("text") or "")
        if ident and needle in text.casefold():
            ids.append(ident)
    return ids


def hypotheses_from_z10(package: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Wrap frozen Z10 candidates. Does not modify the package."""
    text = ((package.get("document") or {}).get("text") or "")
    clauses = segment_source_clauses(text)
    evidence = {row["id"]: row for row in package.get("evidence") or []}
    nodes = {row["id"]: row for row in package.get("nodes") or []}
    rows: list[dict[str, Any]] = []
    for candidate in package.get("candidates") or []:
        if candidate.get("type") not in ATTACH_TYPES:
            continue
        span = _span_from_candidate(candidate, evidence, nodes)
        scope = candidate.get("scope") or {}
        contexts = list(scope.get("contexts") or [])
        rows.append(hypothesis(
            hypothesis_id=f"z10:{candidate['id']}",
            producer="z10",
            span=span,
            clause_ids=clause_ids_for_span(clauses, span),
            z10_candidate_ids=[candidate["id"]],
            hypothesis={
                "kind": candidate["type"],
                "value": candidate.get("value"),
                "arguments": candidate.get("arguments") or {},
            },
            scope={"contexts": contexts or [{"kind": "UNRESOLVED"}]},
            provenance="SOURCE_ASSERTED",
        ))
    for clause in clauses:
        if re.search(_EXCLUSIVITY_SPAN, clause["text"], flags=re.I):
            rows.append(hypothesis(
                hypothesis_id=f"z10-excl:{clause['clause_id']}",
                producer="z10",
                span=clause["text"],
                clause_ids=[clause["clause_id"]],
                hypothesis={"kind": "exclusivity", "value": "not_both"},
                provenance="SOURCE_ASSERTED",
            ))
    return rows


def hypotheses_from_cloze(text: str, accepted: Mapping[str, str],
                          recoveries: Mapping[str, Mapping[str, Any]],
                          clauses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slot, span in accepted.items():
        if str(slot).endswith("implied_process"):
            continue
        recovery = recoveries.get(slot) or {}
        candidate_ids = list(recovery.get("z10_candidate_ids") or [])
        used_z10 = recovery.get("producer") == "parsing_game_Z10" and candidate_ids
        kind = {
            "quantity": "QUANTITY",
            "exclusivity": "exclusivity",
        }.get(slot, "copied_span")
        rows.append(hypothesis(
            hypothesis_id=f"cloze:{slot}",
            producer="z10" if used_z10 else "cloze_copy",
            span=span or "",
            clause_ids=clause_ids_for_span(clauses, span or ""),
            z10_candidate_ids=candidate_ids,
            hypothesis={"kind": kind, "slot": slot, "value": span},
            provenance="SOURCE_ASSERTED",
        ))
    return rows


def hypotheses_from_implied_notes(slots: Mapping[str, Any],
                                  clauses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, value in slots.items():
        if not (str(name).endswith("implied_process") and value):
            continue
        rows.append(hypothesis(
            hypothesis_id=f"llm:{name}",
            producer="llm_note",
            span=str(value),
            clause_ids=clause_ids_for_span(clauses, str(value)),
            hypothesis={"kind": "implied_process", "slot": name, "value": value},
            provenance="WORLD_KNOWLEDGE_HYPOTHESIS",
        ))
    return rows


def assemble_evidence_graph(*, package: Mapping[str, Any] | None = None,
                            text: str = "",
                            clauses: Sequence[Mapping[str, Any]] = (),
                            accepted: Mapping[str, str] | None = None,
                            recoveries: Mapping[str, Mapping[str, Any]] | None = None,
                            slots: Mapping[str, Any] | None = None,
                            world: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    graph: list[dict[str, Any]] = []
    if package is not None:
        graph.extend(hypotheses_from_z10(package))
    clause_rows = list(clauses) or (
        segment_source_clauses(text) if text else []
    )
    graph.extend(hypotheses_from_cloze(text, accepted or {}, recoveries or {}, clause_rows))
    graph.extend(hypotheses_from_implied_notes(slots or {}, clause_rows))
    if world is not None:
        graph = bind_world_atoms(graph, world, clause_rows)
    return graph


def bind_world_atoms(graph: Sequence[Mapping[str, Any]], world: Mapping[str, Any],
                     clauses: Sequence[Mapping[str, Any]],
                     *, producer: str = "cloze_copy") -> list[dict[str, Any]]:
    rows = [dict(row) for row in graph]
    for action in world.get("actions") or []:
        _cite(rows, f"action:{action['action_id']}",
              action.get("intervention") or "",
              list(action.get("clause_ids") or []),
              clauses, producer, "SOURCE_ASSERTED")
    for effect in world.get("effects") or []:
        if effect.get("derivation_operation") == "AVERTED_ALTERNATIVE_HARM":
            continue
        provenance = (
            "STRUCTURALLY_DERIVED"
            if effect.get("derivation_operation") == "EXCLUSIVE_ALLOCATION_COMPLEMENT"
            else "SOURCE_ASSERTED"
        )
        span = effect.get("source_proposition") or effect.get("outcome") or ""
        _cite(rows, f"effect:{effect['effect_id']}", span,
              list(effect.get("clause_ids") or []),
              clauses, producer, provenance)
    for atom_id, span, clause_ids in discourse_bind_rows(world):
        _cite(rows, atom_id, span, clause_ids, clauses, producer, "SOURCE_ASSERTED")
    return rows


def citeable_atoms(world: Mapping[str, Any]) -> list[str]:
    atoms = [f"action:{row['action_id']}" for row in world.get("actions") or []]
    for effect in world.get("effects") or []:
        if effect.get("derivation_operation") == "AVERTED_ALTERNATIVE_HARM":
            continue
        atoms.append(f"effect:{effect['effect_id']}")
    atoms.extend(discourse_citeable_atoms(world))
    return atoms


def citation_errors(graph: Sequence[Mapping[str, Any]] | None,
                    world: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(world, dict):
        return []
    licensed: set[str] = set()
    for row in graph or []:
        if is_licensed(row):
            licensed.update(row.get("atom_ids") or [])
    return [
        f"Authorized world atom {atom} has no licensed hypothesis"
        for atom in citeable_atoms(world)
        if atom not in licensed
    ]


def exclusive_allocation_signals(records: Sequence[Mapping[str, Any]],
                                 package: Mapping[str, Any] | None = None) -> dict[str, Any]:
    quantity_ids: list[str] = []
    quantity_spans: list[str] = []
    option_propositions: set[str] = set()
    conditional: dict[str, Any] = {}
    exclusivity_clause_ids: list[str] = []
    for row in records:
        kind = ((row.get("hypothesis") or {}).get("kind") or "")
        value = (row.get("hypothesis") or {}).get("value") or {}
        arguments = (row.get("hypothesis") or {}).get("arguments") or {}
        if kind == "QUANTITY":
            exact = isinstance(value, dict) and value.get("operator") == "exact"
            copied = not isinstance(value, dict)
            if exact or copied:
                quantity_ids.extend(row.get("z10_candidate_ids") or [])
                if row.get("span"):
                    quantity_spans.append(row["span"])
        if kind == "OPTION_OF" and arguments.get("proposition"):
            option_propositions.add(arguments["proposition"])
        if kind == "CONDITIONAL_ON" and arguments.get("condition"):
            ident = (row.get("z10_candidate_ids") or [row.get("hypothesis_id")])[0]
            conditional[arguments["condition"]] = {
                "id": ident,
                "arguments": arguments,
            }
        span = row.get("span") or ""
        if kind == "exclusivity" or re.search(_EXCLUSIVITY_SPAN, span, flags=re.I):
            exclusivity_clause_ids.extend(row.get("clause_ids") or [])
    if not exclusivity_clause_ids and package is not None:
        text = ((package.get("document") or {}).get("text") or "")
        exclusivity_clause_ids = [
            row["clause_id"] for row in segment_source_clauses(text)
            if re.search(r"\b(?:but\s+)?not\s+both\b", row["text"], flags=re.I)
        ]
    return {
        "quantity_ids": list(dict.fromkeys(quantity_ids)),
        "quantity_spans": quantity_spans,
        "option_propositions": option_propositions,
        "conditional": conditional,
        "exclusivity_clause_ids": list(dict.fromkeys(exclusivity_clause_ids)),
    }


def overlay_hypotheses(graph: Sequence[Mapping[str, Any]],
                       world: Mapping[str, Any]) -> list[dict[str, Any]]:
    kept_actions = {row["action_id"] for row in world.get("actions") or []}
    kept_effects = {row["effect_id"] for row in world.get("effects") or []}
    kept_discourse = {atom.partition(":")[2] for atom in discourse_citeable_atoms(world)}
    rows = []
    for row in graph:
        if row.get("provenance") == "WORLD_KNOWLEDGE_HYPOTHESIS":
            rows.append(dict(row))
            continue
        cites_kept = False
        for atom in row.get("atom_ids") or []:
            kind, _, ident = atom.partition(":")
            if kind == "action" and ident in kept_actions:
                cites_kept = True
            if kind == "effect" and ident in kept_effects:
                cites_kept = True
            if kind in {"proposition", "report", "commitment", "modal", "norm"} and ident in kept_discourse:
                cites_kept = True
        if not cites_kept:
            rows.append(dict(row))
    return rows


def _span_from_candidate(candidate: Mapping[str, Any],
                         evidence: Mapping[str, Mapping[str, Any]],
                         nodes: Mapping[str, Mapping[str, Any]]) -> str:
    for evidence_id in candidate.get("evidence_ids") or []:
        text = (evidence.get(evidence_id) or {}).get("text") or ""
        if text:
            return text
    for value in (candidate.get("arguments") or {}).values():
        if isinstance(value, str) and value in nodes:
            label = nodes[value].get("label") or ""
            if label:
                return label
    return ""


def _cite(graph: list[dict[str, Any]], atom_id: str, span: str,
          clause_ids: Sequence[str], clauses: Sequence[Mapping[str, Any]],
          producer: str, provenance: str) -> None:
    matched = False
    want = (span or "").casefold()
    wanted_clauses = set(clause_ids or [])
    for row in graph:
        row_span = (row.get("span") or "").casefold()
        overlap = bool(want and row_span and (row_span in want or want in row_span))
        shared = bool(wanted_clauses and wanted_clauses & set(row.get("clause_ids") or []))
        if overlap or shared:
            atoms = row.setdefault("atom_ids", [])
            if atom_id not in atoms:
                atoms.append(atom_id)
            matched = True
    if matched:
        return
    host_clauses = list(clause_ids) or clause_ids_for_span(clauses, span)
    graph.append(hypothesis(
        hypothesis_id=f"bind:{atom_id}",
        producer=producer,
        span=span,
        clause_ids=host_clauses,
        hypothesis={"kind": "world_atom", "atom_id": atom_id},
        provenance=provenance,
        atom_ids=[atom_id],
    ))
