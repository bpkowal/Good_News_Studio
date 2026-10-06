"""Additive 1.4 discourse collections beside the 1.3 occurrence subgraph.

Reports, commitments, and modals are admitted as typed objects. Their content
propositions are never occurrence effects. RelEnt may cite the objects; it may
not treat attributed content as established.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from blueprint_kind_license import license_kind
from z10_world_model_adapter import segment_source_clauses


SCHEMA_VERSION = "1.4"
OCCURRENCE_KEYS = (
    "parties", "actions", "effects", "conditions",
    "temporal_relations", "causal_links", "counterfactual_links",
)
DISCOURSE_KEYS = (
    "propositions", "reports", "commitments",
    "modal_operators", "normative_propositions",
)
PROPOSITION_STATUSES = {
    "ASSERTED", "ATTRIBUTED", "COMMITTED_CONTENT", "GOVERNED", "UNASSERTED",
}
NON_OCCURRENCE_STATUSES = {
    "ATTRIBUTED", "COMMITTED_CONTENT", "GOVERNED", "UNASSERTED",
}
MODAL_FORCES = {
    "prediction", "possibility", "ability", "permission", "obligation",
    "unresolved",
}
NORM_FORCES = {"obligation", "permission", "prohibition", "unresolved"}

PROPOSITION_KEYS = {
    "proposition_id", "predication", "polarity", "status", "clause_ids",
}
REPORT_KEYS = {
    "report_id", "source_party_id", "speech_act", "content_proposition_id",
    "action_id", "clause_ids",
}
COMMITMENT_KEYS = {
    "commitment_id", "promisor_party_id", "promisee_party_id",
    "commitment_event", "content_proposition_id", "reliance", "breach",
    "clause_ids",
}
MODAL_KEYS = {
    "modal_id", "force", "bearer_party_id", "governed_proposition_id",
    "clause_ids",
}
NORM_KEYS = {
    "norm_id", "force", "bearer_party_id", "authority_party_id",
    "governed_proposition_id", "clause_ids",
}

_NONE = {"", "none", "n/a", "null", "unknown", "not stated"}


def ensure_schema(world: Mapping[str, Any] | None) -> dict[str, Any]:
    row = dict(world or {})
    row["schema_version"] = SCHEMA_VERSION
    for key in OCCURRENCE_KEYS + DISCOURSE_KEYS:
        row.setdefault(key, [])
    return row


def has_discourse(world: Mapping[str, Any] | None) -> bool:
    world = world or {}
    return any(world.get(key) for key in DISCOURSE_KEYS)


def occurrence_subgraph(world: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.3",
        **{key: list(world.get(key) or []) for key in OCCURRENCE_KEYS},
    }


def merge_discourse(admitted: Mapping[str, Any],
                    original: Mapping[str, Any]) -> dict[str, Any]:
    row = ensure_schema(admitted)
    for key in DISCOURSE_KEYS:
        row[key] = list(original.get(key) or row.get(key) or [])
    return row


def world_for_relent_compile(admitted: Mapping[str, Any],
                             original: Mapping[str, Any]) -> dict[str, Any]:
    """Keep native parse at 1.3 while leaving 1.4 collections on the compile dict."""
    full = merge_discourse(admitted, original)
    compile_world = dict(admitted or {})
    compile_world["schema_version"] = "1.3"
    for key in DISCOURSE_KEYS:
        compile_world[key] = list(full.get(key) or [])
    return compile_world


def attach_discourse_collections(occurrence: Mapping[str, Any],
                                 discourse: Mapping[str, Any]) -> dict[str, Any]:
    """Copy 1.4 collections and new parties onto an occurrence world.

    Promise speech actions stay off the occurrence action set. Extra parties
    (promisee, target) are unioned by label; effects are not merged.
    """
    world = ensure_schema(occurrence)
    remap: dict[str, str] = {}
    parties = list(world.get("parties") or [])
    existing = {
        (row.get("label") or "").casefold(): row["party_id"] for row in parties
    }
    used = {row["party_id"] for row in parties}
    next_index = len(parties)
    for party in discourse.get("parties") or []:
        label = (party.get("label") or "").casefold()
        if label and label in existing:
            remap[party["party_id"]] = existing[label]
            continue
        next_index += 1
        new_id = f"P{next_index}"
        while new_id in used:
            next_index += 1
            new_id = f"P{next_index}"
        row = dict(party)
        row["party_id"] = new_id
        remap[party["party_id"]] = new_id
        parties.append(row)
        used.add(new_id)
        if label:
            existing[label] = new_id
    world["parties"] = parties
    occurrence_actions = {row["action_id"] for row in world.get("actions") or []}
    party_fields = (
        "source_party_id", "promisor_party_id", "promisee_party_id",
        "bearer_party_id", "authority_party_id",
    )
    for key in DISCOURSE_KEYS:
        attached = []
        for row in discourse.get(key) or []:
            item = dict(row)
            for field in party_fields:
                if item.get(field):
                    item[field] = remap.get(item[field], item[field])
            if item.get("action_id") and item["action_id"] not in occurrence_actions:
                item["action_id"] = ""
            attached.append(item)
        world[key] = list(world.get(key) or []) + attached
    return world


def attach_source_discourse(text: str, world: Mapping[str, Any],
                            package: Mapping[str, Any] | None = None
                            ) -> dict[str, Any]:
    """Attach a source-copied promise as collections, never as extra actions."""
    world = ensure_schema(world)
    if world.get("commitments"):
        return world
    slots = promise_slots_from_source(text, package)
    if not slots:
        return world
    discourse, _notes = build_promise_reliance(text, slots, package)
    return attach_discourse_collections(world, discourse)


def promise_slots_from_source(text: str,
                              package: Mapping[str, Any] | None = None
                              ) -> dict[str, str] | None:
    """Copy a promise from Z10; do not invent parties or content."""
    return _promise_slots_from_package(text, package)


def _promise_slots_from_package(text: str,
                                package: Mapping[str, Any] | None
                                ) -> dict[str, str] | None:
    if not package:
        return None
    nodes = {node["id"]: node for node in package.get("nodes") or []}
    evidence = {row["id"]: row for row in package.get("evidence") or []}
    candidates = list(package.get("candidates") or [])
    promise_pred = None
    anchor_id = ""
    for row in candidates:
        if row.get("type") != "PREDICATION":
            continue
        anchor = (row.get("arguments") or {}).get("proposition")
        if (nodes.get(anchor) or {}).get("predicate", "").casefold() != "promise":
            continue
        promise_pred = row
        anchor_id = str(anchor)
        break
    if promise_pred is None:
        return None
    roles: list[tuple[str, str]] = []
    for row in candidates:
        if row.get("type") != "PARTICIPANT":
            continue
        if (row.get("arguments") or {}).get("proposition") != anchor_id:
            continue
        mention = (nodes.get((row.get("arguments") or {}).get("mention")) or {}).get("label")
        if mention:
            roles.append((str(row.get("value") or ""), mention))
    promisor = next(
        (mention for role, mention in roles
         if role in {"subject", "agent", "controller"}),
        "",
    )
    promisee = next(
        (mention for role, mention in roles
         if role in {"destination", "recipient", "beneficiary"}),
        "",
    )
    if not promisee:
        promisee = next(
            (mention for role, mention in roles
             if role == "object" and mention.casefold() != (promisor or "").casefold()),
            "",
        )
    event = ""
    for evidence_id in promise_pred.get("evidence_ids") or []:
        span = (evidence.get(evidence_id) or {}).get("text") or ""
        if re.search(r"\b(?:promised|promises|promise|committed|commits|commit|pledge)\b", span, re.I):
            event = _copy_present(text, span) or span
            break
    if not event:
        event = (_copy_present(text, "promised") or _copy_present(text, "promises")
                 or _copy_present(text, "promise"))
    content = ""
    children = {
        (row.get("arguments") or {}).get("child")
        for row in candidates
        if row.get("type") == "EVENT_LINK"
        and (row.get("arguments") or {}).get("parent") == anchor_id
    }
    for child in children:
        label = (nodes.get(child) or {}).get("label") or ""
        if label and label.casefold() != "promise":
            content = _copy_present(text, label) or label
            break
    if not content:
        match = re.search(r"\bwould\b[^.]*(?=\.|$)", text or "", re.I)
        if match:
            content = _copy_present(text, match.group(0)) or match.group(0)
    if event:
        head = (_copy_present(text, "promised") or _copy_present(text, "promises")
                or _copy_present(text, "committed") or _copy_present(text, "pledged")
                or _copy_present(text, "promise"))
        if head:
            event = head
    if not content:
        content = _promise_clause_complement(text, event)
    if not content:
        excluded = {(promisor or "").casefold(), (promisee or "").casefold()}
        theme = next(
            (mention for role, mention in roles
             if role in {"object", "theme", "patient", "stimulus"}
             and mention.casefold() not in excluded),
            "",
        )
        if theme:
            content = _copy_present(text, theme) or theme
    if not promisor or not event:
        return None
    return {
        "promisor": promisor,
        "promisee": promisee,
        "commitment_event": event,
        "commitment_content": content,
    }


def destination_from_package(package: Mapping[str, Any] | None) -> str:
    if not package:
        return ""
    nodes = {node["id"]: node for node in package.get("nodes") or []}
    for row in package.get("candidates") or []:
        if row.get("type") != "PARTICIPANT" or row.get("value") != "destination":
            continue
        mention = (nodes.get((row.get("arguments") or {}).get("mention")) or {}).get("label")
        if mention:
            return mention
    return ""


def _copy_present(text: str, snippet: str) -> str:
    snippet = " ".join(str(snippet or "").split())
    if not snippet or not text:
        return ""
    pattern = r"\s+".join(re.escape(part) for part in snippet.split())
    match = re.search(pattern, text, re.I)
    return text[match.start():match.end()] if match else ""


def _promise_clause_complement(text: str, event: str) -> str:
    """Copy the rest of the promise sentence; do not take later allocation clauses."""
    event = _filled(event)
    if not event or not text:
        return ""
    for row in segment_source_clauses(text):
        clause = row.get("text") or ""
        match = re.search(re.escape(event), clause, re.I)
        if not match:
            continue
        rest = clause[match.end():].strip(" .,;:")
        if not rest:
            return ""
        return _copy_present(text, rest)
    return ""


def _include_target(text: str, action: str, target: str) -> str:
    action = _filled(action)
    target = _filled(target)
    if not action:
        return action
    if target and target.casefold() in action.casefold():
        return action
    if not target:
        return action
    for glue in (f"{action} to {target}", f"{action} {target}"):
        copied = _copy_present(text, glue)
        if copied:
            return copied
    return action


def _target_from_action(text: str, action: str) -> str:
    match = re.search(r"\bto\s+(.+)$", action or "", re.I)
    if not match:
        return ""
    candidate = match.group(1).strip(" .")
    return _copy_present(text, candidate)


def discourse_citeable_atoms(world: Mapping[str, Any]) -> list[str]:
    atoms = [f"proposition:{row['proposition_id']}"
             for row in world.get("propositions") or []]
    atoms.extend(f"report:{row['report_id']}" for row in world.get("reports") or [])
    atoms.extend(f"commitment:{row['commitment_id']}"
                 for row in world.get("commitments") or [])
    atoms.extend(f"modal:{row['modal_id']}"
                 for row in world.get("modal_operators") or [])
    atoms.extend(f"norm:{row['norm_id']}"
                 for row in world.get("normative_propositions") or [])
    return atoms


def discourse_bind_rows(world: Mapping[str, Any]) -> list[tuple[str, str, list[str]]]:
    rows = []
    for proposition in world.get("propositions") or []:
        rows.append((
            f"proposition:{proposition['proposition_id']}",
            proposition.get("predication") or "",
            list(proposition.get("clause_ids") or []),
        ))
    for report in world.get("reports") or []:
        rows.append((
            f"report:{report['report_id']}",
            report.get("speech_act") or "",
            list(report.get("clause_ids") or []),
        ))
    for commitment in world.get("commitments") or []:
        rows.append((
            f"commitment:{commitment['commitment_id']}",
            commitment.get("commitment_event") or "",
            list(commitment.get("clause_ids") or []),
        ))
    for modal in world.get("modal_operators") or []:
        rows.append((
            f"modal:{modal['modal_id']}",
            _governed_span(world, modal.get("governed_proposition_id"))
            or modal.get("force") or "",
            list(modal.get("clause_ids") or []),
        ))
    for norm in world.get("normative_propositions") or []:
        rows.append((
            f"norm:{norm['norm_id']}",
            _governed_span(world, norm.get("governed_proposition_id"))
            or norm.get("force") or "",
            list(norm.get("clause_ids") or []),
        ))
    return rows


def content_as_effect_errors(world: Mapping[str, Any]) -> list[str]:
    forbidden: list[str] = []
    for proposition in world.get("propositions") or []:
        if proposition.get("status") not in NON_OCCURRENCE_STATUSES:
            continue
        forbidden.append((proposition.get("predication") or "").casefold())
    errors = []
    for effect in world.get("effects") or []:
        outcome = (effect.get("outcome") or "").casefold()
        predicate = (effect.get("predicate") or "").casefold()
        source = (effect.get("source_proposition") or "").casefold()
        for content in forbidden:
            if content and (content == outcome or content == predicate
                            or content == source or content in outcome):
                errors.append(
                    f"{effect.get('effect_id', 'effect')} treats discourse "
                    "content as an occurrence effect")
                break
    return errors


def relent_workspace_split(world: Mapping[str, Any]) -> dict[str, Any]:
    """Split occurrence facts from discourse objects RelEnt may cite."""
    established = []
    for effect in world.get("effects") or []:
        established.append({
            "kind": "occurrence_effect",
            "effect_id": effect["effect_id"],
            "outcome": effect.get("outcome"),
            "predicate": effect.get("predicate"),
            "status": "established",
        })
    not_established = []
    ev_forbidden = []
    for proposition in world.get("propositions") or []:
        if proposition.get("status") in NON_OCCURRENCE_STATUSES:
            not_established.append({
                "kind": "proposition",
                "proposition_id": proposition["proposition_id"],
                "predication": proposition.get("predication"),
                "status": proposition.get("status"),
            })
            ev_forbidden.append(proposition["proposition_id"])
    for report in world.get("reports") or []:
        not_established.append({
            "kind": "report",
            "report_id": report["report_id"],
            "speech_act": report.get("speech_act"),
            "content_proposition_id": report.get("content_proposition_id"),
            "status": "attributed",
        })
    for commitment in world.get("commitments") or []:
        not_established.append({
            "kind": "commitment",
            "commitment_id": commitment["commitment_id"],
            "content_proposition_id": commitment.get("content_proposition_id"),
            "status": "committed",
        })
    for modal in world.get("modal_operators") or []:
        not_established.append({
            "kind": "modal_operator",
            "modal_id": modal["modal_id"],
            "force": modal.get("force"),
            "governed_proposition_id": modal.get("governed_proposition_id"),
            "status": "modal",
        })
    for norm in world.get("normative_propositions") or []:
        not_established.append({
            "kind": "normative_proposition",
            "norm_id": norm["norm_id"],
            "force": norm.get("force"),
            "governed_proposition_id": norm.get("governed_proposition_id"),
            "status": "normative",
        })
    return {
        "established": established,
        "not_established": not_established,
        "duty_ledger_citations": [
            {"kind": "norm", "norm_id": row["norm_id"]}
            for row in world.get("normative_propositions") or []
        ] + [
            {"kind": "commitment", "commitment_id": row["commitment_id"]}
            for row in world.get("commitments") or []
        ],
        "ev_forbidden_proposition_ids": ev_forbidden,
        "factual_status_sources": ["occurrence_effects"],
    }


def discourse_provenance_rows(world: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for collection, id_key, kind in (
        ("propositions", "proposition_id", "proposition"),
        ("reports", "report_id", "report"),
        ("commitments", "commitment_id", "commitment"),
        ("modal_operators", "modal_id", "modal"),
        ("normative_propositions", "norm_id", "norm"),
    ):
        for atom in world.get(collection) or []:
            rows.append({
                "atom_id": f"{kind}:{atom[id_key]}",
                "atom_kind": kind,
                "origin": "SOURCE_ASSERTED",
                "clause_ids": list(atom.get("clause_ids") or []),
                "evidence": [
                    atom.get("predication") or atom.get("speech_act")
                    or atom.get("commitment_event") or atom.get("force") or ""
                ],
                "explanation": "Copied discourse object; content is not occurrence.",
            })
    return rows


def discourse_graph_rows(world: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for proposition in world.get("propositions") or []:
        nodes.append({
            "id": proposition["proposition_id"], "kind": "proposition",
            "label": proposition.get("predication") or "",
            "status": proposition.get("status"),
        })
    for report in world.get("reports") or []:
        nodes.append({
            "id": report["report_id"], "kind": "report",
            "label": report.get("speech_act") or "CLAIMS",
        })
        if report.get("source_party_id"):
            edges.append({
                "source": report["source_party_id"], "relation": "CLAIMS",
                "target": report["report_id"],
            })
        if report.get("content_proposition_id"):
            edges.append({
                "source": report["report_id"], "relation": "CONTENT",
                "target": report["content_proposition_id"],
            })
        if report.get("action_id"):
            edges.append({
                "source": report["action_id"], "relation": "SPEECH_ACT",
                "target": report["report_id"],
            })
    for commitment in world.get("commitments") or []:
        nodes.append({
            "id": commitment["commitment_id"], "kind": "commitment",
            "label": commitment.get("commitment_event") or "PROMISE",
        })
        if commitment.get("promisor_party_id"):
            edges.append({
                "source": commitment["promisor_party_id"], "relation": "PROMISOR",
                "target": commitment["commitment_id"],
            })
        if commitment.get("promisee_party_id"):
            edges.append({
                "source": commitment["commitment_id"], "relation": "PROMISEE",
                "target": commitment["promisee_party_id"],
            })
        if commitment.get("content_proposition_id"):
            edges.append({
                "source": commitment["commitment_id"], "relation": "CONTENT",
                "target": commitment["content_proposition_id"],
            })
    for modal in world.get("modal_operators") or []:
        nodes.append({
            "id": modal["modal_id"], "kind": "modal",
            "label": modal.get("force") or "modal",
        })
        if modal.get("bearer_party_id"):
            edges.append({
                "source": modal["bearer_party_id"], "relation": "BEARER",
                "target": modal["modal_id"],
            })
        if modal.get("governed_proposition_id"):
            edges.append({
                "source": modal["modal_id"], "relation": "GOVERNS",
                "target": modal["governed_proposition_id"],
            })
    for norm in world.get("normative_propositions") or []:
        nodes.append({
            "id": norm["norm_id"], "kind": "norm",
            "label": norm.get("force") or "norm",
        })
        if norm.get("bearer_party_id"):
            edges.append({
                "source": norm["bearer_party_id"], "relation": "BEARER",
                "target": norm["norm_id"],
            })
        if norm.get("authority_party_id"):
            edges.append({
                "source": norm["authority_party_id"], "relation": "AUTHORITY",
                "target": norm["norm_id"],
            })
        if norm.get("governed_proposition_id"):
            edges.append({
                "source": norm["norm_id"], "relation": "GOVERNS",
                "target": norm["governed_proposition_id"],
            })
    return nodes, edges


def build_disputed_report(text: str, slots: Mapping[str, Any],
                          package: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], list[str]]:
    source = _filled(slots.get("source"))
    speech = _filled(slots.get("report_words"))
    content = _filled(slots.get("reported_content"))
    parties: list[dict[str, Any]] = []
    index = {"n": 0}
    source_id = _party(text, parties, index, source, role="source",
                       construction="disputed_report")
    clauses = _clauses(text, speech, content, source)
    actions = []
    if speech:
        actions.append({
            "action_id": "A0",
            "intervention": speech,
            "actor_party_id": source_id,
            "recipient_party_ids": [],
            "effect_ids": [],
            "clause_ids": [row["clause_id"] for row in clauses],
        })
    proposition = {
        "proposition_id": "PR1",
        "predication": content,
        "polarity": _polarity(content),
        "status": "ATTRIBUTED",
        "clause_ids": [row["clause_id"] for row in clauses],
    }
    report = {
        "report_id": "R0",
        "source_party_id": source_id,
        "speech_act": speech,
        "content_proposition_id": "PR1",
        "action_id": "A0" if actions else "",
        "clause_ids": [row["clause_id"] for row in clauses],
    }
    reports = [report]
    competing = _filled(slots.get("competing_report"))
    if competing:
        reports.append({
            "report_id": "R1",
            "source_party_id": source_id,
            "speech_act": competing,
            "content_proposition_id": "PR1",
            "action_id": "",
            "clause_ids": [row["clause_id"] for row in _clauses(text, competing)],
        })
    notes = []
    if _filled(slots.get("reliability")):
        notes.append("Source reliability is copied, not a world fact.")
    world = ensure_schema({
        "parties": parties, "actions": actions, "effects": [],
        "conditions": [], "temporal_relations": [], "causal_links": [],
        "counterfactual_links": [],
        "propositions": [proposition], "reports": reports,
    })
    return world, notes


def build_promise_reliance(text: str, slots: Mapping[str, Any],
                           package: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], list[str]]:
    promisor = _filled(slots.get("promisor"))
    promisee = _filled(slots.get("promisee"))
    event = _filled(slots.get("commitment_event"))
    content = _filled(slots.get("commitment_content"))
    parties: list[dict[str, Any]] = []
    index = {"n": 0}
    promisor_id = _party(text, parties, index, promisor, role="promisor",
                         construction="promise_reliance")
    promisee_id = _party(text, parties, index, promisee, role="promisee",
                         construction="promise_reliance")
    clauses = _clauses(text, event, content, promisor, promisee)
    clause_ids = [row["clause_id"] for row in clauses]
    actions = []
    if event:
        actions.append({
            "action_id": "A0",
            "intervention": event,
            "actor_party_id": promisor_id,
            "recipient_party_ids": [promisee_id] if promisee_id else [],
            "effect_ids": [],
            "clause_ids": clause_ids,
        })
    proposition = {
        "proposition_id": "PR1",
        "predication": content,
        "polarity": _polarity(content),
        "status": "COMMITTED_CONTENT",
        "clause_ids": clause_ids,
    }
    commitment = {
        "commitment_id": "K0",
        "promisor_party_id": promisor_id,
        "promisee_party_id": promisee_id,
        "commitment_event": event,
        "content_proposition_id": "PR1",
        "reliance": _filled(slots.get("reliance")),
        "breach": _filled(slots.get("breach")),
        "clause_ids": clause_ids,
    }
    notes = ["Promise content is not an occurrence of delivery or performance."]
    world = ensure_schema({
        "parties": parties, "actions": actions, "effects": [],
        "conditions": [], "temporal_relations": [], "causal_links": [],
        "counterfactual_links": [],
        "propositions": [proposition], "commitments": [commitment],
    })
    return world, notes


def build_ability_permission(text: str, slots: Mapping[str, Any],
                             package: Mapping[str, Any] | None = None
                             ) -> tuple[dict[str, Any], list[str]]:
    actor = _filled(slots.get("actor"))
    action = _filled(slots.get("modal_action"))
    target = _filled(slots.get("target"))
    words = _filled(slots.get("modal_words")) or _first_modal_span(slots)
    parties: list[dict[str, Any]] = []
    index = {"n": 0}
    actor_id = _party(text, parties, index, actor, role="actor",
                      construction="ability_permission")
    if not target:
        target = destination_from_package(package)
    if not target:
        target = _target_from_action(text, action)
    if target:
        _party(text, parties, index, target, role="target",
               construction="ability_permission")
    action = _include_target(text, action, target)
    clauses = _clauses(text, actor, action, words, target)
    clause_ids = [row["clause_id"] for row in clauses]
    force = _z10_modal_force(package, {"ability", "permission", "possibility"})
    proposition = {
        "proposition_id": "PR1",
        "predication": action,
        "polarity": _polarity(action),
        "status": "GOVERNED",
        "clause_ids": clause_ids,
    }
    modal = {
        "modal_id": "M0",
        "force": force,
        "bearer_party_id": actor_id,
        "governed_proposition_id": "PR1",
        "clause_ids": clause_ids,
    }
    notes = ["A modal does not establish that the governed action occurred."]
    world = ensure_schema({
        "parties": parties, "actions": [], "effects": [],
        "conditions": [], "temporal_relations": [], "causal_links": [],
        "counterfactual_links": [],
        "propositions": [proposition], "modal_operators": [modal],
    })
    return world, notes


def build_deontic_rule(text: str, slots: Mapping[str, Any],
                       package: Mapping[str, Any] | None = None
                       ) -> tuple[dict[str, Any], list[str]]:
    words = _filled(slots.get("deontic_words"))
    governed = _filled(slots.get("governed_action"))
    bearer = _filled(slots.get("bearer"))
    authority = _filled(slots.get("authority"))
    target = _filled(slots.get("target"))
    parties: list[dict[str, Any]] = []
    index = {"n": 0}
    bearer_id = _party(text, parties, index, bearer, role="bearer",
                       construction="deontic_rule")
    authority_id = ""
    if authority:
        authority_id = _party(text, parties, index, authority, role="authority",
                              construction="deontic_rule")
    if not target:
        target = destination_from_package(package)
    if not target:
        target = _target_from_action(text, governed)
    if target:
        _party(text, parties, index, target, role="target",
               construction="deontic_rule")
    governed = _include_target(text, governed, target)
    clauses = _clauses(text, words, governed, bearer, target)
    clause_ids = [row["clause_id"] for row in clauses]
    force = _z10_modal_force(package, {"obligation", "permission"})
    if force == "unresolved":
        force = _deontic_force_from_span(words)
    proposition = {
        "proposition_id": "PR1",
        "predication": governed,
        "polarity": _polarity(governed),
        "status": "GOVERNED",
        "clause_ids": clause_ids,
    }
    norm = {
        "norm_id": "N0",
        "force": force,
        "bearer_party_id": bearer_id,
        "authority_party_id": authority_id,
        "governed_proposition_id": "PR1",
        "clause_ids": clause_ids,
    }
    notes = ["A deontic rule does not establish that the governed action occurred."]
    world = ensure_schema({
        "parties": parties, "actions": [], "effects": [],
        "conditions": [], "temporal_relations": [], "causal_links": [],
        "counterfactual_links": [],
        "propositions": [proposition], "normative_propositions": [norm],
    })
    return world, notes


DISCOURSE_BUILDERS = {
    "disputed_report": build_disputed_report,
    "promise_reliance": build_promise_reliance,
    "ability_permission": build_ability_permission,
    "deontic_rule": build_deontic_rule,
}


def assignment_for(world: Mapping[str, Any]) -> list[str]:
    return [row["intervention"] for row in world.get("actions") or []]


def _governed_span(world: Mapping[str, Any], ident: str | None) -> str:
    if not ident:
        return ""
    for proposition in world.get("propositions") or []:
        if proposition.get("proposition_id") == ident:
            return proposition.get("predication") or ""
    return ""


def _filled(value: Any) -> str:
    text = str(value or "").strip()
    if text.casefold() in _NONE:
        return ""
    return text


def _polarity(span: str) -> str:
    if re.search(r"\b(?:not|n't|never|no)\b", span or "", re.I):
        return "negative"
    return "positive"


def _clauses(text: str, *snippets: str) -> list[dict[str, Any]]:
    clauses = segment_source_clauses(text)
    needed = [snippet for snippet in snippets if snippet]
    if not needed:
        return clauses[:1] if clauses else []
    matched = []
    for row in clauses:
        folded = row["text"].casefold()
        if any(snippet.casefold() in folded for snippet in needed):
            matched.append({"clause_id": row["clause_id"], "text": row["text"]})
    return matched or ([{"clause_id": clauses[0]["clause_id"], "text": clauses[0]["text"]}]
                       if clauses else [])


def _party(text: str, parties: list[dict[str, Any]], index: dict[str, int],
           label: str, *, role: str, construction: str) -> str:
    if not label:
        return ""
    licensed = license_kind(label, role=role, text=text, construction=construction)
    folded = label.casefold()
    for row in parties:
        if row["label"].casefold() == folded:
            return row["party_id"]
    index["n"] += 1
    ident = f"P{index['n']}"
    clauses = segment_source_clauses(text)
    parties.append({
        "party_id": ident, "label": label, "kind": licensed["kind"],
        "kind_origin": licensed["origin"], "quantities": [],
        "clause_ids": [row["clause_id"] for row in clauses
                       if label.casefold() in row["text"].casefold()],
    })
    return ident


def _z10_modal_force(package: Mapping[str, Any] | None,
                     allowed: set[str]) -> str:
    if not package:
        return "unresolved"
    values = [
        str(row.get("value") or "").casefold()
        for row in package.get("candidates") or []
        if row.get("type") == "MODALITY"
    ]
    for value in values:
        if value in allowed:
            return value
    return "unresolved"


def _deontic_force_from_span(words: str) -> str:
    # Witness only. Regex may not invent a 1.3 kind; it may label a copied
    # deontic span when Z10 left the force unresolved.
    folded = (words or "").casefold()
    if re.search(r"\b(?:must not|may not|shall not|forbidden|prohibited)\b", folded):
        return "prohibition"
    if re.search(r"\b(?:must|shall|should|required|obligated)\b", folded):
        return "obligation"
    if re.search(r"\bmay\b", folded):
        return "permission"
    return "unresolved"


def _first_modal_span(slots: Mapping[str, Any]) -> str:
    readings = slots.get("modal_reading")
    if isinstance(readings, list) and readings:
        return str(readings[0])
    return ""
