"""Compile Z10 interpretation candidates into provisional Parliament worlds.

This is the missing boundary between the frozen parser package and Parliament's
existing ``ScenarioWorldModel`` input shape.  The adapter enumerates supported
Z10 readings, preserves their selection witnesses, and emits explicit problems
for semantics that the package does not encode.  It never admits a world and it
never turns a conditional, quantity, or negative remnant into an unsupported
causal or welfare claim.
"""
from __future__ import annotations

from itertools import product
import argparse
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from candidate_validation import empty_selection, validate_candidate_selection


ADAPTER_VERSION = "z10-world-candidates/0.1"
WORLD_SCHEMA_VERSION = "1.3"
_WORDS = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_STOP = {"a", "an", "the", "to", "of", "if", "will", "can", "could", "would", "should"}
_ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc"}


def _tokens(text: str) -> set[str]:
    values = {word.casefold() for word in _WORDS.findall(text)} - _STOP
    expanded = set(values)
    for word in values:
        if len(word) > 4 and word.endswith("ies"):
            expanded.add(word[:-3] + "y")
        elif len(word) > 3 and word.endswith("s"):
            expanded.add(word[:-1])
    return expanded


def segment_source_clauses(text: str) -> list[dict[str, Any]]:
    """Return exact source spans while keeping titles such as ``Dr.`` intact."""
    rows: list[dict[str, Any]] = []
    start = 0
    for match in re.finditer(r"[.!?]|\n", text):
        end = match.end()
        if match.group() == ".":
            prefix = text[start:match.start()]
            last = (_WORDS.findall(prefix)[-1].casefold() if _WORDS.findall(prefix) else "")
            if last in _ABBREVIATIONS:
                continue
        raw = text[start:end]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if right > left:
            rows.append({"clause_id": f"C{len(rows)}", "text": raw[left:right],
                         "start": start + left, "end": start + right})
        start = end
    if start < len(text):
        raw = text[start:]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if right > left:
            rows.append({"clause_id": f"C{len(rows)}", "text": raw[left:right],
                         "start": start + left, "end": start + right})
    return rows


def _dependency_closure(candidate_ids: Iterable[str], candidates: Mapping[str, dict]) -> set[str]:
    selected = set(candidate_ids)
    pending = list(selected)
    while pending:
        ident = pending.pop()
        for required in candidates[ident]["requires"]:
            if required not in selected:
                selected.add(required)
                pending.append(required)
    return selected


def _selection(package: dict, candidate_ids: Iterable[str]) -> tuple[dict, dict]:
    candidates = {row["id"]: row for row in package["candidates"]}
    selected = _dependency_closure(candidate_ids, candidates)
    selection = empty_selection(package)
    selection["selected_candidate_ids"] = sorted(selected)
    nodes: set[str] = set()
    for ident in selected:
        row = candidates[ident]
        nodes.update(row["arguments"].values())
        for context in row["scope"]["contexts"]:
            for key in ("condition_proposition_id", "source_mention_id", "report_proposition_id"):
                if context.get(key):
                    nodes.add(context[key])
    selection["selected_node_ids"] = sorted(nodes)
    validation = validate_candidate_selection(package, selection)
    return selection, validation


def _participants_by_proposition(package: dict) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for row in package["candidates"]:
        if row["type"] == "PARTICIPANT":
            result.setdefault(row["arguments"]["proposition"], []).append(row)
    return result


def _predications(package: dict) -> dict[str, dict]:
    return {row["arguments"]["proposition"]: row for row in package["candidates"]
            if row["type"] == "PREDICATION"}


def _reconstruction_novel_labels(package: dict) -> dict[str, set[str]]:
    nodes = {row["id"]: row for row in package["nodes"]}
    candidates = {row["id"]: row for row in package["candidates"]}
    participants = _participants_by_proposition(package)
    result: dict[str, set[str]] = {}
    for reconstruction in package.get("reconstructions", []):
        antecedent = {
            nodes[row["arguments"]["mention"]]["label"].casefold()
            for row in participants.get(reconstruction["antecedent_proposition_id"], [])
        }
        reconstructed = {
            nodes[candidates[ident]["arguments"]["mention"]]["label"].casefold()
            for ident in reconstruction["participant_candidate_ids"]
        }
        result[reconstruction["proposition_id"]] = reconstructed - antecedent
    return result


def _action_options(package: dict, action: str) -> list[str]:
    nodes = {row["id"]: row for row in package["nodes"]}
    participants = _participants_by_proposition(package)
    predications = _predications(package)
    action_words = _tokens(action)
    option_props = {row["arguments"]["proposition"] for row in package["candidates"]
                    if row["type"] == "OPTION_OF"}
    novel = _reconstruction_novel_labels(package)
    reconstructed_matches = [prop for prop, labels in novel.items()
                             if any(_tokens(label) <= action_words for label in labels if _tokens(label))]
    if reconstructed_matches:
        pool = reconstructed_matches
    elif option_props:
        pool = list(option_props)
    else:
        # A reconstruction is licensed by its explicit remnant.  Do not let
        # its inherited participants outscore the spoken antecedent for an
        # action that does not name that remnant.
        pool = [proposition for proposition in predications if proposition not in novel]
    scored: list[tuple[int, str]] = []
    for proposition in pool:
        node = nodes[proposition]
        predicate_words = _tokens(node.get("predicate", node["label"]))
        if predicate_words and not predicate_words.intersection(action_words):
            continue
        score = 3 * len(predicate_words.intersection(action_words))
        for role in participants.get(proposition, []):
            label_words = _tokens(nodes[role["arguments"]["mention"]]["label"])
            overlap = len(label_words.intersection(action_words))
            score += overlap * {"destination": 5, "patient": 5, "object": 3,
                                "controller": 2, "agent": 2, "subject": 1}.get(role["value"], 1)
        if proposition in option_props:
            score += 2
        if proposition in novel:
            score += 1
        scored.append((score, proposition))
    if not scored:
        return []
    best = max(score for score, _ in scored)
    # Preserve all reconstruction role readings for the same explicit remnant.
    threshold = best - 6 if reconstructed_matches else best
    return sorted(prop for score, prop in scored if score >= threshold)


def _enumerate_assignments(package: dict, actions: Sequence[str], max_drafts: int) -> list[tuple[str, ...]]:
    options = [_action_options(package, action) for action in actions]
    if any(not rows for rows in options):
        return []
    assignments = []
    for values in product(*options):
        if len(set(values)) != len(values):
            continue
        assignments.append(values)
        if len(assignments) >= max_drafts:
            break
    return assignments


def _support_span(package: dict, candidate_rows: Sequence[dict]) -> tuple[str, list[str]]:
    evidence = {row["id"]: row for row in package["evidence"]}
    ids = list(dict.fromkeys(ident for row in candidate_rows for ident in row["evidence_ids"]))
    spans = [evidence[ident] for ident in ids]
    if not spans:
        return "", []
    start, end = min(row["start"] for row in spans), max(row["end"] for row in spans)
    clauses = segment_source_clauses(package["document"]["text"])
    clause_ids = [row["clause_id"] for row in clauses if row["start"] < end and start < row["end"]]
    return package["document"]["text"][start:end], clause_ids


def _modality(candidate_rows: Sequence[dict]) -> str:
    values = {row["value"] for row in candidate_rows if row["type"] == "MODALITY"}
    contexts = {ctx["kind"] for row in candidate_rows for ctx in row["scope"]["contexts"]}
    if "prediction" in values or "conditional" in contexts or "hypothetical" in contexts:
        return "STIPULATED_CONDITIONAL"
    if values.intersection({"ability", "permission", "possibility"}):
        return "POSSIBLE"
    if "unresolved" in values:
        return "UNKNOWN"
    return "CERTAIN"


def _problem(code: str, message: str, candidate_ids: Iterable[str] = ()) -> dict:
    return {"code": code, "message": message,
            "candidate_ids": sorted(set(candidate_ids))}


def _compile_world(package: dict, actions: Sequence[str], assignment: Sequence[str],
                   selection: dict, validation: dict) -> tuple[dict, list[dict], dict]:
    nodes = {row["id"]: row for row in package["nodes"]}
    candidates = {row["id"]: row for row in package["candidates"]}
    selected = [candidates[ident] for ident in selection["selected_candidate_ids"]]
    participants = _participants_by_proposition(package)
    clauses = segment_source_clauses(package["document"]["text"])

    mention_ids = {row["arguments"]["mention"] for row in selected if row["type"] == "PARTICIPANT"}
    mention_ids.update(row["arguments"]["mention"] for row in selected if row["type"] == "QUANTITY")
    by_label: dict[str, list[str]] = {}
    for ident in sorted(mention_ids):
        by_label.setdefault(nodes[ident]["label"].casefold(), []).append(ident)
    party_for_mention: dict[str, str] = {}
    parties = []
    problems: list[dict] = []
    quantity_rows = [row for row in selected if row["type"] == "QUANTITY"]
    for index, (_, mentions) in enumerate(sorted(by_label.items()), 1):
        label = nodes[mentions[0]]["label"]
        party_id = f"P{index}"
        for ident in mentions:
            party_for_mention[ident] = party_id
        related = [row for row in selected if row.get("arguments", {}).get("mention") in mentions]
        _, clause_ids = _support_span(package, related)
        quantities = []
        for row in quantity_rows:
            if row["arguments"]["mention"] in mentions:
                value = row["value"]
                quantities.append(f"{value['amount']} {value['unit']}".strip())
        parties.append({"party_id": party_id, "label": label, "kind": "OTHER",
                        "quantities": quantities, "clause_ids": clause_ids})
    if parties:
        problems.append(_problem("party_kinds_unresolved",
                                 "Z10 mentions do not classify Parliament party kinds."))

    world_actions, effects, conditions = [], [], []
    action_for_prop = {prop: f"A{index}" for index, prop in enumerate(assignment)}
    action_sources: dict[str, dict] = {}
    for index, (action_text, proposition) in enumerate(zip(actions, assignment)):
        action_id = f"A{index}"
        roles = [row for row in participants.get(proposition, [])
                 if row["id"] in selection["selected_candidate_ids"]]
        predication = _predications(package)[proposition]
        rows = [predication, *roles]
        source, clause_ids = _support_span(package, rows)
        actor = next((row for row in roles if row["value"] in {"agent", "controller", "subject"}), None)
        recipients = [row for row in roles if row["value"] in {"destination", "patient"}]
        if not recipients:
            recipients = [row for row in roles if row["value"] == "object"]
        actor_party = party_for_mention.get(actor["arguments"]["mention"], "") if actor else ""
        recipient_ids = list(dict.fromkeys(
            party_for_mention.get(row["arguments"]["mention"], "") for row in recipients
            if party_for_mention.get(row["arguments"]["mention"])))
        novel_labels = _reconstruction_novel_labels(package).get(proposition, set())
        for role in roles:
            label = nodes[role["arguments"]["mention"]]["label"]
            if label.casefold() not in novel_labels:
                continue
            expects_destination = bool(re.search(
                r"\bto\s+(?:the\s+)?" + re.escape(label.casefold()) + r"\b",
                action_text.casefold(), flags=re.IGNORECASE))
            if expects_destination and role["value"] not in {"destination", "patient"}:
                problems.append(_problem("action_role_alignment_unresolved",
                    f"The action text treats {label!r} as a destination, but this Z10 reading assigns role {role['value']!r}.",
                    [role["id"]]))
        effect_id = f"E{len(effects) + 1}"
        target_party = recipient_ids[0] if recipient_ids else actor_party
        polarity = predication["scope"]["polarity"]
        if polarity != "positive":
            problems.append(_problem("operator_scope_unresolved",
                "Negative or unresolved proposition scope cannot be converted into a positive occurrence.",
                [predication["id"]]))
        effects.append({
            "effect_id": effect_id, "action_id": action_id, "party_id": target_party,
            "outcome": source or nodes[proposition]["label"],
            "predicate": nodes[proposition]["predicate"].upper(),
            # Parliament treats the intervention row itself as neutral; any
            # benefit or harm belongs on a separately supported outcome row.
            "polarity": "NEUTRAL", "directness": "DIRECT",
            # An action row denotes one candidate intervention.  Its source
            # scope remains in the selection witness and scope_qualifiers;
            # STIPULATED_CONDITIONAL without a concrete condition would be an
            # invalid Parliament effect.
            "modality": "CERTAIN", "effect_kind": "INTERVENTION",
            "condition_ids": [], "quantities": [], "likelihood_qualifiers": [],
            "overall_likelihood_qualifiers": [], "scope_qualifiers": [polarity],
            "temporal_qualifiers": [], "condition_join": "AND",
            "source_proposition": source, "source_effect_ids": [],
            "derivation_operation": "DIRECT_COPY", "derivation_explanation": "",
            "derivation_assumptions": [], "outcome_type_transformation": "PRESERVED",
            "clause_ids": clause_ids,
        })
        world_actions.append({"action_id": action_id, "intervention": action_text,
                              "actor_party_id": actor_party,
                              "recipient_party_ids": recipient_ids,
                              "effect_ids": [effect_id], "clause_ids": clause_ids})
        action_sources[action_id] = {"clause_ids": clause_ids,
                                     "reason": "Compiled from selected Z10 proposition " + proposition}

    conditional_rows = [row for row in selected if row["type"] == "CONDITIONAL_ON"
                        and row["arguments"]["condition"] in action_for_prop]
    for link in conditional_rows:
        condition_prop = link["arguments"]["condition"]
        consequence_prop = link["arguments"]["consequence"]
        action_id = action_for_prop[condition_prop]
        consequence_roles = [row for row in participants.get(consequence_prop, [])
                             if row["id"] in selection["selected_candidate_ids"]]
        predication = _predications(package)[consequence_prop]
        rows = [link, predication, *consequence_roles]
        source, clause_ids = _support_span(package, rows)
        condition_id = f"K{len(conditions) + 1}"
        conditions.append({"condition_id": condition_id,
                           "description": "Selected action instantiates " + nodes[condition_prop]["label"],
                           "value_status": "UNKNOWN", "decision_relevance": "MATERIAL",
                           # Parliament reserves event_effect_id for a
                           # probabilistic event whose hedge the condition
                           # records.  A chosen intervention is not that kind
                           # of event, so retain the textual gate without
                           # manufacturing a probabilistic parent.
                           "event_effect_id": "",
                           "polarity": "POSITIVE", "operator": "IF", "clause_ids": clause_ids})
        bearer = next((row for row in consequence_roles
                       if row["value"] in {"patient", "destination", "subject", "object"}), None)
        party_id = party_for_mention.get(bearer["arguments"]["mention"], "") if bearer else ""
        effect_id = f"E{len(effects) + 1}"
        modal_rows = [row for row in selected if row["type"] == "MODALITY"
                      and row["arguments"]["proposition"] == consequence_prop]
        effects.append({
            "effect_id": effect_id, "action_id": action_id, "party_id": party_id,
            "outcome": source or nodes[consequence_prop]["label"],
            "predicate": nodes[consequence_prop]["predicate"].upper(),
            "polarity": "UNRESOLVED", "directness": "DOWNSTREAM",
            "modality": _modality([predication, *modal_rows]), "effect_kind": "OTHER",
            "condition_ids": [condition_id], "quantities": [], "likelihood_qualifiers": [],
            "overall_likelihood_qualifiers": [], "scope_qualifiers": ["conditional"],
            "temporal_qualifiers": [], "condition_join": "AND",
            "source_proposition": source, "source_effect_ids": [],
            "derivation_operation": "DIRECT_COPY", "derivation_explanation": "",
            "derivation_assumptions": [], "outcome_type_transformation": "PRESERVED",
            "clause_ids": clause_ids,
        })
        next(row for row in world_actions if row["action_id"] == action_id)["effect_ids"].append(effect_id)
        problems.append(_problem("conditional_causation_unresolved",
            "CONDITIONAL_ON supports a gate but does not by itself license a CAUSES edge.", [link["id"]]))
        problems.append(_problem("outcome_semantics_unresolved",
            "Z10 does not classify this consequence as a benefit, harm, health outcome, or other welfare effect.",
            [predication["id"]]))

    if any(row["kind"] == "scenario_option" for row in package["choice_sets"]):
        problems.append(_problem("choice_exclusivity_unresolved",
            "Scenario options are represented, but Z10 does not license mutual exclusivity unless the choice set says at_most_one."))
    quantity_labels = {nodes[row["arguments"]["mention"]]["label"].casefold() for row in quantity_rows}
    object_labels = {nodes[row["arguments"]["mention"]]["label"].casefold()
                     for row in selected if row["type"] == "PARTICIPANT" and row["value"] == "object"}
    if quantity_labels and quantity_labels.isdisjoint(object_labels):
        problems.append(_problem("resource_quantity_identity_unresolved",
            "The quantified mention is not yet identified with the transferred object; quantity was not propagated."))
    if validation["unresolved_question_ids"]:
        problems.append(_problem("open_z10_questions",
            "Selected candidates remain subject to unresolved Z10 questions.",
            validation["provisional_candidate_ids"]))

    # Stable deduplication keeps diagnostics compact without hiding witnesses.
    deduped = []
    seen = set()
    for row in problems:
        key = (row["code"], tuple(row["candidate_ids"]))
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    world = {"schema_version": WORLD_SCHEMA_VERSION, "parties": parties,
             "actions": world_actions, "effects": effects, "conditions": conditions,
             "temporal_relations": [], "causal_links": [], "counterfactual_links": []}
    evidence = {"clauses": [{k: row[k] for k in ("clause_id", "text")} for row in clauses],
                "actions": action_sources}
    return world, deduped, evidence


def enumerate_candidate_world_models(package: dict, actions: Sequence[str], *,
                                     max_drafts: int = 16) -> dict:
    """Return concrete, non-admitted Parliament candidates and their witnesses."""
    if not actions or any(not isinstance(action, str) or not action.strip() for action in actions):
        raise ValueError("actions must be a nonempty sequence of nonempty strings")
    baseline = validate_candidate_selection(package, empty_selection(package))
    if not baseline["contract_valid"]:
        raise ValueError("Invalid Z10 package: " + repr(baseline["errors"]))
    assignments = _enumerate_assignments(package, actions, max_drafts)
    drafts, rejected = [], []
    predications = _predications(package)
    participants = _participants_by_proposition(package)
    candidates = {row["id"]: row for row in package["candidates"]}
    for assignment in assignments:
        seed: set[str] = set()
        for proposition in assignment:
            seed.add(predications[proposition]["id"])
            seed.update(row["id"] for row in participants.get(proposition, []))
        for row in package["candidates"]:
            if row["type"] == "CONDITIONAL_ON" and row["arguments"]["condition"] in assignment:
                seed.add(row["id"])
                consequence = row["arguments"]["consequence"]
                seed.update(role["id"] for role in participants.get(consequence, []))
                seed.update(modal["id"] for modal in package["candidates"]
                            if modal["type"] == "MODALITY"
                            and modal["arguments"]["proposition"] == consequence
                            and modal["assessment"]["status"] != "unresolved")
            if row["type"] == "OPTION_OF" and row["arguments"]["proposition"] in assignment:
                seed.add(row["id"])
            if row["type"] == "QUANTITY":
                seed.add(row["id"])
        selection, validation = _selection(package, seed)
        if not validation["contract_valid"]:
            rejected.append({"proposition_ids": list(assignment), "errors": validation["errors"]})
            continue
        world, problems, evidence = _compile_world(
            package, actions, assignment, selection, validation)
        drafts.append({
            "draft_id": f"draft_{len(drafts)}", "status": "PROVISIONAL",
            "proposition_ids": list(assignment), "selection": selection,
            "selection_validation": validation, "evidence_binding": evidence,
            "world_model": world, "construction_problems": problems,
            "admission_authorized": False,
        })
    missing = []
    if not assignments:
        missing.append(_problem("action_alignment_failed",
            "No Z10 proposition assignment could be aligned to every requested action."))
    return {"adapter_version": ADAPTER_VERSION,
            "source_package_id": package["package_id"],
            "world_state_schema_version": WORLD_SCHEMA_VERSION,
            "actions": list(actions), "drafts": drafts,
            "enumeration_rejections": rejected, "construction_problems": missing}


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--package", type=Path, required=True,
                     help="Z10 package JSON or advisory packet JSON")
    cli.add_argument("--action", action="append", required=True,
                     help="One Parliament action; repeat for each alternative")
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--max-drafts", type=int, default=16)
    args = cli.parse_args()
    payload = json.loads(args.package.read_text(encoding="utf-8"))
    package = payload.get("package", payload)
    result = enumerate_candidate_world_models(package, args.action,
                                              max_drafts=args.max_drafts)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
