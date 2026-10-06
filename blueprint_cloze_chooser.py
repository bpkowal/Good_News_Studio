"""Choose a blueprint by completing its sentences from the scenario.

Each template is a list of sentences with a blank at the end. A model finishes
the three templates it thinks the text can complete. A completion counts only
when it is a contiguous copy of the scenario and answers that sentence. The
template with the most accepted blanks is passed forward. Empty blanks stay
empty. This module does not admit a world and does not pick a moral act.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Sequence
from urllib import request as urlrequest
from urllib.error import HTTPError

from blueprint_allocation_invariants import (
    complement_clause_ids,
    group_quantity,
    group_span,
)
from blueprint_derivation_license import apply_derivation_license, origin_for_effect
from blueprint_discourse import (
    DISCOURSE_BUILDERS,
    assignment_for,
    attach_source_discourse,
    discourse_provenance_rows,
    ensure_schema,
)
from blueprint_evidence_graph import assemble_evidence_graph, source_copy_validation
from blueprint_kind_license import (
    apply_kind_license,
    kind_problems,
    license_effect_kind,
    license_kind,
    licensed_source_predicate,
)
from blueprint_proposal_contract import (
    candidate as proposal_candidate,
    coverage_metrics,
    proposal as normalized_proposal,
    validate_proposal,
    withheld_proposal,
)
from candidate_graph_blueprints import _clause_holding, match_exclusive_allocation
from z10_world_model_adapter import segment_source_clauses


CLOZE_VERSION = "blueprint-cloze-chooser/0.9"
DEFAULT_OPENAI_ENV = Path(
    "/Users/benjaminkowal/Documents/Python/Python Coding/RAGAIMODEL/.env"
)
_STOP = {
    "a", "an", "the", "to", "of", "or", "and", "who", "that", "in", "on", "for",
}
_ABSTAIN = {"none", "n/a", "null", "unknown", "no", "not stated"}
_EXCLUSIVITY = re.compile(r"\b(?:but\s+)?not\s+both\b|\beither\b[\s\S]{0,80}\bor\b", re.I)
_EXPLICIT_EXCLUSIVITY = re.compile(
    r"\b(?:but\s+)?not\s+both\b|"
    r"\b(?:cannot|can\s+not|can't)\b[^.!?]{0,80}\bboth\b",
    re.I,
)
_INDIVISIBLE_ONE = re.compile(
    r"\b(?:one|1|single)\s+(?:[A-Za-z-]+\s+){0,3}"
    r"(?:dose|vial|unit|organ|seat|ticket|bed|ventilator)\b",
    re.I,
)
_NUMBER = re.compile(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", re.I)
_CHANCE = re.compile(r"\d+(?:\.\d+)?%\s*chance", re.I)
_NONRECEIPT = re.compile(
    r"\b(?:does not|doesn't|do not|don't|not receive|not get|does not get)\b", re.I)
# Same verbs the world-model check accepts on a RESOURCE_TRANSFER outcome.
_TRANSFER_EVENT = re.compile(
    r"\b(?:give|gives|gave|given|giving|receive|receives|received|receiving|"
    r"allocate|allocates|allocated|allocating|assign|assigns|assigned|assigning|"
    r"send|sends|sent|sending|deliver|delivers|delivered|delivering|"
    r"transfer|transfers|transferred|transferring|administer|administers|"
    r"administered|administering|provide|provides|provided|providing|"
    r"supply|supplies|supplied|supplying)\b",
    re.I,
)
_HEDGE = re.compile(r"\b(?:if|unless|whether)\b", re.I)
_MODAL_WORDS = re.compile(r"\b(?:can|could|may|might|able to|permitted to|allowed to)\b", re.I)
_DEONTIC_WORDS = re.compile(
    r"\b(?:must|shall|should|required to|obligated to|permitted to|allowed to|"
    r"forbidden to|prohibited from|may not|must not)\b", re.I)
_OUTCOME_WORDS = re.compile(
    r"\b(?:live|lives|survive|survives|recover|recovers|die|dies|kill|kills|"
    r"harm|harms|lose|loses|gain|gains|sustain|sustains|drown|drowns|"
    r"benefit|benefits|suffer|suffers|stop|stops)\b", re.I)
_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_RELATIVE = re.compile(r"\s+(?:that|who|which)\b", re.I)
_MATRIX_VERB = re.compile(r"\s+(?:must|will|can)\b", re.I)
_NEXT_ALTERNATIVE = re.compile(
    r"\s+(?:or|and)\s+(?=(?:a|an|the|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b)",
    re.I,
)
_PARTY_IDS = {
    "decider", "first_recipient", "second_recipient", "rescuer",
    "first_saved", "second_saved",
    "actor", "bearer", "second_bearer", "promisor", "promisee", "source", "target",
    "affected_party", "second_affected_party", "authority",
}

Complete = Callable[[Sequence[dict[str, str]]], str]

# Sentences the model finishes. `core` blanks are the ones that make a graph.
# `check` rejects a copied span that answers a different question.
TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "blueprint_id": "exclusive_allocation",
        "summary": "Someone assigns one resource to one of two parties.",
        "graph_builder": "implemented",
        "items": (
            {"id": "decider", "core": True, "check": "copy",
             "sentence": "The person who must decide who gets the resource is"},
            {"id": "resource", "core": True, "check": "copy",
             "sentence": "The resource that can be given is"},
            {"id": "assignment", "core": True, "check": "assignment",
             "sentence": "The act that assigns that resource, in the text's own words, is"},
            {"id": "first_transfer", "core": False, "check": "transfer_event",
             "sentence": "The giving, receipt, allocation, delivery, or administration of the resource to the first party is"},
            {"id": "second_transfer", "core": False, "check": "transfer_event",
             "sentence": "The giving, receipt, allocation, delivery, or administration of the resource to the second party is"},
            {"id": "quantity", "core": False, "check": "resource_quantity",
             "sentence": "The exact amount of that resource, stated in the text, is"},
            {"id": "first_recipient", "core": True, "check": "party",
             "sentence": "The first party mentioned in the text who could receive the resource or benefit is"},
            {"id": "second_recipient", "core": True, "check": "party",
             "sentence": "The second party mentioned in the text who could receive the resource or benefit is"},
            {"id": "exclusivity", "core": False, "check": "exclusivity",
             "sentence": "The words that say both parties cannot receive it are"},
            {"id": "first_outcome", "core": False, "check": "copy",
             "sentence": "The words that license what happens for the first party are"},
            {"id": "first_hedge", "core": False, "check": "hedge",
             "sentence": "The if, unless, whether, or percent-chance words inside that first outcome are"},
            {"id": "second_outcome", "core": False, "check": "copy",
             "sentence": "The words that license what happens for the second party are"},
            {"id": "second_hedge", "core": False, "check": "hedge",
             "sentence": "The if, unless, whether, or percent-chance words inside that second outcome are"},
            {"id": "first_implied_process", "core": False, "check": "implied",
             "sentence": 'The process most aligned with the text might explain how "{resource}" leads to "{outcome}" is'},
            {"id": "second_implied_process", "core": False, "check": "implied",
             "sentence": 'The process most aligned with the text might explain how "{resource}" leads to "{outcome}" is'},
            {"id": "first_branch_sentence", "core": False, "check": "copy",
             "sentence": "The sentence that states the first branch and does not state the second is"},
            {"id": "second_branch_sentence", "core": False, "check": "distinct_branch",
             "sentence": "The sentence that states the second branch and does not state the first is"},
            {"id": "survival_chance", "core": False, "check": "chance",
             "sentence": "A stated chance of survival is"},
            {"id": "nonreceipt", "core": False, "check": "nonreceipt",
             "sentence": "The words that say the other party does not receive it are"},
        ),
    },
    {
        "blueprint_id": "rescue_contrast",
        "summary": "Someone can rescue either of two parties, but not both.",
        "graph_builder": "implemented",
        "items": (
            {"id": "rescuer", "core": True, "check": "copy",
             "sentence": "The person who must choose between the rescues is"},
            {"id": "first_saved", "core": True, "check": "rescue_object",
             "sentence": "The first party who can be rescued is"},
            {"id": "second_saved", "core": True, "check": "rescue_object",
             "sentence": "The second, distinct party who can be rescued is"},
            {"id": "rescue_exclusivity", "core": True, "check": "not_both",
             "sentence": "The exact words that say both rescues cannot happen are"},
            {"id": "first_rescue_action", "core": True, "check": "rescue_action",
             "sentence": "The copied first rescue action is"},
            {"id": "second_rescue_action", "core": True, "check": "rescue_action",
             "sentence": "The copied second rescue action is"},
            {"id": "first_benefit", "core": True, "check": "outcome_clause",
             "sentence": "The beneficial outcome in the first rescue branch is"},
            {"id": "first_harm", "core": False, "check": "harm_clause",
             "sentence": "The harmful outcome in the first rescue branch is"},
            {"id": "second_benefit", "core": True, "check": "outcome_clause",
             "sentence": "The beneficial outcome in the second rescue branch is"},
            {"id": "second_harm", "core": False, "check": "harm_clause",
             "sentence": "The harmful outcome in the second rescue branch is"},
            {"id": "scene", "core": False, "check": "copy",
             "sentence": "The copied words that describe the parties already in danger are"},
        ),
    },
    {
        "blueprint_id": "omission_harm",
        "summary": "Doing an action and not doing it each harm someone.",
        "graph_builder": "implemented",
        "items": (
            {"id": "actor", "core": True, "check": "copy",
             "sentence": "The person who may do the action or not do it is"},
            {"id": "done", "core": True, "check": "copy",
             "sentence": "The action the text says is done is"},
            {"id": "omitted", "core": True, "check": "negated_action",
             "sentence": "The copied non-action branch, including the word not, is"},
            {"id": "harm_done", "core": True, "check": "harm_clause",
             "sentence": "The stated harmful outcome if the action is done is"},
            {"id": "harm_omitted", "core": True, "check": "harm_clause",
             "sentence": "The stated harmful outcome if the action is not done is"},
            {"id": "done_hedge", "core": False, "check": "branch_hedge",
             "sentence": "The conditional words attached to the harm from doing the action are"},
            {"id": "omitted_hedge", "core": False, "check": "branch_hedge",
             "sentence": "The conditional words attached to the harm from not doing the action are"},
            {"id": "group_counts", "core": False, "check": "group_count",
             "sentence": "One copied group count in a harmful outcome is"},
            {"id": "instrument", "core": False, "check": "instrument_contrast",
             "sentence": "The contrasting instrument named after but not is"},
        ),
    },
    {
        "blueprint_id": "conditional_outcome",
        "summary": "An outcome is stated inside an if-clause.",
        "graph_builder": "implemented",
        "items": (
            {"id": "actor", "core": False, "check": "copy",
             "sentence": "The person whose action an outcome depends on is"},
            {"id": "condition", "core": True, "check": "if_clause",
             "sentence": "The if-clause the outcome depends on is"},
            {"id": "bearer", "core": True, "check": "party",
             "sentence": "The party that outcome happens to is"},
            {"id": "outcome", "core": True, "check": "copy",
             "sentence": "The outcome stated for that party is"},
            {"id": "second_condition", "core": False, "check": "distinct_if_clause",
             "sentence": "The second if-clause, distinct from the first, is"},
            {"id": "second_bearer", "core": False, "check": "party",
             "sentence": "The party the second outcome happens to is"},
            {"id": "second_outcome", "core": False, "check": "outcome_clause",
             "sentence": "The outcome stated for that second party is"},
            {"id": "chance", "core": False, "check": "chance",
             "sentence": "A stated chance attached to an outcome is"},
        ),
    },
    {
        "blueprint_id": "ability_permission",
        "summary": "A modal states what an actor can, may, or is permitted to do.",
        "graph_builder": "discourse",
        "items": (
            {"id": "actor", "core": True, "check": "party", "sentence": "The actor governed by the modal is"},
            {"id": "modal_action", "core": True, "check": "modal_action", "sentence": "The copied action stated with can, may, able, or permitted is"},
            {"id": "target", "core": True, "check": "party", "sentence": "The target or recipient of that possible action is"},
            {"id": "modal_words", "core": False, "check": "modal_words", "sentence": "The exact modal words are"},
            {"id": "outcome", "core": False, "check": "outcome_clause", "sentence": "A stated outcome of that action is"},
            {"id": "duty_or_prohibition", "core": False, "check": "deontic_words", "sentence": "Any duty or prohibition attached to that action is"},
        ),
    },
    {
        "blueprint_id": "diversion_redirection",
        "summary": "An intervention diverts or redirects a process toward an affected party.",
        "graph_builder": "implemented",
        "items": (
            {"id": "actor", "core": True, "check": "party", "sentence": "The actor who can redirect the process is"},
            {"id": "controllable_process", "core": True, "check": "process_words", "sentence": "The process that can be redirected is"},
            {"id": "intervention", "core": True, "check": "diversion_action", "sentence": "The copied diversion or redirection action is"},
            {"id": "affected_party", "core": True, "check": "party", "sentence": "The party reached or affected by that process is"},
            {"id": "outcome", "core": True, "check": "outcome_clause", "sentence": "The stated outcome for that party is"},
            {"id": "alternative_route", "core": False, "check": "copy", "sentence": "A different route or destination stated in the text is"},
            {"id": "omission_branch", "core": False, "check": "negated_action", "sentence": "The copied non-intervention branch is"},
            {"id": "uncertainty", "core": False, "check": "hedge_any", "sentence": "The words that make the result uncertain are"},
        ),
    },
    {
        "blueprint_id": "deontic_rule",
        "summary": "A rule obligates, permits, or prohibits an action.",
        "graph_builder": "discourse",
        "items": (
            {"id": "deontic_words", "core": True, "check": "deontic_words", "sentence": "The words that state the obligation, permission, or prohibition are"},
            {"id": "governed_action", "core": True, "check": "copy", "sentence": "The action governed by that rule is"},
            {"id": "bearer", "core": True, "check": "party", "sentence": "The person or institution governed by the rule is"},
            {"id": "target", "core": False, "check": "party", "sentence": "The party that action is directed toward is"},
            {"id": "authority", "core": False, "check": "party", "sentence": "The stated source or authority for the rule is"},
            {"id": "exception", "core": False, "check": "exception_words", "sentence": "A stated exception to the rule is"},
            {"id": "sanction", "core": False, "check": "outcome_clause", "sentence": "A stated consequence of violating the rule is"},
        ),
    },
    {
        "blueprint_id": "promise_reliance",
        "summary": "Someone promises or commits to future conduct for another party.",
        "graph_builder": "discourse",
        "items": (
            {"id": "promisor", "core": True, "check": "party", "sentence": "The person who makes the promise or commitment is"},
            {"id": "commitment_event", "core": True, "check": "commitment_words", "sentence": "The copied promise or commitment event is"},
            {"id": "commitment_content", "core": True, "check": "copy", "sentence": "What the promisor commits to do is"},
            {"id": "promisee", "core": True, "check": "party", "sentence": "The party to whom the commitment is made is"},
            {"id": "reliance", "core": False, "check": "reliance_words", "sentence": "What the text says someone relies on is"},
            {"id": "breach", "core": False, "check": "breach_words", "sentence": "The copied breach or failure to perform is"},
        ),
    },
    {
        "blueprint_id": "uncertain_risk",
        "summary": "An action has an explicitly possible or probabilistic outcome.",
        "graph_builder": "implemented",
        "items": (
            {"id": "actor", "core": True, "check": "party", "sentence": "The actor who may perform the risky action is"},
            {"id": "action", "core": True, "check": "copy", "sentence": "The action carrying the stated risk is"},
            {"id": "possible_outcome", "core": True, "check": "outcome_clause", "sentence": "The possible or probabilistic outcome is"},
            {"id": "affected_party", "core": True, "check": "party", "sentence": "The party exposed to that outcome is"},
            {"id": "likelihood", "core": True, "check": "hedge_any", "sentence": "The copied likelihood or uncertainty words are"},
            {"id": "second_action", "core": False, "check": "copy", "sentence": "A second action with a different risk is"},
            {"id": "second_affected_party", "core": False, "check": "party", "sentence": "The party exposed to the second outcome is"},
            {"id": "second_outcome", "core": False, "check": "outcome_clause", "sentence": "The stated outcome of that second action is"},
            {"id": "second_likelihood", "core": False, "check": "hedge_any", "sentence": "The likelihood words attached to the second outcome are"},
            {"id": "expected_quantity", "core": False, "check": "number", "sentence": "A stated quantity affected by the risk is"},
        ),
    },
    {
        "blueprint_id": "disputed_report",
        "summary": "A source reports or believes a proposition that may be disputed.",
        "graph_builder": "discourse",
        "items": (
            {"id": "source", "core": True, "check": "party", "sentence": "The source or speaker of the report is"},
            {"id": "report_words", "core": True, "check": "report_words", "sentence": "The copied reporting, claiming, or belief words are"},
            {"id": "reported_content", "core": True, "check": "copy", "sentence": "The proposition attributed to that source is"},
            {"id": "competing_report", "core": False, "check": "report_words", "sentence": "A copied conflicting report is"},
            {"id": "reliability", "core": False, "check": "reliability_words", "sentence": "What the text says about source reliability is"},
            {"id": "confirmation", "core": False, "check": "confirmation_words", "sentence": "A later confirmation or refutation is"},
        ),
    },
)
_BY_ID = {row["blueprint_id"]: row for row in TEMPLATES}
_META_BLUEPRINTS = {
    "none": "No listed blueprint is supported by the scenario.",
    "composite": "The scenario needs a composition of multiple blueprint families.",
}


def choose_by_cloze(text: str, complete: Complete,
                    question: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ask for the three most likely templates, finish each, and keep the best.

    The ethical question and its alternatives are read from the parser before a
    world graph is built. An exclusive graph is withheld when exclusivity was
    not evidenced, so a later compiler cannot treat "or" as an averted harm.
    """
    question = deepcopy(question) if question is not None else assess_question(text)
    rank_raw = complete(_messages(_rank_prompt(text)))
    ranking = _inject_discourse_ranking(text, _parse_ranking(rank_raw))
    considered = []
    for index, blueprint_id in enumerate(ranking):
        if blueprint_id in _META_BLUEPRINTS:
            considered.append(_meta_attempt(blueprint_id, index))
            continue
        template = _BY_ID[blueprint_id]
        raw = complete(_messages(_cloze_prompt(text, template)))
        scored = _score_template(text, template, _parse_answers(raw), index)
        scored = _enrich_from_z10(text, template, scored)
        considered.append(_ask_implied(text, template, scored, complete))
    winner = considered[0] if considered and considered[0].get("meta_option") else _winner(considered)
    withheld = _world_withheld(question, winner)
    question["eligible_for_world_state"] = bool(
        winner and winner["status"] == "FILLED"
        and winner.get("graph_builder") in {"implemented", "discourse"}
        and not withheld)
    graph = _graph(text, winner, question) if question["eligible_for_world_state"] else None
    proposals = ([graph] if graph else
                 [_withheld_cloze_proposal(text, winner, withheld)] if winner else [])
    candidate_attempts = _candidate_attempts(text, question, considered, winner, graph)
    status = "WITHHELD" if withheld else (winner["status"] if winner else "NO_MATCH")
    return {
        "cloze_version": CLOZE_VERSION,
        "chosen_blueprint_id": winner["blueprint_id"] if winner else None,
        "status": status,
        "ranking": ranking,
        "left_out": [row["blueprint_id"] for row in TEMPLATES if row["blueprint_id"] not in ranking],
        "question": question,
        "world_withheld": withheld,
        "considered": considered,
        "candidate_attempts": candidate_attempts,
        "proposals": proposals,
        "graph": graph,
    }


def _meta_attempt(blueprint_id: str, rank: int) -> dict[str, Any]:
    return {
        "blueprint_id": blueprint_id,
        "graph_builder": "meta",
        "meta_option": True,
        "status": "FILLED",
        "rank": rank,
        "core_filled": 0,
        "core_count": 0,
        "optional_filled": 0,
        "rejected": 0,
        "coverage_metrics": coverage_metrics({}, {}),
        "unfilled": [],
        "unresolved_slots": [],
        "accepted_evidence": {},
        "slots": {},
        "items": [],
    }


def _candidate_attempts(text: str, question: dict[str, Any],
                        considered: list[dict[str, Any]],
                        winner: dict[str, Any] | None,
                        selected_graph: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Materialize every ranked template without changing chooser selection.

    Complete implemented and discourse templates become proposal envelopes.
    Partial and unmatched templates remain visible as withheld attempts with
    their missing slots. This makes the alternatives inspectable without allowing
    a lower-ranked interpretation to silently replace the chosen reading.
    """
    attempts = []
    for row in considered:
        selected = row is winner
        reasons = _world_withheld(question, row)
        proposal = None
        errors: list[str] = []
        if selected and selected_graph is not None:
            proposal = selected_graph
        elif (
            row.get("status") == "FILLED"
            and row.get("graph_builder") in {"implemented", "discourse"}
            and not reasons
        ):
            try:
                proposal = _graph(text, row, {
                    **question,
                    "eligible_for_world_state": True,
                })
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(str(exc))
        if proposal is None:
            withheld_reasons = list(reasons)
            if errors:
                withheld_reasons.extend(
                    "Candidate construction failed: " + error for error in errors
                )
            if row.get("status") != "FILLED":
                missing = [
                    item["id"] for item in row.get("items", [])
                    if item.get("core") and item.get("verdict") != "accepted"
                ]
                withheld_reasons.append(
                    "Required slots remain unfilled: " + ", ".join(missing)
                    if missing else "The template did not match the scenario."
                )
            proposal = _withheld_cloze_proposal(text, row, withheld_reasons)
        contract_errors = validate_proposal(proposal)
        attempts.append({
            "blueprint_id": row["blueprint_id"],
            "rank": row["rank"],
            "template_status": row["status"],
            "selected": selected,
            "core_filled": row["core_filled"],
            "core_count": row["core_count"],
            "optional_filled": row["optional_filled"],
            "unfilled_slots": list(row["unfilled"]),
            "contract_valid": not contract_errors,
            "contract_errors": contract_errors,
            "proposal": proposal,
        })
    return attempts


def assess_question(text: str) -> dict[str, Any]:
    """Record the question and the alternatives before any world graph.

    A scenario-option set does not say the branches cannot both occur.
    Exclusivity stays unspecified unless the text says "not both". The word
    "or" is not that evidence. Outcome participants stay listed even when a
    later recipient trim would drop them.
    """
    import parsing_game_Z10 as z10

    package = z10.export_candidate_graph(text, package_id="question_before_world")
    nodes = {node["id"]: node for node in package["nodes"]}
    by_id = {item["id"]: item for item in package["candidates"]}
    options = []
    for group in package["choice_sets"]:
        if group["kind"] != "scenario_option":
            continue
        labels = []
        for candidate_id in group["candidate_ids"]:
            prop = by_id[candidate_id]["arguments"].get("proposition")
            labels.append(nodes.get(prop, {}).get("label", ""))
        options.append({
            "id": group["id"],
            "selection_rule": group["selection_rule"],
            "exhaustive": group["exhaustive"],
            "options": [label for label in labels if label],
        })
    participants = []
    for item in package["candidates"]:
        if item["type"] != "PARTICIPANT":
            continue
        mention = nodes.get(item["arguments"].get("mention"), {}).get("label")
        predicate = nodes.get(item["arguments"].get("proposition"), {}).get("label")
        if mention and predicate:
            participants.append({
                "role": item.get("value"),
                "mention": mention,
                "predicate": predicate,
            })
    question = next(
        (node.get("label") or "" for node in package["nodes"] if node.get("kind") == "choice_point"),
        "",
    )
    exclusivity_proof = _exclusivity_proof(text, options, participants)
    exclusivity = (
        "evidenced"
        if exclusivity_proof["status"] in {"EXPLICIT", "DERIVED"}
        else "unspecified"
    )
    return {
        "ethical_question": question,
        "scenario_options": options,
        "exclusivity": exclusivity,
        "exclusivity_proof": exclusivity_proof,
        "participants": participants,
    }


def _exclusivity_proof(text: str, options: list[dict[str, Any]],
                       participants: list[dict[str, Any]]) -> dict[str, Any]:
    explicit = _EXPLICIT_EXCLUSIVITY.search(text)
    if explicit:
        host = _clause_for(text, explicit.group(0))
        return {
            "status": "EXPLICIT",
            "evidence": [{
                "text": explicit.group(0), "clause_id": host["clause_id"],
                "start": explicit.start(), "end": explicit.end(),
            }],
            "assumptions": [],
            "explanation": "The source explicitly says that both alternatives cannot occur.",
        }
    has_alternatives = any(len(row.get("options") or []) >= 2 for row in options)
    if not has_alternatives:
        mentions = {row.get("mention") for row in participants if row.get("mention")}
        has_alternatives = len(mentions) >= 2 and bool(re.search(r"\bor\b", text, re.I))
    # Dose/vial/organ lists may witness a countable unit. They may not mint
    # exclusivity. Bare ``or`` stays hypothesized even with "one dose".
    indivisible = _INDIVISIBLE_ONE.search(text)
    generic_one = re.search(r"\b(?:one|1|single)\b", text, re.I)
    unit = indivisible or generic_one
    if unit and has_alternatives:
        evidence = [{"text": unit.group(0)}]
        if indivisible:
            host = _clause_for(text, indivisible.group(0))
            evidence = [{
                "text": indivisible.group(0), "clause_id": host["clause_id"],
                "start": indivisible.start(), "end": indivisible.end(),
            }]
        return {
            "status": "HYPOTHESIZED",
            "evidence": evidence,
            "assumptions": ["the resource is indivisible", "allocation consumes it"],
            "explanation": (
                "A single item is mentioned, but indivisibility is not exclusivity "
                "and bare or is not exclusive."
            ),
        }
    return {
        "status": "UNKNOWN", "evidence": [], "assumptions": [],
        "explanation": "The source does not establish mutual exclusion.",
    }


def _world_withheld(question: dict[str, Any], winner: dict[str, Any] | None) -> list[str]:
    """Hold the graph until the question's alternatives are actually exclusive."""
    if winner is None:
        return []
    if winner.get("meta_option"):
        return [_META_BLUEPRINTS[winner["blueprint_id"]]]
    if winner.get("blueprint_id") != "exclusive_allocation":
        return []
    if winner.get("status") != "FILLED":
        return []
    proof = question.get("exclusivity_proof") or {}
    if (proof.get("status") in {"EXPLICIT", "DERIVED"}
            or question.get("exclusivity") == "evidenced"):
        return []
    kept = {
        winner["slots"].get("first_recipient", ""),
        winner["slots"].get("second_recipient", ""),
    }
    outcome_mentions = [
        row["mention"] for row in question.get("participants") or []
        if row["predicate"] in {"sustain", "lose", "die", "live", "drown", "drowning"}
        and row["mention"] not in kept
    ]
    reasons = [
        "Exclusivity is unspecified. The alternatives are not an exclusive set, "
        "so an averted-alternative effect is not licensed.",
    ]
    if outcome_mentions:
        reasons.append(
            "These outcome participants stay in the question and are not recipient heads: "
            + "; ".join(dict.fromkeys(outcome_mentions)) + "."
        )
    return reasons


def openai_complete(model: str = "gpt-4o-mini",
                    env_path: Path = DEFAULT_OPENAI_ENV) -> Complete:
    """Return a completer that calls a small chat model. The key is not logged."""
    key = os.environ.get("OPENAI_API_KEY") or _env_value(env_path, "OPENAI_API_KEY")
    if not key:
        raise RuntimeError(f"OPENAI_API_KEY is absent from the shell and {env_path}")

    def complete(messages: Sequence[dict[str, str]]) -> str:
        payload = {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": list(messages),
        }
        body = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI request failed ({error.code}): {detail[:500]}") from None
        return data["choices"][0]["message"]["content"]

    return complete


def _messages(user: str) -> list[dict[str, str]]:
    return [
        {"role": "system",
         "content": "You copy words from the scenario into blanks. "
                    "You do not choose a moral action and you do not add facts."},
        {"role": "user", "content": user},
    ]


def _rank_prompt(text: str) -> str:
    lines = ["Which three templates can be completed from this scenario?",
             "Return a JSON object {\"templates\": [id, id, id]} with the most likely first.",
             "Use none first when no listed template fits. Use composite first when two or more "
             "template families are jointly required and no single family is adequate.",
             "", "Scenario:", text, ""]
    for template in TEMPLATES:
        sample = template["items"][0]["sentence"]
        lines.append(f"- {template['blueprint_id']}: {template['summary']} Example: {sample} _______.")
    for blueprint_id, summary in _META_BLUEPRINTS.items():
        lines.append(f"- {blueprint_id}: {summary}")
    return "\n".join(lines)


def _cloze_prompt(text: str, template: dict[str, Any]) -> str:
    lines = [
        f"Template: {template['blueprint_id']}",
        "Finish each sentence with a short contiguous copy of the scenario.",
        "If the scenario does not state that fact, answer NONE.",
        "Do not paraphrase. Do not decide what anyone should do.",
        "A relative clause that states a consequence may fill an outcome sentence.",
    ]
    lines.extend(_template_instructions(template["blueprint_id"]))
    lines.extend(["", "Scenario:", text, ""])
    answers = []
    asked = [item for item in template["items"] if item["check"] != "implied"]
    for index, item in enumerate(asked, 1):
        lines.append(f"{index}. {item['sentence']} _______.")
        answers.append(f'"{item["id"]}"')
    lines.append("")
    lines.append("Return a JSON object {\"answers\": {" + ", ".join(
        f"{name}: \"copy or NONE\"" for name in answers) + "}}.")
    return "\n".join(lines)


def _template_instructions(blueprint_id: str) -> list[str]:
    common = [
        "Do not turn a mentioned party, object, instrument, or report into an action.",
        "Keep distinct mentions distinct. Do not resolve identity.",
    ]
    by_id = {
        "exclusive_allocation": [
            "\"whether ... or ...\" does not say that both parties cannot receive it.",
            "A count of people is not the amount of the resource.",
            "A giving or receipt event must use give, receive, allocate, deliver, administer, provide, or supply. Another verb is NONE.",
            "A hedge is if, unless, whether, or a percent chance inside that outcome. The word can is not a hedge.",
            "If one sentence states both branches, answer NONE for a sentence that would state only one branch.",
        ],
        "rescue_contrast": [
            "The rescue choice requires copied 'not both' evidence.",
            "Keep each rescue action with only the outcomes in its own conditional branch.",
            "Copy a harm only when that branch states it.",
        ],
        "omission_harm": [
            "Copy the complete harmful proposition, including its harm verb.",
            "A brake or other nearby instrument is context unless the text states it as an action branch.",
        ],
        "conditional_outcome": [
            "Keep each if-clause paired only with the outcome in its own sentence.",
            "Do not infer that two conditional branches are exclusive.",
        ],
        "ability_permission": ["Do not choose between ability, permission, and possibility when the wording leaves them ambiguous."],
        "diversion_redirection": ["Do not invent a route, default process, or affected party."],
        "deontic_rule": ["Copy the governed action separately from the deontic words."],
        "promise_reliance": ["A promise does not establish reliance or breach unless the text says so."],
        "uncertain_risk": ["Copy the likelihood words from the same outcome branch."],
        "disputed_report": ["A reported proposition is attributed content, not an admitted fact."],
    }
    return common + by_id.get(blueprint_id, [])


def _ask_implied(text: str, template: dict[str, Any], row: dict[str, Any],
                 complete: Complete) -> dict[str, Any]:
    """Ask the implied-process blanks only after the act and outcome are known."""
    rendered = []
    for item in template["items"]:
        if item["check"] != "implied":
            continue
        sentence = _implied_sentence(row["slots"], item["id"])
        if sentence:
            rendered.append({**item, "sentence": sentence})
    if not rendered:
        return row
    raw = complete(_messages(_implied_prompt(text, template, rendered)))
    follow = _score_template(
        text, {**template, "items": tuple(rendered)}, _parse_answers(raw), row["rank"])
    by_id = {item["id"]: item for item in follow["items"]}
    items = [by_id.get(item["id"], item) for item in row["items"]]
    slots = dict(row["slots"])
    slots.update(follow["slots"])
    optional = [item for item in items if not item["core"]]
    row = dict(row)
    row["items"] = items
    row["slots"] = slots
    row["optional_filled"] = sum(item["verdict"] == "accepted" for item in optional)
    row["rejected"] = sum(item["verdict"] == "rejected" for item in items)
    row["unfilled"] = [item["id"] for item in items if item["verdict"] != "accepted"]
    row["unresolved_slots"] = list(row["unfilled"])
    row["accepted_evidence"] = {
        item["id"]: item["span"] for item in items if item["verdict"] == "accepted"
    }
    row["coverage_metrics"] = coverage_metrics(
        {item["id"]: item["verdict"] == "accepted" for item in items if item["core"]},
        {item["id"]: item["verdict"] == "accepted" for item in items if not item["core"]},
        rejected=row["rejected"],
    )
    if row["core_filled"] == 0 and row["optional_filled"] == 0:
        row["status"] = "NO_MATCH"
    elif row["core_filled"] == row["core_count"]:
        row["status"] = "FILLED"
    return row


def _implied_prompt(text: str, template: dict[str, Any], items: list[dict[str, Any]]) -> str:
    lines = [
        f"Template: {template['blueprint_id']}",
        "Implied processes, using the copies already made.",
        "The quoted resource and the quoted result are earlier answers. Do not replace them.",
        "Finish the blank with what that resource does to produce the quoted result.",
        "Name that physical process. Do not name a decision, an evaluation, or an assessment.",
        "The process words do not have to appear in the scenario.",
        "Answer NONE when the resource does not produce that result.",
        "", "Scenario:", text, "",
    ]
    for index, item in enumerate(items, 1):
        lines.append(f"{index}. {item['sentence']} _______.")
    names = ", ".join(f'"{item["id"]}": "process or NONE"' for item in items)
    lines.append("")
    lines.append("Return a JSON object {\"answers\": {" + names + "}}.")
    return "\n".join(lines)


def _implied_sentence(slots: dict[str, str], item_id: str) -> str:
    which = "first" if item_id.startswith("first") else "second"
    resource = slots.get("resource") or ""
    outcome = _consequence(
        slots.get(f"{which}_outcome") or "", slots.get(f"{which}_recipient") or "")
    if not resource or not outcome:
        return ""
    return (
        "The process most aligned with the text might explain how "
        f'"{resource}" leads to "{outcome}" is'
    )


def _consequence(outcome: str, recipient: str) -> str:
    """Keep the result that follows a party, so the party is not the result."""
    if not outcome or not recipient or not outcome.casefold().startswith(recipient.casefold()):
        return outcome
    rest = outcome[len(recipient):].lstrip(" ,")
    match = re.match(r"(?:that|who|which)\s+(.+)$", rest, re.I)
    return match.group(1) if match else outcome


_DECISION_FRAME = re.compile(
    r"^(?:(?:must|can|could|will|should)\s+)?(?:decide|choose|chooses)\s+whether\s+to\s+",
    re.I,
)
_ALLOCATION_DECISION = re.compile(
    r"\b(?:must|can|could|will|should)\s+(?:decide|choose|chooses)\s+whether\b",
    re.I,
)
_COMMITMENT_VERB = re.compile(
    r"\b(?:promised|promises|promise|committed|commits|commit|pledged|pledges|pledge)\b",
    re.I,
)


def _allocation_locks_discourse(text: str) -> bool:
    """A decision frame or explicit not-both is allocation, not duty or ability."""
    return bool(_EXPLICIT_EXCLUSIVITY.search(text or "")
                or _ALLOCATION_DECISION.search(text or ""))


def _inject_discourse_ranking(text: str, ranking: Sequence[str]) -> list[str]:
    """Score duty/ability when a modal governs a transfer, even if the LLM skipped it."""
    ranking = list(ranking)
    if _allocation_locks_discourse(text):
        return ranking
    if not _TRANSFER_EVENT.search(text or ""):
        return ranking
    injected = []
    if _DEONTIC_WORDS.search(text or ""):
        injected.append("deontic_rule")
    elif _MODAL_WORDS.search(text or ""):
        injected.append("ability_permission")
    extra = [item for item in injected if item not in ranking]
    return extra + ranking


def _assigned_act(text: str, span: str) -> str:
    """A decision frame is not the act that applies the resource."""
    match = _DECISION_FRAME.match(span)
    if not match:
        return span
    rest = span[match.end():].strip()
    copies = _copy_spans(text, rest)[0]
    return copies[0] if copies else span


def _commitment_event_head(text: str, span: str) -> str:
    """Keep the promise verb; the promisee is a party, not part of the event."""
    match = _COMMITMENT_VERB.search(span or "")
    if not match:
        return span
    copies = _copy_spans(text, match.group(0))[0]
    return copies[0] if copies else span


def _include_copied_target(text: str, accepted: dict[str, str],
                           items: list[dict[str, Any]]) -> None:
    """Prefer a licensed copy that already includes the copied destination."""
    target = accepted.get("target") or ""
    for action_key in ("modal_action", "governed_action"):
        action = accepted.get(action_key) or ""
        if not action or not target or target.casefold() in action.casefold():
            continue
        for glue in (f"{action} to {target}", f"{action} {target}"):
            copies = _copy_spans(text, glue)[0]
            if not copies:
                continue
            accepted[action_key] = copies[0]
            for row in items:
                if row["id"] == action_key and row["verdict"] == "accepted":
                    row["span"] = copies[0]
            break


def _branch_transfer(text: str, actor: str, branch_span: str,
                     recipient: str = "") -> str:
    """Recover the exact transfer event from its copied if-branch.

    The model often answers with the base form from the choice sentence
    (``give``), while the branch contains the factual support form (``gives``).
    This recovery strips only the copied IF frame and the copied subject
    (decider or recipient).
    """
    host = _span_host(text, branch_span)
    if not host or not str(branch_span or "").strip():
        return ""
    receipt = re.compile(
        r"\b(?:get|gets|got|getting|receive|receives|received|receiving)\b", re.I)
    match = re.match(r"\s*If\s+(.+?),", host["text"], re.I)
    if match:
        candidate = match.group(1).strip()
        if actor and candidate.casefold().startswith(actor.casefold()):
            rest = candidate[len(actor):].strip()
            if rest and (_TRANSFER_EVENT.search(rest) or receipt.search(rest)):
                copies = _copy_spans(text, rest)[0]
                if copies and not re.search(r"\bnot\s+both\b", copies[0], re.I):
                    return copies[0]
        if _TRANSFER_EVENT.search(candidate) or receipt.search(candidate):
            copies = _copy_spans(text, candidate)[0]
            if copies and not re.search(r"\bnot\s+both\b", copies[0], re.I):
                return copies[0]
    for subject in (recipient, actor):
        if not subject:
            continue
        match = re.match(
            rf"\s*If\s+{re.escape(subject)}\s+(.+?),",
            host["text"], re.I)
        if not match:
            continue
        candidate = match.group(1).strip()
        if not (_TRANSFER_EVENT.search(candidate) or receipt.search(candidate)):
            continue
        copies = _copy_spans(text, candidate)[0]
        if copies:
            return copies[0]
    return ""


def _branch_intervention(text: str, assignment: str, recipient: str,
                         transfer: str, recovered: str = "") -> str:
    """A branch act is a source copy naming that recipient, not the exclusive or-clause."""
    assignment_head = _assigned_act(text, assignment)
    glued = f"{assignment_head} to {recipient}" if assignment_head and recipient else ""
    ranked = []
    if transfer and _span_names_recipient(transfer, recipient):
        ranked.append(transfer)
    if recovered:
        ranked.append(recovered)
    ranked.extend([transfer, glued, assignment_head])
    for candidate in ranked:
        candidate = " ".join(str(candidate or "").split())
        if not candidate or re.search(r"\bnot\s+both\b", candidate, re.I):
            continue
        copies = _copy_spans(text, candidate)[0]
        if copies:
            span = copies[0]
            if not re.search(r"\bnot\s+both\b", span, re.I):
                return span
    return recovered or transfer or assignment_head or assignment


def _span_names_recipient(span: str, recipient: str) -> bool:
    span = (span or "").casefold()
    recipient = (recipient or "").casefold()
    if not span or not recipient:
        return False
    if recipient in span:
        return True
    for prefix in ("a ", "an ", "the ", "one ", "two ", "three ", "four ",
                   "five ", "six ", "seven ", "eight ", "nine ", "ten "):
        if recipient.startswith(prefix):
            recipient = recipient[len(prefix):]
            break
    return bool(recipient) and recipient in span


def _score_template(text: str, template: dict[str, Any], answers: dict[str, str],
                    rank: int) -> dict[str, Any]:
    accepted: dict[str, str] = {}
    items = []
    for item in template["items"]:
        raw = answers.get(item["id"], "NONE")
        spans, limit_reason = _copy_spans(text, raw)
        span = None
        if item["check"] == "implied" and not _abstains(raw) and not limit_reason:
            phrase = " ".join(str(raw).strip().strip("\"'`").split()).rstrip(".")
            if spans:
                span = spans[0]
                accepted[item["id"]] = span
                verdict, reason = "accepted", "copied"
            elif phrase:
                span = phrase
                accepted[item["id"]] = phrase
                verdict, reason = "accepted", "implied"
            else:
                verdict, reason = "empty", "not_stated"
        elif _abstains(raw):
            verdict, reason = "empty", "not_stated"
        elif limit_reason:
            verdict, reason = "rejected", limit_reason
        elif not spans:
            verdict, reason = "rejected", "not_a_copy"
        else:
            verdict, reason = "rejected", "does_not_answer"
            for candidate in spans:
                if not _answers_sentence(text, item, candidate, accepted):
                    continue
                if _duplicates_party(item, candidate, accepted):
                    reason = "same_party"
                    continue
                head = _party_head(text, candidate) if item["id"] in _PARTY_IDS else candidate
                if item["check"] == "assignment":
                    head = _assigned_act(text, head)
                elif item["check"] == "commitment_words":
                    head = _commitment_event_head(text, head)
                if head != candidate and not _answers_sentence(text, item, head, accepted):
                    head = candidate
                span = head
                accepted[item["id"]] = head
                verdict, reason = "accepted", "copied"
                break
        items.append({
            "id": item["id"], "core": item["core"], "sentence": item["sentence"],
            "completion": raw, "span": span if verdict == "accepted" else None,
            "verdict": verdict, "reason": reason,
        })
    if template["blueprint_id"] == "omission_harm":
        recoveries = []
        actor = accepted.get("actor", "")
        done_hedge = accepted.get("done_hedge", "")
        if "done" not in accepted and actor and done_hedge:
            match = re.match(rf"if\s+{re.escape(actor)}\s+(.+)$", done_hedge, re.I)
            if match:
                recoveries.append(("done", match.group(1)))
        omitted_hedge = accepted.get("omitted_hedge", "")
        if "omitted" not in accepted and omitted_hedge:
            match = re.search(r"\b(?:does|do|did)\s+not\s+.+$", omitted_hedge, re.I)
            if match:
                recoveries.append(("omitted", match.group(0)))
        for item_id, recovered in recoveries:
            copied = _copy_spans(text, recovered.strip(" ,."))[0]
            if not copied:
                continue
            accepted[item_id] = copied[0]
            for row in items:
                if row["id"] == item_id:
                    row.update({
                        "span": copied[0],
                        "verdict": "accepted",
                        "reason": "recovered_from_copied_conditional",
                    })
                    break
    _include_copied_target(text, accepted, items)
    if template["blueprint_id"] == "exclusive_allocation" and "assignment" not in accepted:
        # The assignment is often expressed once as a shared infinitive ("give
        # the medicine to Ben or Cara") while a copied branch supplies a valid
        # transfer event. Reuse that source-copied event rather than asking the
        # model to manufacture a second surface form.
        recovered = accepted.get("first_transfer") or accepted.get("second_transfer")
        if recovered:
            accepted["assignment"] = recovered
            for row in items:
                if row["id"] == "assignment":
                    row.update({
                        "span": recovered, "verdict": "accepted",
                        "reason": "recovered_from_copied_transfer",
                    })
                    break
    core = [row for row in items if row["core"]]
    optional = [row for row in items if not row["core"]]
    core_filled = sum(row["verdict"] == "accepted" for row in core)
    optional_filled = sum(row["verdict"] == "accepted" for row in optional)
    rejected = sum(row["verdict"] == "rejected" for row in items)
    status = "FILLED" if core and core_filled == len(core) else "PARTIAL"
    if core_filled == 0 and not any(row["verdict"] == "accepted" for row in optional):
        status = "NO_MATCH"
    return {
        "blueprint_id": template["blueprint_id"],
        "graph_builder": template["graph_builder"],
        "status": status,
        "rank": rank,
        "core_filled": core_filled,
        "core_count": len(core),
        "optional_filled": optional_filled,
        "rejected": rejected,
        "coverage_metrics": coverage_metrics(
            {row["id"]: row["verdict"] == "accepted" for row in core},
            {row["id"]: row["verdict"] == "accepted" for row in optional},
            rejected=rejected,
            unresolved_readings=1 if template["blueprint_id"] in {
                "ability_permission", "deontic_rule", "promise_reliance",
                "disputed_report",
            } else 0,
        ),
        "unfilled": [row["id"] for row in items if row["verdict"] != "accepted"],
        "unresolved_slots": [row["id"] for row in items if row["verdict"] != "accepted"],
        "accepted_evidence": {
            row["id"]: row["span"] for row in items if row["verdict"] == "accepted"
        },
        "slots": accepted,
        "items": items,
    }


def _enrich_conditional_from_z10(text: str, template: dict[str, Any],
                                 row: dict[str, Any]) -> dict[str, Any]:
    """Fill conditional and omission branches from Z10's typed graph.

    The adapter only copies source spans.  It uses ``CONDITIONAL_ON`` to keep
    each condition with its own consequence, PREDICATION polarity to
    distinguish doing from omission, and PARTICIPANT/QUANTITY candidates for
    the actor and affected party.  It does not infer exclusivity or causation.
    """
    import parsing_game_S as parsing
    import parsing_game_Z10 as z10

    package = z10.export_candidate_graph(text, package_id="blueprint_conditional_recovery")
    doc = parsing.get_nlp()(text)
    nodes = {node["id"]: node for node in package["nodes"]}
    candidates = package["candidates"]

    roles: dict[str, list[tuple[str, str]]] = {}
    predications: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if candidate["type"] == "PARTICIPANT":
            prop = candidate["arguments"]["proposition"]
            mention = nodes.get(candidate["arguments"]["mention"], {}).get("label", "")
            if mention:
                roles.setdefault(prop, []).append((candidate["value"], mention))
        elif candidate["type"] == "PREDICATION":
            predications[candidate["arguments"]["proposition"]] = candidate

    def token_for(prop_id: str):
        if not re.fullmatch(r"p\d+", prop_id or ""):
            return None
        index = int(prop_id[1:])
        return doc[index] if 0 <= index < len(doc) else None

    def exact(value: str) -> str:
        value = (value or "").strip(" ,.")
        copies = _copy_spans(text, value)[0]
        return copies[0] if copies else ""

    def recover(item_id: str, value: str, *, replace: bool = True,
                candidate_ids: Sequence[str] = ()) -> None:
        value = exact(value)
        if not value:
            return
        item = next((entry for entry in row["items"] if entry["id"] == item_id), None)
        if item is None:
            return
        current = row["slots"].get(item_id)
        if current and not replace and item["verdict"] == "accepted":
            return
        row["slots"][item_id] = value
        recovery = {
            "producer": "parsing_game_Z10",
            "method": "conditional_candidate_source_alignment",
            "span": value,
        }
        if candidate_ids:
            recovery["z10_candidate_ids"] = list(candidate_ids)
        row.setdefault("semantic_recoveries", {})[item_id] = recovery
        item.update({
            "span": value,
            "verdict": "accepted",
            "reason": "recovered_from_z10_conditional_structure",
        })

    def split_conditional(prop_id: str) -> tuple[str, str, str]:
        token = token_for(prop_id)
        if token is None:
            return "", "", ""
        sentence = token.sent.text.strip().rstrip(".")
        leading = re.match(r"^\s*((?:if|unless)\b[^,]+),\s*(.+)$", sentence, re.I)
        if leading:
            condition, outcome = leading.group(1), leading.group(2)
        else:
            trailing = re.match(
                r"^\s*(.+?)(?:,\s*|\s+)((?:if|unless)\b.+)$", sentence, re.I)
            if not trailing:
                return "", "", ""
            outcome, condition = trailing.group(1), trailing.group(2)
        return exact(condition), exact(outcome), exact(sentence)

    branches: list[dict[str, Any]] = []
    for link in candidates:
        if link["type"] != "CONDITIONAL_ON":
            continue
        condition_prop = link["arguments"]["condition"]
        consequence_prop = link["arguments"]["consequence"]
        condition_token = token_for(condition_prop)
        consequence_token = token_for(consequence_prop)
        if condition_token is None or consequence_token is None:
            continue
        condition, outcome, sentence = split_conditional(condition_prop)
        if not condition or not outcome:
            continue
        condition_roles = roles.get(condition_prop, [])
        consequence_roles = roles.get(consequence_prop, [])
        actors = [mention for role, mention in condition_roles
                  if role in {"agent", "controller"}]
        # Z10 preserves the passive condition but may expose only its surface
        # subject.  Reuse the same dependency parse to recover an explicit
        # ``by Maria`` agent before falling back to that subject.
        for child in condition_token.children:
            if child.dep_ == "agent" or (child.lemma_.casefold() == "by"
                                         and child.dep_ == "prep"):
                actors.extend(grand.text for grand in child.children
                              if grand.dep_ == "pobj")
        actors.extend(mention for role, mention in condition_roles
                      if role == "subject" and mention not in actors)
        bearers = [mention for role, mention in consequence_roles
                   if role in {"subject", "patient", "object"}]
        if not actors or not bearers:
            continue
        polarity = predications.get(condition_prop, {}).get(
            "scope", {}).get("polarity", "unresolved")
        predicate = nodes.get(condition_prop, {}).get("predicate", "").casefold()
        content = re.sub(r"^(?:if|unless)\s+", "", condition, flags=re.I)
        content = re.sub(rf"^{re.escape(actors[0])}\s+", "", content, flags=re.I)
        content = exact(content)
        branches.append({
            "condition": condition,
            "action": content,
            "outcome": outcome,
            "sentence": sentence,
            "actor": actors[0],
            "bearer": bearers[0],
            "polarity": polarity,
            "predicate": predicate,
            "order": condition_token.i,
            "link_id": link["id"],
        })
    branches.sort(key=lambda branch: branch["order"])
    if not branches:
        return row

    blueprint_id = template["blueprint_id"]
    if blueprint_id == "conditional_outcome":
        recover("actor", branches[0]["actor"])
        for index, branch in enumerate(branches[:2]):
            prefix = "" if index == 0 else "second_"
            ids = [branch["link_id"]]
            recover(f"{prefix}condition", branch["condition"], candidate_ids=ids)
            recover(f"{prefix}bearer", branch["bearer"], candidate_ids=ids)
            recover(f"{prefix}outcome", branch["outcome"], candidate_ids=ids)
        # A chance remains attached to its own copied outcome.  The graph
        # builder checks the containing clause before applying this shared slot.
        chance = next((_CHANCE.search(branch["outcome"]) for branch in branches
                       if _CHANCE.search(branch["outcome"])), None)
        if chance:
            recover("chance", chance.group(0))
    elif blueprint_id == "omission_harm":
        positive = next((branch for branch in branches
                         if branch["polarity"] == "positive"), None)
        negative = next((branch for branch in branches
                         if branch["polarity"] == "negative"), None)
        # Omission is supported only when Z10 found opposite polarities of the
        # same predicate and both consequences actually state harm.  Otherwise
        # leave the cloze answer untouched for the general conditional template.
        harm_pattern = re.compile(
            r"\b(?:die|dies|died|kill|kills|killed|harm|harms|harmed|"
            r"drown|drowns|drowned|suffer|suffers|suffered)\b", re.I)
        if not positive or not negative or positive["predicate"] != negative["predicate"]:
            return row
        if not harm_pattern.search(positive["outcome"]) or not harm_pattern.search(negative["outcome"]):
            return row
        recover("actor", positive["actor"])
        recover("done", positive["action"], candidate_ids=[positive["link_id"]])
        recover("omitted", negative["action"], candidate_ids=[negative["link_id"]])
        recover("harm_done", positive["outcome"], candidate_ids=[positive["link_id"]])
        recover("harm_omitted", negative["outcome"], candidate_ids=[negative["link_id"]])
        recover("done_hedge", positive["condition"], candidate_ids=[positive["link_id"]])
        recover("omitted_hedge", negative["condition"], candidate_ids=[negative["link_id"]])
        recover("group_counts", positive["bearer"], replace=False)
    return _refresh_scored_row(row)


def _enrich_discourse_target(text: str, template: dict[str, Any],
                             row: dict[str, Any]) -> dict[str, Any]:
    """Copy a destination participant onto duty/ability when cloze omitted it."""
    import parsing_game_Z10 as z10
    from blueprint_discourse import destination_from_package

    package = z10.export_candidate_graph(text, package_id="blueprint_discourse_target")
    destination = destination_from_package(package)
    copies = _copy_spans(text, destination)[0] if destination else []
    if copies and not row["slots"].get("target"):
        span = copies[0]
        row["slots"]["target"] = span
        for item in row["items"]:
            if item["id"] == "target":
                item.update({"span": span, "verdict": "accepted", "reason": "copied"})
                break
    _include_copied_target(text, row["slots"], row["items"])
    return _refresh_scored_row(row)


def _enrich_from_z10(text: str, template: dict[str, Any],
                     row: dict[str, Any]) -> dict[str, Any]:
    """Recover typed source spans that the cloze model omitted or paraphrased.

    Z10 is the repository's source-aligned semantic parser.  The cloze answers
    still choose a blueprint, but they are no longer the sole source of its
    slots.  This first integration covers the allocation construction where a
    quantity, participant role, conditional branch, or passive surface form is
    already represented in Z10.  Every recovered value remains an exact source
    span; this function never synthesizes an event or resolves a Z10 ambiguity.
    """
    if template["blueprint_id"] in {"conditional_outcome", "omission_harm"}:
        return _enrich_conditional_from_z10(text, template, row)
    if template["blueprint_id"] in {"ability_permission", "deontic_rule"}:
        return _enrich_discourse_target(text, template, row)
    if template["blueprint_id"] != "exclusive_allocation":
        return row

    import parsing_game_S as parsing
    import parsing_game_Z10 as z10

    package = z10.export_candidate_graph(text, package_id="blueprint_slot_recovery")
    doc = parsing.get_nlp()(text)
    nodes = {node["id"]: node for node in package["nodes"]}
    evidence = {item["id"]: item for item in package["evidence"]}
    candidates = package["candidates"]
    propositions = {
        node["id"]: node for node in package["nodes"]
        if node.get("kind") == "proposition"
    }
    roles: dict[str, list[tuple[str, str]]] = {}
    for item in candidates:
        if item["type"] != "PARTICIPANT":
            continue
        prop = item["arguments"]["proposition"]
        mention = nodes.get(item["arguments"]["mention"], {}).get("label", "")
        if mention:
            roles.setdefault(prop, []).append((item["value"], mention))

    transfer_lemmas = {
        "give", "receive", "allocate", "assign", "send", "deliver",
        "transfer", "administer", "provide", "supply",
    }

    def prop_token(prop_id: str):
        if not prop_id.startswith("p"):
            return None
        index = int(prop_id[1:])
        return doc[index] if 0 <= index < len(doc) else None

    def proposition_span(prop_id: str) -> str:
        token = prop_token(prop_id)
        if token is None:
            return ""
        subtree = [part for part in token.subtree if not part.is_punct]
        if not subtree:
            return token.text
        start = min(part.idx for part in subtree)
        end = max(part.idx + len(part.text) for part in subtree)
        span = text[start:end].strip(" ,.")
        span = re.sub(r"^(?:if|unless)\s+", "", span, flags=re.I)
        copies = _copy_spans(text, span)[0]
        return copies[0] if copies else ""

    def predicate_span(prop_id: str) -> str:
        """Return the exact predicate phrase without its surface subject."""
        token = prop_token(prop_id)
        if token is None:
            return ""
        subtree = [part for part in token.subtree if not part.is_punct and part.idx >= token.idx]
        if not subtree:
            return token.text
        start = token.idx
        end = max(part.idx + len(part.text) for part in subtree)
        span = text[start:end].strip(" ,.")
        copies = _copy_spans(text, span)[0]
        return copies[0] if copies else ""

    def sentence_for_prop(prop_id: str) -> str:
        token = prop_token(prop_id)
        return token.sent.text.strip() if token is not None else ""

    def recover(item_id: str, value: str, *, replace: bool = False,
                candidate_ids: Sequence[str] = ()) -> None:
        value = (value or "").strip(" ,.")
        if not value or not _copy_spans(text, value)[0]:
            return
        current = row["slots"].get(item_id)
        item = next((entry for entry in row["items"] if entry["id"] == item_id), None)
        if item is None or (current and not replace and item["verdict"] == "accepted"):
            return
        exact = _copy_spans(text, value)[0][0]
        row["slots"][item_id] = exact
        recovery = {
            "producer": "parsing_game_Z10",
            "method": "typed_candidate_source_alignment",
            "span": exact,
        }
        if candidate_ids:
            recovery["z10_candidate_ids"] = list(candidate_ids)
        row.setdefault("semantic_recoveries", {})[item_id] = recovery
        item.update({
            "span": exact,
            "verdict": "accepted",
            "reason": "recovered_from_z10_semantics",
        })

    # Quantity comes from Z10's typed QUANTITY candidate, including forms such
    # as "a single dose" and "one indivisible dose".
    resource_units = {
        "dose", "vial", "unit", "organ", "seat", "ticket", "bed", "ventilator",
    }
    quantity_rows = [
        item for item in candidates
        if item["type"] == "QUANTITY"
        and str((item.get("value") or {}).get("unit") or "").casefold() in resource_units
    ]
    if quantity_rows:
        mention_id = quantity_rows[0]["arguments"]["mention"]
        recover("quantity", nodes.get(mention_id, {}).get("label", ""),
                candidate_ids=[quantity_rows[0]["id"]])
    else:
        # Z10 intentionally freezes its quantity vocabulary at explicit
        # numerals.  Its aligned mention plus the existing dependency parse
        # still lets this adapter interpret adjectival ``single`` as exact one
        # without changing the frozen package schema.
        for token in doc:
            if token.lemma_.casefold() not in resource_units:
                continue
            if not any(child.lemma_.casefold() == "single" and child.dep_ == "amod"
                       for child in token.children):
                continue
            mention = nodes.get(f"m{token.i}", {}).get("label", "")
            recover("quantity", mention)
            break

    transfer_props = [
        prop_id for prop_id, node in propositions.items()
        if node.get("predicate", "").casefold() in transfer_lemmas
    ]
    main_prop = max(
        transfer_props,
        key=lambda prop_id: (
            sum(role == "destination" and mention.casefold() != "both"
                for role, mention in roles.get(prop_id, [])),
            not any(ctx.get("kind") == "hypothetical"
                    for item in candidates
                    if item["type"] == "PREDICATION"
                    and item["arguments"].get("proposition") == prop_id
                    for ctx in item.get("scope", {}).get("contexts", [])),
            -int(prop_id[1:]),
        ),
        default=None,
    )

    if main_prop:
        token = prop_token(main_prop)
        main_roles = roles.get(main_prop, [])
        actors = [mention for role, mention in main_roles
                  if role in {"subject", "agent", "controller"}
                  and mention.casefold() not in {"it", "they", "both"}]
        if not actors and token is not None:
            for child in token.children:
                if child.dep_ == "agent":
                    actors.extend(grand.text for grand in child.children if grand.dep_ == "pobj")
        malformed_decider = bool(row["slots"].get("decider") and
                                 _TRANSFER_EVENT.search(row["slots"]["decider"]))
        if actors:
            recover("decider", actors[0], replace=malformed_decider)
        objects = [mention for role, mention in main_roles
                   if role == "object" and mention.casefold() not in {"it", "both"}]
        if objects:
            recover("resource", objects[0])
        main_surface = predicate_span(main_prop)
        if main_surface:
            recover(
                "assignment", main_surface,
                replace=bool(re.search(
                    r"\b(?:it|this|that)\b", row["slots"].get("assignment", ""), re.I)),
            )

    # Build branch records from conditional transfer propositions.  Z10 has
    # already distinguished the condition from its consequence and attached
    # participant roles to each proposition.
    conditional_rows = [item for item in candidates if item["type"] == "CONDITIONAL_ON"]
    branches: list[dict[str, str]] = []
    for link in conditional_rows:
        condition = link["arguments"]["condition"]
        consequence = link["arguments"]["consequence"]
        if condition not in transfer_props:
            continue
        participant_rows = roles.get(condition, [])
        recipients = [mention for role, mention in participant_rows
                      if role in {"subject", "destination"}
                      and mention.casefold() not in {"it", "they", "both"}]
        if not recipients:
            continue
        sentence = sentence_for_prop(condition)
        marker = next((evidence[eid]["text"] for eid in link["evidence_ids"]
                       if eid in evidence), "")
        outcome = sentence.strip(" .")
        if re.match(r"\s*(?:if|unless)\b", outcome, re.I) and "," in outcome:
            outcome = outcome.split(",", 1)[1].strip()
        else:
            split = re.split(r"\s+\b(?:if|unless)\b\s+", outcome, maxsplit=1, flags=re.I)
            if len(split) == 2:
                outcome = split[0].strip()
        branches.append({
            "recipient": recipients[0],
            "transfer": predicate_span(condition),
            "outcome": outcome,
            "hedge": marker,
            "sentence": sentence.strip(" ."),
            "link_id": link["id"],
        })

    # Preserve source order and match existing recipient choices when possible.
    by_recipient = {branch["recipient"].casefold(): branch for branch in branches}
    ordered_names = [row["slots"].get("first_recipient"), row["slots"].get("second_recipient")]
    remaining = list(branches)
    ordered: list[dict[str, str]] = []
    for name in ordered_names:
        branch = by_recipient.get((name or "").casefold())
        if branch and branch not in ordered:
            ordered.append(branch)
    ordered.extend(branch for branch in remaining if branch not in ordered)
    for index, branch in enumerate(ordered[:2]):
        prefix = "first" if index == 0 else "second"
        recover(f"{prefix}_recipient", branch["recipient"])
        recover(f"{prefix}_transfer", branch["transfer"],
                candidate_ids=[branch["link_id"]])
        recover(f"{prefix}_outcome", branch["outcome"],
                candidate_ids=[branch["link_id"]])
        recover(f"{prefix}_hedge", branch["hedge"])
        recover(f"{prefix}_branch_sentence", branch["sentence"],
                candidate_ids=[branch["link_id"]])

    explicit = _EXPLICIT_EXCLUSIVITY.search(text)
    if explicit:
        recover("exclusivity", explicit.group(0))

    # A nonreceipt consequence can be expressed lexically ("untreated") or by
    # a compositional modifier ("left without serum").  Require a parsed death
    # proposition in the same source sentence before accepting the clause.
    for prop_id, node in propositions.items():
        if node.get("predicate", "").casefold() != "die":
            continue
        sentence = sentence_for_prop(prop_id).strip(" .")
        if re.search(r"\b(?:untreated|without|not\s+(?:receive|get)|doesn't\s+get)\b",
                     sentence, re.I):
            recover("nonreceipt", sentence)
            break

    return _refresh_scored_row(row)


def _refresh_scored_row(row: dict[str, Any]) -> dict[str, Any]:
    core = [item for item in row["items"] if item["core"]]
    optional = [item for item in row["items"] if not item["core"]]
    row["core_filled"] = sum(item["verdict"] == "accepted" for item in core)
    row["optional_filled"] = sum(item["verdict"] == "accepted" for item in optional)
    row["rejected"] = sum(item["verdict"] == "rejected" for item in row["items"])
    row["unfilled"] = [item["id"] for item in row["items"] if item["verdict"] != "accepted"]
    row["unresolved_slots"] = list(row["unfilled"])
    row["accepted_evidence"] = {
        item["id"]: item["span"] for item in row["items"]
        if item["verdict"] == "accepted"
    }
    row["coverage_metrics"] = coverage_metrics(
        {item["id"]: item["verdict"] == "accepted" for item in core},
        {item["id"]: item["verdict"] == "accepted" for item in optional},
        rejected=row["rejected"],
    )
    if row["core_filled"] == 0 and row["optional_filled"] == 0:
        row["status"] = "NO_MATCH"
    elif row["core_filled"] == len(core):
        row["status"] = "FILLED"
    else:
        row["status"] = "PARTIAL"
    return row


def _winner(considered: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [row for row in considered if row["core_filled"] or row["optional_filled"]]
    if not usable:
        return None
    return max(usable, key=lambda row: (
        row["status"] == "FILLED",
        row["core_filled"],
        row["optional_filled"],
        -row["rejected"],
        -row["rank"],
    ))


def _graph(text: str, winner: dict[str, Any],
           question: dict[str, Any]) -> dict[str, Any]:
    slots = winner["slots"]
    proof = question.get("exclusivity_proof") or {
        "status": "EXPLICIT" if question.get("exclusivity") == "evidenced" else "UNKNOWN",
        "evidence": [], "assumptions": [],
        "explanation": "Compatibility record from the question assessment.",
    }
    import parsing_game_Z10 as z10
    package = z10.export_candidate_graph(text, package_id="cloze_evidence")
    if winner["blueprint_id"] in DISCOURSE_BUILDERS:
        world, notes = DISCOURSE_BUILDERS[winner["blueprint_id"]](text, slots, package)
    else:
        builder = {
            "exclusive_allocation": _allocation_graph,
            "rescue_contrast": _rescue_graph,
            "omission_harm": _omission_graph,
            "conditional_outcome": _conditional_graph,
            "diversion_redirection": _diversion_graph,
            "uncertain_risk": _risk_graph,
        }[winner["blueprint_id"]]
        if winner["blueprint_id"] == "exclusive_allocation":
            world, notes = builder(text, slots, proof)
            world = attach_source_discourse(text, world, package)
        else:
            world, notes = builder(text, slots)
        world = ensure_schema(world)
    clauses = [
        {"clause_id": row["clause_id"], "text": row["text"]}
        for row in segment_source_clauses(text)
    ]
    apply_derivation_license(world, clauses, proof)
    apply_kind_license(world, clauses, winner["blueprint_id"])
    for effect in world.get("effects", []):
        if effect.get("derivation_operation") == "EXCLUSIVE_ALLOCATION_COMPLEMENT":
            # Complements require an explicit exclusivity proof. Dose lists
            # cannot discharge those premises.
            effect["derivation_assumptions"] = []
    relation_alternatives = _relation_alternatives(world, clauses)
    graph = assemble_evidence_graph(
        package=package,
        text=text,
        clauses=clauses,
        accepted=winner.get("accepted_evidence") or {},
        recoveries=winner.get("semantic_recoveries") or {},
        slots=slots,
        world=world,
    )
    slot_bindings = {
        "copied_spans": dict(slots),
        "z10_recovered_slots": dict(winner.get("semantic_recoveries") or {}),
    }
    if winner["blueprint_id"] == "exclusive_allocation":
        match = match_exclusive_allocation(
            package,
            [row["intervention"] for row in world["actions"]],
            evidence_graph=graph,
        )
        slot_bindings["exclusive_allocation_match"] = {
            "matched": match["matched"],
            "unfilled_required_slots": match["unfilled_required_slots"],
            "quantity_candidate_ids": match["quantity_candidate_ids"],
        }
    assignment = assignment_for(world)
    return normalized_proposal(
        proposal_id=f"{winner['blueprint_id']}_cloze",
        blueprint_id=winner["blueprint_id"],
        status="FILLED",
        assignment_kind="intervention_text",
        assignment=assignment,
        slot_bindings=slot_bindings,
        selection=None,
        selection_validation=source_copy_validation(),
        candidate_value=proposal_candidate(
            {
                row["action_id"]: {
                    "clause_ids": list(row["clause_ids"]),
                    "reason": "Filled from copied cloze slots.",
                }
                for row in world["actions"]
            },
            world,
        ),
        clauses=clauses,
        unfilled_required_slots=[
            row["id"] for row in winner["items"]
            if row["core"] and row["verdict"] != "accepted"
        ],
        unresolved_readings=_cloze_unresolved_readings(winner),
        construction_problems=_cloze_construction_problems(winner, world),
        admission_authorized=True,
        notes=notes,
        pre_world_assessment={
            **question,
            "status": "ASSESSED",
        },
        accepted_evidence=dict(winner["accepted_evidence"]),
        construction_provenance=_construction_provenance(world, clauses, slots, proof),
        relation_alternatives=relation_alternatives,
        exclusivity_proof=proof,
        evidence_graph=graph,
    )


def _withheld_cloze_proposal(text: str, winner: dict[str, Any],
                             reasons: list[str]) -> dict[str, Any]:
    clauses = [
        {"clause_id": row["clause_id"], "text": row["text"]}
        for row in segment_source_clauses(text)
    ]
    return withheld_proposal(
        proposal_id=f"{winner['blueprint_id']}_cloze",
        blueprint_id=winner["blueprint_id"],
        assignment=[],
        slot_bindings={"copied_spans": dict(winner["slots"])},
        clauses=clauses,
        unfilled_required_slots=[
            row["id"] for row in winner["items"]
            if row["core"] and row["verdict"] != "accepted"
        ],
        unresolved_readings=_cloze_unresolved_readings(winner),
        construction_problems=[
            {"code": "world_withheld", "message": reason} for reason in reasons
        ] or [{"code": "incomplete_template", "message": "Required slots remain unfilled."}],
        pre_world_assessment={
            **assess_question(text),
            "status": "WITHHELD",
            "eligible_for_world_state": False,
        },
        accepted_evidence=dict(winner["accepted_evidence"]),
        construction_provenance=_withheld_provenance(winner),
        exclusivity_proof=(assess_question(text).get("exclusivity_proof")),
    )


def _construction_provenance(world: dict[str, Any], clauses: list[dict[str, str]],
                             slots: dict[str, str],
                             proof: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Describe construction outside Parliament's closed world-state records."""
    rows: list[dict[str, Any]] = []
    for collection, id_key, kind in (
        ("parties", "party_id", "party"),
        ("actions", "action_id", "action"),
        ("effects", "effect_id", "effect"),
        ("conditions", "condition_id", "condition"),
    ):
        for atom in world.get(collection, []):
            operation = atom.get("derivation_operation", "DIRECT_COPY")
            if kind == "effect":
                origin = origin_for_effect(
                    atom, clauses, world.get("parties") or [], proof)
            elif kind == "party":
                origin = atom.get("kind_origin") or "UNRESOLVED"
            else:
                origin = (
                    "SOURCE_ASSERTED"
                    if operation in {
                        "DIRECT_COPY", "UNSPECIFIED", "SOURCE_STIPULATED_CAUSAL", "",
                    }
                    else "STRUCTURALLY_DERIVED"
                )
            rows.append({
                "atom_id": f"{kind}:{atom[id_key]}",
                "atom_kind": kind,
                "origin": origin,
                "clause_ids": list(atom.get("clause_ids") or []),
                "evidence": [atom.get("source_proposition") or atom.get("label")
                             or atom.get("intervention") or atom.get("description")],
                "explanation": operation,
            })
    clause_text = {row["clause_id"]: row["text"] for row in clauses}
    for index, link in enumerate(world.get("causal_links", [])):
        cue = _explicit_relation_cue(" ".join(
            clause_text.get(cid, "") for cid in link.get("clause_ids", [])))
        rows.append({
            "atom_id": f"relation:{index}", "atom_kind": "relation",
            "origin": "SOURCE_ASSERTED" if cue else "UNRESOLVED",
            "clause_ids": list(link.get("clause_ids") or []),
            "evidence": [cue] if cue else [],
            "explanation": (
                "Explicit relation cue copied from the source."
                if cue else "A conditional association requires a working Parliament relation."
            ),
        })
    for name, value in slots.items():
        if name.endswith("implied_process") and value:
            rows.append({
                "atom_id": f"hypothesis:{name}", "atom_kind": "hypothesis",
                "origin": "WORLD_KNOWLEDGE_HYPOTHESIS", "clause_ids": [],
                "evidence": [value],
                "explanation": "Model-suggested process; excluded from the Parliament world.",
            })
    rows.extend(discourse_provenance_rows(world))
    return rows


def _withheld_provenance(winner: dict[str, Any]) -> list[dict[str, Any]]:
    return [{
        "atom_id": f"slot:{name}", "atom_kind": "slot",
        "origin": "SOURCE_ASSERTED", "clause_ids": [], "evidence": [value],
        "explanation": "Copied cloze evidence; no world atom was authorized.",
    } for name, value in winner.get("accepted_evidence", {}).items()]


def _relation_alternatives(world: dict[str, Any],
                           clauses: list[dict[str, str]]) -> list[dict[str, Any]]:
    clause_text = {row["clause_id"]: row["text"] for row in clauses}
    rows = []
    for index, link in enumerate(world.get("causal_links", [])):
        text = " ".join(clause_text.get(cid, "") for cid in link.get("clause_ids", []))
        cue = _explicit_relation_cue(text)
        selected = link.get("link_relation") or link.get("relation")
        alternatives = [selected] if cue else list(dict.fromkeys([selected, "CAUSES"]))
        conditional = bool(re.search(r"\b(?:if|unless|whether|who\s+(?:does|do)\s+not)\b", text, re.I))
        rows.append({
            "relation_id": f"relation:{index}",
            "source_id": link["source_id"], "target_id": link["target_id"],
            "selected_for_parliament": selected,
            "alternatives": alternatives,
            "status": "EXPLICIT" if cue else "UNRESOLVED",
            "evidence": [cue] if cue else [],
            "condition_ids": list(link.get("condition_ids") or []),
            "structural_basis": "conditional_parent" if conditional else "associated_parent",
        })
    return rows


def _cloze_unresolved_readings(winner: dict[str, Any]) -> list[dict[str, Any]]:
    readings = []
    if winner["blueprint_id"] == "conditional_outcome":
        readings.append({
            "kind": "conditional_relation",
            "alternatives": ["CAUSES", "ENABLES"],
        })
    elif winner["blueprint_id"] == "ability_permission":
        readings.append({
            "kind": "modal_force",
            "alternatives": ["ability", "permission", "possibility"],
        })
    elif winner["blueprint_id"] == "deontic_rule":
        readings.append({"kind": "normative_force", "status": "governed_not_occurrence"})
    elif winner["blueprint_id"] == "promise_reliance":
        readings.append({"kind": "commitment_status", "status": "not_world_occurrence"})
    elif winner["blueprint_id"] == "disputed_report":
        readings.append({"kind": "reported_truth", "status": "unresolved"})
    return readings


def _cloze_construction_problems(
        winner: dict[str, Any],
        world: dict[str, Any] | None = None) -> list[dict[str, str]]:
    problems: list[dict[str, str]] = []
    if world:
        problems.extend(kind_problems(world))
    if winner["blueprint_id"] != "rescue_contrast":
        return problems
    harms = [
        name for name in ("first_harm", "second_harm")
        if winner["slots"].get(name)
    ]
    if not harms:
        return problems
    problems.append({
        "code": "rescue_harm_missing_named_process",
        "message": (
            "The branch states a cross-party harm but names no physical process "
            "between the rescue action and that harm."
        ),
    })
    return problems


def _allocation_graph(text: str, slots: dict[str, str],
                      exclusivity_proof: dict[str, Any] | None = None) -> tuple[dict, list[str]]:
    parties: list[dict[str, Any]] = []
    index = {"n": 0}
    issued = {"n": 0}

    def party(label: str, role: str = "recipient",
              quantities: list[str] | None = None) -> str:
        return _party(
            text, parties, index, label, quantities=quantities,
            role=role, construction="exclusive_allocation")

    def effect_id() -> str:
        issued["n"] += 1
        return f"E{issued['n']}"

    actor = party(slots["decider"], role="actor")
    resource = party(
        slots["resource"], role="resource",
        quantities=[_quantity_token(slots["quantity"])] if slots.get("quantity") else [])
    licensed_resource = next(
        (row for row in parties if row["party_id"] == resource), {"kind": "OTHER"})
    actions, effects, links, conditions = [], [], [], []
    notes = []
    rows = (
        (slots["first_recipient"], slots.get("first_outcome"), slots.get("first_transfer"),
         slots.get("first_hedge"), slots.get("first_branch_sentence")),
        (slots["second_recipient"], slots.get("second_outcome"), slots.get("second_transfer"),
         slots.get("second_hedge"), slots.get("second_branch_sentence")),
    )
    for branch, (recipient_label, outcome, transfer, hedge, branch_sentence) in enumerate(rows):
        recipient = party(
            recipient_label, quantities=group_quantity(_nominal_before_clause(recipient_label)))
        action_id = f"A{branch}"
        recovered_transfer = _branch_transfer(
            text, slots["decider"], branch_sentence or outcome or "",
            recipient_label)
        copied_transfer = transfer or recovered_transfer or (
            slots["assignment"] if _TRANSFER_EVENT.search(slots["assignment"]) else None)
        claimed_kind = "RESOURCE_TRANSFER" if copied_transfer else "INTERVENTION"
        licensed_kind = license_effect_kind(
            claimed_kind,
            span=copied_transfer or slots["assignment"],
            construction="exclusive_allocation",
            parties=parties,
            predicate=licensed_source_predicate(copied_transfer or slots["assignment"]),
        )
        if copied_transfer:
            direct_outcome, direct_kind = copied_transfer, licensed_kind["kind"]
            # Keep the branch surface as the intervention, but bind the direct
            # transfer state to the named resource when a conditional uses an
            # unresolved pronoun ("administers it to Ana").  The shared
            # assignment is itself an exact source copy; no referent is invented.
            if (re.search(r"\b(?:it|this|that)\b", copied_transfer, re.I)
                    and _shares_word(slots["assignment"], slots["resource"])
                    and _TRANSFER_EVENT.search(slots["assignment"])):
                direct_outcome = slots["assignment"]
            if direct_kind != "RESOURCE_TRANSFER":
                notes.append(
                    f"{action_id} has a giving or receipt span, but no licensed "
                    "RESOURCE, so the act stays an intervention.")
        else:
            direct_outcome, direct_kind = slots["assignment"], "INTERVENTION"
            notes.append(f"{action_id} has no source giving or receipt event, so the act stays an intervention.")
        intervention = _branch_intervention(
            text, slots["assignment"], recipient_label,
            transfer or "", recovered_transfer)
        if intervention not in text:
            notes.append(f"{action_id} joins two copied spans that are not adjacent.")
        host = _clause_for(text, branch_sentence or outcome or recipient_label)
        act_host = _clause_for(text, direct_outcome)
        direct_id = effect_id()
        direct_effect = _effect(
            direct_id, action_id, recipient, direct_outcome, _predicate(direct_outcome),
            "NEUTRAL", "DIRECT", direct_kind, "CERTAIN", act_host, [], [],
            source=direct_outcome)
        # The shared either/or sentence identifies the resource; the branch
        # condition identifies the recipient-specific transfer.  Preserve both
        # source clauses when those facts are distributed across the text.
        direct_effect["clause_ids"] = list(dict.fromkeys(
            direct_effect["clause_ids"] + [host["clause_id"]]))
        effects.append(direct_effect)
        effect_ids = [direct_id]
        parent_id = direct_id
        if outcome:
            reading = _outcome_reading(outcome)
            modality, qualifiers = _checked_modality(outcome, hedge)
            bearer_label, quantities = _outcome_bearer(outcome, recipient_label)
            bearer = party(bearer_label, quantities=quantities)
            outcome_id = effect_id()
            outcome_host = _clause_for(text, outcome)
            condition_ids: list[str] = []
            if modality == "POSSIBLE":
                condition_ids = [f"CND{len(conditions) + 1}"]
                conditions.append({
                    "condition_id": condition_ids[0], "description": hedge or "",
                    "value_status": "UNKNOWN", "decision_relevance": "MATERIAL",
                    "clause_ids": [outcome_host["clause_id"]], "polarity": "POSITIVE",
                    "operator": "IF",
                })
            row = _effect(
                outcome_id, action_id, bearer, outcome, reading["predicate"],
                reading["polarity"], "DOWNSTREAM", reading["kind"], modality,
                outcome_host, [parent_id], qualifiers, source=outcome)
            row["condition_ids"] = condition_ids
            effects.append(row)
            effect_ids.append(outcome_id)
            # The source stipulates the action/outcome relation. Probability is
            # carried by the outcome, while the relation itself is certain.
            link = _link(action_id, parent_id, outcome_id, "CERTAIN", outcome_host)
            link["condition_ids"] = condition_ids
            if condition_ids:
                link["modality"] = modality
            links.append(link)
        actions.append({
            "action_id": action_id, "intervention": intervention,
            "actor_party_id": actor,
            "recipient_party_ids": [recipient], "effect_ids": effect_ids,
            "clause_ids": list(dict.fromkeys([act_host["clause_id"], host["clause_id"]])),
        })
    complement_by_action: dict[str, str] = {}
    proof_status = (exclusivity_proof or {}).get("status", "UNKNOWN")
    exclusive = bool(slots.get("exclusivity") or proof_status in {"EXPLICIT", "DERIVED"})
    if slots.get("quantity") and licensed_resource.get("kind") == "RESOURCE" and exclusive:
        quantity_host = _clause_for(text, slots["quantity"])
        for branch, other in ((0, slots["second_recipient"]), (1, slots["first_recipient"])):
            action_id = f"A{branch}"
            other_id = party(other)
            complement_id = effect_id()
            complement = _effect(
                complement_id, action_id, other_id,
                f"does not receive {slots['resource']}", "NOT_RECEIVES",
                "ADVERSE", "DOWNSTREAM", "OTHER", "CERTAIN", quantity_host,
                [actions[branch]["effect_ids"][0]], [],
                source=quantity_host["text"],
            )
            complement["quantities"] = [_quantity_token(slots["quantity"])]
            complement["derivation_operation"] = "EXCLUSIVE_ALLOCATION_COMPLEMENT"
            complement["derivation_explanation"] = (
                f"Only {slots['quantity']} exists, and the "
                f"{proof_status.lower()} exclusivity proof precludes simultaneous "
                "receipt by the other recipient."
            )
            # Keep local quantity provenance on the indivisible resource. The
            # exclusivity sentence names both recipient populations, so citing
            # it here would leak the rival branch's headcount into this effect.
            # Parliament's complement compiler independently binds the global
            # exclusivity evidence and retains that constraint provenance.
            complement["clause_ids"] = complement_clause_ids(
                [quantity_host["clause_id"]])
            effects.append(complement)
            actions[branch]["effect_ids"].append(complement_id)
            complement_by_action[action_id] = complement_id
    if slots.get("nonreceipt") and licensed_resource.get("kind") == "RESOURCE" and exclusive:
        host = _clause_for(text, slots["nonreceipt"])
        # "do not get ... will die" keeps the negation on receiving. The death
        # itself is the "will die" span, so polarity is not read as a negated harm.
        death_copies = _copy_spans(text, "will die")[0]
        death_outcome = death_copies[0] if death_copies and "will die" in slots["nonreceipt"].casefold() else slots["nonreceipt"]
        state_span = slots["nonreceipt"]
        if death_copies and death_copies[0].casefold() in state_span.casefold():
            cut = state_span.casefold().rfind(death_copies[0].casefold())
            state_span = state_span[:cut].strip(" ,") or slots["nonreceipt"]
        for branch, other in ((0, slots["second_recipient"]), (1, slots["first_recipient"])):
            other_id = party(other)
            action_id = f"A{branch}"
            parent = complement_by_action.get(action_id)
            if parent is None:
                # Copied nonreceipt is not a quantity-derived complement.
                # Parliament requires an indivisibility source for that
                # derivation; this state stays a source copy of nonreceipt
                # parented on a resource-bearer copy, not immediately on DIRECT.
                direct_id = actions[branch]["effect_ids"][0]
                resource_id = effect_id()
                resource_state = _effect(
                    resource_id, action_id, resource, state_span,
                    _predicate(state_span), "NEUTRAL", "DOWNSTREAM",
                    "OTHER", "CERTAIN", host, [direct_id], [],
                    source=state_span,
                )
                effects.append(resource_state)
                actions[branch]["effect_ids"].append(resource_id)
                parent = effect_id()
                state = _effect(
                    parent, action_id, other_id, state_span, "NOT_RECEIVES",
                    "ADVERSE", "DOWNSTREAM", "OTHER", "CERTAIN", host,
                    [resource_id], [],
                    source=state_span,
                )
                effects.append(state)
                actions[branch]["effect_ids"].append(parent)
                complement_by_action[action_id] = parent
                links.append(_link(action_id, direct_id, resource_id, "CERTAIN", host))
                links.append(_link(action_id, resource_id, parent, "CERTAIN", host))
            death_id = effect_id()
            effects.append(_effect(
                death_id, f"A{branch}", other_id, death_outcome, "die",
                "ADVERSE", "DOWNSTREAM", "HEALTH_OUTCOME", "CERTAIN",
                host, [parent], [], source=death_outcome))
            actions[branch]["effect_ids"].append(death_id)
            links.append(_link(f"A{branch}", parent, death_id, "CERTAIN", host))
    _note_shared_clause(actions, notes)
    for label, implied in (
            ("first outcome", slots.get("first_implied_process")),
            ("second outcome", slots.get("second_implied_process")),
    ):
        if implied:
            notes.append(
                f"Implied process for the {label}: {implied}. "
                "This is a warning, not a cause copied from the text."
            )
    return _world(parties, actions, effects, links, conditions), notes


def _rescue_graph(text: str, slots: dict[str, str]) -> tuple[dict, list[str]]:
    parties: list[dict[str, Any]] = []
    index = {"n": 0}
    actor = _party(
        text, parties, index, slots["rescuer"], role="actor",
        construction="rescue_contrast")
    actions, effects, links, conditions = [], [], [], []
    notes = []
    if slots.get("scene"):
        notes.append(f"Copied danger scene: {slots['scene']}.")
    exclusivity_host = _clause_for(text, slots["rescue_exclusivity"])
    branches = (
        (slots["first_saved"], slots["first_rescue_action"],
         slots["first_benefit"], slots.get("first_harm")),
        (slots["second_saved"], slots["second_rescue_action"],
         slots["second_benefit"], slots.get("second_harm")),
    )
    issued = 0
    for branch, (saved_label, intervention, benefit, harm) in enumerate(branches):
        action_id = f"A{branch}"
        saved = _party(
            text, parties, index, saved_label, group_quantity(saved_label),
            role="saved", construction="rescue_contrast")
        action_host = _clause_for(text, intervention)
        issued += 1
        direct_id = f"E{issued}"
        effects.append(_effect(
            direct_id, action_id, saved, intervention, "save", "NEUTRAL", "DIRECT",
            "INTERVENTION", "CERTAIN", action_host, [], [], source=intervention))
        effect_ids = [direct_id]
        branch_clauses = [exclusivity_host["clause_id"], action_host["clause_id"]]
        for outcome in (benefit, harm):
            if not outcome:
                continue
            reading = _outcome_reading(outcome)
            bearer_label = _party_head(text, outcome)
            bearer = _party(
                text, parties, index, bearer_label, group_quantity(bearer_label),
                role="bearer", construction="rescue_contrast")
            outcome_host = _clause_for(text, outcome)
            issued += 1
            outcome_id = f"E{issued}"
            effects.append(_effect(
                outcome_id, action_id, bearer, outcome, reading["predicate"],
                reading["polarity"], "DOWNSTREAM", reading["kind"],
                reading["modality"], outcome_host, [direct_id],
                reading["qualifiers"], source=outcome))
            links.append(_link(
                action_id, direct_id, outcome_id, reading["modality"], outcome_host))
            effect_ids.append(outcome_id)
            branch_clauses.append(outcome_host["clause_id"])
        actions.append({
            "action_id": action_id,
            "intervention": intervention,
            "actor_party_id": actor,
            "recipient_party_ids": [saved],
            "effect_ids": effect_ids,
            "clause_ids": list(dict.fromkeys(branch_clauses)),
        })
    return _world(parties, actions, effects, links), notes


def _omission_graph(text: str, slots: dict[str, str]) -> tuple[dict, list[str]]:
    parties: list[dict[str, Any]] = []
    index = {"n": 0}
    actor = _party(
        text, parties, index, slots["actor"], role="actor",
        construction="omission_harm")
    actions, effects, links, conditions = [], [], [], []
    pairs = ((slots["done"], slots["harm_done"], slots.get("done_hedge")),
             (slots["omitted"], slots["harm_omitted"], slots.get("omitted_hedge")))
    for branch, (action_span, harm_span, hedge) in enumerate(pairs):
        reading = _outcome_reading(harm_span)
        copied_bearer = _party_head(text, harm_span)
        bearer_label, quantities = _outcome_bearer(harm_span, copied_bearer)
        bearer = _party(
            text, parties, index, bearer_label,
            quantities or ([_quantity_token(bearer_label)]
                           if _quantity_token(bearer_label) else []),
            role="bearer", construction="omission_harm")
        direct_host = _clause_for(text, action_span)
        outcome_host = _clause_for(text, harm_span)
        modality, qualifiers = _checked_modality(harm_span, hedge)
        if _stipulated_certain_outcome(harm_span):
            # The if-clause is the action branch. "will die" on that branch is
            # stipulated, not an unresolved possibility.
            modality, qualifiers = "CERTAIN", []
        action_id = f"A{branch}"
        direct_id = f"E{len(effects) + 1}"
        condition_id = f"CND{branch + 1}"
        condition_text = hedge or action_span
        conditions.append({
            "condition_id": condition_id,
            "description": condition_text,
            "value_status": "UNKNOWN",
            "decision_relevance": "MATERIAL",
            "clause_ids": [outcome_host["clause_id"]],
            "polarity": "NEGATED" if re.search(r"\bnot\b|n't", condition_text, re.I)
            else "POSITIVE",
            "operator": "IF",
        })
        process_label = _action_object(text, action_span)
        direct_party = (_party(
            text, parties, index, process_label, role="process",
            construction="omission_harm", action_span=action_span)
                        if process_label else actor)
        process_kind = license_effect_kind(
            "PHYSICAL_STATE", span=action_span, construction="omission_harm",
            parties=parties, predicate=_predicate(action_span),
        )["kind"] if process_label else "INTERVENTION"
        effects.append(_effect(
            direct_id, action_id, direct_party, action_span, _predicate(action_span),
            "NEUTRAL", "DIRECT", "INTERVENTION", "CERTAIN", direct_host, [], [],
            source=action_span))
        parent_id = direct_id
        effect_ids = [direct_id]
        if process_label:
            process_id = f"E{len(effects) + 1}"
            effects.append(_effect(
                process_id, action_id, direct_party, action_span,
                _predicate(action_span), "NEUTRAL", "DOWNSTREAM",
                process_kind, "CERTAIN", direct_host, [direct_id], [],
                source=action_span))
            links.append(_link(
                action_id, direct_id, process_id, "CERTAIN", direct_host))
            parent_id = process_id
            effect_ids.append(process_id)
        outcome_id = f"E{len(effects) + 1}"
        death = _effect(
            outcome_id, action_id, bearer, harm_span, reading["predicate"],
            reading["polarity"], "DOWNSTREAM", reading["kind"], modality,
            outcome_host, [parent_id], qualifiers, source=harm_span)
        death["quantities"] = group_quantity(harm_span) or (
            [_quantity_token(harm_span)] if _quantity_token(harm_span) else [])
        effects.append(death)
        link = _link(action_id, parent_id, outcome_id, modality, outcome_host)
        links.append(link)
        effect_ids.append(outcome_id)
        clause_ids = list(dict.fromkeys([direct_host["clause_id"], outcome_host["clause_id"]]))
        actions.append({"action_id": action_id, "intervention": action_span,
                        "actor_party_id": actor, "recipient_party_ids": [direct_party],
                        "effect_ids": effect_ids, "clause_ids": clause_ids})
    notes = []
    if slots.get("instrument"):
        notes.append(f"Instrument named in the text: {slots['instrument']}. It is not an action.")
    return _world(parties, actions, effects, links, conditions), notes


def _condition_action(condition: str, actor: str) -> str:
    """Return the copied action content of an if/unless clause."""
    action = re.sub(r"^(?:if|unless)\s+", "", condition.strip(" ,."), flags=re.I)
    if actor:
        action = re.sub(rf"^{re.escape(actor)}\s+", "", action, flags=re.I)
    return action.strip(" ,.") or condition


def _action_object(text: str, action: str) -> str:
    """Return a copied object that can bear an intervention/process state."""
    try:
        import parsing_game_S as parsing
        parsed = parsing.get_nlp()(action)
        objects = [token for token in parsed
                   if token.dep_ in {"dobj", "obj", "pobj", "attr", "nsubjpass"}
                   and token.pos_ in {"NOUN", "PROPN", "PRON"}]
        if not objects:
            return ""
        token = objects[0]
        parts = [part for part in token.subtree if not part.is_punct]
        start = min(part.idx for part in parts)
        end = max(part.idx + len(part.text) for part in parts)
        local = action[start:end].strip(" ,.")
        copies = _copy_spans(text, local)[0]
        return copies[0] if copies else ""
    except (OSError, RuntimeError, ValueError):
        return ""


def _conditional_graph(text: str, slots: dict[str, str]) -> tuple[dict, list[str]]:
    parties: list[dict[str, Any]] = []
    index = {"n": 0}
    actor_label = slots.get("actor") or slots["bearer"]
    actor = _party(
        text, parties, index, actor_label, role="actor",
        construction="conditional_outcome")
    rows = [(slots["condition"], slots["bearer"], slots["outcome"])]
    if all(slots.get(name) for name in ("second_condition", "second_bearer", "second_outcome")):
        rows.append((slots["second_condition"], slots["second_bearer"], slots["second_outcome"]))
    effects, links, actions, conditions = [], [], [], []
    notes = []
    for branch, (condition, bearer_label, outcome) in enumerate(rows):
        bearer = _party(
            text, parties, index, bearer_label, group_quantity(bearer_label),
            role="bearer", construction="conditional_outcome")
        reading = _outcome_reading(outcome)
        host = _clause_for(text, outcome, condition)
        action_text = _condition_action(condition, actor_label)
        action_host = _clause_for(text, action_text)
        chance = slots.get("chance") if slots.get("chance") and slots["chance"].casefold() in host["text"].casefold() else None
        modality, qualifiers = _checked_modality(outcome, chance or "if")
        if _stipulated_certain_outcome(outcome) and not chance:
            modality, qualifiers = "CERTAIN", []
        action_id = f"A{branch}"
        direct_id = f"E{len(effects) + 1}"
        condition_id = f"CND{branch + 1}"
        conditions.append({
            "condition_id": condition_id,
            "description": condition,
            "value_status": "UNKNOWN",
            "decision_relevance": "MATERIAL",
            "clause_ids": [host["clause_id"]],
            "polarity": "NEGATED" if re.search(r"\bnot\b|n't", condition, re.I)
            else "POSITIVE",
            "operator": "IF",
        })
        transfer = bool(_TRANSFER_EVENT.search(action_text))
        process_label = "" if transfer else _action_object(text, action_text)
        if process_label:
            direct_party = _party(
                text, parties, index, process_label, role="process",
                construction="conditional_outcome", action_span=action_text)
            direct_kind = "INTERVENTION"
        elif transfer:
            claimed = license_effect_kind(
                "RESOURCE_TRANSFER", span=action_text,
                construction="conditional_outcome", parties=parties,
                predicate=_predicate(action_text),
            )
            direct_party = bearer
            direct_kind = claimed["kind"]
        else:
            direct_party, direct_kind = actor, "INTERVENTION"
        direct = _effect(
            direct_id, action_id, direct_party, action_text, _predicate(action_text),
            "NEUTRAL", "DIRECT", direct_kind, "CERTAIN", action_host, [], [],
            source=action_text)
        effects.append(direct)
        parent_id = direct_id
        effect_ids = [direct_id]
        if process_label:
            process_id = f"E{len(effects) + 1}"
            process = _effect(
                process_id, action_id, direct_party, action_text,
                _predicate(action_text), "NEUTRAL", "DOWNSTREAM",
                license_effect_kind(
                    "PHYSICAL_STATE", span=action_text,
                    construction="conditional_outcome", parties=parties,
                    predicate=_predicate(action_text),
                )["kind"],
                "CERTAIN", action_host, [direct_id], [],
                source=action_text)
            effects.append(process)
            links.append(_link(
                action_id, direct_id, process_id, "CERTAIN", action_host))
            parent_id = process_id
            effect_ids.append(process_id)
        outcome_id = f"E{len(effects) + 1}"
        result = _effect(
            outcome_id, action_id, bearer, outcome, reading["predicate"], reading["polarity"],
            "DOWNSTREAM", reading["kind"], modality, host, [parent_id], qualifiers,
            source=outcome)
        result["quantities"] = group_quantity(outcome) or (
            [_quantity_token(outcome)] if _quantity_token(outcome) else [])
        effects.append(result)
        link = _link(action_id, parent_id, outcome_id, modality, host)
        links.append(link)
        effect_ids.append(outcome_id)
        actions.append({
            "action_id": action_id, "intervention": action_text,
            "actor_party_id": actor, "recipient_party_ids": [direct_party],
            "effect_ids": effect_ids,
            "clause_ids": list(dict.fromkeys([
                action_host["clause_id"], host["clause_id"],
            ])),
        })
    partial_second = [
        name for name in ("second_condition", "second_bearer", "second_outcome")
        if slots.get(name)
    ]
    if partial_second and len(partial_second) != 3:
        notes.append(
            "A partial second conditional remains unresolved: " + ", ".join(partial_second) + "."
        )
    return _world(parties, actions, effects, links, conditions), notes


def _diversion_graph(text: str, slots: dict[str, str]) -> tuple[dict, list[str]]:
    parties: list[dict[str, Any]] = []
    index = {"n": 0}
    actor = _party(
        text, parties, index, slots["actor"], role="actor",
        construction="diversion_redirection")
    process = _party(
        text, parties, index, slots["controllable_process"], role="process",
        construction="diversion_redirection")
    affected = _party(
        text, parties, index, slots["affected_party"],
        group_quantity(slots["affected_party"]),
        role="affected", construction="diversion_redirection")
    action_host = _clause_for(text, slots["intervention"])
    outcome_host = _clause_for(text, slots["outcome"])
    reading = _outcome_reading(slots["outcome"])
    modality, qualifiers = _risk_modality(
        slots["outcome"], slots.get("uncertainty"))
    direct = _effect(
        "E1", "A0", process, slots["intervention"],
        _predicate(slots["intervention"]), "NEUTRAL", "DIRECT", "INTERVENTION",
        "CERTAIN", action_host, [], [], source=slots["intervention"])
    process_state = _effect(
        "E2", "A0", process, slots["intervention"],
        _predicate(slots["intervention"]),
        "NEUTRAL", "DOWNSTREAM",
        license_effect_kind(
            "PHYSICAL_STATE", span=slots["intervention"],
            construction="diversion_redirection", parties=parties,
            predicate=_predicate(slots["intervention"]),
        )["kind"],
        "CERTAIN", action_host,
        ["E1"], [], source=slots["intervention"])
    outcome = _effect(
        "E3", "A0", affected, slots["outcome"], reading["predicate"],
        reading["polarity"], "DOWNSTREAM", reading["kind"], modality,
        outcome_host, ["E2"], qualifiers, source=slots["outcome"])
    quantity = _quantity_token(slots["outcome"])
    if quantity and quantity not in outcome["quantities"]:
        outcome["quantities"].append(quantity)
    action = {
        "action_id": "A0", "intervention": slots["intervention"],
        "actor_party_id": actor, "recipient_party_ids": [process],
        "effect_ids": ["E1", "E2", "E3"],
        "clause_ids": list(dict.fromkeys([
            action_host["clause_id"], outcome_host["clause_id"],
        ])),
    }
    notes = []
    if slots.get("alternative_route"):
        notes.append(
            "Copied alternative route remains context; no default-route effect is inferred: "
            + slots["alternative_route"])
    if slots.get("omission_branch"):
        notes.append(
            "Copied omission branch needs its own complete outcome before becoming an action: "
            + slots["omission_branch"])
    return _world(
        parties, [action], [direct, process_state, outcome],
        [
            _link("A0", "E1", "E2", "CERTAIN", action_host),
            _link("A0", "E2", "E3", modality, outcome_host),
        ]), notes


def _risk_graph(text: str, slots: dict[str, str]) -> tuple[dict, list[str]]:
    parties: list[dict[str, Any]] = []
    index = {"n": 0}
    actor = _party(
        text, parties, index, slots["actor"], role="actor",
        construction="uncertain_risk")
    branches = [(
        slots["action"], slots["affected_party"], slots["possible_outcome"],
        slots["likelihood"],
    )]
    if all(slots.get(name) for name in (
            "second_action", "second_affected_party", "second_outcome",
            "second_likelihood")):
        branches.append((
            slots["second_action"], slots["second_affected_party"],
            slots["second_outcome"], slots["second_likelihood"],
        ))
    actions, effects, links, conditions = [], [], [], []
    for branch, (action_text, affected_label, outcome_text, likelihood) in enumerate(branches):
        action_id = f"A{branch}"
        affected = _party(
            text, parties, index, affected_label, group_quantity(affected_label),
            role="affected", construction="uncertain_risk")
        action_host = _clause_for(text, action_text)
        outcome_host = _clause_for(text, outcome_text)
        reading = _outcome_reading(outcome_text)
        modality, qualifiers = _risk_modality(outcome_text, likelihood)
        direct_id, outcome_id = f"E{branch * 2 + 1}", f"E{branch * 2 + 2}"
        effects.append(_effect(
            direct_id, action_id, affected, action_text, _predicate(action_text),
            "NEUTRAL", "DIRECT", "INTERVENTION", "CERTAIN", action_host,
            [], [], source=action_text))
        outcome_effect = _effect(
            outcome_id, action_id, affected, outcome_text, reading["predicate"],
            reading["polarity"], "DOWNSTREAM", reading["kind"], modality,
            outcome_host, [direct_id], qualifiers, source=outcome_text)
        link = _link(action_id, direct_id, outcome_id, modality, outcome_host)
        if modality == "POSSIBLE":
            condition_id = f"CND{branch + 1}"
            conditions.append({
                "condition_id": condition_id,
                "description": likelihood,
                "value_status": "UNKNOWN",
                "decision_relevance": "MATERIAL",
                "clause_ids": [outcome_host["clause_id"]],
                "polarity": "POSITIVE",
                "operator": "IF",
            })
            outcome_effect["condition_ids"] = [condition_id]
            link["condition_ids"] = [condition_id]
        effects.append(outcome_effect)
        links.append(link)
        actions.append({
            "action_id": action_id, "intervention": action_text,
            "actor_party_id": actor, "recipient_party_ids": [affected],
            "effect_ids": [direct_id, outcome_id],
            "clause_ids": list(dict.fromkeys([
                action_host["clause_id"], outcome_host["clause_id"],
            ])),
        })
    notes = []
    if len(branches) == 1 and any(slots.get(name) for name in (
            "second_action", "second_affected_party", "second_outcome",
            "second_likelihood")):
        notes.append("The second risk branch is partial and remains unresolved.")
    return _world(parties, actions, effects, links, conditions), notes


def _risk_modality(outcome: str, likelihood: str | None) -> tuple[str, list[str]]:
    chance = _CHANCE.search(likelihood or "") or _CHANCE.search(outcome or "")
    if chance:
        return "PROBABILISTIC", [chance.group(0)]
    if re.search(
            r"\b(?:if|unless|whether|may|might|could|possibly|probably|"
            r"unlikely|likely)\b", likelihood or "", re.I):
        return "POSSIBLE", []
    if re.search(r"\bwill\b", outcome or "", re.I):
        return "CERTAIN", []
    return "UNKNOWN", []


def _world(parties: list[dict], actions: list[dict], effects: list[dict],
           links: list[dict], conditions: list[dict] | None = None) -> dict[str, Any]:
    return ensure_schema({
        "parties": parties, "actions": actions, "effects": effects,
        "conditions": conditions or [], "temporal_relations": [],
        "causal_links": links, "counterfactual_links": [],
    })


def _effect(effect_id: str, action_id: str, party_id: str, outcome: str, predicate: str,
            polarity: str, directness: str, kind: str, modality: str, host: dict,
            parents: list[str], qualifiers: list[str], source: str | None = None) -> dict[str, Any]:
    proposition = source or host["text"]
    return {
        "effect_id": effect_id, "action_id": action_id, "party_id": party_id,
        "outcome": outcome, "predicate": predicate, "polarity": polarity,
        "directness": directness, "modality": modality, "effect_kind": kind,
        "condition_ids": [], "quantities": group_quantity(proposition),
        "likelihood_qualifiers": qualifiers, "overall_likelihood_qualifiers": [],
        "scope_qualifiers": [], "temporal_qualifiers": [], "condition_join": "AND",
        "source_proposition": proposition, "source_effect_ids": parents,
        # Parenthood only proposes stipulated-causal. The proposal envelope
        # licenses asserted copies and demotes polarity inversions.
        "derivation_operation": "DIRECT_COPY" if not parents else "SOURCE_STIPULATED_CAUSAL",
        "derivation_explanation": "The blank was filled with this copy of the scenario.",
        "derivation_assumptions": [], "outcome_type_transformation": "PRESERVED",
        "clause_ids": [host["clause_id"]],
    }


def _explicit_relation_cue(span: str) -> str:
    patterns = (
        (r"\b(?:cause|causes|caused|causing|lead to|leads to|led to|"
         r"result in|results in|resulted in|produce|produces|produced|trigger|triggers|triggered)\b",
         "CAUSES"),
        (r"\b(?:enable|enables|enabled|allow|allows|allowed|permit|permits|permitted|"
         r"make possible|makes possible|made possible)\b", "ENABLES"),
        (r"\b(?:prevent|prevents|prevented|avoid|avoids|avoided|block|blocks|blocked|"
         r"inhibit|inhibits|inhibited)\b", "PREVENTS"),
    )
    for pattern, relation in patterns:
        match = re.search(pattern, span or "", re.I)
        if match:
            return match.group(0)
    return ""


def _link(action_id: str, source: str, target: str, modality: str, host: dict) -> dict[str, Any]:
    cue = _explicit_relation_cue(host.get("text", ""))
    if cue:
        relation = (
            "CAUSES" if re.search(r"\b(?:cause|lead|led|result|produce|trigger)", cue, re.I)
            else "PREVENTS" if re.search(r"\b(?:prevent|avoid|block|inhibit)", cue, re.I)
            else "ENABLES"
        )
    else:
        # Parliament currently requires a typed parent for downstream human
        # outcomes. ENABLES preserves that topology without silently classifying
        # a bare conditional as direct doing; the proposal envelope retains
        # CAUSES as an unresolved alternative.
        relation = "ENABLES"
    return {
        "action_id": action_id, "source_id": source, "link_relation": relation,
        "target_id": target, "modality": modality, "condition_ids": [],
        "clause_ids": [host["clause_id"]],
    }


_ORIGIN_RANK = {"UNRESOLVED": 0, "STRUCTURALLY_DERIVED": 1, "SOURCE_ASSERTED": 2}


def _party(text: str, parties: list[dict], index: dict, label: str,
           quantities: list[str] | None = None, *, role: str = "",
           construction: str = "", action_span: str = "") -> str:
    licensed = license_kind(
        _nominal_before_clause(label), role=role, text=text,
        construction=construction, quantities=quantities,
        action_span=action_span,
    )
    folded = label.casefold()
    for row in parties:
        if row["label"].casefold() == folded:
            if (_ORIGIN_RANK.get(licensed["origin"], 0)
                    > _ORIGIN_RANK.get(row.get("kind_origin"), 0)):
                row["kind"] = licensed["kind"]
                row["kind_origin"] = licensed["origin"]
            return row["party_id"]
    index["n"] += 1
    ident = f"P{index['n']}"
    clauses = segment_source_clauses(text)
    parties.append({
        "party_id": ident, "label": label, "kind": licensed["kind"],
        "kind_origin": licensed["origin"],
        "quantities": [token for token in (quantities or []) if token],
        "clause_ids": [row["clause_id"] for row in clauses if label.casefold() in row["text"].casefold()],
    })
    return ident


def _clause_for(text: str, *snippets: str) -> dict[str, Any]:
    return _clause_matching(text, snippets, None)


def _clause_matching(text: str, snippets: tuple[str, ...], pattern: str | None) -> dict[str, Any] | None:
    """Prefer a clause that contains every snippet and, when asked, a cue."""
    clauses = segment_source_clauses(text)
    required = [snippet for snippet in snippets if snippet]
    for row in clauses:
        folded = row["text"].casefold()
        if any(snippet.casefold() not in folded for snippet in required):
            continue
        if pattern is None or re.search(pattern, row["text"], re.I):
            return row
    if pattern is not None:
        return None
    for snippet in required:
        host = _clause_holding(clauses, snippet)
        if host:
            return host
    return clauses[0] if clauses else None


def _note_shared_clause(actions: list[dict], notes: list[str]) -> None:
    cited = {tuple(row["clause_ids"]) for row in actions}
    if len(actions) > 1 and len(cited) == 1:
        notes.append("Both options are stated in the same clause.")


def _outcome_bearer(outcome: str, recipient: str) -> tuple[str, list[str]]:
    span = group_span(outcome)
    if not span:
        return recipient, []
    return span, group_quantity(span)


def _outcome_reading(span: str) -> dict[str, Any]:
    folded = span.casefold()
    if re.search(r"drown", folded):
        predicate, polarity, kind = "drown", "ADVERSE", "HEALTH_OUTCOME"
    elif re.search(r"\bkill", folded):
        predicate, polarity, kind = "kill", "ADVERSE", "HEALTH_OUTCOME"
    elif re.search(r"\bdie\b|\bdies\b", folded):
        predicate, polarity, kind = "die", "ADVERSE", "HEALTH_OUTCOME"
    elif re.search(r"\blose\b|\bloses\b", folded):
        predicate, polarity, kind = "lose", "ADVERSE", "WELFARE_OUTCOME"
    elif re.search(r"sustain", folded):
        predicate, polarity, kind = "sustain", "BENEFICIAL", "WELFARE_OUTCOME"
    elif re.search(r"\blive\b|\blives\b", folded):
        predicate, polarity, kind = "live", "BENEFICIAL", "HEALTH_OUTCOME"
    elif re.search(r"surviv|recover", folded):
        predicate, polarity, kind = "survive", "BENEFICIAL", "HEALTH_OUTCOME"
    else:
        predicate, polarity, kind = "outcome", "NEUTRAL", "OTHER"
    qualifier = _CHANCE.search(span)
    if re.search(r"\b(?:can|could|may|might)\b", folded):
        modality = "POSSIBLE"
    elif qualifier:
        modality = "PROBABILISTIC"
    elif re.search(r"\bwill\b", folded):
        modality = "CERTAIN"
    else:
        modality = "UNKNOWN"
    return {"predicate": predicate, "polarity": polarity, "kind": kind, "modality": modality,
            "qualifiers": [qualifier.group(0)] if qualifier else []}


def _stipulated_certain_outcome(span: str) -> bool:
    """A copied 'will die' / 'will live' is stipulated on its action branch."""
    return bool(re.search(
        r"\bwill\s+(?:die|live|drown|survive|recover)\b", span or "", re.I))


def _checked_modality(outcome: str, hedge: str | None) -> tuple[str, list[str]]:
    """Use a hedge the world-model check recognizes. Bare can stays CERTAIN."""
    outcome_chance = _CHANCE.search(outcome or "")
    if outcome_chance:
        return "PROBABILISTIC", [outcome_chance.group(0)]
    chosen = hedge or ""
    if not chosen:
        match = _CHANCE.search(outcome or "")
        chosen = match.group(0) if match else ""
    chance = _CHANCE.search(chosen)
    if chance:
        return "PROBABILISTIC", [chance.group(0)]
    if chosen and _HEDGE.search(chosen):
        return "POSSIBLE", []
    return "CERTAIN", []


def _answers_sentence(text: str, item: dict[str, Any], span: str,
                      accepted: dict[str, str]) -> bool:
    check = item["check"]
    if check == "copy":
        return True
    if check == "party":
        return True
    if check == "assignment":
        resource = accepted.get("resource", "")
        return bool(resource) and _shares_word(span, resource)
    if check == "exclusivity":
        return bool(_EXCLUSIVITY.search(span))
    if check == "not_both":
        return bool(re.search(r"\bnot\s+both\b", span, re.I))
    if check == "resource_quantity":
        resource = accepted.get("resource", "")
        return bool(resource) and bool(_NUMBER.search(span)) and _near(text, span, resource, 80)
    if check == "chance":
        return bool(_CHANCE.search(span))
    if check == "nonreceipt":
        return bool(_NONRECEIPT.search(span))
    if check == "transfer_event":
        return bool(_TRANSFER_EVENT.search(span))
    if check == "hedge":
        if not (_HEDGE.search(span) or _CHANCE.search(span)):
            return False
        outcome = accepted.get("first_outcome" if item["id"] == "first_hedge" else "second_outcome", "")
        return bool(outcome) and span.casefold() in outcome.casefold()
    if check == "process":
        resource = accepted.get("resource", "")
        return bool(resource) and (span.casefold() == resource.casefold() or _shares_word(span, resource))
    if check == "distinct_branch":
        earlier = accepted.get("first_branch_sentence", "")
        if not earlier:
            return False
        first = _clause_for(text, earlier)
        second = _clause_for(text, span)
        return first["clause_id"] != second["clause_id"]
    if check == "if_clause":
        return bool(re.search(r"\bif\b", span, re.I))
    if check == "distinct_if_clause":
        if not re.search(r"\bif\b", span, re.I):
            return False
        earlier = accepted.get("condition", "")
        first = _span_host(text, earlier) if earlier else None
        second = _span_host(text, span)
        return bool(first and second and first["clause_id"] != second["clause_id"])
    if check == "outcome_clause":
        host = _span_host(text, span)
        return bool(host and _OUTCOME_WORDS.search(host["text"]))
    if check == "branch_hedge":
        outcome_id = "harm_done" if item["id"] == "done_hedge" else "harm_omitted"
        outcome = accepted.get(outcome_id, "")
        host = _span_host(text, outcome) if outcome else None
        return bool(host and (_HEDGE.search(span) or _CHANCE.search(span))
                    and span.casefold() in host["text"].casefold())
    if check == "group_count":
        return bool(group_span(span))
    if check == "instrument_contrast":
        actions = " ".join((accepted.get("done", ""), accepted.get("omitted", "")))
        return (_near_pattern(text, span, r"\bbut\s+not\b", 64)
                and span.casefold() not in actions.casefold())
    if check == "number":
        return bool(_NUMBER.search(span))
    if check == "hedge_any":
        return bool(_HEDGE.search(span) or _CHANCE.search(span)
                    or re.search(r"\b(?:may|might|could|possibly|probably|unlikely|likely)\b", span, re.I))
    if check == "modal_words":
        return bool(_MODAL_WORDS.search(span))
    if check == "modal_action":
        return bool(_MODAL_WORDS.search(span) or _near_pattern(text, span, _MODAL_WORDS.pattern, 40))
    if check == "deontic_words":
        return bool(_DEONTIC_WORDS.search(span))
    if check == "process_words":
        return bool(re.search(r"\b(?:flow|flows|trolley|train|water|traffic|current|fire|process)\b", span, re.I))
    if check == "diversion_action":
        return bool(re.search(r"\b(?:divert|diverts|redirect|redirects|switch|switches|turn|turns)\b", span, re.I))
    if check == "exception_words":
        return bool(re.search(r"\b(?:except|unless|exception|only if)\b", span, re.I))
    if check == "commitment_words":
        return bool(re.search(r"\b(?:promise|promises|promised|commit|commits|committed|pledge|pledges)\b", span, re.I))
    if check == "reliance_words":
        return bool(re.search(r"\b(?:rely|relies|relied|depend|depends|expected|expects)\b", span, re.I))
    if check == "breach_words":
        return bool(re.search(r"\b(?:breach|breaks|broke|fails|failed|does not|did not)\b", span, re.I))
    if check == "report_words":
        return bool(re.search(r"\b(?:report|reports|reported|say|says|said|claim|claims|believe|believes|allege|alleges)\b", span, re.I))
    if check == "reliability_words":
        return bool(re.search(r"\b(?:reliable|unreliable|credible|uncertain|trustworthy|false|accurate)\b", span, re.I))
    if check == "confirmation_words":
        return bool(re.search(r"\b(?:confirm|confirms|confirmed|refute|refutes|refuted|verify|verifies|verified)\b", span, re.I))
    if check == "negated_action":
        return bool(re.search(r"\bnot\b|n't", span, re.I))
    if check == "not_remnant":
        return _near_pattern(text, span, r"\bnot\b|cannot|can't", 48)
    if check == "rescue_object":
        return _near_pattern(text, span, r"\b(?:save|saves|rescue|rescues|carry|carries)\b", 48)
    if check == "rescue_action":
        return bool(re.search(r"\b(?:save|saves|rescue|rescues|carry|carries)\b", span, re.I))
    if check == "harm_clause":
        host = _span_host(text, span)
        return bool(host and re.search(
            r"\b(?:die|dies|kill|kills|harm|harms|drown|drowns|suffer|suffers)\b",
            host["text"], re.I))
    return False


def _duplicates_party(item: dict[str, Any], span: str, accepted: dict[str, str]) -> bool:
    if item["id"] not in {
        "first_recipient", "second_recipient", "bearer",
        "first_saved", "second_saved",
    }:
        return False
    folded = span.casefold()
    blocked = [accepted.get("decider"), accepted.get("resource"), accepted.get("actor"),
               accepted.get("rescuer"), accepted.get("first_recipient"),
               accepted.get("first_saved")]
    return folded in {value.casefold() for value in blocked if value}


def _copy_spans(text: str, completion: str) -> tuple[list[str], str]:
    if not isinstance(completion, str):
        return [], ""
    snippet = " ".join(completion.strip().strip("\"'`").split()).rstrip(".")
    if not snippet or snippet.casefold() in _ABSTAIN:
        return [], ""
    if len(snippet.split()) > 22:
        return [], "too_long"
    pattern = r"\s+".join(re.escape(part) for part in snippet.split())
    return [text[match.start():match.end()] for match in re.finditer(pattern, text, re.I)], ""


def _nominal_before_clause(span: str) -> str:
    """The noun phrase before a relative clause, for kind and quantity only."""
    match = _RELATIVE.search(span or "")
    return span[:match.start()].strip(" ,") if match else span


def _party_head(text: str, span: str) -> str:
    """End a party at a clause boundary. A relative clause stays with its noun."""
    relative = _RELATIVE.search(span)
    if relative:
        following = _NEXT_ALTERNATIVE.search(span, relative.end())
        kept = span[:following.start()].strip(" ,") if following else span.strip(" ,")
    else:
        verb = _MATRIX_VERB.search(span)
        kept = span[:verb.start()].strip(" ,") if verb else span
    if not kept:
        return span
    copies = _copy_spans(text, kept)[0]
    return copies[0] if copies else span


def _span_host(text: str, span: str) -> dict[str, Any] | None:
    start = text.find(span)
    if start < 0:
        return None
    for row in segment_source_clauses(text):
        if row["start"] <= start < row["end"]:
            return row
    return None


def _abstains(completion: str) -> bool:
    if not isinstance(completion, str):
        return True
    return " ".join(completion.strip().strip("\"'`").split()).rstrip(".").casefold() in _ABSTAIN or not completion.strip()


def _near(text: str, span: str, other: str, radius: int) -> bool:
    words = _content_words(other)
    if not words:
        return False
    return _near_pattern(text, span, r"\b(?:" + "|".join(re.escape(word) for word in words) + r")\b", radius)


def _near_pattern(text: str, span: str, pattern: str, radius: int) -> bool:
    match = re.search(r"\s+".join(re.escape(part) for part in span.split()), text, re.I)
    if not match:
        return False
    window = text[max(0, match.start() - radius):match.end() + radius]
    return bool(re.search(pattern, window, re.I))


def _shares_word(left: str, right: str) -> bool:
    return bool(set(_content_words(left)) & set(_content_words(right)))


def _content_words(span: str) -> list[str]:
    return [word for word in _WORD.findall(span) if word.casefold() not in _STOP and len(word) > 2]


def _predicate(span: str) -> str:
    """Prefer a licensed source verb. SpaCy ROOT is a suggestion only."""
    copied = licensed_source_predicate(span)
    if copied:
        return copied
    try:
        import parsing_game_S as parsing
        parsed = parsing.get_nlp()(span)
        verbal = [token for token in parsed
                  if token.pos_ in {"VERB", "AUX"} and token.dep_ == "ROOT"]
        if not verbal:
            verbal = [token for token in parsed if token.pos_ == "VERB"]
        if verbal:
            return verbal[0].lemma_.casefold()
    except (OSError, RuntimeError, ValueError):
        pass
    words = [word.casefold() for word in _WORD.findall(span) if word.casefold() not in _STOP]
    return words[0] if words else "act"


def _quantity_token(span: str) -> str:
    match = _NUMBER.search(span or "")
    return match.group(0) if match else ""


def _parse_ranking(raw: str) -> list[str]:
    try:
        data = _load_json(raw)
    except (json.JSONDecodeError, TypeError):
        data = {}
    found = []
    for item in data.get("templates") or data.get("blueprint_ids") or []:
        if (item in _BY_ID or item in _META_BLUEPRINTS) and item not in found:
            found.append(item)
    for item in _BY_ID:
        if len(found) >= 3:
            break
        if item not in found:
            found.append(item)
    return found[:3]


def _parse_answers(raw: str) -> dict[str, str]:
    try:
        data = _load_json(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    answers = data.get("answers", data)
    if not isinstance(answers, dict):
        return {}
    return {str(key): "" if value is None else str(value) for key, value in answers.items()}


def _load_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise TypeError("model JSON was not an object")
    return value


def _env_value(path: Path, name: str) -> str:
    if not path.exists():
        return ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value
    return ""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fill blueprint blanks from a scenario.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--openai-env", type=Path, default=DEFAULT_OPENAI_ENV)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = choose_by_cloze(args.text, openai_complete(args.model, args.openai_env))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({
        "chosen_blueprint_id": result["chosen_blueprint_id"],
        "status": result["status"],
        "ranking": result["ranking"],
        "scores": [{
            "blueprint_id": row["blueprint_id"], "status": row["status"],
            "core_filled": row["core_filled"], "optional_filled": row["optional_filled"],
            "rejected": row["rejected"], "unfilled": row["unfilled"],
        } for row in result["considered"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
