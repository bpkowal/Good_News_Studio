"""Choose a blueprint by completing its sentences from the scenario.

Each template is a list of sentences with a blank at the end. A model finishes
the three templates it thinks the text can complete. A completion counts only
when it is a contiguous copy of the scenario and answers that sentence. The
template with the most accepted blanks is passed forward. Empty blanks stay
empty. This module does not admit a world and does not pick a moral act.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Sequence
from urllib import request as urlrequest
from urllib.error import HTTPError

from candidate_graph_blueprints import _clause_holding
from z10_world_model_adapter import segment_source_clauses


CLOZE_VERSION = "blueprint-cloze-chooser/0.2"
DEFAULT_OPENAI_ENV = Path(
    "/Users/benjaminkowal/Documents/Python/Python Coding/RAGAIMODEL/.env"
)
_STOP = {
    "a", "an", "the", "to", "of", "or", "and", "who", "that", "in", "on", "for",
}
_ABSTAIN = {"none", "n/a", "null", "unknown", "no", "not stated"}
_EXCLUSIVITY = re.compile(r"\b(?:but\s+)?not\s+both\b|\beither\b[\s\S]{0,80}\bor\b", re.I)
_NUMBER = re.compile(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", re.I)
_CHANCE = re.compile(r"\d+(?:\.\d+)?%\s*chance", re.I)
_GROUP = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"(?:\s+[A-Za-z]+){0,2}\s+(people|workers|patients|residents)\b",
    re.I,
)
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
_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_RELATIVE = re.compile(r"\s+(?:that|who|which)\b", re.I)
_MATRIX_VERB = re.compile(r"\s+(?:must|will|can)\b", re.I)
_NEXT_ALTERNATIVE = re.compile(
    r"\s+(?:or|and)\s+(?=(?:a|an|the|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b)",
    re.I,
)
_PARTY_IDS = {
    "decider", "first_recipient", "second_recipient", "rescuer", "saved", "not_saved",
    "actor", "bearer", "harm_done", "harm_omitted",
}

Complete = Callable[[Sequence[dict[str, str]]], str]

# Sentences the model finishes. `core` blanks are the ones that make a graph.
# `check` rejects a copied span that answers a different question.
TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "blueprint_id": "exclusive_allocation",
        "summary": "Someone assigns one resource to one of two parties.",
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
        "summary": "Someone can rescue one party and not another.",
        "items": (
            {"id": "rescuer", "core": True, "check": "copy",
             "sentence": "The person who can rescue someone is"},
            {"id": "saved", "core": True, "check": "rescue_object",
             "sentence": "The party that person can save is"},
            {"id": "not_saved", "core": True, "check": "not_remnant",
             "sentence": "The party that person cannot save is"},
            {"id": "survival", "core": False, "check": "copy",
             "sentence": "What the text says happens when the rescue succeeds is"},
            {"id": "scene", "core": False, "check": "copy",
             "sentence": "The parties named as already in danger are"},
            {"id": "foregone", "core": False, "check": "harm_clause",
             "sentence": "What the text says happens to the party who is not saved is"},
        ),
    },
    {
        "blueprint_id": "omission_harm",
        "summary": "Doing an action and not doing it each harm someone.",
        "items": (
            {"id": "actor", "core": True, "check": "copy",
             "sentence": "The person who may do the action or not do it is"},
            {"id": "done", "core": True, "check": "copy",
             "sentence": "The action the text says is done is"},
            {"id": "omitted", "core": True, "check": "negated_action",
             "sentence": "The action the text says is not done is"},
            {"id": "harm_done", "core": True, "check": "harm_clause",
             "sentence": "Who is harmed if the action is done is"},
            {"id": "harm_omitted", "core": True, "check": "harm_clause",
             "sentence": "Who is harmed if the action is not done is"},
            {"id": "instrument", "core": False, "check": "copy",
             "sentence": "Another instrument named beside the action is"},
        ),
    },
    {
        "blueprint_id": "conditional_outcome",
        "summary": "An outcome is stated inside an if-clause.",
        "items": (
            {"id": "actor", "core": False, "check": "copy",
             "sentence": "The person whose action an outcome depends on is"},
            {"id": "condition", "core": True, "check": "if_clause",
             "sentence": "The if-clause the outcome depends on is"},
            {"id": "bearer", "core": True, "check": "party",
             "sentence": "The party that outcome happens to is"},
            {"id": "outcome", "core": True, "check": "copy",
             "sentence": "The outcome stated for that party is"},
            {"id": "second_conditional", "core": False, "check": "if_clause",
             "sentence": "A second if-then sentence in the text is"},
            {"id": "chance", "core": False, "check": "chance",
             "sentence": "A stated chance attached to an outcome is"},
        ),
    },
)
_BY_ID = {row["blueprint_id"]: row for row in TEMPLATES}


def choose_by_cloze(text: str, complete: Complete) -> dict[str, Any]:
    """Ask for the three most likely templates, finish each, and keep the best.

    The ethical question and its alternatives are read from the parser before a
    world graph is built. An exclusive graph is withheld when exclusivity was
    not evidenced, so a later compiler cannot treat "or" as an averted harm.
    """
    question = assess_question(text)
    rank_raw = complete(_messages(_rank_prompt(text)))
    ranking = _parse_ranking(rank_raw)
    considered = []
    for index, blueprint_id in enumerate(ranking):
        template = _BY_ID[blueprint_id]
        raw = complete(_messages(_cloze_prompt(text, template)))
        scored = _score_template(text, template, _parse_answers(raw), index)
        considered.append(_ask_implied(text, template, scored, complete))
    winner = _winner(considered)
    withheld = _world_withheld(question, winner)
    question["eligible_for_world_state"] = bool(
        winner and winner["status"] == "FILLED" and not withheld)
    graph = _graph(text, winner) if question["eligible_for_world_state"] else None
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
        "graph": graph,
    }


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
    exclusivity = "evidenced" if re.search(r"\bnot both\b", text, re.I) else "unspecified"
    return {
        "ethical_question": question,
        "scenario_options": options,
        "exclusivity": exclusivity,
        "participants": participants,
    }


def _world_withheld(question: dict[str, Any], winner: dict[str, Any] | None) -> list[str]:
    """Hold the graph until the question's alternatives are actually exclusive."""
    if winner is None or winner.get("blueprint_id") != "exclusive_allocation":
        return []
    if winner.get("status") != "FILLED":
        return []
    if question.get("exclusivity") == "evidenced":
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
             "", "Scenario:", text, ""]
    for template in TEMPLATES:
        sample = template["items"][0]["sentence"]
        lines.append(f"- {template['blueprint_id']}: {template['summary']} Example: {sample} _______.")
    return "\n".join(lines)


def _cloze_prompt(text: str, template: dict[str, Any]) -> str:
    lines = [
        f"Template: {template['blueprint_id']}",
        "Finish each sentence with a short contiguous copy of the scenario.",
        "If the scenario does not state that fact, answer NONE.",
        "Do not paraphrase. Do not decide what anyone should do.",
        "A relative clause that states a consequence may fill an outcome sentence.",
        "\"whether ... or ...\" does not say that both parties cannot receive it.",
        "A count of people is not the amount of the resource.",
        "A giving or receipt event must use give, receive, allocate, deliver, administer, provide, or supply. Another verb is NONE.",
        "A hedge is if, unless, whether, or a percent chance inside that outcome. The word can is not a hedge.",
        "If one sentence states both branches, answer NONE for a sentence that would state only one branch.",
        "", "Scenario:", text, "",
    ]
    answers = []
    asked = [item for item in template["items"] if item["check"] != "implied"]
    for index, item in enumerate(asked, 1):
        lines.append(f"{index}. {item['sentence']} _______.")
        answers.append(f'"{item["id"]}"')
    lines.append("")
    lines.append("Return a JSON object {\"answers\": {" + ", ".join(
        f"{name}: \"copy or NONE\"" for name in answers) + "}}.")
    return "\n".join(lines)


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


def _assigned_act(text: str, span: str) -> str:
    """A decision frame is not the act that applies the resource."""
    match = _DECISION_FRAME.match(span)
    if not match:
        return span
    rest = span[match.end():].strip()
    copies = _copy_spans(text, rest)[0]
    return copies[0] if copies else span


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
    core = [row for row in items if row["core"]]
    optional = [row for row in items if not row["core"]]
    core_filled = sum(row["verdict"] == "accepted" for row in core)
    status = "FILLED" if core and core_filled == len(core) else "PARTIAL"
    if core_filled == 0 and not any(row["verdict"] == "accepted" for row in optional):
        status = "NO_MATCH"
    return {
        "blueprint_id": template["blueprint_id"],
        "status": status,
        "rank": rank,
        "core_filled": core_filled,
        "core_count": len(core),
        "optional_filled": sum(row["verdict"] == "accepted" for row in optional),
        "rejected": sum(row["verdict"] == "rejected" for row in items),
        "unfilled": [row["id"] for row in items if row["verdict"] != "accepted"],
        "slots": accepted,
        "items": items,
    }


def _winner(considered: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [row for row in considered if row["core_filled"] or row["optional_filled"]]
    if not usable:
        return None
    return max(usable, key=lambda row: (row["core_filled"], row["optional_filled"], -row["rejected"], -row["rank"]))


def _graph(text: str, winner: dict[str, Any]) -> dict[str, Any]:
    slots = winner["slots"]
    builder = {
        "exclusive_allocation": _allocation_graph,
        "rescue_contrast": _rescue_graph,
        "omission_harm": _omission_graph,
        "conditional_outcome": _conditional_graph,
    }[winner["blueprint_id"]]
    world, notes = builder(text, slots)
    return {
        "proposal_id": f"{winner['blueprint_id']}_cloze",
        "status": "FILLED",
        "notes": notes,
        "candidate": {"world_model": world, "ellipsis_resolutions": []},
    }


def _allocation_graph(text: str, slots: dict[str, str]) -> tuple[dict, list[str]]:
    parties: list[dict[str, Any]] = []
    index = {"n": 0}
    issued = {"n": 0}

    def party(label: str, kind: str | None = None, quantities: list[str] | None = None) -> str:
        return _party(text, parties, index, label, kind or _kind(label), quantities or [])

    def effect_id() -> str:
        issued["n"] += 1
        return f"E{issued['n']}"

    actor = party(slots["decider"])
    resource = party(slots["resource"], "RESOURCE",
                     [_quantity_token(slots["quantity"])] if slots.get("quantity") else [])
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
            recipient_label, quantities=_group_quantity(_nominal_before_clause(recipient_label)))
        action_id = f"A{branch}"
        if transfer:
            direct_outcome, direct_kind = transfer, "RESOURCE_TRANSFER"
        else:
            direct_outcome, direct_kind = slots["assignment"], "INTERVENTION"
            notes.append(f"{action_id} has no source giving or receipt event, so the act stays an intervention.")
        intervention = transfer or f"{slots['assignment']} to {recipient_label}"
        if intervention not in text:
            notes.append(f"{action_id} joins two copied spans that are not adjacent.")
        host = _clause_for(text, branch_sentence or outcome or recipient_label)
        act_host = _clause_for(text, direct_outcome)
        direct_id = effect_id()
        effects.append(_effect(
            direct_id, action_id, recipient, direct_outcome, _predicate(direct_outcome),
            "NEUTRAL", "DIRECT", direct_kind, "CERTAIN", act_host, [], [], source=direct_outcome))
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
            link = _link(action_id, parent_id, outcome_id, modality, outcome_host)
            link["condition_ids"] = condition_ids
            links.append(link)
        actions.append({
            "action_id": action_id, "intervention": intervention,
            "actor_party_id": actor, "resource_party_id": resource,
            "recipient_party_ids": [recipient], "effect_ids": effect_ids,
            "clause_ids": list(dict.fromkeys([act_host["clause_id"], host["clause_id"]])),
        })
    if slots.get("nonreceipt") and slots.get("exclusivity"):
        host = _clause_for(text, slots["nonreceipt"])
        # "do not get ... will die" keeps the negation on receiving. The death
        # itself is the "will die" span, so polarity is not read as a negated harm.
        death_copies = _copy_spans(text, "will die")[0]
        death_outcome = death_copies[0] if death_copies and "will die" in slots["nonreceipt"].casefold() else slots["nonreceipt"]
        for branch, other in ((0, slots["second_recipient"]), (1, slots["first_recipient"])):
            other_id = party(other)
            parent = actions[branch]["effect_ids"][0]
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
    actor = _party(text, parties, index, slots["rescuer"], _kind(slots["rescuer"]))
    saved = _party(text, parties, index, slots["saved"], _kind(slots["saved"]))
    _party(text, parties, index, slots["not_saved"], _kind(slots["not_saved"]))
    host = _clause_for(text, slots["saved"])
    intervention = f"save {slots['saved']}"
    if intervention not in text:
        intervention = slots["saved"]
    effects = [_effect("E1", "A0", saved, intervention, "save", "NEUTRAL", "DIRECT",
                       "INTERVENTION", "CERTAIN", host, [], [])]
    links = []
    notes = [f"The text names {slots['not_saved']} and does not state a harm for that party."
             if not slots.get("foregone") else ""]
    if slots.get("survival"):
        reading = _outcome_reading(slots["survival"])
        outcome_host = _clause_for(text, slots["survival"])
        effects.append(_effect("E2", "A0", saved, slots["survival"], reading["predicate"],
                               reading["polarity"], "DOWNSTREAM", reading["kind"],
                               reading["modality"], outcome_host, ["E1"], reading["qualifiers"]))
        links.append(_link("A0", "E1", "E2", reading["modality"], outcome_host))
    actions = [{"action_id": "A0", "intervention": intervention, "actor_party_id": actor,
                "recipient_party_ids": [saved], "effect_ids": [row["effect_id"] for row in effects],
                "clause_ids": [host["clause_id"]]}]
    return _world(parties, actions, effects, links), [row for row in notes if row]


def _omission_graph(text: str, slots: dict[str, str]) -> tuple[dict, list[str]]:
    parties: list[dict[str, Any]] = []
    index = {"n": 0}
    actor = _party(text, parties, index, slots["actor"], _kind(slots["actor"]))
    actions, effects, links = [], [], []
    pairs = ((slots["done"], slots["harm_done"]),
             (slots["omitted"], slots["harm_omitted"]))
    for branch, (action_span, bearer_label) in enumerate(pairs):
        bearer = _party(text, parties, index, bearer_label, _kind(bearer_label),
                        [_quantity_token(bearer_label)] if _quantity_token(bearer_label) else [])
        direct_host = _clause_for(text, action_span)
        outcome_host = _clause_matching(
            text, (bearer_label,), r"\b(?:die|dies|kill|kills|harm|harms)\b") or direct_host
        action_id = f"A{branch}"
        direct_id, outcome_id = f"E{branch * 2 + 1}", f"E{branch * 2 + 2}"
        effects.append(_effect(direct_id, action_id, bearer, action_span, _predicate(action_span),
                               "NEUTRAL", "DIRECT", "INTERVENTION", "CERTAIN", direct_host, [], []))
        effects.append(_effect(outcome_id, action_id, bearer, outcome_host["text"], "die",
                               "ADVERSE", "DOWNSTREAM", "HEALTH_OUTCOME", "CERTAIN",
                               outcome_host, [direct_id], []))
        links.append(_link(action_id, direct_id, outcome_id, "CERTAIN", outcome_host))
        clause_ids = list(dict.fromkeys([direct_host["clause_id"], outcome_host["clause_id"]]))
        actions.append({"action_id": action_id, "intervention": action_span,
                        "actor_party_id": actor, "recipient_party_ids": [bearer],
                        "effect_ids": [direct_id, outcome_id], "clause_ids": clause_ids})
    notes = []
    if slots.get("instrument"):
        notes.append(f"Instrument named in the text: {slots['instrument']}. It is not an action.")
    return _world(parties, actions, effects, links), notes


def _conditional_graph(text: str, slots: dict[str, str]) -> tuple[dict, list[str]]:
    parties: list[dict[str, Any]] = []
    index = {"n": 0}
    actor_label = slots.get("actor") or slots["bearer"]
    actor = _party(text, parties, index, actor_label, _kind(actor_label))
    bearer = _party(text, parties, index, slots["bearer"], _kind(slots["bearer"]))
    reading = _outcome_reading(slots["outcome"])
    host = _clause_for(text, slots["outcome"], slots["condition"])
    effects = [
        _effect("E1", "A0", bearer, slots["condition"], _predicate(slots["condition"]),
                "NEUTRAL", "DIRECT", "INTERVENTION", "CERTAIN", host, [], []),
        _effect("E2", "A0", bearer, slots["outcome"], reading["predicate"], reading["polarity"],
                "DOWNSTREAM", reading["kind"], reading["modality"], host, ["E1"],
                reading["qualifiers"]),
    ]
    links = [_link("A0", "E1", "E2", reading["modality"], host)]
    actions = [{"action_id": "A0", "intervention": slots["condition"], "actor_party_id": actor,
                "recipient_party_ids": [bearer], "effect_ids": ["E1", "E2"],
                "clause_ids": [host["clause_id"]]}]
    notes = []
    if slots.get("second_conditional"):
        notes.append("A second conditional was copied and is not a separate action yet: "
                     + slots["second_conditional"])
    return _world(parties, actions, effects, links), notes


def _world(parties: list[dict], actions: list[dict], effects: list[dict],
           links: list[dict], conditions: list[dict] | None = None) -> dict[str, Any]:
    return {
        "schema_version": "1.3", "parties": parties, "actions": actions,
        "effects": effects, "conditions": conditions or [], "temporal_relations": [],
        "causal_links": links, "counterfactual_links": [],
    }


def _effect(effect_id: str, action_id: str, party_id: str, outcome: str, predicate: str,
            polarity: str, directness: str, kind: str, modality: str, host: dict,
            parents: list[str], qualifiers: list[str], source: str | None = None) -> dict[str, Any]:
    proposition = source or host["text"]
    return {
        "effect_id": effect_id, "action_id": action_id, "party_id": party_id,
        "outcome": outcome, "predicate": predicate, "polarity": polarity,
        "directness": directness, "modality": modality, "effect_kind": kind,
        "condition_ids": [], "quantities": _group_quantity(proposition),
        "likelihood_qualifiers": qualifiers, "overall_likelihood_qualifiers": [],
        "scope_qualifiers": [], "temporal_qualifiers": [], "condition_join": "AND",
        "source_proposition": proposition, "source_effect_ids": parents,
        "derivation_operation": "DIRECT_COPY" if not parents else "SOURCE_STIPULATED_CAUSAL",
        "derivation_explanation": "The blank was filled with this copy of the scenario.",
        "derivation_assumptions": [], "outcome_type_transformation": "PRESERVED",
        "clause_ids": [host["clause_id"]],
    }


def _link(action_id: str, source: str, target: str, modality: str, host: dict) -> dict[str, Any]:
    return {
        "action_id": action_id, "source_id": source, "link_relation": "CAUSES",
        "target_id": target, "modality": modality, "condition_ids": [],
        "clause_ids": [host["clause_id"]],
    }


def _party(text: str, parties: list[dict], index: dict, label: str, kind: str,
           quantities: list[str] | None = None) -> str:
    folded = label.casefold()
    for row in parties:
        if row["label"].casefold() == folded:
            return row["party_id"]
    index["n"] += 1
    ident = f"P{index['n']}"
    clauses = segment_source_clauses(text)
    parties.append({
        "party_id": ident, "label": label, "kind": kind,
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
    match = _GROUP.search(outcome)
    if not match:
        return recipient, []
    return match.group(0), [match.group(1)]


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


def _checked_modality(outcome: str, hedge: str | None) -> tuple[str, list[str]]:
    """Use a hedge the world-model check recognizes. Bare can stays CERTAIN."""
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
    if check == "negated_action":
        return bool(re.search(r"\bnot\b|n't", span, re.I))
    if check == "not_remnant":
        return _near_pattern(text, span, r"\bnot\b|cannot|can't", 48)
    if check == "rescue_object":
        return _near_pattern(text, span, r"\b(?:save|saves|rescue|rescues|carry|carries)\b", 48)
    if check == "harm_clause":
        host = _span_host(text, span)
        return bool(host and re.search(r"\b(?:die|dies|kill|kills|harm|harms)\b", host["text"], re.I))
    return False


def _duplicates_party(item: dict[str, Any], span: str, accepted: dict[str, str]) -> bool:
    if item["id"] not in {"first_recipient", "second_recipient", "bearer", "saved", "not_saved"}:
        return False
    folded = span.casefold()
    blocked = [accepted.get("decider"), accepted.get("resource"), accepted.get("actor"),
               accepted.get("rescuer"), accepted.get("first_recipient"), accepted.get("saved")]
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


def _kind(label: str) -> str:
    folded = _nominal_before_clause(label).casefold()
    if _GROUP.search(label):
        return "HUMAN_GROUP"
    if re.search(r"\b(?:dog|animal)\b", folded):
        return "ANIMAL"
    if re.search(r"\b(?:medicine|serum|water|dose|resource)\b", folded):
        return "RESOURCE"
    if re.search(r"\b(?:farm|town|city|clinic)\b", folded):
        return "COMMUNITY"
    if re.search(r"\b(?:bot|model|system)\b", folded):
        return "OTHER"
    return "PERSON"


def _predicate(span: str) -> str:
    words = [word.casefold() for word in _WORD.findall(span) if word.casefold() not in _STOP]
    return words[0] if words else "act"


def _quantity_token(span: str) -> str:
    match = _NUMBER.search(span or "")
    return match.group(0) if match else ""


def _group_quantity(span: str) -> list[str]:
    match = _GROUP.search(span or "")
    return [match.group(1)] if match else []


def _parse_ranking(raw: str) -> list[str]:
    try:
        data = _load_json(raw)
    except (json.JSONDecodeError, TypeError):
        data = {}
    found = []
    for item in data.get("templates") or data.get("blueprint_ids") or []:
        if item in _BY_ID and item not in found:
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
