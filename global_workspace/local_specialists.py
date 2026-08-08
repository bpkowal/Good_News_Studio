from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

from .models import CandidateChunk, WorkspaceBroadcast


FRAMEWORK_ROLES = {
    "utilitarian": "Score expected harms, benefits, urgency, and reversibility.",
    "deontological": "Score duties, rights, entitlement, coercion, and universal rules.",
    "virtue": "Score practical wisdom, character, honesty, courage, and habituation.",
    "care": "Score vulnerability, dependency, trust, relationship, and responsiveness.",
    "rawlsian": "Score fairness, equal liberty, public rules, and the least advantaged.",
}

ALLOWED_CONSTRAINTS = {
    "IMMINENT_HARM",
    "RIGHTS",
    "DUTY",
    "FAIRNESS",
    "CARE",
    "CHARACTER",
    "FEASIBILITY",
    "UNCERTAINTY",
    "PUBLIC_RULE",
}

FRAMEWORK_CONSTRAINTS = {
    "utilitarian": {"IMMINENT_HARM", "FEASIBILITY", "UNCERTAINTY"},
    "deontological": {"DUTY", "RIGHTS", "PUBLIC_RULE", "UNCERTAINTY"},
    "virtue": {"CHARACTER", "FEASIBILITY", "UNCERTAINTY"},
    "care": {"CARE", "FEASIBILITY", "UNCERTAINTY"},
    "rawlsian": {"FAIRNESS", "RIGHTS", "PUBLIC_RULE", "UNCERTAINTY"},
}

ALLOWED_UNRESOLVED = {
    "NONE",
    "VERIFY_FACTS",
    "CHECK_FEASIBILITY",
    "CLARIFY_SCENARIO",
}


@lru_cache(maxsize=8)
def _json_grammar(schema_text: str) -> Any | None:
    """Build a llama.cpp grammar when supported; tests and older builds may omit it."""
    try:
        from llama_cpp import LlamaGrammar

        return LlamaGrammar.from_json_schema(schema_text)
    except (ImportError, AttributeError, ValueError):
        return None


def _call_json_llm(llm: Any, prompt: str, *, max_tokens: int, temperature: float, schema: dict[str, Any]):
    kwargs = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    grammar = _json_grammar(json.dumps(schema, sort_keys=True))
    if grammar is not None:
        kwargs["grammar"] = grammar
    return llm(prompt, **kwargs)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Local model returned no JSON object: {text[:160]!r}")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Local model response must be a JSON object")
    return value


def _number(value: Any, default: float = 0.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _strict_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return number


def _candidate_from_data(
    specialist: str,
    actions: Sequence[str],
    data: dict[str, Any],
    broadcast: WorkspaceBroadcast,
    baseline_id: str,
    scenario_facts: dict[str, Any],
) -> CandidateChunk:
    action_ids = [f"A{index}" for index in range(len(actions))]
    raw_scores = data.get("scores")
    if not isinstance(raw_scores, dict) or set(raw_scores) != set(action_ids):
        raise ValueError(f"scores must contain exactly {action_ids}")
    id_scores = {
        action_id: _strict_number(raw_scores[action_id], f"scores.{action_id}")
        for action_id in action_ids
    }
    scores = {
        action: id_scores[action_id]
        for action_id, action in zip(action_ids, actions)
    }

    recommended_id = str(data.get("r", data.get("recommended", ""))).strip().upper()
    if recommended_id not in action_ids:
        raise ValueError("recommended must be a valid action ID")
    if id_scores[recommended_id] != max(id_scores.values()):
        raise ValueError("recommended must have the highest score")

    if baseline_id not in {*action_ids, "NONE"}:
        raise ValueError("baseline must be a valid action ID or NONE")
    if baseline_id == "NONE":
        alignment = "UNCLEAR"
    elif recommended_id == baseline_id:
        alignment = "SUPPORTS"
    else:
        alignment = "RECONSIDERS"
    if broadcast.constraint == "OPEN_DELIBERATION" and baseline_id != "NONE" and recommended_id != baseline_id:
        raise ValueError("the initial recommendation must match the source-testimony baseline")

    rationale = " ".join(str(data.get("w", data.get("why", ""))).split())
    if len(rationale.split()) < 2:
        raise ValueError("why must contain at least two words")

    recipient_facts = scenario_facts.get("survival_chance", {})
    if isinstance(recipient_facts, dict) and recipient_facts:
        selected_action = actions[action_ids.index(recommended_id)].lower()
        selected_recipients = [name for name in recipient_facts if name in selected_action]
        mentioned_recipients = [name for name in recipient_facts if name in rationale.lower()]
        if mentioned_recipients and selected_recipients and not set(selected_recipients) & set(mentioned_recipients):
            raise ValueError("why names a different recipient than the recommended action")
        if "higher survival" in rationale.lower() and selected_recipients:
            highest = max(recipient_facts, key=recipient_facts.get)
            if highest not in selected_recipients:
                raise ValueError("why contradicts the scenario's survival probabilities")

    constraint = str(data.get("c", data.get("constraint", ""))).strip().upper().replace(" ", "_")
    allowed_constraints = FRAMEWORK_CONSTRAINTS[specialist]
    if constraint not in allowed_constraints:
        raise ValueError(f"constraint for {specialist} must be one of {sorted(allowed_constraints)}")
    unresolved = str(data.get("u", data.get("unresolved", "NONE"))).strip().upper().replace(" ", "_")
    if unresolved not in ALLOWED_UNRESOLVED:
        raise ValueError(f"unresolved must be one of {sorted(ALLOWED_UNRESOLVED)}")

    ordered_scores = sorted(id_scores.values(), reverse=True)
    score_gap = ordered_scores[0] - ordered_scores[1] if len(ordered_scores) > 1 else ordered_scores[0]
    confidence = max(0.1, min(1.0, score_gap))
    friction = score_gap
    surprise = 0.7 if alignment == "RECONSIDERS" else (0.3 if alignment == "UNCLEAR" else 0.0)

    return CandidateChunk(
        specialist=specialist,
        constraint=constraint,
        action_scores=scores,
        surprise=surprise,
        friction=friction,
        confidence=confidence,
        unresolved=unresolved,
        rationale=rationale,
        recommended_action=actions[action_ids.index(recommended_id)],
        baseline_action=(actions[action_ids.index(baseline_id)] if baseline_id != "NONE" else ""),
        testimony_alignment=alignment,
    )


def _invalid_candidate(specialist: str, actions: Sequence[str], error: str) -> CandidateChunk:
    return CandidateChunk(
        specialist=specialist,
        constraint="MALFORMED_RESPONSE",
        action_scores={action: 0.5 for action in actions},
        surprise=0.0,
        friction=0.0,
        confidence=0.0,
        unresolved="REVIEW_MODEL_OUTPUT",
        rationale="Delegate output failed semantic validation.",
        schema_valid=False,
        validation_errors=[error[:300]],
    )


@dataclass(slots=True)
class CompactLocalSpecialist:
    name: str
    llm: Any
    testimony: str = ""
    baseline_action_id: str = "NONE"
    scenario_facts: dict[str, Any] | None = None
    max_tokens: int = 128

    def evaluate(
        self,
        scenario: str,
        actions: Sequence[str],
        broadcast: WorkspaceBroadcast,
    ) -> CandidateChunk:
        role = FRAMEWORK_ROLES[self.name]
        testimony = " ".join(self.testimony.split())[:900]
        action_ids = [f"A{index}" for index in range(len(actions))]
        action_legend = {action_id: action for action_id, action in zip(action_ids, actions)}
        allowed_constraints = sorted(FRAMEWORK_CONSTRAINTS[self.name])
        fixed_baseline = self.baseline_action_id if self.baseline_action_id in {*action_ids, "NONE"} else "NONE"
        schema = {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "object",
                    "properties": {
                        action_id: {"type": "number", "minimum": 0, "maximum": 1}
                        for action_id in action_ids
                    },
                    "required": action_ids,
                    "additionalProperties": False,
                },
                "r": {"type": "string", "enum": action_ids},
                "c": {"type": "string", "enum": allowed_constraints},
                "u": {"type": "string", "enum": sorted(ALLOWED_UNRESOLVED)},
                "w": {"type": "string"},
            },
            "required": [
                "scores", "r", "c", "u", "w",
            ],
            "additionalProperties": False,
        }
        prompt = f"""[INST]
You are the {self.name} specialist in a bandwidth-limited ethical workspace.
Task: {role}
Scenario: {' '.join(scenario.split())[:700]}
Your original corpus-grounded testimony: {testimony}
Frozen testimony baseline: {fixed_baseline}
Workspace: {broadcast.compact()}
Scenario facts: {json.dumps(self.scenario_facts or {}, sort_keys=True)}
Action IDs: {json.dumps(action_legend)}

Return ONLY compact JSON like:
{{"scores":{{"A0":0.8,"A1":0.2}},"r":"A0","c":"{allowed_constraints[0]}","u":"NONE","w":"short reason"}}
Return scores for every action ID. recommended must have the highest score.
r=recommended and must have the highest score. The frozen baseline was extracted
separately from your testimony. Python derives whether the result supports or
reconsiders it. During OPEN_DELIBERATION, recommended
must equal a known baseline. Each score means recommendation strength:
1 strongly recommends; 0 strongly rejects. w must be 2-8 words.
c must be one of: {', '.join(allowed_constraints)}.
Choose u only from: {', '.join(sorted(ALLOWED_UNRESOLVED))}.
[/INST]"""
        output = _call_json_llm(
            self.llm,
            prompt,
            max_tokens=self.max_tokens,
            temperature=0.2,
            schema=schema,
        )
        raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
        try:
            data = _extract_json(raw)
            return _candidate_from_data(
                self.name, actions, data, broadcast, fixed_baseline, self.scenario_facts or {}
            )
        except (ValueError, json.JSONDecodeError) as first_error:
            repair_prompt = f"""[INST]
Repair this invalid answer as JSON only: {raw[:400]}
Required fields: scores object for {', '.join(action_ids)}, r, c, u, w.
r must have the highest score and respect frozen baseline {fixed_baseline} initially.
c must be one of: {', '.join(allowed_constraints)}. No prose.
[/INST]"""
            repaired = _call_json_llm(
                self.llm,
                repair_prompt,
                max_tokens=96,
                temperature=0.0,
                schema=schema,
            )
            repaired_raw = repaired["choices"][0]["text"] if isinstance(repaired, dict) else str(repaired)
            try:
                data = _extract_json(repaired_raw)
                return _candidate_from_data(
                    self.name, actions, data, broadcast, fixed_baseline, self.scenario_facts or {}
                )
            except (ValueError, json.JSONDecodeError) as repair_error:
                return _invalid_candidate(
                    self.name,
                    actions,
                    f"initial={first_error}; repair={repair_error}",
                )


def _feasible_actions(data: dict[str, Any]) -> list[str]:
    raw_actions = data.get("actions", [])
    actions = []
    if isinstance(raw_actions, list):
        for item in raw_actions:
            if not isinstance(item, dict) or item.get("e") is not True:
                continue
            try:
                feasibility = _strict_number(item.get("f"), "action feasibility")
            except ValueError:
                continue
            action = " ".join(str(item.get("a", "")).split())[:60]
            if action and feasibility >= 0.65 and action not in actions:
                actions.append(action)
    actions = actions[:4]
    if len(actions) < 2:
        raise ValueError("fewer than two explicitly supported feasible actions")
    return actions


def extract_explicit_actions(scenario: str) -> list[str]:
    """Extract a closed natural-language either/or choice without model generation."""
    cleaned = " ".join(scenario.split())
    match = re.search(
        r"\beither\s+(.+?)\s+or\s+(.+?)(?=[?.]|$)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    actions = []
    for value in match.groups():
        action = value.strip(" ,;:")[:80]
        if action and action.lower() not in {item.lower() for item in actions}:
            actions.append(action[0].upper() + action[1:])
    return actions if len(actions) == 2 else []


def extract_allocation_actions(scenario: str) -> list[str]:
    """Recognize simple one-resource/two-recipient allocation questions."""
    cleaned = " ".join(scenario.split())
    if not re.search(r"\bwho should receive (?:it|the\s+\w+)\b", cleaned, re.IGNORECASE):
        return []
    resource_match = re.search(
        r"\b(?:has|have)\s+(?:only\s+)?one\s+([a-z][a-z -]{0,30}?)(?=\s+and\s+|\s+for\s+|\s+to\s+|[.,])",
        cleaned,
        re.IGNORECASE,
    )
    if not resource_match:
        return []
    resource = " ".join(resource_match.group(1).split())
    recipients = []
    for match in re.finditer(
        r"\b(?:a|an|the)\s+([a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,2})\s+(?=with\b|who\b)",
        cleaned,
        re.IGNORECASE,
    ):
        recipient = " ".join(match.group(1).lower().split())
        if recipient not in recipients and recipient not in {"hospital", "patient"}:
            recipients.append(recipient)
    if len(recipients) != 2:
        return []
    return [f"Give the {resource} to the {recipient}" for recipient in recipients]


def extract_scenario_facts(scenario: str) -> dict[str, Any]:
    """Extract a deliberately small fact table used for contradiction checks."""
    cleaned = " ".join(scenario.lower().split())
    survival: dict[str, float] = {}
    for match in re.finditer(
        r"\b(?:a|an|the)\s+([a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,2})\s+with\s+(?:an?\s+)?(\d{1,3})%\s+(?:survival\s+)?chance",
        cleaned,
    ):
        survival[match.group(1)] = int(match.group(2)) / 100.0
    return {"survival_chance": survival} if survival else {}


def infer_testimony_baseline(
    llm: Any,
    specialist: str,
    testimony: str,
    actions: Sequence[str],
    max_tokens: int = 64,
) -> tuple[str, str]:
    action_ids = [f"A{index}" for index in range(len(actions))]
    legend = {action_id: action for action_id, action in zip(action_ids, actions)}
    schema = {
        "type": "object",
        "properties": {
            "b": {"type": "string", "enum": [*action_ids, "NONE"]},
            "w": {"type": "string"},
        },
        "required": ["b", "w"],
        "additionalProperties": False,
    }
    prompt = f"""[INST]
Framework: {specialist}
Original testimony: {' '.join(testimony.split())[:1000]}
Actions: {json.dumps(legend)}
Which action does the testimony recommend? Return NONE if it gives no conclusion.
Return JSON only: {{"b":"A0","w":"short evidence"}}
[/INST]"""
    output = _call_json_llm(llm, prompt, max_tokens=max_tokens, temperature=0.0, schema=schema)
    raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
    try:
        data = _extract_json(raw)
        baseline = str(data.get("b", "NONE")).strip().upper()
        reason = " ".join(str(data.get("w", "")).split())[:160]
        if baseline not in {*action_ids, "NONE"}:
            return "NONE", "invalid baseline ID"
        return baseline, reason
    except (ValueError, json.JSONDecodeError) as exc:
        return "NONE", f"baseline extraction failed: {exc}"[:160]


def propose_actions(llm: Any, scenario: str, max_tokens: int = 96) -> list[str]:
    explicit_actions = extract_explicit_actions(scenario)
    if explicit_actions:
        return explicit_actions
    allocation_actions = extract_allocation_actions(scenario)
    if allocation_actions:
        return allocation_actions

    action_schema = {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "minItems": 2,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "string"},
                        "f": {"type": "number", "minimum": 0, "maximum": 1},
                        "e": {"type": "boolean"},
                    },
                    "required": ["a", "f", "e"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["actions"],
        "additionalProperties": False,
    }
    prompt = f"""[INST]
Scenario: {' '.join(scenario.split())[:1200]}
Return ONLY JSON: {{"actions":[{{"a":"short action","f":0.9,"e":true}}]}}
Give 2 to 4 distinct actions. f is feasibility before the stated harm (0 to 1).
e is true only when the scenario explicitly supports the action. Do not invent
waiting, warning, authorities, escape, or rescue when time/opportunity is absent.
If the scenario states a closed choice, preserve its stated alternatives.
[/INST]"""
    output = _call_json_llm(
        llm,
        prompt,
        max_tokens=max_tokens,
        temperature=0.2,
        schema=action_schema,
    )
    raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
    try:
        return _feasible_actions(_extract_json(raw))
    except (ValueError, json.JSONDecodeError) as first_error:
        repair_prompt = f"""[INST]
Repair this action plan: {raw[:400]}
Return JSON only: {{"actions":[{{"a":"short action","f":0.9,"e":true}}]}}
Keep only actions explicitly available before harm. Include at least two.
[/INST]"""
        repaired = _call_json_llm(
            llm,
            repair_prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            schema=action_schema,
        )
        repaired_raw = repaired["choices"][0]["text"] if isinstance(repaired, dict) else str(repaired)
        try:
            return _feasible_actions(_extract_json(repaired_raw))
        except (ValueError, json.JSONDecodeError) as repair_error:
            raise ValueError(
                f"Action plan failed feasibility validation: initial={first_error}; "
                f"repair={repair_error}"
            ) from repair_error
