from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def summarize_specialist_contributions(cycles: list[Any]) -> dict[str, dict[str, Any]]:
    """Summarize informational contribution without pretending ethical correctness."""
    records: dict[str, dict[str, Any]] = {}
    constraints: dict[str, list[str]] = {}
    axes: dict[str, list[str]] = {}
    for cycle in cycles:
        if getattr(cycle, "is_hypothetical", False):
            continue
        dissent_name = getattr(getattr(cycle, "dissent", None), "specialist", "")
        valid_recommendations = [
            candidate.recommended_action
            for candidate in getattr(cycle, "candidates", [])
            if candidate.schema_valid and candidate.recommended_action
        ]
        unanimous_cycle = bool(
            len(valid_recommendations) >= 2 and len(set(valid_recommendations)) == 1
        )
        for candidate in getattr(cycle, "candidates", []):
            name = candidate.specialist
            record = records.setdefault(name, {
                "responses": 0,
                "valid_responses": 0,
                "malformed_responses": 0,
                "landscape_searches": 0,
                "productive_landscape_searches": 0,
                "semantic_landscape_failures": 0,
                "causal_mapping_errors": 0,
                "unresolved_tiebreaker_errors": 0,
                "procedural_randomization_warnings": 0,
                "speculative_claims": 0,
                "useful_dissent": 0,
                "grounded_nonconsensus": 0,
                "consensus_repetitions": 0,
                "position_changes": 0,
                "justified_position_changes": 0,
                "unjustified_position_changes": 0,
                "distinct_constraints": 0,
                "repeated_constraints": 0,
                "distinct_decisive_axes": 0,
                "total_preference_strength": 0.0,
                "total_epistemic_confidence": 0.0,
                "strong_preference_low_certainty": 0,
            })
            record["responses"] += 1
            if candidate.schema_valid:
                record["valid_responses"] += 1
            else:
                record["malformed_responses"] += 1
            if candidate.landscape_search_attempted:
                record["landscape_searches"] += 1
            if candidate.landscape_search_complete and candidate.landscape_semantic_valid:
                record["productive_landscape_searches"] += 1
            if candidate.landscape_search_attempted and not candidate.landscape_semantic_valid:
                record["semantic_landscape_failures"] += 1
            landscape_errors = candidate.landscape_validation_errors
            record["causal_mapping_errors"] += sum(
                "case for A" in error for error in landscape_errors
            )
            record["unresolved_tiebreaker_errors"] += sum(
                "tiebreaker fully succeeded" in error for error in landscape_errors
            )
            record["procedural_randomization_warnings"] += sum(
                "procedural randomization" in error for error in landscape_errors
            )
            if candidate.evidence_basis == "UNSTATED_FACTS":
                record["speculative_claims"] += 1
            if name == dissent_name:
                record["useful_dissent"] += 1
            if candidate.independence_bonus:
                record["grounded_nonconsensus"] += 1
            if unanimous_cycle:
                record["consensus_repetitions"] += 1
            if candidate.position_changed:
                record["position_changes"] += 1
                if candidate.change_justification.casefold() != "none":
                    record["justified_position_changes"] += 1
                else:
                    record["unjustified_position_changes"] += 1
            record["total_preference_strength"] += candidate.preference_strength
            record["total_epistemic_confidence"] += candidate.epistemic_confidence
            if candidate.preference_strength >= 0.60 and candidate.epistemic_confidence <= 0.40:
                record["strong_preference_low_certainty"] += 1
            constraints.setdefault(name, []).append(candidate.constraint)
            if candidate.landscape_decisive_axis:
                axes.setdefault(name, []).append(candidate.landscape_decisive_axis.casefold())

    for name, record in records.items():
        seen_constraints = constraints.get(name, [])
        record["distinct_constraints"] = len(set(seen_constraints))
        record["repeated_constraints"] = max(0, len(seen_constraints) - len(set(seen_constraints)))
        record["distinct_decisive_axes"] = len(set(axes.get(name, [])))
    return records


class EpisodicMemory:
    """Append-only storage. Retrieved rules are priors, never automatic commands."""

    def __init__(self, path: Path):
        self.path = path

    def append(self, episode: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        episodes = []
        for line in lines:
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return episodes

    def specialist_profiles(self, limit: int = 50) -> dict[str, dict[str, Any]]:
        """Aggregate recent contribution records for prompt-time calibration."""
        totals: dict[str, dict[str, Any]] = {}
        for episode in self.recent(limit):
            contributions = episode.get("specialist_contributions", {})
            if not isinstance(contributions, dict):
                continue
            for name, contribution in contributions.items():
                if not isinstance(contribution, dict):
                    continue
                profile = totals.setdefault(str(name), {"episodes": 0})
                profile["episodes"] += 1
                for metric, value in contribution.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    profile[metric] = profile.get(metric, 0) + value
        for profile in totals.values():
            searches = max(1, int(profile.get("landscape_searches", 0)))
            responses = max(1, int(profile.get("responses", 0)))
            profile["productive_search_rate"] = round(
                profile.get("productive_landscape_searches", 0) / searches, 3
            )
            profile["speculation_rate"] = round(
                profile.get("speculative_claims", 0) / responses, 3
            )
            profile["mean_preference_strength"] = round(
                profile.get("total_preference_strength", 0.0) / responses, 3
            )
            profile["mean_epistemic_confidence"] = round(
                profile.get("total_epistemic_confidence", 0.0) / responses, 3
            )
        return totals
