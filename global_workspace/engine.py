from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from .models import CandidateChunk, CycleRecord, WorkspaceBroadcast, WorkspaceResult, clamp


class Specialist(Protocol):
    name: str

    def evaluate(
        self,
        scenario: str,
        actions: Sequence[str],
        broadcast: WorkspaceBroadcast,
    ) -> CandidateChunk: ...


@dataclass(slots=True)
class WorkspaceConfig:
    max_cycles: int = 4
    high_urgency_cycles: int = 2
    time_budget_seconds: float = 180.0
    entropy_threshold: float = 0.62
    stable_cycles_required: int = 2
    surprise_weight: float = 0.28
    urgency_weight: float = 0.24
    friction_weight: float = 0.28
    minority_weight: float = 0.12
    redundancy_weight: float = 0.18
    min_valid_specialists: int = 2


class WorkspaceEngine:
    def __init__(self, specialists: Sequence[Specialist], config: WorkspaceConfig | None = None):
        if not specialists:
            raise ValueError("At least one specialist is required")
        self.specialists = list(specialists)
        self.config = config or WorkspaceConfig()

    def _salience(
        self,
        candidate: CandidateChunk,
        broadcast: WorkspaceBroadcast,
        constraint_counts: dict[str, int],
        minority_bonus: float,
    ) -> float:
        if not candidate.schema_valid:
            return 0.0
        cfg = self.config
        redundancy = min(1.0, constraint_counts.get(candidate.constraint, 0) / 2)
        return (
            cfg.surprise_weight * candidate.surprise
            + cfg.urgency_weight * broadcast.urgency
            + cfg.friction_weight * candidate.friction
            + cfg.minority_weight * minority_bonus
            - cfg.redundancy_weight * redundancy
        ) * (0.5 + 0.5 * candidate.confidence)

    @staticmethod
    def _policy(candidates: Sequence[CandidateChunk], actions: Sequence[str]) -> dict[str, float]:
        candidates = [candidate for candidate in candidates if candidate.schema_valid]
        if not candidates:
            return {action: 1.0 / len(actions) for action in actions}
        totals = {action: 0.0 for action in actions}
        weight_sum = 0.0
        for candidate in candidates:
            weight = max(0.05, candidate.confidence)
            weight_sum += weight
            for action in actions:
                totals[action] += weight * candidate.action_scores.get(action, 0.0)
        means = {action: score / weight_sum for action, score in totals.items()}
        temperature = 0.25
        exps = {action: math.exp(score / temperature) for action, score in means.items()}
        denominator = sum(exps.values()) or 1.0
        return {action: value / denominator for action, value in exps.items()}

    @staticmethod
    def _normalized_entropy(policy: dict[str, float]) -> float:
        if len(policy) <= 1:
            return 0.0
        entropy = -sum(p * math.log(p) for p in policy.values() if p > 0)
        return entropy / math.log(len(policy))

    @staticmethod
    def _dissent(candidates: Sequence[CandidateChunk], selected_action: str) -> CandidateChunk | None:
        alternatives = [
            candidate
            for candidate in candidates
            if candidate.schema_valid
            and candidate.action_scores
            and max(candidate.action_scores.values())
            - candidate.action_scores.get(selected_action, 0.0) > 0.15
        ]
        if not alternatives:
            return None
        return max(alternatives, key=lambda c: c.friction * c.confidence)

    def run(
        self,
        scenario: str,
        actions: Sequence[str],
        initial_broadcast: WorkspaceBroadcast | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> WorkspaceResult:
        clean_actions = list(dict.fromkeys(a.strip() for a in actions if a.strip()))[:5]
        if len(clean_actions) < 2:
            raise ValueError("At least two distinct actions are required")

        broadcast = initial_broadcast or WorkspaceBroadcast()
        result = WorkspaceResult(scenario=scenario, actions=clean_actions)
        started = time.monotonic()
        previous_action = ""
        stable_cycles = 0
        constraint_counts: dict[str, int] = {}
        previous_dissent: CandidateChunk | None = None
        cycle_limit = (
            min(self.config.max_cycles, self.config.high_urgency_cycles)
            if broadcast.urgency >= 0.8
            else self.config.max_cycles
        )

        for cycle_number in range(1, cycle_limit + 1):
            if progress:
                progress(f"Cycle {cycle_number}/{cycle_limit} — broadcast {broadcast.constraint}")
            candidates = []
            for index, specialist in enumerate(self.specialists, start=1):
                call_started = time.monotonic()
                if progress:
                    progress(
                        f"  [{index}/{len(self.specialists)}] {specialist.name} delegate thinking..."
                    )
                candidate = specialist.evaluate(scenario, clean_actions, broadcast)
                candidates.append(candidate)
                if progress:
                    validation_note = (
                        f"; error={candidate.validation_errors[0]}"
                        if not candidate.schema_valid and candidate.validation_errors
                        else ""
                    )
                    progress(
                        f"  [{index}/{len(self.specialists)}] {specialist.name} returned "
                        f"{candidate.constraint}; recommends={candidate.recommended_action or 'unknown'}; "
                        f"alignment={candidate.testimony_alignment}; "
                        f"why={candidate.rationale}{validation_note} "
                        f"in {time.monotonic() - call_started:.1f}s"
                    )
            minority_name = previous_dissent.specialist if previous_dissent else ""
            for candidate in candidates:
                bonus = 1.0 if candidate.specialist == minority_name else 0.0
                candidate.salience = self._salience(candidate, broadcast, constraint_counts, bonus)

            valid_candidates = [candidate for candidate in candidates if candidate.schema_valid]
            winner_pool = valid_candidates or candidates
            winner = max(winner_pool, key=lambda c: (c.salience, c.confidence, c.specialist))
            constraint_counts[winner.constraint] = constraint_counts.get(winner.constraint, 0) + 1
            policy = self._policy(candidates, clean_actions)
            selected_action = max(policy, key=policy.get)
            stable_cycles = stable_cycles + 1 if selected_action == previous_action else 1
            previous_action = selected_action
            entropy = self._normalized_entropy(policy)
            dissent = self._dissent(candidates, selected_action)
            elapsed = time.monotonic() - started

            next_broadcast = WorkspaceBroadcast(
                constraint=winner.constraint,
                intent=f"evaluate_{selected_action}",
                urgency=broadcast.urgency,
                danger_probability=broadcast.danger_probability,
                unresolved=(dissent.unresolved if dissent and dissent.unresolved != "NONE" else winner.unresolved),
            )
            result.cycles.append(
                CycleRecord(
                    cycle=cycle_number,
                    broadcast=next_broadcast,
                    candidates=candidates,
                    winner=winner,
                    dissent=dissent,
                    policy=policy,
                    entropy=entropy,
                    stable_cycles=stable_cycles,
                    elapsed_seconds=elapsed,
                )
            )
            previous_dissent = dissent
            broadcast = next_broadcast
            if progress:
                progress(
                    f"  cycle policy: {selected_action} "
                    f"({policy[selected_action]:.2f}); entropy={entropy:.2f}; "
                    f"winner={winner.specialist}:{winner.constraint}"
                )

            if len(valid_candidates) < self.config.min_valid_specialists:
                result.halted_by = "insufficient_valid_candidates"
                if progress:
                    progress(
                        f"  halting: only {len(valid_candidates)} valid delegate response(s); "
                        f"need {self.config.min_valid_specialists}"
                    )
                break

            dissent_addressed = dissent is None or (
                cycle_number > 1
                and result.cycles[-2].broadcast.unresolved == dissent.unresolved
            )
            if elapsed >= self.config.time_budget_seconds:
                result.halted_by = "time_budget"
                break
            if entropy < self.config.entropy_threshold and stable_cycles >= self.config.stable_cycles_required and dissent_addressed:
                result.halted_by = "convergence"
                break
        else:
            result.halted_by = "cycle_budget"

        final = result.cycles[-1]
        if result.halted_by == "insufficient_valid_candidates":
            result.selected_action = "INCONCLUSIVE"
            result.confidence = 0.0
        else:
            result.selected_action = max(final.policy, key=final.policy.get)
            result.confidence = clamp(final.policy[result.selected_action])
        result.moral_residue = sorted({
            candidate.constraint
            for cycle in result.cycles
            for candidate in cycle.candidates
            if result.selected_action != "INCONCLUSIVE"
            and candidate.schema_valid
            and candidate.action_scores.get(result.selected_action, 0.0) < 0.5
        } | {
            cycle.dissent.constraint
            for cycle in result.cycles
            if cycle.dissent is not None and cycle.dissent.schema_valid
        })
        result.reopen_conditions = sorted({
            candidate.unresolved
            for cycle in result.cycles
            for candidate in cycle.candidates
            if candidate.unresolved != "NONE"
        })
        if result.halted_by != "convergence" or not final.winner.schema_valid:
            result.compressed_rule = f"Unavailable: deliberation halted by {result.halted_by}."
        else:
            result.compressed_rule = (
                f"When {final.broadcast.constraint.lower().replace('_', ' ')} is salient, "
                f"prefer {result.selected_action}; reopen if "
                f"{', '.join(result.reopen_conditions) or 'material facts change'}."
            )
        return result
