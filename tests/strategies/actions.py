"""Strategies for declared exclusive versus open action pairs.

The oracle is choice_kind, not _scenario_closes_action_set. CLOSED templates
use the production exclusivity cues; OPEN templates omit them.
"""
from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st

from strategies.worlds import unique_tokens


_CHOICE_KINDS = ("CLOSED", "OPEN")
_CLOSED_CUES = frozenset({
    "only", "sole", "must", "cannot", "between", "otherwise", "option",
    "choice", "action", "first", "second", "other", "either", "split",
    "share", "rotate", "switch", "both",
})
_CLOSED_FRAMES = (
    "{actor} must choose between {a0} and {a1}",
    "the {facility} can only keep one of two isolated options",
    "there is no other way to proceed",
    "{actor} cannot choose otherwise",
)


@dataclass(frozen=True, slots=True)
class ClosedActionSetCase:
    """Two actions plus a share/split third action, exclusive or not."""

    scenario: str
    actions: tuple[str, str]
    third_action: str
    choice_kind: str

    @property
    def should_withhold(self) -> bool:
        return self.choice_kind == "CLOSED"


@st.composite
def closed_action_set_cases(draw) -> ClosedActionSetCase:
    actor, facility, verb0, verb1 = draw(
        unique_tokens(4).filter(
            lambda tokens: not any(token in _CLOSED_CUES for token in tokens)
        )
    )
    choice_kind = draw(st.sampled_from(_CHOICE_KINDS))
    actions = (
        f"{verb0} the {facility}",
        f"{verb1} the {facility}",
    )
    if choice_kind == "CLOSED":
        frame = draw(st.sampled_from(_CLOSED_FRAMES)).format(
            actor=actor, facility=facility, a0=actions[0], a1=actions[1],
        )
        scenario = f"An {actor} faces {actions[0]} or {actions[1]}. {frame}."
    else:
        scenario = f"An {actor} may {actions[0]} or {actions[1]}."
    return ClosedActionSetCase(
        scenario=scenario,
        actions=actions,
        third_action=f"split {verb0} and {verb1}",
        choice_kind=choice_kind,
    )
