# Explicit-track trolley: live pipeline result

**A fresh model run froze one world and reached all five frameworks.**
The final status is FRAMEWORK_PARTIAL because some frameworks abstain or retain
uncertainty. Construction and transport succeeded; this is not an admission failure.

Read the [human report and Mermaid world graph](DEMO_REPORT.md),
[frozen topology](pipeline/world_state_topology.md), or
[framework submissions](frameworks/framework_generation.md).

## Representation

- Maria, one worker and five workers are three distinct human bearers. The copied
  references “that worker”/“the one” and “the five” resolve to their unique earlier
  counted noun phrases. Two equal-sized possible antecedents are not merged.
- The lever bears the direct intervention; the trolley bears the downstream
  “switches the trolley onto a side track” transition.
- Each action branch contains its stated death and spared-group outcome. “Spare”
  is preserved; it is not converted into an invented life expectancy or welfare unit.
- Main-track motion, side-track context and “Maria cannot stop the trolley” remain
  whole source propositions in the world and STIPULATED claims in the native ledger.
- Track nodes connect to source propositions through MENTIONED_IN display edges.
  Spatial meaning remains in those source statements; there is no geometry simulator
  or claim that a modal ability establishes an actual performed action.
- The omission retains its copied “does not” wording and branch condition. Its
  instrument-state node remains separate from the intervention node; no second
  switching event is invented.

## Admission and frameworks

Omission and conditional macros both passed admission. After binding each action
to its source if-clause, their graphs were equivalent and the beam collapsed as
INVARIANT. No threshold was lowered and no reviewer was instructed to select one.

| Framework | Native submission | Ledger | Reported position |
| --- | --- | --- | --- |
| Utilitarian | VALID | COMMITTED | FULL, pull |
| Deontological | VALID | COMMITTED_WITH_UNCERTAINTY | ABSTAIN |
| Care | VALID | COMMITTED_WITH_UNCERTAINTY | ATTENUATED, pull |
| Virtue | VALID | COMMITTED | FULL, pull |
| Rawlsian | VALID | COMMITTED | ABSTAIN |

Each framework used one live adapter call. The world was unchanged during these
assessments. Ledger admission does not prove the normative reasoning. For example,
the recorded utilitarian hinge says “five or more” workers would reverse the ranking;
equal counts would ordinarily remove the strict count advantage, not by themselves
establish the opposite ranking. The report preserves this as a model-authored claim.

An earlier live attempt is retained at
[trolley_scene_live_20261008](../trolley_scene_live_20261008/DEMO_REPORT.md): both
worlds admitted, but capability wording versus conditional wording prevented
selection. The subsequent fix corrected construction identity, rather than
silencing that failure.

## Verification and reuse

The broader regression ran 125 tests successfully with four environment skips.
Following the explicit-negation check, 22 trolley/discourse tests passed. Tests
cover missing/uncertain mechanisms, counted-reference ambiguity, both candidate
paths, native admission, whole-statement ledger recognition and preservation of
negation in “does not pull the lever.” The saved viewer's world equals the report,
frozen trace and framework shared world.

```bash
.venv/bin/python run_parliament_demo.py \
  --scenario demo_scenarios/trolley_explicit_tracks.txt \
  --output diagnostics/trolley_next_run
```

Use a new output directory for each run. This single success supports the bounded
instrument-switching construction; it does not establish a general success rate
for every trolley wording or a governing ethical verdict.
