# Rules with choices and uncertain outcomes — October 16, 2026

The laboratory case now constructs an occurrence model alongside its rule layer.
The final live preparation run selects the amended `deontic_rule` candidate,
passes native occurrence admission, and validates frozen-world replay. No ethical
framework generation or deliberation was run.

## The shared construction

This adds no laboratory-specific template or world schema. The primitive
inventory now recognizes a bounded source pattern before blueprint ranking:

- One explicit binary question names an actor and two positive action options.
- A local `doing so may/might/could ...` links an action to a possible outcome.
- An explicit duration statement, followed by `during that delay`, links the
  other action to a possible outcome during that interval.

Source parsing binds the same actor, action lemma and object. Alternative objects,
unstated outcomes, negated risks and additional unsupported action components
are not silently substituted. Unknown branches retain specific missing-binding
messages. Source text is not normalized before parsing; comparisons normalize
whitespace without changing the evidence.

Existing if-branch composition remains the first path. The new extractor is used
only when that inventory has no conditional branches. It handles different names,
verbs and durations; tests include an unlock/pause variant. Its supported grammar
is deliberately narrow, not a general non-conditional scenario parser.

The existing construction helpers, evidence closure, rule collections and native
admission are reused. The rule-only baseline stays available. A separate amended
candidate combines its rule layer with the choice/outcome construction; the
ordinary primitive-composition candidate is also retained. Selection uses the
existing beam and source-review machinery, without changing scores or validators.

## Final represented content

| Branch | Source-backed process | Possible outcome |
| --- | --- | --- |
| Open now | Open the door | Staff may be exposed to hazardous fumes |
| Stop and wait | Stop the experiment; stopping takes ten minutes | Technician may die during that delay |

There are two actions, five effects and three ENABLES links. The delay has its
own process record and retains `ten minutes` as a quantity/temporal qualifier.
Exposure and death remain POSSIBLE, with their source modal qualifiers. Hazardous
exposure is ADVERSE but OTHER: no particular injury or probability is invented.
No successful rescue, survival complement, certain death or moral verdict is added.

ENABLES supplies a candidate branch association, not an asserted direct-killing
mechanism. Its structural link is represented separately from the possible
outcome event. No separate stochastic gate is stated, so no self-referential
condition on the outcome is manufactured. The proposal retains relation
alternatives and provenance limitations through the existing envelope.

Both norms remain governed propositions, including the full `while running` and
`only after stopping` restrictions. This increment does not add a typed norm-to-
action applicability edge, resolve the second rule's bearer or `it`, merge the
two authority mentions, or determine whether a rule is breached. Other scene
content remains available in the original scenario and parser package; semantic
completeness is not claimed.

## Validation and live evidence

- Nine focused tests pass, including actual native occurrence admission and
  persistence of both POSSIBLE outcomes and both norms in the frozen trace.
- The broader run has 115 passing tests and four skips. One failure concerns the
  existing trolley process-state source-text expectation. A second known trolley
  test was run separately and fails because it expects automatic selection
  between different admitted graphs without source review.
- Both failures reproduce when loading the checkpoint's unchanged cloze module
  (`5230184`) with the same pinned native runtime. Their records are saved; no
  production selection policy or old test expectation was changed to hide them.
- A fresh live chooser plus source review selects the amended graph. Native
  admission is COMMITTED and `frozen_trace_valid` is true. The final source review
  is attributed model interpretation, not proof of semantic completeness.

An earlier live review incorrectly described stopping as enabling technician
“safety.” That explanation was unsupported; it introduced no new graph fact.
The original review remains saved under `live_prepare`. The final run is
`live_final`, with adverse exposure represented explicitly.

The first failed construction probes are retained: question-only support was
insufficient for the direct effect, a duration needed an explicit process parent,
and an outcome must not be used as its own probability gate. The final producer
corrects these representations; native checks and runtime patches are unchanged.

Inspect [the final Mermaid graph](diagnostics/deontic_choice_composition_20261016/live_final/world_state_topology.md),
[all candidate attempts](diagnostics/deontic_choice_composition_20261016/live_final/candidate_graphs.md),
[the run manifest](diagnostics/deontic_choice_composition_20261016/live_final/preparation_manifest.json),
and [regression details](diagnostics/deontic_choice_composition_20261016/RESULT.md).

```bash
./run_blueprint_parliament.sh \
  --scenario-file diagnostics/deontic_lab_template_live_20261015/scenario.txt \
  --output-dir diagnostics/my_lab_preparation \
  --prepare-only
```
