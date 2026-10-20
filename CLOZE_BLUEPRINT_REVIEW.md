# Cloze blueprint review — October 15, 2026

The allocation and trolley blueprints work best when they ask for different kinds
of information separately, rather than asking the model to restate the scenario.
The useful common unit is a source-backed slot, not a fixed number of blanks.

| Strength | Allocation | Trolley | Reusable lesson |
| --- | --- | --- | --- |
| Small actor and participant fills | Decider and two recipients | Actor, process entity and branch patients | Keep entities separate from predicates |
| Branch structure | First/second transfer and outcomes | Doing/omitting action and their outcomes | Preserve which choice each outcome belongs to |
| Quantities | Resource amount differs from people counts | Affected-group counts differ by branch | Do not ask one fill to carry all cardinalities |
| Scope | Conditional and probability evidence | Action/omission and conditional outcomes | Shorter labels must retain operators in evidence |
| Closure | Copied transfer can recover an assignment | Parsed source binds the moving process and omission branch | Reuse supported structure rather than inventing missing events |

## First shared increment

`blueprint_cloze_chooser.py` version 0.16 adds span-shape hints to existing slots
and reuses the existing parser helpers. It does not add required blanks, change
ranking weights, modify native admission, or add a world-state schema.

- Rescue, conditional outcome, diversion and risk: separate an initial copied
  if-condition from its consequence. Action fills keep negation/modal words.
- All eight families outside allocation/omission: recover the actor/speaker's
  noun phrase from an overlong answer. When the answer contains an if-condition,
  use that condition's actor, not the consequence's victim.
- Diversion and risk: recover the patient noun phrase from a copied harm clause;
  retain group counts. Diversion also reuses process noun-phrase extraction.
- Promise/report content stays a full proposition. It is not shortened to a
  verb or promoted into an occurrence effect. Required complements that the
  simple action helper cannot safely shorten stay intact.

The original completion stays in the item record; the normalized hypothesis is
still an exact contiguous source copy. No unsupported referent is synthesized.
Existing allocation/omission slot definitions are unchanged.

## Evidence and limits

The offline [saved-answer replay](diagnostics/cloze_compact_slots_20261015/saved_completion_comparison.json)
covers 14 historical scenarios and 42 template attempts. Their accepted slots
and core-fill counts are unchanged: these answers were already compact. This
does not demonstrate a new increase in admission rates.

The separate [controlled copied-answer probes](diagnostics/cloze_compact_slots_20261015/controlled_span_comparison.json)
exercise overlong fills. For example, `If Ada opens the gate, Ben might drown`
previously occupied actor, action and outcome slots. Those fills now become
`Ada`, `opens the gate` and `Ben might drown`. A diversion harm clause yields
`three workers` as patient and `The trolley` as process. These are controlled
inputs, not new live model samples. Reproduce both comparisons with:

```bash
.venv/bin/python diagnostics/cloze_compact_slots_20261015/replay.py
```

Historical reports and frozen worlds are not rewritten. No API calls were made.
The historical `run_blueprint_variants.py` CLI has a stale import of
`ADMITTED_WORLD_STATUSES`; this replay reads its scenario data literal without
invoking that CLI.

## Remaining construction work

Shorter fills solve only one part of the gap. Rescue can still lack an explicitly
named harm process, diversion may lack a stated non-intervention outcome, and
risk may have a partial second branch. Those problems remain visible. A promise
or disputed report can be represented successfully without being a complete
choice-and-outcomes occurrence model.

The next useful increment is source-backed branch bundles for rescue and risk:
actor, action, affected participant, outcome and its scope, together with any
explicit process. Test a compact answer and an overlong answer against the same
expected graph and native admission result. Do not fill an omitted branch by
assuming its opposite, invent a rescue mechanism, or turn a report into a fact.
