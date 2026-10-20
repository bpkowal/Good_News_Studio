# Deontic rule blueprint — first construction increment

The laboratory scenario exposed an underdeveloped path, not evidence that rule
representation is intrinsically unworkable. Allocation and trolley have source
recovery, role binding and multiple branch builders. Deontic had three core
blanks and one norm builder. Its tests mainly covered a single modal such as
“Ada must deliver medicine,” with already correct copied fills.

The live failure crossed several boundaries:

1. The model combined `requires, allows` and two contents into non-source strings.
2. The operator checks recognized “must” but missed common institutional wording;
   the exception check missed “only after.”
3. One norm could not preserve the obligation and the later conditional permission.
4. Force could come from a Z10 modal elsewhere in the scenario, rather than the
   particular norm being built.
5. The fallback alternatives describe different constructions. “Stop and wait”
   is not a source-copied negation of opening, and exposure is not one of the
   existing narrow harm verbs. Those separate gaps remain.

## Implemented

`blueprint_deontic_rules.py` reads the original parsed text. A matrix institutional
rule operator with one explicit verbal complement yields an operator/content
bundle. Requires/obliges, allows/permits, and forbids/prohibits are paired with
their own complements, bearers and authority mentions. This increment does not
extract mechanistic “allows,” numeric requirements, attributed rules or questions
as positive rule assertions. Negation of a requirement remains distinct from a
prohibition.

Bundles include exact operator/content offsets, source clause IDs and scope
spans. Whitespace is not normalized before parsing. They can recover failed or
absent cloze fills; the original model completion and rejected hypotheses remain
inspectable. Hypothesis IDs remain unique during recovery.

The cloze template has optional second-rule fields and instructions to keep
operators and contents paired. A second independently copied cloze rule can
also build a second norm when its operator and content share a source clause.
Existing single-rule slot IDs remain compatible. Ranking weights and native
admission checks are unchanged.

The builder emits separate schema 1.4 norms and governed propositions. For the
laboratory scenario these are:

- Obligation: `keep a containment door closed while a hazardous experiment is running`.
- Permission: `opening it only after the experiment has stopped`.

The complete restrictions stay in the governed text. Structured scope spans also
remain in the proposal's `slot_bindings.deontic_rule_bindings`. This schema has
no typed norm-to-condition relation; none is fabricated. There is no inference
that opening is permitted now or that the experiment actually stopped.

Mira is the first norm's explicit bearer. The second bearer is unstated; `it`
remains unresolved. `The rule` remains a separate authority mention from
`A laboratory safety rule`. Their identity is not silently merged. These textual
authority mentions are OTHER, not invented people. Neither rule establishes that
an action occurred or that the rule is ethically decisive.

## Evidence

The exact initial live completions are frozen in
`fixtures/deontic_lab_cloze.json`. Their offline replay now selects deontic_rule,
fills its three required slots, produces two norms and passes the proposal
contract. It does not depend on a model rewriting its old answer.

A fresh `gpt-4o-mini` chooser run on the unchanged scenario also selects
deontic_rule. The raw response still joins operators and contents and leaves the
second-rule fields empty; the source parser supplies the correct bundles. This
shows working source recovery, not proven improvement in model-only filling.

The existing preparation path records the result as ADMITTED, discourse committed.
This is the discourse-only admission path: zero occurrence effects, no native
occurrence replay (`frozen_trace_valid: false`), and no framework calls. It is
**not a complete decision world**. Staff exposure, rescue, delay and the choices
still need occurrence construction before an ethical pipeline run would be useful.

The regression command ran 135 tests: 130 passed, five skipped for missing native
test-environment configuration. The ten focused rule tests passed again after
final cleanup. They cover saved-answer recovery, operator locality, separate
bearers, source offsets, scope negation, negative controls and old single-rule
compatibility. Final code produces the same rule-world collections as the saved
live run, checked offline without modifying that trace.

```bash
.venv/bin/python -m unittest test_deontic_rule_bundles -q
```

Inspect the [initial failure](diagnostics/deontic_lab_template_live_20261015/INSPECTION.md),
[new candidate attempts](diagnostics/deontic_lab_template_live_20261015/after_rule_bundles/candidate_graphs.md),
and [generated rule graph](diagnostics/deontic_lab_template_live_20261015/after_rule_bundles/world_state_topology.md).

The next bounded step is to compose this rule layer with an explicit
open-now/stop-and-wait choice-and-outcomes model, retaining the conditional
permission and uncertainty. Better rule extraction alone must not be mistaken
for complete semantic coverage or better ethical argument quality.

Update, October 16: that bounded composition is implemented and has passed live
preparation and native occurrence replay. See
[rules with choices and uncertain outcomes](DEONTIC_CHOICE_COMPOSITION.md) for
the final graph, tests and remaining limits.
