# Source-applicability increment

Four live o3 requests inspected two unchanged saved arguments: one source audit and one local revision per case. No new argument generation, native submission, world alteration or operative recommendation occurred. Both sources are frozen text fixtures, not newly admitted Parliament worlds. This is a small, non-blind diagnostic, not a matched causal estimate of reasoning improvement.

## What changed in the arguments

| Case | Previously missed issue | What the new reply did | What remains unresolved |
| --- | --- | --- | --- |
| Volunteer | Volunteer was treated as untrained; rule violation then assumed | Identified unknown training, superseded the overclaim with “Len enlists volunteer Ivo to handle the control,” and proposed a training-conditional judgment | The proposed conditional judgment referenced an undeclared hypothesis. Its two invalid operations remain rejected. The prior impermissibility conclusion remains inspectable but requires reassessment. Source-rule moral authority and consent sufficiency are still not justified. |
| Privacy | Possible benefit was treated as uniquely necessary, timely prevention of unavoidable harm | Withdrew the categorical disclosure recommendation and explicitly retained the only-timely-option/unavoidable-harm condition | The new support route uses PRACTICAL_REASONING without the required purpose/belief/use premises; it requires reassessment. Normative priority remains an assumption, not established merely by conditionalizing the recommendation. |

Inspect the [volunteer argument diagram](ineffective_volunteer.md) and [privacy argument diagram](privacy_emergency.md), or their [canonical volunteer](ineffective_volunteer.json) and [privacy](privacy_emergency.json) records. The prose is labeled as the critic's proposed conclusion; it is not a substitute for the accepted graph.

The audit is fallible. In the volunteer case it conflates “no emergency priority is stated” with “no emergency exception,” and treats lack of an explicit override as support for undefeated authority. Those are not equivalent. It also does not fully examine whether consent was informed. In the privacy case it recognizes the necessity precondition but insufficiently separates imminence from the normative priority premise. A successful local correction therefore does not certify the whole argument.

## Transport and construction

Both initial audit responses used invalid citation aliases; the volunteer response also paraphrased a claim component. The original strict binder retained the prior argument and requested no revision. We then resolved citations only where the exact quotation matched a unique source location, preserved the raw aliases, labeled paraphrases, and reused the unchanged saved audits. No audit was regenerated.

The volunteer revision initially failed atomically because of the undeclared hypothesis. Audited revision now preserves valid local edits while retaining invalid operations as problems. Supersession also preserves its explicitly supplied support operation. The exact saved replies were replayed offline after these mechanical fixes. Nothing invented a hypothesis, a training status, or a normative exception to make a graph pass.

Original calls: [volunteer audit](../premise_audit_volunteer_live_20261023/ineffective_volunteer/calls.json), [volunteer revision](../premise_audit_volunteer_replay_20261023/ineffective_volunteer/calls.json), [privacy audit](../premise_audit_privacy_live_20261023/privacy_emergency/calls.json), [privacy revision](../premise_audit_privacy_replay_20261023/privacy_emergency/calls.json). Reproduce the final compiler replay with zero API calls:

```bash
.runtime/venv/bin/python diagnostics/premise_audit_20261023/replay.py
```

The replay checks and records original call hashes. Original provider replies remain unchanged. Source judgments and critique judgments stay attributed model reviews, never independent proof or shared world facts.

Verification: **152 tests passed, two skipped**: 83 argument/protocol/native tests and 69 passing application tests. See [native regression log](regression_native.log) and [application regression log](regression_app.log). The application log's temporary missing-runtime message belongs to an intentional negative fixture. Tests cover exact source bindings, unknown versus opposite facts, a supported training control, normative/hypothetical separation, stale audit rejection, partial revision retention and atomic supersession with attached support. Authored source-review fixtures test mechanics; they do not measure the model's semantic accuracy. Live requests totaled four; the final compiler replay used zero requests.

## Assessment

This increment helps an agent correct precise factual applicability errors rather than merely elaborate its initial moral conclusion. It does not yet demonstrate better duty ranking, complete support construction, reliable consent analysis or better final recommendations. The remaining failures are visible in the graphs rather than summarized as successful admission. No extra API rounds were used to make these two examples look resolved.
