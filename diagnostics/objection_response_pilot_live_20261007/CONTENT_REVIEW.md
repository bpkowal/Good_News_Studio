# First bounded objection-response comparison

The experiment completed twelve live o3 revisions: two cases, three frameworks,
two arms. Both arms used exactly the same saved October 6 standard opening,
assigned objection, source world, framework commitment, native prompt, envelope
schema and 4096-token ceiling. All [matching controls](controls.json) pass.
Both admitted worlds remained unchanged. No peer arguments or collective verdict
were generated, and no existing operative ledger was updated.

The focused instruction made some reasoning more explicit. It did not establish
better objection handling in the promise case. More elaborate explanations still
used unsupported empirical assumptions or insufficiently defended normative
priorities. Eleven replies called themselves `answered`; one called itself
`partly_answered`. Those self-reports overstate what this review supports.

## Content assessment

This is one non-blind assistant review, not an independent normative oracle or
reliable treatment-effect estimate. Scores use the prewritten 0–2 rubric:
actual objection addressed; bridge defended; uncertainty and unsupported factual
assumptions handled; reversal condition relevant and concrete. No total score is
used. Framework interpretations are assessed as arguments, not declared true
because the agent invokes Kant or Tronto.

| Case / framework | Ordinary: target / bridge / uncertainty / reversal | Focused: target / bridge / uncertainty / reversal |
| --- | --- | --- |
| Promise / utilitarian | 2 / 1 / 0 / 1 | 2 / 1 / 0 / 1 |
| Promise / deontological | 2 / 1 / 1 / 2 | 2 / 1 / 1 / 2 |
| Promise / Care | 2 / 1 / 1 / 1 | 2 / 1 / 1 / 1 |
| Probability swap / utilitarian | 2 / 1 / 1 / 2 | 2 / 1 / 1 / 2 |
| Probability swap / deontological | 2 / 1 / 1 / 2 | 2 / 2 / 1 / 2 |
| Probability swap / Care | 2 / 2 / 1 / 2 | 2 / 2 / 1 / 2 |

**Utilitarian, promise.** Both arms recognize that survivor counts do not settle
total welfare, then introduce an unprovided bound on promise/trust costs. Ordinary
revision says trust costs rarely approach a life; focused revision appeals to
unnamed empirical studies and calls the welfare loss normally smaller. Neither
supplies source support or a valid common welfare unit. The focused reply improves
its self-report to `partly_answered`, but still uses that assumption to justify a
total-welfare preference. Both native assessments flag an ungrounded numeric
reversal threshold and withhold an operative recommendation.

**Deontology, promise.** Both arms explain a universalization argument for fidelity
and latitude in beneficence, rather than just naming duties. However, they do not
adequately defend why emergency rescue cannot constrain this particular promise.
The focused reply explicitly admits uncertainty about emergency rescue and promise
validity while claiming the objection answered. Both native ledgers commit with
uncertainty and retain the same classification/unsupported-premise warnings.
The review does not assume the promise must win or must lose.

**Care, promise.** Both arms explain competence and trust, but treat a sequence of
care phases as a priority rule. The supplied Tronto excerpt names the four phases;
it does not establish that earlier phases defeat later ones. The focused reply
additionally declares the promise morally defective without defending that step
against Cara's claim. The roughly 90% reversal threshold is an illustrative,
unjustified boundary, not scenario evidence. Both native submissions select while
their care records remain contested; neither commits a Care ledger. Their argument
records remain inspectable rather than disappearing with rejection.

**Utilitarian, probability swap.** Both arms correctly compute the restricted
0.95 versus 0.05 survivor endpoint and name missing welfare information. Both
nevertheless treat absence of other information as symmetry or dominance of all
other welfare effects. The replies also introduce life-year language without a
duration premise. Native numeric admission verifies the separate survivor
calculation; it does not verify the broader welfare claim in the explanation.

**Deontology, probability swap.** Both distinguish an unchanged duty from changed
effectiveness. The focused reply adds a case-specific universalizable maxim and
explains how effectiveness could specify an imperfect duty. This is the clearest
local gain. It still treats the patients as strangers and absence of a stated
special claim as equality; those assumptions should be conditional, not established
facts. One sampled improvement is not evidence of dependable superiority.

**Care, probability swap.** Both explain how effectiveness could decide between
equally urgent needs without inventing a special relationship. The focused reply
identifies the challenged inference more explicitly. Both rely on provisional
relational symmetry because no asymmetry is supplied; unknown relationships remain
unknown. Both give relevant reversal conditions, and native ledgers commit.

## Admission, separately

All twelve submissions are structurally VALID after corrected native transport.
Ten have committed ledgers, including two deontological ledgers with uncertainty.
The two promise-case Care ledgers remain absent. In the probability-swap case all
six ledgers commit and prefer Cara. Admission patterns are identical between arms.
No native objection was retired by the model's `answered` label.

An experiment-adapter bug initially passed a decoded native dictionary instead of
the native `choices` transport envelope. The live provider responses were intact.
All twelve original assessment failures remain saved; `native_replay` re-assessed
the exact saved native fields with zero API calls. Each replay records field hashes
and provenance. This corrects transport, not model reasoning. The adapter is fixed
and a native integration test covers that boundary.

Some explanation fields end mid-sentence at the shared 1000-character schema limit.
The separate bridge/uncertainty/reversal fields survive, but this truncation limits
content comparison. Thirteen related regression tests pass; the preceding broader
suite had 129 passing tests. Neither world schemas nor native validators changed.

## Next bounded change

Test a single follow-up that challenges one unsupported bridge from these exact
saved replies, rather than requesting another generally stronger argument. For
example: “Your trust-cost upper bound is not established. Restate the strongest
conclusion available without it, or identify it explicitly as a defeasible normative
assumption.” For Care, challenge why the four phases create a priority ordering.
Preserve a provisional preference and the objection if the bridge cannot be
defended. That would test whether an agent can correct a specific inference instead
of merely elaborating it. It has not been run in this experiment.

Inspect [all outputs](RESULT.md), [raw arguments and native results](results.json),
and the [experiment design](../../OBJECTION_RESPONSE_PILOT.md).
